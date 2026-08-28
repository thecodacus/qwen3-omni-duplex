"""Direct checkpoint loader — skips transformers' compression pass.

`from_pretrained` takes ~910 s on this checkpoint, and roughly 480 s of that is
`ModelCompressor.compress_model()` walking 18,624 modules at ~29/s. That pass is the
*save* path (pack a dense model to int4) reused on load to build the parameter
structure. Its output is discarded here: this project bypasses compressed-tensors
entirely and dequantizes with its own Triton kernel.

So: build the module tree on `meta` (instant, no memory), then read the shards once
and place each tensor directly where it belongs. `clock.py` already proved the
approach — 9,312 packed linears in 129 s.

Packed triplets (`weight_packed` / `weight_scale` / `weight_shape`) are attached as
buffers on the owning Linear, which is what the Triton path expects. The dense
`weight` those Linears would otherwise have is removed.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

PACKED_SUFFIXES = (".weight_packed", ".weight_scale", ".weight_shape")


def _assign(model: nn.Module, mods: dict, name: str, tensor: torch.Tensor,
            device, dtype) -> bool:
    """Place one checkpoint tensor onto the (meta-built) module tree."""
    owner_name, _, attr = name.rpartition(".")
    mod = mods.get(owner_name)
    if mod is None:
        return False

    if name.endswith(PACKED_SUFFIXES):
        # int4 payloads: keep packed dtype, only scales follow the compute dtype.
        t = tensor.to(device) if attr != "weight_scale" else tensor.to(device, dtype)
        mod.register_buffer(attr, t, persistent=False)
        # the dense weight this Linear was built with is never used
        if attr == "weight_packed" and "weight" in mod._parameters:
            del mod._parameters["weight"]
        return True

    t = tensor.to(device, dtype) if tensor.is_floating_point() else tensor.to(device)
    if attr in mod._parameters:
        mod._parameters[attr] = nn.Parameter(t, requires_grad=False)
        return True
    if attr in mod._buffers:
        mod._buffers[attr] = t
        return True
    return False


def load_direct(model_path: str, device: str = "cuda", dtype=torch.bfloat16,
                skip_vision: bool = True, log=print):
    """Build on meta, then fill from shards. Returns (model, config)."""
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeConfig,
    )
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeForConditionalGeneration,
    )
    from safetensors.torch import load_file

    root = Path(model_path)
    t0 = time.time()
    cfg = Qwen3OmniMoeConfig(**json.loads((root / "config.json").read_text()))
    # never let the quantizer engage; we own dequantization
    for c in (cfg, getattr(cfg, "thinker_config", None)):
        if c is not None and hasattr(c, "quantization_config"):
            try:
                delattr(c, "quantization_config")
            except Exception:
                c.quantization_config = None

    # Parameters on meta, but BUFFERS COMPUTED NORMALLY. Seven tensors in this
    # model are built in __init__ rather than stored in the checkpoint:
    #   thinker.model.rotary_emb.inv_freq              (64)
    #   talker.model.rotary_emb.inv_freq               (64)
    #   talker.code_predictor.model.rotary_emb.inv_freq (64)
    #   code2wav.pre_transformer.rotary_emb.inv_freq   (32)
    #   code2wav.code_offset                        (1,16,1)
    #   thinker.audio_tower..positional_embedding  (1500,1280)
    #   thinker.visual.rotary_pos_emb.inv_freq         (18)
    # A plain `torch.device("meta")` build leaves these on meta, and materialising
    # them as zeros destroys the model silently: inv_freq = 0 gives every position
    # an identical rotation, so the thinker emits "\n\n\n't\n\n\n". code_offset = 0
    # makes every codebook read book 0.
    log("building module tree (params on meta, buffers computed)")
    from accelerate import init_empty_weights
    with init_empty_weights(include_buffers=False):
        model = Qwen3OmniMoeForConditionalGeneration(cfg)
    if skip_vision and getattr(model.thinker, "visual", None) is not None:
        model.thinker.visual = None      # 0.5 GB this pipeline never touches
    log(f"  tree built in {time.time()-t0:.1f}s")

    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k, shard in index.items():
        if skip_vision and k.startswith("thinker.visual."):
            continue
        by_shard.setdefault(shard, []).append(k)

    mods = dict(model.named_modules())
    placed = missed = 0
    for i, (shard, keys) in enumerate(sorted(by_shard.items()), 1):
        st = time.time()
        blob = load_file(str(root / shard))
        for k in keys:
            if _assign(model, mods, k, blob[k], device, dtype):
                placed += 1
            else:
                missed += 1
        del blob
        gc.collect()
        log(f"  shard {i}/{len(by_shard)}: {len(keys)} tensors, {time.time()-st:.1f}s")

    # Anything still on meta was never in the checkpoint (tied weights, buffers
    # built at construction). Materialise ONLY those.
    #
    # Emphatically not `model.to_empty(device=...)`: that reallocates EVERY
    # parameter as uninitialised memory, discarding all 64907 tensors just loaded.
    # It fails loudly only by luck — a `weight_shape` buffer read as garbage gave
    # `sizes=[128, 3329622686324454514, 256]`.
    n_meta = 0
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if not t.is_meta:
            continue
        if "inv_freq" in name or "code_offset" in name or "positional_embedding" in name:
            raise RuntimeError(
                f"computed buffer {name} is still on meta — it would be zeroed, "
                "which silently destroys the model. init_empty_weights should have "
                "built it.")
        owner_name, _, attr = name.rpartition(".")
        mod = mods.get(owner_name)
        if mod is None:
            continue
        fresh = torch.zeros(t.shape, dtype=t.dtype, device=device)
        if attr in mod._parameters:
            mod._parameters[attr] = nn.Parameter(fresh, requires_grad=False)
        else:
            mod._buffers[attr] = fresh
        n_meta += 1
    if n_meta:
        log(f"  materialised {n_meta} tensors absent from the checkpoint")

    log(f"loaded {placed} tensors ({missed} unmatched) in {time.time()-t0:.0f}s")
    return model.eval(), cfg


def main():
    import argparse
    p = argparse.ArgumentParser(description="time the direct loader")
    p.add_argument("--model", default="/root/models/Qwen3-Omni-AWQ")
    p.add_argument("--device", default="cpu",
                   help="cpu just times the load; cuda also places it")
    a = p.parse_args()

    t0 = time.time()
    model, cfg = load_direct(a.model, device=a.device)
    dt = time.time() - t0
    n_packed = sum(1 for m in model.modules() if hasattr(m, "weight_packed"))
    print(f"\ndirect load: {dt:.0f}s   packed linears: {n_packed}")
    print(f"  vs from_pretrained: ~910s   speedup {910/max(dt,1e-9):.1f}x")
    if a.device == "cuda":
        print(f"  GPU resident: {torch.cuda.memory_allocated()/1024**3:.2f} GB")


if __name__ == "__main__":
    main()
