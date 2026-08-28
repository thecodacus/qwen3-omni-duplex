"""The full clock path in one loop, with real weights.

Every realtime number in this project was measured in isolation:

    talker + MTP + vocoder    36.85 ms p99.9   (clock-real)
    Thinker[0:24] MoE         15.3 ms          (Triton, separately)
    Thinker attention         ~4 ms            (estimated)

Nothing had run them together. That matters: the `expandable_segments` surprise
showed a change to one stage adding 68 ms of p99.9 to an untouched one, so
per-stage numbers do not compose by addition until measured.

This assembles the real thing:

    Thinker[0:24]  real weights, int4, Triton dequant-GEMV for the MoE,
                   partial residency (N layers pinned, rest streamed from host)
      -> layer-24 hidden
    Talker         real weights, bf16, fused gather+bmm MoE
    MTP            residual codebooks
    Code2Wav       stateful streaming vocoder, one 80 ms frame per iteration

Built on transformers' own `Qwen3OmniMoeThinkerTextModel` truncated to 24 layers,
with its modules patched, rather than a reimplementation — attention, RoPE and
residuals are easy to get subtly wrong and this project has already paid for one
such bug (see the bias trap in docs/design.md section 6).
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from duplex.streaming.code2wav import StreamingCode2Wav
from duplex.streaming.talker import load_talker
from duplex.thesis.vocoder_standalone import load_code2wav


def load_thinker_prefix(model_path: str, n_layers: int, device: str, dtype: torch.dtype):
    """Build Thinker[0:n_layers] with its packed weights attached as buffers.

    A freshly constructed module has `weight`, not `weight_packed`, so the packed
    tensors are registered manually onto the matching submodules.
    """
    from safetensors.torch import load_file
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerConfig,
    )
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerTextModel,
    )

    root = Path(model_path)
    cfg_all = json.loads((root / "config.json").read_text())
    tcfg = Qwen3OmniMoeThinkerConfig(**cfg_all["thinker_config"]).text_config
    tcfg.num_hidden_layers = n_layers
    model = Qwen3OmniMoeThinkerTextModel(tcfg).to(dtype)

    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    want = {}
    for k, shard in index.items():
        if not k.startswith("thinker.model."):
            continue
        sub = k[len("thinker.model."):]
        if sub.startswith("layers."):
            li = int(sub.split(".")[1])
            if li >= n_layers:
                continue
        want.setdefault(shard, []).append((sub, k))

    mods = dict(model.named_modules())
    plain = {}
    n_packed = 0
    for shard, keys in sorted(want.items()):
        blob = load_file(str(root / shard))
        for sub, full in keys:
            t = blob[full]
            if sub.endswith((".weight_packed", ".weight_scale", ".weight_shape")):
                owner, attr = sub.rsplit(".", 1)
                m = mods.get(owner)
                if m is None:
                    continue
                m.register_buffer(attr, t.to(dtype if attr == "weight_scale" else t.dtype),
                                  persistent=False)
                if attr == "weight_packed":
                    n_packed += 1
            else:
                plain[sub] = t
        del blob
        gc.collect()

    missing, unexpected = model.load_state_dict(plain, strict=False)
    # the packed linears legitimately have no `.weight`
    missing = [m for m in missing if not m.endswith(".weight")]
    if missing:
        print(f"  ! unfilled: {missing[:4]}")
    return model, tcfg, n_packed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/models/Qwen3-Omni-AWQ")
    p.add_argument("--layers", type=int, default=24, help="thinker prefix depth (the tap)")
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--out", default="/root/q3o_out")
    p.add_argument("--pinned", type=int, default=12,
                   help="thinker MoE layers kept on the GPU; the rest stream from "
                        "pinned host RAM. 12/24 is the measured working point (docs §13)")
    p.add_argument("--no-talker", action="store_true", help="time the thinker prefix alone")
    a = p.parse_args()

    dev, dtype = "cuda", torch.bfloat16
    from duplex.streaming.fused_packed_moe import fuse_packed_moe_blocks
    from duplex.quant.dequant import patch_packed_linears

    print(f"loading Thinker[0:{a.layers}] ...", flush=True)
    t0 = time.time()
    th, tcfg, n_packed = load_thinker_prefix(a.model, a.layers, dev, dtype)
    print(f"  {n_packed} packed linears, {time.time()-t0:.0f}s", flush=True)

    # MoE via the Triton dequant-GEMV; attention via the torch dequant path
    # (4 linears/layer, ~7 MB packed each -- not worth a kernel yet).
    fuse_packed_moe_blocks(th, dtype=dtype, device=dev, n_pinned=a.pinned)
    patch_packed_linears(th, compute_device=dev, verbose=False)
    # .to(dev) must not drag the host-resident experts onto the card; they are held
    # as plain attributes rather than buffers precisely so this leaves them alone.
    th = th.to(dev).eval()
    th_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  thinker prefix resident: {th_gb:.2f} GB", flush=True)

    tk = c2w = sc = None
    if not a.no_talker:
        tk, tkcfg, _ = load_talker(a.model, dev, dtype, fuse=True)
        c2w, ccfg, _, _ = load_code2wav(a.model, dev, dtype)
        sc = StreamingCode2Wav(c2w).install()
        sc.reset()
        print(f"  + talker + code2wav: {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)

    ups = 1920
    sr = 24000
    budget = ups / sr * 1000
    H = tcfg.hidden_size
    from transformers.cache_utils import DynamicCache

    th_kv = DynamicCache()
    tk_kv = DynamicCache() if tk is not None else None
    acc_layer = a.layers  # hidden_states[a.layers] is the output of layer a.layers-1

    stage = {k: [] for k in ("thinker", "talker", "mtp", "vocoder", "total")}
    sync = torch.cuda.synchronize

    try:
        with torch.inference_mode():
            for i in range(a.warmup + a.frames):
                emb = torch.randn(1, 1, H, device=dev, dtype=dtype)
                sync(); t0 = time.perf_counter()

                out = th(inputs_embeds=emb, past_key_values=th_kv, use_cache=True,
                         output_hidden_states=True)
                th_kv = out.past_key_values
                tap = out.hidden_states[-1]                     # the layer-24 tap
                sync(); t1 = time.perf_counter()

                if tk is not None:
                    h = tk.model(inputs_embeds=tk.text_projection(tap),
                                 past_key_values=tk_kv, use_cache=True)
                    tk_kv = h.past_key_values
                    hs = h.last_hidden_state
                    sync(); t2 = time.perf_counter()

                    ng = tkcfg.num_code_groups
                    tk.code_predictor.model(inputs_embeds=hs.expand(-1, ng - 1, -1))
                    sync(); t3 = time.perf_counter()

                    codes = torch.randint(0, ccfg.codebook_size,
                                          (1, ccfg.num_quantizers, 1), device=dev)
                    sc.decode(codes)
                    sync(); t4 = time.perf_counter()
                else:
                    t2 = t3 = t4 = t1

                if i >= a.warmup:
                    stage["thinker"].append((t1 - t0) * 1000)
                    stage["talker"].append((t2 - t1) * 1000)
                    stage["mtp"].append((t3 - t2) * 1000)
                    stage["vocoder"].append((t4 - t3) * 1000)
                    stage["total"].append((t4 - t0) * 1000)
    finally:
        if sc is not None:
            sc.remove()

    print(f"\nbudget {budget:.1f} ms/frame   resident "
          f"{torch.cuda.memory_allocated()/1024**3:.2f} GB\n")
    print(f"{'stage':<12}{'mean':>8}{'p50':>8}{'p99':>8}{'p99.9':>8}{'max':>8}")
    res = {}
    for k, v in stage.items():
        if not v or (k not in ("thinker", "total") and tk is None):
            continue
        x = np.array(v)
        res[k] = dict(mean=float(x.mean()), p50=float(np.percentile(x, 50)),
                      p99=float(np.percentile(x, 99)),
                      p999=float(np.percentile(x, 99.9)), max=float(x.max()))
        print(f"{k:<12}{res[k]['mean']:>8.2f}{res[k]['p50']:>8.2f}"
              f"{res[k]['p99']:>8.2f}{res[k]['p999']:>8.2f}{res[k]['max']:>8.2f}")

    tot = res["total"]
    over = sum(1 for x in stage["total"] if x > budget)
    print(f"\nVERDICT: p99.9 {tot['p999']:.2f} ms vs {budget:.1f} ms -> "
          f"{'PASS' if tot['p999'] < budget else 'FAIL'}   ({over}/{len(stage['total'])} over)")
    print(f"  realtime factor: {budget/tot['mean']:.2f}x")

    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump({"stages": res, "budget_ms": budget, "frames": len(stage["total"]),
               "resident_gb": torch.cuda.memory_allocated() / 1024**3,
               "over_budget": over, "layers": a.layers, "pinned": a.pinned},
              open(Path(a.out) / "clock_full.json", "w"), indent=2)


if __name__ == "__main__":
    main()
