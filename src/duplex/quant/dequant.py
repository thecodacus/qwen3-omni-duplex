"""On-the-fly dequantization for the AWQ Thinker, bypassing compressed-tensors.

transformers' compressed-tensors integration fails on this checkpoint: it attaches
`quantized_forward` to plain `nn.Linear` modules that still hold `weight_packed`,
so the forward raises `AttributeError: 'Linear' object has no attribute 'weight'`.
It reproduces with `device_map="cpu"` and on both transformers 4.57.1 and 5.16.1,
so it is not an offload or version issue.

The obvious fix -- `run_compressed=False`, decompressing to dense at load -- is
ruled out by arithmetic, not preference:

    thinker experts = 48 layers x 128 experts x 3 x 2048 x 768 = 29.0B params
    dense bf16                                                  = 58 GB

against 61 GB of host RAM with other services running.

So instead: keep the weights packed (15 GB) and dequantize per call. Only 8 of 128
experts are selected per token, so a token touches ~8/128 of each layer's weights.
Slow -- this is a Python loop over int4 unpacking -- but memory-bounded and
correct, which is what offline quality verification needs. It is emphatically not
a path to realtime; see docs/design.md section 7.

Format, read from the checkpoint rather than assumed:
    weight_packed  int32, 8 nibbles per word
    weight_scale   per group of 32 along the input dim
    weight_shape   original [out, in]
    symmetric (no zero point), num_bits=4, strategy "group"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_BITS = 4
GROUP_SIZE = 32


def dequantize_weight(
    mod: nn.Module,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Reconstruct a dense weight from `weight_packed` / `weight_scale` / `weight_shape`.

    `device` moves the *packed* tensors before unpacking, so the 8-shift unpack runs
    there. Packed weights are small -- one expert linear is 2048x96 int32 = 786 KB --
    so shipping them to the GPU and unpacking in parallel beats unpacking serially on
    the CPU, even counting the transfer.
    """
    from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
        unpack_from_int32,
    )

    shape = mod.weight_shape
    shape = tuple(int(x) for x in (shape.tolist() if torch.is_tensor(shape) else shape))
    packed = mod.weight_packed
    scale = mod.weight_scale
    if device is not None:
        packed = packed.to(device, non_blocking=True)
        scale = scale.to(device, non_blocking=True)
    q = unpack_from_int32(packed, NUM_BITS, shape)  # int8, [out, in]
    out_f, in_f = shape
    dtype = dtype or scale.dtype

    if scale.ndim == 2 and scale.shape[1] > 1:
        n_groups = scale.shape[1]
        if n_groups * GROUP_SIZE < in_f:
            raise ValueError(
                f"scale groups {n_groups} x {GROUP_SIZE} < in_features {in_f}"
            )
        w = q.to(dtype).view(out_f, n_groups, -1) * scale.to(dtype).unsqueeze(-1)
        w = w.view(out_f, -1)[:, :in_f]
    else:
        w = q.to(dtype) * scale.to(dtype).view(-1, 1)
    return w


def _streaming_forward(mod: nn.Module, compute_device=None):
    """Dequantize -> matmul -> free. Never materialises the full dense model."""

    def forward(x):
        dev = compute_device if compute_device is not None else x.device
        w = dequantize_weight(mod, dtype=x.dtype, device=dev)
        b = getattr(mod, "bias", None)
        if b is not None and b.device != dev:
            b = b.to(dev)
        return F.linear(x.to(dev), w, b)

    return forward


def patch_packed_linears(
    model: nn.Module, compute_device=None, verbose: bool = True
) -> int:
    """Install on-the-fly dequantizing forwards on every packed Linear.

    `compute_device` forces unpack+matmul onto that device regardless of where the
    packed weights are stored — the point of `place_thinker`.

    Returns the number of modules patched. Also strips any `quantized_forward`
    wrapper compressed-tensors may have attached, which is the thing that breaks.
    """
    n = 0
    for mod in model.modules():
        if hasattr(mod, "weight_packed") and hasattr(mod, "weight_scale"):
            if hasattr(mod, "_old_forward"):
                # accelerate/compressed-tensors wrapper -- drop it
                try:
                    del mod._old_forward
                except Exception:
                    pass
            mod.forward = _streaming_forward(mod, compute_device)
            n += 1
    if verbose:
        where = compute_device or "input device"
        print(f"patched {n} packed Linear modules (dequant+matmul on {where})")
    return n


def place_unpacked(root: nn.Module, device="cuda", verbose: bool = True) -> dict:
    """Move `root`'s *unpacked* params/buffers to `device`; leave packed ones behind.

    Call this on specific submodules, never on the whole Qwen3OmniMoe model — that
    sweeps in the talker (6.2 GB), code2wav, and both the audio and vision towers,
    which OOMs a 12 GB card before it gets anywhere useful.

    Note what is and is not packed in this checkpoint: 48 x (4 attention linears +
    128 experts x 3) = 18624 packed tensors, so **attention is quantized too**. The
    only dense weights in the thinker are embeddings and norms. Everything packed is
    shipped to the GPU per call by the patched forward instead.
    """
    moved = kept = 0
    packed_ids = set()
    for mod in root.modules():
        if hasattr(mod, "weight_packed"):
            packed_ids.update(id(p) for p in mod.parameters(recurse=False))
            packed_ids.update(id(b) for b in mod.buffers(recurse=False))

    for mod in root.modules():
        for name, p in list(mod.named_parameters(recurse=False)):
            if id(p) in packed_ids:
                kept += p.numel()
                continue
            setattr(mod, name, nn.Parameter(p.data.to(device), requires_grad=False))
            moved += p.numel()
        for name, b in list(mod.named_buffers(recurse=False)):
            if id(b) in packed_ids:
                kept += b.numel()
                continue
            mod.register_buffer(name, b.to(device), persistent=False)
            moved += b.numel()

    stats = {"moved_params": moved, "kept_on_host": kept, "device": str(device)}
    if verbose:
        print(f"  placed {moved/1e6:.0f}M params on {device}, "
              f"{kept/1e6:.0f}M packed elements left in host RAM")
    return stats


# Back-compat alias; prefer place_unpacked on a specific submodule.
place_thinker = place_unpacked


def sanity_check(model: nn.Module, k: int = 3) -> list[dict]:
    """Dequantize a few weights and report statistics.

    A correct int4 symmetric dequantization gives roughly zero-mean weights with
    at most 16 distinct values per group. Garbage unpacking usually shows up as a
    non-zero mean or a wildly wrong scale.
    """
    out = []
    seen = 0
    for name, mod in model.named_modules():
        if not (hasattr(mod, "weight_packed") and hasattr(mod, "weight_scale")):
            continue
        w = dequantize_weight(mod).float()
        shape = tuple(int(x) for x in mod.weight_shape.tolist())
        out.append({
            "name": name,
            "shape": tuple(w.shape),
            "expected_shape": shape,
            "shape_ok": tuple(w.shape) == shape,
            "mean": float(w.mean()),
            "std": float(w.std()),
            "absmax": float(w.abs().max()),
            "n_unique_first_group": int(w[0, :GROUP_SIZE].unique().numel()),
        })
        seen += 1
        if seen >= k:
            break
    return out
