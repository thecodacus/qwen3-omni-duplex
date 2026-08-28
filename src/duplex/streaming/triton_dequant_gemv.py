"""Triton dequant-GEMV: unpack int4 inside the matmul, never materialise dense weights.

`fused_packed_moe.py` gathers packed experts, unpacks them with torch ops, then does
a bmm. That costs 4.078 ms per Thinker MoE layer because it writes 75 MB of dense
bf16 weights and runs five elementwise passes over them — to avoid reading 21 MB.

This does the unpack in registers during accumulation. Each program owns a block of
output rows for one selected expert, streams the packed words, expands nibbles,
applies the group scale, and multiplies into an accumulator. Dense weights never
touch memory.

Packing layout (must match `unpack_from_int32`): unpacked column `j = i + 8*m` comes
from word `m` shifted by `4*i`, so eight consecutive output columns come from one
int32 word. Scales are per group of 32 columns.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _dq_gemv_kernel(
        X, PK, SC, SEL, OUT,
        K, R, NW, s_x,
        s_pk_e, s_pk_r, s_pk_c,
        s_sc_e, s_sc_r, s_sc_g,
        s_out_k, s_out_r,
        GROUP: tl.constexpr,
        BLOCK_R: tl.constexpr,
        NW_BLK: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_k = tl.program_id(1)
        e = tl.load(SEL + pid_k).to(tl.int32)

        rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = rows < R
        acc = tl.zeros((BLOCK_R,), dtype=tl.float32)

        shifts = (tl.arange(0, 8) * 4).to(tl.int32)

        for w0 in range(0, NW, NW_BLK):
            ws = w0 + tl.arange(0, NW_BLK)
            wmask = ws < NW

            p = tl.load(
                PK + e * s_pk_e + rows[:, None] * s_pk_r + ws[None, :] * s_pk_c,
                mask=rmask[:, None] & wmask[None, :], other=0,
            ).to(tl.int32)                                   # [BLOCK_R, NW_BLK]

            # expand each word into its 8 nibbles -> [BLOCK_R, NW_BLK, 8]
            q = ((p[:, :, None] >> shifts[None, None, :]) & 0xF) - 8

            cols = ws[:, None] * 8 + tl.arange(0, 8)[None, :]   # [NW_BLK, 8]
            cmask = cols < K

            g = cols // GROUP
            sc = tl.load(
                SC + e * s_sc_e + rows[:, None, None] * s_sc_r + g[None, :, :] * s_sc_g,
                mask=rmask[:, None, None] & cmask[None, :, :], other=0.0,
            ).to(tl.float32)

            # s_x = 0: every expert reads the same activation vector (gate_up).
            # s_x = K: each expert has its own (down projection), so one launch
            # replaces the per-expert loop.
            xv = tl.load(X + pid_k * s_x + cols, mask=cmask, other=0.0).to(tl.float32)

            prod = q.to(tl.float32) * sc * xv[None, :, :]
            acc += tl.sum(tl.sum(prod, axis=2), axis=1)

        tl.store(OUT + pid_k * s_out_k + rows * s_out_r, acc, mask=rmask)


def dequant_gemv(x, packed, scale, sel, group=32, block_r=64, nw_blk=8):
    """y[j, r] = sum_c W[sel[j], r, c] * x[j, c],  W dequantized from int4 on the fly.

    x       [K] or [k, K]    activations; 1-D is shared across experts
    packed  [E, R, NW] int32 NW = ceil(K/8)
    scale   [E, R, NG]       NG = ceil(K/group)
    sel     [k] int32        selected experts
    returns [k, R] float32
    """
    if not HAVE_TRITON:
        raise RuntimeError("triton not available")
    E, R, NW = packed.shape
    k = sel.numel()
    if x.dim() == 1:
        K, s_x = x.numel(), 0
    else:
        K, s_x = x.shape[-1], x.stride(0)
    x = x.contiguous()
    out = torch.empty(k, R, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(R, block_r), k)
    _dq_gemv_kernel[grid](
        x, packed, scale, sel.to(torch.int32), out,
        K, R, NW, s_x,
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        out.stride(0), out.stride(1),
        GROUP=group, BLOCK_R=block_r, NW_BLK=nw_blk,
    )
    return out


def reference_gemv(x, packed, scale, sel, group=32):
    """Same result via the verified torch path, for correctness checking."""
    from duplex.streaming.fused_packed_moe import FusedPackedMoE

    K = x.numel()
    w = FusedPackedMoE._dequant(packed[sel], scale[sel].to(torch.float32), K)
    return torch.matmul(w, x.to(torch.float32).view(-1, 1)).squeeze(-1)


class TritonPackedLinear(torch.nn.Module):
    """A packed int4 Linear whose forward uses the dequant-GEMV kernel.

    Attention was originally left on the torch dequant path with the note "4
    linears/layer, ~7 MB packed each -- not worth a kernel yet". That sized it by
    *packed* bytes; the cost is set by *dequantized* work. q and o are 4096x2048, so
    attention is 452M params/token materialized to dense across 96 calls, and it
    measured at ~215 ms of the Thinker's 230 ms.

    A plain Linear is the E=1 case of the expert gather, so the same kernel serves.
    """

    def __init__(self, mod, device="cuda"):
        super().__init__()
        shape = tuple(int(x) for x in mod.weight_shape.tolist())
        self.out_features, self.in_features = shape
        self.register_buffer("packed", mod.weight_packed.to(device).unsqueeze(0),
                             persistent=False)
        self.register_buffer("scale", mod.weight_scale.to(device).float().unsqueeze(0),
                             persistent=False)
        b = getattr(mod, "bias", None)
        self.register_buffer("bias", b.to(device) if b is not None else None,
                             persistent=False)
        self.register_buffer("_sel", torch.zeros(1, dtype=torch.int32, device=device),
                             persistent=False)

    def forward(self, x):
        lead, K = x.shape[:-1], x.shape[-1]
        flat = x.reshape(-1, K)
        outs = [dequant_gemv(flat[i], self.packed, self.scale, self._sel)[0]
                for i in range(flat.shape[0])]
        y = torch.stack(outs).to(x.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y.view(*lead, self.out_features)


def patch_attention_to_triton(model, device="cuda", verbose=True) -> int:
    """Swap packed attention projections onto the Triton kernel."""
    n = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if hasattr(child, "weight_packed") and hasattr(child, "weight_scale"):
                setattr(parent, name, TritonPackedLinear(child, device))
                n += 1
    if verbose:
        print(f"moved {n} packed linears onto the Triton kernel")
    return n
