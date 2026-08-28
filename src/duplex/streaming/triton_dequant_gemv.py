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
        K, R, NW,
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

            xv = tl.load(X + cols, mask=cmask, other=0.0).to(tl.float32)  # [NW_BLK, 8]

            prod = q.to(tl.float32) * sc * xv[None, :, :]
            acc += tl.sum(tl.sum(prod, axis=2), axis=1)

        tl.store(OUT + pid_k * s_out_k + rows * s_out_r, acc, mask=rmask)


def dequant_gemv(x, packed, scale, sel, group=32, block_r=64, nw_blk=8):
    """y[j, r] = sum_c W[sel[j], r, c] * x[c],  W dequantized from int4 on the fly.

    x       [K]              activations
    packed  [E, R, NW] int32 NW = ceil(K/8)
    scale   [E, R, NG]       NG = ceil(K/group)
    sel     [k] int32        selected experts
    returns [k, R] float32
    """
    if not HAVE_TRITON:
        raise RuntimeError("triton not available")
    E, R, NW = packed.shape
    K = x.numel()
    k = sel.numel()
    out = torch.empty(k, R, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(R, block_r), k)
    _dq_gemv_kernel[grid](
        x, packed, scale, sel.to(torch.int32), out,
        K, R, NW,
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
