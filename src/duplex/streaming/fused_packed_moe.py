"""Fused int4 MoE: gather packed experts, unpack on the GPU, batched matmul.

The bf16 fusion in `fused_moe.py` took the Talker from 41.95 to 17.36 ms/token by
replacing transformers' Python expert loop with a gather + bmm. The Thinker needs
the same thing, but its experts are packed int4, so the gather has to unpack too.

Layout in this checkpoint (symmetric int4, group_size 32, no zero point):

    gate_proj.weight_shape   [768, 2048]          out=inter, in=hidden
    gate_proj.weight_packed  [768, 256] int32     2048/8 nibbles per row
    gate_proj.weight_scale   [768, 64]  fp16      2048/32 groups per row

Stacked per layer into [E, 2*inter, hidden/8] and [E, hidden, inter/8], plus their
scales. Per token the forward gathers `top_k` experts, unpacks them on the GPU
(8 parallel shifts, not a serial CPU loop), scales by group, and does one bmm.

Memory: `Thinker[0:24]` packed is ~8.2 GB (7.25 GB nibbles + 0.9 GB fp16 scales),
so on a 12 GB card it fits alone but NOT alongside the bf16 talker's 6.1 GB. That
is a real constraint on the full clock path, not an implementation detail — see
docs/design.md section 10.
"""

from __future__ import annotations

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

from duplex.streaming.triton_dequant_gemv import dequant_gemv, HAVE_TRITON

NUM_BITS = 4
PACK = 32 // NUM_BITS      # 8 nibbles per int32
GROUP = 32


def unpack4(packed: torch.Tensor, out_cols: int) -> torch.Tensor:
    """Unpack int32-packed signed int4 along the last dim. Arbitrary leading dims.

    Mirrors compressed_tensors' `unpack_from_int32`, which writes
    `unpacked[:, i::PACK] = value >> (NUM_BITS*i)`: column `i + PACK*m` comes from
    word `m` shifted by `4i`. Flattening a `[..., C, PACK]` stack gives position
    `PACK*m + i`, which is the same index.
    """
    shifts = torch.arange(PACK, device=packed.device, dtype=torch.int32) * NUM_BITS
    u = (packed.unsqueeze(-1) >> shifts) & 0xF          # [..., C, PACK]
    u = u.flatten(-2)                                    # [..., C*PACK]
    return (u[..., :out_cols] - (1 << (NUM_BITS - 1)))   # signed, [-8, 7]


def _stack_packed(linears, dtype, device):
    """Stack a list of packed Linears into (packed, scale, out_features, in_features)."""
    l0 = linears[0]
    shape0 = tuple(int(x) for x in l0.weight_shape.tolist())
    out_f, in_f = shape0
    E = len(linears)
    packed = torch.empty(E, out_f, l0.weight_packed.shape[1],
                         dtype=torch.int32, device=device)
    scale = torch.empty(E, out_f, l0.weight_scale.shape[1], dtype=dtype, device=device)
    for i, l in enumerate(linears):
        packed[i] = l.weight_packed.to(device)
        scale[i] = l.weight_scale.to(device, dtype)
    return packed, scale, out_f, in_f


class FusedPackedMoE(nn.Module):
    """SparseMoeBlock replacement for int4-packed experts."""

    def __init__(self, block: nn.Module, dtype=torch.bfloat16, device="cuda",
                 host_resident: bool = False, cpu_compute: bool = False):
        """`host_resident` keeps the packed experts in pinned host RAM and copies
        only the selected ones to a GPU staging buffer per token.

        Measured cost of that choice, per layer: 0.639 ms resident vs 1.575 ms
        streamed, against 0.316 GB vs 0.022 GB. Neither extreme fits a 12 GB card
        alongside the talker (24 pinned = 8.4 GB) nor the 80 ms budget (24 streamed
        = 37.8 ms), so layers are split between the two. See docs/design.md §13.
        """
        super().__init__()
        self.host_resident = host_resident
        # llama.cpp's --n-cpu-moe: compute non-resident experts ON the CPU and send
        # only activations over PCIe. Weight streaming moves 21.3 MB/layer; an
        # activation is 4 KB each way -- ~2600x less traffic, no per-expert copies
        # (1280 launches/token), no sel.tolist() sync per layer.
        #
        # OFF by default. The architecture is right but the implementation is not
        # competitive here: llama.cpp wins this with hand-written AVX2/AVX512 int4
        # kernels, and dequantising with generic torch ops on the CPU was far slower
        # than streaming the weights. Worth revisiting with a real CPU kernel.
        self.cpu_compute = cpu_compute and host_resident
        experts = block.experts
        self.num_experts = len(experts)
        self.top_k = block.top_k
        self.norm_topk_prob = getattr(block, "norm_topk_prob", True)
        self.gate = block.gate.to(device=device, dtype=dtype)

        # Stack straight onto the destination. Building on the GPU and then moving
        # to host would spike device memory for every host-resident layer.
        build = "cpu" if host_resident else device
        gp, gs, inter, hidden = _stack_packed([e.gate_proj for e in experts], dtype, build)
        up, us, _, _ = _stack_packed([e.up_proj for e in experts], dtype, build)
        dp, ds, hid_out, inter_in = _stack_packed([e.down_proj for e in experts], dtype, build)

        # gate and up share a shape, so concatenate along out_features for one gather
        gu_p, gu_s = torch.cat([gp, up], dim=1), torch.cat([gs, us], dim=1)
        del gp, up, gs, us

        if cpu_compute and host_resident:
            # Weights live on the CPU and stay there; nothing is staged to the GPU.
            self._h_gu_p, self._h_gu_s = gu_p, gu_s
            self._h_dn_p, self._h_dn_s = dp, ds
            del gu_p, gu_s, dp, ds
        elif host_resident:
            # pinned host storage + preallocated device staging for top_k experts
            self._h_gu_p = gu_p.pin_memory()
            self._h_gu_s = gu_s.pin_memory()
            self._h_dn_p = dp.pin_memory()
            self._h_dn_s = ds.pin_memory()
            k = self.top_k
            self.register_buffer("gu_packed", torch.empty(k, *gu_p.shape[1:],
                                 dtype=gu_p.dtype, device=device), persistent=False)
            self.register_buffer("gu_scale", torch.empty(k, *gu_s.shape[1:],
                                 dtype=gu_s.dtype, device=device), persistent=False)
            self.register_buffer("down_packed", torch.empty(k, *dp.shape[1:],
                                 dtype=dp.dtype, device=device), persistent=False)
            self.register_buffer("down_scale", torch.empty(k, *ds.shape[1:],
                                 dtype=ds.dtype, device=device), persistent=False)
            self.register_buffer("_staged_idx",
                                 torch.arange(k, dtype=torch.int32, device=device),
                                 persistent=False)
            del gu_p, gu_s, dp, ds
        else:
            self.register_buffer("gu_packed", gu_p, persistent=False)
            self.register_buffer("gu_scale", gu_s, persistent=False)
            self.register_buffer("down_packed", dp, persistent=False)
            self.register_buffer("down_scale", ds, persistent=False)
            del gu_p, gu_s, dp, ds

        self.inter = inter
        self.hidden = hidden
        self.hid_out = hid_out
        self.inter_in = inter_in

    @staticmethod
    def _dequant(packed, scale, cols):
        """[..., R, C] int32 + [..., R, G] scale -> [..., R, cols] dtype."""
        q = unpack4(packed, cols)                                  # [..., R, cols] int
        lead = q.shape[:-1]
        q = q.view(*lead, scale.shape[-1], -1).to(scale.dtype)     # [..., R, G, cols/G]
        return (q * scale.unsqueeze(-1)).view(*lead, cols)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.view(-1, shape[-1])                      # [T, H]

        w = F.softmax(self.gate(x), dim=1, dtype=torch.float)
        w, sel = torch.topk(w, self.top_k, dim=-1)
        if self.norm_topk_prob:
            w = w / w.sum(dim=-1, keepdim=True)
        w = w.to(x.dtype)
        T, k = sel.shape

        if self.cpu_compute:
            # Route on whichever device the gate ended up on (model.to(dev) moves it
            # back to the GPU regardless of what __init__ did), then send only the
            # activation to the CPU. That is 4 KB each way per layer against
            # 21.3 MB of weights.
            dev_in = x.device
            xc = x.to("cpu", torch.float32)
            wc, selc = w.to("cpu", torch.float32), sel.to("cpu")
            outs = []
            for t in range(T):
                st = selc[t]
                gu = self._dequant(self._h_gu_p[st], self._h_gu_s[st].float(), self.hidden)
                h = torch.matmul(gu, xc[t].view(-1, 1)).squeeze(-1)      # [k, 2I]
                g, u = h.split(self.inter, dim=-1)
                act = (F.silu(g) * u).unsqueeze(-1)                      # [k, I, 1]
                dn = self._dequant(self._h_dn_p[st], self._h_dn_s[st].float(), self.inter_in)
                y = torch.matmul(dn, act).squeeze(-1)                    # [k, H]
                outs.append((y * wc[t].unsqueeze(-1)).sum(dim=0))
            return torch.stack(outs).to(dev_in, hidden_states.dtype).view(shape)

        if HAVE_TRITON:
            # One token at a time. Prefill passes T>1, and the torch path below
            # would materialise [T, k, 2I, H] dense weights for it -- measured at
            # 2.91 GiB for a single layer. Looping never allocates dense weights.
            #
            # Host-resident layers must stage INSIDE this loop: each token selects
            # its own experts, and staging once for token 0 leaves the kernel
            # indexing an 8-slot buffer with expert ids up to 127 (illegal access).
            outs = []
            for t in range(T):
                st = sel[t]
                if self.host_resident:
                    for j, e in enumerate(st.tolist()):
                        self.gu_packed[j].copy_(self._h_gu_p[e], non_blocking=True)
                        self.gu_scale[j].copy_(self._h_gu_s[e], non_blocking=True)
                        self.down_packed[j].copy_(self._h_dn_p[e], non_blocking=True)
                        self.down_scale[j].copy_(self._h_dn_s[e], non_blocking=True)
                    st = self._staged_idx
                gu = dequant_gemv(x[t], self.gu_packed, self.gu_scale, st)   # [k, 2I]
                g, u = gu.split(self.inter, dim=-1)
                act = (F.silu(g) * u).to(x.dtype)                            # [k, I]
                y = dequant_gemv(act, self.down_packed, self.down_scale, st) # [k, H]
                outs.append((y.to(x.dtype) * w[t].unsqueeze(-1)).sum(dim=0))
            return torch.stack(outs).view(shape)

        gu = self._dequant(self.gu_packed[sel], self.gu_scale[sel], self.hidden)
        h = torch.matmul(gu, x.view(T, 1, shape[-1], 1)).squeeze(-1)   # [T, k, 2I]
        del gu
        g, u = h.split(self.inter, dim=-1)
        act = (F.silu(g) * u).unsqueeze(-1)                            # [T, k, I, 1]

        dn = self._dequant(self.down_packed[sel], self.down_scale[sel], self.inter_in)
        y = torch.matmul(dn, act).squeeze(-1)                          # [T, k, H]
        del dn
        y = (y * w.unsqueeze(-1)).sum(dim=1)
        return y.view(shape)


def fuse_packed_moe_blocks(model, dtype=torch.bfloat16, device="cuda",
                           n_pinned: int | None = None, cpu_compute: bool = False,
                           verbose: bool = True) -> int:
    """Replace every packed SparseMoeBlock with a FusedPackedMoE.

    Records only (parent, name) — holding the child modules keeps every original
    expert alive while the stacks accumulate, which doubles peak memory.
    """
    sites = []
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if type(child).__name__.endswith("SparseMoeBlock") and hasattr(child, "experts"):
                gp = getattr(child.experts[0], "gate_proj", None)
                if gp is not None and hasattr(gp, "weight_packed"):
                    sites.append((parent, name))

    n = 0
    for parent, name in sites:
        child = getattr(parent, name)
        # partial residency: the first n_pinned blocks live on the GPU, the rest
        # stream their selected experts from pinned host RAM per token.
        host = n_pinned is not None and n >= n_pinned
        setattr(parent, name, FusedPackedMoE(child, dtype, device, host_resident=host,
                                            cpu_compute=cpu_compute))
        child.experts = None
        del child
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        n += 1
        if verbose and n % 6 == 0:
            used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            print(f"    fused {n}/{len(sites)} packed blocks, {used:.2f} GB", flush=True)

    if verbose:
        if n_pinned is None:
            print(f"fused {n} packed MoE blocks, all GPU-resident")
        else:
            mode = "computed on CPU" if cpu_compute else "streamed from host"
            print(f"fused {n} packed MoE blocks: {min(n_pinned, n)} pinned, "
                  f"{max(0, n - n_pinned)} {mode}")
    return n
