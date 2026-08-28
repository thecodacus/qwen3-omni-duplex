"""Fuse Qwen3-Omni's MoE blocks from a Python expert loop into gather + bmm.

transformers ships the Mixtral-style implementation:

    for expert_idx in expert_hit:            # top_k iterations, in Python
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        ...

At batch-1 decode that is `top_k` iterations per layer, each with a `torch.where`
and three small matmuls: ~720 kernel launches per Talker token across 20 layers.
Measured cost 41.95 ms/token, against ~1.05 ms if it were bandwidth-bound on a
3060 (190M active params -> 380 MB bf16 @ ~360 GB/s). Roughly 40x overhead.

This stacks each layer's experts into contiguous tensors once, then does:

    w = gate_up[selected]        # [k, 2I, H]  gather
    h = bmm(w, x)                # one batched gemv
    y = bmm(silu(g)*u, down[selected])

which is the "gather+gemv strategy for MoE kernels instead of the standard grouped
gemm" Thinking Machines described -- batch-1 decode has no batch to build a GEMM
from, so a gather is the right primitive.

Stacking is done per layer and the ModuleList freed immediately: a whole-model
stack would transiently double the 6 GB of talker experts and OOM a 12 GB card.
Only unquantized blocks are handled here; the packed int4 thinker needs the same
treatment over `weight_packed`, which is a separate job.
"""

from __future__ import annotations

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedMoE(nn.Module):
    """Drop-in replacement for a SparseMoeBlock, with experts stacked."""

    def __init__(self, block: nn.Module):
        super().__init__()
        experts = block.experts
        e0 = experts[0]
        self.num_experts = len(experts)
        self.top_k = block.top_k
        self.norm_topk_prob = getattr(block, "norm_topk_prob", True)
        self.gate = block.gate

        dtype = e0.gate_proj.weight.dtype
        dev = e0.gate_proj.weight.device
        inter, hidden = e0.gate_proj.weight.shape
        self.inter = inter

        gate_up = torch.empty(self.num_experts, 2 * inter, hidden, dtype=dtype, device=dev)
        down = torch.empty(self.num_experts, hidden, inter, dtype=dtype, device=dev)
        for i, e in enumerate(experts):
            gate_up[i, :inter] = e.gate_proj.weight
            gate_up[i, inter:] = e.up_proj.weight
            down[i] = e.down_proj.weight
        self.register_buffer("gate_up", gate_up, persistent=False)
        self.register_buffer("down", down, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.view(-1, shape[-1])                      # [T, H]

        router_logits = self.gate(x)
        w = F.softmax(router_logits, dim=1, dtype=torch.float)
        w, sel = torch.topk(w, self.top_k, dim=-1)                 # [T, k]
        if self.norm_topk_prob:
            w = w / w.sum(dim=-1, keepdim=True)
        w = w.to(x.dtype)

        gu = self.gate_up[sel]                                     # [T, k, 2I, H]
        dn = self.down[sel]                                        # [T, k, H, I]
        T, k = sel.shape

        h = torch.matmul(gu, x.view(T, 1, shape[-1], 1)).squeeze(-1)   # [T, k, 2I]
        g, u = h.split(self.inter, dim=-1)
        act = (F.silu(g) * u).unsqueeze(-1)                            # [T, k, I, 1]
        y = torch.matmul(dn, act).squeeze(-1)                          # [T, k, H]
        y = (y * w.unsqueeze(-1)).sum(dim=1)                           # [T, H]
        return y.view(shape)


def fuse_moe_blocks(model: nn.Module, verbose: bool = True) -> int:
    """Replace every unquantized SparseMoeBlock in `model` with a FusedMoE.

    Skips blocks whose experts hold packed weights -- those need an int4-aware
    gather and are not handled here.
    """
    # Record only (parent, name). Holding the child modules here would keep every
    # original expert alive while the stacked copies accumulate — 6.2 GB of talker
    # experts plus 6 GB of stacks, which OOMs a 12 GB card.
    sites = []
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if type(child).__name__.endswith("SparseMoeBlock") and hasattr(child, "experts"):
                e0 = child.experts[0]
                gp = getattr(e0, "gate_proj", None)
                if gp is None or hasattr(gp, "weight_packed") or not hasattr(gp, "weight"):
                    continue  # packed int4 — separate job
                sites.append((parent, name))

    n = 0
    for parent, name in sites:
        child = getattr(parent, name)
        fused = FusedMoE(child)
        setattr(parent, name, fused)
        # Drop the originals now, per layer, so peak stays at
        # (all experts) + (one stacked layer) rather than double.
        child.experts = None
        del child
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        n += 1
    if verbose:
        print(f"fused {n} MoE blocks (gather+bmm, was a Python expert loop)")
    return n
