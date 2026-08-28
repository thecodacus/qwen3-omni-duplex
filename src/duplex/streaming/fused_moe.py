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

    def __init__(self, block: nn.Module, chunk: int = 8, q8: bool = False):
        """`q8` stores the stacked experts as per-output-channel symmetric int8.

        The talker ships bf16 in this checkpoint and is 6.1 GB — the largest single
        item on the card, and the reason the full model leaves no room for
        activations. int8 halves it to ~3.0 GB. Far less aggressive than the int4
        already running on the thinker, and dequantization is one multiply.
        """
        super().__init__()
        self.chunk = chunk
        self.q8 = q8
        experts = block.experts
        e0 = experts[0]
        self.num_experts = len(experts)
        self.top_k = block.top_k
        self.norm_topk_prob = getattr(block, "norm_topk_prob", True)
        self.gate = block.gate
        # The TALKER's block has a shared expert whose output is added to every
        # token on top of the routed experts:
        #
        #   shared = sigmoid(shared_expert_gate(x)) * shared_expert(x)
        #   out    = routed + shared
        #
        # Dropping it measured 49% relative error against the original block and
        # is what made the generated speech tear. The thinker's block has no
        # shared expert, which is why its fused path was fine and the text stayed
        # coherent while the audio did not.
        self.shared_expert = getattr(block, "shared_expert", None)
        self.shared_expert_gate = getattr(block, "shared_expert_gate", None)

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
        if q8:
            # symmetric per-output-channel: scale over the input dim
            gu_s = gate_up.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
            dn_s = down.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
            gu_q = (gate_up / gu_s).round().clamp_(-127, 127).to(torch.int8)
            dn_q = (down / dn_s).round().clamp_(-127, 127).to(torch.int8)
            del gate_up, down
            self.register_buffer("gate_up", gu_q, persistent=False)
            self.register_buffer("down", dn_q, persistent=False)
            self.register_buffer("gate_up_s", gu_s.to(dtype), persistent=False)
            self.register_buffer("down_s", dn_s.to(dtype), persistent=False)
        else:
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

        T, k = sel.shape

        # Chunk over tokens. The gather is [T, k, 2I, H] + [T, k, H, I], which for
        # a prefill of any length is hundreds of MB — measured OOM at 306 MB for
        # `self.down[sel]` alone. Chunking bounds it while keeping the batched
        # matmul that made this fast in the first place.
        chunk = self.chunk if T > self.chunk else T
        outs = []
        for i in range(0, T, chunk):
            xs, ss, ws = x[i:i + chunk], sel[i:i + chunk], w[i:i + chunk]
            t = xs.shape[0]
            gu = self.gate_up[ss]                                      # [t, k, 2I, H]
            if self.q8:
                gu = gu.to(xs.dtype) * self.gate_up_s[ss]
            h = torch.matmul(gu, xs.view(t, 1, shape[-1], 1)).squeeze(-1)
            del gu
            g, u = h.split(self.inter, dim=-1)
            act = (F.silu(g) * u).unsqueeze(-1)                        # [t, k, I, 1]
            dn = self.down[ss]                                         # [t, k, H, I]
            if self.q8:
                dn = dn.to(xs.dtype) * self.down_s[ss]
            y = torch.matmul(dn, act).squeeze(-1)                      # [t, k, H]
            del dn
            outs.append((y * ws.unsqueeze(-1)).sum(dim=1))
        out = torch.cat(outs, dim=0)

        if self.shared_expert is not None:
            shared = self.shared_expert(x)
            if self.shared_expert_gate is not None:
                shared = F.sigmoid(self.shared_expert_gate(x)) * shared
            out = out + shared
        return out.view(shape)


def fuse_moe_blocks(model: nn.Module, q8: bool = False, verbose: bool = True) -> int:
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
        fused = FusedMoE(child, q8=q8)
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
        print(f"fused {n} MoE blocks (gather+bmm{', int8' if q8 else ''})")
    return n
