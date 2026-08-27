"""
Qwen3-Omni full-duplex feasibility: can the pinned clock path hold an 80ms frame deadline?

Geometry mirrors Qwen/Qwen3-Omni-30B-A3B-Instruct config.json exactly:
  Thinker[0:24]  hidden 2048, MoE 128e/8a, moe_intermediate 768, GQA 32q/4kv
  Talker         hidden 1024, MoE 128e/6a, moe_intermediate 384, 20 layers
  CodePredictor  hidden 1024, dense, 5 layers, 16 code groups (MTP)

Weights are random. This measures memory traffic + kernel overhead, which is what
sets the deadline; what the weights contain is irrelevant to frame time.

Two deliberate pessimisms vs. the real deployment:
  - bf16 experts (9.4 MB) instead of Q4 (2.53 MB) -> 3.7x the per-frame byte traffic
  - uniform-random routing -> worst-case expert locality
If it passes here it passes for real.

Expert pool is reduced to --experts (default 32) so the resident set fits 12GB.
Per-frame traffic is unaffected: only `top_k` experts per layer are ever read.
"""

import argparse, json, statistics, threading, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class MoEBlock(nn.Module):
    def __init__(self, hidden, inter, n_experts, top_k, dtype, dev):
        super().__init__()
        self.top_k, self.n_experts, self.inter, self.hidden = top_k, n_experts, inter, hidden
        self.router = nn.Linear(hidden, n_experts, bias=False, dtype=dtype, device=dev)
        # packed [E, 3*inter, hidden] so a gather is one contiguous read per expert
        self.w = nn.Parameter(torch.randn(n_experts, 3 * inter, hidden, dtype=dtype, device=dev) * 0.02)

    def forward(self, x):  # x: [T, hidden]
        logits = self.router(x)
        w_sel, idx = logits.softmax(-1).topk(self.top_k, dim=-1)  # [T,k], [T,k]
        out = torch.zeros_like(x)
        I = self.inter
        for t in range(x.shape[0]):
            we = self.w[idx[t]]                        # gather: [k, 3I, hidden] — the PCIe/VRAM read
            gu = torch.matmul(we[:, : 2 * I], x[t])    # gemv:   [k, 2I]
            g, u = gu.split(I, dim=-1)
            act = F.silu(g) * u                        # [k, I]
            down = we[:, 2 * I :]                      # [k, I, hidden]
            y = torch.bmm(act.unsqueeze(1), down).squeeze(1)  # [k, hidden]
            out[t] = (y * w_sel[t].unsqueeze(-1)).sum(0)
        return out


class Attn(nn.Module):
    def __init__(self, hidden, n_q, n_kv, dtype, dev):
        super().__init__()
        self.hd = hidden // n_q
        self.n_q, self.n_kv = n_q, n_kv
        self.q = nn.Linear(hidden, n_q * self.hd, bias=False, dtype=dtype, device=dev)
        self.k = nn.Linear(hidden, n_kv * self.hd, bias=False, dtype=dtype, device=dev)
        self.v = nn.Linear(hidden, n_kv * self.hd, bias=False, dtype=dtype, device=dev)
        self.o = nn.Linear(n_q * self.hd, hidden, bias=False, dtype=dtype, device=dev)

    def forward(self, x, kv):
        T = x.shape[0]
        q = self.q(x).view(T, self.n_q, self.hd).transpose(0, 1)
        k = self.k(x).view(T, self.n_kv, self.hd).transpose(0, 1)
        v = self.v(x).view(T, self.n_kv, self.hd).transpose(0, 1)
        kc, vc = kv
        kc = torch.cat([kc, k], dim=1) if kc is not None else k
        vc = torch.cat([vc, v], dim=1) if vc is not None else v
        rep = self.n_q // self.n_kv
        a = F.scaled_dot_product_attention(
            q.unsqueeze(0), kc.repeat_interleave(rep, 0).unsqueeze(0), vc.repeat_interleave(rep, 0).unsqueeze(0)
        )
        return self.o(a.squeeze(0).transpose(0, 1).reshape(T, -1)), (kc, vc)


class Layer(nn.Module):
    def __init__(self, hidden, inter, n_e, k, n_q, n_kv, dtype, dev, dense=False):
        super().__init__()
        self.n1 = nn.RMSNorm(hidden, dtype=dtype, device=dev)
        self.n2 = nn.RMSNorm(hidden, dtype=dtype, device=dev)
        self.attn = Attn(hidden, n_q, n_kv, dtype, dev)
        self.ffn = (
            nn.Sequential(nn.Linear(hidden, inter, bias=False, dtype=dtype, device=dev), nn.SiLU(),
                          nn.Linear(inter, hidden, bias=False, dtype=dtype, device=dev))
            if dense else MoEBlock(hidden, inter, n_e, k, dtype, dev)
        )

    def forward(self, x, kv):
        h, kv = self.attn(self.n1(x), kv)
        x = x + h
        return x + self.ffn(self.n2(x)), kv


class Stack(nn.Module):
    def __init__(self, n_layers, **kw):
        super().__init__()
        self.layers = nn.ModuleList([Layer(**kw) for _ in range(n_layers)])

    def forward(self, x, cache):
        for i, l in enumerate(self.layers):
            x, cache[i] = l(x, cache[i])
        return x


def new_cache(n):
    return [(None, None) for _ in range(n)]


def pcie_hammer(stop, dev, mb, bw_log):
    """Simulate the offloaded Thinker[24:48] streaming experts H2D over PCIe."""
    src = torch.empty(mb * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cpu").pin_memory()
    dst = torch.empty_like(src, device=dev)
    stream = torch.cuda.Stream(device=dev)
    n = 0
    t0 = time.perf_counter()
    with torch.cuda.stream(stream):
        while not stop.is_set():
            dst.copy_(src, non_blocking=True)
            stream.synchronize()
            n += 1
    dt = time.perf_counter() - t0
    bw_log.append(n * mb / 1024 / dt if dt > 0 else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experts", type=int, default=32, help="resident pool; per-frame traffic is top_k-bound")
    p.add_argument("--positions", type=int, default=1, help="thinker positions per 80ms frame")
    p.add_argument("--frames", type=int, default=1500)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--contend", action="store_true", help="run a concurrent PCIe streamer")
    p.add_argument("--contend-mb", type=int, default=256)
    p.add_argument("--ctx", type=int, default=250, help="prefilled KV frames (250 = 20s @12.5Hz)")
    p.add_argument("--json", type=str, default="")
    a = p.parse_args()

    if a.experts < 8:
        raise SystemExit(f"--experts must be >= 8 (thinker routes top-8); got {a.experts}")

    dev, dt = "cuda", torch.bfloat16
    torch.manual_seed(0)

    thinker = Stack(24, hidden=2048, inter=768, n_e=a.experts, k=8, n_q=32, n_kv=4, dtype=dt, dev=dev)
    talker = Stack(20, hidden=1024, inter=384, n_e=a.experts, k=6, n_q=16, n_kv=2, dtype=dt, dev=dev)
    mtp = Stack(5, hidden=1024, inter=2048, n_e=1, k=1, n_q=16, n_kv=2, dtype=dt, dev=dev, dense=True)
    proj = nn.Linear(2048, 1024, bias=False, dtype=dt, device=dev)
    heads = nn.Linear(1024, 2048, bias=False, dtype=dt, device=dev)  # codec head, 2048 codebook

    for m in (thinker, talker, mtp, proj, heads):
        m.requires_grad_(False).eval()

    resident = torch.cuda.memory_allocated(dev) / 1024**3
    print(f"resident weights: {resident:.2f} GB  (experts/layer={a.experts}, bf16)")
    print(f"per-frame expert traffic: thinker {24*8*3*2048*768*2/1024**2:.0f} MB + "
          f"talker {20*6*3*1024*384*2/1024**2:.0f} MB  (bf16; Q4 would be /3.6)")

    tc, kc, mc = new_cache(24), new_cache(20), new_cache(5)

    @torch.inference_mode()
    def frame():
        x = torch.randn(a.positions, 2048, dtype=dt, device=dev)
        h = thinker(x, tc)                       # Thinker[0:24] -> layer-24 hidden
        t = talker(proj(h[-1:]), kc)             # Talker: codebook 0
        c = heads(t)
        mc[:] = new_cache(5)                     # MTP is fixed-step per frame, not streaming
        for _ in range(15):                      # 15 residual codebooks
            c = heads(mtp(t, mc))
        return c

    stop, bw = threading.Event(), []
    if a.contend:
        threading.Thread(target=pcie_hammer, args=(stop, dev, a.contend_mb, bw), daemon=True).start()
        time.sleep(2.0)

    for _ in range(a.warmup):
        frame()
    torch.cuda.synchronize()

    # pre-roll KV so we measure at realistic context depth
    for _ in range(a.ctx):
        frame()
    torch.cuda.synchronize()

    times = []
    for _ in range(a.frames):
        t0 = time.perf_counter()
        frame()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    stop.set()
    time.sleep(0.3)
    times.sort()
    q = lambda p: times[min(int(len(times) * p), len(times) - 1)]
    res = {
        "contended": a.contend, "positions": a.positions, "experts": a.experts,
        "mean": statistics.mean(times), "p50": q(0.50), "p99": q(0.99),
        "p999": q(0.999), "max": times[-1],
        "budget_ms": 80.0, "over_budget": sum(t > 80 for t in times),
        "frames": len(times), "resident_gb": resident,
        "pcie_gbs": statistics.mean(bw) if bw else None,
        "peak_vram_gb": torch.cuda.max_memory_allocated(dev) / 1024**3,
    }
    print(json.dumps(res, indent=2))
    print(f"\nVERDICT: p99.9 = {res['p999']:.1f} ms vs 80 ms budget -> "
          f"{'PASS' if res['p999'] < 80 else 'FAIL'}   "
          f"({res['over_budget']}/{res['frames']} frames over)")
    if a.json:
        open(a.json, "w").write(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
