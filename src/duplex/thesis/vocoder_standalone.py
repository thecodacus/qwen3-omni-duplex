"""Standalone code2wav thesis test — no Thinker, no Talker, no quantization.

The AWQ build leaves code2wav entirely unquantized (230 tensors, 0 packed, all
in the quantization `ignore` list) and confines it to one shard. So the vocoder
loads on its own, in bf16, straight onto the GPU — which sidesteps the
compressed-tensors/accelerate offload bug that blocks loading the full model on
12 GB, and isolates the experiment to the part actually under test.

`chunked_decode` is stateless: each chunk re-runs the full forward over
`[left_context : chunk_end]` and discards the context's output.

    chunk_size=300, left=25  ->  325 codes processed per 300 kept  (1.08x)
    chunk_size=1,   left=25  ->   26 codes processed per   1 kept  (26x)

Two independent questions, swept separately:

  1. CORRECTNESS -- does shrinking chunk_size change the waveform? Chunk
     boundaries are where causal convolutions lose trailing samples.
  2. COST -- how much left context does the receptive field actually need?
     If 25 can drop to 4, per-frame cost falls from 26x to 5x. This is where
     realtime is won or lost.

Codes are synthetic. Chunk-boundary behaviour is a property of the convolution
stack, not of what the codes mean -- but see `--code-mode`: iid-random codes are
unrealistically jumpy, so `smooth` mimics the temporal correlation of real
speech codes and is the default.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def make_codes(T: int, n_q: int, codebook: int, mode: str, seed: int = 0) -> torch.Tensor:
    """Synthetic codec codes, [1, n_q, T]."""
    g = torch.Generator().manual_seed(seed)
    if mode == "iid":
        return torch.randint(0, codebook, (1, n_q, T), generator=g)
    # `smooth`: a random walk that mostly holds, mimicking real speech codes
    out = torch.zeros(1, n_q, T, dtype=torch.long)
    cur = torch.randint(0, codebook, (n_q,), generator=g)
    for t in range(T):
        step = (torch.rand(n_q, generator=g) < 0.25)
        jitter = torch.randint(-3, 4, (n_q,), generator=g)
        cur = torch.where(step, (cur + jitter) % codebook, cur)
        out[0, :, t] = cur
    return out


def spectral_dist(a: np.ndarray, b: np.ndarray, n_fft: int = 1024) -> float:
    n = min(len(a), len(b))
    if n < n_fft:
        return float("nan")
    w = torch.hann_window(n_fft)
    f = lambda x: torch.stft(torch.as_tensor(x[:n]).float(), n_fft, hop_length=n_fft // 4,
                             window=w, return_complex=True).abs().clamp_min(1e-7).log()
    return float((f(a) - f(b)).abs().mean())


def load_code2wav(model_path: str, device: str, dtype: torch.dtype):
    from safetensors.torch import load_file
    from transformers.models.qwen3_omni_moe import Qwen3OmniMoeCode2Wav
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeCode2WavConfig,
    )

    root = Path(model_path)
    cfg_all = json.loads((root / "config.json").read_text())
    cfg = Qwen3OmniMoeCode2WavConfig(**cfg_all["code2wav_config"])

    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted({v for k, v in index.items() if k.startswith("code2wav.")})
    state = {}
    for s in shards:
        blob = load_file(str(root / s))
        for k, v in blob.items():
            if k.startswith("code2wav."):
                state[k[len("code2wav."):]] = v
        del blob

    model = Qwen3OmniMoeCode2Wav(cfg).to(dtype)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if "code_offset" not in m]  # non-persistent buffer
    if missing or unexpected:
        print(f"  ! missing={missing[:4]} unexpected={unexpected[:4]}")
    return model.to(device).eval(), cfg, len(state), shards


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/models/Qwen3-Omni-AWQ")
    p.add_argument("--out", default="/root/q3o_out")
    p.add_argument("--frames", type=int, default=250, help="code frames (250 @12.5Hz = 20s)")
    p.add_argument("--chunks", default="300,100,25,10,5,2,1")
    p.add_argument("--left-ctx", default="25,12,6,4,2,1,0")
    p.add_argument("--code-mode", default="smooth", choices=["smooth", "iid"])
    p.add_argument("--dtype", default="bfloat16")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, a.dtype)

    print("loading code2wav only (unquantized, single shard)...", flush=True)
    t0 = time.time()
    model, cfg, n_tensors, shards = load_code2wav(a.model, dev, dtype)
    ups = int(model.total_upsample)
    sr = getattr(cfg, "sampling_rate", 24000)
    vram = torch.cuda.memory_allocated() / 1024**3 if dev == "cuda" else 0
    print(f"  {n_tensors} tensors from {shards} in {time.time()-t0:.1f}s, {vram:.3f} GB VRAM")
    print(f"  total_upsample={ups}  sr={sr}  -> {ups/sr*1000:.1f} ms audio per code "
          f"({sr/ups:.2f} Hz frame grid)")

    codes = make_codes(a.frames, cfg.num_quantizers, cfg.codebook_size, a.code_mode).to(dev)
    print(f"  codes {tuple(codes.shape)} ({a.code_mode}) = {a.frames*ups/sr:.1f}s of audio\n")

    budget_ms = ups / sr * 1000

    def run(cs: int, lc: int):
        torch.cuda.synchronize() if dev == "cuda" else None
        t = time.time()
        with torch.inference_mode():
            w = model.chunked_decode(codes, chunk_size=cs, left_context_size=lc)
        torch.cuda.synchronize() if dev == "cuda" else None
        dt = time.time() - t
        return w.reshape(-1).float().cpu().numpy(), dt

    import soundfile as sf

    # ---- 1. CORRECTNESS: chunk_size sweep at the reference left context ----
    print(f"[1] chunk_size sweep (left_context=25, the shipped value)")
    print(f"    {'cs':>4} {'ms/chunk':>9} {'samples':>9} {'vs ref':>8} {'mse':>10} {'spec':>7} "
          f"{'ms/frame':>9} {'realtime':>9}")
    ref = None
    rows = []
    for cs in [int(x) for x in a.chunks.split(",")]:
        w, dt = run(cs, 25)
        if ref is None:
            ref = w
        n = min(len(w), len(ref))
        r = {"chunk_size": cs, "left_ctx": 25, "chunk_ms": cs * ups / sr * 1000,
             "samples": len(w), "delta_samples": len(w) - len(ref),
             "mse": float(np.mean((w[:n] - ref[:n]) ** 2)),
             "spec": spectral_dist(w, ref), "total_s": dt,
             "ms_per_frame": dt / a.frames * 1000}
        r["realtime"] = r["ms_per_frame"] < budget_ms
        rows.append(r)
        sf.write(out / f"cs{cs}.wav", w, sr)
        print(f"    {cs:>4} {r['chunk_ms']:>9.1f} {len(w):>9} {r['delta_samples']:>8} "
              f"{r['mse']:>10.2e} {r['spec']:>7.4f} {r['ms_per_frame']:>9.2f} "
              f"{'YES' if r['realtime'] else 'no':>9}")

    # ---- 2. COST: how much left context does chunk_size=1 actually need? ----
    print(f"\n[2] left_context sweep at chunk_size=1 (budget {budget_ms:.1f} ms/frame)")
    print(f"    {'left':>4} {'processed/kept':>15} {'mse vs ref':>11} {'spec':>7} {'ms/frame':>9} {'realtime':>9}")
    ctx_rows = []
    for lc in [int(x) for x in a.left_ctx.split(",")]:
        w, dt = run(1, lc)
        n = min(len(w), len(ref))
        r = {"chunk_size": 1, "left_ctx": lc, "ratio": lc + 1,
             "mse": float(np.mean((w[:n] - ref[:n]) ** 2)),
             "spec": spectral_dist(w, ref), "total_s": dt,
             "ms_per_frame": dt / a.frames * 1000}
        r["realtime"] = r["ms_per_frame"] < budget_ms
        ctx_rows.append(r)
        sf.write(out / f"cs1_ctx{lc}.wav", w, sr)
        print(f"    {lc:>4} {str(lc+1)+'x':>15} {r['mse']:>11.2e} {r['spec']:>7.4f} "
              f"{r['ms_per_frame']:>9.2f} {'YES' if r['realtime'] else 'no':>9}")

    json.dump({"chunk_sweep": rows, "ctx_sweep": ctx_rows, "frames": a.frames,
               "sr": sr, "upsample": ups, "budget_ms": budget_ms,
               "code_mode": a.code_mode},
              open(out / "vocoder_thesis.json", "w"), indent=2)

    # ---- verdict ----
    one = next((r for r in rows if r["chunk_size"] == 1), None)
    best = min((r for r in ctx_rows if r["spec"] < 0.05), key=lambda r: r["ratio"], default=None)
    print("\nVERDICT")
    if one:
        print(f"  chunk_size=1 @ left=25: spec={one['spec']:.4f}  {one['ms_per_frame']:.2f} ms/frame "
              f"vs {budget_ms:.1f} ms budget -> {'FITS' if one['realtime'] else 'EXCEEDS'}")
    if best:
        print(f"  minimum left context holding spec<0.05: {best['left_ctx']} "
              f"({best['ratio']}x work, {best['ms_per_frame']:.2f} ms/frame)")
    else:
        print("  no left_context setting held spec<0.05 -- context cannot be trimmed")
    print(f"  WAVs in {out}/")


if __name__ == "__main__":
    main()
