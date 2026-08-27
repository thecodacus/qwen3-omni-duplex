"""Verify the stateful vocoder reproduces the batch result frame-by-frame.

Three decodes of identical codes:

  reference   model(codes)                              one unchunked forward
  shipped     model.chunked_decode(codes, chunk_size=1) the stateless wrapper
  streaming   StreamingCode2Wav, one code at a time     this project's fix

The bar is exact: `streaming` must match `reference` to the bf16 noise floor AND
emit `total_upsample` samples per frame with no per-frame loss. `shipped` is
included to show the 28.9% it drops.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from duplex.streaming.code2wav import StreamingCode2Wav
from duplex.thesis.vocoder_standalone import load_code2wav, make_codes


def compare(name: str, w: np.ndarray, ref: np.ndarray, ups: int, frames: int) -> dict:
    n = min(len(w), len(ref))
    d = np.abs(w[:n] - ref[:n])
    r = {
        "name": name,
        "samples": len(w),
        "delta_vs_ref": len(w) - len(ref),
        "per_frame": len(w) / frames,
        "max_abs_err": float(d.max()) if n else float("nan"),
        "mean_abs_err": float(d.mean()) if n else float("nan"),
        "loss_pct": 100.0 * (1 - (len(w) / frames) / ups),
    }
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/models/Qwen3-Omni-AWQ")
    p.add_argument("--out", default="/root/q3o_out")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--code-mode", default="smooth", choices=["smooth", "iid"])
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, a.dtype)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model, cfg, n_tensors, _ = load_code2wav(a.model, dev, dtype)
    ups = int(model.total_upsample)
    sr = getattr(cfg, "sampling_rate", 24000)
    budget_ms = ups / sr * 1000
    codes = make_codes(a.frames, cfg.num_quantizers, cfg.codebook_size, a.code_mode).to(dev)
    print(f"code2wav loaded ({n_tensors} tensors), upsample={ups}, sr={sr}")
    print(f"{a.frames} frames = {a.frames*ups/sr:.1f}s, budget {budget_ms:.1f} ms/frame\n")

    # ---- reference: one unchunked forward ----
    with torch.inference_mode():
        ref = model(codes).reshape(-1).float().cpu().numpy()

    # ---- shipped wrapper at chunk_size=1 ----
    t0 = time.time()
    with torch.inference_mode():
        shipped = model.chunked_decode(codes, chunk_size=1, left_context_size=25)
    torch.cuda.synchronize() if dev == "cuda" else None
    shipped_s = time.time() - t0
    shipped = shipped.reshape(-1).float().cpu().numpy()

    # ---- streaming, one code at a time ----
    sc = StreamingCode2Wav(model).install()
    sc.reset()
    pieces, per_frame_ms = [], []
    try:
        for t in range(a.frames):
            torch.cuda.synchronize() if dev == "cuda" else None
            t0 = time.time()
            w = sc.decode(codes[:, :, t : t + 1])
            torch.cuda.synchronize() if dev == "cuda" else None
            per_frame_ms.append((time.time() - t0) * 1000)
            pieces.append(w.reshape(-1).float().cpu().numpy())
    finally:
        sc.remove()
    stream = np.concatenate(pieces)

    rows = [
        compare("reference (unchunked)", ref, ref, ups, a.frames),
        compare("shipped chunked_decode(cs=1)", shipped, ref, ups, a.frames),
        compare("streaming (stateful)", stream, ref, ups, a.frames),
    ]

    print(f"{'path':<32}{'samples':>9}{'/frame':>9}{'loss':>8}{'max err':>10}{'mean err':>10}")
    for r in rows:
        print(f"{r['name']:<32}{r['samples']:>9}{r['per_frame']:>9.1f}"
              f"{r['loss_pct']:>7.1f}%{r['max_abs_err']:>10.5f}{r['mean_abs_err']:>10.6f}")

    pf = np.array(per_frame_ms[1:])  # drop first frame (warm-up + alloc)
    print(f"\nstreaming latency: mean {pf.mean():.2f} ms  p99 {np.percentile(pf,99):.2f} ms  "
          f"max {pf.max():.2f} ms   vs {budget_ms:.1f} ms budget")
    print(f"shipped cs=1 total {shipped_s*1000/a.frames:.2f} ms/frame")

    import soundfile as sf
    sf.write(out / "verify_reference.wav", ref, sr)
    sf.write(out / "verify_streaming.wav", stream, sr)
    sf.write(out / "verify_shipped_cs1.wav", shipped, sr)

    st = rows[2]
    exact = st["max_abs_err"] < 0.02 and abs(st["loss_pct"]) < 0.5
    print("\nVERDICT")
    print(f"  streaming reproduces reference: {'YES' if exact else 'NO'} "
          f"(max err {st['max_abs_err']:.5f}, loss {st['loss_pct']:.2f}%)")
    print(f"  shipped wrapper loses {rows[1]['loss_pct']:.1f}% per frame")
    print(f"  realtime at 80 ms: {'YES' if pf.mean() < budget_ms else 'NO'}")

    json.dump({"rows": rows, "per_frame_ms": per_frame_ms, "budget_ms": budget_ms,
               "frames": a.frames, "sr": sr, "upsample": ups},
              open(out / "stream_verify.json", "w"), indent=2)


if __name__ == "__main__":
    main()
