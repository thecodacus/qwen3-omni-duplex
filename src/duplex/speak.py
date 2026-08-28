"""End-to-end speech from the AWQ build: Thinker -> Talker -> streaming vocoder.

This is the offline quality gate, not a realtime path. The Thinker runs on CPU with
on-the-fly int4 dequantization (see duplex.quant.dequant for why dense is
impossible here), which is slow by construction. Its job is to answer one question
that no amount of latency measurement can: does the speech actually sound right.

Placement:
    thinker   CPU, weights kept packed (~15 GB)
    talker    GPU, bf16 (6.20 GB)
    code2wav  GPU, bf16 (0.40 GB), driven by StreamingCode2Wav at 80 ms frames
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/models/Qwen3-Omni-AWQ")
    p.add_argument("--out", default="/root/q3o_out")
    p.add_argument("--prompt", default="Say one short sentence about running AI locally.")
    p.add_argument("--max-new", type=int, default=24)
    p.add_argument("--cpu", action="store_true", help="force everything onto CPU")
    p.add_argument("--speakers", default="Ethan",
                   help="comma-separated voices; the checkpoint ships Chelsie, Ethan, Aiden. "
                        "The load dominates runtime, so generating several per load is ~free.")
    p.add_argument("--frame-decode", action="store_true",
                   help="also decode with the stateful vocoder frame-by-frame at 80ms, "
                        "compare against the batch path, and time each frame")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from duplex.quant.dequant import patch_packed_linears, place_unpacked, sanity_check

    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32

    print("loading (experts stay packed in host RAM)...", flush=True)
    t0 = time.time()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        a.model, device_map="cpu", dtype=dtype,
    ).eval()
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    for c in sanity_check(model, k=2):
        print(f"  {c['name']}: shape_ok={c['shape_ok']} mean={c['mean']:+.6f} "
              f"std={c['std']:.5f} uniq/group={c['n_unique_first_group']}")

    if dev == "cuda":
        # Selective placement. Moving the whole model sweeps in the audio and
        # vision towers alongside the talker and OOMs a 12GB card at ~9.6GB.
        # The towers are unused for a text prompt, so they stay on the host.
        print("placing on GPU (talker, code2wav, thinker embeddings/norms):", flush=True)
        for name in ("code2wav", "talker"):
            sub = getattr(model, name, None)
            if sub is not None:
                place_unpacked(sub, dev)
                print(f"    after {name}: {torch.cuda.memory_allocated()/1024**3:.2f} GB",
                      flush=True)
        # thinker.model holds all 18624 packed tensors plus 242 dense
        # (embeddings/norms); lm_head sits OUTSIDE it and is dense. audio_tower
        # (525) and visual (351) are dense too but unused for a text prompt, so
        # they stay on the host.
        th = getattr(model, "thinker", None)
        if th is not None:
            for attr in ("model", "lm_head"):
                sub = getattr(th, attr, None)
                if sub is not None:
                    place_unpacked(sub, dev)
                    print(f"    after thinker.{attr}: "
                          f"{torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)
    patch_packed_linears(model, compute_device=dev if dev == "cuda" else None)

    proc = Qwen3OmniMoeProcessor.from_pretrained(a.model)
    conv = [{"role": "user", "content": [{"type": "text", "text": a.prompt}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=text, return_tensors="pt")
    if dev == "cuda":
        inputs = inputs.to(dev)

    import soundfile as sf
    sr = getattr(model.code2wav.config, "sampling_rate", 24000)
    speakers = [x.strip() for x in a.speakers.split(",") if x.strip()]

    # Capture the codes the talker produces, so they can be re-decoded frame by
    # frame. Every vocoder test until now used synthetic codes; this is the first
    # time real speech codes go through the streaming path.
    captured = {}
    if a.frame_decode:
        _orig_cd = model.code2wav.chunked_decode

        def _capture(codes, **kw):
            captured["codes"] = codes.detach().clone()
            return _orig_cd(codes, **kw)

        model.code2wav.chunked_decode = _capture

    for spk in speakers:
        print(f"\ngenerating [{spk}] (max_new_tokens={a.max_new})...", flush=True)
        t0 = time.time()
        with torch.inference_mode():
            seq, wav = model.generate(
                **inputs, return_audio=True, max_new_tokens=a.max_new,
                speaker=spk, thinker_do_sample=False, talker_do_sample=False,
            )
        print(f"  generated in {time.time()-t0:.0f}s", flush=True)

        # Audio first: it is expensive to reproduce, and a text-decode error has
        # already destroyed one run by raising before the write.
        try:
            w = (wav[0] if isinstance(wav, (list, tuple)) else wav)
            w = w.reshape(-1).float().cpu().numpy()
            path = out / f"speak_{spk.lower()}.wav"
            sf.write(path, w, sr)
            print(f"  AUDIO {len(w)/sr:.2f}s -> {path}", flush=True)
        except Exception as e:
            print(f"  audio write FAILED: {type(e).__name__}: {e}")

        try:
            ids = getattr(seq, "sequences", seq)
            if isinstance(ids, (list, tuple)):
                ids = ids[0]
            if torch.is_tensor(ids) and ids.ndim == 1:
                ids = ids.unsqueeze(0)
            # dump raw ids so a bad decode can be diagnosed without a 15min reload
            if torch.is_tensor(ids):
                torch.save(ids.cpu(), out / f"ids_{spk.lower()}.pt")
            said = proc.batch_decode(ids, skip_special_tokens=True)[0]
            print(f"  TEXT: {said[-400:]}")
        except Exception as e:
            print(f"  text decode failed ({type(e).__name__}: {e}); type={type(seq)}")

        if a.frame_decode and "codes" in captured:
            frame_decode_report(model, captured["codes"], sr, out, spk)

    print("\nIf that text is coherent and the audio is intelligible, the int4 "
          "dequantization is correct and the speech stack is sound.")


def frame_decode_report(model, codes, sr, out, spk):
    """Re-decode real talker codes one 80ms frame at a time and compare."""
    import time
    import numpy as np
    import soundfile as sf
    from duplex.streaming.code2wav import StreamingCode2Wav

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c2w = model.code2wav.to(dev).eval()
    codes = codes.to(dev)
    T = codes.shape[-1]
    ups = int(c2w.total_upsample)
    budget = ups / sr * 1000

    with torch.inference_mode():
        batch = c2w.chunked_decode(codes, chunk_size=300, left_context_size=25)
    batch = batch.reshape(-1).float().cpu().numpy()

    sc = StreamingCode2Wav(c2w).install()
    sc.reset()
    pieces, per_frame = [], []
    try:
        for t in range(T):
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            w = sc.decode(codes[:, :, t : t + 1])
            if dev == "cuda":
                torch.cuda.synchronize()
            per_frame.append((time.perf_counter() - t0) * 1000)
            pieces.append(w.reshape(-1).float().cpu().numpy())
    finally:
        sc.remove()
    stream = np.concatenate(pieces)

    sf.write(out / f"stream_{spk.lower()}.wav", stream, sr)
    pf = np.array(per_frame[1:])
    n = min(len(stream), len(batch))
    print(f"  [frame-decode] {T} frames of REAL speech codes")
    print(f"     batch  {len(batch):>7} samples   streaming {len(stream):>7} samples "
          f"(delta {len(stream)-len(batch):+d})")
    print(f"     max|stream-batch| over common span: {np.abs(stream[:n]-batch[:n]).max():.5f}")
    print(f"     per-frame: mean {pf.mean():.2f} ms  p99 {np.percentile(pf,99):.2f} ms  "
          f"max {pf.max():.2f} ms   vs {budget:.1f} ms budget")
    print(f"     realtime factor: {budget/pf.mean():.1f}x faster than playback")
    print(f"     -> {out}/stream_{spk.lower()}.wav")


if __name__ == "__main__":
    main()
