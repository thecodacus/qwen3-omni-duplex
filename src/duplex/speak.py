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
    p.add_argument("--frame-decode", action="store_true",
                   help="decode audio with the streaming vocoder at 80ms frames "
                        "instead of the shipped chunked_decode")
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

    print(f"\ngenerating (max_new_tokens={a.max_new}, CPU + on-the-fly dequant: slow)...",
          flush=True)
    t0 = time.time()
    with torch.inference_mode():
        seq, wav = model.generate(
            **inputs, return_audio=True, max_new_tokens=a.max_new,
            thinker_do_sample=False, talker_do_sample=False,
        )
    gen_s = time.time() - t0

    print(f"  generated in {gen_s:.0f}s", flush=True)

    # Audio first: it is the thing that is expensive to reproduce, and text
    # decoding has already destroyed one run by raising before the write.
    sr = getattr(model.code2wav.config, "sampling_rate", 24000)
    import soundfile as sf
    try:
        w = (wav[0] if isinstance(wav, (list, tuple)) else wav)
        w = w.reshape(-1).float().cpu().numpy()
        sf.write(out / "speak_batch.wav", w, sr)
        print(f"  AUDIO {len(w)/sr:.2f}s -> {out}/speak_batch.wav", flush=True)
    except Exception as e:
        print(f"  audio write FAILED: {type(e).__name__}: {e}")

    # `generate(return_audio=True)` has returned different shapes across versions,
    # so decode defensively rather than assuming a tensor of ids.
    try:
        if isinstance(seq, str):
            said = seq
        else:
            ids = seq
            if isinstance(ids, (list, tuple)):
                ids = ids[0]
            if torch.is_tensor(ids) and ids.ndim == 1:
                ids = ids.unsqueeze(0)
            said = proc.batch_decode(ids, skip_special_tokens=True)[0]
        print(f"  TEXT: {said[-400:]}")
    except Exception as e:
        print(f"  text decode failed ({type(e).__name__}: {e}); seq type={type(seq)}")
        if torch.is_tensor(seq):
            print(f"    seq shape={tuple(seq.shape)} dtype={seq.dtype}")

    print("\nIf that text is coherent and the audio is intelligible, the int4 "
          "dequantization is correct and the speech stack is sound.")


if __name__ == "__main__":
    main()
