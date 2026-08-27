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
    p.add_argument("--frame-decode", action="store_true",
                   help="decode audio with the streaming vocoder at 80ms frames "
                        "instead of the shipped chunked_decode")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from duplex.quant.dequant import patch_packed_linears, sanity_check

    print("loading (thinker stays packed on CPU)...", flush=True)
    t0 = time.time()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        a.model, device_map="cpu", dtype=torch.float32,
    ).eval()
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    for c in sanity_check(model, k=2):
        print(f"  {c['name']}: shape_ok={c['shape_ok']} mean={c['mean']:+.6f} "
              f"std={c['std']:.5f} uniq/group={c['n_unique_first_group']}")
    patch_packed_linears(model)

    proc = Qwen3OmniMoeProcessor.from_pretrained(a.model)
    conv = [{"role": "user", "content": [{"type": "text", "text": a.prompt}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=text, return_tensors="pt")

    print(f"\ngenerating (max_new_tokens={a.max_new}, CPU + on-the-fly dequant: slow)...",
          flush=True)
    t0 = time.time()
    with torch.inference_mode():
        seq, wav = model.generate(
            **inputs, return_audio=True, max_new_tokens=a.max_new,
            thinker_do_sample=False, talker_do_sample=False,
        )
    gen_s = time.time() - t0

    said = proc.batch_decode(seq, skip_special_tokens=True)[0]
    print(f"  generated in {gen_s:.0f}s")
    print(f"  TEXT: {said[-400:]}")

    w = (wav[0] if isinstance(wav, list) else wav).reshape(-1).float().cpu().numpy()
    sr = getattr(model.code2wav.config, "sampling_rate", 24000)
    import soundfile as sf
    sf.write(out / "speak_batch.wav", w, sr)
    print(f"  audio {len(w)/sr:.2f}s -> {out}/speak_batch.wav")
    print("\nIf that text is coherent and the audio is intelligible, the int4 "
          "dequantization is correct and the speech stack is sound.")


if __name__ == "__main__":
    main()
