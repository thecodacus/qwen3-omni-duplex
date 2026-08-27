"""
Thesis test: can Qwen3-Omni's speech stack be driven at 80ms frame granularity?

The reference path decodes audio as:
    code2wav.chunked_decode(codes, chunk_size=300, left_context_size=25)
300 codes @ 12.5Hz = 24 SECONDS per vocoder call. Full duplex needs chunk_size=1.

The source comment names the hazard directly:
    "A chunk of n codes yields slightly fewer than n * total_upsample samples,
     because the causal transposed convolutions drop a fixed number of trailing
     samples per chunk"
At chunk_size=300 that loss is paid once per 24s. At chunk_size=1 it is paid
every frame. This measures whether that destroys the audio.

Stages:
  A  baseline end-to-end generate() -> ground-truth codes + wav (listen to it)
  B  re-decode the SAME codes at chunk_size in {300,100,25,5,2,1}, compare to
     baseline: sample loss, waveform MSE, spectral distance, per-frame latency
  C  verdict on whether chunk_size=1 is viable as-is

Only stage A needs the full model; B is vocoder-only and is the real experiment.
"""

import argparse, json, time
import numpy as np
import torch
import soundfile as sf


def spectral_dist(a, b, n_fft=1024):
    """Log-magnitude STFT distance — catches artefacts that MSE misses after tiny shifts."""
    n = min(len(a), len(b))
    if n < n_fft:
        return float("nan")
    A = torch.stft(torch.as_tensor(a[:n]).float(), n_fft, hop_length=n_fft // 4,
                   window=torch.hann_window(n_fft), return_complex=True).abs().clamp_min(1e-7).log()
    B = torch.stft(torch.as_tensor(b[:n]).float(), n_fft, hop_length=n_fft // 4,
                   window=torch.hann_window(n_fft), return_complex=True).abs().clamp_min(1e-7).log()
    return float((A - B).abs().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/models/Qwen3-Omni-AWQ")
    p.add_argument("--out", default="/root/q3o_out")
    p.add_argument("--prompt", default="Explain in two sentences why running a language model on your own hardware matters.")
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--chunks", default="300,100,25,5,2,1")
    a = p.parse_args()

    import os
    os.makedirs(a.out, exist_ok=True)
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    print("loading model (this offloads to CPU; 12GB VRAM can't hold it all)...", flush=True)
    t0 = time.time()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        a.model, dtype="auto", device_map="auto",
    ).eval()
    proc = Qwen3OmniMoeProcessor.from_pretrained(a.model)
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    sr = getattr(model.code2wav.config, "sampling_rate", 24000)
    ups = int(model.code2wav.total_upsample)
    print(f"code2wav: total_upsample={ups}  sr={sr}  -> {ups/sr*1000:.1f} ms of audio per code")

    # ---------- A: baseline ----------
    conv = [{"role": "user", "content": [{"type": "text", "text": a.prompt}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=text, return_tensors="pt").to(model.device)

    print("\n[A] baseline generate()...", flush=True)
    t0 = time.time()
    with torch.inference_mode():
        seq, wav = model.generate(**inputs, return_audio=True, max_new_tokens=a.max_new,
                                  thinker_do_sample=False, talker_do_sample=False)
    gen_s = time.time() - t0
    wav0 = (wav[0] if isinstance(wav, list) else wav).reshape(-1).float().cpu().numpy()
    sf.write(f"{a.out}/baseline.wav", wav0, sr)
    said = proc.batch_decode(seq, skip_special_tokens=True)[0]
    print(f"  generated in {gen_s:.0f}s   wav {len(wav0)/sr:.2f}s -> {a.out}/baseline.wav")
    print(f"  text: {said[-300:]}")

    # ---------- recover the codes the talker produced ----------
    # re-run just the talker capture path is expensive; instead re-derive codes by
    # inverting the known relationship is not possible, so we regenerate with a hook.
    codes_holder = {}
    orig = model.code2wav.chunked_decode

    def capture(codes, **kw):
        codes_holder["codes"] = codes.detach().clone()
        return orig(codes, **kw)

    model.code2wav.chunked_decode = capture
    with torch.inference_mode():
        model.generate(**inputs, return_audio=True, max_new_tokens=a.max_new,
                       thinker_do_sample=False, talker_do_sample=False)
    model.code2wav.chunked_decode = orig
    codes = codes_holder["codes"]
    T = codes.shape[-1]
    print(f"\n  captured codes: {tuple(codes.shape)}  = {T} frames = {T*ups/sr:.2f}s @ {sr/ups:.1f} Hz")

    # ---------- B: chunk-size sweep ----------
    print("\n[B] re-decoding identical codes at decreasing chunk_size", flush=True)
    ref = None
    rows = []
    for cs in [int(x) for x in a.chunks.split(",")]:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        with torch.inference_mode():
            w = model.code2wav.chunked_decode(codes, chunk_size=cs, left_context_size=25)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        dt = time.time() - t0
        w = w.reshape(-1).float().cpu().numpy()
        if ref is None:
            ref = w
        n = min(len(w), len(ref))
        mse = float(np.mean((w[:n] - ref[:n]) ** 2))
        row = {
            "chunk_size": cs,
            "chunk_ms": cs * ups / sr * 1000,
            "samples": len(w),
            "lost_vs_ref": len(ref) - len(w),
            "mse_vs_ref": mse,
            "spec_dist": spectral_dist(w, ref),
            "decode_s": dt,
            "ms_per_frame": dt / T * 1000,
        }
        rows.append(row)
        sf.write(f"{a.out}/chunk{cs}.wav", w, sr)
        print(f"  cs={cs:<4} {row['chunk_ms']:7.1f}ms/chunk  samples={len(w):>7} "
              f"(lost {row['lost_vs_ref']:>6})  mse={mse:.3e}  spec={row['spec_dist']:.4f}  "
              f"{row['ms_per_frame']:.2f} ms/frame", flush=True)

    json.dump({"rows": rows, "frames": T, "sr": sr, "upsample": ups, "text": said},
              open(f"{a.out}/thesis.json", "w"), indent=2)

    # ---------- C: verdict ----------
    one = next((r for r in rows if r["chunk_size"] == 1), None)
    print("\n[C] VERDICT")
    if one:
        budget = ups / sr * 1000
        print(f"  chunk_size=1 emits {budget:.1f} ms of audio per call, costing "
              f"{one['ms_per_frame']:.2f} ms -> {'FITS' if one['ms_per_frame'] < budget else 'EXCEEDS'} realtime")
        print(f"  sample loss vs 24s-chunk baseline: {one['lost_vs_ref']} "
              f"({one['lost_vs_ref']/max(len(ref),1)*100:.2f}% of the waveform)")
        print(f"  spectral distance: {one['spec_dist']:.4f}  (0 = identical; >0.5 audibly different)")
    print(f"\n  WAVs in {a.out}/ — listen to baseline.wav vs chunk1.wav")


if __name__ == "__main__":
    main()
