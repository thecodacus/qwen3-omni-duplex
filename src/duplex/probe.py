"""Probe: does the Talker already suppress its own speech when told a user is talking?

Everything in this project has assumed Qwen3-Omni cannot do duplex without training.
That assumption was never tested. It is cheap to test.

The Talker's per-step conditioning is:

    if generation_step < trailing_text_hidden.shape[1]:
        inputs_embeds += trailing_text_hidden[:, generation_step]
    else:
        inputs_embeds += tts_pad_embed

Measured: the real text runs out after ~34 steps and the remaining ~75% of frames get
`tts_pad_embed`. So extending `trailing_text_hidden` with the user's audio hiddens —
projected through the Talker's own `hidden_projection`, the same path
`_get_talker_user_parts` uses for a user turn — tells the model "a user is speaking
right now" at every frame past that point, with no code changes and no training.

If duplex behaviour is latent, output energy after the injection point should collapse.
If it is absent, the model keeps talking over the user.

Three conditions, same prompt and same seed:
    baseline   unmodified (pad after text runs out)
    user       pad replaced with real user-audio hiddens
    shuffled   pad replaced with the SAME hiddens, time-shuffled

`shuffled` is the control that matters. If `user` goes quiet but `shuffled` does too,
the model is reacting to "conditioning is not pad", not to speech content.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def frame_energy_db(wav: np.ndarray, sr: int, frame_ms: float = 80.0) -> np.ndarray:
    n = int(sr * frame_ms / 1000)
    return np.array([
        20 * np.log10(np.sqrt(np.mean(wav[i:i + n] ** 2)) + 1e-12)
        for i in range(0, len(wav) - n + 1, n)
    ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/models/Qwen3-Omni-AWQ")
    p.add_argument("--out", default="/root/q3o_out")
    p.add_argument("--user-audio", default=None,
                   help="wav of a user speaking; injected as conditioning")
    p.add_argument("--prompt", default="Tell me about running AI locally, in three sentences.")
    p.add_argument("--max-new", type=int, default=48)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from duplex.quant.dequant import patch_packed_linears
    import soundfile as sf

    print("loading...", flush=True)
    t0 = time.time()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        a.model, device_map="cpu", dtype=torch.float32,
    ).eval()
    patch_packed_linears(model, verbose=False)
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    proc = Qwen3OmniMoeProcessor.from_pretrained(a.model)
    conv = [{"role": "user", "content": [{"type": "text", "text": a.prompt}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=text, return_tensors="pt")
    sr = getattr(model.code2wav.config, "sampling_rate", 24000)

    # Build the injected conditioning: real user audio through the Talker's own
    # hidden_projection, exactly as _get_talker_user_parts does for a user turn.
    inject = {}
    if a.user_audio:
        import librosa
        ua, _ = librosa.load(a.user_audio, sr=16000, mono=True)
        uconv = [{"role": "user", "content": [{"type": "audio", "audio": ua}]}]
        utext = proc.apply_chat_template(uconv, add_generation_prompt=False, tokenize=False)
        uin = proc(text=utext, audio=[ua], return_tensors="pt")
        with torch.inference_mode():
            uo = model.thinker(**uin, output_hidden_states=True)
        acc = model.config.talker_config.accept_hidden_layer
        uh = uo.hidden_states[acc][0]                       # [T, 2048] layer-24
        with torch.inference_mode():
            proj = model.talker.hidden_projection(uh)       # -> talker hidden size
        inject["user"] = proj.unsqueeze(0)
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(proj.shape[0], generator=g)
        inject["shuffled"] = proj[perm].unsqueeze(0)
        print(f"  user conditioning: {tuple(proj.shape)} from {len(ua)/16000:.2f}s of audio", flush=True)

    orig_parts = model._get_talker_assistant_parts
    results = {}

    for cond in ["baseline"] + list(inject):
        if cond == "baseline":
            model._get_talker_assistant_parts = orig_parts
        else:
            extra = inject[cond]

            def patched(*args, _extra=extra, **kw):
                emb, ids, trailing = orig_parts(*args, **kw)
                # Append the injected conditioning so that past the real text the
                # model is told a user is speaking, instead of receiving pad.
                n = 200   # 16s of frames; 600 made CPU generation huge and it was killed
                rep = int(np.ceil(n / _extra.shape[1]))
                tail = _extra.repeat(1, rep, 1)[:, :n].to(trailing.dtype)
                return emb, ids, torch.cat([trailing, tail], dim=1)

            model._get_talker_assistant_parts = patched

        torch.manual_seed(0)
        t0 = time.time()
        try:
            with torch.inference_mode():
                seq, wav = model.generate(
                    **inputs, return_audio=True, max_new_tokens=a.max_new,
                    thinker_do_sample=False, talker_do_sample=False,
                )
        except Exception as e:
            print(f"\n[{cond}] FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        w = (wav[0] if isinstance(wav, (list, tuple)) else wav).reshape(-1).float().cpu().numpy()
        sf.write(out / f"probe_{cond}.wav", w, sr)
        db = frame_energy_db(w, sr)
        results[cond] = (w, db)
        print(f"\n[{cond}] {len(w)/sr:.2f}s in {time.time()-t0:.0f}s -> probe_{cond}.wav", flush=True)
        print(f"   frames {len(db)}  mean {db.mean():.1f} dB  "
              f"silent(<-50dB) {100*(db<-50).mean():.1f}%", flush=True)

    model._get_talker_assistant_parts = orig_parts

    # The text runs out around frame 34; everything after is where injection acts.
    print("\n--- energy after the injection point (frame 40+) ---", flush=True)
    base_db = results["baseline"][1]
    for cond, (_, db) in results.items():
        tail = db[40:] if len(db) > 40 else db
        print(f"  {cond:<10} mean {tail.mean():7.1f} dB   silent {100*(tail<-50).mean():5.1f}%   "
              f"frames {len(tail)}", flush=True)
    if "user" in results:
        d = results["user"][1][40:].mean() - base_db[40:].mean()
        print(f"\n  user vs baseline: {d:+.1f} dB", flush=True)
        print("  A large negative number would mean the model already quiets itself when", flush=True)
        print("  told a user is speaking. Near zero means the behaviour is absent.", flush=True)
        if "shuffled" in results:
            ds = results["shuffled"][1][40:].mean() - base_db[40:].mean()
            print(f"  shuffled vs baseline: {ds:+.1f} dB  (control: if this matches 'user',")
            print("  the model is reacting to non-pad conditioning, not to speech)", flush=True)


if __name__ == "__main__":
    main()
