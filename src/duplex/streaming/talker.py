"""Real-weights clock-path latency: Talker -> code predictor -> streaming vocoder.

`duplex bench` estimated the 80 ms frame budget with synthetic weights at config
geometry. This measures the same budget with the *actual* weights for every stage
except Thinker[0:24], which is the only quantized component and is blocked by a
compressed-tensors bug in the third-party AWQ build.

Loadable standalone because the AWQ repack quantizes only the Thinker:

    talker      8037 tensors, 0 packed  ->  6.20 GB bf16
    code2wav     230 tensors, 0 packed  ->  0.40 GB bf16

Conditioning is synthetic. This measures compute, not speech quality — the
per-frame cost of a Talker decode step does not depend on what the hidden states
mean. Real speech needs a working Thinker.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from duplex.streaming.code2wav import StreamingCode2Wav
from duplex.thesis.vocoder_standalone import load_code2wav


def load_talker(model_path: str, device: str, dtype: torch.dtype, fuse: bool = False):
    from safetensors.torch import load_file
    from transformers.models.qwen3_omni_moe import (
        Qwen3OmniMoeTalkerForConditionalGeneration as TK,
    )
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeTalkerConfig,
    )

    root = Path(model_path)
    cfg_all = json.loads((root / "config.json").read_text())
    cfg = Qwen3OmniMoeTalkerConfig(**cfg_all["talker_config"])
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]

    state = {}
    for s in sorted({v for k, v in index.items() if k.startswith("talker.")}):
        blob = load_file(str(root / s))
        for k, v in blob.items():
            if k.startswith("talker."):
                state[k[len("talker."):]] = v
        del blob

    model = TK(cfg).to(dtype)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"talker weights did not match: {len(missing)} missing, "
            f"{len(unexpected)} unexpected. Wrong transformers version? "
            f"(5.x expects fused experts; this checkpoint is per-expert)"
        )
    if fuse:
        # Fuse on the HOST, before the move. Stacking on-GPU has to hold the
        # originals and the stacks simultaneously, which OOMs a 12GB card; host
        # RAM has 61GB and the stacked model is the same size as the original.
        from duplex.streaming.fused_moe import fuse_moe_blocks
        fuse_moe_blocks(model)
    return model.to(device).eval(), cfg, len(state)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/models/Qwen3-Omni-AWQ")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default="/root/q3o_out")
    p.add_argument("--fuse", action="store_true",
                   help="replace the Python expert loop with gather+bmm")
    p.add_argument("--graph", action="store_true",
                   help="capture the talker decode step in a CUDA graph")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, a.dtype)

    talker, tcfg, n_tk = load_talker(a.model, dev, dtype, fuse=a.fuse)
    tk_gb = torch.cuda.memory_allocated() / 1024**3 if dev == "cuda" else 0
    c2w, ccfg, n_c2w, _ = load_code2wav(a.model, dev, dtype)
    tot_gb = torch.cuda.memory_allocated() / 1024**3 if dev == "cuda" else 0
    ups = int(c2w.total_upsample)
    sr = getattr(ccfg, "sampling_rate", 24000)
    budget = ups / sr * 1000

    print(f"talker   {n_tk} tensors  {tk_gb:.2f} GB")
    print(f"code2wav {n_c2w} tensors  {tot_gb - tk_gb:.2f} GB")
    print(f"clock path resident (minus Thinker[0:24]): {tot_gb:.2f} GB")
    print(f"budget {budget:.1f} ms/frame ({sr/ups:.2f} Hz)\n")

    hidden = tcfg.text_config.hidden_size
    n_groups = tcfg.num_code_groups
    from transformers.cache_utils import DynamicCache

    sc = StreamingCode2Wav(c2w).install()
    sc.reset()
    kv = DynamicCache()

    talk_ms, mtp_ms, wav_ms, tot_ms = [], [], [], []
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)

    try:
        with torch.inference_mode():
            for i in range(a.warmup + a.frames):
                emb = torch.randn(1, 1, hidden, device=dev, dtype=dtype)
                sync(); t0 = time.perf_counter()

                out = talker.model(inputs_embeds=emb, past_key_values=kv, use_cache=True)
                kv = getattr(out, "past_key_values", kv)
                h = out.last_hidden_state
                sync(); t1 = time.perf_counter()

                # code predictor (MTP): backbone gives codebook 0, this gives 1..15
                cp = talker.code_predictor
                cp_in = h.to(dtype)
                cp_out = cp.model(inputs_embeds=cp_in.expand(-1, n_groups - 1, -1))
                _ = cp_out.last_hidden_state
                sync(); t2 = time.perf_counter()

                codes = torch.randint(0, ccfg.codebook_size, (1, ccfg.num_quantizers, 1), device=dev)
                _ = sc.decode(codes)
                sync(); t3 = time.perf_counter()

                if i >= a.warmup:
                    talk_ms.append((t1 - t0) * 1000)
                    mtp_ms.append((t2 - t1) * 1000)
                    wav_ms.append((t3 - t2) * 1000)
                    tot_ms.append((t3 - t0) * 1000)
    finally:
        sc.remove()

    def stats(x):
        x = np.array(x)
        return dict(mean=float(x.mean()), p50=float(np.percentile(x, 50)),
                    p99=float(np.percentile(x, 99)),
                    p999=float(np.percentile(x, 99.9)), max=float(x.max()))

    rows = [("talker decode", stats(talk_ms)), ("code predictor (MTP)", stats(mtp_ms)),
            ("streaming vocoder", stats(wav_ms)), ("TOTAL", stats(tot_ms))]
    print(f"{'stage':<24}{'mean':>8}{'p50':>8}{'p99':>8}{'p99.9':>8}{'max':>8}")
    for name, s in rows:
        print(f"{name:<24}{s['mean']:>8.2f}{s['p50']:>8.2f}{s['p99']:>8.2f}"
              f"{s['p999']:>8.2f}{s['max']:>8.2f}")

    t = rows[-1][1]
    over = sum(1 for x in tot_ms if x > budget)
    print(f"\nVERDICT: p99.9 {t['p999']:.2f} ms vs {budget:.1f} ms budget -> "
          f"{'PASS' if t['p999'] < budget else 'FAIL'}   ({over}/{len(tot_ms)} frames over)")
    print(f"  headroom for Thinker[0:24]: {budget - t['p999']:.1f} ms")

    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump({"rows": {n: s for n, s in rows}, "budget_ms": budget,
               "frames": len(tot_ms), "resident_gb": tot_gb, "over_budget": over},
              open(Path(a.out) / "clock_real.json", "w"), indent=2)


if __name__ == "__main__":
    main()
