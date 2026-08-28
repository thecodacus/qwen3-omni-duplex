# qwen3-omni-duplex

Turning [Qwen3-Omni-30B-A3B](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) into a
**full-duplex** speech model — one that listens while it talks — on a single 12 GB consumer GPU.

Qwen3-Omni is half-duplex: it waits for you to stop speaking, then answers. Thinking Machines'
`TML-Interaction-Small` showed the alternative — a model that emits a frame every 200 ms whether or
not it has anything to say, so interruption and backchannel become next-token prediction instead of
a dialogue state machine. That model is a closed research preview.

This repo asks whether the same architecture can be recovered from open weights. It mostly can:
the split TML describes is already latent in Qwen3-Omni's config.

## The finding

```
accept_hidden_layer: 24        # Talker taps Thinker layer 24 of 48
position_id_per_seconds: 13    # ~12.5 Hz — same clock as Moshi's Mimi
seconds_per_chunk: 2           # current half-duplex granularity
```

The Talker is conditioned on Thinker layer-24 hidden states per token position, and **never sees
layers 25–47**. Depth is feed-forward, so layer 24 cannot depend on anything above it. That makes
layer 24 a **bus boundary**:

| path | components | policy |
|---|---|---|
| clock | AuT → Thinker[0:24] → Talker → MTP → Code2Wav | pinned in VRAM, 80 ms deadline, never faults |
| background | Thinker[24:48] → text, reasoning, tools | offloaded to RAM, deferred bursts, no deadline |

Half-duplex turns out to be *orchestration*, not architecture — the reference implementation runs
`thinker.generate()` to completion and only then invokes the Talker, but the conditioning is
per-token and can be streamed.

Full derivation, with source citations: [`docs/design.md`](docs/design.md).
Chronological log of findings and wrong turns: [`JOURNEY.md`](JOURNEY.md).

## Status

- [x] Architecture verified against `modular_qwen3_omni_moe.py` and `config.json`
- [x] **Clock-path deadline benchmark passes** — p99.9 of 55.5 ms clean / 64.4 ms under PCIe
      contention, against an 80 ms budget, 8.74 GB resident on an RTX 3060
- [x] **Vocoder streams at 80 ms frames** — the shipped `chunked_decode` drops 28.9% of every
      frame; a stateful rewrite is sample-exact at 9.43 ms/frame
- [x] **Quantized Thinker runs** — on-the-fly int4 dequant, verified by coherent speech
- [x] **Realtime clock path: ~66 ms against an 80 ms budget**, 10.74 GB — via a fused MoE
      (41.95 → 17.36 ms), a Triton dequant-GEMV (4.078 → 0.639 ms/layer) and partial residency
- [x] llama.cpp turns out **not** to be required — the dividing line was in-memory vs in-register
      dequantization, not framework
- [x] **Clock path runs end to end: 77.16 ms p99.9 vs an 80 ms budget, 0/150 frames over,
      10.24 GB resident** — measured in one loop, not projected from per-stage numbers
- [ ] Silence-frame semantics — no existing mechanism; the codec special tokens are a
      one-shot mode/voice preamble, not per-frame control (see `docs/design.md` §2)
- [ ] Dual-channel training for interrupt/backchannel behaviour

## Usage

```bash
duplex geometry                       # print the split and its cost
duplex verify /models/Qwen3-Omni-AWQ  # assert constants against a real checkpoint
duplex bench --experts 32 --frames 3000 --contend   # frame-deadline benchmark
duplex thesis --model /models/Qwen3-Omni-AWQ        # vocoder chunk-size sweep
```

`duplex bench` uses synthetic weights at exact config geometry — it needs a GPU but no checkpoint.

### Docker

Weights are never baked into the image (the AWQ build alone is 27.6 GB); mount them.

```bash
docker compose run --rm duplex geometry
MODELS_DIR=/root/models docker compose run --rm duplex thesis --model /models/Qwen3-Omni-AWQ
```

## Hardware

Developed against an RTX 3060 (12 GB) with 62 GB host RAM. The clock path is ~10 GB at Q4 with the
full 128 experts/layer, which is the entire point: it fits, so it never needs an expert cache.

## License

Apache-2.0, matching the upstream model.
