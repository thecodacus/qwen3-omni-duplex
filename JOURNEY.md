# Journey — facts in chronological order

Raw material for a video. Facts and numbers only, in the order they were found,
including the wrong turns. Not a script.

Hardware throughout: one RTX 3060 (12 GB), 62 GB host RAM, PCIe measured at
25.9 GB/s. Model: `Qwen3-Omni-30B-A3B-Instruct`, AWQ 4-bit repack (27.6 GB).

---

## 1. The starting question

Thinking Machines announced `TML-Interaction-Small` — a full-duplex model that
listens while it talks. 276B MoE, 12B active, claimed 0.40 s response latency,
77.8 on FD-bench v1.5.

It is a **closed research preview**. No paper, no weights, no data recipe.

From their blog, four design choices:
- One interleaved stream at 200 ms: `input_0 output_0 input_1 output_1 ...`. The model
  emits every tick; while you speak it emits silence. "No separate dialog management
  component" — interruption is next-token prediction.
- No encoders in the latency path: dMel (train-free binning of log-mel energies) for
  audio in, 40×40 patches through an hMLP for video, a flow head for audio out.
- A background model runs asynchronously for reasoning while the interaction model
  keeps the conversation alive.
- `gather+gemv` for MoE instead of grouped GEMM, because batch-1 decode has no batch.

Question: can that architecture be recovered from open weights, on one consumer GPU?

## 2. The only open target

- **Qwen3.5-Omni** (Mar 2026) — proprietary. API and Qwen Chat only, no weights.
- **Qwen3-Omni-30B-A3B** (Apache 2.0) — three open models: Instruct, Thinking, Captioner.

Qwen's own realtime product (`qwen3.5-omni-*-realtime`, `qwen3-omni-flash-realtime`) is
**streaming half-duplex**: `server_vad` / `semantic_vad` turn detection plus
`response.cancel` for barge-in. The server aborts the model; the model does not choose
to yield. Opposite mechanism to full duplex.

Qwen shipped **no realtime code**. `web_demo.py` is one blocking `model.generate()`.
[Issue #99](https://github.com/QwenLM/Qwen3-Omni/issues/99) asked for realtime audio in
Oct 2025, was labelled `inactive`, and was never answered.

Official VRAM requirement for the bf16 model: **78.85 GB**. That is an A100-80 class card.

## 3. The finding: layer 24 is a bus boundary

In `config.json`:

```
thinker.text   2048 hidden, 48 layers, MoE 128e/8a, moe_intermediate 768
talker.text    1024 hidden, 20 layers, MoE 128e/6a, moe_intermediate 384
accept_hidden_layer: 24
position_id_per_seconds: 13     -> ~12.5 Hz, same clock as Moshi's Mimi
seconds_per_chunk: 2            -> current half-duplex granularity
```

The Talker is conditioned on **Thinker layer 24 of 48**, per token position, and never
sees layers 25–47. Depth is feed-forward, so the speech path is complete at half the
Thinker's depth.

Three more things already present:
- `_get_talker_user_parts()` projects the **user's audio** into the Talker. A listen
  channel exists; it is just fed the whole turn retrospectively.
- The codec vocabulary already has `codec_nothink_id: 2155`, `codec_think_bos_id`,
  `codec_pad_id` — a latent hold mechanism.
- Half-duplex is **orchestration**: `thinker.generate()` runs to completion, and only
  then does the Talker run. The conditioning itself is per-token.

This pattern repeated all project: the pieces for duplex are present; the wrappers are batch.

## 4. First measurement — the deadline

Synthetic weights at exact config geometry, 3000 frames (4 minutes of conversation),
deliberately pessimistic (bf16 = 3.6× the real byte traffic, uniform-random routing):

| condition | mean | p99.9 | over 80 ms |
|---|---|---|---|
| clean | 52.6 ms | 55.5 ms | 0 / 3000 |
| + PCIe streamer @ 23.6 GB/s | 62.0 ms | 64.4 ms | 1 / 3000 |

**Wrong prediction:** PCIe contention was expected to break the clock path, based on a
prior measurement that PCIe DMA steals DRAM bandwidth from cores. It cost **+18%**, not
an order of magnitude.

The measurement was also **overhead-bound, not bandwidth-bound**: 1998 MB/frame at
360 GB/s is 5.5 ms, so ~47 ms was Python and kernel-launch cost. That fact was noted at
the time and then not acted on for several more steps.

## 5. The vocoder — 24-second chunks

The output path ends at:

```python
code2wav.chunked_decode(talker_codes, chunk_size=300, left_context_size=25)
```

**300 codes at 12.5 Hz is 24 seconds of audio per call.** Full duplex needs
`chunk_size=1`. The source comment names the hazard: the causal transposed convolutions
"drop a fixed number of trailing samples per chunk."

Measured, sweeping chunk_size on identical codes:

| chunk_size | audio/chunk | samples kept | lost | ms/frame |
|---|---|---|---|---|
| 300 | 24 s | 479445 | — | 3.55 |
| 25 | 2 s | 474450 | 4995 | 2.33 |
| 1 | **80 ms** | 341250 | **138195 (28.8%)** | 22.45 |

Timing was never the problem — 22.45 ms against an 80 ms budget even doing 26×
redundant work. The problem is that **28.9% of every frame is discarded**.

The loss is exactly **545.8 samples (22.7 ms) per `forward()` call**, and it accounts
for itself: propagating the decoder blocks' right-trims through the upsample chain gives
3 + 12 + 60 + 480 = **555 samples**.

Crucially, what survives is **bit-accurate** (max deviation 0.0035, the bf16 noise
floor). Clean truncation, not corruption.

**Metric error made here:** the MSE and log-STFT columns in that sweep measure the
*time-shift* caused by dropped samples, not artefact severity. They looked like quality
degradation and were not.

## 6. The fix — stateful vocoder, and the bias trap

`chunked_decode` is stateless: each chunk re-runs `forward()` and throws away the conv
tail. The tail is real partial output that later inputs complete.

Stateful version: `CausalConvNet` prepends the previous call's trailing input frames
instead of zero-padding; `CausalTransConvNet` overlap-adds the trimmed tail
(`U[:s] += tail; emit U[:-s]; tail = U[-s:]`); the warm-up trim applies only on call 1.

**The bias trap.** All six transposed convs carry a bias. Two consecutive chunks both
cover the overlap region, so overlap-adding *biased* outputs counts the bias twice. The
convolution must run bias-free and the bias be added once to what is emitted.

That single bug: **max error 0.67 → 0.003**.

It nearly shipped. In bf16 the fixed version reads 0.073, which looks like plausible
precision noise. Only running float32 separated *precision* from *logic*.

Result: sample-exact against the batch path (229845 samples both), **9.43 ms/frame,
p99 9.68 ms** against an 80 ms budget, and ~2.5× faster than the broken path because no
left context is recomputed.

## 7. Two packaging traps

**transformers 5.x silently destroys this checkpoint.** 5.x expects *fused* expert
tensors; this checkpoint stores them *per-expert*. On 5.16.1 all 20 Talker layers'
experts are **randomly initialized with no error raised**. Qwen's own README recommends
transformers ≥ 5.2.0, which is exactly wrong for this AWQ build. 4.57.1 loads it
cleanly: 8037 tensors, 0 missing, 0 unexpected.

**compressed-tensors cannot run the quantized Thinker.** It attaches `quantized_forward`
to plain `nn.Linear` modules still holding `weight_packed`, giving
`AttributeError: 'Linear' object has no attribute 'weight'`. Reproduces with
`device_map="cpu"` on both 4.57.1 and 5.16.1 — **not** an accelerate/offload bug, which
was the first (wrong) diagnosis.

`run_compressed=False` is ruled out by arithmetic: the Thinker's experts are 29.0B
params = **58 GB dense bf16** against 61 GB of host RAM.

## 8. Running the int4 Thinker anyway

Keep the weights packed (15 GB) and dequantize per call — only 8 of 128 experts are
touched per token. Format read from the checkpoint, not assumed: symmetric int4,
group_size 32, no zero point.

Verified against the checkpoint's own metadata: packed (2048, 96) int32 → 96 × 8 = 768
columns matching `weight_shape`; scale (2048, 24) × 32 = 768; result mean −5.1e-05 with
**14 distinct values per 32-element group** (int4 allows 16).

**Also discovered: attention is quantized too.** 48 × (4 attention + 128 experts × 3) =
**18624**, exactly the packed tensor count. The Thinker's only dense weights are
embeddings, norms and `lm_head`.

## 9. First speech

5.74 s of audio, generated in 104 s on CPU.

Signal analysis: 83.3% of energy in 80–4000 Hz, envelope std/mean 0.85 (speech > 0.6,
steady noise < 0.3), 15.4% silent frames, DC −0.0001. Speech, not noise.

Confirmed by ear: *"audio sounds great... clear and clean."*

Later confirmed in text — the model said:

> "Running AI on your own hardware matters because it gives you greater control,
> privacy, and customization while reducing reliance on third-party services and
> ongoing subscription costs."

Coherent output confirms the dequantization is right across all 18624 tensors, not just
the spot-checked ones.

**Three built-in voices**, as codec tokens in `talker_config.speaker_id`: chelsie 2301,
ethan 2302 (default), aiden 2303. Measured F0 225 / 180 / 151 Hz. Because they are
tokens rather than speaker embeddings, there is **no zero-shot voice cloning** — a
fourth voice needs training.

## 10. Realtime, part 1 — the Python expert loop

Real-weights clock path, first measurement:

```
talker decode         41.95 ms   <- dominant
code predictor         4.02
streaming vocoder     10.79
TOTAL p99.9           65.89 ms   vs 80 ms
```

The Talker's 190M active params should read 380 MB and take **~1.05 ms** on a 3060.
41.95 ms is **40× off bandwidth-bound**.

Cause: transformers ships the Mixtral `SparseMoeBlock` — a Python loop over selected
experts, each iteration doing a `torch.where` and three small matmuls. At batch-1 decode
that is **~720 kernel launches per token** across 20 layers.

Fix: stack each layer's experts once, replace the loop with gather + bmm. Which is
exactly the `gather+gemv` TML described, reached from the same batch-1 constraint.

| | unfused | fused |
|---|---|---|
| talker decode | 41.95 / 47.09 | **17.36 / 19.84** |
| TOTAL | 56.76 / 65.89 | **31.79 / 36.85** |
| thinker headroom | 14.1 ms | **43.1 ms** |

**Trap found here:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, added to dodge
an OOM and left on, cost nothing on the mean (33.52 vs 31.79) but added a **68 ms p99.9
to the vocoder** — a stage the fusion does not touch. On a throughput benchmark it would
have looked free.

**Second trap:** fusing on the GPU holds the originals and the stacks simultaneously —
6.2 GB + 6 GB OOMs a 12 GB card. Fuse on the host, where there is 61 GB.

## 11. Realtime, part 2 — the int4 wall

Applying the same gather+bmm to the packed Thinker: **4.078 ms per MoE layer**, so
97.9 ms for 24 layers against 43.1 ms of headroom.

The layer's 8 active experts are 21 MB at int4 = 0.06 ms of bandwidth. Still **68× off**.
The fusion removed launch overhead and added a new cost: dequantization *materialises*
75 MB of dense bf16 weights and runs five elementwise passes over them — to avoid
reading 21 MB.

## 12. Realtime, part 3 — the Triton kernel

A real int4 kernel dequantizes **in registers inside** the matmul, never writing dense
weights. ~90 lines of Triton: each program owns a block of output rows for one selected
expert, streams packed words, expands nibbles, applies the group scale, accumulates.

Verified against the torch path: **rel error 2.5e-07**.

| | per layer | ×24 |
|---|---|---|
| torch gather + unpack + bmm | 4.078 ms | 97.9 ms |
| **Triton dequant-GEMV** | **0.639 ms** | **15.3 ms** |

**The llama.cpp question, settled.** Over the project this flipped three times:
"mandatory" → "not mandatory" → "needed for int4" → "not needed". The dividing line was
never framework. It is whether int4 dequantization happens **in memory or in registers**.

## 13. Where the experts live

Speed solved promoted memory to the binding constraint: `Thinker[0:24]` packed is 8.4 GB
and the fused bf16 Talker is 6.1 GB, against a 12 GB card.

| | thinker MoE | GPU memory | total clock path |
|---|---|---|---|
| A. all pinned | 15.3 ms | 8.4 GB | ~56 ms ✓ / memory ✗ |
| B. all streamed from host | 37.8 ms | 0.5 GB | ~79 ms ✗ / memory ✓ |

B measured at **24.7 GB/s effective** against a 25.9 GB/s PCIe ceiling — already at line
rate, so the only lever is transferring less.

**Partial residency** — pin some layers, stream the rest — is linear and needs no
hit-rate assumption. Each pinned layer costs **0.316 GB** and buys **0.94 ms**:

| N pinned | thinker MoE | thinker GB | + talker/vocoder | |
|---|---|---|---|---|
| 0 | 36.5 ms | 0.02 | 6.52 | fits |
| 8 | 28.9 ms | 2.83 | 9.33 | fits |
| **12** | **25.1 ms** | **4.24** | **10.74** | **fits** |
| 24 | 13.5 ms | 8.46 | 14.96 | over |

**N=12 working point:**

```
talker + MTP + vocoder    36.85 ms  (p99.9)
thinker MoE, 12/12        25.1  ms
thinker attention         ~4    ms
                          --------
                          ~66 ms   vs 80 ms budget
```

Chosen over an LRU expert cache at similar projected latency because it is
**deterministic** — static placement, so p99 tracks mean, with no routing-locality
gamble. A duplex clock path interleaves user-audio-ingestion frames with assistant-text
frames, which plausibly route to different expert sets and could thrash a cache.

## 14. Loose ends closed

**`trailing_text_hidden` is front-loaded.** Measured: 34 assistant text tokens against
129–160 codec frames, so text covers only **21–26%** of the frames. It is injected over
the first ~2.7 s and then persists in the Talker's KV cache while it keeps speaking on
`tts_pad_embed`. The `else: + tts_pad_embed` branch means a streaming Thinker that has
not yet produced the next token is *mechanically* supported. Whether quality holds when
text arrives at ~3.3 tokens/s instead of 12.5 is a **train/test distribution shift and
is untested**.

**The vocoder KV cache stays unbounded.** An attempt to trim it to the 72-frame sliding
window broke correctness — max err 0.26 / 0.77 / 0.99 at 120 / 200 / 400 frames against
0.003 untrimmed. `cache_position` fixes RoPE but the attention mask still treats retained
keys as positions 0..W-1. Reverted; the cost of leaving it is under 200 MB for a
5-minute conversation, and the correct fix is a real `SlidingWindowCache`.

**A decode "bug" that never existed.** The text decode appeared to return only `user`
for three attempts. The decoded string is multi-line (`user\n…\nassistant\n…`) and the
log filter used to read results only matched the first line. Two fixes were written
against a symptom created by the grep.

## 15. What remains

- Wire the clock path into one loop — the pieces are benchmarked separately, not yet run together.
- Silence-frame semantics: `codec_nothink_id` exists, but the model was never trained to emit "nothing" as a positive action.
- Dual-channel conversation data (Fisher-style) for interrupt and backchannel timing. No amount of engineering recovers this; it is the honest ceiling on doing this solo.

## Numbers worth keeping

| | |
|---|---|
| Official VRAM for this model, bf16 | 78.85 GB |
| What it runs on here | 12 GB |
| Vocoder frame loss, shipped wrapper at 80 ms | 28.9% |
| The bias bug's cost | max err 0.67 → 0.003 |
| Talker, Python expert loop → gather+bmm | 41.95 → 17.36 ms |
| Thinker MoE layer, torch → Triton int4 | 4.078 → 0.639 ms |
| `expandable_segments:True` cost, hidden in p99.9 | +68 ms |
| Final clock path | ~66 ms vs 80 ms budget |
| Times the llama.cpp answer flipped | 3 |
