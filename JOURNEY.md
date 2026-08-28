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
- ~~The codec vocabulary already has `codec_nothink_id` — a latent hold mechanism.~~
  **Wrong, corrected later.** Those tokens are a fixed 6-token preamble
  (`nothink, think_bos, think_eos, speaker, pad, bos`) prepended once — a mode-and-voice
  header, not a per-frame silence mechanism. They are ids 2148-2157 against a
  `codebook_size` of 2048, so they are outside the audio codebook and cannot decode to
  sound. Inferred from token names without reading the usage, then repeated in four
  documents before being checked.
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
- Silence-frame semantics. There is **no** existing mechanism — see the correction in section 3. Silence must be ordinary codec tokens decoding to near-zero audio, and must be learned.
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

## 16. End to end on real speech

`duplex speak --frame-decode` — generate, capture the Talker's real codec codes,
re-decode them one 80 ms frame at a time through the stateful vocoder, compare against
the batch path. Every previous vocoder measurement used synthetic codes.

24.03 s of speech, 301 frames:

```
batch      576810 samples
streaming  577365 samples      delta +555
per-frame  mean 10.51 ms   p99 16.19 ms   max 25.39 ms   vs 80 ms budget
realtime   7.6x faster than playback
```

**The +555 is the prediction landing.** Section 5 derived 555 samples analytically from
the decoder blocks' right-trims (3 + 12 + 60 + 480). Here the streaming path emits
exactly 555 more samples than the batch path, on real content — it keeps the tail the
batch decoder discards.

Agreement is essentially exact where the two paths should agree:

```
mean|diff|                    0.000031
median per-80ms-frame error   0.00006      (-77 dB)
worst frame                   0.04843  at frame 299/300  <- the last one
```

299 of 300 frames match to 6e-05. The whole discrepancy is the final frame, which is
where the implementations differ by design. An initial reading of the headline
`max|diff| = 0.052` as "17x worse on real codes than synthetic" was wrong — it is a
boundary effect, not distributed degradation.

Text generated across those 24 seconds:

> "Running a language model on your own hardware gives you full control over data
> privacy, customization, and deployment, but requires significant technical expertise
> and computational resources. In contrast, using an API offers convenience,
> scalability, and lower upfront costs, but limits customization and may involve data
> privacy concerns since inputs are processed externally."

## 17. How the duplex training would actually work

Researched rather than assumed. Moshi's recipe ([kyutai.org/Moshi.pdf](https://kyutai.org/Moshi.pdf)):

```
pre-training    unsupervised audio       1016 H100s / 127 DGX nodes
post-training   simulated multi-stream via PyAnnote diarization, 100K steps, 8h batch
full-duplex     Fisher (2000h dual-channel), 10,000 steps
instruction FT  synthetic
```

**The full-duplex stage is 10,000 steps.** The 1016 H100s went into pre-training, which
is skipped here — the Talker is already trained. What is needed is the last-mile
behaviour fine-tune on a 3B model with ~230M active params.

**Dual-channel data does not require the Fisher licence.** Moshi's own post-training
synthesised multi-stream data by diarizing ordinary single-channel conversation with
PyAnnote and splitting each speaker into a channel. Two-person podcasts and interviews
are unlimited free source material, and the diarization bleed during overlap is exactly
the interruption signal. Fisher was the final polish, not the foundation. A 15-hour open
dual-track set ([arXiv 2509.04093](https://arxiv.org/html/2509.04093), EN + ZH,
speaker-isolated) works as clean validation; CANDOR as OOD eval.

**Possible shortcut, unverified:** [SoulX-Duplug](https://arxiv.org/pdf/2603.14877)
trains a small external "streaming state prediction module" for speak/listen decisions
instead of fine-tuning the model. If that transfers to a Thinker/Talker split it is
consumer-GPU-sized work. Not yet read closely.

**Silence is already producible.** Generated samples measure 15.4% and 10.1% silent
frames — the Talker already emits codec tokens that decode to near-zero audio during
pauses. What is missing is *control* (choosing silence for an unbounded stretch because
someone else is speaking), not *representation*.

This corrects a claim repeated throughout the project that dual-channel data was an
immovable blocker. It is neither immovable nor, apparently, expensive.

## 18. Testing the assumption: is duplex behaviour already latent?

The whole project assumed Qwen3-Omni cannot do duplex without training. That was
never tested, and testing it is cheap.

The Talker takes one conditioning vector per frame:

```python
if generation_step < trailing_text_hidden.shape[1]:
    inputs_embeds += trailing_text_hidden[:, generation_step]
else:
    inputs_embeds += tts_pad_embed
```

Real text runs out around frame 34, so extending `trailing_text_hidden` with the
user's audio hiddens — through the Talker's own `hidden_projection`, the same path
`_get_talker_user_parts` uses — tells the model "a user is speaking" at every frame
past that point. No code change, no training.

Three conditions, same prompt, same seed, **the same 150-frame cap** so length cannot
be confused with quietness:

| | mean | silent | vs baseline |
|---|---|---|---|
| baseline (pad) | −31.9 dB | 13.0% | — |
| user hiddens | −38.0 dB | 20.4% | **−6.1 dB** |
| **shuffled hiddens** | **−39.6 dB** | **25.0%** | **−7.7 dB** |

**No latent duplex.** The model does quiet down under injected conditioning, but
*time-shuffled* conditioning suppresses it slightly **more** than real speech. It is
reacting to "this is not the pad embedding I expect", not to the content. Duplex
behaviour must be trained.

The `shuffled` arm is the entire result. Without it, −6.1 dB and 20.4% silence would
have looked like a discovery.

Two process notes: the first two attempts of this probe were unreadable. One died
when the host rebooted mid-run (silently — no traceback, initially misattributed to
print buffering). The second had no frame cap, so the injected condition simply
generated for 42 minutes against baseline's 6, making "went quiet" indistinguishable
from "ran longer".

## 19. The full clock path, in one loop

Every realtime number until now was measured per stage. Assembling them exposed two
bugs that isolation had hidden.

**First run** (8 pinned / 16 streamed):

```
thinker 245.79   talker 17.28   mtp 3.89   vocoder 10.54   total 277.51
p99.9 288.93 vs 80 ms -> FAIL, 150/150 over
```

Talker, MTP and vocoder matched their isolated benchmarks to within 0.1 ms. The
thinker was 7x over prediction. Running it alone attributed the cost:

```
24 pinned, no streaming   230.86 ms
 8 pinned, 16 streamed    245.79 ms   -> streaming = 0.93 ms/layer, matching config B
```

So streaming was fine. Two things were not.

**Bug 1: attention on the torch dequant path.** Documented in the code as "4
linears/layer, ~7 MB packed each -- not worth a kernel yet". That sized it by
*packed* bytes; cost is set by *dequantized* work. q and o are 4096x2048, so
attention materialises 452M params/token to dense across 96 calls — ~215 ms of the
230. Moving it onto the Triton kernel (a Linear is the E=1 case of the expert
gather): **230.86 -> 110.86 ms**.

**Bug 2: the Triton kernel was never called.** `FusedPackedMoE.forward` used
`_dequant`, the torch path, throughout. The kernel had been written, verified at rel
error 2.5e-07, benchmarked at 6.4x, documented as the solution — and not wired in.
24 x 4.078 = 97.9 ms, which is exactly the "unexplained" remainder that had been
attributed to launch overhead. **110.86 -> 27.67 ms.**

While fixing it, the kernel gained per-expert activations (`s_x` stride) so the
down-projection is one launch rather than eight: two launches per layer, down from
nine.

**Bug 3: the vocoder's growing cache.** Full path then passed on the mean (71.92) but
failed p99.9 at 85.38, and the tail was isolated to the vocoder: 22.88 p99.9 against
a 10.22 mean while every other stage was tight. Cause was a `DynamicCache`
reallocating and copying every frame — O(n^2) over a stream. Preallocating once:
**22.88 -> 13.07 p99.9**, correctness unchanged (max err 0.00608).

**Final, all stages real, one loop, RTX 3060:**

| stage | mean | p99.9 |
|---|---|---|
| thinker (8 pinned / 16 streamed) | 40.95 | 44.45 |
| talker | 17.11 | 19.35 |
| code predictor | 3.85 | 4.45 |
| streaming vocoder | 11.06 | 13.07 |
| **total** | **72.96** | **77.16** |

**PASS: 0/150 frames over an 80 ms budget, 10.24 GB resident, 1.10x realtime.**

Margin is 2.8 ms (3.5%), with synthetic conditioning — real routing could move the
thinker. CUDA graphs and the pinned/streamed split are both untouched levers.

The pattern across all three bugs: a component measured in kinder conditions than it
runs in. That is what the loop was built to catch, and it caught three.
