# Design

How Qwen3-Omni is put together, why a full-duplex retrofit is tractable, and what is actually
unknown. Every claim below cites the source it came from.

## 1. What TML built, as far as they described it

Thinking Machines' `TML-Interaction-Small` (276B MoE, 12B active) is full-duplex through four
choices, per [their blog](https://thinkingmachines.ai/blog/interaction-models/):

1. **One interleaved stream at fixed granularity** — `input_0 output_0 input_1 output_1 ...`, each
   200 ms. The model emits every tick unconditionally; while you speak it emits silence. There is
   "no separate dialog management component" — interruption is next-token prediction.
2. **No encoders in the latency path** — audio via
   [dMel](https://arxiv.org/abs/2407.15835) (train-free binning of log-mel energies) into "a
   light-weighted embedding layer"; video as 40×40 patches through an hMLP; audio out via a flow
   head. All co-trained from scratch with the transformer.
3. **A background model** runs asynchronously for reasoning and tools while the interaction model
   "remains present throughout... and integrates background results."
4. **Serving built for tiny chunks** — persistent GPU buffers, and `gather+gemv` for MoE instead of
   grouped GEMM, because batch-1 decode has no batch to build a GEMM from.

Note (4) also required bitwise trainer/sampler alignment — their earlier
[determinism work](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/),
shipped here at <5% overhead. Behaviour like *when to interrupt* comes from on-policy RL, which is
impossible if trainer and sampler disagree about what the policy just did.

Nothing is reproducible from that post: no paper, no weights, no data recipe, no RL details.

## 2. What Qwen3-Omni already has

Verified in `config.json` and `transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py`.

```
thinker.text  2048 hidden, 48 layers, MoE 128e/8a, moe_intermediate 768, GQA 32q/4kv
talker.text   1024 hidden, 20 layers, MoE 128e/6a, moe_intermediate 384, GQA 16q/2kv
code_predictor 1024 hidden, 5 layers, 16 code groups          (MTP)
accept_hidden_layer: 24
position_id_per_seconds: 13
seconds_per_chunk: 2
```

**The tap.** The Talker consumes two things per token position:

```python
thinker_embed  = cat([hs[0]                    for hs in thinker_result.hidden_states])
thinker_hidden = cat([hs[accept_hidden_layer]  for hs in thinker_result.hidden_states])
```

Layer-0 embeddings and **layer-24 hidden states**. Nothing from layers 25–47. Since depth is
feed-forward, layer 24's activations cannot depend on layers above it — so the speech path is
complete at half the Thinker's depth. This is the single fact the whole design rests on, and it is
guaranteed by architecture rather than something that needs measuring.

**A listen channel already exists.** `_get_talker_user_parts()` projects the *user's* turn into the
Talker: multimodal (audio) positions through `hidden_projection` of layer-24 hiddens, text positions
through `text_projection` of embeddings. The Talker already ingests user audio — it is simply fed
the whole turn retrospectively rather than frame by frame.

**Silence tokens already exist.** The codec vocabulary carries `codec_nothink_id: 2155`,
`codec_think_bos_id`, `codec_think_eos_id`, `codec_pad_id` — a latent hold mechanism.

**Half-duplex is orchestration, not architecture.** The reference `generate()` does:

```python
thinker_result = self.thinker.generate(...)   # runs to completion
...                                           # only then
talker_result = self.talker.generate(...)
```

The conditioning itself is per-token, so this is a sequencing choice, not a structural constraint.

## 3. The split that follows

Layer 24 is a **bus boundary**, not a VRAM boundary — the distinction matters. The background half
does not need to fit anywhere; it is asynchronous by construction, so it can live entirely in host
RAM and absorb any PCIe latency. What must never fault is the clock path.

| | components | policy |
|---|---|---|
| clock | AuT → Thinker[0:24] → Talker → MTP → Code2Wav | pinned VRAM, ~10 GB @ Q4 (128 experts), 80 ms deadline |
| background | Thinker[24:48] | RAM-resident, deferred batched bursts |

A correction worth stating plainly: layers 25–47 **cannot be skipped outright**. The Thinker needs
all 48 to produce a text token — the LM head is at the end. What is true is the *timing* split.
Layers 0–24 run every frame to keep the Talker fed; 25–47 run in deferred bursts only when a new
text token is needed, roughly every 4–6 frames (speech is ~2–3 words/s against a 12.5 Hz grid).
Those bursts are prefill-shaped — multiple positions at once — which is exactly the case where
expert prefetch pays off.

Consequence: **the clock path needs no expert cache.** A cache exists to hide a fetch you cannot
afford; a fully pinned path never fetches.

## 4. Measured — clock-path deadline

RTX 3060 12 GB, synthetic weights at exact config geometry, 3000 frames (4 minutes of conversation
at 12.5 Hz). Two deliberate pessimisms: bf16 experts (9.4 MB) rather than Q4 (2.53 MB), giving
**3.6× the real byte traffic**; and uniform-random routing, the worst case for expert locality.

| condition | mean | p99.9 | max | over 80 ms |
|---|---|---|---|---|
| clean | 52.6 ms | 55.5 ms | 55.7 ms | 0 / 3000 |
| + PCIe streamer @ 23.6 GB/s | 62.0 ms | 64.4 ms | 95.6 ms | 1 / 3000 |

Resident 8.74 GB. **Passes.**

Two caveats that matter more than the pass:

- **The measurement is overhead-bound, not bandwidth-bound.** 1998 MB/frame at the 3060's ~360 GB/s
  is 5.5 ms, so ~47 ms is Python and kernel-launch cost across 119 sequential layer forwards. These
  numbers are a *floor*; a real implementation has more headroom, not less. Corroborated by the
  8-expert run coming in at 48.9 ms against 52.6 ms for 32 experts on identical per-frame traffic —
  that gap is memory pressure, not bytes.
- **Contention cost +18%, not the order of magnitude predicted.** Prior measurements on this machine
  showed PCIe DMA stealing DRAM bandwidth from cores; the prediction was that a background MoE
  streaming over the same bus would break the clock path. It did not. Deadline workloads tolerate
  the contention this architecture creates.

The single 95.6 ms outlier is the real engineering problem — one frame per 4 minutes is one audible
glitch per 4 minutes. Almost certainly allocator or scheduling jitter; CUDA graphs and a priority
stream are the fix.

## 5. Measured — the vocoder thesis

The output path ends at:

```python
talker_wavs = self.code2wav.chunked_decode(talker_codes, chunk_size=300, left_context_size=25)
```

**300 codes at 12.5 Hz is 24 seconds of audio per vocoder call.** Full duplex needs
`chunk_size=1`. The source comment names the hazard:

> A chunk of `n` codes yields slightly fewer than `n * total_upsample` samples, because the causal
> transposed convolutions drop a fixed number of trailing samples per chunk

That loss is currently paid once per 24 s. At `chunk_size=1` it is paid *every frame*. And
`left_context_size=25` means 2 s of context recomputed per 80 ms frame.

### Result: timing passes, `chunked_decode` fails

code2wav is unquantized in the AWQ build (230 tensors, 0 packed, one shard), so it loads standalone
on the GPU in 0.4 GB — which also sidesteps the compressed-tensors/accelerate offload bug that
blocks loading the full 30B model on 12 GB.

**Compute is not the constraint.** At `chunk_size=1`, decode costs **22.45 ms/frame against an
80 ms budget** — despite doing 26× redundant work (26 codes processed to keep 1). Dropping
`left_context_size` to 0 takes it to ~10 ms/frame. Comfortable either way.

**But `chunked_decode` is structurally lossy.** It is stateless: each chunk re-runs `forward()`
and the causal transposed convolutions lose a fixed **545.8-sample (22.7 ms) tail** per call. One
tail per chunk:

| chunk_size | chunk | samples kept | lost vs 24 s baseline | ms/frame |
|---|---|---|---|---|
| 300 | 24 s | 479445 | — | 3.55 |
| 25 | 2 s | 474450 | 4995 | 2.33 |
| 5 | 400 ms | 452250 | 27195 | 5.09 |
| 1 | **80 ms** | 341250 | **138195 (28.8%)** | 22.45 |

At 80 ms frames, **28.9% of every frame is discarded**. Critically, what survives is *bit-accurate*
— `max|cs1 − ref|` over the kept 1365 samples is 0.0035, the bf16 noise floor. So this is clean
truncation, not corruption, and the MSE/log-STFT columns in the sweep are measuring the resulting
time-shift rather than artefact severity. They should be ignored.

`left_context_size` does not affect the loss at all (the tail is the same either way); trimming it
is purely a compute optimisation.

### What that implies

`chunked_decode` is a batch convenience wrapper, never intended for streaming, and no parameter
choice makes it viable at frame granularity. The fix is a **stateful vocoder**: carry the causal
convolutions' tail state across calls instead of discarding and recomputing it. This is standard
streaming-convolution practice and what Mimi/Moshi already does — and it is almost certainly what
Alibaba's Realtime API uses internally, since the shipped wrapper could not support it.

Consequence: it also removes the 26× redundancy, because a stateful decoder needs no left context
at all. Expect well under 10 ms/frame.

## 6. Still unknown after that

- Talker behaviour at 80 ms granularity when it has only ever run at `seconds_per_chunk: 2`
- Quality when the Thinker's back half is deferred rather than synchronous
- Silence-frame semantics: the tokens exist, but the model was never trained to emit "nothing" as a
  positive action
- Interrupt and backchannel behaviour, which needs dual-channel conversation audio (Fisher-style)
  and is the one gap no amount of architecture recovers

## 7. Why not llama.cpp yet

llama.cpp lists Qwen3-Omni with `Capabilities: audio input, vision input` and no audio output. The
Talker has been converted to GGUF by third parties (`talker-f16.gguf` ~6.2 GB) but is explicitly not
integrated; MTP and Code2Wav are absent entirely. There is also no way to expose an intermediate
layer's hidden states as a routable output, which the tap requires.

It is nonetheless the destination. The benchmark says we are overhead-bound, and llama.cpp is what
removes that overhead. The order is: prove the architecture in transformers, then port. Doing it in
ggml first means weeks of C++ to answer a question a Python script answers in an afternoon.

## 6. Solved — the stateful vocoder

`src/duplex/streaming/code2wav.py` wraps `Qwen3OmniMoeCode2Wav` in place and makes its conv stack
stateful. Two kinds of state, installed by walking the module tree:

- **`CausalConvNet`** (stride 1) zero-pads `padding` frames on the left every call. Streaming
  prepends the previous call's trailing `padding` input frames instead.
- **`CausalTransConvNet`** (kernel = 2·stride) emits `(L-1)s + k` samples then trims `s` from each
  end. The right trim is the region later inputs still contribute to, so streaming overlap-adds it:
  `U = conv(x); U[:s] += tail; emit U[:-s]; tail = U[-s:]`. The left trim is warm-up, applied only
  on the first call.

**The bias trap.** All six transposed convs carry a bias. Two consecutive chunks both cover the
shared region, so overlap-adding biased outputs counts the bias twice. The convolution must run
bias-free and the bias be added once to what is emitted. Getting this wrong cost `max err 0.67`;
fixing it dropped that to `0.073` (bf16) / `0.003` (float32).

### Verified (`duplex stream-verify`, 120 frames)

| path | samples | /frame | loss | max err |
|---|---|---|---|---|
| reference (unchunked `forward`) | 229845 | 1915.4 | 0.2% | — |
| shipped `chunked_decode(cs=1)` | 163800 | 1365.0 | **28.9%** | 1.111 |
| **streaming (stateful)** | **229845** | **1915.4** | **0.2%** | **0.003** (fp32) |

Sample-exact against the batch path, and the residual 0.003 is float32 rounding, not logic — it
grows smoothly with stream length (0.0020 @40 → 0.0054 @200 frames) with no discontinuity at the
`sliding_window: 72` boundary, while mean error stays flat at ~0.00004. At −46 dB it is inaudible.

The `/frame` figure converging on 1920 (1906 @40 → 1917 @200) confirms the remaining 0.2% is a
one-time warm-up amortising over the stream, not a per-frame loss.

**Latency: 9.43 ms/frame mean, p99 9.68 ms**, against an 80 ms budget — 8× headroom, and ~2.5×
faster than the broken `chunked_decode(cs=1)` path because no left context is recomputed.

### Known limitation

The `pre_transformer` KV cache grows unbounded. Masking is correct, but a real conversation runs far
past 72 frames, so a windowed cache is needed to bound memory and attention cost. Not a correctness
issue; an efficiency one.
