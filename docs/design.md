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

**No silence mechanism exists — an earlier claim here was wrong.** The codec vocabulary
carries `codec_nothink_id: 2155`, `codec_think_bos_id`, `codec_think_eos_id`,
`codec_pad_id`, and this document previously called that "a latent hold mechanism". It
is not. They are used in exactly one place, as a fixed 6-token preamble prepended once
to the Talker's codec stream:

```python
codec_special_tokens = [[codec_nothink_id, codec_think_bos_id, codec_think_eos_id,
                        speaker_id, codec_pad_id, codec_bos_id]]
```

Read in order that is a mode-and-voice header — no-thinking, empty think span, which
voice, pad, begin — the codec-stream analogue of Qwen3's `/no_think`. It appears at
position 0 and is never emitted during generation. `codebook_size` is 2048 and these ids
are 2148-2157, **outside the audio codebook**, so they cannot decode to sound at all.

The Talker therefore has no way to emit *nothing* for a frame. Silence would have to be
ordinary codec tokens decoding to near-zero audio — learnable, but nothing in the
checkpoint indicates it was learned.

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

## 8. Manual hybrid device placement fights transformers

`place_unpacked()` implements the MoE-offload split — packed experts in host RAM,
everything dense on the GPU, packed tensors shipped per call so activations never
move. It works, and the footprint is right:

```
code2wav              0.40 GB
talker                6.59 GB   (3325M params)
thinker text stack    7.20 GB   (324M dense moved, 4671M packed left in host RAM)
```

4.4 GB spare on a 12 GB card, with ~15 GB of int4 expert weight never resident.

**But `generate()` does not survive it.** transformers assumes uniform placement and
builds helper tensors accordingly. Failures appear one line at a time, each ~15 min
of load away:

- `inputs` left on CPU while embeddings are on GPU
- `thinker.lm_head` sits outside `thinker.model`, so a submodule-scoped placement misses it
- `modeling_qwen3_omni_moe.py:3993` builds `talker_special_tokens` on the CPU and feeds
  it a GPU embedding — library code, not ours

The module map that would have saved three of those cycles is derivable from
`model.safetensors.index.json` without loading anything:

| thinker submodule | packed | dense |
|---|---|---|
| `model` | 18624 | 242 |
| `lm_head` | 0 | 1 |
| `audio_tower` | 0 | 525 |
| `visual` | 0 | 351 |

Note this also shows **attention is quantized**: 48 × (4 attention + 128 experts × 3)
= 18624, exactly the packed count. The only dense weights in the thinker are
embeddings, norms and `lm_head`.

**Conclusion:** hybrid placement needs a real `device_map` through accelerate, not
manual `.to()` calls. For the offline quality gate — which is latency-insensitive —
CPU-only is the right tool, and `--cpu` selects it. The GPU split belongs with the
llama.cpp port, where `--n-cpu-moe` already does this properly.

## 9. Realtime — fused MoE, and llama.cpp is not mandatory

Section 8 concluded transformers could not make the 80 ms deadline. That was wrong:
it was true of transformers' *expert loop*, not of PyTorch.

The shipped `SparseMoeBlock` is the Mixtral implementation — a Python loop over the
selected experts, each iteration doing a `torch.where` and three small matmuls. At
batch-1 decode that is ~720 kernel launches per Talker token across 20 layers, for
190M active params that should read 380 MB and take ~1.05 ms on a 3060. Measured
41.95 ms: **~40x overhead-bound**.

`duplex.streaming.fused_moe` stacks each layer's experts once and replaces the loop
with a gather plus batched matmul — the "gather+gemv strategy for MoE kernels
instead of the standard grouped gemm" Thinking Machines described, arrived at from
the same constraint.

| stage | unfused mean / p99.9 | **fused** mean / p99.9 |
|---|---|---|
| talker decode | 41.95 / 47.09 | **17.36 / 19.84** |
| code predictor (MTP) | 4.02 / 4.58 | 3.91 / 4.28 |
| streaming vocoder | 10.79 / 16.58 | 10.52 / 15.57 |
| **TOTAL** | 56.76 / 65.89 | **31.79 / 36.85** |
| headroom for Thinker[0:24] | 14.1 ms | **43.1 ms** |

RTX 3060, 200 frames, real weights for everything but the Thinker. 0/200 over budget.

Whether the Thinker fits 43.1 ms: it is ~1.13B active params/token (226M attention +
906M experts) over 24 layers at 2x the Talker's hidden size, so scaling the fused
Talker's 17.36 ms suggests ~25-35 ms. Tight but plausible — and it needs the same
fusion over `weight_packed`, which is an int4-aware gather rather than a bf16 one.

### Two traps

**Fuse on the host, not the GPU.** Stacking on-device holds the originals and the
stacks at once — 6.2 GB of talker experts plus 6 GB of stacks OOMs a 12 GB card.
Host RAM has 61 GB and the stacked model is the same size as the original.

**`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is poison for deadline work.**
Added to dodge the OOM above and left on, it cost nothing on the mean (33.52 vs
31.79) but added a **68 ms p99.9 to the vocoder** — a stage the fusion does not
touch — pushing TOTAL p99.9 to 90.43 and failing the budget. On a throughput
benchmark it would have looked free.

## 10. The int4 wall — where a real quantized kernel becomes necessary

Section 9's fusion works for bf16. Applying the same gather+bmm to the packed
Thinker (`fused_packed_moe.py`, gather -> `unpack4` on GPU -> scale -> bmm) does not
get there.

The batched unpack is verified correct — bit-identical (`max|diff| = 0.000e+00`) to
the 2-D `dequantize_weight` path across gate_proj (768×2048), down_proj (2048×768)
and q_proj (4096×2048), and consistent across a broadcast batch dim. The problem is
cost, not correctness:

```
per fused packed MoE layer   4.078 ms mean, 4.118 p99
x24 layers                   97.9 ms
+ attention (~20% more)     ~122 ms
headroom available            43.1 ms
```

**Why 4.078 ms is slow.** That layer's 8 active experts are 37.7M params = 21 MB at
int4, which is 0.06 ms of bandwidth on a 3060. We are 68x off. The fusion removed
the *launch* overhead but added a new one: dequantization materialises full dense
weights (37.7M params -> 75 MB bf16) and runs five separate elementwise passes over
them — shift, mask, subtract, reshape, scale — before the matmul begins. That is
~375 MB of traffic per layer to avoid reading 21 MB.

A real int4 kernel (Marlin, or llama.cpp's quantized GEMV) dequantizes **in registers
inside** the matmul and never writes dense weights to memory. That cannot be
expressed in PyTorch ops; it is a CUDA/Triton kernel.

### Where that leaves realtime

| stage | dtype | PyTorch fusion sufficient? |
|---|---|---|
| Talker, code predictor | bf16 | **yes** — 41.95 -> 17.36 ms |
| streaming vocoder | bf16 | yes — 10.5 ms |
| Thinker[0:24] | **int4** | **no** — needs a fused dequant-GEMV |

Two prior claims in this document were each half right. "transformers cannot make the
deadline, llama.cpp is mandatory" was wrong for the bf16 stages — a gather fixed
those. "llama.cpp is not mandatory" was wrong for the int4 stage. The dividing line
is dtype, not framework.

There is also a memory constraint independent of speed: `Thinker[0:24]` packed is
~8.2 GB (7.25 GB nibbles + 0.9 GB fp16 scales) and the fused bf16 talker is 6.1 GB.
Both do not fit a 12 GB card, so a full-clock-path system needs the talker quantized
too, or the thinker's experts streamed from host RAM per token (453 MB/token over
PCIe at 25.9 GB/s = 17.5 ms, which the 43.1 ms headroom would absorb).

### Options, in increasing order of work

1. **Marlin via vLLM** — the checkpoint is already compressed-tensors format, which
   vLLM serves with Marlin kernels. Does not expose layer-24 hidden states.
2. **A Triton dequant-GEMV** — fuse unpack+scale+matmul into one kernel. Self-contained
   and keeps everything else as-is.
3. **llama.cpp** — has the kernels, lacks the Talker, MTP, Code2Wav and the tap.

## 11. Solved — Triton dequant-GEMV, and llama.cpp is not required

`triton_dequant_gemv.py` does the unpack in registers during accumulation: each
program owns a block of output rows for one selected expert, streams packed words,
expands nibbles, applies the group scale, and multiplies into an accumulator. Dense
weights never reach memory.

Verified against the torch path: **rel error 2.5e-07**, pure fp32 rounding.

| | per Thinker MoE layer | x24 layers |
|---|---|---|
| torch gather + unpack + bmm | 4.078 ms | 97.9 ms |
| **Triton dequant-GEMV** | **0.639 ms** | **15.3 ms** |

6.4x, against 43.1 ms of headroom. Estimated full clock path:

```
talker + MTP + vocoder (fused bf16)   36.85 ms p99.9
Thinker[0:24] MoE (triton int4)       15.3 ms
Thinker[0:24] attention (~25% of MoE)  ~4 ms
                                      -------
                                      ~56 ms  vs 80 ms budget
```

**This settles the llama.cpp question.** The dividing line was never framework, it was
whether int4 dequantization happens in memory or in registers. A ~90-line Triton
kernel moves it into registers and PyTorch is sufficient end to end.

### The binding constraint is now memory, not speed

```
Thinker[0:24] packed   8.4 GB   (0.35 GB/layer x 24)
fused bf16 Talker      6.1 GB
                      ------
                      14.5 GB  vs a 12 GB card
```

Two ways out:

1. **Quantize the Talker too.** It ships bf16 in this build; int4 would take it from
   6.1 GB to ~1.6 GB, giving 10.0 GB total. The Triton kernel already handles the
   format, so the Talker would also get faster.
2. **Stream the Thinker's experts from host RAM.** 8 experts x 24 layers x 2.36 MB =
   453 MB/token, at 25.9 GB/s measured PCIe = 17.5 ms — which the ~24 ms of remaining
   headroom absorbs, though it eats most of it.

Option 1 is better and is the natural next step: it reduces memory *and* latency,
and reuses work already done.

### Remaining optimisations not yet taken

- The down-projection currently issues one kernel per selected expert (8 launches per
  layer) because each expert has its own activation vector. Batching those into one
  kernel is straightforward and worth roughly another 2x on that half.
- Attention's four packed linears per layer still use the torch dequant path and
  should move to the same kernel.
- No CUDA graphs yet on the clock path.

## 12. Where the experts live — measured

Section 11 solved speed and promoted memory to the binding constraint. Three
placements for `Thinker[0:24]`'s packed experts, all measured on the 3060 with the
Triton kernel:

| | thinker MoE (24 layers) | GPU memory | total clock path |
|---|---|---|---|
| **A. pinned** | 15.3 ms | 8.4 GB | ~56 ms ✓ / memory ✗ |
| **B. streamed from host** | 37.8 ms | ~0.5 GB | ~79 ms ✗ / memory ✓ |
| **C. LRU cache @ 90% hit** (projected) | ~17.6 ms | tunable | ~58 ms ✓ / memory ✓ |

### B, measured

```
per-layer transfer, 8 experts   23.6 MB   -> 566 MB/token
transfer only        0.954 ms   -> 24.7 GB/s effective
full layer           1.575 ms   (vs 0.639 pinned)
GPU staging          0.022 GB   (vs 0.35 GB) -- 16x less
```

24.7 GB/s is essentially the machine's measured 25.9 GB/s PCIe ceiling, so the
transfer is at line rate and cannot be made faster — only smaller. It is 61% of the
per-layer time.

Totalling honestly: 36.85 ms (talker+MTP+vocoder, p99.9) + 37.8 ms (streamed MoE,
mean) + ~4 ms attention = **~78.7 ms against 80 ms**. That mixes a p99.9 against a
mean for a 1.3 ms margin, so B should be read as a **practical fail** — its own p99.9
will exceed budget.

### Why C is the answer

The only remaining lever is transferring less, and an LRU expert cache is exactly
that. At the 90% hit rate measured previously on this machine, per-layer transfer
falls to ~0.095 ms, giving ~0.73 ms/layer and ~17.6 ms for 24 layers — near pinned
speed at a fraction of the memory, with cache size as a tunable dial.

C is also the only configuration that does sustained fetching *during* the clock
path, so it is the one where the PCIe-DMA-steals-DRAM-bandwidth effect actually
applies. That was measured at +18% for a deadline workload, which ~17.6 ms absorbs.

## 13. Partial residency — the working configuration

Neither extreme works: pinning all 24 layers needs 8.4 GB and busts the card;
streaming all 24 costs 37.8 ms and busts the budget. Splitting them is linear and
tunable, and needs no hit-rate assumption:

| N pinned | thinker MoE | thinker GB | + talker/vocoder (6.5 GB) | |
|---|---|---|---|---|
| 0 | 36.5 ms | 0.02 | 6.52 | fits |
| 4 | 32.7 ms | 1.43 | 7.93 | fits |
| 8 | 28.9 ms | 2.83 | 9.33 | fits |
| **12** | **25.1 ms** | **4.24** | **10.74** | **fits** |
| 14 | 23.1 ms | 4.94 | 11.44 | over |
| 24 | 13.5 ms | 8.46 | 14.96 | over |

Each pinned layer costs 0.316 GB (fp16 scales) and buys 0.94 ms.

**N=12 is the working point:**

```
talker + MTP + vocoder    36.85 ms  (p99.9)
thinker MoE, 12/12        25.1  ms
thinker attention         ~4    ms
                          --------
                          ~66 ms   vs 80 ms budget, 14 ms margin
```

This is preferable to an LRU expert cache (section 12's config C) despite similar
projected latency, because it is **deterministic**. Placement is static, so cost does
not depend on routing locality — and a duplex clock path interleaves user-audio
ingestion frames with assistant-text frames, which plausibly route to different
expert sets and could thrash a cache. p99 tracks mean here because nothing is
data-dependent.

KV growth does not threaten the margin: thinker 24.5 KB/token + talker 10 KB/token is
under 200 MB for a 5-minute conversation.

N=8 (28.9 ms, 9.33 GB, ~70 ms total) trades 4 ms for 1.4 GB if more headroom is wanted.
