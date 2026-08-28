"""GPU engine: the real generate() path, with every packed matmul on the Triton kernel.

The slow path runs the whole model on CPU through the torch dequant, at roughly 15x
slower than realtime. Everything needed to fix that already exists in this repo and
was only ever measured in benchmarks:

    FusedPackedMoE + Triton dequant-GEMV   thinker MoE, 4.078 -> 0.639 ms/layer
    host_resident mode                     partial residency, 0.316 GB vs 0.022 GB
    TritonPackedLinear                     attention, ~215 ms -> ~8 ms
    fuse_moe_blocks                        talker bf16, 41.95 -> 17.36 ms/token
    StreamingCode2Wav                      vocoder, sample-exact at 80 ms frames

An earlier attempt at hybrid CPU/GPU placement failed because transformers assumes a
single device and builds helper tensors accordingly (`modeling_qwen3_omni_moe.py:3993`
hands a CPU tensor to a GPU embedding). The fix is not to fight that: put *everything*
on the GPU. Host-resident MoE blocks keep their packed experts in pinned host RAM
internally, which is invisible to transformers — the module reports as a GPU module.

The vision tower is dropped: it is 0.5 GB of weights this pipeline never touches.
"""

from __future__ import annotations

import gc
import time

import numpy as np
import torch


class FastEngine:
    """Drop-in replacement for duplex.server.Engine, on the GPU."""

    def __init__(self, model_path: str, speaker: str = "Ethan", n_pinned: int = 4,
                 cpu: bool = False, talker_q8: bool = True,
                 cpu_moe: bool = False, compile_mode: str | None = None):
        self.model_path = model_path
        self.speaker = speaker
        self.n_pinned = n_pinned
        self.talker_q8 = talker_q8
        self.cpu_moe = cpu_moe
        # Three micro-optimisations each landed at ~2% (launch count 1.5%, the
        # device->host sync 1.9%). The remaining ~168 ms/token is diffuse Python
        # dispatch across ~930 ops, which only disappears if the dispatch itself
        # does. torch.compile("reduce-overhead") wraps CUDA graphs.
        self.compile_mode = compile_mode
        self.model = None
        self.proc = None
        self.status = {"ready": False, "phase": "not started", "elapsed": 0.0,
                       "detail": ""}
        self._t0 = None

    # ---- status plumbing, matching Engine ----------------------------------
    def start_background_load(self):
        import threading
        self._t0 = time.time()
        self.status.update(phase="starting", detail="spawning loader")
        threading.Thread(target=self._load_guarded, daemon=True).start()

    def _set(self, phase, detail=""):
        self.status.update(phase=phase, detail=detail,
                           elapsed=round(time.time() - (self._t0 or time.time()), 1))

    def _load_guarded(self):
        try:
            self.load()
            self.status.update(ready=True, phase="ready",
                               elapsed=round(time.time() - self._t0, 1))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status.update(ready=False, phase="failed",
                               detail=f"{type(e).__name__}: {e}")

    # ---- load ---------------------------------------------------------------
    def _talker_cache_path(self):
        from pathlib import Path
        return Path(self.model_path) / "talker_q8_fused.safetensors"

    def _save_talker_cache(self, talker):
        """Persist the fused+quantised talker so restarts skip the work."""
        from safetensors.torch import save_file
        sd = {}
        for name, mod in talker.named_modules():
            if type(mod).__name__ == "FusedMoE":
                for attr in ("gate_up", "down", "gate_up_s", "down_s"):
                    t = getattr(mod, attr, None)
                    if t is not None:
                        sd[f"{name}.{attr}"] = t.cpu()
        if sd:
            save_file(sd, str(self._talker_cache_path()))
        return len(sd)

    def _load_talker_cache(self, talker, device):
        from safetensors.torch import load_file
        path = self._talker_cache_path()
        if not path.exists():
            return 0
        blob = load_file(str(path))
        mods = dict(talker.named_modules())
        n = 0
        for k, v in blob.items():
            owner, _, attr = k.rpartition(".")
            m = mods.get(owner)
            if m is None:
                return 0                      # stale cache, fall back to quantising
            m.register_buffer(attr, v.to(device), persistent=False)
            n += 1
        return n

    def load(self, log=print):
        from transformers import Qwen3OmniMoeProcessor
        from duplex.loader import load_direct
        from duplex.streaming.fused_packed_moe import fuse_packed_moe_blocks
        from duplex.streaming.fused_moe import fuse_moe_blocks
        from duplex.streaming.triton_dequant_gemv import patch_attention_to_triton

        dev, dtype = "cuda", torch.bfloat16
        t0 = time.time()

        # Direct loader, not from_pretrained: that spends ~480s of its ~910s in
        # compressed-tensors' compress_model() building a parameter structure whose
        # output we discard, since dequantization here is our own Triton kernel.
        # Measured: 18s to place 64907 tensors, 0 unmatched.
        self._set("loading weights", "direct shard read, skipping the compress pass")
        model, _cfg = load_direct(self.model_path, device="cpu", dtype=dtype,
                                  skip_vision=True, log=lambda m: self._set(
                                      "loading weights", m))
        th = model.thinker

        self._set("fusing thinker MoE",
                  f"{self.n_pinned} layers pinned, rest streamed from host")
        fuse_packed_moe_blocks(th.model, dtype=dtype, device=dev,
                               n_pinned=self.n_pinned, cpu_compute=self.cpu_moe,
                               verbose=False)

        self._set("attention -> Triton", "packed projections onto the kernel")
        patch_attention_to_triton(th, device=dev, verbose=False)

        cached = 0
        if self.talker_q8 and self._talker_cache_path().exists():
            # Structure first (cheap, stacks then throws the result away), then
            # overwrite with the cached int8 tensors.
            self._set("fusing talker MoE", "restoring int8 from disk cache")
            fuse_moe_blocks(model.talker, q8=True, verbose=False)
            cached = self._load_talker_cache(model.talker, "cpu")
        if not cached:
            self._set("fusing talker MoE",
                      "gather+bmm" + (", int8 (6.1 -> ~3.0 GB)" if self.talker_q8 else ""))
            fuse_moe_blocks(model.talker, q8=self.talker_q8, verbose=False)
            if self.talker_q8:
                self._set("caching talker", "writing int8 to disk for next start")
                n = self._save_talker_cache(model.talker)
                log(f"cached {n} talker tensors to {self._talker_cache_path().name}")

        self._set("placing on GPU", "everything must report as a GPU module")
        model = model.to(dev)

        # code2wav stays bf16. fp32 was tried on the theory that its overlap-add
        # conv stack was causing the torn audio; it changed nothing (env 1.49 vs
        # 1.60, max-jump 0.399 vs 0.379) and cost ~0.4 GB plus fp32 activations,
        # which OOMed. The tearing was FusedMoE dropping the talker's shared
        # expert -- 49% relative error, now 0.004.
        torch.cuda.empty_cache()

        if self.compile_mode:
            self._set("compiling", f"torch.compile({self.compile_mode})")
            model.thinker.model = torch.compile(model.thinker.model,
                                                mode=self.compile_mode,
                                                dynamic=False)
        self.torch = torch
        self.model = model
        self._set("processor", "tokenizer + feature extractor")
        self.proc = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
        self.dev = dev
        self.dtype = dtype
        gb = torch.cuda.memory_allocated() / 1024**3
        log(f"fast engine ready in {time.time()-t0:.0f}s, {gb:.2f} GB resident")
        self._set("ready", f"{gb:.2f} GB on GPU")

    # ---- streaming generate --------------------------------------------------
    def generate_streaming(self, user_audio, on_frame, max_new: int = 48,
                           frames_per_token: int = 6):
        """Emit 80 ms audio frames as the talker produces them.

        The talker's `prepare_inputs_for_generation` builds `residual_codes` -- the
        full 16-codebook frame -- on every step. Wrapping it captures each frame at
        the moment it exists, so audio can be decoded and sent while generation
        continues, instead of waiting for the whole utterance.

        `trailing_text_hidden` is only ever read at `[:, generation_step]`, which is
        why this works without touching the generation loop.

        Returns the text; frames go to `on_frame(float32[1920])`.
        """
        torch = self.torch
        from duplex.streaming.code2wav import StreamingCode2Wav

        conv = [{"role": "user", "content": [{"type": "audio", "audio": user_audio}]}]
        text = self.proc.apply_chat_template(conv, add_generation_prompt=True,
                                             tokenize=False)
        inputs = self.proc(text=text, audio=[user_audio], return_tensors="pt").to(self.dev)
        for k in list(inputs.keys()):
            v = inputs[k]
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(self.dtype)

        talker = self.model.talker
        c2w = self.model.code2wav
        sc = StreamingCode2Wav(c2w).install()
        sc.reset()
        orig_prep = talker.prepare_inputs_for_generation
        n_groups = self.model.config.talker_config.num_code_groups

        import functools

        # transformers inspects this method's signature to validate model_kwargs,
        # so the wrapper must advertise the original's parameters or generate()
        # rejects trailing_text_hidden, tts_pad_embed and friends.
        @functools.wraps(orig_prep)
        def prep(*a, **kw):
            out = orig_prep(*a, **kw)
            codes = out.get("residual_codes")
            if codes is not None and codes.shape[-1] == n_groups:
                with torch.inference_mode():
                    wav = sc.decode(codes.view(1, n_groups, 1).to(self.dev))
                on_frame(wav.reshape(-1).float().cpu().numpy())
            return out

        talker.prepare_inputs_for_generation = prep
        try:
            with torch.inference_mode():
                # Cap the talker. Left uncapped it runs away: 32 text tokens
                # produced 1619 frames = 129.5 s of audio, which was most of a
                # 210 s turn. Speech is ~4-6 frames per text token at 12.5 Hz, so
                # this bounds it without truncating normal replies.
                seq, _ = self.model.generate(
                    **inputs, return_audio=True, max_new_tokens=max_new,
                    talker_max_new_tokens=max_new * frames_per_token,
                    speaker=self.speaker, thinker_do_sample=False,
                    talker_do_sample=False,
                )
        finally:
            talker.prepare_inputs_for_generation = orig_prep
            sc.remove()

        said = ""
        try:
            ids = getattr(seq, "sequences", seq)
            said = self.proc.batch_decode(ids, skip_special_tokens=True)[0]
        except Exception:
            pass
        return said

    # ---- generate -----------------------------------------------------------
    def generate(self, user_audio: np.ndarray, max_new: int = 48):
        torch = self.torch
        conv = [{"role": "user", "content": [{"type": "audio", "audio": user_audio}]}]
        text = self.proc.apply_chat_template(conv, add_generation_prompt=True,
                                             tokenize=False)
        inputs = self.proc(text=text, audio=[user_audio], return_tensors="pt")
        inputs = inputs.to(self.dev)
        # The processor emits float32 features; the model is bf16, and the audio
        # tower's conv2d rejects the mismatch outright.
        for k in list(inputs.keys()):
            v = inputs[k]
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(self.dtype)

        with torch.inference_mode():
            seq, wav = self.model.generate(
                **inputs, return_audio=True, max_new_tokens=max_new,
                speaker=self.speaker, thinker_do_sample=False, talker_do_sample=False,
            )
        w = (wav[0] if isinstance(wav, (list, tuple)) else wav)
        w = w.reshape(-1).float().cpu().numpy()

        said = ""
        try:
            ids = getattr(seq, "sequences", seq)
            said = self.proc.batch_decode(ids, skip_special_tokens=True)[0]
        except Exception:
            pass
        return said, w
