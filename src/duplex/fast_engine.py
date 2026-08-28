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
                 cpu: bool = False):
        self.model_path = model_path
        self.speaker = speaker
        self.n_pinned = n_pinned
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
                               n_pinned=self.n_pinned, verbose=False)

        self._set("attention -> Triton", "packed projections onto the kernel")
        patch_attention_to_triton(th, device=dev, verbose=False)

        self._set("fusing talker MoE", "gather+bmm, replacing the python expert loop")
        fuse_moe_blocks(model.talker, verbose=False)

        self._set("placing on GPU", "everything must report as a GPU module")
        model = model.to(dev)
        torch.cuda.empty_cache()

        self.torch = torch
        self.model = model
        self._set("processor", "tokenizer + feature extractor")
        self.proc = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
        self.dev = dev
        self.dtype = dtype
        gb = torch.cuda.memory_allocated() / 1024**3
        log(f"fast engine ready in {time.time()-t0:.0f}s, {gb:.2f} GB resident")
        self._set("ready", f"{gb:.2f} GB on GPU")

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
