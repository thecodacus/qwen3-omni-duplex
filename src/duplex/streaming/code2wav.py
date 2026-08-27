"""Stateful streaming wrapper for Qwen3-Omni's code2wav vocoder.

The shipped `chunked_decode` is stateless: every chunk re-runs `forward()`, so the
causal stack's warm-up is re-paid and its tail is re-discarded on each call. At
`chunk_size=300` that costs one tail per 24 s. At `chunk_size=1` it destroys 28.9%
of every frame.

The tail is not garbage — it is real partial output that later inputs complete.
This wrapper keeps it.

Two kinds of state, installed by walking the module tree:

`Qwen3OmniMoeCausalConvNet` (stride 1)
    Zero-pads `padding` frames on the left every call. Streaming instead prepends
    the previous call's trailing `padding` input frames.

`Qwen3OmniMoeCausalTransConvNet` (kernel = 2*stride)
    Emits `(L-1)*s + k` samples then trims `s` from each end. The right trim is the
    region later inputs still contribute to, so streaming overlap-adds it:

        U = conv(x); U[..., :s] += tail; emit U[..., :-s]; tail = U[..., -s:]

    The left trim is warm-up and is applied only on the first call.

Concatenating the emitted pieces reproduces the whole-sequence result exactly:
one warm-up at stream start and one tail at stream end, instead of one of each
per chunk. Verify with `duplex stream-verify`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StreamingCode2Wav:
    """Wraps a `Qwen3OmniMoeCode2Wav` in place, making its conv stack stateful.

    Use as a context manager, or call `install()` / `remove()` explicitly. The
    underlying module is restored on exit, so the same object can be used for
    batch decoding afterwards.
    """

    def __init__(self, model):
        self.model = model
        self._orig: list[tuple[nn.Module, callable]] = []
        self._conv_state: dict[int, torch.Tensor | None] = {}
        self._tconv_state: dict[int, torch.Tensor | None] = {}
        self._tconv_started: dict[int, bool] = {}
        self._kv = None
        self._installed = False

    # ---- state -------------------------------------------------------------
    def reset(self):
        """Clear all streaming state — start a new utterance."""
        for d in (self._conv_state, self._tconv_state):
            for k in d:
                d[k] = None
        for k in self._tconv_started:
            self._tconv_started[k] = False
        self._kv = None

    # ---- install / remove --------------------------------------------------
    def install(self):
        if self._installed:
            return self
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeCausalConvNet,
            Qwen3OmniMoeCausalTransConvNet,
        )

        for m in self.model.modules():
            if isinstance(m, Qwen3OmniMoeCausalConvNet):
                if m.stride != 1:
                    raise NotImplementedError(
                        f"streaming CausalConvNet assumes stride 1, got {m.stride}"
                    )
                self._conv_state[id(m)] = None
                self._orig.append((m, m.forward))
                m.forward = self._make_conv_forward(m)
            elif isinstance(m, Qwen3OmniMoeCausalTransConvNet):
                self._tconv_state[id(m)] = None
                self._tconv_started[id(m)] = False
                self._orig.append((m, m.forward))
                m.forward = self._make_tconv_forward(m)

        self._installed = True
        return self

    def remove(self):
        for m, fwd in self._orig:
            m.forward = fwd
        self._orig.clear()
        self._installed = False

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()
        return False

    # ---- patched forwards --------------------------------------------------
    def _make_conv_forward(self, m):
        pad = m.padding

        def forward(x):
            if pad > 0:
                prev = self._conv_state[id(m)]
                if prev is None:
                    prev = x.new_zeros(x.shape[0], x.shape[1], pad)
                x = torch.cat([prev, x], dim=-1)
                self._conv_state[id(m)] = x[..., -pad:].clone()
            return m.conv(x).contiguous()

        return forward

    def _make_tconv_forward(self, m):
        right = m.right_pad
        left = m.left_pad
        conv = m.conv
        bias = conv.bias

        def forward(x):
            # The bias must be kept OUT of the overlap-add. Two consecutive chunks
            # both cover the shared region, so adding a biased output to a biased
            # tail counts the bias twice. Convolve without it, overlap-add the raw
            # contributions, then add the bias once to what is emitted.
            u = nn.functional.conv_transpose1d(
                x, conv.weight, None, conv.stride, conv.padding,
                conv.output_padding, conv.groups, conv.dilation,
            )
            if right > 0:
                tail = self._tconv_state[id(m)]
                if tail is not None:
                    u = u.clone()
                    u[..., : tail.shape[-1]] += tail
                self._tconv_state[id(m)] = u[..., u.shape[-1] - right :].clone()
                u = u[..., : u.shape[-1] - right]
            if bias is not None:
                u = u + bias.view(1, -1, 1)
            if not self._tconv_started[id(m)]:
                u = u[..., left:]           # warm-up, first call only
                self._tconv_started[id(m)] = True
            return u.contiguous()

        return forward

    # ---- decoding ----------------------------------------------------------
    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode `codes` [B, num_quantizers, T] continuing the current stream."""
        model = self.model
        if codes.shape[1] != model.config.num_quantizers:
            raise ValueError(
                f"expected {model.config.num_quantizers} code groups, got {codes.shape[1]}"
            )
        hidden = model.code_embedding(codes + model.code_offset).mean(1)

        from transformers.cache_utils import DynamicCache

        if self._kv is None:
            self._kv = DynamicCache()
        out = model.pre_transformer(
            inputs_embeds=hidden, past_key_values=self._kv, use_cache=True
        )
        self._kv = getattr(out, "past_key_values", self._kv)
        hidden = out.last_hidden_state.permute(0, 2, 1)

        for blocks in model.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in model.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)

    @torch.inference_mode()
    def flush(self) -> torch.Tensor:
        """Emit the final tail the stream is still holding (end of utterance).

        The batch path discards this too, so it is only needed to be strictly
        longer than `forward()`, not to match it.
        """
        parts = [t for t in self._tconv_state.values() if t is not None]
        if not parts:
            return torch.empty(0)
        return parts[-1]
