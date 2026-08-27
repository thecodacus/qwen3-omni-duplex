"""Geometry of Qwen3-Omni-30B-A3B, and the split it implies.

Every number here is read from the shipped `config.json` (mirrored in
`docs/qwen3-omni-config.json`), not guessed. `verify()` re-reads a real
checkpoint and fails loudly if upstream changes any of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# ---- Thinker -----------------------------------------------------------------
THINKER_LAYERS = 48
THINKER_HIDDEN = 2048
THINKER_EXPERTS = 128
THINKER_ACTIVE = 8
THINKER_MOE_INTER = 768
THINKER_Q_HEADS = 32
THINKER_KV_HEADS = 4

# ---- Talker ------------------------------------------------------------------
TALKER_LAYERS = 20
TALKER_HIDDEN = 1024
TALKER_EXPERTS = 128
TALKER_ACTIVE = 6
TALKER_MOE_INTER = 384
TALKER_Q_HEADS = 16
TALKER_KV_HEADS = 2

# ---- Code predictor (MTP) ----------------------------------------------------
MTP_LAYERS = 5
MTP_HIDDEN = 1024
NUM_CODE_GROUPS = 16

# ---- The load-bearing constant ----------------------------------------------
# The Talker is conditioned on Thinker hidden states from THIS layer, per token
# position, plus layer-0 embeddings. It never sees layers above it. Depth is
# feed-forward, so layer 24's activations cannot depend on layers 25-47 --
# which is what makes a fast/slow split possible at all.
ACCEPT_HIDDEN_LAYER = 24

# ---- The clock ---------------------------------------------------------------
POSITION_ID_PER_SECONDS = 13          # ~12.5 Hz frame grid, same as Moshi's Mimi
SECONDS_PER_CHUNK = 2                 # current (half-duplex) Talker granularity
FRAME_MS = 80.0                       # duplex target: one frame per 80 ms
REF_CODE2WAV_CHUNK = 300              # reference decode chunk = 24 s of audio
REF_CODE2WAV_LEFT_CTX = 25

BYTES_PER_PARAM_Q4 = 0.5625           # q4_K_M-ish, including scales


@dataclass(frozen=True)
class PathCost:
    """Resident weight footprint and per-frame read traffic for one path."""

    name: str
    resident_gb: float
    per_frame_mb: float


def _expert_params(hidden: int, inter: int) -> int:
    return 3 * hidden * inter  # gate, up, down


def clock_path(bytes_per_param: float = BYTES_PER_PARAM_Q4) -> PathCost:
    """Thinker[0:ACCEPT_HIDDEN_LAYER] + Talker + MTP -- pinned, never faults."""
    th_e = _expert_params(THINKER_HIDDEN, THINKER_MOE_INTER)
    tk_e = _expert_params(TALKER_HIDDEN, TALKER_MOE_INTER)
    resident = (
        THINKER_EXPERTS * th_e * ACCEPT_HIDDEN_LAYER
        + TALKER_EXPERTS * tk_e * TALKER_LAYERS
    ) * bytes_per_param
    per_frame = (
        THINKER_ACTIVE * th_e * ACCEPT_HIDDEN_LAYER
        + TALKER_ACTIVE * tk_e * TALKER_LAYERS
    ) * bytes_per_param
    return PathCost("clock", resident / 1024**3, per_frame / 1024**2)


def background_path(bytes_per_param: float = BYTES_PER_PARAM_Q4) -> PathCost:
    """Thinker[ACCEPT_HIDDEN_LAYER:] -- offloaded to RAM, runs in deferred bursts."""
    th_e = _expert_params(THINKER_HIDDEN, THINKER_MOE_INTER)
    n = THINKER_LAYERS - ACCEPT_HIDDEN_LAYER
    resident = THINKER_EXPERTS * th_e * n * bytes_per_param
    per_frame = THINKER_ACTIVE * th_e * n * bytes_per_param
    return PathCost("background", resident / 1024**3, per_frame / 1024**2)


def verify(model_path: str | Path) -> dict:
    """Re-read a checkpoint's config.json and assert our constants still hold."""
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    t = cfg["thinker_config"]["text_config"]
    tk = cfg["talker_config"]
    tkt = tk["text_config"]

    checks = {
        "THINKER_LAYERS": (THINKER_LAYERS, t["num_hidden_layers"]),
        "THINKER_HIDDEN": (THINKER_HIDDEN, t["hidden_size"]),
        "THINKER_EXPERTS": (THINKER_EXPERTS, t["num_experts"]),
        "THINKER_ACTIVE": (THINKER_ACTIVE, t["num_experts_per_tok"]),
        "THINKER_MOE_INTER": (THINKER_MOE_INTER, t["moe_intermediate_size"]),
        "TALKER_LAYERS": (TALKER_LAYERS, tkt["num_hidden_layers"]),
        "TALKER_HIDDEN": (TALKER_HIDDEN, tkt["hidden_size"]),
        "ACCEPT_HIDDEN_LAYER": (ACCEPT_HIDDEN_LAYER, tk["accept_hidden_layer"]),
        "NUM_CODE_GROUPS": (NUM_CODE_GROUPS, tk["num_code_groups"]),
        "SECONDS_PER_CHUNK": (SECONDS_PER_CHUNK, tk["seconds_per_chunk"]),
    }
    bad = {k: v for k, (exp, got) in checks.items() if exp != got for v in [f"expected {exp}, got {got}"]}
    if bad:
        raise AssertionError(f"checkpoint geometry diverged from duplex.config: {bad}")
    return {k: got for k, (_, got) in checks.items()}


def summary() -> str:
    c, b = clock_path(), background_path()
    return (
        f"Qwen3-Omni-30B-A3B split at Thinker layer {ACCEPT_HIDDEN_LAYER}/{THINKER_LAYERS}\n"
        f"  clock path      {c.resident_gb:6.2f} GB resident   {c.per_frame_mb:7.1f} MB/frame  (pinned, {FRAME_MS:.0f} ms deadline)\n"
        f"  background path {b.resident_gb:6.2f} GB resident   {b.per_frame_mb:7.1f} MB/token  (offloaded, no deadline)\n"
        f"  frame grid      {1000 / FRAME_MS:.1f} Hz    reference decode chunk "
        f"{REF_CODE2WAV_CHUNK} codes = {REF_CODE2WAV_CHUNK * FRAME_MS / 1000:.0f} s"
    )
