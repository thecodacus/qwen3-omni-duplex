"""Energy VAD and turn-taking state machine.

The simplest thing that produces correct turn-taking, so the conversation loop can
be built and tested before a semantic VAD is introduced. Deliberately dumb: it
decides on loudness, not meaning, and will treat a cough as a turn.

Interface is the part that matters — it is the same shape SoulX-Duplug exposes
(`idle` / `nonidle` / `speak`), so swapping in a semantic detector later is a
constructor change rather than a rewrite:

    vad = EnergyVAD(sample_rate=24000)
    for chunk in audio_chunks:          # 80 ms frames, matching the clock
        state = vad.push(chunk)         # -> State.IDLE | SPEAKING | ENDPOINT

`ENDPOINT` fires once, on the transition from speech to sustained silence, which is
the moment the assistant may take its turn.

Tracks its own noise floor by minimum-following — falls to any quieter frame
immediately, creeps up slowly — so it needs no per-microphone threshold and does
not assume the stream *begins* with silence. (An earlier version calibrated on the
opening frames and failed completely on audio that starts mid-speech: it set the
floor to speech level and then detected 0.4 s of speech in a 12 s utterance.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class State(str, Enum):
    IDLE = "idle"            # no speech
    SPEAKING = "speaking"    # user is talking
    ENDPOINT = "endpoint"    # user just stopped — assistant may respond


@dataclass
class EnergyVAD:
    sample_rate: int = 24000
    frame_ms: float = 80.0           # match the model's 12.5 Hz clock
    onset_frames: int = 2            # consecutive loud frames to start
    hangover_frames: int = 8         # consecutive quiet frames to end (~0.64 s)
    # 25 dB above the floor, measured rather than guessed: across generated
    # samples the silence/speech boundary sat ~30 dB above the noise floor
    # (floor -81.8 -> boundary ~-50; floor -68.6 -> boundary ~-38), while
    # peak-relative was inconsistent (-35 vs -23 dB). An earlier value of 9 dB
    # marked pauses as speech.
    threshold_db: float = 25.0       # dB above the tracked noise floor
    floor_db: float = -60.0          # absolute gate, for a silent room
    floor_rise_db: float = 0.08      # per frame: how fast the floor may creep up
    floor_init_db: float = -35.0     # starting guess before anything quiet is seen

    _floor: float = field(default=None, init=False)
    _loud_run: int = field(default=0, init=False)
    _quiet_run: int = field(default=0, init=False)
    _speaking: bool = field(default=False, init=False)
    _speech: list = field(default_factory=list, init=False)

    @property
    def frame_len(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    @staticmethod
    def _db(x: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)
        return 20.0 * np.log10(rms)

    def reset(self):
        self._floor = None
        self._loud_run = self._quiet_run = 0
        self._speaking = False
        self._speech.clear()

    def push(self, frame: np.ndarray) -> State:
        """Feed one frame. Returns the state after this frame."""
        db = self._db(frame)

        # Minimum-following noise floor: drop to any quieter frame at once, rise
        # only slowly. Speech cannot drag the floor up with it, and a stream that
        # opens mid-sentence still finds the floor at the first pause.
        if self._floor is None:
            self._floor = min(db, self.floor_init_db)
        elif db < self._floor:
            self._floor = db
        else:
            self._floor = min(self._floor + self.floor_rise_db, db)

        loud = db > max(self._floor + self.threshold_db, self.floor_db)

        if not self._speaking:
            self._loud_run = self._loud_run + 1 if loud else 0
            if self._loud_run >= self.onset_frames:
                self._speaking = True
                self._quiet_run = 0
                self._speech.append(frame)
                return State.SPEAKING
            return State.IDLE

        self._speech.append(frame)
        self._quiet_run = 0 if loud else self._quiet_run + 1
        if self._quiet_run >= self.hangover_frames:
            self._speaking = False
            self._loud_run = 0
            return State.ENDPOINT
        return State.SPEAKING

    def take_utterance(self) -> np.ndarray:
        """Return and clear the audio captured since speech onset."""
        if not self._speech:
            return np.zeros(0, dtype=np.float32)
        out = np.concatenate(self._speech)
        self._speech.clear()
        # drop the trailing hangover silence
        trim = self.hangover_frames * self.frame_len
        return out[:-trim] if len(out) > trim else out

    def segment(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """Offline helper: return [(start, end)] sample ranges of speech.

        Offline the whole signal is available, so seed the floor from a low
        percentile of frame energies rather than waiting for a pause.
        """
        self.reset()
        n = self.frame_len
        frames = [audio[i : i + n] for i in range(0, len(audio) - n + 1, n)]
        if frames:
            self._floor = float(min(self._db(f) for f in frames))
        spans, start = [], None
        for i in range(0, len(audio) - n + 1, n):
            st = self.push(audio[i : i + n])
            if st is State.SPEAKING and start is None:
                start = max(0, i - self.onset_frames * n)
            elif st is State.ENDPOINT and start is not None:
                spans.append((start, i - self.hangover_frames * n))
                start = None
        if start is not None:
            spans.append((start, len(audio)))
        return [(a, b) for a, b in spans if b > a]
