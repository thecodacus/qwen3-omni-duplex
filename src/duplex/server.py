"""WebSocket server: audio in, VAD, generation, streaming audio out.

Protocol (one connection = one conversation):

    client -> server   binary   Int16 PCM @ 16 kHz, 80 ms frames (1280 samples)
    client -> server   text     {"type":"reset"}

    server -> client   text     {"type":"state","vad":"idle|speaking|endpoint"}
                                {"type":"status","msg":...}
                                {"type":"text","text":...}
                                {"type":"audio_start","sr":24000}
                                {"type":"audio_end","frames":N,"ms_per_frame":...}
    server -> client   binary   Int16 PCM @ 24 kHz, 80 ms frames (1920 samples)

Generation sits behind `Engine` so the fast decode loop can replace the current
one without touching the server or the client.

Not full duplex: the model has no learned ability to stay quiet while you speak
(measured — see JOURNEY.md §18), so this is VAD-gated turn taking. The transport
is duplex-shaped on purpose — audio flows both directions continuously and the
server keeps running VAD while it speaks — so barge-in and, later, a duplex model
slot in without a protocol change.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from pathlib import Path

import numpy as np

# Module scope, deliberately: this file uses `from __future__ import annotations`,
# so every annotation is a string that FastAPI resolves with get_type_hints against
# MODULE globals. Importing WebSocket inside build_app made it a local, so the
# annotation could not be resolved, `sock` was treated as a query parameter, and the
# handshake was rejected with a bare 403 before the handler ever ran.
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

FRAME_MS = 80.0
IN_SR = 16000
OUT_SR = 24000
IN_FRAME = int(IN_SR * FRAME_MS / 1000)     # 1280
OUT_FRAME = int(OUT_SR * FRAME_MS / 1000)   # 1920


class Engine:
    """Generation backend. `generate` yields 80 ms float32 frames at OUT_SR."""

    def __init__(self, model_path: str, cpu: bool = True, speaker: str = "Ethan"):
        self.model_path = model_path
        self.cpu = cpu
        self.speaker = speaker
        self.model = None
        self.proc = None
        # Loading takes ~15 min, so the server must not block on it: it serves
        # immediately and reports progress, otherwise a restart is indistinguishable
        # from a crash for a quarter of an hour.
        self.status = {"ready": False, "phase": "not started", "elapsed": 0.0,
                       "detail": ""}
        self._t0 = None

    def start_background_load(self):
        self._t0 = time.time()
        self.status.update(phase="starting", detail="spawning loader")
        threading.Thread(target=self._load_guarded, daemon=True).start()

    def _set(self, phase, detail=""):
        self.status.update(phase=phase, detail=detail,
                           elapsed=round(time.time() - (self._t0 or time.time()), 1))

    def _load_guarded(self):
        try:
            self.load(log=lambda m: self._set(self.status["phase"], m))
            self.status.update(ready=True, phase="ready",
                               elapsed=round(time.time() - self._t0, 1))
        except Exception as e:
            self.status.update(ready=False, phase="failed", detail=f"{type(e).__name__}: {e}")

    def load(self, log=print):
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        from duplex.quant.dequant import patch_packed_linears

        t0 = time.time()
        self.torch = torch
        self._set("loading weights", "reading shards, ~13 min")
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_path, device_map="cpu", dtype=torch.float32,
        ).eval()
        self._set("patching", "installing int4 dequant")
        patch_packed_linears(self.model, verbose=False)
        self._set("processor", "tokenizer + feature extractor")
        self.proc = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
        log(f"model ready in {time.time()-t0:.0f}s")

    def generate(self, user_audio: np.ndarray, max_new: int = 48):
        """Yield (text, [frames]) — currently batch, streamed out frame by frame."""
        torch = self.torch
        conv = [{"role": "user", "content": [{"type": "audio", "audio": user_audio}]}]
        text = self.proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = self.proc(text=text, audio=[user_audio], return_tensors="pt")

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


def build_app(engine: Engine, static_dir: Path):
    from duplex.vad import EnergyVAD, State

    app = FastAPI()

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    async def health():
        st = dict(engine.status)
        if not st["ready"] and engine._t0:
            st["elapsed"] = round(time.time() - engine._t0, 1)
        return st

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        vad = EnergyVAD(sample_rate=IN_SR, frame_ms=FRAME_MS)
        vad.reset()
        last_level = {"db": -99.0}
        await sock.send_text(json.dumps({"type": "status", "msg": "connected"}))
        await sock.send_text(json.dumps({"type": "model", **engine.status}))
        await sock.send_text(json.dumps({"type": "vadcfg", **vad.settings()}))
        busy = False
        told_loading = False
        pending = []          # generation tasks that finished while we kept reading
        t0_gen = []
        frameq = []           # audio frames produced by the streaming engine
        sent_frames = [0]

        try:
            while True:
                msg = await sock.receive()
                if "text" in msg and msg["text"]:
                    m = json.loads(msg["text"])
                    if m.get("type") == "reset":
                        vad.reset()
                        await sock.send_text(json.dumps({"type": "status", "msg": "reset"}))
                    elif m.get("type") == "config":
                        st = vad.configure(**{k: m.get(k) for k in
                                              ("onset_frames", "hangover_frames",
                                               "threshold_db")})
                        await sock.send_text(json.dumps({"type": "vadcfg", **st}))
                    continue
                data = msg.get("bytes")
                if not data:
                    continue

                frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if len(frame) < IN_FRAME:
                    continue
                state = vad.push(frame[:IN_FRAME])

                if not engine.status["ready"]:
                    # Keep consuming audio so the client's stream never stalls, but
                    # do not accumulate an utterance against a model that cannot answer.
                    vad.reset()
                    if not told_loading:
                        await sock.send_text(json.dumps({"type": "model", **engine.status}))
                        told_loading = True
                    continue
                if told_loading:
                    await sock.send_text(json.dumps({"type": "model", **engine.status}))
                    told_loading = False

                # Ship any audio frames produced since the last message.
                while frameq:
                    f = frameq.pop(0)
                    pcm = (np.clip(f, -1, 1) * 32767).astype(np.int16)
                    await sock.send_bytes(pcm.tobytes())
                    sent_frames[0] += 1

                if busy:
                    # A turn is being generated. Keep draining the socket -- the
                    # client streams 12.5 frames/s throughout, and blocking here
                    # fills the receive buffer and kills the connection. Reset the
                    # VAD so speech during playback does not queue a second turn;
                    # this is where barge-in will hook in.
                    vad.reset()
                    continue

                if state is State.ENDPOINT:
                    utt = vad.take_utterance()
                    await sock.send_text(json.dumps(
                        {"type": "state", "vad": "endpoint",
                         "utterance_s": round(len(utt) / IN_SR, 2)}))
                    if len(utt) < IN_SR * 0.3:      # ignore coughs
                        continue
                    busy = True
                    await sock.send_text(json.dumps({"type": "status", "msg": "thinking"}))
                    t0_gen.clear(); t0_gen.append(time.time())
                    await sock.send_text(json.dumps({"type": "audio_start", "sr": OUT_SR}))
                    # Stream frames as the talker produces them. First audio lands
                    # ~5x sooner than waiting for the whole utterance.
                    if hasattr(engine, "generate_streaming"):
                        loop = asyncio.get_running_loop()
                        def emit(f, _l=loop):
                            _l.call_soon_threadsafe(frameq.append, f)
                        task = asyncio.create_task(
                            asyncio.to_thread(engine.generate_streaming, utt, emit))
                    else:
                        task = asyncio.create_task(asyncio.to_thread(engine.generate, utt))
                    task.add_done_callback(lambda t: pending.append(t))
                elif state is not State.IDLE:
                    await sock.send_text(json.dumps({"type": "state", "vad": state.value}))
                else:
                    db = vad._db(frame[:IN_FRAME])
                    if abs(db - last_level["db"]) > 3.0:
                        last_level["db"] = db
                        await sock.send_text(json.dumps(
                            {"type": "level", "db": round(db, 1),
                             "floor": round(vad._floor, 1) if vad._floor is not None else None,
                             "thresh": round((vad._floor or -60) + vad.threshold_db, 1)}))

                # Deliver a finished turn, if one completed while we kept reading.
                while pending:
                    t = pending.pop(0)
                    try:
                        res = t.result()
                    except Exception as e:
                        await sock.send_text(json.dumps(
                            {"type": "status", "msg": f"generate failed: {e}"}))
                        busy = False
                        continue
                    gen_s = time.time() - t0_gen[0] if t0_gen else 0.0
                    if isinstance(res, tuple):          # non-streaming engine
                        said, wav = res
                        for i in range(0, len(wav) - OUT_FRAME + 1, OUT_FRAME):
                            pcm = (np.clip(wav[i:i + OUT_FRAME], -1, 1) * 32767).astype(np.int16)
                            await sock.send_bytes(pcm.tobytes())
                            sent_frames[0] += 1
                    else:
                        said = res
                    while frameq:                       # drain the tail
                        pcm = (np.clip(frameq.pop(0), -1, 1) * 32767).astype(np.int16)
                        await sock.send_bytes(pcm.tobytes())
                        sent_frames[0] += 1
                    if said:
                        await sock.send_text(json.dumps({"type": "text", "text": said}))
                    n = sent_frames[0]; sent_frames[0] = 0
                    audio_s = n * FRAME_MS / 1000
                    await sock.send_text(json.dumps(
                        {"type": "audio_end", "frames": n,
                         "gen_s": round(gen_s, 1), "audio_s": round(audio_s, 2),
                         "realtime_factor": round(audio_s / max(gen_s, 1e-9), 3)}))
                    vad.reset()
                    busy = False
        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await sock.send_text(json.dumps({"type": "status", "msg": f"error: {e}"}))
            except Exception:
                pass

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/models/Qwen3-Omni-AWQ")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=877)
    p.add_argument("--speaker", default="Ethan")
    p.add_argument("--cpu", action="store_true",
                   help="use the slow CPU engine instead of the GPU one")
    p.add_argument("--pinned", type=int, default=4,
                   help="thinker MoE layers kept on the GPU (of 48); the rest stream")
    a = p.parse_args()

    import uvicorn

    static = Path(__file__).parent / "static"
    if a.cpu:
        engine = Engine(a.model, speaker=a.speaker)
    else:
        from duplex.fast_engine import FastEngine
        engine = FastEngine(a.model, speaker=a.speaker, n_pinned=a.pinned)
    app = build_app(engine, static)
    engine.start_background_load()
    print(f"serving on http://{a.host}:{a.port} (model loading in background)",
          flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
