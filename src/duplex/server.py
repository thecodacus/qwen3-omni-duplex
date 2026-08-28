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
import time
from pathlib import Path

import numpy as np

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

    def load(self, log=print):
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        from duplex.quant.dequant import patch_packed_linears

        t0 = time.time()
        log("loading model (~15 min, once per server lifetime)")
        self.torch = torch
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_path, device_map="cpu", dtype=torch.float32,
        ).eval()
        patch_packed_linears(self.model, verbose=False)
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
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
    from duplex.vad import EnergyVAD, State

    app = FastAPI()

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    async def health():
        return {"ready": engine.model is not None}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        vad = EnergyVAD(sample_rate=IN_SR, frame_ms=FRAME_MS)
        vad.reset()
        await sock.send_text(json.dumps({"type": "status", "msg": "connected"}))
        busy = False

        try:
            while True:
                msg = await sock.receive()
                if "text" in msg and msg["text"]:
                    if json.loads(msg["text"]).get("type") == "reset":
                        vad.reset()
                        await sock.send_text(json.dumps({"type": "status", "msg": "reset"}))
                    continue
                data = msg.get("bytes")
                if not data:
                    continue

                frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if len(frame) < IN_FRAME:
                    continue
                state = vad.push(frame[:IN_FRAME])

                if state is State.ENDPOINT and not busy:
                    utt = vad.take_utterance()
                    await sock.send_text(json.dumps(
                        {"type": "state", "vad": "endpoint",
                         "utterance_s": round(len(utt) / IN_SR, 2)}))
                    if len(utt) < IN_SR * 0.3:      # ignore coughs
                        continue
                    busy = True
                    await sock.send_text(json.dumps(
                        {"type": "status", "msg": "thinking"}))
                    t0 = time.time()
                    said, wav = await asyncio.to_thread(engine.generate, utt)
                    gen_s = time.time() - t0
                    if said:
                        await sock.send_text(json.dumps({"type": "text", "text": said}))
                    await sock.send_text(json.dumps({"type": "audio_start", "sr": OUT_SR}))
                    n = 0
                    for i in range(0, len(wav) - OUT_FRAME + 1, OUT_FRAME):
                        pcm = (np.clip(wav[i:i + OUT_FRAME], -1, 1) * 32767).astype(np.int16)
                        await sock.send_bytes(pcm.tobytes())
                        n += 1
                        await asyncio.sleep(0)      # let the socket drain
                    await sock.send_text(json.dumps(
                        {"type": "audio_end", "frames": n,
                         "gen_s": round(gen_s, 1),
                         "audio_s": round(len(wav) / OUT_SR, 2),
                         "realtime_factor": round((len(wav) / OUT_SR) / max(gen_s, 1e-9), 3)}))
                    busy = False
                elif state is not State.IDLE:
                    await sock.send_text(json.dumps({"type": "state", "vad": state.value}))
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
    a = p.parse_args()

    import uvicorn

    static = Path(__file__).parent / "static"
    engine = Engine(a.model, speaker=a.speaker)
    engine.load()
    app = build_app(engine, static)
    print(f"serving on http://{a.host}:{a.port}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
