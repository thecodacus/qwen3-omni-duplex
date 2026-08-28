"""CLI entry point: `duplex <command>`."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="duplex",
        description="Full-duplex retrofit of Qwen3-Omni. See docs/design.md for the architecture.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("geometry", help="print the model split and its cost, from config constants")

    v = sub.add_parser("verify", help="re-read a checkpoint and assert the geometry still holds")
    v.add_argument("model_path")

    sub.add_parser("bench", help="clock-path frame-deadline benchmark (synthetic weights, no model needed)",
                   add_help=False)
    sub.add_parser("thesis", help="full-model vocoder sweep (needs the whole checkpoint loaded)",
                   add_help=False)
    sub.add_parser("vocoder", help="standalone code2wav sweep: chunk_size + left_context (no Thinker needed)",
                   add_help=False)
    sub.add_parser("stream-verify", help="prove the stateful vocoder matches the batch result frame-by-frame",
                   add_help=False)
    sub.add_parser("clock-real", help="real-weights clock-path latency (talker + MTP + streaming vocoder)",
                   add_help=False)
    sub.add_parser("speak", help="end-to-end speech via on-the-fly int4 dequant (offline quality gate)",
                   add_help=False)
    sub.add_parser("probe", help="test whether the talker ALREADY quiets itself when told a user is speaking",
                   add_help=False)
    sub.add_parser("clock-full", help="the FULL clock path in one loop with real weights",
                   add_help=False)
    sub.add_parser("serve", help="websocket server + browser UI: talk to it",
                   add_help=False)

    # bench/thesis own the rest of the argv so their flags pass through untouched
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in ("bench", "thesis", "vocoder", "stream-verify", "clock-real", "speak", "probe", "clock-full", "serve"):
        cmd, rest = argv[0], argv[1:]
        sys.argv = [f"duplex {cmd}", *rest]
        if cmd == "bench":
            from duplex.bench.clock import main as run
        elif cmd == "vocoder":
            from duplex.thesis.vocoder_standalone import main as run
        elif cmd == "stream-verify":
            from duplex.streaming.verify import main as run
        elif cmd == "clock-real":
            from duplex.streaming.talker import main as run
        elif cmd == "speak":
            from duplex.speak import main as run
        elif cmd == "probe":
            from duplex.probe import main as run
        elif cmd == "clock-full":
            from duplex.streaming.clock import main as run
        elif cmd == "serve":
            from duplex.server import main as run
        else:
            from duplex.thesis.vocoder import main as run
        run()
        return 0

    a = p.parse_args(argv)
    from duplex import config

    if a.cmd == "geometry":
        print(config.summary())
    elif a.cmd == "verify":
        got = config.verify(a.model_path)
        print("geometry verified against checkpoint:")
        for k, v in got.items():
            print(f"  {k:<22} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
