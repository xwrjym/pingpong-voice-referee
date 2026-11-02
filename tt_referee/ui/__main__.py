"""独立启动计分 UI（演示模式）。"""

from __future__ import annotations

import argparse
import sys
import time

from ..scoring import ScoreEngine, ScoringRules
from .server import ScoreServer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tt_referee.ui", description="乒乓球计分 UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--window-size", default="1024x600", help="例如 1024x600 / 800x480")
    parser.add_argument("--compact", action="store_true",
                        help="隐藏演示按钮与事件流，只显示比分")
    parser.add_argument("--best-of", type=int, choices=[3, 5], default=5)
    parser.add_argument("--players", nargs=2, metavar=("A", "B"), default=("选手 A", "选手 B"))
    parser.add_argument("--title", default="乒乓球智能裁判")
    parser.add_argument("--no-seed", action="store_true", help="不以 6:4 的示例比分启动")
    args = parser.parse_args(argv)

    engine = ScoreEngine(ScoringRules(best_of=args.best_of), log_events=False)
    if not args.no_seed:
        engine.points = {"A": 6, "B": 4}
        engine.sets = {"A": 1, "B": 0}
        engine.set_index = 1
        engine.server = "B"
    try:
        width, height = (int(v) for v in args.window_size.lower().split("x"))
    except ValueError:
        print("--window-size 格式应为 1024x600")
        return 2
    server = ScoreServer(engine, args.host, args.port, allow_demo=True,
                         players={"A": args.players[0], "B": args.players[1]},
                         title=args.title)
    server.start(open_browser=not args.no_browser, window_size=(width, height),
                 compact=args.compact)
    suffix = "?compact=1" if args.compact else ""
    print(f"计分 UI：{server.url}{suffix}（窗口 {width}x{height}，Ctrl+C 退出）")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
