"""计分 UI 的本地服务。

* `GET /`           单页计分牌
* `GET /api/state`  当前比分快照（JSON）
* `POST /api/demo/…` 仅演示模式可用：加分、撤销、重置、待确认

实时模式只读：语音/视觉进程启动服务时传同一个 ScoreEngine，页面每 0.5 秒拉一次状态，
不做双向写入，避免和识别线程抢状态。
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..scoring import (STATE_MATCH_OVER, STATE_PENDING, ScoreEngine)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _other(side: str) -> str:
    return "B" if side == "A" else "A"


def open_display(url: str, window_size: tuple[int, int] = (1024, 600),
                 compact: bool = False) -> None:
    """优先用 Edge/Chrome 的 --app 模式打开固定大小窗口（无地址栏，适合小屏）。"""
    target = url + ("?compact=1" if compact else "")
    width, height = window_size
    if os.name == "nt":
        candidates = [
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                 "Microsoft", "Edge", "Application", "msedge.exe"),
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"),
                 "Microsoft", "Edge", "Application", "msedge.exe"),
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"),
                 "Google", "Chrome", "Application", "chrome.exe"),
        ]
        profile = Path(tempfile.gettempdir()) / "tt_referee_ui_profile"
        for exe in candidates:
            if exe.exists():
                try:
                    subprocess.Popen([str(exe), f"--app={target}",
                                      f"--window-size={width},{height}",
                                      f"--user-data-dir={profile}",
                                      "--no-first-run", "--no-default-browser-check"],
                                     close_fds=True)
                    return
                except OSError:
                    continue
    webbrowser.open(target)


def state_payload(engine: ScoreEngine, players: dict[str, str] | None = None,
                  demo: bool = False, title: str = "乒乓球智能裁判") -> dict:
    rules = engine.rules
    points, sets = engine.points, engine.sets
    deuce = (points["A"] >= rules.points_to_win - 1
             and points["B"] >= rules.points_to_win - 1)
    set_point = ""
    match_point = ""
    if not engine.match_over:
        for side in ("A", "B"):
            lead = points[side] - points[_other(side)]
            if points[side] >= rules.points_to_win - 1 and lead >= rules.win_by - 1:
                set_point = side
                if sets[side] >= rules.sets_to_win - 1:
                    match_point = side
                break
    player_a_left = engine.side_of_a == "left"
    serving_left = (engine.server == "A") == player_a_left
    pending = engine.pending or {}
    recent = []
    for item in engine.actions[-6:]:
        recent.append({
            "ts": item.get("ts", 0.0),
            "action": item.get("action", ""),
            "side": item.get("side", ""),
            "source": item.get("source", ""),
            "reason": item.get("reason", ""),
        })
    return {
        "title": title,
        "players": players or {"A": "选手 A", "B": "选手 B"},
        "points": {"A": points["A"], "B": points["B"]},
        "sets": {"A": sets["A"], "B": sets["B"]},
        "server": engine.server,
        "serving_side": "left" if serving_left else "right",
        "side_of_a": engine.side_of_a,
        "set_index": engine.set_index,
        "best_of": rules.best_of,
        "points_to_win": rules.points_to_win,
        "state": engine.state,
        "match_over": engine.match_over,
        "deuce": deuce,
        "set_point": set_point,
        "match_point": match_point,
        "pending": {
            "side": pending.get("side", ""),
            "reason": pending.get("reason", ""),
            "confidence": pending.get("confidence", 0.0),
        } if pending else None,
        "can_undo": bool(engine.history),
        "recent": recent,
        "demo": demo,
        "updated_at": time.time(),
    }


class _Handler(BaseHTTPRequestHandler):
    server: "_Server"

    def log_message(self, fmt, *args):  # 静默访问日志
        return

    # ------------------------------------------------------------ 工具
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        ctype = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "application/javascript; charset=utf-8",
                 ".svg": "image/svg+xml", ".png": "image/png"}.get(
                     path.suffix.lower(), "application/octet-stream")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _state(self) -> dict:
        with self.server.lock:
            return state_payload(self.server.engine, self.server.players,
                                 demo=self.server.allow_demo, title=self.server.title)

    # ------------------------------------------------------------ 路由
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html")
            return
        if parsed.path.startswith("/static/"):
            relative = parsed.path[len("/static/"):]
            target = (STATIC_DIR / relative).resolve()
            if STATIC_DIR.resolve() not in target.parents:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self._send_file(target)
            return
        if parsed.path == "/api/state":
            self._send_json(self._state())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if not self.server.allow_demo:
            self._send_json({"ok": False, "error": "read_only"}, HTTPStatus.FORBIDDEN)
            return
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        action = parsed.path.rsplit("/", 1)[-1]
        with self.server.lock:
            engine = self.server.engine
            if action == "award":
                side = (query.get("side") or ["A"])[0].upper()
                if side not in ("A", "B"):
                    self._send_json({"ok": False, "error": "bad_side"}, 400)
                    return
                engine.award(side, reason="ui_demo", source="ui")
            elif action == "undo":
                engine.undo()
            elif action == "redo":
                engine.redo()
            elif action == "reset":
                engine.reset_now()
            elif action == "pending":
                engine.propose_pending("B", reason="ui_demo", confidence=0.6, source="ui")
            elif action == "confirm":
                engine.confirm_pending()
            elif action == "reject":
                engine.reject_pending()
            else:
                self._send_json({"ok": False, "error": "unknown_action"}, 404)
                return
        self._send_json(self._state())


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, engine: ScoreEngine, allow_demo: bool,
                 players: dict[str, str] | None, title: str):
        super().__init__(address, _Handler)
        self.engine = engine
        self.allow_demo = allow_demo
        self.players = players
        self.title = title
        self.lock = threading.RLock()


class ScoreServer:
    def __init__(self, engine: ScoreEngine, host: str = "127.0.0.1", port: int = 8765,
                 allow_demo: bool = False, players: dict[str, str] | None = None,
                 title: str = "乒乓球智能裁判"):
        self._server = _Server((host, port), engine, allow_demo, players, title)
        self.host = host
        self.port = self._server.server_address[1]
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}/"

    def start(self, open_browser: bool = False,
              window_size: tuple[int, int] = (1024, 600),
              compact: bool = False) -> "ScoreServer":
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        if open_browser:
            threading.Timer(0.4, lambda: open_display(self.url, window_size, compact)).start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
