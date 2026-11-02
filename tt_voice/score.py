"""比分状态机：语音只改这里的数字，不动原始视频和事件日志。

和方案第 4 点的对接约定：

* 视觉状态机在"疑似得分"时调用 `propose_point("left")` -> 进入待确认
* 语音「确认这一分」提交该分；「撤销」回退到上一步
* 「重置比分」必须说两次（重置 -> 确认），避免比赛中误清零
* 所有动作写进 events.jsonl，带原始识别文本和相似度，方便复盘误识别
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from .config import AppConfig

Side = Literal["left", "right"]

# 个人自用：说"我加一分"时映射到本人所在的一侧（config.self_side）
SELF_ALIASES = {"score_self": "self", "score_other": "other"}


@dataclass
class ActionResult:
    ok: bool
    message: str = ""
    intent: str = ""
    reason: str = ""
    requires_reask: bool = False
    state: dict[str, Any] = field(default_factory=dict)


class ScoreBoard:
    def __init__(self, cfg: AppConfig | None = None, log_events: bool = True):
        self.cfg = cfg or AppConfig()
        self.log_events = log_events          # 离线评测时关掉，别污染现场日志
        self.left = 0
        self.right = 0
        self._history: deque[dict[str, Any]] = deque(maxlen=200)
        self.pending: dict[str, Any] | None = None
        self.awaiting_reset_confirm = False
        self.reset_requested_at = 0.0
        self._log_path = self.cfg.resolve(self.cfg.event_log) if self.cfg.event_log else None
        self.events: list[dict[str, Any]] = []

    # ------------------------------------------------------------ 查询
    def snapshot(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "pending": dict(self.pending) if self.pending else None,
            "awaiting_reset_confirm": self.awaiting_reset_confirm,
            "can_undo": bool(self._history),
        }

    def score_text(self) -> str:
        return f"左方{self.left}比{self.right}右方"

    # ------------------------------------------------------------ 视觉 -> 语音
    def propose_point(self, side: Side, confidence: float = 0.0, note: str = "") -> ActionResult:
        self.pending = {"side": side, "confidence": confidence, "note": note, "at": time.time()}
        side_cn = "左方" if side == "left" else "右方"
        return ActionResult(ok=True, message=f"检测到{side_cn}可能得分，请说「确认这一分」",
                            intent="propose", state=self.snapshot())

    # ------------------------------------------------------------ 语音 -> 比分
    def apply(self, intent: str, raw_text: str = "", score: float = 0.0,
              source: str = "voice") -> ActionResult:
        if intent in SELF_ALIASES:
            mine = getattr(self.cfg, "self_side", "left")
            other = "right" if mine == "left" else "left"
            resolved = mine if SELF_ALIASES[intent] == "self" else other
            intent = f"score_{resolved}"
        handler = getattr(self, f"_do_{intent}", None)
        if handler is None:
            return ActionResult(ok=False, message="没听懂这个口令", intent=intent,
                                reason="unknown_intent", state=self.snapshot())

        before = {"left": self.left, "right": self.right}
        cancelled_reset = False
        if self.awaiting_reset_confirm and intent not in ("confirm", "reset"):
            # 重置待确认期间来了别的口令 -> 视为放弃重置，再执行新口令
            self.awaiting_reset_confirm = False
            cancelled_reset = True
        result = handler(raw_text)
        self._log(intent, raw_text, score, before, result, source,
                  extra={"reset_cancelled": True} if cancelled_reset else None)
        return result

    # ------------------------------------------------------------ 各意图实现
    def _do_score_left(self, raw_text: str = "") -> ActionResult:
        self._push_history("score_left")
        self.left += 1
        self.pending = None
        return self._result(True, f"左方加一分，{self.score_text()}", "score_left")

    def _do_score_right(self, raw_text: str = "") -> ActionResult:
        self._push_history("score_right")
        self.right += 1
        self.pending = None
        return self._result(True, f"右方加一分，{self.score_text()}", "score_right")

    def _do_undo(self, raw_text: str = "") -> ActionResult:
        if not self._history:
            return self._result(False, "没有可以撤销的操作", "undo", reason="nothing_to_undo")
        snap = self._history.pop()
        self.left = snap["left"]
        self.right = snap["right"]
        self.pending = None
        return self._result(True, f"已撤销，当前{self.score_text()}", "undo")

    def _do_reset(self, raw_text: str = "") -> ActionResult:
        if not self.awaiting_reset_confirm:
            self.awaiting_reset_confirm = True
            self.reset_requested_at = time.time()
            return self._result(True, "确认要重置比分吗？请说「确认」", "reset",
                                reason="need_second_confirm")
        return self._commit_reset()

    def _do_confirm(self, raw_text: str = "") -> ActionResult:
        if self.awaiting_reset_confirm:
            return self._commit_reset()
        if self.pending:
            side = self.pending["side"]
            self.pending = None
            return self._do_score_left() if side == "left" else self._do_score_right()
        return self._result(False, "没有需要确认的这一分", "confirm", reason="nothing_to_confirm")

    def _do_report(self, raw_text: str = "") -> ActionResult:
        return self._result(True, f"当前比分，{self.score_text()}", "report")

    def _do_dismiss(self, raw_text: str = "") -> ActionResult:
        self.pending = None
        self.awaiting_reset_confirm = False
        return self._result(True, "已取消", "dismiss")

    def _commit_reset(self) -> ActionResult:
        self.awaiting_reset_confirm = False
        self._push_history("reset")
        self.left = 0
        self.right = 0
        self.pending = None
        return self._result(True, f"比分已重置，{self.score_text()}", "reset")

    # ------------------------------------------------------------ 内部
    def _push_history(self, reason: str) -> None:
        self._history.append({"left": self.left, "right": self.right,
                              "reason": reason, "at": time.time()})

    def _result(self, ok: bool, message: str, intent: str, reason: str = "",
                requires_reask: bool = False) -> ActionResult:
        return ActionResult(ok=ok, message=message, intent=intent, reason=reason,
                            requires_reask=requires_reask, state=self.snapshot())

    def _log(self, intent: str, raw_text: str, score: float, before: dict[str, int],
             result: ActionResult, source: str, extra: dict[str, Any] | None = None) -> None:
        event = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "intent": intent,
            "raw_text": raw_text,
            "similarity": round(float(score), 4),
            "ok": result.ok,
            "reason": result.reason,
            "before": before,
            "after": {"left": self.left, "right": self.right},
        }
        if extra:
            event.update(extra)
        self.events.append(event)
        if self.log_events and self._log_path:
            os.makedirs(os.path.dirname(self._log_path) or ".", exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------ 落盘/恢复
    def save(self, path: str | None = None) -> str:
        target = self.cfg.resolve(path or self.cfg.persist_state)
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        payload = {
            "left": self.left,
            "right": self.right,
            "history": list(self._history),
            "pending": self.pending,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return target

    def load(self, path: str | None = None) -> bool:
        target = self.cfg.resolve(path or self.cfg.persist_state)
        if not os.path.exists(target):
            return False
        try:
            with open(target, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return False
        self.left = int(payload.get("left", 0))
        self.right = int(payload.get("right", 0))
        self._history = deque(payload.get("history", []), maxlen=200)
        self.pending = payload.get("pending")
        return True


# ---------------------------------------------------------------------------
# 共用计分状态机：视觉事件、语音纠错、实体按键共用同一份比分与规则。
# 采用 vendored 的 tt_referee（由 tools/sync_referee.py 从裁判工程同步）；
# 如果单独分发本工程且没有该目录，则退回上面这份本地实现。
try:  # noqa: F811
    from tt_referee.voice_compat import SharedScoreBoard as ScoreBoard
except ImportError:  # pragma: no cover
    pass
