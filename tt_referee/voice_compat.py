"""语音模块兼容层：让语音纠错与视觉、按键共用同一个计分状态机。

保持语音模块 `tt_voice.score.ScoreBoard` 的接口与消息风格：

    board = SharedScoreBoard(cfg)
    board.apply("score_left", raw_text="左方加一分", score=0.97)
    board.propose_point("right", confidence=0.6)   # 视觉提出的待确认
    board.apply("confirm")
    board.snapshot()["left"], board.snapshot()["right"]

内部全部委托给 `ScoreEngine`；日志仍按语音模块的 events.jsonl 格式写，
方便沿用原来的复盘脚本。
"""

from __future__ import annotations

import json
import os
import time

from .scoring import ActionResult, ScoreEngine, ScoringRules


class SharedScoreBoard:
    def __init__(self, cfg=None, rules: ScoringRules | None = None,
                 log_path: str | None = None, log_events: bool = True):
        self.cfg = cfg
        self.log_events = log_events
        a_on_left = bool(getattr(cfg, "a_on_left", True)) if cfg is not None else True
        if rules is None:
            rules = ScoringRules()
            rules.a_on_left = a_on_left
            physical_self = getattr(cfg, "self_side", "left") if cfg is not None else "left"
            if physical_self in ("left", "right"):
                rules.self_side = ("A" if physical_self == "left" else "B") \
                    if a_on_left else ("B" if physical_self == "left" else "A")
        self.engine = ScoreEngine(rules, log_events=False)
        self.events: list[dict] = []
        if log_path:
            self._log_path = os.path.abspath(log_path)
        elif cfg is not None and getattr(cfg, "event_log", None):
            self._log_path = cfg.resolve(cfg.event_log)
        else:
            self._log_path = None
        self._state_path = cfg.resolve(cfg.persist_state) \
            if cfg is not None and getattr(cfg, "persist_state", None) else None

    # ------------------------------------------------------------ 兼容属性
    @property
    def left(self) -> int:
        return self.engine.points["A" if self.engine.side_of_a == "left" else "B"]

    @property
    def right(self) -> int:
        return self.engine.points["A" if self.engine.side_of_a == "right" else "B"]

    @property
    def pending(self):
        if not self.engine.pending:
            return None
        pending = dict(self.engine.pending)
        pending["side"] = self._to_physical(pending.get("side"))
        return pending

    @property
    def awaiting_reset_confirm(self) -> bool:
        return self.engine.awaiting_reset_confirm

    @awaiting_reset_confirm.setter
    def awaiting_reset_confirm(self, value: bool) -> None:
        self.engine.awaiting_reset_confirm = bool(value)

    def score_text(self) -> str:
        return f"左方{self.left}比{self.right}右方"

    # ------------------------------------------------------------ 映射
    def _to_side(self, physical: str | None) -> str | None:
        if physical not in ("left", "right"):
            return physical if physical in ("A", "B") else None
        return ("A" if self.engine.side_of_a == physical else "B")

    def _to_physical(self, side: str | None) -> str | None:
        if side not in ("A", "B"):
            return side
        return "left" if (self.engine.side_of_a == "left") == (side == "A") else "right"

    def _side_cn(self, side: str) -> str:
        return "左方" if self._to_physical(side) == "left" else "右方"

    # ------------------------------------------------------------ 语音意图
    def apply(self, intent: str, raw_text: str = "", score: float = 0.0,
              source: str = "voice") -> ActionResult:
        before = (self.left, self.right)
        result = self.engine.apply_voice_intent(intent, raw_text, score)
        result.message = self._voice_message(intent, result)
        result.state = self.snapshot()
        self._log_voice(intent, raw_text, score, before, result, source)
        return result

    def _voice_message(self, intent: str, result: ActionResult) -> str:
        if intent in ("score_left", "score_right", "score_self", "score_other"):
            if result.ok and result.intent.startswith("score_"):
                side = result.intent.split("_")[1].upper()
                return f"{self._side_cn(side)}加一分，{self.score_text()}"
            return result.message
        if intent == "undo":
            return f"已撤销，当前{self.score_text()}" if result.ok else result.message
        if intent == "reset":
            if result.reason == "need_second_confirm":
                return "确认要重置比分吗？请说「确认」"
            return f"比分已重置，{self.score_text()}" if result.ok else result.message
        if intent == "confirm":
            if result.ok and result.intent.startswith("score_"):
                side = result.intent.split("_")[1].upper()
                return f"{self._side_cn(side)}加一分，{self.score_text()}"
            return result.message
        if intent == "report":
            return f"当前比分，{self.score_text()}"
        if intent == "dismiss":
            return "已取消" if result.ok else result.message
        return result.message

    # ------------------------------------------------------------ 视觉提出
    def propose_point(self, side: str, confidence: float = 0.0, note: str = "",
                      auto_confirm: bool = False) -> ActionResult:
        physical = side
        side_ab = self._to_side(physical if physical in ("left", "right") else side)
        if side_ab not in ("A", "B"):
            result = ActionResult(ok=False, intent="propose", reason="side_unknown",
                                  message="未知的得分方", state=self.snapshot())
            return result
        if auto_confirm and confidence >= self.engine.rules.min_auto_confidence:
            result = self.engine.award(side_ab, reason=note or "vision", confidence=confidence,
                                       source="vision")
            result.message = f"{self._side_cn(side_ab)}加一分，{self.score_text()}"
        else:
            result = self.engine.propose_pending(side_ab, note or "proposed", confidence)
            result.message = f"检测到{self._side_cn(side_ab)}可能得分，请说「确认这一分」"
        result.state = self.snapshot()
        return result

    def confirm_pending(self, side: str | None = None) -> ActionResult:
        side_ab = self._to_side(side) if side else None
        result = self.engine.confirm_pending(side_ab)
        result.message = self._voice_message("confirm", result)
        result.state = self.snapshot()
        return result

    def snapshot(self) -> dict:
        snap = self.engine.snapshot()
        return {"left": self.left, "right": self.right,
                "pending": self.pending,
                "awaiting_reset_confirm": snap["awaiting_reset_confirm"],
                "can_undo": bool(self.engine.history),
                "sets": snap["sets"], "server": snap["server"],
                "match_over": snap["match_over"], "state": snap["state"]}

    # ------------------------------------------------------------ 日志
    def _log_voice(self, intent: str, raw_text: str, score: float,
                   before: tuple[int, int], result: ActionResult, source: str) -> None:
        event = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "source": source,
                 "intent": intent, "raw_text": raw_text,
                 "similarity": round(float(score), 4),
                 "ok": result.ok, "reason": result.reason,
                 "before": {"left": before[0], "right": before[1]},
                 "after": {"left": self.left, "right": self.right}}
        self.events.append(event)
        if self.log_events and self._log_path:
            os.makedirs(os.path.dirname(self._log_path) or ".", exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------ 落盘
    def save(self, path: str | None = None) -> str:
        return self.engine.save(path or self._state_path or "score_state.json")

    def load(self, path: str | None = None) -> bool:
        target = path or self._state_path
        return self.engine.load(target) if target else False
