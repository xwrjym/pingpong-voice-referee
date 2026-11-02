"""公用计分状态机：三个模型、语音纠正、实体按键共用这一套规则。

规则与感知完全解耦：这里不认识像素、模型和事件类型，只接受"这一分给 A/B"
以及少量人工意图。因此同一段比赛：

* YOLO / TrackNet / TTNet 的适配器都可以驱动它；
* 语音纠错与实体按键走同一套 `apply_voice_intent`；
* 换检测模型不会改变乒乓球规则，也能直接对比三者的最终判分准确率。

状态：

    WAIT_SERVE --start_rally--> SERVE --首个落台--> RALLY --判定--> WAIT_SERVE
                                   \--低置信/冲突--> PENDING --确认--> WAIT_SERVE

规则默认值：11 分制、领先 2 分、每 2 分发球、10:10 后每分发球、五局三胜。
每局结束交换发球与场地；决胜局（最后一局）中途不换边。
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

Side = Literal["A", "B"]

STATE_WAIT_SERVE = "WAIT_SERVE"
STATE_SERVE = "SERVE"
STATE_RALLY = "RALLY"
STATE_PENDING = "PENDING"
STATE_MATCH_OVER = "MATCH_OVER"


@dataclass
class ScoringRules:
    points_to_win: int = 11
    win_by: int = 2
    serve_interval: int = 2
    deuce_serve_interval: int = 1
    best_of: int = 5
    final_set_switch_at: int = 0      # 0 = 决胜局中途不换边（本项目采用）
    a_on_left: bool = True
    self_side: str = "A"          # 个人自用："我"是哪一侧
    min_auto_confidence: float = 0.70

    @property
    def sets_to_win(self) -> int:
        return self.best_of // 2 + 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoringRules":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, path: str) -> "ScoringRules":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


@dataclass
class ActionResult:
    ok: bool
    message: str = ""
    intent: str = ""
    reason: str = ""
    requires_reask: bool = False
    state: dict[str, Any] = field(default_factory=dict)


class ScoreEngine:
    def __init__(self, rules: ScoringRules | None = None, log_path: str | None = None,
                 log_events: bool = True):
        self.rules = rules or ScoringRules()
        self.log_events = log_events
        self.log_path = log_path
        self.points: dict[str, int] = {"A": 0, "B": 0}
        self.sets: dict[str, int] = {"A": 0, "B": 0}
        self.server: str = "A"
        self.serves_since_change = 0
        self.set_index = 0
        self.side_of_a = "left" if self.rules.a_on_left else "right"
        self.state = STATE_WAIT_SERVE
        self.pending: dict[str, Any] | None = None
        self.awaiting_reset_confirm = False
        self.match_over = False
        self.last_winner: str | None = None
        self.history: list[dict[str, Any]] = []
        self.redo_stack: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []      # 外部动作审计（可据此重放）
        self.stats_by_source: dict[str, int] = {}
        self.events: list[dict[str, Any]] = []

    # ------------------------------------------------------------ 状态
    def snapshot(self) -> dict[str, Any]:
        return {
            "points": dict(self.points),
            "sets": dict(self.sets),
            "server": self.server,
            "serves_since_change": self.serves_since_change,
            "set_index": self.set_index,
            "side_of_a": self.side_of_a,
            "state": self.state,
            "pending": copy.deepcopy(self.pending),
            "awaiting_reset_confirm": self.awaiting_reset_confirm,
            "match_over": self.match_over,
            "serving_side": self.server,
            "serves_remaining": self._serves_remaining(),
            "points_text": f"{self.points['A']}:{self.points['B']}",
            "sets_text": f"{self.sets['A']}:{self.sets['B']}",
        }

    def display(self) -> str:
        """给 UI/终端用的一行记分牌。"""
        return (f"A {self.points['A']:>2} : {self.points['B']:<2} B  |  "
                f"局 {self.sets['A']}:{self.sets['B']}  |  第 {self.set_index + 1} 局  |  "
                f"发球 {self.server}（剩 {self._serves_remaining()} 个）")

    def _serves_remaining(self) -> int:
        interval = (self.rules.deuce_serve_interval if self._is_deuce()
                    else self.rules.serve_interval)
        return max(0, interval - self.serves_since_change)

    def start_rally(self) -> None:
        if self.match_over or self.state == STATE_PENDING:
            return
        if self.state == STATE_WAIT_SERVE:
            self.state = STATE_SERVE
            self._log({"action": "start_rally", "server": self.server})

    def note_bounce(self) -> None:
        """第一个落台后从"发球"进入"对打"。"""
        if self.state == STATE_SERVE:
            self.state = STATE_RALLY

    def let(self, reason: str = "let_net_serve", source: str = "vision") -> ActionResult:
        """擦网重发：比分、局分、发球方与已发球数都不变，只重置当前回合。"""
        self.pending = None
        self.state = STATE_WAIT_SERVE
        self._log({"action": "let", "reason": reason, "source": source})
        self._record_action("let", reason=reason, source=source)
        return ActionResult(ok=True, intent="let", reason=reason,
                            message="擦网，重发这一分", state=self.snapshot())

    # ------------------------------------------------------------ 视觉判定
    def finish_rally(self, side: str, reason: str, confidence: float = 1.0,
                     source: str = "vision", events: list[Any] | None = None) -> ActionResult:
        if side not in ("A", "B"):
            return self._pending(side="UNKNOWN", reason=reason or "side_unknown",
                                 confidence=confidence, source=source, events=events)
        if confidence < self.rules.min_auto_confidence:
            return self._pending(side=side, reason=reason, confidence=confidence,
                                 source=source, events=events)
        return self.award(side, reason=reason, confidence=confidence, source=source,
                          events=events)

    def _pending(self, side: str, reason: str, confidence: float, source: str,
                 events: list[Any] | None) -> ActionResult:
        self.pending = {"side": side, "reason": reason, "confidence": confidence,
                        "source": source, "events": events or [], "at": time.time()}
        self.state = STATE_PENDING
        self._log({"action": "pending", "side": side, "reason": reason,
                   "confidence": confidence})
        return ActionResult(ok=False, intent="pending",
                            reason="pending_confirmation",
                            message="这一分无法判定，请人工确认" if side == "UNKNOWN"
                                    else f"疑似 {side} 得分，请确认",
                            state=self.snapshot())

    def propose_pending(self, side: str, reason: str = "proposed",
                        confidence: float = 0.0, source: str = "vision") -> ActionResult:
        """视觉提出"疑似得分"，但不直接加分——等语音/按键确认。"""
        if side not in ("A", "B"):
            side = "UNKNOWN"
        self.pending = {"side": side, "reason": reason, "confidence": confidence,
                        "source": source, "events": [], "at": time.time()}
        self.state = STATE_PENDING
        self._log({"action": "propose_pending", "side": side, "reason": reason})
        return ActionResult(ok=True, intent="propose", reason="awaiting_confirmation",
                            message=f"疑似 {side} 得分，等待确认", state=self.snapshot())

    def confirm_pending(self, side: str | None = None) -> ActionResult:
        if self.awaiting_reset_confirm:
            self.awaiting_reset_confirm = False
            return self._commit_reset()
        if not self.pending:
            return ActionResult(ok=False, intent="confirm", reason="nothing_to_confirm",
                                message="没有需要确认的这一分", state=self.snapshot())
        chosen = side or self.pending.get("side")
        reason = f"confirmed:{self.pending.get('reason', '')}"
        confidence = float(self.pending.get("confidence", 0.0))
        self.pending = None
        if chosen not in ("A", "B"):
            self.state = STATE_WAIT_SERVE
            return ActionResult(ok=False, intent="confirm", reason="side_unknown",
                                message="请说明这一分给哪一方", state=self.snapshot())
        return self.award(chosen, reason=reason, confidence=confidence, source="human")

    def reject_pending(self) -> ActionResult:
        self.pending = None
        self.state = STATE_WAIT_SERVE
        self._log({"action": "reject_pending"})
        self._record_action("reject_pending")
        return ActionResult(ok=True, intent="dismiss", message="已取消本次判定",
                            state=self.snapshot())

    # ------------------------------------------------------------ 计分
    def award(self, side: str, reason: str = "manual", confidence: float = 1.0,
              source: str = "vision", events: list[Any] | None = None) -> ActionResult:
        if side not in ("A", "B"):
            raise ValueError("side 只能是 A 或 B")
        if self.match_over:
            return ActionResult(ok=False, intent=f"score_{side.lower()}",
                                reason="match_over", message="比赛已结束",
                                state=self.snapshot())
        if self.state == STATE_PENDING and source == "vision":
            # 待人工确认期间，视觉不能再自动加一分（防止模型连续输出重复加分）
            return ActionResult(ok=False, intent=f"score_{side.lower()}",
                                reason="pending_confirmation",
                                message="存在待确认的回合，请先确认或取消",
                                state=self.snapshot())
        if self.state == STATE_PENDING:
            # 语音/按键是人工纠错：以人工动作为准，取消待确认项
            self.pending = None
        self.history.append(copy.deepcopy(self.snapshot()))
        self.redo_stack.clear()
        before = dict(self.points)
        self.points[side] += 1
        self.last_winner = side
        set_won = self._check_set()
        if not set_won:
            self._update_server()
            self.state = STATE_WAIT_SERVE
        self._log({"action": "award", "side": side, "reason": reason,
                   "confidence": confidence, "source": source,
                   "before": before, "after": dict(self.points),
                   "sets": dict(self.sets)})
        self._record_action("award", side=side, reason=reason, confidence=confidence,
                            source=source)
        self.stats_by_source[source] = self.stats_by_source.get(source, 0) + 1
        return ActionResult(ok=True, intent=f"score_{side.lower()}",
                            message=f"{side} 得分，{self.points_text}",
                            state=self.snapshot())

    @property
    def points_text(self) -> str:
        return f"{self.points['A']}:{self.points['B']}"

    @property
    def sets_text(self) -> str:
        return f"{self.sets['A']}:{self.sets['B']}"

    def _is_deuce(self) -> bool:
        target = self.rules.points_to_win - 1
        return self.points["A"] >= target and self.points["B"] >= target

    def _update_server(self) -> None:
        self.serves_since_change += 1
        interval = (self.rules.deuce_serve_interval if self._is_deuce()
                    else self.rules.serve_interval)
        if self.serves_since_change >= interval:
            self.server = "B" if self.server == "A" else "A"
            self.serves_since_change = 0

    def _check_set(self) -> bool:
        side = "A" if self.points["A"] >= self.points["B"] else "B"
        lead = abs(self.points["A"] - self.points["B"])
        if self.points[side] >= self.rules.points_to_win and lead >= self.rules.win_by:
            self.sets[side] += 1
            self._log({"action": "set_end", "side": side, "sets": dict(self.sets)})
            if self.sets[side] >= self.rules.sets_to_win:
                self.match_over = True
                self.state = STATE_MATCH_OVER
                return True
            self.set_index += 1
            self.points = {"A": 0, "B": 0}
            self.serves_since_change = 0
            self.server = "B" if self.server == "A" else "A"
            self._toggle_sides()
            self.state = STATE_WAIT_SERVE
            return True
        # 决胜局中途换边（默认关闭：0 表示不换）
        if self.rules.final_set_switch_at > 0 and \
                self.set_index == self.rules.best_of - 1 and \
                sum(self.points.values()) == self.rules.final_set_switch_at:
            self._toggle_sides()
        return False

    def _toggle_sides(self) -> None:
        self.side_of_a = "right" if self.side_of_a == "left" else "left"

    # ------------------------------------------------------------ 语音/按键
    def apply_voice_intent(self, intent: str, raw_text: str = "",
                           score: float = 1.0) -> ActionResult:
        if self.awaiting_reset_confirm and intent not in ("confirm", "reset"):
            # 重置等待确认期间说了别的口令：放弃重置，执行新动作
            self.awaiting_reset_confirm = False
        if intent in ("score_left", "score_right"):
            physical = intent.split("_")[1]
            return self._award_physical(physical, intent)
        if intent in ("score_self", "score_other"):
            mine = self.rules.self_side
            other = "B" if mine == "A" else "A"
            return self.award(mine if intent == "score_self" else other,
                              reason=f"voice:{raw_text}", source="voice")
        if intent == "undo":
            return self.undo()
        if intent == "reset":
            if not self.awaiting_reset_confirm:
                self.awaiting_reset_confirm = True
                return ActionResult(ok=True, intent="reset", reason="need_second_confirm",
                                    message="确认要重置比分吗？请说确认",
                                    state=self.snapshot())
            return self._commit_reset()
        if intent == "confirm":
            return self.confirm_pending()
        if intent == "report":
            return ActionResult(ok=True, intent="report",
                                message=f"当前比分 {self.points_text}，局分 {self.sets_text}",
                                state=self.snapshot())
        if intent == "dismiss":
            self.pending = None
            self.awaiting_reset_confirm = False
            if self.state == STATE_PENDING:
                self.state = STATE_WAIT_SERVE
            return ActionResult(ok=True, intent="dismiss", message="已取消",
                                state=self.snapshot())
        return ActionResult(ok=False, intent=intent, reason="unknown_intent",
                            message="未知口令", state=self.snapshot())

    def _award_physical(self, physical: str, intent: str) -> ActionResult:
        side = "A" if self.side_of_a == physical else "B"
        return self.award(side, reason=f"voice:{intent}", source="voice")

    def undo(self) -> ActionResult:
        if not self.history:
            return ActionResult(ok=False, intent="undo", reason="nothing_to_undo",
                                message="没有可以撤销的操作", state=self.snapshot())
        snap = self.history.pop()
        self.redo_stack.append(copy.deepcopy(self.snapshot()))
        self.points = snap["points"]
        self.sets = snap["sets"]
        self.server = snap["server"]
        self.serves_since_change = snap.get("serves_since_change", 0)
        self.set_index = snap["set_index"]
        self.side_of_a = snap["side_of_a"]
        self.match_over = False
        self.state = STATE_WAIT_SERVE
        self.pending = None
        self._log({"action": "undo", "after": dict(self.points)})
        self._record_action("undo")
        return ActionResult(ok=True, intent="undo", message=f"已撤销，{self.points_text}",
                            state=self.snapshot())

    def redo(self) -> ActionResult:
        if not self.redo_stack:
            return ActionResult(ok=False, intent="redo", reason="nothing_to_redo",
                                message="没有可以重做的操作", state=self.snapshot())
        snap = self.redo_stack.pop()
        self.history.append(copy.deepcopy(self.snapshot()))
        self.points = snap["points"]
        self.sets = snap["sets"]
        self.server = snap["server"]
        self.serves_since_change = snap.get("serves_since_change", 0)
        self.set_index = snap["set_index"]
        self.side_of_a = snap["side_of_a"]
        self.match_over = snap["match_over"]
        self.state = STATE_MATCH_OVER if self.match_over else STATE_WAIT_SERVE
        self.pending = None
        self._log({"action": "redo", "after": dict(self.points)})
        self._record_action("redo")
        return ActionResult(ok=True, intent="redo", message=f"已重做，{self.points_text}",
                            state=self.snapshot())

    def _commit_reset(self) -> ActionResult:
        self.history.append(copy.deepcopy(self.snapshot()))
        self.points = {"A": 0, "B": 0}
        self.sets = {"A": 0, "B": 0}
        self.set_index = 0
        self.server = "A"
        self.serves_since_change = 0
        self.side_of_a = "left" if self.rules.a_on_left else "right"
        self.match_over = False
        self.state = STATE_WAIT_SERVE
        self.pending = None
        self.awaiting_reset_confirm = False
        self._log({"action": "reset"})
        self._record_action("reset")
        return ActionResult(ok=True, intent="reset", message="比分已重置",
                            state=self.snapshot())

    def reset_now(self) -> ActionResult:
        """直接重置（审计回放用；人工入口仍走两段式 reset）。"""
        self.awaiting_reset_confirm = True
        return self._commit_reset()

    # ------------------------------------------------------------ 落盘
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = self.snapshot()
        payload["rules"] = self.rules.__dict__
        payload["history"] = self.history
        payload["redo_stack"] = self.redo_stack
        payload["actions"] = self.actions
        payload["stats_by_source"] = self.stats_by_source
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return path

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.points = payload.get("points", {"A": 0, "B": 0})
        self.sets = payload.get("sets", {"A": 0, "B": 0})
        self.server = payload.get("server", "A")
        self.set_index = payload.get("set_index", 0)
        self.side_of_a = payload.get("side_of_a", "left")
        self.match_over = payload.get("match_over", False)
        self.state = payload.get("state", STATE_WAIT_SERVE)
        if self.state == STATE_PENDING:
            self.state = STATE_WAIT_SERVE
        self.history = payload.get("history", [])
        self.redo_stack = payload.get("redo_stack", [])
        self.actions = payload.get("actions", [])
        self.stats_by_source = payload.get("stats_by_source", {})
        return True

    def _record_action(self, action: str, **params: Any) -> None:
        self.actions.append({"ts": time.time(), "action": action, **params})

    def _log(self, data: dict[str, Any]) -> None:
        event = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **data,
                 "state": self.state, "points": dict(self.points),
                 "sets": dict(self.sets)}
        self.events.append(event)
        if self.log_events and self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)) or ".", exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def replay_actions(actions: list[dict[str, Any]],
                   rules: ScoringRules | None = None) -> ScoreEngine:
    """按审计动作重放，重建比分（用于断电恢复校验与赛后复盘）。"""
    engine = ScoreEngine(rules or ScoringRules(), log_events=False)
    for item in actions:
        kind = item.get("action")
        if kind == "award":
            engine.award(item["side"], item.get("reason", "replay"),
                         float(item.get("confidence", 1.0)),
                         item.get("source", "replay"))
        elif kind == "undo":
            engine.undo()
        elif kind == "redo":
            engine.redo()
        elif kind == "reset":
            engine.reset_now()
        elif kind == "reject_pending":
            engine.reject_pending()
        elif kind == "let":
            engine.let(item.get("reason", "let"), item.get("source", "replay"))
    return engine
