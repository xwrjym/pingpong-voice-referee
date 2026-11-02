"""事件序列 -> 回合判定。

把"发生了什么"翻译成"这一分给谁"。只依赖 Event 与当前发球方，不碰像素。

判定规则（第一版，按业余比赛可解释的简化）：

1. 发球必须先在发球方半台落一次；首个落台在接发球方 → 发球失误，接发球方得分。
2. 连续两次落在同一半台 → 该半台没有成功回球，对方得分。
3. 落点在台外 → 落点所在半台得分（把球打飞的一方失分）。
4. 擦网本身不直接判分；擦网后球落在发球方自己半台或出界，判接发球方得分。
   发球擦网后仍合法落在接发球方半台 -> 重发（let），不改变比分与发球方。
5. 球消失、漏检、多模型冲突 → 不判分，进入"待人工确认"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Event, RallyOutcome


def other(side: str) -> str:
    return "B" if side == "A" else "A"


@dataclass
class RallyConfig:
    rally_timeout_s: float = 3.0      # 这么长时间没有新事件 -> 待确认
    net_timeout_s: float = 1.0        # 擦网后多久内没恢复 -> 判擦网失分
    min_confidence: float = 0.55      # 低于它的判定转为待确认


@dataclass
class _Bounce:
    side: str
    inside: bool
    timestamp_s: float
    confidence: float


@dataclass
class RallyJudge:
    cfg: RallyConfig = field(default_factory=RallyConfig)
    server: str = "A"
    started: bool = False
    bounces: list[_Bounce] = field(default_factory=list)
    net_hit: Event | None = None
    last_event_t: float = 0.0
    last_event: Event | None = None
    _events: list[Event] = field(default_factory=list)

    def reset(self, server: str | None = None) -> None:
        if server is not None:
            self.server = server
        self.started = False
        self.bounces.clear()
        self.net_hit = None
        self.last_event = None
        self._events.clear()

    # ------------------------------------------------------------ 主入口
    def on_event(self, event: Event, now: float | None = None) -> RallyOutcome | None:
        now = event.timestamp_s if now is None else now
        self.started = True
        self.last_event_t = max(self.last_event_t, event.timestamp_s)
        self.last_event = event
        self._events.append(event)

        # 擦网事件的侧别可以未知（球的精确落点还没发生），先记录，不当作冲突
        if event.type == "NET_HIT":
            self.net_hit = event
            return None

        if event.conflict or event.side == "UNKNOWN":
            return self._outcome("UNKNOWN", "conflicting_or_unknown_side",
                                 event.confidence, pending=True)

        if event.type == "BOUNCE":
            return self._on_bounce(event)

        if event.type == "OUT":
            return self._on_out(event)

        return None

    def _on_bounce(self, event: Event) -> RallyOutcome | None:
        self.bounces.append(_Bounce(event.side, True, event.timestamp_s, event.confidence))
        if len(self.bounces) == 1:
            if event.side != self.server:
                # 发球直接落在对方半台：发球失误
                return self._outcome(other(self.server), "serve_first_bounce_wrong_side",
                                     event.confidence)
            if self.net_hit is not None:
                # 擦网后落在自己半台：发球擦网失误
                return self._outcome(other(self.server), "serve_net_short",
                                     min(event.confidence, self.net_hit.confidence))
            return None
        prev = self.bounces[-2]
        if len(self.bounces) == 2 and self.net_hit is not None and prev.side != event.side:
            # 发球：第一跳在发球方，擦网后仍合法落在接发球方 -> 重发
            return self._let(self.net_hit, event)
        if prev.side == event.side:
            return self._outcome(other(event.side), "two_bounces_same_side",
                                 min(prev.confidence, event.confidence))
        # 正常来回，擦网记录作废（球已合法落台）
        self.net_hit = None
        return None

    def _on_out(self, event: Event) -> RallyOutcome | None:
        side = event.side
        if not self.bounces and side == self.server:
            return self._outcome(other(self.server), "serve_out_own_side",
                                 event.confidence)
        if self.net_hit is not None and side == self.server:
            return self._outcome(other(self.server), "net_then_out_own_side",
                                 min(event.confidence, self.net_hit.confidence))
        # 落在台外：落点所在半台得分（对方把球打飞）
        return self._outcome(side, "out_of_table", event.confidence)

    # ------------------------------------------------------------ 超时
    def tick(self, now: float) -> RallyOutcome | None:
        """没有新事件时的处理：球消失不是失分。"""
        if not self.started:
            return None
        if self.net_hit is not None and now - self.net_hit.timestamp_s >= self.cfg.net_timeout_s:
            if len(self.bounces) <= 1:
                # 发球擦网后没有合法落台 -> 发球失误
                return self._outcome(other(self.server), "serve_net_no_bounce",
                                     self.net_hit.confidence * 0.8)
            # 回合中的擦网没有后续证据：不猜，转人工
            return self._outcome("UNKNOWN", "net_unresolved", self.net_hit.confidence * 0.5,
                                 pending=True)
        if now - self.last_event_t >= self.cfg.rally_timeout_s:
            return self._outcome("UNKNOWN", "no_event_timeout", 0.0, pending=True)
        return None

    def _outcome(self, side: str, reason: str, confidence: float,
                 pending: bool = False) -> RallyOutcome:
        if confidence < self.cfg.min_confidence:
            pending = True
        outcome = RallyOutcome(side=side, reason=reason,
                               timestamp_s=self.last_event_t,
                               confidence=float(confidence), pending=pending,
                               events=list(self._events))
        self.reset()
        return outcome

    def _let(self, net_hit: Event, bounce: Event) -> RallyOutcome:
        outcome = RallyOutcome(side="UNKNOWN", reason="let_net_serve",
                               timestamp_s=bounce.timestamp_s,
                               confidence=min(net_hit.confidence, bounce.confidence),
                               pending=False, let=True, events=list(self._events))
        self.reset()
        return outcome
