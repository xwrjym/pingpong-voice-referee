"""从观测到比分的串联层。

    Observation -> Tracker -> EventExtractor -> EventFuser -> RallyJudge -> ScoreEngine

三个模型可以同时注册；事件在 EventFuser 里去重与融合，再交给同一个状态机。
语音与按键通过 `apply_voice` 直接改状态机，不经过视觉链路。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .adapters import TTNetAdapter
from .events import EventConfig, EventExtractor, EventFuser
from .geometry import Homography
from .rally import RallyConfig, RallyJudge
from .schema import Event, Observation, RallyOutcome
from .scoring import ActionResult, ScoreEngine, ScoringRules, STATE_PENDING, STATE_WAIT_SERVE
from .tracking import Tracker


@dataclass
class PipelineConfig:
    event: EventConfig = field(default_factory=EventConfig)
    rally: RallyConfig = field(default_factory=RallyConfig)
    source_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class FeedResult:
    events: list[Event] = field(default_factory=list)
    outcome: RallyOutcome | None = None
    action: ActionResult | None = None
    state: dict[str, Any] = field(default_factory=dict)


class RefereePipeline:
    def __init__(self, config: PipelineConfig | None = None,
                 rules: ScoringRules | None = None,
                 homography: Homography | None = None,
                 log_path: str | None = None, log_events: bool = True):
        self.config = config or PipelineConfig()
        self.engine = ScoreEngine(rules, log_path=log_path, log_events=log_events)
        self.homography = homography
        self.trackers: dict[str, Tracker] = {}
        self.extractors: dict[str, EventExtractor] = {}
        self.adapters: dict[str, Any] = {}
        self.fuser = EventFuser(self.config.event, self.config.source_weights)
        self.judge = RallyJudge(self.config.rally)
        self._sync_judge()
        self.last_events: list[Event] = []
        self.last_outcome: RallyOutcome | None = None
        self._ignore_events_until = 0.0     # 判分后的冷却期，避免同一回合的残留事件再判一次

    # ------------------------------------------------------------ 数据源
    def register_source(self, name: str, adapter: Any | None = None) -> None:
        self.trackers[name] = Tracker()
        self.extractors[name] = EventExtractor(source=name, homography=self.homography,
                                               cfg=self.config.event)
        if adapter is not None:
            self.adapters[name] = adapter

    def feed_raw(self, source: str, frame_id: int, timestamp_s: float, raw: Any,
                 image_size: tuple[int, int] = (1920, 1080)) -> FeedResult:
        if source not in self.adapters:
            raise KeyError(f"数据源 {source} 没有注册适配器")
        obs = self.adapters[source].to_observation(frame_id, timestamp_s, raw, image_size)
        return self.feed(obs)

    # ------------------------------------------------------------ 主入口
    def feed(self, obs: Observation) -> FeedResult:
        if obs.source not in self.trackers:
            self.register_source(obs.source)
        result = FeedResult()
        if self.engine.state == STATE_PENDING:
            # 待人工确认期间不接受新的视觉判定，避免连续加分
            result.state = self.engine.snapshot()
            return result
        if obs.timestamp_s < self._ignore_events_until:
            if obs.source not in self.trackers:
                self.register_source(obs.source)
            self.trackers[obs.source].update(obs)
            result.state = self.engine.snapshot()
            return result

        point = self.trackers[obs.source].update(obs)
        candidates: list[Event] = []
        if point is not None:
            candidates = self.extractors[obs.source].on_point(point, obs)
            for event in candidates:
                self.fuser.add(event, obs.timestamp_s)
        fused = self.fuser.ready(obs.timestamp_s)
        result.events, result.outcome, result.action = self._consume(fused, obs.timestamp_s)
        result.state = self.engine.snapshot()
        return result

    def _consume(self, fused: list[Event], now: float):
        events: list[Event] = []
        outcome = None
        action = None
        for event in fused:
            events.append(event)
            if self.engine.state == STATE_WAIT_SERVE:
                self.engine.start_rally()
            if event.type == "BOUNCE":
                self.engine.note_bounce()
            out = self.judge.on_event(event, now)
            if out is not None:
                outcome = out
                if out.let:
                    action = self.engine.let(out.reason, source="vision")
                    self._sync_judge()
                    self.last_outcome = out
                    self._ignore_events_until = event.timestamp_s + 0.8
                    continue
                out.player_side = self._player_of(out.side)
                action = self.engine.finish_rally(out.player_side, out.reason, out.confidence,
                                                  source="vision", events=out.events)
                self._sync_judge()
                self.last_outcome = out
                self._ignore_events_until = event.timestamp_s + 0.8
        if outcome is None:
            deadline = self.judge.tick(now)
            if deadline is not None:
                outcome = deadline
                if deadline.let:
                    action = self.engine.let(deadline.reason, source="vision")
                    self._sync_judge()
                    self.last_outcome = deadline
                    self._ignore_events_until = now + 0.8
                    return events, outcome, action
                deadline.player_side = self._player_of(deadline.side)
                action = self.engine.finish_rally(deadline.player_side, deadline.reason,
                                                  deadline.confidence, source="vision",
                                                  events=deadline.events)
                self._sync_judge()
                self.last_outcome = deadline
                self._ignore_events_until = now + 0.8
        self.last_events.extend(events)
        return events, outcome, action

    def flush(self, now: float) -> FeedResult:
        fused = self.fuser.flush()
        events, outcome, action = self._consume(fused, now)
        if outcome is None:
            deadline = self.judge.tick(now)
            if deadline is not None:
                outcome = deadline
                if deadline.let:
                    action = self.engine.let(deadline.reason, source="vision")
                    self._sync_judge()
                    self.last_outcome = deadline
                    self._ignore_events_until = now + 0.8
                    return FeedResult(events=events, outcome=outcome, action=action,
                                      state=self.engine.snapshot())
                deadline.player_side = self._player_of(deadline.side)
                action = self.engine.finish_rally(deadline.player_side, deadline.reason,
                                                  deadline.confidence, source="vision",
                                                  events=deadline.events)
                self._sync_judge()
                self.last_outcome = deadline
                self._ignore_events_until = now + 0.8
        return FeedResult(events=events, outcome=outcome, action=action,
                          state=self.engine.snapshot())

    # ------------------------------------------------------------ 人工输入
    def apply_voice(self, intent: str, raw_text: str = "", score: float = 1.0) -> ActionResult:
        result = self.engine.apply_voice_intent(intent, raw_text, score)
        self._sync_judge()
        return result

    def confirm_pending(self, side: str | None = None) -> ActionResult:
        result = self.engine.confirm_pending(side)
        self._sync_judge()
        return result

    def reject_pending(self) -> ActionResult:
        result = self.engine.reject_pending()
        self._sync_judge()
        return result

    def snapshot(self) -> dict[str, Any]:
        return self.engine.snapshot()

    # ------------------------------------------------------------ 物理半台/选手
    def _server_physical(self) -> str:
        """当前发球选手对应的物理半台（A=左，B=右）。"""
        player_a_left = self.engine.side_of_a == "left"
        if self.engine.server == "A":
            return "A" if player_a_left else "B"
        return "B" if player_a_left else "A"

    def _player_of(self, physical: str) -> str:
        if physical not in ("A", "B"):
            return "UNKNOWN"
        physical_name = "left" if physical == "A" else "right"
        return "A" if self.engine.side_of_a == physical_name else "B"

    def _sync_judge(self) -> None:
        self.judge.reset(server=self._server_physical())
