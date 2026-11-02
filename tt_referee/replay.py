"""离线回放：把模型导出的 Observation JSONL 跑成事件序列。

这一层只做"轨迹 -> 事件"，不碰计分，方便单独调参和对比三个模型。

输入：每行一个 Observation 字典（字段见 schema.py），或一个 JSON 数组。
输出：融合后的 Event 列表，可写成 JSONL 交给计分状态机或人工复查。
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field

from .events import EventConfig, EventExtractor, EventFuser
from .geometry import Homography
from .schema import Event, Observation
from .timeline import unify_frame_ids
from .tracking import Tracker


def load_observations(path: str, video_fps: float | None = None,
                      unify: bool = True) -> list[Observation]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read().strip()
    if not text:
        return []
    rows = json.loads(text) if text.startswith("[") else \
        [json.loads(line) for line in text.splitlines() if line.strip()]
    observations = [Observation.from_dict(row) for row in rows]
    if unify:
        return unify_frame_ids(observations, video_fps)
    observations.sort(key=lambda o: (o.timestamp_s, o.source, o.frame_id))
    return observations


def save_events(events: list[Event], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
    return path


@dataclass
class EventLayer:
    homography: Homography | None = None
    config: EventConfig = field(default_factory=EventConfig)
    source_weights: dict[str, float] = field(default_factory=dict)
    trackers: dict[str, Tracker] = field(default_factory=dict)
    extractors: dict[str, EventExtractor] = field(default_factory=dict)
    fuser: EventFuser = field(init=False)
    events: list[Event] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.fuser = EventFuser(self.config, self.source_weights)

    def _ensure(self, source: str) -> None:
        if source not in self.trackers:
            self.trackers[source] = Tracker()
            self.extractors[source] = EventExtractor(source, homography=self.homography,
                                                     cfg=self.config)

    def feed(self, obs: Observation) -> list[Event]:
        self._ensure(obs.source)
        point = self.trackers[obs.source].update(obs)
        if point is not None:
            for event in self.extractors[obs.source].on_point(point, obs):
                self.fuser.add(event, obs.timestamp_s)
        fused = self.fuser.ready(obs.timestamp_s)
        self.events.extend(fused)
        return fused

    def flush(self) -> list[Event]:
        fused = self.fuser.flush()
        self.events.extend(fused)
        return fused


def extract_events(observations: list[Observation],
                   homography: Homography | None = None,
                   config: EventConfig | None = None,
                   source_weights: dict[str, float] | None = None) -> tuple[list[Event], dict]:
    layer = EventLayer(homography=homography, config=config or EventConfig(),
                       source_weights=source_weights or {})
    for obs in observations:
        layer.feed(obs)
    layer.flush()
    return layer.events, stats(observations, layer.events)


def stats(observations: list[Observation], events: list[Event]) -> dict:
    counts = Counter(e.type for e in events)
    per_source: dict[str, dict] = {}
    for source in sorted({o.source for o in observations}):
        series = [o for o in observations if o.source == source]
        visible = [o for o in series if o.ball.visible]
        times = [o.timestamp_s for o in series]
        gaps = sum(1 for a, b in zip(times, times[1:]) if b - a > 0.35)
        dt = (times[-1] - times[0]) / max(1, len(times) - 1) if len(times) > 1 else 0.0
        per_source[source] = {"frames": len(series),
                              "visible_ratio": round(len(visible) / max(1, len(series)), 3),
                              "estimated_fps": round(1 / dt, 1) if dt else 0.0,
                              "gaps_over_0_35s": gaps}
    return {"observations": len(observations), "events": len(events),
            "event_counts": dict(counts), "sources": per_source,
            "duration_s": round(observations[-1].timestamp_s - observations[0].timestamp_s, 2)
            if observations else 0.0}
