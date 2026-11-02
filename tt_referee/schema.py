"""统一数据契约：Observation / Event / RallyOutcome。

字段命名与《TTNet、TrackNet、YOLO 共用计分状态机方案》第 7 节保持一致，
方便三个模型的适配器直接对接，也方便把同一段视频的回放结果做逐帧 diff。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Source = str                      # "yolo" / "tracknet" / "ttnet" / 其它
Side = Literal["A", "B", "UNKNOWN"]
EventType = Literal["BOUNCE", "NET_HIT", "OUT", "RALLY_START", "RALLY_END"]


@dataclass
class BallObservation:
    """单帧球的观测。没有检测到球时 visible=False，坐标必须为 None。"""

    visible: bool = False
    center_px: tuple[float, float] | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.visible:
            if self.center_px is None:
                raise ValueError("visible=True 时必须给出 center_px")
            if len(self.center_px) != 2:
                raise ValueError("center_px 必须是 (x, y)")
            if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
                raise ValueError("confidence 必须在 0~1")
        else:
            if self.center_px is not None or self.bbox_xyxy is not None:
                raise ValueError("visible=False 时不能带坐标（避免把预测位置当成真实检测）")

    @classmethod
    def lost(cls) -> "BallObservation":
        return cls(visible=False)

    @classmethod
    def from_bbox(cls, bbox: tuple[float, float, float, float],
                  confidence: float | None = None) -> "BallObservation":
        x1, y1, x2, y2 = bbox
        return cls(visible=True, center_px=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                   bbox_xyxy=tuple(float(v) for v in bbox), confidence=confidence)


@dataclass
class EventScores:
    """模型自带的事件候选分数（不是已确认事件）。"""

    bounce: float | None = None
    net_hit: float | None = None


@dataclass
class Observation:
    frame_id: int
    timestamp_s: float
    source: Source
    image_size: tuple[int, int] = (1920, 1080)
    ball: BallObservation = field(default_factory=BallObservation)
    event_scores: EventScores = field(default_factory=EventScores)

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id 不能为负")
        if self.timestamp_s < 0:
            raise ValueError("timestamp_s 不能为负")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        ball = data.get("ball") or {}
        center = ball.get("center_px")
        bbox = ball.get("bbox_xyxy")
        return cls(
            frame_id=int(data["frame_id"]),
            timestamp_s=float(data["timestamp_s"]),
            source=str(data["source"]),
            image_size=tuple(data.get("image_size", (1920, 1080))),
            ball=BallObservation(
                visible=bool(ball.get("visible", False)),
                center_px=tuple(center) if center else None,
                bbox_xyxy=tuple(bbox) if bbox else None,
                confidence=ball.get("confidence"),
            ),
            event_scores=EventScores(**(data.get("event_scores") or {})),
        )


@dataclass
class Event:
    """事件处理层的输出，也是计分状态机唯一认可的输入。"""

    event_id: str
    timestamp_s: float
    type: EventType
    side: Side = "UNKNOWN"
    position_px: tuple[float, float] | None = None
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)
    conflict: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        pos = data.get("position_px")
        return cls(
            event_id=str(data["event_id"]),
            timestamp_s=float(data["timestamp_s"]),
            type=data["type"],
            side=data.get("side", "UNKNOWN"),
            position_px=tuple(pos) if pos else None,
            confidence=float(data.get("confidence", 0.0)),
            sources=list(data.get("sources", [])),
            conflict=bool(data.get("conflict", False)),
            meta=dict(data.get("meta", {})),
        )


@dataclass
class RallyOutcome:
    """一个回合的判定结果；pending=True 表示需要人工确认。"""

    side: Side
    reason: str
    timestamp_s: float
    confidence: float
    pending: bool = False
    let: bool = False           # 擦网重发：不改变比分与发球方
    events: list[Event] = field(default_factory=list)
    player_side: str = ""       # 物理半台 -> 选手（换边后不同）

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "reason": self.reason,
            "timestamp_s": self.timestamp_s,
            "confidence": self.confidence,
            "pending": self.pending,
            "let": self.let,
            "player_side": self.player_side,
            "events": [e.to_dict() for e in self.events],
        }


def dumps(obj: Any) -> str:
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    return json.dumps(obj, ensure_ascii=False)
