"""轨迹 -> 事件：落台、擦网、出界，以及单源去重与跨模型融合。

三条原则对应方案第 4 节：

1. 用时间戳判断，不用固定帧数；
2. 球消失不产生失分事件，交给回合判定层按超时进入"待人工确认"；
3. 多模型同时跑时先融合去重：同一次落台只产生一个事件，冲突则标 UNKNOWN，
   不能各加一次分，也不能简单取置信度最大值。

单目限制：落点只在**落台时刻**投影到桌面；擦网优先采用模型自带分数
（TTNet），纯轨迹推断只作为低置信候选。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import BounceGeometry, Homography
from .schema import Event, Observation
from .tracking import TrackPoint


@dataclass
class EventConfig:
    window_s: float = 0.15                   # 抛物线拟合时间窗（秒），与帧率无关
    min_window_points: int = 5               # 至少多少帧才拟合
    max_window_points: int = 30
    bounce_min_curvature: float = 600.0      # 二次项阈值（像素/秒^2）
    bounce_min_excursion_px: float = 5.0     # 顶点相对两端的最小落差
    model_event_threshold: float = 0.5      # 模型事件分数阈值
    dedup_window_s: float = 0.30            # 同源同类事件合并窗口
    fuse_window_s: float = 0.30             # 跨源融合窗口
    out_margin_m: float = 0.02              # 台边容差
    net_from_model_only: bool = True        # 单目默认只用模型的擦网分数
    min_confidence: float = 0.30


def _slope(points: list[TrackPoint], index: tuple[int, int]) -> float:
    """用两点的 y 方向斜率（像素/秒），向下为正。"""
    a, b = index
    dt = points[b].timestamp_s - points[a].timestamp_s
    if dt <= 1e-6:
        return 0.0
    return (points[b].px[1] - points[a].px[1]) / dt


@dataclass
class EventExtractor:
    """单个数据源的轨迹->事件提取器。"""

    source: str
    homography: Homography | None = None
    cfg: EventConfig = field(default_factory=EventConfig)
    _recent: list[TrackPoint] = field(default_factory=list)
    _last_events: dict[str, Event] = field(default_factory=dict)
    _counter: int = 0

    def reset(self) -> None:
        self._recent.clear()
        self._last_events.clear()

    # ------------------------------------------------------------ 主入口
    def on_point(self, point: TrackPoint, obs: Observation | None = None) -> list[Event]:
        if point.usable_for_events:
            self._recent.append(point)
            if len(self._recent) > 60:
                del self._recent[:30]
        events: list[Event] = []
        if point.usable_for_events:
            bounce = self._detect_bounce()
            if bounce is not None:
                events.extend(self._split_bounce(bounce))
        if obs is not None:
            events.extend(self._model_events(obs))
        return [e for e in (self._dedup(e) for e in events) if e is not None]

    # ------------------------------------------------------------ 落台
    def _detect_bounce(self) -> Event | None:
        if not self._recent:
            return None
        last_t = self._recent[-1].timestamp_s
        window = [p for p in self._recent if p.timestamp_s >= last_t - self.cfg.window_s]
        if len(window) > self.cfg.max_window_points:
            window = window[-self.cfg.max_window_points:]
        if len(window) < self.cfg.min_window_points:
            window = self._recent[-self.cfg.min_window_points:]
        if len(window) < 3:
            return None
        t0 = window[0].timestamp_s
        ts = np.array([p.timestamp_s - t0 for p in window], dtype=np.float64)
        ys = np.array([p.px[1] for p in window], dtype=np.float64)
        xs = np.array([p.px[0] for p in window], dtype=np.float64)
        # 对整窗拟合抛物线：抖动被平均掉，顶点就是落台时刻
        a, b, c = np.polyfit(ts, ys, 2)
        if a >= -self.cfg.bounce_min_curvature:
            return None
        t_vertex = -b / (2 * a)
        if not (ts[1] <= t_vertex <= ts[-2]):
            return None
        vertex_y = a * t_vertex ** 2 + b * t_vertex + c
        # 顶点必须比窗口两端明显更低（图像 y 更大）
        excursion = vertex_y - max(ys[0], ys[-1])
        if excursion < self.cfg.bounce_min_excursion_px:
            return None
        vertex_time = float(t0 + t_vertex)
        last = self._last_events.get("BOUNCE")
        if last is not None and abs(vertex_time - last.timestamp_s) < self.cfg.dedup_window_s:
            return None
        slope_x, intercept_x = np.polyfit(ts, xs, 1)
        px = (float(slope_x * t_vertex + intercept_x), float(vertex_y))
        geom = self._project(px)
        if geom is None:
            return None
        if geom.inside:
            confidence = 0.60 + 0.35 * min(1.0, excursion / 15.0)
            etype = "BOUNCE"
        else:
            confidence = 0.50 + 0.35 * min(1.0, excursion / 15.0)
            etype = "OUT"
        return self._event(etype, vertex_time, geom.side, px, min(0.99, confidence),
                           meta={"table_xy": geom.table_xy, "inside": geom.inside,
                                 "edge_margin_m": geom.edge_margin_m,
                                 "excursion_px": round(float(excursion), 1),
                                 "curvature": round(float(a), 1),
                                 "detector": "trajectory"})

    def _split_bounce(self, event: Event) -> list[Event]:
        # OUT 是"落台点不在台面"，仍保留落点侧信息供回合判定使用
        return [event]

    def _project(self, px: tuple[float, float]) -> BounceGeometry | None:
        return BounceGeometry.evaluate(self.homography, px, margin_m=self.cfg.out_margin_m)

    # ------------------------------------------------------------ 模型事件
    def _model_events(self, obs: Observation) -> list[Event]:
        events: list[Event] = []
        scores = obs.event_scores
        px = obs.ball.center_px
        if scores.bounce is not None and scores.bounce >= self.cfg.model_event_threshold \
                and obs.ball.visible and px is not None:
            last = self._last_events.get("BOUNCE")
            if last is None or obs.timestamp_s - last.timestamp_s >= self.cfg.dedup_window_s:
                geom = self._project(px)
                if geom is not None:
                    events.append(self._event(
                        "BOUNCE" if geom.inside else "OUT",
                        obs.timestamp_s, geom.side, px,
                        min(0.99, 0.55 + 0.45 * float(scores.bounce)),
                        meta={"table_xy": geom.table_xy, "inside": geom.inside,
                              "from_model": True, "detector": "model"}))
        if scores.net_hit is not None and scores.net_hit >= self.cfg.model_event_threshold:
            last = self._last_events.get("NET_HIT")
            if last is None or obs.timestamp_s - last.timestamp_s >= self.cfg.dedup_window_s:
                side = "UNKNOWN"
                if obs.ball.visible and px is not None:
                    geom = self._project(px)
                    side = geom.side if geom else "UNKNOWN"
                events.append(self._event("NET_HIT", obs.timestamp_s, side, px,
                                          min(0.99, 0.55 + 0.45 * float(scores.net_hit)),
                                          meta={"from_model": True, "detector": "model"}))
        return events

    # ------------------------------------------------------------ 工具
    def _event(self, etype: str, t: float, side: str, px: tuple[float, float] | None,
               confidence: float, meta: dict | None = None) -> Event:
        self._counter += 1
        event = Event(event_id=f"{self.source}_{etype.lower()}_{self._counter:05d}",
                      timestamp_s=float(t), type=etype, side=side,  # type: ignore[arg-type]
                      position_px=px, confidence=float(confidence),
                      sources=[self.source], meta=meta or {})
        return event

    def _dedup(self, event: Event) -> Event | None:
        last = self._last_events.get(event.type)
        if last is not None and last.side == event.side and \
                abs(event.timestamp_s - last.timestamp_s) < self.cfg.dedup_window_s:
            # 同源、同类、同侧且在窗口内：只保留第一次，避免一帧抖动产生两个事件
            return None
        self._last_events[event.type] = event
        return event


@dataclass
class EventFuser:
    """跨模型融合：同一次事件只保留一条，冲突进入 UNKNOWN。"""

    cfg: EventConfig = field(default_factory=EventConfig)
    source_weights: dict[str, float] = field(default_factory=dict)
    _pending: list[tuple[float, Event]] = field(default_factory=list)
    _counter: int = 0

    def add(self, event: Event, now: float | None = None) -> None:
        """按**到达时间**排队：检测有延迟，不能用事件时间做融合切分。"""
        arrival = event.timestamp_s if now is None else now
        self._pending.append((float(arrival), event))
        self._pending.sort(key=lambda item: item[1].timestamp_s)

    def ready(self, now: float) -> list[Event]:
        cutoff = now - self.cfg.fuse_window_s
        ready_items = [(a, e) for a, e in self._pending if a <= cutoff]
        if not ready_items:
            return []
        max_ts = max(e.timestamp_s for _, e in ready_items)
        # 把与已就绪事件时间相近的后续事件一起取走，避免同一时刻的两个模型被拆到两次融合
        extra = [(a, e) for a, e in self._pending
                 if a > cutoff and e.timestamp_s <= max_ts + self.cfg.fuse_window_s]
        ready_items.extend(extra)
        taken = {id(e) for _, e in ready_items}
        self._pending = [(a, e) for a, e in self._pending if id(e) not in taken]
        ready = [e for _, e in ready_items]
        return self._cluster(ready)

    def flush(self) -> list[Event]:
        ready = [e for _, e in self._pending]
        self._pending = []
        return self._cluster(ready)

    def _cluster(self, events: list[Event]) -> list[Event]:
        clusters: list[list[Event]] = []
        for event in sorted(events, key=lambda e: e.timestamp_s):
            for cluster in clusters:
                head = cluster[0]
                if head.type == event.type and \
                        abs(head.timestamp_s - event.timestamp_s) <= self.cfg.fuse_window_s:
                    cluster.append(event)
                    break
            else:
                clusters.append([event])
        return [self._merge(c) for c in clusters]

    def _merge(self, cluster: list[Event]) -> Event:
        self._counter += 1
        sides = {e.side for e in cluster if e.side != "UNKNOWN"}
        conflict = len(sides) > 1
        side = "UNKNOWN" if conflict else (sides.pop() if sides else "UNKNOWN")
        weights = [self.source_weights.get(e.sources[0] if e.sources else "", 1.0)
                   for e in cluster]
        confidences = [e.confidence for e in cluster]
        weighted = float(np.average(confidences, weights=weights)) if confidences else 0.0
        agreement_bonus = 0.03 * (len(cluster) - 1)
        if conflict:
            weighted *= 0.5
        positions = [e.position_px for e in cluster if e.position_px]
        px = tuple(np.mean(np.asarray(positions, dtype=float), axis=0)) if positions else None
        sources = sorted({s for e in cluster for s in e.sources})
        meta = {k: v for e in cluster for k, v in e.meta.items()}
        meta["source_confidences"] = {s: sorted({round(e.confidence, 3) for e in cluster
                                                 if s in e.sources}) for s in sources}
        return Event(event_id=f"fused_{cluster[0].type.lower()}_{self._counter:05d}",
                     timestamp_s=float(np.mean([e.timestamp_s for e in cluster])),
                     type=cluster[0].type, side=side, position_px=px,  # type: ignore[arg-type]
                     confidence=min(0.99, weighted + agreement_bonus),
                     sources=sources, conflict=conflict, meta=meta)
