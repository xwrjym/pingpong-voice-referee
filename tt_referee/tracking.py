"""单源轨迹：平滑、补点、可见性标记。

球在画面里会抖动、被遮挡、短暂漏检。这里用一个轻量的 alpha-beta 滤波器：

* 检测到球：用观测修正位置与速度；
* 没检测到：输出**预测位置**并标记 predicted，绝不伪装成真实检测；
* 断档超过 max_gap_s：放弃这条轨迹，等下一段重新起跟。

事件层只用真实检测点判断落台方向，预测点只用于维持时间连续性。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Observation


@dataclass
class TrackPoint:
    timestamp_s: float
    px: tuple[float, float]
    visible: bool
    predicted: bool
    confidence: float | None = None

    @property
    def usable_for_events(self) -> bool:
        """预测点不能作为事件证据（只用于连续性）。"""
        return self.visible and not self.predicted


@dataclass
class Tracker:
    """单个模型的轨迹跟踪器。"""

    max_gap_s: float = 0.35
    alpha: float = 0.6
    beta: float = 0.2
    max_speed_px_s: float = 6000.0
    _points: list[TrackPoint] = field(default_factory=list)
    _last_t: float | None = None
    _last_px: tuple[float, float] | None = None
    _velocity: tuple[float, float] = (0.0, 0.0)

    def reset(self) -> None:
        self._points.clear()
        self._last_t = None
        self._last_px = None
        self._velocity = (0.0, 0.0)

    def update(self, obs: Observation) -> TrackPoint | None:
        t = obs.timestamp_s
        if self._last_t is not None and t < self._last_t:
            raise ValueError("观测时间戳必须单调不减")

        if not obs.ball.visible:
            if self._last_t is None or self._last_px is None:
                return None
            gap = t - self._last_t
            if gap > self.max_gap_s:
                self.reset()
                return None
            px = (self._last_px[0] + self._velocity[0] * gap,
                  self._last_px[1] + self._velocity[1] * gap)
            point = TrackPoint(t, px, visible=False, predicted=True,
                               confidence=None)
            self._points.append(point)
            return point

        px = obs.ball.center_px
        assert px is not None
        if self._last_px is None or self._last_t is None:
            self._last_px = px
            self._last_t = t
            self._velocity = (0.0, 0.0)
            point = TrackPoint(t, px, visible=True, predicted=False,
                               confidence=obs.ball.confidence)
        else:
            dt = t - self._last_t
            if dt <= 1e-6:
                point = TrackPoint(t, px, visible=True, predicted=False,
                                   confidence=obs.ball.confidence)
                self._points.append(point)
                return point
            pred = (self._last_px[0] + self._velocity[0] * dt,
                    self._last_px[1] + self._velocity[1] * dt)
            residual = (px[0] - pred[0], px[1] - pred[1])
            new_px = (pred[0] + self.alpha * residual[0],
                      pred[1] + self.alpha * residual[1])
            new_vel = (self._velocity[0] + self.beta * residual[0] / dt,
                       self._velocity[1] + self.beta * residual[1] / dt)
            speed = (new_vel[0] ** 2 + new_vel[1] ** 2) ** 0.5
            if speed > self.max_speed_px_s:
                # 明显是误关联（跳点）：按新观测重起
                self._velocity = (0.0, 0.0)
                new_px = px
            else:
                self._velocity = new_vel
            self._last_px = new_px
            px = new_px
            self._last_t = t
            point = TrackPoint(t, px, visible=True, predicted=False,
                               confidence=obs.ball.confidence)
        self._points.append(point)
        if len(self._points) > 240:
            del self._points[:120]
        return point

    # ------------------------------------------------------------ 查询
    @property
    def points(self) -> list[TrackPoint]:
        return list(self._points)

    def recent(self, since_s: float) -> list[TrackPoint]:
        return [p for p in self._points if p.timestamp_s >= since_s]

    def visible_recent(self, count: int) -> list[TrackPoint]:
        found = [p for p in reversed(self._points) if p.usable_for_events]
        return list(reversed(found[:count]))

    @property
    def last_visible(self) -> TrackPoint | None:
        for point in reversed(self._points):
            if point.usable_for_events:
                return point
        return None
