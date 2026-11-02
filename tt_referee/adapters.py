"""三个模型的适配器：把各自原始输出统一成 Observation。

适配器只做坐标还原与字段映射，不参与落台判断，更不输出"谁得分"。
实际部署时按模型仓库的真实输出改 `to_observation` 的取值即可。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .schema import BallObservation, EventScores, Observation


class YoloAdapter:
    """YOLO：检测框 -> 球心。raw 为 [{"bbox":[x1,y1,x2,y2], "score":0.9, "cls":0}, ...]。"""

    source = "yolo"

    def __init__(self, ball_class: int = 0, min_score: float = 0.25):
        self.ball_class = ball_class
        self.min_score = min_score

    def to_observation(self, frame_id: int, timestamp_s: float, raw: list[dict[str, Any]],
                       image_size: tuple[int, int] = (1920, 1080)) -> Observation:
        best = None
        for det in raw:
            if int(det.get("cls", 0)) != self.ball_class:
                continue
            score = float(det.get("score", 0.0))
            if score < self.min_score:
                continue
            if best is None or score > float(best.get("score", 0.0)):
                best = det
        if best is None:
            return Observation(frame_id, timestamp_s, self.source, image_size,
                               BallObservation.lost())
        bbox = tuple(float(v) for v in best["bbox"])
        return Observation(frame_id, timestamp_s, self.source, image_size,
                           BallObservation.from_bbox(bbox, float(best.get("score", 0.0))))


class TrackNetAdapter:
    """TrackNet：热图 -> 峰值。raw 为二维热图（或已经归一化的概率图）。"""

    source = "tracknet"

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def to_observation(self, frame_id: int, timestamp_s: float, raw: np.ndarray,
                       image_size: tuple[int, int] = (1920, 1080)) -> Observation:
        heat = np.asarray(raw, dtype=np.float32)
        if heat.ndim != 2:
            raise ValueError("TrackNet 热图必须是二维数组")
        idx = int(np.argmax(heat))
        y, x = divmod(idx, heat.shape[1])
        value = float(heat[y, x])
        if value < self.threshold:
            return Observation(frame_id, timestamp_s, self.source, image_size,
                               BallObservation.lost())
        return Observation(frame_id, timestamp_s, self.source, image_size,
                           BallObservation(visible=True, center_px=(float(x), float(y)),
                                           confidence=min(1.0, value)))


class TTNetAdapter:
    """TTNet：球位置 + 模型自带的落台/擦网分数。

    raw 形如：{"ball": {"visible": True, "center_px": [x, y], "confidence": 0.9},
               "bounce": 0.8, "net_hit": 0.2}
    """

    source = "ttnet"

    def to_observation(self, frame_id: int, timestamp_s: float, raw: dict[str, Any],
                       image_size: tuple[int, int] = (1920, 1080)) -> Observation:
        ball_raw = raw.get("ball") or {}
        if ball_raw.get("visible") and ball_raw.get("center_px") is not None:
            ball = BallObservation(visible=True,
                                   center_px=tuple(float(v) for v in ball_raw["center_px"]),
                                   confidence=ball_raw.get("confidence"))
        else:
            ball = BallObservation.lost()
        return Observation(frame_id, timestamp_s, self.source, image_size, ball,
                           EventScores(bounce=raw.get("bounce"), net_hit=raw.get("net_hit")))


class CallableAdapter:
    """把任意函数包装成适配器，方便接自定义模型。"""

    def __init__(self, source: str, func: Callable[..., Observation]):
        self.source = source
        self.func = func

    def to_observation(self, frame_id: int, timestamp_s: float, raw: Any,
                       image_size: tuple[int, int] = (1920, 1080)) -> Observation:
        return self.func(frame_id, timestamp_s, raw, image_size)
