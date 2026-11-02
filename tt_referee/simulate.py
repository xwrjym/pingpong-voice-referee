"""合成轨迹：为事件层与计分状态机造可复现的测试输入。

不依赖相机与模型，用"桌面坐标 + 抛物线高度"反投影成图像坐标，
因此落台点的几何位置是精确已知的，适合做回归测试与演示。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import Homography
from .schema import BallObservation, EventScores, Observation

DEFAULT_IMAGE_QUAD = [(320.0, 330.0), (1600.0, 300.0),
                      (1580.0, 860.0), (360.0, 880.0)]


def demo_homography() -> Homography:
    return Homography.from_points(DEFAULT_IMAGE_QUAD)


@dataclass
class SimConfig:
    fps: float = 60.0
    video_fps: float = 60.0            # 统一视频帧号用的帧率（与模型输出帧率无关）
    px_per_meter: float = 450.0
    apex_height_m: float = 0.30
    jitter_px: float = 1.5
    dropout_rate: float = 0.08
    seed: int = 7
    image_size: tuple[int, int] = (1920, 1080)


def _segment_ball(t: float, bounces: list[tuple[float, tuple[float, float]]],
                  cfg: SimConfig) -> tuple[float, float, float]:
    """返回 (table_x, table_y, height_m)。"""
    times = [b[0] for b in bounces]
    if t <= times[0]:
        # 发球前：从发球方一侧更高处下降
        t0 = times[0] - 0.35
        u = max(0.0, min(1.0, (t - t0) / 0.35))
        x0, y0 = bounces[0][1]
        x0 -= 1.1
        x = x0 + (bounces[0][1][0] - x0) * u
        y = y0
        h = cfg.apex_height_m * 1.2 * (1 - u) ** 2
        return x, y, h
    for index in range(len(bounces) - 1):
        t0, p0 = bounces[index]
        t1, p1 = bounces[index + 1]
        if t0 <= t <= t1:
            u = (t - t0) / max(1e-6, t1 - t0)
            x = p0[0] + (p1[0] - p0[0]) * u
            y = p0[1] + (p1[1] - p0[1]) * u
            h = cfg.apex_height_m * 4 * u * (1 - u)
            return x, y, h
    # 最后一次落台后：向上飞走
    t_last, p_last = bounces[-1]
    dt = max(0.0, t - t_last)
    return p_last[0] + 0.25 * dt, p_last[1], min(0.6, 1.2 * dt)


def make_observations(bounces: list[tuple[float, tuple[float, float]]],
                      source: str = "yolo", start_s: float = 0.0,
                      end_s: float = 2.5, cfg: SimConfig | None = None,
                      net_hit_at: float | None = None) -> list[Observation]:
    cfg = cfg or SimConfig()
    homography = demo_homography()
    rng = np.random.default_rng(cfg.seed)
    observations: list[Observation] = []
    t = start_s
    while t <= end_s:
        x, y, h = _segment_ball(t, bounces, cfg)
        px, py = homography.table_to_image((x, y))
        py -= h * cfg.px_per_meter
        px += rng.normal(0, cfg.jitter_px)
        py += rng.normal(0, cfg.jitter_px)
        visible = bool(rng.random() > cfg.dropout_rate)
        scores = EventScores()
        if source == "ttnet":
            bounce_score = max((math.exp(-((t - bt) / 0.06) ** 2) for bt in (b[0] for b in bounces)),
                               default=0.0)
            scores.bounce = bounce_score if bounce_score > 0.2 else None
            if net_hit_at is not None:
                net_score = math.exp(-((t - net_hit_at) / 0.05) ** 2)
                scores.net_hit = net_score if net_score > 0.2 else None
        ball = (BallObservation(visible=True, center_px=(float(px), float(py)),
                                confidence=0.9) if visible else BallObservation.lost())
        observations.append(Observation(frame_id=int(round(t * cfg.video_fps)),
                                        timestamp_s=round(t, 4),
                                        source=source, image_size=cfg.image_size,
                                        ball=ball, event_scores=scores))
        t += 1.0 / cfg.fps
    return observations


SCENARIOS: dict[str, dict] = {
    # 正常来回：A 发球（A 侧、B 侧），B 没接住，球再次落在 B 侧 -> A 得分
    "double_bounce": {
        "bounces": [(0.60, (0.95, 0.70)), (1.15, (1.85, 0.72)), (1.70, (1.88, 0.70))],
        "end_s": 2.2,
    },
    # 发球直接落在对方半台 -> 发球失误，B 得分
    "serve_wrong_side": {"bounces": [(0.60, (1.90, 0.70))], "end_s": 1.2},
    # 球飞出端线 -> 落点所在半台 B 得分
    "out": {"bounces": [(0.60, (0.95, 0.70)), (1.30, (3.05, 0.70))], "end_s": 1.9},
    # 擦网后落在发球方自己半台 -> B 得分（仅 TTNet 有擦网分数）
    "net_short": {"bounces": [(1.00, (0.62, 0.70))], "end_s": 1.6, "net_hit_at": 0.55},
    # 发球擦网但仍合法落在接发球方半台 -> 重发（不改变比分与发球方）
    "net_let": {"bounces": [(0.60, (0.95, 0.70)), (1.15, (1.85, 0.72))],
                "end_s": 1.6, "net_hit_at": 0.85},
    # 只落一次台后球消失 -> 不判分，等待人工确认
    "lost": {"bounces": [(0.60, (0.95, 0.70))], "end_s": 1.0},
}


def build_scenario(name: str) -> dict[str, list[Observation]]:
    if name == "two_sources":
        cfg_a = SimConfig(seed=11)
        cfg_b = SimConfig(seed=23)
        return {
            "yolo": make_observations(SCENARIOS["double_bounce"]["bounces"], "yolo",
                                      end_s=2.2, cfg=cfg_a),
            "ttnet": make_observations(SCENARIOS["double_bounce"]["bounces"], "ttnet",
                                       end_s=2.2, cfg=cfg_b),
        }
    if name == "three_sources":
        return {
            "yolo": make_observations(SCENARIOS["double_bounce"]["bounces"], "yolo",
                                      end_s=2.2, cfg=SimConfig(seed=11)),
            "tracknet": make_observations(SCENARIOS["double_bounce"]["bounces"], "tracknet",
                                          end_s=2.2, cfg=SimConfig(seed=17)),
            "ttnet": make_observations(SCENARIOS["double_bounce"]["bounces"], "ttnet",
                                       end_s=2.2, cfg=SimConfig(seed=23)),
        }
    if name == "conflict":
        cfg_a = SimConfig(seed=31)
        cfg_b = SimConfig(seed=41)
        return {
            "yolo": make_observations([(0.80, (0.95, 0.70))], "yolo", end_s=0.95, cfg=cfg_a),
            "ttnet": make_observations([(0.80, (1.90, 0.70))], "ttnet", end_s=0.95, cfg=cfg_b),
        }
    if name not in SCENARIOS:
        raise KeyError(f"未知场景：{name}，可选 {sorted(SCENARIOS)} + "
                       f"two_sources/three_sources/conflict")
    spec = SCENARIOS[name]
    source = "ttnet" if name in ("net_short", "net_let") else "yolo"
    return {source: make_observations(spec["bounces"], source, end_s=spec["end_s"],
                                      net_hit_at=spec.get("net_hit_at"))}
