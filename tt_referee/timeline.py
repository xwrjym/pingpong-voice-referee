"""统一视频帧号。

方案要求 frame_id 表示**被预测的视频帧**，不是模型输入帧号。
三个模型（YOLO / TrackNet / TTNet）可能帧率、延迟都不同，但同一时刻的观测
必须落到同一个 frame_id；时间轴从 0 开始。

    clock = FrameClock(fps=60.0, start_s=0.0)
    clock.frame_id(1.25)   # -> 75

对已经导出的观测，用 `unify_frame_ids` 一次性重编号：

    obs = load_observations("obs.jsonl", video_fps=60.0)   # 推荐
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import Observation


@dataclass(frozen=True)
class FrameClock:
    fps: float = 60.0
    start_s: float = 0.0

    def frame_id(self, timestamp_s: float) -> int:
        if self.fps <= 0:
            raise ValueError("fps 必须大于 0")
        return int(round((timestamp_s - self.start_s) * self.fps))


def unify_frame_ids(observations: list[Observation], video_fps: float | None = None,
                    start_s: float | None = None) -> list[Observation]:
    """把所有来源的 frame_id 统一到同一条视频时间轴，且从 0 开始。

    * 给了 video_fps：frame_id = round((t - t0) * fps)，与模型输出帧率无关；
    * 没给：按时间排序后用**不同时刻**重新编号 0,1,2,...（保持同一时刻同一编号）。
    """
    if not observations:
        return []
    ordered = sorted(observations, key=lambda o: (o.timestamp_s, o.source, o.frame_id))
    t0 = ordered[0].timestamp_s if start_s is None else start_s
    if video_fps is not None:
        clock = FrameClock(video_fps, t0)
        for obs in ordered:
            obs.frame_id = clock.frame_id(obs.timestamp_s)
        return ordered
    stamps = sorted({round(o.timestamp_s, 6) for o in ordered})
    mapping = {stamp: index for index, stamp in enumerate(stamps)}
    for obs in ordered:
        obs.frame_id = mapping[round(obs.timestamp_s, 6)]
    return ordered
