"""球台四点标定：保存/读取像素坐标，生成单应矩阵。

现场流程：固定相机 → 在画面上点选球台四角（顺序：左远、右远、右近、左近）→
保存成 JSON；之后每帧都复用同一个单应矩阵，不需要重复标定。

侧别按方案 §7：A = 球台左半台，B = 球台右半台（物理半台，不随换边改变）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .geometry import TABLE_CORNERS_M, Homography


@dataclass
class Calibration:
    image_points: list[tuple[float, float]]
    image_size: tuple[int, int] = (1920, 1080)
    table_points: list[tuple[float, float]] = field(
        default_factory=lambda: [tuple(p) for p in TABLE_CORNERS_M])
    name: str = ""

    def __post_init__(self) -> None:
        if len(self.image_points) != 4:
            raise ValueError("需要恰好 4 个球台角点")
        self.image_points = [(float(x), float(y)) for x, y in self.image_points]

    # ------------------------------------------------------------ 校验
    def validate(self) -> list[str]:
        warnings: list[str] = []
        width, height = self.image_size
        for x, y in self.image_points:
            if not (0 <= x <= width and 0 <= y <= height):
                warnings.append(f"角点 ({x:.0f}, {y:.0f}) 超出画面 {width}x{height}")
        area = self._signed_area()
        if abs(area) < 1000:
            warnings.append("四点几乎共线，无法稳定求单应矩阵")
        crosses = self._cross_signs()
        if len(set(crosses)) > 1:
                warnings.append("四点不是凸四边形，或顺序不对（应为 左远→右远→右近→左近）")
        if area < 0:
            warnings.append("角点顺序是逆时针（部分实现要求顺时针，请确认左右半台没有颠倒）")
        return warnings

    def _signed_area(self) -> float:
        pts = self.image_points
        total = 0.0
        for index in range(4):
            x1, y1 = pts[index]
            x2, y2 = pts[(index + 1) % 4]
            total += x1 * y2 - x2 * y1
        return total / 2.0

    def _cross_signs(self) -> list[int]:
        pts = self.image_points
        signs = []
        for index in range(4):
            x1, y1 = pts[index]
            x2, y2 = pts[(index + 1) % 4]
            x3, y3 = pts[(index + 2) % 4]
            cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
            signs.append(1 if cross > 0 else -1)
        return signs

    def homography(self) -> Homography:
        return Homography.from_points(self.image_points, self.table_points)

    # ------------------------------------------------------------ 存取
    def to_dict(self) -> dict:
        return {"name": self.name, "image_points": self.image_points,
                "table_points": self.table_points, "image_size": self.image_size}

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "Calibration":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(image_points=[tuple(p) for p in data["image_points"]],
                   image_size=tuple(data.get("image_size", (1920, 1080))),
                   table_points=[tuple(p) for p in data.get("table_points", TABLE_CORNERS_M)],
                   name=data.get("name", ""))
