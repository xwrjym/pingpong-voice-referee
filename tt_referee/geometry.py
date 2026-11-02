"""球台几何：四点标定、透视变换、左右半台与出界判断。

单目摄像头的限制（方案第 5 节）：

* 透视矩阵只对**桌面平面**成立，空中球的位置不能直接映射成落点；
* 正确顺序是：先在时间序列上判定落台时刻，再把**落台时刻的像素位置**投到桌面，
  否则会把高处的球误判成出界。

本模块只提供"像素 <-> 桌面"的映射，不负责判断什么时候落台（那是 events 层的事）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ITTF 标准球台（米）
TABLE_LENGTH_M = 2.74
TABLE_WIDTH_M = 1.525
HALF_LENGTH_M = TABLE_LENGTH_M / 2.0

# 桌面坐标约定：x 沿球台长边（球网在 x=1.37），y 沿球台短边，单位米。
# 侧别按方案 §7：A = 球台左半台（x < 1.37），B = 球台右半台（x > 1.37）。
# 标定四角顺序：左远 -> 右远 -> 右近 -> 左近。
TABLE_CORNERS_M = np.array([
    [0.0, 0.0],
    [TABLE_LENGTH_M, 0.0],
    [TABLE_LENGTH_M, TABLE_WIDTH_M],
    [0.0, TABLE_WIDTH_M],
], dtype=np.float64)


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley 归一化，提升 DLT 数值稳定性。"""
    center = points.mean(axis=0)
    shifted = points - center
    scale = np.sqrt((shifted ** 2).sum(axis=1).mean())
    scale = scale if scale > 1e-9 else 1.0
    matrix = np.array([
        [np.sqrt(2) / scale, 0.0, -np.sqrt(2) / scale * center[0]],
        [0.0, np.sqrt(2) / scale, -np.sqrt(2) / scale * center[1]],
        [0.0, 0.0, 1.0],
    ])
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    normalized = (matrix @ homogeneous.T).T
    return normalized[:, :2], matrix


class Homography:
    """3x3 单应矩阵；像素坐标 <-> 桌面坐标（米）。"""

    def __init__(self, matrix: np.ndarray):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("单应矩阵必须是 3x3")
        if abs(matrix[2, 2]) < 1e-12:
            raise ValueError("单应矩阵右下角不能为 0")
        self.matrix = matrix / matrix[2, 2]
        self.inverse = np.linalg.inv(self.matrix)

    # ------------------------------------------------------------ 构建
    @classmethod
    def from_points(cls, image_points: list[tuple[float, float]],
                    table_points: list[tuple[float, float]] | None = None) -> "Homography":
        """用四点标定求单应矩阵。

        image_points 顺序：桌角 0,1,2,3 对应桌面坐标 (0,0),(L,0),(L,W),(0,W)。
        现场标定板按这个顺序点选即可；顺序反了会导致左右半台整体颠倒。
        """
        image = np.asarray(image_points, dtype=np.float64)
        table = np.asarray(table_points if table_points is not None else TABLE_CORNERS_M,
                           dtype=np.float64)
        if image.shape != (4, 2) or table.shape != (4, 2):
            raise ValueError("需要恰好 4 个像素点与 4 个桌面点")

        img_n, img_t = _normalize_points(image)
        tbl_n, tbl_t = _normalize_points(table)
        rows = []
        for (x, y), (u, v) in zip(img_n, tbl_n):
            rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
            rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
        h_norm = vt[-1].reshape(3, 3)
        matrix = np.linalg.inv(tbl_t) @ h_norm @ img_t
        return cls(matrix)

    # ------------------------------------------------------------ 映射
    def image_to_table(self, point: tuple[float, float]) -> tuple[float, float]:
        vec = self.matrix @ np.array([point[0], point[1], 1.0])
        if abs(vec[2]) < 1e-12:
            raise ValueError("点在无穷远处，无法映射")
        return float(vec[0] / vec[2]), float(vec[1] / vec[2])

    def table_to_image(self, point: tuple[float, float]) -> tuple[float, float]:
        vec = self.inverse @ np.array([point[0], point[1], 1.0])
        if abs(vec[2]) < 1e-12:
            raise ValueError("点在无穷远处，无法映射")
        return float(vec[0] / vec[2]), float(vec[1] / vec[2])

    def image_to_table_many(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        mapped = (self.matrix @ homogeneous.T).T
        return mapped[:, :2] / mapped[:, 2:3]


def side_of(table_xy: tuple[float, float]) -> str:
    """按球网（x=1.37）判断物理半台：A=左半台，B=右半台（方案 §7）。"""
    x = table_xy[0]
    return "A" if x < HALF_LENGTH_M else "B"


def inside_table(table_xy: tuple[float, float], margin_m: float = 0.0) -> bool:
    """是否落在球台范围内（margin 允许擦边容差）。"""
    x, y = table_xy
    return (-margin_m <= x <= TABLE_LENGTH_M + margin_m
            and -margin_m <= y <= TABLE_WIDTH_M + margin_m)


def distance_to_net_m(table_xy: tuple[float, float]) -> float:
    return abs(table_xy[0] - HALF_LENGTH_M)


@dataclass
class BounceGeometry:
    """一次落台的几何判定结果。"""

    table_xy: tuple[float, float]
    side: str
    inside: bool
    distance_to_net_m: float
    edge_margin_m: float

    @classmethod
    def evaluate(cls, homography: Homography | None, image_xy: tuple[float, float],
                 margin_m: float = 0.02) -> "BounceGeometry | None":
        if homography is None:
            return None
        table_xy = homography.image_to_table(image_xy)
        inside = inside_table(table_xy, margin_m)
        # 到最近台边的距离：出界时用来判断"擦边/明显出界"
        edge = min(table_xy[0], table_xy[1],
                   TABLE_LENGTH_M - table_xy[0], TABLE_WIDTH_M - table_xy[1])
        return cls(table_xy=table_xy, side=side_of(table_xy), inside=inside,
                   distance_to_net_m=distance_to_net_m(table_xy), edge_margin_m=edge)
