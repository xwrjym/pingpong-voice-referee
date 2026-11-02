"""DTW 距离：把两段 MFCC 序列按最优路径对齐后的平均每帧失真。

两种模式：

* 端点对齐（subsequence=False）：两段必须首尾都对齐，用于"整句模板 vs 整句录音"
* 子序列（subsequence=True）：允许模板出现在目标序列的任意位置，用于唤醒词
  和"连着说"的场景（"小裁判左方加一分" 里包含 "小裁判"）

实现用反对角线扫描做向量化，300 帧量级的序列只要几百次 numpy 调用，
比逐格 Python 循环快一两个数量级。
"""

from __future__ import annotations

import numpy as np

NORM_MODE_TEMPLATE = "template"
NORM_MODE_PATH = "path"


def pairwise_cost(a: np.ndarray, b: np.ndarray, metric: str = "euclid") -> np.ndarray:
    """逐帧失真矩阵 (len(a), len(b))。默认用每维平均平方距离。"""
    if metric == "cosine":
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
        b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-6)
        return (1.0 - a_n @ b_n.T).astype(np.float32)
    diff = a[:, None, :] - b[None, :, :]
    return np.mean(diff * diff, axis=2).astype(np.float32)


def dtw(a: np.ndarray, b: np.ndarray, band_ratio: float = 0.3,
        subsequence: bool = False, norm_mode: str = NORM_MODE_TEMPLATE,
        metric: str = "euclid") -> float:
    """返回归一化距离（越小越像）。b 视为模板。"""
    n, m = int(a.shape[0]), int(b.shape[0])
    if n == 0 or m == 0:
        return float("inf")
    cost = pairwise_cost(a, b, metric=metric)
    band = max(abs(n - m), int(band_ratio * max(n, m)))
    inf = np.float32(1e18)
    dist = np.full((n + 1, m + 1), inf, dtype=np.float32)
    steps = np.zeros((n + 1, m + 1), dtype=np.float32)

    if subsequence:
        # 子序列匹配：模板必须**整条**对齐上，只是允许它出现在目标的任意位置。
        # 所以自由的是"目标的前缀"（dist[i, 0] = 0），取最后**一列**的最小值；
        # 反过来（自由的是模板）会允许只匹配模板的一小段，实测会把完全无关的
        # 音频匹配得极像。
        dist[1:, 0] = 0.0
        steps[1:, 0] = 0.0
    else:
        dist[0, 0] = 0.0

    for k in range(2, n + m + 1):
        i = np.arange(max(1, k - m), min(n, k - 1) + 1)
        if i.size == 0:
            continue
        j = k - i
        keep = np.abs(i - j) <= band
        i, j = i[keep], j[keep]
        if i.size == 0:
            continue
        cand = np.stack([dist[i - 1, j], dist[i, j - 1], dist[i - 1, j - 1]])
        pick = np.argmin(cand, axis=0)
        cols = np.arange(i.size)
        dist[i, j] = cost[i - 1, j - 1] + cand[pick, cols]
        prev_steps = np.stack([steps[i - 1, j], steps[i, j - 1], steps[i - 1, j - 1]])[pick, cols]
        steps[i, j] = prev_steps + 1.0

    if subsequence:
        column = dist[1:, m].astype(np.float64)
        valid = np.isfinite(column) & (column < float(inf) / 2)
        if not valid.any():
            return float("inf")
        idx = int(np.argmin(np.where(valid, column, np.inf)))
        total = float(column[idx])
        length = float(steps[idx + 1, m]) or 1.0
    else:
        total = float(dist[n, m])
        length = float(steps[n, m]) or 1.0
        if not np.isfinite(total) or total >= float(inf) / 2:
            return float("inf")

    if norm_mode == NORM_MODE_PATH:
        return total / max(length, 1.0)
    return total / float(m)
