"""模板库：注册阶段存下来的一段段 MFCC，以及每条模板的判据阈值。

阈值不是拍脑袋定的：同一条口令录 N 遍，两两算 DTW 距离，取类内最大距离再放一点余量。
这样"说得快/说得慢"的差异被自动吸收，不同口令之间也不会互相顶掉。

文件格式：npz，键 "meta"（JSON 字符串）+ "feat_i"（float32 数组），不依赖 pickle。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .dtw import dtw

KIND_WAKE = "wake"
KIND_COMMAND = "command"


@dataclass
class Template:
    phrase: str
    intent: str
    kind: str
    feats: np.ndarray
    threshold: float = 0.0          # 0 表示未标定，运行时用配置里的默认阈值

    @property
    def frames(self) -> int:
        return int(self.feats.shape[0])


@dataclass
class TemplateStore:
    templates: list[Template] = field(default_factory=list)
    sample_rate: int = 16000
    feature_dim: int = 26
    metric: str = "cosine"

    def by_kind(self, kind: str) -> list[Template]:
        return [t for t in self.templates if t.kind == kind]

    def phrases(self, kind: str) -> list[str]:
        seen: list[str] = []
        for t in self.by_kind(kind):
            if t.phrase not in seen:
                seen.append(t.phrase)
        return seen

    def calibrate(self, margin: float = 1.3, minimum: float = 0.02,
                  log=print) -> None:
        """按"同一口令不同遍数"的类内距离给每条模板定阈值。"""
        groups: dict[tuple[str, str], list[Template]] = {}
        for t in self.templates:
            groups.setdefault((t.kind, t.phrase), []).append(t)
        for (kind, phrase), items in groups.items():
            if len(items) < 2:
                continue
            subsequence = kind == KIND_WAKE
            dists: list[float] = []
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    dists.append(dtw(items[i].feats, items[j].feats, subsequence=subsequence,
                                     metric=self.metric))
            base = max(dists)
            threshold = max(base * margin, minimum)
            for t in items:
                t.threshold = float(threshold)
            log(f"  [标定] {kind}/{phrase}: 类内距离 {min(dists):.3f}~{base:.3f} "
                f"-> 阈值 {threshold:.3f}")

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "meta": json.dumps([
                {"phrase": t.phrase, "intent": t.intent, "kind": t.kind,
                 "threshold": t.threshold, "frames": t.frames}
                for t in self.templates
            ], ensure_ascii=False),
            "sample_rate": np.array(self.sample_rate),
            "feature_dim": np.array(self.feature_dim),
            "metric": self.metric,
        }
        for index, t in enumerate(self.templates):
            payload[f"feat_{index}"] = t.feats
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str) -> "TemplateStore":
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到模板库：{path}，先跑 python tools/enroll.py")
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            store = cls(sample_rate=int(data["sample_rate"]),
                        feature_dim=int(data["feature_dim"]),
                        metric=str(data["metric"]) if "metric" in data else "cosine")
            for index, item in enumerate(meta):
                store.templates.append(Template(
                    phrase=item["phrase"],
                    intent=item["intent"],
                    kind=item["kind"],
                    feats=data[f"feat_{index}"].astype(np.float32),
                    threshold=float(item.get("threshold", 0.0)),
                ))
        return store

    def describe(self) -> str:
        lines = []
        for kind in (KIND_WAKE, KIND_COMMAND):
            items = self.by_kind(kind)
            if not items:
                continue
            lines.append(f"  {kind}: {len(items)} 条模板 / {len(self.phrases(kind))} 个短语")
            for phrase in self.phrases(kind):
                group = [t for t in items if t.phrase == phrase]
                frames = ",".join(str(t.frames) for t in group)
                lines.append(f"    {phrase:<12} {len(group)} 条  帧数 {frames}  "
                             f"阈值 {group[0].threshold:.3f}")
        return "\n".join(lines)
