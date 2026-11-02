"""前端对比实验：为"选哪个特征 / 哪个距离度量"提供数据，而不是拍脑袋。

用法：

    python tools/compare_frontends.py testdata

对同一批测试音频比较几种前端的可分性：
  类内距离 = 同一句话 clean vs 带噪
  类间距离 = 不同短语之间（越小越容易混）
  比值 = 类间 / 类内，越大越好
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tt_voice.audio_io import WavSource                # noqa: E402
from tt_voice.dtw import dtw                           # noqa: E402
from tt_voice.features import FeatureExtractor         # noqa: E402

FRONTENDS = {
    "CMN": dict(cmn=True, cmvn=False, noise_suppress=False),
    "SS+CMN": dict(cmn=True, cmvn=False, noise_suppress=True),
}
METRICS = {"euclid": "euclid", "cos": "cosine"}


def main() -> None:
    directory = sys.argv[1] if len(sys.argv) > 1 else "testdata"
    with open(os.path.join(directory, "labels.json"), "r", encoding="utf-8") as fh:
        rows = json.load(fh)

    # 每组取 8 个短语：clean + snr10
    groups: dict[str, dict[str, str]] = {}
    intent_of: dict[str, str] = {}
    for row in rows:
        if row["kind"] not in ("cmd", "combo"):
            continue
        phrase = row["text"]
        intent_of[phrase] = row.get("intent", "")
        if phrase not in groups and len(groups) >= 12:
            continue
        groups.setdefault(phrase, {})[row["note"]] = os.path.join(directory, row["file"])

    wake_files = {row["note"]: os.path.join(directory, row["file"])
                  for row in rows if row["kind"] == "wake" and row["text"] == "小裁判"}

    phrases = list(groups)
    print(f"参与比较的短语 {len(phrases)} 个：{phrases}\n")

    for fe_name, fe_opts in FRONTENDS.items():
        extractor = FeatureExtractor(16000, **fe_opts)
        cmn = {p: extractor.extract(WavSource(groups[p]["clean"], 16000, 1600).frames())
               for p in phrases}
        noisy = {p: extractor.extract(WavSource(groups[p]["snr10"], 16000, 1600).frames())
                 for p in phrases}
        for metric in METRICS.values():
            intra = [dtw(noisy[p], cmn[p], subsequence=True, metric=metric) for p in phrases]
            inter_all = []
            inter_intent = []
            for i, a in enumerate(phrases):
                for b in phrases[i + 1:]:
                    dist = dtw(cmn[a], cmn[b], subsequence=True, metric=metric)
                    inter_all.append(dist)
                    if intent_of.get(a, a) != intent_of.get(b, b):
                        inter_intent.append(dist)
            print(f"{fe_name:<14} metric={metric:<7} "
                  f"类内(带噪) 均值 {np.mean(intra):5.3f} 最大 {np.max(intra):5.3f} | "
                  f"跨意图最小 {np.min(inter_intent):5.3f} | "
                  f"类间 最小 {np.min(inter_all):5.3f} 均值 {np.mean(inter_all):5.3f} | "
                  f"跨意图比值 {np.min(inter_intent) / np.max(intra):5.2f}")

        if wake_files:
            ref = extractor.extract(WavSource(wake_files.get("clean", ""), 16000, 1600).frames())
            line = []
            for note, path in wake_files.items():
                f = extractor.extract(WavSource(path, 16000, 1600).frames())
                line.append(f"{note}={dtw(f, ref, subsequence=True, metric='cosine'):.3f}")
            print(f"{fe_name:<14} 唤醒词各信道版本距离: " + "  ".join(line))
        print()


if __name__ == "__main__":
    main()
