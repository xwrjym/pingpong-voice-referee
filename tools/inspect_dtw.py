"""逐段跟踪 DTW 引擎的 VAD/分段/匹配距离，调阈值时用。

用法：

    python tools/inspect_dtw.py testdata/wake__xiao_caipan_1__far.wav wake
    python tools/inspect_dtw.py testdata/cmd__undo_14__snr10.wav command models/dtw_cn/templates_yaoyao.npz
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tt_voice.audio_io import WavSource                  # noqa: E402
from tt_voice.config import load_config                  # noqa: E402
from tt_voice.engines.dtw_engine import DtwEngine        # noqa: E402
from tt_voice.features import trim_silence               # noqa: E402


def run(path: str, mode: str) -> None:
    cfg = load_config()
    if len(sys.argv) > 3:
        cfg.dtw_model = sys.argv[3]
    engine = DtwEngine(cfg, log=print)
    engine.start()
    engine.reset()
    engine.set_mode(mode)
    print(f"\n=== {os.path.basename(path)}  mode={mode} ===")
    original = engine._match_segment

    def traced(segment: bytes):
        feats_all = engine.extractor.extract(segment)
        energy = engine.extractor.frame_energy_db(segment)
        trimmed = trim_silence(feats_all, energy)
        print(f"  [段] {len(segment) / 2 / 16:.0f}ms  特征 {feats_all.shape} "
              f"-> 去静音 {trimmed.shape}  能量跨度 "
              f"{energy.min() if energy.size else 0:.1f}~{energy.max() if energy.size else 0:.1f}dB")
        result = original(segment)
        print("  [段] 结果:", result, engine.last_scores[:3])
        return result

    engine._match_segment = traced
    source = WavSource(path, cfg.sample_rate, cfg.block_size)
    for index, block in enumerate(source):
        speech = engine.vad.is_speech(block)
        events = engine.feed(block)
        if index % 5 == 0 or events or speech:
            print(f"  block {index:>3} speech={int(speech)} buf={len(engine._buf) / 2 / 16:.0f}ms "
                  f"silence={engine._silence_ms:.0f} events={events}")
    tail = engine.flush()
    print("  flush:", tail)
    if engine.last_scores:
        print("  最近5名:", engine.last_scores)

    # 直接看看这一段单独做匹配的距离量级
    whole = WavSource(path, cfg.sample_rate, cfg.block_size).frames()
    trimmed = engine.extractor.extract(whole)
    print(f"  整段特征（已含剪静音+CMN）: {trimmed.shape}")
    for kind in ("wake", "command"):
        templates = engine.store.by_kind(kind)
        scored = []
        from tt_voice.dtw import dtw
        for t in templates:
            scored.append((dtw(trimmed, t.feats, subsequence=True), t.phrase, t.threshold))
        scored.sort()
        print(f"  [{kind}] 最好的 5 条:")
        for dist, phrase, thr in scored[:5]:
            print(f"     {dist:7.3f} / {thr:5.3f}  {phrase}")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "wake")
