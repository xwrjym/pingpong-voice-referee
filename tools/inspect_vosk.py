"""Vosk 后端调试工具：看 grammar 建图日志、样例 wav 的解码结果、事件时序。

    python tools/inspect_vosk.py              # 各类样例解码结果
    python tools/inspect_vosk.py --pipeline   # 走完整状态机，打印事件 phase 时序
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vosk  # noqa: E402

from tt_voice.audio_io import WavSource              # noqa: E402
from tt_voice.config import load_config              # noqa: E402
from tt_voice.engines.vosk_engine import VoskEngine  # noqa: E402


def run(engine, path: str, mode: str) -> list:
    engine.reset()
    engine.set_mode(mode)
    hits = []
    for block in WavSource(path, engine.cfg.sample_rate, engine.cfg.block_size):
        hits += engine.feed(block)
    hits += engine.flush()
    return hits


def main() -> None:
    cfg = load_config()
    print("字典文件:", os.path.exists(os.path.join(cfg.model_path, "graph", "words.txt")))
    vosk.SetLogLevel(0)          # 打开日志，看 grammar 建图有没有报错
    engine = VoskEngine(cfg, log=print)
    engine.start()

    for label, path, mode in (
        ("唤醒词(clean)", r"testdata\wake__xiao_caipan_1__clean.wav", "wake"),
        ("唤醒词(远场)", r"testdata\wake__xiao_caipan_2__far.wav", "wake"),
        ("干扰句", r"testdata\neg__chat_next__clean.wav", "wake"),
        ("口令", r"testdata\cmd__score_left_01__clean.wav", "command"),
        ("口令(带噪)", r"testdata\cmd__undo_12__snr10.wav", "command"),
        ("连说", r"testdata\combo__left_0__clean.wav", "wake"),
    ):
        hits = run(engine, path, mode)
        print(f"  [{label:<12}] mode={mode:<8} -> "
              f"{[(h.kind, h.text, round(h.confidence, 2)) for h in hits]}")

    print("grammar 模式可用性:", "用了 grammar" if "grammar" in " ".join(engine.notes) else
          "（看上面的日志判断）")


def pipeline_probe() -> None:
    """走完整状态机（含唤醒后回灌），打印每条记录的 phase，看时序到底怎么走的。"""
    from tt_voice.app import Pipeline
    from tt_voice.feedback import NullSpeaker

    cfg = load_config()
    cfg.engine = "vosk"
    engine = VoskEngine(cfg, log=lambda *_: None)
    engine.start()
    original = engine.feed

    def traced(pcm: bytes):
        events = original(pcm)
        for event in events:
            print(f"      event {event.kind}/{event.phase} text={event.text!r} "
                  f"conf={event.confidence:.2f} 缓存块={len(engine._preroll)}")
        return events

    engine.feed = traced
    for name in ("combo__undo_0__clean.wav", "combo__right_0__clean.wav",
                 "combo__left_0__clean.wav"):
        path = os.path.join("testdata", name)
        print(f"\n=== {name} ===")
        engine.reset()
        pipe = Pipeline(cfg, engine=engine, speaker=NullSpeaker(), log=lambda *_: None)
        pipe.start()
        now = 0.0
        for block in WavSource(path, cfg.sample_rate, cfg.block_size):
            now += cfg.block_ms / 1000.0
            pipe.feed(block, now=now)
        pipe.flush(now)
        for record in pipe.records:
            print(f"    record {record['kind']}/{record['phase']} text={record['text']!r} "
                  f"intent={record['intent']} accepted={record['accepted']} "
                  f"note={record['note']} score={record['score']}")
        print(f"    状态={pipe.state} 比分={pipe.board.score_text()}")


if __name__ == "__main__":
    main()
    pipeline_probe()
