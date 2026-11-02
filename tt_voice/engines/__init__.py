"""声学层引擎工厂。

目前只有 Vosk（Kaldi 小模型 + 受限词表）这一个后端；
接口刻意保持最小，方便后续换成 sherpa-onnx KWS（更省 CPU，适合常驻唤醒）。
"""

from __future__ import annotations

from .base import AsrEngine, PhraseHit
from .dtw_engine import DtwEngine
from .vosk_engine import VoskEngine

ENGINES = {"vosk": VoskEngine, "dtw": DtwEngine}


def build_engine(cfg, name: str | None = None, log=print) -> AsrEngine:
    name = name or getattr(cfg, "engine", "dtw")
    try:
        cls = ENGINES[name]
    except KeyError:
        raise ValueError(f"未知引擎 {name!r}，可选：{sorted(ENGINES)}") from None
    return cls(cfg, log=log)


__all__ = ["AsrEngine", "PhraseHit", "VoskEngine", "DtwEngine", "build_engine", "ENGINES"]
