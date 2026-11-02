"""声学层接口：只负责「音频帧 -> 候选文本」，不做任何语义判断。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PhraseHit:
    """一次解码结果。kind 由引擎产出时的工作模式决定。"""

    kind: str                                  # "wake" | "command"
    text: str
    confidence: float = 0.0
    words: list[tuple[str, float]] = field(default_factory=list)
    phase: str = "live"                        # "live" | "preroll"（唤醒后回灌的缓存音频）

    def __str__(self) -> str:  # pragma: no cover - 调试用
        return f"<{self.kind}/{self.phase} {self.text!r} conf={self.confidence:.2f}>"


class AsrEngine:
    name = "base"

    def start(self) -> None:
        raise NotImplementedError

    def set_mode(self, mode: str) -> None:
        """mode: "wake"（只认唤醒词）/ "command"（只认口令）。"""
        raise NotImplementedError

    def feed(self, pcm: bytes) -> list[PhraseHit]:
        raise NotImplementedError

    def flush(self) -> list[PhraseHit]:
        """输入结束（wav 播完）时把最后一段强制定稿。"""
        return []

    def take_preroll(self) -> list[bytes]:
        """取走并清空最近缓存的音频帧（唤醒瞬间回灌给口令识别器）。"""
        return []

    def mark_handoff(self) -> None:
        """标记「回灌音频到此为止」。

        回灌之后才定稿的结果，如果它的音频起点落在回灌区间内，说明那是
        回灌进去的唤醒词残音（不是用户真的说了口令），上层会把它当白噪声忽略。
        DTW 后端不需要，空实现即可。
        """

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass
