"""语义层：候选文本 -> 意图 + 是否接受。

两个关键判据，缺一不可：

* `score >= threshold`：像不像某个口令
* `score - runner_up >= margin`：有没有「第二个也很像」的歧义

只有两者都满足才执行动作。歧义时返回 reason="ambiguous"，由上层提示重说一遍——
这比猜一个意图安全得多（比如「左边加一分」听糊了可能同时像左/右）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import AppConfig
from .matcher import (Candidate, Reference, make_reference, pinyin_distance,
                      rank, similarity, subsequence_similarity)
from .textnorm import normalize, normalize_light, syllables

# 太短的口令（≤2 个汉字）容易被无关语句蹭到，单独收紧阈值
SHORT_PHRASE_LEN = 2
SHORT_PHRASE_THRESHOLD = 0.95


@dataclass
class Match:
    text: str
    match_text: str = ""
    intent: str | None = None
    phrase: str | None = None
    score: float = 0.0
    runner_up_score: float = 0.0
    runner_up_intent: str | None = None
    accepted: bool = False
    reason: str = ""
    detail: list[tuple[str, str, float]] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - 调试用
        flag = "OK " if self.accepted else "REJ"
        return (f"{flag} text={self.text!r} intent={self.intent} phrase={self.phrase!r} "
                f"score={self.score:.3f} runner_up={self.runner_up_score:.3f} reason={self.reason}")


class Parser:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.command_refs: list[Reference] = [
            make_reference(intent, phrase) for intent, phrase in cfg.all_command_phrases()
        ]
        self.wake_refs: list[Reference] = [
            make_reference("wake", word) for word in cfg.wake_words
        ]

    # ------------------------------------------------------------ 唤醒词前缀
    def strip_wake_prefix(self, text: str) -> str:
        """去掉句子开头的唤醒词。

        用户经常连着说「小裁判左方加一分」，或唤醒后回灌缓存音频导致识别文本里
        带着唤醒词，这里按拼音前缀把唤醒词剪掉，避免它把相似度拉低。
        """
        norm = normalize_light(text)
        chars = list(norm)
        syls = syllables(text)
        if not chars or len(syls) != len(chars):
            return text
        best: tuple[float, int] | None = None
        for ref in self.wake_refs:
            span = len(ref.syls)
            for k in range(max(1, span - 1), span + 2):
                if k >= len(chars):
                    continue
                dist = pinyin_distance(tuple(syls[:k]), ref.syls)
                if dist > 0.6:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, k)
        if best is None:
            return text
        return "".join(chars[best[1]:])

    # ------------------------------------------------------------ 唤醒词
    def matches_wake(self, text: str) -> Match:
        match = self._decide(
            text,
            self.wake_refs,
            threshold=self.cfg.wake_threshold,
            margin=self.cfg.wake_margin,
            asr_conf=None,
        )
        if match.accepted:
            return match
        if self.cfg.wake_allow_missing_onset:
            partial = self._wake_missing_onset_match(text)
            if partial is not None:
                return partial
        # 唤醒识别器要等静音才出结果，用户"小裁判撤销"一口气说完时，
        # 后面的词会被粘到唤醒词上（实测得到"小裁判小"）。这里允许唤醒词
        # 只占句首前缀，最多容忍多出 2 个音节。
        return self._wake_prefix_match(text) or match

    def _wake_missing_onset_match(self, text: str) -> Match | None:
        """唤醒词第一个字被采集链路吞掉时的兜底（"小裁判"→"裁判"）。

        默认关闭：这等于放宽唤醒词，球友之间说"裁判"两个字就会误唤醒。
        只有在实测发现唤醒率明显受起音影响时才打开。
        """
        norm = normalize_light(text)
        chars = list(norm)
        best: Match | None = None
        for ref in self.wake_refs:
            if len(chars) != len(ref.norm) - 1:
                continue
            candidate = "".join(chars)
            match = self._decide(candidate, [Reference(
                key=ref.key, text=ref.text[1:], norm=ref.norm[1:], syls=ref.syls[1:])],
                threshold=self.cfg.wake_threshold, margin=0.0, asr_conf=None)
            if match.accepted and (best is None or match.score > best.score):
                match.text = text
                match.phrase = ref.text
                best = match
        return best

    def _wake_prefix_match(self, text: str) -> Match | None:
        norm = normalize_light(text)
        chars = list(norm)
        if not chars:
            return None
        best: Match | None = None
        for ref in self.wake_refs:
            span = len(ref.norm)
            for k in range(span, min(len(chars), span + 2) + 1):
                candidate = "".join(chars[:k])
                prefix = self._decide(candidate, [ref],
                                      threshold=self.cfg.wake_threshold, margin=0.0,
                                      asr_conf=None)
                if prefix.accepted and (best is None or prefix.score > best.score):
                    prefix.text = text
                    best = prefix
        return best

    # ------------------------------------------------------------ 口令
    def parse_command(self, text: str, asr_conf: float | None = None,
                      tolerant: bool = False) -> Match:
        """tolerant=True：允许口令出现在句子任意位置（唤醒后回灌阶段用）。"""
        match_text = self.strip_wake_prefix(text)
        if not normalize(match_text):
            match_text = text
        match = self._decide(
            match_text,
            self.command_refs,
            threshold=self.cfg.match_threshold,
            margin=self.cfg.match_margin,
            asr_conf=asr_conf,
            tolerant=tolerant,
        )
        match.text = text
        match.match_text = match_text
        if match.accepted and match.phrase and len(normalize(match.phrase)) <= SHORT_PHRASE_LEN:
            # 短口令必须「几乎精确」，否则一律当歧义处理
            if match.score < SHORT_PHRASE_THRESHOLD:
                match.accepted = False
                match.reason = "short_phrase_low_score"
        return match

    # ------------------------------------------------------------ 内部
    def _decide(self, text: str, refs: list[Reference], threshold: float,
                margin: float, asr_conf: float | None, tolerant: bool = False) -> Match:
        utt_norm = normalize(text)
        if not utt_norm or not syllables(text):
            return Match(text=text, accepted=False, reason="empty")
        if asr_conf is not None and self.cfg.asr_conf_min > 0 and asr_conf < self.cfg.asr_conf_min:
            return Match(text=text, accepted=False, reason="low_asr_conf", score=asr_conf)

        ranked = rank(text, refs)
        if tolerant:
            utt_syls = syllables(text)
            ranked = [
                Candidate(c.reference, max(c.score, 0.95 * subsequence_similarity(
                    utt_syls, c.reference.syls)))
                for c in ranked
            ]
            ranked.sort(key=lambda c: (-c.score, len(c.reference.text)))
        if not ranked:
            return Match(text=text, accepted=False, reason="empty")

        best = ranked[0]
        # 只有「不同意图」的候选才构成歧义：同一意图的不同说法（左方加一分/左加一分）
        # 不该触发重说，否则正确识别反而被拒。
        runner = next((c for c in ranked[1:] if c.reference.key != best.reference.key), None)
        match = Match(
            text=text,
            intent=best.reference.key,
            phrase=best.reference.text,
            score=best.score,
            runner_up_score=runner.score if runner else 0.0,
            runner_up_intent=runner.reference.key if runner else None,
            detail=[(c.reference.key, c.reference.text, round(c.score, 3)) for c in ranked[:3]],
        )
        gap = best.score - (runner.score if runner else 0.0)
        if best.score < threshold:
            match.reason = "low_similarity"
        elif runner is not None and gap < margin:
            match.reason = "ambiguous"
        else:
            match.accepted = True
            match.reason = "ok"
        return match
