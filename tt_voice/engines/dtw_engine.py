"""DTW 模板匹配后端：零模型依赖的离线唤醒 + 固定口令识别。

对第 5 点的意义：

* 不需要下载神经网络模型（外网被墙也能跑），整条链路纯 numpy；
* CPU 占用低：空闲时只在每段语音上做几条模板的 DTW，RK3588 上余量很大；
* 代价是说话人相关：必须先用本人录音注册模板（tools/enroll.py），
  换人或换麦克风要重新注册（或补录几遍做多模板）。

分段策略：能量 VAD 切句（静音 400ms 收尾 / 单段最长 3s）。
唤醒用子序列 DTW（唤醒词可以嵌在"小裁判左方加一分"这类长句里），
命中唤醒词后顺手在同一段里再匹配一次口令，解决"一口气说完"的情况。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AppConfig
from ..dtw import dtw
from ..features import FeatureExtractor
from ..templates import KIND_COMMAND, KIND_WAKE, Template, TemplateStore
from ..vad import EnergyVad
from .base import AsrEngine, PhraseHit


@dataclass
class Decision:
    """一次匹配的判定结果（含为什么拒识，方便调参和复盘）。"""

    phrase: str = ""
    intent: str = ""
    dist: float = float("inf")
    threshold: float = 0.0
    runner_phrase: str = ""
    runner_dist: float = float("inf")
    accepted: bool = False
    reason: str = ""


class DtwEngine(AsrEngine):
    name = "dtw"

    def __init__(self, cfg: AppConfig, log=print):
        self.cfg = cfg
        self.log = log
        self.extractor = FeatureExtractor(cfg.sample_rate)
        self.vad = EnergyVad(cfg.sample_rate, start_db=cfg.vad_start_db,
                             margin_db=cfg.vad_margin_db)
        self.store: TemplateStore | None = None
        self._mode = "wake"
        self._buf = bytearray()
        self._silence_ms = 0.0
        self._speech_seen = False
        self.notes: list[str] = []
        self.last_scores: list[dict] = []
        self.last_decision = Decision()

    # ------------------------------------------------------------ 生命周期
    def start(self) -> None:
        if self.store is not None:
            return
        path = self.cfg.resolve(self.cfg.dtw_model)
        self.store = TemplateStore.load(path)
        if self.store.metric != self.cfg.dtw_metric:
            self._note(f"模板库用 {self.store.metric} 距离标定，当前配置是 "
                       f"{self.cfg.dtw_metric}，建议重新注册模板")
        if self.store.feature_dim != self.extractor.dim:
            self._note(f"模板特征维度 {self.store.feature_dim} 与当前配置 {self.extractor.dim} "
                       f"不一致，需要重新注册模板")
        self.log(f"[引擎] 加载 DTW 模板库：{path}")
        self.log(self.store.describe())

    def close(self) -> None:
        self.store = None

    def reset(self) -> None:
        self._buf.clear()
        self._silence_ms = 0.0
        self._speech_seen = False
        self.vad = EnergyVad(self.cfg.sample_rate, start_db=self.cfg.vad_start_db,
                             margin_db=self.cfg.vad_margin_db)

    def set_mode(self, mode: str) -> None:
        if mode not in ("wake", "command"):
            raise ValueError(f"mode 只能是 wake/command，当前 {mode!r}")
        self._mode = mode

    def take_preroll(self) -> list[bytes]:
        # 本引擎在唤醒的同一段里已经顺手匹配过口令，不需要回灌音频
        return []

    # ------------------------------------------------------------ 音频 -> 文本
    def feed(self, pcm: bytes) -> list[PhraseHit]:
        if self.store is None:
            self.start()
        events: list[PhraseHit] = []
        self._buf += pcm
        block_ms = len(pcm) / 2 * 1000.0 / self.cfg.sample_rate

        if self.vad.is_speech(pcm):
            self._speech_seen = True
            self._silence_ms = 0.0
        elif self._speech_seen:
            self._silence_ms += block_ms

        seg_ms = len(self._buf) / 2 * 1000.0 / self.cfg.sample_rate
        finished = self._speech_seen and (
            self._silence_ms >= self.cfg.dtw_end_silence_ms
            or seg_ms >= self.cfg.dtw_max_segment_ms
        )
        if finished:
            segment = bytes(self._buf)
            events = self._match_segment(segment)
            if self._silence_ms >= self.cfg.dtw_end_silence_ms:
                self._buf.clear()
            else:
                # 还在说话但段太长：留 200ms 上下文再开新段，避免把字切掉
                keep = int(self.cfg.sample_rate * 0.2) * 2
                self._buf = bytearray(self._buf[-keep:])
            self._silence_ms = 0.0
            self._speech_seen = False
        return events

    def flush(self) -> list[PhraseHit]:
        if self._buf and self._speech_seen:
            segment = bytes(self._buf)
            self._buf.clear()
            self._speech_seen = False
            return self._match_segment(segment)
        return []

    # ------------------------------------------------------------ 匹配
    def _match_segment(self, segment: bytes) -> list[PhraseHit]:
        assert self.store is not None
        feats = self.extractor.extract(segment)
        min_frames = max(3, int(self.cfg.dtw_min_segment_ms / 10.0))
        if feats.shape[0] < min_frames:
            return []

        if self._mode == "wake":
            decision = self._decide(feats, KIND_WAKE, self.cfg.dtw_wake_threshold)
            if not decision.accepted:
                return []
            events = [PhraseHit(kind="wake", text=decision.phrase,
                                confidence=self._confidence(decision.dist, decision.threshold))]
            # 同一段里可能跟着口令："小裁判左方加一分"
            follow = self._decide(feats, KIND_COMMAND, self.cfg.dtw_command_threshold)
            if follow.accepted:
                events.append(PhraseHit(kind="command", text=follow.phrase,
                                        confidence=self._confidence(follow.dist, follow.threshold),
                                        phase="preroll"))
            return events

        decision = self._decide(feats, KIND_COMMAND, self.cfg.dtw_command_threshold)
        if not decision.accepted:
            # 明确告诉上层"这段没听懂"，让它走"请再说一遍"的流程
            return [PhraseHit(kind="command", text="", confidence=0.0)]
        return [PhraseHit(kind="command", text=decision.phrase,
                          confidence=self._confidence(decision.dist, decision.threshold))]

    def _decide(self, feats, kind: str, default_threshold: float) -> Decision:
        """两条判据：绝对距离进阈值内 + 与其它意图拉开足够倍数。

        只加绝对阈值会出现"某个短语阈值本来就松，于是什么都匹配到它"，
        幅度余量把这类勉强命中判成歧义，交给上层提示重说，
        比猜一个意图去改比分安全得多。
        """
        assert self.store is not None
        templates = self.store.by_kind(kind)
        scores: list[dict] = []
        scored: list[tuple[float, Template, float]] = []
        for t in templates:
            if feats.shape[0] < 0.5 * t.frames:
                continue
            threshold = t.threshold if t.threshold > 0 else default_threshold
            dist = dtw(feats, t.feats, subsequence=True, metric=self.cfg.dtw_metric)
            scores.append({"phrase": t.phrase, "dist": round(float(dist), 3),
                           "threshold": round(float(threshold), 3), "intent": t.intent})
            scored.append((float(dist), t, float(threshold)))
        scores.sort(key=lambda item: item["dist"])
        self.last_scores = scores[:5]

        if not scored:
            decision = Decision(reason="too_short")
            self.last_decision = decision
            return decision

        scored.sort(key=lambda item: item[0])
        under = [item for item in scored if item[0] <= item[2]]
        if not under:
            dist, template, threshold = scored[0]
            decision = Decision(phrase=template.phrase, intent=template.intent,
                                dist=dist, threshold=threshold, reason="low_similarity")
            self.last_decision = decision
            return decision

        dist, template, threshold = under[0]
        decision = Decision(phrase=template.phrase, intent=template.intent,
                            dist=dist, threshold=threshold, accepted=True, reason="ok")
        runner = next((item for item in scored if item[1].intent != template.intent), None)
        if runner is not None:
            decision.runner_phrase = runner[1].phrase
            decision.runner_dist = runner[0]
            if runner[0] < dist * self.cfg.dtw_margin:
                decision.accepted = False
                decision.reason = "ambiguous"
        self.last_decision = decision
        return decision

    @staticmethod
    def _confidence(dist: float, threshold: float) -> float:
        if threshold <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - dist / threshold))

    def _note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)
            self.log(f"[引擎] {message}")
