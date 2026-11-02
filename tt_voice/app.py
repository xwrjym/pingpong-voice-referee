"""主状态机：待唤醒 -> 聆听 -> 执行 -> 反馈。

设计要点：

* 只有「唤醒后的一小段时间窗口」才收口令，窗口外任何语音都不执行动作，
  这样误识别最多是重说一遍，不会改错比分；
* 唤醒瞬间把引擎缓存的前 1.5s 音频回灌给口令识别器，避免「小裁判左方加一分」
  这种一口气说完的用法丢掉口令；
* 每个动作都产出 ActionResult，UI/语音/日志都从同一个地方拿结果。
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Callable

from .config import AppConfig
from .engines.base import AsrEngine, PhraseHit
from .feedback import Speaker, build_speaker
from .parser import Match, Parser
from .score import ActionResult, ScoreBoard
from .textnorm import normalize
from .vad import EnergyVad

STATE_IDLE = "idle"
STATE_LISTENING = "listening"


class Pipeline:
    def __init__(self, cfg: AppConfig, engine: AsrEngine | None = None,
                 parser: Parser | None = None, board: ScoreBoard | None = None,
                 speaker: Speaker | None = None, log: Callable[[str], None] = print,
                 start_awake: bool = False):
        self.cfg = cfg
        self.engine = engine
        self.parser = parser or Parser(cfg)
        self.board = board or ScoreBoard(cfg)
        self.speaker = speaker if speaker is not None else build_speaker()
        self.log = log

        self.state = STATE_LISTENING if start_awake else STATE_IDLE
        self.deadline: float | None = None
        self.reset_deadline: float | None = None
        self._tolerant_until: float | None = None
        self.commands_in_session = 0
        self.reask_count = 0
        self.speech_seen = False
        self.vad = EnergyVad(cfg.sample_rate, start_db=cfg.vad_start_db, margin_db=cfg.vad_margin_db)
        self.stats: Counter[str] = Counter()
        self.records: list[dict[str, Any]] = []
        self._started = False
        self._mute_while_speaking = False   # 只有"可能被听成口令"的播报才静音

    # ------------------------------------------------------------ 生命周期
    def start(self) -> None:
        if self.engine is not None:
            self.engine.start()
            self.engine.set_mode("command" if self.state == STATE_LISTENING else "wake")
        self._started = True

    def close(self) -> None:
        if self.engine is not None:
            self.engine.close()
        self.speaker.close()
        try:
            self.board.save()
        except OSError:
            pass

    # ------------------------------------------------------------ 输入
    def feed(self, pcm: bytes, now: float | None = None) -> list[ActionResult]:
        now = self._now(now)
        results = self._tick_timeouts(now)
        if self._mute_while_speaking and getattr(self.speaker, "is_speaking", False):
            # 半双工静音只用于"比分播报"这类含口令词的播报：喇叭里的
            # "左方加一分，左方1比0右方"会被自己的麦克风听成口令。
            # "在/请再说一遍/没听到"这类短提示不含口令词，不能占用麦克风——
            # 用户往往一听提示就开口，静音会把整句话丢掉（真人验收实测过）。
            # 静音期间顺延聆听窗口，别让播报吃掉用户的说话时间。
            if self.deadline is not None:
                self.deadline = max(self.deadline, now + 1.0)
            self.stats["muted_while_speaking"] += 1
            return results
        if self.state == STATE_LISTENING:
            if self.vad.is_speech(pcm):
                self.speech_seen = True
                if self.deadline is not None:
                    # 用户还在说，别急着超时
                    self.deadline = max(self.deadline, now + 2.0)
        if self.engine is not None:
            results += self._handle_all(self.engine.feed(pcm), now)
        results += self._tick_timeouts(self._now(now))
        return results

    def flush(self, now: float | None = None) -> list[ActionResult]:
        if self.engine is None:
            return []
        now = self._now(now)
        return self._handle_all(self.engine.flush(), now)

    def push_text(self, text: str, now: float | None = None, kind: str = "command",
                  confidence: float = 0.0) -> list[ActionResult]:
        """文本直接注入（离线调试 / 单元测试 / 将来接别的 ASR）。"""
        now = self._now(now)
        return self._handle_all([PhraseHit(kind=kind, text=text, confidence=confidence)], now)

    def tick(self, now: float | None = None) -> list[ActionResult]:
        return self._tick_timeouts(self._now(now))

    # ------------------------------------------------------------ 事件处理
    def _handle_all(self, events: list[PhraseHit], now: float) -> list[ActionResult]:
        results: list[ActionResult] = []
        for event in events:
            results += self._handle(event, now)
        return results

    def _handle(self, event: PhraseHit, now: float) -> list[ActionResult]:
        if self.state == STATE_IDLE:
            return self._handle_idle(event, now)
        return self._handle_listening(event, now)

    def _handle_idle(self, event: PhraseHit, now: float) -> list[ActionResult]:
        if not event.text.strip():
            # 引擎主动上报"这段没听懂"（例如 DTW 距离超阈值），待机态直接忽略
            return []
        match = self.parser.matches_wake(event.text)
        if event.phase == "partial" and not match.accepted:
            # 部分结果的中间态（"小裁"），不算误唤醒，也不记日志
            return []
        if not match.accepted:
            self.stats["wake_ignored"] += 1
            self._record(event, match, accepted=False, note="not_wake")
            return []
        self.stats["wakes"] += 1
        self._record(event, match, accepted=True, note="wake")
        self.state = STATE_LISTENING
        self.commands_in_session = 0
        self.reask_count = 0
        self.speech_seen = False
        self.deadline = now + self.cfg.listen_timeout_s
        self._tolerant_until = now + self.cfg.wake_tolerant_s
        out = [ActionResult(ok=True, message="在", intent="wake", state=self.board.snapshot())]
        self._say("在", beep="wake", mute=False)
        # 回灌缓存音频，找回「一口气说完」时被唤醒识别器吃掉的口令
        if self.engine is not None:
            self.engine.set_mode("command")
            for block in self.engine.take_preroll():
                for echo in self.engine.feed(block):
                    out += self._handle(echo, now)
            # 回灌结束点：之后才定稿的回声靠它识别（见 VoskEngine._phase_of）
            self.engine.mark_handoff()
        return out

    def _handle_listening(self, event: PhraseHit, now: float) -> list[ActionResult]:
        tolerant = self._tolerant_until is not None and now <= self._tolerant_until
        if not tolerant and event.kind == "command" \
                and self.parser.matches_wake(event.text).accepted:
            # 会话中又喊了一次唤醒词：续期，而不是当成没听懂（用户很自然会这么做）
            self.stats["rewake"] += 1
            self.reask_count = 0
            self.deadline = now + self.cfg.listen_timeout_s
            self._say("在", beep="wake", mute=False)
            return [ActionResult(ok=True, message="在", intent="wake",
                                 state=self.board.snapshot())]
        if event.kind == "wake":
            # 还在唤醒模式就收到唤醒模式结果（模式切换瞬间），忽略
            self.stats["wake_ignored"] += 1
            return []

        # 唤醒后一小段时间内，识别文本里可能混着唤醒词残字（回灌音频、
        # 或唤醒识别器把后面的字粘进来），按"口令可以出现在任意位置"来匹配
        match = self.parser.parse_command(event.text, asr_conf=event.confidence,
                                          tolerant=tolerant)
        self._record(event, match, accepted=match.accepted, note=match.reason,
                     phase=event.phase)
        if match.accepted and match.intent:
            self.stats["commands_ok"] += 1
            self.reask_count = 0
            self._tolerant_until = None
            result = self.board.apply(match.intent, raw_text=event.text, score=match.score,
                                      source="voice")
            if result.reason == "need_second_confirm":
                self.reset_deadline = now + self.cfg.reset_confirm_timeout_s
            elif not self.board.awaiting_reset_confirm:
                self.reset_deadline = None
            self._say(result.message, beep="ok" if result.ok else "reject")
            self.commands_in_session += 1
            if self.commands_in_session >= self.cfg.max_commands_per_wake or not result.ok:
                return [result] + self._leave_listening(now)
            self.deadline = now + self.cfg.listen_timeout_s
            return [result]

        if event.text.strip() and tolerant and self.parser.matches_wake(event.text).accepted:
            # 这段其实就是唤醒词本身（回灌音频里的残字），静默跳过
            self.stats["wake_echo"] += 1
            return []
        if event.phase in ("preroll", "partial"):
            # 引擎回灌/部分结果阶段的拒识不算用户的错，不提示、不计数
            return []
        if tolerant and 0 < len(normalize(event.text)) <= 3:
            # 唤醒后极短的垃圾残字（实测"消操""销才分"），静默丢弃；
            # 真正的长句听不懂仍然照常提示"请再说一遍"
            self.stats["junk_ignored"] += 1
            return []

        self.stats["commands_rejected"] += 1
        if match.reason == "ambiguous":
            self.stats["ambiguous"] += 1
        if not getattr(self.cfg, "reask_on_reject", True):
            # 演示/安静模式：不是口令就完全不反馈，直接退出聆听窗口
            self.stats["silent_rejects"] += 1
            result = ActionResult(ok=False, message="", intent="",
                                  reason=match.reason, state=self.board.snapshot())
            return [result] + self._leave_listening(now)
        self.reask_count += 1
        if self.reask_count > self.cfg.max_reask:
            self._say("没听清", beep="reject", mute=False)
            result = ActionResult(ok=False, message="没听清", intent="", reason=match.reason,
                                  state=self.board.snapshot())
            return [result] + self._leave_listening(now)
        self._say("请再说一遍", beep="reject", mute=False)
        self.deadline = now + self.cfg.listen_timeout_s
        return [ActionResult(ok=False, message="请再说一遍", intent="", reason=match.reason,
                             requires_reask=True, state=self.board.snapshot())]

    # ------------------------------------------------------------ 超时
    def _tick_timeouts(self, now: float) -> list[ActionResult]:
        results: list[ActionResult] = []
        if self.reset_deadline is not None:
            if not self.board.awaiting_reset_confirm:
                self.reset_deadline = None
            elif now >= self.reset_deadline:
                self.board.awaiting_reset_confirm = False
                self.reset_deadline = None
                results.append(ActionResult(ok=False, message="重置已取消", intent="reset",
                                            reason="reset_timeout", state=self.board.snapshot()))
        if self.state == STATE_LISTENING and self.deadline is not None and now >= self.deadline:
            message = "已退出" if self.speech_seen else "没听到"
            self.stats["timeouts"] += 1
            self._say(message, beep="timeout", mute=False)
            results.append(ActionResult(ok=False, message=message, intent="timeout",
                                        reason="timeout", state=self.board.snapshot()))
            results += self._leave_listening(now)
        return results

    def _leave_listening(self, now: float) -> list[ActionResult]:
        self.state = STATE_IDLE
        self.deadline = None
        self.commands_in_session = 0
        self.reask_count = 0
        self.speech_seen = False
        self._tolerant_until = None
        if self.engine is not None:
            self.engine.set_mode("wake")
        return []

    # ------------------------------------------------------------ 工具
    def _say(self, text: str, beep: str = "ok", mute: bool = True) -> None:
        """mute=False 用于不含口令词的短提示（"在""请再说一遍""没听到"）。"""
        if not text:
            return
        self._mute_while_speaking = mute
        self.speaker.beep(beep)
        self.speaker.say(text)

    def _record(self, event: PhraseHit, match: Match, accepted: bool, note: str,
                phase: str = "live") -> None:
        self.records.append({
            "kind": event.kind,
            "phase": phase,
            "text": event.text,
            "asr_conf": round(event.confidence, 3),
            "intent": match.intent,
            "phrase": match.phrase,
            "score": round(match.score, 3),
            "runner_up": match.runner_up_intent,
            "runner_up_score": round(match.runner_up_score, 3),
            "accepted": accepted,
            "note": note,
        })

    @staticmethod
    def _now(now: float | None) -> float:
        return time.monotonic() if now is None else now

    def summary(self) -> dict[str, Any]:
        data = dict(self.stats)
        data["score"] = self.board.snapshot()
        return data
