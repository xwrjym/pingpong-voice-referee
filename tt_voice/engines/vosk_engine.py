"""Vosk（Kaldi）后端：受限词表解码 + 唤醒/口令两种模式。

为什么用 grammar（受限词表）：

* 唤醒常驻运行时，全词表解码的 CPU 开销和误唤醒率都明显更高；
* 口令是固定集合，把候选限定成这几十个词，结果天然不会跑偏；
* 仍保留 open 模式：中文小模型词典可能缺字，缺字时自动降级，不会跑不起来。

一个坑：唤醒词和口令连着说（"小裁判左方加一分"）时，唤醒识别器要等静音才出结果，
这期间的口令音频已经被它吃掉了。所以引擎缓存最近 1.5s 音频，唤醒后回灌给口令识别器。
"""

from __future__ import annotations

import json
import os
from collections import deque

import numpy as np

from ..config import AppConfig
from ..textnorm import normalize, normalize_light
from .base import AsrEngine, PhraseHit


class VoskEngine(AsrEngine):
    name = "vosk"

    def __init__(self, cfg: AppConfig, log=print, preroll_s: float = 1.5):
        self.cfg = cfg
        self.log = log
        self.sample_rate = cfg.sample_rate
        self.model = None
        self._rec = None
        self._mode = "wake"
        self._preroll: deque[bytes] = deque()
        self._preroll_limit = max(1, int(preroll_s * 1000 / max(1, cfg.block_ms)))
        self._blocks_fed = 0          # 已喂入的块数，用来把识别器时间轴映射到缓存块
        self._stream_block_start = 0  # 当前识别器开始时的块号
        self._wake_end_s: float | None = None
        self._wake_ref_block = 0
        self._last_partial = ""
        self._handoff_block: int | None = None   # 回灌结束时的块号（None=还没回灌过）
        self.modes = {"wake": cfg.wake_mode, "command": cfg.command_mode}
        self.notes: list[str] = []
        self._vocab: set[str] | None = None

    # ------------------------------------------------------------ 生命周期
    def start(self) -> None:
        import vosk

        vosk.SetLogLevel(-1)
        model_path = self.cfg.model_path
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"找不到语音模型目录：{model_path}，请先运行 python tools/download_model.py"
            )
        if self.model is not None:
            # 允许多个 Pipeline 复用同一个模型实例（离线批量评估时省掉重复加载）
            return
        self._vocab = self._load_vocab(model_path)
        self.log(f"[引擎] 加载 Vosk 模型：{model_path}")
        self.model = vosk.Model(model_path)
        self._build_recognizer("wake")

    def close(self) -> None:
        self._rec = None
        self.model = None

    def reset(self) -> None:
        self._preroll.clear()
        self._build_recognizer(self._mode)

    def set_mode(self, mode: str) -> None:
        if mode not in ("wake", "command"):
            raise ValueError(f"mode 只能是 wake/command，当前 {mode!r}")
        if mode == self._mode and self._rec is not None:
            return
        self._build_recognizer(mode)
        self._last_partial = ""

    # ------------------------------------------------------------ 音频 -> 文本
    def feed(self, pcm: bytes) -> list[PhraseHit]:
        self._blocks_fed += 1
        self._push_preroll(pcm)
        if self._rec is None:
            return []
        if self._rec.AcceptWaveform(pcm):
            return self._emit(self._rec.Result())
        if self._mode == "wake":
            # 不等静音定稿：唤醒词一冒头就上报，否则用户连着说口令时
            # 唤醒识别器会把整句吞进自己的结果里（实测得到"小裁判小"，后面的"撤销"就没了）
            partial = json.loads(self._rec.PartialResult()).get("partial", "")
            partial = partial.replace(" ", "").replace("[unk]", "")
            if partial and partial != self._last_partial and len(partial) >= 2:
                self._last_partial = partial
                return [PhraseHit(kind="wake", text=partial, confidence=0.0, phase="partial")]
        return []

    def flush(self) -> list[PhraseHit]:
        if self._rec is None:
            return []
        return self._emit(self._rec.FinalResult())

    def take_preroll(self) -> list[bytes]:
        blocks = list(self._preroll)
        self._preroll.clear()
        # 故意不做时间裁剪：唤醒识别器的时间戳在受限词表下不可靠
        # （"小裁判撤销"会被解码成"小裁判小"，结束时间已经越过了口令的开头，
        #  按它裁剪会连口令第一个字一起切掉）。混进来的唤醒词残字由
        # parser 的 tolerant 匹配消化。
        self._wake_end_s = None
        return blocks

    def mark_handoff(self) -> None:
        self._handoff_block = self._blocks_fed

    # ------------------------------------------------------------ 识别器构建
    def _build_recognizer(self, mode: str) -> None:
        import vosk

        policy = self.modes[mode]
        phrases = self.cfg.wake_words if mode == "wake" else [p for _, p in self.cfg.all_command_phrases()]
        if policy == "grammar":
            tokens = self._char_tokens(phrases)
            missing = self._missing_tokens(tokens)
            if missing:
                self._note(
                    f"{mode} 模式：模型词典缺少 {missing}，已自动降级为 open（全词表）模式"
                )
            else:
                grammar = json.dumps(tokens + ["[unk]"], ensure_ascii=False)
                self._rec = vosk.KaldiRecognizer(self.model, self.sample_rate, grammar)
                self._rec.SetWords(True)
                self._mode = mode
                self._stream_block_start = self._blocks_fed
                self._note(f"{mode} 模式：受限词表 {len(tokens)} 条")
                return
        self._rec = vosk.KaldiRecognizer(self.model, self.sample_rate)
        self._rec.SetWords(True)
        self._mode = mode
        self._stream_block_start = self._blocks_fed

    @staticmethod
    def _char_tokens(phrases: list[str]) -> list[str]:
        """把「左方加一分」变成 grammar 需要的 "左 方 加 一 分"。

        注意必须用**保留中文数字**的归一化：Vosk 中文小模型的词典里没有 "1"，
        写成 "加 1 分" 会被整个丢掉，识别结果会变成"加记分"这种鬼东西
        （实测 confidence 还有 0.97，非常隐蔽）。
        """
        tokens: list[str] = []
        for phrase in phrases:
            norm = normalize_light(phrase)
            if not norm:
                continue
            token = " ".join(list(norm))
            if token not in tokens:
                tokens.append(token)
        return tokens

    def _load_vocab(self, model_path: str) -> set[str] | None:
        """读模型自带的词典（graph/words.txt），用来预判 grammar 会不会缺字。"""
        words_path = os.path.join(model_path, "graph", "words.txt")
        if not os.path.exists(words_path):
            return None
        vocab: set[str] = set()
        with open(words_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.split()
                if parts and not parts[0].startswith("#"):
                    vocab.add(parts[0])
        return vocab

    def _missing_tokens(self, tokens: list[str]) -> list[str]:
        if self._vocab is None:
            return []
        missing: list[str] = []
        for token in tokens:
            for piece in token.split():
                if piece not in self._vocab and piece not in missing:
                    missing.append(piece)
        return missing

    # ------------------------------------------------------------ 结果解析
    def _emit(self, payload: str) -> list[PhraseHit]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return []
        words = [(w.get("word", ""), float(w.get("conf", 0.0))) for w in data.get("result", [])]
        if self._mode == "wake" and data.get("result"):
            ends = [float(w.get("end", 0.0)) for w in data["result"] if w.get("word")]
            if ends:
                self._wake_end_s = max(ends)
                self._wake_ref_block = self._stream_block_start
        kept = [(w, c) for w, c in words if w and w != "[unk]"]
        if kept:
            text = "".join(w for w, _ in kept)
        else:
            text = (data.get("text") or "").replace(" ", "").replace("[unk]", "")
            if text and not words:
                # 没有 word 级信息时，用整体文本、置信度按 0 记（上层可关闭该判据）
                kept = [(text, 0.0)]
        if not text:
            return []
        conf = float(np.mean([c for _, c in kept])) if kept else 0.0
        return [PhraseHit(kind=self._mode, text=text, confidence=conf, words=kept,
                          phase=self._phase_of(data))]

    def _phase_of(self, data: dict) -> str:
        """判断这次定稿的音频是不是来自唤醒后回灌的那段缓存。

        只看**第一个字的起点**：回灌进来的唤醒词残音如果和用户真实说的口令
        粘成了同一段（实测会得到"销才分对方加一分"这种混合串），按结束时间判断会漏，
        按起点判断则一定落在回灌区间内。

        （这是另一台电脑做真人验收时定位并修掉的问题：误判成 live 会触发
          本不该有的"请再说一遍"，而那次播报又把用户紧接着说的口令整句丢掉。）
        """
        if self._mode != "command" or self._handoff_block is None:
            return "live"
        starts = [float(w.get("start", 0.0))
                  for w in data.get("result", []) if w.get("word")]
        if not starts:
            return "live"
        block_ms = max(1, self.cfg.block_ms)
        start_block = self._stream_block_start + int(round(min(starts) * 1000.0 / block_ms))
        return "preroll" if start_block <= self._handoff_block else "live"

    # ------------------------------------------------------------ 内部
    def _push_preroll(self, pcm: bytes) -> None:
        self._preroll.append(pcm)
        while len(self._preroll) > self._preroll_limit:
            self._preroll.popleft()

    def _note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)
            self.log(f"[引擎] {message}")
