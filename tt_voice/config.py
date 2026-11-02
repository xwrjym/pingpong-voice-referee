"""配置读写。所有字段都有默认值，config.json 只覆盖需要改的部分。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CommandSpec:
    intent: str
    phrases: list[str] = field(default_factory=list)


DEFAULT_COMMANDS = [
    CommandSpec("score_left", ["左方加一分", "左边加一分", "左方加分", "左加一分"]),
    CommandSpec("score_right", ["右方加一分", "右边加一分", "右方加分", "右加一分"]),
    CommandSpec("undo", ["撤销上一分", "撤销一分", "撤销", "回退一分"]),
    CommandSpec("reset", ["重置比分", "重置分数", "比分清零"]),
    CommandSpec("confirm", ["确认这一分", "确认得分", "确认"]),
    CommandSpec("report", ["报分", "现在比分", "当前比分", "比分多少"]),
    CommandSpec("dismiss", ["退出", "取消操作", "结束", "不用了"]),
]


@dataclass
class AppConfig:
    sample_rate: int = 16000
    block_ms: int = 100
    engine: str = "dtw"                 # dtw（零依赖模板匹配）/ vosk（Kaldi 小模型）
    self_side: str = "left"             # 个人自用时"我"是哪一方：left/right
    model_dir: str = "models/vosk-model-small-cn-0.22"
    dtw_model: str = "models/dtw_cn/templates.npz"
    dtw_wake_threshold: float = 1.6
    dtw_command_threshold: float = 1.6
    dtw_end_silence_ms: float = 400.0
    dtw_max_segment_ms: float = 3000.0
    dtw_min_segment_ms: float = 250.0
    dtw_margin: float = 1.2             # 与其它意图的最小距离倍数（小于它判为歧义）
    dtw_metric: str = "cosine"          # cosine（实测可分性最好）/ euclid

    wake_words: list[str] = field(default_factory=lambda: ["小裁判"])
    wake_mode: str = "grammar"          # grammar=受限词表（省 CPU、抗误唤醒） / open=全词表
    wake_threshold: float = 0.80
    wake_margin: float = 0.05
    wake_allow_missing_onset: bool = False   # 允许唤醒词被削掉第一个字（例如"裁判"）

    command_mode: str = "grammar"
    match_threshold: float = 0.80
    match_margin: float = 0.04
    asr_conf_min: float = 0.0

    commands: list[CommandSpec] = field(default_factory=lambda: list(DEFAULT_COMMANDS))

    listen_timeout_s: float = 6.0
    no_speech_timeout_s: float = 3.0
    max_commands_per_wake: int = 3
    max_reask: int = 1
    reask_on_reject: bool = True        # False: 听不懂时静默退出（演示用）
    reset_confirm_timeout_s: float = 8.0
    wake_tolerant_s: float = 3.0        # 唤醒后这段时间内，口令匹配容忍唤醒词残字

    vad_start_db: float = -42.0
    vad_margin_db: float = 8.0

    persist_state: str = "runtime/score_state.json"
    event_log: str = "runtime/events.jsonl"

    base_dir: str = "."
    source_path: str | None = None

    # ---------------------------------------------------------------- 工具
    @property
    def block_size(self) -> int:
        return int(self.sample_rate * self.block_ms / 1000)

    def all_command_phrases(self) -> list[tuple[str, str]]:
        """[(intent, phrase), ...]"""
        out: list[tuple[str, str]] = []
        for spec in self.commands:
            for p in spec.phrases:
                out.append((spec.intent, p))
        return out

    def resolve(self, path: str) -> str:
        if not path:
            return path
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.base_dir, path))

    @property
    def model_path(self) -> str:
        return self.resolve(self.model_dir)

    # ---------------------------------------------------------------- 序列化
    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: str = ".", source_path: str | None = None) -> "AppConfig":
        cfg = cls()
        cfg.base_dir = base_dir
        cfg.source_path = source_path
        for key, value in (data or {}).items():
            if key == "commands":
                cfg.commands = [
                    CommandSpec(intent=str(c.get("intent", "")), phrases=[str(p) for p in c.get("phrases", [])])
                    for c in value
                ]
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        data = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("base_dir", "source_path")
        }
        data["commands"] = [{"intent": c.intent, "phrases": list(c.phrases)} for c in self.commands]
        return data

    def save(self, path: str | None = None) -> str:
        target = path or self.source_path or os.path.join(self.base_dir, "config.json")
        os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        return target

    def validate(self) -> None:
        if self.engine not in ("dtw", "vosk"):
            raise ValueError(f"engine 只能是 dtw/vosk，当前 {self.engine!r}")
        if self.wake_mode not in ("grammar", "open"):
            raise ValueError(f"wake_mode 只能是 grammar/open，当前 {self.wake_mode!r}")
        if self.command_mode not in ("grammar", "open"):
            raise ValueError(f"command_mode 只能是 grammar/open，当前 {self.command_mode!r}")
        if not self.wake_words:
            raise ValueError("wake_words 不能为空")
        if self.sample_rate not in (8000, 16000, 32000, 44100, 48000):
            raise ValueError(f"不常见的采样率：{self.sample_rate}")
        seen: dict[str, str] = {}
        for intent, phrase in self.all_command_phrases():
            if phrase in seen and seen[phrase] != intent:
                raise ValueError(f"口令 {phrase!r} 同时属于 {seen[phrase]} 和 {intent}，请去掉重复")
            seen[phrase] = intent


def load_config(path: str | None = None) -> AppConfig:
    if path is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "config.json")
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到配置文件：{path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return AppConfig.from_dict(data, base_dir=os.path.dirname(path), source_path=path)
