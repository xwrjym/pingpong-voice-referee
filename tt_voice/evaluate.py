"""离线验收工具：唤醒率 / 误唤醒率 / 口令识别准确率 / 阈值扫描。

用法（测试集可以由 tools/make_test_wav.py 自动生成）：

    python tools/make_test_wav.py --out testdata
    python -m tt_voice.cli eval testdata

测试集约定：目录内 labels.json 描述每个 wav：

    {"file": "cmd__left_01__clean.wav", "kind": "cmd", "text": "左方加一分"}
    {"file": "wake__xiao_caipan_01__clean.wav", "kind": "wake"}
    {"file": "neg__chat_01__clean.wav", "kind": "neg"}
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from .audio_io import WavSource
from .config import AppConfig
from .engines.base import AsrEngine, PhraseHit
from .feedback import NullSpeaker
from .parser import Parser
from .score import ScoreBoard
from .app import Pipeline


@dataclass
class Sample:
    path: str
    kind: str                 # "wake" | "cmd" | "neg"
    text: str = ""            # cmd 的期望口令
    note: str = ""
    expected_intent: str = ""


@dataclass
class SampleResult:
    sample: Sample
    ok: bool
    predicted: str = ""
    accepted: bool = False
    score: float = 0.0
    decoded: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


def load_samples(directory: str, labels_file: str = "labels.json") -> list[Sample]:
    labels_path = os.path.join(directory, labels_file)
    with open(labels_path, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    samples: list[Sample] = []
    for row in rows:
        path = os.path.join(directory, row["file"])
        if not os.path.exists(path):
            continue
        samples.append(Sample(
            path=path,
            kind=row.get("kind", "cmd"),
            text=row.get("text", ""),
            note=row.get("note", ""),
            expected_intent=row.get("intent", ""),
        ))
    return samples


def intent_of(cfg: AppConfig, text: str) -> str:
    """给定的口令文本应该映射到哪个意图（用于 cmd 样本的期望值）。"""
    for intent, phrase in cfg.all_command_phrases():
        if phrase == text:
            return intent
    return ""


def run_samples(cfg: AppConfig, engine: AsrEngine, samples: list[Sample],
                log: Callable[[str], None] = print) -> list[SampleResult]:
    results: list[SampleResult] = []
    for sample in samples:
        name = os.path.basename(sample.path)
        if sample.kind in ("wake", "neg", "combo"):
            # 走真实状态机：唤醒 -> (combo 时) 连着说口令
            engine.reset()
            engine.set_mode("wake")
            pipe = Pipeline(cfg, engine=engine, board=ScoreBoard(cfg, log_events=False),
                            speaker=NullSpeaker(),
                            log=lambda *_: None, start_awake=False)
            now, step = 0.0, cfg.block_ms / 1000.0
            for block in WavSource(sample.path, sample_rate=cfg.sample_rate,
                                   block_size=cfg.block_size):
                now += step
                pipe.feed(block, now=now)
            pipe.flush(now)
            woke = any(r["kind"] == "wake" and r["accepted"] for r in pipe.records)
            accepted = [r for r in pipe.records if r["accepted"] and r["kind"] == "command"]
            decoded = " / ".join(f"{r['kind']}:{r['text']}" for r in pipe.records)
            if sample.kind == "wake":
                ok = woke
                label, extra = "唤醒" if woke else "未唤醒", ""
                predicted = "wake" if woke else ""
            elif sample.kind == "neg":
                ok = not woke
                label, extra = ("误唤醒!" if woke else "正常"), ""
                predicted = "wake" if woke else ""
            else:
                expected = sample.expected_intent or intent_of(
                    cfg, sample.text.replace("，", ",").split(",")[-1])
                predicted = accepted[0]["intent"] if accepted else ""
                ok = woke and bool(accepted) and predicted == expected
                label = "OK " if ok else f"失败(唤醒={woke})"
                extra = f" 期望 {expected} 得到 {predicted or '（未执行）'}"
            score = accepted[0]["score"] if accepted else 0.0
            results.append(SampleResult(sample, ok=ok, predicted=predicted,
                                        accepted=woke if sample.kind != "combo" else (
                                            woke and bool(accepted)),
                                        score=score, decoded=decoded))
            log(f"  [{sample.kind:<5}] {name:<42} -> {label}  {decoded}{extra}")
        else:
            expected = sample.expected_intent or intent_of(cfg, sample.text)
            engine.reset()
            engine.set_mode("command")
            hits: list[PhraseHit] = []
            for block in WavSource(sample.path, sample_rate=cfg.sample_rate,
                                   block_size=cfg.block_size):
                hits += engine.feed(block)
            hits += engine.flush()
            pipe = Pipeline(cfg, engine=None, speaker=NullSpeaker(),
                            board=ScoreBoard(cfg, log_events=False),
                            log=lambda *_: None, start_awake=True)
            for hit in hits:
                pipe.push_text(hit.text, kind=hit.kind, confidence=hit.confidence)
            accepted = [r for r in pipe.records if r["accepted"]]
            best = accepted[0] if accepted else (pipe.records[0] if pipe.records else None)
            ok = bool(accepted) and accepted[0]["intent"] == expected
            results.append(SampleResult(
                sample, ok=ok,
                predicted=accepted[0]["intent"] if accepted else "",
                accepted=bool(accepted),
                score=best["score"] if best else 0.0,
                decoded=" / ".join(r["text"] for r in pipe.records),
            ))
            flag = "OK " if ok else ("误判" if accepted else "漏识别")
            detail = ""
            scores = getattr(engine, "last_scores", None)
            if not ok and scores:
                detail = " 最近5名: " + ", ".join(
                    f"{s['phrase']}={s['dist']}/{s['threshold']}" for s in scores)
            log(f"  [cmd  ] {name:<42} -> {flag} "
                f"{results[-1].decoded!r} 期望 {expected} "
                f"得到 {results[-1].predicted or '（拒识）'} score={results[-1].score:.3f}{detail}")
    return results


# ------------------------------------------------------------------ 阈值扫描
def sweep_thresholds(cfg: AppConfig, engine: AsrEngine, samples: list[Sample],
                     thresholds: list[float], margins: list[float],
                     log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """解码一次，网格扫描 (match_threshold, match_margin)，避免反复跑模型。"""
    cmd_samples = [s for s in samples if s.kind == "cmd"]
    decoded: list[tuple[Sample, str, float, str]] = []
    for sample in cmd_samples:
        expected = sample.expected_intent or intent_of(cfg, sample.text)
        engine.reset()
        engine.set_mode("command")
        hits: list[PhraseHit] = []
        for block in WavSource(sample.path, sample_rate=cfg.sample_rate,
                               block_size=cfg.block_size):
            hits += engine.feed(block)
        hits += engine.flush()
        text = " ".join(h.text for h in hits)
        conf = hits[0].confidence if hits else 0.0
        decoded.append((sample, text, conf, expected))

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for margin in margins:
            trial = Parser(AppConfig.from_dict({**cfg.to_dict(),
                                                "match_threshold": threshold,
                                                "match_margin": margin},
                                               base_dir=cfg.base_dir))
            correct = wrong = missed = 0
            for sample, text, conf, expected in decoded:
                match = trial.parse_command(text, asr_conf=conf)
                if not match.accepted:
                    missed += 1
                elif match.intent == expected:
                    correct += 1
                else:
                    wrong += 1
            total = max(1, len(decoded))
            rows.append({
                "threshold": threshold,
                "margin": margin,
                "accuracy": correct / total,
                "correct": correct,
                "wrong": wrong,
                "missed": missed,
            })
    rows.sort(key=lambda r: (-r["accuracy"], -r["threshold"], -r["margin"]))
    log("  阈值扫描（前 5 名）：threshold / margin -> 准确率 (正确/错判/漏识别)")
    for row in rows[:5]:
        log(f"    {row['threshold']:.2f} / {row['margin']:.2f} -> {row['accuracy']:.1%}"
            f"  ({row['correct']}/{row['wrong']}/{row['missed']})")
    return rows


# ------------------------------------------------------------------ 汇总
def summarize(results: list[SampleResult]) -> dict[str, Any]:
    wake = [r for r in results if r.sample.kind == "wake"]
    neg = [r for r in results if r.sample.kind == "neg"]
    cmd = [r for r in results if r.sample.kind == "cmd"]
    combo = [r for r in results if r.sample.kind == "combo"]
    accepted = [r for r in cmd if r.accepted]
    wrong = [r for r in accepted if not r.ok]
    summary = {
        "唤醒样本": len(wake),
        "唤醒成功": sum(1 for r in wake if r.accepted),
        "唤醒率": (sum(1 for r in wake if r.accepted) / len(wake)) if wake else None,
        "干扰样本": len(neg),
        "误唤醒": sum(1 for r in neg if r.accepted),
        "误唤醒率": (sum(1 for r in neg if r.accepted) / len(neg)) if neg else None,
        "口令样本": len(cmd),
        "口令正确": sum(1 for r in cmd if r.ok),
        "口令准确率": (sum(1 for r in cmd if r.ok) / len(cmd)) if cmd else None,
        "误判": len(wrong),
        "漏识别": len(cmd) - len(accepted),
    }
    if combo:
        summary["连说样本"] = len(combo)
        summary["连说成功率"] = sum(1 for r in combo if r.ok) / len(combo)
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    lines = ["验收汇总："]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.1%}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def write_csv(results: list[SampleResult], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "kind", "expected", "decoded", "predicted",
                         "accepted", "score", "ok", "note"])
        for r in results:
            writer.writerow([
                os.path.basename(r.sample.path), r.sample.kind,
                r.sample.expected_intent or r.sample.text, r.decoded,
                r.predicted, int(r.accepted), f"{r.score:.3f}", int(r.ok), r.sample.note,
            ])
    return path
