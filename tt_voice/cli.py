"""命令行入口。

    python -m tt_voice.cli devices                      列出麦克风
    python -m tt_voice.cli selftest                     检查配置/模型是否就绪
    python -m tt_voice.cli mic [--device 1] [--say]     实时唤醒+口令
    python -m tt_voice.cli wav a.wav b.wav              离线回放 wav（不依赖麦克风）
    python -m tt_voice.cli text 小裁判 左方加一分 撤销     纯文本跑一遍状态机
    python -m tt_voice.cli eval testdata [--sweep]       跑测试集，出唤醒率/准确率
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .app import Pipeline
from .audio_io import MicSource, WavSource, list_input_devices
from .config import load_config
from .engines import build_engine
from .evaluate import (format_summary, load_samples, run_samples, summarize,
                       sweep_thresholds, write_csv)
from .feedback import NullSpeaker, build_speaker
from .score import ScoreBoard


def _print_records(pipe: Pipeline, since: int) -> None:
    for record in pipe.records[since:]:
        print(f"    · {record['kind']}/{record['phase']} text={record['text']!r} "
              f"intent={record['intent']} score={record['score']:.3f} "
              f"{'接受' if record['accepted'] else '拒绝(' + record['note'] + ')'}")


def cmd_devices(args) -> int:
    try:
        devices = list_input_devices()
    except Exception as exc:
        print(f"无法枚举音频设备：{exc}")
        return 1
    if not devices:
        print("没有找到输入设备")
        return 1
    print("可用输入设备：")
    for dev in devices:
        print(f"  [{dev.index}] {dev.name}  ({dev.channels}ch, "
              f"{dev.default_samplerate:.0f}Hz, {dev.host_api})")
    return 0


def cmd_selftest(cfg, args) -> int:
    print(f"配置文件：{cfg.source_path}")
    print(f"采样率：{cfg.sample_rate} Hz，帧长：{cfg.block_ms} ms（{cfg.block_size} 样本）")
    print(f"唤醒词：{cfg.wake_words}（模式 {cfg.wake_mode}，阈值 {cfg.wake_threshold}）")
    print(f"口令：{len(cfg.all_command_phrases())} 条，"
          f"模式 {cfg.command_mode}，阈值 {cfg.match_threshold}，歧义间隔 {cfg.match_margin}")
    for intent, phrase in cfg.all_command_phrases():
        print(f"    {intent:<12} {phrase}")
    print(f"模型目录：{cfg.model_path} -> "
          f"{'存在' if os.path.isdir(cfg.model_path) else '缺失（先跑 tools/download_model.py）'}")
    try:
        engine = build_engine(cfg)
        engine.start()
        print(f"引擎：{engine.name}，"
              f"{'加载成功' if engine.model is not None else '未加载'}")
        engine.close()
    except Exception as exc:
        print(f"引擎加载失败：{exc}")
        return 1
    return 0


def _run_mic(cfg, args) -> int:
    if getattr(args, "no_reask", False):
        cfg.reask_on_reject = False
    engine = build_engine(cfg)
    pipe = Pipeline(cfg, engine=engine,
                    speaker=build_speaker(say=args.say, beep=not args.no_beep),
                    start_awake=False)
    if args.fresh:
        pipe.board.engine.reset_now()
    elif pipe.board.load():
        print(f"[恢复] 上次比分 {pipe.board.score_text()}")
    pipe.start()
    ui_server = None
    if args.ui:
        try:
            from tt_referee.ui.server import ScoreServer
            width, height = (int(v) for v in str(args.ui_window).lower().split("x"))
            ui_server = ScoreServer(pipe.board.engine, port=args.ui_port,
                                    allow_demo=False,
                                    players={"A": "选手 A", "B": "选手 B"},
                                    title="乒乓球智能裁判")
            ui_server.start(open_browser=True, window_size=(width, height), compact=True)
            print(f"[UI] 计分界面：{ui_server.url}（{width}x{height}，实时只读）")
        except Exception as exc:
            print(f"[UI] 启动失败：{exc}")
            ui_server = None
    print(f"开始监听（唤醒词：{'、'.join(cfg.wake_words)}），Ctrl+C 退出……")
    try:
        with MicSource(cfg.sample_rate, cfg.block_size, args.device) as source:
            for block in source:
                since = len(pipe.records)
                for result in pipe.feed(block):
                    if result.ok and result.state:
                        print(f"\n  ===== 当前比分  A {result.state['left']} : "
                              f"{result.state['right']} B =====\n")
                if args.verbose:
                    _print_records(pipe, since)
    except KeyboardInterrupt:
        print("\n收到中断，退出")
    except Exception as exc:
        print(f"麦克风运行失败：{exc}")
        return 1
    finally:
        if ui_server is not None:
            ui_server.stop()
        path = pipe.board.save()
        print(f"比分已保存到 {path}")
        print(f"统计：{json.dumps(pipe.summary(), ensure_ascii=False)}")
        pipe.close()
    return 0


def _run_wav(cfg, args) -> int:
    engine = build_engine(cfg)
    engine.start()
    step = cfg.block_ms / 1000.0
    exit_code = 0
    try:
        for path in args.files:
            if not os.path.exists(path):
                print(f"找不到文件：{path}")
                exit_code = 1
                continue
            engine.reset()
            pipe = Pipeline(cfg, engine=engine, board=ScoreBoard(cfg),
                            speaker=build_speaker(say=False, beep=False),
                            start_awake=args.start_awake)
            pipe.start()
            now = 0.0
            for block in WavSource(path, cfg.sample_rate, cfg.block_size):
                now += step
                pipe.feed(block, now=now)
            pipe.flush(now)
            print(f"\n[{os.path.basename(path)}]")
            _print_records(pipe, 0)
            if not pipe.records:
                print("  -> 没有识别到唤醒/口令（符合预期的干扰句）")
            else:
                last = pipe.records[-1]
                print(f"  -> 最终：{last['intent'] or '（无动作）'}；"
                      f"比分 {pipe.board.score_text()}")
    finally:
        engine.close()
    return exit_code


def _run_text(cfg, args) -> int:
    pipe = Pipeline(cfg, engine=None, speaker=build_speaker(say=False, beep=False),
                    start_awake=args.start_awake)
    base = time.monotonic()
    for index, text in enumerate(args.phrases):
        pipe.push_text(text, now=base + index * 0.5)
    print("\n识别轨迹：")
    for record in pipe.records:
        verdict = "接受" if record["accepted"] else f"拒绝({record['note']})"
        print(f"  text={record['text']!r} -> intent={record['intent']} "
              f"score={record['score']:.3f} runner_up={record['runner_up']} "
              f"{record['runner_up_score']:.3f} {verdict}")
    print(f"\n最终比分：{pipe.board.score_text()}")
    print(f"状态：{json.dumps(pipe.board.snapshot(), ensure_ascii=False)}")
    return 0


def _run_eval(cfg, args) -> int:
    samples = load_samples(args.directory, args.labels)
    if not samples:
        print(f"{args.directory} 下没有可用的样本（检查 labels.json）")
        return 1
    print(f"加载 {len(samples)} 个样本……")
    engine = build_engine(cfg)
    engine.start()
    try:
        results = run_samples(cfg, engine, samples)
        summary = summarize(results)
        print()
        print(format_summary(summary))
        if args.out_csv:
            csv_path = write_csv(results, args.out_csv)
        else:
            tag = (os.path.splitext(os.path.basename(cfg.resolve(cfg.dtw_model)))[0]
                   if cfg.engine == "dtw" else "vosk")
            csv_path = write_csv(results,
                                 os.path.join(args.directory, f"eval_report_{cfg.engine}_{tag}.csv"))
        print(f"逐条明细：{csv_path}")
        if args.sweep:
            print()
            sweep_thresholds(cfg, engine, samples, args.thresholds, args.margins)
    finally:
        engine.close()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tt_voice", description="离线语音唤醒 + 固定口令识别")
    parser.add_argument("--config", default=None, help="配置文件路径（默认用工程根目录的 config.json）")
    parser.add_argument("--engine", default=None, choices=["dtw", "vosk"],
                        help="覆盖配置里的识别引擎")
    parser.add_argument("--dtw-model", default=None, help="覆盖 DTW 模板库路径")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="列出输入设备").set_defaults(func=cmd_devices)
    sub.add_parser("selftest", help="检查配置和模型").set_defaults(func=cmd_selftest)

    mic = sub.add_parser("mic", help="实时麦克风")
    mic.add_argument("--device", type=int, default=None, help="输入设备编号")
    mic.add_argument("--say", action="store_true", help="用中文 TTS 播报（Windows SAPI）")
    mic.add_argument("--no-beep", action="store_true", help="关闭提示音")
    mic.add_argument("--verbose", action="store_true", help="打印每次识别结果")
    mic.add_argument("--fresh", action="store_true", help="忽略上次比分，从 0:0 开始（演示用）")
    mic.add_argument("--no-reask", action="store_true",
                     help="听不懂时不播报，静默退出（演示'非口令无反馈'用）")
    mic.add_argument("--ui", action="store_true", help="启动自适应屏幕计分 UI")
    mic.add_argument("--ui-port", type=int, default=8765)
    mic.add_argument("--ui-window", default="1024x600",
                     help="计分 UI 窗口大小，如 1024x600 / 800x480")
    _add_engine_overrides(mic)
    mic.set_defaults(func=_run_mic)

    wav = sub.add_parser("wav", help="离线回放 wav")
    wav.add_argument("files", nargs="+")
    wav.add_argument("--start-awake", action="store_true", help="跳过唤醒，直接口令模式")
    _add_engine_overrides(wav)
    wav.set_defaults(func=_run_wav)

    text = sub.add_parser("text", help="纯文本跑状态机")
    text.add_argument("phrases", nargs="+")
    text.add_argument("--start-awake", action="store_true", help="跳过唤醒，直接口令模式")
    text.set_defaults(func=_run_text)

    ev = sub.add_parser("eval", help="跑测试集")
    ev.add_argument("directory")
    ev.add_argument("--labels", default="labels.json")
    ev.add_argument("--sweep", action="store_true", help="网格扫描匹配阈值")
    ev.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.70, 0.75, 0.80, 0.85, 0.90])
    ev.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.02, 0.04, 0.06, 0.08])
    ev.add_argument("--out-csv", default=None, help="逐条明细输出路径")
    _add_engine_overrides(ev)
    ev.set_defaults(func=_run_eval)

    return parser


def _add_engine_overrides(sub: argparse.ArgumentParser) -> None:
    """子命令上也允许写 --engine/--dtw-model（默认用 SUPPRESS 以免覆盖全局值）。"""
    sub.add_argument("--engine", default=argparse.SUPPRESS, choices=["dtw", "vosk"])
    sub.add_argument("--dtw-model", default=argparse.SUPPRESS, help="覆盖 DTW 模板库路径")


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 GBK，中文播报会乱码，这里统一按 UTF-8 输出
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_arg_parser().parse_args(argv)
    if args.command == "devices":
        return cmd_devices(args)
    cfg = load_config(args.config)
    engine = getattr(args, "engine", None)
    dtw_model = getattr(args, "dtw_model", None)
    if engine:
        cfg.engine = engine
    if dtw_model:
        cfg.dtw_model = dtw_model
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
