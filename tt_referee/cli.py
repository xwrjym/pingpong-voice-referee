"""命令行入口：合成演示、离线事件回放、标定文件、轨迹检查。"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .calibration import Calibration
from .events import EventConfig
from .inspect import ascii_timeline
from .pipeline import PipelineConfig, RefereePipeline
from .replay import extract_events, load_observations, save_events
from .buttons import ButtonAdapter
from .scoring import ScoreEngine, ScoringRules, replay_actions
from .simulate import SCENARIOS, build_scenario, demo_homography


def _calibration(args) -> Calibration | None:
    if args.calib and os.path.exists(args.calib):
        return Calibration.load(args.calib)
    return None


def run_scenario(name: str, verbose: bool = False, as_json: bool = False) -> dict:
    sources = build_scenario(name)
    pipeline = RefereePipeline(PipelineConfig(), ScoringRules(),
                              homography=demo_homography(), log_events=False)
    for source in sources:
        pipeline.register_source(source)
    all_events = []
    last_outcome = None
    latest = 0.0
    for obs in sorted((o for stream in sources.values() for o in stream),
                      key=lambda o: (o.timestamp_s, o.source)):
        result = pipeline.feed(obs)
        latest = max(latest, obs.timestamp_s)
        if result.outcome is not None:
            last_outcome = result.outcome
        if result.events:
            all_events.extend(result.events)
            if verbose and not as_json:
                for event in result.events:
                    print(f"  [{event.timestamp_s:6.3f}s] {event.type:<8} "
                          f"side={event.side:<7} conf={event.confidence:.2f} "
                          f"sources={','.join(event.sources)}")
        if result.action is not None and not as_json:
            if result.outcome is not None and result.outcome.let:
                verdict = "重发"
            else:
                verdict = "判定" if result.action.ok else "转人工"
            side = result.outcome.side if result.outcome else "?"
            print(f"  -> {verdict}：{side} / {result.outcome.reason}，"
                  f"比分 {result.action.state['points_text']}，状态 {result.action.state['state']}")
    flushed = pipeline.flush(latest + 5.0)
    all_events.extend(flushed.events)
    if flushed.outcome is not None:
        last_outcome = flushed.outcome
    if flushed.action is not None and not as_json:
        print(f"  -> 转人工：{flushed.outcome.side} / {flushed.outcome.reason}，"
              f"状态 {flushed.state['state']}")
    if not as_json:
        print(f"  == 结果：比分 {flushed.state['points_text']}，"
              f"状态 {flushed.state['state']}，"
              f"判定 {last_outcome.reason if last_outcome else '无'}")
    summary = {"scenario": name, "events": [e.to_dict() for e in all_events],
               "outcome": last_outcome.to_dict() if last_outcome else None,
               "state": flushed.state}
    if as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def cmd_demo(args) -> int:
    names = sorted(SCENARIOS) + ["two_sources", "conflict"] \
        if args.scenario == "all" else [args.scenario]
    for name in names:
        if not args.json:
            print(f"\n=== {name} ===")
        run_scenario(name, verbose=args.verbose, as_json=args.json)
    return 0


def cmd_simulate(args) -> int:
    sources = build_scenario(args.scenario)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for stream in sources.values():
            for obs in stream:
                fh.write(json.dumps(obs.to_dict(), ensure_ascii=False) + "\n")
    total = sum(len(s) for s in sources.values())
    print(f"已写出 {total} 条观测（{len(sources)} 个来源）到 {args.out}")
    return 0


def cmd_events(args) -> int:
    observations = load_observations(args.input, video_fps=args.fps)
    calib = _calibration(args)
    homography = calib.homography() if calib else demo_homography()
    events, summary = extract_events(observations, homography, EventConfig())
    if args.out:
        save_events(events, args.out)
        print(f"事件已写入 {args.out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for event in events:
        print(f"  [{event.timestamp_s:6.3f}s] {event.type:<8} side={event.side:<7} "
              f"conf={event.confidence:.2f} sources={','.join(event.sources)}")
    return 0


def cmd_inspect(args) -> int:
    observations = load_observations(args.input, video_fps=args.fps)
    calib = _calibration(args)
    homography = calib.homography() if calib else demo_homography()
    events, summary = extract_events(observations, homography, EventConfig())
    print(ascii_timeline(observations, events))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_calibrate(args) -> int:
    points = []
    for item in args.points.replace(";", " ").split():
        x, y = item.split(",")
        points.append((float(x), float(y)))
    calib = Calibration(image_points=points, image_size=(args.width, args.height),
                        name=args.name)
    warnings = calib.validate()
    print(f"球台四点：{calib.image_points}")
    print("侧别约定：A=球台左半台，B=球台右半台（方案 §7）")
    for warning in warnings:
        print(f"  [警告] {warning}")
    if args.save:
        calib.save(args.save)
        print(f"已保存标定文件：{args.save}")
    center = calib.homography().table_to_image((1.37, 0.7625))
    print(f"球台中心投影像素：({center[0]:.1f}, {center[1]:.1f})")
    return 0


def cmd_score(args) -> int:
    rules = ScoringRules.from_json(args.rules) if args.rules else ScoringRules()
    engine = ScoreEngine(rules, log_events=False)
    buttons = ButtonAdapter(engine)

    def show(tag: str, result) -> None:
        mark = "OK " if result.ok else "REJ"
        print(f"  {mark} {tag:<26} {engine.display()}")

    if args.demo:
        show("视觉：A 得分", engine.finish_rally("A", "bounce", 0.95, source="vision"))
        show("语音：B 加一分", engine.apply_voice_intent("score_right"))
        show("按键：A 得分", buttons.press("point_a"))
        show("视觉：低置信待确认", engine.finish_rally("B", "close_bounce", 0.35, source="vision"))
        show("语音：确认这一分", engine.apply_voice_intent("confirm"))
        show("语音：撤销", engine.apply_voice_intent("undo"))
        show("按键：重做", buttons.press("redo"))
        show("语音：重置（第一步）", engine.apply_voice_intent("reset"))
        show("语音：确认重置", engine.apply_voice_intent("confirm"))
        print(f"\n  来源统计：{engine.stats_by_source}")
        print(f"  审计动作：{len(engine.actions)} 条")
        rebuilt = replay_actions(engine.actions, rules)
        print(f"  回放校验：{'一致' if rebuilt.points == engine.points and rebuilt.sets == engine.sets else '不一致'}")
    else:
        if not args.actions:
            print("需要 --demo 或 --actions <动作 JSONL>")
            return 1
        with open(args.actions, "r", encoding="utf-8") as fh:
            actions = [json.loads(line) for line in fh if line.strip()]
        for item in actions:
            kind = item.get("action")
            if kind == "award":
                show(f"{item.get('source', '?')}：{item['side']} 得分",
                     engine.award(item["side"], item.get("reason", ""),
                                  float(item.get("confidence", 1.0)),
                                  item.get("source", "replay")))
            elif kind == "undo":
                show("撤销", engine.undo())
            elif kind == "redo":
                show("重做", engine.redo())
            elif kind == "reset":
                show("重置", engine.reset_now())
            elif kind == "reject_pending":
                show("取消待确认", engine.reject_pending())
        rebuilt = replay_actions(actions, rules)
        print(f"\n  回放校验：{'一致' if rebuilt.points == engine.points else '不一致'}")
    if args.state_out:
        engine.save(args.state_out)
        print(f"  状态已保存：{args.state_out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(prog="tt_referee", description="乒乓球裁判核心")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="跑合成场景（含计分）")
    demo.add_argument("--scenario", default="all")
    demo.add_argument("--verbose", action="store_true")
    demo.add_argument("--json", action="store_true")
    demo.set_defaults(func=cmd_demo)

    sim = sub.add_parser("simulate", help="把合成观测导出成 JSONL")
    sim.add_argument("--scenario", default="double_bounce")
    sim.add_argument("--out", required=True)
    sim.set_defaults(func=cmd_simulate)

    events = sub.add_parser("events", help="离线回放观测，输出事件")
    events.add_argument("--input", required=True)
    events.add_argument("--calib", default=None)
    events.add_argument("--out", default=None)
    events.add_argument("--fps", type=float, default=None, help="统一帧号用的视频帧率")
    events.set_defaults(func=cmd_events)

    inspect = sub.add_parser("inspect", help="文本查看轨迹与事件")
    inspect.add_argument("--input", required=True)
    inspect.add_argument("--calib", default=None)
    inspect.add_argument("--fps", type=float, default=None, help="统一帧号用的视频帧率")
    inspect.set_defaults(func=cmd_inspect)

    calib = sub.add_parser("calibrate", help="生成四点标定文件")
    calib.add_argument("--points", required=True,
                       help="远左,远右,近右,近左 四个像素点，空格分隔，如 '320,330 1600,300 1580,860 360,880'")
    calib.add_argument("--width", type=int, default=1920)
    calib.add_argument("--height", type=int, default=1080)
    calib.add_argument("--name", default="")
    calib.add_argument("--save", default=None)
    calib.set_defaults(func=cmd_calibrate)

    score = sub.add_parser("score", help="计分状态机演示/审计回放")
    score.add_argument("--demo", action="store_true", help="跑一段视觉+语音+按键的示例")
    score.add_argument("--actions", default=None, help="动作 JSONL（审计日志）")
    score.add_argument("--rules", default=None, help="规则 JSON，例如 11 分/五局三胜")
    score.add_argument("--state-out", default=None)
    score.set_defaults(func=cmd_score)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
