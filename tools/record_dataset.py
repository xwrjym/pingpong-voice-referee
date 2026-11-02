"""用真人录音造一份可验收的测试集（在能用的那台电脑上跑）。

    python tools/record_dataset.py --out realdata --list     # 先看录制计划
    python tools/record_dataset.py --out realdata            # 开始录
    python -m tt_voice.cli eval realdata                     # 出唤醒率/误唤醒率/准确率

按验收口径分四类录：

* wake ：只喊唤醒词（近/远/快三种）
* cmd  ：只念口令（配置里所有说法）
* combo：唤醒词和口令连着说
* neg  ：干扰——正常聊天、喊话、近似句、纯环境音

生成的 wav + labels.json 与 `tools/make_test_wav.py` 格式一致，
所以 `eval` 一字不改就能跑，指标口径和合成集结果可直接对照。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tt_voice.audio_io import MicSource, UtteranceRecorder, write_wav16  # noqa: E402
from tt_voice.config import load_config                                  # noqa: E402
from tt_voice.feedback import build_speaker                              # noqa: E402

WAKE_PROMPTS = [
    ("near", "小裁判", "站在麦克风前 0.5 米，正常音量"),
    ("far", "小裁判", "退到 1.5~2 米，正常音量"),
    ("fast", "小裁判", "站原位，语速快一点"),
]

NEG_PROMPTS = [
    ("chat1", "（和对手正常聊天两句话，不要喊）"),
    ("chat2", "（再说两句无关的话，正常音量）"),
    ("shout", "（对着球台喊一声「好球」）"),
    ("near_miss1", "左边加一份"),
    ("near_miss2", "这球到底出没出"),
    ("noise", "（在击球声中什么都不说，录几秒环境音）"),
]

COMBO_PROMPTS = [
    ("left", "小裁判，左方加一分", "score_left"),
    ("self", "小裁判，我加一分", "score_self"),
    ("undo", "小裁判，撤销", "undo"),
]


@dataclass
class Item:
    kind: str
    name: str
    text: str
    note: str
    intent: str = ""


def build_plan(cfg, wake_repeats: int) -> list[Item]:
    plan: list[Item] = []
    for name, text, note in WAKE_PROMPTS:
        for index in range(wake_repeats):
            plan.append(Item("wake", f"{name}_{index + 1}", text, note))
    for index, (intent, phrase) in enumerate(cfg.all_command_phrases(), start=1):
        plan.append(Item("cmd", f"{intent}_{index:02d}", phrase, "正常音量", intent))
    for name, text, intent in COMBO_PROMPTS:
        plan.append(Item("combo", name, text, "一口气说完", intent))
    for name, prompt in NEG_PROMPTS:
        plan.append(Item("neg", name, prompt, "干扰"))
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="realdata")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", type=int, default=None, help="输入设备编号（见 cli devices）")
    parser.add_argument("--speaker", default="", help="说话人标记，写进 labels.json")
    parser.add_argument("--max-record", type=float, default=4.0, help="单次最长录音秒数")
    parser.add_argument("--wake-repeats", type=int, default=1, help="每类唤醒样本录几遍")
    parser.add_argument("--limit", type=int, default=0, help="只录计划里的前 N 条（快速自检用）")
    parser.add_argument("--beep", action="store_true",
                        help="开口前播提示音（仅蓝牙链路需要；手机/回环麦克风会被它误触发）")
    parser.add_argument("--list", action="store_true", help="只看录制计划")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = os.path.abspath(args.out)
    plan = build_plan(cfg, args.wake_repeats)
    if args.limit > 0:
        plan = plan[:args.limit]

    if args.list:
        print(f"计划录制 {len(plan)} 条到 {out_dir}：")
        for item in plan:
            print(f"  [{item.kind:<5}] {item.name:<18} {item.text}   （{item.note}）")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    print(f"共 {len(plan)} 条，录到 {out_dir}")
    print("每条：按回车 → 听到提示音后再说话（等提示音是为了唤醒蓝牙/AGC 链路），"
          "说完停顿约 0.7 秒自动结束\n")
    rows: list[dict] = []
    speaker = build_speaker(say=False, beep=True)
    # 流只开一次：蓝牙耳机每次开流要 0.5~1.5 秒，逐条开关会把起音吃掉
    with MicSource(cfg.sample_rate, cfg.block_size, args.device) as source:
        recorder = UtteranceRecorder(iter(source), cfg.sample_rate, cfg.block_size,
                                     start_db=cfg.vad_start_db, margin_db=cfg.vad_margin_db)
        try:
            for position, item in enumerate(plan, start=1):
                input(f"[{position}/{len(plan)}] {item.kind:<5} 请说/执行：{item.text} ")
                if args.beep:
                    speaker.beep("ready")
                print("      → 现在说")
                audio = recorder.record(max_seconds=args.max_record)
                seconds = audio.size / cfg.sample_rate
                filename = f"{item.kind}__{item.name}__live.wav"
                write_wav16(os.path.join(out_dir, filename), audio, cfg.sample_rate)
                row: dict = {"file": filename, "kind": item.kind, "note": item.note,
                             "speaker": args.speaker, "seconds": round(seconds, 2)}
                if item.kind != "neg":
                    row["text"] = item.text
                if item.intent:
                    row["intent"] = item.intent
                rows.append(row)
                print(f"      录到 {seconds:.2f}s")
        except KeyboardInterrupt:
            print("\n已中断，保存已录部分")
        finally:
            speaker.close()

    labels = os.path.join(out_dir, "labels.json")
    with open(labels, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    print(f"\n共 {len(rows)} 条，标签写入 {labels}")
    print(f"下一步：python -m tt_voice.cli eval {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
