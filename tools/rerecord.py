"""定点补录：只重录验收集里指定的几条（按 labels.json 的文本提示）。

    python tools/rerecord.py --dir realdata4 --device 12 --files wake__near_2__live.wav wake__fast_1__live.wav

用于个别样本念太快、没说话或环境噪声太大的情况，避免整轮重录。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tt_voice.audio_io import MicSource, UtteranceRecorder, write_wav16  # noqa: E402
from tt_voice.config import load_config                                  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="验收集目录，例如 realdata4")
    parser.add_argument("--files", nargs="+", required=True, help="要重录的 wav 文件名")
    parser.add_argument("--device", type=int, default=None, help="输入设备编号")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-record", type=float, default=4.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    folder = os.path.abspath(args.dir)
    with open(os.path.join(folder, "labels.json"), "r", encoding="utf-8") as fh:
        labels = {row["file"]: row for row in json.load(fh)}

    missing = [name for name in args.files if name not in labels]
    if missing:
        print(f"labels.json 里找不到：{missing}")
        return 1

    print(f"补录 {len(args.files)} 条到 {folder}")
    print("每条：按回车 → 看到「→ 现在说」→ 念 → 停一下\n")
    with MicSource(cfg.sample_rate, cfg.block_size, args.device) as source:
        recorder = UtteranceRecorder(iter(source), cfg.sample_rate, cfg.block_size,
                                     start_db=cfg.vad_start_db, margin_db=cfg.vad_margin_db)
        try:
            for position, name in enumerate(args.files, start=1):
                row = labels[name]
                text = row.get("text") or row.get("note") or "（按标签要求）"
                input(f"[{position}/{len(args.files)}] 请说：{text} ")
                print("      → 现在说")
                audio = recorder.record(max_seconds=args.max_record)
                write_wav16(os.path.join(folder, name), audio, cfg.sample_rate)
                print(f"      已覆盖 {name}（{audio.size / cfg.sample_rate:.2f}s）")
        except KeyboardInterrupt:
            print("\n已中断")
    print("\n补录完成，可以重新评测：")
    print(f"  .venv\\Scripts\\python.exe -m tt_voice.cli eval {args.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
