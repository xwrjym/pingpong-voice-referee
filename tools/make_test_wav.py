"""用 Windows 自带中文 TTS 合成一套离线测试集（不需要麦克风）。

    python tools/make_test_wav.py --out testdata

生成内容：

* wake__*  只有唤醒词（正常语速 / 快 / 慢）
* cmd__*   只有口令
* combo__* 唤醒词和口令连着说（验证"一口气说完"时不会丢口令）
* neg__*   无关语句 + 近似干扰句（验证误唤醒和误执行）

每条语音还会生成多种信道劣化版本：干净 / 加噪（20、10 dB）/ 混响 / 远场，
用来观察唤醒率和识别率随信噪比的变化，方便决定阈值。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tt_voice.augment import make_variant                 # noqa: E402
from tt_voice.audio_io import read_wav_mono16          # noqa: E402
from tt_voice.config import load_config                 # noqa: E402
from tt_voice.feedback import sapi_say_to_wav           # noqa: E402

WAKE = [
    ("xiao_caipan_1", "小裁判", 0),
    ("xiao_caipan_2", "小裁判", -2),
    ("xiao_caipan_3", "小裁判", 2),
]

COMBO = [
    ("left_0", "小裁判，左方加一分", "score_left", 0),
    ("undo_0", "小裁判，撤销", "undo", 0),
    ("right_0", "小裁判，右方加一分", "score_right", -2),
]

NEG = [
    ("chat_weather", "今天天气不错"),
    ("chat_light", "帮我把灯关掉"),
    ("chat_next", "我们再来一局吧"),
    ("near_miss_left", "左边加一份"),
    ("near_miss_right", "右方落后一分"),
    ("near_miss_undo", "这个不要了"),
]


def write_wav(path: str, data: np.ndarray, sr: int = 16000) -> None:
    import wave

    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(data.tobytes())


VARIANTS_FULL = ("clean", "snr20", "snr10", "reverb", "far")
VARIANTS_LIGHT = ("clean", "snr10")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="testdata")
    parser.add_argument("--config", default=None)
    parser.add_argument("--voice", default=None, help="指定 SAPI 音色名称，如 Microsoft Huihui")
    parser.add_argument("--no-augment", action="store_true", help="只生成干净音频")
    parser.add_argument("--augment-seed", type=int, default=777,
                        help="测试集加噪的噪声种子；必须和注册用的种子不同，否则等于泄题")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    rows: list[dict] = []

    def synthesize(text: str, rate: int) -> np.ndarray:
        tmp = os.path.join(out_dir, "_tmp_synth.wav")
        sapi_say_to_wav(text, tmp, rate=rate, voice=args.voice)
        data, sr = read_wav_mono16(tmp)
        os.remove(tmp)
        return data

    def emit(kind: str, name: str, text: str, data: np.ndarray, note: str,
             variants: tuple[str, ...], intent: str = "") -> None:
        for variant in (("clean",) if args.no_augment else variants):
            audio = make_variant(data, cfg.sample_rate, variant, seed=args.augment_seed)
            filename = f"{kind}__{name}__{variant}.wav"
            write_wav(os.path.join(out_dir, filename), audio, cfg.sample_rate)
            rows.append({"file": filename, "kind": kind, "text": text,
                         "intent": intent, "note": variant})
            print(f"  {filename}")

    print("合成唤醒词……")
    for name, text, rate in WAKE:
        emit("wake", name, text, synthesize(text, rate), f"rate{rate}", VARIANTS_FULL)

    print("合成口令……")
    for index, (intent, phrase) in enumerate(cfg.all_command_phrases(), start=1):
        name = f"{intent}_{index:02d}"
        emit("cmd", name, phrase, synthesize(phrase, 0), "rate0", VARIANTS_LIGHT, intent)

    print("合成'连着说'的样本……")
    for name, text, intent, rate in COMBO:
        emit("combo", name, text, synthesize(text, rate), f"rate{rate}", VARIANTS_LIGHT, intent)

    print("合成干扰样本……")
    for name, text in NEG:
        emit("neg", name, text, synthesize(text, 0), "rate0", VARIANTS_LIGHT)

    with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    print(f"\n共 {len(rows)} 个文件，标签写入 {os.path.join(out_dir, 'labels.json')}")
    print("下一步：python -m tt_voice.cli eval " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
