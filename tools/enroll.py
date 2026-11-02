"""注册 DTW 模板库（离线、零模型依赖）。

两种注册方式：

    # 1) 用本人录音注册（正式使用要这么做，换人/换麦克风都要重录）
    python tools/enroll.py --mic --repeats 3

    # 2) 用 Windows 中文 TTS 造模板（没有麦克风也能先把链路跑通，用于自测）
    python tools/enroll.py --tts "Microsoft Huihui" --tts-rates -3 3

同一条短语录多遍会被用来标定判据阈值（类内 DTW 距离 + 余量），
所以每种说法至少给 2 遍，3 遍更稳。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tt_voice.augment import add_noise                            # noqa: E402
from tt_voice.audio_io import MicSource, UtteranceRecorder, read_wav_mono16  # noqa: E402
from tt_voice.config import load_config                           # noqa: E402
from tt_voice.features import FeatureExtractor                    # noqa: E402
from tt_voice.feedback import build_speaker, sapi_run, sapi_say_to_wav  # noqa: E402
from tt_voice.templates import KIND_COMMAND, KIND_WAKE, Template, TemplateStore  # noqa: E402
from tt_voice.vad import EnergyVad                                # noqa: E402


def list_zh_voices() -> list[str]:
    script = ("Add-Type -AssemblyName System.Speech; "
              "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices()"
              " | ForEach-Object { $_.VoiceInfo.Name + '|' + $_.VoiceInfo.Culture.Name }")
    result = sapi_run(script)
    out = []
    for line in result.stdout.decode("gbk", errors="replace").splitlines():
        line = line.strip()
        if "|" in line and line.split("|")[1].startswith("zh"):
            out.append(line.split("|")[0])
    return out


def tts_samples(text: str, voices: list[str], rates: list[int], tmp_dir: str) -> list[np.ndarray]:
    samples = []
    tmp = os.path.join(tmp_dir, "_enroll_tmp.wav")
    for voice in voices:
        for rate in rates:
            sapi_say_to_wav(text, tmp, rate=rate, voice=voice)
            data, _ = read_wav_mono16(tmp)
            samples.append(data)
    if os.path.exists(tmp):
        os.remove(tmp)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None, help="输出 npz（默认用配置里的 dtw_model）")
    parser.add_argument("--mic", action="store_true", help="用麦克风录音注册")
    parser.add_argument("--tts", default=None, help="用 TTS 注册，逗号分隔的音色名")
    parser.add_argument("--tts-rates", type=int, nargs="+", default=[0], help="TTS 语速变体")
    parser.add_argument("--list-voices", action="store_true", help="列出可用的中文 TTS 音色")
    parser.add_argument("--repeats", type=int, default=3, help="每条短语录几遍")
    parser.add_argument("--max-record", type=float, default=3.0, help="单次录音最长秒数")
    parser.add_argument("--device", type=int, default=None, help="输入设备编号")
    parser.add_argument("--only", choices=["all", "wake", "command"], default="all")
    parser.add_argument("--margin", type=float, default=1.3, help="阈值余量（类内最大距离×余量）")
    parser.add_argument("--pad-ms", type=int, default=120, help="TTS 音频前后补的静音")
    parser.add_argument("--augment-snr", type=float, nargs="*", default=[15.0, 10.0],
                        help="注册时额外加入的加噪副本信噪比（dB），不想要就传空")
    parser.add_argument("--augment-seed", type=int, default=101,
                        help="注册增广用的噪声种子；必须和造测试集的种子不同，否则等于泄题")
    args = parser.parse_args()

    if args.list_voices:
        for name in list_zh_voices():
            print(" ", name)
        return 0

    cfg = load_config(args.config)
    out_path = os.path.abspath(args.out or cfg.resolve(cfg.dtw_model))
    tmp_dir = os.path.dirname(out_path)
    os.makedirs(tmp_dir, exist_ok=True)

    items: list[tuple[str, str, str]] = []       # (kind, phrase, intent)
    if args.only in ("all", "wake"):
        items += [(KIND_WAKE, word, "wake") for word in cfg.wake_words]
    if args.only in ("all", "command"):
        items += [(KIND_COMMAND, phrase, intent) for intent, phrase in cfg.all_command_phrases()]

    extractor = FeatureExtractor(cfg.sample_rate)
    store = TemplateStore(sample_rate=cfg.sample_rate, feature_dim=extractor.dim,
                          metric=cfg.dtw_metric)

    voices = [v.strip() for v in args.tts.split(",")] if args.tts else []
    if args.mic and voices:
        print("--mic 和 --tts 只能选一个")
        return 1
    use_tts = bool(voices)
    if not use_tts and not args.mic:
        print("请指定 --mic 或 --tts；先看可用音色可以加 --list-voices")
        return 1
    if use_tts:
        print(f"用 TTS 注册：{voices}，语速变体 {args.tts_rates}")
    else:
        print(f"用麦克风注册：每条短语录 {args.repeats} 遍，"
              f"每次最长 {args.max_record:.1f}s（说话停顿 0.7s 自动结束）")

    pad = np.zeros(int(cfg.sample_rate * args.pad_ms / 1000), dtype=np.int16)
    speaker = build_speaker(say=False, beep=True)
    mic = MicSource(cfg.sample_rate, cfg.block_size, args.device) if not use_tts else None
    recorder = None
    if mic is not None:
        # 流只开一次：蓝牙耳机每次开流要 0.5~1.5 秒，逐条开关会把起音吃掉
        mic.__enter__()
        recorder = UtteranceRecorder(iter(mic), cfg.sample_rate, cfg.block_size,
                                     start_db=cfg.vad_start_db, margin_db=cfg.vad_margin_db)
    for kind, phrase, intent in items:
        samples: list[np.ndarray] = []
        if use_tts:
            samples = tts_samples(phrase, voices, args.tts_rates, tmp_dir)
        else:
            for take in range(1, args.repeats + 1):
                input(f"  [{take}/{args.repeats}] 按回车，听到提示音后朗读：{phrase} ")
                speaker.beep("ready")
                audio = recorder.record(max_seconds=args.max_record)
                samples.append(audio)
                print(f"      录到 {audio.size / cfg.sample_rate:.2f}s")
        for sample in samples:
            padded = np.concatenate([pad, sample, pad])
            variants: list[tuple[str, np.ndarray]] = [("clean", padded)]
            for snr in args.augment_snr:
                # 加噪副本让"阈值"天然覆盖带噪工况；匹配时取最像的那条模板
                variants.append((f"snr{snr:g}",
                                 add_noise(padded, cfg.sample_rate, snr, seed=args.augment_seed)))
            for tag, audio in variants:
                # extract() 内部会先剪静音、再在留下的有声帧上做 CMN，这里不要再剪一次
                feats = extractor.extract(audio)
                if feats.shape[0] < 5:
                    print(f"      跳过：{phrase} {tag} 有效语音太短")
                    continue
                store.templates.append(Template(phrase=phrase, intent=intent,
                                                kind=kind, feats=feats))
        print(f"  {kind:<7} {phrase:<12} 得到 "
              f"{sum(1 for t in store.templates if t.phrase == phrase)} 条模板")

    speaker.close()
    if mic is not None:
        mic.__exit__(None, None, None)

    if not store.templates:
        print("没有生成任何模板")
        return 1

    print("标定阈值……")
    store.calibrate(margin=args.margin, log=print)
    path = store.save(out_path)
    print(f"\n模板库已保存：{path}")
    print(store.describe())
    missing = [t.phrase for t in store.templates if t.threshold <= 0]
    if missing:
        print(f"注意：这些短语只有 1 条模板，用的会是配置里的默认阈值：{sorted(set(missing))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
