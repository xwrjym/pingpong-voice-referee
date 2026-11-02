"""麦克风链路体检：量采集质量 + 当场试识别，用来在两副耳机之间做 A/B 对比。

    python tools/check_mic.py --device 28
    python tools/check_mic.py --device 28 --phrases 撤销上一分 刚才不算 报分 我加一分

每条会告诉你四件事：

1. 起音有没有被吞：说话开头 60ms 有没有能量（被蓝牙噪声门削掉的典型表现是开头一片数字零）；
2. 电平够不够：峰值 / RMS（太低的录音在识别前就已经缺信息）；
3. 带宽够不够：4kHz 以上能量占比。蓝牙通话链路（CVSD/8kHz）会把高频砍掉，
   而"撤/出/报/错"这些字恰好靠高频区分，这是短口令最容易糊的原因；
4. 识别结果：解码文本、置信度、是否被正确解析成口令。

换耳机/换位置/改完设置后再跑一次，直接对比这三列数字。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tt_voice.audio_io import MicSource, UtteranceRecorder         # noqa: E402
from tt_voice.config import load_config                            # noqa: E402
from tt_voice.engines.vosk_engine import VoskEngine                # noqa: E402
from tt_voice.feedback import build_speaker                        # noqa: E402
from tt_voice.parser import Parser                                 # noqa: E402

DEFAULT_PHRASES = [
    "小裁判", "我加一分", "撤销上一分", "现在比分", "刚才不算", "撤销", "报分", "没错",
]


def frame_db(x: np.ndarray, sr: int, frame_ms: float = 10.0) -> np.ndarray:
    frame = max(1, int(sr * frame_ms / 1000))
    n = x.size // frame
    if n == 0:
        return np.zeros(0)
    frames = x[:n * frame].astype(np.float32).reshape(n, frame) / 32768.0
    return 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12) + 1e-9)


def analyse(audio: np.ndarray, sr: int) -> dict:
    peak = 20 * np.log10(max(float(np.max(np.abs(audio))) / 32768.0, 1e-9))
    rms = 20 * np.log10(float(np.sqrt(np.mean((audio / 32768.0) ** 2))) + 1e-9)
    energy = frame_db(audio, sr)
    voiced = np.where(energy > -45)[0] if energy.size else np.zeros(0, dtype=int)
    onset = int(voiced[0]) * 10 if voiced.size else 0
    onset_energy = float(np.max(energy[:6])) if energy.size >= 6 else -120.0
    # 4kHz 以上能量占比：蓝牙通话链路砍高频，这里会明显偏低
    spec = np.abs(np.fft.rfft(audio.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(audio.size, 1 / sr)
    speech = spec[freqs > 150]
    high = spec[freqs > 4000].sum() / max(speech.sum(), 1e-9)
    return {"peak": peak, "rms": rms, "onset_ms": onset, "onset_energy": onset_energy,
            "high_ratio": high, "seconds": audio.size / sr}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", type=int, default=None, help="输入设备编号（cli devices）")
    parser.add_argument("--phrases", nargs="+", default=DEFAULT_PHRASES)
    parser.add_argument("--max-record", type=float, default=4.0)
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="固定时长录音模式：录 N 秒（适合远控/立体声混音这类链路）")
    parser.add_argument("--beep", action="store_true",
                        help="开口前播提示音（仅蓝牙链路需要；回环/虚拟麦克风会被它误触发）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    engine = VoskEngine(cfg, log=lambda *_: None)
    engine.start()
    text_parser = Parser(cfg)
    speaker = build_speaker(say=False, beep=True)

    print("== 麦克风链路体检 ==")
    print("每条：按回车 → 听到提示音后再说话（这条提示音同时用来唤醒蓝牙链路）\n")
    results = []
    with MicSource(cfg.sample_rate, cfg.block_size, args.device) as source:
        if args.seconds > 0:
            input(f"按回车开始，然后对着手机正常说话（录 {args.seconds:.0f} 秒）")
            chunks = []
            need = int(args.seconds * cfg.sample_rate / cfg.block_size) + 1
            for index, block in enumerate(source):
                chunks.append(block)
                if index + 1 >= need:
                    break
            audio = np.frombuffer(b"".join(chunks), dtype="<i2")
            info = analyse(audio, cfg.sample_rate)
            engine.reset()
            engine.set_mode("command")
            hits = []
            step = cfg.block_size
            for i in range(0, audio.size, step):
                block = audio[i:i + step]
                if block.size < step:
                    block = np.pad(block, (0, step - block.size))
                hits += engine.feed(block.tobytes())
            hits += engine.flush()
            text = " ".join(h.text for h in hits)
            print(f"    录制 {info['seconds']:.2f}s | 峰值 {info['peak']:.1f} dBFS | "
                  f"RMS {info['rms']:.1f} dBFS | 起音前静音 {info['onset_ms']}ms")
            print(f"    4kHz 以上能量占比 {info['high_ratio'] * 100:.1f}%")
            print(f"    整段解码：{text!r}")
            speaker.close()
            engine.close()
            return 0
        recorder = UtteranceRecorder(iter(source), cfg.sample_rate, cfg.block_size,
                                     start_db=cfg.vad_start_db, margin_db=cfg.vad_margin_db)
        try:
            for index, phrase in enumerate(args.phrases, start=1):
                input(f"[{index}/{len(args.phrases)}] 请说：{phrase} ")
                if args.beep:
                    speaker.beep("ready")
                audio = recorder.record(max_seconds=args.max_record)
                info = analyse(audio, cfg.sample_rate)

                engine.reset()
                engine.set_mode("wake" if phrase in cfg.wake_words else "command")
                hits = []
                step = cfg.block_size
                for i in range(0, audio.size, step):
                    block = audio[i:i + step]
                    if block.size < step:
                        block = np.pad(block, (0, step - block.size))
                    hits += engine.feed(block.tobytes())
                hits += engine.flush()
                text = " ".join(h.text for h in hits)
                conf = hits[0].confidence if hits else 0.0
                if phrase in cfg.wake_words:
                    match = text_parser.matches_wake(text)
                else:
                    match = text_parser.parse_command(text)
                verdict = "OK" if match.accepted else "拒识"

                onset_flag = "正常" if info["onset_energy"] > -45 else "**起音偏弱/被削**"
                results.append((phrase, info, text, conf, match.intent, verdict))
                print(f"    录制 {info['seconds']:.2f}s | 峰值 {info['peak']:.1f} dBFS | "
                      f"RMS {info['rms']:.1f} dBFS | 起音前静音 {info['onset_ms']}ms | "
                      f"开头 60ms 电平 {info['onset_energy']:.1f} dB ({onset_flag})")
                print(f"    4kHz 以上能量占比 {info['high_ratio'] * 100:.1f}%"
                      f"{'（偏低，像是窄带通话链路）' if info['high_ratio'] < 0.02 else ''}")
                print(f"    解码 {text!r}（置信度 {conf:.2f}）→ 意图 {match.intent}，{verdict}"
                      f"（相似度 {match.score:.2f}）")
        except KeyboardInterrupt:
            print("\n已中断")
        finally:
            speaker.close()

    if results:
        ok = sum(1 for *_, verdict in results if verdict == "OK")
        weak = sum(1 for _, info, *_ in results if info["onset_energy"] <= -45)
        narrow = sum(1 for _, info, *_ in results if info["high_ratio"] < 0.02)
        print(f"\n== 汇总 ==\n  识别正确 {ok}/{len(results)}")
        print(f"  起音偏弱 {weak} 条；疑似窄带链路 {narrow} 条")
        print("  起音偏弱多 → 等提示音后再开口、对准麦克风、换另一副耳机对比；")
        print("  窄带多（4kHz 以上几乎没有能量）→ 这副耳机的通话链路是瓶颈，优先换耳机或改走有线。")
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
