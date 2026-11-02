import unittest

import numpy as np

from tt_voice.audio_io import UtteranceRecorder


def tone(freq: float, seconds: float, sr: int = 16000, amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)


class FakeBlocks:
    """固定 100ms 块序列，模拟"用户一按回车就立刻开口"。"""

    def __init__(self, blocks: list[np.ndarray]):
        self.blocks = blocks
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        if self.index >= len(self.blocks):
            self.index = 0
        block = self.blocks[self.index]
        self.index += 1
        return block.tobytes()


class TestUtteranceRecorder(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        self.block = 1600

    def make_blocks(self, silence_s: float, speech_s: float, tail_s: float = 1.2):
        pcm = np.concatenate([np.zeros(int(self.sr * silence_s), dtype=np.int16),
                              tone(300, speech_s),
                              np.zeros(int(self.sr * tail_s), dtype=np.int16)])
        blocks = [pcm[i:i + self.block] for i in range(0, pcm.size, self.block)]
        return [np.pad(b, (0, self.block - b.size)) for b in blocks]

    def test_preroll_keeps_onset(self):
        """用户立刻开口时，起音不能被切掉（真人验收里"撤销"被吞的就是这个）。"""
        blocks = self.make_blocks(silence_s=0.0, speech_s=0.4)
        recorder = UtteranceRecorder(FakeBlocks(blocks), self.sr, self.block, preroll_s=0.5)
        audio = recorder.record(max_seconds=2.0, silence_stop_s=0.5)
        self.assertGreater(audio.size, int(self.sr * 0.4))
        self.assertGreater(float(np.max(np.abs(audio[:160]))), 1000)

    def test_stops_after_silence(self):
        blocks = self.make_blocks(silence_s=0.3, speech_s=0.4, tail_s=1.5)
        recorder = UtteranceRecorder(FakeBlocks(blocks), self.sr, self.block, preroll_s=0.3)
        audio = recorder.record(max_seconds=3.0, silence_stop_s=0.5)
        seconds = audio.size / self.sr
        self.assertLess(seconds, 1.8)
        self.assertGreater(seconds, 0.7)

    def test_respects_max_seconds(self):
        blocks = self.make_blocks(silence_s=0.0, speech_s=5.0)
        recorder = UtteranceRecorder(FakeBlocks(blocks), self.sr, self.block, preroll_s=0.2)
        audio = recorder.record(max_seconds=1.5, silence_stop_s=0.5)
        self.assertLessEqual(audio.size / self.sr, 1.9)

    def test_short_beep_does_not_start_recording(self):
        """0.1~0.2 秒的提示音不能被当成"开始说话"，否则会在用户开口前就结束录音。"""
        beep = tone(660, 0.1, amp=0.5)
        silence = np.zeros(int(self.sr * 1.0), dtype=np.int16)
        speech = tone(300, 0.6, amp=0.4)
        tail = np.zeros(int(self.sr * 1.2), dtype=np.int16)
        pcm = np.concatenate([silence, beep, silence, speech, tail])
        blocks = [pcm[i:i + self.block] for i in range(0, pcm.size, self.block)]
        blocks = [np.pad(b, (0, self.block - b.size)) for b in blocks]
        recorder = UtteranceRecorder(FakeBlocks(blocks), self.sr, self.block, preroll_s=0.5)
        audio = recorder.record(max_seconds=3.0, silence_stop_s=0.5)
        seconds = audio.size / self.sr
        # 应该录到后面的语音（长度明显超过"提示音+0.5s"，且含 300Hz 主体）
        self.assertGreater(seconds, 1.0)
        spec = np.abs(np.fft.rfft(audio.astype(np.float64)))
        freqs = np.fft.rfftfreq(audio.size, 1 / self.sr)
        peak_freq = float(freqs[int(np.argmax(spec))])
        self.assertLess(abs(peak_freq - 300.0), 40.0)


if __name__ == "__main__":
    unittest.main()
