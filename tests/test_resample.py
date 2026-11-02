import unittest

import numpy as np

from tt_voice.audio_io import StreamResampler, resample_int16


def sine(freq: float, seconds: float, sr: int, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)


class TestStreamResampler(unittest.TestCase):
    def test_integer_ratio_length_and_continuity(self):
        """48k -> 16k：分块处理的总长度要对，块边界不能出现相位跳变。"""
        rs = StreamResampler(48000, 16000, 4800)
        x = sine(440, 1.0, 48000)
        out = np.concatenate([rs.process(x[i:i + 4800]) for i in range(0, x.size, 4800)])
        self.assertLessEqual(abs(out.size - 16000), 2)

        # 440Hz 在 16k 下相邻样本最大增量约 2*pi*440/16000*0.5*32767 ≈ 2830
        limit = 2 * np.pi * 440 / 16000 * 0.5 * 32767 * 1.6
        self.assertLess(float(np.max(np.abs(np.diff(out.astype(np.float32))))), limit)

    def test_integer_ratio_matches_batch(self):
        rs = StreamResampler(48000, 16000, 4800)
        x = sine(300, 0.5, 48000)
        out = np.concatenate([rs.process(x[i:i + 4800]) for i in range(0, x.size, 4800)])
        batch = resample_int16(x, 48000, 16000)
        n = min(out.size, batch.size)
        # 分块用 valid 卷积、整段用 same 卷积，会有不到一个样本的相位差，
        # 所以只要求量级一致（正弦一个样本的偏移本身就能造成上千的均值差）
        self.assertLess(float(np.mean(np.abs(out[:n].astype(np.float32)
                                             - batch[:n].astype(np.float32)))), 900.0)

    def test_non_integer_ratio(self):
        """44.1k -> 16k 走线性插值，长度正确且无跳变。"""
        rs = StreamResampler(44100, 16000, 4410)
        x = sine(500, 0.5, 44100)
        out = np.concatenate([rs.process(x[i:i + 4410]) for i in range(0, x.size, 4410)])
        self.assertLessEqual(abs(out.size - 8000), 4)
        limit = 2 * np.pi * 500 / 16000 * 0.5 * 32767 * 1.6
        self.assertLess(float(np.max(np.abs(np.diff(out.astype(np.float32))))), limit)

    def test_same_rate_passthrough(self):
        rs = StreamResampler(16000, 16000, 1600)
        x = sine(440, 0.1, 16000)
        np.testing.assert_array_equal(rs.process(x), x)


if __name__ == "__main__":
    unittest.main()
