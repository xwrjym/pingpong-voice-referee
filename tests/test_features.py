import unittest

import numpy as np

from tt_voice.features import FeatureExtractor, mel_filterbank, trim_silence


def tone(freq: float, seconds: float, sr: int = 16000, amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor(16000)

    def test_filterbank_shape(self):
        bank = mel_filterbank(16000, 512, 26)
        self.assertEqual(bank.shape, (26, 257))
        self.assertTrue((bank >= 0).all())
        self.assertGreater(bank.sum(axis=1).min(), 0.0)

    def test_mfcc_shape_and_finiteness(self):
        feats = self.extractor.extract(tone(440, 1.0))
        self.assertEqual(feats.shape[1], 26)           # 13 MFCC + 13 Δ
        self.assertGreater(feats.shape[0], 90)         # 1s / 10ms ≈ 100 帧
        self.assertTrue(np.isfinite(feats).all())

    def test_cmn_removes_mean(self):
        feats = self.extractor.extract(tone(440, 1.0))
        self.assertLess(abs(feats[:, :13].mean()), 1e-3)

    def test_gain_invariance(self):
        """增益变化不该显著改变特征（CMN 的作用）。"""
        loud = (tone(440, 1.0, amp=0.8).astype(np.float32))
        quiet = (tone(440, 1.0, amp=0.2).astype(np.float32)).astype(np.int16)
        a = self.extractor.extract(loud.astype(np.int16))
        b = self.extractor.extract(quiet)
        n = min(a.shape[0], b.shape[0])
        diff = np.mean(np.abs(a[:n] - b[:n]))
        self.assertLess(diff, 0.5)

    def test_too_short(self):
        self.assertEqual(self.extractor.extract(tone(440, 0.005)).shape[0], 0)

    def test_extract_trims_silence(self):
        silence = np.zeros(8000, dtype=np.int16)
        pcm = np.concatenate([silence, tone(440, 0.5), silence])
        feats = self.extractor.extract(pcm)
        self.assertGreater(feats.shape[0], 30)
        self.assertLess(feats.shape[0], 70)

    def test_trim_silence_helper(self):
        raw = np.concatenate([np.zeros(80, dtype=np.int16).reshape(-1, 1)] * 26, axis=1)
        energy = np.concatenate([np.full(10, -80.0), np.full(50, -20.0), np.full(10, -80.0)])
        trimmed = trim_silence(raw, energy)
        self.assertLess(trimmed.shape[0], 70)
        self.assertGreater(trimmed.shape[0], 40)

    def test_features_stable_across_silence_padding(self):
        """同一句话前后补不同长度的静音，特征必须几乎不变。

        这是踩过的坑：如果先做 CMN 再剪静音，静音长度会改变倒谱均值，
        导致整段匹配距离 0.79、分段匹配距离 3.76，识别直接崩掉。
        """
        speech = tone(440, 0.5)
        short = np.concatenate([np.zeros(1600, dtype=np.int16), speech,
                                np.zeros(1600, dtype=np.int16)])
        long_ = np.concatenate([np.zeros(16000, dtype=np.int16), speech,
                                np.zeros(16000, dtype=np.int16)])
        a = self.extractor.extract(short)
        b = self.extractor.extract(long_)
        self.assertLess(abs(a.shape[0] - b.shape[0]), 3)
        self.assertLess(float(np.mean(np.abs(a - b))), 0.05)


if __name__ == "__main__":
    unittest.main()
