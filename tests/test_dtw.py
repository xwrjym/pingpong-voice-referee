import unittest

import numpy as np

from tt_voice.dtw import dtw
from tt_voice.features import FeatureExtractor


def synthetic(n_frames: int = 60, dim: int = 26, seed: int = 0,
              signature: float = 0.0) -> np.ndarray:
    """造一段确定性的特征序列；signature 用来给序列一个可辨认的"口型"。"""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, (n_frames, dim)).astype(np.float32)
    if signature:
        base += signature * np.linspace(-1.0, 1.0, n_frames)[:, None]
    return base


class TestDtw(unittest.TestCase):
    """DTW 行为测试用合成特征，不掺进前端归一化，故障定位更干净。"""

    def test_identical_is_zero(self):
        a = synthetic(seed=1)
        self.assertAlmostEqual(dtw(a, a), 0.0, places=5)
        self.assertAlmostEqual(dtw(a, a, metric="cosine"), 0.0, places=5)

    def test_same_sound_closer_than_other(self):
        ref = synthetic(seed=2, signature=1.0)
        rng = np.random.default_rng(20)
        same = ref + rng.normal(0, 0.1, ref.shape).astype(np.float32)
        other = synthetic(seed=3, signature=-1.0)
        self.assertLess(dtw(same, ref), dtw(other, ref))
        self.assertLess(dtw(same, ref, metric="cosine"), dtw(other, ref, metric="cosine"))

    def test_time_shift_within_sequence_is_tolerated(self):
        ref = synthetic(40, seed=4, signature=2.0)
        shifted = np.concatenate([synthetic(30, seed=5), ref, synthetic(30, seed=6)], axis=0)
        for metric in ("euclid", "cosine"):
            with self.subTest(metric=metric):
                self.assertLess(dtw(shifted, ref, subsequence=True, metric=metric),
                                dtw(shifted, ref, metric=metric))

    def test_empty_features(self):
        self.assertTrue(np.isinf(dtw(np.zeros((0, 26)), synthetic(20, seed=7))))

    def test_tempo_change_tolerated(self):
        ref = synthetic(40, seed=8, signature=1.5)
        index = np.round(np.linspace(0, len(ref) - 1, int(len(ref) * 1.6))).astype(int)
        slower = ref[index]
        other = synthetic(40, seed=9, signature=-1.5)
        # 自己放慢 1.6 倍，仍然应该比无关序列更像
        self.assertLess(dtw(slower, ref, subsequence=True), dtw(other, ref, subsequence=True))
        self.assertLessEqual(dtw(slower, ref, subsequence=True), dtw(slower, ref) + 1e-6)

    def test_real_audio_smoke(self):
        """真音频走一遍完整前端（含剪静音/CMN），确认能算出有限距离。"""
        extractor = FeatureExtractor(16000)
        t = np.arange(8000) / 16000
        pcm = (0.4 * 32767 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
        feats = extractor.extract(pcm)
        self.assertGreater(feats.shape[0], 10)
        self.assertAlmostEqual(dtw(feats, feats), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
