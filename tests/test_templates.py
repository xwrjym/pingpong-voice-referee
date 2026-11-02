import os
import tempfile
import unittest

import numpy as np

from tt_voice.features import FeatureExtractor
from tt_voice.templates import KIND_COMMAND, KIND_WAKE, Template, TemplateStore


def tone(freq: float, seconds: float, sr: int = 16000, noise: float = 0.0, seed: int = 0):
    t = np.arange(int(sr * seconds)) / sr
    pcm = 0.4 * 32767 * np.sin(2 * np.pi * freq * t)
    if noise:
        rng = np.random.default_rng(seed)
        pcm = pcm + rng.normal(0, noise * 32767, pcm.size)
    return np.clip(pcm, -32768, 32767).astype(np.int16)


class TestTemplates(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor(16000)

    def build_store(self) -> TemplateStore:
        store = TemplateStore(sample_rate=16000, feature_dim=self.extractor.dim)
        for seed in range(3):
            store.templates.append(Template(
                phrase="左方加一分", intent="score_left", kind=KIND_COMMAND,
                feats=self.extractor.extract(tone(440, 0.7, noise=0.02, seed=seed))))
        for seed in range(3):
            store.templates.append(Template(
                phrase="小裁判", intent="wake", kind=KIND_WAKE,
                feats=self.extractor.extract(tone(900, 0.5, noise=0.02, seed=seed))))
        return store

    def test_phrases_and_kinds(self):
        store = self.build_store()
        self.assertEqual(store.phrases(KIND_WAKE), ["小裁判"])
        self.assertEqual(len(store.by_kind(KIND_COMMAND)), 3)

    def test_calibrate_sets_thresholds(self):
        store = self.build_store()
        store.calibrate(log=lambda *_: None)
        thresholds = {t.threshold for t in store.templates}
        self.assertEqual(len(thresholds), 2)            # 每个短语一个阈值
        self.assertTrue(all(t > 0 for t in thresholds))

    def test_save_and_load_roundtrip(self):
        store = self.build_store()
        store.calibrate(log=lambda *_: None)
        tmp = tempfile.mkdtemp(prefix="tttpl_")
        path = store.save(os.path.join(tmp, "templates.npz"))
        loaded = TemplateStore.load(path)
        self.assertEqual(len(loaded.templates), len(store.templates))
        self.assertEqual(loaded.templates[0].phrase, store.templates[0].phrase)
        self.assertAlmostEqual(loaded.templates[0].threshold, store.templates[0].threshold, places=6)
        np.testing.assert_allclose(loaded.templates[0].feats, store.templates[0].feats, atol=1e-5)
        self.assertEqual(loaded.feature_dim, store.feature_dim)

    def test_load_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            TemplateStore.load(os.path.join(tempfile.gettempdir(), "nope_not_here.npz"))


if __name__ == "__main__":
    unittest.main()
