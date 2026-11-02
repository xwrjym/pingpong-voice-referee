import unittest

from tt_voice.matcher import make_reference, rank, similarity, syllable_cost
from tt_voice.textnorm import syllables


def sim(utt: str, ref: str) -> float:
    return similarity(syllables(utt), syllables(ref), utt, ref)


class TestMatcher(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(sim("左方加一分", "左方加一分"), 1.0, places=6)

    def test_cost_ordering(self):
        self.assertLess(syllable_cost("zhang", "zhong"), syllable_cost("zhang", "liang"))
        self.assertLess(syllable_cost("zhi", "chi"), syllable_cost("zhi", "feng"))
        self.assertEqual(syllable_cost("fen", "fen"), 0.0)

    def test_same_initial_cheaper_than_unrelated(self):
        # 加/减 声母相同：应当比 加/香 更接近
        self.assertLess(sim("左方减一分", "左方加一分"), 1.0)
        self.assertLess(sim("左方香一分", "左方加一分"), sim("左方减一分", "左方加一分"))

    def test_filler_cheap(self):
        self.assertGreater(sim("嗯左方加一分", "左方加一分"), 0.9)

    def test_unrelated_low(self):
        self.assertLess(sim("今天天气不错", "左方加一分"), 0.4)

    def test_rank_prefers_right_intent(self):
        refs = [make_reference(intent, phrase) for intent, phrase in (
            ("score_left", "左方加一分"),
            ("score_right", "右方加一分"),
            ("undo", "撤销"),
        )]
        ranked = rank("右方加一分", refs)
        self.assertEqual(ranked[0].reference.key, "score_right")
        self.assertGreater(ranked[0].score, ranked[1].score)


if __name__ == "__main__":
    unittest.main()
