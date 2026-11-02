import unittest

from tt_voice.config import AppConfig
from tt_voice.parser import Parser


class TestParser(unittest.TestCase):
    def setUp(self):
        self.cfg = AppConfig()
        self.parser = Parser(self.cfg)

    def test_wake_word(self):
        match = self.parser.matches_wake("小裁判")
        self.assertTrue(match.accepted)
        self.assertEqual(match.intent, "wake")

    def test_wake_word_with_filler(self):
        self.assertTrue(self.parser.matches_wake("嗯小裁判").accepted)

    def test_other_speech_is_not_wake(self):
        self.assertFalse(self.parser.matches_wake("我们再来一局吧").accepted)

    def test_basic_commands(self):
        cases = {
            "左方加一分": "score_left",
            "右边加一分": "score_right",
            "撤销上一分": "undo",
            "重置比分": "reset",
            "确认这一分": "confirm",
            "报分": "report",
        }
        for text, intent in cases.items():
            with self.subTest(text=text):
                match = self.parser.parse_command(text)
                self.assertTrue(match.accepted, msg=f"{text} 被拒：{match.reason}")
                self.assertEqual(match.intent, intent)

    def test_homophone_correction(self):
        # 同音错字（佐/份）应当被纠正回正确意图
        match = self.parser.parse_command("佐方加一份")
        self.assertTrue(match.accepted)
        self.assertEqual(match.intent, "score_left")

    def test_left_right_not_confused(self):
        self.assertEqual(self.parser.parse_command("左方加一分").intent, "score_left")
        self.assertEqual(self.parser.parse_command("右方加一分").intent, "score_right")

    def test_wake_prefix_stripped(self):
        match = self.parser.parse_command("小裁判左方加一分")
        self.assertTrue(match.accepted)
        self.assertEqual(match.intent, "score_left")
        self.assertEqual(match.match_text, "左方加一分")

    def test_wake_prefix_not_over_stripped(self):
        # 口令开头和唤醒词不像时不能被剪掉
        self.assertEqual(self.parser.strip_wake_prefix("撤销上一分"), "撤销上一分")

    def test_irrelevant_speech_rejected(self):
        for text in ("今天天气不错", "帮我把灯关掉", "我们再来一局吧"):
            with self.subTest(text=text):
                self.assertFalse(self.parser.parse_command(text).accepted)

    def test_wake_missing_onset_option(self):
        """起音被吞（"小裁判"→"裁判"）默认不放过，打开开关后放过。"""
        self.assertFalse(self.parser.matches_wake("裁判").accepted)
        loose = Parser(AppConfig.from_dict({"wake_allow_missing_onset": True,
                                            "wake_words": ["小裁判"]}))
        match = loose.matches_wake("裁判")
        self.assertTrue(match.accepted)
        self.assertEqual(match.intent, "wake")
        # 开关打开也不能把无关句子当成唤醒
        self.assertFalse(loose.matches_wake("真猜").accepted)

    def test_short_phrase_needs_precision(self):
        # 短口令稍微不像就应当拒识，而不是猜
        match = self.parser.parse_command("撤消了")
        if match.accepted:
            self.assertGreaterEqual(match.score, 0.95)


if __name__ == "__main__":
    unittest.main()
