import unittest

from tt_voice.textnorm import cn_digits_to_arabic, normalize, syllables, to_halfwidth


class TestTextNorm(unittest.TestCase):
    def test_halfwidth(self):
        self.assertEqual(to_halfwidth("ＡＢ１"), "AB1")
        self.assertEqual(to_halfwidth("左　方"), "左 方")

    def test_digits(self):
        self.assertEqual(cn_digits_to_arabic("加一分"), "加1分")
        self.assertEqual(cn_digits_to_arabic("两分"), "2分")

    def test_normalize_strips_punct_and_space(self):
        self.assertEqual(normalize("左方加一分。"), "左方加1分")
        self.assertEqual(normalize(" 小裁判， "), "小裁判")
        self.assertEqual(normalize("左方 加一分！"), "左方加1分")

    def test_syllables(self):
        self.assertEqual(syllables("小裁判"), ("xiao", "cai", "pan"))
        self.assertEqual(syllables("左方加一分"), ("zuo", "fang", "jia", "yi", "fen"))
        self.assertEqual(syllables(""), ())

    def test_homophone_same_syllables(self):
        # 同音错字必须落在同一串拼音上，这是纠错层的基础
        self.assertEqual(syllables("佐方加一份"), syllables("左方加一分"))

    def test_digit_and_chinese_numeral_agree(self):
        # "加1分" 和 "加一分" 必须落在同一串拼音上（ASR 两种写法都可能给）
        self.assertEqual(syllables("左方加1分"), syllables("左方加一分"))


if __name__ == "__main__":
    unittest.main()
