"""文本归一化 + 拼音切分。

ASR 输出和固定口令之间常见的差异有三类：

1. 标点/空格/全半角差异      -> 去掉标点、统一半角
2. 同音字（左/佐、分/份）    -> 用拼音比对，而不是用汉字比对
3. 数字写法（一 vs 1）       -> 中文数字统一成阿拉伯数字
"""

from __future__ import annotations

import re
import unicodedata

from pypinyin import Style, lazy_pinyin

_PUNCT_RE = re.compile(r"[\s\u3000,\.。，、；;：:！!？\?\"'\u201c\u201d\u2018\u2019（）\(\)\[\]【】\-—_·~`]+")

_CN_DIGITS = {
    "零": "0", "〇": "0", "一": "1", "幺": "1", "二": "2", "两": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
}

# 阿拉伯数字要读回拼音，否则 "加1分" 和 "加一分" 会被当成完全不同的音节
_DIGIT_SYLLABLE = {
    "0": "ling", "1": "yi", "2": "er", "3": "san", "4": "si",
    "5": "wu", "6": "liu", "7": "qi", "8": "ba", "9": "jiu",
}

# 口语填充词，在拼音对齐时允许以很低代价被删除
FILLERS = {"呃", "嗯", "啊", "哦", "呀", "诶", "唉", "那个", "这个", "就是", "然后", "麻烦"}


def to_halfwidth(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def cn_digits_to_arabic(text: str) -> str:
    return "".join(_CN_DIGITS.get(ch, ch) for ch in text)


def normalize(text: str) -> str:
    """归一化：全角->半角、去标点空格、中文数字->阿拉伯数字、统一小写。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", to_halfwidth(text))
    text = _PUNCT_RE.sub("", text)
    text = cn_digits_to_arabic(text)
    return text.lower().strip()


def normalize_light(text: str) -> str:
    """轻度归一化：只做全角->半角、去标点空格、小写。

    保留中文数字，好处是"字符数 == 音节数"这个假设成立，
    唤醒词前缀裁剪需要按这个假设把音节位置映射回字符位置。
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", to_halfwidth(text))
    return _PUNCT_RE.sub("", text).lower().strip()


def syllables(text: str) -> tuple[str, ...]:
    """把文本切成无声调拼音音节序列；非汉字按小写拉丁串保留。"""
    norm = normalize_light(text)
    if not norm:
        return ()
    out: list[str] = []
    for token in lazy_pinyin(norm, style=Style.NORMAL):
        token = token.strip().lower()
        if not token:
            continue
        if token.isdigit():
            out.extend(_DIGIT_SYLLABLE.get(ch, ch) for ch in token)
        else:
            out.append(token)
    return tuple(out)


_FILLER_SYLLABLES = frozenset(
    token for filler in FILLERS for token in lazy_pinyin(normalize_light(filler), style=Style.NORMAL)
)


def is_filler(token: str) -> bool:
    """token 是拼音音节（matcher 里按音节调用）。"""
    return token in _FILLER_SYLLABLES


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
