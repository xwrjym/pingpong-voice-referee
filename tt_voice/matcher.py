"""把 ASR 文本和候选口令做「拼音近音对齐」，给相似度分数。

为什么不用汉字直接比对：ASR 的错误绝大多数是**同音/近音替换**（左/佐、分/份、判/盼），
汉字串的编辑距离会把这些都算成完全错误；而拼音层面它们是 0 代价或小代价。

为什么不用拼音直接相等：老人/口音会让 zh-z、n-l、in-ing、前后鼻音混淆，
所以按「声母相同 / 韵母相同 / 常见混淆对」给分级的替换代价。
"""

from __future__ import annotations

from dataclasses import dataclass

from .textnorm import is_filler, levenshtein, normalize, syllables

# 常见混淆：这两组音节差异的代价比"完全不同的音节"低得多
_NEAR_INITIALS = {
    ("z", "zh"), ("c", "ch"), ("s", "sh"), ("n", "l"), ("f", "h"),
    ("l", "r"), ("j", "z"), ("q", "c"), ("x", "s"), ("y", "i"), ("w", "u"),
}
_NEAR_FINALS = {
    ("in", "ing"), ("an", "ang"), ("en", "eng"), ("ian", "iang"),
    ("uan", "uang"), ("un", "ong"), ("ong", "eng"), ("uo", "o"), ("ü", "u"),
}

COST_SAME = 0.0
COST_SAME_INITIAL = 0.45       # 声母相同、韵母不同（加/减、左/右 的一部分情况）
COST_SAME_FINAL = 0.60         # 韵母相同、声母不同
COST_NEAR = 0.25               # 常见混淆对（zh/z 等）
COST_DIFFERENT = 1.0
COST_INSERT_FILLER = 0.25      # 呃/嗯/那个 之类的口语填充
COST_INSERT = 0.9              # 其它多出来的音节
COST_DELETE = 0.9


def split_initial_final(syl: str) -> tuple[str, str]:
    for ini in ("zh", "ch", "sh"):
        if syl.startswith(ini):
            return ini, syl[2:]
    if syl and syl[0] in "bpmfdtnlgkhjqxrzcsyw":
        return syl[0], syl[1:]
    return "", syl


def _pair(a: str, b: str) -> float:
    return (a, b) if (a, b) in _NEAR_INITIALS else (b, a)


def syllable_cost(a: str, b: str) -> float:
    if a == b:
        return COST_SAME
    ai, af = split_initial_final(a)
    bi, bf = split_initial_final(b)
    if ai == bi and ai:
        return COST_SAME_INITIAL
    if af == bf and af:
        return COST_SAME_FINAL
    if _pair(ai, bi) in _NEAR_INITIALS or _pair(af, bf) in _NEAR_FINALS:
        return COST_NEAR
    return COST_DIFFERENT


def pinyin_distance(utt: tuple[str, ...], ref: tuple[str, ...]) -> float:
    """音节级加权编辑距离（插入/删除/替换代价不对称）。"""
    n, m = len(utt), len(ref)
    if n == 0:
        return float(m) * COST_DELETE
    if m == 0:
        return float(n) * COST_INSERT
    prev = [j * COST_DELETE for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [i * (COST_INSERT_FILLER if is_filler(utt[i - 1]) else COST_INSERT)]
        for j in range(1, m + 1):
            sub = prev[j - 1] + syllable_cost(utt[i - 1], ref[j - 1])
            dele = prev[j] + COST_DELETE
            ins = cur[j - 1] + (COST_INSERT_FILLER if is_filler(utt[i - 1]) else COST_INSERT)
            cur.append(min(sub, dele, ins))
        prev = cur
    return prev[m]


def similarity(utt_syls: tuple[str, ...], ref_syls: tuple[str, ...],
               utt_norm: str = "", ref_norm: str = "") -> float:
    """返回 0~1 的相似度：0.8*拼音相似 + 0.2*汉字相似。

    汉字相似度用来区分同音但不同意的口令（例如「加分」和「加分」之外的同音干扰词），
    权重压低是因为 ASR 的同音错字不应该被重罚。
    """
    if not ref_syls:
        return 0.0
    denom = max(len(utt_syls), len(ref_syls))
    syl_sim = 1.0 - pinyin_distance(utt_syls, ref_syls) / denom
    syl_sim = max(0.0, min(1.0, syl_sim))
    if utt_norm and ref_norm:
        cdenom = max(len(utt_norm), len(ref_norm))
        char_sim = 1.0 - levenshtein(utt_norm, ref_norm) / cdenom
    else:
        char_sim = syl_sim
    return 0.8 * syl_sim + 0.2 * max(0.0, min(1.0, char_sim))


def subsequence_similarity(utt_syls: tuple[str, ...], ref_syls: tuple[str, ...]) -> float:
    """只要求口令出现在句子的**任意位置**（起点终点都自由）。

    用在两个场景：唤醒后回灌音频里必然带着唤醒词，以及"小裁判左方加一分"连说，
    识别文本里会混入唤醒词的残字（实测"开开右方加一分"），
    这时按整句对齐会把相似度压到阈值以下。

    注意语义：**口令必须整条都对齐上**，只是允许它出现在句子的任意位置。
    （早期版本写成"取最后一行的最小值"，等于允许只匹配口令的一部分，
      结果"帮我把灯关掉"会因为首字"帮/报"声母相同而被判成"报分"——已修。）
    """
    n, m = len(utt_syls), len(ref_syls)
    if n == 0 or m == 0:
        return 0.0
    # 行 0 只能在"一个音节都没消费"时是 0：否则口令音节会被免费吃掉，
    # 实测会让"小裁判小"整条对齐上"撤销"（相似度 1.0）。
    prev = [0.0] + [float("inf")] * m
    best = float("inf")
    for i in range(1, n + 1):
        cur = [0.0] * (m + 1)       # cur[0] = 0：允许跳过任意前缀
        for j in range(1, m + 1):
            sub = prev[j - 1] + syllable_cost(utt_syls[i - 1], ref_syls[j - 1])
            dele = prev[j] + COST_DELETE
            ins = cur[j - 1] + (COST_INSERT_FILLER if is_filler(utt_syls[i - 1])
                                else COST_INSERT)
            cur[j] = min(sub, dele, ins)
        prev = cur
        best = min(best, cur[m])    # 终点自由：口令对齐完就停，后面的句子内容不罚
    return max(0.0, 1.0 - best / m)


@dataclass(frozen=True)
class Reference:
    """一条候选口令（或唤醒词）。"""

    key: str            # intent 或 "wake"
    text: str           # 口令原文
    norm: str           # 归一化文本
    syls: tuple[str, ...]


def make_reference(key: str, text: str) -> Reference:
    return Reference(key=key, text=text, norm=normalize(text), syls=syllables(text))


@dataclass
class Candidate:
    reference: Reference
    score: float


def rank(utt: str, references: list[Reference]) -> list[Candidate]:
    utt_norm = normalize(utt)
    utt_syls = syllables(utt)
    if not utt_norm:
        return []
    out = [
        Candidate(ref, similarity(utt_syls, ref.syls, utt_norm, ref.norm))
        for ref in references
    ]
    out.sort(key=lambda c: (-c.score, len(c.reference.text)))
    return out
