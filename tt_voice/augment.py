"""音频增广：加噪 / 混响 / 远场，用于造测试集和注册时的鲁棒性扩展。"""

from __future__ import annotations

import numpy as np


def speech_rms_db(x: np.ndarray, sr: int, frame_ms: float = 30.0) -> float:
    """只统计有声段的有效电平（dBFS），用它来定义信噪比才符合直觉。"""
    frame = max(1, int(sr * frame_ms / 1000))
    if x.size < frame:
        return -60.0
    frames = np.lib.stride_tricks.sliding_window_view(
        x.astype(np.float32) / 32768.0, frame)[::frame]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    voiced = rms[rms > 10 ** (-50 / 20)]
    if voiced.size == 0:
        voiced = rms
    return float(20 * np.log10(max(float(np.mean(voiced)), 1e-6)))


def pink_noise(n: int, rng: np.random.Generator, sr: int) -> np.ndarray:
    """粉红噪声比白噪声更接近室内底噪（低频能量更大）。"""
    white = rng.normal(0.0, 1.0, n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    spec = spec / np.sqrt(freqs)
    out = np.fft.irfft(spec, n=n)
    return out / (np.max(np.abs(out)) + 1e-9)


def add_noise(x: np.ndarray, sr: int, snr_db: float, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = pink_noise(x.size, rng, sr) * (10 ** ((speech_rms_db(x, sr) - snr_db) / 20))
    out = x / 32768.0 + noise
    return np.clip(out * 32768.0, -32768, 32767).astype(np.int16)


def room_ir(sr: int, t60: float = 0.25, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = max(1, int(sr * t60))
    decay = np.exp(-6.9 * np.arange(n) / max(1.0, sr * t60 * 0.3))
    h = rng.normal(0.0, 1.0, n) * decay
    h[0] = 1.0
    return h / np.sqrt(np.sum(h ** 2))


def reverb(x: np.ndarray, sr: int, seed: int = 7) -> np.ndarray:
    h = room_ir(sr, seed=seed)
    out = np.convolve(x / 32768.0, h, mode="full")[: x.size]
    peak = float(np.max(np.abs(out))) + 1e-9
    if peak > 0.98:
        out = out * (0.98 / peak)
    return np.clip(out * 32768.0, -32768, 32767).astype(np.int16)


def make_variant(clean: np.ndarray, sr: int, variant: str, seed: int = 11) -> np.ndarray:
    if variant == "clean":
        return clean
    if variant.startswith("snr"):
        return add_noise(clean, sr, float(variant[3:]), seed)
    if variant == "reverb":
        return reverb(clean, sr, seed)
    if variant == "far":
        return add_noise(reverb(clean, sr, seed), sr, 8.0, seed + 1)
    raise ValueError(variant)
