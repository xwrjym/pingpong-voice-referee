"""MFCC 特征（纯 numpy 实现，不依赖 librosa/torch）。

* 只保留 c1~c13（丢掉 c0，能量信息由 VAD 负责）
* 加一阶差分 Δ，拼成 26 维
* 做倒谱均值归一化（CMN），抵消麦克风/增益差异

这套特征配合 DTW 是固定口令识别的经典做法，在 RK3588 上只占很少 CPU。
"""

from __future__ import annotations

import numpy as np


def hz_to_mel(freq: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 26,
                   fmin: float = 20.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax if fmax is not None else sr / 2.0
    n_bins = n_fft // 2 + 1
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mel_points) / sr).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    bank = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center == left:
            center = min(left + 1, n_bins - 1)
        if right == center:
            right = min(center + 1, n_bins - 1)
        for k in range(left, center):
            bank[i - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            bank[i - 1, k] = (right - k) / max(1, right - center)
    return bank


def dct_matrix(n_mels: int, n_mfcc: int) -> np.ndarray:
    """正交 DCT-II 矩阵（n_mels, n_mfcc）。"""
    m = np.arange(n_mels)[:, None]
    k = np.arange(n_mfcc)[None, :]
    mat = np.cos(np.pi * k * (2 * m + 1) / (2 * n_mels))
    mat *= np.sqrt(2.0 / n_mels)
    mat[:, 0] /= np.sqrt(2.0)
    return mat.astype(np.float32)


class FeatureExtractor:
    def __init__(self, sr: int = 16000, n_mfcc: int = 13, frame_ms: float = 25.0,
                 hop_ms: float = 10.0, n_fft: int = 512, n_mels: int = 26,
                 use_delta: bool = True, cmn: bool = True, preemph: float = 0.97,
                 trim: bool = True, trim_margin_db: float = 15.0, cmvn: bool = False,
                 noise_suppress: bool = True, ns_over: float = 1.0, ns_floor: float = 0.1):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.frame_len = int(sr * frame_ms / 1000)
        self.hop = max(1, int(sr * hop_ms / 1000))
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.use_delta = use_delta
        self.cmn = cmn
        self.preemph = preemph
        self.trim = trim
        self.trim_margin_db = trim_margin_db
        self.cmvn = cmvn
        self.noise_suppress = noise_suppress
        self.ns_over = ns_over
        self.ns_floor = ns_floor
        self.dim = n_mfcc * (2 if use_delta else 1)
        self._bank = mel_filterbank(sr, n_fft, n_mels)
        self._dct = dct_matrix(n_mels, n_mfcc)
        self._window = np.hamming(self.frame_len).astype(np.float32)

    # ------------------------------------------------------------ 工具
    @staticmethod
    def _delta(feat: np.ndarray, width: int = 2) -> np.ndarray:
        pad = np.pad(feat, ((width, width), (0, 0)), mode="edge")
        num = pad[2 * width:] - pad[:-2 * width]
        return num / (2 * width)

    def num_frames(self, n_samples: int) -> int:
        if n_samples < self.frame_len:
            return 0
        return 1 + (n_samples - self.frame_len) // self.hop

    # ------------------------------------------------------------ 主流程
    def _mfcc_frames(self, pcm_int16: bytes | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (未做 CMN 的 MFCC，逐帧能量 dB)。"""
        x = np.frombuffer(pcm_int16, dtype="<i2") if isinstance(pcm_int16, (bytes, bytearray)) \
            else np.asarray(pcm_int16)
        if x.size < self.frame_len:
            return np.zeros((0, self.n_mfcc), dtype=np.float32), np.zeros(0, dtype=np.float32)
        x = x.astype(np.float32) / 32768.0
        energy = self._energy_from_float(x)
        if self.preemph:
            x = np.append(x[0], x[1:] - self.preemph * x[:-1])

        frames = np.lib.stride_tricks.sliding_window_view(x, self.frame_len)[::self.hop]
        frames = frames * self._window
        spectrum = np.fft.rfft(frames, n=self.n_fft, axis=1)
        power = (np.abs(spectrum) ** 2) / self.n_fft
        if self.noise_suppress and power.shape[0] >= 5:
            # 谱减：每个频点取时间上的低分位数当作噪声底，减掉并留一个下限，
            # 对稳态加性噪声（房间底噪、风扇）特别有效，代价几乎为 0。
            noise = np.percentile(power, 10, axis=0, keepdims=True)
            power = np.maximum(power - self.ns_over * noise, self.ns_floor * power)
        mel = power @ self._bank.T
        log_mel = np.log(mel + 1e-10)
        mfcc = log_mel @ self._dct
        return mfcc, energy

    def _energy_from_float(self, x: np.ndarray) -> np.ndarray:
        n_frames = self.num_frames(x.size)
        if n_frames <= 0:
            return np.zeros(0, dtype=np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(x, self.frame_len)[::self.hop]
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        return 20.0 * np.log10(rms)

    def extract(self, pcm_int16: bytes | np.ndarray) -> np.ndarray:
        """输入 16bit PCM，输出 (T, dim) float32。

        顺序很关键：**先去静音、再在留下的有声帧上做 CMN**。
        反过来的话，前后静音长度不同会把倒谱均值拖偏，同一句话的特征就漂了
        （踩过这个坑：整段匹配距离 0.79，分段匹配变成 3.76）。
        """
        mfcc, energy = self._mfcc_frames(pcm_int16)
        if mfcc.shape[0] == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.trim:
            mfcc = trim_silence(mfcc, energy, margin_db=self.trim_margin_db)
            if mfcc.shape[0] == 0:
                return np.zeros((0, self.dim), dtype=np.float32)

        if self.cmn and mfcc.shape[0] > 1:
            mean = mfcc.mean(axis=0, keepdims=True)
            mfcc = mfcc - mean
            if self.cmvn:
                # 方差也归一化：加性噪声会把特征方差整体抬高，
                # 只减均值不够，除以标准差后干净/带噪样本能落到同一量级。
                mfcc = mfcc / (mfcc.std(axis=0, keepdims=True) + 1e-3)
        if self.use_delta:
            mfcc = np.concatenate([mfcc, self._delta(mfcc)], axis=1)
        return np.ascontiguousarray(mfcc, dtype=np.float32)

    # ------------------------------------------------------------ 能量（VAD/加权用）
    def frame_energy_db(self, pcm_int16: bytes | np.ndarray) -> np.ndarray:
        x = np.frombuffer(pcm_int16, dtype="<i2") if isinstance(pcm_int16, (bytes, bytearray)) \
            else np.asarray(pcm_int16)
        return self._energy_from_float(x.astype(np.float32) / 32768.0)


def trim_silence(feats: np.ndarray, energy_db: np.ndarray, margin_db: float = 15.0) -> np.ndarray:
    """按能量剪掉首尾静音帧（特征和能量帧率一致）。"""
    if feats.shape[0] == 0 or energy_db.size == 0:
        return feats
    n = min(feats.shape[0], energy_db.size)
    feats, energy_db = feats[:n], energy_db[:n]
    threshold = max(float(np.max(energy_db)) - margin_db, -50.0)
    voiced = np.where(energy_db > threshold)[0]
    if voiced.size == 0:
        return feats[:0]
    start, end = int(voiced[0]), int(voiced[-1]) + 1
    pad = 4
    start = max(0, start - pad)
    end = min(feats.shape[0], end + pad)
    return feats[start:end]
