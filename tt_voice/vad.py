"""极轻量能量 VAD：只用来判断「唤醒后有没有说话」和做超时提示。

刻意不用神经网络 VAD：唤醒链路要在 RK3588 上常驻运行，
一个自适应噪声底 + 能量门限就够支撑 UI 反馈了。
"""

from __future__ import annotations

import numpy as np


class EnergyVad:
    def __init__(self, sample_rate: int = 16000, start_db: float = -42.0,
                 margin_db: float = 8.0, frame_ms: int = 30):
        self.sample_rate = sample_rate
        self.start_db = start_db
        self.margin_db = margin_db
        self.frame_size = max(1, int(sample_rate * frame_ms / 1000))
        self.noise_db = float(start_db)
        self._initialized = False

    @staticmethod
    def _db(pcm: np.ndarray) -> float:
        if pcm.size == 0:
            return -100.0
        rms = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2)))
        return 20.0 * np.log10(max(rms, 1e-6))

    def frame_db(self, pcm16: bytes) -> list[float]:
        pcm = np.frombuffer(pcm16, dtype="<i2")
        if pcm.size == 0:
            return [-100.0]
        return [
            self._db(pcm[i:i + self.frame_size])
            for i in range(0, pcm.size, self.frame_size)
        ]

    def is_speech(self, pcm16: bytes) -> bool:
        values = self.frame_db(pcm16)
        peak = max(values)
        if not self._initialized:
            self._initialized = True
            if peak > self.start_db:
                # 一上来就在说话（例如用户按完回车立刻开口）：先按"有语音"处理，
                # 噪声底退回配置值，否则这段话会被当成噪声、起音整段丢掉。
                self.noise_db = self.start_db
                return True
            # 起手是静音：就拿它当噪声底，这是最常见也最准的情况
            self.noise_db = peak
            return False
        threshold = max(self.noise_db + self.margin_db, -55.0)
        speech = peak > threshold
        if not speech:
            self.noise_db = 0.95 * self.noise_db + 0.05 * peak
        return speech
