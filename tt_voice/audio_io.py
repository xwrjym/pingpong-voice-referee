"""音频输入：麦克风实时流 或 wav 文件回放（离线回归测试用）。"""

from __future__ import annotations

import os
import queue
import wave
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class DeviceInfo:
    index: int
    name: str
    channels: int
    default_samplerate: float
    host_api: str = ""


def list_input_devices() -> list[DeviceInfo]:
    import sounddevice as sd

    out: list[DeviceInfo] = []
    hostapis = sd.query_hostapis()
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            api_index = int(dev.get("hostapi", -1))
            api_name = hostapis[api_index]["name"] if 0 <= api_index < len(hostapis) else ""
            out.append(DeviceInfo(
                index=idx,
                name=str(dev.get("name", "")).strip(),
                channels=int(dev["max_input_channels"]),
                default_samplerate=float(dev.get("default_samplerate", 0.0)),
                host_api=str(api_name),
            ))
    return out


MIC_KEYWORDS = ("麦克风", "mic", "microphone", "input")


def pick_input_device(sample_rate: int = 16000, device: int | None = None) -> int | None:
    """挑一个能用的输入设备。

    很多 Windows 机器没有设置"默认输入设备"（PortAudio 会看到 -1），
    这时按名字找第一个像麦克风的设备；找不到就报错让用户自己 --device。
    """
    import sounddevice as sd

    if device is not None:
        return device
    try:
        sd.query_devices(kind="input")
        return None
    except Exception:
        pass
    candidates = [dev for dev in list_input_devices()
                  if any(word in dev.name.lower() for word in MIC_KEYWORDS)]
    if candidates:
        chosen = candidates[0]
        print(f"[音频] 系统没有默认输入设备，自动选用 [{chosen.index}] {chosen.name}")
        return chosen.index
    raise RuntimeError("没有可用的默认输入设备，请先运行 `python -m tt_voice.cli devices` "
                       "再用 --device 指定输入设备编号")


class MicSource:
    """麦克风 -> 定长 int16 单声道 PCM 帧。

    用回调式流而不是阻塞 read()：部分驱动（例如 WDM-KS）不支持阻塞读取，
    PortAudio 会直接报 "Blocking API not supported yet"。回调里只做入队，
    队列满就丢最旧的，保证采集线程不会被下游拖死。
    """

    def __init__(self, sample_rate: int = 16000, block_size: int = 1600,
                 device: int | None = None, max_queue: int = 40):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device
        self.max_queue = max_queue
        self._stream = None
        self._queue: "queue.Queue[bytes]" = queue.Queue(maxsize=max_queue)
        self.overflows = 0
        self.dropped = 0
        self.channels = 1
        self.device_rate = sample_rate
        self.resampler: StreamResampler | None = None

    def __enter__(self) -> "MicSource":
        import sounddevice as sd

        self.device = pick_input_device(self.sample_rate, self.device)
        info = sd.query_devices(self.device, "input")
        # 声卡常常只有双声道输入，WDM-KS 下 PortAudio 不能自动降混，这里自己平均成单声道
        self.channels = max(1, min(2, int(info["max_input_channels"])))
        device_default = int(round(float(info.get("default_samplerate") or self.sample_rate)))
        try:
            self._open_stream(sd, self.sample_rate, self.block_size)
            self.device_rate = self.sample_rate
        except Exception:
            # 只支持 44.1k/48k 的阵列麦很常见：按原生采样率开，自己流式重采样
            device_block = max(1, int(round(device_default * self.block_size / self.sample_rate)))
            self._open_stream(sd, device_default, device_block)
            self.device_rate = device_default
            self.resampler = StreamResampler(device_default, self.sample_rate, device_block)
            print(f"[音频] 设备不支持 {self.sample_rate}Hz，改用原生 {device_default}Hz + 流式重采样")
        self._stream.start()
        return self

    def _open_stream(self, sd, samplerate: int, blocksize: int) -> None:
        self._stream = sd.RawInputStream(
            samplerate=samplerate,
            blocksize=blocksize,
            device=self.device,
            channels=self.channels,
            dtype="int16",
            callback=self._on_audio,
        )

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            self.overflows += 1
        raw = bytes(indata)
        if self.channels > 1 or self.resampler is not None:
            data = np.frombuffer(raw, dtype="<i2")
            if self.channels > 1:
                data = data.reshape(-1, self.channels).mean(axis=1)
            if self.resampler is not None:
                data = self.resampler.process(data)
            if data.size == 0:
                return
            raw = np.ascontiguousarray(data, dtype=np.int16).tobytes()
        try:
            self._queue.put_nowait(raw)
        except queue.Full:
            self.dropped += 1

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __iter__(self) -> Iterator[bytes]:
        assert self._stream is not None, "MicSource 需要在 with 语句中使用"
        while True:
            yield self._queue.get()


def read_wav_mono16(path: str) -> tuple[np.ndarray, int]:
    """读 wav -> (int16 单声道数组, 采样率)。"""
    with wave.open(path, "rb") as fh:
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        rate = fh.getframerate()
        frames = fh.readframes(fh.getnframes())
    if width != 2:
        raise ValueError(f"{path}: 只支持 16bit PCM wav，当前 {width * 8}bit")
    data = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return data, rate


def resample_int16(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """重采样。整数倍下采样用滑动平均抗混叠，其余走 FFT 插值，不引入额外依赖。"""
    if sr_in == sr_out or data.size == 0:
        return data.astype(np.int16)

    if sr_in % sr_out == 0 and sr_in // sr_out <= 8:
        factor = sr_in // sr_out
        kernel = np.ones(factor, dtype=np.float32) / factor
        smoothed = np.convolve(data.astype(np.float32), kernel, mode="same")
        return smoothed[::factor].astype(np.int16)

    n_out = int(round(data.size * sr_out / sr_in))
    spectrum = np.fft.rfft(data.astype(np.float64))
    n_keep = min(len(spectrum), n_out // 2 + 1)
    resized = np.zeros(n_out // 2 + 1, dtype=np.complex128)
    resized[:n_keep] = spectrum[:n_keep]
    resized *= n_out / max(1, data.size)
    out = np.fft.irfft(resized, n=n_out)
    return np.clip(np.round(out), -32768, 32767).astype(np.int16)


class StreamResampler:
    """流式重采样：把设备原生采样率的音频块连续转成目标采样率。

    整机上的阵列麦常常只支持 48k（16k 是三步抽取，用滑动平均抗混叠），
    也有 44.1k 这种非整数比例（走线性插值）。两种都要**跨块保持相位**，
    否则每块首尾会接错，特征跟着抖，识别就不稳了。
    """

    def __init__(self, sr_in: int, sr_out: int, block_size: int = 0):
        self.sr_in = sr_in
        self.sr_out = sr_out
        self.ratio = sr_in / sr_out
        factor = sr_in // sr_out if sr_in % sr_out == 0 else 0
        self._integer = bool(factor) and (block_size == 0 or block_size % factor == 0)
        self._factor = factor
        self._fir_carry = np.zeros(0, dtype=np.float32)
        self._buf = np.zeros(0, dtype=np.float32)
        self._next_pos = 0.0

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        if abs(self.ratio - 1.0) < 1e-9:
            return np.clip(np.round(block), -32768, 32767).astype(np.int16)
        if self._integer:
            return self._decimate(block)
        return self._linear(block)

    def _decimate(self, block: np.ndarray) -> np.ndarray:
        factor = self._factor
        merged = np.concatenate([self._fir_carry, block])
        if merged.size < factor:
            self._fir_carry = merged
            return np.zeros(0, dtype=np.int16)
        kernel = np.ones(factor, dtype=np.float32) / factor
        filtered = np.convolve(merged, kernel, mode="valid")[::factor]
        self._fir_carry = merged[-(factor - 1):] if factor > 1 else np.zeros(0, np.float32)
        return np.clip(np.round(filtered), -32768, 32767).astype(np.int16)

    def _linear(self, block: np.ndarray) -> np.ndarray:
        self._buf = np.concatenate([self._buf, block])
        if self._buf.size < 2:
            return np.zeros(0, dtype=np.int16)
        usable_end = self._buf.size - 1
        if self._next_pos > usable_end:
            return np.zeros(0, dtype=np.int16)
        n_out = int(np.floor((usable_end - self._next_pos) / self.ratio)) + 1
        if n_out <= 0:
            return np.zeros(0, dtype=np.int16)
        positions = self._next_pos + self.ratio * np.arange(n_out)
        out = np.interp(positions, np.arange(self._buf.size), self._buf)
        self._next_pos = positions[-1] + self.ratio
        keep_from = int(np.floor(self._next_pos)) - 1
        if keep_from > 0:
            self._buf = self._buf[keep_from:]
            self._next_pos -= keep_from
        return np.clip(np.round(out), -32768, 32767).astype(np.int16)


class WavSource:
    """wav 文件 -> 与麦克风等长的 PCM 帧序列（用于离线回归测试）。"""

    def __init__(self, path: str, sample_rate: int = 16000, block_size: int = 1600,
                 lead_silence_s: float = 0.0, tail_silence_s: float = 0.6):
        self.path = path
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.lead_silence_s = lead_silence_s
        self.tail_silence_s = tail_silence_s

    def frames(self) -> np.ndarray:
        data, rate = read_wav_mono16(self.path)
        data = resample_int16(data, rate, self.sample_rate)
        pad_head = np.zeros(int(self.lead_silence_s * self.sample_rate), dtype=np.int16)
        pad_tail = np.zeros(int(self.tail_silence_s * self.sample_rate), dtype=np.int16)
        return np.concatenate([pad_head, data, pad_tail])

    def __iter__(self) -> Iterator[bytes]:
        data = self.frames()
        for start in range(0, data.size, self.block_size):
            block = data[start:start + self.block_size]
            if block.size < self.block_size:
                block = np.pad(block, (0, self.block_size - block.size))
            yield block.tobytes()

    @property
    def duration_s(self) -> float:
        return self.frames().size / self.sample_rate


def write_wav16(path: str, data: np.ndarray, sample_rate: int = 16000) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(np.asarray(data, dtype="<i2").tobytes())
    return path


def record_utterance(sample_rate: int = 16000, block_size: int = 1600,
                     max_seconds: float = 3.0, silence_stop_s: float = 0.7,
                     device: int | None = None, start_db: float = -42.0,
                     margin_db: float = 8.0) -> np.ndarray:
    """录一句话：说话后停顿 silence_stop_s 自动结束，最长 max_seconds。

    用于现场注册模板和录真实测试集（enroll.py / record_dataset.py 共用）。
    """
    from .vad import EnergyVad

    vad = EnergyVad(sample_rate, start_db=start_db, margin_db=margin_db)
    chunks: list[bytes] = []
    silence = 0.0
    speech_seen = False
    step = block_size / sample_rate
    with MicSource(sample_rate, block_size, device) as source:
        for block in source:
            chunks.append(block)
            if vad.is_speech(block):
                speech_seen = True
                silence = 0.0
            elif speech_seen:
                silence += step
            if (speech_seen and silence >= silence_stop_s) or len(chunks) * step >= max_seconds:
                break
    return np.frombuffer(b"".join(chunks), dtype="<i2")


class UtteranceRecorder:
    """在一条**持续打开**的麦克风流上一段一段地录话。

    两个关键点（真人验收实测出来的）：

    1. 流只开一次。蓝牙耳机每次开流要 0.5~1.5 秒（HFP 链路切换），
       逐条录音时用户一按回车就开口，开头几个字根本还没被采集到；
    2. 起音前保留 `preroll_s` 的环形缓存，VAD 一发现人声就把缓存一起带上，
       不会切掉第一个字（现场表现就是"撤销"被听成"认销"这类怪结果）。

    典型用法：with MicSource(...) as src: rec = UtteranceRecorder(iter(src), ...)
    """

    def __init__(self, blocks, sample_rate: int = 16000, block_size: int = 1600,
                 preroll_s: float = 0.5, start_db: float = -42.0, margin_db: float = 8.0):
        from .vad import EnergyVad

        self._blocks = blocks
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.step = block_size / sample_rate
        self._preroll_blocks = max(1, int(preroll_s / self.step))
        self._ring: "queue.deque[bytes]" = queue.deque(maxlen=self._preroll_blocks)
        self._vad_args = {"start_db": start_db, "margin_db": margin_db}
        self._make_vad = lambda: EnergyVad(sample_rate, **self._vad_args)

    def record(self, max_seconds: float = 4.0, silence_stop_s: float = 0.7,
               min_voiced_s: float = 0.3) -> np.ndarray:
        """等用户开口并录到句尾。

        `min_voiced_s` 要求连续有声至少这么久才算"开始说话"。这条是为了挡掉
        短促的提示音或按键声：0.1 秒的提示音如果被当成语音起点，录音会在用户
        开口之前就结束（真实验收里因此产生了大量 1.3 秒的静音文件）。
        """
        vad = self._make_vad()
        self._ring.clear()
        chunks: list[bytes] = []
        silence = 0.0
        voiced_run = 0.0
        speech_seen = False
        total = 0.0
        while True:
            block = next(self._blocks)
            if not speech_seen:
                self._ring.append(block)
                if vad.is_speech(block):
                    voiced_run += self.step
                    if voiced_run >= min_voiced_s:
                        speech_seen = True
                        chunks = list(self._ring)  # 连起音前的缓存一起要
                        total = len(chunks) * self.step
                        silence = 0.0
                else:
                    voiced_run = 0.0
                continue
            chunks.append(block)
            total += self.step
            if vad.is_speech(block):
                silence = 0.0
            else:
                silence += self.step
            if silence >= silence_stop_s or total >= max_seconds:
                break
        return np.frombuffer(b"".join(chunks), dtype="<i2")
