"""反馈：蜂鸣 + 可选的中文 TTS 播报。

TTS 走 Windows 自带的 SAPI（离线、零额外依赖），在后台线程里播，
不阻塞音频采集。将来换成开发板上的播放器时，只要实现 Speaker 接口即可。
"""

from __future__ import annotations

import base64
import os
import queue
import subprocess
import sys
import threading


class Speaker:
    def say(self, text: str) -> None:
        pass

    def beep(self, kind: str = "ok") -> None:
        pass

    def close(self) -> None:
        pass


class ConsoleSpeaker(Speaker):
    def __init__(self, verbose_beep: bool = False):
        self.verbose_beep = verbose_beep

    def say(self, text: str) -> None:
        print(f"[播报] {text}", flush=True)

    def beep(self, kind: str = "ok") -> None:
        if self.verbose_beep:
            print(f"[提示音] {kind}", flush=True)


class NullSpeaker(Speaker):
    """批量评估/单元测试用：什么都不做。"""


_BEEPS = {
    "wake": [(880, 90)],
    "ok": [(1320, 70)],
    "ready": [(660, 60), (880, 60)],
    "reject": [(320, 160)],
    "timeout": [(440, 90), (440, 90)],
    "error": [(240, 200), (240, 200)],
}


class WindowsBeepSpeaker(Speaker):
    """用 winsound 发出不同音高，使用户不看屏幕也能分辨状态。"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._winsound = None
        if enabled and sys.platform.startswith("win"):
            try:
                import winsound

                self._winsound = winsound
            except Exception:
                self._winsound = None

    def beep(self, kind: str = "ok") -> None:
        if not self._winsound or not self.enabled:
            return
        for freq, dur in _BEEPS.get(kind, _BEEPS["ok"]):
            try:
                self._winsound.Beep(freq, dur)
            except Exception:
                return


def _ps_encoded(script: str) -> str:
    """PowerShell -EncodedCommand 用 UTF-16LE base64，避免中文被转义弄乱。"""
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def _sapi_script(text: str, rate: int = 0, volume: int = 100, out_wav: str | None = None,
                 voice: str | None = None) -> str:
    safe_text = text.replace("'", "''")
    lines = [
        "Add-Type -AssemblyName System.Speech",
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
    ]
    if voice:
        lines.append(f"$s.SelectVoice('{voice.replace(chr(39), chr(39) * 2)}')")
    else:
        lines.append("$zh = $s.GetInstalledVoices() | Where-Object "
                     "{ $_.VoiceInfo.Culture.Name -like 'zh*' } | Select-Object -First 1")
        lines.append("if ($zh) { $s.SelectVoice($zh.VoiceInfo.Name) }")
    lines += [
        f"$s.Rate = {rate}",
        f"$s.Volume = {volume}",
    ]
    if out_wav:
        target = out_wav.replace("'", "''")
        lines += [
            ("$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
             "16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
             "[System.Speech.AudioFormat.AudioChannel]::Mono)"),
            f"$s.SetOutputToWaveFile('{target}', $fmt)",
        ]
    else:
        lines.append("$s.SetOutputToDefaultAudioDevice()")
    lines += [f"$s.Speak('{safe_text}')", "$s.Dispose()"]
    return "\n".join(lines)


def sapi_run(script: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-EncodedCommand", _ps_encoded(script)],
        capture_output=True,
        timeout=timeout,
    )


def sapi_say_to_wav(text: str, out_wav: str, rate: int = 0, volume: int = 100,
                    voice: str | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
    result = sapi_run(_sapi_script(text, rate=rate, volume=volume, out_wav=out_wav, voice=voice))
    if result.returncode != 0:
        raise RuntimeError(
            f"SAPI 合成失败：{result.stderr.decode('gbk', errors='replace')[:400]}"
        )


class SapiTtsSpeaker(Speaker):
    """后台线程播报；队列满则丢弃，保证音频采集不被拖住。"""

    def __init__(self, rate: int = 0, volume: int = 100, max_queue: int = 3):
        self.rate = rate
        self.volume = volume
        self.dropped = 0
        self._queue: "queue.Queue[str | None]" = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._speaking = threading.Event()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            self._speaking.set()
            try:
                sapi_run(_sapi_script(item, rate=self.rate, volume=self.volume), timeout=30.0)
            except Exception:
                pass
            finally:
                self._speaking.clear()

    @property
    def is_speaking(self) -> bool:
        """是否正在播报（含排队中）。上层据此暂停识别，避免机器人听到自己说话。"""
        return self._speaking.is_set() or not self._queue.empty()

    def say(self, text: str) -> None:
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            self.dropped += 1

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass


class MultiSpeaker(Speaker):
    def __init__(self, speakers: list[Speaker]):
        self.speakers = speakers

    def say(self, text: str) -> None:
        for spk in self.speakers:
            spk.say(text)

    def beep(self, kind: str = "ok") -> None:
        for spk in self.speakers:
            spk.beep(kind)

    def close(self) -> None:
        for spk in self.speakers:
            spk.close()

    @property
    def is_speaking(self) -> bool:
        return any(getattr(spk, "is_speaking", False) for spk in self.speakers)


def build_speaker(say: bool = False, beep: bool = True, rate: int = 0) -> Speaker:
    speakers: list[Speaker] = []
    if say and sys.platform.startswith("win"):
        speakers.append(SapiTtsSpeaker(rate=rate))
    if beep:
        speakers.append(WindowsBeepSpeaker())
    speakers.append(ConsoleSpeaker())
    return MultiSpeaker(speakers)
