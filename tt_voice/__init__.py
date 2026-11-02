"""乒乓球裁判终端 - 离线语音唤醒与固定口令识别。

分层设计（自下而上）：

    audio_io / vad        ->  麦克风或 wav 文件 -> 16k 单声道 PCM 帧
    engines/vosk_engine   ->  声学层：受限词表解码，输出候选文本 + 置信度
    textnorm / matcher    ->  纠错层：拼音近音对齐 + 分数/边界拒识
    parser                ->  语义层：把文本映射成意图（intent）
    score                 ->  比分状态机（加/减/撤销/重置/确认 + 落盘 + 日志）
    app                   ->  状态机：待唤醒 -> 聆听 -> 执行 -> 反馈
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
