import tempfile
import unittest

from tt_voice.app import Pipeline, STATE_IDLE, STATE_LISTENING
from tt_voice.config import AppConfig
from tt_voice.feedback import NullSpeaker
from tt_voice.score import ScoreBoard


class RecordingSpeaker:
    def __init__(self):
        self.says = []
        self.beeps = []

    def say(self, text):
        self.says.append(text)

    def beep(self, kind="ok"):
        self.beeps.append(kind)

    def close(self):
        pass


def make_pipeline(tmp: str, **kwargs) -> Pipeline:
    cfg = AppConfig.from_dict({"persist_state": "s.json", "event_log": "e.jsonl",
                               "listen_timeout_s": 6.0}, base_dir=tmp)
    return Pipeline(cfg, speaker=NullSpeaker(), log=lambda *_: None, **kwargs)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ttpipe_")

    def test_wake_then_command(self):
        pipe = make_pipeline(self.tmp)
        self.assertEqual(pipe.state, STATE_IDLE)
        wake = pipe.push_text("小裁判", now=0.0)
        self.assertEqual(len(wake), 1)
        self.assertEqual(wake[0].intent, "wake")
        self.assertEqual(pipe.state, STATE_LISTENING)
        results = pipe.push_text("左方加一分", now=0.5)
        self.assertTrue(results[0].ok)
        self.assertEqual(pipe.board.left, 1)
        self.assertEqual(pipe.state, STATE_LISTENING)   # 一次唤醒内可连说多条口令
        pipe.push_text("右方加一分", now=1.0)
        pipe.push_text("右方加一分", now=1.5)
        self.assertEqual(pipe.state, STATE_IDLE)        # 说满 max_commands_per_wake 后退出
        self.assertEqual((pipe.board.left, pipe.board.right), (1, 2))

    def test_command_without_wake_is_ignored(self):
        pipe = make_pipeline(self.tmp)
        self.assertEqual(pipe.push_text("左方加一分", now=0.0), [])
        self.assertEqual(pipe.board.left, 0)
        self.assertEqual(pipe.state, STATE_IDLE)

    def test_reject_then_reask_then_exit(self):
        pipe = make_pipeline(self.tmp)
        pipe.push_text("小裁判", now=0.0)
        first = pipe.push_text("今天天气不错", now=0.5)
        self.assertTrue(first[0].requires_reask)
        self.assertEqual(pipe.state, STATE_LISTENING)
        second = pipe.push_text("帮我把灯关掉", now=1.0)
        self.assertFalse(second[0].ok)
        self.assertEqual(pipe.state, STATE_IDLE)
        self.assertEqual(pipe.stats["commands_rejected"], 2)

    def test_timeout_returns_to_idle(self):
        pipe = make_pipeline(self.tmp)
        pipe.push_text("小裁判", now=0.0)
        results = pipe.tick(now=7.0)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].reason, "timeout")
        self.assertEqual(pipe.state, STATE_IDLE)
        self.assertEqual(pipe.stats["timeouts"], 1)

    def test_reset_flow_through_pipeline(self):
        pipe = make_pipeline(self.tmp)
        pipe.push_text("小裁判", now=0.0)
        pipe.push_text("左方加一分", now=0.5)
        pipe.push_text("重置比分", now=1.0)
        self.assertTrue(pipe.board.awaiting_reset_confirm)
        pipe.push_text("确认", now=1.5)
        self.assertEqual((pipe.board.left, pipe.board.right), (0, 0))

    def test_reset_confirm_times_out(self):
        pipe = make_pipeline(self.tmp)
        pipe.push_text("小裁判", now=0.0)
        pipe.push_text("重置比分", now=0.5)
        self.assertTrue(pipe.board.awaiting_reset_confirm)
        pipe.tick(now=100.0)
        self.assertFalse(pipe.board.awaiting_reset_confirm)

    def test_silent_reject_option(self):
        """非口令在 --no-reask 模式下完全不反馈，并退出聆听。"""
        tmp = tempfile.mkdtemp(prefix="ttsilent_")
        cfg = AppConfig.from_dict({"persist_state": "s.json", "event_log": "e.jsonl",
                                   "reask_on_reject": False}, base_dir=tmp)
        speaker = RecordingSpeaker()
        pipe = Pipeline(cfg, speaker=speaker, log=lambda *_: None)
        pipe.push_text("小裁判", now=0.0)
        result = pipe.push_text("今天天气不错", now=0.5)
        self.assertFalse(result[0].ok)
        self.assertEqual(speaker.says, ["在"])       # 只有唤醒反馈
        self.assertEqual(pipe.state, STATE_IDLE)
        self.assertEqual(pipe.board.snapshot()["left"], 0)
        self.assertEqual(pipe.stats["silent_rejects"], 1)


if __name__ == "__main__":
    unittest.main()
