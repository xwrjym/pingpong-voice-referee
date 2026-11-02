import json
import tempfile
import unittest
import urllib.request

from tt_referee.ui.server import ScoreServer

from tt_voice.config import AppConfig
from tt_voice.score import ScoreBoard


class TestVoiceUIBridge(unittest.TestCase):
    def test_voice_score_reflects_in_ui_state(self):
        tmp = tempfile.mkdtemp(prefix="ttuibridge_")
        cfg = AppConfig.from_dict({"persist_state": "s.json", "event_log": "e.jsonl"},
                                  base_dir=tmp)
        board = ScoreBoard(cfg, log_events=False)
        server = ScoreServer(board.engine, port=0, allow_demo=False).start()
        try:
            board.apply("score_left", raw_text="左方加一分", score=0.99)
            with urllib.request.urlopen(server.url + "api/state", timeout=5) as response:
                state = json.loads(response.read())
            self.assertEqual(state["points"], {"A": 1, "B": 0})
            self.assertFalse(state["demo"])
            self.assertEqual(state["server"], "A")
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
