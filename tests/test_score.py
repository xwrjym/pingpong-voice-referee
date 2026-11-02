import json
import os
import tempfile
import unittest

from tt_voice.config import AppConfig
from tt_voice.score import ScoreBoard


def make_board(tmp: str) -> ScoreBoard:
    cfg = AppConfig.from_dict({
        "persist_state": "state.json",
        "event_log": "events.jsonl",
    }, base_dir=tmp)
    return ScoreBoard(cfg)


class TestScoreBoard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ttscore_")
        self.board = make_board(self.tmp)

    def test_score_and_undo(self):
        self.board.apply("score_left")
        self.board.apply("score_right")
        self.board.apply("score_right")
        self.assertEqual((self.board.left, self.board.right), (1, 2))
        result = self.board.apply("undo")
        self.assertTrue(result.ok)
        self.assertEqual((self.board.left, self.board.right), (1, 1))
        self.assertEqual(self.board.apply("undo").ok, True)
        self.assertEqual((self.board.left, self.board.right), (1, 0))

    def test_undo_when_empty(self):
        result = self.board.apply("undo")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "nothing_to_undo")

    def test_reset_requires_two_steps(self):
        self.board.apply("score_left")
        first = self.board.apply("reset")
        self.assertTrue(first.ok)
        self.assertTrue(self.board.awaiting_reset_confirm)
        self.assertEqual((self.board.left, self.board.right), (1, 0))
        second = self.board.apply("reset")
        self.assertTrue(second.ok)
        self.assertEqual((self.board.left, self.board.right), (0, 0))
        self.assertFalse(self.board.awaiting_reset_confirm)

    def test_reset_then_other_intent_cancels(self):
        self.board.apply("reset")
        self.board.apply("score_left")
        self.assertFalse(self.board.awaiting_reset_confirm)
        self.assertEqual((self.board.left, self.board.right), (1, 0))

    def test_confirm_pending_point(self):
        self.board.propose_point("right", confidence=0.9)
        result = self.board.apply("confirm")
        self.assertTrue(result.ok)
        self.assertEqual((self.board.left, self.board.right), (0, 1))
        self.assertIsNone(self.board.pending)

    def test_confirm_without_pending(self):
        result = self.board.apply("confirm")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "nothing_to_confirm")

    def test_report(self):
        self.board.apply("score_left")
        self.assertIn("左方1比0右方", self.board.apply("report").message)

    def test_persistence(self):
        self.board.apply("score_left")
        self.board.apply("score_left")
        path = self.board.save()
        self.assertTrue(os.path.exists(path))
        other = make_board(self.tmp)
        self.assertTrue(other.load())
        self.assertEqual((other.left, other.right), (2, 0))
        self.assertTrue(other.apply("undo").ok)
        self.assertEqual((other.left, other.right), (1, 0))

    def test_event_log_written(self):
        self.board.apply("score_left", raw_text="左方加一分", score=0.97)
        self.board.apply("undo", raw_text="撤销", score=0.95)
        log_path = self.board.cfg.resolve(self.board.cfg.event_log)
        with open(log_path, "r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["intent"], "score_left")
        self.assertEqual(rows[0]["raw_text"], "左方加一分")
        self.assertEqual(rows[0]["after"], {"left": 1, "right": 0})
        self.assertEqual(rows[1]["after"], {"left": 0, "right": 0})

    def test_self_side_mapping(self):
        """个人自用："我加一分"要落到本人所在的一侧。"""
        cfg = AppConfig.from_dict({"self_side": "right"}, base_dir=self.tmp)
        board = ScoreBoard(cfg)
        board.apply("score_self")
        board.apply("score_other")
        self.assertEqual((board.left, board.right), (1, 1))
        board.apply("score_self")
        self.assertEqual((board.left, board.right), (1, 2))

    def test_self_side_default_left(self):
        board = make_board(self.tmp)
        board.apply("score_self")
        self.assertEqual((board.left, board.right), (1, 0))


if __name__ == "__main__":
    unittest.main()
