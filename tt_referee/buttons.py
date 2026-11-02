"""实体按键适配：与语音、视觉共用同一个计分状态机。

按键是"最后一道保险"：麦克风坏了、模型判不出，裁判仍可用按键改分。
"""

from __future__ import annotations

from .scoring import ActionResult, ScoreEngine

BUTTON_ACTIONS = ("point_a", "point_b", "undo", "redo", "confirm", "reject", "reset", "report")


class ButtonAdapter:
    def __init__(self, engine: ScoreEngine):
        self.engine = engine

    def press(self, action: str, side: str | None = None) -> ActionResult:
        if action == "point_a":
            return self.engine.award("A", reason="button:point_a", source="button")
        if action == "point_b":
            return self.engine.award("B", reason="button:point_b", source="button")
        if action == "undo":
            return self.engine.undo()
        if action == "redo":
            return self.engine.redo()
        if action == "confirm":
            return self.engine.confirm_pending(side)
        if action == "reject":
            return self.engine.reject_pending()
        if action == "reset":
            return self.engine.apply_voice_intent("reset")
        if action == "report":
            return ActionResult(ok=True, intent="report", message=self.engine.display(),
                                state=self.engine.snapshot())
        return ActionResult(ok=False, intent=action, reason="unknown_button",
                            message=f"未知按键：{action}", state=self.engine.snapshot())
