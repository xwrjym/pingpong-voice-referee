"""屏幕计分 UI：本地 HTTP 服务 + 单页计分牌。"""

from .server import ScoreServer, state_payload

__all__ = ["ScoreServer", "state_payload"]
