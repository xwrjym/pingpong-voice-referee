"""把 ../referee/tt_referee 同步到本工程，供语音模块共用计分状态机。

    python tools/sync_referee.py

语音工程的 ScoreBoard 优先使用这份共用实现；同步保证两个工程打分规则一致。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve()
    project = here.parents[1]                  # voice/
    source = here.parents[2] / "referee" / "tt_referee"
    target = project / "tt_referee"
    if not source.is_dir():
        print(f"找不到裁判工程：{source}")
        return 1
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    count = sum(1 for _ in target.rglob("*.py"))
    print(f"已同步 {count} 个 Python 文件：{target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
