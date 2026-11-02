"""文本可视化：没有 matplotlib 也能看轨迹与事件。

输出一张 y(t) 的 ASCII 图（每行一档高度，每列一段时间），
落台/擦网/出界用字母标在对应时间列，方便快速判断"是没检测到，还是判错了"。
"""

from __future__ import annotations

from .schema import Event, Observation

MARKERS = {"BOUNCE": "B", "NET_HIT": "N", "OUT": "O"}


def ascii_timeline(observations: list[Observation], events: list[Event],
                   width: int = 76, height: int = 11) -> str:
    visible = [o for o in observations if o.ball.visible and o.ball.center_px]
    if not visible:
        return "（没有可见的球观测）"
    times = [o.timestamp_s for o in visible]
    ys = [o.ball.center_px[1] for o in visible]
    t0, t1 = min(times), max(times)
    y0, y1 = min(ys), max(ys)
    span_t = max(1e-6, t1 - t0)
    span_y = max(1e-6, y1 - y0)
    grid = [[" "] * width for _ in range(height)]
    for t, y in zip(times, ys):
        col = int((t - t0) / span_t * (width - 1))
        row = int((y - y0) / span_y * (height - 1))
        grid[row][col] = "."
    for event in events:
        if event.timestamp_s < t0 or event.timestamp_s > t1:
            continue
        col = int((event.timestamp_s - t0) / span_t * (width - 1))
        marker = MARKERS.get(event.type, "?")
        row = 0 if event.type != "BOUNCE" else height - 1
        # 尽量放在轨迹附近的空白处
        for candidate in (row, max(0, row - 1), min(height - 1, row + 1)):
            if grid[candidate][col] in (" ", "."):
                grid[candidate][col] = marker
                break
    lines = [f"y(t)  {t0:.2f}s -> {t1:.2f}s   （. 球观测，B 落台，N 擦网，O 出界）"]
    for row in grid:
        lines.append("  |" + "".join(row))
    lines.append("  +" + "-" * width)
    lines.append("   " + " ".join(f"{e.timestamp_s:.2f}s:{MARKERS.get(e.type, '?')}"
                                  for e in events[:12]))
    return "\n".join(lines)
