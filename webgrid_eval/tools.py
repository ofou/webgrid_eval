from __future__ import annotations

import json
import math
import os
import time
from typing import Any

from .screenshot import (
    DEFAULT_CANVAS_SIZE,
    cell_center_pixel,
    pixel_to_cell,
    render_grid_screenshot,
)


def _get_max_xy(state) -> int:
    """Get max pixel coordinate based on state's canvas_size."""
    return getattr(state, "canvas_size", DEFAULT_CANVAS_SIZE) - 1


# ----------------------------
# helpers
# ----------------------------
def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def _ensure_cursor_pixel(state) -> None:
    if getattr(state, "cursor_x", None) is None or getattr(state, "cursor_y", None) is None:
        canvas_size = getattr(state, "canvas_size", DEFAULT_CANVAS_SIZE)
        cx, cy = cell_center_pixel(state.cursor_row, state.cursor_col, state.grid_side, canvas_size)
        state.cursor_x, state.cursor_y = cx, cy


def _set_cursor_pixel(state, x: int, y: int) -> None:
    max_xy = _get_max_xy(state)
    x = _clamp(x, 0, max_xy)
    y = _clamp(y, 0, max_xy)
    canvas_size = getattr(state, "canvas_size", DEFAULT_CANVAS_SIZE)
    row, col = pixel_to_cell(x, y, state.grid_side, canvas_size)
    state.move_cursor(row, col)
    state.cursor_x, state.cursor_y = x, y


def _hud(state, last_click: bool | None = None) -> dict[str, Any]:
    elapsed_ms = (
        (time.time() - state.start_time) * 1000
        if getattr(state, "start_time", None) is not None
        else 0
    )
    t_s = int(elapsed_ms // 1000)
    mm, ss = t_s // 60, t_s % 60
    net = state.score - state.incorrect_count
    ntpm = float(net)
    bps = 0.0
    if net > 0:
        bps = (net / 60.0) * math.log2(state.grid_size**2 - 1)
    return {
        "time": f"{mm:02d}:{ss:02d}",
        "bps": bps,
        "ntpm": ntpm,
        "grid": f"{state.grid_side}×{state.grid_side}",
        "last_click_correct": last_click,
    }


def _packet(state, last_click: bool | None = None) -> dict[str, Any]:
    _ensure_cursor_pixel(state)
    canvas_size = getattr(state, "canvas_size", DEFAULT_CANVAS_SIZE)
    return {
        "hud": _hud(state, last_click=last_click),
        "cursor": {
            "x": int(state.cursor_x),
            "y": int(state.cursor_y),
            "row": int(state.cursor_row),
            "col": int(state.cursor_col),
        },
        "target": {
            "row": int(getattr(state, "target_row", -1)),
            "col": int(getattr(state, "target_col", -1)),
        },
        "grid_side": int(state.grid_side),
        "size_px": int(canvas_size),
    }


def _img_message(b64: str, save_path: str | None) -> dict[str, Any]:
    msg = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
            # TODO: add the time and score to the message
        ],
    }
    if save_path:
        msg["_screenshot_filename"] = os.path.basename(save_path)
    return msg


def _require_int(args: dict[str, Any], k: str) -> tuple[int | None, str | None]:
    if k not in args:
        return None, f"Error: missing '{k}'."
    try:
        return int(args[k]), None
    except (TypeError, ValueError):
        return None, f"Error: '{k}' must be an integer."


TOOLS_MOUSELIKE = [
    {
        "type": "function",
        "function": {
            "name": "screen",
            "description": "Return current HUD + screenshot (like looking at your monitor).",
            "strict": True,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mouse_move",
            "description": "Move cursor by relative mouse deltas (dx, dy). Positive dx=right, positive dy=down.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "dx": {
                        "type": "integer",
                        "description": "Delta x pixels (can be negative).",
                    },
                    "dy": {
                        "type": "integer",
                        "description": "Delta y pixels (can be negative).",
                    },
                },
                "required": ["dx", "dy"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mouse_click",
            "description": "Click at the current cursor position.",
            "strict": True,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

TOOLS_OPENAI = TOOLS_MOUSELIKE


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    state,
    screen_save_path: str | None = None,
    click_save_path: str | None = None,
) -> tuple[str, list[dict] | None]:
    _ensure_cursor_pixel(state)

    if name == "screen":
        pkt = _packet(state, last_click=None)
        pkt["event"] = "screen"
        b64 = render_grid_screenshot(state, save_path=screen_save_path)
        # todo: add the time and score to the packet
        return (
            json.dumps(pkt, separators=(",", ":")),
            [_img_message(b64, screen_save_path)],
        )

    if name == "mouse_move":
        dx, err = _require_int(arguments, "dx")
        if err:
            return err, None
        dy, err = _require_int(arguments, "dy")
        if err:
            return err, None

        assert state.cursor_x is not None and state.cursor_y is not None
        new_x = state.cursor_x + dx
        new_y = state.cursor_y + dy
        _set_cursor_pixel(state, new_x, new_y)

        pkt = _packet(state, last_click=None)
        pkt["event"] = "move"
        return json.dumps(pkt, separators=(",", ":")), None

    if name == "mouse_click":
        assert state.cursor_x is not None and state.cursor_y is not None
        x, y = state.cursor_x, state.cursor_y
        canvas_size = getattr(state, "canvas_size", DEFAULT_CANVAS_SIZE)
        row, col = pixel_to_cell(x, y, state.grid_side, canvas_size)

        correct, _data = state.click_at(row, col)

        pkt = _packet(state, last_click=bool(correct))
        pkt["event"] = "click"
        pkt["correct"] = correct
        pkt["last_click_row"] = state.last_click_row
        pkt["last_click_col"] = state.last_click_col
        pkt["last_click_correct"] = state.last_click_correct
        pkt["last_click_time_ms"] = state.last_click_time_ms

        b64 = render_grid_screenshot(
            state, save_path=click_save_path, last_click_incorrect=not correct
        )
        return (
            json.dumps(pkt, separators=(",", ":")),
            [_img_message(b64, click_save_path)],
        )

    return f"Unknown tool: {name}", None
