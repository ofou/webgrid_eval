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


def _get_max_xy(state: Any) -> int:
    """Get max pixel coordinate based on state's canvas_size."""
    return getattr(state, "canvas_size", DEFAULT_CANVAS_SIZE) - 1


# ----------------------------
# helpers
# ----------------------------
def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def _ensure_cursor_pixel(state: Any) -> None:
    if getattr(state, "cursor_x", None) is None or getattr(state, "cursor_y", None) is None:
        canvas_size = getattr(state, "canvas_size", DEFAULT_CANVAS_SIZE)
        cx, cy = cell_center_pixel(state.cursor_row, state.cursor_col, state.grid_side, canvas_size)
        state.cursor_x, state.cursor_y = cx, cy


def _set_cursor_pixel(state: Any, x: int, y: int) -> None:
    max_xy = _get_max_xy(state)
    x = _clamp(x, 0, max_xy)
    y = _clamp(y, 0, max_xy)
    canvas_size = getattr(state, "canvas_size", DEFAULT_CANVAS_SIZE)
    row, col = pixel_to_cell(x, y, state.grid_side, canvas_size)
    state.move_cursor(row, col)
    state.cursor_x, state.cursor_y = x, y


def _hud(state: Any, last_click: bool | None = None) -> dict[str, Any]:
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
        bps = (net / 60.0) * math.log2(state.grid_size)
    return {
        "time": f"{mm:02d}:{ss:02d}",
        "bps": bps,
        "ntpm": ntpm,
        "grid": f"{state.grid_side}×{state.grid_side}",
        "last_click_correct": last_click,
    }


def _hud_line(state: Any) -> str:
    """Return a compact HUD string (no target position - that would leak the answer to the model).

    Format: 'MM:SS X.XX BPS N NTPM WxH'
    """
    elapsed_ms = (
        (time.time() - state.start_time) * 1000
        if getattr(state, "start_time", None) is not None
        else 0
    )
    t_s = int(elapsed_ms // 1000)
    mm, ss = t_s // 60, t_s % 60
    net = state.score - state.incorrect_count
    bps = 0.0
    if net > 0:
        bps = (net / 60.0) * math.log2(state.grid_size)

    return f"{mm:02d}:{ss:02d} {bps:.2f} BPS {net} NTPM {state.grid_side}×{state.grid_side}"


def _packet(state: Any, last_click: bool | None = None) -> dict[str, Any]:
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


def _img_message(b64: str, save_path: str | None, hud_text: str | None = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        },
    ]
    if hud_text:
        content.append({"type": "text", "text": hud_text})
    msg: dict[str, Any] = {"role": "user", "content": content}
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
            "description": "Move cursor by (dx, dy). Positive dx=right, dy=down.",
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
    state: Any,
    screen_save_path: str | None = None,
    click_save_path: str | None = None,
) -> tuple[str, list[dict] | None]:
    """Execute one tool (screen, mouse_move, mouse_click) and return (content, extra_messages)."""
    _ensure_cursor_pixel(state)

    if name == "screen":
        b64 = render_grid_screenshot(state, save_path=screen_save_path)
        hud = _hud_line(state)
        return (
            hud,
            [_img_message(b64, screen_save_path, hud)],
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

        # Return screenshot after move for visual feedback (like Neuralink)
        b64 = render_grid_screenshot(state, save_path=screen_save_path)
        return "OK", [_img_message(b64, screen_save_path, None)]

    if name == "mouse_click":
        assert state.cursor_x is not None and state.cursor_y is not None
        x, y = state.cursor_x, state.cursor_y
        canvas_size = getattr(state, "canvas_size", DEFAULT_CANVAS_SIZE)
        row, col = pixel_to_cell(x, y, state.grid_side, canvas_size)

        correct, _data = state.click_at(row, col)

        b64 = render_grid_screenshot(
            state, save_path=click_save_path, last_click_incorrect=not correct
        )
        content = json.dumps(
            {
                "correct": correct,
                "ntpm": state.score - state.incorrect_count,
                "grid_side": state.grid_side,
            }
        )
        return (
            content,
            [_img_message(b64, click_save_path, hud_text=None)],
        )

    return f"Unknown tool: {name}", None
