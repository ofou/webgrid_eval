"""Create animated GIF by rendering game state with smooth cursor movement.

Usage: python -m webgrid_eval.make_gif [eval_dir] [--speed 1] [--fps 30]

When no eval_dir is provided, processes all subdirectories under eval/ recursively.

Renders the game grid from history.json, animating cursor movement between clicks.
Does NOT use screenshot images - generates all frames programmatically.
"""

import argparse
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import cairosvg
from PIL import Image, ImageDraw

CURSOR_VIEWBOX = 32
CURSOR_HOTSPOT_XY = (10, 16)
CURSOR_DISPLAY_PX = 24

_CURSOR_CACHE: tuple | None = None


def _load_cursor() -> tuple[Any, int, int]:
    global _CURSOR_CACHE
    if _CURSOR_CACHE is not None:
        return _CURSOR_CACHE
    path = Path(__file__).parent / "assets" / "cursor.svg"
    scale = CURSOR_DISPLAY_PX / CURSOR_VIEWBOX
    png_bytes = cairosvg.svg2png(
        url=str(path),
        output_width=int(CURSOR_VIEWBOX * scale),
        output_height=int(CURSOR_VIEWBOX * scale),
    )
    cur = Image.open(BytesIO(png_bytes)).convert("RGBA")
    hx = max(1, int(CURSOR_HOTSPOT_XY[0] * scale))
    hy = max(1, int(CURSOR_HOTSPOT_XY[1] * scale))
    _CURSOR_CACHE = (cur, hx, hy)
    return _CURSOR_CACHE


# Default canvas size for backward compatibility
DEFAULT_CANVAS_SIZE = 512
DEFAULT_GRID_SIDE = 8
K = 0.002


def get_timestamp_ms(filename: str) -> int | None:
    """Extract timestamp in milliseconds from filename like '01234ms.png'."""
    match = re.match(r"(\d+)ms\.png$", filename)
    if match:
        return int(match.group(1))
    return None


def _line_width(side: int, canvas_size: int = DEFAULT_CANVAS_SIZE) -> int:
    return max(1, round(canvas_size * K))


def _get_cell_rect(
    row: int, col: int, side: int, canvas_size: int = DEFAULT_CANVAS_SIZE
) -> tuple[int, int, int, int]:
    """Cell fill rect: same as screenshot.py (Neuralink)."""
    c = canvas_size / side
    r = _line_width(side, canvas_size)
    inset = r / 2
    x0 = col * c + inset
    y0 = row * c + inset
    x1 = (col + 1) * c - inset
    y1 = (row + 1) * c - inset
    return int(x0), int(y0), int(x1), int(y1)


def detect_target_from_screenshot(img_path: Path, side: int) -> tuple[int, int] | None:
    """Detect the blue target cell from a screenshot. Uses same layout as screenshot.py."""
    if not img_path.exists():
        return None

    img = Image.open(img_path)
    pixels = img.load()
    if pixels is None:
        return None
    canvas_size = img.width  # Get actual size from image
    cell_size = canvas_size / side

    for y in range(canvas_size):
        for x in range(canvas_size):
            pixel = pixels[x, y]
            if isinstance(pixel, (list, tuple)) and len(pixel) >= 3:
                r, g, b = pixel[0], pixel[1], pixel[2]
                if (abs(r - 10) < 20 and abs(g - 132) < 20 and abs(b - 255) < 20) or (
                    abs(r - 7) < 20 and abs(g - 100) < 20 and abs(b - 191) < 20
                ):
                    col = int(x / cell_size)
                    row = int(y / cell_size)
                    if 0 <= row < side and 0 <= col < side:
                        return (row, col)

    return None


def normalized_to_pixel(
    nx: float, ny: float, canvas_size: int = DEFAULT_CANVAS_SIZE
) -> tuple[int, int]:
    """Convert normalized (0-1) coordinates to pixel coordinates."""
    x = int(nx * canvas_size)
    y = int(ny * canvas_size)
    return x, y


def normalized_to_cell(nx: float, ny: float, side: int) -> tuple[int, int]:
    """Convert normalized coordinates to grid cell (row, col)."""
    col = int(nx * side)
    row = int(ny * side)
    col = max(0, min(col, side - 1))
    row = max(0, min(row, side - 1))
    return row, col


def interpolate_position(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    progress: float,
) -> tuple[float, float]:
    """Interpolate position with ease-out easing."""
    t = 1 - (1 - progress) ** 2  # Quadratic ease-out
    x = start_x + (end_x - start_x) * t
    y = start_y + (end_y - start_y) * t
    return x, y


def render_frame(
    target_row: int,
    target_col: int,
    cursor_x: int,
    cursor_y: int,
    side: int,
    error_cell: tuple[int, int] | None = None,
    success_cell: tuple[int, int] | None = None,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
) -> Image.Image:
    """Render a single frame (Neuralink style): cells then grid lines on top."""
    white = (255, 255, 255)
    black = (0, 0, 0)
    active_blue = (10, 132, 255)
    hover_gray = (191, 191, 191)
    active_hover_blue = (7, 100, 191)
    error_red = (255, 59, 48)  # iOS-style red
    success_green = (52, 199, 89)  # iOS-style green

    cell_size = canvas_size / side
    cursor_col = int(cursor_x / cell_size)
    cursor_row = int(cursor_y / cell_size)
    cursor_col = max(0, min(cursor_col, side - 1))
    cursor_row = max(0, min(cursor_row, side - 1))
    cursor_cell = (cursor_row, cursor_col)

    img = Image.new("RGB", (canvas_size, canvas_size), white)
    draw = ImageDraw.Draw(img)

    for r in range(side):
        for c in range(side):
            x0, y0, x1, y1 = _get_cell_rect(r, c, side, canvas_size)
            is_target = (r, c) == (target_row, target_col) and target_row >= 0
            is_cursor_cell = (r, c) == cursor_cell
            is_error_cell = error_cell is not None and (r, c) == error_cell
            is_success_cell = success_cell is not None and (r, c) == success_cell

            if is_error_cell:
                fill = error_red
            elif is_success_cell:
                fill = success_green
            elif is_cursor_cell and is_target:
                fill = active_hover_blue
            elif is_cursor_cell:
                fill = hover_gray
            elif is_target:
                fill = active_blue
            else:
                fill = white

            draw.rectangle([x0, y0, x1, y1], fill=fill)

    line_w = _line_width(side, canvas_size)
    for a in range(side + 1):
        y_center = a * cell_size
        x_center = a * cell_size
        y0 = max(0, int(y_center - line_w / 2))
        y1 = min(canvas_size, int(y_center + line_w / 2) + 1)
        draw.rectangle([0, y0, canvas_size, y1], fill=black)
        x0 = max(0, int(x_center - line_w / 2))
        x1 = min(canvas_size, int(x_center + line_w / 2) + 1)
        draw.rectangle([x0, 0, x1, canvas_size], fill=black)

    tip_x, tip_y = cursor_x, cursor_y
    cur_img, hx, hy = _load_cursor()
    px = tip_x - hx
    py = tip_y - hy
    r, g, b, a = cur_img.split()
    img.paste(cur_img.convert("RGB"), (px, py), a)

    return img


def parse_game_events(history_path: Path, canvas_size: int = DEFAULT_CANVAS_SIZE) -> list[dict]:
    """Parse history.json to extract game events from tool calls and responses.

    Supports multiple formats:
    1. targets.json file (preferred - contains target positions without leaking to model)
    2. Legacy JSON packets in tool responses (structured events)
    3. Current format: tool_calls in assistant messages + plain text responses

    Returns list of events with cursor positions, target positions, and click info.
    """
    if not history_path.exists():
        return []

    with open(history_path) as f:
        history = json.load(f)

    # Try loading targets.json for target positions (new format - not leaked to model)
    targets_path = history_path.parent / "targets.json"
    target_events_from_file: list[dict] = []
    if targets_path.exists():
        try:
            with open(targets_path) as f:
                target_events_from_file = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    events: list[dict[str, Any]] = []

    # Collect screenshot timestamps from user messages (for timing)
    screenshot_times = []
    for entry in history:
        if entry.get("role") == "user":
            content = entry.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if isinstance(url, str) and ".png" in url:
                            fn = url.split("/")[-1] if "/" in url else url
                            ts = get_timestamp_ms(fn)
                            if ts is not None:
                                screenshot_times.append(ts)

    # First, try legacy JSON packet format
    for i, entry in enumerate(history):
        if entry.get("role") != "tool":
            continue
        content = entry.get("content", "")
        if not isinstance(content, str) or not content.strip().startswith("{"):
            continue
        try:
            packet = json.loads(content)
        except json.JSONDecodeError:
            continue

        event_type = packet.get("event")
        cursor = packet.get("cursor", {})
        target = packet.get("target", {})

        cx = cursor.get("x")
        cy = cursor.get("y")
        if cx is None or cy is None:
            continue

        target_row = target.get("row")
        target_col = target.get("col")

        if event_type == "screen" and not events:
            ts = None
            for j in range(i + 1, min(i + 5, len(history))):
                future = history[j]
                if future.get("role") == "user":
                    fc = future.get("content", [])
                    if isinstance(fc, list):
                        for item in fc:
                            if isinstance(item, dict) and item.get("type") == "image_url":
                                url = item.get("image_url", {}).get("url", "")
                                if ".png" in url:
                                    fn = url.split("/")[-1] if "/" in url else url
                                    ts = get_timestamp_ms(fn)
                                    break
                    break
            events.append(
                {
                    "type": "start",
                    "ts": ts or (screenshot_times[0] if screenshot_times else 0),
                    "cursor_x": cx,
                    "cursor_y": cy,
                    "target_row": target_row,
                    "target_col": target_col,
                }
            )
        elif event_type == "move":
            events.append(
                {
                    "type": "move",
                    "ts": None,
                    "cursor_x": cx,
                    "cursor_y": cy,
                    "target_row": target_row,
                    "target_col": target_col,
                }
            )
        elif event_type == "click":
            events.append(
                {
                    "type": "click",
                    "ts": packet.get("last_click_time_ms"),
                    "cursor_x": cx,
                    "cursor_y": cy,
                    "clicked_row": packet.get("last_click_row"),
                    "clicked_col": packet.get("last_click_col"),
                    "new_target_row": target_row,
                    "new_target_col": target_col,
                    "correct": packet.get("correct", False),
                }
            )

    # If legacy format found events, return them
    if events:
        return events

    # Otherwise, parse current format: tool_calls in assistant messages
    last_ntpm = 0
    current_target_row = None
    current_target_col = None

    # Build target update queue from targets.json (if available)
    # This is the new format where target positions are saved separately (not leaked to model)
    target_click_queue: list[tuple[int, int]] = []  # [(target_row, target_col), ...] for each click
    if target_events_from_file:
        for te in target_events_from_file:
            if te.get("type") == "start":
                current_target_row = te.get("target_row")
                current_target_col = te.get("target_col")
            elif te.get("type") == "click":
                # Queue up target positions after each click
                target_click_queue.append((te.get("target_row", -1), te.get("target_col", -1)))
    click_queue_idx = 0  # Index into target_click_queue

    # Parse grid size from first tool response: HUD "00:09 ... 30×30" or JSON {"grid_side": 16}
    grid_side = 30  # default
    if target_events_from_file:
        # Get grid_side from targets.json if available
        for te in target_events_from_file:
            if te.get("grid_side"):
                grid_side = te["grid_side"]
                break
    for entry in history:
        if entry.get("role") == "tool":
            content = entry.get("content", "")
            if isinstance(content, str) and content.strip().startswith("{"):
                try:
                    data = json.loads(content)
                    grid_side = data.get("grid_side", grid_side)
                    ok = (
                        not target_events_from_file
                        and data.get("target_row", -1) >= 0
                        and data.get("target_col", -1) >= 0
                    )
                    if ok:
                        current_target_row = data["target_row"]
                        current_target_col = data["target_col"]
                except json.JSONDecodeError:
                    pass
                break
            elif isinstance(content, str) and "×" in content:
                try:
                    grid_part = content.split("×")[-1].strip()
                    grid_side = int(grid_part.split()[0]) if grid_part else 30
                except (ValueError, IndexError):
                    pass
                # Only parse target from history if not loaded from targets.json
                if not target_events_from_file:
                    target_match = re.search(r"\[T:(\d+),(\d+)\]", content)
                    if target_match:
                        current_target_row = int(target_match.group(1))
                        current_target_col = int(target_match.group(2))
                break

    cell_size = canvas_size / grid_side

    # Track cursor position (starts at center of cell 0,0, NOT center of canvas)
    # These will be updated throughout the loop
    cursor_x = int(cell_size / 2)
    cursor_y = int(cell_size / 2)
    _last_cursor_x, _last_cursor_y = cursor_x, cursor_y
    _last_target_row, _last_target_col = current_target_row, current_target_col

    # Add start event
    if screenshot_times:
        events.append(
            {
                "type": "start",
                "ts": screenshot_times[0],
                "cursor_x": cursor_x,
                "cursor_y": cursor_y,
                "target_row": current_target_row,
                "target_col": current_target_col,
            }
        )

    # Build mapping of tool_call_id to function info
    tool_call_map = {}
    for entry in history:
        if entry.get("role") == "assistant":
            tool_calls = entry.get("tool_calls", [])
            for tc in tool_calls:
                tc_id = tc.get("id")
                func = tc.get("function", {})
                func_name = func.get("name", "")
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                except json.JSONDecodeError:
                    args = {}
                tool_call_map[tc_id] = {"name": func_name, "args": args}

    # Process tool responses to build event timeline
    for i, entry in enumerate(history):
        if entry.get("role") != "tool":
            continue

        tc_id = entry.get("tool_call_id")
        content = entry.get("content", "")

        if tc_id not in tool_call_map:
            continue

        func_info = tool_call_map[tc_id]
        func_name = func_info["name"]
        args = func_info["args"]

        # Get timestamp from next user message's screenshot
        ts = None
        for j in range(i + 1, min(i + 5, len(history))):
            future = history[j]
            if future.get("role") == "user":
                fc = future.get("content", [])
                if isinstance(fc, list):
                    for item in fc:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            if ".png" in url:
                                fn = url.split("/")[-1] if "/" in url else url
                                ts = get_timestamp_ms(fn)
                                break
                break

        # Parse NTPM and target position from content
        # Formats: HUD "00:17 0.00 BPS -1 NTPM 30×30" | JSON {"ntpm": 1, "correct": true} | "OK"
        # Note: target position now comes from targets.json (not leaked to model)
        current_ntpm = last_ntpm
        parsed_target_row = current_target_row
        parsed_target_col = current_target_col
        parsed_correct = None

        if isinstance(content, str) and content.strip().startswith("{"):
            try:
                data = json.loads(content)
                current_ntpm = data.get("ntpm", last_ntpm)
                ok = (
                    not target_events_from_file
                    and data.get("target_row", -1) >= 0
                    and data.get("target_col", -1) >= 0
                )
                if ok:
                    parsed_target_row = data["target_row"]
                    parsed_target_col = data["target_col"]
                parsed_correct = data.get("correct")
            except json.JSONDecodeError:
                pass
        elif isinstance(content, str):
            ntpm_match = re.search(r"(-?\d+)\s*NTPM", content)
            if ntpm_match:
                current_ntpm = int(ntpm_match.group(1))
            # Only parse target from history if not using targets.json (legacy support)
            if not target_events_from_file:
                target_match = re.search(r"\[T:(\d+),(\d+)\]", content)
                if target_match:
                    parsed_target_row = int(target_match.group(1))
                    parsed_target_col = int(target_match.group(2))

        if func_name == "mouse_move":
            dx = args.get("dx", 0)
            dy = args.get("dy", 0)
            cursor_x = max(0, min(canvas_size - 1, cursor_x + dx))
            cursor_y = max(0, min(canvas_size - 1, cursor_y + dy))
            # Update current target from parsed value (only if not using targets.json)
            if not target_events_from_file and parsed_target_row is not None:
                current_target_row = parsed_target_row
                current_target_col = parsed_target_col
            events.append(
                {
                    "type": "move",
                    "ts": ts,
                    "cursor_x": cursor_x,
                    "cursor_y": cursor_y,
                    "target_row": current_target_row,
                    "target_col": current_target_col,
                }
            )

        elif func_name == "mouse_click":
            # Use explicit correct flag from JSON, or infer from NTPM change for legacy HUD format
            correct = parsed_correct if parsed_correct is not None else (current_ntpm > last_ntpm)
            # Store the target BEFORE click (what was clicked)
            clicked_target_row = current_target_row
            clicked_target_col = current_target_col
            # Update to new target AFTER click
            if target_click_queue and click_queue_idx < len(target_click_queue):
                # Use target from targets.json (preferred)
                current_target_row, current_target_col = target_click_queue[click_queue_idx]
                click_queue_idx += 1
            elif parsed_target_row is not None:
                # Fall back to parsing from history (legacy support)
                current_target_row = parsed_target_row
                current_target_col = parsed_target_col
            events.append(
                {
                    "type": "click",
                    "ts": ts,
                    "cursor_x": cursor_x,
                    "cursor_y": cursor_y,
                    "clicked_row": int(cursor_y / cell_size),
                    "clicked_col": int(cursor_x / cell_size),
                    "target_before_click_row": clicked_target_row,
                    "target_before_click_col": clicked_target_col,
                    "new_target_row": current_target_row,
                    "new_target_col": current_target_col,
                    "correct": correct,
                }
            )
            last_ntpm = current_ntpm

        elif func_name == "screen":
            # Screen calls also update target position (only if not using targets.json)
            if not target_events_from_file and parsed_target_row is not None:
                current_target_row = parsed_target_row
                current_target_col = parsed_target_col
            # Update start event if it exists and has no target
            if events and events[0]["type"] == "start" and events[0].get("target_row") is None:
                events[0]["target_row"] = current_target_row
                events[0]["target_col"] = current_target_col

        last_ntpm = current_ntpm

    return events


def create_gif(
    eval_dir: str,
    output: str | None = None,
    speed: float = 1.0,
    fps: float = 30.0,
    loop: int = 0,
) -> str:
    """Create an animated GIF by rendering game state with smooth cursor movement.

    Args:
        eval_dir: Path to eval directory
        output: Output GIF path (default: <eval_dir>/replay.gif)
        speed: Playback speed multiplier (default: 1 = real-time)
        fps: Output frames per second (default: 30)
        loop: Number of loops (0 = infinite)

    Returns:
        Path to created GIF file
    """
    if Image is None:
        raise ImportError("Pillow is required: pip install Pillow")

    eval_path = Path(eval_dir)
    if not eval_path.exists():
        raise FileNotFoundError(f"Directory not found: {eval_dir}")

    # Read grid_side and canvas_size from result.json (matches eval config)
    grid_side = DEFAULT_GRID_SIDE
    canvas_size = DEFAULT_CANVAS_SIZE
    result_path = eval_path / "result.json"
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
            grid_side = int(result.get("grid_side", grid_side))
            canvas_size = int(result.get("size_px", canvas_size))
        except (json.JSONDecodeError, ValueError):
            pass

    # Parse game events from history
    history_path = eval_path / "history.json"
    events = parse_game_events(history_path, canvas_size)

    if not events:
        # Debug: check what's in the history
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            print(f"DEBUG: History has {len(history)} entries")
            roles_list = [e.get("role") for e in history]
            role_counts = {r: roles_list.count(r) for r in set(roles_list)}
            print(f"DEBUG: Roles: {role_counts}")
            tool_entries = [e for e in history if e.get("role") == "tool"]
            print(f"DEBUG: Tool entries: {len(tool_entries)}")
            if tool_entries:
                print(f"DEBUG: First tool content: {tool_entries[0].get('content', '')[:100]}")
        raise ValueError(f"No game events found in {history_path}")

    # Extract click events with timestamps
    clicks = [e for e in events if e["type"] == "click" and e.get("ts") is not None]

    # Debug: show parsed events
    print(f"DEBUG: Parsed {len(events)} events:")
    for e in events:
        if e["type"] == "click":
            print(
                f"  CLICK: ts={e.get('ts')}, pos=({e.get('cursor_x')}, {e.get('cursor_y')}), "
                f"cell=({e.get('clicked_row')}, {e.get('clicked_col')}), correct={e.get('correct')}"
            )
        elif e["type"] == "move":
            print(f"  MOVE: ts={e.get('ts')}, pos=({e.get('cursor_x')}, {e.get('cursor_y')})")
        elif e["type"] == "start":
            print(f"  START: ts={e.get('ts')}, pos=({e.get('cursor_x')}, {e.get('cursor_y')})")

    if not clicks:
        raise ValueError("No click events with timestamps found")

    # Get start event
    start_event = events[0] if events and events[0]["type"] == "start" else None

    # Calculate timing
    start_ts = start_event["ts"] if start_event and start_event["ts"] else 0
    end_ts = clicks[-1]["ts"]
    real_time_ms = end_ts - start_ts
    gif_duration_ms = real_time_ms / speed
    frame_duration_ms = 1000 / fps
    total_frames = max(1, int(gif_duration_ms / frame_duration_ms))

    print(f"Building {total_frames} frames at {fps} fps...")
    print(f"  Events: {len(events)} ({len(clicks)} clicks)")

    # Build cursor position timeline from actual game events
    # Use actual timestamps for time-proportional movement speed
    cursor_timeline = []

    # Start with initial cursor position from start event
    if start_event:
        cursor_timeline.append(
            {
                "ts": start_ts,
                "x": start_event.get("cursor_x", 8),
                "y": start_event.get("cursor_y", 8),
            }
        )

    # Add move and click events to cursor timeline using actual timestamps
    # Movement speed will be proportional to time elapsed between events
    for i, event in enumerate(events):
        if event["type"] == "move":
            event_ts = event.get("ts")
            if event_ts is not None:
                # Use actual timestamp - movement will be time-proportional
                # If next event is a click at same/close timestamp, place move slightly before
                next_event_ts = None
                for j in range(i + 1, len(events)):
                    if events[j].get("ts") is not None:
                        next_event_ts = events[j]["ts"]
                        break
                # If move and next event have same timestamp, offset move by 100ms before
                if next_event_ts is not None and abs(event_ts - next_event_ts) < 50:
                    event_ts = next_event_ts - 100
                cursor_timeline.append(
                    {
                        "ts": event_ts,
                        "x": event["cursor_x"],
                        "y": event["cursor_y"],
                    }
                )
        elif event["type"] == "click" and event.get("ts"):
            cursor_timeline.append(
                {
                    "ts": event["ts"],
                    "x": event["cursor_x"],
                    "y": event["cursor_y"],
                }
            )

    # Sort cursor timeline by timestamp
    cursor_timeline.sort(key=lambda x: x["ts"])

    # Build target timeline from game events
    # Target is shown from start until first click, then changes after each click
    target_timeline = []

    # Initial target from start event
    if start_event and start_event.get("target_row") is not None:
        # Find when this target ends (first click)
        first_click_ts = clicks[0]["ts"] if clicks else end_ts
        target_timeline.append(
            {
                "start_ts": start_ts,
                "end_ts": first_click_ts,
                "target_row": start_event["target_row"],
                "target_col": start_event["target_col"],
            }
        )

    # After each click, show the new target until the next click
    for i, click in enumerate(clicks):
        if click.get("new_target_row") is not None:
            next_click_ts = clicks[i + 1]["ts"] if i + 1 < len(clicks) else end_ts + 1000
            target_timeline.append(
                {
                    "start_ts": click["ts"],
                    "end_ts": next_click_ts,
                    "target_row": click["new_target_row"],
                    "target_col": click["new_target_col"],
                }
            )

    print(f"  Target periods: {len(target_timeline)}")

    # Separate correct and incorrect clicks for visual feedback
    incorrect_clicks = [c for c in clicks if c.get("correct") is False]
    correct_clicks = [c for c in clicks if c.get("correct") is True]

    # Flash duration in real-time ms (will be shown shorter in sped-up GIF)
    # At 10x speed and 24fps, 500ms real = 50ms GIF = ~1.2 frames
    # Use 800ms to ensure at least 2 frames show the flash
    click_flash_duration_ms = 800

    # Generate frames
    frames = []
    frame_duration_int = int(frame_duration_ms)

    for frame_idx in range(total_frames):
        gif_time_ms = frame_idx * frame_duration_ms
        real_time_target = start_ts + (gif_time_ms * speed)

        # Find cursor position by interpolating between timeline points
        # Default to center of cell (0, 0)
        cell_size = canvas_size / grid_side
        cursor_x, cursor_y = int(cell_size / 2), int(cell_size / 2)

        if cursor_timeline:
            # Before first point
            if real_time_target <= cursor_timeline[0]["ts"]:
                cursor_x = cursor_timeline[0]["x"]
                cursor_y = cursor_timeline[0]["y"]
            # After last point
            elif real_time_target >= cursor_timeline[-1]["ts"]:
                cursor_x = cursor_timeline[-1]["x"]
                cursor_y = cursor_timeline[-1]["y"]
            else:
                # Interpolate between points
                for i in range(len(cursor_timeline) - 1):
                    curr = cursor_timeline[i]
                    next_ = cursor_timeline[i + 1]

                    if curr["ts"] <= real_time_target < next_["ts"]:
                        duration = next_["ts"] - curr["ts"]
                        if duration > 0:
                            progress = (real_time_target - curr["ts"]) / duration
                            progress = min(1, max(0, progress))
                            px, py = interpolate_position(
                                curr["x"], curr["y"], next_["x"], next_["y"], progress
                            )
                            cursor_x, cursor_y = int(px), int(py)
                        else:
                            cursor_x, cursor_y = next_["x"], next_["y"]
                        break

        # Find which target is active at this time
        current_target_row = -1  # No target by default
        current_target_col = -1
        for target in target_timeline:
            if target["start_ts"] <= real_time_target < target["end_ts"]:
                current_target_row = target["target_row"]
                current_target_col = target["target_col"]
                break

        # Check for click flash (green for correct, red for incorrect)
        error_cell = None
        success_cell = None

        # Check incorrect clicks - red flash
        for click in incorrect_clicks:
            ts = click["ts"]
            if ts <= real_time_target < ts + click_flash_duration_ms:
                error_cell = (click["clicked_row"], click["clicked_col"])
                break

        # Check correct clicks - green flash (only if no error)
        if error_cell is None:
            for click in correct_clicks:
                ts = click["ts"]
                if ts <= real_time_target < ts + click_flash_duration_ms:
                    success_cell = (click["clicked_row"], click["clicked_col"])
                    break

        # Render frame
        frame = render_frame(
            target_row=current_target_row,
            target_col=current_target_col,
            cursor_x=int(cursor_x),
            cursor_y=int(cursor_y),
            side=grid_side,
            error_cell=error_cell,
            success_cell=success_cell,
            canvas_size=canvas_size,
        )
        frames.append(frame)

    if not frames:
        raise ValueError("No frames generated")

    # Uniform duration for all frames
    durations = [frame_duration_int] * len(frames)

    # Output path
    if output is None:
        output = str(eval_path / "replay.gif")

    # Save GIF
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=loop,
        optimize=True,
    )

    actual_duration = sum(durations) / 1000

    print(f"Created GIF: {output}")
    print(f"  Output: {len(frames)} frames @ {fps} fps")
    print(f"  Real time: {real_time_ms / 1000:.1f}s")
    print(f"  GIF duration: {actual_duration:.1f}s ({speed}x speed)")

    return output


def main() -> None:
    """CLI entrypoint: parse args and generate replay GIF from eval dir."""
    parser = argparse.ArgumentParser(description="Create animated GIF with smooth cursor movement")
    parser.add_argument(
        "eval_dir",
        nargs="?",
        default=None,
        help="Path to eval directory (default: process all subdirs under eval/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output GIF path (default: <eval_dir>/replay.gif)",
    )
    parser.add_argument(
        "--speed",
        "-s",
        type=float,
        default=10.0,
        help="Playback speed (default: 1 = real-time)",
    )
    parser.add_argument(
        "--fps",
        "-f",
        type=float,
        default=24.0,
        help="Frames per second (default: 24)",
    )
    parser.add_argument(
        "--loop",
        "-l",
        type=int,
        default=0,
        help="Loop count (0 = infinite)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate GIFs for all dirs (ignore existing replay.gif)",
    )
    args = parser.parse_args()

    # If no eval_dir provided, process all subdirectories under eval/
    if args.eval_dir is None:
        results_base = Path("eval")
        if not results_base.exists():
            print("Error: eval/ directory not found")
            return

        subdirs = [d for d in results_base.iterdir() if d.is_dir()]
        if not subdirs:
            print("No subdirectories found under eval/")
            return

        # Only process dirs that don't have replay.gif yet (unless --force)
        sorted_subdirs = sorted(subdirs)
        if args.force:
            to_process = sorted_subdirs
        else:
            to_process = [d for d in sorted_subdirs if not (d / "replay.gif").exists()]
            skipped = len(sorted_subdirs) - len(to_process)
            if skipped:
                print(f"Skipping {skipped} dir(s) that already have replay.gif")
            if not to_process:
                print("Nothing to do (all dirs already have replay.gif)")
                return

        for subdir in to_process:
            print(f"\n{'=' * 60}")
            print(f"Processing {subdir}")
            print("=" * 60)
            try:
                create_gif(
                    eval_dir=str(subdir),
                    output=args.output,
                    speed=args.speed,
                    fps=args.fps,
                    loop=args.loop,
                )
            except Exception as e:
                print(f"Error processing {subdir}: {e}")
    else:
        create_gif(
            eval_dir=args.eval_dir,
            output=args.output,
            speed=args.speed,
            fps=args.fps,
            loop=args.loop,
        )


if __name__ == "__main__":
    main()
