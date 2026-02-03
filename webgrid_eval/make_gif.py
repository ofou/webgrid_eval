"""Create animated GIF by rendering game state with smooth cursor movement.

Usage: python -m webgrid_eval.make_gif [eval_dir] [--speed 1] [--fps 30]

When no eval_dir is provided, processes all subdirectories under results/ recursively.

Renders the game grid from history.json, animating cursor movement between clicks.
Does NOT use screenshot images - generates all frames programmatically.
"""

import argparse
import json
import re
from io import BytesIO
from pathlib import Path

import cairosvg

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

CURSOR_VIEWBOX = 32
CURSOR_HOTSPOT_XY = (10, 16)
CURSOR_DISPLAY_PX = 24

_CURSOR_CACHE: tuple | None = None


def _load_cursor():
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
DEFAULT_CANVAS_SIZE = 256
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
    canvas_size = img.width  # Get actual size from image
    cell_size = canvas_size / side

    for y in range(canvas_size):
        for x in range(canvas_size):
            pixel = pixels[x, y]
            if len(pixel) >= 3:
                r, g, b = pixel[:3]
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
    canvas_size: int = DEFAULT_CANVAS_SIZE,
) -> Image.Image:
    """Render a single frame matching screenshot.py (Neuralink style): cells then grid lines on top."""
    white = (255, 255, 255)
    black = (0, 0, 0)
    active_blue = (10, 132, 255)
    hover_gray = (191, 191, 191)
    active_hover_blue = (7, 100, 191)
    error_red = (255, 0, 0)

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

            if is_error_cell:
                fill = error_red
            elif is_cursor_cell and is_target:
                fill = active_hover_blue
            elif is_cursor_cell:
                fill = hover_gray
            elif is_target:
                fill = active_blue
            else:
                fill = white

            draw.rectangle([x0, y0, x1, y1], fill=fill)

    c = canvas_size / side
    r = _line_width(side, canvas_size)
    for a in range(side + 1):
        y_center = a * c
        x_center = a * c
        y0 = max(0, int(y_center - r / 2))
        y1 = min(canvas_size, int(y_center + r / 2) + 1)
        draw.rectangle([0, y0, canvas_size, y1], fill=black)
        x0 = max(0, int(x_center - r / 2))
        x1 = min(canvas_size, int(x_center + r / 2) + 1)
        draw.rectangle([x0, 0, x1, canvas_size], fill=black)

    tip_x, tip_y = cursor_x, cursor_y
    cur_img, hx, hy = _load_cursor()
    px = tip_x - hx
    py = tip_y - hy
    r, g, b, a = cur_img.split()
    img.paste(cur_img.convert("RGB"), (px, py), a)

    return img


def parse_game_events(history_path: Path, canvas_size: int = DEFAULT_CANVAS_SIZE) -> list[dict]:
    """Parse history.json to extract game events from tool-result packets.

    Scans tool messages for event field. Returns:
    - {"type": "start", "ts": int}
    - {"type": "click", "ts": int | None, "nx": float, "ny": float, "correct": bool}
    """
    if not history_path.exists():
        return []

    with open(history_path) as f:
        history = json.load(f)

    events = []

    # Collect screenshot timestamps from user messages (for fallback ts)
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

    # Scan tool responses for event packets
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
        if event_type == "move":
            pass  # Could use cursor position for smoother GIF trajectory (bonus)

        elif event_type == "click":
            cursor = packet.get("cursor", {})
            cx = cursor.get("x")
            cy = cursor.get("y")
            last_row = packet.get("last_click_row")
            last_col = packet.get("last_click_col")
            if cx is not None and cy is not None:
                nx = float(cx) / canvas_size
                ny = float(cy) / canvas_size
            elif last_row is not None and last_col is not None:
                try:
                    from .screenshot import cell_center_pixel
                except ImportError:
                    from screenshot import cell_center_pixel

                side = 8  # Default; create_gif reads from result.json
                px, py = cell_center_pixel(last_row, last_col, side, canvas_size)
                nx = float(px) / canvas_size
                ny = float(py) / canvas_size
            else:
                continue

            correct = packet.get("correct")
            if correct is None and "hud" in packet:
                correct = packet["hud"].get("last_click_correct")
            if correct is None:
                correct = False

            ts = packet.get("last_click_time_ms")
            if ts is None:
                for j in range(i + 1, min(i + 5, len(history))):
                    future = history[j]
                    if future.get("role") != "user":
                        continue
                    fc = future.get("content", [])
                    if not isinstance(fc, list):
                        continue
                    for item in fc:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            if isinstance(url, str) and ".png" in url:
                                fn = url.split("/")[-1] if "/" in url else url
                                ts = get_timestamp_ms(fn)
                                break
                    if ts is not None:
                        break

            events.append(
                {
                    "type": "click",
                    "ts": ts,
                    "nx": nx,
                    "ny": ny,
                    "correct": correct,
                }
            )

    # Prepend start event
    if screenshot_times:
        events.insert(
            0,
            {
                "type": "start",
                "ts": screenshot_times[0],
                "target_row": None,
                "target_col": None,
            },
        )

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
        raise ValueError(f"No game events found in {history_path}")

    # Extract click events with timestamps
    clicks = [e for e in events if e["type"] == "click" and e["ts"] is not None]

    if not clicks:
        raise ValueError("No click events with timestamps found")

    # Calculate timing
    start_ts = events[0]["ts"] if events[0]["ts"] else 0
    end_ts = clicks[-1]["ts"]
    real_time_ms = end_ts - start_ts
    gif_duration_ms = real_time_ms / speed
    frame_duration_ms = 1000 / fps
    total_frames = max(1, int(gif_duration_ms / frame_duration_ms))

    print(f"Building {total_frames} frames at {fps} fps...")
    print(f"  Events: {len(events)} ({len(clicks)} clicks)")

    # Build cursor position timeline
    # Cursor moves from click to click
    cursor_timeline = []

    # Start position (top-left of first target cell or center of grid)
    if clicks:
        first_click = clicks[0]
        # Use cell center of first click as starting position
        px, py = normalized_to_pixel(first_click["nx"], first_click["ny"], canvas_size)
        cursor_timeline.append(
            {
                "ts": start_ts,
                "x": px,
                "y": py,
            }
        )

        for click in clicks:
            px, py = normalized_to_pixel(click["nx"], click["ny"], canvas_size)
            cursor_timeline.append(
                {
                    "ts": click["ts"],
                    "x": px,
                    "y": py,
                }
            )

    # Build target timeline by detecting targets from screenshots
    # Each screenshot shows where the target was at that moment
    target_timeline = []

    # Get all screenshot files sorted by timestamp
    screenshot_files = []
    for f in eval_path.glob("*.png"):
        ts = get_timestamp_ms(f.name)
        if ts is not None:
            screenshot_files.append((ts, f))
    screenshot_files.sort(key=lambda x: x[0])

    # Detect target from each screenshot
    screenshot_targets = []
    for ts, path in screenshot_files:
        target = detect_target_from_screenshot(path, grid_side)
        if target:
            screenshot_targets.append({"ts": ts, "target_row": target[0], "target_col": target[1]})

    # Build target timeline from detected positions
    # Target stays until it's clicked correctly (which means it changes)
    prev_target = None
    prev_ts = start_ts if screenshot_targets else start_ts

    for i, st in enumerate(screenshot_targets):
        current_target = (st["target_row"], st["target_col"])

        if prev_target is None:
            # First target
            prev_target = current_target
            prev_ts = st["ts"]
        elif current_target != prev_target:
            # Target changed - previous target was valid until now
            target_timeline.append(
                {
                    "start_ts": prev_ts,
                    "end_ts": st["ts"],
                    "target_row": prev_target[0],
                    "target_col": prev_target[1],
                }
            )
            prev_target = current_target
            prev_ts = st["ts"]

    # Add the last target (valid until end of animation)
    if prev_target and screenshot_files:
        end_ts = screenshot_files[-1][0] + 1000  # Extend 1 second past last screenshot
        target_timeline.append(
            {
                "start_ts": prev_ts,
                "end_ts": end_ts,
                "target_row": prev_target[0],
                "target_col": prev_target[1],
            }
        )

    print(
        f"  Targets detected: {len(target_timeline)} periods from {len(screenshot_targets)} screenshots"
    )

    # Track incorrect clicks separately for red flash
    incorrect_clicks = [c for c in clicks if c.get("correct") == False]

    # Generate frames
    frames = []
    frame_duration_int = int(frame_duration_ms)

    # Track game state for rendering
    current_target_row = grid_side // 2  # Default to center
    current_target_col = grid_side // 2
    last_click_incorrect = False
    incorrect_flash_until = 0

    for frame_idx in range(total_frames):
        gif_time_ms = frame_idx * frame_duration_ms
        real_time_target = start_ts + (gif_time_ms * speed)

        # Find cursor position by interpolating between timeline points
        cursor_x, cursor_y = 10, 10  # Default

        for i in range(len(cursor_timeline) - 1):
            curr = cursor_timeline[i]
            next_ = cursor_timeline[i + 1]

            if curr["ts"] <= real_time_target < next_["ts"]:
                # Interpolate between these points
                duration = next_["ts"] - curr["ts"]
                if duration > 0:
                    progress = (real_time_target - curr["ts"]) / duration
                    progress = min(1, max(0, progress))
                    cursor_x, cursor_y = interpolate_position(
                        curr["x"], curr["y"], next_["x"], next_["y"], progress
                    )
                else:
                    cursor_x, cursor_y = next_["x"], next_["y"]
                break
            elif real_time_target >= cursor_timeline[-1]["ts"]:
                # After last point
                cursor_x = cursor_timeline[-1]["x"]
                cursor_y = cursor_timeline[-1]["y"]
                break

        # Find which target is active at this time
        # Target is visible from start_ts to end_ts
        current_target_row = -1  # No target by default
        current_target_col = -1
        for target in target_timeline:
            if target["start_ts"] <= real_time_target < target["end_ts"]:
                current_target_row = target["target_row"]
                current_target_col = target["target_col"]
                break

        # Check for incorrect click flash (200ms duration)
        # The clicked cell flashes red
        error_cell = None
        for click in incorrect_clicks:
            ts = click["ts"]
            if ts <= real_time_target < ts + 200:
                # Flash the cell that was incorrectly clicked
                row, col = normalized_to_cell(click["nx"], click["ny"], grid_side)
                error_cell = (row, col)
                break

        # Render frame
        frame = render_frame(
            target_row=current_target_row,
            target_col=current_target_col,
            cursor_x=int(cursor_x),
            cursor_y=int(cursor_y),
            side=grid_side,
            error_cell=error_cell,
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
    parser = argparse.ArgumentParser(description="Create animated GIF with smooth cursor movement")
    parser.add_argument(
        "eval_dir",
        nargs="?",
        default=None,
        help="Path to results directory (default: process all subdirs under results/)",
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

    # If no eval_dir provided, process all subdirectories under results/
    if args.eval_dir is None:
        results_base = Path("results")
        if not results_base.exists():
            print("Error: results/ directory not found")
            return

        subdirs = [d for d in results_base.iterdir() if d.is_dir()]
        if not subdirs:
            print("No subdirectories found under results/")
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
