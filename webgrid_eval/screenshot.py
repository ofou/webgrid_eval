"""Generate grid screenshot matching Neuralink Webgrid (neuralink.com/webgrid).

Site: neuralink.com/webgrid/
CSS: webgrid.CtloUddh.css (canvas: ._canvas_1wslk_27 border .2rem solid black)
Game: canvas_size x canvas_size canvas, r = p*K (K=0.002), fillRect inset r/2.
Click: Math.floor(px/cellSize), cellSize = width/gridSize. No border in drawable area.

Literal coordinates:
  width = canvas_size (default 256), cellSize = width/side
  col = floor(px / cellSize), row = floor(py / cellSize)
  Origin: (0, 0) = top-left of canvas; x right, y down. Pixel range [0, canvas_size-1].
"""

import base64
import io
import math
import os

import cairosvg
from PIL import Image, ImageDraw

# Default canvas size (backward compatible with original Neuralink 256x256)
DEFAULT_CANVAS_SIZE = 256

# Keep GRID_SIZE_PX for backward compatibility with existing code
GRID_SIZE_PX = DEFAULT_CANVAS_SIZE

# Line width constant
K = 0.002

CURSOR_VIEWBOX = 32
CURSOR_HOTSPOT_XY = (10, 16)
CURSOR_DISPLAY_PX = 24

_CURSOR_CACHE: tuple[Image.Image, int, int] | None = None


def _load_cursor() -> tuple[Image.Image, int, int]:
    global _CURSOR_CACHE
    if _CURSOR_CACHE is not None:
        return _CURSOR_CACHE
    path = os.path.join(os.path.dirname(__file__), "assets", "cursor.svg")
    scale = CURSOR_DISPLAY_PX / CURSOR_VIEWBOX
    png_bytes = cairosvg.svg2png(
        url=path,
        output_width=int(CURSOR_VIEWBOX * scale),
        output_height=int(CURSOR_VIEWBOX * scale),
    )
    cur = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    hx = max(1, int(CURSOR_HOTSPOT_XY[0] * scale))
    hy = max(1, int(CURSOR_HOTSPOT_XY[1] * scale))
    _CURSOR_CACHE = (cur, hx, hy)
    return _CURSOR_CACHE


def _line_width(canvas_size: int) -> int:
    """Line width in pixels (Neuralink: r = width * K)."""
    return max(1, round(canvas_size * K))


def _get_cell_rect(row: int, col: int, side: int, canvas_size: int) -> tuple[int, int, int, int]:
    """Cell fill rect: inset by r/2 from boundaries (Neuralink fillRect)."""
    c = canvas_size / side
    r = _line_width(canvas_size)
    inset = r / 2
    x0 = col * c + inset
    y0 = row * c + inset
    x1 = (col + 1) * c - inset
    y1 = (row + 1) * c - inset
    return int(x0), int(y0), int(x1), int(y1)


def pixel_to_cell(
    x: int, y: int, side: int, canvas_size: int = DEFAULT_CANVAS_SIZE
) -> tuple[int, int]:
    """Convert pixel (x, y) to grid cell (row, col). Same as Neuralink: col = floor(px/cellSize), row = floor(py/cellSize), cellSize = canvas_size/side."""
    cell_size = canvas_size / side
    col = math.floor(x / cell_size)
    row = math.floor(y / cell_size)
    col = max(0, min(col, side - 1))
    row = max(0, min(row, side - 1))
    return row, col


def cell_center_pixel(
    row: int, col: int, side: int, canvas_size: int = DEFAULT_CANVAS_SIZE
) -> tuple[int, int]:
    """Return pixel (x, y) at the center of cell (row, col). Use for snap-to-center and cursor tracking."""
    cell_size = canvas_size / side
    x = int((col + 0.5) * cell_size)
    y = int((row + 0.5) * cell_size)
    x = max(0, min(canvas_size - 1, x))
    y = max(0, min(canvas_size - 1, y))
    return x, y


def normalized_to_cell(
    nx: float, ny: float, side: int, canvas_size: int = DEFAULT_CANVAS_SIZE
) -> tuple[int, int]:
    """Convert normalized (nx, ny) in [0,1] to cell."""
    x = int(nx * (canvas_size - 1)) if 0 <= nx <= 1 else 0
    y = int(ny * (canvas_size - 1)) if 0 <= ny <= 1 else 0
    return pixel_to_cell(x, y, side, canvas_size)


def render_grid_screenshot(
    state,
    save_path: str | None = None,
    last_click_incorrect: bool = False,
) -> str:
    """Draw grid matching Neuralink Webgrid: canvas_size x canvas_size, K=0.002 lines, rgb(10,132,255) target.

    Canvas size is read from state.canvas_size (default: 256).
    """
    side = state.grid_side
    # Get canvas_size from state, fallback to default for backward compatibility
    canvas_size = getattr(state, "canvas_size", DEFAULT_CANVAS_SIZE)

    c = canvas_size / side
    r = _line_width(canvas_size)
    white = (255, 255, 255)
    black = (0, 0, 0)
    active_blue = (10, 132, 255)
    active_hover_blue = (7, 100, 191)
    hover_gray = (191, 191, 191)
    error_red = (255, 0, 0)

    # Cursor tip at pixel coords; fallback to cell center if None
    cx = getattr(state, "cursor_x", None)
    cy = getattr(state, "cursor_y", None)
    if cx is None or cy is None:
        tip_x, tip_y = cell_center_pixel(state.cursor_row, state.cursor_col, side, canvas_size)
    else:
        tip_x = max(0, min(int(cx), canvas_size - 1))
        tip_y = max(0, min(int(cy), canvas_size - 1))

    hover_row, hover_col = pixel_to_cell(tip_x, tip_y, side, canvas_size)
    hovered_cell = (hover_row, hover_col)
    last_click_cell = (
        (state.last_click_row, state.last_click_col)
        if getattr(state, "last_click_row", None) is not None
        and getattr(state, "last_click_col", None) is not None
        else None
    )

    img = Image.new("RGB", (canvas_size, canvas_size), white)
    draw = ImageDraw.Draw(img)

    target_row, target_col = state.target_row, state.target_col

    for row in range(side):
        for col in range(side):
            x0, y0, x1, y1 = _get_cell_rect(row, col, side, canvas_size)

            is_target = (row, col) == (target_row, target_col)
            is_hovered = (row, col) == hovered_cell
            is_last_click_wrong = (
                last_click_cell is not None
                and (row, col) == last_click_cell
                and last_click_incorrect
            )

            if is_last_click_wrong:
                fill = error_red
            elif is_hovered and is_target:
                fill = active_hover_blue
            elif is_hovered:
                fill = hover_gray
            elif is_target:
                fill = active_blue
            else:
                fill = white

            draw.rectangle([x0, y0, x1, y1], fill=fill)

    # Draw grid lines first, then cursor on top so cursor is never obscured
    for a in range(side + 1):
        y_center = a * c
        x_center = a * c
        y0 = max(0, int(y_center - r / 2))
        y1 = min(canvas_size, int(y_center + r / 2) + 1)
        draw.rectangle([0, y0, canvas_size, y1], fill=black)
        x0 = max(0, int(x_center - r / 2))
        x1 = min(canvas_size, int(x_center + r / 2) + 1)
        draw.rectangle([x0, 0, x1, canvas_size], fill=black)

    # Cursor: tip at pixel (cursor_x, cursor_y)
    cur_img, hx, hy = _load_cursor()
    px = tip_x - hx
    py = tip_y - hy
    r, g, b, a = cur_img.split()
    img.paste(cur_img.convert("RGB"), (px, py), a)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    if save_path:
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(png_bytes)
    return base64.b64encode(png_bytes).decode("ascii")
