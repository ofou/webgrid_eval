"""Browser-based Webgrid eval — Neuralink-identical UI for computer-use agents.

Serves a fullscreen web game matching neuralink.com/webgrid layout and dimensions.
The grid is rendered server-side with Pillow (same as the API tool pipeline) and
displayed via HTML canvas. Agents play by clicking on the blue target cell.

Usage:
    make play                              # start on port 8000
    make play ARGS="--port 8080"           # custom port
    make play ARGS="--grid-size 64"        # 8x8 grid
    python -m webgrid_eval.web_game        # direct invocation
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from .game_state import GameState
from .main import compute_ntpm_bps
from .screenshot import cell_center_pixel, pixel_to_cell, render_grid_screenshot

DEFAULT_GRID_SIZE = 900  # 30x30 (Neuralink standard)
DEFAULT_CANVAS_SIZE = 991  # Neuralink's exact internal canvas resolution
DEFAULT_MAX_SECONDS = 70
DEFAULT_PORT = 8000

_state: GameState | None = None
_wall_start: float = 0
_grid_size: int = DEFAULT_GRID_SIZE
_canvas_size: int = DEFAULT_CANVAS_SIZE
_max_seconds: int = DEFAULT_MAX_SECONDS
_results_log: list[dict[str, Any]] = []

app = FastAPI(title="Webgrid Eval")


def _grid_side() -> int:
    return int(_grid_size**0.5)


def _html_page() -> str:
    """Return the game HTML, matching neuralink.com/webgrid layout."""
    side = _grid_side()
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Play Webgrid | Neuralink</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html, body {{ width:100%; height:100%; overflow:hidden; background:#fff;
               font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               Helvetica, Arial, sans-serif; }}
  .game {{ display:flex; width:100%; height:100%; }}
  .hud-panel {{ display:flex; flex-direction:column; justify-content:center;
               align-items:flex-start; padding:0 60px; min-width:320px; }}
  .timer {{ font-size:72px; font-weight:300; color:#000; letter-spacing:-2px; }}
  .bps   {{ font-size:52px; font-weight:300; color:#000; margin-top:8px; }}
  .ntpm  {{ font-size:22px; font-weight:400; color:#aaa; margin-top:6px; }}
  .grid-panel {{ flex:1; display:flex; align-items:center; justify-content:center; }}
  #grid {{ border:3px solid #000; display:block; }}
  .overlay {{ position:fixed; inset:0; background:#fff; z-index:100;
             display:flex; flex-direction:column; align-items:center;
             justify-content:center; }}
  .overlay h2 {{ font-size:22px; font-weight:400; color:#888; }}
  .overlay .big-bps {{ font-size:80px; font-weight:300; color:#000; margin:8px 0; }}
  .overlay .big-ntpm {{ font-size:52px; font-weight:300; color:#bbb; }}
  .overlay .ref {{ font-size:15px; color:#888; max-width:440px; text-align:center;
                  margin:28px 0 20px; line-height:1.7; }}
  .overlay .ref u {{ text-decoration:underline; }}
  .overlay button {{ font-size:16px; padding:14px 36px; border:1px solid #ddd;
                    border-radius:999px; background:#fff; cursor:pointer;
                    font-family:inherit; margin:8px; }}
  .overlay button:hover {{ background:#f5f5f5; }}
  .intro {{ position:fixed; inset:0; background:#fff; z-index:200;
           display:flex; flex-direction:column; align-items:center;
           justify-content:center; }}
  .intro h1 {{ font-size:32px; font-weight:400; margin-bottom:24px; }}
  .intro p {{ font-size:16px; color:#555; max-width:480px; text-align:center;
             line-height:1.7; margin-bottom:12px; }}
  .intro button {{ font-size:16px; padding:14px 36px; border:1px solid #ddd;
                  border-radius:999px; background:#fff; cursor:pointer;
                  font-family:inherit; margin-top:20px; }}
</style>
</head>
<body>

<div class="intro" id="intro">
  <h1>Play Webgrid</h1>
  <p>At Neuralink, we use a game called Webgrid to test how precisely you can
  control a computer.</p>
  <p>The goal is to click targets on a grid as fast as possible while minimizing
  misclicks. Your score, measured in bits per second (BPS), is derived from net
  correct targets selected per minute (NTPM) and grid size.</p>
  <p>Our eighth clinical trial participant achieved a score of 10.39 BPS
  controlling his computer with his brain.</p>
  <p><strong>How well can you do?</strong></p>
  <button onclick="beginGame()">Start Game</button>
</div>

<div class="game" id="game" style="display:none">
  <div class="hud-panel">
    <div class="timer" id="timer">01:10</div>
    <div class="bps" id="bps">0.00 BPS</div>
    <div class="ntpm" id="ntpm">0 NTPM &middot; {side}&times;{side}</div>
  </div>
  <div class="grid-panel">
    <canvas id="grid" width="{_canvas_size}" height="{_canvas_size}"></canvas>
  </div>
</div>

<div class="overlay" id="results" style="display:none">
  <h2>Your peak score:</h2>
  <div class="big-bps" id="r-bps">0.00 BPS</div>
  <div class="big-ntpm" id="r-ntpm">(0 NTPM)</div>
  <div class="ref">Using his <u>N1 Implant</u>, our eighth clinical trial participant
  reached 10.39 BPS on a 16-inch MacBook Pro in fullscreen mode.</div>
  <div><button onclick="beginGame()">Play Again</button></div>
</div>

<script>
const CANVAS = {_canvas_size}, SIDE = {side};
const canvas = document.getElementById('grid');
const ctx = canvas.getContext('2d');
let running = false, iv = null;

function fitCanvas() {{
  const h = window.innerHeight - 40;
  canvas.style.width = h + 'px';
  canvas.style.height = h + 'px';
}}
window.addEventListener('resize', fitCanvas);

async function beginGame() {{
  document.getElementById('intro').style.display = 'none';
  document.getElementById('results').style.display = 'none';
  document.getElementById('game').style.display = 'flex';
  fitCanvas();
  const r = await fetch('/api/start', {{method:'POST'}});
  const d = await r.json();
  running = true;
  drawImg(d.image);
  updateHud(d);
  iv = setInterval(tick, 500);
}}

function drawImg(b64) {{
  const img = new Image();
  img.onload = () => ctx.drawImage(img, 0, 0, CANVAS, CANVAS);
  img.src = 'data:image/png;base64,' + b64;
}}

function updateHud(d) {{
  document.getElementById('bps').textContent = (d.bps || 0).toFixed(2) + ' BPS';
  document.getElementById('ntpm').textContent =
    (d.ntpm || 0) + ' NTPM \\u00b7 ' + SIDE + '\\u00d7' + SIDE;
}}

async function tick() {{
  if (!running) return;
  const r = await fetch('/api/time');
  const d = await r.json();
  const s = Math.max(0, Math.ceil(d.remaining));
  document.getElementById('timer').textContent =
    String(Math.floor(s/60)).padStart(2,'0') + ':' + String(s%60).padStart(2,'0');
  if (d.remaining <= 0) {{ clearInterval(iv); running = false; await showResults(); }}
}}

async function showResults() {{
  const r = await fetch('/api/result');
  const d = await r.json();
  document.getElementById('r-bps').textContent = (d.bps || 0).toFixed(2) + ' BPS';
  document.getElementById('r-ntpm').textContent = '(' + (d.ntpm || 0) + ' NTPM)';
  document.getElementById('results').style.display = 'flex';
  document.getElementById('game').style.display = 'none';
}}

canvas.addEventListener('click', async (e) => {{
  if (!running) return;
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((e.clientX - rect.left) / rect.width * CANVAS);
  const y = Math.floor((e.clientY - rect.top) / rect.height * CANVAS);
  const r = await fetch('/api/click', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{x, y}})
  }});
  const d = await r.json();
  drawImg(d.image);
  updateHud(d);
  if (d.done) {{ clearInterval(iv); running = false; await showResults(); }}
}});
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the game page."""
    return _html_page()


@app.post("/api/start")
def api_start() -> JSONResponse:
    """Start a new game session."""
    global _state, _wall_start
    _state = GameState(grid_size=_grid_size, canvas_size=_canvas_size)
    _state.start_time = time.time()
    _state.select_random_target()
    cx, cy = cell_center_pixel(0, 0, _state.grid_side, _canvas_size)
    _state.cursor_x, _state.cursor_y = cx, cy
    _wall_start = time.time()
    b64 = render_grid_screenshot(_state)
    return JSONResponse(
        {"image": b64, "score": 0, "incorrect": 0, "ntpm": 0, "bps": 0.0, "done": False}
    )


@app.post("/api/click")
def api_click(body: dict) -> JSONResponse:
    """Process a click at pixel (x, y) on the canvas."""
    if _state is None:
        return JSONResponse({"error": "not started"}, status_code=400)
    if (time.time() - _wall_start) >= _max_seconds:
        return _finish()
    x = max(0, min(int(body["x"]), _canvas_size - 1))
    y = max(0, min(int(body["y"]), _canvas_size - 1))
    _state.cursor_x, _state.cursor_y = x, y
    row, col = pixel_to_cell(x, y, _state.grid_side, _canvas_size)
    _state.click_at(row, col)
    b64 = render_grid_screenshot(_state)
    ntpm, bps = compute_ntpm_bps(_state.score, _state.incorrect_count, _grid_size)
    done = (time.time() - _wall_start) >= _max_seconds
    return JSONResponse(
        {
            "image": b64,
            "score": _state.score,
            "incorrect": _state.incorrect_count,
            "ntpm": int(ntpm),
            "bps": round(bps, 2),
            "done": done,
        }
    )


@app.get("/api/time")
def api_time() -> JSONResponse:
    """Return remaining game time in seconds."""
    return JSONResponse({"remaining": round(max(0, _max_seconds - (time.time() - _wall_start)), 1)})


@app.get("/api/result")
def api_result() -> JSONResponse:
    """Return final game results and log them."""
    return _finish()


def _finish() -> JSONResponse:
    if _state is None:
        return JSONResponse({"error": "not started"}, status_code=400)
    ntpm, bps = compute_ntpm_bps(_state.score, _state.incorrect_count, _grid_size)
    result = {
        "score": _state.score,
        "incorrect": _state.incorrect_count,
        "ntpm": int(ntpm),
        "bps": round(bps, 2),
        "grid_side": _state.grid_side,
        "elapsed_seconds": round(time.time() - _wall_start, 1),
        "done": True,
    }
    _results_log.append(result)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "web_games.json").write_text(json.dumps(_results_log, indent=2))

    return JSONResponse(result)


@app.get("/api/results")
def api_results() -> JSONResponse:
    """Return all logged game results."""
    return JSONResponse({"results": _results_log, "count": len(_results_log)})


def main() -> None:
    """CLI entrypoint: start the browser-based Webgrid game server."""
    global _grid_size, _canvas_size, _max_seconds

    parser = argparse.ArgumentParser(description="Webgrid browser game (Neuralink-style UI)")
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=f"Total grid cells, must be perfect square (default: {DEFAULT_GRID_SIZE} = 30x30)",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=DEFAULT_CANVAS_SIZE,
        help=f"Canvas resolution in pixels (default: {DEFAULT_CANVAS_SIZE})",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=DEFAULT_MAX_SECONDS,
        help=f"Game duration in seconds (default: {DEFAULT_MAX_SECONDS})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Server port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    side = int(args.grid_size**0.5)
    if side * side != args.grid_size:
        parser.error(f"grid-size must be a perfect square, got {args.grid_size}")

    _grid_size = args.grid_size
    _canvas_size = args.canvas_size
    _max_seconds = args.seconds

    print(f"Webgrid browser game: {side}x{side} grid, {_canvas_size}px canvas, {_max_seconds}s")
    print(f"Open http://localhost:{args.port} in your browser (F11 for fullscreen)")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
