"""FastAPI app: CORS, POST /api/session/start (agentic loop), POST /api/eval/run (multi-model)."""

import asyncio
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from .game_state import GameState
from .openrouter import DEFAULT_MODEL, run_agentic_loop

DEFAULT_GRID_SIDE = 8  # 8x8 grid (easier task, 32px cells on 256 canvas)
DEFAULT_GRID_SIZE = DEFAULT_GRID_SIDE * DEFAULT_GRID_SIDE


def compute_ntpm_bps(
    correct: int, incorrect: int, elapsed_seconds: float, grid_size: int
) -> tuple[float, float]:
    """Net = correct - incorrect.
    NTPM = Net (raw count).
    BPS (Neuralink): (net / 60) * log2(grid_size² - 1).

    Note: grid_size is the number of cells (e.g. 64 for 8×8).
    """
    net = correct - incorrect
    # Frontend: NTPM is just the net count (displayed as "NTPM: {net}")
    ntpm = float(net)

    # Neuralink: BPS = (net / 60) * log2(N² - 1), N = grid_size (number of cells)
    # If net <= 0, it returns 0.
    if net <= 0:
        return ntpm, 0.0

    bps = (net / 60.0) * math.log2(grid_size**2 - 1)
    return ntpm, bps


def _format_peak_score(bps: float, ntpm: float) -> str:
    """Format peak score like frontend: '9.98 BPS (61 NTPM)'."""
    return f"{bps:.2f} BPS ({int(ntpm)} NTPM)"


app = FastAPI(title="Webgrid Eval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionStartRequest(BaseModel):
    model: str = DEFAULT_MODEL
    grid_size: int = DEFAULT_GRID_SIZE  # 8x8
    max_seconds: int | None = None  # stop eval after this many seconds per model
    max_images: int | None = None  # cap images per request (e.g. 8 for Mistral)
    base_url: str | None = None  # LLM API base URL (from YAML or env)
    api_key: str | None = None  # LLM API key (from YAML or env)
    canvas_size: int = 256  # screenshot canvas size in pixels

    @field_validator("grid_size")
    @classmethod
    def validate_grid_size(cls, v: int) -> int:
        """Validate that grid_size is a perfect square."""
        side = int(v**0.5)
        if side * side != v:
            raise ValueError(f"grid_size must be a perfect square, got {v}")
        return v


class SessionStartResponse(BaseModel):
    model: str
    score: int
    incorrect: int
    elapsed_seconds: float
    ntpm: float
    bps: float
    peak_score: str
    grid_side: int
    size_px: int
    messages_count: int
    history: list[dict[str, Any]] | None = None
    screenshots_dir: str | None = None


def _model_to_dir_name(model: str) -> str:
    """e.g. moonshotai/kimi-k2.5 -> moonshotai-kimi-k2.5"""
    return model.replace("/", "-").replace(" ", "_")


def _messages_for_dump(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy messages and replace inline base64 image URLs with a placeholder so the dump stays small."""
    out: list[dict[str, Any]] = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        # Extract filename if present (internal metadata)
        filename = m.get("_screenshot_filename")

        # Remove internal keys for the dump
        m = {k: v for k, v in m.items() if not k.startswith("_")}

        if isinstance(content, list):
            content = [
                (
                    {**part, "image_url": {"url": filename}}
                    if filename and part.get("type") == "image_url"
                    else (
                        {**part, "image_url": {"url": "(image saved in this folder)"}}
                        if part.get("type") == "image_url"
                        and isinstance(part.get("image_url"), dict)
                        and str(part.get("image_url", {}).get("url", "")).startswith("data:image")
                        else part
                    )
                )
                for part in content
            ]
            m["content"] = content
        out.append(m)
    return out


@app.post("/api/session/start", response_model=SessionStartResponse)
def session_start(req: SessionStartRequest) -> SessionStartResponse:
    """Start a session: init game state, run agentic loop with OpenRouter (screen, mouse_move, mouse_click), return score and optional history. Screenshots are saved under results/<model_name>/<unixtime>.png."""
    state = GameState(grid_size=req.grid_size, canvas_size=req.canvas_size)
    state.start_time = time.time()
    state.select_random_target()
    save_dir = str(Path("results") / _model_to_dir_name(req.model))
    # Clear existing folder to ensure fresh results
    save_dir_path = Path(save_dir)
    if save_dir_path.exists():
        shutil.rmtree(save_dir_path)
    max_sec = float(req.max_seconds) if req.max_seconds is not None else 70.0
    score, messages = run_agentic_loop(
        state,
        model=req.model,
        save_dir=save_dir,
        max_seconds=max_sec,
        max_images=req.max_images,
        base_url=req.base_url,
        api_key=req.api_key,
    )
    # Match official implementation: fixed duration (e.g. 70s), not variable actual elapsed
    elapsed = max_sec
    ntpm, bps = compute_ntpm_bps(state.score, state.incorrect_count, elapsed, req.grid_size)
    resp = SessionStartResponse(
        model=req.model,
        score=state.score,
        incorrect=state.incorrect_count,
        elapsed_seconds=elapsed,
        ntpm=ntpm,
        bps=bps,
        peak_score=_format_peak_score(bps, ntpm),
        grid_side=state.grid_side,
        size_px=state.canvas_size,
        messages_count=len(messages),
        history=messages,
        screenshots_dir=save_dir,
    )

    result_path = Path(save_dir) / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result_path.write_text(json.dumps(resp.model_dump(exclude={"history"}), indent=2))
    except OSError as e:
        raise RuntimeError(f"Failed to write result.json to {result_path}: {e}")
    history_path = Path(save_dir) / "history.json"
    history_path.write_text(json.dumps(_messages_for_dump(messages), indent=2, default=str))
    return resp


class EvalRunRequest(BaseModel):
    models: list[str] = []
    models_file: str | None = (
        None  # path to YAML with "models: [...]" and optional base_url, api_key
    )
    grid_size: int = DEFAULT_GRID_SIZE  # 8x8
    max_seconds: int | None = None  # stop eval after this many seconds per model
    max_images: int | None = None  # cap images per request (e.g. 8 for Mistral)
    base_url: str | None = None  # LLM API base URL (from YAML or body)
    api_key: str | None = None  # LLM API key (from YAML or body)
    canvas_size: int = 256  # screenshot canvas size in pixels


class EvalModelResult(BaseModel):
    model: str
    score: int = 0
    incorrect: int = 0
    elapsed_seconds: float = 0.0
    ntpm: float = 0.0
    bps: float = 0.0
    peak_score: str = "0.00 BPS (0 NTPM)"
    grid_side: int = 8
    size_px: int = 256
    screenshots_dir: str | None = None
    error: str | None = None  # set when eval failed (e.g. rate limit, spend limit)


class EvalRunResponse(BaseModel):
    results: list[EvalModelResult]


def _load_config_from_yaml(
    path: str,
) -> tuple[list[str], str | None, str | None, int, float, int | None, int]:
    """Load config from YAML. Returns (models, base_url, api_key, grid_size, max_seconds, max_images, canvas_size)."""
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / path
    if not p.exists():
        raise FileNotFoundError(f"Models file not found: {p}")
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    models = list(data.get("models") or [])
    base_url = data.get("base_url") or None
    api_key = data.get("api_key") or None
    grid_size = data.get("grid_size", DEFAULT_GRID_SIZE)
    max_seconds = float(data.get("max_seconds", 70))
    max_images = data.get("max_images") if "max_images" in data else None
    canvas_size = data.get("canvas_size", 256)
    return models, base_url, api_key, grid_size, max_seconds, max_images, canvas_size


async def _eval_single_model(
    model: str,
    grid_size: int,
    max_seconds: float,
    max_images: int | None,
    base_url: str | None = None,
    api_key: str | None = None,
    canvas_size: int = 256,
) -> EvalModelResult:
    """Evaluate a single model (runs in executor to avoid blocking)."""
    loop = asyncio.get_event_loop()

    def _run_sync():
        state = GameState(grid_size=grid_size, canvas_size=canvas_size)
        state.start_time = time.time()
        state.select_random_target()
        save_dir = str(Path("results") / _model_to_dir_name(model))
        # Clear existing folder to ensure fresh results
        save_dir_path = Path(save_dir)
        if save_dir_path.exists():
            shutil.rmtree(save_dir_path)
        max_sec = max_seconds
        _, messages = run_agentic_loop(
            state,
            model=model,
            save_dir=save_dir,
            max_seconds=max_sec,
            max_images=max_images,
            base_url=base_url,
            api_key=api_key,
        )
        history_path = Path(save_dir) / "history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(_messages_for_dump(messages), indent=2, default=str))
        # Match official implementation: fixed duration (e.g. 70s), not variable actual elapsed
        elapsed = max_sec
        ntpm, bps = compute_ntpm_bps(state.score, state.incorrect_count, elapsed, grid_size)
        result = EvalModelResult(
            model=model,
            score=state.score,
            incorrect=state.incorrect_count,
            elapsed_seconds=elapsed,
            ntpm=ntpm,
            bps=bps,
            peak_score=_format_peak_score(bps, ntpm),
            grid_side=state.grid_side,
            size_px=state.canvas_size,
            screenshots_dir=save_dir,
        )
        result_path = Path(save_dir) / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result.model_dump(), indent=2))
        return result

    # Run blocking I/O in executor to avoid blocking event loop
    return await loop.run_in_executor(None, _run_sync)


@app.post("/api/eval/run", response_model=EvalRunResponse)
async def eval_run(req: EvalRunRequest) -> EvalRunResponse:
    """Run sessions in parallel; return BPS and NTPM per model. When models_file is set, server loads YAML (base_url, api_key, grid_size, max_seconds, max_images); body overrides."""
    models = list(req.models)
    base_url = req.base_url
    api_key = req.api_key
    grid_size = req.grid_size
    max_sec = float(req.max_seconds) if req.max_seconds is not None else 70.0
    max_images = req.max_images

    canvas_size = req.canvas_size
    if req.models_file:
        try:
            (
                file_models,
                file_base_url,
                file_api_key,
                file_grid_size,
                file_max_seconds,
                file_max_images,
                file_canvas_size,
            ) = _load_config_from_yaml(req.models_file)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        if not models:
            models = file_models
        if file_base_url is not None:
            base_url = file_base_url
        if file_api_key is not None:
            api_key = file_api_key
        grid_size = file_grid_size
        max_sec = file_max_seconds
        canvas_size = file_canvas_size
        if file_max_images is not None:
            max_images = file_max_images
        if req.grid_size != DEFAULT_GRID_SIZE:
            grid_size = req.grid_size
        if req.max_seconds is not None:
            max_sec = float(req.max_seconds)
        if req.max_images is not None:
            max_images = req.max_images
        if req.canvas_size != 256:
            canvas_size = req.canvas_size

    if not models:
        return EvalRunResponse(results=[])

    # OpenRouter free tier: run sequentially to avoid 429 burst; local: run in parallel
    is_openrouter = base_url and "openrouter" in base_url.lower()
    if is_openrouter:
        results = []
        for model in models:
            try:
                r = await _eval_single_model(
                    model=model,
                    grid_size=grid_size,
                    max_seconds=max_sec,
                    max_images=max_images,
                    base_url=base_url,
                    api_key=api_key,
                    canvas_size=canvas_size,
                )
                results.append(r)
            except BaseException as e:
                results.append(e)
    else:
        tasks = [
            _eval_single_model(
                model=model,
                grid_size=grid_size,
                max_seconds=max_sec,
                max_images=max_images,
                base_url=base_url,
                api_key=api_key,
                canvas_size=canvas_size,
            )
            for model in models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        results = list(results)
    # Build final results: successful evals + error entries for failed models (partial results)
    final_results: list[EvalModelResult] = []
    for i, r in enumerate(results):
        if isinstance(r, BaseException):
            final_results.append(
                EvalModelResult(
                    model=models[i],
                    error=str(r),
                    grid_side=grid_size,
                )
            )
        elif isinstance(r, EvalModelResult):
            final_results.append(r)

    # Save aggregated results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    sorted_results = sorted(
        final_results,
        key=lambda r: (r.error is not None, -r.score, -r.ntpm),
    )
    (results_dir / "results.json").write_text(
        json.dumps({"results": [r.model_dump() for r in sorted_results]}, indent=2)
    )
    return EvalRunResponse(results=sorted_results)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
