"""FastAPI app: CORS, POST /api/session/start (agentic loop), POST /api/eval/run (multi-model)."""

import asyncio
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, cast

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

    # Neuralink Webgrid frontend formula: BPS = (net / 60) * log2(N)
    # where N = total grid cells (e.g., 900 for 30x30)
    # Reference: _enviroment/src/features/game/components/grid/hooks/useBitsPerSecond.tsx
    # If net <= 0, it returns 0.
    if net <= 0:
        return ntpm, 0.0

    bps = (net / 60.0) * math.log2(grid_size)
    return ntpm, bps


def _format_peak_score(bps: float, ntpm: float) -> str:
    """Format peak score like frontend: '9.98 BPS (61 NTPM)'."""
    return f"{bps:.2f} BPS ({int(ntpm)} NTPM)"


def _aggregate_results_json(results_dir: Path) -> list[dict[str, Any]]:
    """Scan results_dir/*/result.json and return merged results sorted by score."""
    all_results: list[dict[str, Any]] = []
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        result_file = subdir / "result.json"
        if not result_file.exists():
            continue
        try:
            data = json.loads(result_file.read_text())
            if not isinstance(data, dict):
                continue
            if "screenshots_dir" not in data:
                data["screenshots_dir"] = str(subdir)
            all_results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    # Sort by BPS descending, then NTPM descending
    all_results.sort(key=lambda r: (-r.get("bps", 0), -r.get("ntpm", 0)))
    return all_results


app = FastAPI(title="Webgrid Eval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionStartRequest(BaseModel):
    """Request body for starting an eval session."""

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
    """Response after session completes (score, NTPM, BPS, etc.)."""

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


def _model_to_dir_name(model: str, params: dict[str, Any] | None = None) -> str:
    """Generate folder name from model + params.

    Examples:
        - "gemini-3-flash-preview" -> "gemini-3-flash-preview"
        - "gemini-3-flash-preview", {"reasoning_effort": "low"} -> "gemini-3-flash-preview:low"
        - "openai/gpt-4o" -> "openai-gpt-4o"
    """
    name = model.replace("/", "-").replace(" ", "_")
    if params:
        if "reasoning_effort" in params:
            name = f"{name}:{params['reasoning_effort']}"
        else:
            for k, v in sorted(params.items()):
                name = f"{name}:{k}={v}"
    return name


def parse_model_config(model_entry: str | dict) -> tuple[str, dict[str, Any]]:
    """Parse a model entry from config.

    Supports both formats:
      - Simple string: "gemini-3-flash-preview"
      - Dict with params: {id: "gemini-3-flash-preview", reasoning_effort: "low", ...}

    Returns:
        Tuple of (model_id, extra_params)
    """
    if isinstance(model_entry, str):
        return model_entry, {}
    elif isinstance(model_entry, dict):
        model_id = model_entry.get("id") or model_entry.get("model")
        if not model_id:
            raise ValueError(f"Model config missing 'id' field: {model_entry}")
        extra_params = {k: v for k, v in model_entry.items() if k not in ("id", "model")}
        return model_id, extra_params
    else:
        raise ValueError(f"Invalid model entry type: {type(model_entry)}")


def _messages_for_dump(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy messages and replace base64 image URLs with a placeholder so the dump stays small."""
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
    """Start a session: run agentic loop (screen, mouse_move, mouse_click), return score."""
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
        raise RuntimeError(f"Failed to write result.json to {result_path}: {e}") from e
    history_path = Path(save_dir) / "history.json"
    history_path.write_text(json.dumps(_messages_for_dump(messages), indent=2, default=str))
    return resp


class EvalRunRequest(BaseModel):
    """Request body for batch eval (models list or models_file path)."""

    models: list[str | dict[str, Any]] = []  # string or dict with {id, reasoning_effort, ...}
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
    """Single model result (score, NTPM, BPS) or error."""

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
    """Batch eval response: list of EvalModelResult."""

    results: list[EvalModelResult]


def _load_config_from_yaml(
    path: str,
) -> tuple[list[str | dict[str, Any]], str | None, str | None, int, float, int | None, int]:
    """Load YAML. Returns models, base_url, api_key, grid_size, max_seconds, max_images, canvas."""
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
    model_params: dict[str, Any] | None = None,
) -> EvalModelResult:
    """Evaluate a single model (runs in executor to avoid blocking)."""
    loop = asyncio.get_event_loop()
    reasoning_effort = model_params.get("reasoning_effort") if model_params else None

    def _run_sync() -> EvalModelResult:
        state = GameState(grid_size=grid_size, canvas_size=canvas_size)
        state.start_time = time.time()
        state.select_random_target()
        save_dir = str(Path("eval") / _model_to_dir_name(model, model_params))
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
            reasoning_effort=reasoning_effort,
        )
        history_path = Path(save_dir) / "history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(_messages_for_dump(messages), indent=2, default=str))
        # Match official implementation: fixed duration (e.g. 70s), not variable actual elapsed
        elapsed = max_sec
        ntpm, bps = compute_ntpm_bps(state.score, state.incorrect_count, elapsed, grid_size)
        # Use folder name (model:param) as the result model name
        result_model_name = _model_to_dir_name(model, model_params)
        result = EvalModelResult(
            model=result_model_name,
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
    """Run sessions in parallel; return BPS and NTPM per model.

    When models_file is set, server loads YAML (base_url, api_key, etc.); body overrides.
    """
    models: list[str | dict[str, Any]] = list(req.models)
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
            models = list(file_models)
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

    # Parse model configs (support both string and dict formats)
    parsed_models = []
    for model_entry in models:
        model_id, model_params = parse_model_config(model_entry)
        parsed_models.append((model_id, model_params))

    # OpenRouter free tier: run sequentially to avoid 429 burst; local: run in parallel
    is_openrouter = base_url and "openrouter" in base_url.lower()
    if is_openrouter:
        results: list[EvalModelResult | BaseException] = []
        for model_id, model_params in parsed_models:
            try:
                r = await _eval_single_model(
                    model=model_id,
                    grid_size=grid_size,
                    max_seconds=max_sec,
                    max_images=max_images,
                    base_url=base_url,
                    api_key=api_key,
                    canvas_size=canvas_size,
                    model_params=model_params,
                )
                results.append(r)
            except BaseException as e:
                results.append(e)
    else:
        tasks = [
            _eval_single_model(
                model=model_id,
                grid_size=grid_size,
                max_seconds=max_sec,
                max_images=max_images,
                base_url=base_url,
                api_key=api_key,
                canvas_size=canvas_size,
                model_params=model_params,
            )
            for model_id, model_params in parsed_models
        ]
        gathered: list[EvalModelResult | BaseException] = list(
            await asyncio.gather(*tasks, return_exceptions=True)
        )
        results = gathered
    # Build final results: successful evals + error entries for failed models (partial results)
    final_results: list[EvalModelResult] = []
    for i, raw in enumerate(results):
        if isinstance(raw, BaseException):
            model_id, model_params = parsed_models[i]
            # Include reasoning_effort in model name for error results
            model_name = _model_to_dir_name(model_id, model_params) if model_params else model_id
            final_results.append(
                EvalModelResult(
                    model=model_name,
                    error=str(raw),
                    grid_side=grid_size,
                )
            )
        elif isinstance(raw, EvalModelResult):
            final_results.append(cast(EvalModelResult, raw))

    # Save aggregated results from all subdirectories
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = _aggregate_results_json(results_dir)
    (results_dir / "results.json").write_text(json.dumps({"results": all_results}, indent=2))

    # Return results from this run
    sorted_results = sorted(
        final_results,
        key=lambda r: (r.error is not None, -r.score, -r.ntpm),
    )
    return EvalRunResponse(results=sorted_results)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
