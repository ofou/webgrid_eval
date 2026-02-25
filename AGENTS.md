## Cursor Cloud specific instructions

### Project overview

Webgrid Eval is a Python (FastAPI) benchmark that evaluates LLM vision + tool-use on Neuralink's cursor control task. Single service, no database, no Docker.

### Common commands

All commands use `uv run` via `Makefile` targets — see `Makefile` for the full list:

- **Dev server**: `make dev` (uvicorn on port 8000 with `--reload`)
- **Lint**: `make lint` (ruff + mypy)
- **Format**: `make format` (black + ruff --fix)
- **Test**: `make test` (pytest with coverage)
- **Build**: `uv build`

### Caveats

- `uv` must be on PATH. It installs to `~/.local/bin`; the env is sourced via `source $HOME/.local/bin/env`.
- The `POST /api/session/start` and `POST /api/eval/run` endpoints require a live LLM API (configured via YAML + API key env var). Unit/integration tests mock the LLM calls, so `make test` works without any API key.
- `cairosvg` depends on system `libcairo2` — already present in the Cloud Agent base image (Ubuntu). If rendering fails with a Cairo error, install it: `apt-get install -y libcairo2-dev`.
- The ruff config in `pyproject.toml` uses deprecated top-level keys (`select`, `ignore`); ruff emits a warning but still works.
