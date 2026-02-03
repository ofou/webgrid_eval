.PHONY: clean install dev test eval help lint format docs

help:
	@echo "Available targets:"
	@echo "  clean    - Remove evaluation results, screenshots, and cache files"
	@echo "  install  - Install project dependencies"
	@echo "  install-dev - Install project with development dependencies"
	@echo "  dev      - Run FastAPI server in development mode"
	@echo "  test     - Run unit tests"
	@echo "  eval     - Run model evaluation (use ARGS='--seconds 30' for options)"
	@echo "  gif      - Generate replay GIFs from evaluation results"
	@echo "  lint     - Run code linters (ruff, mypy)"
	@echo "  format   - Format code with black and ruff"
	@echo "  docs     - Build documentation"

clean:
	@echo "Cleaning up..."
	rm -rf eval/
	rm -rf results/
	rm -rf webgrid_eval/__pycache__/
	rm -rf webgrid_eval/*.pyc
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	@echo "Clean complete!"

install:
	@echo "Installing dependencies..."
	uv sync
	@echo "Installation complete!"

install-dev:
	@echo "Installing dependencies with dev tools..."
	uv sync --extra dev
	uv run pre-commit install
	@echo "Installation complete!"

dev:
	@echo "Starting FastAPI server..."
	uv run uvicorn webgrid_eval.main:app --reload --host 0.0.0.0 --port 8000

test:
	@echo "Running tests..."
	uv run pytest --cov=webgrid_eval --cov-report=term-missing

eval:
	@echo "Running evaluation..."
	uv run python -m webgrid_eval.run_eval $(ARGS)

gif:
	@echo "Generating GIFs..."
	uv run python -m webgrid_eval.make_gif $(ARGS)

lint:
	@echo "Running linters..."
	uv run ruff check webgrid_eval tests
	uv run mypy webgrid_eval

format:
	@echo "Formatting code..."
	uv run black webgrid_eval tests
	uv run ruff check --fix webgrid_eval tests

docs:
	@echo "Building documentation..."
	@echo "Documentation not yet implemented"
