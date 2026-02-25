# Webgrid Eval

[![CI](https://github.com/ofou/webgrid_eval/actions/workflows/ci.yml/badge.svg)](https://github.com/ofou/webgrid_eval/actions/workflows/ci.yml)
[![wakatime](https://wakatime.com/badge/github/ofou/webgrid_eval.svg)](https://wakatime.com/badge/github/ofou/webgrid_eval)

Benchmark LLM vision + tool-use capabilities on Neuralink's cursor control task.

## Overview

At Neuralink, a game called [Webgrid](https://neuralink.com/webgrid) tests how precisely users can control a cursor. This benchmark evaluates LLMs on the same task: the model sees a grid with one blue target cell and must click it as fast as possible.

### Demo

<!-- markdownlint-disable MD033 -->
<figure align="center">
  <video src="./docs/img/claude-computer-use-webgrid-demo.mp4" controls width="600"></video>
  <figcaption><em>claude-4.6-opus (computer use) on 30×30 grid — 0.33 BPS (2 NTPM), 70s game</em></figcaption>
</figure>

<details>
<summary>gemini-3-flash-preview (API tool pipeline)</summary>
<figure align="center">
  <img src="./docs/img/gemini-3-flash-preview.gif" alt="gemini-3-flash-preview at 1x speed" width="400">
  <figcaption><em>gemini-3-flash-preview on 30×30 grid — 4 correct, 3 misclicks, 0.16 BPS (1 NTPM), in 70s task</em></figcaption>
</figure>
</details>
<!-- markdownlint-enable MD033 -->

### Human Baseline

For comparison: Neuralink's eighth clinical trial participant achieved 10.39 BPS controlling his computer with his brain; the highest mouse-based score mentioned is 17.1 BPS on a 35x35 grid (Neuralink employee).

### Metrics

The goal is to click targets on the grid as quickly as possible while minimizing misclicks. Score is measured in bits per second (BPS), derived from net correct clicks (NTPM) and grid size.

- **NTPM**: Net correct clicks = correct - incorrect
- **BPS**: `max((NTPM / 60) * log2(N - 1), 0)` where N = grid cells (e.g., 900 for 30×30)

Verified against the [Neuralink Webgrid frontend source](https://neuralink.com/webgrid):
`function E(f, t) { return Math.max(Math.log2(t * t - 1) * f / 60, 0) }`

### Benchmark Results

Results from 10 rounds on the browser-based eval (`make play`, 30×30 grid, 991px canvas, 70s, fullscreen):

| Model | Modality | Grid | Canvas | Round | NTPM | BPS |
| ----- | -------- | ---- | ------ | ----- | ---- | --- |
| claude-4.6-opus (computer use) | Browser click | 30×30 | 991px | 1 | 5 | 0.82 |
| claude-4.6-opus (computer use) | Browser click | 30×30 | 991px | 2 | 5 | 0.82 |
| claude-4.6-opus (computer use) | Browser click | 30×30 | 991px | 3 | 5 | 0.82 |
| claude-4.6-opus (computer use) | Browser click | 30×30 | 991px | 4 | 7 | 1.14 |
| claude-4.6-opus (computer use) | Browser click | 30×30 | 991px | 5 | 7 | 1.14 |
| claude-4.6-opus (computer use) | Browser click | 30×30 | 991px | 6 | 5 | 0.82 |
| claude-4.6-opus (computer use) | Browser click | 30×30 | 991px | 7 | 2 | 0.33 |
| claude-4.6-opus (computer use) | Browser click | 30×30 | 991px | 8 | 6 | 0.98 |
| claude-4.6-opus (computer use) | Browser click | 30×30 | 991px | 9 | 3 | 0.49 |
| claude-4.6-opus (computer use) | Browser click | 30×30 | 991px | 10 | 4 | 0.65 |
| | | | | **Avg** | **4.9** | **0.80** |

Comparison with other players:

| Player | Method | Grid | Best BPS | Avg BPS |
| ------ | ------ | ---- | -------- | ------- |
| Bliss Chapman | Mouse | 35×35 | **17.10** | — |
| Neuralink P8 | N1 Brain Implant | 30×30 | **10.39** | — |
| claude-4.6-opus | Computer use (browser click) | 30×30 | **1.14** | **0.80** |
| gemini-3-flash-preview | API tool pipeline | 30×30 | **0.16** | **~0.16** |

## Quick Start

### Installation

```bash
git clone git@github.com:ofou/webgrid_eval.git
cd webgrid_eval
make install-dev
```

### Play the game (default eval mode)

```bash
make play
# Open http://localhost:8000 in your browser (F11 for fullscreen)
```

### Run API-based evaluation (requires LLM API key)

```bash
# 1. Start the API server
make dev
```

```bash
# 2. In another terminal, run the evaluation
make eval ARGS="configs/openrouter.yaml"
```

## Usage

### Browser Game (default eval)

```bash
# Start the game (30×30 grid, 991px canvas, Neuralink-identical UI)
make play
# Open http://localhost:8000 → F11 for fullscreen → click blue cells
```

Results are logged to `results/web_games.json`.

### Configure Models (API eval)

Create a YAML configuration file (see `configs/` for examples):

```yaml
# configs/my_models.yaml
base_url: https://openrouter.ai/api/v1
grid_size: 64 # 8×8 grid (64 cells)
canvas_size: 256 # screenshot size in pixels
max_seconds: 70 # evaluation duration per model

models:
  - google/gemini-3-flash-preview
  - qwen/qwen3-vl-235b-a22b-instruct
```

Available configs:

- `configs/openrouter.yaml` - OpenRouter API (many models)
- `configs/google.yaml` - Google AI API (Gemini models)
- `configs/local.yaml` - Local LLM server (e.g., LM Studio, Ollama)

### Run Evaluation

```bash
# Run with a config file
make eval ARGS="configs/openrouter.yaml"

# With custom duration (seconds)
make eval ARGS="configs/openrouter.yaml --seconds 120"

# Cap images per API request (for models with limits)
make eval ARGS="configs/openrouter.yaml --max-images 8"
```

### API Endpoints

When the server is running (`make dev`):

- `GET /health` - Health check
- `POST /api/session/start` - Run single model evaluation
- `POST /api/eval/run` - Run batch evaluation (multiple models)

### Generate Replay GIFs

```bash
# Generate GIFs for all evaluation results
make gif

# Or for a specific evaluation folder
make gif ARGS="eval/model-name"
```

## Tools

The LLM agent has access to three tools:

| Tool          | Description                                                     |
| ------------- | --------------------------------------------------------------- |
| `screen`      | Returns current HUD + screenshot (like looking at your monitor) |
| `mouse_move`  | Move cursor by (dx, dy) pixels. Positive dx=right, dy=down      |
| `mouse_click` | Click at the current cursor position                            |

## Citation

If you use this software in your research, please cite:

```bibtex
@software{olivares2026webgrid,
  author  = {Olivares Urrutia, Omar},
  title   = {{Webgrid Eval: Benchmark for LLM Vision and Tool-Use Capabilities}},
  year    = {2026},
  month   = feb,
  url     = {https://github.com/ofou/webgrid_eval},
}
```

## Acknowledgments

- Inspired by [Neuralink's Webgrid](https://neuralink.com/webgrid)

## Contributing

Contributions are welcome!
