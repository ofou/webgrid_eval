import argparse
import json
import sys
from pathlib import Path

import requests
import yaml


def _recalculate_results_json(results_dir: Path) -> None:
    """Scan results/*/result.json and write results/results.json with all results merged."""
    all_results: list[dict] = []
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
    all_results.sort(key=lambda r: (-r.get("score", 0), -r.get("ntpm", 0)))
    (results_dir / "results.json").write_text(json.dumps({"results": all_results}, indent=2))


def main() -> None:
    """CLI entrypoint: load YAML, POST to eval API, print results."""
    parser = argparse.ArgumentParser(description="Run webgrid eval from models YAML")
    parser.add_argument(
        "models_yaml",
        nargs="?",
        default=None,
        help="Path to YAML with 'models:' list (default: configs/models.yaml)",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=70,
        metavar="N",
        help="Stop eval per model after N seconds (default: 70, e.g. --seconds 70)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        metavar="N",
        help="Cap images per API request (e.g. 8 for Mistral)",
    )
    args = parser.parse_args()
    yaml_path = (
        Path(args.models_yaml)
        if args.models_yaml is not None
        else Path(__file__).parent.parent / "configs" / "models.yaml"
    )
    yaml_path = yaml_path.resolve()
    if not yaml_path.exists():
        print(f"Models file not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    models = list(data.get("models") or [])
    if not models:
        print(
            "No models in YAML (expect top-level key 'models': list of model IDs).",
            file=sys.stderr,
        )
        sys.exit(1)

    eval_api_base = data.get("eval_api_base") or data.get("server_url") or "http://127.0.0.1:8000"
    # Send models_file so server loads same YAML (single source of truth)
    models_file_str = str(yaml_path)
    body = {
        "models_file": models_file_str,
        "models": models,
        "grid_size": data.get("grid_size", 64),
        "canvas_size": data.get("canvas_size", 256),
        "max_seconds": (args.seconds if args.seconds is not None else data.get("max_seconds", 70)),
    }
    if data.get("base_url") is not None:
        body["base_url"] = data["base_url"]
    if data.get("api_key") is not None:
        body["api_key"] = data["api_key"]
    # Only send max_images when set in YAML or CLI (otherwise API uses None = no cap)
    max_images = data.get("max_images") if "max_images" in data else None
    if args.max_images is not None:
        max_images = args.max_images
    if max_images is not None:
        body["max_images"] = max_images
    resp = requests.post(
        f"{eval_api_base}/api/eval/run",
        json=body,
        timeout=7200,
    )
    if not resp.ok:
        try:
            err_body = resp.json()
            print(
                f"Server error ({resp.status_code}): {err_body.get('detail', resp.text)}",
                file=sys.stderr,
            )
        except Exception:
            print(resp.text, file=sys.stderr)
        resp.raise_for_status()
    out = resp.json()
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results = out.get("results") or []
    # Aggregate all results from subdirectories (includes all previous runs)
    _recalculate_results_json(results_dir)
    print(json.dumps({"results": results}, indent=2))
    for r in sorted(results, key=lambda x: (x.get("error") is not None, -x["score"], -x["ntpm"])):
        if r.get("error"):
            err = r["error"]
            print(f" {r['model']}: failed — {err[:80]}{'...' if len(err) > 80 else ''}")
        else:
            print(
                f" {r['model']}: BPS={r['bps']:.2f}, NTPM={r['ntpm']:.1f}, "
                f"score={r['score']}, incorrect={r['incorrect']}"
            )


if __name__ == "__main__":
    main()
