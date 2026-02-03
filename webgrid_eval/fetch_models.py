"""Fetch models from OpenRouter or local LM Studio. Updates models.yaml options."""

import json
import urllib.request
from pathlib import Path
from typing import Any

import yaml  # Add this import; install via pip if needed

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
LOCAL_URL = (
    "http://localhost:1234/v1/models"  # Adjusted endpoint if needed; check your local API docs
)
MODELS_YAML = Path(__file__).parent / "openrouter.yaml"
LOCAL_YAML = Path(__file__).parent / "local.yaml"


def _fetch_openrouter(base_url: str) -> list[str]:
    req = urllib.request.Request(base_url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    models = data.get("data", [])

    def supported(m: dict[str, Any]) -> list[Any]:
        return m.get("supported_parameters") or []

    image_tools = [
        m["id"]
        for m in models
        if "image" in (m.get("architecture") or {}).get("input_modalities", [])
        and "tools" in supported(m)
        and "response_format" in supported(m)
    ]
    image_tools.sort()
    return image_tools


def _fetch_local(base_url: str) -> list[str]:
    req = urllib.request.Request(base_url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    # Adjust based on actual API response structure; assumes list of models with 'name' or 'id'
    models = data.get("data", [])
    model_ids = [m.get("id") or m.get("name") for m in models]
    model_ids.sort()
    return model_ids


def update_yaml(yaml_path: Path, model_ids: list[str]) -> None:
    """Update YAML file with given model_ids in the options key."""
    # Load existing YAML safely
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Update the 'options' key (assumes the YAML has a top-level 'options' list; adjust if nested)
    data["options"] = model_ids  # Or data['some_section']['options'] if nested

    # Write back with safe_dump to preserve YAML formatting
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    print(f"Updated {yaml_path}")


def main() -> None:
    """Fetch local and OpenRouter models and update YAML files."""
    local_model_ids = _fetch_local(LOCAL_URL)
    openrouter_model_ids = _fetch_openrouter(OPENROUTER_URL)
    print(local_model_ids)
    print(openrouter_model_ids)
    update_yaml(LOCAL_YAML, local_model_ids)
    update_yaml(MODELS_YAML, openrouter_model_ids)


if __name__ == "__main__":
    main()
