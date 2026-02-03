"""Tests for FastAPI main module.

Tests for API endpoints and utility functions.
"""

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from webgrid_eval.main import (
    EvalModelResult,
    EvalRunRequest,
    SessionStartRequest,
    _format_peak_score,
    _load_config_from_yaml,
    _messages_for_dump,
    _model_to_dir_name,
    app,
    compute_ntpm_bps,
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestComputeNTPMBPS:
    """Test NTPM and BPS calculations."""

    def test_compute_positive_net(self):
        """Test calculation with positive net score."""
        correct, incorrect, elapsed, grid_size = 10, 2, 60.0, 64
        ntpm, bps = compute_ntpm_bps(correct, incorrect, elapsed, grid_size)

        net = correct - incorrect  # 8
        assert ntpm == float(net)
        expected_bps = (net / 60.0) * math.log2(grid_size)
        assert abs(bps - expected_bps) < 0.01

    def test_compute_zero_net(self):
        """Test calculation with zero net score."""
        correct, incorrect, elapsed, grid_size = 5, 5, 60.0, 64
        ntpm, bps = compute_ntpm_bps(correct, incorrect, elapsed, grid_size)

        assert ntpm == 0.0
        assert bps == 0.0

    def test_compute_negative_net(self):
        """Test calculation with negative net score."""
        correct, incorrect, elapsed, grid_size = 3, 10, 60.0, 64
        ntpm, bps = compute_ntpm_bps(correct, incorrect, elapsed, grid_size)

        assert ntpm == -7.0
        assert bps == 0.0  # BPS is 0 when net <= 0

    def test_compute_different_grid_sizes(self):
        """Test with different grid sizes."""
        # Larger grid should result in higher BPS for same net
        ntpm1, bps1 = compute_ntpm_bps(10, 0, 60.0, 64)  # 8x8
        ntpm2, bps2 = compute_ntpm_bps(10, 0, 60.0, 256)  # 16x16

        assert ntpm1 == ntpm2 == 10.0
        assert bps2 > bps1  # Larger grid = higher BPS


class TestFormatPeakScore:
    """Test peak score formatting."""

    def test_format_basic(self):
        """Test basic formatting."""
        result = _format_peak_score(9.98, 61.0)
        assert "9.98 BPS" in result
        assert "61 NTPM" in result

    def test_format_zero(self):
        """Test formatting zero values."""
        result = _format_peak_score(0.0, 0.0)
        assert "0.00 BPS" in result
        assert "0 NTPM" in result

    def test_format_rounding(self):
        """Test that BPS is rounded to 2 decimal places."""
        result = _format_peak_score(9.987654, 61.123)
        assert "9.99 BPS" in result
        assert "61 NTPM" in result


class TestModelToDirName:
    """Test model name to directory name conversion."""

    def test_model_with_slash(self):
        """Test model with slash separator."""
        assert _model_to_dir_name("moonshotai/kimi-k2.5") == "moonshotai-kimi-k2.5"

    def test_model_with_space(self):
        """Test model with space."""
        assert _model_to_dir_name("gpt 4") == "gpt_4"

    def test_model_simple(self):
        """Test simple model name."""
        assert _model_to_dir_name("gpt-4") == "gpt-4"


class TestMessagesForDump:
    """Test message formatting for dump."""

    def test_messages_basic(self):
        """Test basic message formatting."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        result = _messages_for_dump(messages)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_messages_with_image(self):
        """Test message with image URL."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc123"},
                    }
                ],
                "_screenshot_filename": "test.png",
            }
        ]

        result = _messages_for_dump(messages)

        # Should replace base64 with placeholder
        assert "data:image" not in result[0]["content"][0]["image_url"]["url"]
        assert "test.png" in result[0]["content"][0]["image_url"]["url"]

    def test_messages_removes_internal_keys(self):
        """Test that internal keys starting with _ are removed."""
        messages = [
            {
                "role": "user",
                "content": "Hello",
                "_internal_key": "should be removed",
            }
        ]

        result = _messages_for_dump(messages)

        assert "_internal_key" not in result[0]


class TestLoadConfigFromYAML:
    """Test YAML configuration loading."""

    def test_load_valid_yaml(self, tmp_path):
        """Test loading valid YAML file."""
        yaml_content = """
models:
  - model1
  - model2
base_url: http://localhost:8000
api_key: test-key
grid_size: 128
max_seconds: 60
max_images: 8
canvas_size: 512
"""
        yaml_file = tmp_path / "test_models.yaml"
        yaml_file.write_text(yaml_content)

        (
            models,
            base_url,
            api_key,
            grid_size,
            max_seconds,
            max_images,
            canvas_size,
        ) = _load_config_from_yaml(str(yaml_file))

        assert models == ["model1", "model2"]
        assert base_url == "http://localhost:8000"
        assert api_key == "test-key"
        assert grid_size == 128
        assert max_seconds == 60.0
        assert max_images == 8
        assert canvas_size == 512

    def test_load_defaults(self, tmp_path):
        """Test loading YAML with default values."""
        yaml_content = """
models:
  - model1
"""
        yaml_file = tmp_path / "test_models.yaml"
        yaml_file.write_text(yaml_content)

        (
            models,
            base_url,
            api_key,
            grid_size,
            max_seconds,
            max_images,
            canvas_size,
        ) = _load_config_from_yaml(str(yaml_file))

        assert models == ["model1"]
        assert base_url is None
        assert api_key is None
        assert grid_size == 64  # Default
        assert max_seconds == 70.0  # Default
        assert max_images is None
        assert canvas_size == 256  # Default

    def test_load_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            _load_config_from_yaml("/nonexistent/path/models.yaml")


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns ok."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestSessionStartRequest:
    """Test SessionStartRequest model."""

    def test_default_values(self):
        """Test default field values."""
        req = SessionStartRequest()

        assert req.model == "gpt-5.2"  # DEFAULT_MODEL
        assert req.grid_size == 64  # DEFAULT_GRID_SIZE
        assert req.max_seconds is None
        assert req.max_images is None
        assert req.base_url is None
        assert req.api_key is None
        assert req.canvas_size == 256

    def test_custom_values(self):
        """Test custom field values."""
        req = SessionStartRequest(
            model="custom-model",
            grid_size=144,  # 12x12 perfect square
            max_seconds=60,
            max_images=8,
            base_url="http://localhost:8000",
            api_key="test-key",
            canvas_size=512,
        )

        assert req.model == "custom-model"
        assert req.grid_size == 144
        assert req.max_seconds == 60
        assert req.max_images == 8
        assert req.base_url == "http://localhost:8000"
        assert req.api_key == "test-key"
        assert req.canvas_size == 512


class TestEvalRunRequest:
    """Test EvalRunRequest model."""

    def test_default_values(self):
        """Test default field values."""
        req = EvalRunRequest()

        assert req.models == []
        assert req.models_file is None
        assert req.grid_size == 64
        assert req.max_seconds is None
        assert req.max_images is None
        assert req.base_url is None
        assert req.api_key is None
        assert req.canvas_size == 256


class TestEvalModelResult:
    """Test EvalModelResult model."""

    def test_default_values(self):
        """Test default field values."""
        result = EvalModelResult(model="test-model")

        assert result.model == "test-model"
        assert result.score == 0
        assert result.incorrect == 0
        assert result.elapsed_seconds == 0.0
        assert result.ntpm == 0.0
        assert result.bps == 0.0
        assert result.peak_score == "0.00 BPS (0 NTPM)"
        assert result.grid_side == 8
        assert result.size_px == 256
        assert result.screenshots_dir is None
        assert result.error is None

    def test_with_results(self):
        """Test with actual results."""
        result = EvalModelResult(
            model="test-model",
            score=10,
            incorrect=2,
            elapsed_seconds=60.0,
            ntpm=8.0,
            bps=1.5,
            peak_score="1.50 BPS (8 NTPM)",
        )

        assert result.score == 10
        assert result.incorrect == 2
        assert result.peak_score == "1.50 BPS (8 NTPM)"

    def test_with_error(self):
        """Test with error field."""
        result = EvalModelResult(
            model="test-model",
            error="Rate limit exceeded",
        )

        assert result.error == "Rate limit exceeded"
        assert result.score == 0


class TestSessionStartEndpoint:
    """Test session start endpoint."""

    @patch("webgrid_eval.main.run_agentic_loop")
    @patch("webgrid_eval.main.GameState")
    def test_session_start_success(self, mock_gamestate, mock_run_loop, client):
        """Test successful session start."""
        # Setup mock
        mock_state = MagicMock()
        mock_state.score = 10
        mock_state.incorrect_count = 2
        mock_state.grid_side = 8
        mock_state.canvas_size = 256
        mock_gamestate.return_value = mock_state

        mock_run_loop.return_value = (10, [{"role": "user", "content": "test"}])

        response = client.post(
            "/api/session/start",
            json={
                "model": "test-model",
                "grid_size": 64,
                "max_seconds": 60,
                "canvas_size": 256,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test-model"
        assert data["score"] == 10
        assert data["incorrect"] == 2
        assert "bps" in data
        assert "ntpm" in data

    def test_session_start_invalid_json(self, client):
        """Test session start with invalid JSON."""
        response = client.post(
            "/api/session/start",
            data="invalid json",
        )

        assert response.status_code == 422  # Validation error


class TestEvalRunEndpoint:
    """Test eval run endpoint."""

    @patch("webgrid_eval.main._eval_single_model")
    @pytest.mark.asyncio
    async def test_eval_run_success(self, mock_eval, client):
        """Test successful eval run."""
        mock_eval.return_value = EvalModelResult(
            model="test-model",
            score=10,
            incorrect=2,
            bps=1.5,
            ntpm=8.0,
        )

        response = client.post(
            "/api/eval/run",
            json={
                "models": ["test-model"],
                "grid_size": 64,
                "max_seconds": 60,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["model"] == "test-model"

    def test_eval_run_empty_models(self, client):
        """Test eval run with empty models list."""
        response = client.post(
            "/api/eval/run",
            json={"models": []},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []


class TestResultsDirectory:
    """Test that results are stored in correct directory."""

    def test_results_directory_created(self, client, tmp_path):
        """Test that results directory is created during eval."""
        import shutil

        # Clean up any existing results
        if Path("results").exists():
            shutil.rmtree("results")

        # Mock the eval to avoid actual API calls
        with patch("webgrid_eval.main.run_agentic_loop") as mock_run:
            mock_run.return_value = (5, [])

            response = client.post(
                "/api/session/start",
                json={"model": "test-model", "grid_size": 64},
            )

            assert response.status_code == 200

        # Check that results directory was created
        assert Path("results").exists()
        assert Path("results/test-model").exists()
        assert Path("results/test-model/result.json").exists()

    def test_results_json_structure(self, client):
        """Test that results.json has correct structure."""
        import shutil

        # Clean up any existing results
        if Path("results").exists():
            shutil.rmtree("results")

        with patch("webgrid_eval.main.run_agentic_loop") as mock_run:
            mock_run.return_value = (5, [])

            client.post(
                "/api/session/start",
                json={"model": "test-model", "grid_size": 64},
            )

        # Check result.json structure
        with open("results/test-model/result.json") as f:
            data = json.load(f)

        assert "model" in data
        assert "score" in data
        assert "bps" in data
        assert "ntpm" in data
