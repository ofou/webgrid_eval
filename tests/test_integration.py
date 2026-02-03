"""Integration tests for FastAPI API endpoints.

Tests that actually start the server and make real HTTP requests.
Uses FastAPI TestClient and can also test with actual HTTP calls.
"""

import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

from webgrid_eval.main import app


class TestAPIIntegration(unittest.TestCase):
    """Integration tests using unittest framework."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running all tests."""
        cls.client = TestClient(app)
        cls.base_url = "http://127.0.0.1:8000"

        # Clean up results directory
        if Path("results").exists():
            shutil.rmtree("results")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Clean up results
        if Path("results").exists():
            shutil.rmtree("results")

    def setUp(self):
        """Set up before each test."""
        # Clean results before each test
        if Path("results").exists():
            shutil.rmtree("results")

    # ==================== Health Endpoint Tests ====================

    def test_health_endpoint_returns_ok(self):
        """Test that /health returns status ok."""
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")

    # ==================== Session Start Endpoint Tests ====================

    @patch("webgrid_eval.main.run_agentic_loop")
    def test_session_start_basic(self, mock_run_loop):
        """Test basic session start with mocked agentic loop."""

        # Setup mock that modifies state.score to set the score
        def mock_side_effect(state, *args, **kwargs):
            state.score = 5  # Set the score on the state object
            return (5, [{"role": "user", "content": "test message"}])

        mock_run_loop.side_effect = mock_side_effect

        response = self.client.post(
            "/api/session/start",
            json={
                "model": "test-model",
                "grid_size": 64,
                "max_seconds": 10,
                "canvas_size": 256,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify response structure
        self.assertEqual(data["model"], "test-model")
        self.assertEqual(data["score"], 5)
        self.assertEqual(data["grid_side"], 8)
        self.assertEqual(data["size_px"], 256)
        self.assertIn("bps", data)
        self.assertIn("ntpm", data)
        self.assertIn("peak_score", data)
        self.assertIn("history", data)

        # Verify that run_agentic_loop was called
        mock_run_loop.assert_called_once()

    @patch("webgrid_eval.main.run_agentic_loop")
    def test_session_start_creates_results_directory(self, mock_run_loop):
        """Test that session start creates results directory structure."""
        mock_run_loop.return_value = (3, [])

        self.client.post(
            "/api/session/start",
            json={
                "model": "integration-test-model",
                "grid_size": 64,
                "max_seconds": 5,
            },
        )

        # Check that results directory was created
        results_dir = Path("results")
        self.assertTrue(results_dir.exists())

        # Check that model directory was created
        model_dir = results_dir / "integration-test-model"
        self.assertTrue(model_dir.exists())

        # Check that result.json was created
        result_file = model_dir / "result.json"
        self.assertTrue(result_file.exists())

        # Check that history.json was created
        history_file = model_dir / "history.json"
        self.assertTrue(history_file.exists())

    @patch("webgrid_eval.main.run_agentic_loop")
    def test_session_start_result_json_content(self, mock_run_loop):
        """Test that result.json has correct content."""

        # Setup mock that modifies state.score to set the score
        def mock_side_effect(state, *args, **kwargs):
            state.score = 10  # Set the score on the state object
            return (10, [{"role": "assistant", "content": "test"}])

        mock_run_loop.side_effect = mock_side_effect

        self.client.post(
            "/api/session/start",
            json={
                "model": "json-test-model",
                "grid_size": 64,
                "max_seconds": 5,
            },
        )

        # Read and verify result.json
        with open("results/json-test-model/result.json") as f:
            data = json.load(f)

        self.assertEqual(data["model"], "json-test-model")
        self.assertEqual(data["score"], 10)
        self.assertEqual(data["grid_side"], 8)
        self.assertIn("bps", data)
        self.assertIn("ntpm", data)
        self.assertIn("screenshots_dir", data)

    def test_session_start_invalid_grid_size(self):
        """Test session start with invalid grid size."""
        response = self.client.post(
            "/api/session/start",
            json={
                "model": "test-model",
                "grid_size": 65,  # Not a perfect square
                "max_seconds": 5,
            },
        )

        # Should fail with validation error
        self.assertEqual(response.status_code, 422)

    # ==================== Eval Run Endpoint Tests ====================

    @patch("webgrid_eval.main._eval_single_model")
    def test_eval_run_single_model(self, mock_eval):
        """Test eval run with single model."""
        from webgrid_eval.main import EvalModelResult

        # For async functions, we can return the result directly
        # TestClient handles the async execution
        mock_eval.return_value = EvalModelResult(
            model="test-model",
            score=15,
            incorrect=2,
            ntpm=13.0,
            bps=2.0,
            elapsed_seconds=60.0,
        )

        response = self.client.post(
            "/api/eval/run",
            json={
                "models": ["test-model"],
                "grid_size": 64,
                "max_seconds": 60,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 1)

        result = data["results"][0]
        self.assertEqual(result["model"], "test-model")
        self.assertEqual(result["score"], 15)

    @patch("webgrid_eval.main._eval_single_model")
    def test_eval_run_multiple_models(self, mock_eval):
        """Test eval run with multiple models."""
        from webgrid_eval.main import EvalModelResult

        def side_effect(*args, **kwargs):
            model = kwargs.get("model", "unknown")
            return EvalModelResult(
                model=model,
                score=10 if model == "model1" else 20,
                incorrect=0,
                ntpm=10.0 if model == "model1" else 20.0,
            )

        mock_eval.side_effect = side_effect

        response = self.client.post(
            "/api/eval/run",
            json={
                "models": ["model1", "model2"],
                "grid_size": 64,
                "max_seconds": 60,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(len(data["results"]), 2)

        # Results should be sorted by score (descending)
        scores = [r["score"] for r in data["results"]]
        self.assertEqual(scores, [20, 10])

    @patch("webgrid_eval.main._eval_single_model")
    def test_eval_run_empty_models(self, mock_eval):
        """Test eval run with empty models list."""
        response = self.client.post("/api/eval/run", json={"models": []})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["results"], [])

    @patch("webgrid_eval.main._eval_single_model")
    def test_eval_run_creates_results_json(self, mock_eval):
        """Test that eval run creates results.json (aggregates from results/ dir)."""
        from webgrid_eval.main import EvalModelResult

        mock_eval.return_value = EvalModelResult(
            model="test-model",
            score=10,
            incorrect=2,
        )

        self.client.post(
            "/api/eval/run",
            json={
                "models": ["test-model"],
                "grid_size": 64,
                "max_seconds": 60,
            },
        )

        # Check that results.json was created
        results_json = Path("results/results.json")
        self.assertTrue(results_json.exists())

        # Verify content - aggregation scans results/ dir (session results),
        # not eval/ dir where _eval_single_model writes. So expect empty aggregation
        # when only eval/run is called (no session/start calls).
        with open(results_json) as f:
            data = json.load(f)

        self.assertIn("results", data)
        # Aggregation is empty since eval results go to eval/, not results/
        self.assertEqual(len(data["results"]), 0)

    # ==================== Error Handling Tests ====================

    @patch("webgrid_eval.main._eval_single_model")
    def test_eval_run_with_error(self, mock_eval):
        """Test eval run when model evaluation fails."""
        from webgrid_eval.main import EvalModelResult

        mock_eval.return_value = EvalModelResult(
            model="error-model",
            error="Rate limit exceeded",
        )

        response = self.client.post(
            "/api/eval/run",
            json={
                "models": ["error-model"],
                "grid_size": 64,
                "max_seconds": 60,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        result = data["results"][0]
        self.assertEqual(result["model"], "error-model")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Rate limit exceeded")

    def test_invalid_endpoint(self):
        """Test that invalid endpoint returns 404."""
        response = self.client.get("/invalid/endpoint")
        self.assertEqual(response.status_code, 404)

    def test_invalid_method(self):
        """Test that wrong HTTP method returns 405."""
        response = self.client.get("/api/session/start")  # Should be POST
        self.assertEqual(response.status_code, 405)


@pytest.mark.skip(reason="HTTP integration tests - run explicitly with --run-http-tests")
class TestAPIWithActualHTTP(unittest.TestCase):
    """Integration tests that start actual server and make HTTP requests.

    These tests are skipped by default because they start a real server.
    Run them explicitly with: pytest tests/test_integration.py --run-http-tests
    """

    @classmethod
    def setUpClass(cls):
        """Start the server in background for actual HTTP tests."""
        import subprocess
        import time

        # Clean up
        if Path("results").exists():
            shutil.rmtree("results")

        # Start server as subprocess
        cls.server_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "webgrid_eval.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8001",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start
        cls.base_url = "http://127.0.0.1:8001"
        max_retries = 30
        for _i in range(max_retries):
            try:
                response = requests.get(f"{cls.base_url}/health", timeout=1)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(0.5)
        else:
            cls.server_process.terminate()
            raise RuntimeError("Server failed to start")

    @classmethod
    def tearDownClass(cls):
        """Stop the server."""
        cls.server_process.terminate()
        cls.server_process.wait()

        # Clean up
        if Path("results").exists():
            shutil.rmtree("results")

    def setUp(self):
        """Clean results before each test."""
        if Path("results").exists():
            shutil.rmtree("results")

    def test_01_health_check_actual_http(self):
        """Test health endpoint with actual HTTP request."""
        response = requests.get(f"{self.base_url}/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")

    @patch("webgrid_eval.main.run_agentic_loop")
    def test_02_session_start_actual_http(self, mock_run_loop):
        """Test session start with actual HTTP request."""
        mock_run_loop.return_value = (8, [{"role": "assistant", "content": "test"}])

        response = requests.post(
            f"{self.base_url}/api/session/start",
            json={
                "model": "http-test-model",
                "grid_size": 64,
                "max_seconds": 5,
                "canvas_size": 256,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["model"], "http-test-model")
        self.assertEqual(data["score"], 8)
        self.assertEqual(data["grid_side"], 8)

    @patch("webgrid_eval.main._eval_single_model")
    def test_03_eval_run_actual_http(self, mock_eval):
        """Test eval run with actual HTTP request."""
        from webgrid_eval.main import EvalModelResult

        mock_eval.return_value = EvalModelResult(
            model="http-eval-model",
            score=12,
            incorrect=1,
            ntpm=11.0,
            bps=1.8,
        )

        response = requests.post(
            f"{self.base_url}/api/eval/run",
            json={
                "models": ["http-eval-model"],
                "grid_size": 64,
                "max_seconds": 10,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["model"], "http-eval-model")
        self.assertEqual(data["results"][0]["score"], 12)


class TestResultsDirectoryStructure(unittest.TestCase):
    """Tests for verifying results directory structure."""

    def setUp(self):
        """Clean up before each test."""
        if Path("results").exists():
            shutil.rmtree("results")

    def tearDown(self):
        """Clean up after each test."""
        if Path("results").exists():
            shutil.rmtree("results")

    @patch("webgrid_eval.main.run_agentic_loop")
    def test_model_name_sanitization(self, mock_run_loop):
        """Test that model names with special characters are sanitized."""
        mock_run_loop.return_value = (5, [])

        client = TestClient(app)

        # Test model with slash (like moonshotai/kimi-k2.5)
        client.post(
            "/api/session/start",
            json={
                "model": "moonshotai/kimi-k2.5",
                "grid_size": 64,
                "max_seconds": 5,
            },
        )

        # Directory should use hyphen instead of slash
        model_dir = Path("results/moonshotai-kimi-k2.5")
        self.assertTrue(model_dir.exists())

    @patch("webgrid_eval.main.run_agentic_loop")
    def test_model_name_with_spaces(self, mock_run_loop):
        """Test that model names with spaces are sanitized."""
        mock_run_loop.return_value = (5, [])

        client = TestClient(app)

        client.post(
            "/api/session/start",
            json={
                "model": "gpt 4 turbo",
                "grid_size": 64,
                "max_seconds": 5,
            },
        )

        # Directory should use underscore instead of space
        model_dir = Path("results/gpt_4_turbo")
        self.assertTrue(model_dir.exists())


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)
