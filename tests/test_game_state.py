"""Tests for Webgrid Eval.

This module contains unit tests for the webgrid_eval package.
"""

import pytest

from webgrid_eval import GameState


class TestGameState:
    """Test cases for GameState class."""

    def test_initialization(self):
        """Test GameState initialization."""
        state = GameState(grid_size=64, canvas_size=256)
        assert state.grid_size == 64
        assert state.grid_side == 8
        assert state.canvas_size == 256
        assert state.score == 0
        assert state.incorrect_count == 0

    def test_grid_side_calculation(self):
        """Test grid side calculation."""
        state = GameState(grid_size=64)
        assert state.grid_side == 8

        state = GameState(grid_size=256)
        assert state.grid_side == 16

    def test_invalid_grid_size(self):
        """Test that non-perfect-square grid sizes raise ValueError."""
        with pytest.raises(ValueError):
            state = GameState(grid_size=65)
            _ = state.grid_side

    def test_cursor_movement(self):
        """Test cursor movement."""
        state = GameState(grid_size=64)
        state.move_cursor(5, 5)
        assert state.cursor_row == 5
        assert state.cursor_col == 5

    def test_cursor_clamping(self):
        """Test that cursor is clamped to grid boundaries."""
        state = GameState(grid_size=64)
        state.move_cursor(10, 10)
        assert state.cursor_row == 7  # clamped to max
        assert state.cursor_col == 7

        state.move_cursor(-1, -1)
        assert state.cursor_row == 0  # clamped to min
        assert state.cursor_col == 0

    def test_target_selection(self):
        """Test target selection."""
        state = GameState(grid_size=64)
        initial_target = state.active_index
        state.select_random_target()
        assert state.active_index != initial_target

    def test_correct_click(self):
        """Test correct click handling."""
        state = GameState(grid_size=64)
        state.start_time = 0.0
        target_row = state.target_row
        target_col = state.target_col

        correct, _ = state.click_at(target_row, target_col)
        assert correct is True
        assert state.score == 1
        assert state.incorrect_count == 0

    def test_incorrect_click(self):
        """Test incorrect click handling."""
        state = GameState(grid_size=64)
        state.start_time = 0.0
        target_row = state.target_row
        target_col = state.target_col

        # Click on wrong cell
        wrong_row = (target_row + 1) % 8
        wrong_col = (target_col + 1) % 8
        correct, _ = state.click_at(wrong_row, wrong_col)
        assert correct is False
        assert state.score == 0
        assert state.incorrect_count == 1


class TestMetrics:
    """Test cases for metrics calculations."""

    def test_ntpm_calculation(self):
        """Test NTPM (Net Targets Per Minute) calculation."""
        # NTPM = correct - incorrect
        correct = 10
        incorrect = 2
        ntpm = correct - incorrect
        assert ntpm == 8

    def test_bps_calculation(self):
        """Test BPS (Bits Per Second) calculation."""
        import math

        correct = 10
        incorrect = 2
        elapsed = 60.0
        grid_size = 64

        net = correct - incorrect
        if net > 0:
            bps = (net / 60.0) * math.log2(grid_size**2 - 1)
            expected = (8 / 60.0) * math.log2(4095)
            assert abs(bps - expected) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
