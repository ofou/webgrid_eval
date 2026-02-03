"""Tests for tools module.

Tests for tool execution and helper functions.
"""

import json
import math
from unittest.mock import patch

from webgrid_eval import GameState
from webgrid_eval.screenshot import DEFAULT_CANVAS_SIZE
from webgrid_eval.tools import (
    TOOLS_MOUSELIKE,
    TOOLS_OPENAI,
    _clamp,
    _ensure_cursor_pixel,
    _get_max_xy,
    _hud,
    _img_message,
    _packet,
    _require_int,
    _set_cursor_pixel,
    execute_tool,
)


class TestClamp:
    """Test clamp function."""

    def test_clamp_within_range(self):
        """Test value within range is unchanged."""
        assert _clamp(5, 0, 10) == 5
        assert _clamp(0, 0, 10) == 0
        assert _clamp(10, 0, 10) == 10

    def test_clamp_below_range(self):
        """Test value below range is clamped to lower bound."""
        assert _clamp(-5, 0, 10) == 0
        assert _clamp(-100, 0, 10) == 0

    def test_clamp_above_range(self):
        """Test value above range is clamped to upper bound."""
        assert _clamp(15, 0, 10) == 10
        assert _clamp(100, 0, 10) == 10


class TestGetMaxXY:
    """Test getting max pixel coordinate."""

    def test_get_max_xy_default(self):
        """Test with default canvas size."""
        state = GameState(grid_size=64)
        assert _get_max_xy(state) == DEFAULT_CANVAS_SIZE - 1

    def test_get_max_xy_custom(self):
        """Test with custom canvas size."""
        state = GameState(grid_size=64, canvas_size=512)
        assert _get_max_xy(state) == 511


class TestEnsureCursorPixel:
    """Test cursor pixel position initialization."""

    def test_ensure_cursor_pixel_sets_position(self):
        """Test that cursor pixel position is set from cell position."""
        state = GameState(grid_size=64)
        state.cursor_row = 4
        state.cursor_col = 4

        _ensure_cursor_pixel(state)

        # Should set cursor_x and cursor_y based on cell center
        assert state.cursor_x is not None
        assert state.cursor_y is not None

    def test_ensure_cursor_pixel_preserves_existing(self):
        """Test that existing cursor position is preserved."""
        state = GameState(grid_size=64)
        state.cursor_x = 100
        state.cursor_y = 100

        _ensure_cursor_pixel(state)

        # Should preserve existing values
        assert state.cursor_x == 100
        assert state.cursor_y == 100


class TestSetCursorPixel:
    """Test setting cursor pixel position."""

    def test_set_cursor_pixel_updates_position(self):
        """Test that cursor pixel position is updated."""
        state = GameState(grid_size=64)
        state.cursor_row = 0
        state.cursor_col = 0

        _set_cursor_pixel(state, 100, 100)

        assert state.cursor_x == 100
        assert state.cursor_y == 100

    def test_set_cursor_pixel_clamps(self):
        """Test that position is clamped to canvas bounds."""
        state = GameState(grid_size=64, canvas_size=256)

        # Below bounds
        _set_cursor_pixel(state, -100, -100)
        assert state.cursor_x == 0
        assert state.cursor_y == 0

        # Above bounds
        _set_cursor_pixel(state, 1000, 1000)
        assert state.cursor_x == 255
        assert state.cursor_y == 255

    def test_set_cursor_pixel_updates_cell(self):
        """Test that cell position is updated from pixel."""
        state = GameState(grid_size=64)
        state.cursor_row = 0
        state.cursor_col = 0

        # Move to center of cell 4,4 (approximately)
        _set_cursor_pixel(state, 144, 144)

        assert state.cursor_row == 4
        assert state.cursor_col == 4


class TestHUD:
    """Test HUD data generation."""

    def test_hud_basic(self):
        """Test basic HUD generation."""
        state = GameState(grid_size=64)
        state.score = 10
        state.incorrect_count = 2
        state.start_time = 0.0

        with patch("time.time", return_value=60.0):
            hud = _hud(state, last_click=True)

        assert hud["time"] == "01:00"
        assert hud["ntpm"] == 8.0  # 10 - 2
        assert hud["grid"] == "8×8"
        assert hud["last_click_correct"] is True

    def test_hud_bps_calculation(self):
        """Test BPS calculation in HUD."""
        state = GameState(grid_size=64)
        state.score = 10
        state.incorrect_count = 2
        state.start_time = 0.0

        with patch("time.time", return_value=60.0):
            hud = _hud(state)

        net = 8
        expected_bps = (net / 60.0) * math.log2(64**2 - 1)
        assert abs(hud["bps"] - expected_bps) < 0.01

    def test_hud_zero_net(self):
        """Test HUD when net score is zero or negative."""
        state = GameState(grid_size=64)
        state.score = 2
        state.incorrect_count = 2
        state.start_time = 0.0

        with patch("time.time", return_value=60.0):
            hud = _hud(state)

        assert hud["ntpm"] == 0.0
        assert hud["bps"] == 0.0

    def test_hud_no_start_time(self):
        """Test HUD when start_time is not set."""
        state = GameState(grid_size=64)

        hud = _hud(state)

        assert hud["time"] == "00:00"


class TestPacket:
    """Test packet generation."""

    def test_packet_structure(self):
        """Test packet structure and content."""
        state = GameState(grid_size=64, canvas_size=256)
        state.cursor_row = 3
        state.cursor_col = 4
        state.cursor_x = 80
        state.cursor_y = 80
        state.start_time = 0.0
        state.score = 5
        state.incorrect_count = 1

        with patch("time.time", return_value=30.0):
            packet = _packet(state, last_click=True)

        assert "hud" in packet
        assert "cursor" in packet
        assert "target" in packet
        assert "grid_side" in packet
        assert "size_px" in packet

        assert packet["cursor"]["row"] == 3
        assert packet["cursor"]["col"] == 4
        assert packet["cursor"]["x"] == 80
        assert packet["cursor"]["y"] == 80
        assert packet["grid_side"] == 8
        assert packet["size_px"] == 256

    def test_packet_ensures_cursor(self):
        """Test that packet ensures cursor pixel position."""
        state = GameState(grid_size=64)
        state.cursor_row = 0
        state.cursor_col = 0
        # cursor_x and cursor_y not set

        packet = _packet(state)

        # Should set cursor position
        assert packet["cursor"]["x"] is not None
        assert packet["cursor"]["y"] is not None


class TestImgMessage:
    """Test image message generation."""

    def test_img_message_structure(self):
        """Test image message structure."""
        b64 = "test_base64_string"
        msg = _img_message(b64, None)

        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "image_url"
        assert "data:image/png;base64," in msg["content"][0]["image_url"]["url"]

    def test_img_message_with_path(self):
        """Test image message with save path."""
        b64 = "test_base64_string"
        save_path = "/path/to/screenshot.png"
        msg = _img_message(b64, save_path)

        assert "_screenshot_filename" in msg
        assert msg["_screenshot_filename"] == "screenshot.png"


class TestRequireInt:
    """Test integer argument validation."""

    def test_require_int_valid(self):
        """Test with valid integer."""
        val, err = _require_int({"x": 5}, "x")
        assert val == 5
        assert err is None

    def test_require_int_missing(self):
        """Test with missing key."""
        val, err = _require_int({}, "x")
        assert val is None
        assert err is not None
        assert "missing" in err.lower()

    def test_require_int_invalid(self):
        """Test with non-integer value."""
        val, err = _require_int({"x": "abc"}, "x")
        assert val is None
        assert err is not None
        assert "integer" in err.lower()

    def test_require_int_float(self):
        """Test with float value."""
        val, err = _require_int({"x": 5.7}, "x")
        assert val == 5  # Should convert to int
        assert err is None


class TestToolsDefinitions:
    """Test tool definitions."""

    def test_tools_mouselike_structure(self):
        """Test that TOOLS_MOUSELIKE is a list of valid tools."""
        assert isinstance(TOOLS_MOUSELIKE, list)
        assert len(TOOLS_MOUSELIKE) == 3

        tool_names = [t["function"]["name"] for t in TOOLS_MOUSELIKE]
        assert "screen" in tool_names
        assert "mouse_move" in tool_names
        assert "mouse_click" in tool_names

    def test_tools_openai_same_as_mouselike(self):
        """Test that TOOLS_OPENAI is the same as TOOLS_MOUSELIKE."""
        assert TOOLS_OPENAI is TOOLS_MOUSELIKE

    def test_mouse_move_tool_params(self):
        """Test mouse_move tool parameters."""
        mouse_move_tool = next(t for t in TOOLS_MOUSELIKE if t["function"]["name"] == "mouse_move")

        params = mouse_move_tool["function"]["parameters"]
        assert "dx" in params["properties"]
        assert "dy" in params["properties"]
        assert "dx" in params["required"]
        assert "dy" in params["required"]


class TestExecuteTool:
    """Test tool execution."""

    def test_execute_screen_tool(self):
        """Test screen tool execution."""
        state = GameState(grid_size=64, canvas_size=256)
        state.start_time = 0.0
        state.select_random_target()

        result, messages = execute_tool("screen", {}, state)

        assert isinstance(result, str)
        # Result should be JSON
        data = json.loads(result)
        assert "hud" in data
        assert "cursor" in data
        assert data["event"] == "screen"

        # Should return image messages
        assert messages is not None
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_execute_mouse_move(self):
        """Test mouse_move tool execution."""
        state = GameState(grid_size=64, canvas_size=256)
        state.cursor_x = 100
        state.cursor_y = 100

        result, messages = execute_tool("mouse_move", {"dx": 10, "dy": -5}, state)

        # Should update cursor position
        assert state.cursor_x == 110
        assert state.cursor_y == 95

        assert isinstance(result, str)
        data = json.loads(result)
        assert data["event"] == "move"

        # No image messages for move
        assert messages is None

    def test_execute_mouse_move_invalid_args(self):
        """Test mouse_move with invalid arguments."""
        state = GameState(grid_size=64)

        result, messages = execute_tool("mouse_move", {}, state)

        assert "missing" in result.lower()
        assert messages is None

    def test_execute_mouse_click_correct(self):
        """Test mouse_click on correct target."""
        state = GameState(grid_size=64, canvas_size=256)
        state.start_time = 0.0
        state.select_random_target()

        # Move cursor to target
        target_row = state.target_row
        target_col = state.target_col
        from webgrid_eval.screenshot import cell_center_pixel

        state.cursor_x, state.cursor_y = cell_center_pixel(target_row, target_col, 8, 256)

        result, messages = execute_tool("mouse_click", {}, state)

        assert isinstance(result, str)
        data = json.loads(result)
        assert data["event"] == "click"
        assert data["correct"] is True
        assert state.score == 1

        # Should return image messages
        assert messages is not None

    def test_execute_mouse_click_incorrect(self):
        """Test mouse_click on wrong target."""
        state = GameState(grid_size=64, canvas_size=256)
        state.start_time = 0.0
        state.select_random_target()

        # Move cursor to wrong cell
        from webgrid_eval.screenshot import cell_center_pixel

        wrong_row = (state.target_row + 1) % 8
        wrong_col = (state.target_col + 1) % 8
        state.cursor_x, state.cursor_y = cell_center_pixel(wrong_row, wrong_col, 8, 256)

        result, messages = execute_tool("mouse_click", {}, state)

        data = json.loads(result)
        assert data["correct"] is False
        assert state.incorrect_count == 1

    def test_execute_unknown_tool(self):
        """Test execution of unknown tool."""
        state = GameState(grid_size=64)

        result, messages = execute_tool("unknown_tool", {}, state)

        assert "unknown" in result.lower()
        assert messages is None
