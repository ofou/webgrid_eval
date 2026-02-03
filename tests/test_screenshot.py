"""Tests for screenshot module.

Tests for coordinate conversion and grid rendering functions.
"""

import base64
from io import BytesIO

from PIL import Image

from webgrid_eval import GameState
from webgrid_eval.screenshot import (
    DEFAULT_CANVAS_SIZE,
    K,
    _get_cell_rect,
    _line_width,
    _load_cursor,
    cell_center_pixel,
    normalized_to_cell,
    pixel_to_cell,
    render_grid_screenshot,
)


class TestCoordinateConversion:
    """Test pixel to cell coordinate conversions."""

    def test_pixel_to_cell_basic(self):
        """Test basic pixel to cell conversion."""
        # 8x8 grid on 256px canvas = 32px per cell
        row, col = pixel_to_cell(0, 0, 8, 256)
        assert row == 0
        assert col == 0

    def test_pixel_to_cell_mid_cell(self):
        """Test pixel in middle of first cell."""
        row, col = pixel_to_cell(16, 16, 8, 256)
        assert row == 0
        assert col == 0

    def test_pixel_to_cell_next_cell(self):
        """Test pixel in second cell."""
        row, col = pixel_to_cell(32, 32, 8, 256)
        assert row == 1
        assert col == 1

    def test_pixel_to_cell_edge(self):
        """Test pixel at cell boundary."""
        row, col = pixel_to_cell(31, 31, 8, 256)
        assert row == 0
        assert col == 0

        row, col = pixel_to_cell(32, 32, 8, 256)
        assert row == 1
        assert col == 1

    def test_pixel_to_cell_clamping(self):
        """Test that out-of-bounds pixels are clamped."""
        # Negative pixels
        row, col = pixel_to_cell(-1, -1, 8, 256)
        assert row == 0
        assert col == 0

        # Pixels beyond canvas
        row, col = pixel_to_cell(300, 300, 8, 256)
        assert row == 7
        assert col == 7

    def test_pixel_to_cell_different_sizes(self):
        """Test with different grid sizes."""
        # 16x16 grid on 256px = 16px per cell
        row, col = pixel_to_cell(16, 16, 16, 256)
        assert row == 1
        assert col == 1

        # 4x4 grid on 256px = 64px per cell
        row, col = pixel_to_cell(64, 64, 4, 256)
        assert row == 1
        assert col == 1


class TestCellCenterPixel:
    """Test cell to pixel center conversion."""

    def test_cell_center_basic(self):
        """Test center of first cell."""
        x, y = cell_center_pixel(0, 0, 8, 256)
        # Cell 0,0 is from (0,0) to (32,32), center at (16,16)
        assert x == 16
        assert y == 16

    def test_cell_center_second_cell(self):
        """Test center of second cell."""
        x, y = cell_center_pixel(1, 1, 8, 256)
        # Cell 1,1 is from (32,32) to (64,64), center at (48,48)
        assert x == 48
        assert y == 48

    def test_cell_center_last_cell(self):
        """Test center of last cell."""
        x, y = cell_center_pixel(7, 7, 8, 256)
        # Cell 7,7 is from (224,224) to (256,256), center at (240,240)
        assert x == 240
        assert y == 240

    def test_cell_center_clamping(self):
        """Test that center coordinates are clamped to canvas."""
        # Cell beyond grid should still return valid pixel
        x, y = cell_center_pixel(10, 10, 8, 256)
        assert x <= 255
        assert y <= 255


class TestNormalizedToCell:
    """Test normalized coordinate conversion."""

    def test_normalized_to_cell_corners(self):
        """Test conversion from normalized coordinates."""
        # Top-left corner
        row, col = normalized_to_cell(0.0, 0.0, 8, 256)
        assert row == 0
        assert col == 0

        # Bottom-right corner
        row, col = normalized_to_cell(1.0, 1.0, 8, 256)
        assert row == 7
        assert col == 7

    def test_normalized_to_cell_center(self):
        """Test center of canvas."""
        # 0.5 normalized on 256px = pixel 127.5
        # 127.5 / 32 (cell size) = 3.98, floor = 3
        row, col = normalized_to_cell(0.5, 0.5, 8, 256)
        assert row == 3
        assert col == 3

    def test_normalized_out_of_bounds(self):
        """Test out-of-bounds normalized coordinates."""
        # Negative
        row, col = normalized_to_cell(-0.1, -0.1, 8, 256)
        assert row == 0
        assert col == 0

        # Greater than 1
        row, col = normalized_to_cell(1.5, 1.5, 8, 256)
        assert row == 0  # Should use fallback
        assert col == 0


class TestLineWidth:
    """Test line width calculations."""

    def test_line_width_default(self):
        """Test default line width calculation."""
        width = _line_width(256)
        # r = width * K = 256 * 0.002 = 0.512, rounded to 1
        assert width == 1

    def test_line_width_larger_canvas(self):
        """Test line width with larger canvas."""
        width = _line_width(512)
        # 512 * 0.002 = 1.024, rounded to 1
        assert width == 1

    def test_line_width_minimum(self):
        """Test that minimum line width is 1."""
        width = _line_width(100)
        assert width >= 1


class TestCellRect:
    """Test cell rectangle calculations."""

    def test_cell_rect_basic(self):
        """Test first cell rectangle."""
        x0, y0, x1, y1 = _get_cell_rect(0, 0, 8, 256)
        # With line width 1, inset is 0.5
        assert x0 < x1
        assert y0 < y1
        assert x0 >= 0
        assert y0 >= 0

    def test_cell_rect_dimensions(self):
        """Test that cell dimensions are approximately correct."""
        x0, y0, x1, y1 = _get_cell_rect(0, 0, 8, 256)
        # Cell should be roughly 32px minus line width
        width = x1 - x0
        height = y1 - y0
        assert 30 <= width <= 32
        assert 30 <= height <= 32


class TestCursorLoading:
    """Test cursor asset loading."""

    def test_load_cursor(self):
        """Test that cursor loads successfully."""
        cursor, hx, hy = _load_cursor()
        assert cursor is not None
        assert isinstance(cursor, Image.Image)
        assert hx > 0
        assert hy > 0

    def test_cursor_caching(self):
        """Test that cursor is cached after first load."""
        cursor1, hx1, hy1 = _load_cursor()
        cursor2, hx2, hy2 = _load_cursor()
        # Should return same cached object
        assert cursor1 is cursor2
        assert hx1 == hx2
        assert hy1 == hy2


class TestRenderGridScreenshot:
    """Test grid screenshot rendering."""

    def test_render_returns_base64(self):
        """Test that rendering returns valid base64 string."""
        state = GameState(grid_size=64, canvas_size=256)
        state.start_time = 0.0
        state.select_random_target()

        b64 = render_grid_screenshot(state)

        # Should return base64 string
        assert isinstance(b64, str)
        # Should be decodable
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_render_creates_valid_image(self):
        """Test that rendered image is valid PNG."""
        state = GameState(grid_size=64, canvas_size=256)
        state.start_time = 0.0
        state.select_random_target()

        b64 = render_grid_screenshot(state)
        decoded = base64.b64decode(b64)

        # Should be valid PNG
        img = Image.open(BytesIO(decoded))
        assert img.format == "PNG"
        assert img.size == (256, 256)

    def test_render_with_save_path(self, tmp_path):
        """Test that rendering saves to file when path provided."""
        state = GameState(grid_size=64, canvas_size=256)
        state.start_time = 0.0
        state.select_random_target()

        save_path = tmp_path / "test.png"
        b64 = render_grid_screenshot(state, save_path=str(save_path))

        # File should exist
        assert save_path.exists()
        # Should still return base64
        assert isinstance(b64, str)

    def test_render_different_canvas_sizes(self):
        """Test rendering with different canvas sizes."""
        for size in [128, 256, 512]:
            state = GameState(grid_size=64, canvas_size=size)
            state.start_time = 0.0
            state.select_random_target()

            b64 = render_grid_screenshot(state)
            decoded = base64.b64decode(b64)
            img = Image.open(BytesIO(decoded))

            assert img.size == (size, size), f"Failed for canvas size {size}"

    def test_render_with_incorrect_click(self):
        """Test rendering with last click incorrect flag."""
        state = GameState(grid_size=64, canvas_size=256)
        state.start_time = 0.0
        state.select_random_target()
        state.last_click_row = 0
        state.last_click_col = 0
        state.last_click_correct = False

        b64 = render_grid_screenshot(state, last_click_incorrect=True)
        assert isinstance(b64, str)


class TestConstants:
    """Test module constants."""

    def test_default_canvas_size(self):
        """Test default canvas size constant."""
        assert DEFAULT_CANVAS_SIZE == 256

    def test_k_constant(self):
        """Test line width constant."""
        assert K == 0.002
        assert isinstance(K, float)
