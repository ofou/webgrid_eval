"""Session game state: grid, target, cursor, score."""

import random
import time
from dataclasses import dataclass, field


@dataclass
class GameState:
    """In-memory game state for one session."""

    grid_size: int
    canvas_size: int = 256  # Screenshot canvas size in pixels (default: 256)
    active_index: int = 0
    cursor_row: int = 0
    cursor_col: int = 0
    cursor_x: int | None = None  # pixel position (synced from cell or moves)
    cursor_y: int | None = None
    score: int = 0  # correct clicks
    incorrect_count: int = 0  # wrong clicks (for NTPM / BPS)
    cursor_speed: float = 1000.0  # pixels per second
    start_time: float | None = field(default=None, repr=False)
    end_time: float | None = field(default=None, repr=False)
    last_click_row: int | None = None
    last_click_col: int | None = None
    last_click_correct: bool | None = None
    last_click_time_ms: int | None = None

    @property
    def grid_side(self) -> int:
        """Return grid side length (e.g. 8 for 8x8)."""
        side = int(self.grid_size**0.5)
        if side * side != self.grid_size:
            raise ValueError(f"grid_size must be a perfect square, got {self.grid_size}")
        return side

    @property
    def target_row(self) -> int:
        """Return target row index."""
        return self.active_index // self.grid_side

    @property
    def target_col(self) -> int:
        """Return target column index."""
        return self.active_index % self.grid_side

    def select_random_target(self) -> None:
        """Pick a new random target different from current."""
        while True:
            random_index = random.randint(0, self.grid_size - 1)
            if random_index != self.active_index:
                self.active_index = random_index
                return

    def move_cursor(self, row: int, col: int) -> None:
        """Set cursor to (row, col) clamped to grid."""
        side = self.grid_side
        self.cursor_row = max(0, min(row, side - 1))
        self.cursor_col = max(0, min(col, side - 1))

    def click_at(self, row: int, col: int) -> tuple[bool, dict]:
        """Click at cell (row, col). Updates cursor and processes click. Returns (correct, data)."""
        self.move_cursor(row, col)
        return self._click_at(row, col)

    def _click_at(self, row: int, col: int) -> tuple[bool, dict]:
        """Check if (row, col) is the target; advance and return (True, data) or (False, {})."""
        side = self.grid_side
        self.last_click_row = row
        self.last_click_col = col
        self.last_click_time_ms = (
            int((time.time() - self.start_time) * 1000) if self.start_time is not None else None
        )
        if row < 0 or row >= side or col < 0 or col >= side:
            self.incorrect_count += 1
            self.last_click_correct = False
            return False, {}
        index = row * side + col
        if index != self.active_index:
            self.incorrect_count += 1
            self.last_click_correct = False
            return False, {}
        self.score += 1
        self.last_click_correct = True
        self.select_random_target()
        return True, {
            "row": self.target_row,
            "col": self.target_col,
            "index": self.active_index,
        }
