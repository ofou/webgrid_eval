"""Webgrid Eval - Benchmark LLM vision + tool-use capabilities.

This package provides tools for evaluating Large Language Models on Neuralink's
Webgrid cursor control task, testing vision understanding and tool-use capabilities.
"""

__version__ = "0.1.0"
__author__ = "Omar Olivares"
__email__ = "omar@olivares.cl"

from .game_state import GameState
from .openrouter import get_client, run_agentic_loop
from .screenshot import render_grid_screenshot
from .tools import TOOLS_OPENAI, execute_tool

__all__ = [
    "GameState",
    "execute_tool",
    "TOOLS_OPENAI",
    "render_grid_screenshot",
    "run_agentic_loop",
    "get_client",
]
