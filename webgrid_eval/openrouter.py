"""OpenAI Chat Completions API client; agentic loop with tool_calls."""

import json
import os
import time
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI
from openai import RateLimitError as OpenAIRateLimitError

from .game_state import GameState
from .tools import TOOLS_OPENAI, execute_tool

DEFAULT_MODEL = "gpt-5.2"
MAX_TOOL_ROUNDS = 1000


def get_client(
    base_url: str | None = None,
    api_key: str | None = None,
) -> OpenAI:
    """Returns OpenAI client. Use base_url/api_key from YAML or env: WEBGRID_API_BASE_URL (local), OPENROUTER_API_KEY/OPENAI_API_KEY (OpenRouter)."""
    if base_url:
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        if api_key is not None:
            key = api_key
        elif "openrouter" in base_url.lower():
            key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
            if not key:
                raise ValueError(
                    "OPENROUTER_API_KEY or OPENAI_API_KEY environment variable required when using OpenRouter (base_url from YAML)."
                )
        else:
            key = "lm-studio"
        return OpenAI(base_url=base_url, api_key=key)
    env_base = os.environ.get("WEBGRID_API_BASE_URL")
    if env_base:
        env_base = env_base.rstrip("/")
        if not env_base.endswith("/v1"):
            env_base += "/v1"
        return OpenAI(base_url=env_base, api_key="lm-studio")
    key = (
        api_key
        if api_key is not None
        else (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


# Official Neuralink Webgrid instructions (https://neuralink.com/webgrid)
SYSTEM_PROMPT = """Play Webgrid

At Neuralink, we use a game called Webgrid to test how precisely you can control a computer.

The goal of the game is to click targets on a grid as fast as possible while minimizing misclicks. Your score, measured in bits per second (BPS), is derived from net correct targets selected per minute (NTPM) and grid size.

Our eighth clinical trial participant achieved a score of 10.39 BPS controlling his computer with his brain.

Do not repeat the same coordinates after an incorrect click—the target moves. Find the new blue cell each time.

How well can you do?"""


def _truncate_messages_to_max_images(
    messages: list[dict[str, Any]], max_images: int
) -> list[dict[str, Any]]:
    """Keep at most max_images. Prioritizes error screenshots for better learning.

    Preserves (user, assistant, tool) structure. For max_images=1, keeps only the
    latest image plus all assistant/tool text. When max_images > 1, prioritizes
    rounds with incorrect clicks to help the model learn from mistakes.
    """
    total = sum(
        1
        for m in messages
        for part in (m.get("content") or [])
        if isinstance(part, dict) and part.get("type") == "image_url"
    )
    if total <= max_images:
        return messages

    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]

    # Collect rounds: each round = user (with img) + assistant + tool(s)
    rounds: list[list[dict]] = []
    i = 0
    while i < len(rest):
        m = rest[i]
        if m.get("role") == "user":
            content = m.get("content") or []
            has_img = any(
                isinstance(p, dict) and p.get("type") == "image_url"
                for p in (content if isinstance(content, list) else [])
            )
            if has_img:
                round_msgs = [m]
                i += 1
                while i < len(rest) and rest[i].get("role") in ("assistant", "tool"):
                    round_msgs.append(rest[i])
                    i += 1
                rounds.append(round_msgs)
                continue
        i += 1

    if max_images == 1:
        # Only the latest image: instructions + latest img in one user msg, all assistant/tool
        last_round = rounds[-1] if rounds else []
        first_user = rounds[0][0] if rounds else None
        last_user = last_round[0] if last_round else None
        if not first_user or not last_user:
            return system + rest
        # Get instruction text from first user, latest image from last user
        first_content = first_user.get("content") or []
        last_content = last_user.get("content") or []
        if not isinstance(first_content, list):
            first_content = [{"type": "text", "text": str(first_content)}]
        if not isinstance(last_content, list):
            last_content = []
        instruction_parts = [
            p for p in first_content if isinstance(p, dict) and p.get("type") == "text"
        ]
        latest_img = next(
            (p for p in last_content if isinstance(p, dict) and p.get("type") == "image_url"),
            None,
        )
        new_content = instruction_parts + ([latest_img] if latest_img else [])
        combined_user = {"role": "user", "content": new_content}
        if last_user.get("_screenshot_filename"):
            combined_user["_screenshot_filename"] = last_user["_screenshot_filename"]
        # All assistant + tool from all rounds
        asst_tool = [msg for r in rounds for msg in r[1:] if r]
        return system + [combined_user] + asst_tool

    # Smart prioritization: keep initial + prioritize error rounds + recent successful rounds
    if len(rounds) <= max_images:
        keep = rounds
    else:
        # Separate error rounds from successful rounds
        error_rounds = []
        normal_rounds = []

        for idx, round_msgs in enumerate(rounds):
            user_msg = round_msgs[0]
            content = user_msg.get("content", [])
            has_error = any(
                "Wrong click" in str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
            if has_error:
                error_rounds.append((idx, round_msgs))
            else:
                normal_rounds.append((idx, round_msgs))

        # Always keep the first round (initial instructions)
        keep = [rounds[0]]
        remaining = max_images - 1

        # Prioritize recent errors (up to half the available slots)
        error_slots = min(len(error_rounds), max(1, remaining // 2))
        if error_rounds:
            # Add recent error rounds, ensuring no duplicates if first round was an error
            added_error_rounds = []
            for _, r in error_rounds:
                if r is not rounds[0]:  # Avoid adding first round again if it was an error
                    added_error_rounds.append(r)
            keep.extend(added_error_rounds[-error_slots:])
            remaining -= len(added_error_rounds[-error_slots:])  # Subtract actual added count

        # Fill remaining with recent successful rounds
        if remaining > 0 and normal_rounds:
            # Exclude first round if already included (which it always is)
            recent_normal = [r for idx, r in normal_rounds if idx > 0]
            keep.extend(recent_normal[-remaining:])

        # Sort by original order to maintain conversation flow
        # Use a temporary list of (original_index, round_msgs) to sort
        temp_keep_with_indices = []
        for r_idx, original_round in enumerate(rounds):
            if original_round in keep:
                temp_keep_with_indices.append((r_idx, original_round))

        temp_keep_with_indices.sort(key=lambda x: x[0])
        keep = [r for _, r in temp_keep_with_indices]

    flat = [msg for r in keep for msg in r]
    return system + flat


def run_agentic_loop(
    state: GameState,
    model: str = DEFAULT_MODEL,
    extra_headers: dict[str, str] | None = None,
    save_dir: str | None = None,
    max_seconds: float | None = None,
    max_images: int | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Run the agentic loop using OpenAI Chat Completions API.
    Session starts with an initial grid screenshot so the model sees the grid first.
    If save_dir is set, every screenshot (initial + each click) is saved there as <unixtime_ms>.png.
    If max_seconds is set, stop after that many seconds per model.
    Returns (final_score, messages).
    """
    from .screenshot import render_grid_screenshot

    client = get_client(base_url=base_url, api_key=api_key)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    def _get_relative_path(save_dir: str, start_time: float) -> str:
        return os.path.join(save_dir, f"{int((time.time() - start_time) * 1000):05d}ms.png")

    initial_save = (
        _get_relative_path(save_dir, state.start_time) if save_dir and state.start_time else None
    )
    b64 = render_grid_screenshot(state, save_path=initial_save)

    msg = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "You cannot click at coordinates—only relative moves. Use screen() to see the current state. "
                    "Move the cursor with mouse_move(dx, dy) (positive dx = right, dy = down). "
                    "Click with mouse_click() at the current cursor position. Hit the blue target; wrong clicks move the target. Prefer accuracy over speed."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
        ],
    }
    if initial_save:
        msg["_screenshot_filename"] = os.path.basename(initial_save)
    messages.append(msg)

    wall_clock_start = time.time()

    for _ in range(MAX_TOOL_ROUNDS):
        if max_seconds is not None:
            if time.time() - wall_clock_start >= max_seconds:
                break

        # Sanitize messages: remove internal keys (e.g. _screenshot_filename)
        sanitized_messages = [
            {k: v for k, v in m.items() if not k.startswith("_")} for m in messages
        ]
        if max_images is not None:
            sanitized_messages = _truncate_messages_to_max_images(sanitized_messages, max_images)

        # Call Chat Completions API with tools (retry on rate limits, errors, with exponential backoff)
        resp = None
        max_retries = 6
        base_delay = 5.0

        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=sanitized_messages,
                    tools=TOOLS_OPENAI,
                    timeout=180.0,
                )
                break
            except (OpenAIRateLimitError, APIConnectionError, APITimeoutError) as e:
                # Exponential backoff with jitter: delay = base * 2^attempt + random(0-2s)
                delay = base_delay * (2**attempt) + (hash(str(time.time())) % 200) / 100
                if attempt < max_retries - 1:
                    print(
                        f"  ⏳ {model}: {type(e).__name__}, retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                else:
                    raise
            except APIError as e:
                # Retry on any API error (402 payment required, 429 rate limit, 5xx server errors)
                delay = base_delay * (2**attempt) + (hash(str(time.time())) % 200) / 100
                if attempt < max_retries - 1:
                    error_code = getattr(e, "code", None) or getattr(e, "status_code", "unknown")
                    print(
                        f"  ⏳ {model}: API error {error_code}, retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                else:
                    raise
        if not resp or not resp.choices or len(resp.choices) == 0:
            raise ValueError(
                f"Model {model} returned no choices (empty or filtered response). "
                "Try another model or check provider status."
            )
        msg = resp.choices[0].message

        # Extract tool_calls from API response
        tool_calls_raw = getattr(msg, "tool_calls", None) or []
        tool_calls_list: list[dict[str, Any]] = []
        for tc in tool_calls_raw:
            tid = tc.id if hasattr(tc, "id") else tc.get("id")
            fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
            name = fn.name if hasattr(fn, "name") else fn.get("name", "")
            args_str = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", "{}")
            tool_calls_list.append(
                {
                    "id": tid,
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                }
            )

        assistant_msg = {
            "role": "assistant",
            "content": msg.content or None,
            "tool_calls": tool_calls_list,
        }
        messages.append(assistant_msg)

        if not tool_calls_list:
            # Model returned no tool calls - prompt it to continue playing
            continue_save_path = (
                _get_relative_path(save_dir, state.start_time)
                if save_dir and state.start_time
                else None
            )
            b64 = render_grid_screenshot(state, save_path=continue_save_path)
            prompt_msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Continue playing. Use screen() to see the grid, then mouse_move and mouse_click.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
            if continue_save_path:
                prompt_msg["_screenshot_filename"] = os.path.basename(continue_save_path)
            messages.append(prompt_msg)
            continue

        for tc in tool_calls_list:
            tid = tc["id"]
            name = tc["function"]["name"].strip()
            args_str = tc["function"].get("arguments", "{}")
            try:
                arguments = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
            except json.JSONDecodeError:
                arguments = {}
            screen_save_path = (
                _get_relative_path(save_dir, state.start_time)
                if save_dir and name == "screen" and state.start_time
                else None
            )
            click_save_path = (
                _get_relative_path(save_dir, state.start_time)
                if save_dir and name == "mouse_click" and state.start_time
                else None
            )
            content, extra_messages = execute_tool(
                name,
                arguments,
                state,
                screen_save_path=screen_save_path,
                click_save_path=click_save_path,
            )
            messages.append({"role": "tool", "tool_call_id": tid, "content": content})
            if extra_messages:
                for m in extra_messages:
                    messages.append(m)

            # Adaptive hints: detect consecutive errors and provide stronger guidance
            if name == "mouse_click":
                try:
                    result = (
                        json.loads(content)
                        if isinstance(content, str) and content.startswith("{")
                        else {"correct": False}
                    )
                    is_correct = result.get("correct", False)

                    if not is_correct:
                        # Count consecutive errors
                        consecutive_errors = 0
                        for msg in reversed(messages):
                            if msg.get("role") == "tool" and "correct" in msg.get("content", ""):
                                if '"correct": false' in msg.get("content", ""):
                                    consecutive_errors += 1
                                else:
                                    break

                        # Provide adaptive hint after 3 consecutive errors
                        if consecutive_errors >= 3:
                            hint_msg = {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            f"ATTENTION: You've had {consecutive_errors} consecutive incorrect clicks.\n\n"
                                            f"DEBUGGING STRATEGY:\n"
                                            f"1. Carefully examine the current image\n"
                                            f"2. Locate the BLUE cell (ignore any RED cells from previous errors)\n"
                                            f"3. Use mouse_move(dx, dy) to move the cursor toward the blue cell\n"
                                            f"4. When the cursor is over the blue cell, call mouse_click()\n"
                                            f"5. Double-check: the gray cell shows where your cursor is\n\n"
                                            f"TIP: The blue target moves after each wrong click. Always look for the NEW blue cell!"
                                        ),
                                    }
                                ],
                            }
                            messages.append(hint_msg)
                except (json.JSONDecodeError, ValueError):
                    pass

    # Update state end_time
    state.end_time = time.time()

    return state.score, messages
