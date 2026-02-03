"""OpenAI Chat Completions API client; agentic loop with tool_calls."""

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
    """Return OpenAI client. Use base_url/api_key from YAML or env."""
    if base_url:
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        if api_key is not None:
            key = api_key
        elif "openrouter" in base_url.lower():
            key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
            if not key:
                raise ValueError("OPENROUTER_API_KEY or OPENAI_API_KEY required for OpenRouter.")
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

The goal is to click targets on a grid as fast as possible while minimizing misclicks.
Your score (BPS) is derived from net correct targets per minute (NTPM) and grid size.

Our eighth clinical trial participant achieved 10.39 BPS controlling his computer with his brain.

Do not repeat the same coordinates after an incorrect click—the target moves.
Find the new blue cell each time.

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
    reasoning_effort: str | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Run the agentic loop using OpenAI Chat Completions API.

    Session starts with an initial grid screenshot so the model sees the grid first.
    If save_dir is set, every screenshot (initial + each click) is saved there as <unixtime_ms>.png.
    If max_seconds is set, stop after that many seconds per model.
    If reasoning_effort is set (minimal, low, medium, high), pass to Gemini 3 models.
    Returns (final_score, messages).
    """
    from .screenshot import render_grid_screenshot

    client = get_client(base_url=base_url, api_key=api_key)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Track target positions for GIF maker (saved to targets.json, not sent to model)
    target_events: list[dict[str, Any]] = []

    def _get_relative_path(save_dir: str, start_time: float) -> str:
        return os.path.join(save_dir, f"{int((time.time() - start_time) * 1000):05d}ms.png")

    initial_save = (
        _get_relative_path(save_dir, state.start_time) if save_dir and state.start_time else None
    )
    b64 = render_grid_screenshot(state, save_path=initial_save)

    # Record initial target position
    target_events.append(
        {
            "type": "start",
            "ts_ms": 0,
            "target_row": getattr(state, "target_row", -1),
            "target_col": getattr(state, "target_col", -1),
            "grid_side": state.grid_side,
        }
    )

    msg = {
        "role": "user",
        "content": [
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
        elapsed = time.time() - wall_clock_start
        if max_seconds is not None:
            if elapsed >= max_seconds:
                break
            # Dynamic timeout: remaining time + small buffer, capped at 60s
            remaining = max_seconds - elapsed
            api_timeout = min(remaining + 5.0, 60.0)
        else:
            api_timeout = 60.0

        # Sanitize messages: remove internal keys (e.g. _screenshot_filename)
        sanitized_messages = [
            {k: v for k, v in m.items() if not k.startswith("_")} for m in messages
        ]
        if max_images is not None:
            sanitized_messages = _truncate_messages_to_max_images(sanitized_messages, max_images)

        # Call Chat Completions API with tools (retry on rate limits / errors)
        resp = None
        max_retries = 6
        base_delay = 5.0

        for attempt in range(max_retries):
            try:
                # Build API call kwargs
                api_kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": sanitized_messages,
                    "tools": TOOLS_OPENAI,
                    "timeout": api_timeout,
                }
                # Add reasoning_effort for Gemini 3 thinking control
                # Maps to Gemini's thinking_level: minimal, low, medium, high
                extra_body: dict[str, Any] = {}
                if reasoning_effort:
                    extra_body["reasoning_effort"] = reasoning_effort
                # For Google's direct API, request actual thinking content (not just signatures)
                # This returns readable thought summaries via include_thoughts
                # Note: Google's OpenAI-compat API requires nested extra_body structure
                if "generativelanguage.googleapis.com" in (base_url or ""):
                    extra_body["extra_body"] = {
                        "google": {"thinking_config": {"include_thoughts": True}}
                    }
                if extra_body:
                    api_kwargs["extra_body"] = extra_body

                resp = client.chat.completions.create(**api_kwargs)
                break
            except (OpenAIRateLimitError, APIConnectionError, APITimeoutError) as e:
                # Exponential backoff with jitter: delay = base * 2^attempt + random(0-2s)
                delay = base_delay * (2**attempt) + (hash(str(time.time())) % 200) / 100
                if attempt < max_retries - 1:
                    print(
                        f"  ⏳ {model}: {type(e).__name__}, retrying in {delay:.1f}s... "
                        f"(attempt {attempt + 1}/{max_retries})"
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
                        f"  ⏳ {model}: API error {error_code}, retrying in {delay:.1f}s... "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                else:
                    raise
        if not resp or not resp.choices or len(resp.choices) == 0:
            raise ValueError(
                f"Model {model} returned no choices (empty or filtered response). "
                "Try another model or check provider status."
            )

        # Check time again after API call (in case it took a long time)
        if max_seconds is not None and time.time() - wall_clock_start >= max_seconds:
            break

        msg = resp.choices[0].message

        # Extract thinking content from Google's API response (when include_thoughts is enabled)
        # The thinking text may come in extra_content.google.thinking or similar fields
        thinking_content = None
        msg_extra_content = getattr(msg, "extra_content", None)
        if msg_extra_content and isinstance(msg_extra_content, dict):
            google_extra = msg_extra_content.get("google", {})
            thinking_content = google_extra.get("thinking") or google_extra.get("thought_summary")

        # Debug: print raw response structure to identify thinking content location
        if os.environ.get("DEBUG_THINKING"):
            import json

            print(f"DEBUG msg.content type: {type(msg.content)}")
            is_str = isinstance(msg.content, str) and msg.content
            content_preview = msg.content if is_str else msg.content
            print(f"DEBUG msg.content: {content_preview}")
            print(f"DEBUG msg_extra_content: {msg_extra_content}")
            if hasattr(resp.choices[0], "extra_content"):
                print(f"DEBUG choice.extra_content: {resp.choices[0].extra_content}")

        # Extract tool_calls from API response, preserving thought_signature for Gemini 3
        tool_calls_raw = getattr(msg, "tool_calls", None) or []
        tool_calls_list: list[dict[str, Any]] = []
        for tc in tool_calls_raw:
            tid = tc.id if hasattr(tc, "id") else tc.get("id")
            fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
            name = fn.name if hasattr(fn, "name") else fn.get("name", "")
            args_str = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", "{}")

            tool_call_entry: dict[str, Any] = {
                "id": tid,
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            }

            # Preserve extra_content (including thought_signature and thinking) for Gemini 3
            extra_content = None
            if hasattr(tc, "extra_content"):
                extra_content = tc.extra_content
            elif isinstance(tc, dict) and "extra_content" in tc:
                extra_content = tc["extra_content"]

            # Check for thinking content in tool call extra_content
            if extra_content and isinstance(extra_content, dict):
                google_tc = extra_content.get("google", {})
                tc_thinking = google_tc.get("thinking") or google_tc.get("thought_summary")
                if tc_thinking and not thinking_content:
                    thinking_content = tc_thinking

            # Note: extra_content (thought_signature) is intentionally NOT saved to history
            # to avoid bloating the JSON with large opaque signatures

            tool_calls_list.append(tool_call_entry)

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or None,
            "tool_calls": tool_calls_list,
        }
        # Include thinking content in the message if available
        if thinking_content:
            assistant_msg["thinking"] = thinking_content
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
                        "text": (
                            "Continue playing. Use screen() to see the grid, "
                            "then mouse_move and mouse_click."
                        ),
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

            # Track target position after each click (for GIF maker)
            if name == "mouse_click":
                ts_ms = int((time.time() - state.start_time) * 1000) if state.start_time else 0
                target_events.append(
                    {
                        "type": "click",
                        "ts_ms": ts_ms,
                        "target_row": getattr(state, "target_row", -1),
                        "target_col": getattr(state, "target_col", -1),
                    }
                )

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
                                            f"ATTENTION: You've had {consecutive_errors} "
                                            "consecutive incorrect clicks.\n\n"
                                            "DEBUGGING STRATEGY:\n"
                                            "1. Carefully examine the current image\n"
                                            "2. Locate the BLUE cell (ignore RED)\n"
                                            "3. Use mouse_move(dx, dy) toward the blue cell\n"
                                            "4. When cursor over blue cell, call mouse_click()\n"
                                            "5. Double-check: gray cell shows cursor position\n\n"
                                            "TIP: After wrong click find NEW blue cell!"
                                        ),
                                    }
                                ],
                            }
                            messages.append(hint_msg)
                except (json.JSONDecodeError, ValueError):
                    pass

    # Update state end_time
    state.end_time = time.time()

    # Save target events for GIF maker (not sent to model)
    if save_dir and target_events:
        targets_path = os.path.join(save_dir, "targets.json")
        with open(targets_path, "w") as f:
            json.dump(target_events, f, indent=2)

    return state.score, messages
