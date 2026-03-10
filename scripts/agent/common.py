"""Shared utilities for AI review-cycle scripts.

Contains the CLI-based reviewer abstraction, preset definitions,
git helpers, comment persistence, and common argument parsing used
by both design_doc_review_cycle.py and plan_doc_review_cycle.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


# ---------------------------------------------------------------------------
# Reviewer abstraction
# ---------------------------------------------------------------------------


@dataclass
class Reviewer:
    """A CLI-based AI reviewer.

    If *fallback* is set and the primary command fails (non-zero exit
    code or timeout), the fallback reviewer is tried automatically.

    If *rate_limit_fallback* is set and a rate-limit error is detected
    in the command output, this cross-provider fallback is used instead
    of the regular *fallback*.
    """

    name: str
    cmd: str  # shell command template (prompt is appended as last arg)
    activity_probe: str = "generic"
    fallback: Reviewer | None = None
    rate_limit_fallback: Reviewer | None = None

    def run(
        self,
        prompt: str,
        timeout: int | None = None,
        stall_timeout: int | None = None,
        heartbeat_secs: int | None = 30,
    ) -> str:
        """Invoke the CLI tool, falling back to an alternative on failure.

        If a rate-limit error is detected in the output,
        ``rate_limit_fallback`` (or ``fallback``) is tried automatically.

        Otherwise, if the primary command fails (non-zero exit code or
        timeout) and a ``fallback`` reviewer is configured, the fallback
        is tried.  If the fallback also fails the error propagates
        normally.
        """
        try:
            raw_output, stderr_output, returncode = self._execute(
                prompt,
                timeout=timeout,
                stall_timeout=stall_timeout,
                heartbeat_secs=heartbeat_secs,
            )
        except subprocess.TimeoutExpired as exc:
            if self.fallback:
                timeout_value = (
                    exc.timeout
                    if exc.timeout is not None
                    else timeout
                )
                timeout_desc = (
                    f"{timeout_value}s"
                    if timeout_value is not None
                    else "the configured limit"
                )
                print(
                    f"\n[FALLBACK] {self.name} timed out after {timeout_desc}, "
                    f"falling back to {self.fallback.name}",
                    file=sys.stderr,
                    flush=True,
                )
                return self.fallback.run(
                    prompt,
                    timeout=timeout,
                    stall_timeout=stall_timeout,
                    heartbeat_secs=heartbeat_secs,
                )
            raise

        # Rate-limit check takes priority over generic non-zero exit.
        if _is_rate_limit_error(raw_output, stderr_output):
            fb = self.rate_limit_fallback or self.fallback
            if fb:
                print(
                    f"\n[RATE LIMIT] {self.name} hit a rate limit, "
                    f"falling back to {fb.name}",
                    file=sys.stderr,
                    flush=True,
                )
                return fb.run(
                    prompt,
                    timeout=timeout,
                    stall_timeout=stall_timeout,
                    heartbeat_secs=heartbeat_secs,
                )
            print(
                f"\n[RATE LIMIT] {self.name} hit a rate limit "
                "and no fallback is available",
                file=sys.stderr,
                flush=True,
            )

        if returncode != 0:
            print(
                f"[WARNING] {self.name} exited with code {returncode}",
                file=sys.stderr,
                flush=True,
            )
            if self.fallback:
                print(
                    f"[FALLBACK] Falling back to {self.fallback.name}",
                    file=sys.stderr,
                    flush=True,
                )
                return self.fallback.run(
                    prompt,
                    timeout=timeout,
                    stall_timeout=stall_timeout,
                    heartbeat_secs=heartbeat_secs,
                )

        return _extract_reviewer_output(self.activity_probe, raw_output)

    def _execute(
        self,
        prompt: str,
        timeout: int | None = None,
        stall_timeout: int | None = None,
        heartbeat_secs: int | None = 30,
    ) -> tuple[str, str, int]:
        """Run the subprocess and return ``(stdout, stderr, returncode)``.

        The prompt is written to a temporary file and that file is used
        as stdin to avoid hitting OS ARG_MAX limits and to prevent
        blocking on a large pipe write before timeout polling starts.

        Output is redirected to temporary log files to avoid
        pipe-buffering hangs.  File paths are printed to stderr so
        the user can ``tail -f`` them for real-time monitoring.

        Raises ``subprocess.TimeoutExpired`` if *timeout* or
        *stall_timeout* is exceeded.
        """
        tmpdir = tempfile.mkdtemp(
            prefix=f"reviewer_{sanitize_name(self.name)}_",
        )
        prompt_path = os.path.join(tmpdir, "prompt.txt")
        stdout_path = os.path.join(tmpdir, "stdout.log")
        stderr_path = os.path.join(tmpdir, "stderr.log")
        Path(prompt_path).write_text(prompt, encoding="utf-8")

        prompt_f = open(prompt_path, "rb")
        stdout_f = open(stdout_path, "wb")
        stderr_f = open(stderr_path, "wb")
        timed_out = False

        try:
            proc = subprocess.Popen(
                shlex.split(self.cmd),
                stdin=prompt_f,
                stdout=stdout_f,
                stderr=stderr_f,
            )

            print(
                f"[{self.name}] stdout -> {stdout_path}",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"[{self.name}] stderr -> {stderr_path}",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"[{self.name}] pid={proc.pid}",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"[{self.name}] activity_probe={self.activity_probe}",
                file=sys.stderr,
                flush=True,
            )

            # Poll until exit, enforcing timeout and surfacing liveness.
            deadline = (time.monotonic() + timeout) if timeout else None
            monitor_start = time.monotonic()
            last_activity = monitor_start
            last_output = monitor_start
            next_heartbeat = (
                monitor_start + heartbeat_secs
                if heartbeat_secs and heartbeat_secs > 0
                else None
            )
            timeout_value = timeout
            last_size = 0
            last_cpu = _read_proc_cpu_seconds(proc.pid)
            last_cpu_tree = _read_proc_tree_cpu_seconds(proc.pid)
            probe_state = _init_probe_state(self.activity_probe)
            last_progress_emit = monitor_start
            last_progress_count = 0
            POLL_INTERVAL = 5.0

            while proc.poll() is None:
                now = time.monotonic()
                if deadline is not None:
                    remaining = deadline - now
                    if remaining <= 0:
                        _terminate_process(proc)
                        timed_out = True
                        timeout_value = timeout
                        break
                    sleep_time = min(POLL_INTERVAL, remaining)
                else:
                    sleep_time = POLL_INTERVAL

                if next_heartbeat is not None:
                    sleep_time = min(
                        sleep_time,
                        max(0.1, next_heartbeat - now),
                    )

                # Wake up immediately when the process exits instead of
                # always sleeping the full poll interval.
                try:
                    proc.wait(timeout=sleep_time)
                    break
                except subprocess.TimeoutExpired:
                    pass

                # Stall detection via output file growth.
                try:
                    current_size = (
                        os.path.getsize(stdout_path)
                        + os.path.getsize(stderr_path)
                    )
                except OSError:
                    current_size = last_size

                if current_size > last_size:
                    last_size = current_size
                    last_activity = time.monotonic()
                    last_output = last_activity

                cpu_now = _read_proc_cpu_seconds(proc.pid)
                if cpu_now is not None and last_cpu is not None and cpu_now > last_cpu:
                    last_activity = time.monotonic()
                if cpu_now is not None:
                    last_cpu = cpu_now
                cpu_tree_now = _read_proc_tree_cpu_seconds(proc.pid)
                if (
                    cpu_tree_now is not None
                    and last_cpu_tree is not None
                    and cpu_tree_now > last_cpu_tree
                ):
                    last_activity = time.monotonic()
                if cpu_tree_now is not None:
                    last_cpu_tree = cpu_tree_now

                (
                    progress_count,
                    progress_last,
                    final_count,
                    final_last,
                ) = _update_probe_state(
                    self.activity_probe, stdout_path, probe_state,
                )
                if progress_count > last_progress_count:
                    last_activity = time.monotonic()
                    now = time.monotonic()
                    should_emit = (
                        (progress_count - last_progress_count) >= 20
                        or (now - last_progress_emit) >= 15
                    )
                    if should_emit:
                        print(
                            f"[PROGRESS] {self.name}: "
                            f"progress_events={progress_count}, "
                            f"last={progress_last or 'n/a'}",
                            file=sys.stderr,
                            flush=True,
                        )
                        last_progress_emit = now
                    last_progress_count = progress_count

                now = time.monotonic()
                if next_heartbeat is not None and now >= next_heartbeat:
                    elapsed = _format_duration(now - monitor_start)
                    idle = _format_duration(now - last_activity)
                    out_idle = _format_duration(now - last_output)
                    out_size = _format_bytes(last_size)
                    cpu_total = (
                        f"{cpu_now:.1f}s"
                        if cpu_now is not None
                        else "n/a"
                    )
                    cpu_tree_total = (
                        f"{cpu_tree_now:.1f}s"
                        if cpu_tree_now is not None
                        else "n/a"
                    )
                    probe_info = ""
                    if self.activity_probe != "generic":
                        probe_info = (
                            f", progress_events={progress_count}, "
                            f"last_progress={progress_last or 'n/a'}, "
                            f"final_events={final_count}, "
                            f"last_final={final_last or 'n/a'}"
                        )
                    print(
                        f"[HEARTBEAT] {self.name}: "
                        f"elapsed={elapsed}, idle={idle}, "
                        f"output_idle={out_idle}, output={out_size}, "
                        f"cpu_root={cpu_total}, cpu_tree={cpu_tree_total}"
                        f"{probe_info}",
                        file=sys.stderr,
                        flush=True,
                    )
                    next_heartbeat = now + heartbeat_secs

                if stall_timeout and stall_timeout > 0:
                    stalled_for = now - last_activity
                    if stalled_for >= stall_timeout:
                        print(
                            f"[STALL-TIMEOUT] {self.name}: no output or CPU "
                            f"activity for {_format_duration(stalled_for)} "
                            f"(limit: {_format_duration(stall_timeout)}). "
                            "Terminating reviewer.",
                            file=sys.stderr,
                            flush=True,
                        )
                        _terminate_process(proc)
                        timed_out = True
                        timeout_value = stall_timeout
                        break
        finally:
            prompt_f.close()
            stdout_f.close()
            stderr_f.close()

        proc.wait()
        output = Path(stdout_path).read_text(errors="replace")
        stderr_output = Path(stderr_path).read_text(errors="replace")
        try:
            final_size = (
                os.path.getsize(stdout_path)
                + os.path.getsize(stderr_path)
            )
        except OSError:
            final_size = last_size

        print(
            f"[{self.name}] finished rc={proc.returncode}, "
            f"output={_format_bytes(final_size)}",
            file=sys.stderr,
            flush=True,
        )

        if timed_out:
            raise subprocess.TimeoutExpired(
                self.cmd, timeout_value,  # type: ignore[arg-type]
                output=output,
                stderr=stderr_output,
            )

        return output, stderr_output, proc.returncode


# ---------------------------------------------------------------------------
# Rate-limit detection
# ---------------------------------------------------------------------------

_RATE_LIMIT_PATTERNS = [
    "rate_limit",
    "rate limit",
    "usage_limit",
    "usage limit",
    "429 too many",
    "http 429",
    "too many requests",
]


def _is_rate_limit_error(stdout: str, stderr: str) -> bool:
    """Return ``True`` if stdout/stderr contains a rate-limit indicator."""
    combined = (stdout + "\n" + stderr).lower()
    return any(pattern in combined for pattern in _RATE_LIMIT_PATTERNS)


def _init_probe_state(activity_probe: str) -> dict[str, object]:
    """Initialize per-probe parsing state."""
    if activity_probe in ("codex_json", "claude_stream_json"):
        return {
            "offset": 0,
            "tail": "",
            "progress_count": 0,
            "final_count": 0,
            "last_progress": None,
            "last_final": None,
        }
    return {}


def _update_probe_state(
    activity_probe: str,
    stdout_path: str,
    state: dict[str, object],
) -> tuple[int, str | None, int, str | None]:
    """Update probe state from stdout and return progress/final counters."""
    if activity_probe == "generic":
        return 0, None, 0, None

    chunk = _read_new_stdout_chunk(stdout_path, state)
    if not chunk:
        p_count = int(state.get("progress_count", 0))
        f_count = int(state.get("final_count", 0))
        p_last = state.get("last_progress")
        f_last = state.get("last_final")
        return (
            p_count,
            str(p_last) if isinstance(p_last, str) else None,
            f_count,
            str(f_last) if isinstance(f_last, str) else None,
        )

    tail = str(state.get("tail", ""))
    text = tail + chunk
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith(("\n", "\r")):
        state["tail"] = lines[-1]
        lines = lines[:-1]
    else:
        state["tail"] = ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        event = _extract_probe_event(activity_probe, obj)
        if not event:
            continue
        kind, label = event
        if kind == "progress":
            state["progress_count"] = int(state.get("progress_count", 0)) + 1
            state["last_progress"] = label
        elif kind == "final":
            state["final_count"] = int(state.get("final_count", 0)) + 1
            state["last_final"] = label

    p_count = int(state.get("progress_count", 0))
    f_count = int(state.get("final_count", 0))
    p_last = state.get("last_progress")
    f_last = state.get("last_final")
    return (
        p_count,
        str(p_last) if isinstance(p_last, str) else None,
        f_count,
        str(f_last) if isinstance(f_last, str) else None,
    )


def _read_new_stdout_chunk(
    stdout_path: str,
    state: dict[str, object],
) -> str:
    """Read newly appended stdout content based on tracked file offset."""
    offset = int(state.get("offset", 0))
    try:
        with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            chunk = f.read()
            state["offset"] = f.tell()
            return chunk
    except OSError:
        return ""


def _extract_probe_event(
    activity_probe: str,
    obj: object,
) -> tuple[str, str] | None:
    """Return (kind, label), where kind is 'progress' or 'final'."""
    if not isinstance(obj, dict):
        return None

    if activity_probe == "codex_json":
        return _extract_codex_event(obj)
    if activity_probe == "claude_stream_json":
        return _extract_claude_event(obj)
    return None


def _extract_codex_event(obj: dict[str, object]) -> tuple[str, str] | None:
    """Classify Codex JSONL events as progress or final."""
    top_type = _extract_nested_str(obj, ("type",))
    if not top_type:
        return None

    if top_type in ("thread.started", "turn.started", "item.started"):
        return "progress", _compact_event_label(top_type)
    if top_type in ("turn.completed", "thread.completed"):
        return "final", _compact_event_label(top_type)

    if top_type == "item.completed":
        item_type = _extract_nested_str(obj, ("item", "type"))
        if not item_type:
            return "progress", "item.completed"
        if item_type in ("reasoning", "tool_call", "tool_result"):
            return "progress", f"item.{_compact_event_label(item_type)}"
        if item_type in ("agent_message", "assistant_message"):
            return "final", f"item.{_compact_event_label(item_type)}"
        return "progress", f"item.{_compact_event_label(item_type)}"

    if top_type in ("error", "turn.failed"):
        return "final", _compact_event_label(top_type)

    return "progress", _compact_event_label(top_type)


def _extract_claude_event(obj: dict[str, object]) -> tuple[str, str] | None:
    """Classify Claude stream-json events as progress or final."""
    top_type = _extract_nested_str(obj, ("type",))
    if not top_type:
        return None

    if top_type in ("result", "assistant"):
        return "final", _compact_event_label(top_type)
    if top_type == "system":
        subtype = _extract_nested_str(obj, ("subtype",))
        label = f"system.{subtype}" if subtype else "system"
        return "progress", _compact_event_label(label)

    if top_type == "stream_event":
        event_type = _extract_nested_str(obj, ("event", "type"))
        if not event_type:
            return "progress", "stream_event"
        if event_type in ("message_stop",):
            return "final", _compact_event_label(f"stream.{event_type}")
        return "progress", _compact_event_label(f"stream.{event_type}")

    return "progress", _compact_event_label(top_type)


def _extract_nested_str(data: object, path: tuple[str, ...]) -> str | None:
    """Return a nested string value if all path segments are present."""
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, str):
        return cur
    return None


def _compact_event_label(label: str) -> str:
    """Normalize an event label for compact heartbeat output."""
    text = re.sub(r"\s+", " ", label.strip())
    return text[:80]


def _extract_reviewer_output(activity_probe: str, raw_output: str) -> str:
    """Return user-facing reviewer output, stripping probe-only telemetry."""
    if activity_probe == "codex_json":
        return _extract_codex_final_output(raw_output)
    if activity_probe == "claude_stream_json":
        return _extract_claude_final_output(raw_output)
    return raw_output


def _extract_codex_final_output(raw_output: str) -> str:
    """Extract final assistant text from Codex JSONL output."""
    final_text: str | None = None
    for obj in _iter_json_lines(raw_output):
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "item.completed":
            continue
        item = obj.get("item")
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type not in ("agent_message", "assistant_message"):
            continue
        text = item.get("text")
        if isinstance(text, str):
            final_text = text
    return final_text if final_text is not None else raw_output


def _extract_claude_final_output(raw_output: str) -> str:
    """Extract final assistant text from Claude stream-json output."""
    result_text: str | None = None
    assistant_text: str | None = None

    for obj in _iter_json_lines(raw_output):
        if not isinstance(obj, dict):
            continue
        top_type = obj.get("type")
        if top_type == "result":
            value = obj.get("result")
            if isinstance(value, str):
                result_text = value
        elif top_type == "assistant":
            msg_text = _extract_claude_message_text(obj.get("message"))
            if msg_text is not None:
                assistant_text = msg_text

    if result_text is not None:
        return result_text
    if assistant_text is not None:
        return assistant_text
    return raw_output


def _extract_claude_message_text(message: object) -> str | None:
    """Extract concatenated text blocks from a Claude assistant message."""
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    if not parts:
        return None
    return "".join(parts)


def _iter_json_lines(raw_output: str) -> list[object]:
    """Parse newline-delimited JSON objects; ignore malformed lines."""
    parsed: list[object] = []
    for line in raw_output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            parsed.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue
    return parsed


def _read_proc_cpu_seconds(pid: int) -> float | None:
    """Best-effort process CPU time from /proc/<pid>/stat (Linux)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    fields = stat.split()
    if len(fields) < 15:
        return None
    try:
        ticks = int(fields[13]) + int(fields[14])
    except ValueError:
        return None
    hz = os.sysconf("SC_CLK_TCK")
    return ticks / hz


def _read_proc_tree_cpu_seconds(root_pid: int) -> float | None:
    """Best-effort total CPU time for a process tree rooted at *root_pid*."""
    seen: set[int] = set()
    queue: list[int] = [root_pid]
    total = 0.0
    any_sample = False

    while queue:
        pid = queue.pop()
        if pid in seen:
            continue
        seen.add(pid)

        cpu = _read_proc_cpu_seconds(pid)
        if cpu is not None:
            total += cpu
            any_sample = True

        for child in _read_proc_children(pid):
            if child not in seen:
                queue.append(child)

    return total if any_sample else None


def _read_proc_children(pid: int) -> list[int]:
    """Read child PIDs from /proc/<pid>/task/<pid>/children (Linux)."""
    children_path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        raw = children_path.read_text(encoding="utf-8").strip()
    except OSError:
        return []
    if not raw:
        return []
    out: list[int] = []
    for token in raw.split():
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out


def _terminate_process(proc: subprocess.Popen[bytes]) -> None:
    """Terminate a subprocess, escalating to kill if needed."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as HhMmSs (compact)."""
    total = max(0, int(seconds))
    mins, sec = divmod(total, 60)
    hrs, mins = divmod(mins, 60)
    if hrs:
        return f"{hrs}h{mins:02d}m{sec:02d}s"
    if mins:
        return f"{mins}m{sec:02d}s"
    return f"{sec}s"


def _format_bytes(num: int) -> str:
    """Format bytes into a short human-readable unit."""
    value = float(max(0, num))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{int(value)}B"


# ---------------------------------------------------------------------------
# Reviewer presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, tuple[str, str, str]] = {
    # key -> (display_name, shell_command, activity_probe)
    "claude": (
        "Claude CLI",
        "claude --dangerously-skip-permissions -p "
        "--output-format stream-json --include-partial-messages --verbose",
        "claude_stream_json",
    ),
    "codex": (
        "Codex CLI",
        "codex --dangerously-bypass-approvals-and-sandbox exec --json",
        "codex_json",
    ),
    "cursor-opus": (
        "Cursor / Opus 4.6 Thinking",
        "cursor agent -p -f --model opus-4.6-thinking",
        "generic",
    ),
    "cursor-gpt": (
        "Cursor / GPT 5.2 Codex XHigh",
        "cursor agent -p -f --model gpt-5.2-codex-xhigh",
        "generic",
    ),
}

PRESET_KEYS = list(PRESETS.keys())

# Automatic fallback mapping: if the primary preset fails (non-zero exit
# or timeout), the mapped preset is tried instead.
FALLBACK_MAP: dict[str, str] = {
    "claude": "codex",
    "codex": "cursor-gpt",
}

# Rate-limit fallback: same as FALLBACK_MAP by default.  Override entries
# here when you want rate-limited presets to fall back to a *different*
# preset than the generic failure fallback (e.g. cross-provider).
RATE_LIMIT_FALLBACK_MAP: dict[str, str] = {
    "claude": "codex",            # Anthropic -> OpenAI direct
    "codex": "cursor-gpt",        # OpenAI direct -> OpenAI via Cursor
    "cursor-opus": "cursor-gpt",  # Anthropic via Cursor -> OpenAI via Cursor
    "cursor-gpt": "cursor-opus",  # OpenAI via Cursor -> Anthropic via Cursor
}


def pick_reviewer(role: str) -> Reviewer:
    """Interactively ask the user to pick a reviewer preset."""
    print(f"\nSelect {role}:")
    for i, key in enumerate(PRESET_KEYS, 1):
        name, cmd, _probe = PRESETS[key]
        print(f"  {i}) {name}  [{key}]")
    while True:
        choice = input(f"Choice [1-{len(PRESET_KEYS)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(PRESET_KEYS):
            key = PRESET_KEYS[int(choice) - 1]
            name, cmd, probe = PRESETS[key]
            return Reviewer(name=name, cmd=cmd, activity_probe=probe)
        if choice in PRESETS:
            name, cmd, probe = PRESETS[choice]
            return Reviewer(name=name, cmd=cmd, activity_probe=probe)
        print(f"  Invalid choice. Enter 1-{len(PRESET_KEYS)} or a preset key.")


def resolve_reviewers(args: argparse.Namespace) -> list[Reviewer]:
    """Resolve the full list of reviewers from CLI args or interactive input.

    Resolution order:
      1. Preset-based reviewers (``--reviewer``, repeatable)
      2. Custom reviewers (``--reviewer-cmd`` / ``--reviewer-name`` pairs)
      3. If neither is provided, interactive fallback
    """
    reviewers: list[Reviewer] = []

    # 1. Presets
    no_fallback = getattr(args, "no_fallback", False)
    for preset_key in (args.reviewer or []):
        if preset_key not in PRESETS:
            sys.exit(
                f"Unknown preset '{preset_key}'. "
                f"Choose from: {', '.join(PRESET_KEYS)}"
            )
        disp, shell_cmd, probe = PRESETS[preset_key]
        fallback = None
        rate_limit_fallback = None
        if not no_fallback:
            fb_key = FALLBACK_MAP.get(preset_key)
            if fb_key:
                fb_name, fb_cmd, fb_probe = PRESETS[fb_key]
                fallback = Reviewer(
                    name=fb_name, cmd=fb_cmd, activity_probe=fb_probe,
                )
            rl_fb_key = RATE_LIMIT_FALLBACK_MAP.get(preset_key)
            if rl_fb_key:
                rl_fb_name, rl_fb_cmd, rl_fb_probe = PRESETS[rl_fb_key]
                rate_limit_fallback = Reviewer(
                    name=rl_fb_name, cmd=rl_fb_cmd, activity_probe=rl_fb_probe,
                )
        reviewers.append(Reviewer(
            name=disp, cmd=shell_cmd,
            activity_probe=probe,
            fallback=fallback,
            rate_limit_fallback=rate_limit_fallback,
        ))

    # 2. Custom command/name pairs
    custom_cmds: list[str] = args.reviewer_cmd or []
    custom_names: list[str] = args.reviewer_name or []
    if len(custom_cmds) != len(custom_names):
        sys.exit(
            "--reviewer-cmd and --reviewer-name must be specified the "
            f"same number of times (got {len(custom_cmds)} commands "
            f"and {len(custom_names)} names)"
        )
    for cmd, name in zip(custom_cmds, custom_names):
        reviewers.append(Reviewer(name=name, cmd=cmd))

    # 3. Interactive fallback
    if not reviewers:
        n_str = input("\nHow many reviewers? [2]: ").strip()
        count = int(n_str) if n_str.isdigit() and int(n_str) >= 1 else 2
        for i in range(1, count + 1):
            reviewers.append(pick_reviewer(f"Reviewer {i}"))

    if not reviewers:
        sys.exit("At least one reviewer is required.")

    return reviewers


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def git_commit(doc_path: str, message: str) -> None:
    """Stage a document and create a signed commit.

    Uses ``git commit --only`` so that only *doc_path* is included in the
    commit, even if other files happen to be staged in the index.
    """
    subprocess.run(["git", "add", doc_path], check=True)
    subprocess.run(
        ["git", "commit", "--only", doc_path, "-s", "-m", message],
        check=True,
    )


# ---------------------------------------------------------------------------
# Comment persistence
# ---------------------------------------------------------------------------


def comments_dir(base: Path, doc_path: str) -> Path:
    """Return a per-document comments directory under *base*.

    For paths under ``tasks/<slug>/`` (e.g. ``tasks/20260208-foo/plan.md``),
    uses the slug directory name so all plans don't collide into
    ``base/plan/``.  Otherwise falls back to the filename stem.

    Examples:
        tasks/20260208-foo/plan.md  -> base/20260208-foo/
        docs/autoep-design.md       -> base/autoep-design/
    """
    p = Path(doc_path)
    # If the parent looks like a slug directory (not repo root / generic),
    # use the parent directory name as the key.
    if p.parent.name and p.parent.name not in (".", "docs", "todo"):
        return base / p.parent.name
    return base / p.stem


def sanitize_name(name: str) -> str:
    """Convert a display name to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save_comments(
    comments_dir: Path,
    cycle: int,
    reviewer_name: str,
    output: str,
    label: str = "",
) -> Path:
    """Persist reviewer output to a per-document comments directory."""
    comments_dir.mkdir(parents=True, exist_ok=True)
    slug = sanitize_name(reviewer_name)
    tag = f"_{label}" if label else ""
    filename = comments_dir / f"cycle{cycle}_{slug}{tag}.txt"
    filename.write_text(output, encoding="utf-8")
    return filename


# ---------------------------------------------------------------------------
# Common CLI arguments
# ---------------------------------------------------------------------------


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add reviewer, timeout, cycle, and commit arguments to *parser*."""
    preset_list = ", ".join(PRESET_KEYS)

    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="Number of review cycles (default: 3)",
    )

    # --- Preset-based selection (repeatable) ----------------------------
    parser.add_argument(
        "--reviewer",
        action="append",
        choices=PRESET_KEYS,
        default=None,
        dest="reviewer",
        metavar="PRESET",
        help=(
            f"Add a reviewer by preset ({preset_list}). "
            "Can be repeated for multiple reviewers."
        ),
    )

    # --- Fully custom (repeatable, matched by position) -----------------
    parser.add_argument(
        "--reviewer-cmd",
        action="append",
        default=None,
        dest="reviewer_cmd",
        metavar="CMD",
        help=(
            "Custom shell command for a reviewer. Can be repeated; "
            "each must be paired with a --reviewer-name."
        ),
    )
    parser.add_argument(
        "--reviewer-name",
        action="append",
        default=None,
        dest="reviewer_name",
        metavar="NAME",
        help=(
            "Display name for a custom reviewer (must pair with "
            "--reviewer-cmd by position)."
        ),
    )

    # --- Additional context -----------------------------------------------
    parser.add_argument(
        "--context",
        action="append",
        default=None,
        dest="context",
        metavar="TEXT",
        help=(
            "Additional context to include in every prompt. "
            "Can be repeated; all values are concatenated."
        ),
    )
    parser.add_argument(
        "--context-file",
        action="append",
        default=None,
        dest="context_file",
        metavar="FILE",
        help=(
            "Path to a file whose contents are included as additional "
            "context in every prompt. Can be repeated."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help=(
            "Maximum seconds to wait for each reviewer invocation "
            "(default: 1800 = 30 min). Use 0 for no limit."
        ),
    )
    parser.add_argument(
        "--stall-timeout",
        type=int,
        default=0,
        help=(
            "Maximum seconds of no output/CPU activity before killing a "
            "reviewer (default: disabled). Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--heartbeat-secs",
        type=int,
        default=30,
        help=(
            "Seconds between liveness heartbeats while a reviewer runs "
            "(default: 30). Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Skip git commits after each review",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help=(
            "Disable automatic fallback to alternative reviewers on "
            "failure (by default claude falls back to cursor-opus, "
            "codex falls back to cursor-gpt)"
        ),
    )
    parser.add_argument(
        "--comments-dir",
        default=None,
        dest="comments_dir",
        metavar="DIR",
        help="Directory for reviewer comments (overrides default)",
    )


def postprocess_common_args(args: argparse.Namespace) -> None:
    """Apply common post-parse fixups (e.g. timeout=0 → None)."""
    if args.timeout == 0:
        args.timeout = None
    if getattr(args, "stall_timeout", 0) == 0:
        args.stall_timeout = None
    if getattr(args, "heartbeat_secs", 0) <= 0:
        args.heartbeat_secs = None

    # Resolve --context and --context-file into a single string.
    parts: list[str] = []
    for text in (args.context or []):
        parts.append(text)
    for filepath in (args.context_file or []):
        p = Path(filepath)
        if not p.exists():
            sys.exit(f"Error: context file not found: {filepath}")
        parts.append(p.read_text(encoding="utf-8"))
    args.additional_context = "\n\n".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def append_context(prompt: str, additional_context: str | None) -> str:
    """Append additional user-supplied context to a prompt, if any."""
    if not additional_context:
        return prompt
    return (
        f"{prompt}\n\n"
        "--- ADDITIONAL CONTEXT ---\n"
        f"{additional_context}"
    )


# ---------------------------------------------------------------------------
# Review-loop helper
# ---------------------------------------------------------------------------


def run_review_loop(
    *,
    reviewers: list[Reviewer],
    cycles: int,
    doc_path: str,
    build_prompt_fn: "BuildPromptFn",
    cdir: Path,
    no_commit: bool,
    timeout: int | None,
    stall_timeout: int | None,
    heartbeat_secs: int | None,
    cycle_label: str = "reviewing",
) -> None:
    """Execute the iterative review loop shared by both scripts.

    Parameters
    ----------
    reviewers :
        Ordered list of reviewers to cycle through.
    cycles :
        Number of full review rounds.
    doc_path :
        Path to the document being committed after each review.
    build_prompt_fn :
        Callable ``(prev_comments_path, prev_reviewer_name) -> str``
        that builds the review prompt for the current iteration.
    cdir :
        Directory where reviewer comments are saved.
    no_commit :
        If ``True``, skip git commits.
    timeout :
        Per-reviewer wall-clock timeout in seconds (``None`` = no limit).
    stall_timeout :
        No-activity timeout in seconds (``None`` = disabled). Activity is
        stdout/stderr growth or process CPU-time growth.
    heartbeat_secs :
        Heartbeat period in seconds (``None`` = disabled).
    cycle_label :
        Short verb phrase for the log banner (e.g. ``"reviewing"``
        or ``"reviewing plan"``).
    """
    prev_comments_path: Path | None = None
    prev_reviewer_name: str | None = None

    for cycle in range(1, cycles + 1):
        for reviewer in reviewers:
            prompt = build_prompt_fn(prev_comments_path, prev_reviewer_name)

            sep = "=" * 60
            print(f"\n{sep}")
            print(
                f"  Cycle {cycle}/{cycles} — "
                f"{reviewer.name} {cycle_label}"
            )
            print(f"{sep}\n")

            try:
                output = reviewer.run(
                    prompt,
                    timeout=timeout,
                    stall_timeout=stall_timeout,
                    heartbeat_secs=heartbeat_secs,
                )
            except subprocess.TimeoutExpired as exc:
                limit_value = (
                    exc.timeout if exc.timeout is not None else timeout
                )
                limit_desc = (
                    f"{limit_value}s"
                    if limit_value is not None
                    else "the configured limit"
                )
                print(
                    f"\n[TIMEOUT] {reviewer.name} exceeded {limit_desc} "
                    f"during cycle {cycle}. Partial output (if any) "
                    "was printed above.",
                    file=sys.stderr,
                )
                output = exc.output or ""

            prev_comments_path = save_comments(
                cdir, cycle, reviewer.name, output,
            )
            prev_reviewer_name = reviewer.name
            print(f"\n[Saved comments to {prev_comments_path}]")

            if not no_commit:
                try:
                    git_commit(
                        doc_path,
                        f"update {doc_path} by {reviewer.name} "
                        f"(cycle {cycle})",
                    )
                    print(
                        f"\n[Committed: update {doc_path} by "
                        f"{reviewer.name} (cycle {cycle})]"
                    )
                except subprocess.CalledProcessError as exc:
                    print(
                        f"\n[Git commit failed: {exc}]", file=sys.stderr,
                    )



class BuildPromptFn(Protocol):
    """Signature for the prompt-builder passed to ``run_review_loop``."""

    def __call__(
        self,
        prev_comments_path: Path | None,
        prev_reviewer_name: str | None,
    ) -> str: ...
