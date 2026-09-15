#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Read an agent step's session: what it thought, what it ran, what it changed.

Agent steps run with `--output-format stream-json`, so their `stdout.log` is JSONL --
one event per turn, written as it happens. That is the right thing for a machine and
the wrong thing for a person: a single assistant turn is one line thousands of
characters wide. This renders it.

    # newest run, newest iteration, the implement step -- the common case
    agent_log.py

    # follow it live while the step is still running
    agent_log.py --follow

    # a specific run / step / iteration
    agent_log.py --run runs/test-engine-kernel/20260915T033903Z-a232 --step review --iter 2

    # just the session id, to reopen the conversation
    agent_log.py --session-id

`--session-id` prints the CLI's own session id for that step. `claude --resume <id>`
reopens the actual conversation, with full history, from the directory the step ran
in. The id is also recorded in `run.json` under the step's outputs, so a finished run
does not lose the reasoning that produced it.

Exit codes: 0 rendered, 1 nothing to render (no run, no step, empty log).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RUNS_DEFAULT = Path(__file__).resolve().parent.parent / "runs"

#: Enough of a tool call to recognise it without reproducing it. A full Write input is
#: the file that is already on disk; a full Bash input is the command, which is short.
TOOL_PREVIEW_KEYS = (
    "command",
    "file_path",
    "pattern",
    "query",
    "path",
    "prompt",
    "description",
    "notebook_path",
    "url",
    "old_string",
)


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _find_latest(path: Path) -> Path | None:
    entries = sorted((p for p in path.iterdir() if p.is_dir()), key=lambda p: p.name)
    return entries[-1] if entries else None


def _resolve_step_dir(run: Path, step: str, iteration: int | None) -> Path | None:
    """A step is either top-level (`<run>/<step>`) or inside a loop
    (`<run>/<loop>/iter-NN/<step>`). Both are searched, loop first, because the
    interesting agent steps live in the loop."""
    candidates: list[Path] = []
    for loop_dir in sorted(p for p in run.iterdir() if p.is_dir() and p.name != step):
        iters = sorted(
            p for p in loop_dir.iterdir() if p.is_dir() and p.name.startswith("iter-")
        )
        if not iters:
            continue
        if iteration is None:
            chosen = [iters[-1]]
        else:
            chosen = [p for p in iters if p.name == f"iter-{iteration:02d}"]
        candidates += [p / step for p in chosen]
    candidates.append(run / step)
    for candidate in candidates:
        if (candidate / "stdout.log").is_file():
            return candidate
    return None


def _render(event: dict, show_thinking: bool, width: int) -> list[str]:
    kind = event.get("type")

    if kind == "system" and event.get("subtype") == "init":
        return [
            f"-- session {event.get('session_id', '?')}  cwd {event.get('cwd', '?')}"
        ]

    if kind == "assistant":
        lines = []
        for block in event.get("message", {}).get("content", []):
            btype = block.get("type")
            if btype == "thinking" and show_thinking:
                for para in str(block.get("thinking", "")).strip().split("\n"):
                    if para.strip():
                        lines.append(f"   . {_clip(para, width)}")
            elif btype == "text":
                for para in str(block.get("text", "")).strip().split("\n"):
                    if para.strip():
                        lines.append(f"   {_clip(para, width)}")
            elif btype == "tool_use":
                args = block.get("input", {}) or {}
                preview = next(
                    (
                        f"{key}={_clip(args[key], width - 24)}"
                        for key in TOOL_PREVIEW_KEYS
                        if key in args and args[key]
                    ),
                    _clip(", ".join(sorted(args)), width - 24),
                )
                lines.append(f"  -> {block.get('name', '?')}  {preview}")
        return lines

    if kind == "user":
        # Tool results come back as user turns. Only the failures are worth surfacing;
        # a successful Read echoing a file back is noise.
        lines = []
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "tool_result" and block.get("is_error"):
                content = block.get("content")
                if isinstance(content, list):
                    content = " ".join(
                        str(c.get("text", "")) for c in content if isinstance(c, dict)
                    )
                lines.append(f"  !! error: {_clip(content or '', width)}")
        return lines

    if kind == "result":
        cost = event.get("total_cost_usd")
        return [
            "-- done"
            + (
                f"  {event.get('num_turns')} turns"
                if event.get("num_turns") is not None
                else ""
            )
            + (
                f"  {event.get('duration_ms', 0) / 1000:.0f}s"
                if event.get("duration_ms")
                else ""
            )
            + (f"  ${cost:.2f}" if isinstance(cost, (int, float)) else "")
            + (f"  stop={event.get('stop_reason')}" if event.get("stop_reason") else "")
            + ("  ERROR" if event.get("is_error") else "")
        ]

    return []


def main() -> int:
    # A Windows console defaults to cp1252, and an agent transcript is full of em
    # dashes and arrows. Without this the viewer dies on UnicodeEncodeError partway
    # through a session -- worst of all while --follow is watching a live step.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", help="run directory; default is the newest run of any flow"
    )
    parser.add_argument("--flow", help="flow name to pick the newest run from")
    parser.add_argument(
        "--step", default="implement", help="step id (default: implement)"
    )
    parser.add_argument(
        "--iter", type=int, help="loop iteration; default is the newest"
    )
    parser.add_argument(
        "--follow", action="store_true", help="keep reading as the step writes"
    )
    parser.add_argument(
        "--session-id", action="store_true", help="print the session id and exit"
    )
    parser.add_argument(
        "--no-thinking", action="store_true", help="hide reasoning blocks"
    )
    parser.add_argument(
        "--width", type=int, default=160, help="wrap width (default 160)"
    )
    args = parser.parse_args()

    if args.run:
        run = Path(args.run).resolve()
    else:
        root = RUNS_DEFAULT
        if args.flow:
            root = root / args.flow
        if not root.is_dir():
            print(f"error: no runs under {root}", file=sys.stderr)
            return 1
        flow_dir = root if args.flow else _find_latest(root)
        run = _find_latest(flow_dir) if flow_dir else None
        if run is None:
            print(f"error: no run directories under {root}", file=sys.stderr)
            return 1

    if not run.is_dir():
        print(f"error: not a run directory: {run}", file=sys.stderr)
        return 1

    step_dir = _resolve_step_dir(run, args.step, args.iter)
    if step_dir is None:
        print(
            f"error: no stdout.log for step '{args.step}' under {run}", file=sys.stderr
        )
        return 1
    log = step_dir / "stdout.log"

    if args.session_id:
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("session_id"):
                print(event["session_id"])
                return 0
        print("error: no session id in the log", file=sys.stderr)
        return 1

    print(f"== {step_dir.relative_to(run)}  ({run.name})")

    def emit(handle) -> None:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                # Not every line is an event: a tool can write to the same stream.
                print(f"   {_clip(line, args.width)}")
                continue
            for rendered in _render(event, not args.no_thinking, args.width):
                print(rendered, flush=True)

    with log.open("r", encoding="utf-8", errors="replace") as handle:
        emit(handle)
        if args.follow:
            # The step is still running; `run.json` flipping this step off "running"
            # is the signal to stop, but a plain interrupt is the usual exit.
            try:
                while True:
                    where = handle.tell()
                    line = handle.readline()
                    if line:
                        handle.seek(where)
                        emit(handle)
                    else:
                        time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n-- stopped following")
    return 0


if __name__ == "__main__":
    sys.exit(main())
