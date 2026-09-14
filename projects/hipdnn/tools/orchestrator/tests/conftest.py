# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Shared fixtures. Every test is hermetic: the only executable launched is this Python."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Stands in for an agent CLI: writes a JSON result file and prints a JSON envelope on
#: stdout, exactly like `claude --print --output-format json` does.
FAKE_AGENT = textwrap.dedent(
    """
    import argparse, json, pathlib, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--result")
    parser.add_argument("--payload", default="{}")
    parser.add_argument("--counter")
    parser.add_argument("--sequence", default="")
    parser.add_argument("--skip-result", action="store_true")
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args()

    index = 0
    if args.counter:
        counter = pathlib.Path(args.counter)
        index = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(index + 1))

    payload = json.loads(args.payload)
    if args.sequence:
        steps = args.sequence.split(",")
        value = int(steps[min(index, len(steps) - 1)])
        payload["critical_count"] = value
        payload["verdict"] = "pass" if value == 0 else "changes_required"
        payload["critical_issues"] = [{"title": f"issue {n}"} for n in range(value)]
        payload["feedback"] = f"round {index}: {value} critical issue(s)"

    if not args.skip_result and args.result:
        pathlib.Path(args.result).write_text(json.dumps(payload))
    print(json.dumps({"session_id": "session-123", "result": "ok", "round": index}))
    sys.exit(args.exit_code)
    """
)


@pytest.fixture
def agent_script(tmp_path: Path) -> Path:
    script = tmp_path / "fake_agent.py"
    script.write_text(FAKE_AGENT, encoding="utf-8")
    return script


@pytest.fixture
def registry_file(tmp_path: Path) -> Path:
    path = tmp_path / "tools.yaml"
    path.write_text(
        "version: 1\n"
        "vars:\n"
        f"  work: {tmp_path.as_posix()}\n"
        "tools:\n"
        "  agent:\n"
        f"    exe: {Path(sys.executable).as_posix()}\n",
        encoding="utf-8",
    )
    return path
