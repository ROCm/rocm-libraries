#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""One `hipdnn_bench` call against one descriptor tree, with nothing else deciding the answer.

Standard library only, plus `uhd_gen.bench_env`; imported by validate_l1.py, validate_l2.py and
candidate_tree.py.

`isolated_env` is `uhd_gen.bench_env.bench_environment`, the environment generate collects
with: only the named descriptor tree, never benchmarking, logging to stderr. A benchmark
winner record in `HIPDNN_CACHE_DIR` takes precedence over every heuristic
(KernelIngestorStateManager.hpp), and a `--collect-immediate` run writes one, so `call` gives
each run an empty cache of its own and removes it afterwards.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    from uhd_gen.bench_env import bench_environment
except ImportError:
    # Run from a checkout without PYTHONPATH: uhd_gen sits in the same tools directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from uhd_gen.bench_env import bench_environment


def isolated_env(
    tree: Optional[Path] = None, base: Optional[Mapping[str, str]] = None
) -> dict[str, str]:
    """`base` (default: this process's environment) reading only `tree`, or only the build's
    own tree when None (`uhd_gen.bench_env.bench_environment`)."""
    return bench_environment(base, descriptor_dir=tree)


def first_json_object(text: str) -> dict[str, Any]:
    """The first JSON object printed in `text`; ValueError when there is none."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except ValueError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    raise ValueError("hipdnn_bench emitted no JSON object")


def call(
    command: list[str], env: Mapping[str, str], timeout: int, scratch: Path
) -> dict[str, Any]:
    """Run `command` with a new empty `HIPDNN_CACHE_DIR` under `scratch`.

    Returns `ok` (exit 0 and a JSON object printed), `returncode`, `stdout`, `stderr`,
    `timed_out`, and `response` (the object) or `error`.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cache-", dir=scratch) as cache:
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=dict(env, HIPDNN_CACHE_DIR=cache),
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "timed_out": True,
                "error": f"no answer within {timeout} s",
            }
    result: dict[str, Any] = {
        "ok": False,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "timed_out": False,
    }
    if proc.returncode != 0:
        result["error"] = (proc.stderr or proc.stdout)[-800:]
        return result
    try:
        result["response"] = first_json_object(proc.stdout)
    except ValueError as error:
        result["error"] = str(error)
        return result
    result["ok"] = True
    return result


def graphs_of(
    sample: Optional[str], roots: list[str], limit: Optional[int]
) -> list[Path]:
    """The graphs a validation asks about: an `eval_slice.py sample` document's, or every
    `*.fb` / `*.json` graph under `roots` (manifests excluded); the first `limit` of them.
    """
    if sample:
        document = json.loads(Path(sample).read_text(encoding="utf-8"))
        found = [Path(g["path"]) for g in document.get("graphs") or []]
    else:
        found = []
        for root in roots:
            path = Path(root)
            if path.is_file():
                found.append(path)
                continue
            for pattern in ("*.fb", "*.json"):
                found.extend(
                    p for p in path.rglob(pattern) if not p.name.startswith("manifest")
                )
        found = sorted(set(found))
    return found[:limit] if limit else found
