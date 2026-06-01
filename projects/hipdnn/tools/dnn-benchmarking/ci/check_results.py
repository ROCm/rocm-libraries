# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Gate checker for the hipDNN benchmark presubmit CI.

Parses a ``results.json`` produced by ``python -m dnn_benchmarking --output``
and asserts that the benchmark tool actually ran something successfully:

    pass_combinations > 0  AND  error_combinations == 0

This is a *tool-health* gate, not a performance or correctness gate. It exists
to catch two failure modes the tool's own exit code does not:

* ``error_combinations > 0`` -> a graph/engine combination errored at runtime.
  (The tool already exits 1 here, but we re-assert defensively.)
* ``pass_combinations == 0`` -> every combination was *skipped* (e.g. the plugin
  path was wrong or the plugin was missing), which the tool reports as exit 0.
  Without this check that silent "nothing ran" case would pass CI.

Schema contract (``reporting/suite_results.py`` ``SuiteResult.to_dict``):
the counts live under the top-level ``metadata`` object. A missing field means
the schema changed; we fail loudly rather than silently passing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple


class GateError(Exception):
    """Raised when results.json is unreadable or missing required fields."""


def _read_counts(results_path: Path) -> Tuple[int, int]:
    """Return (pass_combinations, error_combinations) from results.json.

    Raises GateError if the file is missing/unparseable or either required
    field is absent (schema drift must not silently pass the gate).
    """
    try:
        raw = results_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GateError(f"cannot read results file '{results_path}': {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GateError(
            f"results file '{results_path}' is not valid JSON: {exc}"
        ) from exc

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise GateError(
            "results file is missing the 'metadata' object "
            "(schema changed?) — refusing to pass the gate"
        )

    missing = [
        k for k in ("pass_combinations", "error_combinations") if k not in metadata
    ]
    if missing:
        raise GateError(
            f"results metadata missing required field(s): {', '.join(missing)} "
            "(schema changed?) — refusing to pass the gate"
        )

    return int(metadata["pass_combinations"]), int(metadata["error_combinations"])


def evaluate(results_path: Path) -> Tuple[bool, str]:
    """Evaluate the gate. Returns (ok, message)."""
    passed, errored = _read_counts(results_path)
    if errored > 0:
        return False, (
            f"benchmark gate FAILED: error_combinations={errored} (> 0) — "
            "at least one graph/engine combination errored at runtime"
        )
    if passed == 0:
        return False, (
            "benchmark gate FAILED: pass_combinations=0 — nothing ran "
            "successfully (all combinations skipped/unsupported; check the "
            "plugin path and that engines are installed)"
        )
    return True, (
        f"benchmark gate PASSED: pass_combinations={passed}, "
        f"error_combinations={errored}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="?",
        default="results.json",
        type=Path,
        help="path to results.json (default: results.json)",
    )
    args = parser.parse_args(argv)

    try:
        ok, message = evaluate(args.results)
    except GateError as exc:
        print(f"benchmark gate ERROR: {exc}", file=sys.stderr)
        return 1

    print(message, file=sys.stderr if not ok else sys.stdout)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
