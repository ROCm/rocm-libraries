# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compatibility entry point for the unified prefill-2D benchmark harness.

New experiments should use ``benchmark_prefill2d_fastkv_regp.py`` directly.
This wrapper preserves the old trace-combo defaults:

    PYTHONPATH=python python/ck_dsl/.venv/bin/python \\
      python/ck_dsl/examples/attention/benchmark_prefill2d_traces.py

It forwards to the unified harness with ``--smart-dispatch-policy trace_combo``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from benchmark_prefill2d_fastkv_regp import main as _unified_main


ROOT = Path(__file__).resolve().parents[4]  # projects/composablekernel
DEFAULT_OUTPUT_JSON = Path("/tmp/ckdsl_prefill2d_trace_combo.json")
DEFAULT_OUTPUT_CSV = Path("/tmp/ckdsl_prefill2d_trace_combo.csv")
DEFAULT_JOINED_CSV = (
    ROOT
    / "python"
    / "ck_dsl"
    / "examples"
    / "attention"
    / "prefill2d_bf16_triton_ckdsl_perf.csv"
)


def _has_option(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def _translate_args(argv: list[str]) -> list[str]:
    forwarded = ["--smart-dispatch-policy", "trace_combo"]

    if not _has_option(argv, "--output-json"):
        forwarded.extend(["--output-json", str(DEFAULT_OUTPUT_JSON)])
    if not _has_option(argv, "--output-csv"):
        forwarded.extend(["--output-csv", str(DEFAULT_OUTPUT_CSV)])
    if not _has_option(argv, "--joined-csv") and not _has_option(
        argv, "--combined-csv"
    ):
        forwarded.extend(["--joined-csv", str(DEFAULT_JOINED_CSV)])

    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--combined-csv":
            forwarded.append("--joined-csv")
            if index + 1 < len(argv):
                forwarded.append(argv[index + 1])
                index += 2
            else:
                index += 1
            continue
        if arg.startswith("--combined-csv="):
            forwarded.append("--joined-csv=" + arg.split("=", 1)[1])
            index += 1
            continue
        forwarded.append(arg)
        index += 1

    return forwarded


def main(argv: list[str] | None = None) -> int:
    return _unified_main(_translate_args(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    sys.exit(main())
