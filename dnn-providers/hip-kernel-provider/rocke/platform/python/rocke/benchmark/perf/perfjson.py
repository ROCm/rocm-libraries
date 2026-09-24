# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The `PerfJSON:` line - the machine-readable contract between a kernel launcher
and any tool that measures it.

A benchmark script prints one `PerfJSON:` line with its own timing; `harness` parses
it for the record's `wall` / `profiled` sections plus tflops/gbs, which the profiler
cannot supply. Emitting it is OPTIONAL - without it the harness falls back to
rocprofv3 dispatch durations and reports `timing_source="rocprofv3_duration"` - but a
launcher that emits it gets real-world (un-profiled) wall timing and throughput.

Use this instead of formatting the line by hand, and never scrape a human-readable
"Perf: ... TFlops" string: that text is for people and its format is not a contract.

    from rocke.benchmark.perf import perfjson
    perfjson.emit(ms=ms, tflops=tflops, gbps=gbps)

`payload` / `format_line` are pure; `emit` writes only to its caller-selected stream
(stdout by default). Stdlib only, no persistence.
"""
from __future__ import annotations

import json
import math
import sys
from typing import IO, Any

PREFIX = "PerfJSON:"


def payload(
    *,
    ms: float | None = None,
    tflops: float | None = None,
    gbps: float | None = None,
    pct_peak: float | None = None,
    max_abs_diff: float | None = None,
    bad_count: int | None = None,
    total: int | None = None,
    **extra: Any,
) -> dict:
    """The payload dict, with absent and non-finite values dropped.

    Non-finite timing is dropped rather than serialized as NaN/Infinity: `json.dumps`
    would emit bare `NaN`, which is not valid JSON, and the harness rejects a
    non-finite `ms` anyway. Dropping it makes the launcher's failure legible as a
    missing field instead of an unparseable line.
    """
    out: dict = {}
    named = {
        "ms": ms,
        "tflops": tflops,
        "gbps": gbps,
        "pct_peak": pct_peak,
        "max_abs_diff": max_abs_diff,
        "bad_count": bad_count,
        "total": total,
    }
    for key, value in {**named, **extra}.items():
        if value is None:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        out[key] = value
    return out


def format_line(**fields: Any) -> str:
    """Render one `PerfJSON:` line (no trailing newline). Same kwargs as `payload`."""
    return f"{PREFIX} {json.dumps(payload(**fields))}"


def emit(*, stream: IO[str] | None = None, **fields: Any) -> str:
    """Print the `PerfJSON:` line (default stdout, where the harness reads it).

    Returns the line so a caller can log it too.
    """
    line = format_line(**fields)
    print(line, file=stream if stream is not None else sys.stdout)
    return line
