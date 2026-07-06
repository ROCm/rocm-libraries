# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Measurement-record schema — the contract every rocke.benchmark.perf component speaks.

One record per (run, kernel, shape, config), composed from several primitives:

  run       : invocation metadata (run_id, arch, commit, timestamp, ...)
  kernel    : identity + launch config (name, op, shape, grid/block, dispatch_ref)
              - grid/block + dispatch_ref let a consumer cross-reference an ATT
                trace of the same kernel (the deep kernel-trace-analysis skill).
  wall      : wall-time primitive (ms_median, spread, tflops, gbs, % peak)   [GPU]
  counters  : PMU-counter primitive (rocprofv3): cycles/cache/waves/insts/stalls [GPU]
  resources : occupancy primitive (ELF notes, NO GPU): vgpr/agpr/sgpr/lds/occupancy
  derived   : busy_fraction, l2_hit_rate, ...
  captured_counters : which normalized counters this arch/run actually captured
  verify    : correctness (ok, max_abs_diff)

`counters`/`resources`/`derived` are nullable — a record may carry only wall
(profiler unavailable) or only resources (no-GPU occupancy check). Keep the schema
**additive-only** so old records stay readable. Stdlib only.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

SCHEMA_VERSION = "rocke.bench.measurement/v1"

# Comparison identity: a kernel is compared only against its own baseline on the
# same GPU and shape. Kernel-AGNOSTIC: (arch, kernel_name, shape-signature) - the
# shape signature is a generic serialization of whatever the shape dict holds
# (GEMM: M/N/K; conv: N/H/W/C/...; attention: batch/heads/seqlen/...), so no op is
# privileged. (config hash is an optional tiebreaker the caller may add.)
IDENTITY_KEYS = ("arch", "kernel_name", "shape")

# Primary regression metric is clock-invariant (cycles); wall time is the fallback
# when the profiler was unavailable.
PRIMARY_METRIC = "busy_cycles"      # from record["counters"]
FALLBACK_METRIC = "ms_median"       # from record["wall"]

# Static resource fields (from ELF notes; no GPU) — the occupancy primitive.
RESOURCE_KEYS = ("vgpr", "agpr", "sgpr", "lds_bytes", "occupancy")

# Fields that, when present, form the diagnostic panel a regression report shows
# (dynamic counters + static resources — both help localize a change).
PANEL_KEYS = ("busy_fraction", "l2_hit_rate", "waves", "wait_cycles", "occupancy", "lds_bytes")

_REQUIRED_TOP = ("schema", "run", "kernel", "wall")
_REQUIRED_RUN = ("run_id", "arch", "timestamp")
_REQUIRED_KERNEL = ("kernel_name", "op", "shape")


class SchemaError(ValueError):
    """Raised when a record does not satisfy the measurement schema."""


def validate(record: Mapping[str, Any]) -> None:
    """Raise SchemaError if `record` is missing required structure.

    Deliberately shallow: checks the load-bearing keys the pipeline relies on,
    not every optional field (counters/derived are nullable by design).
    """
    for k in _REQUIRED_TOP:
        if k not in record:
            raise SchemaError(f"missing top-level key: {k!r}")
    if record["schema"] != SCHEMA_VERSION:
        raise SchemaError(
            f"schema mismatch: {record['schema']!r} != {SCHEMA_VERSION!r}"
        )
    for k in _REQUIRED_RUN:
        if k not in record["run"]:
            raise SchemaError(f"missing run.{k}")
    for k in _REQUIRED_KERNEL:
        if k not in record["kernel"]:
            raise SchemaError(f"missing kernel.{k}")
    shape = record["kernel"]["shape"]
    if not isinstance(shape, Mapping):
        raise SchemaError("kernel.shape must be an object")


def shape_signature(shape: Mapping[str, Any]) -> str:
    """Generic, op-agnostic serialization of a shape dict (sorted keys).

    GEMM -> 'K=512,M=512,N=512'; conv/attention -> their own dims. No op privileged.
    """
    return ",".join(f"{k}={shape[k]}" for k in sorted(shape or {}))


def identity(record: Mapping[str, Any]) -> tuple:
    """The (arch, kernel_name, shape-signature) tuple a record is compared on."""
    run = record.get("run", {})
    kern = record.get("kernel", {})
    return (
        str(run.get("arch", "")),
        str(kern.get("kernel_name", "")),
        shape_signature(kern.get("shape", {})),
    )


def metric(record: Mapping[str, Any]) -> tuple[Optional[float], str]:
    """Return (value, which) — the primary cycle metric if present, else wall.

    `which` is "busy_cycles" or "ms_median" so callers/reports know which was used
    (never compare a cycles number against a wall number).
    """
    counters = record.get("counters") or {}
    val = counters.get(PRIMARY_METRIC)
    if val is not None:
        return float(val), PRIMARY_METRIC
    wall = record.get("wall") or {}
    val = wall.get(FALLBACK_METRIC)
    if val is not None:
        return float(val), FALLBACK_METRIC
    return None, ""


def captured(record: Mapping[str, Any]) -> Sequence[str]:
    """Which normalized counters this record actually captured (arch-dependent)."""
    return list(record.get("captured_counters") or [])
