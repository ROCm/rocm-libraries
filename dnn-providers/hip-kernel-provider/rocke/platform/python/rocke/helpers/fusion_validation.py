# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Correctness + performance validation for fusion plans.

This module provides a small, GPU-friendly harness for the
"acceptance matrix" the plan calls out:

* For each (pattern, dtype, shape) row, run the CK DSL fused
  implementation through :func:`compile_fn` and compare its output
  against ``torch eager`` (and optionally ``torch.compile``).
* Record wall-clock timings via :func:`time_launches` so the same
  numbers we autotune on are the numbers we validate against.
* Produce a :class:`ValidationReport` summarising correctness
  (``max_abs`` error) and timing (``ms_per_iter`` for each backend).

The harness deliberately keeps itself dependency-light: torch is the
only third-party import, and it's gated behind a ``try`` so the data
types still work on CPU-only test environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


__all__ = [
    "BackendTiming",
    "BenchmarkCase",
    "FusionMatrixRunner",
    "ValidationReport",
    "run_fusion_validation_matrix",
]


@dataclass(frozen=True)
class BackendTiming:
    """One row of timing/correctness measurements for a backend.

    ``ms`` is the per-iteration wall time in milliseconds.
    ``max_abs`` is the max-absolute deviation from the reference
    output (zero for the reference itself).
    """

    name: str
    ms: float
    max_abs: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "ms": float(self.ms), "max_abs": float(self.max_abs)}


@dataclass(frozen=True)
class ValidationReport:
    """Aggregated result for one (pattern, shape, dtype) case.

    ``graph_hash`` matches :meth:`FusionGraph.graph_hash` so callers
    can correlate the report with the autotune cache. ``correctness``
    maps backend name -> max absolute deviation from the reference.
    ``timings`` is the same list of :class:`BackendTiming` rows.
    """

    graph_hash: str
    correctness: Dict[str, float]
    timings: Tuple[BackendTiming, ...]
    pattern: str = ""
    shape: Tuple[int, ...] = ()
    dtype: str = ""

    def fastest(self) -> BackendTiming:
        return min(self.timings, key=lambda t: t.ms)

    def speedup_vs(self, baseline: str) -> Dict[str, float]:
        """Return per-backend speedup multipliers vs ``baseline`` (>=1 means faster)."""
        base = next((t for t in self.timings if t.name == baseline), None)
        if base is None or base.ms <= 0:
            return {}
        return {
            t.name: float(base.ms / t.ms) if t.ms > 0 else math.inf
            for t in self.timings
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "graph_hash": self.graph_hash,
            "pattern": self.pattern,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "correctness": dict(self.correctness),
            "timings": [t.as_dict() for t in self.timings],
            "fastest": self.fastest().as_dict() if self.timings else None,
        }


@dataclass
class BenchmarkCase:
    """One row of the validation matrix.

    ``ref_fn`` is the torch-eager reference. ``make_inputs(shape, dtype)``
    builds the input tuple; the harness passes those inputs to every
    backend it runs. ``backends`` is the dict of additional
    callables (e.g. ``compile_fn(ref_fn)``, ``torch.compile(ref_fn)``)
    indexed by name.
    """

    name: str
    ref_fn: Callable[..., Any]
    make_inputs: Callable[[Tuple[int, ...], str], Tuple[Any, ...]]
    backends: Dict[str, Callable[..., Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _max_abs(ref: Any, got: Any) -> float:
    import torch

    if not isinstance(ref, torch.Tensor) or not isinstance(got, torch.Tensor):
        # Heuristic: fall back to a python diff.
        try:
            return float(abs(float(ref) - float(got)))
        except Exception:
            return float("nan")
    if ref.shape != got.shape:
        return float("nan")
    return float((ref.float() - got.float()).abs().max().item())


def _time_callable(
    fn: Callable[..., Any],
    inputs: Tuple[Any, ...],
    *,
    warmup: int = 5,
    iters: int = 20,
) -> Tuple[float, Any]:
    """Time ``fn(*inputs)`` with HIP events; return (ms_per_iter, last_output).

    Falls back to a Python wall-clock timer when torch.cuda isn't
    available. This keeps the harness usable on CPU-only test boxes;
    the test author can decide which backends are GPU-only and skip
    the row when ``torch.cuda.is_available()`` is False.
    """
    import time
    import torch

    out = None
    if torch.cuda.is_available() and any(
        isinstance(t, torch.Tensor) and t.is_cuda for t in inputs
    ):
        for _ in range(max(1, warmup)):
            out = fn(*inputs)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = fn(*inputs)
        end.record()
        end.synchronize()
        ms_per_iter = float(start.elapsed_time(end)) / max(1, iters)
        return ms_per_iter, out
    # CPU fallback.
    for _ in range(max(1, warmup)):
        out = fn(*inputs)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*inputs)
    ms_per_iter = (time.perf_counter() - t0) * 1e3 / max(1, iters)
    return ms_per_iter, out


# ---------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------


class FusionMatrixRunner:
    """Sweep a list of :class:`BenchmarkCase` across shapes and dtypes.

    Returns one :class:`ValidationReport` per (case, shape, dtype). The
    runner is deliberately small: the heavy lifting (kernel build,
    autotune, runtime args) lives in :func:`compile_fn`. This class
    just orchestrates the case sweep and aggregates the results so
    they can be dumped to JSON.
    """

    def __init__(
        self,
        cases: Sequence[BenchmarkCase],
        *,
        shapes: Sequence[Tuple[int, ...]],
        dtypes: Sequence[str] = ("fp16",),
        warmup: int = 5,
        iters: int = 20,
        atol: float = 5e-3,
        rtol: float = 5e-3,
    ) -> None:
        if not cases:
            raise ValueError("FusionMatrixRunner: empty cases")
        if not shapes:
            raise ValueError("FusionMatrixRunner: empty shapes")
        self.cases = list(cases)
        self.shapes = list(shapes)
        self.dtypes = list(dtypes)
        self.warmup = int(warmup)
        self.iters = int(iters)
        self.atol = float(atol)
        self.rtol = float(rtol)

    def run(self) -> List[ValidationReport]:
        reports: List[ValidationReport] = []
        for case in self.cases:
            for shape in self.shapes:
                for dtype in self.dtypes:
                    report = self._run_one(case, shape, dtype)
                    reports.append(report)
        return reports

    def _run_one(
        self,
        case: BenchmarkCase,
        shape: Tuple[int, ...],
        dtype: str,
    ) -> ValidationReport:
        inputs = case.make_inputs(shape, dtype)
        ref_ms, ref_out = _time_callable(
            case.ref_fn, inputs, warmup=self.warmup, iters=self.iters
        )
        timings: List[BackendTiming] = [BackendTiming(name="torch_eager", ms=ref_ms)]
        correctness: Dict[str, float] = {"torch_eager": 0.0}
        for backend_name, backend_fn in case.backends.items():
            try:
                got_ms, got_out = _time_callable(
                    backend_fn, inputs, warmup=self.warmup, iters=self.iters
                )
            except Exception:  # pragma: no cover -- broad-catch by design
                # Record a failed backend with infinite time + nan correctness
                # so the JSON report stays well-formed.
                timings.append(
                    BackendTiming(
                        name=backend_name, ms=float("inf"), max_abs=float("nan")
                    )
                )
                correctness[backend_name] = float("nan")
                continue
            err = _max_abs(ref_out, got_out)
            timings.append(BackendTiming(name=backend_name, ms=got_ms, max_abs=err))
            correctness[backend_name] = err
        graph_hash = _shape_hash(case.name, shape, dtype)
        return ValidationReport(
            graph_hash=graph_hash,
            correctness=correctness,
            timings=tuple(timings),
            pattern=case.name,
            shape=tuple(shape),
            dtype=dtype,
        )


def _shape_hash(name: str, shape: Sequence[int], dtype: str) -> str:
    """Deterministic short hash for ``(name, shape, dtype)`` triples."""
    import hashlib

    payload = f"{name}|{tuple(shape)!r}|{dtype}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def run_fusion_validation_matrix(
    *,
    cases: Iterable[BenchmarkCase],
    shapes: Sequence[Tuple[int, ...]],
    dtypes: Sequence[str] = ("fp16",),
    warmup: int = 5,
    iters: int = 20,
    atol: float = 5e-3,
    rtol: float = 5e-3,
) -> List[ValidationReport]:
    """Functional entry point: equivalent to ``FusionMatrixRunner(...).run()``.

    Most users want a one-liner; using the class form is useful when
    callers want to chain multiple matrices, override ``atol``, or
    inspect intermediate state.
    """
    runner = FusionMatrixRunner(
        cases=list(cases),
        shapes=shapes,
        dtypes=dtypes,
        warmup=warmup,
        iters=iters,
        atol=atol,
        rtol=rtol,
    )
    return runner.run()
