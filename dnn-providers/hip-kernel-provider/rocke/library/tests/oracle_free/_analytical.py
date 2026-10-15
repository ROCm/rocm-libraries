# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free closed forms and diagnostics for the P3 attention probe.

There is intentionally no general attention implementation in this module.
Every expected value is derived directly from legal-key membership and the
uniform probability ``1 / |A(q)|`` forced by zero Q.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


_DTYPES = frozenset({"fp16", "bf16"})
_MASKS = frozenset({"causal", "window"})
P3_BASIS_OFFSETS = (0, 64, 128, 192)
P3_FOCUSED_ROWS = (0, 62, 63, 64, 127, 128, 255)


@dataclass(frozen=True)
class P3Case:
    """Logical dense-attention shape and semantics for one P3 case."""

    batch: int
    seqlen_q: int
    seqlen_k: int
    nhead_q: int
    nhead_k: int
    head_dim: int
    dtype: str
    mask: str
    window: int = 0

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "seqlen_q",
            "seqlen_k",
            "nhead_q",
            "nhead_k",
            "head_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.nhead_q % self.nhead_k:
            raise ValueError("nhead_q must be a multiple of nhead_k")
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
        if self.mask not in _MASKS:
            raise ValueError(f"mask must be one of {sorted(_MASKS)}")
        if self.mask == "window":
            if self.window <= 0:
                raise ValueError("window mask requires window > 0")
        elif self.window:
            raise ValueError("window must be zero unless mask='window'")

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        """Return the runner's logical ``[B, Sq, Hq, D]`` output shape."""
        return (self.batch, self.seqlen_q, self.nhead_q, self.head_dim)

    @property
    def key_shape(self) -> tuple[int, int, int, int]:
        """Return the runner's logical ``[B, Sk, Hkv, D]`` K/V shape."""
        return (self.batch, self.seqlen_k, self.nhead_k, self.head_dim)


P3_CAUSAL_FP16 = P3Case(1, 256, 256, 4, 4, 64, "fp16", "causal")
P3_WINDOW_BF16 = P3Case(1, 256, 256, 4, 4, 64, "bf16", "window", 64)


@dataclass(frozen=True)
class TolerancePolicy:
    """Fixed absolute tolerance for one P3 metric in both storage dtypes."""

    fp16: float
    bf16: float

    def for_dtype(self, dtype: str) -> float:
        """Return the predeclared absolute ceiling for ``dtype``."""
        if dtype not in _DTYPES:
            raise KeyError(dtype)
        return float(getattr(self, dtype))


# These fixed candidates include storage rounding and kernel arithmetic. They
# must be ratified, and tightened where stable, on matching gfx942 hardware.
P3_TOLERANCES: Mapping[str, TolerancePolicy] = MappingProxyType(
    {
        "element": TolerancePolicy(fp16=5e-4, bf16=4e-3),
        "row_mass": TolerancePolicy(fp16=5e-3, bf16=2e-2),
    }
)


@dataclass(frozen=True)
class Measurement:
    """Worst residual and its logical output coordinate."""

    maximum_error: float
    coordinate: tuple[int, int, int, int]
    expected: float
    observed: float
    invalid_count: int


@dataclass(frozen=True)
class P3Diagnostics:
    """Uniform-probability residuals and exact structural checks."""

    elements: Mapping[int, Measurement]
    row_mass_error: float
    zero_leakage_count: int
    invalid_count: int
    focused_row_errors: Mapping[int, float]


def p3_tolerance(metric: str, dtype: str) -> float:
    """Look up one fixed, never run-derived P3 absolute tolerance."""
    return P3_TOLERANCES[metric].for_dtype(dtype)


def legal_key_bounds(
    case: P3Case, query: int, *, rule: str = "correct"
) -> tuple[int, int]:
    """Return inclusive legal-key bounds, or ``(0, -1)`` for an empty set.

    ``rule`` exposes the two explicit off-by-one controls used by falsifiability
    tests. These formulas are independent of all kernel mask helpers.
    """
    if not 0 <= query < case.seqlen_q:
        raise IndexError(query)
    if rule not in {"correct", "causal_exclusive", "window_inclusive_lower"}:
        raise ValueError(f"unknown legal-key rule {rule!r}")

    upper = min(query, case.seqlen_k - 1)
    lower = 0
    if case.mask == "window":
        if rule == "window_inclusive_lower":
            lower = max(0, query - case.window)
        else:
            lower = max(0, query - case.window + 1)
    if case.mask == "causal" and rule == "causal_exclusive":
        upper = min(query - 1, case.seqlen_k - 1)
    return (lower, upper) if lower <= upper else (0, -1)


def legal_keys(case: P3Case, query: int, *, rule: str = "correct") -> tuple[int, ...]:
    """Enumerate the independently-derived legal key set for one query."""
    lower, upper = legal_key_bounds(case, query, rule=rule)
    return tuple(range(lower, upper + 1))


def _deterministic_k(case: P3Case) -> np.ndarray:
    """Construct finite, nonconstant, exactly representable K values."""
    batch = np.arange(case.batch, dtype=np.int64)[:, None, None, None]
    key = np.arange(case.seqlen_k, dtype=np.int64)[None, :, None, None]
    head = np.arange(case.nhead_k, dtype=np.int64)[None, None, :, None]
    dim = np.arange(case.head_dim, dtype=np.int64)[None, None, None, :]
    values = ((11 * batch + 3 * key + 5 * head + 7 * dim) % 17) - 8
    return (values.astype(np.float32) / 8.0).copy()


def p3_inputs(
    case: P3Case, basis_offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the P3 zero-Q, nonconstant-K, shifted-identity-V construction."""
    if basis_offset < 0:
        raise ValueError("basis_offset must be non-negative")
    q = np.zeros(case.output_shape, dtype=np.float32)
    k = _deterministic_k(case)
    v = np.zeros(case.key_shape, dtype=np.float32)
    for dim in range(case.head_dim):
        key = basis_offset + dim
        if key < case.seqlen_k:
            v[:, key, :, dim] = 1.0
    return q, k, v


def p3_expected(
    case: P3Case, basis_offset: int, *, rule: str = "correct"
) -> np.ndarray:
    """Compute P3 directly from legal membership and ``1 / |A(q)|``."""
    expected = np.zeros(case.output_shape, dtype=np.float64)
    for query in range(case.seqlen_q):
        lower, upper = legal_key_bounds(case, query, rule=rule)
        count = max(0, upper - lower + 1)
        if not count:
            continue
        first_dim = max(0, lower - basis_offset)
        stop_dim = min(case.head_dim, upper - basis_offset + 1)
        if first_dim < stop_dim:
            expected[:, query, :, first_dim:stop_dim] = 1.0 / count
    return expected


def _measurement(observed: np.ndarray, expected: np.ndarray) -> Measurement:
    """Reduce one tensor comparison without changing its logical coordinates."""
    observed = np.asarray(observed)
    expected = np.asarray(expected)
    if observed.shape != expected.shape or observed.ndim != 4:
        raise ValueError(
            f"observed/expected must share a 4-D shape, got "
            f"{observed.shape} and {expected.shape}"
        )
    finite = np.isfinite(observed) & np.isfinite(expected)
    invalid = ~finite
    residual = np.abs(observed.astype(np.float64) - expected.astype(np.float64))
    residual = np.where(finite, residual, np.inf)
    flat_index = int(np.argmax(residual))
    coordinate = tuple(int(x) for x in np.unravel_index(flat_index, observed.shape))
    return Measurement(
        maximum_error=float(residual[coordinate]),
        coordinate=coordinate,
        expected=float(expected[coordinate]),
        observed=float(observed[coordinate]),
        invalid_count=int(np.count_nonzero(invalid)),
    )


def _validate_basis_partition(
    case: P3Case, outputs: Mapping[int, np.ndarray], offsets: Sequence[int]
) -> None:
    """Require the supplied basis windows to expose every key exactly once."""
    if len(set(offsets)) != len(offsets) or set(outputs) != set(offsets):
        raise ValueError(f"P3 outputs must have the unique offsets {tuple(offsets)}")
    exposed = [
        offset + dim
        for offset in offsets
        for dim in range(case.head_dim)
        if offset + dim < case.seqlen_k
    ]
    if exposed != list(range(case.seqlen_k)):
        raise ValueError("P3 basis windows must expose every key exactly once")


def analyze_p3(
    case: P3Case,
    outputs: Mapping[int, np.ndarray],
    *,
    rule: str = "correct",
    offsets: Sequence[int] = P3_BASIS_OFFSETS,
    focused_rows: Sequence[int] = P3_FOCUSED_ROWS,
) -> P3Diagnostics:
    """Check all P3 probabilities, exact zero leakage, and combined row mass."""
    _validate_basis_partition(case, outputs, offsets)
    elements: dict[int, Measurement] = {}
    expected_by_offset: dict[int, np.ndarray] = {}
    mass = np.zeros(case.output_shape[:-1], dtype=np.float64)
    leakage_count = 0
    invalid_count = 0
    for offset in offsets:
        observed = np.asarray(outputs[offset])
        expected = p3_expected(case, offset, rule=rule)
        expected_by_offset[offset] = expected
        result = _measurement(observed, expected)
        elements[offset] = result
        invalid_count += result.invalid_count
        zero_mask = expected == 0.0
        leakage_count += int(np.count_nonzero(observed[zero_mask] != 0.0))
        mass += observed.astype(np.float64).sum(axis=-1)

    row_mass_error = (
        float(np.max(np.abs(mass - 1.0))) if np.isfinite(mass).all() else float("inf")
    )
    focus: dict[int, float] = {}
    for query in focused_rows:
        if not 0 <= query < case.seqlen_q:
            continue
        maximum = 0.0
        for offset in offsets:
            observed = np.asarray(outputs[offset])[:, query, :, :]
            expected = expected_by_offset[offset][:, query, :, :]
            if not np.isfinite(observed).all():
                maximum = float("inf")
                break
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(
                            observed.astype(np.float64) - expected.astype(np.float64)
                        )
                    )
                ),
            )
        focus[query] = maximum
    return P3Diagnostics(
        elements=MappingProxyType(elements),
        row_mass_error=row_mass_error,
        zero_leakage_count=leakage_count,
        invalid_count=invalid_count,
        focused_row_errors=MappingProxyType(focus),
    )


def round_to_storage(values: np.ndarray, dtype: str) -> np.ndarray:
    """Decode values after IEEE FP16 or round-to-nearest-even BF16 storage."""
    values = np.asarray(values, dtype=np.float32)
    if dtype == "fp16":
        return values.astype(np.float16).astype(np.float32)
    if dtype != "bf16":
        raise ValueError(dtype)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def format_measurement(measurement: Measurement) -> str:
    """Render maximum-residual failure details compactly."""
    batch, query, head, dim = measurement.coordinate
    return (
        f"max_abs={measurement.maximum_error:.6g} at "
        f"b:q:h:d={batch}:{query}:{head}:{dim}, "
        f"expected={measurement.expected:.9g}, observed={measurement.observed:.9g}, "
        f"invalid={measurement.invalid_count}"
    )
