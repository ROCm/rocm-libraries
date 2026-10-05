# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The fixed SDPA corpus and its absolute-maximum error contract.

Qualification uses float64 NumPy SDPA on the exact fp16/bf16 input values.
Replay generates old answers on the GPU. Every accepted comparison satisfies
``old_error_bound + current_distance + margin <= original_tolerance``.
This is a per-input guarantee relative to the recorded independent reference,
not a bound on unobserved inputs or on the reference's mathematical error.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2
INPUT_GENERATOR = {
    "algorithm": "numpy-pcg64-normal-f32-v1",
    "seed": 0,
    "tensor_order": ["q", "k", "v"],
    "quantization": "fp16-rne-or-bf16-rne",
}


@dataclass(frozen=True)
class Case:
    """One dense SDPA parameterization shared by architecture-specific cohorts."""

    dtype: str
    head_dim: int
    query_heads: int
    kv_heads: int
    persistent: bool
    causal: bool
    batch: int = 1
    sequence_length: int = 512

    @property
    def id(self) -> str:
        grid = "persistent" if self.persistent else "default"
        mask = "causal" if self.causal else "full"
        return (
            f"{self.dtype}-d{self.head_dim}-h{self.query_heads}"
            f"-kv{self.kv_heads}-{grid}-{mask}"
        )

    @property
    def tolerance(self) -> float:
        return 0.02 if self.dtype == "fp16" else 0.04

    @property
    def margin(self) -> float:
        # Reserve 5% of the existing tolerance in addition to conservative
        # evaluation of both measured distances. Never widen that tolerance.
        return 0.001 if self.dtype == "fp16" else 0.002

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.batch, self.sequence_length, self.query_heads, self.head_dim

    @property
    def scale(self) -> float:
        # The launch ABI stores scale as f32. Qualify that same scalar value.
        return float(np.float32(1.0 / math.sqrt(self.head_dim)))


def encode(values: np.ndarray, dtype: str) -> np.ndarray:
    """Store fp16 values or round-to-nearest-even bf16 bits without Torch."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    if dtype == "fp16":
        return values.astype("<f2")
    if dtype != "bf16":
        raise ValueError(f"unsupported SDPA dtype: {dtype}")
    bits = values.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return ((bits + bias) >> 16).astype("<u2")


def decode(values: np.ndarray, dtype: str) -> np.ndarray:
    """Decode tensor storage, checking its dtype rather than reinterpreting it."""
    expected = np.dtype("<f2" if dtype == "fp16" else "<u2")
    if dtype not in ("fp16", "bf16") or values.dtype != expected:
        raise ValueError(f"invalid {dtype} tensor storage: {values.dtype}")
    if dtype == "fp16":
        return values.astype(np.float32)
    return (values.astype(np.uint32) << 16).view(np.float32)


def make_inputs(case: Case) -> dict[str, np.ndarray]:
    """Regenerate the versioned corpus in Q/K/V order using a fresh PCG64 stream.

    Keep this algorithm fixed. NumPy distribution implementation drift is
    detected by the qualified per-tensor digests before GPU execution; a seed
    alone is not treated as a cross-version reproducibility guarantee.
    """
    rng = np.random.Generator(np.random.PCG64(0))
    kv_shape = (*case.shape[:2], case.kv_heads, case.head_dim)
    return {
        name: encode(rng.standard_normal(shape, dtype=np.float32), case.dtype)
        for name, shape in (("q", case.shape), ("k", kv_shape), ("v", kv_shape))
    }


def checked_inputs(case: Case, digests: dict[str, str]) -> dict[str, np.ndarray]:
    """Require generated input bytes to match the independently qualified corpus."""
    arrays = make_inputs(case)
    if {name: array_digest(array) for name, array in arrays.items()} != digests:
        raise ValueError(
            f"generated SDPA input digest mismatch: {case.id}; "
            "the generator or NumPy implementation differs from qualification"
        )
    return arrays


def independent_reference(case: Case, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate stable SDPA in float64, expanding GQA heads before attention.

    This deliberately uses NumPy matmul/exp, independently of rocKE's GPU
    implementation. Inputs have already been quantized to the declared dtype.
    The result is kept in float64, including during qualification.
    """
    q, k, v = (
        decode(inputs[name], case.dtype).astype(np.float64).transpose(0, 2, 1, 3)
        for name in ("q", "k", "v")
    )
    repetitions = case.query_heads // case.kv_heads
    k = np.repeat(k, repetitions, axis=1)
    v = np.repeat(v, repetitions, axis=1)
    scores = (q @ k.swapaxes(-1, -2)) * case.scale
    if case.causal:
        masked = np.triu(np.ones(scores.shape[-2:], dtype=bool), k=1)
        scores[..., masked] = -np.inf
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return (probabilities @ v).transpose(0, 2, 1, 3)


def max_abs_upper(left: np.ndarray, right: np.ndarray) -> float:
    """Bound max(abs(left-right)) for the represented finite floating values.

    Both operands are promoted before subtraction. Rounding the largest f64
    difference upward by one representable value encloses subtraction's
    round-to-nearest error; max/abs introduce no further rounding. An exact
    zero stays zero. No reductions, relative denominators, or squared norms
    are silently substituted for the existing absolute maximum norm.
    """
    if left.shape != right.shape or not left.size:
        raise ValueError(f"incompatible or empty outputs: {left.shape}, {right.shape}")
    left, right = left.astype(np.float64), right.astype(np.float64)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("SDPA comparison contains non-finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        distance = float(np.max(np.abs(left - right)))
    if not math.isfinite(distance):
        raise ValueError("SDPA distance overflowed")
    return math.nextafter(distance, math.inf) if distance else 0.0


@dataclass(frozen=True)
class ErrorBudget:
    """A conservative triangle-inequality budget with strictly positive margin."""

    tolerance: float
    baseline_error_bound: float
    margin: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) for x in asdict(self).values()):
            raise ValueError("non-finite SDPA error budget")
        if self.baseline_error_bound < 0 or self.margin <= 0 or self.tolerance <= 0:
            raise ValueError("invalid SDPA error budget")
        if self.remaining <= 0:
            raise ValueError("pinned SDPA reference leaves no comparison budget")

    @property
    def remaining(self) -> Fraction:
        return (
            Fraction(self.tolerance)
            - Fraction(self.baseline_error_bound)
            - Fraction(self.margin)
        )

    @property
    def comparison_limit(self) -> float:
        # Fraction avoids cancellation/rounding while allocating the budget.
        # Round downward so a binary float threshold cannot overdraw it.
        return math.nextafter(float(self.remaining), -math.inf)

    def check(self, distance: float) -> None:
        """Reject a comparison that does not certify the original tolerance."""
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("invalid SDPA comparison distance")
        if distance > self.comparison_limit:
            raise AssertionError(
                f"SDPA max_abs={distance:.9g} exceeds remaining limit "
                f"{self.comparison_limit:.9g}; baseline_bound="
                f"{self.baseline_error_bound:.9g}, margin={self.margin:.9g}, "
                f"original_tolerance={self.tolerance:.9g}"
            )


def array_digest(array: np.ndarray) -> str:
    """Hash the logical tensor contract and contiguous little-endian bytes."""
    array = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    header = json.dumps(
        {"shape": array.shape, "dtype": array.dtype.str}, sort_keys=True
    ).encode()
    return hashlib.sha256(header + b"\n" + array.tobytes()).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_digests(root: Path) -> dict[str, str]:
    """Describe a source/artifact tree without interpreter-generated caches."""
    return {
        path.relative_to(root).as_posix(): file_digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
