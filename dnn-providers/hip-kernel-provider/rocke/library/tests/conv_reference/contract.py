# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Deterministic 2-D forward corpus and offline NumPy qualification oracle.

The metric is max absolute error divided by a frozen reference scale. CI never
computes the independent reference. Its scalar scale and error certificate are
bound to exact input and baseline-output digests by the reviewed manifest lock.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np

from reference_common.numeric import array_digest, decode, encode, max_abs_upper

SCHEMA_VERSION = 2
INPUT_GENERATOR = {
    "algorithm": "numpy-pcg64-uniform-f32-conv-v1",
    "seed": 0,
    "tensor_order": ["a", "b"],
    "quantization": "fp16-rne-or-bf16-rne",
}


@dataclass(frozen=True)
class Case:
    """One forward NHWC/KYXC convolution; weights use C/groups channels."""

    name: str
    dtype: str
    N: int = 1
    Hi: int = 8
    Wi: int = 8
    C: int = 32
    K: int = 32
    Y: int = 3
    X: int = 3
    sH: int = 1
    sW: int = 1
    pH: int = 1
    pW: int = 1
    dH: int = 1
    dW: int = 1
    groups: int = 1

    def __post_init__(self) -> None:
        if self.dtype not in ("fp16", "bf16"):
            raise ValueError("unsupported convolution dtype")
        if not self.name or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in self.name
        ):
            raise ValueError("invalid convolution case name")
        for key, value in asdict(self).items():
            if key in ("name", "dtype"):
                continue
            if type(value) is not int or value < (0 if key in ("pH", "pW") else 1):
                raise ValueError(f"invalid convolution dimension: {key}")
        if self.C % self.groups or self.K % self.groups or min(self.output_shape) <= 0:
            raise ValueError("invalid grouped convolution geometry")

    @property
    def id(self) -> str:
        return f"fwd-{self.name}-{self.dtype}"

    @property
    def tolerance(self) -> float:
        return 0.05

    @property
    def margin(self) -> float:
        return 0.0025

    @property
    def input_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "a": (self.N, self.Hi, self.Wi, self.C),
            "b": (self.K, self.Y, self.X, self.C // self.groups),
        }

    @property
    def output_shape(self) -> tuple[int, ...]:
        ho = (self.Hi + 2 * self.pH - self.dH * (self.Y - 1) - 1) // self.sH + 1
        wo = (self.Wi + 2 * self.pW - self.dW * (self.X - 1) - 1) // self.sW + 1
        return self.N, ho, wo, self.K

    def problem_fields(self) -> dict:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in ("name", "dtype")
        }


def make_inputs(case: Case) -> dict[str, np.ndarray]:
    """Fixed PCG64 uniform corpus, cast to f32 before storage quantization."""
    rng = np.random.Generator(np.random.PCG64(INPUT_GENERATOR["seed"]))
    return {
        name: encode(rng.uniform(-1, 1, shape).astype(np.float32), case.dtype)
        for name, shape in case.input_shapes.items()
    }


def checked_inputs(case: Case, digests: dict[str, str]) -> dict[str, np.ndarray]:
    arrays = make_inputs(case)
    if {name: array_digest(value) for name, value in arrays.items()} != digests:
        raise ValueError(f"generated convolution input digest mismatch: {case.id}")
    return arrays


def independent_reference(case: Case, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Direct grouped cross-correlation in float64 on exact quantized inputs.

    No rocKE geometry, indexing, or GPU implementation is used by this oracle.
    As in the existing forward test, the result is rounded to output storage
    before comparison. This defines a new independently qualified input corpus;
    it does not claim bitwise identity with Torch's float32 accumulation.
    """
    if set(inputs) != set(case.input_shapes):
        raise ValueError("incorrect convolution operands")
    arrays = {}
    for name, shape in case.input_shapes.items():
        value = decode(inputs[name], case.dtype).astype(np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid convolution operand: {name}")
        arrays[name] = value
    a, b = arrays["a"], arrays["b"]
    out = np.zeros(case.output_shape, dtype=np.float64)
    cpg, kpg = case.C // case.groups, case.K // case.groups
    for group in range(case.groups):
        cs = slice(group * cpg, (group + 1) * cpg)
        ks = slice(group * kpg, (group + 1) * kpg)
        for ho in range(out.shape[1]):
            for wo in range(out.shape[2]):
                for y in range(case.Y):
                    hi = ho * case.sH - case.pH + y * case.dH
                    if not 0 <= hi < case.Hi:
                        continue
                    for x in range(case.X):
                        wi = wo * case.sW - case.pW + x * case.dW
                        if 0 <= wi < case.Wi:
                            out[:, ho, wo, ks] += a[:, hi, wi, cs] @ b[ks, y, x, :].T
    # The reference rounding recipe is explicit and versioned in the manifest:
    # float64 oracle -> float32 -> fp16/bf16 RNE -> float64 represented output.
    return decode(encode(out, case.dtype), case.dtype).astype(np.float64)


def reference_scale(reference: np.ndarray) -> float:
    if not reference.size or not np.isfinite(reference).all():
        raise ValueError("invalid independent reference")
    return max(float(np.max(np.abs(reference))), 1.0)


def normalized_distance(left: np.ndarray, right: np.ndarray, scale: float) -> float:
    """Round division upward; the denominator is the offline reference scale."""
    if not math.isfinite(scale) or scale < 1:
        raise ValueError("invalid convolution reference scale")
    distance = max_abs_upper(left, right)
    if not distance:
        return 0.0
    exact = Fraction(distance) / Fraction(scale)
    result = float(exact)
    if Fraction(result) < exact:
        result = math.nextafter(result, math.inf)
    return result
