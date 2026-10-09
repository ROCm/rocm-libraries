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

import math
from dataclasses import dataclass

import numpy as np

from reference_common.numeric import array_digest, decode, encode

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
