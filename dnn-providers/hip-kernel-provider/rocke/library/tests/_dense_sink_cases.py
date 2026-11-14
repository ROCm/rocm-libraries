# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Sink fixtures shared by the gfx950 dense fp32-sink tests.

The exact case: q = k = 0 and v = 1, so a row with c visible keys has
LSE = log(c + e^s) and O = c / (c + e^s). Sq 256 / Skv 64 / window 64 leaves rows
127..255 without keys (LSE = s, O = 0). Numpy only, so the CPU lane proves every
control breaks the bound the GPU lane asserts."""

from __future__ import annotations

import numpy as np

EXACT_SQ, EXACT_SKV, EXACT_WINDOW = 256, 64, 64
# Not representable in bf16 or fp16 (test_attention_dense_sink_dtype checks it).
EXACT_SINKS = np.array([3.01, -2.37, 30.03, 0.7], dtype=np.float32)
# Absolute LSE bound for the exact case on the fp32-sink path.
EXACT_LSE_ATOL = 4e-5


def exact_counts() -> np.ndarray:
    """Visible keys per query row of the exact case."""
    rows = np.arange(EXACT_SQ)
    upper = np.minimum(rows + 1, EXACT_SKV)
    lower = np.clip(rows - (EXACT_WINDOW - 1), 0, None)
    return np.clip(upper - lower, 0, None)


def exact_lse(sinks: np.ndarray) -> np.ndarray:
    """[Hq, Sq] natural-log LSE of the exact case, float64."""
    counts = exact_counts().astype(np.float64)
    return np.log(counts[None, :] + np.exp(sinks.astype(np.float64))[:, None])


def exact_out(sinks: np.ndarray) -> np.ndarray:
    """[Hq, Sq] output value of the exact case (every head-dim element), float64."""
    counts = exact_counts().astype(np.float64)
    return counts[None, :] / (
        counts[None, :] + np.exp(sinks.astype(np.float64))[:, None]
    )


def round_bf16(x: np.ndarray) -> np.ndarray:
    """Round to nearest even bf16, returned as float32."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return r.astype(np.uint32).view(np.float32)


def round_fp16(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


ROUND = {"bf16": round_bf16, "fp16": round_fp16}


def two_byte_read(x: np.ndarray, dtype: str) -> np.ndarray:
    """What a 2-byte load at index h of the float32 buffer ``x`` returns: its h-th
    little-endian 16-bit half, read as ``dtype``."""
    halves = np.ascontiguousarray(x, dtype=np.float32).view(np.uint16)[: x.size]
    if dtype == "bf16":
        return (halves.astype(np.uint32) << 16).view(np.float32)
    return halves.view(np.float16).astype(np.float32)
