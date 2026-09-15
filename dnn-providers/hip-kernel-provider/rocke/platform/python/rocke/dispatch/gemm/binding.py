# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side problem bindings for the RCR UniversalGemm families.

One binding serves fp16 and bf16: the two families share an args signature
(``A, B, C, M, N, K``) and an RCR reference (``C = A @ B.T``), differing only
in element encoding. Keeping them together means a change to the calling
convention cannot land for one dtype and miss the other.

The launch geometry is taken from the :class:`DispatchResult` rather than
recomputed from tile fields. The legacy manifest runner re-derives it from
``block_m`` / ``block_n`` / ``grid_order``, which is the same arithmetic
written a second time and free to drift from the candidate's ``grid``.
"""

from __future__ import annotations

import struct
from typing import Any, Tuple

from ..core import ProblemBinding

# The legacy GEMM manifest runner's corpus, reused verbatim so a binding and
# the adapter it replaces verify against the same numbers.
_SEED = 0xC0FFEE

# Relative tolerance per output encoding. fp16 keeps the legacy 1e-2. bf16 has
# 8 significand bits (eps 2^-8 ~= 3.9e-3), so output rounding alone can consume
# most of a 1e-2 budget on unlucky values; 5e-2 leaves headroom without hiding
# a real miscompute, which shows up as a wrong magnitude rather than a last-bit
# difference.
_TOLERANCE = {"fp16": 1e-2, "bf16": 5e-2}


def _bf16_from_f32(np, x):
    """Round-to-nearest-even f32 -> bf16, carried as uint16."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF).astype(np.uint16)


def _f32_from_bf16(np, u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def gemm_rcr_binding(result: Any, verify: bool, *, dtype: str) -> ProblemBinding:
    """Bind an RCR UniversalGemm selection to a runnable problem."""
    from ...runtime.host_buffers import as_u8_buffer, nbytes, require_numpy

    np = require_numpy()
    if dtype not in _TOLERANCE:
        raise ValueError(f"unsupported gemm binding dtype {dtype!r}")

    req = result.request
    M, N, K = int(req.M), int(req.N), int(req.K)

    # Small integers are exact in both encodings, so the reference is limited
    # by the output rounding and the device's accumulate order, not by the
    # inputs.
    rng = np.random.default_rng(_SEED)
    a_int = rng.integers(-5, 6, size=(M, K), dtype=np.int16)
    b_int = rng.integers(-5, 6, size=(N, K), dtype=np.int16)

    if dtype == "fp16":
        a_host = a_int.astype(np.float16)
        b_host = b_int.astype(np.float16)
        c_host = np.empty((M, N), dtype=np.float16)

        def decode(buf):
            return buf.astype(np.float32)

    else:
        a_host = _bf16_from_f32(np, a_int.astype(np.float32))
        b_host = _bf16_from_f32(np, b_int.astype(np.float32))
        c_host = np.empty((M, N), dtype=np.uint16)

        def decode(buf):
            return _f32_from_bf16(np, buf)

    def make_args(rt: Any) -> Tuple[bytes, Tuple[int, ...]]:
        a_dev = rt.alloc(nbytes(a_host))
        b_dev = rt.alloc(nbytes(b_host))
        c_dev = rt.alloc(nbytes(c_host))
        rt.memcpy_h2d(a_dev, as_u8_buffer(a_host), nbytes(a_host))
        rt.memcpy_h2d(b_dev, as_u8_buffer(b_host), nbytes(b_host))
        rt.memset(c_dev, 0, nbytes(c_host))
        return (
            struct.pack("<QQQiii", a_dev, b_dev, c_dev, M, N, K),
            (a_dev, b_dev, c_dev),
        )

    def check(rt: Any, ptrs: Tuple[int, ...]) -> Tuple[float, int, int]:
        if not verify:
            return 0.0, 0, c_host.size
        rt.memcpy_d2h(as_u8_buffer(c_host), ptrs[2], nbytes(c_host))
        ref = a_int.astype(np.float32) @ b_int.astype(np.float32).T
        got = decode(c_host)
        tol = _TOLERANCE[dtype]
        err = np.abs(got - ref)
        bad = err > tol + tol * np.abs(ref)
        return float(err.max()), int(np.count_nonzero(bad)), c_host.size

    return ProblemBinding(
        grid=tuple(int(x) for x in result.grid),
        block=tuple(int(x) for x in result.block),
        make_args=make_args,
        check=check,
        flop=2.0 * M * N * K,
        bytes_moved=2.0 * (M * K + N * K + M * N),
    )
