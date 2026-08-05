# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side problem binding for the RCR block-scale fp8 GEMM family.

Separate from :mod:`rocke.dispatch.gemm.binding` because the calling convention
is genuinely different, not just the element encoding: this family takes two
scale pointers between B and C, and its output encoding is a spec field rather
than the input dtype. Folding it in would put a signature branch in a module
whose whole point is that one args layout serves both of its dtypes.

Operands follow the RCR layout ``torch._scaled_mm`` requires -- A row-major
``(M, K)``, B the weight matrix row-major ``(N, K)`` -- so a binding built here
is callable with the buffers a framework already holds, without a transpose.

Encodings are hand-rolled on top of numpy rather than taken from ``ml_dtypes``,
for the reason :mod:`rocke.runtime.host_buffers` gives: this path stays free of
both torch and a low-bit dtype dependency.
"""

from __future__ import annotations

import struct
from typing import Any, Tuple

from ..core import ProblemBinding
from .binding import f32_from_bf16

_SEED = 0xC0FFEE

# Output-encoding tolerance, matching the reasoning in the fp16/bf16 binding:
# bf16 carries 8 significand bits, so a K-deep accumulation spends most of a
# tighter budget on output rounding alone. A miscompute shows up as a wrong
# magnitude, not a last-bit difference.
_TOLERANCE = {"f32": 1e-3, "bf16": 5e-2, "f16": 1e-2}

# Scales the reference applies. Deliberately neither 1.0 nor equal to each
# other: a kernel that dropped one scale, or applied one twice, would still
# pass against unit scales.
_A_SCALE = 0.75
_B_SCALE = 1.25

# Raw bytes for the corpus magnitudes below, written out rather than computed so
# they can be read against the format definitions: e4m3 is s.eeee.mmm with
# exponent bias 7, e5m2 is s.eeeee.mm with bias 15.
_LOW_BIT_CODES = {
    "fp8e4m3": {0.0: 0x00, 0.5: 0x30, 1.0: 0x38, 1.5: 0x3C, 2.0: 0x40},
    "bf8e5m2": {0.0: 0x00, 0.5: 0x38, 1.0: 0x3C, 1.5: 0x3E, 2.0: 0x40},
}


def _encode_low_bit(np, values, mantissa: str):
    """Encode an f32 array of corpus magnitudes to raw fp8 / bf8 bytes.

    Refuses anything outside the table instead of silently emitting zeros, so
    widening the corpus without extending the table is a loud failure.
    """
    try:
        codes = _LOW_BIT_CODES[mantissa]
    except KeyError:
        raise ValueError(f"no host encoding for mantissa {mantissa!r}") from None
    out = np.zeros(values.shape, dtype=np.uint8)
    covered = np.zeros(values.shape, dtype=bool)
    for magnitude, code in codes.items():
        for signed, byte in ((magnitude, code), (-magnitude, code | 0x80)):
            hit = values == signed
            out[hit] = byte
            covered |= hit
    if not covered.all():
        missing = sorted(set(np.unique(values[~covered]).tolist()))
        raise ValueError(
            f"{mantissa} host encoding has no entry for {missing}; "
            f"extend _LOW_BIT_CODES or keep the corpus to {sorted(codes)}"
        )
    return out


def block_scale_gemm_fp8_binding(result: Any, verify: bool) -> ProblemBinding:
    """Bind an RCR block-scale fp8 selection to a runnable problem."""
    from ...runtime.host_buffers import as_u8_buffer, nbytes, require_numpy

    np = require_numpy()

    req = result.request
    spec = result.spec
    M, N, K = int(req.M), int(req.N), int(req.K)
    if spec.dtype_c not in _TOLERANCE:
        raise ValueError(f"unsupported fp8 gemm binding dtype_c {spec.dtype_c!r}")

    # Half-integer magnitudes are exact in both e4m3 and e5m2, so the reference
    # is limited by the output encoding and the device's accumulate order rather
    # than by input quantisation.
    rng = np.random.default_rng(_SEED)
    a_f32 = rng.integers(-4, 5, size=(M, K)).astype(np.float32) * 0.5
    b_f32 = rng.integers(-4, 5, size=(N, K)).astype(np.float32) * 0.5
    a_host = _encode_low_bit(np, a_f32, spec.mantissa_dtype)
    b_host = _encode_low_bit(np, b_f32, spec.mantissa_dtype)
    a_scale = np.array([_A_SCALE], dtype=np.float32)
    b_scale = np.array([_B_SCALE], dtype=np.float32)

    if spec.dtype_c == "f32":
        c_host = np.zeros((M, N), dtype=np.float32)

        def decode(buf):
            return buf.astype(np.float32)

    elif spec.dtype_c == "f16":
        c_host = np.zeros((M, N), dtype=np.float16)

        def decode(buf):
            return buf.astype(np.float32)

    else:
        c_host = np.zeros((M, N), dtype=np.uint16)

        def decode(buf):
            return f32_from_bf16(np, buf)

    def make_args(rt: Any) -> Tuple[bytes, Tuple[int, ...]]:
        a_dev = rt.alloc(nbytes(a_host))
        b_dev = rt.alloc(nbytes(b_host))
        as_dev = rt.alloc(nbytes(a_scale))
        bs_dev = rt.alloc(nbytes(b_scale))
        c_dev = rt.alloc(nbytes(c_host))
        rt.memcpy_h2d(a_dev, as_u8_buffer(a_host), nbytes(a_host))
        rt.memcpy_h2d(b_dev, as_u8_buffer(b_host), nbytes(b_host))
        rt.memcpy_h2d(as_dev, as_u8_buffer(a_scale), nbytes(a_scale))
        rt.memcpy_h2d(bs_dev, as_u8_buffer(b_scale), nbytes(b_scale))
        rt.memset(c_dev, 0, nbytes(c_host))
        return (
            struct.pack("<QQQQQiii", a_dev, b_dev, as_dev, bs_dev, c_dev, M, N, K),
            (a_dev, b_dev, as_dev, bs_dev, c_dev),
        )

    def check(rt: Any, ptrs: Tuple[int, ...]) -> Tuple[float, int, int]:
        if not verify:
            return 0.0, 0, c_host.size
        rt.memcpy_d2h(as_u8_buffer(c_host), ptrs[4], nbytes(c_host))
        ref = (a_f32 @ b_f32.T) * (_A_SCALE * _B_SCALE)
        got = decode(c_host)
        tol = _TOLERANCE[spec.dtype_c]
        err = np.abs(got - ref)
        bad = err > tol + tol * np.abs(ref)
        return float(err.max()), int(np.count_nonzero(bad)), c_host.size

    # A and B are one byte per element; C is the output encoding. The two scale
    # floats do not move the needle and stay out of the denominator.
    out_bytes = 4 if spec.dtype_c == "f32" else 2
    return ProblemBinding(
        grid=tuple(int(x) for x in result.grid),
        block=tuple(int(x) for x in result.block),
        make_args=make_args,
        check=check,
        flop=2.0 * M * N * K,
        bytes_moved=float(M * K + N * K + M * N * out_bytes),
    )
