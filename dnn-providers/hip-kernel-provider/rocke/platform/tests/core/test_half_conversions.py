# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The f32 <-> f16 and f32 <-> bf16 conversions on IEEE edge values, on the GPU.

``trunc_f32_to_f16`` / ``trunc_f32_to_bf16`` lower to ``fptrunc`` and must round
to nearest even under the kernel's floating-point mode (round to nearest even,
subnormals kept, FP16 overflow clear; see ``test_dtypes.py``), so:

* a value past the largest finite half becomes +-inf, and so does the tie
  exactly halfway to the next power of two (the largest finite value has an odd
  significand, so the even neighbour is the infinity);
* a tie rounds to the neighbour with the even significand, also among the
  subnormals, and a value below half the smallest subnormal rounds to a signed
  zero;
* NaN stays NaN, the infinities and both zeros keep their sign.

``cast_to_f32`` widens exactly: every one of the 65536 f16 codes and of the 65536
bf16 codes comes back as its f32 value, subnormals included.

The references are numpy's ``float16`` conversion and rocke's torch-free bf16
codec (``rocke.numeric.bf16``, held to ``ml_dtypes`` by
``tests/numeric/test_bf16_codec.py``). Results are compared bit for bit; only a
NaN's sign and payload are left open.
"""

from __future__ import annotations

import numpy as np
import pytest
from rocke.core.ir import BF16, F16, F32, IRBuilder, PtrType
from rocke.helpers.compile import compile_kernel
from rocke.helpers.spec import sig_param
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
from rocke.tests.gpu import launch_with_buffers, supported_device_arch
from rocke.tests.ieee_compare import assert_ieee_identical

pytestmark = pytest.mark.gpu

# The arches this test has run on (gfx950 narrows to bf16 with
# v_cvt_pk_bf16_f32, gfx90a in integer instructions, so the two round
# differently). Add an arch here once the test has run green on it.
_RUN_ARCHES = ("gfx90a", "gfx942", "gfx950")

_F16_MAX = float(np.finfo(np.float16).max)  # 65504, significand 0x3ff (odd)
_F16_TINY = 2.0**-24  # the smallest f16 subnormal
_BF16_MAX = float(bf16_to_f32(np.array([0x7F7F], np.uint16))[0])  # significand odd
_BF16_TINY = 2.0**-133  # the smallest bf16 subnormal

# Values exactly halfway between two neighbouring halves, where round to nearest
# even differs from rounding half away from zero, plus their neighbours at the
# overflow threshold.
_F16_EDGES = [
    1 + 2**-11,  # between 1 and 1 + 2**-10: rounds down to the even 1
    1 + 3 * 2**-11,  # rounds up to the even 1 + 2**-9
    -(1 + 3 * 2**-11),
    _F16_TINY / 2,  # between 0 and the smallest subnormal: the even 0
    -_F16_TINY / 2,  # ... -0
    3 * _F16_TINY / 2,  # between subnormals 2**-24 and 2**-23: the even 2**-23
    _F16_TINY,
    2.0**-14 - _F16_TINY,  # the largest subnormal
    2.0**-14,  # the smallest normal
    _F16_MAX,
    _F16_MAX + 15.0,  # below the halfway point: rounds down to the max
    _F16_MAX + 16.0,  # 65520, the tie (half an f16 ulp): rounds to the even inf
    -(_F16_MAX + 16.0),
    1e6,
]
_BF16_EDGES = [
    1 + 2**-8,  # between 1 and 1 + 2**-7: rounds down to the even 1
    1 + 3 * 2**-8,  # rounds up to the even 1 + 2**-6
    -(1 + 3 * 2**-8),
    _BF16_TINY / 2,  # the even 0
    -_BF16_TINY / 2,  # ... -0
    3 * _BF16_TINY / 2,  # the even 2**-132
    _BF16_TINY,
    2.0**-149,  # the smallest f32 subnormal: rounds to 0
    2.0**-126,  # the smallest normal
    _BF16_MAX,
    _BF16_MAX + 2.0**119 - 2.0**104,  # one f32 ulp below the halfway point
    _BF16_MAX + 2.0**119,  # the tie (half a bf16 ulp): rounds to the even inf
    -(_BF16_MAX + 2.0**119),
    float(np.finfo(np.float32).max),
]
_SPECIALS = [0.0, -0.0, np.inf, -np.inf, np.nan]


def _narrowing_inputs() -> np.ndarray:
    ordinary = np.random.default_rng(7).standard_normal(64) * 1000.0
    values = np.concatenate([ordinary, _F16_EDGES, _BF16_EDGES, _SPECIALS]).astype(
        np.float32
    )
    # Every edge value must be an f32 value, or the f32 input already rounded it.
    edges = np.array(_F16_EDGES + _BF16_EDGES, dtype=np.float64)
    assert np.array_equal(edges.astype(np.float32).astype(np.float64), edges)
    return values


def _narrowing_kernel():
    """F16[i] / BF16[i] = X[i] narrowed; F16_BACK[i] / BF16_BACK[i] = widened again."""
    b = IRBuilder("half_narrowing")
    x = b.param("X", PtrType(F32, "global"))
    out = {
        name: b.param(name, PtrType(dtype, "global"))
        for name, dtype in (
            ("F16", F16),
            ("BF16", BF16),
            ("F16_BACK", F32),
            ("BF16_BACK", F32),
        )
    }
    tid = b.thread_id_x()
    value = b.global_load(x, tid, F32, align=4)
    as_f16 = b.trunc_f32_to_f16(value)
    as_bf16 = b.trunc_f32_to_bf16(value)
    b.global_store(out["F16"], tid, as_f16, align=2)
    b.global_store(out["BF16"], tid, as_bf16, align=2)
    b.global_store(out["F16_BACK"], tid, b.cast_to_f32(as_f16), align=4)
    b.global_store(out["BF16_BACK"], tid, b.cast_to_f32(as_bf16), align=4)
    b.ret()
    return b.kernel


def test_narrowing_rounds_to_nearest_even_on_edge_values_on_gpu():
    x = _narrowing_inputs()
    buffers = {
        "X": x.copy(),
        "F16": np.zeros(x.size, np.float16),
        "BF16": np.zeros(x.size, np.uint16),
        "F16_BACK": np.zeros(x.size, np.float32),
        "BF16_BACK": np.zeros(x.size, np.float32),
    }
    # run_on_device names each pointer by its host dtype; bf16 travels as raw
    # uint16 bits, so launch through the explicit signature instead.
    kernel = _narrowing_kernel()
    arch = supported_device_arch(kernel, _RUN_ARCHES)
    launch_with_buffers(
        compile_kernel(kernel, arch=arch),
        [
            sig_param("X", "f32"),
            sig_param("F16", "f16"),
            sig_param("BF16", "bf16"),
            sig_param("F16_BACK", "f32"),
            sig_param("BF16_BACK", "f32"),
        ],
        buffers,
        grid=(1, 1, 1),
        block=(x.size, 1, 1),
    )

    with np.errstate(over="ignore"):  # the overflow to inf is a wanted case
        want_f16 = x.astype(np.float16)
    want_bf16 = f32_to_bf16(x)
    assert_ieee_identical(buffers["F16"], want_f16)
    # bf16_to_f32 only moves the bits, so this compares the bf16 bits exactly
    # except for NaN, which must be NaN where expected.
    assert_ieee_identical(bf16_to_f32(buffers["BF16"]), bf16_to_f32(want_bf16))
    assert_ieee_identical(buffers["F16_BACK"], want_f16.astype(np.float32))
    assert_ieee_identical(buffers["BF16_BACK"], bf16_to_f32(want_bf16))


def _widening_kernel():
    """F16_WIDE[i] = f32(F16_CODES[i]), BF16_WIDE[i] = f32(BF16_CODES[i])."""
    b = IRBuilder("half_widening")
    f16_codes = b.param("F16_CODES", PtrType(F16, "global"))
    bf16_codes = b.param("BF16_CODES", PtrType(BF16, "global"))
    f16_wide = b.param("F16_WIDE", PtrType(F32, "global"))
    bf16_wide = b.param("BF16_WIDE", PtrType(F32, "global"))
    index = b.add(b.mul(b.block_id_x(), b.const_i32(256)), b.thread_id_x())
    b.global_store(
        f16_wide,
        index,
        b.cast_to_f32(b.global_load(f16_codes, index, F16, align=2)),
        align=4,
    )
    b.global_store(
        bf16_wide,
        index,
        b.cast_to_f32(b.global_load(bf16_codes, index, BF16, align=2)),
        align=4,
    )
    b.ret()
    return b.kernel


def test_widening_is_exact_for_every_code_on_gpu():
    codes = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    buffers = {
        "F16_CODES": codes.copy(),
        "BF16_CODES": codes.copy(),
        "F16_WIDE": np.zeros(codes.size, np.float32),
        "BF16_WIDE": np.zeros(codes.size, np.float32),
    }
    kernel = _widening_kernel()
    arch = supported_device_arch(kernel, _RUN_ARCHES)
    launch_with_buffers(
        compile_kernel(kernel, arch=arch),
        [
            sig_param("F16_CODES", "f16"),
            sig_param("BF16_CODES", "bf16"),
            sig_param("F16_WIDE", "f32"),
            sig_param("BF16_WIDE", "f32"),
        ],
        buffers,
        grid=(codes.size // 256, 1, 1),
        block=(256, 1, 1),
    )
    assert_ieee_identical(
        buffers["F16_WIDE"], codes.view(np.float16).astype(np.float32)
    )
    assert_ieee_identical(buffers["BF16_WIDE"], bf16_to_f32(codes))
