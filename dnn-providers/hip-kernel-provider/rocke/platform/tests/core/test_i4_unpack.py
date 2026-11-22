# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The packed-i4 unpack helpers give the right nibble order and sign on the GPU.

``unpack_i4_byte_to_pair_i32`` reads one byte as two signed 4-bit values: the low
nibble (bits 3..0) first, the high nibble (bits 7..4) second, each sign-extended
from bit 3 into the range ``[-8, 7]``. The f32 and f16 helpers convert those
values. Every one of the 256 bytes runs through each helper on the GPU, with and
without the IR optimizer, and the results are compared bit for bit with a numpy
decode written here from the bit fields. A swapped nibble order, a zero-extended
nibble or a sign taken from the wrong bit changes some of the 256 results.
"""

from __future__ import annotations

import numpy as np
import pytest
from rocke.core.ir import F16, F32, I8, I32, IRBuilder, KernelDef, PtrType
from rocke.core.passes import optimize_kernel
from rocke.helpers.i4_dequant import (
    unpack_i4_byte_to_pair_f16,
    unpack_i4_byte_to_pair_f32,
    unpack_i4_byte_to_pair_i32,
)
from rocke.tests.gpu import run_on_device

# The arches this test has run on. Every arch executes these integer and
# conversion instructions; add an arch here once the test has run green on it.
_RUN_ARCHES = ("gfx90a", "gfx942", "gfx950")

_BYTES = np.arange(256, dtype=np.uint8)

# helper -> (element type of the output buffer, numpy dtype of that buffer)
_UNPACKERS = {
    "i32": (unpack_i4_byte_to_pair_i32, I32, np.int32),
    "f32": (unpack_i4_byte_to_pair_f32, F32, np.float32),
    "f16": (unpack_i4_byte_to_pair_f16, F16, np.float16),
}


def _signed_nibbles(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(low, high) signed 4-bit values of each byte, from the bit fields."""
    low = (packed & 0x0F).astype(np.int32)
    high = (packed >> 4).astype(np.int32)
    return np.where(low >= 8, low - 16, low), np.where(high >= 8, high - 16, high)


def test_reference_decode_covers_the_whole_i4_range():
    # The reference itself: every byte's nibbles land in [-8, 7], and each of the
    # 16 values appears 16 times in each position, so the GPU check below sees
    # every value in both nibble positions.
    low, high = _signed_nibbles(_BYTES)
    for nibble in (low, high):
        values, counts = np.unique(nibble, return_counts=True)
        assert values.tolist() == list(range(-8, 8))
        assert counts.tolist() == [16] * 16
    assert (low[0x8F], high[0x8F]) == (-1, -8)
    assert (low[0x7F], high[0x7F]) == (-1, 7)


def _unpack_kernel(name: str) -> KernelDef:
    """Thread t unpacks byte t: LOW[t] and HIGH[t] get its two nibbles."""
    unpack, out_type, _ = _UNPACKERS[name]
    b = IRBuilder(f"i4_unpack_{name}")
    packed = b.param("Packed", PtrType(I8, "global"))
    low_out = b.param("Low", PtrType(out_type, "global"))
    high_out = b.param("High", PtrType(out_type, "global"))
    tid = b.thread_id_x()
    low, high = unpack(b, b.global_load_i8(packed, tid))
    align = 2 if out_type is F16 else 4
    b.global_store(low_out, tid, low, align=align)
    b.global_store(high_out, tid, high, align=align)
    b.ret()
    return b.kernel


# The optimizer runs only when asked; run the bytes through both paths.
@pytest.mark.parametrize("optimize_ir", [False, True], ids=["plain", "optimized"])
@pytest.mark.parametrize("name", sorted(_UNPACKERS))
def test_every_byte_unpacks_to_its_signed_nibbles_on_gpu(name, optimize_ir):
    kernel = _unpack_kernel(name)
    if optimize_ir:
        optimize_kernel(kernel)
    out_dtype = _UNPACKERS[name][2]
    buffers = {
        "Packed": _BYTES.copy(),
        "Low": np.full(_BYTES.size, 99, dtype=out_dtype),
        "High": np.full(_BYTES.size, 99, dtype=out_dtype),
    }
    run_on_device(kernel, buffers, _BYTES.size, _RUN_ARCHES)

    want_low, want_high = _signed_nibbles(_BYTES)
    for position, got, want in (
        ("low", buffers["Low"], want_low),
        ("high", buffers["High"], want_high),
    ):
        # The values are small integers, exact in every output type, so the
        # comparison is bit for bit after converting the reference.
        want = want.astype(out_dtype)
        wrong = np.flatnonzero(
            got.view(f"u{got.itemsize}") != want.view(f"u{want.itemsize}")
        )
        assert wrong.size == 0, (
            f"{name} {position} nibble wrong for {wrong.size} bytes, first "
            f"0x{int(wrong[0]):02x}: got {got[wrong[0]]}, want {want[wrong[0]]}"
        )
