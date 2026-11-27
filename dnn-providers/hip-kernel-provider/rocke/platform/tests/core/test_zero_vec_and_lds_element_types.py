# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Element types that ``zero_vec`` and LDS loads and stores accept or refuse.

``zero_vec`` of f64, i16 and u8, LDS loads and stores of i16 and i64 (i64 also
in HIP), LDS loads of the 8-bit float types at their byte width, and the refusal
of an LDS tile whose element has no whole-byte width.

``zero_vec`` is what ``fill_fragment`` and ``load_fragment`` build on, so without f64 an f64
fragment could not be zero-filled, and without u8 a u8 fragment could not be built. Lowering
goes through the active engine, so a strict both-engines run
(``ROCKE_BACKEND=both ROCKE_CPP_STRICT=1``) also proves the two engines agree. The numeric
test runs only on a visible GPU whose arch it names, and skips otherwise.
"""

import numpy as np
import pytest
from rocke.core.ir import (
    BF8E5M2,
    F64,
    FP8E4M3,
    I1,
    I16,
    I64,
    TF32,
    U8,
    IRBuilder,
    PtrType,
    Type,
)
from rocke.core.isa.backend import wired_arches
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.tests.engines import (
    ENGINES,
    assert_engine_rejects,
    lower_hip_with_both_engines,
)
from rocke.tests.gpu import run_on_device
from rocke.tests.ir_validity import assert_target_accepts


def test_zero_vec_accepts_f64_i16_and_u8():
    b = IRBuilder("zero_vectors")
    b.zero_vec(I16, 4)
    b.zero_vec(F64, 2)
    unsigned_bytes = b.zero_vec(U8, 8)
    assert unsigned_bytes.type.name == "vec<u8x8>"
    text = lower_kernel_to_llvm(b.kernel, arch="gfx90a")
    assert "<4 x i16> zeroinitializer" in text
    assert "<2 x double> zeroinitializer" in text
    # LLVM has no unsigned integer types: a u8 vector is an i8 vector.
    assert "<8 x i8> zeroinitializer" in text
    assert_target_accepts(text, "gfx90a")
    with pytest.raises(ValueError, match="^zero_vec unsupported elem tf32$"):
        IRBuilder("tf32_zero").zero_vec(TF32, 4)


def _lds_round_trip(elem: Type, n: int) -> IRBuilder:
    """DST[i*n : i*n+n] = SRC[i*n : i*n+n], staged through LDS as <n x elem>."""
    b = IRBuilder(f"lds_round_trip_{elem.name}_{n}")
    src = b.param("SRC", PtrType(elem, "global"))
    dst = b.param("DST", PtrType(elem, "global"))
    tid = b.thread_id_x()
    base = b.mul(tid, b.const_i32(n))
    smem = b.smem_alloc(elem, [64 * n])
    elems = [b.global_load(src, b.add(base, b.const_i32(j)), elem) for j in range(n)]
    value = elems[0] if n == 1 else b.vec_pack(elems, elem)
    b.smem_store_vN(smem, [base], value, n)
    loaded = b.smem_load_vN(smem, base, dtype=elem, n=n)
    for j in range(n):
        b.global_store(dst, b.add(base, b.const_i32(j)), b.vec_extract(loaded, j))
    return b


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_lds_loads_and_stores_i16_at_natural_alignment(n):
    text = lower_kernel_to_llvm(_lds_round_trip(I16, n).kernel, arch="gfx90a")
    lds_lines = [line for line in text.splitlines() if "ptr addrspace(3)" in line]
    loads = [line for line in lds_lines if " = load " in line]
    stores = [line for line in lds_lines if line.strip().startswith("store ")]
    assert loads and stores
    for line in loads + stores:
        assert "i16" in line and line.rstrip().endswith(f"align {2 * n}"), line
    if n > 1:
        assert any(f"<{n} x i16>" in line for line in loads)
    assert_target_accepts(text, "gfx90a")


@pytest.mark.parametrize("n", [1, 2])
def test_lds_loads_and_stores_i64_at_natural_alignment(n):
    text = lower_kernel_to_llvm(_lds_round_trip(I64, n).kernel, arch="gfx90a")
    lds_lines = [line for line in text.splitlines() if "ptr addrspace(3)" in line]
    loads = [line for line in lds_lines if " = load " in line]
    stores = [line for line in lds_lines if line.strip().startswith("store ")]
    assert loads and stores
    for line in loads + stores:
        assert "i64" in line and line.rstrip().endswith(f"align {8 * n}"), line
    if n > 1:
        assert any("<2 x i64>" in line for line in loads + stores)
    assert_target_accepts(text, "gfx90a")


@pytest.mark.parametrize("n", [1, 2])
def test_lds_loads_and_stores_i64_lower_to_hip_as_i64_vectors(n):
    hip = lower_hip_with_both_engines(_lds_round_trip(I64, n).kernel, "gfx90a")
    vector = f"i64x{n}"
    assert f"using {vector} = int64_t __attribute__((ext_vector_type({n})));" in hip
    assert f"*reinterpret_cast<const {vector}*>(&" in hip  # the LDS load
    if n > 1:  # one lane is stored as a scalar
        assert f"*reinterpret_cast<{vector}*>(&" in hip  # the LDS store


@pytest.mark.parametrize("dtype", [FP8E4M3, BF8E5M2], ids=lambda t: t.name)
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16])
def test_lds_loads_of_8_bit_floats_claim_their_byte_width(dtype, n):
    # An <n x i8> LDS load claims only n bytes of alignment: a claim of 2n
    # (the 16-bit width) is more than an address at an odd multiple of n
    # guarantees, which LLVM treats as undefined behaviour.
    b = IRBuilder(f"lds_load_{dtype.name}_{n}")
    dst = b.param("DST", PtrType(dtype, "global"))
    tid = b.thread_id_x()
    smem = b.smem_alloc(dtype, [64 * n])
    loaded = b.smem_load_vN(smem, b.mul(tid, b.const_i32(n)), dtype=dtype, n=n)
    b.global_store(dst, tid, b.vec_extract(loaded, 0))
    text = lower_kernel_to_llvm(b.kernel, arch="gfx950")
    loads = [
        line
        for line in text.splitlines()
        if "ptr addrspace(3)" in line and " = load " in line
    ]
    assert loads
    for line in loads:
        assert line.rstrip().endswith(f"align {n}"), line
    assert_target_accepts(text, "gfx950")


_NO_BYTE_WIDTH = (
    "type_bytes failed - type=i1 (1 bits), expected a whole number of bytes"
)


def _lds_tile_of_i1():
    # An i1 element has no whole-byte width, so neither the tile's LDS segment
    # size nor the store's alignment exists; the lowering refuses rather than
    # assume a width.
    b = IRBuilder("lds_i1")
    tid = b.thread_id_x()
    smem = b.smem_alloc(I1, [64])
    b.smem_store_vN(smem, [tid], b.cmp_lt(tid, b.const_i32(3)), n=1)
    return b.kernel


@pytest.mark.parametrize("engine", ENGINES)
def test_an_lds_tile_without_a_byte_width_is_refused(engine):
    assert_engine_rejects(
        _lds_tile_of_i1(),
        "gfx942",
        engine=engine,
        error_type=ValueError,
        message=_NO_BYTE_WIDTH,
    )


def test_lds_loads_outside_the_element_rules_still_raise():
    b = IRBuilder("lds_bad_loads")
    smem = b.smem_alloc(I16, [64])
    idx = b.thread_id_x()
    with pytest.raises(
        ValueError, match="^unsupported vector width 3 for smem_load_vN"
    ):
        b.smem_load_vN(smem, idx, dtype=I16, n=3)
    with pytest.raises(
        ValueError,
        match=(
            r"^smem_load_vN supports f16 / bf16 / i16 / f32 / i32 / tf32 / f64 / "
            r"i64 / fp8e4m3 / bf8e5m2 / i8, got u8$"
        ),
    ):
        b.smem_load_vN(smem, idx, dtype=U8, n=1)


# Every arch rocke lowers for executes i16 LDS accesses.
_EVERY_ARCH = wired_arches()


@pytest.mark.gpu
def test_lds_round_trip_of_i16_on_gpu():
    n = 4
    # Span the whole i16 range, sign bit included, so a truncated or sign-extended
    # lane shows up as a wrong value.
    values = (np.arange(64 * n, dtype=np.int64) * 257 - 32768).astype(np.int16)
    buffers = {"SRC": values.copy(), "DST": np.zeros_like(values)}
    run_on_device(_lds_round_trip(I16, n).kernel, buffers, 64, _EVERY_ARCH)
    np.testing.assert_array_equal(buffers["DST"], values)


# The arches the i64 round trip has run green on; add one once it has.
_I64_RUN_ARCHES = ("gfx90a", "gfx942", "gfx950")


@pytest.mark.gpu
def test_lds_round_trip_of_i64_on_gpu():
    n = 2
    # Both halves of every word vary, the sign bit included, and every fourth
    # word is an f64 NaN pattern (quiet or signalling, either sign), so a
    # dropped half, a sign extension or a float canonicalisation shows up as a
    # wrong value.
    lanes = np.arange(64 * n, dtype=np.uint64)
    words = (lanes * np.uint64(0x9E3779B97F4A7C15)) ^ (lanes << np.uint64(59))
    words[::8] = np.uint64(0x7FF0000000000001) + lanes[::8]
    words[4::8] = np.uint64(0xFFF8000000000000) | lanes[4::8]
    values = words.view(np.int64)
    buffers = {"SRC": values.copy(), "DST": np.zeros_like(values)}
    run_on_device(_lds_round_trip(I64, n).kernel, buffers, 64, _I64_RUN_ARCHES)
    np.testing.assert_array_equal(buffers["DST"], values)
