# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The F64 scalar type, and the i16 / f64 element support the memory ops gained with it.

Every lowering here pins the Python engine: the C++ engine does not mirror this yet.
The lowered modules are checked with the compile-and-link helper; the numeric tests run
only on a visible GPU whose arch they name, and skip otherwise; there, a compiler
rejection fails the test.
"""

import struct

import numpy as np
import pytest
from rocke.core.ir import F32, F64, I16, TF32, IRBuilder, PtrType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.core.isa.backend import wired_arches
from rocke.tests.gpu import run_on_device
from rocke.tests.ir_validity import assert_target_accepts


def _lower(b: IRBuilder) -> str:
    return _lower_kernel_to_llvm_python(b.kernel, arch="gfx90a")


def _f64_kernel() -> IRBuilder:
    """y[i] = x[i] + 1.0, and lo[i] = f32(x[i]), hi[i] = f64(f32(x[i]))."""
    b = IRBuilder("f64_add_and_casts")
    x = b.param("X", PtrType(F64, "global"))
    y = b.param("Y", PtrType(F64, "global"))
    lo = b.param("LO", PtrType(F32, "global"))
    hi = b.param("HI", PtrType(F64, "global"))
    tid = b.thread_id_x()
    value = b.global_load(x, tid, F64, align=8)
    b.global_store(y, tid, b.fadd(value, b.const_f64(1.0)), align=8)
    narrowed = b.cvt_f64_to_f32(value)
    b.global_store(lo, tid, narrowed, align=4)
    b.global_store(hi, tid, b.cvt_f32_to_f64(narrowed), align=8)
    return b


def test_f64_lowers_to_double_and_links():
    text = _lower(_f64_kernel())
    assert "load double" in text
    assert "fadd double" in text
    assert "0x3FF0000000000000" in text  # const_f64(1.0)
    assert "fptrunc double" in text and "to float" in text
    assert "fpext float" in text and "to double" in text
    assert_target_accepts(text, "gfx90a")


def test_f64_vectors_load_add_and_store():
    b = IRBuilder("f64_vector")
    p = b.param("P", PtrType(F64, "global"))
    tid = b.thread_id_x()
    pair = b.global_load_vN(p, tid, F64, 2)
    b.global_store_vN(p, tid, b.vector_add(pair, pair), 2)
    text = _lower(b)
    assert "load <2 x double>" in text and "align 16" in text
    assert "fadd <2 x double>" in text
    assert "store <2 x double>" in text
    assert_target_accepts(text, "gfx90a")


def test_zero_vec_accepts_f64_and_i16():
    b = IRBuilder("zero_vectors")
    b.zero_vec(I16, 4)
    b.zero_vec(F64, 2)
    text = _lower(b)
    assert "<4 x i16> zeroinitializer" in text
    assert "<2 x double> zeroinitializer" in text
    assert_target_accepts(text, "gfx90a")
    with pytest.raises(ValueError, match="zero_vec unsupported elem tf32"):
        IRBuilder("tf32_zero").zero_vec(TF32, 4)


def _lds_round_trip(dtype, n: int) -> IRBuilder:
    """DST[i*n : i*n+n] = SRC[i*n : i*n+n], staged through LDS as <n x dtype>."""
    b = IRBuilder(f"lds_round_trip_{dtype.name}_{n}")
    src = b.param("SRC", PtrType(dtype, "global"))
    dst = b.param("DST", PtrType(dtype, "global"))
    tid = b.thread_id_x()
    base = b.mul(tid, b.const_i32(n))
    smem = b.smem_alloc(dtype, [64 * n])
    elems = [b.global_load(src, b.add(base, b.const_i32(j)), dtype) for j in range(n)]
    value = elems[0] if n == 1 else b.vec_pack(elems, dtype)
    b.smem_store_vN(smem, [base], value, n)
    loaded = b.smem_load_vN(smem, base, dtype=dtype, n=n)
    for j in range(n):
        b.global_store(dst, b.add(base, b.const_i32(j)), b.vec_extract(loaded, j))
    return b


@pytest.mark.parametrize(
    "dtype,n,align",
    [(I16, 1, 2), (I16, 2, 4), (I16, 4, 8), (F64, 1, 8), (F64, 2, 16)],
)
def test_lds_load_and_store_i16_and_f64_at_natural_alignment(dtype, n, align):
    text = _lower(_lds_round_trip(dtype, n))
    element = "i16" if dtype is I16 else "double"
    vector = f"<{n} x {element}>"
    lds_lines = [line for line in text.splitlines() if "ptr addrspace(3)" in line]
    loads = [line for line in lds_lines if " = load " in line]
    stores = [line for line in lds_lines if line.strip().startswith("store ")]
    assert loads and stores
    for line in loads + stores:
        assert element in line and line.rstrip().endswith(f"align {align}"), line
    if n > 1:
        assert any(vector in line for line in loads)
    assert_target_accepts(text, "gfx90a")


def test_lds_widths_outside_the_element_rules_still_raise():
    b = IRBuilder("lds_bad_widths")
    smem = b.smem_alloc(F64, [64])
    idx = b.thread_id_x()
    with pytest.raises(ValueError, match="unsupported vector width 4"):
        b.smem_load_vN(smem, idx, dtype=F64, n=4)
    from rocke.core.ir import I64

    with pytest.raises(ValueError, match="smem_load_vN supports"):
        b.smem_load_vN(smem, idx, dtype=I64, n=1)


def test_lds_pool_sizes_f64_allocations_at_eight_bytes():
    b = IRBuilder("lds_pool_f64")
    tid = b.thread_id_x()
    doubles = b.smem_alloc(F64, [64])
    halves = b.smem_alloc(I16, [64])
    b.smem_store_vN(doubles, [tid], b.const_f64(1.0), 1)
    b.smem_store_vN(halves, [tid], b.vec_extract(b.zero_vec(I16, 2), 0), 1)
    text = _lower(b)
    # Both allocations are live at once, so the pool holds 64 x 8 + 64 x 2 bytes; an f64
    # element sized at the 2-byte default would make the two overlap.
    assert "global [640 x i8]" in text


# ---- on GPU (only where the visible GPU supports these types) --------------------

# Every arch rocke lowers for executes f64 and i16 arithmetic and LDS accesses.
_EVERY_ARCH = wired_arches()


def test_f64_arithmetic_and_casts_on_gpu():
    x = np.array(
        [0.1 * i - 3.0 for i in range(64)] + [1e300, -1e-300], dtype=np.float64
    )
    x = np.resize(x, 64)
    buffers = {
        "X": x.copy(),
        "Y": np.zeros(64, np.float64),
        "LO": np.zeros(64, np.float32),
        "HI": np.zeros(64, np.float64),
    }
    run_on_device(_f64_kernel().kernel, buffers, 64, _EVERY_ARCH)
    np.testing.assert_array_equal(buffers["Y"], x + 1.0)
    np.testing.assert_array_equal(buffers["LO"], x.astype(np.float32))
    np.testing.assert_array_equal(
        buffers["HI"], x.astype(np.float32).astype(np.float64)
    )


@pytest.mark.parametrize("dtype,n", [(I16, 4), (F64, 2)])
def test_lds_round_trip_of_i16_and_f64_on_gpu(dtype, n):
    host = np.int16 if dtype is I16 else np.float64
    values = np.arange(64 * n).astype(host)
    if dtype is F64:
        values = values * 0.5 + struct.unpack("<d", struct.pack("<Q", 1))[0]
    buffers = {"SRC": values.copy(), "DST": np.zeros_like(values)}
    run_on_device(_lds_round_trip(dtype, n).kernel, buffers, 64, _EVERY_ARCH)
    np.testing.assert_array_equal(buffers["DST"], values)


def test_the_f32_promote_and_demote_helpers_do_not_take_f64():
    # f64 conversions are their own methods, so helpers that promote to or demote from f32
    # (and the code built on them) never pick up an f64 conversion by accident.
    b = IRBuilder("no_implicit_f64")
    x = b.param("X", F64)
    y = b.param("Y", F32)
    with pytest.raises(ValueError, match="cast_to_f32 unsupported from f64"):
        b.cast_to_f32(x)
    with pytest.raises(ValueError, match="cast_f32_to unsupported to f64"):
        b.cast_f32_to(y, F64)
    with pytest.raises(ValueError, match="cvt_f64_to_f32 expects f64 input, got f32"):
        b.cvt_f64_to_f32(y)
    with pytest.raises(ValueError, match="cvt_f32_to_f64 expects f32 input, got f64"):
        b.cvt_f32_to_f64(x)
