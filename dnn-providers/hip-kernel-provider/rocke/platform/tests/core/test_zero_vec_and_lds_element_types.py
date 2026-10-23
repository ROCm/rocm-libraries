# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``zero_vec`` of f64 and i16, and LDS loads and stores of i16.

``zero_vec`` is what ``fill_fragment`` builds on, so without f64 an f64 fragment could
not be zero-filled. Lowering goes through the active engine, so a strict both-engines run
(``ROCKE_BACKEND=both ROCKE_CPP_STRICT=1``) also proves the two engines agree. The numeric
test runs only on a visible GPU whose arch it names, and skips otherwise.
"""

import numpy as np
import pytest
from rocke.core.ir import F64, I16, I64, TF32, IRBuilder, PtrType
from rocke.core.isa.backend import wired_arches
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.tests.gpu import run_on_device
from rocke.tests.ir_validity import assert_target_accepts


def test_zero_vec_accepts_f64_and_i16():
    b = IRBuilder("zero_vectors")
    b.zero_vec(I16, 4)
    b.zero_vec(F64, 2)
    text = lower_kernel_to_llvm(b.kernel, arch="gfx90a")
    assert "<4 x i16> zeroinitializer" in text
    assert "<2 x double> zeroinitializer" in text
    assert_target_accepts(text, "gfx90a")
    with pytest.raises(ValueError, match="^zero_vec unsupported elem tf32$"):
        IRBuilder("tf32_zero").zero_vec(TF32, 4)


def _i16_lds_round_trip(n: int) -> IRBuilder:
    """DST[i*n : i*n+n] = SRC[i*n : i*n+n], staged through LDS as <n x i16>."""
    b = IRBuilder(f"lds_round_trip_i16_{n}")
    src = b.param("SRC", PtrType(I16, "global"))
    dst = b.param("DST", PtrType(I16, "global"))
    tid = b.thread_id_x()
    base = b.mul(tid, b.const_i32(n))
    smem = b.smem_alloc(I16, [64 * n])
    elems = [b.global_load(src, b.add(base, b.const_i32(j)), I16) for j in range(n)]
    value = elems[0] if n == 1 else b.vec_pack(elems, I16)
    b.smem_store_vN(smem, [base], value, n)
    loaded = b.smem_load_vN(smem, base, dtype=I16, n=n)
    for j in range(n):
        b.global_store(dst, b.add(base, b.const_i32(j)), b.vec_extract(loaded, j))
    return b


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_lds_loads_and_stores_i16_at_natural_alignment(n):
    text = lower_kernel_to_llvm(_i16_lds_round_trip(n).kernel, arch="gfx90a")
    lds_lines = [line for line in text.splitlines() if "ptr addrspace(3)" in line]
    loads = [line for line in lds_lines if " = load " in line]
    stores = [line for line in lds_lines if line.strip().startswith("store ")]
    assert loads and stores
    for line in loads + stores:
        assert "i16" in line and line.rstrip().endswith(f"align {2 * n}"), line
    if n > 1:
        assert any(f"<{n} x i16>" in line for line in loads)
    assert_target_accepts(text, "gfx90a")


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
            r"fp8e4m3 / bf8e5m2 / i8, got i64$"
        ),
    ):
        b.smem_load_vN(smem, idx, dtype=I64, n=1)


# Every arch rocke lowers for executes i16 LDS accesses.
_EVERY_ARCH = wired_arches()


def test_lds_round_trip_of_i16_on_gpu():
    n = 4
    # Span the whole i16 range, sign bit included, so a truncated or sign-extended
    # lane shows up as a wrong value.
    values = (np.arange(64 * n, dtype=np.int64) * 257 - 32768).astype(np.int16)
    buffers = {"SRC": values.copy(), "DST": np.zeros_like(values)}
    run_on_device(_i16_lds_round_trip(n).kernel, buffers, 64, _EVERY_ARCH)
    np.testing.assert_array_equal(buffers["DST"], values)
