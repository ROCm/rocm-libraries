# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Relations stated between kernel parameters feed the tiling layer's build-time proofs.

A relation (``ldc ge N``, ``sb ge product_of(M, ldc)``) is checked before each launch through
``KernelLauncher.from_artifact``, so the build may rely on it. It is stated on whichever of its
parameters is declared last, so one fact takes several shapes; ``stated_relations`` reads each in
one shape, ``big >= small``, moving a ``quotient_of`` to a product and a ``difference_of`` to a sum.
These tests pin:

- the normal shape of every way a relation can be stated, and the relations left out;
- the aliasing proof: a padded C whose ``ldc`` is stated at least ``N``, and a batched C whose batch
  stride is stated at least ``M * ldc``, are proven not to alias in every shape the ABI order allows,
  and the proof names exactly the relations it used; without them the view is decided neither way;
- the 32-bit index proof: a proven window over runtime lengths whose largest offset the numbers
  cannot bound takes the one-index form when a stated relation bounds it, and
  ``explain_bounds_checks`` names that relation -- only that one, only when the proof uses it, and
  without it the window takes the 64-bit form; both engines lower the kernel the same.
"""

from __future__ import annotations

import pytest

from rocke.core.intexpr import difference_of, product_of, quotient_of, sum_of
from rocke.core.ir import F16, I32, IRBuilder, PtrType
from rocke.helpers.tiling import (
    exit_if_tile_outside,
    explain_bounds_checks,
    fill_fragment,
    make_tensor_desc,
    make_tile_desc,
    make_window,
    store_fragment,
)
from rocke.helpers.tiling.clipping import GlobalForm
from rocke.helpers.tiling.fragments import make_fragment
from rocke.helpers.tiling.stated_relations import relations_among
from rocke.tests.engines import lower_with_both_engines

WAVE = 64
LIMIT = (1 << 31) - 1
TILE = make_tile_desc(
    shape=[16, 16],
    thread_tile=[1, 4],
    thread_dist=[16, 4],
    thread_order=[1, 0],
    wave_size=WAVE,
)


def _builder(name: str = "relations") -> IRBuilder:
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = WAVE
    return b


def _shapes(*values) -> set[tuple[tuple, tuple, str]]:
    return {(r.big, r.small, r.text) for r in relations_among(values)}


# --- the normal shape ------------------------------------------------------------------------------


def test_every_way_to_state_a_relation_reads_as_big_at_least_small():
    b = _builder()
    m = b.param("M", I32, ge=1, le=1 << 16)
    n = b.param("N", I32, ge=1, le=1 << 16)
    ldc = b.param("ldc", I32, ge=n, le=1 << 16)
    sb = b.param("sb", I32, ge=product_of(m, ldc))
    assert _shapes(ldc, sb) == {
        (((("ldc",), 1),), ((("N",), 1),), "ldc ge N"),
        (((("sb",), 1),), ((("M", "ldc"), 1),), "sb ge product_of(M, ldc)"),
    }


def test_a_quotient_moves_to_a_product_and_a_difference_to_a_sum():
    # ABI (sb, M, ldc): sb >= M * ldc is stated on ldc as le=quotient_of(sb, M).
    b = _builder()
    sb = b.param("sb", I32, ge=1)
    m = b.param("M", I32, ge=1, le=1 << 16)
    ldc = b.param("ldc", I32, ge=1, le=quotient_of(sb, m))
    assert _shapes(ldc) == {
        (((("sb",), 1),), ((("M", "ldc"), 1),), "ldc le quotient_of(sb, M)"),
    }
    # ABI (total, a, b): total >= a + b is stated on b as le=difference_of(total, a).
    b2 = _builder()
    total = b2.param("total", I32, ge=0)
    a = b2.param("a", I32, ge=0)
    last = b2.param("b", I32, ge=0, le=difference_of(total, a))
    assert _shapes(last) == {
        (
            ((("total",), 1),),
            ((("a",), 1), (("b",), 1)),
            "b le difference_of(total, a)",
        ),
    }


def test_lt_gt_and_eq_shift_and_double_the_shape():
    b = _builder()
    n = b.param("N", I32, ge=1, le=64)
    k = b.param("K", I32, gt=n, lt=sum_of(n, 64))
    assert _shapes(k) == {
        (((("K",), 1),), (((), 1), (("N",), 1)), "K gt N"),
        ((((), 64), (("N",), 1)), (((), 1), (("K",), 1)), "K lt sum_of(N, 64)"),
    }
    b2 = _builder()
    n2 = b2.param("N", I32, ge=1, le=64)
    e = b2.param("E", I32, eq=n2)
    assert _shapes(e) == {
        (((("E",), 1),), ((("N",), 1),), "E eq N"),
        (((("N",), 1),), ((("E",), 1),), "E eq N"),
    }


def test_a_quotient_on_the_smaller_side_is_left_out():
    # x ge quotient_of(a, y) gives no exact product.
    b = _builder()
    a = b.param("a", I32, ge=0, le=1 << 20)
    y = b.param("y", I32, ge=1, le=64)
    x = b.param("x", I32, ge=quotient_of(a, y))
    assert _shapes(x) == set()


# --- the aliasing proof ------------------------------------------------------------------------------


def _padded_c(stated: str):
    """C (M, N) with its own leading dimension, in the ABI order and shape ``stated`` gives."""
    b = _builder()
    if stated == "ldc ge N":
        m = b.param("M", I32, ge=1, le=1 << 16)
        n = b.param("N", I32, ge=1, le=1 << 16)
        ldc = b.param("ldc", I32, ge=n, le=1 << 16)
    elif stated == "N le ldc":
        m = b.param("M", I32, ge=1, le=1 << 16)
        ldc = b.param("ldc", I32, ge=1, le=1 << 16)
        n = b.param("N", I32, ge=1, le=ldc)
    elif stated == "ldc ge sum_of(N, 8)":
        m = b.param("M", I32, ge=1, le=1 << 16)
        n = b.param("N", I32, ge=1, le=1 << 16)
        ldc = b.param("ldc", I32, ge=sum_of(n, 8), le=1 << 17)
    else:
        m = b.param("M", I32, ge=1, le=1 << 16)
        n = b.param("N", I32, ge=1, le=1 << 16)
        ldc = b.param("ldc", I32, ge=1, le=1 << 16)
    return b, make_tensor_desc((m, n), (ldc, 1), F16)


@pytest.mark.parametrize("stated", ["ldc ge N", "N le ldc", "ldc ge sum_of(N, 8)"])
def test_a_padded_c_stated_ldc_at_least_n_is_proven_not_to_alias(stated):
    _, c = _padded_c(stated)
    assert c._injective_proof() == (stated,)
    assert c._aliasing_positions() is None


def test_without_the_relation_the_padded_c_is_decided_neither_way():
    _, c = _padded_c("nothing")
    assert c._injective_proof() is None
    assert c._aliasing_positions() is None


def _batched_c(abi: str):
    """C (G, M, N) with ldc and a batch stride, the batch stride related to M * ldc in the shape
    the ABI order allows."""
    b = _builder()
    g = b.param("G", I32, ge=1, le=64)
    if abi == "sb last":
        m = b.param("M", I32, ge=1, le=1 << 12)
        n = b.param("N", I32, ge=1, le=1 << 12)
        ldc = b.param("ldc", I32, ge=n, le=1 << 12)
        sb = b.param("sb", I32, ge=product_of(m, ldc))
    elif abi == "sb first":
        sb = b.param("sb", I32, ge=1)
        m = b.param("M", I32, ge=1, le=1 << 12)
        n = b.param("N", I32, ge=1, le=1 << 12)
        ldc = b.param("ldc", I32, ge=n, le=quotient_of(sb, m))
    else:
        m = b.param("M", I32, ge=1, le=1 << 12)
        n = b.param("N", I32, ge=1, le=1 << 12)
        ldc = b.param("ldc", I32, ge=n, le=1 << 12)
        sb = b.param("sb", I32, ge=1)
    return b, make_tensor_desc((g, m, n), (sb, ldc, 1), F16)


def test_a_batched_c_stated_sb_at_least_m_times_ldc_is_proven_not_to_alias():
    _, c = _batched_c("sb last")
    assert c._injective_proof() == ("ldc ge N", "sb ge product_of(M, ldc)")
    _, c = _batched_c("sb first")
    assert c._injective_proof() == ("ldc ge N", "ldc le quotient_of(sb, M)")


def test_without_the_batch_relation_the_batched_c_is_decided_neither_way_and_stores():
    b, c = _batched_c("no batch relation")
    assert c._injective_proof() is None
    assert c._aliasing_positions() is None
    ptr = b.param("C", PtrType(F16, "global"), align=16)
    fragment = make_fragment(TILE, F16)
    fill_fragment(b, fragment, 0)
    store_fragment(b, ptr, make_window(c, (0, 0, 0)), fragment, b.thread_id_x())


# --- the 32-bit index proof --------------------------------------------------------------------------

ROW = 4096


def _proven_window(variant: str):
    """A proven window over C (M, N) with the int row stride 4096: both lengths runtime, tile
    multiples, both origins guarded. ``variant`` decides what bounds ``4096 * M + N``:

    - ``"relation"``: M up to 2^20 (the numbers reach 2^32) and N stated at most
      ``difference_of(2^31 - 1, product_of(M, 4096))``, which keeps it in i32;
    - ``"none"``: the same lengths with N at most 4096 and no relation;
    - ``"numbers"``: N at most 4096 first, then M stated at most
      ``quotient_of(difference_of(2^31 - 1, N), 4096)`` -- the same relation, whose numbers alone
      already keep every offset in i32.
    """
    b = _builder()
    ptr = b.param("C", PtrType(F16, "global"), align=16)
    if variant == "numbers":
        n = b.param("N", I32, ge=16, le=ROW, multiple_of=16)
        bound = quotient_of(difference_of(LIMIT, n), ROW)
        m = b.param("M", I32, ge=16, le=bound, multiple_of=16)
    else:
        m = b.param("M", I32, ge=16, le=1 << 20, multiple_of=16)
        bound = (
            difference_of(LIMIT, product_of(m, ROW)) if variant == "relation" else ROW
        )
        n = b.param("N", I32, ge=16, le=bound, multiple_of=16)
    c = make_tensor_desc((m, n), (ROW, 1), F16)
    m0 = exit_if_tile_outside(b, b.mul(b.block_id_y(), b.const_i32(16)), m)
    n0 = exit_if_tile_outside(b, b.mul(b.block_id_x(), b.const_i32(16)), n)
    return b, ptr, make_window(c, (m0, n0))


STATED = f"N le difference_of({LIMIT}, product_of(M, {ROW}))"


def test_a_stated_relation_proves_the_32_bit_index_and_is_named():
    b, ptr, window = _proven_window("relation")
    plan = explain_bounds_checks(b, ptr, window, TILE)
    assert plan.form is GlobalForm.INDEX32
    assert plan.address_facts == (STATED,)
    assert str(plan).splitlines()[-2:] == [
        "  addresses: one 32-bit element index from the pointer",
        f"  the 32-bit index is proven by the stated relations: {STATED}",
    ]


def test_the_named_relation_is_what_lifts_the_form():
    # The same window without it takes the 64-bit form and names nothing.
    b, ptr, window = _proven_window("none")
    plan = explain_bounds_checks(b, ptr, window, TILE)
    assert plan.form is GlobalForm.PLACED
    assert plan.address_facts == ()


def test_a_relation_the_proof_does_not_need_is_not_named():
    b, ptr, window = _proven_window("numbers")
    plan = explain_bounds_checks(b, ptr, window, TILE)
    assert plan.form is GlobalForm.INDEX32
    assert plan.address_facts == ()


def test_the_window_the_relation_proves_lowers_the_same_in_both_engines():
    b, ptr, window = _proven_window("relation")
    fragment = make_fragment(TILE, F16)
    fill_fragment(b, fragment, 0)
    store_fragment(b, ptr, window, fragment, b.thread_id_x())
    b.ret()
    for flavor in ("llvm20", "llvm22", "llvm23"):
        lower_with_both_engines(b.kernel, "gfx90a", llvm_flavor=flavor)


def test_gpu_the_window_the_relation_proves_copies_exactly():
    # A copy through the proven window, launched through the checking door with M = N = 32: every
    # element of the 32 x 32 corner lands, and nothing past N in a row is written.
    import numpy as np

    from rocke.helpers.compile import compile_kernel
    from rocke.helpers.tiling import load_fragment
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import (
        DeviceMem,
        KernelLauncher,
        LaunchConfig,
        synchronize_and_release,
    )
    from rocke.tests.gpu import current_device_arch

    arch = current_device_arch()
    b, ptr, window = _proven_window("relation")
    source = b.param("X", PtrType(F16, "global"), align=16)
    lane = b.thread_id_x()
    store_fragment(b, ptr, window, load_fragment(b, source, window, TILE, lane), lane)
    b.ret()
    launcher = KernelLauncher.from_artifact(compile_kernel(b.kernel, arch=arch))
    rows, cols = 32, 32
    x = (np.arange(rows * ROW) % 2039).astype(np.float16)
    y = np.full(rows * ROW, -1.0, dtype=np.float16)
    rt = Runtime()
    x_dev, y_dev = DeviceMem(x.nbytes), DeviceMem(y.nbytes)
    rt.memcpy_h2d(x_dev.ptr(), as_u8_buffer(x), x.nbytes)
    rt.memcpy_h2d(y_dev.ptr(), as_u8_buffer(y), y.nbytes)
    launcher(
        {"C": y_dev, "M": rows, "N": cols, "X": x_dev},
        config=LaunchConfig(grid=(cols // 16, rows // 16, 1), block=(WAVE, 1, 1)),
    )
    synchronize_and_release()
    rt.memcpy_d2h(as_u8_buffer(y), y_dev.ptr(), y.nbytes)
    got, want = y.reshape(rows, ROW), x.reshape(rows, ROW)
    assert np.array_equal(got[:, :cols], want[:, :cols])
    assert np.all(got[:, cols:] == -1.0)
