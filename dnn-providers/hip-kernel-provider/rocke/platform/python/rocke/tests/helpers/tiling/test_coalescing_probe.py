# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Offline tests for the ASM-backed coalescing orchestrator (no GPU / no toolchain): the arch line-size gate,
the objdump width extractor, and the model-vs-ASM HARD gate. The compile seam (:func:`gate_recorded_store`) is
exercised only where a toolchain exists; these tests pin the pure, machine-independent glue.
"""

import pytest

from rocke.core.ir import F16, F32, IRBuilder, PtrType
from rocke.helpers.tiling import (
    TileMma,
    Tiling,
    fill_fragment,
    load_fragment,
    make_fragment,
    make_tensor_desc,
    make_window,
)
from rocke.helpers.tiling.analysis import analyze_coalescing
from rocke.helpers.tiling.coalescing_probe import (
    achieved_widths,
    gate_report,
    line_bytes_for,
    report_for_transaction,
)
from rocke.helpers.tiling.tiling_recorder import record_build
from rocke.helpers.tiling.transforms import interleave_idx


def _native_c():
    return {
        (L, mi * 16 + nj * 4 + r): (mi + 4 * (4 * (L // 16) + r), nj + 4 * (L % 16))
        for L in range(64)
        for mi in range(4)
        for nj in range(4)
        for r in range(4)
    }


def _shuffle(fwd, perm):
    return {(L, i): fwd[(L, perm[i])] for L in range(64) for i in range(len(perm))}


def test_line_bytes_required_never_assumed():
    assert line_bytes_for("gfx90a") == 128
    with pytest.raises(KeyError, match="no cache-line size"):
        line_bytes_for("gfx-imaginary")


def test_achieved_widths_reads_dword_ladder():
    # a global f32 store family: two dwordx4 (b128) and one dwordx2 (b64) -> {4:2, 2:1} elements
    text = """
      global_store_dwordx4 v[0:3], v4, s[0:1]
      global_store_dwordx4 v[4:7], v8, s[0:1]
      global_store_dwordx2 v[8:9], v10, s[0:1]
      global_load_dwordx4  v[0:3], v4, s[2:3]
    """
    st = achieved_widths(text, direction="store", space="global", dtype_bits=32)
    assert st == {4: 2, 2: 1}  # loads excluded; widths in ELEMENTS
    ld = achieved_widths(text, direction="load", space="global", dtype_bits=32)
    assert ld == {4: 1}


def test_achieved_widths_lds_b_ladder_and_dtype_scaling():
    # ds_write_b128 of f16 = 16 bytes / 2 = 8 elements; a global family filter must NOT catch ds_*
    text = "ds_write_b128 v0, v[1:4]\n ds_read_b64 v[5:6], v7\n"
    assert achieved_widths(text, direction="store", space="lds", dtype_bits=16) == {
        8: 1
    }
    assert achieved_widths(text, direction="load", space="lds", dtype_bits=16) == {4: 1}
    assert achieved_widths(text, direction="store", space="global", dtype_bits=16) == {}


def test_gate_report_hard_fails_on_underwidth():
    # a real CRC N-contig store report (b128-ideal VW=4 f32); if the ASM only reached dwordx2 the gate RAISES.
    native = _native_c()
    rep = analyze_coalescing(
        _shuffle(native, interleave_idx(1, 4, 64)),
        ("M", "N"),
        (64, 1),
        32,
        direction="store",
        line_bytes=128,
    )
    assert rep.ideal_vw_elems == 4
    good = "global_store_dwordx4 v[0:3], v4, s[0:1]\n" * 16
    _r, hist, note = gate_report(rep, good, space="global")  # matches ideal -> passes
    assert hist == {4: 16} and "consistent" in note
    bad = "global_store_dwordx2 v[0:1], v4, s[0:1]\n" * 32
    with pytest.raises(AssertionError, match="ASM does not back"):
        gate_report(rep, bad, space="global")


def test_gate_report_raises_when_asm_lacks_the_access():
    native = _native_c()
    rep = analyze_coalescing(
        _shuffle(native, interleave_idx(1, 4, 64)),
        ("M", "N"),
        (64, 1),
        32,
        direction="store",
        line_bytes=128,
    )
    with pytest.raises(AssertionError, match="ASM shows NO"):
        gate_report(rep, "s_nop 0\n", space="global")


def _gemm_operand_loads():
    """A and B of one wave-tile MMA, loaded from K-contiguous global tensors and fed to TileMma.

    The wave tile is (M, N, K) = (32, 64, 8) over 32x32x8 atoms, so A's and B's tile descriptors differ
    and the probe can attribute each load to its operand. Both tensors are written in the GEMM axis
    order, A (M, K) and B (N, K), with stride 1 on K.
    """
    mma = TileMma(
        (32, 64, 8),
        a="f16",
        b="f16",
        c="f32",
        target="gfx90a",
        tiling=Tiling(atom_shape=(32, 32, 8)),
    )
    m_len, n_len, k_len = mma.shape
    b = IRBuilder("coalescing_probe_operands")
    a_ptr = b.param("A", PtrType(F16, "global"))
    b_ptr = b.param("B", PtrType(F16, "global"))
    lane = b.thread_id_x()
    a_td = make_tensor_desc((m_len, k_len), (k_len, 1), F16)
    b_td = make_tensor_desc((n_len, k_len), (k_len, 1), F16)
    a_fragment = load_fragment(b, a_ptr, make_window(a_td, (0, 0)), mma.a_desc, lane)
    b_fragment = load_fragment(b, b_ptr, make_window(b_td, (0, 0)), mma.b_desc, lane)
    accumulator = make_fragment(mma.c_desc, F32)
    fill_fragment(b, accumulator, 0)
    mma(b, a_fragment, b_fragment, accumulator)
    b.ret()
    return b.kernel


def test_report_names_b_axes_in_gemm_order():
    # The probe names a recorded load's axes from the MMA operand it feeds. B's tile descriptor is
    # (N, K), so a K-contiguous B must be reported with axes (N, K) and its stride-1 axis K; naming
    # B's axes (K, N) would report the same strides as N-contiguous.
    _kernel, pipeline = record_build(_gemm_operand_loads)
    loads = [t for t in pipeline.transactions if t.kind == "load"]
    a_report, b_report = (
        report_for_transaction(t, arch="gfx90a", pipeline=pipeline) for t in loads
    )
    assert a_report.dims == ("M", "K") and a_report.stride1_axis == "K"
    assert b_report.dims == ("N", "K") and b_report.stride1_axis == "K"
