# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The attention capability table (rocke.helpers.attention_arch_caps).

Checks the table against the arch catalog for every attention arch, pins the
values the kernels rely on, and checks the C++ twin
(helper_rocke.helpers.attention_arch_caps.h) agrees with Python field by field
and message by message when the engine is built.
"""

from __future__ import annotations

import dataclasses

import pytest

from rocke.core.arch import LDS_DMA_ENCODED_DWORDS, ArchTarget
from rocke.helpers import attention_arch_caps as C

ARCHES = ("gfx942", "gfx950", "gfx1151", "gfx1201", "gfx1250")

# arch: (wave, path, LDS-DMA dwords, transpose-read bits, AGPRs)
EXPECTED = {
    "gfx942": (64, "mfma", (1,), (), 256),
    "gfx950": (64, "mfma", (1, 3, 4), (64,), 256),
    "gfx1151": (32, "wmma", (), (), 0),
    "gfx1201": (32, "wmma", (), (), 0),
    "gfx1250": (32, "wmma", (), (), 0),
}


def test_arches_are_the_ticketed_set():
    assert C.ATTN_CAPS_ARCHES == ARCHES


@pytest.mark.parametrize("arch", ARCHES)
def test_row_matches_the_arch_catalog(arch):
    c = C.attn_arch_caps(arch)
    t = ArchTarget.from_gfx(arch)
    assert c.wave_size == t.wave_size
    assert c.matrix_path == ("mfma" if t.has_mfma else "wmma")
    assert c.lds_capacity_bytes == t.lds_capacity_bytes
    assert set(c.lds_dma_dwords) <= set(LDS_DMA_ENCODED_DWORDS)
    assert list(c.lds_dma_dwords) == sorted(c.lds_dma_dwords)
    assert (c.lds_dma_dwords == ()) == (not t.memory.has_async_lds)
    if c.lds_dma_dwords:
        assert c.max_lds_dma_dwords == t.async_lds_max_dwords
    assert c.has_tr_read_b64 == t.memory.has_ds_read_tr
    assert c.agprs == t.limits.agprs
    assert c.arch_vgprs == t.limits.vgprs - t.limits.agprs
    for dtype in ("fp16", "bf16"):
        for op in C.attn_atoms(arch, dtype):
            assert (op.m, op.n) == (16, 16)
            assert op.wave_size == t.wave_size
            assert op in t.mma.ops


@pytest.mark.parametrize("arch", ARCHES)
def test_expected_values(arch):
    wave, path, dma, tr, agprs = EXPECTED[arch]
    c = C.attn_arch_caps(arch)
    assert (c.wave_size, c.matrix_path, c.lds_dma_dwords, c.tr_read_bits, c.agprs) == (
        wave,
        path,
        dma,
        tr,
        agprs,
    )
    assert c.native_fp32_atomic_add is True
    # The native 16x16x32 atom: MFMA on gfx950, WMMA on gfx1250.
    assert C.wide_k_atom_available(arch, "bf16") == (arch in ("gfx950", "gfx1250"))


def test_unknown_arch_is_refused():
    with pytest.raises(ValueError, match="no attention capability row for 'gfx90a'"):
        C.attn_arch_caps("gfx90a")


def test_asm_hazard_table():
    assert C.mfma_result_wait_states("gfx942", 16) == 7
    assert C.mfma_result_wait_states("gfx950", 16) == 8
    assert C.mfma_result_wait_states("gfx950", 32) == 8
    assert C.asm_tail_nop_min("gfx942", "bf16", 16) == 6
    assert C.asm_tail_nop_min("gfx950", "fp16", 32) == 7
    with pytest.raises(ValueError, match="gfx950-only"):
        C.mfma_asm_mnemonic("gfx942", "bf16", 32)
    for arch in ("gfx1151", "gfx1201", "gfx1250"):
        with pytest.raises(ValueError, match="has no AGPRs"):
            C.mfma_asm_mnemonic(arch, "bf16", 16)
    with pytest.raises(ValueError, match="legal: 6..15"):
        C.check_asm_tail_nop("gfx942", "bf16", 16, 5)


# ----------------------------------------------------------------- C++ twin


def _py_check(fn, arch, dtype, a, k):
    """Run one Python check as (ok, value, message), like the binding."""
    try:
        if fn == "check_lds_bytes":
            return True, C.check_lds_bytes(arch, a), ""
        if fn == "check_lds_dma_width":
            return True, C.check_lds_dma_width(arch, a), ""
        if fn == "check_transpose_read":
            return True, C.check_transpose_read(arch, a), ""
        if fn == "wide_k_atom_available":
            return True, C.wide_k_atom_available(arch, dtype), ""
        if fn == "mfma_asm_mnemonic":
            return True, C.mfma_asm_mnemonic(arch, dtype, k), ""
        if fn == "mfma_result_wait_states":
            return True, C.mfma_result_wait_states(arch, k), ""
        if fn == "asm_tail_nop_min":
            return True, C.asm_tail_nop_min(arch, dtype, k), ""
        if fn == "check_asm_tail_nop":
            return True, C.check_asm_tail_nop(arch, dtype, k, a), ""
    except ValueError as e:
        return False, None, str(e)
    raise AssertionError(fn)


def _check_cases():
    for arch in ARCHES:
        c = C.attn_arch_caps(arch)
        for n in (0, c.lds_capacity_bytes, c.lds_capacity_bytes + 1, -1):
            yield ("check_lds_bytes", arch, "", n, 0)
        for d in (1, 2, 3, 4):
            yield ("check_lds_dma_width", arch, "", d, 0)
        for bits in (32, 64, 128):
            yield ("check_transpose_read", arch, "", bits, 0)
        for dt in ("fp16", "bf16"):
            yield ("wide_k_atom_available", arch, dt, 0, 0)
            for k in (16, 32):
                yield ("mfma_asm_mnemonic", arch, dt, 0, k)
                yield ("asm_tail_nop_min", arch, dt, 0, k)
                for nop in (5, 6, 7, 15, 16):
                    yield ("check_asm_tail_nop", arch, dt, nop, k)
        yield ("mfma_asm_mnemonic", arch, "fp8", 0, 16)
        for k in (16, 32):
            yield ("mfma_result_wait_states", arch, "", 0, k)


def test_cpp_twin_matches_python():
    engine = pytest.importorskip("rocke_engine")
    if not hasattr(engine, "attention_arch_caps"):
        pytest.skip("rocke_engine predates the attention capability table")
    for arch in ARCHES:
        py = dataclasses.asdict(C.attn_arch_caps(arch))
        assert engine.attention_arch_caps(arch) == py, arch
    with pytest.raises(ValueError) as cpp_err:
        engine.attention_arch_caps("gfx90a")
    with pytest.raises(ValueError) as py_err:
        C.attn_arch_caps("gfx90a")
    assert str(cpp_err.value) == str(py_err.value)
    for case in _check_cases():
        py_ok, py_value, py_msg = _py_check(*case)
        cpp_ok, cpp_value, cpp_msg = engine.attention_arch_caps_check(*case)
        assert (cpp_ok, cpp_msg) == (py_ok, py_msg), case
        if py_ok and py_value is not None:
            assert cpp_value == py_value, case
