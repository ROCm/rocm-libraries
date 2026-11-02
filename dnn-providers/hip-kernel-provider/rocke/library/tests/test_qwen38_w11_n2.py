# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the N2 (out_proj K-split) stage functions (``qwen38_mono/stages2/n2.py``, true-mono WP17)."""

from __future__ import annotations

import os

import pytest

from kernels.gfx950.qwen38_mono import stage_b3 as SB
from kernels.gfx950.qwen38_mono.stages2 import gemv as GV
from kernels.gfx950.qwen38_mono.stages2 import n2 as N2

pytest.importorskip("rocke.core.lower_llvm")
from rocke.core.lower_llvm import lower_kernel_to_llvm  # noqa: E402

os.environ.setdefault("ROCKE_LLVM_FLAVOR", "llvm22")

NUMA_SPECS = {v: SB.TUNED_SPECS_NUMA[v] for v in ("r1", "r2", "r3", "r4", "r2s", "r3s", "r4s")}


def _layout():
    from kernels.gfx950.qwen38_mono.mono_layout import from_atom

    try:
        return from_atom(None)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ATOM layout not importable: {e}")


def _shape(v):
    r, s, g = GV.parse_variant(v)
    return SB.B3Shape(r, s, g)


@pytest.mark.parametrize("variant", sorted(NUMA_SPECS))
def test_off_path_is_the_stock_b3(variant):
    """n2=None through the patched emit lowers to exactly stage_b3's IR (the hooks are Python-level only)."""
    lay = _layout()
    k = SB.knobs_from_spec(NUMA_SPECS[variant])
    sh = _shape(variant)
    for part in ("body", "pre") if k.pre == "dma" else ("body",):
        ref = lower_kernel_to_llvm(SB.build_b3_function_kernel(sh, k, lay, part), arch="gfx950")
        got = lower_kernel_to_llvm(N2.build_b3n2_kernel(sh, k, None, lay, part), arch="gfx950")
        assert ref == got


@pytest.mark.parametrize("variant", sorted(NUMA_SPECS))
def test_b3n2_lowers_with_one_mfma_per_tile(variant):
    lay = _layout()
    k = SB.knobs_from_spec(NUMA_SPECS[variant])
    ir = lower_kernel_to_llvm(N2.build_b3n2_kernel(_shape(variant), k, N2.N2Knobs(), lay), arch="gfx950")
    calls = [ln for ln in ir.splitlines() if "mfma" in ln and "call" in ln]
    assert len(calls) == N2.TILES // k.split // 8


@pytest.mark.parametrize("variant", sorted(NUMA_SPECS))
@pytest.mark.parametrize("kind", ["delta", "delta_ple"])
def test_c1n2_lowers_without_gemv(variant, kind):
    r, s, g = GV.parse_variant(variant)
    sh = GV.GemvShape("c1", kind, r, s, g)
    ir = lower_kernel_to_llvm(N2.build_c1n2_kernel(sh, GV.GemvKnobs(numa=1, mb_impl="ir"), N2.N2Knobs()), arch="gfx950")
    assert "mfma" not in ir


def test_mutants_change_ir():
    lay = _layout()
    k = SB.knobs_from_spec(NUMA_SPECS["r4s"])
    sh = _shape("r4s")
    base_b3 = lower_kernel_to_llvm(N2.build_b3n2_kernel(sh, k, N2.N2Knobs(), lay), arch="gfx950")
    for m in ("kit", "ascale"):
        assert lower_kernel_to_llvm(N2.build_b3n2_kernel(sh, k, N2.N2Knobs(mutate=m), lay), arch="gfx950") != base_b3
    csh = GV.GemvShape("c1", "delta", 4, True, 0)
    ck = GV.GemvKnobs(numa=1, mb_impl="ir")
    base_c1 = lower_kernel_to_llvm(N2.build_c1n2_kernel(csh, ck, N2.N2Knobs()), arch="gfx950")
    for m in ("drop", "dup"):
        assert lower_kernel_to_llvm(N2.build_c1n2_kernel(csh, ck, N2.N2Knobs(mutate=m)), arch="gfx950") != base_c1


def test_region_bounds_against_atom():
    try:
        from atom.models.qwen4_exp_mono.kernels import layout as LY
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ATOM not importable: {e}")
    N2.check_region(LY.mailbox_bytes(LY.MAX_ROWS), 0)
    assert N2.N2_OFF % 16384 == 0
    with pytest.raises(ValueError):
        N2.check_region(N2.N2_OFF + 1, 0)


def test_refusals_and_plan():
    lay = _layout()
    with pytest.raises(ValueError):
        N2.build_b3n2_kernel(_shape("r4s"), SB.knobs_from_spec(SB.TUNED_SPECS["r4s"]), N2.N2Knobs(), lay)
    with pytest.raises(ValueError):
        N2.N2Knobs(hpt=5)
    with pytest.raises(ValueError):
        N2.plan(4, 1)  # 48 groups x 16 x 4 rows > 512 threads
    for rows in (1, 2, 3, 4):
        g, h = N2.plan(rows)
        assert g * h == N2.HEADS and rows * 16 * g <= 512
    assert N2.knobs_from_spec(N2.knobs_spec(N2.N2Knobs(issue="loader", hpt=3))) == N2.N2Knobs(issue="loader", hpt=3)
    assert N2.pair_id(N2.N2Knobs()) != N2.pair_id(N2.N2Knobs(mutate="drop"))


@pytest.mark.parametrize("variant", ["r1", "r4s"])
@pytest.mark.parametrize("kind", ["delta", "qsa"])
def test_c1s_twin_keeps_the_stock_chain(variant, kind):
    """The C1 streaming twin runs the stock chain: 12 MFMA steps per chain wave, operands from the mailboxes."""
    r, s, g = GV.parse_variant(variant)
    sh = GV.GemvShape("c1", kind, r, s, g)
    ck = GV.GemvKnobs(numa=1, mb_impl="ir")
    for sg in (1, 3, 12):
        ir = lower_kernel_to_llvm(N2.build_c1s_kernel(sh, ck, N2.C1sKnobs(sgrp=sg)), arch="gfx950")
        assert len([ln for ln in ir.splitlines() if "mfma" in ln and "call" in ln]) == N2.CH
    base = lower_kernel_to_llvm(N2.build_c1s_kernel(sh, ck, N2.C1sKnobs()), arch="gfx950")
    for m in ("kit", "exp"):
        assert lower_kernel_to_llvm(N2.build_c1s_kernel(sh, ck, N2.C1sKnobs(mutate=m)), arch="gfx950") != base
    with pytest.raises(ValueError):
        N2.C1sKnobs(sgrp=5)
