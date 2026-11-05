# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``kernels.gfx950.dsv41_mono.moe_floor``: the V4.1 routed-MoE bytes floor (W12 probe P4). CPU only (byte model and
the emitted LLVM text).

The GPU proofs (exact XOR-fold coverage of every timed arm at two union sizes, the must-fail twins ``skip3`` /
``stride``, timing against ATOM's ``fused_moe`` in the same job) run on gfx950: ``hackathon-prep/dsv41/wide/p4_floor.py``.
Results: ``experiments/dsv41-mono/wide/P4.md``.
"""

from __future__ import annotations

import pytest
from kernels.gfx950.dsv41_mono import moe_floor as F
from rocke.core.lower_llvm import lower_kernel_to_llvm


def _ir(**kn):
    return lower_kernel_to_llvm(F.build_floor_kernel(F.FloorKnobs(**kn)), arch="gfx950")


def _loads(text):
    return [x for x in text.splitlines() if "@llvm.amdgcn.raw.ptr.buffer.load.v4i32(" in x and "call" in x]


def test_expert_bytes_served_shape():
    # ATOM serves FusedMoE(pad_align=128): TP2 intermediate 1152 unpadded, TP1 2304
    assert F.expert_bytes(2) == {"w13": 2 * 1152 * 2560, "s13": 2 * 1152 * 160, "w2": 5120 * 576, "s2": 5120 * 36}
    assert F.floor_bytes(2, 1) == 9400320
    assert F.floor_bytes(1, 1) == 2 * 9400320
    # the class default pad 256 (not served) pads TP2 to 1280: 1/9 more bytes
    assert F.floor_bytes(2, 9, pad_align=256) == F.floor_bytes(2, 10)


def test_regions_are_whole_lanes():
    for tp in (1, 2):
        for v in F.expert_bytes(tp).values():
            assert v % 16 == 0


def test_knob_checks():
    for bad in ({"depth": 0}, {"depth": 13}, {"unroll": 3}, {"wcp": "sc1"}, {"mutant": "skip3"}, {"mutant": "x", "dbg": True}):
        with pytest.raises(ValueError):
            F.FloorKnobs(**bad).check()
    F.FloorKnobs(dbg=True, mutant="stride").check()


def test_names_list_deviations():
    assert F.floor_name(F.FloorKnobs()) == "dsv41_moe_floor_d4_u1_nt"
    assert F.floor_name(F.FloorKnobs(depth=8, unroll=2, wcp="default", dbg=True, mutant="skip3")) == \
        "dsv41_moe_floor_d8_u2_default_dbg_mskip3"


@pytest.mark.parametrize("depth,unroll", [(4, 1), (6, 2), (4, 4)])
def test_four_regions_each_with_a_ring_of_nt_loads(depth, unroll):
    t = _ir(depth=depth, unroll=unroll)
    loads = _loads(t)
    # per region: depth * unroll loads in the prologue + unroll refills in the loop
    assert len(loads) == 4 * (depth * unroll + unroll)
    assert all(x.rstrip().endswith("i32 2)") for x in loads)  # nontemporal (CPOL_NT)
    d = _ir(depth=depth, unroll=unroll, wcp="default")
    assert all(x.rstrip().endswith("i32 0)") for x in _loads(d))


def test_skip3_drops_a_region_and_stride_changes_the_ir():
    base = _ir(dbg=True)
    assert len(_loads(_ir(dbg=True, mutant="skip3"))) == 3 * len(_loads(base)) // 4
    assert _ir(dbg=True, mutant="stride") != base
    assert _ir(dbg=True) != _ir()  # dbg stores every lane's fold
