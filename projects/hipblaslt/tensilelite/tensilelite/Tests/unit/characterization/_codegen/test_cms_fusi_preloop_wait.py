# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os

import pytest

from config_harness import derive_states, emit_kernels_from_config


pytestmark = pytest.mark.unit

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx950",
    "cms_fusi_preloop_wait.yaml",
)
_PREPACK_CONFIG = (
    "Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling_tf32.yaml"
)


@pytest.fixture(scope="module")
def cms_fusi_results():
    results = emit_kernels_from_config(_CONFIG, limit=2, arch="gfx950")
    assert len(results) == 2
    return results


def test_cms_fusi_preloop_wait_dominates_all_entry_branches(cms_fusi_results):
    results = [
        result for result in cms_fusi_results
        if "label_LoopBody_NTA0_NTB0:" not in result[1]
    ]
    assert len(results) == 1

    _base, source, error = results[0]
    assert error == 0

    preloop_start = source.index("/* local read prefetch a */")
    open_loop = source.index("label_openLoopL:", preloop_start)
    loop_begin = source.index("label_LoopBeginL:", open_loop)
    first_mfma = source.index("v_mfma", loop_begin)

    preloop = source[preloop_start:open_loop]
    preloop_reads = [
        line.strip()
        for line in preloop.splitlines()
        if line.strip().startswith("ds_read_") and "vgprValu" in line
    ]
    assert len(preloop_reads) == 16
    assert all("vgprValuA" in line for line in preloop_reads[:8])
    assert all("vgprValuB" in line for line in preloop_reads[8:])

    wait_comment = "complete one-time pre-loop local reads"
    assert source.count(wait_comment) == 1
    wait_line = next(line for line in preloop.splitlines() if wait_comment in line)
    assert wait_line.strip().startswith("s_waitcnt lgkmcnt(0)")
    wait = source.index(wait_comment, preloop_start)
    last_preloop_read = source.rindex("ds_read_", preloop_start, open_loop)
    first_entry_branch = source.index("s_cbranch", open_loop)

    assert last_preloop_read < wait < open_loop
    assert wait < first_entry_branch < loop_begin < first_mfma

    entry = source[open_loop:loop_begin]
    assert "label_toPGR1" in entry
    assert "label_LoopEndL" in entry


def test_cms_fusi_adaptive_wait_dominates_each_entry_body(cms_fusi_results):
    results = [
        result for result in cms_fusi_results
        if "label_LoopBody_NTA0_NTB0:" in result[1]
    ]
    assert len(results) == 1

    _base, source, error = results[0]
    assert error == 0

    wait_comment = "complete one-time pre-loop local reads"
    nt_combos = ((0, 0), (0, 4), (4, 0))
    assert source.count(wait_comment) == len(nt_combos)

    for index, (nta, ntb) in enumerate(nt_combos):
        body_label = f"label_LoopBody_NTA{nta}_NTB{ntb}:"
        next_label = (
            f"label_LoopBody_NTA{nt_combos[index + 1][0]}_"
            f"NTB{nt_combos[index + 1][1]}:"
            if index + 1 < len(nt_combos)
            else "label_LoopBody_NTA_NTB_Done:"
        )
        body_start = source.index(body_label)
        body_end = source.index(next_label, body_start)
        body = source[body_start:body_end]

        open_loop = body.index(f"label_openLoopL_NTA{nta}_NTB{ntb}:")
        loop_begin = body.index(f"label_LoopBeginL_NTA{nta}_NTB{ntb}:")
        first_mfma = body.index("v_mfma", loop_begin)
        wait = body.index(wait_comment)
        first_entry_branch = body.index("s_cbranch", open_loop)

        assert body.count(wait_comment) == 1
        assert wait < open_loop < first_entry_branch < loop_begin < first_mfma


def test_cms_fusi_prepack_drain_does_not_get_a_second_wait():
    states = derive_states(_PREPACK_CONFIG, arch="gfx950", limit_solutions=1)
    assert len(states) == 1
    assert {
        key: states[0][key]
        for key in (
            "UseCustomMainLoopSchedule",
            "ForceUnrollSubIter",
            "UsePLRPack",
        )
    } == {
        "UseCustomMainLoopSchedule": 1,
        "ForceUnrollSubIter": True,
        "UsePLRPack": 1,
    }

    results = emit_kernels_from_config(_PREPACK_CONFIG, limit=1, arch="gfx950")
    assert len(results) == 1

    _base, source, error = results[0]
    assert error == 0

    prepack_comment = "Wait for LRA and LRB to complete (for pre Pack code)"
    fallback_comment = "complete one-time pre-loop local reads"
    assert source.count(prepack_comment) == 1
    assert fallback_comment not in source
    prepack_line = next(line for line in source.splitlines() if prepack_comment in line)
    assert prepack_line.strip().startswith("s_waitcnt lgkmcnt(0)")

    preloop_start = source.index("/* local read prefetch a */")
    prepack_wait = source.index(prepack_comment, preloop_start)
    open_loop = source.index("label_openLoopL:", prepack_wait)
    first_mfma = source.index("v_mfma", open_loop)
    assert prepack_wait < open_loop < first_mfma
