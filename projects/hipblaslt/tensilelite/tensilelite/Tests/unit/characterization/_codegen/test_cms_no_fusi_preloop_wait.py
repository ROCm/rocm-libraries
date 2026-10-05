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
    "cms_no_fusi_preloop_wait.yaml",
)


@pytest.fixture(scope="module")
def cms_no_fusi_result():
    states = derive_states(_CONFIG, arch="gfx950", limit_solutions=1)
    assert len(states) == 1
    assert {
        key: states[0][key]
        for key in (
            "UseCustomMainLoopSchedule",
            "ForceUnrollSubIter",
            "PrefetchLocalRead",
            "UsePLRPack",
            "DirectToLdsA",
            "DirectToLdsB",
        )
    } == {
        "UseCustomMainLoopSchedule": 1,
        "ForceUnrollSubIter": False,
        "PrefetchLocalRead": 1,
        "UsePLRPack": 0,
        "DirectToLdsA": True,
        "DirectToLdsB": True,
    }

    results = emit_kernels_from_config(_CONFIG, limit=1, arch="gfx950")
    assert len(results) == 1
    return results[0]


def test_cms_no_fusi_preloop_wait_dominates_all_entry_branches(cms_no_fusi_result):
    _base, source, error = cms_no_fusi_result
    assert error == 0

    preloop_start = source.index("/* local read prefetch a */")
    open_loop = source.index("label_openLoopL:", preloop_start)
    loop_begin = source.index("label_LoopBeginL:", open_loop)
    loop_end = source.index("label_LoopEndL:", loop_begin)
    first_main_mfma = source.index("v_mfma", loop_begin)
    first_no_load_mfma = source.index("v_mfma", loop_end)

    preloop = source[preloop_start:open_loop]
    preloop_reads = [
        line.strip()
        for line in preloop.splitlines()
        if line.strip().startswith("ds_read_") and "vgprValu" in line
    ]
    assert len(preloop_reads) == 16

    wait_comment = "complete one-time pre-loop local reads"
    assert source.count(wait_comment) == 1
    wait_line = next(line for line in preloop.splitlines() if wait_comment in line)
    assert wait_line.strip().startswith("s_waitcnt lgkmcnt(0)")

    wait = source.index(wait_comment, preloop_start)
    last_preloop_read = source.rindex("ds_read_", preloop_start, open_loop)
    first_entry_branch = source.index("s_cbranch", open_loop)

    assert last_preloop_read < wait < open_loop
    assert wait < first_entry_branch < loop_begin < first_main_mfma
    assert wait < loop_end < first_no_load_mfma

    entry = source[open_loop:loop_begin]
    assert "label_toPGR1" in entry
    assert "label_LoopEndL" in entry
