# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import re

import pytest

from config_harness import derive_states, emit_kernels_from_config


pytestmark = pytest.mark.unit

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx950",
    "streamk_dtl_pgr2.yaml",
)

_DTL_LOAD = re.compile(r"^buffer_load\S* [^/\n]* lds\b", re.M)


@pytest.fixture(scope="module")
def streamk_dtl_result():
    states = derive_states(_CONFIG, arch="gfx950", limit_solutions=1)
    assert len(states) == 1
    assert {
        key: states[0][key]
        for key in ("PrefetchGlobalRead", "DirectToLdsA", "DirectToLdsB", "1LDSBuffer", "NumLdsBlk")
    } == {
        "PrefetchGlobalRead": 2,
        "DirectToLdsA": True,
        "DirectToLdsB": True,
        "1LDSBuffer": 0,
        "NumLdsBlk": 3,
    }

    results = emit_kernels_from_config(_CONFIG, limit=1, arch="gfx950")
    assert len(results) == 1
    return results[0]


def test_streamk_dtl_first_prefetch_waits_for_previous_tile_lds_reads(streamk_dtl_result):
    """ROCM-32152, ROCM-32277: the next tile's first DirectToLds prefetch overwrites
    LDS that slower waves may still read in the previous tile's epilogue, so every
    wave must finish its ds_reads and meet at a barrier before the prefetch."""
    _base, source, error = streamk_dtl_result
    assert error == 0

    loop_start = source.index("label_PersistentLoopStart:")
    first_dtl_load = _DTL_LOAD.search(source, loop_start)
    assert first_dtl_load is not None, "no DirectToLds load after the persistent loop start"
    setup = source[loop_start : first_dtl_load.start()]
    barrier = setup.rfind("s_barrier")
    assert barrier != -1, "no s_barrier between the persistent loop start and the first DirectToLds load"
    assert "s_waitcnt lgkmcnt(0)" in setup[:barrier], "no LDS wait before that barrier"
