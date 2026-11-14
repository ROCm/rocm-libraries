# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import re

import pytest

from config_harness import emit_kernels_from_config


pytestmark = pytest.mark.unit

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx950",
    "dtl_pgr4_main_loop_war.yaml",
)

_DS_READ = re.compile(r"^ds_read\S*", re.M)
_BARRIER = re.compile(r"^s_barrier\b", re.M)


def test_dtl_main_loop_barrier_waits_for_local_reads():
    """DirectToLds loads after a main-loop barrier refill an LDS buffer that this
    or the previous trip read before it, and s_barrier does not wait for
    outstanding ds_reads, so each barrier must follow an LDS wait for them.
    rocjitsu reports the missing wait as an LDS race on gfx950. Both
    ScheduleGROverBarrier settings."""
    results = emit_kernels_from_config(_CONFIG, limit=2, arch="gfx950")
    assert len(results) == 2
    for base, source, error in results:
        assert error == 0, base
        loop = source[source.index("label_LoopBeginL:") : source.index("label_LoopEndL:")]
        # The loop body repeats, so reads at its end precede barriers at its start.
        trips = loop + loop
        barriers = [m.start() + len(loop) for m in _BARRIER.finditer(loop)]
        assert barriers, base
        for barrier in barriers:
            last_read = [m.start() for m in _DS_READ.finditer(trips, 0, barrier)][-1]
            assert (
                "s_waitcnt lgkmcnt(0)" in trips[last_read:barrier]
            ), f"{base}: a main-loop barrier leaves earlier ds_reads outstanding"
