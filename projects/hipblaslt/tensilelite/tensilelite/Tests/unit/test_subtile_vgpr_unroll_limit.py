# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Regression: gfx950 MXFP8 skinny tiles with an uneven multi-DU partition split.

The free-list VGPR tile assignment of such a split repeats with a period past
the unroll limit; it used to abort codegen with an AssertionError.
"""

import pytest

from tensilelite.Components.Subtile.LogicalScheduler import (
    LogicalScheduler,
    Pass,
    ReadGranularity,
    SchedulerConfig,
    VgprTileUnrollLimitError,
)

pytestmark = pytest.mark.unit


def _mxfp8_skinny_cfg(partitionSizeM):
    """Scheduler config of the gfx950 MXFP8 MT448x128 DU256 PGR1 kernel (MIWaveTile [28,2])."""
    gran = ReadGranularity
    return SchedulerConfig(
        numMFMATilesM=28, numMFMATilesN=2, numSubIterK=1,
        lrA=gran(mn=1, k=1), lrB=gran(mn=1, k=1),
        grA=gran(mn=2, k=1), grB=gran(mn=1, k=1),
        lrSA=gran(mn=2, k=2), lrSB=gran(mn=2, k=2),
        grSA=gran(mn=28, k=2), grSB=gran(mn=2, k=2),
        partitionSizeM=partitionSizeM, partitionSizeN=2, pgr=1)


def test_slow_rotating_partition_raises_runtime_error():
    cfg = _mxfp8_skinny_cfg(10)
    assert cfg.partitionSizesM == [10, 10, 8]
    with pytest.raises(VgprTileUnrollLimitError):
        LogicalScheduler(cfg).build(stop_after=Pass.VGPR_TILES)
    # Per-kernel codegen only rejects RuntimeError; anything else aborts the job.
    assert issubclass(VgprTileUnrollLimitError, RuntimeError)


@pytest.mark.parametrize("partitionSizeM, sizes, unroll", [
    (14, [14, 14], 2),
    (12, [12, 12, 4], 6),
    (8, [8, 8, 8, 4], 4),
])
def test_uneven_partition_unroll_matches_partition_size(partitionSizeM, sizes, unroll):
    cfg = _mxfp8_skinny_cfg(partitionSizeM)
    sched = LogicalScheduler(cfg)
    sched.build(stop_after=Pass.VGPR_TILES)
    assert cfg.partitionSizesM == sizes
    assert sched.unroll_factor == unroll
