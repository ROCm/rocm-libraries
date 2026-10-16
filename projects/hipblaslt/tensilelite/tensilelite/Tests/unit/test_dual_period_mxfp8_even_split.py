################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
################################################################################

"""The dual-period MXFP8 partition split must never produce an odd cycle.

Partition-directed prefetch is a cycle: partition pi local-reads the operands
that partition pi+1 will consume, and the last partition wraps to partition 0 of
the next DepthU.  Which operand VGPR set a partition uses is decided by its
index parity, so the cycle is 2-colourable only when its length is even.  An odd
cycle makes the last partition prefetch into the very set it is still reading --
the operands stay in bounds and nothing faults, the result is just wrong.

Halving the MIWaveTile dim and rounding DOWN to the local-read granularity used
to produce three partitions for every dim whose dim/mn is odd (6, 10, 14, 18).
These tests pin the rounding direction and the even-cycle invariant it exists to
guarantee.
"""

import pytest

from tensilelite.Components.Subtile.LogicalScheduler import SchedulerConfig

# The MX scale local read fetches two M-tiles at a time, so partition sizes must
# be multiples of this.  It is MXScaleLRGeometry.subtileShape[0] in the emitter.
MN = 2

# MIWaveTile dims reachable from the gfx950 dual-period MXFP8 macrotiles.
DIMS = list(range(1, 21))

# dim/mn odd -> the pre-fix rounding split these three ways.
ODD_BEFORE = [6, 10, 14, 18]


def _sizes(dim, mn=MN):
    """Partition sizes the scheduler derives for a MIWaveTile dim."""
    return SchedulerConfig._normalize_partition_sizes(
        SchedulerConfig._even_split_size(dim, mn), dim, 'M', mn)


@pytest.mark.parametrize("dim", DIMS)
def test_split_is_never_an_odd_cycle(dim):
    """Every dim splits into an even number of partitions, or into exactly one.

    A single partition is fine: there is no multi-node cycle to colour.
    """
    sizes = _sizes(dim)
    assert sum(sizes) == dim, f"partitions must cover the dim, got {sizes}"
    assert len(sizes) == 1 or len(sizes) % 2 == 0, \
        f"dim {dim} split into an odd {len(sizes)}-cycle: {sizes}"


@pytest.mark.parametrize("dim,expected", [(6, [4, 2]), (10, [6, 4]),
                                          (14, [8, 6]), (18, [10, 8])])
def test_dims_that_used_to_split_three_ways(dim, expected):
    """The regression itself: these produced [2,2,2] / [4,2,4] / [6,2,6] / [8,2,8]."""
    assert _sizes(dim) == expected


@pytest.mark.parametrize("dim", DIMS)
def test_partition_sizes_respect_local_read_granularity(dim):
    """Every partition is a whole number of MX scale local reads, when split."""
    sizes = _sizes(dim)
    if len(sizes) > 1:
        assert all(s % MN == 0 for s in sizes), \
            f"dim {dim} produced sizes not divisible by mn={MN}: {sizes}"


def test_rounding_is_up_not_down():
    """Pin the direction: rounding down is what created the odd splits."""
    for dim in ODD_BEFORE:
        down = max(1, dim // 2)
        up = SchedulerConfig._even_split_size(dim, MN)
        assert up > down, f"dim {dim}: expected round-up, got {up} <= {down}"
        assert len(SchedulerConfig._normalize_partition_sizes(down, dim, 'M', MN)) == 3, \
            f"dim {dim}: round-down no longer reproduces the 3-way split"


@pytest.mark.parametrize("mn", [1, 2, 4])
def test_even_split_size_is_in_range(mn):
    """The helper never returns a size outside [1, total]."""
    for dim in DIMS:
        s = SchedulerConfig._even_split_size(dim, mn)
        assert 1 <= s <= dim, f"dim {dim}, mn {mn}: size {s} out of range"
        assert s % mn == 0 or s == dim, \
            f"dim {dim}, mn {mn}: size {s} is not an mn multiple"
