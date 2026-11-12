################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################

"""Scheduling contracts of the uniform PGR2 epoch.

The uniform epoch issues every A/B global read for a body's future epoch
before the first slot that consumes it, so one VMEM wait at that frontier can
publish the whole epoch.  Three pieces have to agree for that to be correct:

  * the frontier is the *earliest* next-epoch A/B read, not merely some read;
  * the wait placed there counts exactly the requests issued before it, and
    counts them per unrolled body, because a dual-period body omits the scale
    requests the first body issues;
  * the first future-write cohort is legal in partition zero, which means any
    current-epoch read overlapping it is hoisted there first -- splitting a
    read that only partly overlaps, rather than hoisting or deferring it whole.

These exercise the scheduler's placement structures directly.  The assembly
level is covered by test_dual_period_mxfp8_emit.py; this file pins the
decisions that produce it, and runs without an assembler.
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensilelite.Components.Subtile.InstructionEmitter import (
    SCALE_SUBTILES_PER_GROUP,
    InstructionEmitter,
)
from tensilelite.Components.Subtile.LogicalScheduler import (
    GRPlacement,
    LogicalScheduler,
    LRPlacement,
    MFMATileRange,
    ReadGranularity,
    SchedulerConfig,
    SubIterKSlot,
    WaitGRCounts,
    WaitGROp,
)


def _config(**over):
    """A minimal uniform-epoch config; A/B load one tile over one K step.

    Partition-directed prefetch pins numSubIterK to 1 and double-buffers A/B,
    so the flat slot index is just the partition index.
    """
    params = dict(
        numMFMATilesM=8, numMFMATilesN=8, numSubIterK=1,
        lrA=ReadGranularity(mn=2, k=1), lrB=ReadGranularity(mn=2, k=1),
        grA=ReadGranularity(mn=1, k=1), grB=ReadGranularity(mn=1, k=1),
        grSA=ReadGranularity(mn=1, k=1), grSB=ReadGranularity(mn=1, k=1),
        # 8 tiles split 4+4 on each axis gives a 2x2 grid, so numPartitions=4.
        partitionSizeM=4, partitionSizeN=4,
        pgr=2, plr=0, prefetchNextPartition=True,
        doubleBufferTensors=('A', 'B'), uniformPgrEpoch=True,
    )
    params.update(over)
    return SchedulerConfig(**params)


def _cohort2_config(**over):
    """Shaped so the derived cohort is two atoms.

    4+4 single-tile A/B requests over a 2x2 partition grid give ceil(8/4)=2.
    Keeping the granularity at one tile means an atom is a tile, so the
    hoist's tile ranges stay readable.
    """
    params = dict(numMFMATilesM=4, numMFMATilesN=4,
                  partitionSizeM=2, partitionSizeN=2)
    params.update(over)
    return _config(**params)


def _single_partition_config():
    """A non-partitioned config, for the hoist pass's early return."""
    return SchedulerConfig(
        numMFMATilesM=8, numMFMATilesN=8, numSubIterK=1,
        lrA=ReadGranularity(mn=2, k=1), lrB=ReadGranularity(mn=2, k=1),
        grA=ReadGranularity(mn=1, k=1), grB=ReadGranularity(mn=1, k=1),
        pgr=2, plr=0)


def _scheduler(cfg, partitions):
    """A scheduler positioned at the point these passes run.

    The passes under test read only the config and the placement grid, so
    bypassing __init__ keeps the fixture free of the toolchain and ISA setup
    a full construction would require.
    """
    sched = LogicalScheduler.__new__(LogicalScheduler)
    sched.config = cfg
    sched._partitions = partitions
    return sched


def _lr(tensor, mt, k_slot, t_start, t_end, partition=0):
    return LRPlacement(tensor=tensor, mtIteration=mt,
                       tiles=MFMATileRange(k_slot, k_slot + 1, t_start, t_end),
                       subIterK_slot=k_slot, partition=partition)


def _gr(tensor, mt, k_slot, t_start, t_end, partition=0):
    return GRPlacement(tensor=tensor, mtIteration=mt,
                       tiles=MFMATileRange(k_slot, k_slot + 1, t_start, t_end),
                       subIterK_slot=k_slot, partition=partition)


def _grid(numK, numPartitions):
    return [[SubIterKSlot(subIterK=k) for k in range(numK)]
            for _ in range(numPartitions)]


# ── config invariants ────────────────────────────────────────

def test_uniform_epoch_requires_pgr2():
    with pytest.raises(AssertionError, match="PrefetchGlobalRead=2"):
        _config(pgr=1)


def test_uniform_epoch_requires_partition_directed_prefetch():
    with pytest.raises(AssertionError, match="partition-directed prefetch"):
        _config(prefetchNextPartition=False)


def test_a_valid_uniform_epoch_config_is_accepted():
    assert _config().uniformPgrEpoch is True


# ── the derived cohort width ─────────────────────────────────

def test_an_atom_covers_one_read_granule():
    """A and B each contribute one request per gr granule of tiles."""
    cfg = _config(grA=ReadGranularity(mn=2, k=1),
                  grB=ReadGranularity(mn=4, k=1))
    assert cfg.uniformEpochVmemAtoms == 8 // 2 + 8 // 4


def test_a_partial_granule_still_costs_a_whole_atom():
    cfg = _config(numMFMATilesM=7, grA=ReadGranularity(mn=2, k=1))
    assert cfg.uniformEpochVmemAtoms == 4 + 8      # ceil(7/2) = 4


def test_scaled_tensors_add_one_atom_each():
    """One scale dword covers the epoch, so each scale tensor is one read."""
    plain = _config()
    scaled = _config(lrSA=ReadGranularity(mn=2, k=1),
                     lrSB=ReadGranularity(mn=2, k=1))
    assert scaled.uniformEpochVmemAtoms == plain.uniformEpochVmemAtoms + 2


def test_the_cohort_spreads_the_epoch_over_every_slot():
    """_distribute_grs sends atom i to slot i//cohort.

    The width is right exactly when the last atom lands in the last slot and
    no earlier: that is what makes the epoch span the partitions.
    """
    cfg = _config()
    atoms, cohort = cfg.uniformEpochVmemAtoms, cfg.uniformEpochVmemCohortSize
    slots = cfg.numPartitions * cfg.numSubIterK
    assert (atoms - 1) // cohort == slots - 1


@pytest.mark.parametrize("grA_mn, grB_mn, scaled, expected", [
    (2, 2, False, 2),      # 4+4 atoms over 4 slots
    (2, 1, False, 3),      # 4+8
    (1, 1, False, 4),      # 8+8
    (1, 1, True, 5),       # 8+8 plus two scale reads
])
def test_the_cohort_width_follows_the_shape(grA_mn, grB_mn, scaled, expected):
    """The width is a property of the shape, not a tuned constant."""
    extra = {}
    if scaled:
        extra = dict(lrSA=ReadGranularity(mn=2, k=1),
                     lrSB=ReadGranularity(mn=2, k=1))
    cfg = _config(grA=ReadGranularity(mn=grA_mn, k=1),
                  grB=ReadGranularity(mn=grB_mn, k=1), **extra)
    assert cfg.uniformEpochVmemCohortSize == expected


def test_fewer_atoms_than_slots_still_gives_a_cohort_of_one():
    """An epoch smaller than the grid cannot reach every slot.

    The width floors at one rather than zero, which would make the slot index
    undefined; the epoch then simply occupies the first few slots.
    """
    cfg = _config(numMFMATilesM=4, numMFMATilesN=4,
                  grA=ReadGranularity(mn=4, k=1),
                  grB=ReadGranularity(mn=4, k=1),
                  partitionSizeM=2, partitionSizeN=2)
    assert cfg.uniformEpochVmemAtoms == 2
    assert cfg.uniformEpochVmemCohortSize == 1


# ── the frontier ─────────────────────────────────────────────

def test_frontier_is_the_earliest_next_epoch_read():
    """Not just any next-epoch read: the earliest in flat slot order.

    A later partition also reads the next epoch, so a pass that took the
    first one it encountered rather than the minimum would still find a
    frontier -- and would publish the epoch after some of its consumers.
    """
    cfg = _config()
    parts = _grid(numK=1, numPartitions=4)
    parts[3][0].lrs.append(_lr('A', 1, 0, 0, 2, partition=3))
    parts[1][0].lrs.append(_lr('B', 1, 0, 0, 2, partition=1))
    # current-epoch reads must not be mistaken for the frontier
    parts[0][0].lrs.append(_lr('A', 0, 0, 0, 2, partition=0))

    # flat = partition * numSubIterK + subIterK, and numSubIterK is 1.
    assert _scheduler(cfg, parts)._uniform_epoch_frontier_slot() == 1


def test_frontier_ignores_scale_reads():
    cfg = _config()
    parts = _grid(numK=1, numPartitions=4)
    parts[0][0].lrs.append(_lr('SA', 1, 0, 0, 2, partition=0))
    parts[2][0].lrs.append(_lr('A', 1, 0, 0, 2, partition=2))
    assert _scheduler(cfg, parts)._uniform_epoch_frontier_slot() == 2


def test_frontier_requires_a_next_epoch_read():
    cfg = _config()
    parts = _grid(numK=1, numPartitions=4)
    parts[0][0].lrs.append(_lr('A', 0, 0, 0, 2, partition=0))
    with pytest.raises(ValueError, match="next-epoch LR frontier"):
        _scheduler(cfg, parts)._uniform_epoch_frontier_slot()


# ── the frontier wait ────────────────────────────────────────

def _epoch_wait_fixture(dual_period=False, scale_reuse=2):
    # dualPeriodScale is derived: scaled tensors present and reused.
    cfg = _config(lrSA=ReadGranularity(mn=2, k=1),
                  lrSB=ReadGranularity(mn=2, k=1),
                  scaleReuse=scale_reuse if dual_period else 1)
    assert cfg.dualPeriodScale is dual_period
    parts = _grid(numK=1, numPartitions=4)
    # Future-epoch requests issued before the frontier.
    parts[0][0].grs_by_unroll[0] = [
        _gr('A', 2, 0, 0, 3), _gr('B', 2, 0, 0, 2),
        _gr('SA', 2, 0, 0, 1), _gr('SB', 2, 0, 0, 1),
    ]
    # A current-epoch request, which the wait must not count.
    parts[0][0].grs_by_unroll[0].append(_gr('A', 0, 0, 0, 4))
    # The frontier itself, with a stale per-partition wait to be replaced.
    consumer = _lr('A', 1, 0, 0, 2, partition=1)
    consumer.preOps.append(WaitGROp(has_sync=False))
    parts[1][0].lrs.append(consumer)
    # A later read also carrying a stale wait.
    trailing = _lr('B', 1, 0, 0, 2, partition=2)
    trailing.preOps.append(WaitGROp(has_sync=False))
    parts[2][0].lrs.append(trailing)
    return cfg, parts, consumer, trailing


def test_epoch_wait_replaces_every_per_partition_wait():
    cfg, parts, consumer, trailing = _epoch_wait_fixture()
    _scheduler(cfg, parts)._apply_uniform_epoch_wait()

    waits = [op for op in consumer.preOps if isinstance(op, WaitGROp)]
    assert len(waits) == 1, "frontier must carry exactly one wait"
    assert not [op for op in trailing.preOps if isinstance(op, WaitGROp)], \
        "later reads must not keep their own producer wait"
    assert waits[0].has_sync is True


def test_epoch_wait_counts_only_future_requests_before_the_frontier():
    cfg, parts, consumer, _ = _epoch_wait_fixture()
    _scheduler(cfg, parts)._apply_uniform_epoch_wait()
    counts = [op for op in consumer.preOps if isinstance(op, WaitGROp)][0]

    # grA/grB granularity is one tile per K step, so the atom count is the
    # tile span.  The mtIteration=0 request is below cfg.pgr and excluded.
    assert (counts.wait_gr_counts.A, counts.wait_gr_counts.B) == (3, 2)
    assert (counts.wait_gr_counts.SA, counts.wait_gr_counts.SB) == (1, 1)


def test_dual_period_second_body_drops_the_scale_requests():
    """Body C1 reuses C0's scales, so counting them would over-wait."""
    cfg, parts, consumer, _ = _epoch_wait_fixture(dual_period=True)
    _scheduler(cfg, parts)._apply_uniform_epoch_wait()
    wait = [op for op in consumer.preOps if isinstance(op, WaitGROp)][0]

    by_unroll = wait.wait_gr_counts_by_unroll
    assert len(by_unroll) == cfg.scaleReuse
    assert (by_unroll[0].SA, by_unroll[0].SB) == (1, 1)
    assert (by_unroll[1].SA, by_unroll[1].SB) == (0, 0)
    # The data requests are present in both bodies.
    assert (by_unroll[1].A, by_unroll[1].B) == (3, 2)
    # wait_gr_counts stays in step with the first body.
    assert wait.wait_gr_counts is by_unroll[0]


def test_single_period_has_one_body():
    cfg, parts, consumer, _ = _epoch_wait_fixture(dual_period=False)
    _scheduler(cfg, parts)._apply_uniform_epoch_wait()
    wait = [op for op in consumer.preOps if isinstance(op, WaitGROp)][0]
    assert len(wait.wait_gr_counts_by_unroll) == 1


def test_epoch_wait_without_a_consumer_is_rejected():
    """The frontier slot must still hold the read that defined it."""
    cfg = _config()
    parts = _grid(numK=1, numPartitions=4)
    parts[1][0].lrs.append(_lr('A', 1, 0, 0, 2, partition=1))
    sched = _scheduler(cfg, parts)
    # Drive the frontier to a slot with no next-epoch consumer.
    sched._uniform_epoch_frontier_slot = lambda: 0
    with pytest.raises(ValueError, match="no next-epoch A/B consumer"):
        sched._apply_uniform_epoch_wait()


# ── partition-zero cohort hoist ──────────────────────────────

def test_hoist_is_a_noop_without_a_second_partition():
    cfg = _single_partition_config()
    parts = _grid(numK=1, numPartitions=1)
    parts[0][0].lrs.append(_lr('A', 0, 0, 0, 2, partition=0))
    _scheduler(cfg, parts)._move_uniform_epoch_p0_overwrite_reads(parts)
    assert len(parts[0][0].lrs) == 1


def test_overlapping_current_epoch_read_is_hoisted_into_partition_zero():
    """A read the first cohort will overwrite must drain before it."""
    cfg = _cohort2_config()
    parts = _grid(numK=1, numPartitions=4)
    parts[1][0].lrs.append(_lr('A', 0, 0, 0, 1, partition=1))
    _scheduler(cfg, parts)._move_uniform_epoch_p0_overwrite_reads(parts)

    assert parts[1][0].lrs == [], "the read must leave its own partition"
    hoisted = parts[0][0].lrs
    assert len(hoisted) == 1
    assert hoisted[0].partition == 0 and hoisted[0].subIterK_slot == 0
    assert hoisted[0].mustPrecedeP0Overwrite is True
    assert (hoisted[0].tiles.tileId_start, hoisted[0].tiles.tileId_end) == (0, 1)


def test_a_read_straddling_the_cohort_boundary_is_split():
    """Only the overlapping segment moves; the rest stays where it was.

    Hoisting the whole read would pull tiles into partition zero that nothing
    overwrites, and deferring the whole read would leave the cohort illegal.
    """
    cfg = _cohort2_config()
    parts = _grid(numK=1, numPartitions=4)
    # grA granularity is one tile, so the first cohort is A tiles [0,1) and
    # [1,2).  This read spans [0,4) and so straddles the boundary at 2.
    parts[1][0].lrs.append(_lr('A', 0, 0, 0, 4, partition=1))
    _scheduler(cfg, parts)._move_uniform_epoch_p0_overwrite_reads(parts)

    hoisted = parts[0][0].lrs
    remaining = parts[1][0].lrs
    assert [(x.tiles.tileId_start, x.tiles.tileId_end) for x in hoisted] \
        == [(0, 1), (1, 2)]
    assert [(x.tiles.tileId_start, x.tiles.tileId_end) for x in remaining] \
        == [(2, 4)]
    assert all(x.mustPrecedeP0Overwrite for x in hoisted)
    assert not any(x.mustPrecedeP0Overwrite for x in remaining)
    # Splitting must not carry the parent's edges onto the segments.
    assert all(x.deps == [] and x.preOps == [] for x in hoisted)


def test_next_epoch_and_scale_reads_are_left_alone():
    """Only current-epoch A/B reads can collide with the future write."""
    cfg = _cohort2_config()
    parts = _grid(numK=1, numPartitions=4)
    parts[1][0].lrs.append(_lr('A', 1, 0, 0, 1, partition=1))
    parts[1][0].lrs.append(_lr('SA', 0, 0, 0, 1, partition=1))
    _scheduler(cfg, parts)._move_uniform_epoch_p0_overwrite_reads(parts)
    assert len(parts[1][0].lrs) == 2
    assert parts[0][0].lrs == []


def test_a_read_clear_of_the_cohort_stays_put():
    cfg = _cohort2_config()
    parts = _grid(numK=1, numPartitions=4)
    parts[1][0].lrs.append(_lr('A', 0, 0, 2, 4, partition=1))
    _scheduler(cfg, parts)._move_uniform_epoch_p0_overwrite_reads(parts)
    assert len(parts[1][0].lrs) == 1
    assert parts[0][0].lrs == []


def test_cohort_spans_into_b_once_a_is_exhausted():
    """The epoch is ordered A then B, so an oversized cohort reaches B."""
    # A holds only two atoms, so a width-3 cohort has to reach into B.
    cfg = _config(numMFMATilesM=2, numMFMATilesN=10,
                  partitionSizeM=1, partitionSizeN=5)
    parts = _grid(numK=1, numPartitions=4)
    parts[1][0].lrs.append(_lr('B', 0, 0, 0, 1, partition=1))
    _scheduler(cfg, parts)._move_uniform_epoch_p0_overwrite_reads(parts)
    assert parts[1][0].lrs == []
    assert len(parts[0][0].lrs) == 1


# ── preloop helpers ──────────────────────────────────────────

def test_wait_gr_counts_for_ops_counts_only_global_reads():
    cfg = _config()
    sched = _scheduler(cfg, _grid(numK=1, numPartitions=4))
    ops = [_gr('A', 2, 0, 0, 3), _gr('SB', 2, 0, 0, 2),
           _lr('A', 0, 0, 0, 2)]
    counts = sched._wait_gr_counts_for_ops(ops)
    assert (counts.A, counts.B, counts.SA, counts.SB) == (3, 0, 0, 2)


def test_wait_gr_counts_for_an_empty_sequence_is_zero():
    sched = _scheduler(_config(), _grid(numK=1, numPartitions=4))
    counts = sched._wait_gr_counts_for_ops([])
    assert (counts.A, counts.B, counts.SA, counts.SB) == (0, 0, 0, 0)


def test_internal_label_op_builds_that_label():
    op = LogicalScheduler._make_internal_label_op("SubtilePreloopWaitDone")
    assert op.label == "SubtilePreloopWaitDone"
    text = str(op.build(None))
    assert "SubtilePreloopWaitDone" in text


def test_internal_branch_op_targets_the_label():
    op = LogicalScheduler._make_internal_branch_op("SubtilePreloopWaitDone")
    assert op.label == "branch_SubtilePreloopWaitDone"
    text = str(op.build(None))
    assert "s_branch" in text and "SubtilePreloopWaitDone" in text


# ── the emitter half of the per-body contract ────────────────

class _TileInfo:
    loadRatioGR = 1.0
    numGRTotal = 0
    subtileShape = (1, 1)


class _VgprPool:
    """Hands out distinct indices and records what came back."""

    def __init__(self, first=100):
        self._next = first
        self.checked_in = []

    def checkOut(self, num, tag=None, *args, **kwargs):
        idx = self._next
        self._next += num
        return idx

    def checkIn(self, idx):
        self.checked_in.append(idx)


class _SgprBlock:
    def __init__(self, idx):
        self.idx = idx


class _Writer:
    """Only the register bookkeeping the emit paths ask for.

    The real KernelWriter brings the assembler and the ISA tables with it;
    emit_mask_k touches none of that, so a pool and a scratch-sgpr
    allocator are the whole contract.
    """

    def __init__(self):
        self.vgprPool = _VgprPool()
        self.states = SimpleNamespace(laneSGPRCount=2, unrollIdx=0)

    @contextlib.contextmanager
    def allocTmpSgpr(self, num, alignment=None, tag=None):
        yield _SgprBlock(80)


def _dtype(num_bytes):
    """A data type that answers only what the mask path dispatches on."""
    d = MagicMock()
    d.numBytes.return_value = num_bytes
    d.numRegisters.return_value = num_bytes / 4
    d.isBFloat16.return_value = (num_bytes == 2)
    d.is8bitFloat.return_value = (num_bytes == 1)
    return d


def _emitter(config=None, scale=False, writer=None, **kwargs):
    scale_info = _TileInfo() if scale else None
    return InstructionEmitter(
        writer=writer if writer is not None else _Writer(),
        kernel=kwargs.pop("kernel", {"enableTDMA": False, "enableTDMB": False}),
        config=config if config is not None else _config(),
        tileInfoA=_TileInfo(), tileInfoB=_TileInfo(), dtileInfo=_TileInfo(),
        vgprTilesA=kwargs.pop("vgprTilesA", {}),
        vgprTilesB=kwargs.pop("vgprTilesB", {}),
        scaleTileInfoA=scale_info, scaleTileInfoB=scale_info,
        **kwargs)


def _counts(a, b, sa, sb):
    c = WaitGRCounts()
    c.A, c.B, c.SA, c.SB = a, b, sa, sb
    return c


def test_wait_gr_selects_the_count_for_this_body():
    """Body C1 must wait on its own request count, not C0's.

    This is the consumer of wait_gr_counts_by_unroll that
    test_dual_period_second_body_drops_the_scale_requests produces.
    """
    op = WaitGROp(wait_gr_counts=_counts(3, 2, 1, 1),
                  wait_gr_counts_by_unroll=(_counts(3, 2, 1, 1),
                                            _counts(3, 2, 0, 0)))
    em = _emitter()
    assert em.emit_wait_gr(op, unroll_iter=0)[0].vlcnt == 7
    assert em.emit_wait_gr(op, unroll_iter=1)[0].vlcnt == 5


def test_wait_gr_wraps_the_body_index():
    """Bodies repeat, so an index past the last one wraps rather than lifts."""
    op = WaitGROp(wait_gr_counts=_counts(1, 0, 0, 0),
                  wait_gr_counts_by_unroll=(_counts(1, 0, 0, 0),
                                            _counts(4, 0, 0, 0)))
    em = _emitter()
    assert em.emit_wait_gr(op, unroll_iter=2)[0].vlcnt == 1
    assert em.emit_wait_gr(op, unroll_iter=3)[0].vlcnt == 4


def test_wait_gr_falls_back_to_the_single_count():
    """Emitters that never set the per-body tuple keep the legacy behavior."""
    op = WaitGROp(wait_gr_counts=_counts(2, 2, 0, 0))
    assert _emitter().emit_wait_gr(op, unroll_iter=1)[0].vlcnt == 4


def test_wait_gr_without_counts_emits_nothing():
    assert _emitter().emit_wait_gr(WaitGROp()) == []


# ── the dual-period scale phase in the tail ──────────────────

def _mask_config(scale_reuse=2, scale_stride=4):
    """An MXFP8 tail config: one scale dword serves scale_stride K-steps."""
    return SchedulerConfig(
        numMFMATilesM=4, numMFMATilesN=4, numSubIterK=8,
        lrA=ReadGranularity(mn=2, k=1), lrB=ReadGranularity(mn=2, k=1),
        grA=ReadGranularity(mn=1, k=1), grB=ReadGranularity(mn=1, k=1),
        lrSA=ReadGranularity(mn=2, k=scale_stride),
        lrSB=ReadGranularity(mn=2, k=scale_stride),
        grSA=ReadGranularity(mn=1, k=1), grSB=ReadGranularity(mn=1, k=1),
        pgr=2, plr=0, scaleReuse=scale_reuse)


def _mask_emitter(scale_reuse=2, scale_stride=4, depth_u=128):
    writer = _Writer()
    kernel = {"MatrixInstK": 128, "DepthU": depth_u,
              "ProblemType": {"DataTypeA": _dtype(1)}}
    em = _emitter(config=_mask_config(scale_reuse, scale_stride),
                  scale=True, writer=writer, kernel=kernel,
                  vgprTilesA={0: [10, 11]}, vgprTilesB={0: [20, 21]},
                  vgprTilesSA={0: [200]}, vgprTilesSB={0: [210]})
    # emit_mask_k_init stages this; its own value is irrelevant to the
    # branch under test, only that the init ran.
    em._tail_vDiff = 50
    em._tail_boundaryMask = None
    return em


def _mask_text(em, subIterK):
    source = SimpleNamespace(
        subIterK=subIterK,
        vgpr_tile_map={'A': [{0: 0}], 'B': [{0: 0}],
                       'SA': [{0: 0}], 'SB': [{0: 0}]})
    return "\n".join(str(i) for i in em.emit_mask_k(source))


def test_tail_shifts_the_scale_dword_on_an_odd_depthu_count():
    """The tail is built for phase 0, so an odd count must rewind to it.

    DepthU=128 makes bit 7 of the unrolled size the completed-iteration
    parity; on odd the tail continues the epoch and owns byte 2, so the
    dword shifts down one phase.
    """
    text = _mask_text(_mask_emitter(), subIterK=0)
    assert "s_bitcmp1_b32" in text and "SizesSum+0" in text
    assert "s_cbranch_scc0" in text
    assert "SubtileTailScalePhase0_k0" in text
    # One shift per scale vgpr, moving it down by a whole subtile group.
    shifts = [l for l in text.splitlines() if "v_lshrrev_b32" in l]
    assert len(shifts) == 2
    assert all(", %d," % (8 * SCALE_SUBTILES_PER_GROUP) in l for l in shifts)


def _phase_bit(text):
    line, = [l for l in text.splitlines() if "s_bitcmp1_b32" in l]
    return int(line.split("//")[0].rsplit(",", 1)[1])


@pytest.mark.parametrize("depth_u, bit", [(64, 6), (128, 7), (256, 8)])
def test_the_phase_bit_tracks_depthu(depth_u, bit):
    """Parity lives in the bit that one DepthU iteration advances."""
    text = _mask_text(_mask_emitter(depth_u=depth_u), subIterK=0)
    assert _phase_bit(text) == bit


def test_only_the_step_that_loaded_the_dword_shifts_it():
    """The shift rewrites the dword, so it must not repeat on reuse steps."""
    text = _mask_text(_mask_emitter(), subIterK=1)
    assert "s_bitcmp1_b32" not in text
    assert "v_lshrrev_b32" not in text


def test_single_period_scale_never_shifts_phase():
    """With no reuse every step loads its own dword, already at phase 0."""
    em = _mask_emitter(scale_reuse=1)
    assert em.config.dualPeriodScale is False
    text = _mask_text(em, subIterK=0)
    assert "s_bitcmp1_b32" not in text
    assert "v_lshrrev_b32" not in text


def test_scale_lanes_still_get_the_k_mask_every_step():
    """Masks nest across the steps that share a dword, so re-ANDing is safe."""
    for subIterK in (0, 1):
        text = _mask_text(_mask_emitter(), subIterK=subIterK)
        assert text.count("v_and_b32") == 2, subIterK


def test_mxfp8_skips_the_redundant_data_mask():
    """8-bit MX takes the K mask through the scale lanes, not the data lanes."""
    text = _mask_text(_mask_emitter(), subIterK=0)
    for v in (10, 11, 20, 21):
        assert "v[%d]" % v not in text
