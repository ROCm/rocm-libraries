# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for GL2 prefetch in the subtile LogicalScheduler.

Tests the placement of GL2PrefetchOp / GL2PrefetchIncOp in the
mainloop and preloop schedules for PrefetchGL2 = 0, 1, 2.
No GPU required -- tests the Python scheduler layer only.
"""

import os
import sys
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

from Tensile.Components.Subtile.LogicalScheduler import (
    LogicalScheduler,
    SchedulerConfig,
    ReadGranularity,
    GL2PrefetchOp,
    GL2PrefetchIncOp,
)


def _make_config(pgl=0):
    """Minimal SchedulerConfig for a 4x4 tile, 2 subIterK."""
    return SchedulerConfig(
        numMFMATilesM=4,
        numMFMATilesN=4,
        numSubIterK=2,
        lrA=ReadGranularity(mn=4, k=1),
        lrB=ReadGranularity(mn=4, k=1),
        grA=ReadGranularity(mn=4, k=2),
        grB=ReadGranularity(mn=4, k=2),
        pgr=2,
        pgl=pgl,
    )


def _count_ops(emitted_3d, kind):
    """Count ops of a given kind in a 3D emitted structure."""
    count = 0
    for partition in emitted_3d:
        for group in partition:
            for em in group:
                if em.opType == kind:
                    count += 1
    return count


def _has_op(emitted_3d, kind):
    return _count_ops(emitted_3d, kind) > 0


class TestGL2PrefetchScheduler:
    """Test GL2 prefetch op placement in the LogicalScheduler."""

    def test_pgl0_no_prefetch_ops(self):
        """PGL=0: no GL2 ops in mainloop or preloop."""
        sched = LogicalScheduler(_make_config(pgl=0))
        sched.build()
        mainloop = sched._emitted
        preloop = sched.build_preloop()

        assert not _has_op(mainloop, 'gl2_prefetch')
        assert not _has_op(mainloop, 'gl2_prefetch_inc')
        assert not _has_op(preloop, 'gl2_prefetch')
        assert not _has_op(preloop, 'gl2_prefetch_inc')

    def test_pgl1_mainloop_has_prefetch(self):
        """PGL=1: mainloop has GL2 inc + load ops."""
        sched = LogicalScheduler(_make_config(pgl=1))
        sched.build()
        mainloop = sched._emitted

        assert _has_op(mainloop, 'gl2_prefetch')
        assert _has_op(mainloop, 'gl2_prefetch_inc')

    def test_pgl1_preloop_one_prefetch(self):
        """PGL=1: preloop has exactly one GL2 prefetch, no inc."""
        sched = LogicalScheduler(_make_config(pgl=1))
        sched.build()
        preloop = sched.build_preloop()

        assert _count_ops(preloop, 'gl2_prefetch') == 1
        assert _count_ops(preloop, 'gl2_prefetch_inc') == 0

    def test_pgl2_mainloop_has_prefetch(self):
        """PGL=2: mainloop has GL2 inc + load ops (same as PGL=1)."""
        sched = LogicalScheduler(_make_config(pgl=2))
        sched.build()
        mainloop = sched._emitted

        assert _has_op(mainloop, 'gl2_prefetch')
        assert _has_op(mainloop, 'gl2_prefetch_inc')

    def test_pgl2_preloop_two_prefetches(self):
        """PGL=2: preloop has two GL2 prefetches and one inc."""
        sched = LogicalScheduler(_make_config(pgl=2))
        sched.build()
        preloop = sched.build_preloop()

        assert _count_ops(preloop, 'gl2_prefetch') == 2
        assert _count_ops(preloop, 'gl2_prefetch_inc') == 1

    def test_ngll_strips_prefetch(self):
        """NGLL should have no GL2 prefetch ops for any PGL value."""
        for pgl in (1, 2):
            sched = LogicalScheduler(_make_config(pgl=pgl))
            sched.build()
            ngll = sched.build_ngll()

            assert not _has_op(ngll, 'gl2_prefetch'), f"PGL={pgl}: NGLL has gl2_prefetch"
            assert not _has_op(ngll, 'gl2_prefetch_inc'), f"PGL={pgl}: NGLL has gl2_prefetch_inc"

    def test_nll_strips_prefetch(self):
        """NLL should have no GL2 prefetch ops for any PGL value."""
        for pgl in (1, 2):
            sched = LogicalScheduler(_make_config(pgl=pgl))
            sched.build()
            nll = sched.build_nll()

            assert not _has_op(nll, 'gl2_prefetch'), f"PGL={pgl}: NLL has gl2_prefetch"
            assert not _has_op(nll, 'gl2_prefetch_inc'), f"PGL={pgl}: NLL has gl2_prefetch_inc"

    def test_mainloop_prefetch_after_last_gr(self):
        """GL2 ops should appear as postOps on the last GR in the mainloop."""
        sched = LogicalScheduler(_make_config(pgl=1))
        sched.build()

        # Walk partitions to find the last GR and check its postOps
        last_gr = None
        for slots in sched._partitions:
            for slot in slots:
                for gr in slot.grs:
                    last_gr = gr

        assert last_gr is not None, "No GR found in schedule"
        post_kinds = [op.kind for op in last_gr.postOps]
        assert 'gl2_prefetch_inc' in post_kinds
        assert 'gl2_prefetch' in post_kinds
