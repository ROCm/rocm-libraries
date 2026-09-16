# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the reusable WMMA-class scheduler (helpers/schedule.py)."""

from __future__ import annotations

import unittest

from rocke.helpers.schedule import (
    DS_READ,
    MFMA,
    WMMA,
    SchedulePolicy,
    WmmaHotLoopInstList,
)


class _RecordingBuilder:
    """Minimal stub that records sched-hint emissions (no real IR)."""

    def __init__(self):
        self.group = []  # (mask, count, group)
        self.barriers = 0

    def sched_group_barrier(self, mask, count, group):
        self.group.append((mask, int(count), group))

    def sched_barrier(self, mask):
        self.barriers += 1


class TestWmmaSchedule(unittest.TestCase):
    def test_wmma_mask_aliases_mfma(self):
        # On the GFX12 programming model the MFMA sched-group mask matches WMMA.
        self.assertEqual(WMMA, MFMA)

    def test_inst_list_counts_gate_up(self):
        # gfx1250 16x16x32 gate+up tile: tile_m=16, tile_n=256, tile_k=32,
        # mfmas_m=1, mfmas_n=4, dual-B (gate+up), frag len 16 (=> 2 ds_read/frag).
        il = WmmaHotLoopInstList.from_geometry(
            block_size=128,
            m_per_block=16,
            n_per_block=256,
            k_per_block=32,
            m_repeat=1,
            n_repeat=4,
            m_per_wmma=16,
            n_per_wmma=16,
            k_per_wmma=32,
            a_frag_len=16,
            b_frag_len=16,
            num_b_operands=2,
        )
        # k_atoms = 1; n_wmma = 1*4*2*1 = 8.
        self.assertEqual(il.n_wmma, 8)
        # ds_read: A 1*1*2 + B 2*4*1*2 = 2 + 16 = 18.
        self.assertEqual(il.n_ds_read, 18)
        # compute-only region by default -> no loads/stores counted.
        self.assertEqual(il.n_ds_write, 0)
        self.assertEqual(il.n_vmem_read, 0)

    def test_compute_schedule_emits_interleaved_groups(self):
        il = WmmaHotLoopInstList.from_geometry(
            block_size=128,
            m_per_block=16,
            n_per_block=256,
            k_per_block=32,
            m_repeat=1,
            n_repeat=4,
            m_per_wmma=16,
            n_per_wmma=16,
            k_per_wmma=32,
            a_frag_len=16,
            b_frag_len=16,
            num_b_operands=2,
        )
        rec = _RecordingBuilder()
        SchedulePolicy.for_pipeline("wmma_v1").emit_wmma_compute_schedule(rec, il)
        wmma_groups = [g for g in rec.group if g[0] == WMMA]
        read_groups = [g for g in rec.group if g[0] == DS_READ]
        # One WMMA group per WMMA op; all ds_reads accounted for; one fence.
        self.assertEqual(len(wmma_groups), il.n_wmma)
        self.assertEqual(sum(c for _, c, _ in read_groups), il.n_ds_read)
        self.assertEqual(rec.barriers, 1)

    def test_mem_policy_emits_nothing(self):
        rec = _RecordingBuilder()
        il = WmmaHotLoopInstList.from_geometry(
            block_size=128,
            m_per_block=16,
            n_per_block=256,
            k_per_block=32,
            m_repeat=1,
            n_repeat=4,
            m_per_wmma=16,
            n_per_wmma=16,
            k_per_wmma=32,
            a_frag_len=16,
            b_frag_len=16,
        )
        SchedulePolicy.for_pipeline("mem").emit_wmma_compute_schedule(rec, il)
        self.assertEqual(rec.group, [])
        self.assertEqual(rec.barriers, 0)


if __name__ == "__main__":
    unittest.main()
