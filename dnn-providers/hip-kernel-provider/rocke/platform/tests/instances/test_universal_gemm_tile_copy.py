# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Coverage for UniversalGemm tile copies and runtime K schedules."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from rocke.helpers.loads import AsyncTileLoaderSlot, CoalescedTileLoader
from rocke.instances.common import gemm_universal
from rocke.instances.common._gemm_kloop import GemmKLoop
from rocke.instances.common.gemm_universal import (
    TraitSpec,
    build_universal_gemm,
)


from rocke.tests.universal_gemm_gpu_gate import tile_copy_cases, tile_copy_spec


def _build(trait: TraitSpec):
    spec = tile_copy_spec(trait)
    copies = []
    schedules = []
    visit_chunks = CoalescedTileLoader.visit_chunks
    single = GemmKLoop.single_buffered
    ping_pong = GemmKLoop.ping_pong

    def record_copy(loader, builder, **kwargs):
        copies.append((loader.tile_rows, loader.tile_cols, loader.load_vec))
        return visit_chunks(loader, builder, **kwargs)

    def record_single(loop):
        schedules.append("single")
        return single(loop)

    def record_ping_pong(loop, *, direct_to_lds):
        schedules.append("async" if direct_to_lds else "sync")
        return ping_pong(loop, direct_to_lds=direct_to_lds)

    with (
        patch.object(CoalescedTileLoader, "visit_chunks", record_copy),
        patch.object(GemmKLoop, "single_buffered", record_single),
        patch.object(GemmKLoop, "ping_pong", record_ping_pong),
        patch.object(
            AsyncTileLoaderSlot,
            "issue_bytes",
            autospec=True,
            side_effect=AsyncTileLoaderSlot.issue_bytes,
        ) as dtl_issues,
        patch.object(
            gemm_universal,
            "load_clipped_global_fragment",
            wraps=gemm_universal.load_clipped_global_fragment,
        ) as clipped_reads,
    ):
        kernel = build_universal_gemm(spec, arch="gfx950")
    return (
        spec,
        kernel,
        copies,
        schedules,
        clipped_reads.call_count,
        dtl_issues.call_count,
    )


class TestUniversalGemmTileCopy(unittest.TestCase):
    def _build_case(self, label, trait, schedule):
        spec, kernel, copies, schedules, clipped_reads, dtl_issues = _build(trait)
        expected = [(64, 32, 8)] * {"single": 2, "sync": 4, "async": 0}[schedule]
        if trait.epilogue == "cshuffle":
            expected.append((64, 64, 8))
        self.assertEqual(copies, expected)
        self.assertEqual(schedules, [schedule])
        self.assertEqual(bool(clipped_reads), trait.pad_k)
        self.assertEqual(dtl_issues, 4 if schedule == "async" else 0)
        return spec, kernel

    def test_copy_and_schedule_selection(self):
        for label, trait, schedule, _ in tile_copy_cases():
            with self.subTest(case=label):
                self._build_case(label, trait, schedule)


if __name__ == "__main__":
    unittest.main(verbosity=2)
