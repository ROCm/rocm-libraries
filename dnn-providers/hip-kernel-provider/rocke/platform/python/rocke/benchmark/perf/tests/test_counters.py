# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for counter probing/grouping (pure)."""

from __future__ import annotations

import unittest

from rocke.benchmark.perf import counters

_CDNA9 = [
    "GRBM_COUNT",
    "GRBM_GUI_ACTIVE",
    "SQ_BUSY_CYCLES",
    "SQ_WAVES",
    "SQ_WAIT_ANY",
    "SQ_INSTS_VALU",
    "SQ_INSTS_LDS",
    "TCC_HIT",
    "TCC_MISS",
]
_RDNA9 = [
    "GRBM_COUNT",
    "GRBM_GUI_ACTIVE",
    "SQ_BUSY_CYCLES",
    "SQ_WAVES",
    "SQ_WAIT_ANY",
    "SQ_INSTS_WAVE32_VALU",
    "SQ_INSTS_WAVE32_LDS",
    "GL2C_HIT",
    "GL2C_MISS",
]


class TestBlockOf(unittest.TestCase):
    def test_prefixes(self):
        self.assertEqual(counters._block_of("GRBM_COUNT"), "GRBM")
        self.assertEqual(counters._block_of("SQ_INSTS_WAVE32_VALU"), "SQ")
        self.assertEqual(counters._block_of("TCC_HIT"), "TCC")
        self.assertEqual(counters._block_of("GL2C_MISS"), "GL2C")
        self.assertEqual(counters._block_of("TCP_TOTAL_CACHE_ACCESSES"), "TCP")

    def test_unknown_block_fallback(self):
        self.assertEqual(counters._block_of("FOO_BAR_BAZ"), "FOO")


class TestGroupCounters(unittest.TestCase):
    def test_full_cdna_set_is_one_pass(self):
        groups = counters.group_counters(_CDNA9)
        self.assertEqual(len(groups), 1)  # GRBM 2, SQ 5, TCC 2 -> one pass
        self.assertEqual(set(groups[0]), set(_CDNA9))

    def test_full_rdna_set_is_one_pass(self):
        groups = counters.group_counters(_RDNA9)
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0]), set(_RDNA9))

    def test_cross_block_share_a_pass(self):
        groups = counters.group_counters(["GRBM_COUNT", "SQ_WAVES", "TCC_HIT"])
        self.assertEqual(len(groups), 1)  # different blocks -> same pass

    def test_same_block_overflow_splits(self):
        sq = [f"SQ_C{i}" for i in range(10)]  # SQ limit is 8
        groups = counters.group_counters(sq)
        self.assertEqual([len(g) for g in groups], [8, 2])

    def test_ratio_partners_stay_together_under_overflow(self):
        # overflow SQ, but GRBM/TCC ratio partners must each share a pass
        raws = [f"SQ_C{i}" for i in range(10)] + [
            "GRBM_COUNT",
            "GRBM_GUI_ACTIVE",
            "TCC_HIT",
            "TCC_MISS",
        ]
        groups = counters.group_counters(raws)

        def group_of(name):
            return next(i for i, g in enumerate(groups) if name in g)

        self.assertEqual(group_of("GRBM_COUNT"), group_of("GRBM_GUI_ACTIVE"))
        self.assertEqual(group_of("TCC_HIT"), group_of("TCC_MISS"))

    def test_empty(self):
        self.assertEqual(counters.group_counters([]), [])


class TestParseAndSelect(unittest.TestCase):
    def test_parse_both_formats_and_select(self):
        text = "Name:\tgfx950\nCounter_Name        :\tTCC_HIT\nName:\tGRBM_COUNT\n"
        avail = counters.parse_list_avail(text)
        self.assertIn("TCC_HIT", avail)
        self.assertIn("GRBM_COUNT", avail)
        sel = counters.select("gfx950", avail)
        self.assertEqual(sel.get("l2_hit"), "TCC_HIT")
        self.assertEqual(sel.get("total_clocks"), "GRBM_COUNT")


if __name__ == "__main__":
    unittest.main()
