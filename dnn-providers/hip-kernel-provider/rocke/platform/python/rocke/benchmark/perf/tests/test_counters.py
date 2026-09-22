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

    def test_keep_together_survives_block_overflow(self):
        # A ratio's inputs are meaningless across executions, so an overflowing
        # block must split BETWEEN units, never through one.
        raws = [f"SQ_C{i}" for i in range(7)] + ["SQ_NUM", "SQ_DEN"]
        groups = counters.group_counters(raws, keep_together=[["SQ_NUM", "SQ_DEN"]])

        def group_of(name):
            return next(i for i, g in enumerate(groups) if name in g)

        self.assertEqual(group_of("SQ_NUM"), group_of("SQ_DEN"))
        self.assertTrue(all(len(g) <= counters._BLOCK_SLOTS["SQ"] for g in groups))

    def test_unit_larger_than_block_budget_raises(self):
        # Better to fail loudly than emit a ratio spanning two executions.
        with self.assertRaises(ValueError):
            counters.group_counters(
                ["GRBM_A", "GRBM_B", "GRBM_C"],
                keep_together=[["GRBM_A", "GRBM_B", "GRBM_C"]],  # GRBM has 2 slots
            )

    def test_overlapping_units_exceeding_budget_raise(self):
        with self.assertRaisesRegex(ValueError, "block GRBM has only 2"):
            counters.group_counters(
                ["GRBM_A", "GRBM_B", "GRBM_C"],
                keep_together=[["GRBM_A", "GRBM_B"], ["GRBM_B", "GRBM_C"]],
            )

    def test_overlapping_units_merge_transitively_before_packing(self):
        prefix = [f"SQ_C{i}" for i in range(6)]
        raws = prefix + ["SQ_A", "SQ_B", "SQ_C", "SQ_D", "SQ_E"]
        groups = counters.group_counters(
            raws,
            keep_together=[
                prefix,
                ["SQ_A", "SQ_B"],
                ["SQ_D", "SQ_E"],
                ["SQ_B", "SQ_C"],
                ["SQ_C", "SQ_D"],
            ],
        )
        self.assertEqual(groups, [prefix, ["SQ_A", "SQ_B", "SQ_C", "SQ_D", "SQ_E"]])

    def test_duplicate_constraint_members_use_one_slot(self):
        self.assertEqual(
            counters.group_counters(
                ["GRBM_A", "GRBM_B"],
                keep_together=[
                    ["GRBM_A", "GRBM_A", "GRBM_B", "GRBM_B"],
                    ["GRBM_B", "GRBM_A"],
                ],
            ),
            [["GRBM_A", "GRBM_B"]],
        )

    def test_constraints_ignore_unselected_members(self):
        self.assertEqual(
            counters.group_counters(
                ["GRBM_A", "GRBM_C"],
                keep_together=[["GRBM_A", "GRBM_B"], ["GRBM_B", "GRBM_C"]],
            ),
            [["GRBM_A", "GRBM_C"]],
        )

    def test_full_rdna_diagnostic_set_is_one_pass(self):
        sel = counters.wanted_map("gfx1201")
        groups = counters.group_counters(
            list(sel.values()), keep_together=counters.ratio_units(sel)
        )
        self.assertEqual(len(groups), 1)
        self.assertCountEqual(groups[0], sel.values())

    def test_cross_block_unit_raises(self):
        with self.assertRaises(ValueError):
            counters.group_counters(
                ["SQ_X", "TCC_Y"], keep_together=[["SQ_X", "TCC_Y"]]
            )

    def test_cdna_diagnostic_set_keeps_every_ratio_in_one_pass(self):
        # gfx90a: 14 counters overflow SQ (8 slots) into 2 passes; every ratio's
        # inputs must still land together or the ratio mixes two kernel runs.
        sel = counters.wanted_map("gfx90a")
        self.assertEqual(len(sel), 14)
        groups = counters.group_counters(
            list(sel.values()), keep_together=counters.ratio_units(sel)
        )
        self.assertEqual(len(groups), 2)
        self.assertCountEqual([r for group in groups for r in group], sel.values())
        for group in groups:
            for block in {counters._block_of(r) for r in group}:
                self.assertLessEqual(
                    sum(counters._block_of(r) == block for r in group),
                    counters._BLOCK_SLOTS.get(block, counters._DEFAULT_SLOTS),
                )

        def group_of(name):
            return next(i for i, g in enumerate(groups) if name in g)

        for num, den in (
            ("valu_active_cycles", "cu_busy_cycles"),
            ("lds_bank_conflict", "lds_idx_active"),
            ("l2_hit", "l2_miss"),
            ("busy_cycles", "total_clocks"),
            ("mfma_insts", "valu_insts"),
        ):
            self.assertEqual(
                group_of(sel[num]), group_of(sel[den]), f"{num}/{den} split"
            )

    def test_wavescope_rule_inputs_share_one_pass_on_cdna(self):
        # WaveScope reads ONE uploaded CSV, so every input of a given PMC rule has
        # to be in the same pass or that rule silently cannot evaluate.
        sel = counters.wanted_map("gfx90a")
        groups = counters.group_counters(
            list(sel.values()), keep_together=counters.ratio_units(sel)
        )

        def group_of(name):
            return next(i for i, g in enumerate(groups) if name in g)

        rules = {
            "lds-bank-conflict": ("SQ_LDS_BANK_CONFLICT", "SQ_LDS_IDX_ACTIVE"),
            "cache-hit": ("TCC_HIT", "TCC_MISS"),
            "valu-util": (
                "SQ_ACTIVE_INST_VALU",
                "SQ_BUSY_CU_CYCLES",
                "SQ_INSTS_MFMA",
                "SQ_INSTS_VALU",
            ),
        }
        for rule, inputs in rules.items():
            self.assertEqual(
                {group_of(r) for r in inputs}, {0}, f"{rule} inputs missing first pass"
            )


class TestDerive(unittest.TestCase):
    def test_ratios_from_full_capture(self):
        d = counters.derive(
            {
                "busy_cycles": 900,
                "total_clocks": 1000,
                "l2_hit": 75,
                "l2_miss": 25,
                "lds_bank_conflict": 25,
                "lds_idx_active": 75,
                "valu_active_cycles": 400,
                "cu_busy_cycles": 1000,
                "mfma_insts": 30,
                "valu_insts": 120,
            }
        )
        self.assertAlmostEqual(d["busy_fraction"], 0.9)
        self.assertAlmostEqual(d["l2_hit_rate"], 0.75)
        self.assertAlmostEqual(d["lds_bank_conflict_rate"], 1 / 3)
        self.assertAlmostEqual(d["valu_utilization"], 0.4)
        self.assertAlmostEqual(d["matrix_share"], 0.25)

    def test_partial_capture_omits_rather_than_guesses(self):
        d = counters.derive({"busy_cycles": 900, "lds_bank_conflict": 25})
        self.assertNotIn("busy_fraction", d)  # total_clocks absent
        self.assertNotIn("lds_bank_conflict_rate", d)  # lds_idx_active absent

    def test_zero_denominator_omits_rather_than_raises(self):
        d = counters.derive({"busy_cycles": 0, "total_clocks": 0})
        self.assertEqual(d, {})

    def test_zero_lds_activity_omits_conflict_rate(self):
        for conflicts in (0, 25):
            with self.subTest(conflicts=conflicts):
                d = counters.derive(
                    {"lds_bank_conflict": conflicts, "lds_idx_active": 0}
                )
                self.assertNotIn("lds_bank_conflict_rate", d)

    def test_lds_conflict_ratio_can_exceed_one(self):
        d = counters.derive({"lds_bank_conflict": 150, "lds_idx_active": 75})
        self.assertEqual(d["lds_bank_conflict_rate"], 2)

    def test_ratio_units_skips_uncaptured_pairs(self):
        # RDNA captures no VALU-activity pair, so it must not be pinned.
        units = counters.ratio_units(counters.wanted_map("gfx1201"))
        flat = {r for u in units for r in u}
        self.assertNotIn("SQ_BUSY_CU_CYCLES", flat)
        self.assertIn("SQC_LDS_BANK_CONFLICT", flat)


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
