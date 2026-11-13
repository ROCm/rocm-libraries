# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests for gemm_perf_fp8/clock_probe.py (in-kernel clock, health check).

The IR splice is tested on a real lowered kernel (lowering needs no GPU); the
analysis on synthetic stamps, including the two device states the probe
exists to tell apart: a realtime base offset between groups of workgroups
(harmless) and workgroups that need more cycles for the same work (degraded).
"""

from __future__ import annotations

import re
import unittest

import numpy as np
from rocke.examples.gfx1250.gemm_perf_fp8 import clock_probe as cp
from rocke.examples.gfx1250.gemm_perf_fp8.fp8_trait_ab import ARCH, arm_spec

RT_HZ = 100e6
WAVES = 16


def _lowered(arm: str = "early_refill") -> str:
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.instances.common.gemm_universal import build_universal_gemm

    # tensor_load_to_lds needs the llvm23 flavor; pin it rather than take the
    # environment's, as core/test_gfx1250_tdm.py does.
    return lower_kernel_to_llvm(
        build_universal_gemm(arm_spec(arm), arch=ARCH), arch=ARCH, llvm_flavor="llvm23"
    )


def _stamps(n_wg=8, life_ticks=40000, cycles=70000, start_offset=None, slow=None):
    """[n_wg * WAVES, 4] stamps: all start at 0 and live ``life_ticks``."""
    rows = []
    for wg in range(n_wg):
        base = 0 if start_offset is None else start_offset(wg)
        cyc = cycles * (1.25 if slow and wg in slow else 1.0)
        life = life_ticks * (1.25 if slow and wg in slow else 1.0)
        for _ in range(WAVES):
            rows.append([1000, base, 1000 + int(cyc), base + int(life)])
    return np.array(rows, dtype=np.uint64)


class TestInstrumentLl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ll = _lowered()
        cls.out = cp.instrument_ll(cls.ll, waves_per_wg=WAVES, tile_n=256)

    def test_both_counters_are_read_twice_and_declared(self):
        for name in ("readcyclecounter", "readsteadycounter"):
            with self.subTest(name=name):
                self.assertEqual(
                    len(re.findall(rf"call i64 @llvm\.{name}\(\)", self.out)), 2
                )
                self.assertRegex(self.out, rf"(?m)^declare i64 @llvm\.{name}\(\)")

    def test_start_stamp_opens_the_entry_block_and_end_stamp_precedes_ret(self):
        entry = self.out.index("\nentry:\n")
        first = self.out.index("call i64 @llvm.readcyclecounter()")
        self.assertLess(entry, first)
        self.assertNotIn(
            "%tid", self.out[entry:first]
        )  # nothing of the kernel before it
        body = self.out[self.out.rindex("call i64 @llvm.readcyclecounter()") :]
        self.assertRegex(body, r"store volatile i64 %clk\.rt1[^\n]*\n\s*ret void")

    def test_four_volatile_stores_past_c_and_nothing_else_changes(self):
        self.assertEqual(len(re.findall(r"store volatile i64 %clk\.", self.out)), 4)
        self.assertIn("%clk.base = lshr i64 %clk.mn, 2", self.out)
        stripped = "\n".join(
            line
            for line in self.out.splitlines()
            if "%clk." not in line and not line.startswith("declare i64 @llvm.read")
        )
        self.assertEqual(stripped.split(), self.ll.split())

    def test_ir_it_cannot_place_is_rejected(self):
        with self.assertRaises(ValueError):
            cp.instrument_ll(
                self.ll.replace("%C", "%Q"), waves_per_wg=WAVES, tile_n=256
            )
        with self.assertRaises(ValueError):
            cp.instrument_ll(
                self.ll.replace("ret void", "ret void\n  ret void"),
                waves_per_wg=WAVES,
                tile_n=256,
            )


class TestAnalysis(unittest.TestCase):
    def test_a_healthy_launch(self):
        s = cp.summarize(np, _stamps(), waves_per_wg=WAVES, rt_hz=RT_HZ)
        self.assertEqual(s["start_max_ms"], 0.0)
        self.assertAlmostEqual(s["life_med_ms"], 0.4)
        self.assertAlmostEqual(s["clock_med_mhz"], 70000 / 0.4e-3 / 1e6)
        self.assertTrue(cp.realtime_bases_agree(s, launch_ms=0.41))
        self.assertEqual(cp.health_verdict(s, bases_agree=True), "ok")

    def test_a_realtime_base_offset_is_reported_but_not_a_fault(self):
        # Half the workgroups read realtime 0.46 ms ahead; lifetimes equal.
        off = cp.summarize(
            np,
            _stamps(start_offset=lambda wg: 46000 if wg % 2 else 0),
            waves_per_wg=WAVES,
            rt_hz=RT_HZ,
        )
        self.assertAlmostEqual(off["start_max_ms"], 0.46)
        self.assertFalse(cp.realtime_bases_agree(off, launch_ms=0.41))
        self.assertEqual(cp.health_verdict(off, bases_agree=False), "ok")
        self.assertIn("start late", cp.health_verdict(off, bases_agree=True))

    def test_extra_cycles_for_the_same_work_are_flagged(self):
        sick = cp.summarize(np, _stamps(slow={5}), waves_per_wg=WAVES, rt_hz=RT_HZ)
        self.assertAlmostEqual(sick["cycles_slow"] / sick["cycles_rest"], 1.25)
        self.assertIn("cycles", cp.health_verdict(sick, bases_agree=True))

    def test_stragglers_are_chosen_by_lifetime_not_by_an_offset_end(self):
        # The offset group ends later on the realtime axis but is not slow;
        # the genuinely slow workgroup (wg 2) must still be among the picked.
        mixed = cp.summarize(
            np,
            _stamps(start_offset=lambda wg: 46000 if wg % 2 else 0, slow={2}),
            waves_per_wg=WAVES,
            rt_hz=RT_HZ,
        )
        self.assertGreater(mixed["cycles_slow"], mixed["cycles_rest"])


if __name__ == "__main__":
    unittest.main()
