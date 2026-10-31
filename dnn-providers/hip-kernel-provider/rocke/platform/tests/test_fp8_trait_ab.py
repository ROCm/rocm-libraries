# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests for the gfx1250 fp8 case-study tools in gemm_perf_fp8/.

``fp8_data`` (the two input sets), ``fp8_trait_ab`` (the interleaved trait
A/B) and ``wmma_roofline_probe`` (the data-matched ceiling). Everything here
runs without a GPU; the tools' on-device behaviour is exercised by running
them, as the case study describes.
"""

from __future__ import annotations

import os
import unittest
from dataclasses import fields
from types import SimpleNamespace
from unittest import mock

import numpy as np
from rocke.examples.gfx1250.gemm_perf_fp8 import fp8_data, fp8_trait_ab
from rocke.examples.gfx1250.gemm_perf_fp8 import wmma_roofline_probe as probe
from rocke.instances.common.manifest_runner.gemm import _fp8e4m3_decode_table


class TestE4m3Round(unittest.TestCase):
    def test_every_finite_value_round_trips(self):
        table = _fp8e4m3_decode_table(np)
        codes = np.flatnonzero(np.isfinite(table) & (np.arange(256) != 0x80))
        self.assertTrue(
            np.array_equal(
                fp8_data.e4m3_round(np, table[codes]), codes.astype(np.uint8)
            )
        )

    def test_ties_go_to_the_even_code(self):
        # 1.0 = 0x38, 1.125 = 0x39, 1.25 = 0x3A: midpoints go to 0x38 and 0x3A.
        x = np.array([1.0625, 1.1875, -1.0625], dtype=np.float32)
        self.assertEqual(list(fp8_data.e4m3_round(np, x)), [0x38, 0x3A, 0xB8])

    def test_rounds_to_nearest_and_saturates(self):
        x = np.array([1.05, 1.08, 1000.0, -1000.0, 0.0], dtype=np.float32)
        self.assertEqual(
            list(fp8_data.e4m3_round(np, x)), [0x38, 0x39, 0x7E, 0xFE, 0x00]
        )

    def test_nan_is_rejected(self):
        with self.assertRaises(ValueError):
            fp8_data.e4m3_round(np, np.array([np.nan], dtype=np.float32))


class TestInputSets(unittest.TestCase):
    def test_int_set_is_exact_and_in_range(self):
        a8, b8, a, b = fp8_data.int_operands(np, 8, 4, 64)
        self.assertTrue(np.array_equal(fp8_data.e4m3_decode(np, a8), a))
        self.assertTrue(np.array_equal(fp8_data.e4m3_decode(np, b8), b))
        self.assertLessEqual(float(np.abs(a).max()), 5.0)

    def test_rand_set_returns_what_the_kernel_multiplies(self):
        a8, b8, a, b = fp8_data.rand_operands(np, 8, 4, 64)
        self.assertTrue(np.array_equal(fp8_data.e4m3_decode(np, a8), a))
        self.assertTrue(np.array_equal(fp8_data.e4m3_decode(np, b8), b))
        # Seeded: the same bytes on every run.
        again = fp8_data.rand_operands(np, 8, 4, 64)
        self.assertTrue(np.array_equal(again[0], a8))

    def test_rand_bound_covers_bf16_output_rounding(self):
        self.assertGreater(fp8_data.rand_rel_error_bound(), 2.0**-8)


class TestTraitAbArms(unittest.TestCase):
    def test_every_arm_is_a_valid_gfx1250_spec(self):
        for arm in fp8_trait_ab.ARMS:
            with self.subTest(arm=arm):
                fp8_trait_ab.arm_spec(arm)  # raises if invalid

    def test_arms_share_the_geometry_and_differ_only_in_their_traits(self):
        base = fp8_trait_ab.champion_spec()
        self.assertEqual(base.block_size, 512)
        self.assertEqual(base.tile.warp_tile_k, 128)
        for arm, (_desc, overrides) in fp8_trait_ab.ARMS.items():
            with self.subTest(arm=arm):
                spec = fp8_trait_ab.arm_spec(arm)
                self.assertEqual(spec.tile, base.tile)
                self.assertEqual(spec.data, base.data)
                changed = {
                    f.name
                    for f in fields(base.trait)
                    if getattr(spec.trait, f.name) != getattr(base.trait, f.name)
                }
                self.assertEqual(changed, set(overrides))

    def test_each_arm_adds_to_the_one_before(self):
        prev: set = set()
        for _desc, overrides in fp8_trait_ab.ARMS.values():
            self.assertTrue(prev <= set(overrides))
            prev = set(overrides)

    def test_arm_order_alternates_by_round(self):
        arms = ["a", "b", "c"]
        self.assertEqual(fp8_trait_ab.interleaved_order(arms, 0), ["a", "b", "c"])
        self.assertEqual(fp8_trait_ab.interleaved_order(arms, 1), ["c", "b", "a"])

    def test_process_report_excludes_this_process(self):
        listing = (
            "PID\tPROCESS NAME\tGPU(s)\n"
            f"{os.getpid()}\tpython3\t1\n"
            "4242\tother_job\t1\n"
        )
        with mock.patch.object(
            fp8_trait_ab.shutil, "which", return_value="rocm-smi"
        ), mock.patch.object(
            fp8_trait_ab.subprocess, "run", return_value=SimpleNamespace(stdout=listing)
        ):
            self.assertEqual(fp8_trait_ab.other_gpu_processes(), ["4242"])
        empty = SimpleNamespace(stdout="No KFD PIDs currently running")
        with mock.patch.object(
            fp8_trait_ab.shutil, "which", return_value="rocm-smi"
        ), mock.patch.object(fp8_trait_ab.subprocess, "run", return_value=empty):
            self.assertEqual(fp8_trait_ab.other_gpu_processes(), [])

    def test_bad_arguments_fail_before_touching_the_gpu(self):
        for argv in (["--arms", "nope"], ["--data", "fp16"], ["--k", str(1 << 20)]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                fp8_trait_ab.main(argv)


class TestRooflineProbe(unittest.TestCase):
    def test_data_tables_decode_to_the_gemm_input_sets(self):
        ints = fp8_data.e4m3_decode(np, probe._data_table("int"))
        self.assertTrue(np.array_equal(ints, np.round(ints)))
        self.assertLessEqual(float(np.abs(ints).max()), 5.0)
        rand = fp8_data.e4m3_decode(np, probe._data_table("rand"))
        self.assertTrue(np.isfinite(rand).all())
        self.assertGreater(len(set(rand.tolist())), 8)

    def test_generated_source_uses_the_requested_data(self):
        name = "f32_16x16x128_f8f6f4.fp8"
        case = {
            "id": "x",
            "inst": name,
            "chains": 16,
            "blk": 128,
            "passes": 2,
            "blocks": 8,
            "kname": "k_x",
        }
        saved = probe.DATA_MODE
        try:
            probe.DATA_MODE = "const"
            const_src = probe.gen_source([case], 4, 1, 1, 1, 0)
            probe.DATA_MODE = "rand"
            rand_src = probe.gen_source([case], 4, 1, 1, 1, 0)
        finally:
            probe.DATA_MODE = saved
        for src in (const_src, rand_src):
            self.assertIn("__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4", src)
        self.assertNotIn("T[64]", const_src)
        self.assertIn("T[64]", rand_src)

    def test_sets_resolve_and_unknown_names_fail(self):
        self.assertIn("f32_16x16x128_f8f6f4.fp8", probe.resolve("@ceiling"))
        with self.assertRaises(SystemExit):
            probe.resolve("no_such_instruction")


if __name__ == "__main__":
    unittest.main()
