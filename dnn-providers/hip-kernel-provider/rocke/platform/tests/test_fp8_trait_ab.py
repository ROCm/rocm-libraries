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


def _smi_listing(stdout: str):
    """Make ``rocm-smi --showpids`` print ``stdout`` to the contention check."""
    from contextlib import ExitStack

    from rocke.examples.gfx1250.gemm_perf_fp8 import gpu_contention

    stack = ExitStack()
    stack.enter_context(
        mock.patch.object(gpu_contention.shutil, "which", return_value="rocm-smi")
    )
    stack.enter_context(
        mock.patch.object(
            gpu_contention.subprocess,
            "run",
            return_value=SimpleNamespace(stdout=stdout),
        )
    )
    return stack


class _FakeRuntime:
    """Records host->device copies; enough for PreparedProblem.create."""

    def __init__(self):
        self.copies = {}
        self._next = 0

    def alloc(self, nbytes):
        self._next += 1
        return self._next

    def memcpy_h2d(self, ptr, buf, nbytes):
        self.copies[ptr] = bytes(buf)[:nbytes]

    def memset(self, ptr, value, nbytes):
        pass


class TestSweepExactSet(unittest.TestCase):
    def test_sweep_uploads_fp8_data_int_operands(self):
        """The sweep's exact set is fp8_data's, so the two cannot drift."""
        from rocke.examples.gfx1250.gemm_perf_fp8 import fp8_gemm_sweep

        m, n, k, padded_n = 8, 6, 16, 8
        with mock.patch.object(fp8_gemm_sweep, "Runtime", _FakeRuntime):
            prob = fp8_gemm_sweep.PreparedProblem.create(
                (m, n, k), padded_n, with_reference=True
            )
        a8, b8, a32, b32 = fp8_data.int_operands(np, m, n, k)
        b_padded = np.zeros((padded_n, k), dtype=np.uint8)
        b_padded[:n] = b8
        self.assertEqual(prob.rt.copies[prob.a_dev], a8.tobytes())
        self.assertEqual(prob.rt.copies[prob.b_dev], b_padded.tobytes())
        from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16

        ref = bf16_to_f32(f32_to_bf16(a32 @ b32.T))
        self.assertTrue(np.array_equal(prob.reference, ref))


class TestInputSets(unittest.TestCase):
    def test_the_random_set_rounds_with_the_shared_codec(self):
        from rocke.numeric.fp8 import fp8e4m3_round

        self.assertIs(fp8_data.fp8e4m3_round, fp8e4m3_round)
        self.assertFalse(hasattr(fp8_data, "e4m3_round"))
        self.assertFalse(hasattr(fp8_data, "E4M3_MAX"))

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
        with _smi_listing(listing):
            self.assertEqual(fp8_trait_ab.other_gpu_processes(), ["4242"])
        with _smi_listing("No KFD PIDs currently running"):
            self.assertEqual(fp8_trait_ab.other_gpu_processes(), [])

    def test_sweep_and_trait_ab_share_one_contention_check(self):
        """Both tools must name the same foreign processes from one listing."""
        from rocke.examples.gfx1250.gemm_perf_fp8 import fp8_gemm_sweep

        listing = (
            "PID\tPROCESS NAME\tGPU(s)\tVRAM USED\tSDMA USED\tCU OCCUPANCY\n"
            f"{os.getpid()}\tpython3\t1\t1024\t0\t0\n"
            "4242\tother_job\t1\t2048\t0\t0\n"
        )
        with _smi_listing(listing):
            state = fp8_gemm_sweep.machine_state("t")
            pids = fp8_trait_ab.other_gpu_processes()
        self.assertEqual(state["foreign_kfd_count"], 1)
        self.assertFalse(state["clean"])
        self.assertEqual(pids, ["4242"])

    def test_a_nan_error_fails_the_random_bound(self):
        # max() and <= both let a NaN through; the verify step must not.
        bound = fp8_data.rand_rel_error_bound()
        self.assertGreater(fp8_trait_ab._finite_or_inf(float("nan")), bound)
        self.assertEqual(fp8_trait_ab._finite_or_inf(0.5), 0.5)

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
        # The fp8 GEMM's instruction, and the f8f6f4 form it is compared with.
        self.assertIn("f32_16x16x128_fp8_fp8", probe.resolve("@ceiling"))
        self.assertIn("f32_16x16x128_f8f6f4.fp8", probe.resolve("@ceiling"))
        with self.assertRaises(SystemExit):
            probe.resolve("no_such_instruction")


if __name__ == "__main__":
    unittest.main()
