#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests for fixed A/B/C global vector widths in the GEMM bridge.

A GEMM whose contiguous A/B/C extent is not a multiple of the native 16-byte
vector width is rejected by the kernel's IsSupportedArgument. The bridge can
build the same tile with narrower fixed widths instead; these tests lock in:

  * the per-problem width helper (gcd of extent and native width);
  * canonical resolution (native -> all zero, else min(requested, native)) and
    the per-tile legality checks;
  * that gemm_utils, the codegen KernelNaming and the generated kernel agree on
    the kernel name and actually pass the widths to the CK problem/epilogue;
  * expand_sweep emitting fallback configs and counting rejects.

Run: python3 -m pytest tests/test_gemm_vector_sizes.py -v
"""

import sys
import unittest
from dataclasses import replace
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.append(str(DISPATCHER_DIR.parent / "tile_engine" / "ops" / "gemm"))

from codegen_common import (  # noqa: E402
    TileConfig,
    _native_ab_vector_size,
    gemm_native_vector_sizes,
    gemm_problem_vector_sizes,
    gemm_vector_size_suffix,
    gemm_lockstep_vector_bytes,
    gemm_vector_size_sweep,
    resolve_gemm_vector_sizes,
)
from unified_gemm_codegen import (  # noqa: E402
    CKTileKernelGenerator,
    GemmVariant,
    KernelConfig,
    KernelNaming,
    TraitConfig,
)
from gemm_utils import GemmKernelConfig  # noqa: E402
from gemm_vector_fallback import VectorFallback  # noqa: E402

TILE = dict(tile=(256, 256, 64), waves=(2, 2, 1), warp_tile=(32, 32, 16))

# (layout, M, N, K) -> widest legal bf16 (A, B, C) widths
ACCEPTANCE = [
    ("rcr", 393216, 256, 257, (1, 1, 8)),
    ("rrr", 196608, 641, 384, (8, 1, 1)),
    ("crr", 316, 256, 3072, (4, 8, 8)),
    ("rcr", 1792, 2048, 5972, (4, 4, 8)),
]


def _resolve(layout, requested, **kw):
    args = dict(
        dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", layout=layout,
        gfx_arch="gfx950", requested=requested, **TILE,
    )
    args.update(kw)
    return resolve_gemm_vector_sizes(**args)


def _bridge_config(vec=(1, 1, 8), variant="standard", pipeline="compv3"):
    return GemmKernelConfig(
        dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32",
        layout_a="row", layout_b="col", layout_c="row", tile_m=256, tile_n=256, tile_k=64,
        wave_m=2, wave_n=2, wave_k=1,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
        pipeline=pipeline, scheduler="intrawave", epilogue="cshuffle",
        pad_m=True, pad_n=True, pad_k=True, gfx_arch="gfx950", variant=variant,
    ).with_vector_sizes(vec)


def _codegen_config(vec, variant=GemmVariant.STANDARD):
    tile = TileConfig(
        tile_m=256, tile_n=256, tile_k=64, warp_m=2, warp_n=2, warp_k=1,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
    )
    trait = TraitConfig(
        pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
        pad_m=True, pad_n=True, pad_k=True, persistent=False,
        vector_size_a=vec[0], vector_size_b=vec[1], vector_size_c=vec[2],
    )
    return KernelConfig(tile=tile, trait=trait, variant=variant)


class TestProblemWidths(unittest.TestCase):
    def test_acceptance_shapes(self):
        for layout, m, n, k, want in ACCEPTANCE:
            got = gemm_problem_vector_sizes(m, n, k, layout, "bf16", "bf16", "bf16")
            self.assertEqual(got, want, (layout, m, n, k))

    def test_aligned_problem_is_native(self):
        self.assertEqual(
            gemm_problem_vector_sizes(1024, 1024, 1024, "rcr", "fp16", "fp16", "fp16"),
            (8, 8, 8),
        )
        self.assertEqual(
            gemm_problem_vector_sizes(1024, 1024, 1024, "rcr", "fp8", "fp8", "fp16"),
            (16, 16, 8),
        )


class TestResolution(unittest.TestCase):
    def test_acceptance_shapes_resolve_legal(self):
        for layout, _, _, _, want in ACCEPTANCE:
            self.assertEqual(_resolve(layout, want), (want, None), layout)

    def test_native_request_is_canonical_zero(self):
        native = gemm_native_vector_sizes(
            dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", layout="rcr",
            gfx_arch="gfx950", **TILE,
        )
        self.assertEqual(native, (8, 8, 8))
        for req in [(0, 0, 0), (8, 8, 8), (16, 16, 16), native]:
            self.assertEqual(_resolve("rcr", req), ((0, 0, 0), None), req)

    def test_resolution_is_idempotent(self):
        for req in [(1, 1, 8), (0, 2, 0), (4, 16, 1)]:
            vec, reason = _resolve("rcr", req)
            self.assertIsNone(reason)
            self.assertEqual(_resolve("rcr", vec), (vec, None))

    def test_rejects_non_power_of_two(self):
        self.assertIn("power of two", _resolve("rcr", (3, 8, 8))[1])

    def test_rejects_unsupported_scope(self):
        for kw, word in [
            (dict(pipeline="comp_async"), "pipeline"),
            (dict(epilogue="default"), "epilogue"),
            (dict(variant="grouped_quant"), "variant"),
            (dict(variant="preshuffle", pipeline="preshufflev2"), "variant"),
        ]:
            self.assertIn(word, _resolve("rcr", (1, 1, 8), **kw)[1], kw)

    def test_native_width_skips_sub_element_steps(self):
        # Like CK: fp16 never uses the 4-byte (2-wide) step.
        self.assertEqual(_native_ab_vector_size(2, 64, 2, 64, 256), 1)
        self.assertEqual(_native_ab_vector_size(4, 64, 2, 64, 256), 2)

    def test_rejects_wave64_lds_with_too_few_warps(self):
        # Narrow col-major A / row-major B on wave64 would divide by zero in the LDS descriptor.
        tile = dict(tile=(64, 64, 64), waves=(1, 1, 1), warp_tile=(32, 32, 16))
        self.assertIn("LDS", _resolve("crr", (1, 0, 0), gfx_arch="gfx942", **tile)[1])
        self.assertIn("LDS", _resolve("rrr", (0, 1, 0), gfx_arch="gfx942", **tile)[1])
        self.assertIsNone(_resolve("crr", (1, 0, 0), gfx_arch="gfx1250", **tile)[1])

    def test_rejects_stream_k_sub_dword_atomic_c(self):
        # bf16 C width 1 = 2 bytes: below the 4-byte buffer atomic.
        self.assertIn("atomic", _resolve("rrr", (8, 1, 1), variant="stream_k")[1])
        self.assertIsNone(_resolve("rrr", (8, 1, 2), variant="stream_k")[1])
        self.assertIsNone(_resolve("rrr", (8, 1, 1))[1])

    def test_rejects_warp_tile_k_not_multiple(self):
        _, reason = _resolve("rcr", (4, 8, 8), warp_tile=(32, 32, 2))
        self.assertIsNotNone(reason)


class TestNamingAgreement(unittest.TestCase):
    """gemm_utils, KernelNaming and the kernel header must agree on the name."""

    def test_suffix(self):
        self.assertEqual(gemm_vector_size_suffix((0, 0, 0)), "")
        self.assertEqual(gemm_vector_size_suffix((1, 1, 8)), "_vec1_1_8")

    def test_bridge_name_matches_codegen(self):
        for vec in [(1, 1, 8), (0, 0, 0)]:
            cfg, reason = _bridge_config(vec=vec)
            self.assertIsNone(reason)
            want = KernelNaming.generate(_codegen_config(cfg.vector_sizes), "bf16", "rcr")
            self.assertEqual(cfg.name, want)
            self.assertEqual("_vec" in cfg.name, any(vec))

    def test_codegen_key_names_distinct(self):
        self.assertNotEqual(
            _codegen_config((1, 1, 8)).key_name("bf16", "rcr"),
            _codegen_config((0, 0, 0)).key_name("bf16", "rcr"),
        )

    def test_effective_vector_sizes(self):
        self.assertEqual(_bridge_config(vec=(0, 0, 0))[0].effective_vector_sizes, (8, 8, 8))
        self.assertEqual(_bridge_config(vec=(1, 1, 8))[0].effective_vector_sizes, (1, 1, 8))

    def test_bridge_reject_names_tile(self):
        _, reason = _bridge_config(vec=(1, 1, 8), pipeline="comp_async")
        self.assertIn("256x256x64", reason)
        self.assertIn("vec1_1_8", reason)


class TestGeneratedKernel(unittest.TestCase):
    def _src(self, vec, variant=GemmVariant.STANDARD):
        return CKTileKernelGenerator("bf16", "rcr").generate(_codegen_config(vec, variant))

    def test_fixed_widths_reach_problem_and_epilogue(self):
        for variant in (GemmVariant.STANDARD, GemmVariant.BATCHED):
            src = self._src((1, 1, 8), variant)
            self.assertIn("ADataType, BDataType, true, 1, 1>", src, variant)
            self.assertIn("true, 8, 1, DoubleSmemBuffer>", src, variant)
            self.assertIn("_vec1_1_8", src, variant)

    def test_native_kernel_unchanged(self):
        src = self._src((0, 0, 0))
        self.assertNotIn("_vec", src)
        self.assertNotIn("ADataType, BDataType, true", src)
        self.assertIn("NumWaveGroups, false, 1,", src)

    def test_every_problem_site_gets_widths_and_lockstep(self):
        # bf16 (1, 1, 8): VectorSizeA/B = 1 and _VectorSize = min(1*2, 1*2) = 2 bytes.
        for variant in GemmVariant:
            src = self._src((1, 1, 8), variant)
            self.assertRegex(src, r"(ADataType, BDataType|AsDataType, BsDataType), true, 1, 1>", variant)
            self.assertRegex(src, r"Preshuffle, 2>", variant)
            self.assertIn("true, 8", src, variant)
            native = self._src((0, 0, 0), variant)
            self.assertNotIn("DataType, true,", native, variant)
            self.assertNotIn("Preshuffle, ", native, variant)

    def test_lockstep_bytes(self):
        self.assertEqual(gemm_lockstep_vector_bytes((1, 8, 8), "bf16", "bf16"), 2)
        self.assertEqual(gemm_lockstep_vector_bytes((4, 8, 8), "fp8", "fp8"), 4)
        self.assertEqual(gemm_lockstep_vector_bytes((8, 8, 8), "bf16", "bf16"), 16)

    def test_lockstep_bytes_skip_gfx1250_transpose_load(self):
        # gfx1250 transpose-loads col-major A / row-major B from LDS with a
        # 16-byte pack: if either operand does, the shared knob stays at 16.
        lock = gemm_lockstep_vector_bytes
        self.assertEqual(lock((8, 1, 1), "bf16", "bf16", "rrr", "gfx1250"), 16)
        self.assertEqual(lock((1, 8, 8), "bf16", "bf16", "crr", "gfx1250:xnack-"), 16)
        self.assertEqual(lock((1, 8, 8), "bf16", "bf16", "rcr", "gfx1250"), 2)
        self.assertEqual(lock((4, 8, 8), "fp16", "fp16", "rrr", "gfx1250"), 16)
        self.assertEqual(lock((8, 4, 8), "fp16", "fp16", "ccr", "gfx1250"), 16)
        self.assertEqual(lock((2, 1, 4), "fp32", "fp32", "crr", "gfx1250"), 4)
        self.assertEqual(lock((8, 1, 1), "bf16", "bf16", "rrr", "gfx950"), 2)


class TestWidthSweep(unittest.TestCase):
    def test_aligned_problem_is_native_only(self):
        self.assertEqual(gemm_vector_size_sweep((8, 8, 8), "bf16", "bf16", "bf16"), [(0, 0, 0)])

    def test_only_offending_tensor_sweeps(self):
        self.assertEqual(
            gemm_vector_size_sweep((4, 8, 8), "bf16", "bf16", "bf16"),
            [(1, 8, 8), (2, 8, 8), (4, 8, 8)],
        )
        self.assertEqual(gemm_vector_size_sweep((1, 1, 8), "bf16", "bf16", "bf16"), [(1, 1, 8)])
        self.assertEqual(len(gemm_vector_size_sweep((2, 4, 8), "bf16", "bf16", "bf16")), 6)


class TestHeaderLookup(unittest.TestCase):
    """Name-based header fallbacks must never pick a reduced-width kernel."""

    def test_fixed_vector_headers_skipped(self):
        import tempfile

        import ctypes_utils

        stem = "gemm_bf16_rcr_compv3_cshuffle_intrawave_True_True_True_False_256x256x64_2x2x1_32x32x16"
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            vec = d / f"{stem}_vec1_1_8.hpp"
            vec.touch()
            self.assertIsNone(ctypes_utils._parse_gemm_header_metadata(vec))
            self.assertEqual(ctypes_utils._glob_native(d, "gemm_*.hpp"), [])
            native = d / f"{stem}.hpp"
            native.touch()
            self.assertEqual(ctypes_utils._glob_native(d, "gemm_*.hpp"), [native])
            self.assertIsNotNone(ctypes_utils._parse_gemm_header_metadata(native))


class TestExpandSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gemm_utils import expand_sweep

        cfg = (
            DISPATCHER_DIR.parent / "tile_engine" / "ops" / "gemm" / "configs"
            / "default_ci_config.json"
        )
        # Driver path: K=257 makes the fallback sweep (0, 0, 0) and (1, 1, 8).
        cls.vfb = VectorFallback([dict(M=393216, N=256, K=257)], "rcr", "bf16", "standard")
        cls.cfgs = expand_sweep(
            str(cfg), "gfx950", dtype="bf16", layout="rcr", variant="standard",
            **cls.vfb.expand_kwargs,
        )

    def test_fallback_configs_and_rejects(self):
        cfgs, rejects = self.cfgs, self.vfb.rejects
        self.assertEqual(self.vfb.expand_kwargs["vector_sizes"], [(0, 0, 0), (1, 1, 8)])
        native = [c for c in cfgs if not any(c.vector_sizes)]
        fixed = [c for c in cfgs if any(c.vector_sizes)]
        self.assertTrue(native and fixed)
        self.assertTrue(all(c.epilogue == "cshuffle" for c in fixed))
        self.assertTrue(all(c.name.endswith("_vec1_1_8") for c in fixed))
        # The CI config also sweeps the default epilogue, which cannot take fixed widths.
        self.assertTrue(any("epilogue default" in r for r in rejects))
        self.assertEqual(len({c.name for c in cfgs}), len(cfgs))

    def test_max_kernels_keeps_fixed_width_variants(self):
        # --max-kernels counts native kernels and keeps their fixed-width
        # variants, else --max-kernels 1 leaves K=257 without a kernel.
        one = self.vfb.limit_base_kernels(self.cfgs, 1)
        self.assertEqual([any(c.vector_sizes) for c in one], [False, True])
        self.assertEqual(self.vfb.limit_base_kernels(self.cfgs, 0), self.cfgs)
        # Without the fallback it stays the plain slice.
        off = VectorFallback([], "rcr", "bf16", "standard", disabled=True)
        self.assertEqual(off.limit_base_kernels(self.cfgs, 2), self.cfgs[:2])

    def test_fixed_widths_force_padding(self):
        cfg = GemmKernelConfig(
            dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32",
            layout_a="row", layout_b="col", layout_c="row", pipeline="compv3",
            epilogue="cshuffle", gfx_arch="gfx950", pad_m=False, pad_n=False, pad_k=False,
            tile_m=256, tile_n=256, tile_k=64, wave_m=2, wave_n=2, wave_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
        )
        self.assertEqual(cfg.with_vector_sizes((0, 0, 0))[0].pad_k, False)
        fixed, _ = cfg.with_vector_sizes((1, 1, 8))
        self.assertTrue(fixed.pad_m and fixed.pad_n and fixed.pad_k)

    def test_codegen_json_carries_widths(self):
        cfg, _ = _bridge_config(vec=(1, 1, 8))
        tr = cfg.to_codegen_json()["trait_config"]
        self.assertEqual(
            (tr["vector_size_a"], tr["vector_size_b"], tr["vector_size_c"]),
            ([1], [1], [8]),
        )
        self.assertEqual(replace(cfg).to_dict()["vector_sizes"], [1, 1, 8])


if __name__ == "__main__":
    unittest.main()
