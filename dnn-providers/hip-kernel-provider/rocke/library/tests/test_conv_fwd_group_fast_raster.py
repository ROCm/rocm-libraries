# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests for the forward implicit-GEMM group-fast launch raster
(``ImplicitGemmConvSpec.group_fast_raster``, "gfr"): validator gates, kernel
names, the launch grid, the identity / label round trip, cache serving
(grouped problems only), the RDNA4 AOT rows, and that the default
leaves names, hashes and IR untouched."""

import unittest

from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_implicit_gemm import (
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
    implicit_gemm_conv_grid,
    is_valid_spec,
)
from kernels.conv.aot_jobs import (
    CACHE_FWD_RDNA4_KNOB_ROWS,
    _fwd_rdna4_knob_sets,
    job_for_identity,
)
from kernels.conv.kernel_identity import KernelIdentity, identity_supports_problem
from rocke.core.ir_print import print_ir

_WMMA = "gfx1201"
_PROBLEM = ConvProblem(N=2, Hi=14, Wi=14, C=448, K=448, Y=3, X=3, pH=1, pW=1, groups=8)


def _spec(problem=_PROBLEM, **kw):
    base = dict(
        problem=problem,
        data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"),
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        wave_size=32,
        pipeline="mem",
        epilogue="cshuffle",
        vector_size_a=8,
        vector_size_b=8,
        vector_size_c=8,
    )
    base.update(kw)
    return ImplicitGemmConvSpec(**base)


def _label(**tags):
    extra = "".join(f"-{t}" for t in tags.get("tags", ()))
    return (
        "fwd-implicit_gemm-fp16-2d-t64x64x64-w2x2-a16x16x16-v8x8x8-mem-cshuffle-grp"
        f"{extra}-wave32-gfx1201-llvm24"
    )


class TestSpec(unittest.TestCase):
    def test_name_and_grid(self):
        off, on = _spec(), _spec(group_fast_raster=True)
        self.assertNotIn("gfr", off.kernel_name())
        self.assertTrue(on.kernel_name().endswith("_gfr"))
        gn, gm, g = implicit_gemm_conv_grid(off)
        self.assertEqual(implicit_gemm_conv_grid(on), (gn, g, gm))

    def test_launch_grid_swap_lives_in_one_place(self):
        """ConvArgs.grid and the sweep's grid agree with the kernel's grid."""
        from benchmarks.common.benchmark_implicit_gemm_conv import _grid_for_spec
        from kernels.common.conv_args import ConvArgs

        for kw in (dict(), dict(group_fast_raster=True)):
            spec = _spec(**kw)
            args = ConvArgs.from_problem(
                spec.problem, tile_m=spec.tile_m, tile_n=spec.tile_n
            )
            gfr = spec.group_fast_raster
            want = implicit_gemm_conv_grid(spec)
            self.assertEqual(args.grid(group_fast_raster=gfr), want)
            self.assertEqual(_grid_for_spec(spec, spec.problem), want)
        gx, gy, gz = args.grid()
        self.assertEqual(args.grid(group_fast_raster=True), (gx, gz, gy))
        self.assertEqual(args.grid(group_merge=1, split_k=1), (gx, gy, gz))

    def test_validator(self):
        self.assertTrue(is_valid_spec(_spec(group_fast_raster=True), _WMMA)[0])
        for kw, why in (
            (dict(split_k=2), "split-K"),
            (dict(pipeline="wavelet"), "wavelet"),
        ):
            ok, msg = is_valid_spec(_spec(group_fast_raster=True, **kw), _WMMA)
            self.assertFalse(ok)
            self.assertIn(why, msg)

    def test_grid_refuses_bad_group_merge(self):
        """A merge factor that is not a positive divisor of the group count
        would silently drop group slices or give a non-integer grid axis."""
        from kernels.common.conv_args import ConvArgs

        args = ConvArgs.from_problem(_PROBLEM, tile_m=64, tile_n=64)
        for bad in (0, -1, 3, 16, 2.0, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                args.grid(group_merge=bad)
        self.assertEqual(args.grid(group_merge=2)[2], 4)
        self.assertEqual(args.grid(group_merge=8)[2], 1)

    def test_default_ir_untouched(self):
        def ir(**kw):
            return print_ir(build_implicit_gemm_conv(_spec(**kw), arch=_WMMA))

        self.assertNotEqual(ir(), ir(group_fast_raster=True))
        self.assertEqual(ir(), ir(group_fast_raster=False))


class TestRdna4OnlyKnobs(unittest.TestCase):
    """uniform_k_decode / lds_swizzle / group_fast_raster are checked on the
    gfx12 RDNA targets only, so every other target refuses them."""

    _KNOBS = ("uniform_k_decode", "lds_swizzle", "group_fast_raster")

    def test_gfx12_rdna_accepts(self):
        for arch in ("gfx1201", "gfx1200"):
            for knob in self._KNOBS:
                ok, why = is_valid_spec(_spec(**{knob: True}), arch)
                self.assertTrue(ok, f"{arch} {knob}: {why}")

    def test_other_wmma_targets_refuse(self):
        for arch, atom_k in (("gfx1151", 16), ("gfx1250", 32)):
            for knob in self._KNOBS:
                spec = _spec(warp_tile_k=atom_k, **{knob: True})
                self.assertTrue(is_valid_spec(_spec(warp_tile_k=atom_k), arch)[0])
                ok, why = is_valid_spec(spec, arch)
                self.assertFalse(ok, f"{arch} {knob}")
                self.assertIn("only on the gfx12 RDNA targets", why)
                self.assertIn(knob, why)

    def test_mfma_targets_refuse(self):
        from kernels.common.conv_implicit_gemm import fwd_rdna4_knob_reason
        from rocke.core.arch import ArchTarget

        for arch in ("gfx942", "gfx950"):
            target = ArchTarget.from_gfx(arch)
            for knob in self._KNOBS:
                why = fwd_rdna4_knob_reason(_spec(**{knob: True}), target)
                self.assertIsNotNone(why, f"{arch} {knob}")
                self.assertIn(knob, why)
        self.assertIsNone(fwd_rdna4_knob_reason(_spec(), ArchTarget.from_gfx("gfx950")))


class TestIdentity(unittest.TestCase):
    def test_label_round_trip_and_hash(self):
        ident = KernelIdentity.from_label(_label(tags=("gfr", "ukd")))
        self.assertTrue(ident.group_fast_raster)
        self.assertEqual(KernelIdentity.from_label(ident.label()), ident)
        plain = KernelIdentity.from_label(_label(tags=("ukd",)))
        self.assertFalse(plain.group_fast_raster)
        self.assertNotIn("gfr", plain.label())
        self.assertNotEqual(plain.stable_hash(), ident.stable_hash())

    def test_serves_grouped_problems_only(self):
        ident = KernelIdentity.from_label(_label(tags=("gfr",)))
        ok, _ = identity_supports_problem(ident, _PROBLEM, llvm_flavor="llvm24")
        self.assertTrue(ok)
        dense = ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1)
        ok, why = identity_supports_problem(ident, dense, llvm_flavor="llvm24")
        self.assertFalse(ok)
        self.assertIn("grouped", why)


class TestAot(unittest.TestCase):
    _GFR_ROWS = [r for r in CACHE_FWD_RDNA4_KNOB_ROWS if r[8]]

    def _geometry(self, row, pipeline="mem"):
        """The gfx1201 forward geometry of a knob row (16x16 warp tiles,
        cshuffle) on ``pipeline``."""
        from kernels.conv.aot_jobs import _geometries
        from rocke.core.arch import ArchTarget

        target = ArchTarget.from_gfx("gfx1201")
        for g in _geometries(target, "wmma", "fp16", "fp16", "fwd"):
            if tuple(g[:5]) == tuple(row[:5]) and g[5:8] == (16, pipeline, "cshuffle"):
                return g
        raise AssertionError(f"no geometry for {row}")

    def test_rows_rdna_only(self):
        self.assertTrue(self._GFR_ROWS)
        geo = self._geometry(self._GFR_ROWS[0])
        for arch in ("gfx1201", "gfx1200", "gfx12-generic"):
            sets = _fwd_rdna4_knob_sets(arch, geo)
            self.assertTrue(any(k.get("group_fast_raster") for k in sets))
        for arch in ("gfx1250", "gfx950", "gfx11-generic"):
            self.assertEqual(_fwd_rdna4_knob_sets(arch, geo), [])
        wavelet = self._geometry(self._GFR_ROWS[0], "wavelet")
        self.assertEqual(_fwd_rdna4_knob_sets("gfx1201", wavelet), [])

    def test_every_row_is_a_job(self):
        for tm, tn, tk, wm, wn, ukd, lxs, rtp, _gfr in self._GFR_ROWS:
            tags = ["gfr"]
            if lxs:
                tags.append("lxs")
            if rtp:
                tags.append(f"rtp{rtp}")
            if ukd:
                tags.append("ukd")
            label = (
                f"fwd-implicit_gemm-fp16-2d-t{tm}x{tn}x{tk}-w{wm}x{wn}-a16x16x16-"
                f"v8x8x8-mem-cshuffle-grp-{'-'.join(tags)}-wave32-gfx1201-llvm24"
            )
            ident = KernelIdentity.from_label(label)
            job = job_for_identity(ident)
            self.assertTrue(job.spec_kwargs["group_fast_raster"])


if __name__ == "__main__":
    unittest.main()
