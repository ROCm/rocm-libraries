# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side gate for the RDNA4 forward additions: the tile_k 128 axis and
the listed forward rows of the AOT grid, and the forward register-prefetch K
loop (``regt_prefetch``).

No GPU and no comgr: the spec gate, the kernel name, the lowered IR and the
AOT grid. The numerical check is ``--run-from-cache --verify`` of the cached
kernels.
"""

from __future__ import annotations

import unittest


def _fwd(**kw):
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
    from kernels.common.conv_implicit_gemm import ImplicitGemmConvSpec

    base = dict(
        problem=ConvProblem(N=2, Hi=14, Wi=14, C=64, K=96, Y=3, X=3, pH=1, pW=1),
        data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"),
        tile_m=128,
        tile_n=256,
        tile_k=64,
        warp_m=2,
        warp_n=4,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        wave_size=32,
        epilogue="cshuffle",
        vector_size_a=8,
        vector_size_b=8,
        vector_size_c=8,
    )
    base.update(kw)
    return ImplicitGemmConvSpec(**base)


class TestFwdRegtPrefetch(unittest.TestCase):
    def test_default_is_off_and_untagged(self):
        self.assertNotIn("rtp", _fwd().kernel_name())
        self.assertTrue(_fwd(regt_prefetch=1).kernel_name().endswith("_rtp1"))

    def test_gates(self):
        from kernels.common.conv_implicit_gemm import is_valid_spec

        for arch in ("gfx1201", "gfx1200", "gfx12-generic"):
            with self.subTest(arch=arch):
                self.assertEqual(
                    is_valid_spec(_fwd(regt_prefetch=1), arch), (True, "ok")
                )
        # The LDS-only barrier that keeps the prefetch in flight is gfx12
        # RDNA's; the other WMMA targets refuse the loop.
        for arch in ("gfx11-generic", "gfx1151", "gfx1250"):
            with self.subTest(arch=arch):
                kw = dict(warp_tile_k=32) if arch == "gfx1250" else {}
                ok, why = is_valid_spec(_fwd(regt_prefetch=1, **kw), arch)
                self.assertFalse(ok)
                self.assertIn("gfx12 RDNA", why)
        ok, why = is_valid_spec(
            _fwd(regt_prefetch=1, wave_size=64, warp_tile_m=32, warp_tile_n=32),
            "gfx950",
        )
        self.assertFalse(ok)
        self.assertIn("gfx12 RDNA", why)
        ok, why = is_valid_spec(_fwd(regt_prefetch=1, pipeline="wavelet"), "gfx1201")
        self.assertFalse(ok)
        for mode in (2, 3):
            ok, why = is_valid_spec(_fwd(regt_prefetch=mode), "gfx1201")
            self.assertFalse(ok)
            self.assertIn("must be 0 or 1", why)

    def test_lowered_loop_keeps_vmem_in_flight(self):
        from kernels.common.conv_implicit_gemm import build_implicit_gemm_conv
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        plain = lower_kernel_to_llvm(
            build_implicit_gemm_conv(_fwd(), arch="gfx1201"), arch="gfx1201"
        )
        rpf = lower_kernel_to_llvm(
            build_implicit_gemm_conv(_fwd(regt_prefetch=1), arch="gfx1201"),
            arch="gfx1201",
        )
        self.assertNotIn("s_barrier_signal", plain)
        # Two LDS-only inline-asm barriers in the loop, and the load pin.
        self.assertEqual(rpf.count("s_wait_dscnt 0x0"), 2)
        self.assertIn("sched.barrier", rpf)


_ARCHS = (
    ("gfx1201", 32, "wmma", True),
    ("gfx1200", 32, "wmma", True),
    ("gfx12-generic", 32, "wmma", True),
    ("gfx11-generic", 32, "wmma", False),
    ("gfx1250", 32, "wmma", False),
    ("gfx950", 64, "mma", False),
)


def _knob_tags(identity) -> tuple:
    return (
        int(identity.uniform_k_decode),
        int(identity.lds_swizzle),
        identity.regt_prefetch,
        int(identity.group_fast_raster),
    )


class TestFwdRdnaGrid(unittest.TestCase):
    def test_knob_rows_are_rdna_only_and_each_one_job(self):
        from kernels.conv.aot_jobs import (
            CACHE_FWD_RDNA4_KNOB_ROWS,
            CACHE_FWD_RDNA4_VEC,
            _fwd_jobs,
            _geometries,
        )
        from rocke.core.arch import ArchTarget

        tiles = {tuple(r[:5]) for r in CACHE_FWD_RDNA4_KNOB_ROWS}
        want = sorted(tuple(r) for r in CACHE_FWD_RDNA4_KNOB_ROWS)
        self.assertEqual(len(set(want)), len(want))
        for arch, wave, family, rdna in _ARCHS:
            with self.subTest(arch=arch):
                t = ArchTarget.from_gfx(arch)
                geos = [
                    g
                    for g in _geometries(t, family, "fp16", "fp16", "fwd")
                    if tuple(g[:5]) in tiles
                ]
                jobs = list(_fwd_jobs(arch, "fp16", wave, family, t, geometries=geos))
                knob = [j for j in jobs if any(_knob_tags(j.identity))]
                if not rdna:
                    self.assertEqual(knob, [])
                    continue
                got = sorted(
                    (
                        j.spec_kwargs["tile_m"],
                        j.spec_kwargs["tile_n"],
                        j.spec_kwargs["tile_k"],
                        j.spec_kwargs["warp_m"],
                        j.spec_kwargs["warp_n"],
                        *_knob_tags(j.identity),
                    )
                    for j in knob
                )
                self.assertEqual(got, want)
                for j in knob:
                    self.assertTrue(j.identity.grouped)
                    self.assertEqual(j.spec_kwargs["pipeline"], "mem")
                    self.assertEqual(
                        j.spec_kwargs["vector_size_a"], CACHE_FWD_RDNA4_VEC
                    )

    def test_tile_k_128_builds_only_the_listed_rows(self):
        from kernels.conv.aot_jobs import (
            CACHE_FWD_RDNA4_DEEP_K_ROWS,
            CACHE_FWD_RDNA4_KNOB_ROWS,
            _fwd_jobs,
            _geometries,
        )
        from rocke.core.arch import ArchTarget

        t = ArchTarget.from_gfx("gfx1201")
        geos = [g for g in _geometries(t, "wmma", "fp16", "fp16", "fwd") if g[2] == 128]
        jobs = list(_fwd_jobs("gfx1201", "fp16", 32, "wmma", t, geometries=geos))
        plain = sorted(
            (
                j.spec_kwargs["tile_m"],
                j.spec_kwargs["tile_n"],
                j.spec_kwargs["warp_m"],
                j.spec_kwargs["warp_n"],
                j.spec_kwargs["pipeline"],
                j.spec_kwargs.get("group_merge", 0),
            )
            for j in jobs
            if not any(_knob_tags(j.identity))
        )
        self.assertEqual(plain, sorted(tuple(r) for r in CACHE_FWD_RDNA4_DEEP_K_ROWS))
        n_knob = sum(1 for r in CACHE_FWD_RDNA4_KNOB_ROWS if r[2] == 128)
        self.assertEqual(len(jobs), len(plain) + n_knob)

    def test_tile_k_128_axis_is_rdna4_forward_only_and_appended(self):
        from kernels.conv.aot_jobs import (
            CACHE_TILE_K,
            CACHE_WGRAD_DEEPK_GEOMETRIES,
            _geometries,
            _grid_axes,
        )
        from rocke.core.arch import ArchTarget

        for arch, wave, family, rdna4 in (
            ("gfx1201", 32, "wmma", True),
            ("gfx11-generic", 32, "wmma", False),
            ("gfx1250", 32, "wmma", False),
            ("gfx950", 64, "mma", False),
        ):
            for direction in ("fwd", "wgrad", "dgrad"):
                with self.subTest(arch=arch, direction=direction):
                    t = ArchTarget.from_gfx(arch)
                    # WMMA wgrad's own deep-K geometries (one is tile_k 128)
                    # are not this axis.
                    geos = [
                        g
                        for g in _geometries(t, family, "fp16", "fp16", direction)
                        if tuple(g[:5]) not in CACHE_WGRAD_DEEPK_GEOMETRIES
                    ]
                    deep = [i for i, g in enumerate(geos) if g[2] == 128]
                    self.assertEqual(bool(deep), rdna4 and direction == "fwd")
                    if deep:
                        # One contiguous block after every geometry of the
                        # base product (the direction's tile and warp axes at
                        # CACHE_TILE_K). Explicitly listed extra tiles are not
                        # part of the product and may come before or after.
                        self.assertEqual(deep, list(range(deep[0], deep[-1] + 1)))
                        tile_mn, warp_mn = _grid_axes(direction)
                        base = [
                            i
                            for i, g in enumerate(geos)
                            if g[2] in CACHE_TILE_K
                            and g[0] in tile_mn
                            and g[1] in tile_mn
                            and g[3] in warp_mn
                            and g[4] in warp_mn
                        ]
                        self.assertTrue(base)
                        self.assertLess(max(base), min(deep))


class TestFwdRpfWithDecodeKnobs(unittest.TestCase):
    """regt_prefetch combined with uniform_k_decode (its own mode-1 loop)."""

    def test_gates(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem
        from kernels.common.conv_implicit_gemm import is_valid_spec

        self.assertEqual(
            is_valid_spec(_fwd(regt_prefetch=1, uniform_k_decode=True), "gfx1201"),
            (True, "ok"),
        )
        self.assertTrue(
            _fwd(regt_prefetch=1, uniform_k_decode=True)
            .kernel_name()
            .endswith("_ukd_rtp1")
        )

    def test_ukd_loop_keeps_vmem_in_flight(self):
        from kernels.common.conv_implicit_gemm import build_implicit_gemm_conv
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        ir = lower_kernel_to_llvm(
            build_implicit_gemm_conv(
                _fwd(regt_prefetch=1, uniform_k_decode=True), arch="gfx1201"
            ),
            arch="gfx1201",
        )
        self.assertEqual(ir.count("s_wait_dscnt 0x0"), 2)
        self.assertIn("sched.barrier", ir)


if __name__ == "__main__":
    unittest.main()
