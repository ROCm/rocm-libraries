# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the GEMM coverage / correctness levers.

* The launch rules (``launch_shape_error``) refuse every configuration class the
  can fault or compute wrong results: split-K slices
  that are not whole tiles under ``pad_k``, ragged M / N tiles whose loads read
  past the operand, DTL bodies on K % tile_k != 0 (no K mask), grids past the
  HIP limits.
* ``pad_row_clamp``, ``pad_k_fast``, ``split_k_aligned`` and ``dtl_k_tail`` make
  those classes correct (GPU-checked by a fuzz harness); each is
  default-off, scoped by ``is_valid_spec`` and flagged in the kernel name;
  all but ``split_k_aligned`` (mirrored by the C++ engine) are Python-engine
  only.
* The dispatcher admits every RCR shape on gfx942 / gfx950 (f16 and bf16) with
  a spec the launch rules accept; aligned shapes keep their previous kernels.
"""

from __future__ import annotations

import contextlib
import random
import re
import unittest
import warnings
from dataclasses import replace
from unittest import mock

from rocke.core import backend as _be
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.dispatch.gemm import (
    GemmRequest,
    dispatch_gemm_bf16,
    dispatch_gemm_bf16_all,
    dispatch_gemm_fp16,
    dispatch_gemm_fp16_all,
)
from rocke.dispatch.gemm import common as _gc
from rocke.dispatch.gemm import skinny as _sk
from rocke.dispatch.gemm.support import request_shape_supported
from rocke.instances.common.gemm_skinny import SkinnyGemmSpec
from rocke.instances.common.gemm_skinny import is_valid_spec as skinny_is_valid_spec
from rocke.instances.common.gemm_skinny import skinny_launch_error
from rocke.instances.common.gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
    launch_shape_error,
    mono_data_spec,
)

_G1 = ("pad_row_clamp", "pad_k_fast", "split_k_aligned", "dtl_k_tail")
_DEC = TileSpec(16, 64, 32, 1, 4, 1, 16, 16, 32)
_GEN = TileSpec(128, 128, 32, 2, 2, 1, 32, 32, 16)
_PADALL = TraitSpec(
    pipeline="compv4", epilogue="default", pad_m=True, pad_n=True, pad_k=True
)
# the small ring (DTL, 3 stages, exact bounds)
_RING = TraitSpec(
    pipeline="compv4",
    epilogue="default",
    pad_m=True,
    pad_n=True,
    pad_k=True,
    direct_to_lds=True,
    dtl_prefetch=True,
    dtl_stages=3,
    dtl_tight_rsrc=True,
    dtl_hoist=True,
    dtl_frag_prefetch=True,
    lds_swizzle=True,
    chiplet_swizzle=True,
    chiplet_chunk_size=32,
    dtl_cache_b=0,
    mfma_interleave=True,
    ilv_merge=True,
    ilv_barrier_after=2,
    interwave_setprio=True,
    pad_fast_tile=True,
)
_RT = TileSpec(128, 128, 64, 2, 2, 1, 16, 16, 32)


def _g1_only():
    """The coverage dispatch rules in isolation (the gemm_skinny region /
    fallback and its small-ring candidate switched off; they outrank the
    padded fallbacks on the decode and few-tile shapes these tests pin)."""
    return _sk_knobs(False)


class _sk_knobs(contextlib.ContextDecorator):
    def __init__(self, on: bool):
        self._ps = [
            mock.patch.object(_sk, k, on)
            for k in ("SKINNY_DISPATCH", "SKINNY_FALLBACK", "RING_ANY")
        ]

    def __enter__(self):
        for p in self._ps:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._ps:
            p.stop()
        return False


def _spec(tile, trait, dt="bf16"):
    return UniversalGemmSpec(name="t", tile=tile, trait=trait, data=mono_data_spec(dt))


def _ir(s, arch="gfx950"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lower_kernel_to_llvm(build_universal_gemm(s, arch=arch), arch=arch)


class TestLaunchRules(unittest.TestCase):
    """The failing configuration classes are refused; their lever twins admitted."""

    def test_split_k_pad_k_ragged_slice(self):
        # 5x4736x2624: K / 4 = 656 is not whole tiles of 32 (wrong results)
        s = _spec(_DEC, replace(_PADALL, split_k=4, pad_row_clamp=True))
        err = launch_shape_error(s, 5, 4736, 2624)
        self.assertIn("split_k_aligned", err)
        ok = replace(s, trait=replace(s.trait, split_k_aligned=True))
        self.assertIsNone(launch_shape_error(ok, 5, 4736, 2624))
        # aligned slices need no K % split_k either
        self.assertIsNone(launch_shape_error(ok, 5, 4736, 2623))
        # whole-tile slices stay legal without the lever
        self.assertIsNone(launch_shape_error(s, 5, 4736, 4096))

    def test_ragged_rows_need_bounded_loads(self):
        # M = 2 on a 128-row padded tile: the loads of rows 2..127 read past A
        s = _spec(replace(_DEC, tile_m=128, warp_m=1), _PADALL)
        for m, n in ((2, 192), (6614, 609), (16, 5056)):
            with self.subTest(m=m, n=n):
                self.assertIn("pad_row_clamp", launch_shape_error(s, m, n, 792))
        c = replace(s, trait=replace(s.trait, pad_row_clamp=True))
        self.assertIsNone(launch_shape_error(c, 2, 192, 8128))
        # aligned rows need nothing
        self.assertIsNone(launch_shape_error(s, 256, 128, 8128))

    def test_dtl_ragged_rows_need_tight_bounds(self):
        s = _spec(_RT, replace(_RING, dtl_tight_rsrc=False))
        self.assertIn("dtl_tight_rsrc", launch_shape_error(s, 5174, 521, 128))
        self.assertIsNone(launch_shape_error(_spec(_RT, _RING), 5174, 521, 128))

    def test_dtl_k_tail(self):
        s = _spec(_RT, _RING)
        for k in (96, 8480, 2528):
            with self.subTest(k=k):
                self.assertIn("dtl_k_tail", launch_shape_error(s, 1024, 1024, k))
        kt = replace(s, trait=replace(s.trait, dtl_k_tail=True))
        self.assertIsNone(launch_shape_error(kt, 1024, 1024, 8480))
        self.assertIn("multiple of 8", launch_shape_error(kt, 1024, 1024, 8484))

    def test_grid_and_range(self):
        s = _spec(_DEC, replace(_PADALL, pad_row_clamp=True))
        self.assertIn("grid.y", launch_shape_error(s, 16 * 65536, 64, 64))
        self.assertIn("i32", launch_shape_error(s, 600000, 64, 4096))
        self.assertIn("empty", launch_shape_error(s, 0, 64, 64))


class TestLeverScope(unittest.TestCase):
    def test_default_off_and_python_only(self):
        tr = TraitSpec()
        listed = dict(_be._PYTHON_ONLY_GEMM_LEVERS)
        for f in _G1:
            with self.subTest(f=f):
                self.assertFalse(getattr(tr, f))
                if f == "split_k_aligned":
                    # mirrored by the C++ engine: converted, not refused
                    self.assertNotIn(f, listed)
                    _be._refuse_python_only_levers(replace(tr, **{f: True}))
                    continue
                self.assertIn(f, listed)
                self.assertEqual(listed[f], False)
                with self.assertRaises(NotImplementedError):
                    _be._refuse_python_only_levers(replace(tr, **{f: True}))

    def test_scope(self):
        def ok(tile, **kw):
            return is_valid_spec(_spec(tile, replace(_PADALL, **kw)), arch="gfx950")[0]

        self.assertTrue(ok(_DEC, pad_row_clamp=True))
        self.assertFalse(ok(_DEC, pad_row_clamp=True, pad_m=False, pad_n=False))
        self.assertTrue(ok(_DEC, pad_k_fast=True))
        self.assertFalse(ok(_DEC, pad_k_fast=True, pad_k=False))
        self.assertTrue(ok(_DEC, split_k=4, split_k_aligned=True))
        self.assertFalse(ok(_DEC, split_k_aligned=True))
        self.assertFalse(ok(_DEC, split_k=4, split_k_aligned=True, pad_k=False))
        self.assertFalse(ok(_DEC, dtl_k_tail=True))
        ring = lambda **kw: is_valid_spec(  # noqa: E731
            _spec(_RT, replace(_RING, **kw)), arch="gfx950"
        )[0]
        self.assertTrue(ring(dtl_k_tail=True))
        self.assertFalse(ring(dtl_k_tail=True, dtl_stages=0, dtl_tight_rsrc=False))
        self.assertFalse(ring(pad_row_clamp=True))
        self.assertFalse(ring(pad_k_fast=True))
        # the builder raises on it: the gate must refuse it
        self.assertFalse(ok(_DEC, dtl_prefetch=True))

    def test_names_distinct(self):
        base = _spec(_DEC, replace(_PADALL, split_k=4))
        names = {base.kernel_name()}
        for f in _G1:
            tr = replace(base.trait, **{f: True})
            names.add(replace(base, trait=tr).kernel_name())
        self.assertEqual(len(names), 1 + len(_G1))


class TestEmission(unittest.TestCase):
    def test_dtl_k_tail_peeled(self):
        # mode 2: the K loop is unchanged; one extra tile of DMA issues (with
        # the tail offsets) and MFMAs after it, a 0/1-trip loop
        s = _spec(_RT, _RING)
        base = _ir(s)
        pk = _ir(replace(s, trait=replace(s.trait, dtl_k_tail=2)))
        sel = _ir(replace(s, trait=replace(s.trait, dtl_k_tail=1)))
        dma = re.compile(r"call void @llvm\.amdgcn\.[a-z.]*buffer\.load\.lds")
        n_base, n_pk = len(dma.findall(base)), len(dma.findall(pk))
        passes = n_pk - n_base
        self.assertGreater(passes, 0)
        # the tail selects exist only once per tail DMA, not per loop issue
        d_pk = pk.count("= select") - base.count("= select")
        d_sel = sel.count("= select") - base.count("= select")
        self.assertLessEqual(d_pk, 2 * passes)  # tail offsets + the tail resources
        self.assertGreater(d_sel, d_pk)  # mode 1 selects at every issue
        self.assertGreater(pk.count("mfma"), base.count("mfma"))
        self.assertIn(
            "dkt2", replace(s, trait=replace(s.trait, dtl_k_tail=2)).kernel_name()
        )
        self.assertFalse(
            is_valid_spec(_spec(_RT, replace(_RING, dtl_k_tail=3)), arch="gfx950")[0]
        )

    def test_row_clamp(self):
        s = _spec(_GEN, _PADALL)
        base = _ir(s)
        clamp = _ir(replace(s, trait=replace(s.trait, pad_row_clamp=True)))
        self.assertEqual(base.count("smin"), 0)
        self.assertGreater(clamp.count("smin"), 0)
        # only the padded operand is clamped
        only_m = _ir(
            replace(s, trait=replace(s.trait, pad_n=False, pad_row_clamp=True))
        )
        self.assertLess(only_m.count("smin"), clamp.count("smin"))
        self.assertGreater(only_m.count("smin"), 0)

    def test_pad_k_fast_branches_to_vector_loads(self):
        s = _spec(_GEN, _PADALL)
        vec = re.compile(r"load <\d+ x (half|bfloat|i16)>, ptr addrspace\(1\)")
        base = _ir(s)
        fast = _ir(replace(s, trait=replace(s.trait, pad_k_fast=True)))
        self.assertEqual(len(vec.findall(base)), 0)
        self.assertGreater(len(vec.findall(fast)), 0)
        self.assertGreater(fast.count(" br i1"), base.count(" br i1"))

    def test_split_k_aligned_slices(self):
        s = _spec(_DEC, replace(_PADALL, split_k=4))
        legacy = _ir(s)
        al = _ir(replace(s, trait=replace(s.trait, split_k_aligned=True)))
        self.assertNotEqual(legacy, al)
        self.assertEqual(legacy.count("smin"), 0)
        self.assertGreater(al.count("smin"), 0)  # k_hi = min(K, k_lo + ks)
        # masked lanes of an empty slice (k_off >= K) read the row start, not
        # k_off past it: the safe column is -k_off
        neg = re.compile(r"= sub (nsw )?i32 0, ")
        self.assertGreater(len(neg.findall(al)), len(neg.findall(legacy)))

    def test_dtl_k_tail_selects(self):
        s = _spec(_RT, _RING)
        base = _ir(s)
        kt = _ir(replace(s, trait=replace(s.trait, dtl_k_tail=True)))
        # one tail select per pass in the loop-invariant offsets + one per DMA
        # issue (the issue takes the tail offsets on the last K tile)
        n_dma = len(
            re.findall(r"call void @llvm\.amdgcn\.[a-z.]*buffer\.load\.lds", kt)
        )
        self.assertGreater(n_dma, 0)
        self.assertGreaterEqual(kt.count("= select") - base.count("= select"), n_dma)
        self.assertNotEqual(base, kt)


def _rand_shapes(seed, n):
    rng = random.Random(seed)
    out = [(1, 1, 1), (3, 5, 7), (129, 77, 13), (30000, 17, 999), (7, 192, 26816)]
    out += [(4099, 4101, 1000), (8192, 8192, 8200), (2, 4096, 8192), (5, 4736, 2624)]
    for _ in range(n):
        dims = []
        for lo, hi in ((1, 32768), (1, 32768), (1, 32768)):
            v = int(2 ** rng.uniform(0, 15))
            a = rng.choice((256, 64, 8, 1))
            dims.append(max(lo, min(hi, v - v % a if v >= a else v)))
        M, N, K = dims
        while M * K > (1 << 31) or N * K > (1 << 31) or M * N > (1 << 31):
            K = max(1, K // 2)
            M = max(1, M // 2) if M * N > (1 << 31) else M
        out.append((M, N, K))
    return out


class TestCoverage(unittest.TestCase):
    """Every RCR shape gets a kernel the launch rules accept."""

    def test_no_refusal(self):
        for arch in ("gfx950", "gfx942"):
            for dt, fn in (("fp16", dispatch_gemm_fp16), ("bf16", dispatch_gemm_bf16)):
                for M, N, K in _rand_shapes(7, 60):
                    with self.subTest(arch=arch, dt=dt, shape=(M, N, K)):
                        req = GemmRequest(M=M, N=N, K=K, arch=arch, dtype=dt)
                        r = fn(req)
                        if isinstance(r.spec, SkinnyGemmSpec):
                            # gemm_skinny serves it
                            self.assertTrue(skinny_is_valid_spec(r.spec, arch)[0])
                            self.assertIsNone(
                                skinny_launch_error(r.spec, M, N, K, r.spec.split)
                            )
                            continue
                        self.assertIsNone(launch_shape_error(r.spec, M, N, K))
                        self.assertTrue(request_shape_supported(req, r.spec)[0])
                        if r.spec.trait.split_k > 1:
                            names = [a["name"] for a in r.signature]
                            if r.spec.trait.split_k_serial:
                                # in-launch split-K reduction
                                self.assertEqual(names[2:5], ["C", "W", "Cnt"])
                            else:
                                self.assertIn("Cf32", names)

    def test_alternatives_launchable(self):
        for M, N, K in ((4608, 4608, 288), (3000, 5000, 808), (4099, 4101, 1000)):
            for dt, fn in (
                ("fp16", dispatch_gemm_fp16_all),
                ("bf16", dispatch_gemm_bf16_all),
            ):
                for r in fn(GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dt)):
                    with self.subTest(shape=(M, N, K), c=r.candidate.name):
                        if isinstance(r.spec, SkinnyGemmSpec):
                            self.assertIsNone(
                                skinny_launch_error(r.spec, M, N, K, r.spec.split)
                            )
                            continue
                        self.assertIsNone(launch_shape_error(r.spec, M, N, K))

    def test_aligned_specs_untouched(self):
        # g1_adapt_spec is the identity on shapes the body tiles exactly
        for M, N, K in ((8192, 8192, 8192), (4096, 4096, 4160), (1536, 1440, 576)):
            req = GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype="bf16")
            for r in dispatch_gemm_bf16_all(req):
                if isinstance(r.spec, SkinnyGemmSpec):
                    continue
                with self.subTest(shape=(M, N, K), c=r.candidate.name):
                    t, tr = r.spec.tile, r.spec.trait
                    if M % t.tile_m == 0 and N % t.tile_n == 0 and K % t.tile_k == 0:
                        self.assertFalse(any(getattr(tr, f) for f in _G1))

    @_g1_only()
    def test_knobs_restore_refusal(self):
        req = GemmRequest(M=130, N=128, K=512, arch="gfx950", dtype="fp16")
        self.assertEqual(
            dispatch_gemm_fp16(req).candidate.name,
            "universal_gemm_fp16_cdna_dtl_ring_any",
        )
        with mock.patch.object(_gc, "G1_RING_ANY", False):
            self.assertEqual(
                dispatch_gemm_fp16(req).candidate.name, "universal_gemm_fp16_cdna_pad"
            )
            with mock.patch.object(_gc, "G1_PAD_FALLBACK", False):
                # the universal padded fallback (lowest
                # priority, registered at import) is what remains
                self.assertEqual(
                    dispatch_gemm_fp16(req).candidate.name,
                    "universal_gemm_fp16_cdna_universal_fallback",
                )
        dec = GemmRequest(M=5, N=4736, K=2624, arch="gfx950", dtype="bf16")
        self.assertTrue(dispatch_gemm_bf16(dec).spec.trait.pad_row_clamp)
        with mock.patch.object(_gc, "G1_ADAPT", False):
            # without the row clamp the global-load decode body may not take a
            # ragged M (the DTL ring with exact bounds still may)
            r = dispatch_gemm_bf16(dec)
            # (with G1_ADAPT off the bf16 decode body keeps the
            # guarded loader, kc_row_guard, which bounds the rows instead)
            if r.candidate.name == "universal_gemm_bf16_cdna_decode":
                self.assertTrue(r.spec.trait.kc_row_guard)
                self.assertFalse(r.spec.trait.pad_row_clamp)
            self.assertIsNone(launch_shape_error(r.spec, 5, 4736, 2624))

    def test_k_tail_needs_enough_tiles(self):
        # K = 96 on tile_k 64 keeps the exact-tile pick
        r = dispatch_gemm_bf16(
            GemmRequest(M=4608, N=4608, K=96, arch="gfx950", dtype="bf16")
        )
        self.assertFalse(r.spec.trait.dtl_k_tail)
        r = dispatch_gemm_bf16(
            GemmRequest(M=8192, N=8192, K=8200, arch="gfx950", dtype="bf16")
        )
        self.assertTrue(r.spec.trait.dtl_k_tail)
        # the beat body of the aligned class (peeled K tail), not the DTL ring
        self.assertTrue(r.spec.trait.hb_sched)
        self.assertEqual(r.spec.trait.dtl_k_tail, 2)

    @_g1_only()
    def test_aligned_split_beats_shallow_factor(self):
        r = dispatch_gemm_bf16(
            GemmRequest(M=7, N=192, K=26816, arch="gfx950", dtype="bf16")
        )
        self.assertTrue(r.spec.trait.split_k_aligned)
        self.assertGreater(r.spec.trait.split_k, 2)
        with mock.patch.dict("os.environ", {"ROCKE_GEMM_SPLIT_K": "off"}):
            r = dispatch_gemm_bf16(
                GemmRequest(M=7, N=192, K=26816, arch="gfx950", dtype="bf16")
            )
            self.assertEqual(r.spec.trait.split_k, 1)


class TestRingAny(unittest.TestCase):
    @_g1_only()
    def test_split_rule_and_retile(self):
        # few tiles + long K: the split-K fallback, not one ring pass
        r = dispatch_gemm_bf16(
            GemmRequest(M=504, N=216, K=23624, arch="gfx950", dtype="bf16")
        )
        self.assertEqual(r.candidate.name, "universal_gemm_bf16_cdna_pad")
        self.assertGreater(r.spec.trait.split_k, 1)
        # under half a wave of 128x128 tiles: a smaller ring tile
        req = GemmRequest(M=968, N=816, K=1880, arch="gfx950", dtype="bf16")
        r = dispatch_gemm_bf16(req)
        self.assertEqual(r.candidate.name, "universal_gemm_bf16_cdna_dtl_ring_any")
        self.assertLess(r.spec.tile.tile_m * r.spec.tile.tile_n, 128 * 128)
        self.assertIsNone(launch_shape_error(r.spec, 968, 816, 1880))
        with mock.patch.object(_gc, "G1_RING_ANY_TILES", ()):
            r0 = dispatch_gemm_bf16(req)
            self.assertEqual((r0.spec.tile.tile_m, r0.spec.tile.tile_n), (128, 128))
        with mock.patch.object(_gc, "G1_RING_ANY_SPLIT_RULE", ()):
            r1 = dispatch_gemm_bf16(
                GemmRequest(M=504, N=216, K=23624, arch="gfx950", dtype="bf16")
            )
            self.assertEqual(r1.candidate.name, "universal_gemm_bf16_cdna_dtl_ring_any")


class TestAdaptModes(unittest.TestCase):
    def test_k_tail_mode_per_body(self):
        req = GemmRequest(M=4096, N=4096, K=4104, arch="gfx950", dtype="bf16")
        s = _spec(_RT, replace(_RING, pad_k=False))
        self.assertEqual(_gc.g1_adapt_spec(req, s).trait.dtl_k_tail, 2)
        ig = replace(s, trait=replace(s.trait, iglp_opt=1))
        # the K-tail form of an iglp body drops iglp_opt (compile
        # cliff) and takes the peeled tail; the knob restores mode 1
        a = _gc.g1_adapt_spec(req, ig)
        self.assertEqual((a.trait.dtl_k_tail, a.trait.iglp_opt), (2, None))
        with mock.patch.object(_gc, "G1_K_TAIL_DROP_IGLP", False):
            self.assertEqual(_gc.g1_adapt_spec(req, ig).trait.dtl_k_tail, 1)
        # aligned K: untouched
        req2 = GemmRequest(M=4096, N=4096, K=4096, arch="gfx950", dtype="bf16")
        self.assertIs(_gc.g1_adapt_spec(req2, s), s)


class TestSplitKBinding(unittest.TestCase):
    @_g1_only()
    def test_workspace_is_f32(self):
        from rocke.dispatch.gemm.binding import gemm_rcr_binding

        try:
            import numpy  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("numpy")
        r = dispatch_gemm_bf16(
            GemmRequest(M=5, N=4736, K=2624, arch="gfx950", dtype="bf16")
        )
        self.assertGreater(r.spec.trait.split_k, 1)
        sizes = []

        class _RT:
            def alloc(self, n):
                sizes.append(n)
                return 4096 * len(sizes)

            def memcpy_h2d(self, *a):
                pass

            def memset(self, *a):
                pass

        b = gemm_rcr_binding(r, False, dtype="bf16")
        b.make_args(_RT())
        self.assertEqual(sizes[2], 5 * 4736 * 4)  # f32 Cf32 workspace


if __name__ == "__main__":
    unittest.main()
