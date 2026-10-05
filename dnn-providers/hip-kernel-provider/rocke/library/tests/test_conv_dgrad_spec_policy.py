# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only policy tests for the implicit-GEMM dgrad spec (no GPU needed).

Covers the stride-1 specialization gates (``static_sub_gemm`` /
``tap_outer_k``), their kernel-name tags, the IR shape each one emits, the
gfx950 rules derived from them (load batching, the waves_per_eu hint, the
XCD-contiguous tile order of grouped problems), and the validator rules that
replaced silently-ignored knobs:

  - async_dma / unroll_k / chiplet_swizzle are rejected (the dgrad builder has
    no such path);
  - an accumulator tile above 256 fp32 registers per lane is rejected (the
    regression: a 256x256 tile with 1x2 warps of the 16x16x32 atom spilled
    ~1000 VGPRs and returned wrong dX on a large dense bf16 3x3 problem);
  - compv4 is a schedule policy here and is no longer charged a second LDS
    buffer.
"""

from __future__ import annotations

import unittest

import kernels.common.conv_implicit_gemm_dgrad as dgrad_mod
from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_implicit_gemm import ConvDataSpec
from kernels.common.conv_implicit_gemm_dgrad import (
    DgradConvSpec,
    build_implicit_gemm_conv_dgrad,
    is_valid_dgrad_spec,
    xcd_contiguous_tile_order,
)
from rocke.helpers.grid import python_chiplet_transform_chunked

_ARCH = "gfx950"


def _problem(**kw):
    base = {
        "N": 2,
        "Hi": 14,
        "Wi": 14,
        "C": 64,
        "K": 64,
        "Y": 3,
        "X": 3,
        "pH": 1,
        "pW": 1,
    }
    base.update(kw)
    return ConvProblem(**base)


def _spec(problem=None, dtype="bf16", **kw):
    base = {
        "problem": problem if problem is not None else _problem(),
        "data": ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
        "tile_m": 64,
        "tile_n": 64,
        "tile_k": 64,
        "warp_m": 2,
        "warp_n": 2,
        "warp_tile_m": 32,
        "warp_tile_n": 32,
        "warp_tile_k": 16,
        "pipeline": "mem",
        "epilogue": "cshuffle",
        "lds_k_outer": True,
    }
    base.update(kw)
    return DgradConvSpec(**base)


def _op_names(kernel):
    names = []

    def walk(ops):
        for op in ops:
            names.append(op.name)
            for region in getattr(op, "regions", ()) or ():
                walk(region.ops)

    walk(kernel.body.ops if hasattr(kernel, "body") else kernel.ops)
    return names


class TestStride1Gates(unittest.TestCase):
    def test_default_stride1_folds_and_runs_tap_outer(self):
        spec = _spec()
        self.assertTrue(spec.folds_sub_gemm_record)
        self.assertTrue(spec.uses_tap_outer_k)

    def test_strided_keeps_runtime_record(self):
        spec = _spec(problem=_problem(sH=2, sW=2))
        self.assertGreater(len(spec.compute_sub_gemms()), 1)
        self.assertFalse(spec.folds_sub_gemm_record)
        self.assertFalse(spec.uses_tap_outer_k)

    def test_ungrouped_pointwise_keeps_runtime_record(self):
        # Its descriptors are already divide-free; folding would only make
        # the K-loop trip count a constant that LLVM then unrolls.
        pointwise = _problem(Y=1, X=1, pH=0, pW=0)
        spec = _spec(problem=pointwise)
        self.assertEqual(len(spec.compute_sub_gemms()), 1)
        self.assertFalse(spec.folds_sub_gemm_record)
        self.assertNotIn("dynrec", spec.kernel_name())
        # Same IR as the explicit opt-out apart from the kernel name.
        k_on = build_implicit_gemm_conv_dgrad(spec, arch=_ARCH)
        k_off = build_implicit_gemm_conv_dgrad(
            _spec(problem=pointwise, static_sub_gemm=False), arch=_ARCH
        )
        self.assertEqual(_op_names(k_on), _op_names(k_off))
        # Grouped pointwise has no such fast path and still folds.
        grouped = _problem(Y=1, X=1, pH=0, pW=0, C=128, K=128, groups=2)
        self.assertTrue(_spec(problem=grouped).folds_sub_gemm_record)

    def test_tap_outer_needs_tile_k_to_divide_kpg(self):
        # K=96 with tile_k=64: a K tile would straddle two taps.
        self.assertFalse(_spec(problem=_problem(K=96)).uses_tap_outer_k)
        self.assertTrue(_spec(problem=_problem(K=96), tile_k=32).uses_tap_outer_k)

    def test_tap_outer_excluded_cases(self):
        self.assertFalse(_spec(split_k=2).uses_tap_outer_k)
        pointwise = _problem(Y=1, X=1, pH=0, pW=0)
        self.assertFalse(_spec(problem=pointwise).uses_tap_outer_k)
        self.assertFalse(_spec(tap_outer_k=False).uses_tap_outer_k)
        self.assertFalse(_spec(static_sub_gemm=False).uses_tap_outer_k)
        # Grouped stride-1 qualifies (per-group kpg divides tile_k).
        grouped = _problem(C=128, K=256, groups=2)
        self.assertTrue(_spec(problem=grouped).uses_tap_outer_k)

    def test_kernel_name_tags_only_the_opt_outs(self):
        self.assertNotIn("dynrec", _spec().kernel_name())
        self.assertNotIn("flatk", _spec().kernel_name())
        self.assertIn("dynrec", _spec(static_sub_gemm=False).kernel_name())
        self.assertIn("flatk", _spec(tap_outer_k=False).kernel_name())

    def test_folded_record_emits_no_record_loads(self):
        def n_scalar_loads(spec):
            k = build_implicit_gemm_conv_dgrad(spec, arch=_ARCH)
            return sum(1 for n in _op_names(k) if n == "memref.global_load_typed")

        self.assertEqual(n_scalar_loads(_spec()), 0)
        # Runtime record: binary search probes + 21 record fields.
        self.assertGreater(n_scalar_loads(_spec(static_sub_gemm=False)), 21)

    def test_tap_outer_emits_nested_k_loop(self):
        def n_loops(spec):
            k = build_implicit_gemm_conv_dgrad(spec, arch=_ARCH)
            return sum(1 for n in _op_names(k) if n == "scf.for")

        self.assertEqual(n_loops(_spec(tap_outer_k=False)) + 1, n_loops(_spec()))


def _k_loop_staging(spec, arch=_ARCH):
    """Global-load / LDS-store op names, in order, of the K loop's body."""
    k = build_implicit_gemm_conv_dgrad(spec, arch=arch)
    keep = ("tile.buffer_load_vN", "tile.smem_store_vN")
    found = []

    def walk(ops):
        for op in ops:
            if op.name == "scf.for":
                body = [o.name for r in op.regions for o in r.ops]
                if "tile.buffer_load_vN" in body:
                    found.append([n for n in body if n in keep])
            for region in getattr(op, "regions", ()) or ():
                walk(region.ops)

    walk(k.body.ops if hasattr(k, "body") else k.ops)
    assert len(found) == 1, found
    return found[0]


def _batched(staging):
    """True when every global load precedes the first LDS store."""
    first_store = staging.index("tile.smem_store_vN")
    return "tile.buffer_load_vN" not in staging[first_store:]


# Grouped, cpg 36 / kpg 72 (kpg % 64 != 0): folded record, flat loop.
_FLAT_GROUPED = {"N": 4, "Hi": 7, "Wi": 7, "C": 72, "K": 144, "groups": 2}
# Dense C = 50 (cpg % 4 != 0), K = 72.
_FLAT_DENSE = {"N": 16, "C": 50, "K": 72}


class TestFlatFoldLoadBatching(unittest.TestCase):
    """The flat K loop of a folded record issues all of the tile's global
    loads before its first LDS store on gfx950 (as the tap-outer loop does).
    With per-vector load->store pairs the scheduler serialised each load
    behind its store, which made the folded flat loop slower than the
    runtime-record build on grouped problems with cpg % 8 != 0 and on dense
    problems with cpg % 4 != 0. Not under the waves_per_eu hint (its
    occupancy ceiling already batches the loads), not on other targets (not
    measured), and never with the runtime record."""

    def test_flat_folded_loop_batches_its_loads(self):
        for kw in (_FLAT_GROUPED, _FLAT_DENSE):
            for dtype in ("bf16", "fp16"):
                with self.subTest(dtype=dtype, **kw):
                    spec = _spec(problem=_problem(**kw), dtype=dtype)
                    self.assertTrue(spec.folds_sub_gemm_record)
                    self.assertFalse(spec.uses_tap_outer_k)
                    self.assertTrue(_batched(_k_loop_staging(spec)))

    def test_tap_outer_loop_batches_its_loads(self):
        spec = _spec(problem=_problem(C=72, K=128, groups=2))
        self.assertTrue(spec.uses_tap_outer_k)
        self.assertTrue(_batched(_k_loop_staging(spec)))

    def test_per_vector_pairs_elsewhere(self):
        grouped = _problem(**_FLAT_GROUPED)
        # Runtime record.
        self.assertFalse(
            _batched(_k_loop_staging(_spec(problem=grouped, static_sub_gemm=False)))
        )
        # Other wave64 target: not measured, keeps the per-vector pairs.
        gfx942 = _spec(problem=grouped, dtype="fp16", warp_tile_k=8, lds_k_outer=False)
        self.assertTrue(gfx942.folds_sub_gemm_record)
        self.assertFalse(_batched(_k_loop_staging(gfx942, "gfx942")))
        # Under the waves_per_eu hint (16-byte loads, dispatch warp tile).
        hinted = _spec(problem=_problem(C=144, K=144, groups=2, N=4, Hi=7, Wi=7))
        k = build_implicit_gemm_conv_dgrad(hinted, arch=_ARCH)
        self.assertEqual(k.attrs.get("waves_per_eu"), (2, 6))
        self.assertFalse(_batched(_k_loop_staging(hinted)))


# cpg 96 / kpg 160 (kpg % 64 == 32, partial N tile): flat loop.
_XCD_FLAT = {"N": 1, "Hi": 13, "Wi": 11, "C": 288, "K": 480, "groups": 3}
# cpg 80 / kpg 128: tap-outer loop.
_XCD_TAP = {"N": 2, "Hi": 9, "Wi": 7, "C": 400, "K": 640, "groups": 5}
# 1x1, cpg 48 (one N tile) / kpg 192: every dY element is read by one tile.
_XCD_POINTWISE_ONE_N = {"N": 4, "Hi": 8, "Wi": 8, "C": 192, "K": 768, "groups": 4}
_XCD_POINTWISE_ONE_N.update({"Y": 1, "X": 1, "pH": 0, "pW": 0})


class TestXcdContiguousTileOrder(unittest.TestCase):
    """Grouped problems with a folded record launch their (group, tile) space
    in XCD-contiguous order on gfx950 (see ``xcd_contiguous_tile_order``):
    launch order sends the tiles that share a group's operands to different
    XCDs, so every XCD refetches them into its own L2. Never for ungrouped
    problems, the runtime record, split-K or other targets, nor where every
    dY element is read by one tile (1x1 filter and a single N tile)."""

    def _ops(self, spec, arch=_ARCH):
        return _op_names(build_implicit_gemm_conv_dgrad(spec, arch=arch))

    def _ops_launch_order(self, spec, arch=_ARCH):
        saved = dgrad_mod._XCD_TILE_ORDER_ARCHES
        dgrad_mod._XCD_TILE_ORDER_ARCHES = {}
        try:
            return self._ops(spec, arch)
        finally:
            dgrad_mod._XCD_TILE_ORDER_ARCHES = saved

    def test_grouped_folded_record_gets_the_order(self):
        for kw in (_XCD_FLAT, _XCD_TAP, _FLAT_GROUPED):
            with self.subTest(**kw):
                spec = _spec(problem=_problem(**kw))
                self.assertTrue(spec.folds_sub_gemm_record)
                self.assertEqual(xcd_contiguous_tile_order(spec, _ARCH), 8)
                remapped = self._ops(spec)
                plain = self._ops_launch_order(spec)
                self.assertGreater(len(remapped), len(plain))
                # The remap selects between the swizzled id and the
                # launch-order remainder.
                self.assertGreater(
                    remapped.count("arith.select"), plain.count("arith.select")
                )

    def test_single_n_tile_needs_a_shared_dy_row(self):
        # One N tile and one tap: no dY row is shared between tiles.
        one_n = _problem(**_XCD_POINTWISE_ONE_N)
        # Same 1x1 problem with two N tiles (cpg 80): the N tiles share dY.
        two_n = _problem(**{**_XCD_POINTWISE_ONE_N, "C": 320})
        # Same single-N-tile problem with a 3x3 filter: neighbouring M tiles
        # share the dY halo rows.
        taps = _problem(**{**_XCD_POINTWISE_ONE_N, "Y": 3, "X": 3, "pH": 1, "pW": 1})
        # Strided 1x1 with one N tile: still one tap per dY element.
        strided = _problem(
            **{**_XCD_POINTWISE_ONE_N, "Hi": 16, "Wi": 16, "sH": 2, "sW": 2}
        )
        for label, problem, xcds in (
            ("1x1, one N tile", one_n, 0),
            ("1x1 strided, one N tile", strided, 0),
            ("1x1, two N tiles", two_n, 8),
            ("3x3, one N tile", taps, 8),
        ):
            with self.subTest(case=label):
                spec = _spec(problem=problem)
                self.assertTrue(spec.folds_sub_gemm_record)
                self.assertGreaterEqual(
                    spec.compute_sub_gemms()[-1].block_end * problem.groups, 8
                )
                self.assertEqual(xcd_contiguous_tile_order(spec, _ARCH), xcds)
                same = self._ops(spec) == self._ops_launch_order(spec)
                self.assertEqual(same, xcds == 0)

    def test_withheld_elsewhere(self):
        flat = _problem(**_XCD_FLAT)
        cases = {
            "ungrouped": _spec(problem=_problem(**_FLAT_DENSE)),
            "runtime record": _spec(problem=flat, static_sub_gemm=False),
            "split-K": _spec(problem=flat, split_k=2),
            "strided": _spec(problem=_problem(**_XCD_FLAT, sH=2, sW=2)),
            "fewer workgroups than XCDs": _spec(
                problem=_problem(N=1, Hi=4, Wi=4, C=32, K=96, groups=2)
            ),
        }
        for label, spec in cases.items():
            with self.subTest(case=label):
                self.assertEqual(xcd_contiguous_tile_order(spec, _ARCH), 0)
                self.assertEqual(self._ops(spec), self._ops_launch_order(spec))
        self.assertEqual(xcd_contiguous_tile_order(_spec(problem=flat), "gfx942"), 0)

    def test_order_is_a_permutation_with_contiguous_xcd_ranges(self):
        for num_wgs in (8, 18, 20, 40, 3200, 6656 + 5):
            chunk = num_wgs // 8
            logical = [
                python_chiplet_transform_chunked(
                    w, num_wgs=num_wgs, num_xcds=8, chunk_size=chunk
                )
                for w in range(num_wgs)
            ]
            self.assertEqual(sorted(logical), list(range(num_wgs)))
            for xcd in range(8):
                ids = sorted(logical[w] for w in range(xcd, chunk * 8, 8))
                self.assertEqual(ids, list(range(xcd * chunk, (xcd + 1) * chunk)))


class TestFlatFoldAccumulatorHint(unittest.TestCase):
    """``waves_per_eu`` floor on the flat folded K loop (see
    ``flat_fold_acc_waves_per_eu``): emitted only for the folded record with
    the flat loop, 16-byte dY and W loads, the mem pipeline, gfx950, one of
    the two dispatch warp tiles, and no explicit ``waves_per_eu``."""

    def _wpe(self, spec, arch=_ARCH):
        k = build_implicit_gemm_conv_dgrad(spec, arch=arch)
        return k.attrs.get("waves_per_eu")

    def test_flat_folded_loop_gets_the_floor(self):
        # kpg 48 is not a multiple of tile_k 64: folded record, flat loop.
        spec = _spec(problem=_problem(K=48))
        self.assertTrue(spec.folds_sub_gemm_record)
        self.assertFalse(spec.uses_tap_outer_k)
        self.assertEqual(self._wpe(spec), (2, 6))
        # tap_outer_k off forces the flat loop on a tap-aligned problem.
        self.assertEqual(self._wpe(_spec(tap_outer_k=False)), (2, 6))
        # Grouped, large filter, small grid (kpg 16 < tile_k).
        grouped = _problem(
            N=1, Hi=13, Wi=13, C=64, K=64, Y=7, X=7, pH=3, pW=3, groups=4
        )
        self.assertEqual(self._wpe(_spec(problem=grouped)), (2, 6))
        self.assertEqual(self._wpe(_spec(problem=grouped, dtype="fp16")), (2, 6))

    def test_withheld_outside_the_flat_folded_loop(self):
        self.assertIsNone(self._wpe(_spec()))  # tap-outer loop
        self.assertIsNone(
            self._wpe(_spec(problem=_problem(K=48), static_sub_gemm=False))
        )
        self.assertIsNone(self._wpe(_spec(problem=_problem(sH=2, sW=2))))  # strided
        pointwise = _problem(Y=1, X=1, pH=0, pW=0)
        self.assertIsNone(self._wpe(_spec(problem=pointwise)))
        self.assertIsNone(self._wpe(_spec(problem=_problem(K=48), pipeline="compv3")))

    def test_withheld_with_narrow_loads(self):
        # K and C multiples of 4 but not 8: 8-byte dY and W loads.
        narrow = _problem(C=100, K=68, Hi=13, Wi=11)
        spec = _spec(problem=narrow)
        self.assertTrue(spec.folds_sub_gemm_record)
        self.assertFalse(spec.uses_tap_outer_k)
        self.assertIsNone(self._wpe(spec))
        grouped = _problem(C=200, K=136, Hi=19, Wi=7, groups=2)
        self.assertIsNone(self._wpe(_spec(problem=grouped, dtype="fp16")))

    def test_withheld_above_128_accumulators_per_lane(self):
        # Above 128 fp32 accumulators per lane the flat K loop keeps the
        # runtime record (the folded flat loop spills where it does not) and
        # gets no hint (the two-wave floor caps VGPRs + AGPRs at 256 per lane,
        # so a 256-accumulator tile under it spills heavily).
        flat = _problem(C=128, K=48)
        big = (
            {"tile_m": 256, "tile_n": 128, "warp_m": 1},  # 32x32 atom, 1x2 waves
            {
                "tile_m": 256,
                "tile_n": 256,
                "warp_tile_m": 16,
                "warp_tile_n": 16,
                "warp_tile_k": 32,
            },  # 16x16 atom, 2x2 waves
        )
        for kw in big:
            with self.subTest(**kw):
                spec = _spec(problem=flat, **kw)
                self.assertTrue(is_valid_dgrad_spec(spec, _ARCH)[0])
                self.assertFalse(spec.folds_sub_gemm_record)
                self.assertFalse(spec.uses_tap_outer_k)
                self.assertIsNone(self._wpe(spec))
                # tap_outer_k off forces the flat loop: same exclusion.
                forced = _spec(problem=_problem(C=128, K=128), tap_outer_k=False, **kw)
                self.assertFalse(forced.folds_sub_gemm_record)
                self.assertIsNone(self._wpe(forced))
                # A tap-aligned problem keeps the fold with the tap-outer loop,
                # also when the N tile is wider than the input channels.
                for c in (128, 64):
                    tap = _spec(problem=_problem(C=c, K=128), **kw)
                    self.assertTrue(tap.folds_sub_gemm_record)
                    self.assertTrue(tap.uses_tap_outer_k)
                    self.assertIsNone(self._wpe(tap))
        # 128 accumulators per lane (256x128 tile, 2x2 waves) keeps the fold
        # but not the hint (see test_hint_only_on_the_dispatch_warp_tiles).
        edge = _spec(problem=flat, tile_m=256, tile_n=128)
        self.assertTrue(edge.folds_sub_gemm_record)
        self.assertFalse(edge.uses_tap_outer_k)
        self.assertIsNone(self._wpe(edge))

    def test_hint_only_on_the_dispatch_warp_tiles(self):
        # Outside the two dispatch warp tiles the hint speeds up some configs
        # and slows down (or spills) others, with no accumulator-count, atom,
        # warp-count or epilogue rule that separates them, so the folded flat
        # loop keeps its fold but gets no hint.
        flat = _problem(C=256, K=48)
        a16 = {"warp_tile_m": 16, "warp_tile_n": 16, "warp_tile_k": 32}
        for kw in (
            {"tile_m": 64, "tile_n": 128, "warp_m": 1, "warp_n": 1, **a16},
            {"tile_m": 256, "tile_n": 64, "warp_m": 2, "warp_n": 1, **a16},
            {"tile_m": 128, "tile_n": 128, "warp_m": 2, "warp_n": 1},  # 32x32
            # 64 accumulators per lane or fewer, not a dispatch tile:
            {"tile_m": 256, "tile_n": 64},  # 2x2 waves, 32x32
            {"tile_m": 64, "tile_n": 32, "warp_m": 1, "warp_n": 2, **a16},
            {"tile_m": 256, "tile_n": 32, "warp_m": 1, "warp_n": 2, **a16},
            {"tile_m": 128, "tile_n": 128, "warp_m": 4, "warp_n": 1},  # 32x32
            {"tile_m": 64, "tile_n": 64, "warp_m": 1, "warp_n": 1, **a16},
            {"tile_m": 128, "tile_n": 64, "warp_m": 1, "warp_n": 2},  # 32x32
            {"tile_m": 128, "tile_n": 128, "tile_k": 32, **a16},  # tile_k 32
        ):
            with self.subTest(**kw):
                spec = _spec(problem=flat, **kw)
                self.assertTrue(is_valid_dgrad_spec(spec, _ARCH)[0])
                self.assertTrue(spec.folds_sub_gemm_record)
                self.assertFalse(spec.uses_tap_outer_k)
                self.assertIsNone(self._wpe(spec))
        # Both dispatch tiles, either epilogue: 64x64x64 2x2 waves 32x32x16
        # and 128x128x64 2x2 waves 16x16x32.
        for kw in ({}, {"tile_m": 128, "tile_n": 128, **a16}):
            for epilogue in ("cshuffle", "default"):
                with self.subTest(epilogue=epilogue, **kw):
                    spec = _spec(problem=flat, epilogue=epilogue, **kw)
                    self.assertEqual(self._wpe(spec), (2, 6))

    def test_explicit_waves_per_eu_wins(self):
        self.assertEqual(self._wpe(_spec(problem=_problem(K=48), waves_per_eu=1)), 1)

    def test_withheld_on_other_arches(self):
        # gfx942 has AGPRs too but was not measured; its 32x32 fp16 atom is K=8.
        spec = _spec(
            problem=_problem(K=48), dtype="fp16", warp_tile_k=8, lds_k_outer=False
        )
        self.assertIsNone(self._wpe(spec, arch="gfx942"))
        gfx950 = _spec(problem=_problem(K=48), dtype="fp16", lds_k_outer=False)
        self.assertEqual(self._wpe(gfx950, arch="gfx950"), (2, 6))


class TestValidatorRules(unittest.TestCase):
    def test_ignored_knobs_are_rejected(self):
        for kw in ({"async_dma": True}, {"unroll_k": True}, {"chiplet_swizzle": True}):
            with self.subTest(**kw):
                ok, why = is_valid_dgrad_spec(_spec(lds_k_outer=False, **kw), _ARCH)
                self.assertFalse(ok)
                self.assertIn("does not implement", why)

    def test_spilling_accumulator_tile_rejected(self):
        # A 256x256 tile with 1x2 warps of the 16x16x32 atom: 512 fp32
        # accumulators per lane, the whole wave64 register file.
        bad = _spec(
            problem=_problem(N=4, Hi=16, Wi=16, C=512, K=512),
            tile_m=256,
            tile_n=256,
            tile_k=64,
            warp_m=1,
            warp_n=2,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
        )
        ok, why = is_valid_dgrad_spec(bad, _ARCH)
        self.assertFalse(ok)
        self.assertIn("accumulator", why)
        # 256 accumulators per lane (2x2 warps) is the largest accepted.
        ok, why = is_valid_dgrad_spec(
            _spec(
                problem=_problem(N=4, Hi=16, Wi=16, C=512, K=512),
                tile_m=256,
                tile_n=256,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
            ),
            _ARCH,
        )
        self.assertTrue(ok, why)

    def test_compv4_is_single_buffered(self):
        # A/B single-buffered ~137 KB fits the 160 KB cap; a doubled charge
        # would not.
        spec = _spec(
            problem=_problem(C=256, K=256),
            tile_m=256,
            tile_n=256,
            tile_k=128,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
            pipeline="compv4",
        )
        ok, why = is_valid_dgrad_spec(spec, _ARCH)
        self.assertTrue(ok, why)


if __name__ == "__main__":
    unittest.main(verbosity=2)
