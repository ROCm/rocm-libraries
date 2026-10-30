# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Split-K forward implicit-GEMM conv (``ImplicitGemmConvSpec.split_k``).

CPU: the validator gate, the ABI, that ``split_k = 1`` builds the same
kernel as before, and the auto (``-1``) degree policy and its resolver. GPU
(gfx1250): Stage 1 + the Stage 2 reduce against the
reference across degrees, tiles, pipelines, epilogues, groups, strides, a
partial last K tile and degrees past the K-tile count (empty slices).
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import unittest
from unittest import mock

from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_abi import conv_arg_names
from kernels.common.conv_implicit_gemm import (
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
    is_valid_spec,
)

_HAS_TORCH = importlib.util.find_spec("torch") is not None

_WMMA = dict(
    warp_tile_m=16,
    warp_tile_n=16,
    warp_tile_k=32,
    wave_size=32,
    vector_size_a=8,
    vector_size_b=8,
)
_PROBE = ConvProblem(N=1, Hi=16, Wi=16, C=128, K=128, Y=3, X=3, pH=1, pW=1, groups=2)
_BF16 = ConvDataSpec("bf16", "bf16", "bf16")


def _spec(problem=_PROBE, data=_BF16, **kw):
    cfg = dict(
        _WMMA,
        tile_m=16,
        tile_n=16,
        tile_k=64,
        warp_m=1,
        warp_n=1,
        pipeline="wavelet",
        epilogue="cshuffle",
        vector_size_c=8,
        num_load_waves=1,
    )
    cfg.update(kw)
    return ImplicitGemmConvSpec(problem=problem, data=data, **cfg)


class TestSplitKGate(unittest.TestCase):
    def test_wmma_split_k_is_valid(self):
        ok, why = is_valid_spec(_spec(split_k=2), "gfx1250")
        self.assertTrue(ok, why)

    def test_mfma_is_rejected(self):
        spec = _spec(
            split_k=2,
            wave_size=64,
            pipeline="mem",
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            tile_m=64,
            tile_n=64,
        )
        ok, why = is_valid_spec(spec, "gfx950")
        self.assertFalse(ok)
        self.assertIn("WMMA-only", why)

    def test_group_merge_is_rejected(self):
        dw = ConvProblem(N=1, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=64)
        ok, why = is_valid_spec(
            _spec(
                problem=dw,
                split_k=2,
                group_merge=4,
                vector_size_a=None,
                vector_size_b=None,
                vector_size_c=None,
            ),
            "gfx1250",
        )
        self.assertFalse(ok)

    def test_non_identity_acc_epilogue_is_rejected(self):
        # Stage 1 stores raw partials and the reduction only adds them: a bias
        # would be added once per slice, ReLU / clamp applied before the sum.
        from kernels.common.conv_implicit_gemm import ConvAccumulatorEpilogue

        for epi in (
            ConvAccumulatorEpilogue(bias=1.0),
            ConvAccumulatorEpilogue(scale=2.0),
            ConvAccumulatorEpilogue(relu=True),
            ConvAccumulatorEpilogue(clamp_min=-1.0, clamp_max=1.0),
        ):
            for kw in (
                dict(split_k=2),
                dict(split_k=2, split_k_cluster=4, pipeline="mem"),
            ):
                with self.subTest(epi=epi, **kw):
                    ok, why = is_valid_spec(_spec(acc_epilogue=epi, **kw), "gfx1250")
                    self.assertFalse(ok)
                    self.assertIn("identity acc_epilogue", why)
        # The unsplit kernel keeps its fused epilogue.
        ok, why = is_valid_spec(
            _spec(acc_epilogue=ConvAccumulatorEpilogue(relu=True)), "gfx1250"
        )
        self.assertTrue(ok, why)

    def test_nonpositive_split_k_is_rejected(self):
        for sk in (0, -1, -4):
            with self.subTest(split_k=sk):
                ok, why = is_valid_spec(_spec(split_k=sk), "gfx1250")
                self.assertFalse(ok)
                self.assertIn("split_k must be >= 1", why)

    def test_name_tags_the_variant(self):
        self.assertTrue(_spec(split_k=2).kernel_name().endswith("_splitk"))
        self.assertNotIn("splitk", _spec().kernel_name())

    def test_abi_appends_workspace_and_slice(self):
        base = [n for n, _ in conv_arg_names(direction="fwd")]
        split = [n for n, _ in conv_arg_names(direction="fwd", two_stage=True)]
        self.assertEqual(split[: len(base)], base)
        self.assertEqual(split[len(base) :], ["ws_ptr", "ws_bytes", "ks", "ks_count"])

    def test_kernel_params_follow_the_abi(self):
        kernel = build_implicit_gemm_conv(_spec(split_k=2), arch="gfx1250")
        names = [p.name for p in kernel.params]
        abi = [n for n, _ in conv_arg_names(direction="fwd", two_stage=True)]
        self.assertEqual(names, abi)


class TestClusterSplitKGate(unittest.TestCase):
    """``split_k_cluster``: the single-launch fold through a workgroup cluster."""

    def test_mem_cluster_is_valid_and_sets_the_cluster(self):
        spec = _spec(split_k=2, split_k_cluster=4, pipeline="mem")
        ok, why = is_valid_spec(spec, "gfx1250")
        self.assertTrue(ok, why)
        kernel = build_implicit_gemm_conv(spec, arch="gfx1250")
        self.assertEqual(kernel.cluster_dims, (1, 1, 4))
        self.assertTrue(spec.kernel_name().endswith("_splitk_skc4"))
        # The ABI is the split-K one: the degree still arrives as ks_count.
        abi = [n for n, _ in conv_arg_names(direction="fwd", two_stage=True)]
        self.assertEqual([p.name for p in kernel.params], abi)

    def test_rejections(self):
        for kw, why in (
            (dict(split_k=1, split_k_cluster=4, pipeline="mem"), "split_k > 1"),
            (dict(split_k=2, split_k_cluster=4), "mem pipeline"),
            (dict(split_k=2, split_k_cluster=16, pipeline="mem"), "2..15"),
        ):
            ok, reason = is_valid_spec(_spec(**kw), "gfx1250")
            self.assertFalse(ok, kw)
            self.assertIn(why, reason)

    def test_off_by_default(self):
        kernel = build_implicit_gemm_conv(_spec(split_k=2), arch="gfx1250")
        self.assertIsNone(kernel.cluster_dims)

    def test_target_without_cluster_launch_is_rejected(self):
        spec = _spec(split_k=2, split_k_cluster=4, pipeline="mem", warp_tile_k=16)
        ok, why = is_valid_spec(spec, "gfx1201")
        self.assertFalse(ok)
        self.assertIn("workgroup-cluster launch", why)
        # The two-stage split-K form needs no cluster.
        ok, why = is_valid_spec(_spec(split_k=2, warp_tile_k=16), "gfx1201")
        self.assertTrue(ok, why)

    def test_channels_not_a_multiple_of_4_are_rejected(self):
        # The fold reads 4-channel quads bounded by their first channel: with
        # kpg = 10 the quad at channel 8 would run past the row.
        for problem in (
            ConvProblem(N=1, Hi=7, Wi=7, C=256, K=10, Y=3, X=3, pH=1, pW=1),
            ConvProblem(N=1, Hi=7, Wi=7, C=128, K=20, Y=3, X=3, pH=1, pW=1, groups=2),
        ):
            with self.subTest(K=problem.K, groups=problem.groups):
                spec = _spec(
                    problem=problem,
                    split_k=2,
                    split_k_cluster=4,
                    pipeline="mem",
                    vector_size_c=2,
                )
                ok, why = is_valid_spec(spec, "gfx1250")
                self.assertFalse(ok)
                self.assertIn("multiples of 4", why)

    def test_launch_admission_checks_the_launched_problem(self):
        # The binary is shape-generic: a spec built on a good problem must
        # still refuse a launched one whose channels break the quad rule.
        from kernels.common.conv_implicit_gemm import is_valid_spec_for_problem
        from kernels.conv.kernel_identity import (
            KernelIdentity,
            identity_supports_problem,
        )

        spec = _spec(split_k=2, split_k_cluster=4, pipeline="mem", vector_size_c=2)
        good = ConvProblem(N=1, Hi=7, Wi=7, C=256, K=12, Y=3, X=3, pH=1, pW=1)
        odd = ConvProblem(N=1, Hi=7, Wi=7, C=256, K=10, Y=3, X=3, pH=1, pW=1)
        grouped_odd = ConvProblem(
            N=1, Hi=7, Wi=7, C=256, K=12, Y=3, X=3, pH=1, pW=1, groups=2
        )
        self.assertEqual(is_valid_spec_for_problem(spec, good, "gfx1250"), (True, "ok"))
        for problem in (odd, grouped_odd):
            ok, why = is_valid_spec_for_problem(spec, problem, "gfx1250")
            self.assertFalse(ok)
            self.assertIn("multiples of 4", why)

        ident = KernelIdentity.from_label(
            "fwd-implicit_gemm-bf16-2d-t16x16x64-w1x1-a16x16x32-v8x8x2-mem-"
            "cshuffle-grp-sk2-skc4-2stage-wave32-gfx1250-llvm23"
        )
        self.assertTrue(identity_supports_problem(ident, good, "llvm23")[0])
        for problem in (odd, grouped_odd):
            ok, why = identity_supports_problem(ident, problem, "llvm23")
            self.assertFalse(ok)
            self.assertIn("multiples of 4", why)


class TestClusterSplitKCache(unittest.TestCase):
    """The AOT cache carries one cluster binary per degree, named apart."""

    def test_identity_keeps_its_hash_at_zero(self):
        from dataclasses import asdict

        from kernels.conv.kernel_identity import KernelIdentity

        ident = KernelIdentity.from_label(
            "fwd-implicit_gemm-bf16-2d-t32x32x64-w2x1-a16x16x32-v8x8x8-mem-"
            "cshuffle-grp-sk2-2stage-wave32-gfx1250-llvm23"
        )
        self.assertEqual(ident.split_k_cluster, 0)
        # An identity written before the field existed hashes the same.
        old = asdict(ident)
        del old["split_k_cluster"]
        self.assertEqual(
            KernelIdentity.from_dict(old).stable_hash(), ident.stable_hash()
        )

    def test_only_the_two_stage_form_needs_stage2(self):
        # Both split-K forward forms take the workspace ABI (two_stage); only
        # the two-stage one is followed by the Stage 2 reduce launch.
        from kernels.conv.kernel_identity import KernelIdentity

        prefix = "fwd-implicit_gemm-bf16-2d-t32x32x64-w2x1-a16x16x32-v8x8x8-mem-"
        two_stage = KernelIdentity.from_label(
            prefix + "cshuffle-grp-sk2-2stage-wave32-gfx1250-llvm23"
        )
        cluster = KernelIdentity.from_label(
            prefix + "cshuffle-grp-sk2-skc4-2stage-wave32-gfx1250-llvm23"
        )
        unsplit = KernelIdentity.from_label(
            prefix + "cshuffle-grp-wave32-gfx1250-llvm23"
        )
        self.assertTrue(two_stage.two_stage and two_stage.needs_stage2)
        self.assertTrue(cluster.two_stage)
        self.assertFalse(cluster.needs_stage2)
        self.assertFalse(unsplit.needs_stage2)

    def test_grid_has_one_binary_per_degree(self):
        from rocke.core.arch import ArchTarget

        from benchmarks.common.kernel_sweep import enumerate_jobs
        from kernels.conv.aot_jobs import (
            CACHE_FWD_SPLIT_K_CLUSTER_DEGREES,
            CACHE_FWD_SPLIT_K_CLUSTER_TILES,
        )
        from kernels.conv.kernel_identity import KernelIdentity

        # Cluster jobs exist only at the flavor that lowers clusters, so the
        # grid is enumerated there, whatever the host's own flavor is.
        with mock.patch.dict(os.environ, {"ROCKE_LLVM_FLAVOR": "llvm23"}):
            jobs = [
                j
                for j in enumerate_jobs(
                    arch="gfx1250",
                    dtype="bf16",
                    target=ArchTarget.from_gfx("gfx1250"),
                    directions=["fwd"],
                )
                if j.identity.split_k_cluster
            ]
        self.assertEqual(
            len(jobs),
            len(CACHE_FWD_SPLIT_K_CLUSTER_DEGREES)
            * len(CACHE_FWD_SPLIT_K_CLUSTER_TILES),
        )
        self.assertEqual(len({j.identity.stable_hash() for j in jobs}), len(jobs))
        for j in jobs:
            label = j.identity.label()
            self.assertIn(f"skc{j.identity.split_k_cluster}", label)
            self.assertEqual(KernelIdentity.from_label(label), j.identity)

    def test_launch_gate(self):
        from benchmarks.common.benchmark_implicit_gemm_conv import _runtime_split_ks
        from kernels.conv.kernel_identity import KernelIdentity

        ident = KernelIdentity.from_label(
            "fwd-implicit_gemm-bf16-2d-t64x32x64-w2x1-a16x16x32-v8x8x8-mem-"
            "cshuffle-grp-sk2-skc4-2stage-wave32-gfx1250-llvm23"
        )
        small = ConvProblem(N=1, Hi=7, Wi=7, C=512, K=512, Y=3, X=3, pH=1, pW=1)
        many_tiles = ConvProblem(
            N=64, Hi=7, Wi=7, C=1024, K=1024, Y=3, X=3, pH=1, pW=1, groups=32
        )
        self.assertEqual(_runtime_split_ks(small, ident), (4,))
        self.assertEqual(_runtime_split_ks(small, ident, 8), ())
        self.assertEqual(_runtime_split_ks(many_tiles, ident), ())
        # The fold's quad rule holds at launch too.
        odd = ConvProblem(N=1, Hi=7, Wi=7, C=512, K=510, Y=3, X=3, pH=1, pW=1)
        self.assertEqual(_runtime_split_ks(odd, ident), ())


class TestSplitKDegrees(unittest.TestCase):
    """``--split-k`` / ``--split-k-degrees`` on a cached split-K forward binary."""

    _IDENT = (
        "fwd-implicit_gemm-bf16-2d-t16x16x64-w1x1-a16x16x32-v8x8x8-wavelet-"
        "cshuffle-grp-sk2-2stage-wave32-gfx1250-llvm23"
    )
    # 9*512 = 4608 = 72 K tiles of 64: every degree below fits. 4 x 4 output
    # tiles, so the policy splits as far as it goes (32).
    _SMALL = ConvProblem(N=1, Hi=7, Wi=7, C=512, K=64, Y=3, X=3, pH=1, pW=1)
    # 8*28*28/16 * 64/16 = 1568 output tiles of 16 x 16 fill the device unsplit.
    _FULL = ConvProblem(N=8, Hi=28, Wi=28, C=64, K=64, Y=3, X=3, pH=1, pW=1)

    def test_explicit_sweep_uses_the_given_degrees(self):
        from benchmarks.common.benchmark_implicit_gemm_conv import _runtime_split_ks
        from kernels.conv.kernel_identity import KernelIdentity

        ident = KernelIdentity.from_label(self._IDENT)
        small = self._SMALL
        # 0 sweeps the list plus the resolver's pick, as for wgrad.
        self.assertEqual(_runtime_split_ks(small, ident, 0, (3,)), (32, 3))
        self.assertEqual(_runtime_split_ks(small, ident, 0, (5, 3)), (32, 5, 3))
        self.assertEqual(_runtime_split_ks(small, ident, 6, (3,)), (6,))

    def test_auto_launches_the_resolved_degree(self):
        from benchmarks.common.benchmark_implicit_gemm_conv import _runtime_split_ks
        from kernels.common.conv_split_k import resolve_conv_split_k
        from kernels.conv.kernel_identity import KernelIdentity

        ident = KernelIdentity.from_label(self._IDENT)
        resolved = resolve_conv_split_k(
            "fwd",
            self._SMALL,
            arch="gfx1250",
            dtype="bf16",
            tile_m=16,
            tile_n=16,
            tile_k=64,
            block_size=32,
        )
        self.assertEqual(resolved.split_k, 32)
        # Whatever the list says: -1 is the resolver's pick only.
        self.assertEqual(_runtime_split_ks(self._SMALL, ident, -1, (3,)), (32,))

    def test_auto_skips_the_split_binary_where_splitting_does_not_pay(self):
        from benchmarks.common.benchmark_implicit_gemm_conv import _runtime_split_ks
        from kernels.conv.kernel_identity import KernelIdentity

        ident = KernelIdentity.from_label(self._IDENT)
        big = TestSplitKDegrees._FULL
        self.assertEqual(_runtime_split_ks(big, ident, -1), ())
        # An explicit degree is still the caller's.
        self.assertEqual(_runtime_split_ks(big, ident, 2), (2,))

    def test_the_override_variable_forces_the_auto_degree(self):
        from benchmarks.common.benchmark_implicit_gemm_conv import _runtime_split_ks
        from kernels.conv.kernel_identity import KernelIdentity

        ident = KernelIdentity.from_label(self._IDENT)
        with mock.patch.dict(os.environ, {"ROCKE_CONV_SPLIT_K": "4"}):
            self.assertEqual(_runtime_split_ks(self._SMALL, ident, -1), (4,))
        with mock.patch.dict(os.environ, {"ROCKE_CONV_SPLIT_K": "off"}):
            self.assertEqual(_runtime_split_ks(self._SMALL, ident, -1), ())


class TestSplitKPolicy(unittest.TestCase):
    """``select_split_k_fwd``: the auto forward degree from the GEMM, the tile
    and the arch."""

    def _pick(self, M, N, K, tile=(16, 16, 64), groups=1, arch="gfx1250"):
        from rocke.helpers.split_k import select_split_k_fwd

        tm, tn, tk = tile
        return select_split_k_fwd(
            M=M, N=N, K=K, tile_m=tm, tile_n=tn, tile_k=tk, groups=groups, arch=arch
        ).split_k

    def test_fills_toward_the_target_grid(self):
        from rocke.helpers.split_k import FWD_SPLIT_K_TARGET_CTAS_PER_CU

        target = 256 * FWD_SPLIT_K_TARGET_CTAS_PER_CU
        # Deep enough that only the grid limits the degree.
        for tiles in (1, 3, 64, 100, 255, 256, target // 2 - 1):
            with self.subTest(tiles=tiles):
                got = self._pick(16 * tiles, 16, 1 << 20)
                want = min(32, 1 << ((target // tiles).bit_length() - 1))
                self.assertEqual(got, want)
                self.assertLessEqual(got * tiles, target)

    def test_a_grid_that_fills_the_device_is_not_split(self):
        from rocke.helpers.split_k import FWD_SPLIT_K_TARGET_CTAS_PER_CU

        target = 256 * FWD_SPLIT_K_TARGET_CTAS_PER_CU
        self.assertEqual(self._pick(16 * (target // 2), 16, 1 << 20), 1)
        # Groups multiply the grid.
        self.assertEqual(self._pick(16 * 8, 16, 1 << 20, groups=target // 16), 1)
        self.assertGreater(self._pick(16 * 8, 16, 1 << 20, groups=2), 1)

    def test_shallow_reductions_are_not_split(self):
        from rocke.helpers.split_k import FWD_SPLIT_K_MIN_GEMM_K

        self.assertEqual(self._pick(16, 16, FWD_SPLIT_K_MIN_GEMM_K - 1), 1)
        self.assertGreater(self._pick(16, 16, FWD_SPLIT_K_MIN_GEMM_K), 1)

    def test_every_slice_keeps_two_k_tiles(self):
        for k_tiles in (8, 9, 15, 16, 33, 64, 65):
            with self.subTest(k_tiles=k_tiles):
                got = self._pick(16, 16, 64 * k_tiles)
                self.assertGreaterEqual(k_tiles // got, 2)
                self.assertEqual(got & (got - 1), 0)
        # The same reduction in narrower K tiles splits further.
        self.assertGreater(
            self._pick(16, 16, 640, (16, 16, 32)), self._pick(16, 16, 640)
        )

    def test_a_smaller_part_splits_less(self):
        from rocke.helpers.split_k import select_split_k_fwd

        kw = dict(M=16 * 64, N=16, K=1 << 20, tile_m=16, tile_n=16, tile_k=64)
        full = select_split_k_fwd(**kw).split_k
        self.assertEqual(select_split_k_fwd(**kw, num_cus=128).split_k, full // 2)

    def test_degrees_stay_on_the_ladder_and_under_the_cap(self):
        from rocke.helpers.split_k import FWD_SPLIT_K_MAX

        for M in (1, 49, 196, 784, 3136):
            for K in (512, 2304, 4608, 17280):
                with self.subTest(M=M, K=K):
                    got = self._pick(M, 256, K, (32, 32, 64))
                    self.assertEqual(got & (got - 1), 0)
                    self.assertLessEqual(got, FWD_SPLIT_K_MAX)


class TestSplitKResolver(unittest.TestCase):
    """``resolve_conv_split_k("fwd", ...)``: where the forward splits, the
    explicit and override paths, and the launch limits."""

    def _resolve(self, problem, arch="gfx1250", tile=(16, 16, 64), **kw):
        from kernels.common.conv_split_k import resolve_conv_split_k

        tm, tn, tk = tile
        return resolve_conv_split_k(
            "fwd",
            problem,
            arch=arch,
            dtype="bf16",
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            block_size=32,
            **kw,
        )

    def test_auto_follows_the_policy_on_gfx1250(self):
        from rocke.helpers.split_k import select_split_k_fwd

        p = TestSplitKDegrees._SMALL
        got = self._resolve(p)
        want = select_split_k_fwd(
            M=p.M, N=p.kpg, K=p.K_gemm, tile_m=16, tile_n=16, tile_k=64
        ).split_k
        self.assertEqual((got.split_k, got.two_stage), (want, True))
        self.assertGreater(want, 1)

    def test_no_split_where_the_kernel_is_not_built(self):
        p = TestSplitKDegrees._SMALL
        for arch in ("gfx950", "gfx942", "gfx1201"):
            with self.subTest(arch=arch):
                r = self._resolve(p, arch=arch)
                self.assertEqual((r.split_k, r.two_stage), (1, False))
                # Even when forced: the kernel cannot be built there.
                self.assertEqual(self._resolve(p, arch=arch, override=8).split_k, 1)
        pointwise = ConvProblem(N=1, Hi=7, Wi=7, C=4096, K=64, Y=1, X=1)
        self.assertEqual(self._resolve(pointwise).split_k, 1)
        three_d = ConvProblem(
            N=1,
            Hi=4,
            Wi=4,
            C=512,
            K=64,
            Y=3,
            X=3,
            pH=1,
            pW=1,
            Di=4,
            Z=3,
            sD=1,
            pD=1,
            dD=1,
        )
        self.assertEqual(self._resolve(three_d).split_k, 1)

    def test_explicit_degrees_are_kept(self):
        p = TestSplitKDegrees._SMALL
        for sk in (1, 2, 3, 64):
            with self.subTest(split_k=sk):
                r = self._resolve(p, split_k=sk)
                self.assertEqual((r.split_k, r.two_stage), (sk, sk > 1))

    def test_override_is_held_to_the_launch_limits(self):
        p = TestSplitKDegrees._SMALL
        self.assertEqual(self._resolve(p, override=1).split_k, 1)
        self.assertEqual(self._resolve(p, override=6).split_k, 6)
        # gridDim.z carries group * split_k + slice.
        grouped = ConvProblem(
            N=1, Hi=7, Wi=7, C=40000, K=40000, Y=3, X=3, pH=1, pW=1, groups=10000
        )
        self.assertEqual(self._resolve(grouped, override=8).split_k, 6)
        # The fp32 slabs are addressed through an i32 byte count.
        from kernels.common.conv_fwd_split_k_reduce import fwd_split_k_workspace_nbytes
        from kernels.common.conv_split_k import MAX_FWD_WS_BYTES

        wide = ConvProblem(N=16, Hi=128, Wi=128, C=64, K=256, Y=3, X=3, pH=1, pW=1)
        got = self._resolve(wide, override=64).split_k
        self.assertLess(got, 64)
        self.assertLessEqual(
            fwd_split_k_workspace_nbytes(wide.M, wide.K, got), MAX_FWD_WS_BYTES
        )
        self.assertGreater(
            fwd_split_k_workspace_nbytes(wide.M, wide.K, 2 * got), MAX_FWD_WS_BYTES
        )

    def test_a_split_k_binary_holder_gets_an_error_not_degree_1(self):
        big = TestSplitKDegrees._FULL
        self.assertEqual(self._resolve(big).split_k, 1)
        with self.assertRaisesRegex(ValueError, "does not serve this problem"):
            self._resolve(big, allow_unsplit=False)

    def test_the_cache_builds_where_the_resolver_splits(self):
        from kernels.common.conv_split_k import FWD_SPLIT_K_ARCHS
        from kernels.conv.aot_jobs import CACHE_FWD_SPLIT_K_ARCHS

        self.assertEqual(CACHE_FWD_SPLIT_K_ARCHS, FWD_SPLIT_K_ARCHS)


class TestSplitKAutoBuild(unittest.TestCase):
    """A spec with ``split_k=-1`` builds the binary its resolved degree
    selects, in both engines' lowering entry."""

    def test_auto_builds_the_split_binary_where_it_pays(self):
        # 128 output tiles over 9 K tiles: the policy splits.
        auto = build_implicit_gemm_conv(_spec(split_k=-1), arch="gfx1250")
        split = build_implicit_gemm_conv(_spec(split_k=2), arch="gfx1250")
        self.assertEqual(auto.name, split.name)
        self.assertIn("ks_count", [p.name for p in auto.params])

    def test_auto_builds_the_plain_kernel_where_it_does_not(self):
        big = TestSplitKDegrees._FULL
        auto = build_implicit_gemm_conv(_spec(problem=big, split_k=-1), arch="gfx1250")
        plain = build_implicit_gemm_conv(_spec(problem=big), arch="gfx1250")
        self.assertEqual(auto.name, plain.name)
        self.assertNotIn("ks_count", [p.name for p in auto.params])

    def test_auto_with_a_spec_the_split_cannot_take_builds_the_plain_kernel(self):
        # The problem would split (see above), but a fused accumulator
        # epilogue keeps the spec out of the split form: auto resolves to 1
        # and builds the plain kernel with its epilogue, not an error.
        from kernels.common.conv_implicit_gemm import (
            ConvAccumulatorEpilogue,
            resolve_fwd_split_k_spec,
        )

        for kw in (
            dict(acc_epilogue=ConvAccumulatorEpilogue(relu=True)),
            dict(acc_epilogue=ConvAccumulatorEpilogue(bias=1.0)),
            dict(acc_epilogue=ConvAccumulatorEpilogue(clamp_min=-1.0, clamp_max=1.0)),
            dict(acc_epilogue=ConvAccumulatorEpilogue(scale=2.0)),
        ):
            with self.subTest(**kw):
                spec = _spec(split_k=-1, **kw)
                self.assertEqual(resolve_fwd_split_k_spec(spec, "gfx1250").split_k, 1)
        epi = dict(acc_epilogue=ConvAccumulatorEpilogue(relu=True))
        auto = build_implicit_gemm_conv(_spec(split_k=-1, **epi), arch="gfx1250")
        plain = build_implicit_gemm_conv(_spec(**epi), arch="gfx1250")
        self.assertEqual(auto.name, plain.name)
        self.assertNotIn("ks_count", [p.name for p in auto.params])
        # The identity epilogue still splits.
        self.assertEqual(
            resolve_fwd_split_k_spec(_spec(split_k=-1), "gfx1250").split_k, 4
        )

    def test_the_backend_resolves_auto_for_the_cpp_engine(self):
        from rocke.core import backend

        seen = []

        def _fake_lower_family(family, spec, *a, **kw):
            seen.append(spec.split_k)

        with mock.patch.object(
            backend, "_lower_family", _fake_lower_family
        ), mock.patch.object(backend, "_import_engine", lambda: None):
            backend.lower_conv_implicit_gemm(_spec(split_k=-1), arch="gfx1250")
            backend.lower_conv_implicit_gemm(_spec(split_k=-1), arch="gfx950")
        self.assertEqual(seen, [4, 1])


class TestSplitKAotScope(unittest.TestCase):
    """The AOT grid builds split-K forward jobs only where they are supported:
    gfx1250, and the cluster ones only at the flavor that lowers clusters.
    ``count_jobs`` agrees with the walk in every case."""

    def _walk(self, arch, flavor):
        from rocke.core.arch import ArchTarget

        from kernels.conv import aot_jobs as aj

        target = ArchTarget.from_gfx(arch)
        jobs = list(
            aj._fwd_jobs(
                arch, "bf16", target.wave_size, "wmma", target, llvm_flavor=flavor
            )
        )
        counted = aj.count_jobs(
            dtype="bf16", target=target, directions=["fwd"], llvm_flavor=flavor
        )
        split = [j for j in jobs if j.identity.split_k > 1]
        cluster = [j for j in split if j.identity.split_k_cluster]
        return len(jobs), counted, len(split), len(cluster)

    def test_gfx1250_llvm23_has_both(self):
        walked, counted, split, cluster = self._walk("gfx1250", "llvm23")
        self.assertEqual(walked, counted)
        self.assertGreater(cluster, 0)
        self.assertGreater(split, cluster)

    def test_gfx1250_llvm22_has_no_cluster_jobs(self):
        walked, counted, split, cluster = self._walk("gfx1250", "llvm22")
        self.assertEqual(walked, counted)
        self.assertGreater(split, 0)
        self.assertEqual(cluster, 0)

    def test_other_wmma_targets_have_no_split_jobs(self):
        for arch in ("gfx1201", "gfx1151"):
            with self.subTest(arch=arch):
                walked, counted, split, _ = self._walk(arch, "llvm23")
                self.assertEqual(walked, counted)
                self.assertEqual(split, 0)


class TestSplitKSpecDict(unittest.TestCase):
    """The Python-to-C++ spec dict carries the split-K fields as they are."""

    def test_fields_are_forwarded(self):
        from rocke.core.backend import conv_implicit_gemm_spec_to_dict

        d = conv_implicit_gemm_spec_to_dict(
            _spec(split_k=2, split_k_cluster=4, pipeline="mem", num_load_waves=3)
        )
        self.assertEqual(
            (d["split_k"], d["split_k_cluster"], d["num_load_waves"]), (2, 4, 3)
        )

    def test_a_spec_without_them_is_rejected(self):
        # A spec object missing a field is malformed: it must not be read as
        # an unsplit, cluster-free kernel with four load waves.
        from rocke.core.backend import conv_implicit_gemm_spec_to_dict

        spec = _spec(split_k=2)
        for missing in ("split_k", "split_k_cluster", "num_load_waves"):

            class _Without:
                def __getattr__(self, name, _missing=missing):
                    if name == _missing:
                        raise AttributeError(name)
                    return getattr(spec, name)

            with self.subTest(missing=missing):
                with self.assertRaises(AttributeError):
                    conv_implicit_gemm_spec_to_dict(_Without())


def _engine_or_skip(case):
    try:
        import rocke_engine  # type: ignore
    except ImportError:
        case.skipTest("rocke_engine (the C++ engine extension) is not built")
    return rocke_engine


class TestEngineAgreement(unittest.TestCase):
    """The Python and C++ engines, reached through the backend, agree on the
    accumulator epilogue and on split-K admission."""

    def test_split_k_with_a_fused_epilogue_is_rejected_by_both(self):
        _engine_or_skip(self)
        from kernels.common.conv_implicit_gemm import ConvAccumulatorEpilogue
        from rocke.core.backend import lower_conv_implicit_gemm

        for epi in (
            ConvAccumulatorEpilogue(bias=1.0),
            ConvAccumulatorEpilogue(relu=True),
            ConvAccumulatorEpilogue(clamp_min=-1.0, clamp_max=1.0),
        ):
            spec = _spec(split_k=2, acc_epilogue=epi)
            for backend in ("python", "cpp"):
                with self.subTest(epi=epi, backend=backend):
                    with self.assertRaisesRegex(Exception, "identity acc_epilogue"):
                        lower_conv_implicit_gemm(spec, arch="gfx1250", backend=backend)

    def test_unsplit_fused_epilogue_lowers_identically(self):
        _engine_or_skip(self)
        from kernels.common.conv_implicit_gemm import ConvAccumulatorEpilogue
        from rocke.core.backend import lower_conv_implicit_gemm

        # fp16: the C++ conv implicit-GEMM builder stages A/B in LDS as f16
        # whatever the dtype (a separate, pre-existing mirror gap), so bf16
        # cannot be compared through this path yet.
        fp16 = ConvDataSpec("fp16", "fp16", "fp16")
        spec = _spec(
            data=fp16,
            acc_epilogue=ConvAccumulatorEpilogue(
                bias=0.5, scale=2.0, relu=True, clamp_max=6.0
            ),
        )
        py = lower_conv_implicit_gemm(spec, arch="gfx1250", backend="python")
        cpp = lower_conv_implicit_gemm(spec, arch="gfx1250", backend="cpp")
        self.assertEqual(py.llvm_text, cpp.llvm_text)
        plain = lower_conv_implicit_gemm(
            _spec(data=fp16), arch="gfx1250", backend="cpp"
        )
        self.assertNotEqual(cpp.llvm_text, plain.llvm_text)

    def test_reduce_rejects_an_explicit_none_dtype_in_both(self):
        engine = _engine_or_skip(self)
        from kernels.common.conv_fwd_split_k_reduce import (
            FwdSplitKReduceSpec,
            build_conv_fwd_split_k_reduce,
        )

        with self.assertRaisesRegex(ValueError, r"dtype_d must be fp16 or bf16"):
            build_conv_fwd_split_k_reduce(
                FwdSplitKReduceSpec(dtype_d=None), arch="gfx1250"
            )
        with self.assertRaisesRegex(
            Exception, r"dtype_d must be fp16 or bf16 \(got None\)"
        ):
            engine.conv_fwd_split_k_reduce_lower_llvm(
                dict(dtype_d=None, vec=4), arch="gfx1250"
            )
        # An absent key keeps the engine default (the dataclass default, bf16).
        ll = engine.conv_fwd_split_k_reduce_lower_llvm(dict(vec=4), arch="gfx1250")
        self.assertIn("bf16", ll)


class TestSplitKReduceEngine(unittest.TestCase):
    """The C++ Stage 2 builder fails cleanly on a name it cannot hold."""

    def test_overlong_name_is_an_error_not_a_stale_builder(self):
        try:
            import rocke_engine  # type: ignore
        except ImportError:
            self.skipTest("rocke_engine (the C++ engine extension) is not built")
        spec = dict(dtype_d="bf16", vec=4, name="x" * 300)
        with self.assertRaisesRegex(Exception, "kernel name does not fit"):
            rocke_engine.conv_fwd_split_k_reduce_lower_llvm(spec, arch="gfx1250")
        with self.assertRaisesRegex(Exception, "kernel name does not fit"):
            rocke_engine.conv_fwd_split_k_reduce_serialize_ir(spec, arch="gfx1250")
        # A name that fits still builds.
        ll = rocke_engine.conv_fwd_split_k_reduce_lower_llvm(
            dict(dtype_d="bf16", vec=4), arch="gfx1250"
        )
        self.assertIn("conv_fwd_splitk_reduce", ll)


# ---------------------------------------------------------------------------
# GPU numerics
# ---------------------------------------------------------------------------


def _gpu_skip() -> str:
    if not _HAS_TORCH:
        return "torch not importable"
    try:
        from rocke.runtime.hip_module import get_device_arch

        arch = get_device_arch()
    except Exception as e:  # pragma: no cover - no device
        return f"no GPU ({e})"
    return (
        "" if arch == "gfx1250" else f"split-K fwd is WMMA (gfx1250); device is {arch}"
    )


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _run(problem, spec_kw, degrees, dtype="bf16", unrolled=True, cluster=False):
    """Max relative error per degree of one split-K binary on ``problem``."""
    import torch

    from rocke import compile_kernel
    from builders.common.conv_reference import conv_reference_gfx1250
    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_fwd_split_k_reduce import (
        FwdSplitKReduceSpec,
        build_conv_fwd_split_k_reduce,
        fwd_split_k_reduce_grid,
        fwd_split_k_reduce_signature,
        fwd_split_k_reduce_vec,
        fwd_split_k_workspace_nbytes,
    )
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import (
        KernelLauncher,
        LaunchConfig,
        synchronize_and_release,
    )

    data = ConvDataSpec(dtype, dtype, dtype)

    def _stage1(degree):
        # The cluster binary bakes its degree (one per degree); the plain one
        # serves every degree.
        spec = _spec(
            problem=problem,
            data=data,
            split_k=2,
            split_k_cluster=degree if cluster else 0,
            **spec_kw,
        )
        ok, why = is_valid_spec(spec, "gfx1250")
        assert ok, why
        art = compile_kernel(
            build_implicit_gemm_conv(spec, arch="gfx1250"), arch="gfx1250"
        )
        return spec, KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=conv_args_signature(dtype, two_stage=True),
            cluster_dims=(1, 1, degree) if cluster else None,
        )

    if not cluster:
        spec, s1 = _stage1(2)
    elems = problem.M * problem.K

    def _stage2(degree):
        # The degree-specialised fold (what the benchmark launches), or the
        # runtime-loop one.
        r_spec = FwdSplitKReduceSpec(
            dtype_d=dtype,
            vec=fwd_split_k_reduce_vec(elems),
            degree=degree if unrolled else 0,
        )
        r_art = compile_kernel(build_conv_fwd_split_k_reduce(r_spec), arch="gfx1250")
        return r_spec, KernelLauncher(
            hsaco=r_art.hsaco,
            kernel_name=r_art.kernel_name,
            signature=fwd_split_k_reduce_signature(r_spec),
        )

    tdt = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    torch.manual_seed(0)
    A = (
        torch.empty(problem.N, problem.Hi, problem.Wi, problem.C)
        .uniform_(-1, 1)
        .to(tdt)
    )
    B = (
        torch.empty(problem.K, problem.Y, problem.X, problem.cpg)
        .uniform_(-1, 1)
        .to(tdt)
    )
    ref = conv_reference_gfx1250(A, B, problem, out_dtype=tdt).float().cpu()
    D = torch.empty(problem.N, problem.Ho, problem.Wo, problem.K, dtype=tdt)
    # Host <-> device copies go through pinned buffers only: a pageable copy
    # registers the host range with the driver, which must not overlap a
    # kernel in flight.
    A, B = A.pin_memory(), B.pin_memory()

    rt = Runtime()
    bufs = [rt.alloc(A.nbytes), rt.alloc(B.nbytes), rt.alloc(D.nbytes)]
    rt.memcpy_h2d(bufs[0], _u8(A), A.nbytes)
    rt.memcpy_h2d(bufs[1], _u8(B), B.nbytes)
    errs = {}
    try:
        for degree in degrees:
            if cluster:
                spec, s1 = _stage1(degree)
            r_spec, s2 = _stage2(degree)
            ws_bytes = fwd_split_k_workspace_nbytes(problem.M, problem.K, degree)
            ws = rt.alloc(ws_bytes)
            # Poison the workspace and the output: Stage 1 must write every slab
            # element, Stage 2 every output element.
            rt.memset(ws, 0x7F, ws_bytes)
            rt.memset(bufs[2], 0x7F, D.nbytes)
            args = ConvArgs.from_problem(
                problem, tile_m=spec.tile_m, tile_n=spec.tile_n, tile_k=spec.tile_k
            )
            values = args.to_launch_values(
                *[int(x) for x in bufs],
                A.nbytes,
                B.nbytes,
                D.nbytes,
                split_k=degree,
                ws_ptr=int(ws),
                ws_bytes=ws_bytes,
            )
            s1_cfg = LaunchConfig(
                grid=args.grid(degree),
                block=(spec.launch_block_size, 1, 1),
                fence=True,
            )
            if cluster:
                # One launch: the cluster folds the slabs into D itself. Run
                # it through bind() too, the path the benchmark times.
                s1(values, config=s1_cfg)
                out = torch.empty_like(D).pin_memory()
                rt.memcpy_d2h(_u8(out), bufs[2], D.nbytes)
                errs[degree] = float(
                    (out.float() - ref).abs().max() / ref.abs().max().clamp(min=1.0)
                )
                rt.memset(bufs[2], 0x7F, D.nbytes)
                s1.bind(values, config=s1_cfg)()
                synchronize_and_release()
                rt.memcpy_d2h(_u8(out), bufs[2], D.nbytes)
                rt.free(ws)
                diff = (out.float() - ref).abs().max()
                errs[degree] = max(
                    errs[degree], float(diff / ref.abs().max().clamp(min=1.0))
                )
                continue
            s1(values, config=s1_cfg)
            s2(
                dict(
                    ws_ptr=int(ws),
                    d_ptr=int(bufs[2]),
                    elems=elems,
                    ws_bytes=ws_bytes,
                    d_bytes=D.nbytes,
                    ks_count=degree,
                ),
                config=LaunchConfig(
                    grid=fwd_split_k_reduce_grid(r_spec, elems),
                    block=(r_spec.block_size, 1, 1),
                    fence=True,
                ),
            )
            out = torch.empty_like(D).pin_memory()
            rt.memcpy_d2h(_u8(out), bufs[2], D.nbytes)
            rt.free(ws)
            diff = (out.float() - ref).abs().max()
            errs[degree] = float(diff / ref.abs().max().clamp(min=1.0))
    finally:
        for x in bufs:
            rt.free(x)
    return errs


_GPU_SKIP = _gpu_skip()

# (problem, spec knobs, degrees)
_CASES = [
    # small M, deep K; 9*256 = 2304 = 36 K tiles of 64
    (
        "small_m_wavelet",
        ConvProblem(N=1, Hi=7, Wi=7, C=256, K=64, Y=3, X=3, pH=1, pW=1),
        {},
        (1, 2, 4, 16, 64),
    ),
    (
        "mem_cshuffle_64",
        ConvProblem(N=2, Hi=12, Wi=12, C=128, K=96, Y=3, X=3, pH=1, pW=1),
        dict(tile_m=64, tile_n=64, warp_m=2, warp_n=2, pipeline="mem"),
        (1, 3, 8),
    ),
    (
        "mem_default",
        ConvProblem(N=1, Hi=9, Wi=9, C=64, K=40, Y=3, X=3, pH=1, pW=1),
        dict(
            tile_m=32,
            tile_n=32,
            tile_k=32,
            warp_m=2,
            warp_n=1,
            pipeline="mem",
            epilogue="default",
            vector_size_c=1,
        ),
        (2, 5),
    ),
    # partial last K tile: 9*24 = 216 is not a multiple of 64
    (
        "k_tail_stride2",
        ConvProblem(N=2, Hi=15, Wi=13, C=24, K=32, Y=3, X=3, sH=2, sW=2, pH=1, pW=1),
        {},
        (2, 3, 4),
    ),
    (
        "grouped",
        ConvProblem(N=1, Hi=10, Wi=10, C=128, K=64, Y=3, X=3, pH=1, pW=1, groups=2),
        {},
        (1, 4),
    ),
    (
        "pointwise_shape_2x2s2",
        ConvProblem(N=1, Hi=14, Wi=14, C=256, K=128, Y=2, X=2, sH=2, sW=2),
        {},
        (2, 8),
    ),
]


@unittest.skipIf(_GPU_SKIP, _GPU_SKIP)
class TestSplitKNumerics(unittest.TestCase):
    def test_cases(self):
        for dtype in ("bf16", "fp16"):
            tol = 2e-2
            for name, problem, kw, degrees in _CASES:
                with self.subTest(case=name, dtype=dtype):
                    for unrolled in (True, False):
                        errs = _run(problem, kw, degrees, dtype, unrolled)
                        for degree, err in errs.items():
                            self.assertLess(
                                err,
                                tol,
                                f"{name} {dtype} degree {degree} "
                                f"unrolled={unrolled}: {err}",
                            )


@unittest.skipIf(_GPU_SKIP, _GPU_SKIP)
class TestSplitKAutoNumerics(unittest.TestCase):
    """The binary an auto (-1) spec builds, launched at the resolved degree."""

    def test_small_m_at_the_resolved_degree(self):
        from kernels.common.conv_split_k import resolve_conv_split_k

        problem = _CASES[0][1]
        spec = _spec(problem=problem, split_k=-1)
        degree = resolve_conv_split_k(
            "fwd",
            problem,
            arch="gfx1250",
            dtype="bf16",
            tile_m=spec.tile_m,
            tile_n=spec.tile_n,
            tile_k=spec.tile_k,
            block_size=spec.block_size,
        ).split_k
        self.assertGreater(degree, 1)
        auto = build_implicit_gemm_conv(spec, arch="gfx1250")
        split = build_implicit_gemm_conv(
            _spec(problem=problem, split_k=2), arch="gfx1250"
        )
        self.assertEqual(auto.name, split.name)
        err = _run(problem, {}, (degree,))[degree]
        self.assertLess(err, 2e-2, f"degree {degree}: {err}")


# (problem, spec knobs, degrees) for the single-launch cluster fold (mem only).
_CLUSTER_CASES = [
    (
        "small_m",
        ConvProblem(N=1, Hi=7, Wi=7, C=256, K=64, Y=3, X=3, pH=1, pW=1),
        dict(pipeline="mem"),
        (2, 4, 15),
    ),
    (
        "mem_cshuffle_64",
        ConvProblem(N=2, Hi=12, Wi=12, C=128, K=96, Y=3, X=3, pH=1, pW=1),
        dict(tile_m=64, tile_n=64, warp_m=2, warp_n=2, pipeline="mem"),
        (3, 8),
    ),
    (
        "k_tail_stride2",
        ConvProblem(N=2, Hi=15, Wi=13, C=24, K=32, Y=3, X=3, sH=2, sW=2, pH=1, pW=1),
        dict(tile_m=32, tile_n=32, warp_m=2, warp_n=1, pipeline="mem"),
        (2, 4),
    ),
    (
        "grouped",
        ConvProblem(N=1, Hi=10, Wi=10, C=128, K=64, Y=3, X=3, pH=1, pW=1, groups=2),
        dict(tile_m=32, tile_n=32, warp_m=1, warp_n=2, pipeline="mem"),
        (2, 4),
    ),
]


@unittest.skipIf(_GPU_SKIP, _GPU_SKIP)
class TestClusterSplitKNumerics(unittest.TestCase):
    def test_cases(self):
        for dtype in ("bf16", "fp16"):
            for name, problem, kw, degrees in _CLUSTER_CASES:
                with self.subTest(case=name, dtype=dtype):
                    errs = _run(problem, kw, degrees, dtype, cluster=True)
                    for degree, err in errs.items():
                        self.assertLess(err, 2e-2, f"{name} {dtype} S={degree}: {err}")


if __name__ == "__main__":
    unittest.main()
