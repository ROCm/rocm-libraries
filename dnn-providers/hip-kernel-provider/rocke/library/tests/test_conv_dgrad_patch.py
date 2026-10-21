# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side gate for the patch-embedding dgrad (``DgradConvSpec.patch``) and
the width fold the grouped-convolution dispatcher runs dgrad problems through
(``fold_width``).

No GPU and no comgr: the fold's arithmetic (checked against torch on the CPU
when torch is installed), the patch record, the validity gates, the lowered
IR, the cache identity, the AOT job grid and admission. The two-engine
counterpart is configs 34-35 of the dgrad parity emitters.
"""

from __future__ import annotations

import importlib.util
import re
import unittest
from dataclasses import replace

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _problem(**kw):
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    base = {
        "N": 2,
        "Hi": 32,
        "Wi": 32,
        "C": 3,
        "K": 64,
        "Y": 4,
        "X": 4,
        "sH": 4,
        "sW": 4,
    }
    base.update(kw)
    return ConvProblem(**base)


def _spec(**kw):
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec
    from kernels.common.conv_implicit_gemm_dgrad import DgradConvSpec

    base = {
        "problem": _problem(),
        "data": ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
        "tile_m": 128,
        "tile_n": 64,
        "tile_k": 64,
        "warp_m": 2,
        "warp_n": 2,
        "warp_tile_m": 16,
        "warp_tile_n": 16,
        "warp_tile_k": 32,
        "pipeline": "mem",
        "epilogue": "cshuffle",
        "vector_size_a": 8,
        "vector_size_b": 4,
        "vector_size_c": 4,
        "lds_k_outer": True,
        "uniform_k_decode": True,
    }
    base.update(kw)
    return DgradConvSpec(**base)


def _lower(spec, arch="gfx950") -> str:
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    from kernels.common.conv_implicit_gemm_dgrad import build_implicit_gemm_conv_dgrad

    return lower_kernel_to_llvm(
        build_implicit_gemm_conv_dgrad(spec, arch=arch), arch=arch
    )


def _sdivs(ll: str) -> int:
    # Divisions by a runtime value; a constant divisor folds to shifts.
    return len(re.findall(r"= sdiv i32 %[\w.]+, %", ll))


class TestFoldWidth(unittest.TestCase):
    def test_fold_factor(self):
        from kernels.common.conv_implicit_gemm_dgrad import width_fold

        self.assertEqual(width_fold(_problem(Wi=224, X=16, sW=16)), 16)
        # 384 is not a multiple of 14: only the common factor 2 folds.
        self.assertEqual(width_fold(_problem(Wi=384, X=14, sW=14)), 2)
        self.assertEqual(width_fold(_problem(Wi=30, X=4, sW=2, pW=2)), 2)
        # Stride 1, a dilated width or groups: nothing to fold.
        self.assertEqual(width_fold(_problem(X=3, sW=1, pW=1)), 1)
        self.assertEqual(width_fold(_problem(dW=2)), 1)
        self.assertEqual(width_fold(_problem(C=4, K=64, groups=2)), 1)

    def test_folded_problem_keeps_the_tensors(self):
        from kernels.common.conv_implicit_gemm_dgrad import fold_width

        p = _problem(N=3, Hi=37, Wi=48, C=5, K=24, Y=2, X=4, sH=3, sW=4)
        q = fold_width(p)
        self.assertEqual((q.C, q.X, q.sW, q.Wi, q.pW), (20, 1, 1, 12, 0))
        # Same dY, and the same element counts of W and dX.
        self.assertEqual((q.Ho, q.Wo, q.K), (p.Ho, p.Wo, p.K))
        self.assertEqual(q.Wi * q.C, p.Wi * p.C)
        self.assertEqual(q.Y * q.X * q.C, p.Y * p.X * p.C)
        # Nothing to fold: the problem itself.
        unit = _problem(X=3, sW=1, pW=1)
        self.assertIs(fold_width(unit), unit)

    def test_rejects_a_factor_that_does_not_divide(self):
        from kernels.common.conv_implicit_gemm_dgrad import fold_width

        with self.assertRaises(ValueError):
            fold_width(_problem(Wi=30, X=4, sW=4), 4)

    @unittest.skipUnless(_HAS_TORCH, "torch is optional")
    def test_folded_dgrad_is_the_same_numbers(self):
        import torch

        from kernels.common.conv_implicit_gemm_dgrad import fold_width

        def dgrad(p, dy, w):
            # NHWC dY / KYXC W -> NHWC dX, through torch's NCHW reference.
            dx = torch.nn.grad.conv2d_input(
                (p.N, p.C, p.Hi, p.Wi),
                w.permute(0, 3, 1, 2),
                dy.permute(0, 3, 1, 2),
                stride=(p.sH, p.sW),
                padding=(p.pH, p.pW),
            )
            return dx.permute(0, 2, 3, 1).contiguous()

        torch.manual_seed(0)
        for kw in (
            {
                "N": 2,
                "Hi": 12,
                "Wi": 12,
                "C": 3,
                "K": 8,
                "Y": 4,
                "X": 4,
                "sH": 4,
                "sW": 4,
            },
            {
                "N": 1,
                "Hi": 29,
                "Wi": 30,
                "C": 2,
                "K": 5,
                "Y": 3,
                "X": 6,
                "sH": 3,
                "sW": 6,
            },
            {
                "N": 2,
                "Hi": 11,
                "Wi": 16,
                "C": 3,
                "K": 4,
                "Y": 3,
                "X": 4,
                "sH": 2,
                "sW": 2,
                "pH": 1,
                "pW": 2,
            },
        ):
            p = _problem(**kw)
            q = fold_width(p)
            self.assertNotEqual(q, p)
            dy = torch.randn(p.N, p.Ho, p.Wo, p.K, dtype=torch.float64)
            w = torch.randn(p.K, p.Y, p.X, p.C, dtype=torch.float64)
            want = dgrad(p, dy, w)
            got = dgrad(q, dy, w.reshape(q.K, q.Y, q.X, q.C))
            torch.testing.assert_close(got.reshape(want.shape), want)


class TestPatchRecord(unittest.TestCase):
    def test_is_patch_problem(self):
        from kernels.common.conv_implicit_gemm_dgrad import is_patch_problem

        self.assertTrue(is_patch_problem(_problem()))
        self.assertTrue(is_patch_problem(_problem(sH=5, sW=6)))
        self.assertFalse(is_patch_problem(_problem(sH=3)))
        self.assertFalse(is_patch_problem(_problem(pW=1)))
        self.assertFalse(is_patch_problem(_problem(C=4, groups=2)))

    def test_one_record_over_the_filter(self):
        from kernels.common.conv_implicit_gemm_dgrad import patch_sub_gemms

        p = _problem(N=2, Hi=34, Wi=35, C=3, K=72, Y=4, X=4, sH=4, sW=4)
        (sg,) = patch_sub_gemms(p, 128, 64, tile_k=64)
        self.assertEqual(sg.gemm_m, p.N * p.Ho * p.Wo)
        self.assertEqual(sg.gemm_n, p.Y * p.X * p.C)
        self.assertEqual(sg.gemm_k, p.K)
        self.assertEqual((sg.y_dot_slice, sg.x_dot_slice), (1, 1))
        self.assertEqual((sg.d_h_stride, sg.d_w_stride), (p.sH, p.sW))
        self.assertEqual((sg.d_h_offset, sg.d_w_offset), (0, 0))
        # The K tail (ydot = 1) must land outside the filter.
        self.assertEqual((sg.b_y_stride, sg.b_y_offset), (p.Y, 0))

    def test_refuses_a_problem_that_is_not_a_patch_embedding(self):
        from kernels.common.conv_implicit_gemm_dgrad import patch_sub_gemms

        with self.assertRaises(ValueError):
            patch_sub_gemms(_problem(sH=2), 128, 64)


class TestPatchSpec(unittest.TestCase):
    def test_default_is_off_and_untagged(self):
        spec = _spec()
        self.assertFalse(spec.patch)
        self.assertNotIn("patch", spec.kernel_name())
        self.assertTrue(replace(spec, patch=True).kernel_name().endswith("_patch"))

    def test_rejections(self):
        from kernels.common.conv_implicit_gemm_dgrad import is_valid_dgrad_spec

        ok, why = is_valid_dgrad_spec(_spec(patch=True))
        self.assertTrue(ok, why)
        for kw in (
            {"split_k": 2},
            {"problem": _problem(sH=2, sW=2)},
            {"problem": _problem(pH=1, pW=1, Hi=34, Wi=34)},
            {"problem": _problem(C=4, groups=2)},
        ):
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                ok, _ = is_valid_dgrad_spec(_spec(patch=True, **kw))
                self.assertFalse(ok)
                with self.assertRaises(ValueError):
                    _spec(patch=True, **kw).validate()

    def test_only_the_store_changes(self):
        off = _lower(_spec())
        on = _lower(_spec(patch=True))
        self.assertNotEqual(off, on)
        # The column split: one runtime division per store address.
        self.assertGreater(_sdivs(on), _sdivs(off))
        # The K loop is the same: same number of buffer loads.
        loads = r"call [^\n]*amdgcn\.raw\.(?:ptr\.)?buffer\.load"
        self.assertEqual(len(re.findall(loads, on)), len(re.findall(loads, off)))


class TestPatchIdentity(unittest.TestCase):
    LABEL = (
        "dgrad-implicit_gemm-bf16-2d-t128x256x64-w2x4-a16x16x32-v8x8x8-mem-"
        "cshuffle-kouter-ukd-patch-msg1024-wave64-gfx950-llvm22"
    )

    def test_label_round_trip_and_hash(self):
        from kernels.conv.kernel_identity import KernelIdentity

        ident = KernelIdentity.from_label(self.LABEL)
        self.assertTrue(ident.patch)
        self.assertFalse(ident.grouped)
        self.assertEqual(ident.label(), self.LABEL)
        off = replace(ident, patch=False)
        self.assertNotEqual(off.stable_hash(), ident.stable_hash())
        self.assertNotIn("patch", off.label())

    def test_off_keeps_the_historical_hash(self):
        import hashlib
        import json
        from dataclasses import asdict

        from kernels.conv.kernel_identity import KernelIdentity

        ident = replace(KernelIdentity.from_label(self.LABEL), patch=False)
        fields = asdict(ident)
        for name in ("ws_replicas", "prefetch_addr", "patch"):
            if not fields[name]:
                del fields[name]
        if fields["streamk"] == "off" and not fields["streamk_reduction"]:
            del fields["streamk"], fields["streamk_reduction"]
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        self.assertEqual(
            ident.stable_hash(), hashlib.sha1(payload.encode()).hexdigest()
        )

    def test_aot_job_builds_it(self):
        from kernels.conv.aot_jobs import job_for_identity
        from kernels.conv.kernel_identity import KernelIdentity

        job = job_for_identity(KernelIdentity.from_label(self.LABEL))
        self.assertTrue(job.spec_kwargs["patch"])
        self.assertTrue(job.caps["patch"])

    def test_admission(self):
        from kernels.conv.kernel_identity import (
            KernelIdentity,
            dgrad_sub_gemms,
            identity_supports_problem,
        )

        ident = KernelIdentity.from_label(self.LABEL)
        vit = _problem(N=8, Hi=224, Wi=224, C=3, K=768, Y=16, X=16, sH=16, sW=16)
        ok, why = identity_supports_problem(ident, vit, ident.llvm_flavor)
        self.assertTrue(ok, why)
        self.assertEqual(len(dgrad_sub_gemms(ident, vit)), 1)
        # Not a patch embedding.
        ok, _ = identity_supports_problem(
            ident, _problem(sH=2, sW=2), ident.llvm_flavor
        )
        self.assertFalse(ok)
        # X*C = 42: W's run (Y*X*C = 588) takes a width of 4, dX's only 2.
        vitl = _problem(N=8, Hi=224, Wi=224, C=3, K=1024, Y=14, X=14, sH=14, sW=14)
        ok, _ = identity_supports_problem(ident, vitl, ident.llvm_flavor)
        self.assertFalse(ok)
        four = replace(ident, vector_size_b=4, vector_size_c=4)
        ok, why = identity_supports_problem(four, vitl, ident.llvm_flavor)
        self.assertFalse(ok)
        self.assertIn("dX run", why)
        narrow = replace(ident, vector_size_b=2, vector_size_c=2)
        ok, why = identity_supports_problem(narrow, vitl, ident.llvm_flavor)
        self.assertTrue(ok, why)


class TestPatchAotGrid(unittest.TestCase):
    def test_patch_jobs_are_the_listed_geometries_at_every_width(self):
        from rocke.core.arch import ArchTarget

        from kernels.conv import aot_jobs as aj

        target = ArchTarget.from_gfx("gfx950")
        jobs = [
            j
            for j in aj._dgrad_jobs("gfx950", "bf16", 64, "mma", target, 1024)
            if j.identity.patch
        ]
        self.assertEqual(
            len(jobs), len(aj.CACHE_DGRAD_PATCH_GEOMETRIES) * len(aj._VECS)
        )
        for j in jobs:
            i = j.identity
            self.assertFalse(i.grouped)
            self.assertEqual(i.epilogue, "cshuffle")
            self.assertEqual(i.vector_size_b, i.vector_size_c)

    def test_count_jobs_counts_them(self):
        from rocke.core.arch import ArchTarget

        from kernels.conv import aot_jobs as aj

        target = ArchTarget.from_gfx("gfx950")
        counted = aj.count_jobs(dtype="bf16", target=target, directions=["dgrad"])
        generated = sum(
            1 for _ in aj._dgrad_jobs("gfx950", "bf16", 64, "mma", target, 1024)
        )
        self.assertEqual(counted, generated)

    def test_no_patch_jobs_on_wmma(self):
        # The patch store is MFMA-only; the wave32 grids match the same
        # geometries but must not yield jobs that cannot build.
        from rocke.core.arch import ArchTarget

        from kernels.conv import aot_jobs as aj

        for arch in ("gfx1201", "gfx1250"):
            with self.subTest(arch=arch):
                target = ArchTarget.from_gfx(arch)
                jobs = list(aj._dgrad_jobs(arch, "bf16", 32, "wmma", target, 1024))
                self.assertTrue(jobs)
                self.assertFalse([j for j in jobs if j.identity.patch])
                counted = aj.count_jobs(
                    dtype="bf16", target=target, directions=["dgrad"]
                )
                self.assertEqual(counted, len(jobs))

    def test_validation_rejects_patch_on_wmma(self):
        from kernels.common.conv_implicit_gemm_dgrad import is_valid_dgrad_spec

        ok, why = is_valid_dgrad_spec(
            _spec(
                patch=True,
                wave_size=32,
                warp_tile_k=32,
                lds_k_outer=False,
                vector_size_b=None,
                vector_size_c=None,
            ),
            "gfx1250",
        )
        self.assertFalse(ok)
        self.assertIn("patch", why)


class TestDispatchFold(unittest.TestCase):
    def _req(self, **kw):
        from dispatch.grouped_convolution import ConvGroupedRequest

        base = {
            "N": 8,
            "C": 3,
            "K": 768,
            "Hi": 224,
            "Wi": 224,
            "Y": 16,
            "X": 16,
            "stride_h": 16,
            "stride_w": 16,
            "arch": "gfx950",
            "dtype": "bf16",
            "direction": "dgrad",
        }
        base.update(kw)
        return ConvGroupedRequest(**base)

    def test_dgrad_runs_folded(self):
        from dispatch.grouped_convolution import _gemm_problem

        p = _gemm_problem(self._req())
        self.assertEqual((p.C, p.X, p.sW, p.Wi), (48, 1, 1, 14))
        # Forward is not folded.
        p = _gemm_problem(self._req(direction="fwd"))
        self.assertEqual((p.C, p.X, p.sW, p.Wi), (3, 16, 16, 224))


if __name__ == "__main__":
    unittest.main()
