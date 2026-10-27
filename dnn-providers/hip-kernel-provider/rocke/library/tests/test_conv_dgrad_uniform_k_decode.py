# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side gate for dgrad ``uniform_k_decode`` (per-tile reduction decode).

No GPU and no comgr: spec construction, the validity gates, the selection
policy, the lowered IR, the AOT job generator, the JIT sweep worker and the
cache identity. The numerical counterpart is ``TestConvDgradUniformKDecode`` in
``test_conv_dgrad_correctness.py``; the two-engine counterpart is configs 15-20
of the dgrad parity emitters.
"""

from __future__ import annotations

import re
import unittest
from dataclasses import replace


def _spec(**kw):
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
    from kernels.common.conv_implicit_gemm_dgrad import DgradConvSpec

    base = {
        "problem": ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1),
        "data": ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
        "tile_m": 64,
        "tile_n": 64,
        "tile_k": 64,
        "warp_m": 2,
        "warp_n": 2,
        "warp_tile_m": 32,
        "warp_tile_n": 32,
        "warp_tile_k": 16,
        "wave_size": 64,
        "pipeline": "mem",
        "epilogue": "cshuffle",
        "lds_k_outer": True,
    }
    base.update(kw)
    return DgradConvSpec(**base)


def _lower(spec, arch="gfx950") -> str:
    from kernels.common.conv_implicit_gemm_dgrad import build_implicit_gemm_conv_dgrad
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    return lower_kernel_to_llvm(
        build_implicit_gemm_conv_dgrad(spec, arch=arch), arch=arch
    )


def _buffer_loads(ll: str) -> int:
    return len(re.findall(r"call [^\n]*amdgcn\.raw\.(?:ptr\.)?buffer\.load", ll))


def _sdivs(ll: str) -> int:
    # Divisions by a runtime value; a constant divisor folds to shifts.
    return len(re.findall(r"= s(?:div|rem) i32 %[\w.]+, %", ll))


class TestDgradUniformKDecodeGate(unittest.TestCase):
    def test_default_is_off_and_untagged(self):
        spec = _spec()
        self.assertFalse(spec.uniform_k_decode)
        self.assertNotIn("ukd", spec.kernel_name())
        self.assertTrue(
            replace(spec, uniform_k_decode=True).kernel_name().endswith("_ukd")
        )

    def test_off_emits_the_plain_loop_only(self):
        ll = _lower(_spec())
        self.assertNotIn("uk_k", ll)
        self.assertNotIn("fb_acc", ll)

    def test_on_adds_the_decode_loop_and_keeps_the_plain_one(self):
        off = _lower(_spec())
        on = _lower(_spec(uniform_k_decode=True))
        # The decode loop carries (k_out, xdot, ydot) next to the accumulators.
        for name in ("uk_k", "uk_x", "uk_y"):
            self.assertRegex(on, rf"%{name}\w* = phi i32")
        # The plain loop is still there, as the fallback, renamed.
        self.assertRegex(on, r"%fb_acc_m0_n0\w* = phi")
        # Two loops' worth of reads: the decode loop issues the same chunks.
        self.assertEqual(_buffer_loads(on), 2 * _buffer_loads(off))

    def test_decode_loop_divides_only_ahead_of_the_loop(self):
        """The decode loop's own per-chunk addressing has no division.

        Every division the flag adds is uniform set-up (decoding k_lo, the
        kpg % tile_k gate). The k decode is gone from the loop, and the
        loop-invariant (n, ho, wo) row decode goes through the record's magic
        pairs, as in the plain loop.
        """
        off = _lower(_spec())
        on = _lower(_spec(uniform_k_decode=True))
        uk_body = on[on.index("%uk_k") :]
        uk_body = uk_body[: uk_body.index("fb_acc_m0_n0")]
        self.assertEqual(_sdivs(uk_body), 0)
        self.assertGreaterEqual(_sdivs(off), _sdivs(uk_body))

    def test_every_sync_pipeline_is_admitted(self):
        from kernels.common.conv_implicit_gemm_dgrad import is_valid_dgrad_spec

        for pipeline in ("mem", "compv3", "compv4", "basic"):
            for kouter in (False, True):
                with self.subTest(pipeline=pipeline, kouter=kouter):
                    spec = _spec(
                        pipeline=pipeline, lds_k_outer=kouter, uniform_k_decode=True
                    )
                    spec.validate()
                    ok, why = is_valid_dgrad_spec(spec, "gfx950")
                    self.assertTrue(ok, why)
                    self.assertIn("%uk_k", _lower(spec))

    def test_split_k_grouped_strided_pointwise_are_admitted(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem
        from kernels.common.conv_implicit_gemm_dgrad import is_valid_dgrad_spec

        problems = {
            "split_k": None,
            "grouped": ConvProblem(
                N=2, Hi=14, Wi=14, C=128, K=128, Y=3, X=3, pH=1, pW=1, groups=2
            ),
            "strided": ConvProblem(
                N=2, Hi=15, Wi=15, C=64, K=64, Y=4, X=4, sH=2, sW=2, pH=1, pW=1
            ),
            "dilated": ConvProblem(
                N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, dH=2, dW=2, pH=2, pW=2
            ),
            "pointwise": ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=1, X=1),
        }
        for label, problem in problems.items():
            with self.subTest(label):
                kw = {"uniform_k_decode": True}
                if problem is None:
                    kw.update({"split_k": 2, "epilogue": "default", "tile_k": 32})
                else:
                    kw["problem"] = problem
                spec = _spec(**kw)
                spec.validate()
                ok, why = is_valid_dgrad_spec(spec, "gfx950")
                self.assertTrue(ok, why)
                self.assertIn("%uk_k", _lower(spec))

    def test_wmma_mem_is_admitted(self):
        from kernels.common.conv_implicit_gemm_dgrad import is_valid_dgrad_spec

        spec = _spec(
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
            wave_size=32,
            epilogue="default",
            lds_k_outer=False,
            data=_spec().data.__class__(dtype_a="fp16", dtype_b="fp16", dtype_d="fp32"),
            uniform_k_decode=True,
        )
        ok, why = is_valid_dgrad_spec(spec, "gfx1201")
        self.assertTrue(ok, why)
        self.assertIn("%uk_k", _lower(spec, arch="gfx1201"))

    def test_rejects_the_wavelet_loop(self):
        from kernels.common.conv_implicit_gemm_dgrad import is_valid_dgrad_spec

        spec = _spec(
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
            wave_size=32,
            tile_k=32,
            pipeline="wavelet",
            epilogue="default",
            lds_k_outer=False,
            data=_spec().data.__class__(dtype_a="fp16", dtype_b="fp16", dtype_d="fp32"),
            uniform_k_decode=True,
        )
        with self.assertRaisesRegex(ValueError, "wavelet"):
            spec.validate()
        ok, why = is_valid_dgrad_spec(spec, "gfx1250")
        self.assertFalse(ok)
        self.assertIn("uniform_k_decode", why)
        # Without the flag the same loop is admitted, so the flag is what the
        # gate rejects.
        ok, why = is_valid_dgrad_spec(replace(spec, uniform_k_decode=False), "gfx1250")
        self.assertTrue(ok, why)


class TestDefaultUniformKDecode(unittest.TestCase):
    """The selection policy: a pure function of the binary's configuration."""

    def _pick(self, **kw):
        from kernels.common.conv_implicit_gemm_dgrad import DgradConvSpec

        args = {"arch": "gfx950", "wave_size": 64, "pipeline": "mem"}
        args.update(kw)
        return DgradConvSpec.default_uniform_k_decode(**args)

    def test_on_for_every_gfx950_sync_pipeline(self):
        for pipeline in ("mem", "compv3", "compv4", "basic"):
            with self.subTest(pipeline=pipeline):
                self.assertTrue(self._pick(pipeline=pipeline))

    def test_off_outside_the_measured_regime(self):
        self.assertFalse(self._pick(arch="gfx942"))
        self.assertFalse(self._pick(arch="gfx1250", wave_size=32))
        self.assertFalse(self._pick(arch="gfx1201", wave_size=32))
        self.assertFalse(self._pick(pipeline="wavelet"))

    def test_policy_agrees_with_validate(self):
        """Whatever the policy turns on must validate and build."""
        from kernels.common.conv_implicit_gemm_dgrad import is_valid_dgrad_spec

        for pipeline in ("mem", "compv3", "compv4", "basic"):
            spec = _spec(pipeline=pipeline)
            spec = replace(spec, uniform_k_decode=self._pick(pipeline=pipeline))
            self.assertTrue(spec.uniform_k_decode)
            spec.validate()
            self.assertTrue(is_valid_dgrad_spec(spec, "gfx950")[0])

    def test_aot_jobs_carry_it_in_spec_and_identity(self):
        from kernels.conv import aot_jobs as ks
        from rocke.core.arch import ArchTarget

        target = ArchTarget.from_gfx("gfx950")
        jobs = list(
            ks._dgrad_jobs(
                "gfx950",
                "bf16",
                64,
                "mma",
                target,
                64,
                geometries=[
                    (128, 256, 64, 2, 4, 16, "mem", "cshuffle", _atom16x16x32())
                ],
            )
        )
        self.assertTrue(jobs)
        for job in jobs:
            # The stride-1 tap-outer variant replaces the decode loop.
            tapk = job.spec_kwargs.get("tap_outer_k", False)
            self.assertEqual(job.spec_kwargs["uniform_k_decode"], not tapk)
            self.assertEqual(
                job.identity.uniform_k_decode, job.spec_kwargs["uniform_k_decode"]
            )
            self.assertIn("tapk" if tapk else "ukd", job.identity.label())

    def test_jit_sweep_worker_uses_the_policy(self):
        from benchmarks.common.benchmark_implicit_gemm_conv import _build_dgrad_one
        from kernels.common._conv_implicit_gemm_common import ConvProblem

        problem = ConvProblem(
            N=2, Hi=16, Wi=16, C=64, K=64, Y=2, X=2, sH=2, sW=2, groups=1
        )
        combo = (64, 64, 64, 2, 2, 32, "mem", "cshuffle", 1)
        out = _build_dgrad_one((combo, problem, "bf16", "gfx950", 8, 8, 8))
        self.assertIsNotNone(out)
        _combo, spec, _sk, kernel = out
        self.assertTrue(spec.uniform_k_decode)
        self.assertTrue(kernel.name.endswith("_ukd"))


def _atom16x16x32():
    from rocke.core.arch import ArchTarget

    return ArchTarget.from_gfx("gfx950").mma.select_largest_k(
        family="mma", a_dtype="bf16", b_dtype="bf16", c_dtype="fp32", m=16, n=16
    )


class TestKernelIdentityUniformKDecode(unittest.TestCase):
    def _ident(self, **kw):
        from benchmarks.common.kernel_cache import KernelIdentity

        base = {
            "arch": "gfx950",
            "direction": "dgrad",
            "algorithm": "implicit_gemm",
            "dtype_a": "bf16",
            "dtype_b": "bf16",
            "dtype_d": "bf16",
            "tile_m": 128,
            "tile_n": 256,
            "tile_k": 64,
            "warp_m": 2,
            "warp_n": 4,
            "warp_tile_m": 16,
            "warp_tile_n": 16,
            "warp_tile_k": 32,
            "pipeline": "mem",
            "epilogue": "cshuffle",
            "wave_size": 64,
            "vector_size_a": 8,
            "vector_size_b": 8,
            "vector_size_c": 8,
            "lds_k_outer": True,
            "grouped": True,
            "stride_h": 2,
            "stride_w": 2,
            "dilation_h": 1,
            "dilation_w": 1,
            "max_sub_gemms": 64,
            "llvm_flavor": "llvm22",
        }
        base.update(kw)
        return KernelIdentity(**base)

    def test_off_keeps_the_historical_hash(self):
        # Pinned from the identity before the field existed: existing cache
        # entries must keep resolving.
        self.assertEqual(
            self._ident().stable_hash(), "ce9cd6ba8e05628e8fd717165800e95c7587e3be"
        )

    def test_on_is_a_distinct_identity(self):
        on = self._ident(uniform_k_decode=True)
        self.assertNotEqual(on.stable_hash(), self._ident().stable_hash())
        self.assertIn("ukd", on.label())


if __name__ == "__main__":
    unittest.main(verbosity=2)
