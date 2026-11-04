# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side gate for wgrad ``prefetch_addr`` (carried K-loop byte offsets).

No GPU and no comgr: spec construction, the validity gates, the selection
policy, the lowered IR and the loader split. The numerical counterpart is
``TestWgradPrefetchAddrNumerics`` in ``test_conv_wgrad_correctness.py``; the
two-engine counterpart is configs 21-37 of the wgrad parity emitters.
"""

from __future__ import annotations

import re
import unittest
from dataclasses import replace


def _spec(**kw):
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
    from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

    base = dict(
        problem=ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1),
        data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
        wave_size=64,
        pipeline="mem",
        epilogue="cshuffle",
        lds_k_outer=True,
    )
    base.update(kw)
    return WgradConvSpec(**base)


def _lower(spec, arch="gfx950") -> str:
    from kernels.common.conv_implicit_gemm_wgrad import build_implicit_gemm_conv_wgrad
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    return lower_kernel_to_llvm(
        build_implicit_gemm_conv_wgrad(spec, arch=arch), arch=arch
    )


def _buffer_loads(ll: str) -> int:
    return len(re.findall(r"call [^\n]*amdgcn\.raw\.(?:ptr\.)?buffer\.load", ll))


class TestWgradPrefetchAddrGate(unittest.TestCase):
    def test_default_is_off_and_untagged(self):
        spec = _spec()
        self.assertFalse(spec.prefetch_addr)
        self.assertNotIn("pfa", spec.kernel_name())
        self.assertTrue(
            replace(spec, prefetch_addr=True).kernel_name().endswith("_pfa")
        )

    def test_offsets_are_loop_carried(self):
        off = _lower(_spec())
        on = _lower(_spec(prefetch_addr=True))
        self.assertNotIn("pfa_off", off)
        # One carried offset per A and B chunk: 64x64 K-outer tiles at vec 8
        # over 256 threads are two chunks per thread per operand.
        self.assertEqual(len(re.findall(r"%pfa_off\d+ = phi", on)), 4)
        # The loads are regrouped, not duplicated.
        self.assertEqual(_buffer_loads(on), _buffer_loads(off))

    def test_every_sync_pipeline_is_admitted(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        for pipeline in ("mem", "compv3", "compv4", "basic"):
            for kouter in (False, True):
                with self.subTest(pipeline=pipeline, kouter=kouter):
                    spec = _spec(
                        pipeline=pipeline, lds_k_outer=kouter, prefetch_addr=True
                    )
                    spec.validate()
                    ok, why = is_valid_wgrad_spec(spec, "gfx950")
                    self.assertTrue(ok, why)
                    self.assertIn("pfa_off0", _lower(spec))

    def test_split_k_and_two_stage_are_admitted(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        for kw in (
            dict(split_k=2),
            dict(split_k=2, two_stage=True, epilogue="default"),
        ):
            with self.subTest(**kw):
                spec = _spec(prefetch_addr=True, **kw)
                ok, why = is_valid_wgrad_spec(spec, "gfx950")
                self.assertTrue(ok, why)
                self.assertIn("pfa_off0", _lower(spec))

    def test_rejects_loops_it_does_not_rewrite(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        for kw, word in (
            (dict(async_dma=True, epilogue="default", vector_size_a=None), "async_dma"),
            (dict(unroll_k=True), "unroll_k"),
        ):
            with self.subTest(word=word):
                spec = _spec(prefetch_addr=True, **kw)
                with self.assertRaisesRegex(ValueError, word):
                    spec.validate()
                ok, why = is_valid_wgrad_spec(spec, "gfx950")
                self.assertFalse(ok)
                self.assertIn("prefetch_addr", why)
                # Without the flag the same loop is admitted, so the flag is
                # what the gate rejects.
                ok, why = is_valid_wgrad_spec(
                    replace(spec, prefetch_addr=False), "gfx950"
                )
                self.assertTrue(ok, why)

    def test_rejected_on_wmma(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        spec = _spec(
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
            wave_size=32,
            tile_m=32,
            tile_n=32,
            tile_k=32,
            warp_m=1,
            warp_n=1,
            epilogue="default",
            lds_k_outer=False,
            data=_spec().data.__class__(dtype_a="fp16", dtype_b="fp16", dtype_d="fp32"),
        )
        ok, why = is_valid_wgrad_spec(spec, "gfx1151")
        self.assertTrue(ok, why)
        ok, why = is_valid_wgrad_spec(replace(spec, prefetch_addr=True), "gfx1151")
        self.assertFalse(ok)
        self.assertIn("prefetch_addr", why)


class TestDefaultPrefetchAddr(unittest.TestCase):
    """The selection policy: a pure function of the binary's configuration."""

    def _pick(self, tile_m, tile_n, tile_k, **kw):
        from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

        args = dict(arch="gfx950", lds_k_outer=True, wave_size=64)
        args.update(kw)
        return WgradConvSpec.default_prefetch_addr(
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, **args
        )

    def test_footprint_split(self):
        for tile, want in (
            ((64, 64, 64), True),
            ((128, 64, 64), True),
            ((128, 128, 64), True),
            ((256, 64, 64), True),
            ((128, 256, 32), True),
            ((256, 128, 32), True),
            ((16, 256, 64), True),
            ((256, 128, 64), False),
            ((128, 256, 64), False),
        ):
            with self.subTest(tile=tile):
                self.assertEqual(self._pick(*tile), want)

    def test_off_outside_the_measured_regime(self):
        self.assertFalse(self._pick(64, 64, 64, arch="gfx942"))
        self.assertFalse(self._pick(64, 64, 64, arch="gfx1250", wave_size=32))
        self.assertFalse(self._pick(64, 64, 64, lds_k_outer=False))
        self.assertFalse(self._pick(64, 64, 64, async_dma=True))
        self.assertFalse(self._pick(64, 64, 64, unroll_k=True))

    def test_policy_agrees_with_validate(self):
        """Whatever the policy turns on must validate and build."""
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        for pipeline in ("mem", "compv3", "compv4", "basic"):
            spec = _spec(pipeline=pipeline)
            pfa = self._pick(spec.tile_m, spec.tile_n, spec.tile_k)
            self.assertTrue(pfa)
            spec = replace(spec, prefetch_addr=pfa)
            spec.validate()
            self.assertTrue(is_valid_wgrad_spec(spec, "gfx950")[0])

    def test_aot_job_generator_uses_the_policy(self):
        from kernels.conv.aot_jobs import _wgrad_prefetch_addr

        cfg = dict(tile_m=128, tile_n=64, tile_k=64, wave_size=64, lds_k_outer=True)
        self.assertTrue(_wgrad_prefetch_addr("gfx950", cfg))
        self.assertFalse(_wgrad_prefetch_addr("gfx950", dict(cfg, lds_k_outer=False)))
        self.assertFalse(_wgrad_prefetch_addr("gfx950", dict(cfg, tile_n=256)))
        # A job without the K-outer axis (the spec default) keeps it off.
        cfg.pop("lds_k_outer")
        self.assertFalse(_wgrad_prefetch_addr("gfx950", cfg))

    def test_jobs_carry_it_in_spec_and_identity(self):
        from kernels.conv import aot_jobs as ks
        from rocke.core.arch import ArchTarget

        target = ArchTarget.from_gfx("gfx950")
        jobs = list(
            ks._wgrad_jobs(
                "gfx950",
                "bf16",
                64,
                "mma",
                target,
                (2,),
                geometries=[
                    (128, 64, 64, 2, 2, 16, "mem", "cshuffle", _atom16x16x32())
                ],
            )
        )
        self.assertTrue(jobs)
        for job in jobs:
            self.assertEqual(
                job.identity.prefetch_addr, job.spec_kwargs["prefetch_addr"]
            )
            self.assertEqual(
                job.spec_kwargs["prefetch_addr"],
                ks._wgrad_prefetch_addr("gfx950", job.spec_kwargs),
            )


def _atom16x16x32():
    from rocke.core.arch import ArchTarget

    return ArchTarget.from_gfx("gfx950").mma.select_largest_k(
        family="mma", a_dtype="bf16", b_dtype="bf16", c_dtype="fp32", m=16, n=16
    )


class TestCoalescedTileLoaderSlots(unittest.TestCase):
    """The plan/offset/issue split emits the ops ``load`` emits, regrouped."""

    def _ops(self, use_split: bool, vector_axis: str):
        from rocke.core.ir import BF16, I32, IRBuilder, PtrType
        from rocke.helpers.loads import CoalescedTileLoader

        b = IRBuilder("k")
        p = b.param("p", PtrType(BF16, "global"))
        n = b.param("n", I32)
        rsrc = b.buffer_rsrc(p, n)
        smem = b.smem_alloc(BF16, (64, 72), name_hint="s")
        tid = b.thread_id_x()
        ld = CoalescedTileLoader.from_tile(
            tile_rows=64,
            tile_cols=64,
            block_size=256,
            elem_dtype=BF16,
            vector_axis=vector_axis,
        )

        def desc(b_, row, col):
            return b_.add(b_.mul(row, n), col), b_.cmp_lt(row, n)

        if use_split:
            slots = ld.plan_slots(b, tid=tid)
            offs = ld.slot_offsets(b, slots=slots, descriptor=desc)
            ld.store_lds(
                b,
                smem_dst=smem,
                staged=ld.issue_slots(b, rsrc=rsrc, slots=slots, offsets=offs),
            )
        else:
            ld.load(b, tid=tid, smem_dst=smem, descriptor=desc, rsrc=rsrc)
        return sorted(op.name for op in b.kernel.body.ops)

    def test_same_ops_as_load(self):
        for axis in ("col", "row"):
            with self.subTest(vector_axis=axis):
                self.assertEqual(self._ops(True, axis), self._ops(False, axis))


class TestKernelIdentityPrefetchAddr(unittest.TestCase):
    def _ident(self, **kw):
        from benchmarks.common.kernel_cache import KernelIdentity

        base = dict(
            arch="gfx950",
            direction="wgrad",
            algorithm="implicit_gemm",
            dtype_a="bf16",
            dtype_b="bf16",
            dtype_d="bf16",
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
            wave_size=64,
            vector_size_a=8,
            vector_size_b=8,
            vector_size_c=8,
            llvm_flavor="llvm20",
        )
        base.update(kw)
        return KernelIdentity(**base)

    def test_off_keeps_the_historical_hash(self):
        import hashlib
        import json
        from dataclasses import asdict

        ident = self._ident()
        fields = asdict(ident)
        del fields["prefetch_addr"]
        del fields["ws_replicas"]
        del fields["uniform_k_decode"]  # dgrad's flag, likewise hash-neutral
        del fields["patch"]  # dgrad's patch store, likewise
        del fields["tap_outer_k"]  # dgrad's tap-outer K loop, off
        del fields["dy_halo"], fields["halo_max_wo"]  # dgrad dY halo, off
        del fields["dy_halo_setprio"], fields["dy_halo_kouter_pad"]
        del fields["streamk"], fields["streamk_reduction"]  # stream-K, off
        legacy = hashlib.sha1(
            json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(ident.stable_hash(), legacy)

    def test_on_is_a_distinct_identity(self):
        on = self._ident(prefetch_addr=True)
        self.assertNotEqual(on.stable_hash(), self._ident().stable_hash())
        self.assertIn("pfa", on.label())


if __name__ == "__main__":
    unittest.main(verbosity=2)
