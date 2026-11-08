# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side gate for wgrad ``prefetch_regs`` (the register-prefetch K loop).

No GPU and no comgr: spec construction, the validity gates, the lowered IR,
the kernel identity and the AOT grid, plus the C++ engine's IR where the
``rocke_engine`` binding is importable. The numerical counterpart is
``TestWgradPrefetchRegsNumerics`` in ``test_conv_wgrad_correctness.py``; the
two-engine counterpart is configs 73-84 of the wgrad parity emitters.
"""

from __future__ import annotations

import dataclasses
import re
import unittest
from dataclasses import replace


def _spec(dtype="fp16", **kw):
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
    from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

    base = dict(
        problem=ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1),
        data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
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


def _loop_body(ll: str) -> str:
    """The K loop body: from the block the loop header enters to the latch."""
    header = ll.index("%pfra0 = phi")
    start = ll.index("\nfor.body", ll.index("br i1", header))
    return ll[start : ll.index("for.latch", start + 1)]


# s_waitcnt immediates on gfx9: lgkmcnt(0) alone leaves vmcnt at its maximum;
# a full drain zeroes vmcnt too.
_WAIT_LGKM_ONLY = "s.waitcnt(i32 49279)"


class TestWgradPrefetchRegsGate(unittest.TestCase):
    def test_default_is_off_and_untagged(self):
        spec = _spec()
        self.assertFalse(spec.prefetch_regs)
        self.assertNotIn("pfr", spec.kernel_name())
        self.assertTrue(
            replace(spec, prefetch_regs=True).kernel_name().endswith("_pfr")
        )

    def test_staged_values_are_loop_carried(self):
        off = _lower(_spec())
        on = _lower(_spec(prefetch_regs=True))
        self.assertNotIn("%pfra0", off)
        # One carried vector per A and B chunk: 64x64 K-outer tiles at vec 8
        # over 256 threads are two chunks per thread per operand.
        self.assertEqual(len(re.findall(r"%pfr[ab]\d+ = phi <8 x half>", on)), 4)
        # A prologue issues the first tile, so every read appears twice.
        self.assertEqual(_buffer_loads(on), 2 * _buffer_loads(off))

    def test_next_tile_is_read_after_the_barrier_and_before_the_mfmas(self):
        body = _loop_body(_lower(_spec(prefetch_regs=True)))

        def first(pat: str) -> int:
            return re.search(pat, body).start()

        lds = first(r"store [^\n]*addrspace\(3\)")
        barrier = first(r"amdgcn\.s\.barrier")
        load = first(r"amdgcn\.raw\.(?:ptr\.)?buffer\.load")
        mfma = first(r"llvm\.amdgcn\.mfma")
        # Write the carried tile, publish it, read the next one, then MFMA.
        self.assertLess(lds, barrier)
        self.assertLess(barrier, load)
        self.assertLess(load, mfma)

    def test_loop_barriers_do_not_drain_the_global_reads(self):
        body = _loop_body(_lower(_spec(prefetch_regs=True)))
        waits = re.findall(r"s\.waitcnt\(i32 \d+\)", body)
        self.assertTrue(waits)
        self.assertTrue(all(w == _WAIT_LGKM_ONLY for w in waits), waits)
        # The plain loop drains everything at its barriers.
        plain = _lower(_spec())
        self.assertNotIn(_WAIT_LGKM_ONLY, plain)

    def test_every_sync_pipeline_is_admitted(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        for dtype in ("fp16", "bf16"):
            for pipeline in ("mem", "compv3", "compv4", "basic"):
                for kouter in (False, True):
                    with self.subTest(dtype=dtype, pipeline=pipeline, kouter=kouter):
                        spec = _spec(
                            dtype,
                            pipeline=pipeline,
                            lds_k_outer=kouter,
                            prefetch_regs=True,
                        )
                        spec.validate()
                        ok, why = is_valid_wgrad_spec(spec, "gfx950")
                        self.assertTrue(ok, why)
                        self.assertIn("%pfra0 = phi", _lower(spec))

    def test_split_k_and_two_stage_are_admitted(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        for dtype in ("fp16", "bf16"):
            for kw in (
                dict(split_k=2),
                dict(split_k=2, two_stage=True, epilogue="default"),
            ):
                with self.subTest(dtype=dtype, **kw):
                    spec = _spec(dtype, prefetch_regs=True, **kw)
                    ok, why = is_valid_wgrad_spec(spec, "gfx950")
                    self.assertTrue(ok, why)
                    self.assertIn("%pfra0 = phi", _lower(spec))

    def test_rejects_loops_it_does_not_rewrite(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        for kw, word in (
            (dict(async_dma=True, epilogue="default", vector_size_a=None), "async_dma"),
            (dict(unroll_k=True), "unroll_k"),
            (dict(prefetch_addr=True), "prefetch_addr"),
            (dict(streamk="dp_sk", epilogue="default"), "streamk"),
        ):
            with self.subTest(word=word):
                spec = _spec(prefetch_regs=True, **kw)
                with self.assertRaisesRegex(ValueError, word):
                    spec.validate()
                ok, why = is_valid_wgrad_spec(spec, "gfx950")
                self.assertFalse(ok)
                self.assertIn("prefetch_regs", why)
                # Without the flag the same loop is admitted, so the flag is
                # what the gate rejects.
                ok, why = is_valid_wgrad_spec(
                    replace(spec, prefetch_regs=False), "gfx950"
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
        ok, why = is_valid_wgrad_spec(replace(spec, prefetch_regs=True), "gfx1151")
        self.assertFalse(ok)
        self.assertIn("prefetch_regs", why)


def _ident(**kw):
    from kernels.conv.kernel_identity import KernelIdentity

    base = dict(
        arch="gfx950",
        direction="wgrad",
        algorithm="implicit_gemm",
        dtype_a="fp16",
        dtype_b="fp16",
        dtype_d="fp16",
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


class TestKernelIdentityPrefetchRegs(unittest.TestCase):
    def test_off_keeps_the_hash(self):
        from kernels.conv.kernel_identity import KernelIdentity

        ident = _ident()
        self.assertFalse(ident.prefetch_regs)
        # An identity written before the field existed hashes the same: this
        # is the hash it had then.
        self.assertEqual(
            ident.stable_hash(), "1466094dbf6e49654ee8a31556a9f117b2e191c7"
        )
        old = dataclasses.asdict(ident)
        del old["prefetch_regs"]
        self.assertEqual(KernelIdentity.from_dict(old), ident)

    def test_on_is_a_distinct_identity_with_a_label_tag(self):
        from kernels.conv.kernel_identity import KernelIdentity

        on = _ident(prefetch_regs=True)
        self.assertNotEqual(on.stable_hash(), _ident().stable_hash())
        self.assertIn("-pfr-", on.label())
        self.assertNotIn("pfr", _ident().label())
        self.assertEqual(KernelIdentity.from_label(on.label()), on)


def _geometry(tm, tn, tk, wm, wn, wt, pipeline, epilogue):
    from rocke.core.arch import ArchTarget

    atom = ArchTarget.from_gfx("gfx950").mma.select_largest_k(
        family="mma", a_dtype="fp16", b_dtype="fp16", c_dtype="fp32", m=wt, n=wt
    )
    return (tm, tn, tk, wm, wn, wt, pipeline, epilogue, atom)


class TestAotPrefetchRegsGrid(unittest.TestCase):
    def _jobs(self, arch, geometries, dtype="fp16"):
        from kernels.conv import aot_jobs as aj
        from rocke.core.arch import ArchTarget

        target = ArchTarget.from_gfx(arch)
        return list(
            aj._wgrad_jobs(
                arch,
                dtype,
                target.wave_size,
                "mma",
                target,
                aj.CACHE_WGRAD_SPLIT_KS,
                geometries=geometries,
            )
        )

    def test_built_on_the_cover_only(self):
        from kernels.conv import aot_jobs as aj

        cover = aj.CACHE_WGRAD_PREFETCH_REGS_GEOMETRIES[0]
        off_cover = (16, 64, 64, 1, 4, 16)
        self.assertNotIn(off_cover, aj.CACHE_WGRAD_PREFETCH_REGS_GEOMETRIES)
        for epilogue in ("default", "cshuffle"):
            with self.subTest(epilogue=epilogue):
                jobs = self._jobs("gfx950", [_geometry(*cover, "mem", epilogue)])
                pfr = [j for j in jobs if j.identity.prefetch_regs]
                # One per (8, 8) stage the cover takes: cshuffle at split_k 1
                # and atomic split-K, the default epilogue under two-stage.
                self.assertEqual(len(pfr), 2 if epilogue == "cshuffle" else 1)
                for j in pfr:
                    i = j.identity
                    self.assertEqual((i.vector_size_a, i.vector_size_b), (8, 8))
                    self.assertFalse(i.prefetch_addr)
                    self.assertTrue(j.spec_kwargs["prefetch_regs"])
                    self.assertFalse(j.spec_kwargs["prefetch_addr"])
                    if epilogue == "default":
                        self.assertTrue(i.two_stage)
                    else:
                        self.assertFalse(i.two_stage)
                    # Next to the plain-loop job it is derived from.
                    twin = replace(
                        i,
                        prefetch_regs=False,
                        prefetch_addr=aj._wgrad_prefetch_addr("gfx950", j.spec_kwargs),
                    )
                    self.assertIn(twin, [x.identity for x in jobs])
                jobs = self._jobs("gfx950", [_geometry(*off_cover, "mem", epilogue)])
                self.assertFalse([j for j in jobs if j.identity.prefetch_regs])

    def test_other_arches_do_not_build_it(self):
        from kernels.conv import aot_jobs as aj

        cover = aj.CACHE_WGRAD_PREFETCH_REGS_GEOMETRIES[0]
        jobs = self._jobs("gfx942", [_geometry(*cover, "mem", "cshuffle")])
        self.assertTrue(jobs)
        self.assertFalse([j for j in jobs if j.identity.prefetch_regs])

    def test_jobs_validate_and_rebuild_from_their_identity(self):
        from kernels.conv import aot_jobs as aj

        cover = aj.CACHE_WGRAD_PREFETCH_REGS_GEOMETRIES[0]
        for dtype in ("fp16", "bf16"):
            for epilogue in ("default", "cshuffle"):
                jobs = self._jobs(
                    "gfx950", [_geometry(*cover, "compv3", epilogue)], dtype
                )
                for j in [j for j in jobs if j.identity.prefetch_regs]:
                    with self.subTest(dtype=dtype, label=j.identity.label()):
                        self.assertTrue(aj._spec_is_valid(j, "gfx950", dtype))
                        self.assertEqual(
                            aj.job_for_identity(j.identity).identity, j.identity
                        )

    def test_count_matches_the_walk(self):
        """count_jobs() counts the register-prefetch jobs the walk yields."""
        from kernels.conv import aot_jobs as aj
        from rocke.core.arch import ArchTarget

        target = ArchTarget.from_gfx("gfx950")
        jobs = list(
            aj._wgrad_jobs("gfx950", "fp16", 64, "mma", target, aj.CACHE_WGRAD_SPLIT_KS)
        )
        n_pfr = sum(1 for j in jobs if j.identity.prefetch_regs)
        self.assertGreater(n_pfr, 0)
        self.assertEqual(
            len(jobs),
            aj.count_jobs(
                dtype="fp16",
                target=target,
                directions=["wgrad"],
                split_ks=aj.CACHE_WGRAD_SPLIT_KS,
            ),
        )


class TestPrefetchRegsCrossEngine(unittest.TestCase):
    """The C++ engine emits the Python IR byte for byte and rejects alike."""

    def setUp(self):
        from rocke.core.backend import BackendError, _import_engine

        try:
            self.eng = _import_engine()
        except BackendError as e:
            self.skipTest(f"C++ engine not importable: {e}")

    def _dict(self, spec):
        d = {
            f.name: getattr(spec, f.name)
            for f in dataclasses.fields(spec)
            if f.name not in ("problem", "data", "acc_epilogue", "lds_layout")
        }
        d = {k: v for k, v in d.items() if v is not None}
        d["problem"] = {
            k: v for k, v in dataclasses.asdict(spec.problem).items() if v is not None
        }
        d["dtype_a"] = spec.data.dtype_a
        d["dtype_b"] = spec.data.dtype_b
        d["dtype_d"] = spec.data.dtype_d
        return d

    def test_identical_ir(self):
        # fp16 operands: the C++ synchronous wgrad loader loads 16-bit
        # elements as f16, so bf16 dW is covered through the output dtype.
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        bf16_out = ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="bf16")
        for kw in (
            dict(),
            dict(lds_k_outer=False, epilogue="default"),
            dict(pipeline="compv3", split_k=2),
            dict(pipeline="compv4", split_k=2, two_stage=True, epilogue="default"),
            dict(split_k=2, data=bf16_out),
            dict(
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
                tile_m=128,
                tile_n=128,
                split_k=2,
                two_stage=True,
                epilogue="default",
            ),
        ):
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                spec = _spec(prefetch_regs=True, **kw)
                self.assertEqual(
                    self.eng.conv_wgrad_lower_llvm(self._dict(spec), "gfx950"),
                    _lower(spec),
                )

    def test_rejects_alike(self):
        for kw in (
            dict(prefetch_addr=True),
            dict(unroll_k=True),
            dict(streamk="dp_sk", epilogue="default"),
        ):
            with self.subTest(**kw):
                spec = _spec(prefetch_regs=True, **kw)
                with self.assertRaises(ValueError):
                    _lower(spec)
                with self.assertRaises(RuntimeError):
                    self.eng.conv_wgrad_lower_llvm(self._dict(spec), "gfx950")


if __name__ == "__main__":
    unittest.main(verbosity=2)
