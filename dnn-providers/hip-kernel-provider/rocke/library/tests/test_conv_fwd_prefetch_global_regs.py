# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side gate for forward ``prefetch_global_regs`` (the register-prefetch K loop).

No GPU and no comgr: spec construction, the validity gates, the lowered IR,
the kernel identity and the AOT grid, plus the C++ engine's IR where the
``rocke_library_engine`` binding is importable. The numerical counterpart is
``TestFwdPrefetchGlobalRegsNumerics`` in ``test_conv_fwd_correctness.py``; the
two-engine counterpart is configs 86-97 of the forward parity emitters. The
loop itself is shared with wgrad (``emit_prefetch_global_regs_kloop``), whose
gate is ``test_conv_wgrad_prefetch_global_regs.py``.
"""

from __future__ import annotations

import dataclasses
import re
import unittest
from dataclasses import replace

from rocke.tests.engines import skip_without_cpp_engine


def _spec(dtype="fp16", **kw):
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
    from kernels.common.conv_implicit_gemm import ImplicitGemmConvSpec

    base = dict(
        problem=ConvProblem(N=2, Hi=12, Wi=12, C=128, K=96, Y=3, X=3, pH=1, pW=1),
        data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
        wave_size=64,
        pipeline="mem",
        epilogue="cshuffle",
        vector_size_a=8,
        vector_size_b=8,
        vector_size_c=8,
    )
    base.update(kw)
    return ImplicitGemmConvSpec(**base)


def _lower(spec, arch="gfx950") -> str:
    from kernels.common.conv_implicit_gemm import build_implicit_gemm_conv
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    return lower_kernel_to_llvm(build_implicit_gemm_conv(spec, arch=arch), arch=arch)


def _buffer_loads(ll: str) -> int:
    return len(re.findall(r"call [^\n]*amdgcn\.raw\.(?:ptr\.)?buffer\.load", ll))


def _loop_body(ll: str) -> str:
    """The K loop body: from the block the loop header enters to the latch."""
    header = ll.index("%pgra0 = phi")
    start = ll.index("\nfor.body", ll.index("br i1", header))
    return ll[start : ll.index("for.latch", start + 1)]


# s_waitcnt immediates on gfx9: lgkmcnt(0) alone leaves vmcnt at its maximum;
# a full drain zeroes vmcnt too.
_WAIT_LGKM_ONLY = "s.waitcnt(i32 49279)"


class TestFwdPrefetchGlobalRegsGate(unittest.TestCase):
    def test_default_is_off_and_untagged(self):
        spec = _spec()
        self.assertFalse(spec.prefetch_global_regs)
        self.assertNotIn("pgr", spec.kernel_name())
        self.assertTrue(
            replace(spec, prefetch_global_regs=True).kernel_name().endswith("_pgr")
        )

    def test_staged_values_are_loop_carried(self):
        off = _lower(_spec())
        on = _lower(_spec(prefetch_global_regs=True))
        self.assertNotIn("%pgra0", off)
        # One carried vector per A and B chunk: 64x64 tiles at vec 8 over 256
        # threads are two chunks per thread per operand.
        self.assertEqual(len(re.findall(r"%pgr[ab]\d+ = phi <8 x half>", on)), 4)
        # A prologue issues the first tile, so every read appears twice.
        self.assertEqual(_buffer_loads(on), 2 * _buffer_loads(off))

    def test_next_tile_is_read_after_the_barrier_and_before_the_mfmas(self):
        body = _loop_body(_lower(_spec(prefetch_global_regs=True)))

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
        body = _loop_body(_lower(_spec(prefetch_global_regs=True)))
        waits = re.findall(r"s\.waitcnt\(i32 \d+\)", body)
        self.assertTrue(waits)
        self.assertTrue(all(w == _WAIT_LGKM_ONLY for w in waits), waits)
        # The plain loop drains everything at its barriers.
        self.assertNotIn(_WAIT_LGKM_ONLY, _lower(_spec()))

    def test_every_sync_pipeline_is_admitted(self):
        from kernels.common.conv_implicit_gemm import is_valid_spec

        for dtype in ("fp16", "bf16"):
            for pipeline in ("mem", "compv3", "compv4", "basic"):
                for lds_k_pad in (None, 16):
                    with self.subTest(dtype=dtype, pipeline=pipeline, pad=lds_k_pad):
                        spec = _spec(
                            dtype,
                            pipeline=pipeline,
                            lds_k_pad=lds_k_pad,
                            prefetch_global_regs=True,
                        )
                        spec.validate()
                        ok, why = is_valid_spec(spec, "gfx950")
                        self.assertTrue(ok, why)
                        self.assertIn("%pgra0 = phi", _lower(spec))

    def test_split_k_group_merge_and_default_epilogue_are_admitted(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem
        from kernels.common.conv_implicit_gemm import is_valid_spec

        dw = ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=64)
        for kw in (
            dict(split_k=2),
            dict(epilogue="default", vector_size_c=1),
            dict(
                problem=dw,
                group_merge=8,
                vector_size_a=None,
                vector_size_b=None,
                vector_size_c=None,
            ),
        ):
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                spec = _spec(prefetch_global_regs=True, **kw)
                ok, why = is_valid_spec(spec, "gfx950")
                self.assertTrue(ok, why)
                self.assertIn("%pgra0 = phi", _lower(spec))

    def test_rejects_loops_it_does_not_rewrite(self):
        from kernels.common.conv_implicit_gemm import is_valid_spec

        for kw, word in (
            (dict(async_dma=True), "async_dma"),
            (dict(unroll_k=True), "unroll_k"),
        ):
            with self.subTest(word=word):
                spec = _spec(prefetch_global_regs=True, **kw)
                with self.assertRaisesRegex(ValueError, word):
                    spec.validate()
                ok, why = is_valid_spec(spec, "gfx950")
                self.assertFalse(ok)
                self.assertIn("prefetch_global_regs", why)
                # Without the flag the same loop is admitted, so the flag is
                # what the gate rejects.
                ok, why = is_valid_spec(
                    replace(spec, prefetch_global_regs=False), "gfx950"
                )
                self.assertTrue(ok, why)

    def test_rejects_the_wavelet_pipeline(self):
        spec = _spec(pipeline="wavelet", prefetch_global_regs=True)
        with self.assertRaisesRegex(ValueError, "pipeline='wavelet'"):
            spec.validate()

    def test_staged_loads_are_capped_per_operand(self):
        """Both engines carry at most 32 global loads per thread per operand."""
        from kernels.common.conv_implicit_gemm import is_valid_spec

        # Scalar loads over 256 threads: a 256x64 A tile is 64 loads per thread.
        spec = _spec(
            tile_m=256, vector_size_a=1, vector_size_b=1, prefetch_global_regs=True
        )
        with self.assertRaisesRegex(ValueError, "the A tile needs 64"):
            spec.validate()
        ok, why = is_valid_spec(spec, "gfx950")
        self.assertFalse(ok)
        self.assertIn("at most 32 global loads per thread per operand", why)
        # The plain loop stages nothing across iterations and is unaffected.
        ok, why = is_valid_spec(replace(spec, prefetch_global_regs=False), "gfx950")
        self.assertTrue(ok, why)
        # 128x64 is exactly 32 per thread: admitted and built.
        spec = replace(spec, tile_m=128)
        ok, why = is_valid_spec(spec, "gfx950")
        self.assertTrue(ok, why)
        self.assertEqual(len(re.findall(r"%pgra\d+ = phi", _lower(spec))), 32)
        # The B operand alone can exceed the cap too.
        spec = _spec(tile_n=256, vector_size_b=1, prefetch_global_regs=True)
        ok, why = is_valid_spec(spec, "gfx950")
        self.assertFalse(ok)
        self.assertIn("the B tile needs 64", why)

    def test_bf16_operands_are_carried_as_bfloat(self):
        ll = _lower(_spec("bf16", prefetch_global_regs=True))
        self.assertEqual(len(re.findall(r"%pgr[ab]\d+ = phi <8 x bfloat>", ll)), 4)
        self.assertNotIn("x half>", _loop_body(ll))

    def test_rejected_on_wmma(self):
        from kernels.common.conv_implicit_gemm import is_valid_spec

        spec = _spec(
            tile_m=32,
            tile_n=32,
            tile_k=32,
            warp_m=2,
            warp_n=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
            wave_size=32,
        )
        ok, why = is_valid_spec(spec, "gfx1250")
        self.assertTrue(ok, why)
        ok, why = is_valid_spec(replace(spec, prefetch_global_regs=True), "gfx1250")
        self.assertFalse(ok)
        self.assertIn("prefetch_global_regs", why)


def _ident(**kw):
    from kernels.conv.kernel_identity import KernelIdentity

    base = dict(
        arch="gfx950",
        direction="fwd",
        algorithm="implicit_gemm",
        dtype_a="fp16",
        dtype_b="fp16",
        dtype_d="fp16",
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
        pipeline="mem",
        epilogue="cshuffle",
        wave_size=64,
        vector_size_a=8,
        vector_size_b=8,
        vector_size_c=8,
        grouped=True,
        llvm_flavor="llvm20",
    )
    base.update(kw)
    return KernelIdentity(**base)


class TestKernelIdentityFwdPrefetchGlobalRegs(unittest.TestCase):
    def test_off_keeps_the_hash(self):
        from kernels.conv.kernel_identity import KernelIdentity

        ident = _ident()
        self.assertFalse(ident.prefetch_global_regs)
        old = dataclasses.asdict(ident)
        del old["prefetch_global_regs"]
        self.assertEqual(KernelIdentity.from_dict(old), ident)
        self.assertEqual(
            KernelIdentity.from_dict(old).stable_hash(), ident.stable_hash()
        )

    def test_on_is_a_distinct_identity_with_a_label_tag(self):
        from kernels.conv.kernel_identity import KernelIdentity

        on = _ident(prefetch_global_regs=True)
        self.assertNotEqual(on.stable_hash(), _ident().stable_hash())
        self.assertIn("-pgr-", on.label())
        self.assertNotIn("pgr", _ident().label())
        self.assertEqual(KernelIdentity.from_label(on.label()), on)


def _geometry(tm, tn, tk, wm, wn, wt, pipeline, epilogue):
    from rocke.core.arch import ArchTarget

    atom = ArchTarget.from_gfx("gfx950").mma.select_largest_k(
        family="mma", a_dtype="fp16", b_dtype="fp16", c_dtype="fp32", m=wt, n=wt
    )
    return (tm, tn, tk, wm, wn, wt, pipeline, epilogue, atom)


class TestAotFwdPrefetchGlobalRegsGrid(unittest.TestCase):
    def _jobs(self, arch, geometries, dtype="fp16"):
        from kernels.conv import aot_jobs as aj
        from rocke.core.arch import ArchTarget

        target = ArchTarget.from_gfx(arch)
        return list(
            aj._fwd_jobs(
                arch, dtype, target.wave_size, "mma", target, geometries=geometries
            )
        )

    def test_built_on_the_cover_only(self):
        from kernels.conv import aot_jobs as aj

        cover = aj.CACHE_FWD_PREFETCH_GLOBAL_REGS_GEOMETRIES[0]
        off_cover = (64, 64, 64, 1, 1, 32)
        self.assertNotIn(off_cover, aj.CACHE_FWD_PREFETCH_GLOBAL_REGS_GEOMETRIES)
        for pipeline in aj.CACHE_FWD_LOOP_VARIANT_PIPELINES:
            with self.subTest(pipeline=pipeline):
                jobs = self._jobs("gfx950", [_geometry(*cover, pipeline, "cshuffle")])
                pgr = [
                    j
                    for j in jobs
                    if j.identity.prefetch_global_regs and not j.identity.lds_k_pad
                ]
                # One per cover geometry and pipeline: the 8 / 8 / 8 plain loop.
                self.assertEqual(len(pgr), 1)
                i = pgr[0].identity
                self.assertEqual(
                    (i.vector_size_a, i.vector_size_b, i.vector_size_c), (8, 8, 8)
                )
                self.assertFalse(i.unroll_k or i.async_dma)
                self.assertTrue(pgr[0].spec_kwargs["prefetch_global_regs"])
                # Next to the plain-loop job it is derived from.
                twin = replace(i, prefetch_global_regs=False)
                self.assertIn(twin, [x.identity for x in jobs])
                jobs = self._jobs(
                    "gfx950", [_geometry(*off_cover, pipeline, "cshuffle")]
                )
                self.assertFalse([j for j in jobs if j.identity.prefetch_global_regs])
        # Never on the default epilogue.
        jobs = self._jobs("gfx950", [_geometry(*cover, "mem", "default")])
        self.assertFalse([j for j in jobs if j.identity.prefetch_global_regs])

    def test_other_arches_do_not_build_it(self):
        from kernels.conv import aot_jobs as aj

        cover = aj.CACHE_FWD_PREFETCH_GLOBAL_REGS_GEOMETRIES[0]
        jobs = self._jobs("gfx942", [_geometry(*cover, "mem", "cshuffle")])
        self.assertTrue(jobs)
        self.assertFalse([j for j in jobs if j.identity.prefetch_global_regs])

    def test_jobs_validate_and_rebuild_from_their_identity(self):
        from kernels.conv import aot_jobs as aj

        for dtype in ("fp16", "bf16"):
            for cover in aj.CACHE_FWD_PREFETCH_GLOBAL_REGS_GEOMETRIES:
                jobs = self._jobs(
                    "gfx950", [_geometry(*cover, "compv3", "cshuffle")], dtype
                )
                for j in [j for j in jobs if j.identity.prefetch_global_regs]:
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
        jobs = list(aj._fwd_jobs("gfx950", "fp16", 64, "mma", target))
        n_pgr = sum(
            1
            for j in jobs
            if j.identity.prefetch_global_regs
            and not j.identity.lds_k_pad
            and not j.identity.patch
        )
        self.assertEqual(
            n_pgr,
            len(aj.CACHE_FWD_PREFETCH_GLOBAL_REGS_GEOMETRIES)
            * len(aj.CACHE_FWD_LOOP_VARIANT_PIPELINES),
        )
        self.assertEqual(
            len(jobs),
            aj.count_jobs(dtype="fp16", target=target, directions=["fwd"]),
        )


class TestFwdPrefetchGlobalRegsCrossEngine(unittest.TestCase):
    """The C++ engine emits the Python IR byte for byte and rejects alike."""

    def setUp(self):
        from rocke.core import BackendError, import_engine

        from kernels.conv.engine_bridge import ENGINE_MODULE

        try:
            self.eng = import_engine(ENGINE_MODULE)
        except BackendError as e:
            skip_without_cpp_engine(e, f"C++ engine not importable: {e}")

    def _lower_cpp(self, spec):
        from kernels.conv.engine_bridge import conv_implicit_gemm_spec_to_dict

        return self.eng.conv_implicit_gemm_lower_llvm(
            conv_implicit_gemm_spec_to_dict(spec), "gfx950"
        )

    def test_identical_ir(self):
        for kw in (
            dict(),
            dict(dtype="bf16", pipeline="compv4", tile_m=128, tile_n=128),
            dict(pipeline="compv3", lds_k_pad=16),
            dict(pipeline="basic", epilogue="default", vector_size_c=1),
            dict(split_k=2),
            dict(tile_m=128, vector_size_a=1, vector_size_b=1),
        ):
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                spec = _spec(prefetch_global_regs=True, **kw)
                self.assertEqual(self._lower_cpp(spec), _lower(spec))

    def test_rejects_alike(self):
        for kw, why in (
            (dict(async_dma=True), "not supported with async_dma"),
            (dict(unroll_k=True), "not supported with unroll_k"),
            (
                dict(tile_m=256, vector_size_a=1, vector_size_b=1),
                "the A tile needs 64",
            ),
        ):
            with self.subTest(why=why):
                spec = _spec(prefetch_global_regs=True, **kw)
                with self.assertRaisesRegex(ValueError, why):
                    _lower(spec)
                with self.assertRaisesRegex(RuntimeError, why):
                    self._lower_cpp(spec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
