# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side gate for wgrad ``prefetch_global_regs`` (the register-prefetch K loop).

No GPU and no comgr: spec construction, the validity gates, the lowered IR,
the kernel identity and the AOT grid, plus the C++ engine's IR where the
``rocke_library_engine`` binding is importable. The numerical counterpart is
``TestWgradPrefetchGlobalRegsNumerics`` in ``test_conv_wgrad_correctness.py``; the
two-engine counterpart is configs 89-105 of the wgrad parity emitters.
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
    header = ll.index("%pgra0 = phi")
    start = ll.index("\nfor.body", ll.index("br i1", header))
    return ll[start : ll.index("for.latch", start + 1)]


# s_waitcnt immediates on gfx9: lgkmcnt(0) alone leaves vmcnt at its maximum;
# a full drain zeroes vmcnt too.
_WAIT_LGKM_ONLY = "s.waitcnt(i32 49279)"


class TestWgradPrefetchGlobalRegsGate(unittest.TestCase):
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
        # One carried vector per A and B chunk: 64x64 K-outer tiles at vec 8
        # over 256 threads are two chunks per thread per operand.
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
                            prefetch_global_regs=True,
                        )
                        spec.validate()
                        ok, why = is_valid_wgrad_spec(spec, "gfx950")
                        self.assertTrue(ok, why)
                        self.assertIn("%pgra0 = phi", _lower(spec))

    def test_split_k_and_two_stage_are_admitted(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        for dtype in ("fp16", "bf16"):
            for kw in (
                dict(split_k=2),
                dict(split_k=2, two_stage=True, epilogue="default"),
            ):
                with self.subTest(dtype=dtype, **kw):
                    spec = _spec(dtype, prefetch_global_regs=True, **kw)
                    ok, why = is_valid_wgrad_spec(spec, "gfx950")
                    self.assertTrue(ok, why)
                    self.assertIn("%pgra0 = phi", _lower(spec))

    def test_rejects_loops_it_does_not_rewrite(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        for kw, word in (
            (dict(async_dma=True, epilogue="default", vector_size_a=None), "async_dma"),
            (dict(unroll_k=True), "unroll_k"),
            (dict(prefetch_addr=True), "prefetch_addr"),
            (dict(streamk="dp_sk", epilogue="default"), "streamk"),
        ):
            with self.subTest(word=word):
                spec = _spec(prefetch_global_regs=True, **kw)
                with self.assertRaisesRegex(ValueError, word):
                    spec.validate()
                ok, why = is_valid_wgrad_spec(spec, "gfx950")
                self.assertFalse(ok)
                self.assertIn("prefetch_global_regs", why)
                # Without the flag the same loop is admitted, so the flag is
                # what the gate rejects.
                ok, why = is_valid_wgrad_spec(
                    replace(spec, prefetch_global_regs=False), "gfx950"
                )
                self.assertTrue(ok, why)

    def test_staged_loads_are_capped_per_operand(self):
        """Both engines carry at most 32 global loads per thread per operand."""
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        # Scalar loads over 256 threads: a 64x256 tile is 64 loads per thread.
        spec = _spec(
            tile_k=256, vector_size_a=1, vector_size_b=1, prefetch_global_regs=True
        )
        with self.assertRaisesRegex(ValueError, "the A tile needs 64"):
            spec.validate()
        ok, why = is_valid_wgrad_spec(spec, "gfx950")
        self.assertFalse(ok)
        self.assertIn("at most 32 global loads per thread per operand", why)
        # The plain loop stages nothing across iterations and is unaffected.
        ok, why = is_valid_wgrad_spec(
            replace(spec, prefetch_global_regs=False), "gfx950"
        )
        self.assertTrue(ok, why)
        # 64x128 is exactly 32 per thread: admitted and built.
        spec = replace(spec, tile_k=128)
        ok, why = is_valid_wgrad_spec(spec, "gfx950")
        self.assertTrue(ok, why)
        self.assertEqual(len(re.findall(r"%pgra\d+ = phi", _lower(spec))), 32)
        # On the M-outer tile the B operand alone can exceed the cap.
        spec = _spec(
            lds_k_outer=False,
            tile_n=128,
            tile_k=128,
            vector_size_b=1,
            prefetch_global_regs=True,
        )
        ok, why = is_valid_wgrad_spec(spec, "gfx950")
        self.assertFalse(ok)
        self.assertIn("the B tile needs 64", why)

    def test_bf16_operands_are_carried_as_bfloat(self):
        spec = _spec("bf16", prefetch_global_regs=True)
        ll = _lower(spec)
        self.assertEqual(len(re.findall(r"%pgr[ab]\d+ = phi <8 x bfloat>", ll)), 4)
        self.assertNotIn("x half>", _loop_body(ll))

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
        ok, why = is_valid_wgrad_spec(
            replace(spec, prefetch_global_regs=True), "gfx1151"
        )
        self.assertFalse(ok)
        self.assertIn("prefetch_global_regs", why)


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


class TestKernelIdentityPrefetchGlobalRegs(unittest.TestCase):
    def test_off_keeps_the_hash(self):
        from kernels.conv.kernel_identity import KernelIdentity

        ident = _ident()
        self.assertFalse(ident.prefetch_global_regs)
        # An identity written before the field existed hashes the same: this
        # is the hash it had then.
        self.assertEqual(
            ident.stable_hash(), "1466094dbf6e49654ee8a31556a9f117b2e191c7"
        )
        old = dataclasses.asdict(ident)
        del old["prefetch_global_regs"]
        self.assertEqual(KernelIdentity.from_dict(old), ident)

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


class TestAotPrefetchGlobalRegsGrid(unittest.TestCase):
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

        cover = aj.CACHE_WGRAD_PREFETCH_GLOBAL_REGS_GEOMETRIES[0]
        off_cover = (16, 64, 64, 1, 4, 16)
        self.assertNotIn(off_cover, aj.CACHE_WGRAD_PREFETCH_GLOBAL_REGS_GEOMETRIES)
        for epilogue in ("default", "cshuffle"):
            with self.subTest(epilogue=epilogue):
                jobs = self._jobs("gfx950", [_geometry(*cover, "mem", epilogue)])
                pgr = [j for j in jobs if j.identity.prefetch_global_regs]
                # One per (8, 8) stage the cover takes: cshuffle at split_k 1
                # and atomic split-K, the default epilogue under two-stage.
                self.assertEqual(len(pgr), 2 if epilogue == "cshuffle" else 1)
                for j in pgr:
                    i = j.identity
                    self.assertEqual((i.vector_size_a, i.vector_size_b), (8, 8))
                    self.assertFalse(i.prefetch_addr)
                    self.assertTrue(j.spec_kwargs["prefetch_global_regs"])
                    self.assertFalse(j.spec_kwargs["prefetch_addr"])
                    if epilogue == "default":
                        self.assertTrue(i.two_stage)
                    else:
                        self.assertFalse(i.two_stage)
                    # Next to the plain-loop job it is derived from.
                    twin = replace(
                        i,
                        prefetch_global_regs=False,
                        prefetch_addr=aj._wgrad_prefetch_addr("gfx950", j.spec_kwargs),
                    )
                    self.assertIn(twin, [x.identity for x in jobs])
                jobs = self._jobs("gfx950", [_geometry(*off_cover, "mem", epilogue)])
                self.assertFalse([j for j in jobs if j.identity.prefetch_global_regs])

    def test_other_arches_do_not_build_it(self):
        from kernels.conv import aot_jobs as aj

        cover = aj.CACHE_WGRAD_PREFETCH_GLOBAL_REGS_GEOMETRIES[0]
        jobs = self._jobs("gfx942", [_geometry(*cover, "mem", "cshuffle")])
        self.assertTrue(jobs)
        self.assertFalse([j for j in jobs if j.identity.prefetch_global_regs])

    def test_jobs_validate_and_rebuild_from_their_identity(self):
        from kernels.conv import aot_jobs as aj

        cover = aj.CACHE_WGRAD_PREFETCH_GLOBAL_REGS_GEOMETRIES[0]
        for dtype in ("fp16", "bf16"):
            for epilogue in ("default", "cshuffle"):
                jobs = self._jobs(
                    "gfx950", [_geometry(*cover, "compv3", epilogue)], dtype
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
        jobs = list(
            aj._wgrad_jobs("gfx950", "fp16", 64, "mma", target, aj.CACHE_WGRAD_SPLIT_KS)
        )
        n_pgr = sum(1 for j in jobs if j.identity.prefetch_global_regs)
        self.assertGreater(n_pgr, 0)
        self.assertEqual(
            len(jobs),
            aj.count_jobs(
                dtype="fp16",
                target=target,
                directions=["wgrad"],
                split_ks=aj.CACHE_WGRAD_SPLIT_KS,
            ),
        )


class TestPrefetchGlobalRegsCrossEngine(unittest.TestCase):
    """The C++ engine emits the Python IR byte for byte and rejects alike."""

    def setUp(self):
        from rocke.core import BackendError, import_engine

        from kernels.conv.engine_bridge import ENGINE_MODULE

        try:
            self.eng = import_engine(ENGINE_MODULE)
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
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        bf16_out = ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="bf16")
        bf16 = ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16")
        for kw in (
            # bf16 A/B: the carried loads and their LDS stores are bfloat.
            dict(data=bf16),
            dict(data=bf16, vector_size_a=1, vector_size_b=1),
            dict(
                data=bf16,
                pipeline="compv4",
                split_k=2,
                two_stage=True,
                epilogue="default",
            ),
            # Scalar loads at the 32-per-thread cap.
            dict(tile_k=128, vector_size_a=1, vector_size_b=1),
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
                spec = _spec(prefetch_global_regs=True, **kw)
                self.assertEqual(
                    self.eng.conv_wgrad_lower_llvm(self._dict(spec), "gfx950"),
                    _lower(spec),
                )

    def test_rejects_alike(self):
        for kw in (
            dict(prefetch_addr=True),
            dict(unroll_k=True),
            dict(streamk="dp_sk", epilogue="default"),
            dict(tile_k=256, vector_size_a=1, vector_size_b=1),
        ):
            with self.subTest(**kw):
                spec = _spec(prefetch_global_regs=True, **kw)
                with self.assertRaises(ValueError) as py:
                    _lower(spec)
                with self.assertRaises(RuntimeError) as cc:
                    self.eng.conv_wgrad_lower_llvm(self._dict(spec), "gfx950")
                # Same reason text from both engines.
                self.assertIn(str(py.exception), str(cc.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
