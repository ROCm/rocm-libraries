# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side gate for the wgrad register-transpose operand load
(``reg_transpose``) and the knobs built on it.

No GPU and no comgr: the spec gates, the kernel names, the AOT identity and
grid, and the lowered IR. The numerical check is ``--run-from-cache --verify``
of the cached kernels. The loader's own work split is covered by
``platform/tests/core/test_reg_transpose_tile_loader.py``.
"""

from __future__ import annotations

import re
import unittest
from dataclasses import replace


def _wgrad(**kw):
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
    from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

    base = dict(
        problem=ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1),
        data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"),
        tile_m=128,
        tile_n=128,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        wave_size=32,
        vector_size_a=4,
        vector_size_b=4,
        vector_size_c=1,
        reg_transpose=True,
    )
    base.update(kw)
    return WgradConvSpec(**base)


def _ident(**kw):
    from kernels.conv.kernel_identity import KernelIdentity

    base = dict(
        arch="gfx1201",
        direction="wgrad",
        algorithm="implicit_gemm",
        dtype_a="fp16",
        dtype_b="fp16",
        dtype_d="fp16",
        tile_m=128,
        tile_n=128,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        pipeline="mem",
        epilogue="default",
        wave_size=32,
        vector_size_a=4,
        vector_size_b=4,
        vector_size_c=1,
    )
    base.update(kw)
    return KernelIdentity(**base)


def _ll(spec, arch="gfx1201"):
    from kernels.common.conv_implicit_gemm_wgrad import build_implicit_gemm_conv_wgrad
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    return _lower_kernel_to_llvm_python(
        build_implicit_gemm_conv_wgrad(spec, arch=arch), arch=arch
    )


def _ir(spec, arch="gfx1201"):
    from kernels.common.conv_implicit_gemm_wgrad import build_implicit_gemm_conv_wgrad
    from rocke.core.ir_print import print_ir

    return print_ir(build_implicit_gemm_conv_wgrad(spec, arch=arch))


class TestWgradRegTransposeGate(unittest.TestCase):
    def test_default_is_off_and_untagged(self):
        self.assertNotIn("regt", _wgrad(reg_transpose=False).kernel_name())
        self.assertIn("_regt", _wgrad().kernel_name())

    def test_arch_gate(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        for arch in ("gfx1201", "gfx1200", "gfx12-generic"):
            with self.subTest(arch=arch):
                self.assertEqual(is_valid_wgrad_spec(_wgrad(), arch), (True, "ok"))
        ok, why = is_valid_wgrad_spec(
            _wgrad(wave_size=64, warp_tile_m=32, warp_tile_n=32), "gfx950"
        )
        self.assertFalse(ok)
        self.assertIn("WMMA", why)

    def test_conflicts(self):
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        for kw in (
            dict(lds_k_outer=True, warp_tile_k=32, tile_k=32),
            dict(data=ConvDataSpec(dtype_a="fp32", dtype_b="fp32", dtype_d="fp32")),
            dict(prefetch_addr=True),
            dict(prefetch_global_regs=True),
            # An exact width the 1024-thread tile cannot be split into.
            dict(
                tile_m=256,
                tile_n=192,
                tile_k=16,
                warp_m=8,
                warp_n=4,
                vector_size_a=8,
                vector_size_b=8,
            ),
        ):
            with self.subTest(kw=sorted(kw)):
                with self.assertRaises(ValueError):
                    _wgrad(**kw).validate()

    def test_lowered_body_stores_wide(self):
        on, off = _ll(_wgrad()), _ll(_wgrad(reg_transpose=False))
        lds16 = r"store half [^\n]*addrspace\(3\)"
        self.assertTrue(re.search(lds16, off))
        self.assertFalse(re.search(lds16, on))
        # 4 rows x 4 dwords per operand: one v_perm_b32 per output dword.
        self.assertEqual(len(re.findall(r"call i32 @llvm\.amdgcn\.perm\(", on)), 32)


class TestRegTransposeIdentity(unittest.TestCase):
    def test_label_round_trip_and_hash(self):
        from kernels.conv.kernel_identity import KernelIdentity

        base = _ident()
        regt = replace(base, reg_transpose=True)
        self.assertNotIn("regt", base.label())
        self.assertIn("-regt-", regt.label())
        self.assertEqual(KernelIdentity.from_label(regt.label()), regt)
        self.assertNotEqual(base.stable_hash(), regt.stable_hash())


class TestRegtPrefetch(unittest.TestCase):
    def test_names_and_gates(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        self.assertIn("_rtp3", _wgrad(regt_prefetch=3).kernel_name())
        self.assertNotIn("rtp", _wgrad().kernel_name())
        with self.assertRaises(ValueError):
            _wgrad(reg_transpose=False, regt_prefetch=3).validate()
        # Only mode 3 is built.
        for mode in (1, 2, 4, 5):
            with self.assertRaises(ValueError):
                _wgrad(regt_prefetch=mode).validate()
        for kw in (dict(unroll_k=True), dict(register_prefetch=True)):
            with self.assertRaises(ValueError):
                _wgrad(regt_prefetch=3, **kw).validate()
        for arch in ("gfx1201", "gfx1200", "gfx12-generic"):
            self.assertEqual(
                is_valid_wgrad_spec(_wgrad(regt_prefetch=3), arch), (True, "ok")
            )
        # Measured (and its LDS-only barrier written) for gfx12 RDNA only.
        for arch in ("gfx1250", "gfx11-generic"):
            ok, why = is_valid_wgrad_spec(_wgrad(regt_prefetch=3), arch)
            self.assertFalse(ok)
            self.assertIn("regt_prefetch", why)

    def test_identity_label(self):
        from kernels.conv.kernel_identity import KernelIdentity

        base = _ident(reg_transpose=True)
        rtp = replace(base, regt_prefetch=3)
        self.assertNotIn("rtp", base.label())
        self.assertIn("-rtp3-regt-", rtp.label())
        self.assertEqual(KernelIdentity.from_label(rtp.label()), rtp)
        self.assertNotEqual(base.stable_hash(), rtp.stable_hash())

    def test_mode_3_keeps_the_prefetch_in_flight(self):
        def ll(mode):
            return _ll(_wgrad(regt_prefetch=mode, warp_m=4, warp_n=2))

        intr = r"call void @llvm\.amdgcn\.s\.barrier\(\)"
        plain, three = ll(0), ll(3)
        self.assertIn("llvm.amdgcn.sched.barrier", three)
        self.assertNotIn("llvm.amdgcn.sched.barrier", plain)
        # The in-loop barriers are LDS-only -- the full drain is asked for in
        # front of the prologue's barrier only.
        full = "call void @llvm.amdgcn.s.waitcnt(i32 7)"
        self.assertEqual(three.count(full), 1)
        # ... as the split-barrier inline asm (lds_barrier_keep_vmem): one
        # intrinsic barrier left (the prologue's), two asm ones in the loop.
        self.assertEqual(len(re.findall(intr, three)), 1)
        self.assertEqual(len(re.findall(r"s_barrier_signal -1", three)), 2)
        # The loads of the plain loop, plus the prologue's.
        loads = r"call <\d+ x i32> @llvm\.amdgcn\.raw\.ptr\.buffer\.load"
        self.assertEqual(
            len(re.findall(loads, three)), 2 * len(re.findall(loads, plain))
        )


class TestRegtAddrStep(unittest.TestCase):
    def test_tag_and_gates(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec
        from kernels.conv.kernel_identity import KernelIdentity

        self.assertIn("_ras", _wgrad(regt_addr_step=True).kernel_name())
        self.assertEqual(
            is_valid_wgrad_spec(
                _wgrad(regt_addr_step=True, regt_prefetch=3), "gfx1201"
            ),
            (True, "ok"),
        )
        for kw in (
            dict(reg_transpose=False, regt_addr_step=True),
            dict(regt_addr_step=True, streamk="dp_sk"),
        ):
            with self.assertRaises(ValueError):
                _wgrad(**kw).validate()
        ras = _ident(reg_transpose=True, regt_prefetch=3, regt_addr_step=True)
        self.assertIn("-rtp3-regt-ras-", ras.label())
        self.assertEqual(KernelIdentity.from_label(ras.label()), ras)
        self.assertNotEqual(
            ras.stable_hash(), replace(ras, regt_addr_step=False).stable_hash()
        )

    def test_same_loads_fewer_decodes_on_the_fast_path(self):
        off = _ir(_wgrad())
        on = _ir(_wgrad(regt_addr_step=True))
        loads = r"buffer_load"
        # Both branches carry their loads' offsets; the loads themselves are
        # issued once, after the branch.
        self.assertEqual(len(re.findall(loads, on)), len(re.findall(loads, off)))
        self.assertIn("scf.if_else", on)
        # And under the mode-3 prefetch loop.
        self.assertIn("scf.if_else", _ir(_wgrad(regt_addr_step=True, regt_prefetch=3)))


class TestWmmaAtomicPair(unittest.TestCase):
    def test_tag_and_gates(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        self.assertIn("_apr", _wgrad(split_k=2, wmma_atomic_pair=True).kernel_name())
        self.assertNotIn("_apr", _wgrad(split_k=2).kernel_name())
        self.assertEqual(
            is_valid_wgrad_spec(_wgrad(split_k=2, wmma_atomic_pair=True), "gfx1201"),
            (True, "ok"),
        )
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        for kw in (
            dict(wmma_atomic_pair=True),  # unsplit
            dict(wmma_atomic_pair=True, split_k=2, two_stage=True),
            dict(
                wmma_atomic_pair=True,
                split_k=2,
                data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp32"),
            ),
        ):
            with self.assertRaises(ValueError):
                _wgrad(**kw).validate()

    def test_half_the_atomics_one_dpp_each(self):
        off = _ir(_wgrad(split_k=2))
        on = _ir(_wgrad(split_k=2, wmma_atomic_pair=True))
        atom = r"global_atomic_add_pk_f16"
        n_off, n_on = len(re.findall(atom, off)), len(re.findall(atom, on))
        self.assertGreater(n_off, 0)
        self.assertEqual(2 * n_on, n_off)
        self.assertEqual(len(re.findall(r"tile\.dpp_xor", on)), n_on)

    def test_identity_label(self):
        from kernels.conv.kernel_identity import KernelIdentity

        lab = (
            "wgrad-implicit_gemm-fp16-2d-t64x64x128-w1x4-a16x16x16-v8x8x1-mem-"
            "default-grp-sk2-rtp3-regt-apr-wave32-gfx1201-llvm24"
        )
        ident = KernelIdentity.from_label(lab)
        self.assertTrue(ident.wmma_atomic_pair)
        self.assertEqual(ident.label(), lab)
        self.assertNotEqual(
            ident.stable_hash(), replace(ident, wmma_atomic_pair=False).stable_hash()
        )

    def test_identity_supports_problem_refuses_odd_wg_n(self):
        """Split-K wgrad identity refuses a grouped fp16 problem with an odd dW row.

        The packed <2 x fp16> atomic needs an even dW row length wg_N = Y*X*cpg;
        an odd wg_N makes the pair misaddress silently. The
        ``identity_supports_problem`` gate must reject this case before the
        binary is offered to the launcher.
        """
        from kernels.common._conv_implicit_gemm_common import ConvProblem
        from kernels.conv.kernel_identity import (
            KernelIdentity,
            identity_supports_problem,
        )

        # Three groups, C=3*3=9 -> cpg=3, K=3*8=24 -> kpg=8 (divisible by
        # vector_size_a=8), Y=X=1 -> wg_N = 1*1*3 = 3 (odd).
        # vector_size_b=1 is compatible with any cpg.
        odd_wg_n = ConvProblem(N=1, Hi=4, Wi=4, C=9, K=24, Y=1, X=1, groups=3)
        # label: v8x1x1 = vector_size_a=8, vector_size_b=1, vector_size_c=1
        lab = (
            "wgrad-implicit_gemm-fp16-2d-t64x64x32-w1x4-a16x16x16-v8x1x1-mem-"
            "default-grp-sk2-wave32-gfx1201-llvm24"
        )
        ident = KernelIdentity.from_label(lab)
        self.assertEqual(ident.split_k, 2)
        ok, why = identity_supports_problem(ident, odd_wg_n)
        self.assertFalse(ok, f"expected refusal, got ok (why={why!r})")
        self.assertIn("even dW row", why)

        # Verify that an even-wg_N grouped problem is accepted.
        # C=12, groups=3 -> cpg=4, K=24 -> kpg=8; wg_N = 1*1*4 = 4 (even).
        even_wg_n = ConvProblem(N=1, Hi=4, Wi=4, C=12, K=24, Y=1, X=1, groups=3)
        ok_even, _ = identity_supports_problem(ident, even_wg_n)
        self.assertTrue(ok_even, "even-wg_N grouped split-K should be accepted")


class TestXRun(unittest.TestCase):
    def _spec(self, **kw):
        from kernels.common._conv_implicit_gemm_common import ConvProblem

        base = dict(
            problem=ConvProblem(N=2, Hi=64, Wi=64, C=3, K=64, Y=16, X=16, sH=16, sW=16),
            reg_transpose=False,
            tile_k=16,
            vector_size_a=8,
            vector_size_b=8,
            x_run=True,
        )
        base.update(kw)
        return _wgrad(**base)

    def test_run_tag_and_problem_gates(self):
        from kernels.common.conv_implicit_gemm_wgrad import (
            is_valid_wgrad_spec,
            wgrad_x_run_problem_reason,
        )
        from kernels.common._conv_implicit_gemm_common import ConvProblem

        sp = self._spec()
        self.assertEqual(sp.b_run, 48)
        self.assertEqual(self._spec(x_run=False).b_run, 3)
        self.assertIn("_xrun", sp.kernel_name())
        self.assertEqual(is_valid_wgrad_spec(sp, "gfx1201"), (True, "ok"))
        # without x_run a C=3 problem cannot take an 8-wide X vector
        self.assertFalse(is_valid_wgrad_spec(self._spec(x_run=False), "gfx1201")[0])
        for bad in (
            dict(pW=1),
            dict(dW=2, Wi=80),
            dict(C=6, groups=2, K=64),
        ):
            p = ConvProblem(
                **{
                    **dict(N=2, Hi=64, Wi=64, C=3, K=64, Y=16, X=16, sH=16, sW=16),
                    **bad,
                }
            )
            self.assertIsNotNone(wgrad_x_run_problem_reason(p))
        with self.assertRaises(ValueError):
            self._spec(lds_k_outer=True).validate()

    def test_wider_x_loads(self):
        on = _ll(self._spec())
        off = _ll(self._spec(x_run=False, vector_size_b=1))
        loads = r"call <\d+ x i32> @llvm\.amdgcn\.raw\.ptr\.buffer\.load"
        self.assertGreater(len(re.findall(loads, on)), len(re.findall(loads, off)))

    def test_identity_supports_only_eligible_problems(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem
        from kernels.conv.kernel_identity import (
            KernelIdentity,
            identity_supports_problem,
        )

        lab = (
            "wgrad-implicit_gemm-bf16-2d-t144x128x16-w1x8-a16x16x16-v8x8x1-mem-"
            "default-xrun-wave32-gfx1201-llvm24"
        )
        ident = KernelIdentity.from_label(lab)
        self.assertTrue(ident.x_run)
        self.assertEqual(ident.label(), lab)
        ok = ConvProblem(N=2, Hi=64, Wi=64, C=3, K=64, Y=16, X=16, sH=16, sW=16)
        kw = dict(llvm_flavor="llvm24")
        self.assertTrue(identity_supports_problem(ident, ok, **kw)[0])
        padded = ConvProblem(
            N=2, Hi=64, Wi=64, C=3, K=64, Y=16, X=16, sH=16, sW=16, pW=1
        )
        self.assertFalse(identity_supports_problem(ident, padded, **kw)[0])
        odd = ConvProblem(N=2, Hi=64, Wi=64, C=3, K=64, Y=3, X=3, sH=1, sW=1)
        self.assertFalse(identity_supports_problem(ident, odd, **kw)[0])  # X*C = 9


class TestWgradRdna4Grid(unittest.TestCase):
    def _jobs(self, arch, n=40, geos=None):
        from kernels.conv.aot_jobs import _geometry_list, _wgrad_jobs
        from rocke.core.arch import ArchTarget

        target = ArchTarget.from_gfx(arch)
        geo = geos or _geometry_list(target, "fp16", "wgrad")[:n]
        family = "wmma" if target.wave_size == 32 else "mma"
        return list(
            _wgrad_jobs(
                arch, "fp16", target.wave_size, family, target, (1, 2), geometries=geo
            )
        )

    @staticmethod
    def _geos(arch, geometry):
        from kernels.conv.aot_jobs import _geometry_list
        from rocke.core.arch import ArchTarget

        geos = _geometry_list(ArchTarget.from_gfx(arch), "fp16", "wgrad")
        return [g for g in geos if tuple(g[:6]) == geometry]

    def test_regt_jobs_on_gfx12_rdna_only(self):
        for arch, want in (
            ("gfx1201", True),
            ("gfx1200", True),
            ("gfx12-generic", True),
            ("gfx1250", False),
            ("gfx11-generic", False),
            ("gfx950", False),
        ):
            with self.subTest(arch=arch):
                geos = self._geos(arch, (64, 64, 128, 2, 2, 16)) or None
                jobs = self._jobs(arch, geos=geos)
                got = any(j.identity.reg_transpose for j in jobs)
                self.assertEqual(got, want)

    def test_regt_jobs_are_listed(self):
        from kernels.conv.rdna4_wgrad_regt_kernels import CACHE_WGRAD_REGT_KERNELS

        geometry = (64, 64, 128, 2, 2, 16)
        ids = [
            j.identity
            for j in self._jobs("gfx1201", geos=self._geos("gfx1201", geometry))
            if j.identity.reg_transpose
        ]
        self.assertTrue(ids)
        for i in ids:
            v = (
                i.vector_size_a,
                i.vector_size_b,
                i.split_k,
                i.regt_prefetch,
                i.two_stage,
                i.regt_addr_step,
                i.wmma_atomic_pair,
            )
            self.assertIn(geometry + v, CACHE_WGRAD_REGT_KERNELS, i.label())
        # Width 2 is in the grid but not in the list.
        self.assertFalse(any(i.vector_size_a == 2 for i in ids))

    def test_atomic_split_k_is_regt_only(self):
        for j in self._jobs("gfx1201"):
            i = j.identity
            if i.split_k > 1 and not i.two_stage:
                self.assertTrue(i.reg_transpose, i.label())

    def test_twin_helpers(self):
        from kernels.conv.aot_jobs import (
            _regt_prefetch_modes,
            _wgrad_atomic_pairs,
            _wgrad_regt_addr_steps,
        )

        self.assertEqual(
            _wgrad_atomic_pairs("gfx1201", 2, False, "fp16"), (False, True)
        )
        self.assertEqual(
            _wgrad_atomic_pairs("gfx1200", 2, False, "bf16"), (False, True)
        )
        self.assertEqual(_wgrad_atomic_pairs("gfx1201", 2, True, "fp16"), (False,))
        self.assertEqual(_wgrad_atomic_pairs("gfx1201", 1, False, "bf16"), (False,))
        self.assertEqual(_wgrad_atomic_pairs("gfx1201", 2, False, "fp32"), (False,))
        self.assertEqual(_wgrad_atomic_pairs("gfx1250", 2, False, "fp16"), (False,))
        # Mode 3 on the listed tiles only.
        self.assertEqual(_regt_prefetch_modes((64, 64, 64, 2, 4, 16)), (0,))
        self.assertEqual(_regt_prefetch_modes((64, 64, 128, 2, 2, 16)), (0, 3))
        self.assertEqual(_regt_prefetch_modes((64, 64, 16, 2, 2, 16)), (0,))
        self.assertEqual(_wgrad_regt_addr_steps("gfx1201", 64, 64), (True, False))
        self.assertEqual(_wgrad_regt_addr_steps("gfx1201", 256, 64), (True,))

    def test_x_run_twins_are_ungrouped(self):
        geos = self._geos("gfx1201", (64, 64, 128, 2, 2, 16))
        xr = [j for j in self._jobs("gfx1201", geos=geos) if j.identity.x_run]
        self.assertTrue(xr)
        for j in xr:
            self.assertFalse(j.identity.grouped)
            self.assertEqual(j.caps, dict(grouped=False, xrun=True))
            self.assertTrue(j.identity.regt_prefetch)

    def test_two_stage_rdna4_cover(self):
        from kernels.conv.aot_jobs import (
            CACHE_WGRAD_WMMA_GEOMETRIES,
            CACHE_WGRAD_WMMA_TWO_STAGE_RDNA4_GEOMETRIES,
            _geometry_list,
        )
        from rocke.core.arch import ArchTarget

        self.assertFalse(
            set(CACHE_WGRAD_WMMA_TWO_STAGE_RDNA4_GEOMETRIES)
            & set(CACHE_WGRAD_WMMA_GEOMETRIES)
        )
        for arch, want in (("gfx1201", True), ("gfx1250", False)):
            geos = [
                g
                for g in _geometry_list(ArchTarget.from_gfx(arch), "fp16", "wgrad")
                if tuple(g[:5]) in ((32, 16, 32, 2, 1), (128, 256, 64, 2, 8))
                and g[7] == "default"
            ]
            ids = [j.identity for j in self._jobs(arch, geos=geos)]
            with self.subTest(arch=arch):
                self.assertEqual(
                    any(i.two_stage and i.group_merge == 1 for i in ids), want
                )
                # The cshuffle epilogue keeps the gfx1250 cover.
                self.assertFalse(any(i.epilogue == "cshuffle" for i in ids))
        # Every RDNA4 cover geometry is a geometry of the gfx1201 grid.
        grid = {
            tuple(g[:5])
            for g in _geometry_list(ArchTarget.from_gfx("gfx1201"), "fp16", "wgrad")
        }
        self.assertLessEqual(set(CACHE_WGRAD_WMMA_TWO_STAGE_RDNA4_GEOMETRIES), grid)

    def test_tile_k_128_axis_is_wgrad_regt_only(self):
        from kernels.conv.aot_jobs import (
            CACHE_RDNA4_EXTRA_TILE_K_DIRECTIONS,
            CACHE_WGRAD_DEEPK_GEOMETRIES,
            _geometry_list,
            _wgrad_regt_listed_geometries,
        )
        from rocke.core.arch import ArchTarget

        listed = _wgrad_regt_listed_geometries()

        def at_128(target, direction):
            # WMMA wgrad's own deep-K geometries (one is tile_k 128, cshuffle)
            # are a separate list, not this axis.
            return [
                g
                for g in _geometry_list(target, "fp16", direction)
                if g[2] == 128 and tuple(g[:5]) not in CACHE_WGRAD_DEEPK_GEOMETRIES
            ]

        for arch in ("gfx1201", "gfx1200", "gfx12-generic"):
            target = ArchTarget.from_gfx(arch)
            for direction in ("fwd", "dgrad"):
                with self.subTest(arch=arch, direction=direction):
                    # Forward follows the tuple (it is ("fwd", "wgrad") once
                    # the forward PR is in); dgrad is never on the axis, so a
                    # tuple that gains "dgrad" fails here.
                    if (
                        direction == "fwd"
                        and direction in CACHE_RDNA4_EXTRA_TILE_K_DIRECTIONS
                    ):
                        self.assertTrue(at_128(target, direction))
                    else:
                        self.assertFalse(at_128(target, direction))
            geos = at_128(target, "wgrad")
            with self.subTest(arch=arch, direction="wgrad"):
                self.assertTrue(geos)
                # Only geometries of a listed register-transpose kernel.
                for g in geos:
                    self.assertIn(tuple(g[:6]), listed)
                    self.assertEqual(g[7], "default")
                # The jobs of those geometries are all register-transpose.
                jobs = self._jobs(arch, geos=geos[:60])
                self.assertTrue(jobs)
                self.assertTrue(all(j.identity.reg_transpose for j in jobs))
        # WMMA wgrad's deep-K geometry at tile_k 128 (cshuffle) is not on the
        # register-transpose-only axis: it still builds non-regt kernels.
        deep = [
            g
            for g in _geometry_list(ArchTarget.from_gfx("gfx1201"), "fp16", "wgrad")
            if tuple(g[:5]) in CACHE_WGRAD_DEEPK_GEOMETRIES and g[2] == 128
        ]
        with self.subTest(deep_k=True):
            self.assertTrue(deep)
            self.assertTrue(
                [
                    j
                    for j in self._jobs("gfx1201", geos=deep)
                    if not j.identity.reg_transpose
                ]
            )
        # The filter is the wgrad axis's: were forward walked on the axis too
        # (a combined ("fwd", "wgrad") tuple), its tile_k 128 geometries would
        # stay unfiltered.
        from unittest import mock

        import kernels.conv.aot_jobs as aot_jobs

        with mock.patch.object(
            aot_jobs, "CACHE_RDNA4_EXTRA_TILE_K_DIRECTIONS", ("fwd", "wgrad")
        ):
            for arch in ("gfx1201", "gfx1200", "gfx12-generic"):
                target = ArchTarget.from_gfx(arch)
                fwd = [g for g in _geometry_list(target, "fp16", "fwd") if g[2] == 128]
                wgrad = [
                    g for g in _geometry_list(target, "fp16", "wgrad") if g[2] == 128
                ]
                with self.subTest(arch=arch, combined=True):
                    self.assertTrue(fwd)
                    self.assertTrue(
                        [g for g in fwd if tuple(g[:6]) not in listed],
                        "forward must not inherit the wgrad register-transpose filter",
                    )
                    for g in wgrad:
                        self.assertIn(tuple(g[:6]), listed)
        # Every other arch keeps its geometry lists.
        for arch in ("gfx1250", "gfx950"):
            target = ArchTarget.from_gfx(arch)
            for direction in ("fwd", "dgrad", "wgrad"):
                with self.subTest(arch=arch, direction=direction):
                    self.assertFalse(at_128(target, direction))

    def test_count_jobs_matches_the_walk(self):
        from kernels.conv.aot_jobs import (
            CACHE_DGRAD_MAX_SUB_GEMMS,
            CACHE_WGRAD_SPLIT_KS,
            _direction_jobs,
            count_jobs,
        )
        from rocke.core.arch import ArchTarget

        for arch in ("gfx1201", "gfx1200", "gfx1250"):
            target = ArchTarget.from_gfx(arch)
            n = sum(
                1
                for _ in _direction_jobs(
                    "wgrad",
                    arch,
                    "bf16",
                    target,
                    CACHE_WGRAD_SPLIT_KS,
                    CACHE_DGRAD_MAX_SUB_GEMMS,
                )
            )
            c = count_jobs(dtype="bf16", target=target, directions=("wgrad",))
            self.assertEqual(n, c, arch)


class TestXRunAndRasRefusals(unittest.TestCase):
    """x_run combination refusals and ras (regt_addr_step) 3-D refusal."""

    def _xrun_spec(self, **kw):
        from kernels.common._conv_implicit_gemm_common import ConvProblem

        base = dict(
            problem=ConvProblem(N=2, Hi=64, Wi=64, C=3, K=64, Y=16, X=16, sH=16, sW=16),
            reg_transpose=False,
            tile_k=16,
            vector_size_a=8,
            vector_size_b=8,
            x_run=True,
        )
        base.update(kw)
        return _wgrad(**base)

    def _assert_only_x_run_refuses(self, **kw):
        """The spec validates without x_run and is refused for x_run alone."""
        self._xrun_spec(x_run=False, **kw).validate()
        with self.assertRaisesRegex(ValueError, "x_run widens the sync X loader"):
            self._xrun_spec(x_run=True, **kw).validate()

    def test_x_run_refused_with_async_dma(self):
        # async_dma rides on lds_k_outer, which needs the deeper 32-wide atom.
        self._assert_only_x_run_refuses(
            async_dma=True, lds_k_outer=True, tile_k=32, warp_tile_k=32
        )

    def test_x_run_refused_with_lds_k_outer(self):
        self._assert_only_x_run_refuses(lds_k_outer=True, tile_k=32, warp_tile_k=32)

    def test_x_run_refused_with_group_merge(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem

        # group_merge is depthwise-only (and divides the groups); wave64.
        dw = ConvProblem(N=2, Hi=64, Wi=64, C=8, K=8, Y=3, X=3, pH=1, pW=1, groups=8)
        self._assert_only_x_run_refuses(
            problem=dw, group_merge=2, wave_size=64, tile_k=32, warp_tile_k=32
        )

    def test_x_run_refused_with_stream_k(self):
        # dp_sk is a stream-K mode ("full" is not), wave64.
        self._assert_only_x_run_refuses(
            streamk="dp_sk", wave_size=64, tile_k=32, warp_tile_k=32
        )

    def test_x_run_refused_with_3d(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem

        p3d = ConvProblem(
            N=2,
            Di=8,
            Hi=64,
            Wi=64,
            C=3,
            K=64,
            Z=3,
            Y=3,
            X=3,
            sD=1,
            sH=1,
            sW=1,
            pD=1,
            pH=1,
            pW=1,
            dD=1,
        )
        with self.assertRaisesRegex(ValueError, "x_run widens the sync X loader"):
            self._xrun_spec(problem=p3d).validate()

    def test_ras_refused_with_3d(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        p3d = ConvProblem(
            N=2,
            Di=8,
            Hi=14,
            Wi=14,
            C=64,
            K=64,
            Z=3,
            Y=3,
            X=3,
            sD=1,
            sH=1,
            sW=1,
            pD=1,
            pH=1,
            pW=1,
            dD=1,
        )
        spec = _wgrad(problem=p3d, regt_addr_step=True)
        ok, why = is_valid_wgrad_spec(spec, "gfx1201")
        self.assertFalse(ok)
        self.assertIn("3-D", why)


if __name__ == "__main__":
    unittest.main()
