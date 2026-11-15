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
        # Prologue + two per tile.
        self.assertEqual(len(re.findall(intr, three)), 3)
        self.assertIn("llvm.amdgcn.sched.barrier", three)
        self.assertNotIn("llvm.amdgcn.sched.barrier", plain)
        # The in-loop barriers are LDS-only -- the full drain is asked for in
        # front of the prologue's barrier only.
        full = "call void @llvm.amdgcn.s.waitcnt(i32 7)"
        self.assertEqual(three.count(full), 1)
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


if __name__ == "__main__":
    unittest.main()
