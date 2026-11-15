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


if __name__ == "__main__":
    unittest.main()
