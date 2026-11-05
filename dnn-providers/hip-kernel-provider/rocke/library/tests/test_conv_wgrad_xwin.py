# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side gate for wgrad ``xwin`` (X-window reuse across the x taps).

No GPU: the window stride and its bank behaviour, spec gates, the name and
label tags, the identity capability rule, and lowering (the window tile and
the alignbyte-built odd taps).
"""

from __future__ import annotations

import unittest
from dataclasses import replace


def _problem(**kw):
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    base = dict(N=1, Hi=18, Wi=34, C=64, K=64, Y=3, X=3, pH=0, pW=0)
    base.update(kw)
    return ConvProblem(**base)


def _spec(**kw):
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec
    from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

    base = dict(
        problem=_problem(),
        data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
        tile_m=64,
        tile_n=192,
        tile_k=32,
        warp_m=1,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        wave_size=64,
        pipeline="mem",
        epilogue="default",
        lds_k_pad=4,
        prefetch_regs=1,
        lds_kpack=2,
        split_k=2,
        two_stage=True,
        vector_size_a=8,
        vector_size_b=8,
        vector_size_c=1,
    )
    base.update(kw)
    return WgradConvSpec(**base)


def _ident(**kw):
    from kernels.conv.kernel_identity import KernelIdentity

    base = dict(
        arch="gfx942",
        direction="wgrad",
        algorithm="implicit_gemm",
        dtype_a="bf16",
        dtype_b="bf16",
        dtype_d="bf16",
        tile_m=64,
        tile_n=192,
        tile_k=32,
        warp_m=1,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        pipeline="mem",
        epilogue="default",
        wave_size=64,
        vector_size_a=8,
        vector_size_b=8,
        vector_size_c=1,
        split_k=2,
        two_stage=True,
        xwin=3,
    )
    base.update(kw)
    return KernelIdentity(**base)


class TestWgradXwin(unittest.TestCase):
    def test_window_stride(self):
        from kernels.common.conv_implicit_gemm_wgrad import xwin_window_stride

        self.assertEqual(xwin_window_stride(16, 3), 20)
        self.assertEqual(xwin_window_stride(32, 3), 36)
        self.assertEqual(xwin_window_stride(64, 3), 68)
        for tk in (16, 32, 64):
            s = xwin_window_stride(tk, 3)
            # 16 channel rows of one fragment read land on 16 distinct even
            # banks: each lane's 8-byte read takes its own bank pair.
            banks = {(c * s // 2) % 32 for c in range(16)}
            self.assertEqual(len(banks), 16)
            self.assertTrue(all(b % 2 == 0 for b in banks))

    def test_default_is_off_and_untagged(self):
        spec = _spec()
        self.assertEqual((spec.xwin, spec.xwin_rows), (0, 1))
        self.assertNotIn("_xw", spec.kernel_name())
        self.assertIn("_xw3", replace(spec, xwin=3).kernel_name())
        self.assertIn(
            "_xwr3",
            replace(spec, xwin=3, xwin_rows=3, tile_n=576, warp_n=6).kernel_name(),
        )

    def test_gates(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        for good in (
            dict(xwin=3),
            dict(xwin=3, prefetch_regs=2),
            dict(xwin=3, prefetch_regs=3),
            dict(xwin=3, warp_tile_m=32, warp_tile_n=32, warp_tile_k=8),
            dict(xwin=3, xwin_rows=3, tile_n=576, warp_n=6),
        ):
            ok, why = is_valid_wgrad_spec(_spec(**good), arch="gfx942")
            self.assertTrue(ok, (good, why))
        for bad in (
            dict(xwin=3, prefetch_regs=0, lds_kpack=1),
            dict(xwin=3, two_stage=False),
            dict(xwin=3, sep_addr=2),
            dict(xwin=3, tile_n=256),
            dict(xwin=3, problem=_problem(X=1, Y=1)),
            dict(xwin=3, problem=_problem(sW=2)),
            dict(xwin=3, xwin_rows=3),  # 3 rows over 2 waves
            dict(xwin_rows=3),
        ):
            ok, _ = is_valid_wgrad_spec(_spec(**bad), arch="gfx942")
            self.assertFalse(ok, bad)

    def test_capability(self):
        from kernels.conv.kernel_identity import identity_supports_problem

        ident = _ident()
        ok, why = identity_supports_problem(ident, _problem())
        self.assertTrue(ok, why)
        self.assertTrue(identity_supports_problem(ident, _problem(C=128))[0])
        for bad in (
            _problem(X=5, Wi=36),  # filter width
            _problem(sW=2, Wi=65),  # W stride
            _problem(C=96),  # channel blocks of 64
            _problem(Wi=30),  # Wo = 28: not whole K tiles
            _problem(C=128, K=128, groups=2),
        ):
            self.assertFalse(identity_supports_problem(ident, bad)[0], bad)
        r3 = replace(ident, tile_n=576, xwin_rows=3)
        self.assertTrue(identity_supports_problem(r3, _problem())[0])
        self.assertFalse(identity_supports_problem(r3, _problem(Y=2, Hi=17))[0])

    def test_label_and_hash(self):
        from kernels.conv.kernel_identity import KernelIdentity

        ident = _ident(tile_n=576, xwin_rows=3)
        lab = ident.label()
        self.assertIn("-xw3-", lab)
        self.assertIn("-xwr3-", lab)
        self.assertEqual(KernelIdentity.from_label(lab), ident)
        # Off, neither field reaches the hash.
        off = _ident(xwin=0)
        self.assertNotIn("xwin", off.label())

    def test_lowers_with_window_and_alignbyte(self):
        from kernels.common.conv_implicit_gemm_wgrad import build_implicit_gemm_conv_wgrad
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        ll = lower_kernel_to_llvm(
            build_implicit_gemm_conv_wgrad(_spec(xwin=3), arch="gfx942"), arch="gfx942"
        )
        # A 64 x (32 + 4) and the window (64 + 8 trash rows) x 36, bf16.
        self.assertIn(f"[{(64 * 36 + 72 * 36) * 2} x i8]", ll)
        self.assertIn("alignbyte", ll)


class TestWgradXwinRound4(unittest.TestCase):
    """xwin_groups / xwin_pp (block ping-pong) / xwin_mrow and the scratch
    slab count of the cached two-stage launcher."""

    def test_defaults_untagged(self):
        spec = _spec(xwin=3)
        self.assertEqual((spec.xwin_groups, spec.xwin_pp, spec.xwin_mrow), (1, 0, 1))
        name = spec.kernel_name()
        for tag in ("_xg", "_xpp", "_xmr"):
            self.assertNotIn(tag, name)
        self.assertEqual(spec.block_size, spec.group_block_size)

    def test_groups_and_pp_gates(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        pp = _spec(xwin=3, xwin_groups=4, xwin_pp=8, waves_per_eu=2)
        ok, why = is_valid_wgrad_spec(pp, arch="gfx942")
        self.assertTrue(ok, why)
        for gn in (1, 2):
            ok, why = is_valid_wgrad_spec(replace(pp, xwin_gn=gn), arch="gfx942")
            self.assertTrue(ok, (gn, why))
        self.assertEqual(pp.block_size, 512)
        self.assertIn("_xg4_xpp8", pp.kernel_name())
        for bad in (
            dict(xwin=3, xwin_pp=8),  # 2 waves, not 8
            dict(xwin=3, xwin_groups=3, xwin_pp=8),  # odd group count
            dict(xwin=3, xwin_groups=4, xwin_pp=99),
            dict(xwin_groups=4),  # needs xwin
            dict(xwin=3, xwin_groups=4, xwin_gn=2),  # shared dY needs the ping-pong
            dict(xwin=3, xwin_groups=4, xwin_pp=8, xwin_gn=3),
        ):
            ok, _ = is_valid_wgrad_spec(_spec(**bad), arch="gfx942")
            self.assertFalse(ok, bad)

    def test_mrow_window_and_capability(self):
        from kernels.common.conv_implicit_gemm_wgrad import (
            is_valid_wgrad_spec,
            xwin_window_pitch,
            xwin_window_stride,
        )
        from kernels.conv.kernel_identity import identity_supports_problem

        self.assertEqual(xwin_window_pitch(32, 3, 2), 20)
        self.assertEqual(xwin_window_stride(32, 3, 2), 44)
        self.assertEqual(xwin_window_stride(32, 3, 1), xwin_window_stride(32, 3))
        ok, why = is_valid_wgrad_spec(_spec(xwin=3, xwin_mrow=2), arch="gfx942")
        self.assertTrue(ok, why)
        self.assertFalse(is_valid_wgrad_spec(_spec(xwin=3, xwin_mrow=3), arch="gfx942")[0])
        ident = _ident(xwin_mrow=2)
        self.assertIn("-xmr2-", ident.label())
        self.assertTrue(identity_supports_problem(ident, _problem(Wi=18, Hi=18))[0])
        self.assertFalse(identity_supports_problem(ident, _problem(Wi=34))[0])  # Wo 32
        self.assertFalse(identity_supports_problem(ident, _problem(Wi=18, Hi=17))[0])  # Ho 15

    def test_label_round_trip(self):
        from kernels.conv.kernel_identity import KernelIdentity

        ident = _ident(xwin_groups=4, xwin_pp=8, waves_per_eu=2)
        lab = ident.label()
        self.assertIn("-xg4-xpp8-", lab)
        self.assertEqual(KernelIdentity.from_label(lab), ident)
        self.assertNotEqual(ident.stable_hash(), _ident(waves_per_eu=2).stable_hash())

    def test_ws_slabs_used(self):
        from benchmarks.common.benchmark_implicit_gemm_conv import _ws_slabs_used

        ident = _ident(ws_replicas=8)
        p = _problem()
        self.assertEqual(_ws_slabs_used(p, ident, 2), 2)
        self.assertEqual(_ws_slabs_used(p, ident, 760), 8)
        self.assertEqual(_ws_slabs_used(_problem(C=128, K=128, groups=2), ident, 2), 8)

    def test_stage2_vec_fold(self):
        from kernels.common.conv_wgrad_workspace_reduce import (
            WgradReduceSpec,
            build_conv_wgrad_workspace_reduce,
            wgrad_reduce_grid,
        )
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        p = _problem()
        s1 = WgradReduceSpec(problem=p, dtype_d="bf16", ws_replicas=2)
        s4 = WgradReduceSpec(problem=p, dtype_d="bf16", ws_replicas=2, vec=4)
        self.assertNotIn("_v4", s1.kernel_name())
        self.assertIn("_v4_", s4.kernel_name())
        gx1, gy1, _ = wgrad_reduce_grid(s1)
        gx4, gy4, _ = wgrad_reduce_grid(s4)
        self.assertEqual(gy1, gy4)
        self.assertEqual(gx4, -(-s1.wg_N // (s1.tile_n * 4)))
        ll = lower_kernel_to_llvm(build_conv_wgrad_workspace_reduce(s4), arch="gfx942")
        self.assertIn("<4 x float>", ll)

    def test_lowers_pingpong(self):
        from kernels.common.conv_implicit_gemm_wgrad import build_implicit_gemm_conv_wgrad
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        spec = _spec(xwin=3, xwin_groups=4, xwin_pp=8, waves_per_eu=2)
        ll = lower_kernel_to_llvm(build_implicit_gemm_conv_wgrad(spec, arch="gfx942"), arch="gfx942")
        # four groups' A + window tiles
        self.assertIn(f"[{4 * (64 * 36 + 72 * 36) * 2} x i8]", ll)
        self.assertIn("s.setprio", ll)


if __name__ == "__main__":
    unittest.main()
