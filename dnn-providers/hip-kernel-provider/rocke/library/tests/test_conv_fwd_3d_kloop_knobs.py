# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU correctness of the forward implicit-GEMM K-loop knobs on 3-D problems.

Covers the knobs that change the forward K loop or its reduction:

* ``prefetch_regs`` -- register-staged global prefetch, on the single-buffer
  loop and (with ``unroll_k``) the ping-pong loop, including an odd tile count
  (the trailing prefetch reads past K_gemm and must read zero);
* ``split_addr_k_decode`` -- the per-tile (z, y, x) decode (cpg % tile_k == 0);
* ``split_k`` -- the split binary at several launch degrees, atomic-adding
  into a zeroed D, grouped and ungrouped;

each against the fp32 torch reference, on the AOT launch path (ConvArgs).
Also checks the 3-D split addressing every 3-D kernel now takes, with padding
and stride so the bounds checks are exercised.
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _skip_reason() -> str:
    if not _HAS_TORCH:
        return "torch not importable"
    try:
        arch = get_device_arch()
    except Exception as e:  # pragma: no cover - no device
        return f"no GPU: {e}"
    if not arch.startswith(("gfx94", "gfx95")):
        return f"MFMA-only knobs; device is {arch}"
    return ""


_SKIP = _skip_reason()


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _run(problem, *, degree=1, tile=(64, 64, 32, 2, 2, 32), full_tiles=None, **knobs):
    """Build, launch and verify one 3-D fwd kernel; returns the max rel error."""
    import torch

    from builders.common.conv_reference import conv_reference
    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_implicit_gemm import (
        ConvDataSpec,
        ImplicitGemmConvSpec,
        build_implicit_gemm_conv,
        is_valid_spec_for_problem,
    )
    from rocke import compile_kernel
    from rocke.core.arch import ArchTarget
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    arch = get_device_arch().split(":")[0]
    tm, tn, tk, wm, wn, wt = tile
    atom = ArchTarget.from_gfx(arch).mma.select_largest_k(
        family="mma", a_dtype="bf16", b_dtype="bf16", c_dtype="fp32", m=wt, n=wt, k_max=tk
    )
    tag = "".join(f"_{k}{int(v)}" for k, v in sorted(knobs.items()))
    spec = ImplicitGemmConvSpec(
        problem=problem,
        name=f"test_fwd3d_knobs{tag}",
        data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        warp_m=wm,
        warp_n=wn,
        warp_tile_m=wt,
        warp_tile_n=wt,
        warp_tile_k=atom.k,
        pipeline="compv4",
        epilogue="cshuffle",
        groups=problem.groups,
        **knobs,
    )
    ok, why = is_valid_spec_for_problem(spec, problem, arch)
    assert ok, why
    art = compile_kernel(build_implicit_gemm_conv(spec, arch=arch), arch=arch)
    split = spec.split_k > 1
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=conv_args_signature(
            "bf16", is_3d=True, split_k=split, split_tail=spec.split_tail
        ),
    )
    p = problem
    torch.manual_seed(0)
    A = torch.empty(p.N, p.Di, p.Hi, p.Wi, p.C).uniform_(-1, 1).to(torch.bfloat16)
    B = torch.empty(p.K, p.Z, p.Y, p.X, p.cpg).uniform_(-1, 1).to(torch.bfloat16)
    D = torch.empty(p.N, p.Do, p.Ho, p.Wo, p.K, dtype=torch.bfloat16)
    ref = conv_reference(A, B, p, out_dtype=torch.bfloat16)
    rt = Runtime()
    ptrs = [rt.alloc(t.nbytes) for t in (A, B, D)]
    try:
        rt.memcpy_h2d(ptrs[0], _u8(A), A.nbytes)
        rt.memcpy_h2d(ptrs[1], _u8(B), B.nbytes)
        # Garbage in D: the split binary must be launched on a zeroed D, the
        # others must overwrite every element.
        rt.memset(ptrs[2], 0x7F, D.nbytes)
        args = ConvArgs.from_problem(p, tile_m=tm, tile_n=tn, tile_k=tk)
        if split:
            # split_tail: only the tail's rows are zeroed; the full tiles
            # must overwrite the garbage above them.
            off = 0
            if spec.split_tail and full_tiles:
                from kernels.common.conv_implicit_gemm import fwd_split_tail_zero_rows

                row0 = fwd_split_tail_zero_rows(
                    args.gemm_m, tm, args.num_pid_n, full_tiles
                )
                off = (D.nbytes // args.gemm_m) * row0
            if D.nbytes > off:
                rt.memset(ptrs[2] + off, 0, D.nbytes - off)
        values = args.to_launch_values(
            *ptrs, A.nbytes, B.nbytes, D.nbytes, split_k=degree if split else 1,
            full_tiles=full_tiles if spec.split_tail else None,
        )
        gx, gy, gz = args.grid()
        launcher(
            values,
            config=LaunchConfig(
                grid=(gx, gy, gz * (degree if split else 1)),
                block=(spec.block_size, 1, 1),
                fence=True,
            ),
        )
        out = torch.empty_like(D)
        rt.memcpy_d2h(_u8(out), ptrs[2], D.nbytes)
    finally:
        for ptr in ptrs:
            rt.free(ptr)
    o = out.float().to(ref.device)
    return float((o - ref).abs().max() / ref.abs().max().clamp(min=1.0))


def _problem(**kw):
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    base = dict(N=1, Hi=6, Wi=5, C=64, K=64, Y=3, X=3, pH=1, pW=1, Di=5, Z=3, sD=1, pD=1, dD=1)
    base.update(kw)
    return ConvProblem(**base)


_TOL = 5e-2


@unittest.skipIf(_SKIP, _SKIP)
class TestFwd3dKLoopKnobs(unittest.TestCase):
    def _check(self, problem, **kw):
        err = _run(problem, **kw)
        self.assertLess(err, _TOL, f"{kw}: rel err {err:.3e}")

    def test_split_addressing_padded_strided(self):
        # Padding on every axis and a stride: the per-row origin and the
        # per-tap bounds of the 3-D split addressing.
        self._check(_problem())
        self._check(_problem(sH=2, sW=2, sD=2, Hi=9, Wi=7, Di=7))

    def test_prefetch_regs(self):
        # 3*3*3*64 = 1728 = 54 tiles of 32: even count.
        self._check(_problem(), prefetch_regs=True)
        # C=48: 1296 = 40.5 tiles -> a partial last tile.
        self._check(_problem(C=48), prefetch_regs=True)

    def test_prefetch_regs_unroll_odd_tiles(self):
        # C=32: 864 = 27 tiles of 32, odd: the ping-pong's trailing prefetch
        # and MFMA run past K_gemm on zeros.
        self._check(_problem(C=32), prefetch_regs=True, unroll_k=True)
        self._check(_problem(), prefetch_regs=True, unroll_k=True)

    def test_split_addr_k_decode(self):
        self._check(_problem(), split_addr_k_decode=True)
        self._check(_problem(), split_addr_k_decode=True, prefetch_regs=True)

    def test_split_addr_k_decode_rejects_straddling_tiles(self):
        from kernels.common.conv_implicit_gemm import ImplicitGemmConvSpec

        with self.assertRaises(ValueError):
            ImplicitGemmConvSpec(
                problem=_problem(C=48), tile_k=32, split_addr_k_decode=True
            ).validate()

    def test_split_k(self):
        for degree in (2, 3, 5):
            with self.subTest(degree=degree):
                self._check(_problem(), split_k=2, degree=degree)
                self._check(
                    _problem(), split_k=2, degree=degree, prefetch_regs=True,
                    split_addr_k_decode=True,
                )

    def test_split_k_grouped(self):
        self._check(_problem(C=64, K=64, groups=2), split_k=2, degree=3)

    def test_kloop_schedule_knobs_reg_prefetch(self):
        # lds_ahead / defer_mfma / vmem_interleave on the reg_prefetch=1 loop
        # (uniform_k_decode, 64-wide K tiles: 2 kk steps of the 32x32x8 atom
        # would leave defer nothing to keep, so use tile_k 64).
        tile = (64, 64, 64, 2, 2, 32)
        for knobs in (
            dict(lds_ahead=True),
            dict(lds_ahead=True, defer_mfma=1),
            dict(lds_ahead=True, defer_mfma=1, vmem_interleave=2),
            dict(vmem_interleave=4),
            dict(lds_ahead=True, defer_mfma=1, vmem_interleave=4, lds_xor=True),
            dict(reg_prefetch=2, lds_xor=True, lds_ahead=True, vmem_interleave=2),
            dict(lds_ahead=True, defer_mfma=1, rpf_setprio=1),
            dict(lds_ahead=True, defer_mfma=3, vmem_interleave=4, lds_xor=True,
                 lds_kpack_read=True, uk_nopad=False, defer_mfma_per_write=1),
            dict(lds_ahead=True, defer_mfma=4, vmem_interleave=2, lds_xor=True,
                 lds_kpack_read=True, defer_mfma_per_write=3),
            dict(reg_prefetch=2, lds_ahead=True, rpf_setprio=2),
        ):
            with self.subTest(**knobs):
                kw = dict(reg_prefetch=1)
                kw.update(knobs)
                self._check(_problem(), tile=tile, uniform_k_decode=True, **kw)

    def test_kloop_schedule_knobs_prefetch_regs(self):
        # The single-buffer prefetch_regs loop: lds_xor stores + lookahead +
        # interleave, unsplit and split.
        tile = (64, 64, 64, 2, 2, 32)
        for knobs in (
            dict(lds_xor=True),
            dict(lds_xor=True, lds_ahead=True, vmem_interleave=2),
        ):
            with self.subTest(**knobs):
                self._check(_problem(), tile=tile, prefetch_regs=True, **knobs)
                self._check(
                    _problem(), tile=tile, prefetch_regs=True,
                    split_addr_k_decode=True, split_k=2, degree=3, **knobs,
                )

    def test_split_k_uniform_k_decode(self):
        # split_k on the uniform_k_decode + reg_prefetch=1 loop: each slice
        # starts its tap state mid-reduction (both tap orders), and the
        # plain fallback loop (cpg % tile_k != 0) takes the slice range too.
        tile = (64, 64, 64, 2, 2, 32)
        for degree in (2, 3, 5):
            for knobs in (
                dict(),
                dict(uk_tap_fast=True),
                dict(lds_xor=True, lds_ahead=True, defer_mfma=1, vmem_interleave=4),
            ):
                with self.subTest(degree=degree, **knobs):
                    self._check(
                        _problem(), tile=tile, uniform_k_decode=True,
                        reg_prefetch=1, split_k=2, degree=degree, **knobs,
                    )
        self._check(_problem(C=96), tile=tile, uniform_k_decode=True,
                    reg_prefetch=1, split_k=2, degree=3)
        self._check(_problem(C=64, K=64, groups=2), tile=tile,
                    uniform_k_decode=True, reg_prefetch=1, split_k=2, degree=3)

    def test_uk_nopad(self):
        # Unpadded (halo-form) problems without the per-tap bounds checks,
        # incl. an M tail (M = 4*4*3 = 48 < 64) and a stride.
        tile = (64, 64, 64, 2, 2, 32)
        for prob in (
            _problem(Hi=8, Wi=7, Di=7, pH=0, pW=0, pD=0),
            _problem(Hi=6, Wi=5, Di=6, pH=0, pW=0, pD=0),
            _problem(Hi=9, Wi=9, Di=9, pH=0, pW=0, pD=0, sH=2, sW=2, sD=2),
        ):
            for knobs in (
                dict(),
                dict(lds_xor=True, lds_ahead=True, defer_mfma=1, vmem_interleave=4),
                dict(split_k=2, degree=3),
            ):
                with self.subTest(prob=(prob.Hi, prob.Wi, prob.Di), **knobs):
                    self._check(prob, tile=tile, uniform_k_decode=True,
                                reg_prefetch=1, uk_nopad=True, **knobs)

    def test_split_tail(self):
        # split_tail: the first full_tiles tiles (launch order) run whole and
        # store plainly, the rest are split and atomic-added. 6x5x5 output,
        # M=150 -> 3 M tiles x 1 N tile on 64x64 tiles.
        tile = (64, 64, 64, 2, 2, 32)
        prob = _problem(Hi=8, Wi=7, Di=7, pH=0, pW=0, pD=0)
        # 12x10x10 output, K=128: M=1200 -> 19 M tiles x 2 N tiles = 38 tiles.
        big = _problem(Hi=12, Wi=10, Di=10, pH=1, pW=1, pD=1, K=128)
        for full in (0, 12, 24, 36, 38):
            with self.subTest(big=True, full=full):
                self._check(big, tile=tile, split_k=2, degree=3, split_tail=True,
                            full_tiles=full, uniform_k_decode=True, reg_prefetch=1,
                            xcd_remap=6, lds_xor=True, lds_ahead=True,
                            defer_mfma=1, vmem_interleave=4)
        for full in (0, 1, 2, 3):
            for knobs in (
                dict(uniform_k_decode=True, reg_prefetch=1),
                dict(uniform_k_decode=True, reg_prefetch=1, xcd_remap=6,
                     lds_xor=True, lds_ahead=True, defer_mfma=1, vmem_interleave=4),
                dict(prefetch_regs=True),
            ):
                with self.subTest(full=full, **knobs):
                    self._check(prob, tile=tile, split_k=2, degree=3,
                                split_tail=True, full_tiles=full, **knobs)


class TestFwdKLoopScheduleSpec(unittest.TestCase):
    """CPU: where the scheduling knobs are rejected, and the AOT env axis."""

    def _spec(self, **kw):
        from kernels.common.conv_implicit_gemm import ImplicitGemmConvSpec

        base = dict(
            problem=_problem(), tile_m=64, tile_n=64, tile_k=64, warp_m=2,
            warp_n=2, warp_tile_m=32, warp_tile_n=32, warp_tile_k=8,
            pipeline="compv4",
        )
        base.update(kw)
        return ImplicitGemmConvSpec(**base)

    def test_rejections(self):
        from kernels.common.conv_implicit_gemm import _fwd_kloop_conflict

        ok = self._spec(uniform_k_decode=True, reg_prefetch=1, lds_ahead=True,
                        defer_mfma=1, vmem_interleave=2)
        self.assertEqual(_fwd_kloop_conflict(ok), "")
        for kw in (
            dict(lds_ahead=True),  # plain loop
            dict(uniform_k_decode=True, reg_prefetch=1, defer_mfma=8),
            dict(prefetch_regs=True, defer_mfma=1),
            dict(uniform_k_decode=True, reg_prefetch=1, vmem_interleave=2,
                 pipeline="mem"),
            dict(prefetch_regs=True, unroll_k=True, vmem_interleave=2),
            dict(uniform_k_decode=True, reg_prefetch=1, uk_tap_fast=True,
                 uk_chunk_group=2, split_k=2),
        ):
            with self.subTest(**kw):
                self.assertNotEqual(_fwd_kloop_conflict(self._spec(**kw)), "")

    def test_default_name_unchanged(self):
        a = self._spec(uniform_k_decode=True, reg_prefetch=1)
        b = self._spec(uniform_k_decode=True, reg_prefetch=1, lds_ahead=True,
                       defer_mfma=1, vmem_interleave=4)
        self.assertNotIn("lah", a.kernel_name())
        self.assertTrue(b.kernel_name().endswith("_lah_dfr1_vmi4"), b.kernel_name())

    def test_aot_env_axis(self):
        import os
        from unittest import mock

        from kernels.conv import aot_jobs

        with mock.patch.dict(os.environ, {"ROCKE_AOT_FWD_SCHED": "a1d1v2, a1d0v0,a1d1v2"}):
            self.assertEqual(
                aot_jobs._fwd_sched_variants(),
                ({"lds_ahead": True, "defer_mfma": 1, "vmem_interleave": 2},
                 {"lds_ahead": True}),
            )
        with mock.patch.dict(os.environ, {"ROCKE_AOT_FWD_SCHED": "x1"}):
            with self.assertRaises(ValueError):
                aot_jobs._fwd_sched_variants()
        with mock.patch.dict(os.environ, {"ROCKE_AOT_FWD_SCHED": ""}):
            self.assertEqual(aot_jobs._fwd_sched_variants(), ())

    def test_uk_nopad_serves_unpadded_problems_only(self):
        from dataclasses import replace

        from kernels.conv.kernel_identity import KernelIdentity, identity_supports_problem

        ident = KernelIdentity(
            arch="gfx942", direction="fwd", algorithm="implicit_gemm",
            llvm_flavor="llvm22", dtype_a="bf16", dtype_b="bf16", dtype_d="bf16",
            grouped=True, is_3d=True, tile_m=64, tile_n=64, tile_k=64, warp_m=2,
            warp_n=2, warp_tile_m=32, warp_tile_n=32, warp_tile_k=8,
            pipeline="compv4", epilogue="cshuffle", uniform_k_decode=True,
            reg_prefetch=1, uk_nopad=True, wave_size=64, vector_size_a=8,
            vector_size_b=8, vector_size_c=8,
        )
        unpadded = _problem(Hi=8, Wi=7, Di=7, pH=0, pW=0, pD=0)
        self.assertTrue(identity_supports_problem(ident, unpadded)[0])
        self.assertFalse(identity_supports_problem(ident, _problem())[0])
        self.assertTrue(
            identity_supports_problem(replace(ident, uk_nopad=False), _problem())[0]
        )

    def test_aot_opt_in_scope(self):
        import os
        from unittest import mock

        from kernels.conv import aot_jobs

        ukd = dict(uniform_k_decode=True, reg_prefetch=1)
        caps = dict(grouped=False, is_3d=True)
        env = {
            "ROCKE_AOT_FWD_UKD_SPLIT": "1",
            "ROCKE_AOT_FWD_SPLIT_TAIL": "1",
            "ROCKE_AOT_FWD_R2_SCOPE": "128x128x64w2x2a32",
        }
        geo_in = (128, 128, 64, 2, 2, 32)
        geo_out = (64, 64, 32, 2, 2, 32)
        with mock.patch.dict(os.environ, env):
            ins = aot_jobs._fwd_job_variants(ukd, geo_in, "cshuffle", caps)
            out = aot_jobs._fwd_job_variants(ukd, geo_out, "cshuffle", caps)
            self.assertEqual(ins, [(1, False, False), (2, False, False), (2, False, True)])
            self.assertEqual(out, [(1, False, False)])
            sched = dict(ukd, lds_ahead=True, uk_nopad=True)
            self.assertTrue(aot_jobs._fwd_r2_in_scope(sched, *geo_in))
            self.assertFalse(aot_jobs._fwd_r2_in_scope(sched, *geo_out))
            self.assertTrue(aot_jobs._fwd_r2_in_scope(ukd, *geo_out))

if __name__ == "__main__":
    unittest.main(verbosity=2)
