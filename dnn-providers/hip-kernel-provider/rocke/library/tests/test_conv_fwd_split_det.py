# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Deterministic forward split-K (``ImplicitGemmConvSpec.split_det``).

GPU: modes 1 / 2 (owner-ordered fold) on the plain and the split_tail grids,
mode 3 (f32 slabs + the workspace fold) on the plain grid: against the fp32
reference, D garbage-filled before every launch, several launches whose
outputs must be bitwise identical (the counters reset themselves). CPU:
rejections, the name tag, the label round trip, the ABI.
"""

from __future__ import annotations

import unittest

import pytest

from tests.test_conv_fwd_3d_kloop_knobs import _SKIP, _TOL, _problem

_PP5 = dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True, lds_kpack_read=True,
            rpf_pingpong=5, vmem_interleave=4)
_T256 = (256, 128, 64, 4, 2, 16)


def _spec(problem, tile, **knobs):
    from kernels.common.conv_implicit_gemm import ConvDataSpec, ImplicitGemmConvSpec

    tm, tn, tk, wm, wn, wt = tile
    return ImplicitGemmConvSpec(
        problem=problem, name="test_fwd_split_det",
        data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
        tile_m=tm, tile_n=tn, tile_k=tk, warp_m=wm, warp_n=wn, warp_tile_m=wt,
        warp_tile_n=wt, warp_tile_k=16, pipeline="mem", epilogue="cshuffle",
        groups=problem.groups, **knobs,
    )


def _run_det(problem, *, degree, tile, launches=4, full_tiles=None, **knobs):
    """``(max rel err, bitwise identical over the launches)``."""
    import torch

    from builders.common.conv_reference import conv_reference
    from kernels.common._conv_implicit_gemm_common import ConvProblem
    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_implicit_gemm import (
        build_implicit_gemm_conv,
        fwd_split_det_slabs,
        fwd_split_det_slots,
    )
    from kernels.common.conv_wgrad_workspace_reduce import (
        WgradReduceSpec,
        build_conv_wgrad_workspace_reduce,
        wgrad_reduce_grid,
        wgrad_reduce_signature,
    )
    from rocke import compile_kernel
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig
    from tests.test_conv_fwd_3d_kloop_knobs import _u8, get_device_arch

    arch = get_device_arch().split(":")[0]
    tm, tn, tk = tile[:3]
    spec = _spec(problem, tile, **knobs)
    art = compile_kernel(build_implicit_gemm_conv(spec, arch=arch), arch=arch)
    launcher = KernelLauncher(
        hsaco=art.hsaco, kernel_name=art.kernel_name,
        signature=conv_args_signature("bf16", is_3d=True, split_k=True,
                                      split_tail=spec.split_tail, split_det=True),
    )
    p = problem
    torch.manual_seed(0)
    A = torch.empty(p.N, p.Di, p.Hi, p.Wi, p.C).uniform_(-1, 1).to(torch.bfloat16)
    B = torch.empty(p.K, p.Z, p.Y, p.X, p.cpg).uniform_(-1, 1).to(torch.bfloat16)
    D = torch.empty(p.N, p.Do, p.Ho, p.Wo, p.K, dtype=torch.bfloat16)
    ref = conv_reference(A, B, p, out_dtype=torch.bfloat16)
    args = ConvArgs.from_problem(p, tile_m=tm, tile_n=tn, tile_k=tk)
    tiles = args.num_pid_m * args.num_pid_n
    fb = 8 * tiles * max(1, p.groups)
    if spec.split_det == 3:
        slabs = fwd_split_det_slabs(tiles, degree)
        pb = 4 * args.gemm_m * p.K * slabs
    else:
        pb = 4 * tm * tn * max(1, fwd_split_det_slots(
            tiles, max(1, p.groups), degree, split_tail=spec.split_tail,
            full_tiles=full_tiles or 0))
    rt = Runtime()
    ptrs = [rt.alloc(t.nbytes) for t in (A, B, D)]
    ws = rt.alloc(fb + pb)
    outs, errs = [], []
    try:
        rt.memcpy_h2d(ptrs[0], _u8(A), A.nbytes)
        rt.memcpy_h2d(ptrs[1], _u8(B), B.nbytes)
        rt.memset(ws, 0, fb + pb)
        values = args.to_launch_values(
            *ptrs, A.nbytes, B.nbytes, D.nbytes, split_k=degree, sk_flags=ws,
            sk_partials=int(ws) + fb, full_tiles=full_tiles if spec.split_tail else None,
        )
        fold = None
        if spec.split_det == 3:
            fspec = WgradReduceSpec(
                problem=ConvProblem(N=1, Hi=1, Wi=1, C=p.K, K=args.gemm_m, Y=1, X=1),
                dtype_d="bf16", ws_replicas=slabs, vec=4,
            )
            fart = compile_kernel(build_conv_wgrad_workspace_reduce(fspec), arch=arch)
            fold = (KernelLauncher(hsaco=fart.hsaco, kernel_name=fart.kernel_name,
                                   signature=wgrad_reduce_signature(fspec)),
                    {"ws_ptr": int(ws) + fb, "dw_ptr": ptrs[2], "wg_M": fspec.wg_M,
                     "wg_N": fspec.wg_N, "ws_bytes": pb, "dw_bytes": D.nbytes, "groups": 1},
                    LaunchConfig(grid=wgrad_reduce_grid(fspec), block=(fspec.block_size, 1, 1),
                                 fence=True))
        gx, gy, gz = args.grid()
        for _ in range(launches):
            rt.memset(ptrs[2], 0x7F, D.nbytes)  # garbage
            launcher(values, config=LaunchConfig(grid=(gx, gy, gz * degree),
                                                 block=(spec.block_size, 1, 1), fence=True))
            if fold is not None:
                fold[0](fold[1], config=fold[2])
            out = torch.empty_like(D)
            rt.memcpy_d2h(_u8(out), ptrs[2], D.nbytes)
            outs.append(out.view(torch.int16).clone())
            o = out.float().to(ref.device)
            errs.append(float((o - ref).abs().max() / ref.abs().max().clamp(min=1.0)))
    finally:
        for ptr in ptrs + [ws]:
            rt.free(ptr)
    return max(errs), all(bool((o == outs[0]).all()) for o in outs[1:])


@pytest.mark.gpu
@unittest.skipIf(_SKIP, _SKIP)
class TestFwdSplitDet(unittest.TestCase):
    def test_owner_fold(self):
        p = _problem(C=128, K=128, Hi=18, Wi=18, Di=6, pH=0, pW=0, pD=0)
        for mode in (1, 2):
            for degree in (2, 3, 5):
                with self.subTest(mode=mode, degree=degree):
                    err, same = _run_det(p, degree=degree, tile=_T256, **_PP5, uk_nopad=True,
                                         split_k=2, split_det=mode)
                    self.assertLess(err, _TOL)
                    self.assertTrue(same)
            with self.subTest(mode=mode, split_tail=True):
                err, same = _run_det(p, degree=3, tile=_T256, full_tiles=2, **_PP5,
                                     uk_nopad=True, split_k=2, split_tail=True, split_det=mode)
                self.assertLess(err, _TOL)
                self.assertTrue(same)

    def test_slabs(self):
        p = _problem(C=128, K=128, Hi=18, Wi=18, Di=6, pH=0, pW=0, pD=0)
        for degree in (2, 3, 5):
            with self.subTest(degree=degree):
                err, same = _run_det(p, degree=degree, tile=_T256, **_PP5, uk_nopad=True,
                                     split_k=2, split_det=3)
                self.assertLess(err, _TOL)
                self.assertTrue(same)


class TestFwdSplitDetSpec(unittest.TestCase):
    def _s(self, **kw):
        return _spec(_problem(C=128, K=128, pH=0, pW=0, pD=0), _T256, **_PP5, **kw)

    def test_rejections(self):
        from kernels.common.conv_implicit_gemm import fwd_split_k_conflict

        for kw in (dict(split_k=2, split_det=1), dict(split_k=2, split_det=2),
                   dict(split_k=2, split_det=3), dict(split_k=2, split_det=2, split_tail=True),
                   dict(split_k=2, split_det=3, split_hybrid=16)):
            with self.subTest(**kw):
                self.assertEqual(fwd_split_k_conflict(self._s(**kw)), "")
        for kw in (dict(split_det=2),                              # needs split_k
                   dict(split_k=2, split_det=4),
                   dict(split_k=2, split_det=3, split_tail=True),
                   dict(split_k=2, split_det=2, split_streamk=True),
                   dict(split_k=2, split_det=2, split_zero_first=True),
                   dict(split_k=2, split_det=2, split_flag_store=True)):
            with self.subTest(**kw):
                self.assertNotEqual(fwd_split_k_conflict(self._s(**kw)), "")

    def test_name_label_abi(self):
        from kernels.common.conv_abi import conv_arg_names
        from kernels.conv.kernel_identity import KernelIdentity

        self.assertIn("_sdet3", self._s(split_k=2, split_det=3).kernel_name())
        self.assertNotIn("sdet", self._s(split_k=2).kernel_name())
        names = [n for n, _ in conv_arg_names(is_3d=True, split_k=True, split_det=True)]
        self.assertEqual(names[-4:], ["ks", "ks_count", "sk_flags", "sk_partials"])
        base = ("fwd-implicit_gemm-bf16-3d-t256x128x64-w4x2-a16x16x16-v8x8x8-mem-cshuffle-grp-"
                "sk2-ukd-rpf1-lxor-tapf-nopad-vmi5-kprd-pp5-ppbd-sapl-skx6-psf3-fbse-wave64-"
                "gfx942-llvm22")
        det = base.replace("-wave64-", "-sdet3-wave64-")
        i0, i3 = KernelIdentity.from_label(base), KernelIdentity.from_label(det)
        self.assertEqual((i0.split_det, i3.split_det), (0, 3))
        self.assertEqual(i3.label(), det)
        self.assertNotEqual(i0.stable_hash(), i3.stable_hash())


if __name__ == "__main__":
    unittest.main()
