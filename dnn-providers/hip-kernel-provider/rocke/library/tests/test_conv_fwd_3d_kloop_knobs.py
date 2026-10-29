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


def _run(problem, *, degree=1, tile=(64, 64, 32, 2, 2, 32), **knobs):
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
        signature=conv_args_signature("bf16", is_3d=True, split_k=split),
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
        if split:
            rt.memset(ptrs[2], 0, D.nbytes)
        args = ConvArgs.from_problem(p, tile_m=tm, tile_n=tn, tile_k=tk)
        values = args.to_launch_values(
            *ptrs, A.nbytes, B.nbytes, D.nbytes, split_k=degree if split else 1
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
