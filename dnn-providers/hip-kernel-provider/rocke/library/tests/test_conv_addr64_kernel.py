# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""End-to-end GPU correctness of an addr64 implicit-GEMM forward conv kernel
(Phase 1 slice A -- 2D, MFMA, non-split, non-grouped).

Builds a real kernel with ``ImplicitGemmConvSpec(addr64=True)``, launches it
on a small conv problem, and compares against the same float32 torch
reference ``test_conv_fwd_correctness.py`` uses for the i32 path. Small
shapes only here (this is a correctness proof of the addressing mechanism,
not a >2 GiB capacity test -- that is covered at the primitive level by
test_conv_addr64_descriptor.py and platform/tests/instances/
test_addr64_gpu_numeric.py, which exercise a real >4 GiB allocation).

**Not part of the check_byte_identity parity family**: this builder has no
C++ mirror yet (see the plan doc's "known gap" note) -- this test exercises
the Python engine only.

Requires a ROCm GPU and torch; skips otherwise.
"""

from __future__ import annotations

import importlib.util
import unittest

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch

    torch.cuda.is_available()

GPU_ARCH = get_device_arch(0)
_IS_MFMA = GPU_ARCH in ("gfx942", "gfx950")


def _skip_reason() -> str:
    if not GPU_ARCH:
        return "no ROCm GPU detected"
    if not _HAS_TORCH:
        return "torch not importable"
    if not _IS_MFMA:
        return f"addr64 is MFMA-only (Phase 1 slice A); device is {GPU_ARCH}"
    return ""


_SKIP_REASON = _skip_reason()


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON)
class TestConvAddr64Kernel(unittest.TestCase):
    def _run(self, *, problem, epilogue="default", tile_n=64, vector_size_c=1):
        import ctypes

        from rocke import compile_kernel
        from rocke.core.arch import ArchTarget
        from builders.common.conv_reference import conv_reference
        from kernels.common.conv_abi import conv_args_signature
        from kernels.common.conv_args import ConvArgs
        from kernels.common.conv_implicit_gemm import (
            ConvDataSpec,
            ImplicitGemmConvSpec,
            build_implicit_gemm_conv,
        )
        from rocke.runtime import synchronize_and_release
        from rocke.runtime.hip_module import Runtime
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        arch = GPU_ARCH
        target = ArchTarget.from_gfx(arch)

        spec = ImplicitGemmConvSpec(
            problem=problem,
            name="test_conv_addr64",
            data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"),
            tile_m=64,
            tile_n=tile_n,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
            wave_size=target.wave_size,
            epilogue=epilogue,
            vector_size_c=vector_size_c,
            addr64=True,
        )
        kernel = build_implicit_gemm_conv(spec, arch=arch)
        artifact = compile_kernel(kernel, arch=arch)

        torch.manual_seed(0)
        A_t = torch.empty(problem.N, problem.Hi, problem.Wi, problem.C).uniform_(
            -1.0, 1.0
        ).to(torch.float16)
        B_t = torch.empty(problem.K, problem.Y, problem.X, problem.C).uniform_(
            -1.0, 1.0
        ).to(torch.float16)
        D_t = torch.empty(
            problem.N, problem.Ho, problem.Wo, problem.K, dtype=torch.float16
        )
        ref = conv_reference(A_t, B_t, problem, out_dtype=torch.float16)

        def u8(t):
            return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

        rt = Runtime()
        A_dev = rt.alloc(A_t.nbytes)
        B_dev = rt.alloc(B_t.nbytes)
        D_dev = rt.alloc(D_t.nbytes)
        rt.memcpy_h2d(A_dev, u8(A_t), A_t.nbytes)
        rt.memcpy_h2d(B_dev, u8(B_t), B_t.nbytes)
        rt.memset(D_dev, 0xFF, D_t.nbytes)

        sig = conv_args_signature("fp16", addr64=True)
        launcher = KernelLauncher(
            hsaco=artifact.hsaco, kernel_name=artifact.kernel_name, signature=sig
        )
        gx = (spec.grid_N_gemm + spec.tile_n - 1) // spec.tile_n
        gy = (spec.grid_M + spec.tile_m - 1) // spec.tile_m
        grid = (gx, gy, spec.grid_groups)
        block = (spec.launch_block_size, 1, 1)
        values = ConvArgs.from_problem(
            problem, tile_m=spec.tile_m, tile_n=spec.tile_n
        ).to_launch_values(
            int(A_dev),
            int(B_dev),
            int(D_dev),
            A_t.nbytes,
            B_t.nbytes,
            D_t.nbytes,
            addr64=True,
        )
        launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))
        synchronize_and_release()

        D_cpu = torch.empty_like(D_t)
        rt.memcpy_d2h(u8(D_cpu), D_dev, D_t.nbytes)
        rt.free(A_dev)
        rt.free(B_dev)
        rt.free(D_dev)

        out_f32 = D_cpu.float()
        ref_f32 = ref.cpu().float()
        abs_diff = (out_f32 - ref_f32).abs()
        ref_scale = ref_f32.abs().max().clamp(min=1.0)
        rel_err = float(abs_diff.max() / ref_scale)
        self.assertLess(rel_err, 2e-2, f"rel_err={rel_err:.3e}")

    def test_small_shape_matches_torch_reference(self):
        from kernels.common.conv_implicit_gemm import ConvProblem

        problem = ConvProblem(
            N=2, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1
        )
        self._run(problem=problem)

    def test_cshuffle_epilogue_matches_torch_reference(self):
        from kernels.common.conv_implicit_gemm import ConvProblem

        problem = ConvProblem(
            N=2, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1
        )
        self._run(problem=problem, epilogue="cshuffle", vector_size_c=None)

    def test_strided_padded_shape_matches_torch_reference(self):
        """Hi=Wi=15, sH=sW=2, pH=pW=1, Y=X=3 -> Ho=Wo=8, so Ho*Wo == tile_m
        (64): every tile stays within one image. A shape where Ho*Wo is not
        a multiple of tile_m exercises the documented image-boundary
        precondition in make_a_descriptor_addr64 / make_d_descriptor_addr64
        (not yet supported in this slice -- see plan section 2.2's
        p_image_aligned), not a bug in this test."""
        from kernels.common.conv_implicit_gemm import ConvProblem

        problem = ConvProblem(
            N=3, Hi=15, Wi=15, C=64, K=64, Y=3, X=3, sH=2, sW=2, pH=1, pW=1, dH=1, dW=1
        )
        self.assertEqual(problem.Ho * problem.Wo, 64)
        self._run(problem=problem)

    def test_partial_last_tile_exercises_the_m_bounds_mask(self):
        """M = N*Ho*Wo not a multiple of tile_m: the last tile's OOB rows
        are masked by the epilogue's m_origin_i64 bounds check (plan
        section 3) -- exactly the path this slice changed."""
        from kernels.common.conv_implicit_gemm import ConvProblem

        problem = ConvProblem(
            N=1, Hi=9, Wi=9, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1
        )
        # M = 1*9*9 = 81, not a multiple of tile_m=64 -> a partial last tile.
        self._run(problem=problem)


if __name__ == "__main__":
    unittest.main()
