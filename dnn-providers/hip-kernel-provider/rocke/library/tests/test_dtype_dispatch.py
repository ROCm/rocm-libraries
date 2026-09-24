# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Dtype-dispatch sanity tests for direct-conv and implicit-gemm families.

Two test tiers:

  Level 1 — IR smoke (no GPU required).
    Lowers each (family × direction × dtype) combination to LLVM-IR via the
    Python engine and asserts that the parameter pointer element type matches
    the requested dtype.  Specifically:
      - bf16 specs → the function signature must contain ``bfloat addrspace``
      - fp16 specs → must NOT contain ``bfloat addrspace``

    This would have caught the pre-fix bug where ``make_dgrad_fprop_spec``
    silently dropped ``dtype`` from ``transposed_problem``, so a bf16 dgrad
    spec emitted fp16 IR for the underlying fprop kernel.

  Level 2 — Execution round-trip (GPU + torch required).
    Runs a minimal kernel (N=1, tiny spatial) with inputs crafted to
    distinguish fp16 from bf16 at the bit level:

        1.0 as bf16 → bits 0x3F80  (interpreted as fp16 → value ≈ 1.5)
        1.0 as fp16 → bits 0x3C00  (interpreted as bf16 → value ≈ 1e-4)

    A kernel that internally reinterprets bf16 bits as fp16 (or vice-versa)
    computes a convolution with the *wrong* inputs and the result diverges
    from the float32 reference by far more than the dtype tolerance.

Run without GPU (IR only):
    PYTHONPATH=rocke/platform/python:rocke/library pytest \\
        rocke/library/tests/test_dtype_dispatch.py -k "ir"

Run with GPU (all tests):
    PYTHONPATH=rocke/platform/python:rocke/library <torch-python> -m pytest \\
        rocke/library/tests/test_dtype_dispatch.py
"""
from __future__ import annotations

import ctypes
import importlib.util
import unittest
from dataclasses import dataclass, field
from typing import Tuple

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None

if _HAS_TORCH:
    import torch

    torch.cuda.is_available()

GPU_ARCH = get_device_arch(0)
_IS_MFMA = GPU_ARCH in ("gfx942", "gfx950")


def _skip_reason_gpu() -> str:
    if not GPU_ARCH:
        return "no ROCm GPU detected"
    if not _HAS_TORCH:
        return "torch not importable"
    if not _IS_MFMA:
        return f"unsupported arch {GPU_ARCH!r} (need gfx942 or gfx950)"
    return ""


_SKIP_GPU = _skip_reason_gpu()

_ARCH = GPU_ARCH if GPU_ARCH in ("gfx942", "gfx950") else "gfx950"

# Tolerances (relative error against float32 reference).
_TOL_FP16 = 5e-2
_TOL_BF16 = 1e-1


# ---------------------------------------------------------------------------
# Shared lower helper
# ---------------------------------------------------------------------------


def _lower(kernel, arch: str) -> str:
    try:
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _f
    except ImportError:
        from rocke import lower_kernel_to_llvm as _f
    return _f(kernel, arch=arch)


# ---------------------------------------------------------------------------
# Level 1 helpers
# ---------------------------------------------------------------------------


def _assert_ir_dtype(ir: str, dtype: str, kernel_label: str) -> None:
    """Assert that the LLVM-IR load/store element type matches *dtype*.

    Modern LLVM uses opaque pointers (``ptr addrspace(N)``), so the element
    type appears in the load/store instruction, not in the function signature.
    We check for ``load <N x bfloat>`` / ``load <N x half>`` in the IR body.
    """
    if dtype == "bf16":
        # The IR must reference 'bfloat' for load/store/MFMA element type.
        # Scalar dgrad uses individual bfloat loads; vector kernels use
        # <N x bfloat> loads.  Either form proves dtype reached the builder.
        assert "bfloat" in ir, (
            f"{kernel_label}: expected 'bfloat' in IR for bf16 spec "
            f"but not found — dtype was likely not forwarded to the kernel builder"
        )
    else:
        assert "bfloat" not in ir, f"{kernel_label}: found 'bfloat' in IR for fp16 spec"


# ===========================================================================
# Level 1 — IR dtype checks (no GPU)
# ===========================================================================


class TestDirectConvIRDtype(unittest.TestCase):
    """IR-level dtype checks for the direct-conv family (no GPU required).

    Builds each (variant × direction × dtype) combination and verifies that
    the emitted LLVM-IR parameter pointers use the correct element type.
    """

    # ---- forward -----------------------------------------------------------

    def _check_fwd_ir(self, cpg: int, dtype: str) -> None:
        from kernels.common.conv_direct_grouped import (
            DirectConvProblem,
            DirectConvSpec,
            build_direct_conv,
        )

        # groups=8 satisfies default block_groups=8 for 8c/16c/32c.
        groups = 8
        p = DirectConvProblem(
            N=1,
            H=8,
            W=8,
            groups=groups,
            cpg=cpg,
            kpg=cpg,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype=dtype,
        )
        spec = DirectConvSpec(problem=p, name=f"ir_fwd_{cpg}c_{dtype}")
        kernel = build_direct_conv(spec, arch=_ARCH)
        ir = _lower(kernel, arch=_ARCH)
        _assert_ir_dtype(ir, dtype, f"direct_fwd cpg={cpg} {dtype}")

    def test_fwd_8c_fp16(self):
        self._check_fwd_ir(8, "fp16")

    def test_fwd_8c_bf16(self):
        self._check_fwd_ir(8, "bf16")

    def test_fwd_16c_fp16(self):
        self._check_fwd_ir(16, "fp16")

    def test_fwd_16c_bf16(self):
        self._check_fwd_ir(16, "bf16")

    def test_fwd_32c_fp16(self):
        self._check_fwd_ir(32, "fp16")

    def test_fwd_32c_bf16(self):
        self._check_fwd_ir(32, "bf16")

    # ---- dgrad (scalar-FMA path, cpg=kpg=1 → 16c-as-dgrad) ---------------
    # The scalar-FMA dgrad (build_direct_conv_dgrad) uses _io_type(p.dtype)
    # directly.  The MFMA dgrad path (cpg>1 → make_dgrad_fprop_spec) is the
    # one that previously dropped dtype; it maps to DirectConvDgradSpec.

    def _check_dgrad_scalar_ir(self, dtype: str) -> None:
        from kernels.common.conv_direct_grouped import (
            DirectConvDgradSpec,
            DirectConvProblem,
            build_direct_conv_dgrad,
        )

        # cpg=kpg=1 → scalar FMA dgrad (single kernel, no weight-transpose)
        p = DirectConvProblem(
            N=1,
            H=8,
            W=8,
            groups=8,
            cpg=1,
            kpg=1,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype=dtype,
        )
        spec = DirectConvDgradSpec(
            problem=p, name=f"ir_dgrad_dw_{dtype}", block_q=8, block_groups=8
        )
        kernel = build_direct_conv_dgrad(spec, arch=_ARCH)
        ir = _lower(kernel, arch=_ARCH)
        _assert_ir_dtype(ir, dtype, f"direct_dgrad dw {dtype}")

    def test_dgrad_scalar_fp16(self):
        self._check_dgrad_scalar_ir("fp16")

    def test_dgrad_scalar_bf16(self):
        self._check_dgrad_scalar_ir("bf16")

    # ---- dgrad MFMA path (cpg>1 → transposed fprop) -----------------------
    # THIS test would have caught the pre-fix bug: make_dgrad_fprop_spec
    # dropped dtype, so the transposed fprop kernel always emitted fp16 IR.

    def _check_dgrad_mfma_ir(self, cpg: int, dtype: str) -> None:
        from kernels.common.conv_direct_grouped import (
            DirectConvProblem,
            make_dgrad_fprop_spec,
            build_direct_conv,
        )

        p = DirectConvProblem(
            N=1,
            H=8,
            W=8,
            groups=8,
            cpg=cpg,
            kpg=cpg,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype=dtype,
        )
        spec = make_dgrad_fprop_spec(p, block_q=16, block_groups=8 if cpg <= 16 else 4)
        kernel = build_direct_conv(spec, arch=_ARCH)
        ir = _lower(kernel, arch=_ARCH)
        _assert_ir_dtype(ir, dtype, f"direct_dgrad mfma cpg={cpg} {dtype}")

    def test_dgrad_mfma_16c_fp16(self):
        self._check_dgrad_mfma_ir(16, "fp16")

    def test_dgrad_mfma_16c_bf16(self):
        # Regression test: before the fix this emitted 'half addrspace' even
        # though dtype="bf16" because make_dgrad_fprop_spec dropped dtype.
        self._check_dgrad_mfma_ir(16, "bf16")

    def test_dgrad_mfma_32c_fp16(self):
        self._check_dgrad_mfma_ir(32, "fp16")

    def test_dgrad_mfma_32c_bf16(self):
        self._check_dgrad_mfma_ir(32, "bf16")


class TestImplicitGemmIRDtype(unittest.TestCase):
    """IR-level dtype checks for the implicit-gemm family (no GPU required)."""

    def _make_problem(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem

        # Tiny shape that satisfies tile alignment constraints.
        # C=K=64 so default_vector_sizes returns vec_c=8, but default epilogue
        # does not support vec_c > 1, so we override vector_size_c=1 below.
        return ConvProblem(N=1, Hi=8, Wi=8, C=64, K=64, Y=3, X=3, pH=1, pW=1)

    def _check_fwd_ir(self, dtype: str) -> None:
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec
        from kernels.common.conv_implicit_gemm import (
            ImplicitGemmConvSpec,
            build_implicit_gemm_conv,
        )

        p = self._make_problem()
        data = ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype)
        spec = ImplicitGemmConvSpec(
            problem=p,
            name=f"ir_igemm_fwd_{dtype}",
            data=data,
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            # vec_c > 1 is incompatible with epilogue="default"; force 1.
            vector_size_c=1,
        )
        kernel = build_implicit_gemm_conv(spec, arch=_ARCH)
        ir = _lower(kernel, arch=_ARCH)
        _assert_ir_dtype(ir, dtype, f"igemm_fwd {dtype}")

    def test_fwd_fp16(self):
        self._check_fwd_ir("fp16")

    def test_fwd_bf16(self):
        self._check_fwd_ir("bf16")

    def _check_wgrad_ir(self, dtype: str) -> None:
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec
        from kernels.common.conv_implicit_gemm_wgrad import (
            WgradConvSpec,
            build_implicit_gemm_conv_wgrad,
        )

        p = self._make_problem()
        data = ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype)
        spec = WgradConvSpec(
            problem=p,
            name=f"ir_igemm_wgrad_{dtype}",
            data=data,
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        )
        kernel = build_implicit_gemm_conv_wgrad(spec, arch=_ARCH)
        ir = _lower(kernel, arch=_ARCH)
        _assert_ir_dtype(ir, dtype, f"igemm_wgrad {dtype}")

    def test_wgrad_fp16(self):
        self._check_wgrad_ir("fp16")

    def test_wgrad_bf16(self):
        self._check_wgrad_ir("bf16")

    def _check_dgrad_ir(self, dtype: str) -> None:
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec
        from kernels.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            build_implicit_gemm_conv_dgrad,
        )

        p = self._make_problem()
        data = ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype)
        spec = DgradConvSpec(
            problem=p,
            name=f"ir_igemm_dgrad_{dtype}",
            data=data,
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        )
        kernel = build_implicit_gemm_conv_dgrad(spec, arch=_ARCH)
        ir = _lower(kernel, arch=_ARCH)
        _assert_ir_dtype(ir, dtype, f"igemm_dgrad {dtype}")

    def test_dgrad_fp16(self):
        self._check_dgrad_ir("fp16")

    def test_dgrad_bf16(self):
        self._check_dgrad_ir("bf16")


# ===========================================================================
# Level 2 — GPU execution with dtype-distinguishing inputs
# ===========================================================================
#
# Key insight: 1.0 represented as bf16 has bits 0x3F80.
# Those same bits as fp16 decode to ≈ 1.5 (sign=0, exp=15→0, frac=0.5).
# So a kernel that silently reinterprets bf16 data as fp16 computes a
# convolution with inputs ≈ 1.5 instead of 1.0, producing wrong results.
# The relative error against the correct float32 reference is >> tolerance.


def _u8(t):
    import torch  # noqa: F401

    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _conv_ref_f32(A_t, B_t, p_stride: int, p_pad: int, groups: int):
    """Return NHWK float32 reference computed by torch.nn.functional.conv2d."""
    import torch
    import torch.nn.functional as F

    A_f = A_t.float()
    B_f = B_t.float()
    A_nchw = A_f.permute(0, 3, 1, 2)
    B_nchw = B_f.permute(0, 3, 1, 2)
    out = F.conv2d(A_nchw, B_nchw, padding=p_pad, stride=p_stride, groups=groups)
    return out.permute(0, 2, 3, 1).contiguous().cuda()


def _run_direct_fwd(arch: str, cpg: int, dtype: str) -> Tuple[bool, str]:
    import torch
    from rocke import compile_kernel
    from rocke.helpers.manifest import conv_args_signature
    from kernels.common.conv_direct_grouped import (
        DirectConvProblem,
        DirectConvSpec,
        build_direct_conv,
        is_valid_spec,
    )
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    groups = 8
    N, H, W = 1, 8, 8
    p = DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=cpg,
        kpg=cpg,
        KH=3,
        KW=3,
        PAD=1,
        stride=1,
        dtype=dtype,
    )
    spec = DirectConvSpec(
        problem=p, name=f"exec_fwd_{cpg}c_{dtype}", block_groups=groups
    )
    ok, reason = is_valid_spec(spec, arch=arch)
    if not ok:
        return False, f"skip {reason}"

    kernel = build_direct_conv(spec, arch=arch)
    artifact = compile_kernel(kernel, arch=arch)

    td = torch.bfloat16 if dtype == "bf16" else torch.float16
    torch.manual_seed(0)
    A = torch.ones(N, H, W, groups * cpg, dtype=td).cuda()
    B = torch.ones(groups * cpg, 3, 3, cpg, dtype=td).cuda()
    D = torch.zeros(N, p.Ho, p.Wo, groups * cpg, dtype=td).cuda()

    ref = _conv_ref_f32(A.cpu(), B.cpu(), p.stride, p.PAD, groups)

    rt = Runtime()
    A_dev = rt.alloc(A.nbytes)
    rt.memcpy_h2d(A_dev, _u8(A), A.nbytes)
    B_dev = rt.alloc(B.nbytes)
    rt.memcpy_h2d(B_dev, _u8(B), B.nbytes)
    D_dev = rt.alloc(D.nbytes)
    rt.memset(D_dev, 0, D.nbytes)

    sig = conv_args_signature(dtype)
    try:
        launcher = KernelLauncher(
            hsaco=artifact.hsaco, kernel_name=artifact.kernel_name, signature=sig
        )
    except HipError as e:
        for dev in (A_dev, B_dev, D_dev):
            rt.free(dev)
        return False, f"load failed: {e}"

    q_tiles = (p.Wo + spec.block_q - 1) // spec.block_q
    g_tiles = p.groups // spec.block_groups
    launcher(
        {
            "A": A_dev,
            "B": B_dev,
            "D": D_dev,
            "A_bytes": A.nbytes,
            "B_bytes": B.nbytes,
            "D_bytes": D.nbytes,
        },
        config=LaunchConfig(
            grid=(q_tiles, g_tiles, N), block=(spec.threads_per_block, 1, 1), fence=True
        ),
    )

    D_cpu = torch.empty_like(D.cpu())
    rt.memcpy_d2h(_u8(D_cpu), D_dev, D.nbytes)
    for dev in (A_dev, B_dev, D_dev):
        rt.free(dev)
    synchronize_and_release(0)

    rel = float(
        (D_cpu.float() - ref.float().cpu()).abs().max()
        / ref.float().cpu().abs().max().clamp(min=1.0)
    )
    tol = _TOL_BF16 if dtype == "bf16" else _TOL_FP16
    return (rel < tol), f"rel_err={rel:.3e} tol={tol:.1e}"


def _run_direct_dgrad(arch: str, cpg: int, dtype: str) -> Tuple[bool, str]:
    import torch
    from rocke import compile_kernel
    from rocke.helpers.manifest import conv_args_signature
    from kernels.common.conv_direct_grouped import (
        DirectConvDgradSpec,
        DirectConvProblem,
        build_direct_conv_dgrad,
        is_valid_dgrad_spec,
    )
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    groups = 8
    N, H, W = 1, 8, 8
    kpg = cpg
    p = DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=cpg,
        kpg=kpg,
        KH=3,
        KW=3,
        PAD=1,
        stride=1,
        dtype=dtype,
    )
    spec = DirectConvDgradSpec(
        problem=p, name=f"exec_dgrad_{cpg}c_{dtype}", block_q=16, block_groups=groups
    )
    ok, reason = is_valid_dgrad_spec(spec, arch=arch)
    if not ok:
        return False, f"skip {reason}"

    kernel = build_direct_conv_dgrad(spec, arch=arch)
    artifact = compile_kernel(kernel, arch=arch)

    td = torch.bfloat16 if dtype == "bf16" else torch.float16
    torch.manual_seed(42)
    total_c = groups * cpg
    total_k = groups * kpg
    dY = torch.ones(N, p.Ho, p.Wo, total_k, dtype=td).cuda()
    Wt = torch.ones(total_k, 3, 3, cpg, dtype=td).cuda()
    dX = torch.zeros(N, H, W, total_c, dtype=td).cuda()

    dY_nchw = dY.float().permute(0, 3, 1, 2)
    W_nchw = Wt.float().permute(0, 3, 1, 2)
    h_base = (p.Ho - 1) * p.stride - 2 * p.PAD + p.KH
    w_base = (p.Wo - 1) * p.stride - 2 * p.PAD + p.KW
    ref_nchw = torch.nn.functional.conv_transpose2d(
        dY_nchw,
        W_nchw,
        padding=p.PAD,
        stride=p.stride,
        groups=groups,
        output_padding=(H - h_base, W - w_base),
    )
    ref = ref_nchw.permute(0, 2, 3, 1).contiguous()

    rt = Runtime()
    dY_dev = rt.alloc(dY.nbytes)
    rt.memcpy_h2d(dY_dev, _u8(dY), dY.nbytes)
    W_dev = rt.alloc(Wt.nbytes)
    rt.memcpy_h2d(W_dev, _u8(Wt), Wt.nbytes)
    dX_dev = rt.alloc(dX.nbytes)
    rt.memset(dX_dev, 0, dX.nbytes)

    sig = conv_args_signature(dtype)
    try:
        launcher = KernelLauncher(
            hsaco=artifact.hsaco, kernel_name=artifact.kernel_name, signature=sig
        )
    except HipError as e:
        for dev in (dY_dev, W_dev, dX_dev):
            rt.free(dev)
        return False, f"load failed: {e}"

    q_tiles = (W + spec.block_q - 1) // spec.block_q
    g_tiles = (total_c + spec.threads_per_block - 1) // spec.threads_per_block
    launcher(
        {
            "A": dY_dev,
            "B": W_dev,
            "D": dX_dev,
            "A_bytes": dY.nbytes,
            "B_bytes": Wt.nbytes,
            "D_bytes": dX.nbytes,
        },
        config=LaunchConfig(
            grid=(q_tiles, g_tiles, N * H),
            block=(spec.threads_per_block, 1, 1),
            fence=True,
        ),
    )

    dX_cpu = torch.empty_like(dX.cpu())
    rt.memcpy_d2h(_u8(dX_cpu), dX_dev, dX.nbytes)
    for dev in (dY_dev, W_dev, dX_dev):
        rt.free(dev)
    synchronize_and_release(0)

    scale = ref.float().abs().max().clamp(min=1.0)
    rel = float((dX_cpu.float() - ref.float().cpu()).abs().max() / scale)
    tol = _TOL_BF16 if dtype == "bf16" else _TOL_FP16
    return (rel < tol), f"rel_err={rel:.3e} tol={tol:.1e}"


@unittest.skipUnless(not _SKIP_GPU, _SKIP_GPU or "no GPU")
class TestDirectConvExecDtype(unittest.TestCase):
    """GPU execution tests: verify dtype-correct output for direct-conv.

    Uses all-ones inputs so the expected result is known analytically
    (sum of kernel-window elements) and distinguishes fp16 vs bf16 because
    their bit patterns for the same float value differ:
        1.0_bf16 = 0x3F80  → read as fp16 → ≈ 1.5
        1.0_fp16 = 0x3C00  → read as bf16 → ≈ 9.77e-4
    A kernel with wrong dtype reinterpretation fails the relative-error gate.
    """

    def _fwd(self, cpg: int, dtype: str) -> None:
        ok, msg = _run_direct_fwd(_ARCH, cpg, dtype)
        if msg.startswith("skip"):
            self.skipTest(msg)
        self.assertTrue(ok, f"FAIL direct_fwd cpg={cpg} {dtype}: {msg}")

    def _dgrad(self, cpg: int, dtype: str) -> None:
        ok, msg = _run_direct_dgrad(_ARCH, cpg, dtype)
        if msg.startswith("skip"):
            self.skipTest(msg)
        self.assertTrue(ok, f"FAIL direct_dgrad cpg={cpg} {dtype}: {msg}")

    # -- forward fp16
    def test_exec_fwd_16c_fp16(self):
        self._fwd(16, "fp16")

    def test_exec_fwd_8c_fp16(self):
        self._fwd(8, "fp16")

    # -- forward bf16
    def test_exec_fwd_16c_bf16(self):
        self._fwd(16, "bf16")

    def test_exec_fwd_8c_bf16(self):
        self._fwd(8, "bf16")

    # -- dgrad fp16
    def test_exec_dgrad_16c_fp16(self):
        self._dgrad(16, "fp16")

    # -- dgrad bf16  (regression: make_dgrad_fprop_spec used to drop dtype)
    def test_exec_dgrad_16c_bf16(self):
        self._dgrad(16, "bf16")
