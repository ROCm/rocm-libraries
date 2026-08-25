# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Comprehensive GPU numeric tests for all three wgrad epilogue paths on gfx942.

Tests all three execution paths defined in conv_implicit_gemm_wgrad.py:

  Path 1 — split_k=1, two_stage=False:
      Direct store epilogue.  Always deterministic.  No workspace, no atomics.

  Path 2 — split_k>1, two_stage=False:
      global_atomic_add epilogue.  Non-deterministic.  Caller must zero-init dW.
      No workspace buffer needed.

  Path 3 — split_k>1, two_stage=True:
      Stage 1 writes f32 partial sums to a workspace; Stage 2 reduces them in a
      fixed sequential order.  Bit-exact reproducible.

Each path is tested across a wide range of:
  - Kernel sizes: 1×1, 1×3, 3×3, 5×5
  - Padding:      0, 1, 2
  - Stride:       1, 2
  - Dilation:     1, 2
  - Problem sizes: various N, C, K, Hi, Wi
  - Pipelines:    "mem", "compv4"
  - split_k:      1, 4, 8 (where applicable)

Run on a gfx942 device with the rocke venv:
  ROCKE_COMGR_LIB=/opt/rocm/lib/libamd_comgr.so.3 \
  PYTHONPATH=python python -m pytest tests/instances/test_wgrad_numeric_comprehensive.py -v
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest

_HAS_TORCH = importlib.util.find_spec("torch") is not None

try:
    from rocke.runtime.hip_module import Runtime, get_device_arch

    _ARCH = get_device_arch(0) or ""
except Exception:
    _ARCH = ""

_CDNA_ARCHES = ("gfx942", "gfx950")
_SKIP_REASON = (
    f"needs a CDNA GPU (gfx942/gfx950) + torch; detected arch={_ARCH!r}"
    if _HAS_TORCH
    else f"needs torch; detected arch={_ARCH!r}"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _u8(t):
    """Return a ctypes byte array backed by ``t``'s data pointer."""
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _cpu_wgrad_reference(X_f32, dY_f32, p):
    """CPU torch reference for the weight gradient."""
    import torch

    X_nchw = X_f32.float().permute(0, 3, 1, 2).contiguous()
    dY_nchw = dY_f32.float().permute(0, 3, 1, 2).contiguous()
    dW_nchw = torch.nn.grad.conv2d_weight(
        X_nchw,
        weight_size=(p.K, p.C // p.groups, p.Y, p.X),
        grad_output=dY_nchw,
        stride=(p.sH, p.sW),
        padding=(p.pH, p.pW),
        dilation=(p.dH, p.dW),
        groups=p.groups,
    )
    return dW_nchw.permute(0, 2, 3, 1).contiguous()


def _run_wgrad(spec, arch, rt, dY_t, X_t):
    """Unified launcher for all three wgrad paths.

    Dispatches based on ``spec.two_stage`` and ``spec.split_k``:
      split_k=1  + two_stage=False → direct store
      split_k>1  + two_stage=False → atomic add (caller must zero-init dW)
      split_k>1  + two_stage=True  → workspace store + sequential reduce

    Returns dW as a CPU fp32 tensor.
    """
    import torch
    from rocke.helpers.compile import compile_kernel
    from rocke.helpers.manifest import conv_args_signature
    from rocke.instances.common.conv_implicit_gemm_wgrad import (
        build_implicit_gemm_conv_wgrad,
    )
    from rocke.instances.common.conv_implicit_gemm_wgrad_two_stage import (
        build_implicit_gemm_conv_wgrad_two_stage,
        wgrad_two_stage_workspace_nbytes,
    )
    from rocke.instances.common.conv_wgrad_workspace_reduce import (
        WgradReduceSpec,
        wgrad_reduce_grid,
    )
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig, PipelineLauncher

    p = spec.problem
    dW_t = torch.zeros(p.K, p.Y, p.X, p.C, dtype=torch.float16)

    dY_dev = rt.alloc(dY_t.nbytes)
    X_dev = rt.alloc(X_t.nbytes)
    dW_dev = rt.alloc(dW_t.nbytes)

    try:
        rt.memcpy_h2d(dY_dev, _u8(dY_t), dY_t.nbytes)
        rt.memcpy_h2d(X_dev, _u8(X_t), X_t.nbytes)
        # Always zero dW: required for atomic path, harmless for others.
        rt.memset(dW_dev, 0, dW_t.nbytes)

        if spec.two_stage:
            # ---- Path 3: two-stage workspace + sequential reduce ----
            ws_nbytes = wgrad_two_stage_workspace_nbytes(spec)
            ws_dev = rt.alloc(ws_nbytes)
            rt.memset(ws_dev, 0, ws_nbytes)
            try:
                pipeline, _ = build_implicit_gemm_conv_wgrad_two_stage(spec, arch=arch)
                s2_spec = WgradReduceSpec(
                    problem=spec.problem, dtype_d=spec.data.dtype_d
                )
                s2_grid = wgrad_reduce_grid(s2_spec)
                s1_grid = (
                    (spec.wg_N + spec.tile_n - 1) // spec.tile_n,
                    (spec.wg_M + spec.tile_m - 1) // spec.tile_m,
                    spec.split_k,
                )
                s1_vals = {
                    "A": dY_dev,
                    "B": X_dev,
                    "D": dW_dev,
                    "A_bytes": dY_t.nbytes,
                    "B_bytes": X_t.nbytes,
                    "D_bytes": dW_t.nbytes,
                    "ws_ptr": ws_dev,
                    "ws_bytes": ws_nbytes,
                }
                s2_vals = {
                    "ws_ptr": ws_dev,
                    "dw_ptr": dW_dev,
                    "wg_M": spec.wg_M,
                    "wg_N": spec.wg_N,
                    "split_k": spec.split_k,
                    "ws_bytes": ws_nbytes,
                    "dw_bytes": dW_t.nbytes,
                }
                s1_cfg = LaunchConfig(
                    grid=s1_grid, block=(spec.block_size, 1, 1), stream=0, fence=False
                )
                s2_cfg = LaunchConfig(
                    grid=s2_grid, block=(s2_spec.block_size, 1, 1), stream=0, fence=True
                )
                pipeline(
                    values_per_stage=[s1_vals, s2_vals],
                    configs_per_stage=[s1_cfg, s2_cfg],
                    stream=0,
                )
            finally:
                rt.free(ws_dev)
        else:
            # ---- Path 1 (split_k=1 direct store) or Path 2 (split_k>1 atomic) ----
            kernel = build_implicit_gemm_conv_wgrad(spec, arch)
            artifact = compile_kernel(kernel, arch=arch)
            sig = list(conv_args_signature(spec.data.dtype_a))
            launcher = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=sig,
                cache_key=("wgrad_numeric_comp", spec.kernel_name()),
            )
            grid = (
                (spec.wg_N + spec.tile_n - 1) // spec.tile_n,
                (spec.wg_M + spec.tile_m - 1) // spec.tile_m,
                max(1, spec.split_k),
            )
            vals = {
                "A": dY_dev,
                "B": X_dev,
                "D": dW_dev,
                "A_bytes": dY_t.nbytes,
                "B_bytes": X_t.nbytes,
                "D_bytes": dW_t.nbytes,
            }
            cfg = LaunchConfig(
                grid=grid, block=(spec.block_size, 1, 1), stream=0, fence=True
            )
            PipelineLauncher([launcher])(
                values_per_stage=[vals], configs_per_stage=[cfg], stream=0
            )

        dW_out = torch.empty_like(dW_t)
        rt.memcpy_d2h(_u8(dW_out), dW_dev, dW_t.nbytes)
    finally:
        rt.free(dY_dev)
        rt.free(X_dev)
        rt.free(dW_dev)

    return dW_out.float()


def _make_spec(
    arch,
    N,
    C,
    K,
    Hi,
    Wi,
    Y,
    X,
    sH=1,
    sW=1,
    pH=0,
    pW=0,
    dH=1,
    dW=1,
    split_k=1,
    two_stage=False,
    pipeline="mem",
    tile_m=64,
    tile_n=32,
    tile_k=16,
    warp_m=2,
    warp_n=2,
    warp_tile_m=16,
    warp_tile_n=16,
    warp_tile_k=16,
):
    from rocke.instances.common._conv_implicit_gemm_common import (
        ConvDataSpec,
        ConvProblem,
    )
    from rocke.instances.common.conv_implicit_gemm_wgrad import WgradConvSpec

    p = ConvProblem(
        N=N, Hi=Hi, Wi=Wi, C=C, K=K, Y=Y, X=X, sH=sH, sW=sW, pH=pH, pW=pW, dH=dH, dW=dW
    )
    return WgradConvSpec(
        problem=p,
        data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"),
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        warp_m=warp_m,
        warp_n=warp_n,
        warp_tile_m=warp_tile_m,
        warp_tile_n=warp_tile_n,
        warp_tile_k=warp_tile_k,
        pipeline=pipeline,
        epilogue="default",
        split_k=split_k,
        two_stage=two_stage,
    )


# ---------------------------------------------------------------------------
# Shape catalogue
# (label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW)
# ---------------------------------------------------------------------------

# All shapes must produce valid (Ho > 0, Wo > 0) convolution output.
# For split_k>1 the atomic path additionally requires even C (packed fp16 atomics).
_SHAPES = [
    # --- Standard 3x3 with various paddings ---
    ("3x3_pad1", 2, 16, 32, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1),  # Ho=8, Wo=8
    ("3x3_pad0", 2, 16, 32, 8, 8, 3, 3, 1, 1, 0, 0, 1, 1),  # Ho=6, Wo=6
    ("3x3_pad2", 2, 16, 32, 10, 10, 3, 3, 1, 1, 2, 2, 1, 1),  # Ho=10, Wo=10
    # --- Stride ---
    ("3x3_stride2_p1", 2, 32, 32, 16, 16, 3, 3, 2, 2, 1, 1, 1, 1),  # Ho=8, Wo=8
    ("3x3_stride2_p0", 2, 32, 32, 16, 16, 3, 3, 2, 2, 0, 0, 1, 1),  # Ho=7, Wo=7
    # --- Dilation ---
    ("3x3_dil2_p2", 2, 16, 32, 10, 10, 3, 3, 1, 1, 2, 2, 2, 2),  # Ho=10, Wo=10
    ("3x3_dil2_p0", 2, 16, 32, 10, 10, 3, 3, 1, 1, 0, 0, 2, 2),  # Ho=6, Wo=6
    # --- Kernel size variety ---
    ("1x1_pointwise", 4, 64, 64, 14, 14, 1, 1, 1, 1, 0, 0, 1, 1),  # Ho=14, Wo=14
    ("1x3_nonsquare", 2, 16, 32, 8, 8, 1, 3, 1, 1, 0, 1, 1, 1),  # Ho=8, Wo=8
    ("5x5_pad2", 2, 16, 32, 12, 12, 5, 5, 1, 1, 2, 2, 1, 1),  # Ho=12, Wo=12
    # --- Non-tile-aligned K (tests OOB guard in Stage 1) ---
    ("3x3_K40_align", 2, 16, 40, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1),  # wg_M=40 < tile_m=64
    # --- Larger spatial / batch ---
    ("3x3_largeN", 8, 16, 16, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1),  # N=8
    ("3x3_large_sp", 2, 16, 32, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1),  # Hi=Wi=28
]

# Shapes valid for split_k>1 (C must be even for fp16 packed atomics — all above have even C).
# K=40 is allowed: C=16 is even, which is the constraint.
_SHAPES_SPLITK = _SHAPES  # same catalogue; C is even in all cases


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


@unittest.skipUnless(_ARCH in _CDNA_ARCHES and _HAS_TORCH, _SKIP_REASON)
class TestWgradDirectStore(unittest.TestCase):
    """Path 1: split_k=1, two_stage=False — direct store epilogue.

    Always deterministic; no workspace; no atomics.
    """

    ARCH = _ARCH
    TOL = 5e-2

    @classmethod
    def setUpClass(cls):
        cls.rt = Runtime()

    def _check(
        self, label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW, pipeline="mem"
    ):
        import torch
        from rocke.instances.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        spec = _make_spec(
            self.ARCH,
            N,
            C,
            K,
            Hi,
            Wi,
            Y,
            X,
            sH,
            sW,
            pH,
            pW,
            dH,
            dW,
            split_k=1,
            two_stage=False,
            pipeline=pipeline,
        )
        ok, reason = is_valid_wgrad_spec(spec, arch=self.ARCH)
        if not ok:
            self.skipTest(f"{label}: invalid spec ({reason})")

        p = spec.problem
        torch.manual_seed(abs(hash(label)) % (2**31))
        X_f32 = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1.0, 1.0)
        dY_f32 = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0)

        dW_ours = _run_wgrad(spec, self.ARCH, self.rt, dY_f32.half(), X_f32.half())
        dW_ref = _cpu_wgrad_reference(X_f32, dY_f32, p)

        max_abs = dW_ref.abs().max().item()
        if max_abs < 1e-6:
            self.skipTest(f"{label}: reference near-zero, problem degenerate")
        rel_err = (dW_ours - dW_ref).abs().max().item() / max_abs
        self.assertLess(
            rel_err,
            self.TOL,
            f"[direct_store/{pipeline}] {label}: rel_err={rel_err:.3e} >= {self.TOL}",
        )

    def test_3x3_shapes_mem(self):
        for label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW in _SHAPES:
            with self.subTest(shape=label):
                self._check(
                    label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW, pipeline="mem"
                )

    def test_compv4_pipeline(self):
        """compv4 software-pipelined path for a representative subset."""
        subset = [
            s
            for s in _SHAPES
            if s[0]
            in (
                "3x3_pad1",
                "3x3_stride2_p1",
                "1x1_pointwise",
                "3x3_large_sp",
            )
        ]
        for label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW in subset:
            with self.subTest(shape=label):
                self._check(
                    label,
                    N,
                    C,
                    K,
                    Hi,
                    Wi,
                    Y,
                    X,
                    sH,
                    sW,
                    pH,
                    pW,
                    dH,
                    dW,
                    pipeline="compv4",
                )

    def test_split_k1_is_deterministic(self):
        """Direct store is bit-exact across two runs."""
        import torch

        label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW = _SHAPES[0]
        spec = _make_spec(
            self.ARCH,
            N,
            C,
            K,
            Hi,
            Wi,
            Y,
            X,
            sH,
            sW,
            pH,
            pW,
            dH,
            dW,
            split_k=1,
            two_stage=False,
        )
        p = spec.problem
        torch.manual_seed(11)
        X_t = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1.0, 1.0).half()
        dY_t = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0).half()
        r1 = _run_wgrad(spec, self.ARCH, self.rt, dY_t, X_t)
        r2 = _run_wgrad(spec, self.ARCH, self.rt, dY_t, X_t)
        self.assertTrue(torch.equal(r1, r2), "direct-store output not bit-exact")

    def test_large_tile_64x64(self):
        """64×64 tile with mem pipeline on several shapes."""
        subset = [
            s
            for s in _SHAPES
            if s[0]
            in (
                "3x3_pad1",
                "1x1_pointwise",
                "3x3_large_sp",
            )
        ]
        for label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW in subset:
            with self.subTest(shape=label):
                import torch
                from rocke.instances.common.conv_implicit_gemm_wgrad import (
                    is_valid_wgrad_spec,
                )

                spec = _make_spec(
                    self.ARCH,
                    N,
                    C,
                    K,
                    Hi,
                    Wi,
                    Y,
                    X,
                    sH,
                    sW,
                    pH,
                    pW,
                    dH,
                    dW,
                    split_k=1,
                    two_stage=False,
                    pipeline="mem",
                    tile_m=64,
                    tile_n=64,
                    tile_k=64,
                    warp_m=2,
                    warp_n=2,
                    warp_tile_m=16,
                    warp_tile_n=16,
                    warp_tile_k=16,
                )
                ok, reason = is_valid_wgrad_spec(spec, arch=self.ARCH)
                if not ok:
                    self.skipTest(f"{label} 64x64: {reason}")
                p = spec.problem
                torch.manual_seed(abs(hash(label + "64x64")) % (2**31))
                X_f32 = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1.0, 1.0)
                dY_f32 = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0)
                dW_ours = _run_wgrad(
                    spec, self.ARCH, self.rt, dY_f32.half(), X_f32.half()
                )
                dW_ref = _cpu_wgrad_reference(X_f32, dY_f32, p)
                max_abs = dW_ref.abs().max().item()
                if max_abs < 1e-6:
                    continue
                rel_err = (dW_ours - dW_ref).abs().max().item() / max_abs
                self.assertLess(rel_err, self.TOL, f"{label} 64x64: {rel_err:.3e}")


@unittest.skipUnless(_ARCH in _CDNA_ARCHES and _HAS_TORCH, _SKIP_REASON)
class TestWgradAtomic(unittest.TestCase):
    """Path 2: split_k>1, two_stage=False — global_atomic_add epilogue.

    Non-deterministic but fast.  No workspace.  Caller zero-inits dW.
    """

    ARCH = _ARCH
    TOL = 5e-2

    @classmethod
    def setUpClass(cls):
        cls.rt = Runtime()

    def _check(
        self,
        label,
        N,
        C,
        K,
        Hi,
        Wi,
        Y,
        X,
        sH,
        sW,
        pH,
        pW,
        dH,
        dW,
        split_k=4,
        pipeline="mem",
    ):
        import torch
        from rocke.instances.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        spec = _make_spec(
            self.ARCH,
            N,
            C,
            K,
            Hi,
            Wi,
            Y,
            X,
            sH,
            sW,
            pH,
            pW,
            dH,
            dW,
            split_k=split_k,
            two_stage=False,
            pipeline=pipeline,
        )
        ok, reason = is_valid_wgrad_spec(spec, arch=self.ARCH)
        if not ok:
            self.skipTest(f"{label}: invalid spec ({reason})")

        p = spec.problem
        torch.manual_seed(abs(hash(label + "atomic")) % (2**31))
        X_f32 = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1.0, 1.0)
        dY_f32 = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0)

        dW_ours = _run_wgrad(spec, self.ARCH, self.rt, dY_f32.half(), X_f32.half())
        dW_ref = _cpu_wgrad_reference(X_f32, dY_f32, p)

        max_abs = dW_ref.abs().max().item()
        if max_abs < 1e-6:
            self.skipTest(f"{label}: reference near-zero")
        rel_err = (dW_ours - dW_ref).abs().max().item() / max_abs
        self.assertLess(
            rel_err,
            self.TOL,
            f"[atomic/spk{split_k}/{pipeline}] {label}: rel_err={rel_err:.3e} >= {self.TOL}",
        )

    def test_split_k4_all_shapes(self):
        for label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW in _SHAPES_SPLITK:
            with self.subTest(shape=label):
                self._check(
                    label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW, split_k=4
                )

    def test_split_k8_subset(self):
        """split_k=8 exercises more atomic-add partitions."""
        subset = [
            s
            for s in _SHAPES_SPLITK
            if s[0]
            in (
                "3x3_pad1",
                "3x3_dil2_p2",
                "1x1_pointwise",
                "3x3_large_sp",
            )
        ]
        for label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW in subset:
            with self.subTest(shape=label):
                self._check(
                    label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW, split_k=8
                )

    def test_compv4_pipeline(self):
        subset = [
            s
            for s in _SHAPES_SPLITK
            if s[0]
            in (
                "3x3_pad1",
                "3x3_stride2_p1",
                "1x1_pointwise",
            )
        ]
        for label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW in subset:
            with self.subTest(shape=label):
                self._check(
                    label,
                    N,
                    C,
                    K,
                    Hi,
                    Wi,
                    Y,
                    X,
                    sH,
                    sW,
                    pH,
                    pW,
                    dH,
                    dW,
                    split_k=4,
                    pipeline="compv4",
                )

    def test_large_tile_128x64(self):
        """128×64 tile exercising a different warp grid layout."""
        subset = [
            s
            for s in _SHAPES_SPLITK
            if s[0]
            in (
                "3x3_pad1",
                "1x1_pointwise",
                "3x3_large_sp",
            )
        ]
        for label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW in subset:
            with self.subTest(shape=label):
                import torch
                from rocke.instances.common.conv_implicit_gemm_wgrad import (
                    is_valid_wgrad_spec,
                )

                spec = _make_spec(
                    self.ARCH,
                    N,
                    C,
                    K,
                    Hi,
                    Wi,
                    Y,
                    X,
                    sH,
                    sW,
                    pH,
                    pW,
                    dH,
                    dW,
                    split_k=4,
                    two_stage=False,
                    pipeline="mem",
                    tile_m=128,
                    tile_n=64,
                    tile_k=64,
                    warp_m=4,
                    warp_n=2,
                    warp_tile_m=16,
                    warp_tile_n=16,
                    warp_tile_k=16,
                )
                ok, reason = is_valid_wgrad_spec(spec, arch=self.ARCH)
                if not ok:
                    self.skipTest(f"{label} 128x64: {reason}")
                p = spec.problem
                torch.manual_seed(abs(hash(label + "128x64at")) % (2**31))
                X_f32 = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1.0, 1.0)
                dY_f32 = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0)
                dW_ours = _run_wgrad(
                    spec, self.ARCH, self.rt, dY_f32.half(), X_f32.half()
                )
                dW_ref = _cpu_wgrad_reference(X_f32, dY_f32, p)
                max_abs = dW_ref.abs().max().item()
                if max_abs < 1e-6:
                    continue
                rel_err = (dW_ours - dW_ref).abs().max().item() / max_abs
                self.assertLess(rel_err, self.TOL, f"{label} 128x64: {rel_err:.3e}")


@unittest.skipUnless(_ARCH in _CDNA_ARCHES and _HAS_TORCH, _SKIP_REASON)
class TestWgradTwoStageComprehensive(unittest.TestCase):
    """Path 3: split_k>1, two_stage=True — workspace store + sequential reduce.

    Bit-exact reproducible.  Wide shape coverage beyond the existing
    test_wgrad_two_stage_numeric.py.
    """

    ARCH = _ARCH
    TOL = 5e-2

    @classmethod
    def setUpClass(cls):
        cls.rt = Runtime()

    def _check(
        self,
        label,
        N,
        C,
        K,
        Hi,
        Wi,
        Y,
        X,
        sH,
        sW,
        pH,
        pW,
        dH,
        dW,
        split_k=4,
        pipeline="mem",
        tile_m=64,
        tile_n=32,
        tile_k=16,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        check_determinism=False,
    ):
        import torch
        from rocke.instances.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        spec = _make_spec(
            self.ARCH,
            N,
            C,
            K,
            Hi,
            Wi,
            Y,
            X,
            sH,
            sW,
            pH,
            pW,
            dH,
            dW,
            split_k=split_k,
            two_stage=True,
            pipeline=pipeline,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_tile_m=warp_tile_m,
            warp_tile_n=warp_tile_n,
            warp_tile_k=warp_tile_k,
        )
        ok, reason = is_valid_wgrad_spec(spec, arch=self.ARCH)
        if not ok:
            self.skipTest(f"{label}: invalid spec ({reason})")

        p = spec.problem
        torch.manual_seed(abs(hash(label + "ts")) % (2**31))
        X_f32 = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1.0, 1.0)
        dY_f32 = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0)

        dW_ours = _run_wgrad(spec, self.ARCH, self.rt, dY_f32.half(), X_f32.half())
        dW_ref = _cpu_wgrad_reference(X_f32, dY_f32, p)

        max_abs = dW_ref.abs().max().item()
        if max_abs < 1e-6:
            self.skipTest(f"{label}: reference near-zero")
        rel_err = (dW_ours - dW_ref).abs().max().item() / max_abs
        self.assertLess(
            rel_err,
            self.TOL,
            f"[two_stage/spk{split_k}/{pipeline}] {label}: rel_err={rel_err:.3e} >= {self.TOL}",
        )

        if check_determinism:
            dW_run2 = _run_wgrad(spec, self.ARCH, self.rt, dY_f32.half(), X_f32.half())
            self.assertTrue(
                torch.equal(dW_ours, dW_run2),
                f"[two_stage] {label}: output not bit-exact across runs",
            )

    def test_all_shapes_split_k4(self):
        for label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW in _SHAPES_SPLITK:
            with self.subTest(shape=label):
                self._check(
                    label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW, split_k=4
                )

    def test_split_k8_subset(self):
        subset = [
            s
            for s in _SHAPES_SPLITK
            if s[0]
            in (
                "3x3_pad1",
                "3x3_dil2_p2",
                "1x1_pointwise",
                "3x3_K40_align",
                "3x3_large_sp",
            )
        ]
        for label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW in subset:
            with self.subTest(shape=label):
                self._check(
                    label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW, split_k=8
                )

    def test_determinism_across_shapes(self):
        """Two-stage output is bit-exact on a representative subset."""
        subset = [
            s
            for s in _SHAPES_SPLITK
            if s[0]
            in (
                "3x3_pad1",
                "3x3_stride2_p1",
                "3x3_dil2_p2",
                "1x1_pointwise",
                "3x3_K40_align",
            )
        ]
        for label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW in subset:
            with self.subTest(shape=label):
                self._check(
                    label,
                    N,
                    C,
                    K,
                    Hi,
                    Wi,
                    Y,
                    X,
                    sH,
                    sW,
                    pH,
                    pW,
                    dH,
                    dW,
                    split_k=4,
                    check_determinism=True,
                )

    def test_compv4_pipeline(self):
        subset = [
            s
            for s in _SHAPES_SPLITK
            if s[0]
            in (
                "3x3_pad1",
                "3x3_stride2_p1",
                "1x1_pointwise",
                "3x3_large_sp",
            )
        ]
        for label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW in subset:
            with self.subTest(shape=label):
                self._check(
                    label,
                    N,
                    C,
                    K,
                    Hi,
                    Wi,
                    Y,
                    X,
                    sH,
                    sW,
                    pH,
                    pW,
                    dH,
                    dW,
                    split_k=4,
                    pipeline="compv4",
                )

    def test_large_tile_64x64(self):
        """64×64 tile with different warp grid than the default 64×32."""
        subset = [
            s
            for s in _SHAPES_SPLITK
            if s[0]
            in (
                "3x3_pad1",
                "1x1_pointwise",
                "3x3_large_sp",
            )
        ]
        for label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW in subset:
            with self.subTest(shape=label):
                self._check(
                    label,
                    N,
                    C,
                    K,
                    Hi,
                    Wi,
                    Y,
                    X,
                    sH,
                    sW,
                    pH,
                    pW,
                    dH,
                    dW,
                    split_k=4,
                    pipeline="mem",
                    tile_m=64,
                    tile_n=64,
                    tile_k=64,
                    warp_m=2,
                    warp_n=2,
                    warp_tile_m=16,
                    warp_tile_n=16,
                    warp_tile_k=16,
                )

    def test_large_tile_128x64(self):
        """128×64 tile for two-stage path."""
        subset = [
            s
            for s in _SHAPES_SPLITK
            if s[0]
            in (
                "3x3_pad1",
                "1x1_pointwise",
                "3x3_large_sp",
            )
        ]
        for label, N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW in subset:
            with self.subTest(shape=label):
                self._check(
                    label,
                    N,
                    C,
                    K,
                    Hi,
                    Wi,
                    Y,
                    X,
                    sH,
                    sW,
                    pH,
                    pW,
                    dH,
                    dW,
                    split_k=4,
                    pipeline="mem",
                    tile_m=128,
                    tile_n=64,
                    tile_k=64,
                    warp_m=4,
                    warp_n=2,
                    warp_tile_m=16,
                    warp_tile_n=16,
                    warp_tile_k=16,
                )


@unittest.skipUnless(_ARCH in _CDNA_ARCHES and _HAS_TORCH, _SKIP_REASON)
class TestWgradAllPathsCrossCheck(unittest.TestCase):
    """Cross-path correctness: all 3 paths must agree on the same input.

    For large split_k the atomic path may accumulate more rounding error than
    the two-stage path, but both must stay within tolerance of the CPU reference.
    The two-stage path must additionally be bit-exact against itself.
    """

    ARCH = _ARCH
    TOL = 5e-2

    @classmethod
    def setUpClass(cls):
        cls.rt = Runtime()

    def _run_all_three(
        self,
        label,
        N,
        C,
        K,
        Hi,
        Wi,
        Y,
        X,
        sH=1,
        sW=1,
        pH=0,
        pW=0,
        dH=1,
        dW=1,
        split_k=4,
    ):
        import torch
        from rocke.instances.common._conv_implicit_gemm_common import (
            ConvDataSpec,
            ConvProblem,
        )
        from rocke.instances.common.conv_implicit_gemm_wgrad import (
            WgradConvSpec,
            is_valid_wgrad_spec,
        )

        p = ConvProblem(
            N=N,
            Hi=Hi,
            Wi=Wi,
            C=C,
            K=K,
            Y=Y,
            X=X,
            sH=sH,
            sW=sW,
            pH=pH,
            pW=pW,
            dH=dH,
            dW=dW,
        )
        data = ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16")
        kwargs = dict(
            problem=p,
            data=data,
            tile_m=64,
            tile_n=32,
            tile_k=16,
            warp_m=2,
            warp_n=2,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="default",
        )

        spec_direct = WgradConvSpec(**kwargs, split_k=1, two_stage=False)
        spec_atomic = WgradConvSpec(**kwargs, split_k=split_k, two_stage=False)
        spec_two_s = WgradConvSpec(**kwargs, split_k=split_k, two_stage=True)

        for name, spec in [
            ("direct", spec_direct),
            ("atomic", spec_atomic),
            ("two_stage", spec_two_s),
        ]:
            ok, reason = is_valid_wgrad_spec(spec, arch=self.ARCH)
            if not ok:
                self.skipTest(f"{label}/{name}: {reason}")

        torch.manual_seed(abs(hash(label + "cross")) % (2**31))
        X_f32 = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1.0, 1.0)
        dY_f32 = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0)
        X_t, dY_t = X_f32.half(), dY_f32.half()

        dW_ref = _cpu_wgrad_reference(X_f32, dY_f32, p)
        max_abs = dW_ref.abs().max().item()
        if max_abs < 1e-6:
            self.skipTest(f"{label}: reference near-zero")

        for name, spec in [
            ("direct", spec_direct),
            ("atomic", spec_atomic),
            ("two_stage", spec_two_s),
        ]:
            with self.subTest(path=name):
                dW_ours = _run_wgrad(spec, self.ARCH, self.rt, dY_t, X_t)
                rel_err = (dW_ours - dW_ref).abs().max().item() / max_abs
                self.assertLess(
                    rel_err,
                    self.TOL,
                    f"[{name}] {label}: rel_err={rel_err:.3e} >= {self.TOL}",
                )

    def test_cross_check_3x3_pad1(self):
        self._run_all_three(
            "3x3_pad1", N=2, C=16, K=32, Hi=8, Wi=8, Y=3, X=3, pH=1, pW=1
        )

    def test_cross_check_3x3_stride2(self):
        self._run_all_three(
            "3x3_stride2",
            N=2,
            C=32,
            K=32,
            Hi=16,
            Wi=16,
            Y=3,
            X=3,
            sH=2,
            sW=2,
            pH=1,
            pW=1,
        )

    def test_cross_check_1x1_pointwise(self):
        self._run_all_three("1x1", N=4, C=64, K=64, Hi=14, Wi=14, Y=1, X=1)

    def test_cross_check_3x3_dilation2(self):
        self._run_all_three(
            "3x3_dil2", N=2, C=16, K=32, Hi=10, Wi=10, Y=3, X=3, pH=2, pW=2, dH=2, dW=2
        )

    def test_cross_check_5x5(self):
        self._run_all_three("5x5", N=2, C=16, K=32, Hi=12, Wi=12, Y=5, X=5, pH=2, pW=2)

    def test_cross_check_nonaligned_K(self):
        self._run_all_three("K40", N=2, C=16, K=40, Hi=8, Wi=8, Y=3, X=3, pH=1, pW=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
