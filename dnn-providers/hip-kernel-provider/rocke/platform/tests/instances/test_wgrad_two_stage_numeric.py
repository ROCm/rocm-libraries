# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU numerical correctness tests for the two-stage deterministic wgrad path.

Verifies that the two-stage pipeline (Stage 1: workspace-store epilogue +
Stage 2: sequential reduction) produces weight gradients that match a CPU
torch reference within the expected floating-point tolerance.

Also verifies:
- Bit-exact reproducibility across two consecutive runs (determinism claim).

Run on a gfx942 (or gfx950) device with the rocke venv:
  PYTHONPATH=python /root/rocke-venv/bin/python -m pytest \\
      tests/instances/test_wgrad_two_stage_numeric.py -v
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


def _u8(t):
    """Return a ctypes byte array backed by ``t``'s data pointer."""
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _cpu_wgrad_reference(X_f32, dY_f32, p):
    """CPU torch reference for weight gradient. No GPU required."""
    import torch

    X_nchw = X_f32.float().permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
    dY_nchw = dY_f32.float().permute(0, 3, 1, 2).contiguous()  # NHWK -> NKHW
    dW_nchw = torch.nn.grad.conv2d_weight(
        X_nchw,
        weight_size=(p.K, p.C // p.groups, p.Y, p.X),
        grad_output=dY_nchw,
        stride=(p.sH, p.sW),
        padding=(p.pH, p.pW),
        dilation=(p.dH, p.dW),
        groups=p.groups,
    )
    return dW_nchw.permute(0, 2, 3, 1).contiguous()  # KCHW -> KYXC


def _run_two_stage(spec, arch, rt, dY_t, X_t):
    """Compile and launch the two-stage pipeline; return dW as a CPU fp32 tensor."""
    import torch
    from rocke.instances.common.conv_implicit_gemm_wgrad_two_stage import (
        build_implicit_gemm_conv_wgrad_two_stage,
    )
    from rocke.instances.common.conv_wgrad_workspace_reduce import (
        WgradReduceSpec,
        wgrad_reduce_grid,
    )
    from rocke.runtime.launcher import LaunchConfig

    pipeline, ws_nbytes = build_implicit_gemm_conv_wgrad_two_stage(spec, arch=arch)
    assert ws_nbytes > 0, "workspace must be non-empty for split_k > 1"

    p = spec.problem
    dW_t = torch.zeros(p.K, p.Y, p.X, p.C, dtype=torch.float16)

    dY_dev = rt.alloc(dY_t.nbytes)
    X_dev = rt.alloc(X_t.nbytes)
    dW_dev = rt.alloc(dW_t.nbytes)
    ws_dev = rt.alloc(ws_nbytes)

    try:
        rt.memcpy_h2d(dY_dev, _u8(dY_t), dY_t.nbytes)
        rt.memcpy_h2d(X_dev, _u8(X_t), X_t.nbytes)
        rt.memset(dW_dev, 0, dW_t.nbytes)
        # Belt-and-suspenders: Stage 2 wraps its reduction in an OOB scf_if
        # and never loads out-of-bounds workspace elements, so zero-init is
        # not required for correctness.  It is kept here as a safety measure.
        rt.memset(ws_dev, 0, ws_nbytes)

        s2_spec = WgradReduceSpec(problem=spec.problem, dtype_d=spec.data.dtype_d)
        s2_grid = wgrad_reduce_grid(s2_spec)
        s2_block = (s2_spec.block_size, 1, 1)

        s1_grid = (
            (spec.wg_N + spec.tile_n - 1) // spec.tile_n,
            (spec.wg_M + spec.tile_m - 1) // spec.tile_m,
            spec.split_k,
        )
        s1_block = (spec.block_size, 1, 1)

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

        s1_cfg = LaunchConfig(grid=s1_grid, block=s1_block, stream=0, fence=False)
        s2_cfg = LaunchConfig(grid=s2_grid, block=s2_block, stream=0, fence=True)

        pipeline(
            values_per_stage=[s1_vals, s2_vals],
            configs_per_stage=[s1_cfg, s2_cfg],
            stream=0,
        )

        dW_out = torch.empty_like(dW_t)
        rt.memcpy_d2h(_u8(dW_out), dW_dev, dW_t.nbytes)
    finally:
        rt.free(dY_dev)
        rt.free(X_dev)
        rt.free(dW_dev)
        rt.free(ws_dev)

    return dW_out.float()


def _run_two_stage_grouped(spec, arch, rt, dY_t, X_t):
    """Compile and launch two-stage pipeline for grouped conv; return dW as CPU fp32.

    Workspace shape: ``[groups*split_k, wg_M, wg_N]`` f32, where wg_M and wg_N
    are the per-group GEMM dimensions (kpg and Y*X*cpg).  Stage 1 covers all
    groups in one launch (z = group*split_k + k_id).  Stage 2 runs once per
    group with ws_ptr and dw_ptr shifted to that group's region.
    """
    import torch
    from rocke.instances.common.conv_implicit_gemm_wgrad_two_stage import (
        build_implicit_gemm_conv_wgrad_two_stage,
    )
    from rocke.instances.common.conv_wgrad_workspace_reduce import (
        WgradReduceSpec,
        wgrad_reduce_grid,
    )
    from rocke.runtime.launcher import LaunchConfig, PipelineLauncher

    p = spec.problem
    groups = p.groups
    # dW packed shape for grouped: [K, Y, X, cpg].
    dW_t = torch.zeros(p.K, p.Y, p.X, p.cpg, dtype=torch.float16)

    pipeline, ws_nbytes = build_implicit_gemm_conv_wgrad_two_stage(spec, arch=arch)

    dY_dev = rt.alloc(dY_t.nbytes)
    X_dev = rt.alloc(X_t.nbytes)
    dW_dev = rt.alloc(dW_t.nbytes)
    ws_dev = rt.alloc(ws_nbytes)

    try:
        rt.memcpy_h2d(dY_dev, _u8(dY_t), dY_t.nbytes)
        rt.memcpy_h2d(X_dev, _u8(X_t), X_t.nbytes)
        rt.memset(dW_dev, 0, dW_t.nbytes)
        rt.memset(ws_dev, 0, ws_nbytes)

        s2_spec = WgradReduceSpec(problem=spec.problem, dtype_d=spec.data.dtype_d)
        s2_grid = wgrad_reduce_grid(s2_spec)
        s2_block = (s2_spec.block_size, 1, 1)

        # Stage 1: one launch covering all groups (z = group*split_k + k_id).
        s1_grid = (
            (spec.wg_N + spec.tile_n - 1) // spec.tile_n,
            (spec.wg_M + spec.tile_m - 1) // spec.tile_m,
            groups * spec.split_k,
        )
        s1_block = (spec.block_size, 1, 1)
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
        s1_cfg = LaunchConfig(grid=s1_grid, block=s1_block, stream=0, fence=False)

        # Stage 2: one launch per group.
        # Group g's workspace region starts at g*split_k*wg_M*wg_N f32 elements.
        # Group g's dW region starts at g*wg_M*wg_N dtype_d elements.
        per_group_ws_bytes = spec.split_k * spec.wg_M * spec.wg_N * 4
        per_group_dw_bytes = dW_t.nbytes // groups

        s1_launcher = pipeline._stages[0]
        s2_launcher = pipeline._stages[1]
        all_launchers = [s1_launcher] + [s2_launcher] * groups
        all_vals = [s1_vals]
        all_cfgs = [s1_cfg]
        for g in range(groups):
            all_vals.append(
                {
                    "ws_ptr": ws_dev + g * per_group_ws_bytes,
                    "dw_ptr": dW_dev + g * per_group_dw_bytes,
                    "wg_M": spec.wg_M,
                    "wg_N": spec.wg_N,
                    "split_k": spec.split_k,
                    "ws_bytes": per_group_ws_bytes,
                    "dw_bytes": per_group_dw_bytes,
                }
            )
            all_cfgs.append(
                LaunchConfig(
                    grid=s2_grid, block=s2_block, stream=0, fence=(g == groups - 1)
                )
            )

        PipelineLauncher(all_launchers)(
            values_per_stage=all_vals,
            configs_per_stage=all_cfgs,
            stream=0,
        )

        dW_out = torch.empty_like(dW_t)
        rt.memcpy_d2h(_u8(dW_out), dW_dev, dW_t.nbytes)
    finally:
        rt.free(dY_dev)
        rt.free(X_dev)
        rt.free(dW_dev)
        rt.free(ws_dev)

    return dW_out.float()


def _make_spec(arch, N=2, Hi=8, Wi=8, C=16, K=32, Y=3, X=3, split_k=4, groups=1):
    """Build a WgradConvSpec with two_stage=True."""
    from rocke.instances.common._conv_implicit_gemm_common import (
        ConvDataSpec,
        ConvProblem,
    )
    from rocke.instances.common.conv_implicit_gemm_wgrad import WgradConvSpec

    p = ConvProblem(N=N, Hi=Hi, Wi=Wi, C=C, K=K, Y=Y, X=X, groups=groups)
    # 16x16x16 MFMA atoms work on both gfx942 and gfx950.
    return WgradConvSpec(
        problem=p,
        data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"),
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
        split_k=split_k,
        two_stage=True,
    )


@unittest.skipUnless(_ARCH in _CDNA_ARCHES and _HAS_TORCH, _SKIP_REASON)
class TestWgradTwoStageNumeric(unittest.TestCase):
    """End-to-end GPU numerical tests for the two-stage deterministic wgrad path."""

    ARCH = _ARCH
    TOL_FP16 = 5e-2  # relative error tolerance for fp16 kernels

    @classmethod
    def setUpClass(cls):
        cls.rt = Runtime()

    def test_two_stage_matches_reference_fp16(self):
        """Stage 1 + Stage 2 output matches CPU torch reference within fp16 tolerance."""
        import torch

        spec = _make_spec(self.ARCH)
        p = spec.problem

        torch.manual_seed(42)
        X_f32 = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1.0, 1.0)
        dY_f32 = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0)
        X_t = X_f32.half()
        dY_t = dY_f32.half()

        dW_ours = _run_two_stage(spec, self.ARCH, self.rt, dY_t, X_t)
        dW_ref = _cpu_wgrad_reference(X_f32, dY_f32, p)

        max_abs = dW_ref.abs().max().item()
        if max_abs < 1e-6:
            self.skipTest("reference output is near-zero — problem degenerate")

        rel_err = (dW_ours - dW_ref).abs().max().item() / max_abs
        self.assertLess(
            rel_err,
            self.TOL_FP16,
            f"two-stage fp16 relative error {rel_err:.3e} >= tol {self.TOL_FP16}",
        )

    def test_two_stage_is_deterministic(self):
        """Two consecutive runs on identical inputs produce bit-exact output."""
        import torch

        spec = _make_spec(self.ARCH)
        p = spec.problem

        torch.manual_seed(99)
        X_t = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1.0, 1.0).half()
        dY_t = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0).half()

        dW_run1 = _run_two_stage(spec, self.ARCH, self.rt, dY_t, X_t)
        dW_run2 = _run_two_stage(spec, self.ARCH, self.rt, dY_t, X_t)

        self.assertTrue(
            torch.equal(dW_run1, dW_run2),
            f"two-stage output is not bit-exact across runs; "
            f"max diff = {(dW_run1 - dW_run2).abs().max().item():.3e}",
        )

    def test_two_stage_split_k_8(self):
        """split_k=8 exercises more workspace slices; output still matches reference."""
        import torch

        spec = _make_spec(self.ARCH, N=1, Hi=14, Wi=14, C=32, K=32, Y=3, X=3, split_k=8)
        p = spec.problem

        torch.manual_seed(7)
        X_f32 = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1.0, 1.0)
        dY_f32 = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0)
        X_t = X_f32.half()
        dY_t = dY_f32.half()

        dW_ours = _run_two_stage(spec, self.ARCH, self.rt, dY_t, X_t)
        dW_ref = _cpu_wgrad_reference(X_f32, dY_f32, p)

        max_abs = dW_ref.abs().max().item()
        rel_err = (dW_ours - dW_ref).abs().max().item() / max(max_abs, 1e-6)
        self.assertLess(
            rel_err,
            self.TOL_FP16,
            f"split_k=8 rel_err {rel_err:.3e} >= tol {self.TOL_FP16}",
        )

    def test_non_tile_aligned_shape_matches_reference(self):
        """Non-tile-aligned wg_M (K=40, not divisible by tile_m=64) produces correct output.

        This exercises Stage 1's OOB guard: the single M-tile covers rows 0-63 but
        wg_M=40, so rows 40-63 are skipped by the scf_if guard.  Stage 2 reads only
        rows 0-39 (workspace is allocated as wg_M*wg_N, not tile_m*wg_N), so the
        unwritten rows are never accessed.  The result must still match the reference.
        """
        import torch

        spec = _make_spec(self.ARCH, K=40)
        self.assertNotEqual(spec.wg_M % spec.tile_m, 0, "expected non-aligned wg_M")
        p = spec.problem

        torch.manual_seed(55)
        X_f32 = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1.0, 1.0)
        dY_f32 = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0)
        X_t = X_f32.half()
        dY_t = dY_f32.half()

        dW_ours = _run_two_stage(spec, self.ARCH, self.rt, dY_t, X_t)
        dW_ref = _cpu_wgrad_reference(X_f32, dY_f32, p)

        max_abs = dW_ref.abs().max().item()
        rel_err = (dW_ours - dW_ref).abs().max().item() / max(max_abs, 1e-6)
        self.assertLess(
            rel_err,
            self.TOL_FP16,
            f"non-aligned wg_M={spec.wg_M} rel_err {rel_err:.3e} >= tol {self.TOL_FP16}",
        )

    def test_two_stage_grouped_matches_reference(self):
        """groups=2, split_k=4 two-stage output matches CPU torch reference.

        Verifies the workspace partitioning for grouped conv:
        workspace shape [groups*split_k, wg_M, wg_N] where wg_M=kpg and
        wg_N=Y*X*cpg are per-group dimensions.  Stage 1 launches with
        z=groups*split_k; z=group*split_k+k_id uniquely indexes each region.
        Stage 2 runs once per group with shifted ws_ptr/dw_ptr pointers.
        """
        import torch

        # C=16, K=16, groups=2 → cpg=8 (even ✓), kpg=8.
        # Y=3,X=3 avoids the grouped-pointwise restriction.
        spec = _make_spec(
            self.ARCH, N=2, Hi=8, Wi=8, C=16, K=16, Y=3, X=3, split_k=4, groups=2
        )
        self.assertEqual(spec.problem.groups, 2)
        p = spec.problem

        torch.manual_seed(13)
        X_f32 = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1.0, 1.0)
        dY_f32 = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0)
        X_t = X_f32.half()
        dY_t = dY_f32.half()

        dW_ours = _run_two_stage_grouped(spec, self.ARCH, self.rt, dY_t, X_t)
        dW_ref = _cpu_wgrad_reference(X_f32, dY_f32, p)

        max_abs = dW_ref.abs().max().item()
        if max_abs < 1e-6:
            self.skipTest("reference output is near-zero — problem degenerate")

        rel_err = (dW_ours - dW_ref).abs().max().item() / max_abs
        self.assertLess(
            rel_err,
            self.TOL_FP16,
            f"grouped two-stage rel_err {rel_err:.3e} >= tol {self.TOL_FP16}",
        )

    def test_two_stage_grouped_g3_and_g11(self):
        """groups=3 and groups=11 with split_k=4 match the CPU torch reference.

        Exercises non-power-of-two group counts where wg_M and wg_N are small
        relative to the tile (wg_M=kpg << tile_m), stressing the OOB guard
        path in the Stage 1 epilogue.

        Problem shapes:
          groups=3:  C=24, K=24 → cpg=8, kpg=8,  wg_N=72
          groups=11: C=44, K=44 → cpg=4, kpg=4,  wg_N=36
        """
        import torch

        configs = [
            dict(groups=3, C=24, K=24),
            dict(groups=11, C=44, K=44),
        ]
        for cfg in configs:
            with self.subTest(**cfg):
                spec = _make_spec(
                    self.ARCH,
                    N=2,
                    Hi=8,
                    Wi=8,
                    C=cfg["C"],
                    K=cfg["K"],
                    Y=3,
                    X=3,
                    split_k=4,
                    groups=cfg["groups"],
                )
                p = spec.problem

                torch.manual_seed(cfg["groups"])
                X_f32 = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1.0, 1.0)
                dY_f32 = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0)

                dW_ours = _run_two_stage_grouped(
                    spec, self.ARCH, self.rt, dY_f32.half(), X_f32.half()
                )
                dW_ref = _cpu_wgrad_reference(X_f32, dY_f32, p)

                max_abs = dW_ref.abs().max().item()
                if max_abs < 1e-6:
                    self.skipTest("reference output is near-zero — problem degenerate")

                rel_err = (dW_ours - dW_ref).abs().max().item() / max_abs
                self.assertLess(
                    rel_err,
                    self.TOL_FP16,
                    f"groups={cfg['groups']} rel_err {rel_err:.3e} >= tol {self.TOL_FP16}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
