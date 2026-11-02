# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The implicit-GEMM sweep's stream-K wgrad launch helper.

``benchmark_implicit_gemm_conv._streamk_launch`` sets up one stream-K launch
for the JIT sweep and for ``--run-from-cache``: it sizes and allocates the
reduction's workspace, fills the kernargs and wraps the per-reduction zeroing
contract (and the Stage 2 fold for ``workspace``) in a ``run()`` closure.

The case pinned here is a partition with no stream-K CTAs at all: when every
output tile is data-parallel the workspace is zero bytes, and the helper must
still hand the kernel valid flag / partial pointers instead of failing (an
uncaught error there aborts the whole sweep). The host tests drive the helper
with a recording runtime; the GPU test launches the real kernel through it
and checks dW, and skips without a gfx942/gfx950 GPU and torch.
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest

from rocke.runtime.hip_module import get_device_arch

from benchmarks.common.benchmark_implicit_gemm_conv import _streamk_launch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
GPU_ARCH = get_device_arch(0)

# Every output tile is data-parallel under the default gfx950 pool: 1 x 9
# tiles of 64x64 over K=64, Y*X*C=576, so the stream-K CTA count is 0.
_ALL_DP = dict(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1)
# 459 tiles: more than the default pool, so stream-K CTAs share some of them.
_SPLITS = dict(N=2, Hi=12, Wi=12, C=192, K=1056, Y=3, X=3, pH=1, pW=1)
# (reduction, dW dtype): "atomic" needs fp32 dW.
_REDUCTIONS = (
    ("linear", "fp16"),
    ("tree", "fp16"),
    ("atomic", "fp32"),
    ("workspace", "bf16"),
)


def _spec(shape, *, arch, reduction, dtype_d, mode="dp_sk"):
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
    from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

    ab = "bf16" if dtype_d == "bf16" else "fp16"
    edge = 32 if arch == "gfx950" else 16
    return WgradConvSpec(
        problem=ConvProblem(**shape),
        data=ConvDataSpec(dtype_a=ab, dtype_b=ab, dtype_d=dtype_d),
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=edge,
        warp_tile_n=edge,
        warp_tile_k=16,
        streamk=mode,
        streamk_reduction=reduction,
    )


class _RecordingRuntime:
    """The subset of ``Runtime`` the helper uses, recording every call."""

    def __init__(self):
        self.allocs: dict = {}
        self.freed: list = []
        self.memsets: list = []
        self._next = 0x10000

    def alloc(self, nbytes):
        ptr = self._next
        self._next += max(nbytes, 1) + 0x1000
        self.allocs[ptr] = nbytes
        return ptr

    def free(self, ptr):
        self.freed.append(ptr)

    def memset(self, ptr, value, nbytes):
        self.memsets.append((ptr, value, nbytes))


class _RecordingLauncher:
    def __init__(self):
        self.calls: list = []

    def __call__(self, values, *, config):
        self.calls.append((values, config))


def _launch_host(spec, arch):
    from rocke.runtime.launcher import LaunchConfig

    rt = _RecordingRuntime()
    launcher = _RecordingLauncher()
    stage2 = _RecordingLauncher()
    # Pre-seed the Stage 2 memo so the workspace reduction needs no compiler.
    stage2_memo = {(spec.data.dtype_d, spec.ws_replicas): stage2}
    run, free, pool = _streamk_launch(
        rt=rt,
        launcher=launcher,
        spec=spec,
        problem=spec.problem,
        arch=arch,
        ptrs=(0x100, 0x200, 0x300),
        sizes=(64, 64, 64),
        stage2=stage2_memo,
        KernelLauncher=None,
        LaunchConfig=LaunchConfig,
    )
    return rt, launcher, stage2, run, free, pool


class TestStreamKSweepLaunchHost(unittest.TestCase):
    """Host-only: the helper's workspace handling, with a recording runtime."""

    ARCH = "gfx950"

    def test_all_dp_partition_has_no_workspace(self):
        # The premise of the other cases: this shape takes no stream-K CTAs.
        from kernels.common.conv_implicit_gemm_wgrad import (
            wgrad_streamk_partition,
            wgrad_streamk_workspace_nbytes,
        )

        spec = _spec(_ALL_DP, arch=self.ARCH, reduction="linear", dtype_d="fp16")
        part = wgrad_streamk_partition(spec, arch=self.ARCH)
        self.assertEqual(part.sk_ctas, 0)
        self.assertEqual(wgrad_streamk_workspace_nbytes(spec, arch=self.ARCH), 0)

    def test_all_dp_partition_launches(self):
        for reduction, dtype_d in _REDUCTIONS:
            with self.subTest(reduction=reduction):
                spec = _spec(
                    _ALL_DP, arch=self.ARCH, reduction=reduction, dtype_d=dtype_d
                )
                rt, launcher, stage2, run, free, _ = _launch_host(spec, self.ARCH)
                # One (stub) allocation, never a zero-byte one.
                self.assertEqual(len(rt.allocs), 1)
                (ws,) = rt.allocs
                self.assertGreaterEqual(rt.allocs[ws], 4)
                run()
                self.assertEqual(len(launcher.calls), 1)
                values, _ = launcher.calls[0]
                if reduction in ("linear", "tree"):
                    self.assertEqual(values["sk_flags"], ws)
                    self.assertEqual(values["sk_partials"], ws)
                self.assertEqual(len(stage2.calls), reduction == "workspace")
                free()
                self.assertEqual(rt.freed, [ws])

    def test_split_partition_lays_out_flags_then_partials(self):
        from kernels.common.conv_implicit_gemm_wgrad import (
            wgrad_streamk_partition,
            wgrad_streamk_workspace_layout,
        )

        spec = _spec(_SPLITS, arch=self.ARCH, reduction="linear", dtype_d="fp16")
        self.assertGreater(wgrad_streamk_partition(spec, arch=self.ARCH).sk_ctas, 0)
        flags_bytes, partials_bytes = wgrad_streamk_workspace_layout(
            spec, arch=self.ARCH
        )
        self.assertGreater(flags_bytes, 0)
        rt, launcher, _, run, free, _ = _launch_host(spec, self.ARCH)
        (ws,) = rt.allocs
        self.assertEqual(rt.allocs[ws], flags_bytes + partials_bytes)
        run()
        values, _ = launcher.calls[0]
        self.assertEqual(values["sk_flags"], ws)
        self.assertEqual(values["sk_partials"], ws + flags_bytes)
        # linear/tree reset only the flag table before each launch.
        self.assertEqual(rt.memsets, [(ws, 0, flags_bytes)])
        free()
        self.assertEqual(rt.freed, [ws])


def _gpu_skip_reason() -> str:
    if not GPU_ARCH:
        return "no ROCm GPU detected"
    if not _HAS_TORCH:
        return "torch not importable"
    if GPU_ARCH not in ("gfx942", "gfx950"):
        return f"stream-K wgrad is gfx942/gfx950 only; got {GPU_ARCH}"
    return ""


_GPU_SKIP_REASON = _gpu_skip_reason()


@unittest.skipUnless(not _GPU_SKIP_REASON, _GPU_SKIP_REASON or "no GPU")
class TestStreamKSweepLaunchGpu(unittest.TestCase):
    """The helper's ``run()`` on the GPU for the all-data-parallel shape."""

    def _check(self, reduction, dtype_d):
        import torch

        from rocke import compile_kernel
        from rocke.runtime import synchronize_and_release
        from rocke.runtime.hip_module import Runtime
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig
        from kernels.common.conv_implicit_gemm_wgrad import (
            build_implicit_gemm_conv_wgrad,
            wgrad_streamk_signature,
        )

        arch = GPU_ARCH
        spec = _spec(_ALL_DP, arch=arch, reduction=reduction, dtype_d=dtype_d)
        p = spec.problem
        art = compile_kernel(build_implicit_gemm_conv_wgrad(spec, arch=arch), arch=arch)
        launcher = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=wgrad_streamk_signature(spec),
        )
        dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        torch.manual_seed(0)
        X = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1, 1).to(dt[spec.data.dtype_b])
        dY = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1, 1).to(dt[spec.data.dtype_a])
        dW = torch.empty(p.K, p.Y, p.X, p.C, dtype=dt[dtype_d])
        ref = torch.nn.grad.conv2d_weight(
            X.float().permute(0, 3, 1, 2),
            weight_size=(p.K, p.C, p.Y, p.X),
            grad_output=dY.float().permute(0, 3, 1, 2),
            padding=(p.pH, p.pW),
        ).permute(0, 2, 3, 1)

        def u8(t):
            return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

        rt = Runtime()
        bufs = [rt.alloc(t.nbytes) for t in (dY, X, dW)]
        try:
            rt.memcpy_h2d(bufs[0], u8(dY), dY.nbytes)
            rt.memcpy_h2d(bufs[1], u8(X), X.nbytes)
            # Poison dW: a reduction that leaves it alone must overwrite it.
            rt.memset(bufs[2], 0xFF if reduction != "atomic" else 0, dW.nbytes)
            run, free, _ = _streamk_launch(
                rt=rt,
                launcher=launcher,
                spec=spec,
                problem=p,
                arch=arch,
                ptrs=tuple(int(b) for b in bufs),
                sizes=(dY.nbytes, X.nbytes, dW.nbytes),
                stage2={},
                KernelLauncher=KernelLauncher,
                LaunchConfig=LaunchConfig,
            )
            try:
                run()
                synchronize_and_release(0)
            finally:
                free()
            rt.memcpy_d2h(u8(dW), bufs[2], dW.nbytes)
        finally:
            for b in bufs:
                rt.free(b)
        out = dW.float()
        rel = float((out - ref).abs().max() / ref.abs().max().clamp(min=1.0))
        tol = {"fp32": 1e-4, "fp16": 2e-3, "bf16": 2e-2}[dtype_d]
        self.assertLess(rel, tol, f"{reduction}: rel_err={rel:.3e}")

    def test_all_dp_linear(self):
        self._check("linear", "fp16")

    def test_all_dp_workspace(self):
        self._check("workspace", "bf16")


if __name__ == "__main__":
    unittest.main()
