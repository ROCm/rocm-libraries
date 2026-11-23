# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU correctness for the native-NCHW wgrad prototype (experimental spike).

:mod:`kernels.common.conv_wgrad_nchw_proto` is not a production kernel family
-- see its module docstring -- so this test checks what a spike needs: every
knob combination compiles, launches, and matches ``torch.nn.grad.conv2d_weight``
computed directly in NCHW / NKHW / KCYX (torch's native layouts, no permute on
either side).

Tolerance: max |dW - ref| / max(max |ref|, 1) below ``_TOL``, the same metric
and the same fp16 / bf16 bound ``test_conv_wgrad_correctness.py`` uses for the
production wgrad split-K kernels (each split slice rounds its partial to 16
bits before the packed atomic add, as there).

Run as a script if pytest reports "No HIP GPUs are available"::

    python library/tests/test_conv_wgrad_nchw_proto.py
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest
from dataclasses import replace

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch

from kernels.common.conv_direct_grouped import DirectConvProblem  # noqa: E402
from kernels.common.conv_wgrad_nchw_proto import (  # noqa: E402
    WgradNchwProtoSpec,
    build_conv_wgrad_nchw_proto,
    is_valid_wgrad_nchw_proto_spec,
    wgrad_nchw_proto_launch_values,
    wgrad_nchw_proto_signature,
)

_TOL = {"fp16": 5e-2, "bf16": 5e-2}  # test_conv_wgrad_correctness.py's _TOL

GPU_ARCH = get_device_arch(0)


def _gpu_skip_reason() -> str:
    if not GPU_ARCH:
        return "no ROCm GPU detected"
    if not _HAS_TORCH:
        return "torch not importable"
    if GPU_ARCH != "gfx950":
        return f"prototype targets gfx950 (got {GPU_ARCH!r})"
    return ""


_GPU_SKIP = _gpu_skip_reason()


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _problem(N, C, K, H, W, stride=1, k=3, dtype="fp16"):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=1,
        cpg=C,
        kpg=K,
        KH=k,
        KW=k,
        PAD=k // 2,
        stride=stride,
        dtype=dtype,
    )


def reference_wgrad_nchw(X_t, dY_t, p: DirectConvProblem):
    """IEEE fp32 weight gradient on the CPU, native NCHW in, KCYX out."""
    return torch.nn.grad.conv2d_weight(
        X_t.float(),
        (p.total_k, p.cpg, p.KH, p.KW),
        dY_t.float(),
        stride=p.stride,
        padding=p.PAD,
    )


def run_wgrad_nchw_proto(arch: str, spec: WgradNchwProtoSpec, split_k: int = 1):
    """Compile, launch and check one spec; returns ``(ok, rel_err)``."""
    from rocke import compile_kernel
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    p = spec.problem
    ok, why = is_valid_wgrad_nchw_proto_spec(spec, arch=arch)
    if not ok:
        raise AssertionError(f"invalid spec {spec.kernel_name()}: {why}")
    art = compile_kernel(build_conv_wgrad_nchw_proto(spec, arch=arch), arch=arch)

    td = torch.bfloat16 if p.dtype == "bf16" else torch.float16
    torch.manual_seed(0)
    X_t = torch.empty(p.N, p.total_c, p.H, p.W).uniform_(-1.0, 1.0).to(td)
    dY_t = torch.empty(p.N, p.total_k, p.Ho, p.Wo).uniform_(-1.0, 1.0).to(td)
    ref = reference_wgrad_nchw(X_t, dY_t, p)
    dW_t = torch.zeros(p.total_k, p.cpg, p.KH, p.KW, dtype=td)

    rt = Runtime()
    bufs = [rt.alloc(t.nbytes) for t in (dY_t, X_t, dW_t)]
    try:
        rt.memcpy_h2d(bufs[0], _u8(dY_t), dY_t.nbytes)
        rt.memcpy_h2d(bufs[1], _u8(X_t), X_t.nbytes)
        rt.memset(bufs[2], 0, dW_t.nbytes)  # the atomic epilogue accumulates
        launcher = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=wgrad_nchw_proto_signature(spec),
        )
        launcher(
            wgrad_nchw_proto_launch_values(
                bufs[0], bufs[1], bufs[2], dY_t.nbytes, X_t.nbytes, split_k
            ),
            config=LaunchConfig(
                grid=spec.grid(split_k), block=(spec.threads, 1, 1), fence=True
            ),
        )
        out = torch.empty_like(dW_t)
        rt.memcpy_d2h(_u8(out), bufs[2], dW_t.nbytes)
    finally:
        for buf in bufs:
            rt.free(buf)
        synchronize_and_release(0)

    out = out.float()
    if not torch.isfinite(out).all():
        return False, float("inf")
    rel = float((out - ref).abs().max() / ref.abs().max().clamp(min=1.0))
    return rel < _TOL[p.dtype], rel


@unittest.skipIf(bool(_GPU_SKIP), _GPU_SKIP or "gpu ok")
class TestWgradNchwProto(unittest.TestCase):
    def _check(self, spec, split_k=1):
        with self.subTest(spec=spec.kernel_name(), split_k=split_k):
            ok, rel = run_wgrad_nchw_proto(GPU_ARCH, spec, split_k)
            self.assertTrue(ok, f"{spec.kernel_name()} spk{split_k}: rel_err={rel:.3e}")

    def test_fp16_3x3_shapes(self):
        """Default knobs over stride, batch, channel counts and image size."""
        for stride in (1, 2):
            for N in (1, 2):
                for C, K in ((32, 64), (64, 64)):
                    for HW in (16, 32):
                        p = _problem(N, C, K, HW, HW, stride=stride)
                        spec = WgradNchwProtoSpec(problem=p, tile_m=64, tile_c=32)
                        self._check(spec, split_k=1)

    def test_fp16_knobs(self):
        """Atom, wave split, reduction depth, LDS pad and split-K degree."""
        for stride in (1, 2):
            p = _problem(2, 64, 64, 32, 32, stride=stride)
            variants = [
                dict(atom="16x16x32", tile_c=16, tile_m=32, waves_m=2, tile_r=64),
                dict(atom="32x32x16", waves_m=2, waves_n=1, tile_r=128),
                dict(atom="32x32x16", waves_m=1, waves_n=3, tile_m=32, tile_r=96),
                dict(atom="32x32x16", waves_m=2, waves_n=3, tile_r=96, lds_pad=0),
                dict(atom="16x16x32", tile_c=32, tile_m=64, waves_m=4, tile_r=128),
                dict(atom="16x16x32", tile_c=16, tile_m=32, waves_m=2, y_groups=3),
                dict(
                    atom="32x32x16",
                    waves_m=1,
                    waves_n=3,
                    tile_m=32,
                    tile_r=96,
                    y_groups=3,
                ),
            ]
            for kw in variants:
                spec = WgradNchwProtoSpec(problem=p, **kw)
                for sk in (1, 3, 8):
                    self._check(spec, split_k=sk)

    def test_fp16_reduction_tail_and_nonsquare(self):
        """Reduction not a multiple of tile_r, H != W, split past the step count."""
        for stride in (1, 2):
            p = _problem(1, 32, 64, 12 * stride, 16 * stride, stride=stride)
            spec = WgradNchwProtoSpec(problem=p, tile_m=64, tile_c=32, tile_r=128)
            for sk in (1, 2, 64):
                self._check(spec, split_k=sk)

    def test_fp16_5x5(self):
        for stride in (1, 2):
            p = _problem(1, 32, 32, 32, 32, stride=stride, k=5)
            for kw in (
                dict(waves_m=1, waves_n=5, tile_m=32, tile_r=80),
                dict(atom="16x16x32", tile_c=16, tile_m=16, waves_m=1),
                dict(atom="16x16x32", tile_c=16, tile_m=16, waves_m=1, y_groups=5),
            ):
                spec = WgradNchwProtoSpec(problem=p, **kw)
                self._check(spec, split_k=4)

    def test_bf16(self):
        for stride in (1, 2):
            p = _problem(2, 64, 64, 32, 32, stride=stride, dtype="bf16")
            spec = WgradNchwProtoSpec(problem=p)
            self._check(spec, split_k=4)
            self._check(replace(spec, atom="16x16x32", tile_c=16, tile_m=32), split_k=1)


if __name__ == "__main__":
    unittest.main()
