# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU correctness for the native-NCHW backward-data (dgrad) direct convolution
family.

:mod:`kernels.common.conv_direct_nchw_dgrad` is AOT-buildable and C++-mirrored
(see its module docstring) but not yet wired into runtime dispatch. This test
checks that it compiles, launches and matches
``torch.nn.grad.conv2d_input`` computed directly in NCHW (dY NKHW,
W KCYX, dX NCHW -- no host transpose of the activations; the weights are
repacked once on the host, as a plan-time step would).

Tolerance: max |dX - ref| / max(|ref|, 1) below 5e-3 (fp16) / 1.5e-2 (bf16),
the forward NCHW family's bound and stricter than the 2e-2
relative-to-peak bound ``test_conv_dgrad_correctness.py`` applies to dgrad.

Run as a script if pytest reports "No HIP GPUs are available"::

    python library/tests/test_conv_direct_nchw_dgrad.py
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

    torch.cuda.is_available()

from kernels.common.conv_direct_grouped import DirectConvProblem  # noqa: E402
from kernels.common.conv_direct_nchw_dgrad import (  # noqa: E402
    DirectNchwDgradSpec,
    build_direct_conv_nchw_dgrad,
    is_valid_direct_nchw_dgrad_spec,
    phase_taps,
    repack_weights_cyxk,
)

_ARCHES = ("gfx950", "gfx942")
TOL = {"fp16": 5e-3, "bf16": 1.5e-2}

GPU_ARCH = get_device_arch(0)


def _gpu_skip_reason() -> str:
    if not GPU_ARCH:
        return "no ROCm GPU detected"
    if not _HAS_TORCH:
        return "torch not importable"
    if GPU_ARCH not in _ARCHES:
        return f"unsupported arch {GPU_ARCH!r}"
    return ""


_GPU_SKIP = _gpu_skip_reason()


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def dgrad_reference(dY, W, p: DirectConvProblem):
    """fp32 CPU ``conv2d_input`` in native NCHW."""
    with torch.no_grad():
        return torch.nn.grad.conv2d_input(
            (p.N, p.total_c, p.H, p.W),
            W.float(),
            dY.float(),
            stride=p.stride,
            padding=p.PAD,
        )


def run_dgrad_nchw_spec(arch: str, spec: DirectNchwDgradSpec, artifact=None):
    """Compile, launch and check one spec; returns ``(ok, rel_err)``."""
    from rocke import compile_kernel
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    p = spec.problem
    ok, why = is_valid_direct_nchw_dgrad_spec(spec, arch=arch)
    if not ok:
        raise AssertionError(f"invalid spec {spec.kernel_name()}: {why}")
    if artifact is None:
        artifact = compile_kernel(
            build_direct_conv_nchw_dgrad(spec, arch=arch), arch=arch
        )

    td = torch.bfloat16 if p.dtype == "bf16" else torch.float16
    torch.manual_seed(0)
    dY = torch.empty(p.N, p.total_k, p.Ho, p.Wo, dtype=td).uniform_(-1.0, 1.0)
    W = torch.empty(p.total_k, p.total_c, p.KH, p.KW, dtype=td).uniform_(-1.0, 1.0)
    dX = torch.empty(p.N, p.total_c, p.H, p.W, dtype=td)
    ref = dgrad_reference(dY, W, p)
    W_dev = repack_weights_cyxk(W)

    rt = Runtime()
    bufs = [rt.alloc(t.nbytes) for t in (dY, W_dev, dX)]
    try:
        rt.memcpy_h2d(bufs[0], _u8(dY), dY.nbytes)
        rt.memcpy_h2d(bufs[1], _u8(W_dev), W_dev.nbytes)
        dX.fill_(float("nan"))  # an unwritten element cannot pass as 0
        rt.memcpy_h2d(bufs[2], _u8(dX), dX.nbytes)
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=conv_direct_args_signature(p.dtype),
        )
        launcher(
            ConvArgs.from_problem(p).to_launch_values(
                int(bufs[0]),
                int(bufs[1]),
                int(bufs[2]),
                dY.nbytes,
                W_dev.nbytes,
                dX.nbytes,
            ),
            config=LaunchConfig(
                grid=spec.grid(), block=(spec.threads_per_block, 1, 1), fence=True
            ),
        )
        out = torch.empty_like(dX)
        rt.memcpy_d2h(_u8(out), bufs[2], dX.nbytes)
    finally:
        for buf in bufs:
            rt.free(buf)
        synchronize_and_release(0)
    if not torch.isfinite(out.float()).all():
        return False, float("inf")
    rel = float((out.float() - ref).abs().max() / ref.abs().max().clamp(min=1.0))
    return rel < TOL[p.dtype], rel


def _problem(N, H, W, C, K, F=3, P=1, stride=1, dtype="fp16"):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=1,
        cpg=C,
        kpg=K,
        KH=F,
        KW=F,
        PAD=P,
        stride=stride,
        dtype=dtype,
    )


def _spec(p, **kw):
    base = dict(tile_i=8, tile_j=32, tile_c=64, ck=32, waves_m=2, waves_n=2)
    if p.stride == 2:
        base.update(tile_i=4, waves_n=2, tile_c=64)
    base.update(kw)
    return DirectNchwDgradSpec(problem=p, **base)


class TestDgradNchwProtoHost(unittest.TestCase):
    def test_phase_taps(self):
        # 3x3 pad 1: stride 1 is one phase with offsets +1/0/-1; stride 2 is
        # tap 1 on even rows and taps 0 (+1) / 2 (0) on odd rows.
        self.assertEqual(phase_taps(3, 1, 1), [[(0, 1), (1, 0), (2, -1)]])
        self.assertEqual(phase_taps(3, 1, 2), [[(1, 0)], [(0, 1), (2, 0)]])
        self.assertEqual(sum(len(t) for t in phase_taps(5, 2, 2)), 5)

    def test_validation(self):
        ok, _ = is_valid_direct_nchw_dgrad_spec(
            _spec(_problem(1, 16, 16, 64, 64)), "gfx950"
        )
        self.assertTrue(ok)
        ok, why = is_valid_direct_nchw_dgrad_spec(
            _spec(_problem(1, 16, 12, 64, 64)), "gfx950"
        )
        self.assertFalse(ok)
        self.assertIn("Wo", why)
        ok, why = is_valid_direct_nchw_dgrad_spec(
            _spec(_problem(1, 16, 16, 64, 64, dtype="fp32")), "gfx950"
        )
        self.assertFalse(ok)

    def test_build_and_lower(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        for stride in (1, 2):
            spec = _spec(_problem(1, 32, 32, 64, 64, stride=stride))
            ll = lower_kernel_to_llvm(
                build_direct_conv_nchw_dgrad(spec, "gfx950"), arch="gfx950"
            )
            self.assertIn("llvm.amdgcn.mfma", ll)


@unittest.skipUnless(not _GPU_SKIP, _GPU_SKIP or "no GPU")
class TestDgradNchwProtoGpu(unittest.TestCase):
    """On silicon: dX against torch's native-NCHW ``conv2d_input``."""

    def _check(self, p, **kw):
        spec = _spec(p, **kw)
        ok, rel = run_dgrad_nchw_spec(GPU_ARCH, spec)
        tag = f"{spec.kernel_name()} N{p.N}H{p.H}W{p.W}C{p.total_c}K{p.total_k}"
        print(f"  {'PASS' if ok else 'FAIL'} {tag} rel={rel:.2e}", flush=True)
        self.assertTrue(ok, f"{tag}: rel_err {rel:.3e}")

    def test_3x3_stride1(self):
        # Edge tiles in h (H % tile_i) and w (W % tile_j), C != K, N > 1.
        for dtype in ("fp16", "bf16"):
            for N, H, W, C, K in [
                (1, 16, 32, 64, 64),
                (2, 20, 40, 64, 96),
                (1, 9, 24, 128, 32),
            ]:
                with self.subTest(dtype=dtype, shape=(N, H, W, C, K)):
                    self._check(_problem(N, H, W, C, K, stride=1, dtype=dtype))

    def test_3x3_stride2(self):
        # Even and odd H; W gives Wo % 8 == 0.
        for dtype in ("fp16", "bf16"):
            for N, H, W, C, K in [
                (1, 16, 16, 64, 64),
                (2, 23, 48, 64, 96),
                (1, 32, 80, 128, 32),
            ]:
                with self.subTest(dtype=dtype, shape=(N, H, W, C, K)):
                    self._check(_problem(N, H, W, C, K, stride=2, dtype=dtype))

    def test_5x5(self):
        for stride in (1, 2):
            for N, H, W, C, K in [(1, 16, 32, 64, 64), (2, 18, 48, 64, 32)]:
                with self.subTest(stride=stride, shape=(N, H, W, C, K)):
                    self._check(_problem(N, H, W, C, K, F=5, P=2, stride=stride))

    def test_knobs(self):
        """Every staging group, both atom sizes, wider tiles."""
        cases = []
        for stride in (1, 2):
            p = _problem(2, 24, 64, 64, 64, stride=stride)
            for g in (1, 2, 4, 8):
                cases.append((p, dict(stage_group=g)))
            cases.append((p, dict(atom="16x16x32", tile_j=32, waves_m=2, waves_n=2)))
            cases.append((p, dict(atom="16x16x16", ck=16)))
            cases.append((p, dict(tile_j=64, tile_i=2, waves_n=1, waves_m=2, ck=16)))
            cases.append((p, dict(lds_pad=0)))
        for p, kw in cases:
            with self.subTest(stride=p.stride, **kw):
                self._check(p, **kw)

    def test_one_binary_serves_every_shape(self):
        """The binary depends on N/H/W/C/K only through kernargs."""
        from rocke import compile_kernel

        for stride in (1, 2):
            spec0 = _spec(_problem(1, 16, 32, 64, 64, stride=stride))
            art = compile_kernel(
                build_direct_conv_nchw_dgrad(spec0, GPU_ARCH), arch=GPU_ARCH
            )
            for N, H, W, C, K in [(2, 32, 32, 128, 64), (1, 17, 48, 64, 128)]:
                case = replace(spec0, problem=_problem(N, H, W, C, K, stride=stride))
                with self.subTest(stride=stride, shape=(N, H, W, C, K)):
                    ok, rel = run_dgrad_nchw_spec(GPU_ARCH, case, artifact=art)
                    self.assertTrue(ok, f"rel_err {rel:.3e}")


if __name__ == "__main__":
    unittest.main()
