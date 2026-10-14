# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""4c (cpg = kpg = 4) direct conv: bf16 support.

Covers:

* CPU only: the bf16 4c kernel emits only the ``mfma_f32_4x4x4_bf16``
  (``_1k``) atom while the fp16 kernel keeps the f16 atom, and the kernel name
  carries the dtype;
* on-device numerics (gfx942 / gfx950 + torch): the 4c forward kernel in bf16
  over adversarial shapes (odd H / W, N = 1, W not a multiple of block_q, 1x1
  filters, several block_q / block_groups knobs), and one binary built on a
  probe problem launched on several shapes (the AOT contract: extents are
  kernargs).

Every launch goes through the direct forward ABI (``ConvArgs`` /
``conv_direct_args_signature``) with the grid of ``direct_launch_geometry``.

Elementwise rule (``manifest_runner/conv.py``): an output element is bad when
``|out - ref| > tol + tol * |ref|`` with ``tol = 1e-2``; a case passes iff no
element is bad. The output is NaN-filled before each launch, so an element the
kernel never writes is caught as bad too.

Run (GPU part as a script, as the other numeric suites):
    PYTHONPATH=rocke/platform/python:rocke/library python \\
        rocke/library/tests/test_conv_dgrad_4c.py
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest

from kernels.common.conv_direct_grouped import (
    DirectConv4cSpec,
    DirectConvProblem,
    build_direct_conv_4c,
    direct_launch_geometry,
)
from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    # Claim the HIP context for torch before rocke's runtime touches it (see
    # test_direct_conv_correctness.py).
    import torch

    torch.cuda.is_available()

GPU_ARCH = get_device_arch(0)
_GPU_SKIP = (
    ""
    if (GPU_ARCH in ("gfx942", "gfx950") and _HAS_TORCH)
    else f"needs gfx942/gfx950 + torch (arch={GPU_ARCH!r}, torch={_HAS_TORCH})"
)
_TOL = 1e-2

_F16_ATOM = "@llvm.amdgcn.mfma.f32.4x4x4f16("
_BF16_ATOM = "@llvm.amdgcn.mfma.f32.4x4x4bf16.1k("


def _problem(N, H, W, groups, dtype="bf16", K=3, pad=None, stride=1, cpg=4, kpg=4):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=cpg,
        kpg=kpg,
        KH=K,
        KW=K,
        PAD=(K - 1) // 2 if pad is None else pad,
        stride=stride,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# CPU-only: emitted atom and naming
# ---------------------------------------------------------------------------


class Test4cBf16Spec(unittest.TestCase):
    def test_fp16_kernel_name_has_no_dtype_flag(self):
        p = _problem(N=1, H=8, W=8, groups=16, dtype="fp16")
        name = DirectConv4cSpec(problem=p).kernel_name()
        self.assertEqual(name, "direct_conv_4c_N1H8W8_g16_c4k4_bq4_bg16")
        bf = DirectConv4cSpec(problem=_problem(N=1, H=8, W=8, groups=16))
        self.assertEqual(bf.kernel_name(), name + "_bf16")

    def test_bf16_atom_emitted(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        for arch in ("gfx942", "gfx950"):
            with self.subTest(arch=arch):
                p = _problem(N=1, H=8, W=8, groups=16)
                ll = lower_kernel_to_llvm(
                    build_direct_conv_4c(DirectConv4cSpec(problem=p), arch=arch),
                    arch=arch,
                )
                self.assertIn(_BF16_ATOM, ll)
                self.assertNotIn(_F16_ATOM, ll)
                self.assertIn("<4 x bfloat>", ll)
                ll16 = lower_kernel_to_llvm(
                    build_direct_conv_4c(
                        DirectConv4cSpec(problem=_problem(1, 8, 8, 16, dtype="fp16")),
                        arch=arch,
                    ),
                    arch=arch,
                )
                self.assertIn(_F16_ATOM, ll16)
                self.assertNotIn(_BF16_ATOM, ll16)

    def test_launch_geometry(self):
        p = _problem(N=3, H=13, W=17, groups=64)
        spec = DirectConv4cSpec(problem=p, block_q=8, block_groups=32)
        self.assertEqual(direct_launch_geometry(spec), ((3, 2, 3), (128, 1, 1)))


# ---------------------------------------------------------------------------
# GPU numerics
# ---------------------------------------------------------------------------


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _check(out, ref) -> tuple[bool, str]:
    out_f = out.float().cpu()
    ref_f = ref.float().cpu()
    diff = (out_f - ref_f).abs()
    # NaN-safe: a NaN (e.g. unwritten, NaN-filled) element fails `<=`, so it
    # counts as bad; `diff > tol` would let it pass.
    n_bad = int((~(diff <= _TOL + _TOL * ref_f.abs())).sum())
    max_abs = float(diff.nan_to_num(nan=float("inf")).max())
    return n_bad == 0, f"bad={n_bad} max_abs={max_abs:.3e}"


def _rand(shape, dtype):
    td = torch.bfloat16 if dtype == "bf16" else torch.float16
    return torch.empty(*shape, dtype=td).uniform_(-1.0, 1.0)


def _compile(kernel):
    from rocke import compile_kernel

    return compile_kernel(kernel, arch=GPU_ARCH)


def _launch_direct(art, spec, run_problem, abi_problem, a, b, d, direction="fwd"):
    """Launch ``art`` (built from ``spec``'s shape class) on ``run_problem``.

    ``abi_problem`` is the problem the kernarg block describes (for a dgrad run
    on the transposed problem this is the transposed one); ``a`` / ``b`` / ``d``
    are host tensors bound to the A, B and D slots. Returns the D tensor.
    """
    from dataclasses import replace

    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    rt = Runtime()
    bufs = [rt.alloc(t.nbytes) for t in (a, b, d)]
    rt.memcpy_h2d(bufs[0], _u8(a), a.nbytes)
    rt.memcpy_h2d(bufs[1], _u8(b), b.nbytes)
    # NaN-fill the output so an unwritten element can never pass.
    rt.memset(bufs[2], 0xFF, d.nbytes)
    grid, block = direct_launch_geometry(replace(spec, problem=abi_problem))
    values = ConvArgs.from_problem(abi_problem, direction=direction).to_launch_values(
        int(bufs[0]), int(bufs[1]), int(bufs[2]), a.nbytes, b.nbytes, d.nbytes
    )
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=conv_direct_args_signature(run_problem.dtype, direction=direction),
    )
    launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))
    rt.memcpy_d2h(_u8(d), bufs[2], d.nbytes)
    for buf in bufs:
        rt.free(buf)
    return d


def _run_4c_fprop(spec: DirectConv4cSpec, p: DirectConvProblem, art=None):
    """Run the 4c forward kernel of ``spec`` (built once, on any shape of its
    class) on ``p``; ``(ok, msg)``."""
    if art is None:
        art = _compile(build_direct_conv_4c(spec, arch=GPU_ARCH))
    torch.manual_seed(0)
    x = _rand((p.N, p.H, p.W, p.total_c), p.dtype)
    w = _rand((p.total_k, p.KH, p.KW, p.cpg), p.dtype)
    y = torch.zeros(p.N, p.Ho, p.Wo, p.total_k, dtype=x.dtype)
    ref = torch.nn.functional.conv2d(
        x.float().permute(0, 3, 1, 2),
        w.float().permute(0, 3, 1, 2),
        padding=p.PAD,
        groups=p.groups,
    ).permute(0, 2, 3, 1)
    y = _launch_direct(art, spec, p, p, x, w, y)
    return _check(y, ref)


@unittest.skipIf(bool(_GPU_SKIP), _GPU_SKIP or "gpu")
class Test4cBf16Fprop(unittest.TestCase):
    def test_bf16_fprop(self):
        cases = [
            (_problem(N=2, H=14, W=14, groups=16), 4, 16),
            (_problem(N=1, H=13, W=13, groups=32), 8, 32),
            (_problem(N=1, H=7, W=17, groups=64), 16, 64),
            (_problem(N=2, H=15, W=15, groups=32, K=1), 32, 16),
        ]
        for p, bq, bg in cases:
            with self.subTest(p=p.short(), K=p.KH, bq=bq, bg=bg):
                spec = DirectConv4cSpec(problem=p, block_q=bq, block_groups=bg)
                ok, msg = _run_4c_fprop(spec, p)
                self.assertTrue(ok, msg)

    def test_one_binary_serves_every_shape(self):
        # Built once on a probe problem, launched on unrelated extents.
        probe = _problem(N=1, H=64, W=64, groups=64)
        spec = DirectConv4cSpec(problem=probe, block_q=8, block_groups=16)
        art = _compile(build_direct_conv_4c(spec, arch=GPU_ARCH))
        for p in (
            _problem(N=2, H=5, W=11, groups=16),
            _problem(N=1, H=17, W=3, groups=48),
            _problem(N=3, H=9, W=24, groups=32),
        ):
            with self.subTest(p=p.short()):
                ok, msg = _run_4c_fprop(spec, p, art=art)
                self.assertTrue(ok, msg)


if __name__ == "__main__":
    unittest.main()
