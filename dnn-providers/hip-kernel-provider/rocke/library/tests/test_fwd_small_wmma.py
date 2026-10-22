# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the wave32 WMMA forward kernels of tiny channel counts
(``conv_fwd_small_wmma``: few input channels / few output channels).

CPU: the spec validators and the AOT grid. GPU (gfx1250): 3x3 stride 1 / 2 and
7x7 stride 2 small-C stems with image edges on every side, several tiles per
block, the TDM-store epilogue and a partial output-channel block; the small-K
kernel with edges, stride 2, several channel chunks and K 4 / 8 -- checked
numerically against torch's forward convolution.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library <torch-python> -m pytest \\
        rocke/library/tests/test_fwd_small_wmma.py -v
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch

    torch.cuda.is_available()

from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.common.conv_fwd_small_wmma import (
    FwdSmallcWmmaSpec,
    FwdSmallkWmmaSpec,
    build_fwd_smallc_wmma,
    build_fwd_smallk_wmma,
    fwd_smallc_reason,
    fwd_smallk_reason,
)

GPU_ARCH = get_device_arch(0)


def _problem(N=2, H=32, W=48, C=3, K=64, KH=3, S=1, PAD=1, groups=1, dtype="bf16"):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=C // groups,
        kpg=K // groups,
        KH=KH,
        KW=KH,
        PAD=PAD,
        stride=S,
        dtype=dtype,
    )


class TestSmallSpecs(unittest.TestCase):
    def test_smallc_rejects(self):
        ok = FwdSmallcWmmaSpec(problem=_problem())
        self.assertIsNone(fwd_smallc_reason(ok))
        self.assertIn("wave32", fwd_smallc_reason(ok, "gfx950"))
        for problem, knobs, needle in (
            (_problem(C=8), {}, "C 8"),
            (_problem(K=60), {}, "multiple of 8"),
            (_problem(C=4, K=32, groups=2), {}, "ungrouped"),
            (_problem(dtype="fp32"), {}, "dtype"),
            (_problem(), dict(out_h=4), "GEMM pixels"),
            (_problem(), dict(tiles_k=8, tiles_p=8, out_h=16), "VGPRs"),
        ):
            why = fwd_smallc_reason(FwdSmallcWmmaSpec(problem=problem, **knobs))
            self.assertIsNotNone(why, knobs)
            self.assertIn(needle, why)

    def test_smallk_rejects(self):
        p = _problem(C=64, K=4)
        ok = FwdSmallkWmmaSpec(problem=p)
        self.assertIsNone(fwd_smallk_reason(ok))
        self.assertIn("wave32", fwd_smallk_reason(ok, "gfx950"))
        for problem, knobs, needle in (
            (_problem(C=64, K=16), {}, "K 16"),
            (_problem(C=64, K=3), {}, "K 3"),
            (_problem(C=60, K=4), {}, "multiple of 8"),
            (p, dict(tiles_c=4), "block_c"),
            (p, dict(out_h=32, out_w=32), "input window"),
            (p, dict(ck=48), "ck 48"),
        ):
            why = fwd_smallk_reason(FwdSmallkWmmaSpec(problem=problem, **knobs))
            self.assertIsNotNone(why, knobs)
            self.assertIn(needle, why)

    def test_aot_grid(self):
        from kernels.conv.direct_aot_jobs import (
            DIRECT_CAPABILITIES,
            _knob_grid,
            probe_problem,
        )
        from kernels.conv.direct_variants import make_direct_spec, validate_direct_spec

        for variant in ("direct_fwd_smallc_wmma", "direct_fwd_smallk_wmma"):
            _, caps_list = DIRECT_CAPABILITIES[variant]
            self.assertEqual(
                list(_knob_grid(variant, "gfx950", "bf16", caps_list[0])), []
            )
            for caps in caps_list:
                probe = probe_problem(caps, "bf16", variant)
                n_ok = 0
                for knobs in _knob_grid(variant, "gfx1250", "bf16", caps):
                    spec = make_direct_spec(variant, probe, knobs)
                    n_ok += validate_direct_spec(variant, spec, "gfx1250")[0]
                self.assertGreater(n_ok, 0, (variant, caps))


def _gpu_skip() -> str:
    if GPU_ARCH != "gfx1250":
        return f"needs gfx1250 (got {GPU_ARCH!r})"
    if not _HAS_TORCH:
        return "torch not importable"
    return ""


_SMALLC_CASES = (
    (dict(H=37, W=29), dict()),
    (dict(H=37, W=29, K=24), dict(tdm_store=True)),
    (
        dict(N=2, H=50, W=70, K=24, S=2),
        dict(out_h=4, out_w=16, tiles_p=1, tdm_store=True),
    ),
    (
        dict(H=40, W=36, KH=7, S=2, PAD=3, dtype="fp16"),
        dict(tiles_k=2, out_h=16, out_w=16, waves_p=2, tiles_p=8),
    ),
    (
        dict(N=2, H=33, W=45, C=4, K=40),
        dict(waves_k=2, tiles_k=1, tiles_per_block=3, tdm_store=True),
    ),
    (
        dict(N=2, H=64, W=64, K=128),
        dict(waves_k=2, tiles_k=4, tiles_per_block=4),
    ),
)
_SMALLK_CASES = (
    (dict(C=64, K=4, H=33, W=27), dict()),
    (dict(N=2, C=320, K=4, H=40, W=44), dict(ck=64)),
    (
        dict(C=64, K=8, H=20, W=20, dtype="fp16"),
        dict(tiles_c=5, prefetch=True, sched="iterative-ilp"),
    ),
    (
        dict(C=64, K=4, H=30, W=30, S=2),
        dict(out_h=8, out_w=8, tiles_c=3, tiles_p=5),
    ),
)


@unittest.skipUnless(not _gpu_skip(), _gpu_skip() or "no GPU")
class TestSmallGpu(unittest.TestCase):
    def _check(self, spec, build):
        from kernels.common.conv_abi import conv_direct_args_signature
        from kernels.common.conv_args import ConvArgs
        from rocke import compile_kernel
        from rocke.runtime.hip_module import Runtime
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        p = spec.problem
        art = compile_kernel(build(spec, GPU_ARCH), arch=GPU_ARCH)
        td = torch.bfloat16 if p.dtype == "bf16" else torch.float16
        torch.manual_seed(0)
        x = torch.empty(p.N, p.H, p.W, p.total_c).uniform_(-1, 1).to(td)
        w = torch.empty(p.total_k, p.KH, p.KW, p.cpg).uniform_(-1, 1).to(td)
        y = torch.zeros(p.N, p.Ho, p.Wo, p.total_k, dtype=td)
        rt = Runtime()

        def u8(t):
            return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

        bufs = [rt.alloc(t.nbytes) for t in (x, w, y)]
        try:
            rt.memcpy_h2d(bufs[0], u8(x), x.nbytes)
            rt.memcpy_h2d(bufs[1], u8(w), w.nbytes)
            rt.memset(bufs[2], 0x7F, y.nbytes)  # poison: every element is written
            vals = ConvArgs.from_problem(p, direction="fwd").to_launch_values(
                *(int(v) for v in bufs), x.nbytes, w.nbytes, y.nbytes
            )
            grid, block = spec.launch_geometry()
            KernelLauncher(
                hsaco=art.hsaco,
                kernel_name=art.kernel_name,
                signature=conv_direct_args_signature(p.dtype, direction="fwd"),
            )(vals, config=LaunchConfig(grid=grid, block=block, fence=True))
            rt.memcpy_d2h(u8(y), bufs[2], y.nbytes)
        finally:
            for v in bufs:
                rt.free(v)
        ref = torch.nn.functional.conv2d(
            x.float().permute(0, 3, 1, 2),
            w.float().permute(0, 3, 1, 2),
            stride=p.stride,
            padding=p.PAD,
        ).permute(0, 2, 3, 1)
        err = float((y.float() - ref).abs().max() / ref.abs().max())
        self.assertLess(err, 1e-2, f"{spec}: rel err {err:.3g}")

    def test_smallc(self):
        for pkw, knobs in _SMALLC_CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                spec = FwdSmallcWmmaSpec(problem=_problem(**pkw), **knobs)
                self.assertIsNone(fwd_smallc_reason(spec, GPU_ARCH))
                self._check(spec, build_fwd_smallc_wmma)

    def test_smallk(self):
        for pkw, knobs in _SMALLK_CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                spec = FwdSmallkWmmaSpec(problem=_problem(**pkw), **knobs)
                self.assertIsNone(fwd_smallk_reason(spec, GPU_ARCH))
                self._check(spec, build_fwd_smallk_wmma)
