# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the wave32 WMMA stem dgrad kernels (``conv_dgrad_patch_wmma``,
``conv_dgrad_smallc_wmma``).

CPU: the spec validators and the AOT grid. GPU (gfx1250): patch sizes with and
without an uncovered border, padded GEMM columns, several patch-row blocks;
3x3 stride 1 / 2 and 7x7 stride 2 small-C stems with image edges -- checked
numerically against torch's dgrad.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library <torch-python> -m pytest \\
        rocke/library/tests/test_dgrad_stem_wmma.py -v
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

from kernels.common.conv_dgrad_patch_wmma import (
    DgradPatchWmmaSpec,
    build_dgrad_patch_wmma,
    dgrad_patch_reason,
)
from kernels.common.conv_dgrad_smallc_wmma import (
    DgradSmallcWmmaSpec,
    build_dgrad_smallc_wmma,
    dgrad_smallc_reason,
)
from kernels.common.conv_direct_grouped import DirectConvProblem

GPU_ARCH = get_device_arch(0)


def _problem(N=2, H=32, W=48, C=3, K=64, KH=4, S=4, PAD=0, groups=1, dtype="bf16"):
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


class TestStemSpecs(unittest.TestCase):
    def test_patch_rejects(self):
        ok = DgradPatchWmmaSpec(
            problem=_problem(), waves_c=3, waves_p=1, tiles_c=1, r_blk=4
        )
        self.assertIsNone(dgrad_patch_reason(ok))
        self.assertIn("wave32", dgrad_patch_reason(ok, "gfx950"))
        for problem, knobs, needle in (
            (_problem(PAD=1), {}, "patch embedding"),
            (_problem(S=2), {}, "patch embedding"),
            (_problem(K=60), dict(waves_c=3, tiles_c=1), "multiple of 8"),
            (_problem(), dict(waves_c=3, tiles_c=1, r_blk=3), "divide P"),
            (_problem(), dict(waves_c=4, tiles_c=1, r_blk=4), "block_c"),
            (_problem(), dict(waves_c=3, tiles_c=1, r_blk=4, ck=48), "ck 48"),
        ):
            why = dgrad_patch_reason(DgradPatchWmmaSpec(problem=problem, **knobs))
            self.assertIsNotNone(why, knobs)
            self.assertIn(needle, why)

    def test_smallc_rejects(self):
        p = _problem(KH=3, S=1, PAD=1)
        ok = DgradSmallcWmmaSpec(
            problem=p, waves_c=1, waves_p=4, tiles_c=2, tiles_p=6, out_h=16, out_w=16
        )
        self.assertIsNone(dgrad_smallc_reason(ok))
        for problem, knobs, needle in (
            (_problem(C=8, KH=3, S=1, PAD=1), {}, "C 8"),
            (_problem(), {}, "patch embedding"),
            (_problem(KH=3, S=2, PAD=1), dict(out_h=15), "multiples of S"),
            (p, dict(tiles_c=3), "block_c"),
            (p, dict(out_h=32, out_w=32), "dY window"),
        ):
            why = dgrad_smallc_reason(DgradSmallcWmmaSpec(problem=problem, **knobs))
            self.assertIsNotNone(why, knobs)
            self.assertIn(needle, why)

    def test_aot_grid(self):
        from kernels.conv.direct_aot_jobs import (
            DIRECT_CAPABILITIES,
            _knob_grid,
            probe_problem,
        )
        from kernels.conv.direct_variants import make_direct_spec, validate_direct_spec

        for variant in ("direct_dgrad_patch_wmma", "direct_dgrad_smallc_wmma"):
            _, caps_list = DIRECT_CAPABILITIES[variant]
            self.assertEqual(
                list(_knob_grid(variant, "gfx950", "bf16", caps_list[0])), []
            )
            n_ok = 0
            for caps in caps_list:
                probe = probe_problem(caps, "bf16", variant)
                for knobs in _knob_grid(variant, "gfx1250", "bf16", caps):
                    spec = make_direct_spec(variant, probe, knobs)
                    n_ok += validate_direct_spec(variant, spec, "gfx1250")[0]
            self.assertGreater(n_ok, 0, variant)


def _gpu_skip() -> str:
    if GPU_ARCH != "gfx1250":
        return f"needs gfx1250 (got {GPU_ARCH!r})"
    if not _HAS_TORCH:
        return "torch not importable"
    return ""


_PATCH_CASES = (
    (dict(), dict(waves_c=3, waves_p=1, tiles_c=1, tiles_p=4, tile_w=8, r_blk=4)),
    (
        dict(dtype="fp16", K=40),
        dict(waves_c=1, waves_p=2, tiles_c=1, tiles_p=2, r_blk=1),
    ),
    # 14x14 over 45x50: an uncovered border; padded columns (2*42 -> 96).
    (
        dict(N=3, H=45, W=50, KH=14, S=14, K=96),
        dict(waves_c=2, waves_p=1, tiles_c=3, tiles_p=1, tile_w=4, r_blk=2),
    ),
    (
        dict(N=2, H=64, W=32, KH=16, S=16, K=72),
        dict(
            waves_c=2,
            waves_p=1,
            tiles_c=6,
            tiles_p=1,
            tile_w=4,
            r_blk=4,
            ck=64,
            prefetch=True,
            sched="iterative-ilp",
        ),
    ),
)
_SMALLC_CASES = (
    (
        dict(KH=3, S=1, PAD=1, C=4, H=37, W=29),
        dict(waves_c=1, waves_p=4, tiles_c=3, tiles_p=6, out_h=16, out_w=16),
    ),
    (
        dict(KH=3, S=2, PAD=1, K=24, H=50, W=70),
        dict(waves_c=2, waves_p=4, tiles_c=1, tiles_p=5, out_h=32, out_w=32),
    ),
    (
        dict(KH=7, S=2, PAD=3, H=40, W=36, dtype="fp16"),
        dict(
            waves_c=2,
            waves_p=2,
            tiles_c=5,
            tiles_p=4,
            out_h=16,
            out_w=16,
            ck=64,
            prefetch=True,
            sched="iterative-ilp",
        ),
    ),
)


@unittest.skipUnless(not _gpu_skip(), _gpu_skip() or "no GPU")
class TestStemGpu(unittest.TestCase):
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
        dY = torch.empty(p.N, p.Ho, p.Wo, p.total_k).uniform_(-1, 1).to(td)
        w = torch.empty(p.total_k, p.KH, p.KW, p.cpg).uniform_(-1, 1).to(td)
        dX = torch.zeros(p.N, p.H, p.W, p.total_c, dtype=td)
        rt = Runtime()

        def u8(t):
            return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

        bufs = [rt.alloc(t.nbytes) for t in (dY, w, dX)]
        try:
            rt.memcpy_h2d(bufs[0], u8(dY), dY.nbytes)
            rt.memcpy_h2d(bufs[1], u8(w), w.nbytes)
            rt.memset(bufs[2], 0x7F, dX.nbytes)  # poison: every element is written
            vals = ConvArgs.from_problem(p, direction="dgrad").to_launch_values(
                *(int(x) for x in bufs), dY.nbytes, w.nbytes, dX.nbytes
            )
            KernelLauncher(
                hsaco=art.hsaco,
                kernel_name=art.kernel_name,
                signature=conv_direct_args_signature(p.dtype, direction="dgrad"),
            )(
                vals,
                config=LaunchConfig(
                    grid=spec.grid(), block=(spec.threads_per_block, 1, 1), fence=True
                ),
            )
            rt.memcpy_d2h(u8(dX), bufs[2], dX.nbytes)
        finally:
            for x in bufs:
                rt.free(x)
        ref = torch.nn.grad.conv2d_input(
            (p.N, p.total_c, p.H, p.W),
            w.float().permute(0, 3, 1, 2),
            dY.float().permute(0, 3, 1, 2),
            stride=p.stride,
            padding=p.PAD,
        ).permute(0, 2, 3, 1)
        err = float((dX.float() - ref).abs().max() / ref.abs().max())
        self.assertLess(err, 1e-2, f"{spec}: rel err {err:.3g}")

    def test_patch(self):
        for pkw, knobs in _PATCH_CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                spec = DgradPatchWmmaSpec(problem=_problem(**pkw), **knobs)
                self.assertIsNone(dgrad_patch_reason(spec, GPU_ARCH))
                self._check(spec, build_dgrad_patch_wmma)

    def test_smallc(self):
        for pkw, knobs in _SMALLC_CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                spec = DgradSmallcWmmaSpec(problem=_problem(**pkw), **knobs)
                self.assertIsNone(dgrad_smallc_reason(spec, GPU_ARCH))
                self._check(spec, build_dgrad_smallc_wmma)
