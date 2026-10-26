# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the depthwise dgrad additions.

- Channel-packed row-streaming dgrad (``DirectDepthwiseDgradStreamSpec.cpl``).
- Stride-1 depthwise dgrad through the output-stationary forward kernel with
  the filter rotated by 180 degrees (``DirectDepthwiseTiledSpec(dgrad=True)``),
  including per-axis padding for non-square filters.

CPU: validators, the AOT grid entries and the shape gate of
``direct_spec_for_problem``. GPU (wave32 targets): numerics against torch.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library <torch-python> -m pytest \\
        rocke/library/tests/test_direct_depthwise_dgrad_v2.py -v
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    # Claim the HIP context for torch before rocke's runtime touches it.
    import torch

    torch.cuda.is_available()

from kernels.common import conv_direct_grouped as dc
from kernels.conv.direct_variants import (
    DIRECT_DEPTHWISE_TILED_DGRAD,
    direct_spec_for_problem,
)

GPU_ARCH = get_device_arch(0)


def _problem(
    N=2, H=13, W=21, C=64, KH=3, KW=None, PAD=1, pad_w=-1, stride=1, dtype="bf16"
):
    return dc.DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=C,
        cpg=1,
        kpg=1,
        KH=KH,
        KW=KH if KW is None else KW,
        PAD=PAD,
        stride=stride,
        dtype=dtype,
        pad_w=pad_w,
    )


def _stream(p, **kn):
    return dc.DirectDepthwiseDgradStreamSpec(problem=p, wave_size=32, **kn)


def _tiled(p, **kn):
    return dc.DirectDepthwiseTiledSpec(problem=p, wave_size=32, dgrad=True, **kn)


class TestDepthwiseDgradSpecs(unittest.TestCase):
    def test_cpl_validation(self):
        ok = dc.is_valid_depthwise_dgrad_stream_spec
        self.assertTrue(ok(_stream(_problem(), cpl=2), "gfx1250")[0])
        self.assertFalse(ok(_stream(_problem(), cpl=3), "gfx1250")[0])
        self.assertFalse(ok(_stream(_problem(C=66), cpl=4), "gfx1250")[0])
        # 7x7 at 8 channels per lane would hold 392 weights per lane.
        why = ok(_stream(_problem(KH=7, PAD=3), cpl=8), "gfx1250")[1]
        self.assertIn("cpl=8", why)

    def test_cpl1_name_unchanged(self):
        p = _problem()
        self.assertEqual(_stream(p).kernel_name(), _stream(p, cpl=1).kernel_name())
        self.assertIn("cpl2", _stream(p, cpl=2).kernel_name())

    def test_tiled_dgrad_validation(self):
        ok = dc.is_valid_depthwise_tiled_spec
        kn = dict(block_w=8, block_h=2)
        self.assertTrue(ok(_tiled(_problem(KH=31, PAD=15), **kn), "gfx1250")[0])
        self.assertTrue(
            ok(_tiled(_problem(KH=51, KW=5, PAD=25, pad_w=2), **kn), "gfx1250")[0]
        )
        self.assertFalse(ok(_tiled(_problem(stride=2), **kn), "gfx1250")[0])
        # The forward kernel keeps refusing per-axis padding.
        fwd = dc.DirectDepthwiseTiledSpec(
            problem=_problem(KH=5, KW=3, PAD=2, pad_w=1), wave_size=32, **kn
        )
        self.assertFalse(ok(fwd, "gfx1250")[0])

    def test_pad_w_gate(self):
        p = _problem(KH=5, KW=3, PAD=2, pad_w=1)
        spec, why = direct_spec_for_problem(
            "direct_depthwise_dgrad", p, {"block_w": 4, "block_waves": 1}, "gfx1250"
        )
        self.assertIsNone(spec)
        self.assertIn("one padding", why)
        spec, why = direct_spec_for_problem(
            DIRECT_DEPTHWISE_TILED_DGRAD,
            p,
            {"block_w": 8, "block_h": 2, "block_waves": 1},
            "gfx1250",
            wave_size=32,
        )
        self.assertIsNotNone(spec, why)

    def test_tiled_dgrad_name_carries_per_axis_padding(self):
        kn = dict(block_w=8, block_h=2)
        n0 = _tiled(_problem(KH=5, PAD=2, pad_w=0), **kn).kernel_name()
        n1 = _tiled(_problem(KH=5, PAD=2, pad_w=1), **kn).kernel_name()
        self.assertNotEqual(n0, n1)
        self.assertIn("_f5x5_p2x0_", n0)
        # An equal padding keeps its historical name, however it is spelled.
        sym = _tiled(_problem(KH=5, PAD=2, pad_w=2), **kn).kernel_name()
        self.assertEqual(sym, _tiled(_problem(KH=5, PAD=2), **kn).kernel_name())
        self.assertNotIn("_p2x", sym)

    def test_pad_w_below_minus_one_refused(self):
        kn = dict(block_w=8, block_h=2)
        for p in (_problem(KH=5, PAD=1, pad_w=-2), _problem(pad_w=-3)):
            ok, why = dc.is_valid_depthwise_tiled_spec(_tiled(p, **kn), "gfx1250")
            self.assertFalse(ok)
            self.assertIn("pad_w", why)
            fwd = dc.DirectDepthwiseTiledSpec(problem=p, wave_size=32, **kn)
            self.assertFalse(dc.is_valid_depthwise_tiled_spec(fwd, "gfx1250")[0])
            ok, why = dc.is_valid_depthwise_dgrad_stream_spec(_stream(p), "gfx1250")
            self.assertFalse(ok)
            self.assertIn("pad_w", why)
            with self.assertRaisesRegex(ValueError, "pad_w"):
                _stream(p).validate()
            spec, why = direct_spec_for_problem(
                "direct_depthwise", p, {"block_w": 8, "block_waves": 1}, "gfx1250"
            )
            self.assertIsNone(spec)
            self.assertIn("pad_w", why)

    def test_stream_refuses_per_axis_padding(self):
        # The row-streaming kernel offsets columns by PAD.
        p = _problem(KH=5, PAD=2, pad_w=1)
        ok, why = dc.is_valid_depthwise_dgrad_stream_spec(_stream(p), "gfx1250")
        self.assertFalse(ok)
        self.assertIn("one padding", why)
        with self.assertRaisesRegex(ValueError, "one padding"):
            dc.build_direct_depthwise_dgrad_streaming(_stream(p), arch="gfx1250")
        # pad_w equal to PAD is the same problem as pad_w=-1.
        p = _problem(KH=5, PAD=2, pad_w=2)
        self.assertTrue(
            dc.is_valid_depthwise_dgrad_stream_spec(_stream(p), "gfx1250")[0]
        )

    def test_aot_grid(self):
        from rocke.core.arch import ArchTarget

        from kernels.conv.direct_aot_jobs import DIRECT_CAPABILITIES, _caps_jobs

        direction, caps_list = DIRECT_CAPABILITIES[DIRECT_DEPTHWISE_TILED_DGRAD]
        self.assertEqual(direction, "dgrad")
        filters = {(c.KH, c.KW, c.PAD, c.PAD_W) for c in caps_list}
        self.assertIn((31, 31, 15, 15), filters)
        self.assertIn((51, 5, 25, 2), filters)
        self.assertIn((5, 51, 2, 25), filters)
        wave = ArchTarget.from_gfx("gfx1250").wave_size
        rect = next(c for c in caps_list if c.KW == 51)
        _, jobs = _caps_jobs(
            "gfx1250", wave, DIRECT_DEPTHWISE_TILED_DGRAD, "dgrad", rect, "bf16"
        )
        self.assertTrue(jobs)
        ident = jobs[0][1].identity
        self.assertEqual((ident.filter_h, ident.filter_w), (5, 51))
        self.assertEqual((ident.pad_h, ident.pad_w), (2, 25))


def _gpu_skip() -> str:
    if GPU_ARCH != "gfx1250":
        return f"needs gfx1250 (got {GPU_ARCH!r})"
    if not _HAS_TORCH:
        return "torch not importable"
    return ""


_CASES = (
    # (problem kwargs, "stream"/"tiled", knobs)
    (dict(), "stream", dict(block_w=4, block_waves=1, cpl=2)),
    (dict(dtype="fp16", C=96), "stream", dict(block_w=8, block_waves=2, cpl=4)),
    (dict(H=12, W=12, stride=2), "stream", dict(block_w=8, block_waves=1, cpl=8)),
    (
        dict(H=15, W=9, KH=7, PAD=3, C=48),
        "stream",
        dict(block_w=7, block_waves=1, cpl=2),
    ),
    (
        dict(N=1, H=9, W=10, stride=2, PAD=0, C=32),
        "stream",
        dict(block_w=4, block_waves=1, cpl=2),
    ),
    (dict(H=20, W=19, KH=13, PAD=6), "tiled", dict(block_w=16, block_h=2)),
    (dict(N=1, H=14, W=14, KH=31, PAD=15, C=32), "tiled", dict(block_w=8, block_h=3)),
    (
        dict(N=1, H=14, W=14, KH=51, KW=5, PAD=25, pad_w=2, C=32),
        "tiled",
        dict(block_w=4, block_h=7),
    ),
    (
        dict(N=1, H=14, W=14, KH=5, KW=51, PAD=2, pad_w=25, C=32, dtype="fp16"),
        "tiled",
        dict(block_w=8, block_h=2),
    ),
    (dict(H=10, W=11, KH=3, PAD=0), "tiled", dict(block_w=4, block_h=1)),
)


@unittest.skipUnless(not _gpu_skip(), _gpu_skip() or "no GPU")
class TestDepthwiseDgradGpu(unittest.TestCase):
    def _check(self, pkw, kind, knobs):
        from kernels.common.conv_abi import conv_direct_args_signature
        from kernels.common.conv_args import ConvArgs
        from rocke import compile_kernel
        from rocke.runtime.hip_module import Runtime
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        p = _problem(**pkw)
        if kind == "stream":
            spec = _stream(p, **knobs)
            ok, why = dc.is_valid_depthwise_dgrad_stream_spec(spec, GPU_ARCH)
            kernel = dc.build_direct_depthwise_dgrad_streaming(spec, GPU_ARCH)
        else:
            spec = _tiled(p, **knobs)
            ok, why = dc.is_valid_depthwise_tiled_spec(spec, GPU_ARCH)
            kernel = dc.build_direct_depthwise_tiled(spec, GPU_ARCH)
        self.assertTrue(ok, why)
        art = compile_kernel(kernel, arch=GPU_ARCH)
        grid, block = dc.direct_launch_geometry(spec)
        td = torch.bfloat16 if p.dtype == "bf16" else torch.float16
        torch.manual_seed(0)
        dY = torch.empty(p.N, p.Ho, p.Wo, p.groups).uniform_(-1, 1).to(td)
        w = torch.empty(p.groups, p.KH, p.KW, 1).uniform_(-1, 1).to(td)
        dX = torch.zeros(p.N, p.H, p.W, p.groups, dtype=td)
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
            )(vals, config=LaunchConfig(grid=grid, block=block, fence=True))
            rt.memcpy_d2h(u8(dX), bufs[2], dX.nbytes)
        finally:
            for x in bufs:
                rt.free(x)
        ref = torch.nn.grad.conv2d_input(
            (p.N, p.groups, p.H, p.W),
            w.float().permute(0, 3, 1, 2),
            dY.float().permute(0, 3, 1, 2),
            stride=p.stride,
            padding=(p.PAD, p.PAD_W),
            groups=p.groups,
        ).permute(0, 2, 3, 1)
        err = float((dX.float() - ref).abs().max() / ref.abs().max())
        self.assertLess(err, 1e-2, f"{pkw} {kind} {knobs}: rel err {err:.3g}")

    def test_cases(self):
        for pkw, kind, knobs in _CASES:
            with self.subTest(problem=pkw, kind=kind, knobs=knobs):
                self._check(pkw, kind, knobs)


if __name__ == "__main__":
    unittest.main()
