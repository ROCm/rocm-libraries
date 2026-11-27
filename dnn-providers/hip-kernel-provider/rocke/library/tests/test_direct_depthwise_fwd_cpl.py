# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for channel-packed depthwise forward (``cpl``).

``DirectDepthwiseSpec.cpl`` (row-streaming, weights preloaded) and
``DirectDepthwiseTiledSpec.cpl`` (output-stationary): every lane owns ``cpl``
adjacent channels, input and output move as one vector per lane and the taps
run as packed ``<2 x f32>`` FMAs over channel pairs. ``cpl == 1`` is the
unchanged kernel.

CPU: validators, kernel names and the AOT grid entries. GPU (gfx1250):
numerics against a float64 numpy reference of the stored operands.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library python -m pytest \\
        rocke/library/tests/test_direct_depthwise_fwd_cpl.py -v
"""

from __future__ import annotations

import json
import unittest

import pytest
from rocke.runtime.hip_module import get_device_arch
from rocke.tests.lanes import arch_skip_reason

from kernels.common import conv_direct_grouped as dc

from ._direct_conv_numeric import direct_fwd_peak_error

GPU_ARCH = get_device_arch(0)


def _problem(N=2, H=13, W=21, C=64, KH=3, PAD=1, stride=1, dtype="bf16"):
    return dc.DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=C,
        cpg=1,
        kpg=1,
        KH=KH,
        KW=KH,
        PAD=PAD,
        stride=stride,
        dtype=dtype,
    )


def _stream(p, **kn):
    return dc.DirectDepthwiseSpec(problem=p, wave_size=32, **kn)


def _tiled(p, **kn):
    return dc.DirectDepthwiseTiledSpec(problem=p, wave_size=32, **kn)


class TestDepthwiseFwdCplSpecs(unittest.TestCase):
    def test_stream_validation(self):
        ok = dc.is_valid_depthwise_spec
        self.assertTrue(ok(_stream(_problem(), block_w=4, cpl=2), "gfx1250")[0])
        self.assertTrue(ok(_stream(_problem(), block_w=4, cpl=4), "gfx1250")[0])
        self.assertFalse(ok(_stream(_problem(), cpl=3), "gfx1250")[0])
        self.assertFalse(ok(_stream(_problem(C=66), cpl=4), "gfx1250")[0])
        # 11x11 at 2 channels per lane would hold 242 weights per lane.
        why = ok(_stream(_problem(KH=11, PAD=5), cpl=2), "gfx1250")[1]
        self.assertIn("cpl=2", why)

    def test_tiled_validation(self):
        ok = dc.is_valid_depthwise_tiled_spec
        kn = dict(block_w=8, block_h=2)
        self.assertTrue(ok(_tiled(_problem(KH=7, PAD=3), cpl=2, **kn), "gfx1250")[0])
        self.assertFalse(ok(_tiled(_problem(C=66), cpl=4, **kn), "gfx1250")[0])
        # The packed registers count against the tile's budget.
        big = dict(block_w=7, block_h=7)
        self.assertTrue(ok(_tiled(_problem(KH=7, PAD=3), **big), "gfx1250")[0])
        self.assertFalse(ok(_tiled(_problem(KH=7, PAD=3), cpl=2, **big), "gfx1250")[0])

    def test_stride_skip_validation(self):
        ok = dc.is_valid_depthwise_spec
        s2 = _problem(KH=5, PAD=2, stride=2)
        self.assertTrue(ok(_stream(s2, cpl=2, stride_skip=True), "gfx1250")[0])
        self.assertFalse(ok(_stream(s2, stride_skip=True), "gfx1250")[0])
        self.assertFalse(ok(_stream(_problem(), cpl=2, stride_skip=True), "gfx1250")[0])
        # 3 accumulator rows instead of 5: block_w 24 fits only when skipping.
        self.assertFalse(ok(_stream(s2, block_w=24, cpl=2), "gfx1250")[0])
        self.assertTrue(
            ok(_stream(s2, block_w=24, cpl=2, stride_skip=True), "gfx1250")[0]
        )
        self.assertIn("_ss", _stream(s2, cpl=2, stride_skip=True).kernel_name())
        # Row blocks are a packed-kernel knob.
        self.assertTrue(ok(_stream(s2, cpl=2, block_h=4), "gfx1250")[0])
        self.assertFalse(ok(_stream(s2, block_h=4), "gfx1250")[0])
        grid, _ = dc.direct_launch_geometry(_stream(s2, cpl=2, block_h=4))
        self.assertEqual(grid[2], s2.N * -(-s2.Ho // 4))

    def test_cpl1_name_unchanged(self):
        p = _problem()
        self.assertEqual(_stream(p).kernel_name(), _stream(p, cpl=1).kernel_name())
        self.assertEqual(_tiled(p).kernel_name(), _tiled(p, cpl=1).kernel_name())
        self.assertIn("cpl2", _stream(p, cpl=2).kernel_name())
        self.assertIn("cpl2", _tiled(p, cpl=2).kernel_name())

    def test_launch_geometry(self):
        p = _problem(C=96)
        grid, block = dc.direct_launch_geometry(_stream(p, block_w=4, cpl=4))
        self.assertEqual(block[0], 32)
        self.assertEqual(grid[1], 1)  # 32 lanes x 4 channels cover 96

    def test_aot_grid(self):
        from rocke.core.arch import ArchTarget

        from kernels.conv.direct_aot_jobs import DIRECT_CAPABILITIES, _caps_jobs

        wave = ArchTarget.from_gfx("gfx1250").wave_size
        for variant in ("direct_depthwise", "direct_depthwise_tiled"):
            _, caps_list = DIRECT_CAPABILITIES[variant]
            caps = next(c for c in caps_list if c.KH == 7 and c.stride == 1)
            _, jobs = _caps_jobs("gfx1250", wave, variant, "fwd", caps, "bf16")
            knobs = [json.loads(j.identity.knobs) for _, j in jobs]
            self.assertTrue(any(k.get("cpl") == 2 for k in knobs), variant)
            # cpl == 1 entries keep the knob-less identity.
            self.assertTrue(any("cpl" not in k for k in knobs), variant)


def _gpu_skip() -> str:
    if GPU_ARCH != "gfx1250":
        return arch_skip_reason(("gfx1250",), f"needs gfx1250 (got {GPU_ARCH!r})")
    return ""


_CASES = (
    # (problem kwargs, "stream"/"tiled", knobs)
    (dict(), "stream", dict(block_w=4, block_waves=1, cpl=2)),
    (dict(dtype="fp16", C=96), "stream", dict(block_w=8, block_waves=2, cpl=4)),
    (dict(H=12, W=12, stride=2), "stream", dict(block_w=7, block_waves=1, cpl=2)),
    (
        dict(H=15, W=9, KH=7, PAD=3, C=48),
        "stream",
        dict(block_w=7, block_waves=1, cpl=2),
    ),
    (
        dict(H=11, W=17, KH=5, PAD=2, stride=2, C=40),
        "stream",
        dict(block_w=4, block_waves=2, cpl=4),
    ),
    # stride_skip: output-row accumulators only (period S * ceil(KH / S)).
    (
        dict(H=13, W=17, KH=5, PAD=2, stride=2),
        "stream",
        dict(block_w=4, block_waves=1, cpl=2, stride_skip=True),
    ),
    (
        dict(N=1, H=56, W=56, KH=5, PAD=2, stride=2, C=96, dtype="fp16"),
        "stream",
        dict(block_w=16, block_waves=2, cpl=2, stride_skip=True),
    ),
    (
        dict(H=14, W=9, stride=2),
        "stream",
        dict(block_w=8, block_waves=1, cpl=2, stride_skip=True),
    ),
    (
        dict(H=15, W=19, KH=7, PAD=3, stride=2, C=32),
        "stream",
        dict(block_w=4, block_waves=1, cpl=4, stride_skip=True),
    ),
    (
        dict(H=10, W=11, stride=3),
        "stream",
        dict(block_w=4, block_waves=1, cpl=2, stride_skip=True),
    ),
    # Row blocks (block_h, grid z = n * row blocks): ragged last block,
    # stride 2 with and without stride_skip, a block shorter than the filter.
    (
        dict(H=13, W=17, KH=5, PAD=2, C=96),
        "stream",
        dict(block_w=4, block_waves=1, cpl=2, block_h=4),
    ),
    (
        dict(H=29, W=17, KH=5, PAD=2, stride=2),
        "stream",
        dict(block_w=4, block_waves=2, cpl=2, stride_skip=True, block_h=3),
    ),
    (
        dict(H=14, W=9, stride=2, dtype="fp16"),
        "stream",
        dict(block_w=2, block_waves=1, cpl=2, block_h=2),
    ),
    (
        dict(N=3, H=31, W=19, KH=7, PAD=3, stride=2, C=32),
        "stream",
        dict(block_w=4, block_waves=1, cpl=4, stride_skip=True, block_h=5),
    ),
    (
        dict(H=11, W=12, KH=5, PAD=2),
        "stream",
        dict(block_w=8, block_waves=1, cpl=2, block_h=1),
    ),
    (dict(), "tiled", dict(block_w=8, block_h=2, block_waves=1, cpl=2)),
    (
        dict(H=15, W=9, KH=7, PAD=3, C=48, dtype="fp16"),
        "tiled",
        dict(block_w=4, block_h=4, block_waves=2, cpl=2),
    ),
    (
        dict(H=12, W=13, stride=2, C=96),
        "tiled",
        dict(block_w=8, block_h=2, block_waves=2, cpl=2),
    ),
    (
        dict(H=20, W=19, KH=13, PAD=6),
        "tiled",
        dict(block_w=8, block_h=1, block_waves=1, cpl=2),
    ),
    (
        dict(H=10, W=11, PAD=0, C=32),
        "tiled",
        dict(block_w=4, block_h=2, block_waves=1, cpl=4, unroll_rows=True),
    ),
)


@pytest.mark.gpu
@unittest.skipUnless(not _gpu_skip(), _gpu_skip() or "no GPU")
class TestDepthwiseFwdCplGpu(unittest.TestCase):
    def _check(self, pkw, kind, knobs):
        from rocke import compile_kernel

        p = _problem(**pkw)
        if kind == "stream":
            spec = _stream(p, **knobs)
            ok, why = dc.is_valid_depthwise_spec(spec, GPU_ARCH)
            self.assertTrue(ok, why)
            kernel = dc.build_direct_depthwise(spec, arch=GPU_ARCH)
        else:
            spec = _tiled(p, **knobs)
            ok, why = dc.is_valid_depthwise_tiled_spec(spec, GPU_ARCH)
            self.assertTrue(ok, why)
            kernel = dc.build_direct_depthwise_tiled(spec, GPU_ARCH)
        art = compile_kernel(kernel, arch=GPU_ARCH)
        grid, block = dc.direct_launch_geometry(spec)
        err = direct_fwd_peak_error(art, p, grid=grid, block=block)
        self.assertLess(err, 1e-2, f"{pkw} {kind} {knobs}: rel err {err:.3g}")

    def test_cases(self):
        for pkw, kind, knobs in _CASES:
            with self.subTest(problem=pkw, kind=kind, knobs=knobs):
                self._check(pkw, kind, knobs)


if __name__ == "__main__":
    unittest.main()
