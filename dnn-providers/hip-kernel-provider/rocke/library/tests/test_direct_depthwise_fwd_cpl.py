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


def _spatial(p, **kn):
    return dc.DirectDepthwiseSpatialSpec(problem=p, wave_size=32, **kn)


class TestDepthwiseSpatialSpecs(unittest.TestCase):
    """Row blocks and the rotated-filter dgrad of the small-group spatial kernel."""

    def test_refusals(self):
        ok = dc.is_valid_depthwise_spatial_spec
        p = _problem(C=3, KH=3, PAD=1)
        self.assertTrue(ok(_spatial(p, block_h=4), "gfx1250")[0])
        # block_h is non-negative: validate() raises, the validator reports.
        with self.assertRaisesRegex(ValueError, "block_h must be >= 0"):
            _spatial(p, block_h=-1).validate()
        self.assertIn("block_h", ok(_spatial(p, block_h=-1), "gfx1250")[1])
        # The dgrad mode is a stride-1 pass: validate() raises, the validator reports.
        s2 = _problem(C=3, KH=3, PAD=1, stride=2)
        with self.assertRaisesRegex(ValueError, "needs stride 1"):
            _spatial(s2, dgrad=True).validate()
        self.assertIn("stride 1", ok(_spatial(s2, dgrad=True), "gfx1250")[1])
        # ... and needs a square filter ("same" padding keeps the rotation in place).
        wide = dc.DirectConvProblem(
            N=2, H=13, W=21, groups=3, cpg=1, kpg=1, KH=3, KW=5, PAD=1, dtype="bf16"
        )
        with self.assertRaisesRegex(ValueError, "square filter"):
            _spatial(wide, dgrad=True).validate()
        self.assertIn("square filter", ok(_spatial(wide, dgrad=True), "gfx1250")[1])
        # A column padding of its own is not one the row stream offsets by PAD.
        padw = dc.DirectConvProblem(
            N=2, H=13, W=21, groups=3, cpg=1, kpg=1, KH=3, KW=3, PAD=1, pad_w=0
        )
        self.assertFalse(ok(_spatial(padw), "gfx1250")[0])
        self.assertFalse(ok(_spatial(padw, dgrad=True), "gfx1250")[0])
        self.assertIn("one padding", ok(_spatial(padw), "gfx1250")[1])

    def test_names(self):
        p = _problem(C=3)
        self.assertNotIn("bh", _spatial(p).kernel_name().replace("bh0", ""))
        self.assertIn("bh4", _spatial(p, block_h=4).kernel_name())
        self.assertIn("dg", _spatial(p, dgrad=True).kernel_name())

    def test_ir_builds(self):
        # Row-block, dgrad and dgrad row-block bodies lower to IR on a wave32 arch.
        for kn in (dict(block_h=4), dict(dgrad=True), dict(dgrad=True, block_h=4)):
            spec = _spatial(_problem(C=3, KH=5, PAD=2), **kn)
            kernel = dc.build_direct_depthwise_spatial(spec, arch="gfx1250")
            self.assertTrue(kernel.name, kn)
            self.assertEqual("bh4" in kernel.name, bool(kn.get("block_h")), kn)

    def test_stride_skip_5x5_ir_builds(self):
        spec = _stream(
            _problem(KH=5, PAD=2, stride=2), block_w=4, cpl=2, stride_skip=True
        )
        self.assertTrue(dc.build_direct_depthwise(spec, arch="gfx1250").name)

    def test_aot_rows_stay_on_wave32(self):
        from rocke.core.arch import ArchTarget

        from kernels.conv.direct_aot_jobs import DIRECT_CAPABILITIES, _caps_jobs

        for variant, direction in (
            ("direct_depthwise_spatial", "fwd"),
            ("direct_depthwise_spatial_dgrad", "dgrad"),
        ):
            _, caps_list = DIRECT_CAPABILITIES[variant]
            caps = next(c for c in caps_list if c.KH == 3)
            seen = {}
            for arch in ("gfx950", "gfx1250"):
                wave = ArchTarget.from_gfx(arch).wave_size
                _, jobs = _caps_jobs(arch, wave, variant, direction, caps, "bf16")
                seen[arch] = [json.loads(j.identity.knobs) for _, j in jobs]
            # The wave64 packaged spatial variant takes no block_h knob.
            self.assertFalse(any("block_h" in k for k in seen["gfx950"]), variant)
            self.assertTrue(any("block_h" in k for k in seen["gfx1250"]), variant)
            self.assertTrue(seen["gfx1250"], variant)
        # The 3x3 Toeplitz dgrad form is cached for the wave32 WMMA targets only.
        from kernels.conv.direct_aot_jobs import _dw_win_policy_reason

        win_caps = DIRECT_CAPABILITIES["direct_depthwise_dgrad_win"][1]
        caps = next(c for c in win_caps if c.KH == 3 and c.PAD == 1)
        self.assertIn("wave32", _dw_win_policy_reason(caps, {"mfma": True}, "gfx950"))
        self.assertIsNone(_dw_win_policy_reason(caps, {"mfma": True}, "gfx1250"))


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
