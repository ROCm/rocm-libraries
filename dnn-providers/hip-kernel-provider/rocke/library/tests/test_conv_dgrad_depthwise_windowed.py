# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Windowed depthwise dgrad (``build_direct_depthwise_dgrad_windowed``).

Host-only: spec validation, kernel naming and grid geometry.
GPU (gfx942 / gfx950; ``dot2`` and ``mfma`` cases gfx950 only): numeric check
of dX against an fp32 NumPy reference with the manifest conv rule --
``bad == 0`` where an element is bad when ``|out - ref| > tol + tol * |ref|``
(tol = 1e-2 for fp16/bf16). dX is NaN-filled before each launch, so an
element the kernel never writes is caught as bad too.

Every launch goes through the direct dgrad ABI (``ConvArgs`` /
``conv_direct_args_signature(direction="dgrad")``): the extents are kernargs.
The AOT cases build one binary on a probe problem and launch it on other
shapes, the property the kernel cache relies on.

The GPU shapes are adversarial for this kernel: widths that no block_w
divides, odd heights, N = 1, channel counts that are not a multiple of the
64-lane block, padding wider than half the filter, tiles taller than the
image and ragged row tiles, non-square filters, 1x1, and every ch_per_lane /
dot2 path. The Toeplitz MFMA form (``mfma``, gfx950) gets its own set: odd N
under w_fold (a half-empty last tile), images narrower than the tile, two W
tiles, partial channel blocks, ragged row tiles, pad 0 and pad > K/2, filters
larger than the image, 3x3 to 11x11 and non-square filters, fp16 and bf16.

Non-finite inputs: the VALU forms keep an Inf / NaN in dY inside its
receptive field and channel exactly. The MFMA form multiplies one-hot
weights against 8 channels at once, so a non-finite dY value may reach every
channel of its 8-channel group, in its receptive field widened to
4 * ceil((KW + 1) / 4) columns (the zero-weight taps of its passes);
TestNonFiniteGradients asserts that bound for it.

Run (GPU part as a script, as the other numeric suites):
    PYTHONPATH=rocke/platform/python:rocke/library python \\
        rocke/library/tests/test_conv_dgrad_depthwise_windowed.py
"""

from __future__ import annotations

import ctypes
import unittest
import unittest.mock
from dataclasses import dataclass, replace

import numpy as np

try:
    # Optional. When torch is installed its bundled HIP runtime must load
    # before rocke's launcher does; otherwise the launcher pulls in a second
    # libamdhip64 and torch.cuda reports no GPUs for every later test in this
    # pytest process.
    import torch  # noqa: F401
except ImportError:
    pass
from kernels.common.conv_direct_grouped import (
    DW_DGRAD_MFMA_CH,
    DW_DGRAD_WIN_MAX_UNROLL,
    DirectConvProblem,
    DirectDepthwiseDgradWindowedSpec,
    build_direct_depthwise_dgrad_windowed,
    direct_launch_geometry,
    is_valid_depthwise_dgrad_win_spec,
)
from rocke.runtime.hip_module import get_device_arch

GPU_ARCH = get_device_arch(0)
_TOL = 1e-2


def _problem(N, H, W, C, KH, KW, PAD, dtype="fp16", stride=1):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=C,
        cpg=1,
        kpg=1,
        KH=KH,
        KW=KW,
        PAD=PAD,
        stride=stride,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Host-only
# ---------------------------------------------------------------------------


class TestWindowedSpec(unittest.TestCase):
    def _ok(self, spec, arch="gfx950"):
        return is_valid_depthwise_dgrad_win_spec(spec, arch=arch)

    def test_rejections(self):
        base = _problem(2, 8, 8, 64, 3, 3, 1)
        cases = [
            ({"problem": _problem(2, 8, 8, 64, 3, 3, 1, stride=2)}, "stride=1"),
            ({"problem": _problem(2, 8, 8, 66, 3, 3, 1), "ch_per_lane": 4}, "multiple"),
            ({"problem": base, "ch_per_lane": 3}, "ch_per_lane must be"),
            ({"problem": base, "ch_per_lane": 2, "dot2": True}, "dot2 requires"),
            ({"problem": base, "block_w": 0}, "block_w"),
            ({"problem": base, "block_waves": 17}, "block_waves"),
            # The image height is a kernarg: no whole-image block.
            ({"problem": base, "block_h": 0}, "block_h"),
            ({"problem": base, "block_h": -1}, "block_h"),
            ({"problem": _problem(2, 8, 8, 64, 3, 3, 1, dtype="fp32")}, "dtype"),
            ({"problem": _problem(2, 2, 8, 64, 5, 5, 0)}, "degenerate"),
        ]
        for kw, needle in cases:
            ok, why = self._ok(DirectDepthwiseDgradWindowedSpec(**kw))
            self.assertFalse(ok, kw)
            self.assertIn(needle, why, kw)
            with self.assertRaises(ValueError):
                build_direct_depthwise_dgrad_windowed(
                    DirectDepthwiseDgradWindowedSpec(**kw)
                )

    def test_dot2_is_gfx950_only(self):
        spec = DirectDepthwiseDgradWindowedSpec(
            problem=_problem(2, 8, 8, 64, 3, 3, 1, "bf16"), dot2=True
        )
        self.assertTrue(self._ok(spec, "gfx950")[0])
        ok, why = self._ok(spec, "gfx942")
        self.assertFalse(ok)
        self.assertIn("dot2", why)
        self.assertTrue(
            self._ok(DirectDepthwiseDgradWindowedSpec(problem=spec.problem), "gfx942")[
                0
            ]
        )

    def test_unroll_cap_follows_the_tile_not_the_image(self):
        p = _problem(1, 64, 64, 64, 7, 7, 3)
        tall = DirectDepthwiseDgradWindowedSpec(problem=p, block_w=16, block_h=64)
        self.assertGreater(tall.unrolled_fmas(), DW_DGRAD_WIN_MAX_UNROLL)
        ok, why = self._ok(tall)
        self.assertFalse(ok)
        self.assertIn("unrolled body too large", why)
        split = DirectDepthwiseDgradWindowedSpec(problem=p, block_w=16, block_h=16)
        self.assertTrue(self._ok(split)[0])
        # A tile taller than the image is fine: the rows past it are guarded.
        short = replace(split, problem=_problem(1, 5, 64, 64, 7, 7, 3))
        self.assertTrue(self._ok(short)[0])
        self.assertEqual(short.unrolled_fmas(), split.unrolled_fmas())

    def test_sentinel_range_cap(self):
        # dX = 2 * N*H*W*C bytes must stay below 2**30 for the sentinel adds.
        p = _problem(4096, 64, 64, 64, 3, 3, 1)
        ok, why = self._ok(DirectDepthwiseDgradWindowedSpec(problem=p, block_h=8))
        self.assertFalse(ok)
        self.assertIn("sentinel", why)

    def test_grid_and_name(self):
        p = _problem(3, 13, 17, 130, 5, 5, 2, "bf16")
        spec = DirectDepthwiseDgradWindowedSpec(
            problem=p, block_w=9, block_waves=2, block_h=4, dot2=True
        )
        self.assertEqual(spec.h_tiles, 4)
        self.assertEqual(spec.grid(), (2, 2, 12))
        self.assertEqual(direct_launch_geometry(spec), ((2, 2, 12), (128, 1, 1)))
        name = spec.kernel_name()
        for part in ("r5s5p2", "bw9", "wv2", "cpl1", "bh4", "dot2", "bf16"):
            self.assertIn(part, name)
        # The filter and pad reach the name; the extents (kernargs) do not.
        other = replace(spec, problem=_problem(3, 13, 17, 130, 3, 3, 1, "bf16"))
        self.assertNotEqual(name, other.kernel_name())
        self.assertEqual(
            name,
            replace(
                spec, problem=_problem(1, 40, 9, 64, 5, 5, 2, "bf16")
            ).kernel_name(),
        )
        # block_h is always spelled, even where the tile covers the image.
        whole = DirectDepthwiseDgradWindowedSpec(problem=p, block_h=16)
        self.assertEqual(whole.h_tiles, 1)
        self.assertIn("_bh16", whole.kernel_name())

    def test_dot2_lowers_to_fdot2(self):
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        for dtype, intrin in (("bf16", "fdot2.f32.bf16"), ("fp16", "fdot2(")):
            spec = DirectDepthwiseDgradWindowedSpec(
                problem=_problem(1, 6, 6, 64, 3, 3, 1, dtype),
                block_w=6,
                block_h=6,
                dot2=True,
            )
            ll = _lower_kernel_to_llvm_python(
                build_direct_depthwise_dgrad_windowed(spec), arch="gfx950"
            )
            self.assertIn(f"@llvm.amdgcn.{intrin}", ll)


class TestMfmaSpec(unittest.TestCase):
    """Host-only checks of the Toeplitz MFMA form (``mfma``)."""

    def _ok(self, spec, arch="gfx950"):
        return is_valid_depthwise_dgrad_win_spec(spec, arch=arch)

    def _mfma(self, problem=None, **kw):
        base = {
            "block_waves": 8,
            "block_h": 14,
            "mfma": True,
            "w_fold": 2,
            "prefetch_rows": 2,
        }
        base.update(kw)
        return DirectDepthwiseDgradWindowedSpec(
            problem=problem or _problem(4, 14, 14, 64, 7, 7, 3, "bf16"), **base
        )

    def test_rejections(self):
        cases = [
            ({"problem": _problem(2, 14, 14, 12, 7, 7, 3)}, "groups % 8"),
            ({"dot2": True}, "mfma excludes dot2"),
            ({"ch_per_lane": 2}, "mfma excludes dot2"),
            ({"w_fold": 3}, "w_fold must be 1, 2 or 4"),
            ({"prefetch_rows": 0}, "prefetch_rows must be >= 1"),
            ({"prefetch_rows": 5}, "prefetch_rows must be <= 4"),
            ({"block_waves": 16}, "block_waves <= 8"),
            ({"problem": _problem(1, 24, 24, 64, 15, 15, 7)}, "VGPRs"),
            ({"block_h": 224}, "MFMAs"),
        ]
        for kw, needle in cases:
            with self.subTest(kw=kw):
                ok, why = self._ok(self._mfma(**kw))
                self.assertFalse(ok)
                self.assertIn(needle, why)
                with self.assertRaises(ValueError):
                    build_direct_depthwise_dgrad_windowed(self._mfma(**kw))
        ok, why = self._ok(self._mfma(), "gfx942")
        self.assertFalse(ok)
        self.assertIn("mfma needs one of", why)
        # A tall image fits any bounded row tile.
        tall = self._mfma(_problem(1, 224, 16, 64, 7, 7, 3))
        self.assertTrue(self._ok(tall)[0])

    def test_mfma_only_knobs_need_mfma(self):
        p = _problem(2, 8, 8, 64, 3, 3, 1)
        for kw in ({"w_fold": 2}, {"prefetch_rows": 2}):
            ok, why = self._ok(DirectDepthwiseDgradWindowedSpec(problem=p, **kw))
            self.assertFalse(ok, kw)
            self.assertIn("require mfma", why)

    def test_lds_bytes_and_budget(self):
        # Two double buffers of (block_ch + 8)-channel pixels: 64 staged dY
        # window columns and 32 dX columns at 8 waves / w_fold 2.
        self.assertEqual(self._mfma().mfma_lds_bytes(), 2 * (64 + 32) * 72 * 2)
        self.assertEqual(
            self._mfma(block_waves=4).mfma_lds_bytes(), 2 * (64 + 32) * 40 * 2
        )
        budget = "kernels.common.conv_direct_grouped.DW_DGRAD_MFMA_LDS_BUDGET"
        with unittest.mock.patch(budget, 16 * 1024):
            ok, why = self._ok(self._mfma())
        self.assertFalse(ok)
        self.assertIn("LDS bytes", why)

    def test_grid_and_name(self):
        p = _problem(3, 13, 17, 136, 5, 5, 2, "bf16")
        spec = self._mfma(p, block_waves=4, w_fold=2, block_h=4)
        self.assertEqual(spec.block_ch, 4 * DW_DGRAD_MFMA_CH)
        self.assertEqual(spec.tile_w, 16)
        # W tiles of 16 columns, channel blocks of 32, ceil(3 / 2) image
        # tiles of 4 row tiles each.
        self.assertEqual(spec.grid(), (2, 5, 8))
        name = spec.kernel_name()
        for part in ("r5s5p2", "wv4", "bh4", "f2", "pf2", "mfma", "bf16"):
            self.assertIn(part, name)
        # block_w / ch_per_lane do not apply: neither reaches the name, and
        # specs that differ only there name (and build) the same kernel.
        self.assertNotIn("_bw", name)
        self.assertNotIn("_cpl", name)
        other = self._mfma(p, block_waves=4, w_fold=2, block_h=4, block_w=3)
        self.assertEqual(name, other.kernel_name())
        for kw in ({"w_fold": 4}, {"prefetch_rows": 1}, {"block_waves": 8}):
            args = {"block_waves": 4, "w_fold": 2, "block_h": 4}
            args.update(kw)
            self.assertNotEqual(name, self._mfma(p, **args).kernel_name(), kw)
        # The VALU name does not grow MFMA tags.
        valu = DirectDepthwiseDgradWindowedSpec(problem=p, block_w=9)
        self.assertNotIn("_f1", valu.kernel_name())
        self.assertNotIn("_pf", valu.kernel_name())

    def test_lowers_to_mfma_with_lds_staging(self):
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        for dtype, intrin in (("bf16", "16x16x32.bf16"), ("fp16", "16x16x32.f16")):
            spec = self._mfma(
                _problem(2, 7, 9, 16, 5, 5, 2, dtype), block_waves=2, block_h=7
            )
            ll = _lower_kernel_to_llvm_python(
                build_direct_depthwise_dgrad_windowed(spec), arch="gfx950"
            )
            self.assertIn(f"@llvm.amdgcn.mfma.f32.{intrin}", ll)
            self.assertIn("addrspace(3)", ll)


# ---------------------------------------------------------------------------
# GPU numerics
# ---------------------------------------------------------------------------


def _to_bf16_bits(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _bits_to_f32(bits: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        return (bits.astype(np.uint32) << 16).view(np.float32)
    return bits.view(np.float16).astype(np.float32)


def _reference(dy: np.ndarray, w: np.ndarray, p: DirectConvProblem) -> np.ndarray:
    """dX[n, h, x, c] = sum_{r, s} W[c, r, s] * dY[n, h + PAD - r, x + PAD - s, c]."""
    dx = np.zeros((p.N, p.H, p.W, p.groups), np.float32)
    for r in range(p.KH):
        for s in range(p.KW):
            h0, h1 = max(0, r - p.PAD), min(p.H, p.Ho + r - p.PAD)
            x0, x1 = max(0, s - p.PAD), min(p.W, p.Wo + s - p.PAD)
            if h0 >= h1 or x0 >= x1:
                continue
            dx[:, h0:h1, x0:x1, :] += (
                w[:, r, s][None, None, None, :]
                * dy[
                    :,
                    h0 + p.PAD - r : h1 + p.PAD - r,
                    x0 + p.PAD - s : x1 + p.PAD - s,
                    :,
                ]
            )
    return dx


_ARTIFACTS: dict = {}


def _compiled(spec: DirectDepthwiseDgradWindowedSpec, arch: str):
    """Compile ``spec`` once per kernel name: the extents are kernargs, so a
    binary serves every problem with the spec's filter, padding and knobs."""
    from rocke import compile_kernel

    key = (spec.kernel_name(), arch)
    if key not in _ARTIFACTS:
        _ARTIFACTS[key] = compile_kernel(
            build_direct_depthwise_dgrad_windowed(spec, arch=arch), arch=arch
        )
    return _ARTIFACTS[key]


def _launch(
    spec: DirectDepthwiseDgradWindowedSpec,
    arch: str,
    problem: DirectConvProblem | None = None,
    inf_at=(),
    nan_at=(),
):
    """Launch ``spec``'s binary on ``problem`` (default ``spec.problem``) with
    a NaN-filled dX; return ``(dx, ref)``.

    ``inf_at`` / ``nan_at`` list ``(n, ho, wo, c)`` dY entries set to +Inf /
    NaN before the 16-bit rounding (both survive it).
    """
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    art = _compiled(spec, arch)
    p = problem or spec.problem
    run_spec = replace(spec, problem=p)
    ok, why = is_valid_depthwise_dgrad_win_spec(run_spec, arch=arch)
    assert ok, why
    rng = np.random.default_rng(1234)
    dy32 = rng.uniform(-1, 1, (p.N, p.Ho, p.Wo, p.groups)).astype(np.float32)
    w32 = rng.uniform(-1, 1, (p.groups, p.KH, p.KW)).astype(np.float32)
    for idx in inf_at:
        dy32[idx] = np.inf
    for idx in nan_at:
        dy32[idx] = np.nan
    if p.dtype == "bf16":
        dy_b, w_b = _to_bf16_bits(dy32), _to_bf16_bits(w32)
    else:
        dy_b = dy32.astype(np.float16).view(np.uint16)
        w_b = w32.astype(np.float16).view(np.uint16)
    ref = _reference(_bits_to_f32(dy_b, p.dtype), _bits_to_f32(w_b, p.dtype), p)

    rt = Runtime()
    dx_bytes = p.N * p.H * p.W * p.groups * 2
    dY_d, W_d, dX_d = rt.alloc(dy_b.nbytes), rt.alloc(w_b.nbytes), rt.alloc(dx_bytes)
    try:
        rt.memcpy_h2d(dY_d, dy_b.ctypes.data_as(ctypes.c_void_p), dy_b.nbytes)
        rt.memcpy_h2d(W_d, w_b.ctypes.data_as(ctypes.c_void_p), w_b.nbytes)
        rt.memset(dX_d, 0xFF, dx_bytes)  # NaN in both fp16 and bf16
        launcher = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=conv_direct_args_signature(p.dtype, direction="dgrad"),
        )
        values = ConvArgs.from_problem(p, direction="dgrad").to_launch_values(
            int(dY_d), int(W_d), int(dX_d), dy_b.nbytes, w_b.nbytes, dx_bytes
        )
        grid, block = direct_launch_geometry(run_spec)
        launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))
        out = np.empty(dx_bytes // 2, np.uint16)
        rt.memcpy_d2h(out.ctypes.data_as(ctypes.c_void_p), dX_d, dx_bytes)
    finally:
        rt.free(dY_d)
        rt.free(W_d)
        rt.free(dX_d)
        synchronize_and_release(0)
    return _bits_to_f32(out, p.dtype).reshape(ref.shape), ref


def _elementwise_ok(o: np.ndarray, ref: np.ndarray):
    """Per element: within tolerance where ``ref`` is finite, equal to the
    reference's Inf where it is not. Returns ``(ok_mask, abs_err)``."""
    with np.errstate(invalid="ignore"):
        fin = np.isfinite(ref)
        err = np.where(fin, np.abs(o - ref), 0.0)
        ok = np.where(fin, err <= _TOL + _TOL * np.abs(ref), o == ref)
    return ok, err


def run_windowed(spec, arch, problem=None, inf_at=()):
    """Launch ``spec``; return ``(bad, max_err)`` vs the reference."""
    o, ref = _launch(spec, arch, problem=problem, inf_at=inf_at)
    ok, err = _elementwise_ok(o, ref)
    return int(np.count_nonzero(~ok)), float(np.nanmax(err))


@dataclass(frozen=True)
class _Case:
    id: str
    N: int
    H: int
    W: int
    C: int
    KH: int
    KW: int
    PAD: int
    dtype: str
    block_w: int
    block_waves: int = 1
    ch_per_lane: int = 1
    block_h: int = 16
    dot2: bool = False


_CASES = (
    _Case("w13_3x3_c72", 2, 9, 13, 72, 3, 3, 1, "fp16", 4),
    _Case("w17_5x5_n1_c66_bf16", 1, 7, 17, 66, 5, 5, 2, "bf16", 5, ch_per_lane=2),
    _Case("w7_7x7_c130_split", 3, 15, 7, 130, 7, 7, 3, "bf16", 8, 2, block_h=3),
    _Case("w15_pad0_split_ragged", 2, 11, 15, 64, 3, 3, 0, "fp16", 7, block_h=4),
    _Case("bigpad_c10", 1, 5, 6, 10, 3, 3, 2, "fp16", 4, block_h=8),
    _Case("w14_7x7_cpl2", 2, 14, 14, 128, 7, 7, 3, "bf16", 7, 1, 2, block_h=14),
    _Case("w12_3x3_cpl4", 2, 12, 12, 256, 3, 3, 1, "fp16", 6, 2, 4, block_h=12),
    _Case("w9_1x1", 2, 9, 9, 64, 1, 1, 0, "fp16", 9),
    _Case("w11_3x5_nonsquare", 2, 7, 11, 96, 3, 5, 1, "bf16", 11, 2, block_h=7),
    _Case("w16_even_split", 2, 16, 16, 64, 5, 5, 2, "fp16", 16, block_h=8),
    _Case("w13_5x5_dot2", 2, 9, 13, 72, 5, 5, 2, "fp16", 13, dot2=True),
    _Case(
        "w17_7x7_dot2_split_bf16",
        1,
        13,
        17,
        130,
        7,
        7,
        3,
        "bf16",
        9,
        2,
        block_h=5,
        dot2=True,
    ),
    _Case("w14_4x4_dot2_even_k", 2, 10, 14, 64, 4, 4, 1, "fp16", 7, dot2=True),
    _Case("w7_3x3_dot2_bigpad", 1, 7, 7, 66, 3, 3, 2, "bf16", 7, dot2=True),
    _Case("w9_1x1_dot2", 2, 9, 9, 64, 1, 1, 0, "bf16", 3, dot2=True),
)

# (N, H, W, C) problems one binary per form is launched on after it was
# built for another: odd batches and channel counts, images shorter and
# taller than the tile, widths no tile divides.
_AOT_SHAPES = ((1, 5, 9, 72), (3, 23, 13, 136), (2, 14, 30, 64))


@unittest.skipUnless(
    GPU_ARCH in ("gfx942", "gfx950"), f"needs gfx942/gfx950, got {GPU_ARCH!r}"
)
class TestWindowedNumerics(unittest.TestCase):
    def test_adversarial_shapes(self):
        ran = 0
        for c in _CASES:
            if c.dot2 and GPU_ARCH != "gfx950":
                continue
            spec = DirectDepthwiseDgradWindowedSpec(
                problem=_problem(c.N, c.H, c.W, c.C, c.KH, c.KW, c.PAD, c.dtype),
                block_w=c.block_w,
                block_waves=c.block_waves,
                ch_per_lane=c.ch_per_lane,
                block_h=c.block_h,
                dot2=c.dot2,
            )
            with self.subTest(case=c.id):
                bad, max_err = run_windowed(spec, GPU_ARCH)
                self.assertEqual(bad, 0, f"{c.id}: bad={bad} max_err={max_err:.3e}")
                ran += 1
        self.assertGreater(ran, 0)

    def test_one_binary_serves_every_shape(self):
        for dot2 in (False, True) if GPU_ARCH == "gfx950" else (False,):
            probe = _problem(1, 64, 64, 256, 5, 5, 1, "bf16")
            spec = DirectDepthwiseDgradWindowedSpec(
                problem=probe, block_w=8, block_waves=2, block_h=7, dot2=dot2
            )
            for N, H, W, C in _AOT_SHAPES:
                p = _problem(N, H, W, C, 5, 5, 1, "bf16")
                with self.subTest(dot2=dot2, shape=(N, H, W, C)):
                    bad, max_err = run_windowed(spec, GPU_ARCH, problem=p)
                    self.assertEqual(bad, 0, f"bad={bad} max_err={max_err:.3e}")


@dataclass(frozen=True)
class _MfmaCase:
    id: str
    N: int
    H: int
    W: int
    C: int
    KH: int
    KW: int
    PAD: int
    dtype: str
    block_waves: int
    w_fold: int
    prefetch_rows: int
    block_h: int = 16


_MFMA_CASES = (
    _MfmaCase("7x7_odd_n_fold2", 3, 14, 14, 64, 7, 7, 3, "bf16", 8, 2, 2, 14),
    _MfmaCase("7x7_n1_fold4_w7", 1, 7, 7, 64, 7, 7, 3, "bf16", 4, 4, 2, 8),
    _MfmaCase(
        "7x7_partial_ch_ragged_split", 5, 13, 15, 40, 7, 7, 3, "fp16", 4, 2, 2, 4
    ),
    _MfmaCase("5x5_two_w_tiles_fold1", 2, 9, 37, 24, 5, 5, 2, "fp16", 2, 1, 1),
    _MfmaCase("7x7_pad0", 2, 10, 12, 16, 7, 7, 0, "bf16", 2, 2, 2, 12),
    _MfmaCase("7x7_pad5_split", 2, 9, 9, 24, 7, 7, 5, "fp16", 1, 2, 3, 3),
    _MfmaCase("7x7_image_smaller_than_k", 2, 5, 3, 16, 7, 7, 3, "fp16", 2, 4, 1, 7),
    _MfmaCase("9x9_fold2", 2, 9, 11, 64, 9, 9, 4, "fp16", 8, 2, 2, 12),
    _MfmaCase("11x11_fold4_split", 2, 13, 13, 64, 11, 11, 5, "bf16", 8, 4, 2, 4),
    _MfmaCase("7x5_nonsquare", 2, 11, 15, 32, 7, 5, 2, "bf16", 4, 2, 2, 12),
    _MfmaCase("5x7_nonsquare_split_pf4", 1, 12, 10, 16, 5, 7, 3, "fp16", 2, 2, 4, 5),
    _MfmaCase("1x5_single_row_taps", 2, 6, 10, 8, 1, 5, 2, "bf16", 1, 2, 2),
    _MfmaCase("3x3_one_pass", 2, 12, 12, 64, 3, 3, 1, "fp16", 4, 2, 1, 12),
    _MfmaCase("7x7_tall_split_fold1", 1, 40, 20, 64, 7, 7, 3, "bf16", 8, 1, 2, 14),
)


def _mfma_spec(c: _MfmaCase) -> DirectDepthwiseDgradWindowedSpec:
    return DirectDepthwiseDgradWindowedSpec(
        problem=_problem(c.N, c.H, c.W, c.C, c.KH, c.KW, c.PAD, c.dtype),
        block_waves=c.block_waves,
        block_h=c.block_h,
        mfma=True,
        w_fold=c.w_fold,
        prefetch_rows=c.prefetch_rows,
    )


@unittest.skipUnless(GPU_ARCH == "gfx950", f"needs gfx950, got {GPU_ARCH!r}")
class TestMfmaNumerics(unittest.TestCase):
    def test_adversarial_shapes(self):
        for c in _MFMA_CASES:
            spec = _mfma_spec(c)
            with self.subTest(case=c.id):
                bad, max_err = run_windowed(spec, GPU_ARCH)
                self.assertEqual(bad, 0, f"{c.id}: bad={bad} max_err={max_err:.3e}")

    def test_one_binary_serves_every_shape(self):
        for fold in (1, 2, 4):
            probe = _problem(1, 64, 64, 256, 7, 7, 3, "fp16")
            spec = DirectDepthwiseDgradWindowedSpec(
                problem=probe,
                block_waves=4,
                block_h=8,
                mfma=True,
                w_fold=fold,
                prefetch_rows=2,
            )
            for N, H, W, C in _AOT_SHAPES:
                p = _problem(N, H, W, C, 7, 7, 3, "fp16")
                with self.subTest(fold=fold, shape=(N, H, W, C)):
                    bad, max_err = run_windowed(spec, GPU_ARCH, problem=p)
                    self.assertEqual(bad, 0, f"bad={bad} max_err={max_err:.3e}")


def _mfma_spread_zone(p: DirectConvProblem, bad_inputs) -> np.ndarray:
    """dX elements a non-finite dY value may reach in the MFMA form: its
    receptive field rows, its columns widened by the ``4 * passes - KW``
    zero-weight taps on either side (where they fall depends on the column
    parity), in every channel of its 8-channel group."""
    zone = np.zeros((p.N, p.H, p.W, p.groups), bool)
    extra = 4 * ((p.KW + 4) // 4) - p.KW
    for n, ho, wo, c in bad_inputs:
        h0, w0 = ho - p.PAD, wo - p.PAD
        g0 = c - c % DW_DGRAD_MFMA_CH
        zone[
            n,
            max(0, h0) : max(0, h0 + p.KH),
            max(0, w0 - extra) : max(0, w0 + p.KW + extra),
            g0 : g0 + DW_DGRAD_MFMA_CH,
        ] = True
    return zone


@unittest.skipUnless(GPU_ARCH == "gfx950", f"needs gfx950, got {GPU_ARCH!r}")
class TestNonFiniteGradients(unittest.TestCase):
    """An Inf in dY reaches only the dX columns in its receptive field.

    With dot2 an odd KW pads the last tap pair with a zero weight; that pair
    must not read a dY column outside the field (0 * Inf would put a NaN in a
    dX column whose true value is finite). Each Inf sits in its own channel,
    on the first, an inner and the last dY column.
    """

    def test_inf_stays_in_receptive_field(self):
        cases = (
            # (N, H, W, C, KH, KW, PAD, dtype, block_w, block_h, dot2)
            (1, 9, 13, 8, 7, 7, 3, "bf16", 13, 12, True),
            (1, 9, 13, 8, 3, 3, 1, "fp16", 5, 4, True),
            (1, 7, 11, 8, 3, 5, 0, "fp16", 4, 7, True),
            (1, 6, 9, 8, 1, 1, 0, "bf16", 9, 8, True),
            (1, 9, 13, 8, 4, 4, 2, "fp16", 7, 9, True),
            (1, 9, 13, 8, 7, 7, 3, "bf16", 13, 12, False),
        )
        for N, H, W, C, KH, KW, PAD, dt, bw, bh, dot2 in cases:
            p = _problem(N, H, W, C, KH, KW, PAD, dt)
            inf_at = (
                (0, p.Ho // 2, 0, 1),
                (0, p.Ho // 2, p.Wo // 2, 3),
                (0, 0, p.Wo - 1, 5),
            )
            spec = DirectDepthwiseDgradWindowedSpec(
                problem=p, block_w=bw, block_h=bh, dot2=dot2
            )
            with self.subTest(case=f"{KH}x{KW}_{dt}_bw{bw}_bh{bh}_dot2{int(dot2)}"):
                bad, _ = run_windowed(spec, GPU_ARCH, inf_at=inf_at)
                self.assertEqual(bad, 0)

    def test_mfma_spread_stays_group_local(self):
        """MFMA form: a non-finite dY value stays inside its 8-channel group
        and its receptive field widened by the zero-weight taps of its
        passes; everything else is exact, and no reference Inf / NaN turns
        finite."""
        cases = (
            # (N, H, W, C, KH, KW, PAD, dtype, block_waves, w_fold, pf, block_h)
            (2, 9, 13, 16, 7, 7, 3, "bf16", 2, 2, 2, 12),
            (1, 11, 21, 24, 5, 5, 2, "fp16", 1, 1, 1, 4),
            (3, 8, 7, 16, 7, 5, 2, "bf16", 2, 4, 2, 8),
            (1, 9, 13, 16, 9, 9, 4, "fp16", 2, 2, 3, 12),
        )
        for N, H, W, C, KH, KW, PAD, dt, wv, fold, pf, bh in cases:
            p = _problem(N, H, W, C, KH, KW, PAD, dt)
            inf_at = (
                (0, p.Ho // 2, 0, 1),
                (0, p.Ho // 2, p.Wo // 2, 3),
                (N - 1, 0, p.Wo - 1, 9),
            )
            nan_at = ((N - 1, p.Ho - 1, p.Wo // 3, C - 1),)
            spec = DirectDepthwiseDgradWindowedSpec(
                problem=p,
                block_waves=wv,
                block_h=bh,
                mfma=True,
                w_fold=fold,
                prefetch_rows=pf,
            )
            with self.subTest(case=spec.kernel_name()):
                o, ref = _launch(spec, GPU_ARCH, inf_at=inf_at, nan_at=nan_at)
                zone = _mfma_spread_zone(p, inf_at + nan_at)
                ok, _ = _elementwise_ok(o, ref)
                outside_bad = int(np.count_nonzero(~ok & ~zone))
                self.assertEqual(outside_bad, 0, "non-finite spread past the bound")
                lost = int(np.count_nonzero(~np.isfinite(ref) & np.isfinite(o)))
                self.assertEqual(
                    lost, 0, "a non-finite reference value came out finite"
                )

    def test_mfma_single_inf_spread_width(self):
        """MFMA form, one Inf per (image, 8-channel group): its bad dX
        elements stay in its receptive field rows, its image and group, and
        span at most ``4 * ceil((KW + 1) / 4)`` contiguous columns (odd and
        even filter widths alike)."""
        N, H, W, C = 2, 9, 21, 32
        for KH, KW, fold in ((5, 5, 2), (6, 6, 1), (7, 7, 2), (8, 8, 4), (9, 9, 1)):
            pad = KH // 2
            p = _problem(N, H, W, C, KH, KW, pad, "bf16")
            # One Inf per (n, group) on the first, an odd, an even inner and
            # the last dY column.
            cols = (0, 3, p.Wo // 2 + (p.Wo // 2) % 2, p.Wo - 1)
            inf_at = tuple(
                (n, p.Ho // 2, cols[(2 * n + g) % len(cols)], 8 * g + (2 * g + n) % 8)
                for n in range(N)
                for g in range(C // DW_DGRAD_MFMA_CH)
            )
            spec = DirectDepthwiseDgradWindowedSpec(
                problem=p,
                block_waves=4,
                block_h=12,
                mfma=True,
                w_fold=fold,
                prefetch_rows=2,
            )
            span = 4 * ((KW + 4) // 4)
            with self.subTest(case=spec.kernel_name()):
                o, ref = _launch(spec, GPU_ARCH, inf_at=inf_at)
                ok, _ = _elementwise_ok(o, ref)
                lost = int(np.count_nonzero(~np.isfinite(ref) & np.isfinite(o)))
                self.assertEqual(lost, 0)
                for n, ho, wo, c in inf_at:
                    g0 = c - c % DW_DGRAD_MFMA_CH
                    bad = ~ok[n, :, :, g0 : g0 + DW_DGRAD_MFMA_CH]
                    rows = np.flatnonzero(bad.any(axis=(1, 2)))
                    h0 = ho - pad
                    self.assertTrue(
                        set(rows.tolist()) <= set(range(max(0, h0), h0 + KH)),
                        f"rows {rows} past the field of {(n, ho, wo, c)}",
                    )
                    bcols = np.flatnonzero(bad.any(axis=(0, 2)))
                    if bcols.size:
                        self.assertLessEqual(
                            int(bcols.max() - bcols.min() + 1),
                            span,
                            f"columns {bcols} of {(n, ho, wo, c)}",
                        )
                    ok[n, :, :, g0 : g0 + DW_DGRAD_MFMA_CH] = True
                # Every other image / group is exact.
                self.assertEqual(int(np.count_nonzero(~ok)), 0)


if __name__ == "__main__":
    unittest.main()
