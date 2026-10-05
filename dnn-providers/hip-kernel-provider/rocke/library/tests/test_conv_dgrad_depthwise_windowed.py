# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Windowed depthwise dgrad (``build_direct_depthwise_dgrad_windowed``).

Host-only: spec validation, kernel naming and grid geometry.
GPU (gfx942 / gfx950; ``dot2`` cases gfx950 only): numeric check of dX against
an fp32 NumPy reference with the manifest conv rule -- ``bad == 0`` where an
element is bad when ``|out - ref| > tol + tol * |ref|`` (tol = 1e-2 for
fp16/bf16). dX is NaN-filled before each launch, so an element the kernel
never writes is caught as bad too.

The GPU shapes are adversarial for this kernel: widths that no block_w
divides (7, 13, 15, 17), odd heights, N = 1, channel counts that are not a
multiple of the 64-lane block, padding wider than half the filter, ragged and
even H splits, non-square filters, 1x1, and every ch_per_lane / dot2 path.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library <python> -m pytest \\
        rocke/library/tests/test_conv_dgrad_depthwise_windowed.py -v
"""

from __future__ import annotations

import ctypes
import unittest
from dataclasses import dataclass

import numpy as np

try:
    # Optional. When torch is installed its bundled HIP runtime must load
    # before rocke's launcher does; otherwise the launcher pulls in a second
    # libamdhip64 and torch.cuda reports no GPUs for every later test in this
    # pytest process.
    import torch  # noqa: F401
except ImportError:
    pass
from rocke.runtime.hip_module import get_device_arch

from kernels.common.conv_direct_grouped import (
    DW_DGRAD_WIN_MAX_UNROLL,
    DirectConvProblem,
    DirectDepthwiseDgradWindowedSpec,
    build_direct_depthwise_dgrad_windowed,
    is_valid_depthwise_dgrad_win_spec,
)

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

    def test_unroll_cap_and_block_h_relief(self):
        p = _problem(1, 64, 64, 64, 7, 7, 3)
        wide = DirectDepthwiseDgradWindowedSpec(problem=p, block_w=16)
        self.assertGreater(wide.unrolled_fmas(), DW_DGRAD_WIN_MAX_UNROLL)
        ok, why = self._ok(wide)
        self.assertFalse(ok)
        self.assertIn("unrolled body too large", why)
        split = DirectDepthwiseDgradWindowedSpec(problem=p, block_w=16, block_h=16)
        self.assertTrue(self._ok(split)[0])

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
        self.assertEqual(spec.rows_per_block, 4)
        self.assertEqual(spec.h_tiles, 4)
        self.assertEqual(spec.grid(), (2, 2, 12))
        name = spec.kernel_name()
        for part in ("r5s5p2", "bw9", "wv2", "cpl1", "bh4", "dot2", "bf16"):
            self.assertIn(part, name)
        # The filter and pad must reach the name: DirectConvProblem.short()
        # omits them, and the compile cache keys on kernel names.
        other = DirectDepthwiseDgradWindowedSpec(
            problem=_problem(3, 13, 17, 130, 3, 3, 1, "bf16"),
            block_w=9,
            block_waves=2,
            block_h=4,
            dot2=True,
        )
        self.assertNotEqual(name, other.kernel_name())
        whole = DirectDepthwiseDgradWindowedSpec(problem=p, block_h=13)
        self.assertEqual(whole.h_tiles, 1)
        self.assertNotIn("_bh", whole.kernel_name())

    def test_dot2_lowers_to_fdot2(self):
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        for dtype, intrin in (("bf16", "fdot2.f32.bf16"), ("fp16", "fdot2(")):
            spec = DirectDepthwiseDgradWindowedSpec(
                problem=_problem(1, 6, 6, 64, 3, 3, 1, dtype), block_w=6, dot2=True
            )
            ll = _lower_kernel_to_llvm_python(
                build_direct_depthwise_dgrad_windowed(spec), arch="gfx950"
            )
            self.assertIn(f"@llvm.amdgcn.{intrin}", ll)


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


def run_windowed(
    spec: DirectDepthwiseDgradWindowedSpec,
    arch: str,
    grid=None,
    block=None,
    inf_at=(),
):
    """Compile and launch ``spec``; return ``(bad, max_err)`` vs the reference.

    ``inf_at`` lists ``(n, ho, wo, c)`` dY entries set to +Inf; a dX element
    then counts as bad unless it is finite exactly where the reference is (and
    within tolerance there) and equals the reference's Inf elsewhere.
    """
    from rocke import compile_kernel
    from rocke.helpers.manifest import conv_args_signature
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    p = spec.problem
    rng = np.random.default_rng(1234)
    dy32 = rng.uniform(-1, 1, (p.N, p.Ho, p.Wo, p.groups)).astype(np.float32)
    w32 = rng.uniform(-1, 1, (p.groups, p.KH, p.KW)).astype(np.float32)
    for idx in inf_at:
        dy32[idx] = np.inf
    if p.dtype == "bf16":
        dy_b, w_b = _to_bf16_bits(dy32), _to_bf16_bits(w32)
    else:
        dy_b = dy32.astype(np.float16).view(np.uint16)
        w_b = w32.astype(np.float16).view(np.uint16)
    ref = _reference(_bits_to_f32(dy_b, p.dtype), _bits_to_f32(w_b, p.dtype), p)

    art = compile_kernel(
        build_direct_depthwise_dgrad_windowed(spec, arch=arch), arch=arch
    )
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
            signature=conv_args_signature(p.dtype),
        )
        launcher(
            {
                "A": dY_d,
                "B": W_d,
                "D": dX_d,
                "A_bytes": dy_b.nbytes,
                "B_bytes": w_b.nbytes,
                "D_bytes": dx_bytes,
            },
            config=LaunchConfig(
                grid=grid or spec.grid(),
                block=block or (spec.threads_per_block, 1, 1),
                fence=True,
            ),
        )
        out = np.empty(dx_bytes // 2, np.uint16)
        rt.memcpy_d2h(out.ctypes.data_as(ctypes.c_void_p), dX_d, dx_bytes)
    finally:
        rt.free(dY_d)
        rt.free(W_d)
        rt.free(dX_d)
        synchronize_and_release(0)
    o = _bits_to_f32(out, p.dtype).reshape(ref.shape)
    with np.errstate(invalid="ignore"):
        fin = np.isfinite(ref)
        err = np.where(fin, np.abs(o - ref), 0.0)
        ok = np.where(fin, err <= _TOL + _TOL * np.abs(ref), o == ref)
    bad = int(np.count_nonzero(~ok))
    return bad, float(np.nanmax(err))


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
    block_h: int = 0
    dot2: bool = False


_CASES = (
    _Case("w13_3x3_c72", 2, 9, 13, 72, 3, 3, 1, "fp16", 4),
    _Case("w17_5x5_n1_c66_bf16", 1, 7, 17, 66, 5, 5, 2, "bf16", 5, ch_per_lane=2),
    _Case("w7_7x7_c130_split", 3, 15, 7, 130, 7, 7, 3, "bf16", 8, 2, block_h=3),
    _Case("w15_pad0_split_ragged", 2, 11, 15, 64, 3, 3, 0, "fp16", 7, block_h=4),
    _Case("bigpad_c10", 1, 5, 6, 10, 3, 3, 2, "fp16", 4),
    _Case("w14_7x7_cpl2", 2, 14, 14, 128, 7, 7, 3, "bf16", 7, 1, 2),
    _Case("w12_3x3_cpl4", 2, 12, 12, 256, 3, 3, 1, "fp16", 6, 2, 4),
    _Case("w9_1x1", 2, 9, 9, 64, 1, 1, 0, "fp16", 9),
    _Case("w11_3x5_nonsquare", 2, 7, 11, 96, 3, 5, 1, "bf16", 11, 2),
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
            (1, 9, 13, 8, 7, 7, 3, "bf16", 13, 0, True),
            (1, 9, 13, 8, 3, 3, 1, "fp16", 5, 4, True),
            (1, 7, 11, 8, 3, 5, 0, "fp16", 4, 0, True),
            (1, 6, 9, 8, 1, 1, 0, "bf16", 9, 0, True),
            (1, 9, 13, 8, 4, 4, 2, "fp16", 7, 0, True),
            (1, 9, 13, 8, 7, 7, 3, "bf16", 13, 0, False),
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


@unittest.skipUnless(GPU_ARCH == "gfx950", f"needs gfx950, got {GPU_ARCH!r}")
class TestDispatchEndToEnd(unittest.TestCase):
    """dispatch_conv_grouped -> candidate build -> launch with its grid/block."""

    def test_dispatched_kernels_are_correct(self):
        from dispatch.grouped_convolution import (
            ConvGroupedRequest,
            dispatch_conv_grouped,
        )

        reqs = (
            {
                "N": 2,
                "C": 72,
                "Hi": 14,
                "Wi": 14,
                "Y": 7,
                "X": 7,
                "pad": 3,
                "dtype": "bf16",
            },
            {
                "N": 1,
                "C": 130,
                "Hi": 13,
                "Wi": 17,
                "Y": 5,
                "X": 5,
                "pad": 2,
                "dtype": "fp16",
            },
            {
                "N": 2,
                "C": 256,
                "Hi": 12,
                "Wi": 12,
                "Y": 3,
                "X": 3,
                "pad": 1,
                "dtype": "fp16",
            },
            {
                "N": 1,
                "C": 96,
                "Hi": 40,
                "Wi": 28,
                "Y": 7,
                "X": 7,
                "pad": 3,
                "dtype": "bf16",
            },
            {
                "N": 1,
                "C": 64,
                "Hi": 35,
                "Wi": 56,
                "Y": 3,
                "X": 3,
                "pad": 1,
                "dtype": "fp16",
            },
        )
        for r in reqs:
            req = ConvGroupedRequest(
                N=r["N"],
                C=r["C"],
                K=r["C"],
                G=r["C"],
                Hi=r["Hi"],
                Wi=r["Wi"],
                Y=r["Y"],
                X=r["X"],
                pad_h=r["pad"],
                pad_w=r["pad"],
                dtype=r["dtype"],
                arch="gfx950",
                direction="dgrad",
            )
            res = dispatch_conv_grouped(req)
            with self.subTest(req=r, kernel=res.spec.kernel_name()):
                bad, max_err = run_windowed(
                    res.spec.instance, "gfx950", grid=res.grid, block=res.block
                )
                self.assertEqual(bad, 0, f"bad={bad} max_err={max_err:.3e}")


if __name__ == "__main__":
    unittest.main()
