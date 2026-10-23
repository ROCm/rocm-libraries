# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the wave32 WMMA direct forward (``conv_direct_fwd_wmma``).

CPU: the spec validator, the AOT grid (gfx1250 only) and the shape filter of
``direct_spec_for_problem``. GPU (gfx1250): every spec branch -- edges and
ragged tiles, groups, PAD 0 / 5x5, several images per block, a 3-deep ring,
two WMMA k-steps per chunk, fragment prefetch, the upper VGPR banks, packed
groups, stride 2, dilation -- checked numerically against torch's conv2d.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library <torch-python> -m pytest \\
        rocke/library/tests/test_direct_fwd_wmma.py -v
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest

import pytest
from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    # Claim the HIP context for torch before rocke's runtime touches it (see
    # test_direct_conv_correctness.py).
    import torch

    torch.cuda.is_available()

from kernels.common.conv_direct_fwd_wmma import (
    DirectFwdWmmaSpec,
    build_direct_fwd_wmma,
    fwd_wmma_reason,
)
from kernels.common.conv_direct_grouped import DirectConvProblem

GPU_ARCH = get_device_arch(0)


def _problem(
    N=2,
    H=13,
    W=21,
    C=64,
    K=96,
    groups=1,
    KH=3,
    PAD=1,
    dtype="bf16",
    stride=1,
    dil_h=1,
    dil_w=1,
):
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
        stride=stride,
        dtype=dtype,
        dil_h=dil_h,
        dil_w=dil_w,
    )


class TestFwdWmmaSpec(unittest.TestCase):
    def test_wave32_tdm_only(self):
        spec = DirectFwdWmmaSpec(problem=_problem())
        self.assertIsNone(fwd_wmma_reason(spec, "gfx1250"))
        self.assertIn("wave32", fwd_wmma_reason(spec, "gfx950"))

    def test_rejects(self):
        for problem, knobs, needle in (
            (_problem(), dict(ck=48), "ck 48"),
            (_problem(C=36), {}, "multiples of 8"),
            (_problem(K=36), {}, "multiples of 8"),
            (_problem(), dict(tile_w=12), "tile_w"),
            (_problem(), dict(waves_k=1, waves_p=1), "wave"),
            (_problem(), dict(tile_w=8, tile_n=8), "tile_n"),
            (_problem(), dict(waves_k=1, waves_p=2, tiles_k=8, tiles_p=8), "VGPRs"),
            (
                _problem(),
                dict(tiles_k=8, tiles_p=8, waves_k=1, waves_p=8, ck=128),
                "LDS",
            ),
        ):
            why = fwd_wmma_reason(DirectFwdWmmaSpec(problem=problem, **knobs))
            self.assertIsNotNone(why, knobs)
            self.assertIn(needle, why)
        p = DirectConvProblem(
            N=1, H=8, W=8, groups=1, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=3
        )
        self.assertIn("stride", fwd_wmma_reason(DirectFwdWmmaSpec(problem=p)))
        for pkw, knobs, needle in (
            (dict(dil_h=2, dil_w=2, PAD=5), {}, "PAD"),
            (dict(C=32, K=32, groups=8), dict(pack_groups=3), "multiple"),
            (dict(C=24, K=24, groups=8), dict(pack_groups=8), "packed cpg"),
            (dict(C=64, K=64, groups=8), dict(pack_groups=8), "packed cpg"),
            (
                dict(C=32, K=32, groups=8),
                dict(pack_groups=8, tiles_k=1, waves_k=1),
                "block_k",
            ),
            (dict(stride=2), dict(tap_reuse=True), "tap_reuse"),
            ({}, dict(tap_reuse=True, row_ring=True), "tap_reuse"),
            ({}, dict(tap_reuse=True, tile_w=8), "tap_reuse"),
            ({}, dict(tap_shift=True), "tap_shift"),
        ):
            why = fwd_wmma_reason(DirectFwdWmmaSpec(problem=_problem(**pkw), **knobs))
            self.assertIsNotNone(why, (pkw, knobs))
            self.assertIn(needle, why)

    def test_pingpong_rejects(self):
        for knobs, needle in (
            (dict(pingpong=True, prefetch=True), "pingpong replaces"),
            (dict(pingpong=True, split_barrier=True), "pingpong replaces"),
            (dict(pingpong=True, row_ring=True), "pingpong replaces"),
            (dict(pingpong=True, wave_epilogue=True), "block epilogue"),
            (dict(pingpong=True, waves_k=1, waves_p=3), "even wave count"),
            (dict(pingpong=True, tile_w=8, tile_n=4), "rank 0"),
            (dict(pp_group=2), "pp_group"),
            (dict(pingpong=True, pp_group=0), "pp_group"),
        ):
            why = fwd_wmma_reason(DirectFwdWmmaSpec(problem=_problem(), **knobs))
            self.assertIsNotNone(why, knobs)
            self.assertIn(needle, why)
        self.assertIsNone(
            fwd_wmma_reason(
                DirectFwdWmmaSpec(problem=_problem(), pingpong=True, pp_group=3)
            )
        )

    def test_rejects_bad_knobs_without_raising(self):
        """Bad knobs are reasons, not exceptions (validate_direct_spec only
        catches ValueError)."""
        for knobs, needle in (
            (dict(tile_n=0), "tile_n"),
            (dict(tile_w=0), "tile_w"),
            (dict(tile_w=-16), "tile_w"),
            (dict(sched="no-such-strategy"), "scheduler_strategy"),
            (dict(sched=""), "scheduler_strategy"),
        ):
            why = fwd_wmma_reason(DirectFwdWmmaSpec(problem=_problem(), **knobs))
            self.assertIsNotNone(why, knobs)
            self.assertIn(needle, why)

    def test_rejects_i32_overflow(self):
        """Global byte offsets are i32: a tensor of 2 GiB or more is refused."""
        for pkw in (
            dict(N=8, H=768, W=768, C=256, K=128),  # X 2.4 GB
            dict(N=1, H=8, W=8, C=4096, K=32768),  # W 2.4 GB
        ):
            why = fwd_wmma_reason(DirectFwdWmmaSpec(problem=_problem(**pkw)))
            self.assertIsNotNone(why, pkw)
            self.assertIn("i32", why)
        self.assertIsNone(
            fwd_wmma_reason(
                DirectFwdWmmaSpec(problem=_problem(N=2, H=512, W=512, C=256, K=256))
            )
        )

    def test_ir_is_shape_independent(self):
        """One binary serves every extent, group count and channel count."""
        from rocke.core.ir_serialize import serialize

        a = build_direct_fwd_wmma(DirectFwdWmmaSpec(problem=_problem()))
        b = build_direct_fwd_wmma(
            DirectFwdWmmaSpec(problem=_problem(N=7, H=40, W=9, C=256, K=32, groups=2))
        )
        self.assertEqual(serialize(a), serialize(b))

    def test_aot_grid(self):
        from kernels.conv.direct_aot_jobs import (
            DIRECT_CAPABILITIES,
            _caps_jobs,
            _knob_grid,
        )

        direction, caps_list = DIRECT_CAPABILITIES["direct_fwd_wmma"]
        self.assertEqual(direction, "fwd")
        for arch, wave in (("gfx1250", 32), ("gfx950", 64)):
            for caps in caps_list:
                # Every knob of the cell's table builds on gfx1250 (none on
                # a wave64 arch). The tap_reuse copies of the dense rows are
                # offered to the validator, which drops the ones it cannot
                # serve.
                knobs = list(_knob_grid("direct_fwd_wmma", arch, "bf16", caps))
                base = [k for k in knobs if not k.get("tap_reuse")]
                if arch == "gfx950":
                    self.assertEqual(knobs, [])
                _, jobs = _caps_jobs(arch, wave, "direct_fwd_wmma", "fwd", caps, "bf16")
                self.assertGreaterEqual(len(jobs), len(base), (arch, caps))
                self.assertLessEqual(len(jobs), len(knobs), (arch, caps))

    def test_spec_for_problem_filters_padding(self):
        from kernels.conv.direct_variants import direct_spec_for_problem

        knobs = dict(waves_k=2, waves_p=4, tiles_k=4, tiles_p=8, tile_w=16)
        spec, why = direct_spec_for_problem(
            "direct_fwd_wmma", _problem(N=1, H=64, W=64, K=128), knobs, "gfx1250", 32
        )
        self.assertIsNotNone(spec, why)
        spec, why = direct_spec_for_problem(
            "direct_fwd_wmma", _problem(N=1, H=8, W=8, K=32), knobs, "gfx1250", 32
        )
        self.assertIsNone(spec)
        self.assertIn("tile efficiency", why)


def _gpu_skip() -> str:
    if GPU_ARCH != "gfx1250":
        return f"needs gfx1250 (got {GPU_ARCH!r})"
    if not _HAS_TORCH:
        return "torch not importable"
    return ""


_CASES = (
    # (problem kwargs, spec knobs)
    (dict(), dict(waves_k=1, waves_p=2, tiles_k=1, tiles_p=1)),
    (dict(dtype="fp16"), dict()),
    (dict(N=1, H=18, W=33, C=256, K=128), dict(tiles_k=2, tiles_p=2, stages=3)),
    (dict(H=8, W=8, groups=2), dict(tiles_k=1, tiles_p=2, tile_w=8)),
    (dict(N=3, H=7, W=9, C=40, K=32, PAD=0), dict(tiles_k=1, tiles_p=2, tile_w=8)),
    (dict(N=3, H=7, W=9, C=40, K=32, KH=5, PAD=2), dict(tiles_k=1, tiles_p=2)),
    (
        dict(N=5, H=7, W=9, C=64, K=64, dtype="fp16"),
        dict(tiles_k=2, tiles_p=2, tile_w=8, tile_n=2, stages=3),
    ),
    (
        dict(N=3, H=6, W=6, C=96, K=32),
        dict(waves_k=1, waves_p=8, tiles_k=1, tiles_p=2, tile_w=4, tile_n=4),
    ),
    (
        dict(N=1, H=20, W=40, C=160, K=128),
        dict(waves_k=1, waves_p=4, tiles_k=4, tiles_p=4, ck=64),
    ),
    # Grouped, kpg 48 over 32-channel blocks: the weight load and the Y store
    # clip at the group boundary.
    (dict(N=2, H=9, W=12, C=64, K=96, groups=2), dict(tiles_k=1, tiles_p=2)),
    # PAD = K - 1: no halo cell is left unwritten by the edge zero fill.
    (dict(N=2, H=9, W=12, C=32, K=32, PAD=2), dict(tiles_k=1, tiles_p=2)),
    (
        dict(N=1, H=30, W=40, C=64, K=256),
        dict(tiles_k=4, tiles_p=8, tile_w=32, prefetch=True, sched="iterative-ilp"),
    ),
    # Packed small groups: 8 groups of 4, 4 groups of 8 (ragged pixels, 2
    # super-groups), 2 groups of 16 with prefetch, and stride 2.
    (
        dict(N=2, H=9, W=12, C=32, K=32, groups=8),
        dict(waves_k=1, tiles_k=2, pack_groups=8),
    ),
    (
        dict(N=3, H=7, W=10, C=64, K=64, groups=8),
        dict(waves_k=1, tiles_k=2, tiles_p=2, tile_w=8, tile_n=2, pack_groups=4),
    ),
    (
        dict(N=2, H=9, W=12, C=64, K=64, groups=4, dtype="fp16"),
        dict(tiles_k=1, pack_groups=2, prefetch=True, sched="iterative-ilp"),
    ),
    (
        dict(N=2, H=13, W=11, C=64, K=64, groups=8, stride=2),
        dict(waves_k=1, tiles_k=2, tiles_p=2, tile_w=8, pack_groups=4),
    ),
    # Stride 2: odd and even extents, several images per block, the 2x2
    # patch merge, and fp16 with prefetch.
    (dict(N=2, H=13, W=21, C=96, K=64, stride=2), dict(tiles_k=1, tiles_p=2)),
    (
        dict(N=3, H=8, W=10, C=64, K=32, stride=2),
        dict(tiles_k=1, tiles_p=2, tile_w=4, tile_n=2),
    ),
    (dict(N=2, H=14, W=10, C=96, K=64, KH=2, PAD=0, stride=2), dict(tiles_k=1)),
    (
        dict(N=2, H=12, W=17, C=64, K=48, stride=2, dtype="fp16"),
        dict(tiles_k=2, tiles_p=1, ck=64, prefetch=True, sched="iterative-ilp"),
    ),
    # Stride 2 with the column phases in separate sub-halos: odd / even
    # widths and padding, several images, 2x2, packed groups, dilation.
    (dict(N=2, H=13, W=21, C=96, K=64, stride=2), dict(tiles_k=1, phase_split=True)),
    (
        dict(N=3, H=8, W=10, C=64, K=32, stride=2),
        dict(tiles_k=1, tiles_p=2, tile_w=4, tile_n=2, phase_split=True),
    ),
    (
        dict(N=2, H=14, W=11, C=96, K=64, KH=2, PAD=0, stride=2),
        dict(tiles_k=1, phase_split=True),
    ),
    (
        dict(N=2, H=12, W=17, C=64, K=48, stride=2, dtype="fp16"),
        dict(tiles_k=2, tiles_p=1, ck=64, prefetch=True, phase_split=True),
    ),
    (
        dict(N=2, H=13, W=11, C=64, K=64, groups=8, stride=2),
        dict(
            waves_k=1, tiles_k=2, tiles_p=2, tile_w=8, pack_groups=4, phase_split=True
        ),
    ),
    (
        dict(N=2, H=14, W=15, C=64, K=32, PAD=1, dil_h=3, stride=2),
        dict(tiles_k=1, tiles_p=2, tile_w=8, phase_split=True),
    ),
    # Ping-pong ranks: several chunks, a 3-deep ring, ragged channels and
    # pixels, phase groups that do not divide the steps, several images,
    # stride 2 with split halos, packed groups and fp16.
    (dict(N=1, H=18, W=33, C=256, K=128), dict(tiles_k=2, tiles_p=2, pingpong=True)),
    (
        dict(N=2, H=17, W=20, C=200, K=96),
        dict(tiles_k=2, tiles_p=2, stages=3, pingpong=True, pp_group=2),
    ),
    (
        dict(N=2, H=17, W=20, C=96, K=96),
        dict(waves_k=1, waves_p=8, tiles_k=2, tiles_p=2, pingpong=True, pp_group=3),
    ),
    (
        dict(N=5, H=7, W=9, C=64, K=64, dtype="fp16"),
        dict(tiles_k=2, tiles_p=2, tile_w=8, tile_n=2, pingpong=True),
    ),
    (
        dict(N=2, H=13, W=21, C=96, K=64, stride=2),
        dict(tiles_k=1, phase_split=True, pingpong=True, pp_group=2),
    ),
    (
        dict(N=2, H=9, W=12, C=32, K=32, groups=8),
        dict(waves_k=1, tiles_k=2, pack_groups=8, pingpong=True),
    ),
    (
        dict(N=2, H=13, W=11, C=64, K=32, PAD=1, dil_h=3),
        dict(tiles_k=1, tiles_p=2, tile_w=8, pingpong=True, sched="iterative-ilp"),
    ),
    # Weights one filter row per stage (row_ring): several chunks, a 3-deep
    # ring, ragged channels, several images, stride 2 (split halo), 5x5,
    # dilation and the per-wave epilogue.
    (dict(N=1, H=18, W=33, C=256, K=128), dict(tiles_k=2, tiles_p=2, row_ring=True)),
    (
        dict(N=2, H=17, W=20, C=200, K=96),
        dict(tiles_k=2, tiles_p=2, stages=3, prefetch=True, row_ring=True),
    ),
    (
        dict(N=5, H=7, W=9, C=64, K=64, dtype="fp16"),
        dict(tiles_k=2, tiles_p=2, tile_w=8, tile_n=2, ck=64, row_ring=True),
    ),
    (
        dict(N=2, H=13, W=21, C=96, K=64, stride=2),
        dict(tiles_k=1, phase_split=True, row_ring=True),
    ),
    (dict(N=3, H=7, W=9, C=40, K=32, KH=5, PAD=2), dict(tiles_k=1, row_ring=True)),
    (
        dict(N=2, H=13, W=11, C=64, K=32, PAD=1, dil_h=3),
        dict(tiles_k=1, tiles_p=2, row_ring=True),
    ),
    (
        dict(N=2, H=30, W=28, C=128, K=64),
        dict(
            waves_k=1,
            waves_p=7,
            tiles_k=4,
            tiles_p=7,
            tile_w=28,
            wave_epilogue=True,
            row_ring=True,
        ),
    ),
    # One tap per weight slot (tap_ring): deep chunks, a 3-deep ring, stride
    # 2 and dilation.
    (
        dict(N=1, H=18, W=33, C=256, K=128),
        dict(tiles_k=2, tiles_p=2, ck=128, tap_ring=True),
    ),
    (
        dict(N=2, H=17, W=20, C=200, K=96),
        dict(tiles_k=2, tiles_p=2, stages=3, prefetch=True, tap_ring=True),
    ),
    (
        dict(N=2, H=13, W=21, C=96, K=64, stride=2, PAD=1, dil_w=2),
        dict(tiles_k=1, tiles_p=2, tile_w=8, ck=64, tap_ring=True),
    ),
    # Padding past the filter's reach along the undilated axis: whole
    # (sub-)halos of the edge tiles are padding.
    (
        dict(N=1, H=8, W=12, C=64, K=32, PAD=4, dil_w=4),
        dict(waves_k=1, waves_p=2, tiles_k=2, tiles_p=1, tile_w=16),
    ),
    (
        dict(N=2, H=9, W=12, C=64, K=32, PAD=4, dil_w=4),
        dict(waves_k=2, waves_p=2, tiles_k=1, tiles_p=1, tile_w=16, ck=64),
    ),
    (
        dict(N=1, H=8, W=12, C=64, K=32, PAD=4, dil_w=4),
        dict(waves_k=1, waves_p=2, tiles_k=2, tiles_p=1, tile_w=16, row_ring=True),
    ),
    # Dilation, per axis, and with stride 2.
    (dict(N=2, H=13, W=21, C=64, K=32, PAD=2, dil_w=2), dict(tiles_k=1, tiles_p=2)),
    (dict(N=2, H=13, W=11, C=64, K=32, PAD=1, dil_h=3), dict(tiles_k=1, tiles_p=2)),
    (dict(N=1, H=16, W=16, C=32, K=64, PAD=0, dil_h=2, dil_w=2), dict(tiles_k=2)),
    # tap_reuse: B fragments reused across filter rows (whole rows per wave,
    # 16- and 32-wide tiles, a 5x5 filter, dilation); tap_shift on top
    # rotates them across filter columns with v_permlane16 (dilation 2 is a
    # 2-lane rotation with 2 edge lanes).
    (
        dict(N=2, H=20, W=33, C=96, K=64),
        dict(waves_k=1, waves_p=4, tiles_k=2, tiles_p=2, tap_reuse=True),
    ),
    (
        dict(N=1, H=30, W=40, C=64, K=128),
        dict(tiles_k=2, tiles_p=4, tile_w=32, prefetch=True, tap_reuse=True),
    ),
    (
        dict(N=2, H=13, W=19, C=40, K=32, KH=5, PAD=2),
        dict(tiles_k=1, tiles_p=2, tap_reuse=True, tap_shift=True),
    ),
    (
        dict(N=2, H=20, W=33, C=96, K=64, dtype="fp16"),
        dict(
            waves_k=1,
            waves_p=4,
            tiles_k=2,
            tiles_p=2,
            prefetch=True,
            tap_reuse=True,
            tap_shift=True,
        ),
    ),
    (
        dict(N=1, H=21, W=37, C=64, K=64, PAD=2, dil_h=2, dil_w=2),
        dict(tiles_k=2, tiles_p=2, tile_w=32, tap_reuse=True, tap_shift=True),
    ),
    (
        dict(N=2, H=14, W=15, C=64, K=32, PAD=1, dil_h=3, stride=2),
        dict(tiles_k=1, tiles_p=2, tile_w=8),
    ),
    # Split barrier over a 3-deep ring (4 chunks), and with prefetch.
    (
        dict(N=1, H=18, W=33, C=128, K=256),
        dict(tiles_k=2, tiles_p=2, stages=3, split_barrier=True),
    ),
    (
        dict(N=2, H=17, W=20, C=96, K=128),
        dict(tiles_k=4, tiles_p=4, prefetch=True, split_barrier=True),
    ),
    # Tile widths that are not a power of two, several images per block (the
    # last block's images past N), the per-wave epilogue.
    (
        dict(N=3, H=24, W=24, C=64, K=128),
        dict(
            waves_k=1,
            waves_p=8,
            tiles_k=2,
            tiles_p=6,
            tile_w=24,
            tile_n=4,
            split_barrier=True,
            wave_epilogue=True,
        ),
    ),
    (
        dict(N=2, H=30, W=28, C=64, K=64),
        dict(waves_k=1, waves_p=7, tiles_k=4, tiles_p=7, tile_w=28),
    ),
    # Per-wave epilogue on a grouped problem with ragged edges.
    (
        dict(N=2, H=9, W=12, C=64, K=96, groups=2),
        dict(tiles_k=1, tiles_p=2, tile_w=8, wave_epilogue=True),
    ),
)


@unittest.skipUnless(not _gpu_skip(), _gpu_skip() or "no GPU")
class TestFwdWmmaGpu(unittest.TestCase):
    def _check(self, pkw, knobs):
        from kernels.common.conv_abi import conv_direct_args_signature
        from kernels.common.conv_args import ConvArgs
        from rocke import compile_kernel
        from rocke.runtime.hip_module import Runtime
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        p = _problem(**pkw)
        spec = DirectFwdWmmaSpec(problem=p, **knobs)
        self.assertIsNone(fwd_wmma_reason(spec, GPU_ARCH))
        art = compile_kernel(build_direct_fwd_wmma(spec, GPU_ARCH), arch=GPU_ARCH)
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
            KernelLauncher(
                hsaco=art.hsaco,
                kernel_name=art.kernel_name,
                signature=conv_direct_args_signature(p.dtype, direction="fwd"),
            )(
                vals,
                config=LaunchConfig(
                    grid=spec.grid(), block=(spec.threads_per_block, 1, 1), fence=True
                ),
            )
            rt.memcpy_d2h(u8(y), bufs[2], y.nbytes)
        finally:
            for v in bufs:
                rt.free(v)
        ref = torch.nn.functional.conv2d(
            x.float().permute(0, 3, 1, 2),
            w.float().permute(0, 3, 1, 2),
            stride=p.stride,
            padding=p.PAD,
            dilation=(p.dil_h, p.dil_w),
            groups=p.groups,
        ).permute(0, 2, 3, 1)
        err = float((y.float() - ref).abs().max() / ref.abs().max())
        self.assertLess(err, 1e-2, f"{pkw} {knobs}: rel err {err:.3g}")

    def test_cases(self):
        for pkw, knobs in _CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                self._check(pkw, knobs)


if __name__ == "__main__":
    unittest.main()
