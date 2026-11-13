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
    POLY_ZERO_LANE_CH,
    DirectFwdWmmaSpec,
    build_direct_fwd_wmma,
    fwd_wmma_reason,
    poly_view_zero_channels,
)
from kernels.common.conv_direct_grouped import DirectConvProblem

GPU_ARCH = get_device_arch(0)


# A one-row tiling a 1-D poly_view takes (tile_w == block_p).
_PV1 = dict(waves_k=1, waves_p=4, tiles_k=1, tiles_p=2, tile_w=128, stages=2)


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
    KW=None,
    pad_w=-1,
):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=C // groups,
        kpg=K // groups,
        KH=KH,
        KW=KH if KW is None else KW,
        PAD=PAD,
        pad_w=pad_w,
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
            (dict(KH=1, KW=3, PAD=0, pad_w=3, dil_w=3, stride=2), {}, "1 x R"),
            (dict(KH=3, KW=1, PAD=2, pad_w=0, dil_h=2), {}, "1 x R"),
            (dict(KH=7, KW=1, PAD=3, pad_w=0), dict(tap_reuse=True), "tap_reuse"),
            (dict(KH=1, KW=7, PAD=0, pad_w=3), dict(phase_split=True), "phase split"),
            (dict(KH=1, KW=7, PAD=0, pad_w=19, dil_w=3), {}, "PAD_W 19"),
            (
                dict(H=1, W=600, KH=1, KW=3, PAD=0, pad_w=1, stride=2),
                dict(poly_view=True),
                "R = m * S",
            ),
            (
                dict(H=2, W=600, KH=1, KW=4, PAD=0, pad_w=1, stride=2),
                dict(poly_view=True),
                "1-D",
            ),
            ({}, dict(poly_view=True), "1-D"),
            (
                dict(H=1, W=512, KH=1, KW=8, PAD=0, pad_w=1, stride=4),
                dict(poly_view=True),
                "S / 2",
            ),
            (
                dict(H=1, W=600, KH=1, KW=4, PAD=0, pad_w=1, stride=2, C=288, K=64),
                dict(poly_view=True, ck=288, **_PV1),
                "at most 256 channels",
            ),
            (
                dict(H=1, W=600, KH=1, KW=4, PAD=0, pad_w=1, stride=2, C=512, K=64),
                dict(poly_view=True, ck=512, **_PV1),
                "at most 256 channels",
            ),
            (
                dict(C=128, K=128, groups=2, KH=2, PAD=0, stride=2),
                dict(strided_view=True),
                "ungrouped",
            ),
            (dict(KH=2, PAD=0, stride=2), dict(strided_view=True, ck=0), "ck 0"),
            (
                dict(H=1, W=600, KH=1, KW=4, PAD=0, pad_w=1, stride=2),
                dict(poly_view=True, ck=0),
                "ck 0",
            ),
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

    @staticmethod
    def _extent_families():
        """(spec, Python validator, engine family) for every user of the
        forward WMMA gate: direct, grouped direct, gstream, tap-outer, small C
        and small K."""
        from kernels.common.conv_direct_fwd_gstream_wmma import (
            FwdGstreamWmmaSpec,
            fwd_gstream_reason,
        )
        from kernels.common.conv_direct_fwd_tapouter_wmma import (
            DirectFwdTapOuterSpec,
            tapouter_reason,
        )
        from kernels.common.conv_fwd_small_wmma import (
            FwdSmallcWmmaSpec,
            FwdSmallkWmmaSpec,
            fwd_smallc_reason,
            fwd_smallk_reason,
        )

        return (
            (
                DirectFwdWmmaSpec(problem=_problem()),
                fwd_wmma_reason,
                "direct_fwd_wmma",
            ),
            (
                DirectFwdWmmaSpec(problem=_problem(C=64, K=64, groups=4)),
                fwd_wmma_reason,
                "direct_fwd_wmma",
            ),
            (
                FwdGstreamWmmaSpec(problem=_problem(C=128, K=128, groups=8)),
                fwd_gstream_reason,
                "direct_fwd_gstream_wmma",
            ),
            (
                DirectFwdTapOuterSpec(
                    problem=_problem(H=29, W=37, PAD=12, dil_h=12, dil_w=12)
                ),
                tapouter_reason,
                "direct_fwd_tapouter_wmma",
            ),
            (
                FwdSmallcWmmaSpec(problem=_problem(H=32, W=48, C=3, K=64)),
                fwd_smallc_reason,
                "direct_fwd_smallc_wmma",
            ),
            (
                FwdSmallkWmmaSpec(problem=_problem(H=32, W=48, C=64, K=4)),
                fwd_smallk_reason,
                "direct_fwd_smallk_wmma",
            ),
        )

    @staticmethod
    def _non_positive_extents(base):
        """(spec, refusal text) for each extent of ``base`` set to 0, -1, -16."""
        import dataclasses

        from kernels.common.conv_direct_grouped import PROBLEM_DIMS

        for name in PROBLEM_DIMS:
            for value in (0, -1, -16):
                problem = dataclasses.replace(base.problem, **{name: value})
                spec = dataclasses.replace(base, problem=problem)
                yield spec, f"{name} must be >= 1 (got {value})"

    def test_gate_refuses_non_positive_extents(self):
        """Every user of the forward WMMA gate refuses a zero or negative
        extent first, with problem_dims_reason's text. groups = 0 used to
        pass: K = 0 is a multiple of every block_k and the tensors are 0
        bytes. Needs no C++ engine, so CI runs it."""
        for base, reason, _ in self._extent_families():
            self.assertIsNone(reason(base, "gfx1250"), base)
            for spec, want in self._non_positive_extents(base):
                self.assertEqual(reason(spec, "gfx1250"), want, spec)

    def test_engines_refuse_non_positive_extents(self):
        """The C++ engine refuses the same extents with the same text."""
        eng = pytest.importorskip("rocke_library_engine")
        from kernels.conv import engine_bridge as be

        engines = {
            "direct_fwd_wmma": (
                eng.conv_direct_fwd_wmma_describe,
                be.conv_direct_fwd_wmma_spec_to_dict,
            ),
            "direct_fwd_gstream_wmma": (
                eng.conv_direct_fwd_gstream_wmma_describe,
                be.conv_direct_fwd_gstream_wmma_spec_to_dict,
            ),
            "direct_fwd_tapouter_wmma": (
                eng.conv_direct_fwd_tapouter_wmma_describe,
                be.conv_direct_fwd_tapouter_wmma_spec_to_dict,
            ),
            "direct_fwd_smallc_wmma": (
                eng.conv_direct_fwd_smallc_wmma_describe,
                be.conv_direct_fwd_smallc_wmma_spec_to_dict,
            ),
            "direct_fwd_smallk_wmma": (
                eng.conv_direct_fwd_smallk_wmma_describe,
                be.conv_direct_fwd_smallk_wmma_spec_to_dict,
            ),
        }
        for base, _, family in self._extent_families():
            describe, to_dict = engines[family]
            self.assertTrue(describe(to_dict(base), "gfx1250")[0], base)
            for spec, want in self._non_positive_extents(base):
                ok, why_cpp, *_ = describe(to_dict(spec), "gfx1250")
                self.assertFalse(ok, spec)
                self.assertEqual(why_cpp, want, spec)

    def test_engines_agree_on_specs(self):
        """The C++ engine's validator, kernel name, grid and LDS size match
        Python's over the AOT grid and a set of rejected specs (the
        byte-identity gate only covers the IR of specs both engines build)."""
        eng = pytest.importorskip("rocke_library_engine")
        from kernels.conv.direct_aot_jobs import (
            DIRECT_CAPABILITIES,
            _knob_grid,
            probe_problem,
        )
        from kernels.conv.engine_bridge import conv_direct_fwd_wmma_spec_to_dict

        specs = []
        _, caps_list = DIRECT_CAPABILITIES["direct_fwd_wmma"]
        for caps in caps_list:
            for dtype in ("bf16", "fp16"):
                probe = probe_problem(caps, dtype, "direct_fwd_wmma")
                for knobs in _knob_grid("direct_fwd_wmma", "gfx1250", dtype, caps):
                    specs.append(DirectFwdWmmaSpec(problem=probe, **knobs))
        for pkw, knobs in (
            ({}, dict(ck=48)),
            (dict(C=36), {}),
            ({}, dict(tile_w=12)),
            ({}, dict(waves_k=1, waves_p=1)),
            ({}, dict(tile_w=8, tile_n=8)),
            ({}, dict(waves_k=1, waves_p=2, tiles_k=8, tiles_p=8)),
            ({}, dict(tiles_k=8, tiles_p=8, waves_k=1, waves_p=8, ck=128)),
            ({}, dict(tile_n=0)),
            ({}, dict(tile_w=-16)),
            ({}, dict(sched="no-such-strategy")),
            ({}, dict(sched="")),
            (dict(stride=3), {}),
            (dict(stride=0), {}),
            (dict(dil_h=2, dil_w=2, PAD=5), {}),
            (dict(dil_h=0), {}),
            (dict(KH=3, PAD=1, H=1, W=1), {}),
            (dict(C=32, K=32, groups=8), dict(pack_groups=3)),
            (dict(C=24, K=24, groups=8), dict(pack_groups=8)),
            (dict(C=64, K=64, groups=8), dict(pack_groups=8)),
            (dict(C=32, K=32, groups=8), dict(pack_groups=8, tiles_k=1, waves_k=1)),
            ({}, dict(pack_groups=0)),
            (dict(stride=2), dict(tap_reuse=True)),
            ({}, dict(tap_reuse=True, row_ring=True)),
            ({}, dict(tap_reuse=True, tile_w=8)),
            ({}, dict(tap_reuse=True, pingpong=True)),
            ({}, dict(tap_shift=True)),
            (dict(PAD=16, dil_w=16), dict(tap_reuse=True, tap_shift=True)),
            ({}, dict(phase_split=True)),
            (dict(stride=2), dict(phase_split=True, tile_w=8, tile_n=4)),
            ({}, dict(strided_view=True)),
            (dict(KH=1, PAD=0, stride=2), dict(strided_view=True)),
            (dict(KH=1, PAD=0, stride=2), dict(strided_view=True, phase_split=True)),
            (dict(KH=2, PAD=0, stride=2, C=48), dict(strided_view=True, ck=64)),
            (dict(KH=2, PAD=0, stride=2, H=14), dict(strided_view=True, fold_n=True)),
            (dict(KH=2, PAD=0, stride=2, H=13), dict(strided_view=True, fold_n=True)),
            ({}, dict(fold_n=True)),
            (dict(KH=2, PAD=0, stride=2), dict(strided_view=True, ck=0)),
            (
                dict(H=1, W=600, KH=1, KW=4, PAD=0, pad_w=1, stride=2),
                dict(poly_view=True, ck=0),
            ),
            (
                dict(C=128, K=128, groups=2, KH=2, PAD=0, stride=2),
                dict(strided_view=True),
            ),
            (
                dict(C=128, K=128, groups=2, KH=1, PAD=0, stride=2),
                dict(strided_view=True),
            ),
            (
                dict(H=1, W=512, KH=1, KW=8, PAD=0, pad_w=1, stride=4),
                dict(poly_view=True),
            ),
            (
                dict(H=1, W=600, KH=1, KW=4, PAD=0, pad_w=1, stride=2, C=288, K=64),
                dict(poly_view=True, ck=288, **_PV1),
            ),
            (
                dict(H=1, W=600, KH=1, KW=4, PAD=0, pad_w=0, stride=2, C=288, K=64),
                dict(poly_view=True, ck=288, **_PV1),
            ),
            (
                dict(H=1, W=600, KH=1, KW=4, PAD=0, pad_w=1, stride=2, C=256, K=64),
                dict(poly_view=True, ck=256, **_PV1),
            ),
            (
                dict(H=1, W=600, KH=1, KW=4, PAD=0, pad_w=1, stride=2),
                dict(poly_view=True, tile_w=128, waves_k=2, waves_p=2),
            ),
            (
                dict(H=1, W=600, KH=1, KW=3, PAD=0, pad_w=1, stride=2),
                dict(poly_view=True),
            ),
            (
                dict(H=1, W=600, KH=1, KW=4, PAD=0, pad_w=1, stride=2),
                dict(poly_view=True, strided_view=True),
            ),
            (
                dict(H=1, W=600, KH=1, KW=4, PAD=0, pad_w=1, stride=2, C=96),
                dict(poly_view=True, ck=64),
            ),
            ({}, dict(wave_epilogue=True, tiles_p=3)),
            ({}, dict(row_ring=True, tap_ring=True)),
            ({}, dict(row_ring=True, split_barrier=True)),
            ({}, dict(stages=1)),
            ({}, dict(pingpong=True, prefetch=True)),
            ({}, dict(pingpong=True, wave_epilogue=True)),
            ({}, dict(pingpong=True, waves_k=1, waves_p=3)),
            ({}, dict(pingpong=True, tile_w=8, tile_n=4)),
            ({}, dict(pp_group=2)),
            ({}, dict(waves_k=8, waves_p=8)),
            (dict(N=8, H=768, W=768, C=256, K=128), {}),
            (dict(N=1, H=8, W=8, C=4096, K=32768), {}),
            (dict(dtype="fp32"), {}),
            # 2^31 channels: total_c / total_k must not wrap in C++.
            (dict(N=1, H=8, W=8, C=1 << 31, K=1 << 31, groups=1 << 25), {}),
        ):
            specs.append(DirectFwdWmmaSpec(problem=_problem(**pkw), **knobs))
        for arch in ("gfx1250", "gfx950"):
            for spec in specs:
                ok, _, name, grid, lds = eng.conv_direct_fwd_wmma_describe(
                    conv_direct_fwd_wmma_spec_to_dict(spec), arch
                )
                why = fwd_wmma_reason(spec, arch)
                self.assertEqual(ok, why is None, (arch, spec, why))
                self.assertEqual(name, spec.kernel_name())
                self.assertEqual(lds, spec.lds_bytes())
                if why is None:
                    self.assertEqual(tuple(grid), spec.grid())

    def test_engines_agree_on_scheduler_reason(self):
        """The shared validator tail refuses an unknown scheduler with
        normalize_scheduler_strategy's text, choices included, in both
        engines."""
        eng = pytest.importorskip("rocke_library_engine")
        from kernels.conv.engine_bridge import conv_direct_fwd_wmma_spec_to_dict
        from rocke.core.codegen_policy import normalize_scheduler_strategy

        for sched in ("no-such-strategy", ""):
            with self.assertRaises(ValueError) as cm:
                normalize_scheduler_strategy(sched)
            spec = DirectFwdWmmaSpec(problem=_problem(), sched=sched)
            why = fwd_wmma_reason(spec)
            self.assertEqual(why, str(cm.exception))
            self.assertIn("expected one of: iterative-ilp", why)
            ok, why_cpp, *_ = eng.conv_direct_fwd_wmma_describe(
                conv_direct_fwd_wmma_spec_to_dict(spec), "gfx1250"
            )
            self.assertFalse(ok)
            self.assertEqual(why_cpp, why)

    def test_binding_rejects_bad_input(self):
        """A rejected spec returns the rejection (no grid is computed from its
        zero tile extents); a missing, mistyped or partial problem raises."""
        eng = pytest.importorskip("rocke_library_engine")
        from kernels.conv.engine_bridge import conv_direct_fwd_wmma_spec_to_dict

        for knobs in (dict(waves_k=0), dict(tiles_k=0), dict(waves_p=0)):
            sd = conv_direct_fwd_wmma_spec_to_dict(
                DirectFwdWmmaSpec(problem=_problem(), **knobs)
            )
            ok, why, _, grid, _ = eng.conv_direct_fwd_wmma_describe(sd, "gfx1250")
            self.assertFalse(ok, knobs)
            self.assertIn("waves and tiles", why)
            self.assertEqual(tuple(grid), (0, 0, 0))
        # A zero stride is rejected, not divided by -- also by the phase-split
        # LDS size describe reports for a rejected spec.
        sd = conv_direct_fwd_wmma_spec_to_dict(
            DirectFwdWmmaSpec(problem=_problem(stride=0), phase_split=True)
        )
        ok, why, _, grid, lds = eng.conv_direct_fwd_wmma_describe(sd, "gfx1250")
        self.assertFalse(ok)
        self.assertIn("stride 0", why)
        self.assertEqual((tuple(grid), lds), ((0, 0, 0), 0))
        with self.assertRaises(Exception):
            eng.conv_direct_fwd_wmma_lower_llvm(sd, arch="gfx1250")
        good = conv_direct_fwd_wmma_spec_to_dict(DirectFwdWmmaSpec(problem=_problem()))
        # An explicit dtype=None is rejected (Python rejects the spec too),
        # not read as the fp16 default.
        self.assertIsNotNone(
            fwd_wmma_reason(DirectFwdWmmaSpec(problem=_problem(dtype=None)))
        )
        sd = dict(good, problem=dict(good["problem"], dtype=None))
        with self.assertRaisesRegex(ValueError, "dtype"):
            eng.conv_direct_fwd_wmma_describe(sd, "gfx1250")
        with self.assertRaisesRegex(ValueError, "dtype"):
            eng.conv_direct_fwd_wmma_lower_llvm(sd, arch="gfx1250")
        partial = dict(good["problem"])
        del partial["kpg"]
        for problem in (None, "N2H13", partial):
            sd = dict(good)
            if problem is None:
                del sd["problem"]
            else:
                sd["problem"] = problem
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                eng.conv_direct_fwd_wmma_describe(sd, "gfx1250")
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                eng.conv_direct_fwd_wmma_lower_llvm(sd, arch="gfx1250")

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
        # A 1 x R problem keeps its own column padding.
        row = _problem(N=1, H=1, W=300, KH=1, KW=7, PAD=0, pad_w=9, dil_w=3)
        knobs = dict(waves_k=1, waves_p=4, tiles_k=2, tiles_p=4, tile_w=256)
        spec, why = direct_spec_for_problem(
            "direct_fwd_wmma", row, knobs, "gfx1250", 32
        )
        self.assertIsNotNone(spec, why)
        self.assertEqual(spec.problem.PAD_W, 9)

    def test_padded_poly_view_chunk_fits_one_zeroing_wave(self):
        """A leading-pad chunk's first view pixel is zeroed by one wave, 8
        channels a lane: with column padding, ck above 256 would leave the
        rest of the chunk stale, so it is refused; ck = 256, or any ck
        without column padding, passes this rule."""
        self.assertEqual(POLY_ZERO_LANE_CH, 8)
        self.assertEqual(poly_view_zero_channels(32), 256)
        for C, ck, pad_w, refused in (
            (288, 288, 1, True),
            (512, 512, 1, True),
            (256, 256, 1, False),
            (288, 288, 0, False),
        ):
            p = _problem(H=1, W=600, KH=1, KW=4, PAD=0, pad_w=pad_w, stride=2, C=C)
            why = fwd_wmma_reason(
                DirectFwdWmmaSpec(problem=p, poly_view=True, ck=ck, **_PV1)
            )
            if refused:
                self.assertEqual(
                    why,
                    f"a padded poly_view zeroes at most 256 channels of a chunk (ck {ck})",
                )
            else:
                self.assertIsNone(why, (C, ck, pad_w))

    def test_shipped_strided_views_refuse_grouped_patches(self):
        """A 2x2 stride-2 patch run is KW * cpg contiguous channels only when
        one group holds a pixel's channels: every shipped 2x2 strided_view row
        refuses a grouped problem, while a 1x1 stride-2 one still serves it."""
        from kernels.conv.direct_aot_jobs import _fwd_wmma_sv_knobs
        from kernels.conv.direct_variants import direct_spec_for_problem

        rows = list(_fwd_wmma_sv_knobs())
        self.assertTrue(rows)
        for kh in (1, 2):
            dense = _problem(N=4, H=64, W=64, C=256, K=256, KH=kh, PAD=0, stride=2)
            grouped = _problem(
                N=4, H=64, W=64, C=256, K=256, groups=2, KH=kh, PAD=0, stride=2
            )
            for knobs in rows:
                if knobs.get("fold_n"):
                    continue
                spec, why = direct_spec_for_problem(
                    "direct_fwd_wmma", dense, knobs, "gfx1250", 32
                )
                self.assertIsNotNone(spec, (kh, knobs, why))
                spec, why = direct_spec_for_problem(
                    "direct_fwd_wmma", grouped, knobs, "gfx1250", 32
                )
                if kh == 1:
                    self.assertIsNotNone(spec, (knobs, why))
                else:
                    self.assertIsNone(spec, knobs)
                    self.assertIn("ungrouped", why)


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
    # Strided GEMM (strided_view): 1x1 s2 on odd extents with a ragged last
    # chunk, grouped; the batch folded into the rows (tiles across images);
    # 2x2 s2 with several chunks per patch run, folded, with prefetch.
    (
        dict(N=3, H=13, W=19, C=40, K=48, KH=1, PAD=0, stride=2),
        dict(tiles_k=1, tiles_p=2, tile_w=8, strided_view=True),
    ),
    (
        dict(N=2, H=14, W=18, C=128, K=64, groups=2, KH=1, PAD=0, stride=2),
        dict(tiles_k=1, tiles_p=2, tile_w=8, strided_view=True),
    ),
    (
        dict(N=5, H=14, W=22, C=96, K=80, KH=1, PAD=0, stride=2, dtype="fp16"),
        dict(tiles_k=2, tiles_p=2, tile_w=8, ck=64, strided_view=True, fold_n=True),
    ),
    (
        dict(N=3, H=10, W=18, C=48, K=64, KH=2, PAD=0, stride=2),
        dict(tiles_k=2, tiles_p=2, tile_w=8, strided_view=True, fold_n=True),
    ),
    (
        dict(N=2, H=12, W=14, C=64, K=96, KH=2, PAD=0, stride=2, dtype="fp16"),
        dict(
            tiles_k=1,
            tiles_p=2,
            tile_w=16,
            ck=64,
            stages=3,
            prefetch=True,
            sched="iterative-ilp",
            strided_view=True,
        ),
    ),
    # n1 deep 1x1 s2 downsample rows: 32-channel x 64-pixel blocks, 128-channel
    # chunks.
    (
        dict(N=1, H=32, W=32, C=256, K=96, KH=1, PAD=0, stride=2),
        dict(
            waves_k=2,
            waves_p=1,
            tiles_k=1,
            tiles_p=1,
            tile_w=8,
            ck=128,
            stages=3,
            prefetch=True,
            sched="iterative-ilp",
            strided_view=True,
        ),
    ),
    # Row-ring stride-2 rows: phase-split 40-wide tiles of a 40x40 output,
    # and four images of 8x8 outputs per block.
    (
        dict(H=80, W=80, C=64, K=128, stride=2, dtype="fp16"),
        dict(
            waves_k=2,
            waves_p=5,
            tiles_k=4,
            tiles_p=4,
            tile_w=40,
            stages=3,
            phase_split=True,
            row_ring=True,
            prefetch=True,
            sched="iterative-ilp",
        ),
    ),
    (
        dict(N=6, H=16, W=16, C=64, K=128, stride=2),
        dict(
            waves_k=2,
            waves_p=4,
            tiles_k=4,
            tiles_p=4,
            tile_w=8,
            tile_n=4,
            stages=3,
            row_ring=True,
        ),
    ),
    # 1-D strided filters over a polyphase view (one output row per block).
    (
        dict(H=1, W=512, C=64, K=64, KH=1, KW=16, PAD=0, pad_w=4, stride=8),
        dict(
            waves_k=1,
            waves_p=4,
            tiles_k=4,
            tiles_p=4,
            tile_w=256,
            ck=64,
            poly_view=True,
            prefetch=True,
            sched="iterative-ilp",
        ),
    ),
    (
        dict(N=3, H=1, W=600, KH=1, KW=4, PAD=0, pad_w=1, stride=2, dtype="fp16"),
        dict(
            waves_k=2,
            waves_p=2,
            tiles_k=2,
            tiles_p=4,
            tile_w=128,
            stages=3,
            poly_view=True,
        ),
    ),
    (
        dict(N=2, H=1, W=1000, C=128, K=32, KH=1, KW=16, PAD=0, pad_w=4, stride=8),
        dict(
            waves_k=2,
            waves_p=2,
            tiles_k=1,
            tiles_p=4,
            tile_w=128,
            ck=64,
            stages=3,
            poly_view=True,
        ),
    ),
    # A column padding of its own on a square filter.
    (dict(N=3, H=11, W=19, PAD=1, pad_w=0), dict(tiles_k=1, tiles_p=2, tile_w=8)),
    # 1 x R "same" filters on 1-D signals (column padding of their own).
    (
        dict(H=1, W=301, C=64, K=64, KH=1, KW=3, PAD=0, pad_w=1),
        dict(waves_k=1, waves_p=4, tiles_k=2, tiles_p=4, tile_w=256),
    ),
    (
        dict(H=1, W=257, KH=1, KW=7, PAD=0, pad_w=9, dil_w=3, dtype="fp16"),
        dict(
            waves_k=2,
            waves_p=4,
            tiles_k=4,
            tiles_p=4,
            tile_w=256,
            stages=3,
            prefetch=True,
            split_barrier=True,
        ),
    ),
    (
        dict(N=3, H=1, W=200, C=32, K=64, KH=1, KW=11, PAD=0, pad_w=25, dil_w=5),
        dict(waves_k=1, waves_p=8, tiles_k=2, tiles_p=4, tile_w=512),
    ),
    (
        dict(H=1, W=400, C=128, K=128, KH=1, KW=11, PAD=0, pad_w=5),
        dict(waves_k=2, waves_p=4, tiles_k=4, tiles_p=8, tile_w=512, pingpong=True),
    ),
    # Rectangular filters with per-axis "same" padding (Inception 17x17 and
    # 8x8 blocks), two images of 4x20 / 8x8 per block.
    (
        dict(N=3, H=17, W=17, KH=7, KW=1, PAD=3, pad_w=0),
        dict(waves_k=1, waves_p=4, tiles_k=4, tiles_p=5, tile_w=20, tile_n=2),
    ),
    (
        dict(N=3, H=17, W=17, KH=1, KW=7, PAD=0, pad_w=3, dtype="fp16"),
        dict(
            waves_k=2,
            waves_p=2,
            tiles_k=1,
            tiles_p=5,
            tile_w=20,
            tile_n=2,
            stages=3,
            prefetch=True,
            sched="iterative-ilp",
        ),
    ),
    (
        dict(N=3, H=8, W=8, KH=3, KW=1, PAD=1, pad_w=0),
        dict(waves_k=4, waves_p=2, tiles_k=2, tiles_p=2, tile_w=8, stages=3),
    ),
    (
        dict(N=3, H=8, W=8, KH=1, KW=3, PAD=0, pad_w=1),
        dict(waves_k=2, waves_p=2, tiles_k=2, tiles_p=4, tile_w=8, tile_n=2),
    ),
    # 3x3 stride 2 "valid" on odd maps (35x35 -> 17x17, 17x17 -> 8x8).
    (
        dict(N=3, H=35, W=35, C=64, K=64, PAD=0, stride=2),
        dict(waves_k=2, waves_p=2, tiles_k=2, tiles_p=5, tile_w=20, tile_n=2),
    ),
    (
        dict(N=3, H=17, W=17, C=64, K=64, PAD=0, stride=2, dtype="fp16"),
        dict(
            waves_k=4,
            waves_p=2,
            tiles_k=2,
            tiles_p=5,
            tile_w=20,
            tile_n=2,
            phase_split=True,
            prefetch=True,
            sched="iterative-ilp",
        ),
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
            padding=(p.PAD, p.PAD_W),
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
