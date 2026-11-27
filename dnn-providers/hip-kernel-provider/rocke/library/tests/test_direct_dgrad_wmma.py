# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the wave32 WMMA direct dgrad (``conv_direct_dgrad_wmma``).

CPU: the spec validator, the AOT grid (gfx1250 only), the shape filter of
``direct_spec_for_problem`` and the shipped path: ``direct_plans`` over a
gfx1250 cache of placeholder binaries (capability match, re-validation, launch
grid, block and kernargs). GPU (gfx1250): every spec branch -- edges and
ragged tiles, groups, PAD 0 / 5x5, several images per block, a 3-deep ring,
two WMMA k-steps per chunk, fragment prefetch, the upper VGPR banks -- checked
numerically against a float64 numpy reference of the stored operands.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library python -m pytest \\
        rocke/library/tests/test_direct_dgrad_wmma.py -v
"""

from __future__ import annotations

import json
import re
import unittest

import numpy as np
import pytest
from rocke.numeric import storage
from rocke.numeric.compare import peak_relative_error
from rocke.runtime.hip_module import get_device_arch
from rocke.tests.lanes import BUILD_VARIANT, arch_skip_reason, import_or_skip_until

from kernels.common.conv_direct_dgrad_wmma import (
    DirectDgradWmmaSpec,
    build_direct_dgrad_wmma,
    dgrad_wmma_reason,
)
from kernels.common.conv_direct_grouped import DirectConvProblem

from ._direct_conv_numeric import direct_dgrad_reference, launch_direct_conv, poisoned

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
        stride=stride,
        dtype=dtype,
        dil_h=dil_h,
        dil_w=dil_w,
        pad_w=pad_w,
    )


class TestDgradWmmaSpec(unittest.TestCase):
    def test_wave32_tdm_only(self):
        spec = DirectDgradWmmaSpec(problem=_problem())
        self.assertIsNone(dgrad_wmma_reason(spec, "gfx1250"))
        self.assertIn("wave32", dgrad_wmma_reason(spec, "gfx950"))

    def test_rejects(self):
        for problem, knobs, needle in (
            (_problem(), dict(ck=48), "ck 48"),
            (_problem(C=36), {}, "multiples of 8"),
            (_problem(), dict(tile_w=12), "tile_w"),
            (_problem(), dict(waves_c=1, waves_p=1), "wave"),
            (_problem(), dict(tile_w=8, tile_n=8), "tile_n"),
            (_problem(), dict(waves_c=1, waves_p=2, tiles_c=8, tiles_p=8), "VGPRs"),
            (_problem(), dict(tiles_c=8, tiles_p=4, waves_c=4, ck=64), "LDS"),
        ):
            why = dgrad_wmma_reason(DirectDgradWmmaSpec(problem=problem, **knobs))
            self.assertIsNotNone(why, knobs)
            self.assertIn(needle, why)
        # Only dX channels need 16-byte rows; the dY tail is zero-filled.
        for k in (36, 37):
            self.assertIsNone(
                dgrad_wmma_reason(DirectDgradWmmaSpec(problem=_problem(K=k)))
            )
        p = DirectConvProblem(
            N=1, H=8, W=8, groups=1, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=3
        )
        self.assertIn("stride", dgrad_wmma_reason(DirectDgradWmmaSpec(problem=p)))
        for pkw, knobs, needle in (
            (dict(stride=2, dil_h=2, dil_w=2, PAD=2), {}, "strided dilated"),
            (dict(dil_h=2, dil_w=2, PAD=5), {}, "PAD"),
            (dict(C=32, K=32, groups=8), dict(pack_groups=3), "multiple"),
            (dict(C=24, K=24, groups=8), dict(pack_groups=8), "packed cpg"),
            (dict(C=64, K=64, groups=8), dict(pack_groups=8), "packed kpg"),
            (
                dict(C=32, K=32, groups=8),
                dict(pack_groups=8, tiles_c=1, waves_c=1),
                "block_c",
            ),
        ):
            why = dgrad_wmma_reason(
                DirectDgradWmmaSpec(problem=_problem(**pkw), **knobs)
            )
            self.assertIsNotNone(why, (pkw, knobs))
            self.assertIn(needle, why)

    def test_rejects_nonpositive_filter(self):
        """A zero or negative filter extent is a rejection reason, not an
        exception from the empty tap plan."""
        for KH, KW in ((0, 3), (3, 0), (0, 0), (-1, 3), (3, -2)):
            spec = DirectDgradWmmaSpec(
                problem=_problem(KH=KH, KW=KW),
                prefetch=True,
                sched="iterative-ilp",
            )
            why = dgrad_wmma_reason(spec)
            self.assertEqual(why, f"filter {KH}x{KW} is not positive")
            with self.assertRaises(ValueError):
                build_direct_dgrad_wmma(spec)

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
            why = dgrad_wmma_reason(DirectDgradWmmaSpec(problem=_problem(), **knobs))
            self.assertIsNotNone(why, knobs)
            self.assertIn(needle, why)

    def test_rejects_i32_overflow(self):
        """Global byte offsets are i32: a tensor of 2 GiB or more is refused."""
        for pkw in (
            dict(N=8, H=768, W=768, C=256, K=256),  # dX and dY 2.4 GB
            dict(N=1, H=8, W=8, C=4096, K=32768),  # W 2.4 GB
        ):
            why = dgrad_wmma_reason(DirectDgradWmmaSpec(problem=_problem(**pkw)))
            self.assertIsNotNone(why, pkw)
            self.assertIn("i32", why)
        self.assertIsNone(
            dgrad_wmma_reason(
                DirectDgradWmmaSpec(problem=_problem(N=2, H=512, W=512, C=256, K=256))
            )
        )

    def test_engines_agree_on_specs(self):
        """The C++ engine's validator, kernel name, grid and LDS size match
        Python's over every cell of the AOT grid (stride 2, dilation, packed
        groups, the split barrier and the per-wave epilogue included) and a
        set of rejected specs (the byte-identity gate only covers the IR of
        specs both engines build)."""
        eng = import_or_skip_until("rocke_library_engine", BUILD_VARIANT)
        from kernels.conv.direct_aot_jobs import DIRECT_CAPABILITIES, _knob_grid
        from kernels.conv.engine_bridge import conv_direct_dgrad_wmma_spec_to_dict

        def caps_problem(caps, dtype):
            # A packed cell bakes cpg = kpg; 32 groups cover every pack size.
            if caps.cpg:
                ch = dict(groups=32, C=32 * caps.cpg, K=32 * caps.kpg)
            else:
                ch = {}
            return _problem(
                N=3,
                H=29,
                W=37,
                KH=caps.KH,
                KW=caps.KW,
                PAD=caps.PAD,
                pad_w=caps.pad_w,
                stride=caps.stride,
                dil_h=caps.dil_h,
                dil_w=caps.dil_w,
                dtype=dtype,
                **ch,
            )

        specs = []
        _, caps_list = DIRECT_CAPABILITIES["direct_dgrad_wmma"]
        for dtype in ("bf16", "fp16"):
            for caps in caps_list:
                for knobs in _knob_grid("direct_dgrad_wmma", "gfx1250", dtype, caps):
                    specs.append(
                        DirectDgradWmmaSpec(problem=caps_problem(caps, dtype), **knobs)
                    )
        n_grid = len(specs)
        self.assertGreater(n_grid, 300)
        for pkw, knobs in (
            ({}, dict(ck=48)),
            ({}, dict(tile_n=0)),
            ({}, dict(sched="bogus")),
            (dict(C=36), {}),
            (dict(N=8, H=768, W=768, C=256, K=256), {}),
            (dict(KH=3, PAD=3), {}),
            # Stride / dilation.
            (dict(stride=3), {}),
            (dict(stride=2), {}),  # four phases of 4x4 tiles: VGPRs
            (dict(stride=2, PAD=2, dil_h=2, dil_w=2), dict(tiles_c=2)),
            (dict(dil_h=2, dil_w=2, PAD=5), {}),
            (dict(dil_h=0), {}),
            # Nonpositive filter extents (an empty tap plan).
            (dict(KH=0, KW=3), dict(prefetch=True)),
            (dict(KH=3, KW=0), {}),
            (dict(KH=-1, KW=3), {}),
            (dict(KH=3, KW=-2), dict(prefetch=True)),
            (dict(N=8, H=768, W=768, C=256, K=256, stride=2), dict(tiles_c=2)),
            # Packed groups.
            (dict(C=32, K=32, groups=8), dict(pack_groups=3)),
            (dict(C=24, K=24, groups=8), dict(pack_groups=8)),
            (dict(C=64, K=64, groups=8), dict(pack_groups=8)),
            (dict(C=32, K=32, groups=8), dict(pack_groups=8, tiles_c=1, waves_c=1)),
            (dict(C=32, K=32, groups=8), dict(pack_groups=0)),
            # Split barrier / per-wave epilogue.
            (dict(stride=2), dict(tiles_c=2, split_barrier=True)),
            (dict(C=32, K=32, groups=8), dict(pack_groups=4, wave_epilogue=True)),
            ({}, dict(tiles_p=1, waves_p=4, tile_w=32, wave_epilogue=True)),
            ({}, dict(tile_w=12)),
            # Per-axis padding, ping-pong and the weight rings.
            (dict(stride=2, pad_w=0), dict(tiles_c=2)),
            (dict(pad_w=-2), {}),
            (dict(KH=1, KW=3, PAD=0, pad_w=3), {}),
            ({}, dict(pingpong=True, prefetch=True)),
            ({}, dict(pingpong=True, split_barrier=True)),
            (dict(stride=2), dict(pingpong=True, tiles_c=2)),
            (dict(C=32, K=32, groups=8), dict(pingpong=True, pack_groups=4)),
            ({}, dict(pingpong=True, waves_c=1, waves_p=3, tile_w=12)),
            ({}, dict(pingpong=True, tile_w=8, tile_n=4)),
            ({}, dict(pingpong=True, wave_epilogue=True)),
            ({}, dict(pp_group=2)),
            ({}, dict(pingpong=True, pp_group=0)),
            ({}, dict(pingpong=True, tiles_c=4, tiles_p=8, pp_group=3)),
            ({}, dict(tap_ring=True)),
            ({}, dict(row_ring=True, prefetch=True)),
            ({}, dict(pingpong=True, row_ring=True, tap_ring=True)),
            (dict(C=512, K=512), dict(tiles_c=8, tiles_p=2, pingpong=True)),
        ):
            specs.append(DirectDgradWmmaSpec(problem=_problem(**pkw), **knobs))
        n_rejected = 0
        for arch in ("gfx1250", "gfx950"):
            for i, spec in enumerate(specs):
                ok, why_c, name, grid, lds = eng.conv_direct_dgrad_wmma_describe(
                    conv_direct_dgrad_wmma_spec_to_dict(spec), arch
                )
                why = dgrad_wmma_reason(spec, arch)
                self.assertEqual(ok, why is None, (arch, spec, why))
                if why is not None:
                    self.assertEqual(why_c, why, (arch, spec))
                self.assertEqual(name, spec.kernel_name())
                self.assertEqual(lds, spec.lds_bytes())
                if why is None:
                    self.assertEqual(tuple(grid), spec.grid())
                elif arch == "gfx1250":
                    if i < n_grid:
                        # The AOT grid drops these: a dilation above 2 widens
                        # the dY halo past the LDS of the largest tiles.
                        p = spec.problem
                        self.assertGreater(max(p.dil_h, p.dil_w), 2, why)
                        self.assertIn("LDS", why)
                    else:
                        n_rejected += 1
        # Every extra spec is a reject on gfx1250.
        self.assertEqual(n_rejected, len(specs) - n_grid)

    def test_engines_quote_the_scheduler_like_repr(self):
        """The C++ refusal of an unknown scheduler quotes the name as Python's
        ``{value!r}`` does: the quote character, the escapes, a printable
        non-ASCII name (the subset rocke_py_str_repr documents)."""
        eng = import_or_skip_until("rocke_library_engine", BUILD_VARIANT)
        from kernels.conv.engine_bridge import conv_direct_dgrad_wmma_spec_to_dict

        for name in ("bogus", "it's", "a\\b", "x'\"y", "tab\there", "caf\u00e9"):
            with self.subTest(name=name):
                spec = DirectDgradWmmaSpec(problem=_problem(), sched=name)
                why = dgrad_wmma_reason(spec, "gfx1250")
                self.assertTrue(
                    why.startswith(f"unsupported scheduler_strategy {name!r};"), why
                )
                ok, why_c, *_ = eng.conv_direct_dgrad_wmma_describe(
                    conv_direct_dgrad_wmma_spec_to_dict(spec), "gfx1250"
                )
                self.assertFalse(ok)
                self.assertEqual(why_c, why)

    def test_binding_rejects_bad_input(self):
        """A rejected spec returns the rejection (no grid is computed from its
        zero tile extents); a missing, mistyped or partial problem raises."""
        eng = import_or_skip_until("rocke_library_engine", BUILD_VARIANT)
        from kernels.conv.engine_bridge import conv_direct_dgrad_wmma_spec_to_dict

        for knobs in (dict(waves_c=0), dict(tiles_c=0), dict(waves_p=0)):
            sd = conv_direct_dgrad_wmma_spec_to_dict(
                DirectDgradWmmaSpec(problem=_problem(), **knobs)
            )
            ok, why, _, grid, _ = eng.conv_direct_dgrad_wmma_describe(sd, "gfx1250")
            self.assertFalse(ok, knobs)
            self.assertIn("waves and tiles", why)
            self.assertEqual(tuple(grid), (0, 0, 0))
        good = conv_direct_dgrad_wmma_spec_to_dict(
            DirectDgradWmmaSpec(problem=_problem())
        )
        partial = dict(good["problem"])
        del partial["kpg"]
        for problem in (None, "N2H13", partial):
            sd = dict(good)
            if problem is None:
                del sd["problem"]
            else:
                sd["problem"] = problem
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                eng.conv_direct_dgrad_wmma_describe(sd, "gfx1250")
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                eng.conv_direct_dgrad_wmma_lower_llvm(sd, arch="gfx1250")

    def test_binding_rejects_nonpositive_filter(self):
        """The C++ engine refuses a zero or negative filter extent with
        Python's reason, before planning its (empty) taps."""
        eng = import_or_skip_until("rocke_library_engine", BUILD_VARIANT)
        from kernels.conv.engine_bridge import conv_direct_dgrad_wmma_spec_to_dict

        for KH, KW in ((0, 3), (3, 0), (-1, 3), (3, -2)):
            spec = DirectDgradWmmaSpec(
                problem=_problem(KH=KH, KW=KW), prefetch=True, sched="iterative-ilp"
            )
            sd = conv_direct_dgrad_wmma_spec_to_dict(spec)
            ok, why, _, _, _ = eng.conv_direct_dgrad_wmma_describe(sd, "gfx1250")
            self.assertFalse(ok, (KH, KW))
            self.assertEqual(why, dgrad_wmma_reason(spec))
            with self.assertRaisesRegex(RuntimeError, "is not positive"):
                eng.conv_direct_dgrad_wmma_lower_llvm(sd, arch="gfx1250")

    def test_ir_is_shape_independent(self):
        """One binary serves every extent, group count and channel count."""
        from rocke.core.ir_serialize import serialize

        a = build_direct_dgrad_wmma(DirectDgradWmmaSpec(problem=_problem()))
        b = build_direct_dgrad_wmma(
            DirectDgradWmmaSpec(problem=_problem(N=7, H=40, W=9, C=256, K=32, groups=2))
        )
        self.assertEqual(serialize(a), serialize(b))

    def test_aot_grid(self):
        from kernels.conv.direct_aot_jobs import (
            DIRECT_CAPABILITIES,
            _caps_jobs,
            _knob_grid,
            probe_problem,
        )
        from kernels.conv.direct_variants import (
            make_direct_spec,
            validate_direct_spec,
        )

        direction, caps_list = DIRECT_CAPABILITIES["direct_dgrad_wmma"]
        self.assertEqual(direction, "dgrad")
        for arch, wave in (("gfx1250", 32), ("gfx950", 64)):
            for caps in caps_list:
                # Every knob of the cell's table builds on gfx1250 (none on
                # a wave64 arch), except that a dilation above 2 widens the
                # dY halo past the LDS budget of the largest tiles.
                grid = list(_knob_grid("direct_dgrad_wmma", arch, "bf16", caps))
                if arch == "gfx950":
                    self.assertEqual(grid, [])
                _, jobs = _caps_jobs(
                    arch, wave, "direct_dgrad_wmma", "dgrad", caps, "bf16"
                )
                if arch == "gfx950" or max(caps.dil_h, caps.dil_w) <= 2:
                    self.assertEqual(len(jobs), len(grid), (arch, caps))
                    continue
                self.assertTrue(jobs, (arch, caps))
                probe = probe_problem(caps, "bf16", "direct_dgrad_wmma")
                checks = [
                    validate_direct_spec(
                        "direct_dgrad_wmma",
                        make_direct_spec("direct_dgrad_wmma", probe, k, wave),
                        arch,
                    )
                    for k in grid
                ]
                self.assertEqual(sum(ok for ok, *_ in checks), len(jobs), (arch, caps))
                for ok, why, *_ in checks:
                    if not ok:
                        self.assertIn("LDS", why, (arch, caps))

    def test_spec_for_problem_filters_padding(self):
        from kernels.conv.direct_variants import direct_spec_for_problem

        knobs = dict(waves_c=2, waves_p=4, tiles_c=4, tiles_p=8, tile_w=16)
        spec, why = direct_spec_for_problem(
            "direct_dgrad_wmma", _problem(N=1, H=64, W=64, C=128), knobs, "gfx1250"
        )
        self.assertIsNotNone(spec, why)
        spec, why = direct_spec_for_problem(
            "direct_dgrad_wmma", _problem(N=1, H=8, W=8, C=32), knobs, "gfx1250"
        )
        self.assertIsNone(spec)
        self.assertIn("tile efficiency", why)


# ---------------------------------------------------------------------------
# The shipped path: the gfx1250 AOT cache (placeholder binaries, no GPU) and
# direct_plans, which matches capabilities, re-validates the knobs on the
# real problem and plans the launch.
# ---------------------------------------------------------------------------
_ARCH = "gfx1250"
_WMMA = "direct_dgrad_wmma"


@pytest.fixture(scope="module")
def wmma_cache(tmp_path_factory):
    """Every cached direct_dgrad_wmma job of the gfx1250 grid, each holding a
    placeholder HSACO."""
    from benchmarks.common import direct_kernel_sweep as dks
    from benchmarks.common.kernel_cache import KernelCache
    from rocke.core.arch import ArchTarget

    jobs = [
        j
        for j in dks.enumerate_direct_jobs(
            arch=_ARCH, target=ArchTarget.from_gfx(_ARCH), directions=("dgrad",)
        )
        if j.identity.algorithm == _WMMA
    ]
    assert jobs
    cache = KernelCache(tmp_path_factory.mktemp("dgrad_wmma_cache"), _ARCH)
    for job in jobs:
        meta = {"kernel_name": job.identity.label(), "spec_wave_size": 32}
        cache.put(job.identity, b"placeholder", meta)
    return cache


def _wmma_plans(cache, problem):
    from benchmarks.common import direct_kernel_sweep as dks

    plans, rejected = dks.direct_plans(cache, problem, "dgrad", _ARCH)
    return (
        [p for p in plans if p.identity.algorithm == _WMMA],
        [(i, why) for i, why in rejected if i.algorithm == _WMMA],
    )


def _cache_problem(**kw):
    # The cache is built at the direct-conv AOT dtype.
    from benchmarks.common.direct_kernel_sweep import DIRECT_DTYPE

    return _problem(**dict(dict(N=2, H=28, W=28, dtype=DIRECT_DTYPE), **kw))


def _ceil(a, b):
    return -(-a // b)


@pytest.mark.parametrize(
    "pkw, pack",
    [
        (dict(C=128, K=128), 1),
        (dict(C=64, K=96, PAD=0), 1),
        (dict(C=64, K=64, groups=2), 1),
        (dict(C=64, K=64, stride=2), 1),
        (dict(C=64, K=64, KH=2, PAD=0, stride=2), 1),
        (dict(C=64, K=64, PAD=2, dil_h=2, dil_w=2), 1),
        (dict(C=64, K=64, PAD=4, dil_h=4, dil_w=4), 1),
        (dict(C=64, K=64, PAD=6, dil_h=6, dil_w=6), 1),
        (dict(C=128, K=128, groups=32), 8),
        (dict(C=256, K=256, groups=32), 4),
        (dict(C=256, K=256, groups=16), 2),
        # Small groups at stride 2: packed super-group + phase decomposition.
        (dict(C=128, K=128, groups=32, stride=2), 8),
        (dict(C=256, K=256, groups=32, stride=2), 4),
        (dict(C=512, K=512, groups=32, stride=2), 2),
    ],
)
def test_aot_cache_plans_launch(wmma_cache, pkw, pack):
    """A gfx1250 dgrad problem gets direct_dgrad_wmma binaries built for its
    exact filter, padding, stride and dilation (and, for small groups, its
    baked channel counts with 32 // cpg packed groups); each plan is one
    dgrad-ABI launch over dY, W, dX whose grid, block and kernargs follow the
    runtime problem."""
    from benchmarks.common import direct_kernel_sweep as dks
    from kernels.common.conv_args import ConvArgs

    problem = _cache_problem(**pkw)
    plans, _ = _wmma_plans(wmma_cache, problem)
    assert plans, pkw
    assert pack in {json.loads(p.identity.knobs).get("pack_groups", 1) for p in plans}
    S = problem.stride
    for plan in plans:
        ident = plan.identity
        assert (ident.filter_h, ident.filter_w) == (problem.KH, problem.KW)
        assert (ident.pad_h, ident.stride_h) == (problem.PAD, S)
        assert (ident.dilation_h, ident.dilation_w) == (problem.dil_h, problem.dil_w)
        knobs = json.loads(ident.knobs)
        # cpg 8 and 16 are served by the runtime-channel binaries too.
        g = knobs.get("pack_groups", 1)
        assert g in (1, pack)
        assert (ident.cpg, ident.kpg) == (
            (problem.cpg, problem.kpg) if g > 1 else (0, 0)
        )
        (step,) = plan.steps
        assert step.buffers == ("dy", "w", "dx")
        assert step.views is None or not any(step.views)
        threads = knobs["waves_c"] * knobs["waves_p"] * 32
        assert step.block == (threads, 1, 1)
        tile_n = knobs.get("tile_n", 1)
        block_p = knobs["waves_p"] * knobs["tiles_p"] * 16
        tile_h = block_p // (knobs["tile_w"] * tile_n)
        block_c = knobs["waves_c"] * knobs["tiles_c"] * 16
        assert step.grid == (
            _ceil(problem.N, tile_n)
            * (problem.groups // g)
            * _ceil(_ceil(problem.H, S), tile_h)
            * _ceil(_ceil(problem.W, S), knobs["tile_w"])
            * _ceil(g * problem.cpg, block_c),
            1,
            1,
        ), ident.label()
        assert step.conv_args == ConvArgs.from_problem(problem, direction="dgrad")
        values = dks.launch_values(
            step,
            {"dy": 1 << 20, "w": 2 << 20, "dx": 3 << 20},
            {"dy": 1, "w": 2, "dx": 3},
        )
        assert (values["A"], values["B"], values["D"]) == (1 << 20, 2 << 20, 3 << 20)
        assert (values["p_Ho"], values["p_Wo"]) == (problem.Ho, problem.Wo)
        # The kernel divides the runtime group count by pack_groups itself.
        assert values["p_groups"] == problem.groups
        assert values["p_total_c"] == problem.total_c
        assert values["p_total_k"] == problem.total_k


@pytest.mark.parametrize(
    "pkw, needle",
    [
        # 12 groups of 4 channels: the cpg-4 binaries pack 8 groups.
        (dict(C=48, K=48, groups=12), "not a multiple of pack_groups"),
        # 6 groups of 8: the cpg-8 binaries pack 4.
        (dict(C=48, K=48, groups=6), "not a multiple of pack_groups"),
    ],
)
def test_aot_cache_rejects_unpackable_group_counts(wmma_cache, pkw, needle):
    plans, rejected = _wmma_plans(wmma_cache, _cache_problem(**pkw))
    assert plans == []
    packed = [why for i, why in rejected if json.loads(i.knobs).get("pack_groups")]
    assert packed and all(needle in why for why in packed)


@pytest.mark.parametrize(
    "pkw",
    [
        # Dilations the cache has no binary for.
        dict(C=64, K=64, PAD=3, dil_h=3, dil_w=3),
        dict(C=64, K=64, PAD=2, dil_h=2),
        dict(C=64, K=64, PAD=2, dil_w=2),
        dict(C=64, K=64, PAD=1, dil_h=3),
        # A strided dilated dgrad is not supported.
        dict(C=64, K=64, PAD=2, dil_h=2, dil_w=2, stride=2),
        # Stride 2 with "valid" padding: only "same" 3x3 is cached.
        dict(C=64, K=64, PAD=0, stride=2),
    ],
)
def test_aot_cache_offers_nothing_for_uncached_geometry(wmma_cache, pkw):
    """No binary of another dilation, stride or padding is offered (the
    capability match is exact), so these problems get no direct_dgrad_wmma
    plan at all."""
    plans, rejected = _wmma_plans(wmma_cache, _cache_problem(**pkw))
    assert plans == [] and rejected == []


def _gpu_skip() -> str:
    if GPU_ARCH != "gfx1250":
        return arch_skip_reason(("gfx1250",), f"needs gfx1250 (got {GPU_ARCH!r})")
    return ""


_CASES = (
    # (problem kwargs, spec knobs)
    (dict(), dict(waves_c=1, waves_p=2, tiles_c=1, tiles_p=1)),
    (dict(dtype="fp16"), dict()),
    (dict(N=1, H=18, W=33, C=256, K=128), dict(tiles_c=2, tiles_p=2, stages=3)),
    (dict(H=8, W=8, groups=2), dict(tiles_c=1, tiles_p=2, tile_w=8)),
    (dict(N=3, H=7, W=9, C=32, K=40, PAD=0), dict(tiles_c=1, tiles_p=2, tile_w=8)),
    (dict(N=3, H=7, W=9, C=32, K=40, KH=5, PAD=2), dict(tiles_c=1, tiles_p=2)),
    (
        dict(N=5, H=7, W=9, C=64, K=64, dtype="fp16"),
        dict(tiles_c=2, tiles_p=2, tile_w=8, tile_n=2, stages=3),
    ),
    (
        dict(N=3, H=6, W=6, C=32, K=96),
        dict(waves_c=1, waves_p=8, tiles_c=1, tiles_p=2, tile_w=4, tile_n=4),
    ),
    (
        dict(N=1, H=20, W=40, C=128, K=160),
        dict(waves_c=1, waves_p=4, tiles_c=4, tiles_p=4, ck=64),
    ),
    # Grouped, cpg 48 over 32-channel blocks: the weight load and the dX
    # store clip at the group boundary.
    (dict(N=2, H=9, W=12, C=96, K=64, groups=2), dict(tiles_c=1, tiles_p=2)),
    # kpg not a multiple of 8: the last dY chunk is clipped and zero-filled
    # by TDM, with 72-byte (and odd-element) pixel strides.
    (dict(N=2, H=13, W=21, C=64, K=36), dict(tiles_c=1, tiles_p=2, ck=64)),
    (dict(N=2, H=9, W=12, C=64, K=74, groups=2), dict(tiles_c=1, tiles_p=2)),
    (dict(N=2, H=13, W=21, C=64, K=37, dtype="fp16"), dict(tiles_c=1, tiles_p=2)),
    # PAD = K - 1: no halo cell is left unwritten (no zero fill).
    (dict(N=2, H=9, W=12, C=32, K=32, PAD=2), dict(tiles_c=1, tiles_p=2)),
    (
        dict(N=1, H=30, W=40, C=256, K=64),
        dict(tiles_c=4, tiles_p=8, tile_w=32, prefetch=True, sched="iterative-ilp"),
    ),
    # Packed small groups: 8 groups of 4, 4 groups of 8 (ragged pixels, 2
    # super-groups), 2 groups of 16 with prefetch.
    (dict(N=2, H=9, W=12, C=32, K=32, groups=8), dict(tiles_c=1, pack_groups=8)),
    (
        dict(N=3, H=7, W=10, C=64, K=64, groups=8),
        dict(waves_c=1, tiles_c=2, tiles_p=2, tile_w=8, tile_n=2, pack_groups=4),
    ),
    (
        dict(N=2, H=9, W=12, C=64, K=64, groups=4, dtype="fp16"),
        dict(tiles_c=1, pack_groups=2, prefetch=True, sched="iterative-ilp"),
    ),
    # Packed small groups at stride 2 (4 phases of accumulators per block):
    # 4 groups of 8 with ragged extents, 2 groups of 16 in fp16 with
    # prefetch, and the ResNeXt downsampling layers (cpg 8 and 16).
    (
        dict(N=2, H=13, W=21, C=64, K=64, groups=8, stride=2),
        dict(waves_c=1, tiles_c=2, tiles_p=2, pack_groups=4),
    ),
    (
        dict(N=3, H=12, W=17, C=64, K=64, groups=4, stride=2, dtype="fp16"),
        dict(
            waves_c=1,
            tiles_c=2,
            tiles_p=4,
            pack_groups=2,
            prefetch=True,
            sched="iterative-ilp",
        ),
    ),
    (
        dict(N=16, H=56, W=56, C=256, K=256, groups=32, stride=2),
        dict(waves_c=1, waves_p=4, tiles_c=2, tiles_p=4, pack_groups=4),
    ),
    (
        dict(N=16, H=28, W=28, C=512, K=512, groups=32, stride=2, dtype="fp16"),
        dict(waves_c=1, waves_p=8, tiles_c=2, tiles_p=4, tile_w=8, pack_groups=2),
    ),
    # Stride 2: odd and even extents, several images per block, the 2x2
    # patch merge, and fp16 with prefetch.
    (dict(N=2, H=13, W=21, C=64, K=96, stride=2), dict(tiles_c=1, tiles_p=2)),
    (
        dict(N=3, H=8, W=10, C=32, K=64, stride=2),
        dict(tiles_c=1, tiles_p=2, tile_w=4, tile_n=2),
    ),
    (dict(N=2, H=14, W=10, C=64, K=96, KH=2, PAD=0, stride=2), dict(tiles_c=1)),
    (
        dict(N=2, H=12, W=17, C=48, K=64, stride=2, dtype="fp16"),
        dict(tiles_c=2, tiles_p=2, ck=64, prefetch=True, sched="iterative-ilp"),
    ),
    # Dilation, per axis.
    (dict(N=2, H=13, W=21, C=32, K=64, PAD=2, dil_w=2), dict(tiles_c=1, tiles_p=2)),
    (dict(N=2, H=13, W=11, C=32, K=64, PAD=1, dil_h=3), dict(tiles_c=1, tiles_p=2)),
    (dict(N=1, H=16, W=16, C=64, K=32, PAD=0, dil_h=2, dil_w=2), dict(tiles_c=2)),
    # The symmetric "same" dilations the AOT cache carries beyond 2.
    (dict(N=2, H=20, W=19, C=64, K=64, PAD=4, dil_h=4, dil_w=4), dict(tiles_c=2)),
    (
        dict(N=1, H=26, W=23, C=32, K=64, PAD=6, dil_h=6, dil_w=6),
        dict(tiles_c=1, tiles_p=2),
    ),
    # Split barrier over a 3-deep ring (4 chunks), and with prefetch.
    (
        dict(N=1, H=18, W=33, C=256, K=128),
        dict(tiles_c=2, tiles_p=2, stages=3, split_barrier=True),
    ),
    (
        dict(N=2, H=17, W=20, C=128, K=96),
        dict(tiles_c=4, tiles_p=4, prefetch=True, split_barrier=True),
    ),
    # Tile widths that are not a power of two, several images per block (the
    # last block's images past N), the per-wave epilogue.
    (
        dict(N=3, H=24, W=24, C=128, K=64),
        dict(
            waves_c=1,
            waves_p=8,
            tiles_c=2,
            tiles_p=6,
            tile_w=24,
            tile_n=4,
            split_barrier=True,
            wave_epilogue=True,
        ),
    ),
    (
        dict(N=2, H=30, W=28, C=64, K=64),
        dict(waves_c=1, waves_p=7, tiles_c=4, tiles_p=7, tile_w=28),
    ),
    # Per-wave epilogue on a grouped problem with ragged edges.
    (
        dict(N=2, H=9, W=12, C=96, K=64, groups=2),
        dict(tiles_c=1, tiles_p=2, tile_w=8, wave_epilogue=True),
    ),
)
# Ping-pong (two wave ranks a barrier apart): one and several chunks, step
# groups of 1-3 (the last group short), a 3-deep ring, two images per block,
# PAD 0, grouped, and the 50-wide tile on a ragged map.
_PP_CASES = (
    (dict(), dict(waves_c=1, waves_p=4, tiles_c=1, tiles_p=1, pingpong=True)),
    (
        dict(N=1, H=18, W=33, C=256, K=160, dtype="fp16"),
        dict(tiles_c=2, tiles_p=2, pingpong=True, pp_group=2),
    ),
    (
        dict(N=1, H=18, W=33, C=128, K=192),
        dict(tiles_c=2, tiles_p=2, stages=3, pingpong=True, pp_group=3),
    ),
    (
        dict(N=3, H=7, W=9, C=64, K=96, PAD=0),
        dict(tiles_c=2, tiles_p=2, tile_w=8, tile_n=2, pingpong=True),
    ),
    (
        dict(N=2, H=9, W=12, C=96, K=64, groups=2),
        dict(tiles_c=1, tiles_p=2, pingpong=True, sched="iterative-ilp"),
    ),
    (
        dict(N=2, H=19, W=57, C=128, K=96),
        dict(
            waves_c=2,
            waves_p=5,
            tiles_c=4,
            tiles_p=5,
            tile_w=50,
            pingpong=True,
            pp_group=2,
            sched="iterative-ilp",
        ),
    ),
)

# Ping-pong with a weight ring (tap_ring / row_ring): several chunks and ring
# steps, ragged K and C, 3 stages, pp_group not dividing a step (row_ring:
# 3 steps per slot), two images per block, fp16, a 256-channel tile, 1x3 /
# 3x1 per-axis padding and PAD 0.
_PP_RING_CASES = (
    (
        dict(N=2, H=13, W=21, C=64, K=96),
        dict(tiles_c=2, tiles_p=2, pingpong=True, tap_ring=True),
    ),
    (
        dict(N=1, H=18, W=33, C=88, K=200, dtype="fp16"),
        dict(tiles_c=2, tiles_p=2, pingpong=True, row_ring=True, stages=3, pp_group=2),
    ),
    (
        dict(N=1, H=11, W=19, C=160, K=136),
        dict(tiles_c=2, tiles_p=2, ck=64, pingpong=True, tap_ring=True, stages=3),
    ),
    (
        dict(N=3, H=7, W=9, C=64, K=96, PAD=0),
        dict(tiles_c=2, tiles_p=2, tile_w=8, tile_n=2, pingpong=True, row_ring=True),
    ),
    (
        dict(N=1, H=12, W=20, C=384, K=96),
        dict(
            waves_c=2,
            waves_p=4,
            tiles_c=8,
            tiles_p=2,
            ck=32,
            pingpong=True,
            tap_ring=True,
            pp_group=2,
        ),
    ),
    (
        dict(N=2, H=9, W=14, C=64, K=64, KH=1, KW=3, PAD=0, pad_w=1),
        dict(tiles_c=2, tiles_p=2, pingpong=True, tap_ring=True, stages=3),
    ),
    (
        dict(N=2, H=9, W=14, C=64, K=96, KH=3, KW=1, PAD=1, pad_w=0, dtype="fp16"),
        dict(tiles_c=2, tiles_p=2, pingpong=True, row_ring=True),
    ),
)

# Packed small groups at stride 2 with the widest pack: 8 groups of 4
# channels per super-group (cpg = kpg = 4, pack_groups 8), one case per dtype
# on ragged extents (odd H and W, an odd W alone), with the knobs of two
# DGRAD_WMMA_PACKED_STRIDED_KNOBS rows (the second with prefetch).
_PACK8_STRIDED_CASES = (
    (
        dict(N=2, H=13, W=21, C=64, K=64, groups=16, stride=2),
        dict(waves_c=1, waves_p=4, tiles_c=2, tiles_p=2, tile_w=16, pack_groups=8),
    ),
    (
        dict(N=3, H=12, W=17, C=32, K=32, groups=8, stride=2, dtype="fp16"),
        dict(
            waves_c=1,
            waves_p=4,
            tiles_c=2,
            tiles_p=4,
            tile_w=16,
            pack_groups=8,
            prefetch=True,
            sched="iterative-ilp",
        ),
    ),
)


class TestDgradWmmaPingpong(unittest.TestCase):
    def test_cases_validate_and_lower(self):
        for pkw, knobs in _PP_CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                spec = DirectDgradWmmaSpec(problem=_problem(**pkw), **knobs)
                self.assertIsNone(dgrad_wmma_reason(spec, "gfx1250"))
                self.assertIn("_pp", spec.kernel_name())
                build_direct_dgrad_wmma(spec, "gfx1250")

    def test_rejects(self):
        pp = dict(pingpong=True)
        for pkw, knobs, needle in (
            ({}, dict(pp, split_barrier=True), "replaces"),
            ({}, dict(pp, prefetch=True), "replaces"),
            (dict(stride=2), dict(pp, tiles_c=2), "stride-1"),
            (dict(C=32, K=32, groups=8), dict(pp, pack_groups=4), "stride-1"),
            ({}, dict(pp, waves_c=1, waves_p=3), "even wave count"),
            ({}, dict(pp, tile_w=8, tile_n=4), "rank 0"),
            ({}, dict(pp, wave_epilogue=True), "block epilogue"),
            ({}, dict(pp_group=2), "pp_group"),
            ({}, dict(pp, pp_group=0), "pp_group"),
            ({}, dict(pp, tiles_c=4, tiles_p=8, pp_group=3), "VGPRs"),
        ):
            why = dgrad_wmma_reason(
                DirectDgradWmmaSpec(problem=_problem(**pkw), **knobs)
            )
            self.assertIsNotNone(why, (pkw, knobs))
            self.assertIn(needle, why)


class TestDgradWmmaPingpongRing(unittest.TestCase):
    def test_cases_validate_and_lower(self):
        for pkw, knobs in _PP_RING_CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                spec = DirectDgradWmmaSpec(problem=_problem(**pkw), **knobs)
                self.assertIsNone(dgrad_wmma_reason(spec, "gfx1250"))
                self.assertRegex(spec.kernel_name(), "_pp.*_(rr|tr)")
                build_direct_dgrad_wmma(spec, "gfx1250")

    def test_ring_shrinks_the_weight_slots(self):
        p = _problem(C=512, K=512)
        knobs = dict(waves_c=2, waves_p=4, tiles_c=8, tiles_p=2, pingpong=True)
        full = DirectDgradWmmaSpec(problem=p, **knobs)
        tap = DirectDgradWmmaSpec(problem=p, tap_ring=True, **knobs)
        self.assertEqual(full.wei_elems, 9 * tap.wei_elems)
        self.assertIn("LDS", dgrad_wmma_reason(full))
        self.assertIsNone(dgrad_wmma_reason(tap))

    def test_rejects(self):
        for pkw, knobs, needle in (
            ({}, dict(tap_ring=True), "pingpong"),
            (dict(stride=2), dict(pingpong=True, tap_ring=True, tiles_c=2), "stride-1"),
            (
                dict(C=32, K=32, groups=8),
                dict(pingpong=True, row_ring=True, pack_groups=4),
                "stride-1",
            ),
            ({}, dict(pingpong=True, tap_ring=True, row_ring=True), "exclusive"),
        ):
            why = dgrad_wmma_reason(
                DirectDgradWmmaSpec(problem=_problem(**pkw), **knobs)
            )
            self.assertIsNotNone(why, (pkw, knobs))
            self.assertIn(needle, why)


class TestDgradWmmaNewKnobsStructure(unittest.TestCase):
    """IR structure and builder refusals of per-axis padding, pingpong,
    pp_group and the weight rings (CPU only)."""

    @staticmethod
    def _ops(spec):
        from rocke.core.ir_serialize import serialize

        text = serialize(build_direct_dgrad_wmma(spec, "gfx1250"))
        return {
            op: len(re.findall(rf"\b{op}\b", text))
            for op in ("s_barrier_bare", "sched_barrier", "tensor_load_to_lds", "mma")
        }

    def test_pingpong_barrier_pairs(self):
        """One barrier pair per group of pp_group steps (the 3x3 filter has
        nine steps per 32-channel chunk), plus the rank-1 lead barrier and the
        rank-0 catch-up barrier; the WMMA count does not change."""
        base = dict(tiles_c=2, tiles_p=2)
        plain = self._ops(DirectDgradWmmaSpec(problem=_problem(), **base))
        self.assertEqual(plain["s_barrier_bare"], 0)
        for pg in (1, 2, 3, 4):
            with self.subTest(pp_group=pg):
                ops = self._ops(
                    DirectDgradWmmaSpec(
                        problem=_problem(), pingpong=True, pp_group=pg, **base
                    )
                )
                self.assertEqual(ops["s_barrier_bare"], 2 * -(-9 // pg) + 2)
                self.assertEqual(ops["sched_barrier"], ops["s_barrier_bare"])
                self.assertEqual(ops["mma"], plain["mma"])

    def test_rings_issue_one_weight_slot_per_step(self):
        """tap_ring refills one tap per ring step (nine weight transfers per
        chunk), row_ring one filter row (three); each chunk issues one halo,
        and the prologue the first halo plus stages - 1 weight steps."""
        base = dict(tiles_c=2, tiles_p=2, pingpong=True)
        for ring, steps in (("tap_ring", 9), ("row_ring", 3)):
            for stages in (2, 3):
                with self.subTest(ring=ring, stages=stages):
                    ops = self._ops(
                        DirectDgradWmmaSpec(
                            problem=_problem(), stages=stages, **{ring: True}, **base
                        )
                    )
                    loads = 1 + (stages - 1) + steps + 1
                    self.assertEqual(ops["tensor_load_to_lds"], loads)

    def test_per_axis_padding_names_and_plans(self):
        """A column padding of its own is spelled in the kernel name and
        moves the column plan only."""
        spec = DirectDgradWmmaSpec(problem=_problem(KH=1, KW=3, PAD=0, pad_w=1))
        self.assertIsNone(dgrad_wmma_reason(spec))
        self.assertIn("_f1x3_p0x1_", spec.kernel_name())
        self.assertEqual(spec.plan_w[2], 1)
        self.assertEqual(spec.plan_h[2], 0)
        same = DirectDgradWmmaSpec(problem=_problem(pad_w=1))
        self.assertIn("_p1_", same.kernel_name())

    def test_builder_refuses_with_the_reason(self):
        """The Python builder raises ValueError carrying the validator's
        reason for every new refusal."""
        for pkw, knobs, reason in (
            (
                dict(stride=2, pad_w=0),
                dict(tiles_c=2),
                "a column padding of its own serves stride 1 only",
            ),
            (
                {},
                dict(pingpong=True, prefetch=True),
                "pingpong replaces split_barrier and prefetch",
            ),
            ({}, dict(pp_group=2), "pp_group 2: >= 1, and > 1 only with pingpong"),
            (
                {},
                dict(tap_ring=True),
                "row_ring / tap_ring serve stride-1 unpacked pingpong tiles",
            ),
            (
                {},
                dict(pingpong=True, row_ring=True, tap_ring=True),
                "row_ring and tap_ring are exclusive",
            ),
        ):
            with self.subTest(problem=pkw, knobs=knobs):
                spec = DirectDgradWmmaSpec(problem=_problem(**pkw), **knobs)
                self.assertEqual(dgrad_wmma_reason(spec), reason)
                with self.assertRaisesRegex(ValueError, re.escape(reason)):
                    build_direct_dgrad_wmma(spec, "gfx1250")

    def test_engine_refuses_with_the_reason(self):
        """The C++ engine refuses the same specs with Python's reason."""
        eng = import_or_skip_until("rocke_library_engine", BUILD_VARIANT)
        from kernels.conv.engine_bridge import conv_direct_dgrad_wmma_spec_to_dict

        for pkw, knobs in (
            (dict(stride=2, pad_w=0), dict(tiles_c=2)),
            ({}, dict(pingpong=True, prefetch=True)),
            ({}, dict(pp_group=2)),
            ({}, dict(tap_ring=True)),
            ({}, dict(pingpong=True, row_ring=True, tap_ring=True)),
        ):
            with self.subTest(problem=pkw, knobs=knobs):
                spec = DirectDgradWmmaSpec(problem=_problem(**pkw), **knobs)
                reason = dgrad_wmma_reason(spec)
                sd = conv_direct_dgrad_wmma_spec_to_dict(spec)
                with self.assertRaisesRegex(RuntimeError, re.escape(reason)):
                    eng.conv_direct_dgrad_wmma_lower_llvm(sd, arch="gfx1250")


class TestDgradWmmaPack8Strided(unittest.TestCase):
    def test_cases_validate_and_lower(self):
        for pkw, knobs in _PACK8_STRIDED_CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                p = _problem(**pkw)
                self.assertEqual((p.cpg, p.kpg, p.stride), (4, 4, 2))
                spec = DirectDgradWmmaSpec(problem=p, **knobs)
                self.assertIsNone(dgrad_wmma_reason(spec, "gfx1250"))
                build_direct_dgrad_wmma(spec, "gfx1250")


@pytest.mark.gpu
@unittest.skipUnless(not _gpu_skip(), _gpu_skip() or "no GPU")
class TestDgradWmmaGpu(unittest.TestCase):
    def _check(self, pkw, knobs):
        from rocke import compile_kernel

        p = _problem(**pkw)
        spec = DirectDgradWmmaSpec(problem=p, **knobs)
        self.assertIsNone(dgrad_wmma_reason(spec, GPU_ARCH))
        art = compile_kernel(build_direct_dgrad_wmma(spec, GPU_ARCH), arch=GPU_ARCH)
        rng = np.random.default_rng(0)
        dy = storage.encode(rng.uniform(-1, 1, (p.N, p.Ho, p.Wo, p.total_k)), p.dtype)
        w = storage.encode(rng.uniform(-1, 1, (p.total_k, p.KH, p.KW, p.cpg)), p.dtype)
        dx = launch_direct_conv(
            art,
            p,
            (dy, w),
            # Poisoned: an element the kernel does not write fails.
            poisoned((p.N, p.H, p.W, p.total_c), p.dtype),
            grid=spec.grid(),
            block=(spec.threads_per_block, 1, 1),
            direction="dgrad",
        )
        ref = direct_dgrad_reference(
            p, storage.decode(dy, p.dtype), storage.decode(w, p.dtype)
        )
        err = peak_relative_error(storage.decode(dx, p.dtype), ref)
        self.assertLess(err, 1e-2, f"{pkw} {knobs}: rel err {err:.3g}")

    def test_cases(self):
        for pkw, knobs in _CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                self._check(pkw, knobs)

    def test_pingpong_cases(self):
        for pkw, knobs in _PP_CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                self._check(pkw, knobs)

    def test_pingpong_ring_cases(self):
        for pkw, knobs in _PP_RING_CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                self._check(pkw, knobs)

    def test_pack8_strided_cases(self):
        for pkw, knobs in _PACK8_STRIDED_CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                self._check(pkw, knobs)


if __name__ == "__main__":
    unittest.main()
