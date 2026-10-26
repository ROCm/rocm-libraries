# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the wave32 WMMA direct dgrad (``conv_direct_dgrad_wmma``).

CPU: the spec validator, the AOT grid (gfx1250 only), the shape filter of
``direct_spec_for_problem`` and the shipped path: ``direct_plans`` over a
gfx1250 cache of placeholder binaries (capability match, re-validation, launch
grid, block and kernargs). GPU (gfx1250): every spec branch -- edges and
ragged tiles, groups, PAD 0 / 5x5, several images per block, a 3-deep ring,
two WMMA k-steps per chunk, fragment prefetch, the upper VGPR banks -- checked
numerically against torch's dgrad.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library <torch-python> -m pytest \\
        rocke/library/tests/test_direct_dgrad_wmma.py -v
"""

from __future__ import annotations

import ctypes
import json
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

from kernels.common.conv_direct_dgrad_wmma import (
    DirectDgradWmmaSpec,
    build_direct_dgrad_wmma,
    dgrad_wmma_reason,
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
    KW=None,
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
        eng = pytest.importorskip("rocke_engine")
        from kernels.conv.direct_aot_jobs import DIRECT_CAPABILITIES, _knob_grid
        from rocke.core.backend import conv_direct_dgrad_wmma_spec_to_dict

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
                PAD=caps.PAD,
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
        ):
            specs.append(DirectDgradWmmaSpec(problem=_problem(**pkw), **knobs))
        n_rejected = 0
        for arch in ("gfx1250", "gfx950"):
            for spec in specs:
                ok, _, name, grid, lds = eng.conv_direct_dgrad_wmma_describe(
                    conv_direct_dgrad_wmma_spec_to_dict(spec), arch
                )
                why = dgrad_wmma_reason(spec, arch)
                self.assertEqual(ok, why is None, (arch, spec, why))
                self.assertEqual(name, spec.kernel_name())
                self.assertEqual(lds, spec.lds_bytes())
                if why is None:
                    self.assertEqual(tuple(grid), spec.grid())
                elif arch == "gfx1250":
                    n_rejected += 1
        # Every AOT knob builds on gfx1250; every extra spec is a reject.
        self.assertEqual(n_rejected, len(specs) - n_grid)

    def test_binding_rejects_bad_input(self):
        """A rejected spec returns the rejection (no grid is computed from its
        zero tile extents); a missing, mistyped or partial problem raises."""
        eng = pytest.importorskip("rocke_engine")
        from rocke.core.backend import conv_direct_dgrad_wmma_spec_to_dict

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
        eng = pytest.importorskip("rocke_engine")
        from rocke.core.backend import conv_direct_dgrad_wmma_spec_to_dict

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
        )

        direction, caps_list = DIRECT_CAPABILITIES["direct_dgrad_wmma"]
        self.assertEqual(direction, "dgrad")
        for arch, wave in (("gfx1250", 32), ("gfx950", 64)):
            for caps in caps_list:
                # Every knob of the cell's table builds on gfx1250 (none on
                # a wave64 arch).
                want = len(list(_knob_grid("direct_dgrad_wmma", arch, "bf16", caps)))
                if arch == "gfx950":
                    self.assertEqual(want, 0)
                _, jobs = _caps_jobs(
                    arch, wave, "direct_dgrad_wmma", "dgrad", caps, "bf16"
                )
                self.assertEqual(len(jobs), want, (arch, caps))

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
        (dict(C=64, K=64, PAD=2, dil_w=2), 1),
        (dict(C=64, K=64, PAD=1, dil_h=3), 1),
        (dict(C=64, K=64, PAD=2, dil_h=2, dil_w=2), 1),
        (dict(C=128, K=128, groups=32), 8),
        (dict(C=256, K=256, groups=32), 4),
        (dict(C=256, K=256, groups=16), 2),
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
        return f"needs gfx1250 (got {GPU_ARCH!r})"
    if not _HAS_TORCH:
        return "torch not importable"
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


@unittest.skipUnless(not _gpu_skip(), _gpu_skip() or "no GPU")
class TestDgradWmmaGpu(unittest.TestCase):
    def _check(self, pkw, knobs):
        from kernels.common.conv_abi import conv_direct_args_signature
        from kernels.common.conv_args import ConvArgs
        from rocke import compile_kernel
        from rocke.runtime.hip_module import Runtime
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        p = _problem(**pkw)
        spec = DirectDgradWmmaSpec(problem=p, **knobs)
        self.assertIsNone(dgrad_wmma_reason(spec, GPU_ARCH))
        art = compile_kernel(build_direct_dgrad_wmma(spec, GPU_ARCH), arch=GPU_ARCH)
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
            dilation=(p.dil_h, p.dil_w),
            groups=p.groups,
        ).permute(0, 2, 3, 1)
        err = float((dX.float() - ref).abs().max() / ref.abs().max())
        self.assertLess(err, 1e-2, f"{pkw} {knobs}: rel err {err:.3g}")

    def test_cases(self):
        for pkw, knobs in _CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                self._check(pkw, knobs)


if __name__ == "__main__":
    unittest.main()
