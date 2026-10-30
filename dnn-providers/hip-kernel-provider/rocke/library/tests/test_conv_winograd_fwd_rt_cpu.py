# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the shape-generic Winograd transform kernels
(:mod:`kernels.common.conv_winograd_fwd_rt`).

* ``magic_div``: the multiply-high division the kernels use is exact for the
  31-bit indices they divide;
* spec validation refuses what the kernels cannot build;
* the AOT grid: unique kernel names, every spec builds, and the default pick of
  any problem is a member of it;
* ``stage_launch`` spells every scalar kernarg of the stage's signature, and
  the kernel's parameters are that signature;
* refusals of problems a binary cannot run.
"""

from __future__ import annotations

import random

import pytest

from kernels.common import conv_winograd_fwd_rt as R
from kernels.common._conv_implicit_gemm_common import ConvProblem


def _q(n: int, d: int) -> int:
    magic, shift = R.magic_div(d)
    if magic == 0:
        return n
    return ((n * magic) >> 32) >> shift


@pytest.mark.parametrize("d", [1, 2, 3, 5, 7, 9, 31, 32, 33, 96, 255, 641, 1000, 65537])
def test_magic_div_exact(d):
    rng = random.Random(d)
    ns = [0, 1, d - 1, d, d + 1, 2**31 - 1, 2**31 - d, 2**30]
    ns += [rng.randrange(0, 2**31) for _ in range(2000)]
    for n in ns:
        assert _q(n, d) == n // d, (n, d)


def test_magic_div_every_small_divisor():
    rng = random.Random(0)
    for d in range(1, 4097):
        magic, _ = R.magic_div(d)
        assert 0 <= magic < 2**32
        for n in (rng.randrange(0, 2**31) for _ in range(16)):
            assert _q(n, d) == n // d


def test_magic_div_rejects_zero():
    with pytest.raises(ValueError):
        R.magic_div(0)


@pytest.mark.parametrize(
    "make",
    [
        lambda: R.RtFilterSpec(mh=3),
        lambda: R.RtFilterSpec(io_dtype="bf16", operand_dtype="bf16"),
        lambda: R.RtFilterSpec(filter_rows=4),
        lambda: R.RtOutputSpec(vec=3),
        lambda: R.RtOutputSpec(block_size=96),
        lambda: R.RtInputSpec(fused=True),
        lambda: R.RtInputSpec(mh=2, mw=2, lds=(2, 3), tiles_per_thread=4),
        lambda: R.RtInputSpec(mh=2, mw=2, lds=(8, 8), lds_lanes=32),
        lambda: R.RtPrescaleSpec(vec=16),
    ],
)
def test_invalid_specs_are_refused(make):
    with pytest.raises(ValueError):
        make()


def test_aot_grid_names_are_unique():
    specs = R.aot_stage_specs()
    names = [s.kernel_name() for s in specs]
    assert len(names) == len(set(names))
    stages = {s.stage for s in specs}
    assert stages == {"K1", "K2a", "K2", "K4"}


def test_aot_grid_builds():
    """Every binary of the grid builds, with its signature as parameters."""
    for spec in R.aot_stage_specs():
        k = R.build_stage(spec, arch="gfx1250")
        assert [p.name for p in k.params] == [
            a["name"] for a in R.stage_signature(spec)
        ]
        assert k.name == spec.kernel_name()


_PROBLEMS = [
    ConvProblem(N=2, Hi=20, Wi=24, C=64, K=96, Y=3, X=3, pH=1, pW=1),
    ConvProblem(N=1, Hi=13, Wi=11, C=36, K=20, Y=3, X=3, pH=0, pW=2),
    ConvProblem(N=1, Hi=40, Wi=40, C=128, K=64, Y=3, X=3, pH=2, pW=2, dH=2, dW=2),
    ConvProblem(N=2, Hi=16, Wi=16, C=128, K=128, Y=3, X=3, pH=1, pW=1, groups=4),
    ConvProblem(N=1, Hi=8, Wi=8, C=16384, K=64, Y=3, X=3, pH=1, pW=1),
    ConvProblem(N=1, Hi=5, Wi=7, C=3, K=5, Y=3, X=3, pH=1, pW=1),
]


@pytest.mark.parametrize("problem", _PROBLEMS)
@pytest.mark.parametrize("io", ["fp16", "bf16"])
def test_default_pick_is_in_the_grid(problem, io):
    grid = {s.kernel_name() for s in R.aot_stage_specs()}
    for tile in R.AOT_TILES:
        specs = R.default_stage_specs(problem, tile, io)
        _, prescale = R.pipeline_dtypes(tile, io)
        assert ("K2a" in specs) == (prescale and not specs["K2"].fused)
        geo = R.RtGeometry(problem, *tile)
        for spec in specs.values():
            assert spec.kernel_name() in grid
            assert spec.reason(geo) is None


def test_fused_prescale_falls_back_to_split_for_wide_c():
    p = _PROBLEMS[4]  # C / vec = 4096 lanes: more than one workgroup
    specs = R.default_stage_specs(p, (4, 4), "bf16")
    assert "K2a" in specs and not specs["K2"].fused
    narrow = R.default_stage_specs(_PROBLEMS[0], (4, 4), "bf16")
    assert "K2a" not in narrow and narrow["K2"].fused


@pytest.mark.parametrize("problem", _PROBLEMS[:4])
def test_stage_launch_spells_every_scalar(problem):
    for tile in ((4, 4), (2, 2), (6, 4)):
        geo = R.RtGeometry(problem, *tile)
        for io in ("fp16", "bf16"):
            for spec in R.default_stage_specs(problem, tile, io).values():
                grid, block, vals = R.stage_launch(spec, geo)
                assert min(grid) >= 1 and 32 <= block <= 1024
                sig = [a["name"] for a in R.stage_signature(spec)]
                pointers = {"w", "u", "x", "v", "e", "mm", "y"}
                scalars = [
                    n for n in sig if n not in pointers and not n.endswith("_bytes")
                ]
                assert sorted(scalars) == sorted(vals)
                for name, val in vals.items():
                    assert -(2**31) <= val < 2**31, name
                    if name.endswith("_mg"):
                        d = vals[name[: -len("_mg")]]
                        magic, shift = R.magic_div(d)
                        assert val % 2**32 == magic and vals[name[:-3] + "_sh"] == shift


def test_stage_launch_refuses_unfit_problems():
    p = ConvProblem(N=1, Hi=8, Wi=8, C=36, K=20, Y=3, X=3, pH=1, pW=1)
    geo = R.RtGeometry(p, 4, 4)
    with pytest.raises(ValueError, match="multiple of vec"):
        R.stage_launch(R.RtInputSpec(vec=8), geo)
    with pytest.raises(ValueError, match="multiple of vec"):
        R.stage_launch(R.RtOutputSpec(vec=8), geo)
    with pytest.raises(ValueError, match="tile mismatch"):
        R.stage_launch(R.RtFilterSpec(mh=2, mw=2, vec=4), geo)
    wide = ConvProblem(N=1, Hi=8, Wi=8, C=4096, K=64, Y=3, X=3, pH=1, pW=1)
    with pytest.raises(ValueError, match="fused prescale"):
        R.stage_launch(
            R.RtInputSpec(io_dtype="bf16", vec=1, prescale=True, fused=True),
            R.RtGeometry(wide, 4, 4),
        )


def test_geometry_refuses_non_3x3_and_huge_buffers():
    p5 = ConvProblem(N=1, Hi=8, Wi=8, C=8, K=8, Y=5, X=5, pH=2, pW=2)
    assert R.RtGeometry(p5, 4, 4).check(False)
    big = ConvProblem(N=64, Hi=512, Wi=512, C=256, K=256, Y=3, X=3, pH=1, pW=1)
    assert "32-bit" in R.RtGeometry(big, 4, 4).check(False)
