# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the Winograd F(m,3) forward pipeline (no GPU).

* The Toom-Cook matrices are exact and satisfy the correlation identity.
* The numpy model of the 4-stage algorithm (same tiling, dilation sub-grids,
  bin-major group layout and border handling as the kernels) matches a direct
  convolution in fp64, and stays within the 16-bit tolerance when U/V/M are
  rounded to fp16 the way the kernels store them.
* Spec validation and the grouped 1x1 GEMM problem.
* The three transform kernels build to IR.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_winograd_fwd import (
    WinogradFwdSpec,
    at_1d,
    bt_1d,
    build_winograd_filter_transform,
    build_winograd_input_transform,
    build_winograd_output_transform,
    filter_matrix_int,
    toom_cook,
    transform_stages,
    tuned_transform_options,
    winograd_fwd_reference_np,
    winograd_gemm_problem,
)


def _direct(x, w, p):
    y = np.zeros((p.N, p.Ho, p.Wo, p.K))
    xp = np.pad(x, ((0, 0), (p.pH, p.pH), (p.pW, p.pW), (0, 0)))
    for g in range(p.groups):
        cs = slice(g * p.cpg, (g + 1) * p.cpg)
        ks = slice(g * p.kpg, (g + 1) * p.kpg)
        for ky in range(3):
            for kx in range(3):
                h0, w0 = ky * p.dH, kx * p.dW
                xs = xp[:, h0 : h0 + p.Ho, w0 : w0 + p.Wo, cs]
                y[..., ks] += np.einsum("nhwc,kc->nhwk", xs, w[ks, ky, kx])
    return y


_PROBLEMS = [
    ConvProblem(N=2, Hi=8, Wi=8, C=8, K=8, Y=3, X=3, pH=1, pW=1),
    ConvProblem(N=1, Hi=7, Wi=9, C=5, K=6, Y=3, X=3),  # pad 0, odd extents
    ConvProblem(N=1, Hi=13, Wi=11, C=4, K=4, Y=3, X=3, pH=2, pW=2, dH=2, dW=2),
    ConvProblem(N=1, Hi=10, Wi=9, C=6, K=4, Y=3, X=3, pH=1, pW=3, dH=1, dW=3),
    ConvProblem(N=2, Hi=6, Wi=5, C=12, K=8, Y=3, X=3, pH=1, pW=1, groups=2),
    ConvProblem(N=1, Hi=9, Wi=9, C=8, K=12, Y=3, X=3, pH=4, pW=4, dH=4, dW=4, groups=4),
]


@pytest.mark.parametrize("m", [2, 4])
def test_toom_cook_identity_exact(m):
    at, g, bt = toom_cook(m)
    n = m + 2
    for i in range(m):
        for q in range(3):
            for p in range(n):
                s = sum(at[i][j] * g[j][q] * bt[j][p] for j in range(n))
                assert s == Fraction(1 if p == i + q else 0)
    # A^T and B^T are small integers; G's numerators are integers over den.
    assert all(v.denominator == 1 for r in at for v in r)
    assert all(v.denominator == 1 for r in bt for v in r)
    gi, den = filter_matrix_int(m)
    assert den == {2: 2, 4: 24}[m]
    assert all(Fraction(gi[j][k], den) == g[j][k] for j in range(n) for k in range(3))


def test_toom_cook_matches_numerics_tool_values():
    # Spot values of an independent Toom-Cook construction.
    at, g, bt = toom_cook(2)
    assert [list(map(int, r)) for r in bt] == [
        [-1, 0, 1, 0],
        [0, 1, 1, 0],
        [0, -1, 1, 0],
        [0, -1, 0, 1],
    ]
    at4, g4, bt4 = toom_cook(4)
    assert list(map(int, bt4[0])) == [4, 0, -5, 0, 1, 0]
    assert g4[3] == (Fraction(1, 24), Fraction(1, 12), Fraction(1, 6))


@pytest.mark.parametrize("m", [2, 4])
@pytest.mark.parametrize("pi", range(len(_PROBLEMS)))
def test_four_stage_model_matches_direct(m, pi):
    p = _PROBLEMS[pi]
    rng = np.random.default_rng(pi * 10 + m)
    x = rng.standard_normal((p.N, p.Hi, p.Wi, p.C))
    w = rng.standard_normal((p.K, 3, 3, p.cpg))
    y, U, V, M = winograd_fwd_reference_np(x, w, p, m)
    ref = _direct(x, w, p)
    assert y.shape == ref.shape
    assert np.abs(y - ref).max() <= 1e-9 * max(1.0, np.abs(ref).max())
    spec = WinogradFwdSpec(problem=p, m=m)
    n2 = spec.n2
    assert U.shape == (n2, p.K, p.cpg)
    assert V.shape == (spec.num_tiles, n2, p.C)
    assert M.shape == (spec.num_tiles, n2, p.K)


@pytest.mark.parametrize("m", [2, 4])
def test_four_stage_model_fp16_operands_within_tolerance(m):
    p = ConvProblem(N=1, Hi=12, Wi=12, C=64, K=16, Y=3, X=3, pH=1, pW=1)
    rng = np.random.default_rng(m)
    x = rng.uniform(-1, 1, (p.N, p.Hi, p.Wi, p.C)).astype(np.float16)
    w = rng.uniform(-1, 1, (p.K, 3, 3, p.cpg)).astype(np.float16)

    def rnd(_name, a):
        return np.asarray(np.float32(a).astype(np.float16), np.float64)

    y, *_ = winograd_fwd_reference_np(x, w, p, m, round_fn=rnd)
    ref = _direct(x.astype(np.float64), w.astype(np.float64), p)
    err = np.abs(y.astype(np.float16).astype(np.float64) - ref).max()
    assert err / max(np.abs(ref).max(), 1.0) < 5e-2


def test_gemm_problem_is_grouped_pointwise():
    p = ConvProblem(N=2, Hi=10, Wi=10, C=32, K=48, Y=3, X=3, pH=1, pW=1, groups=2)
    for m, n2 in ((2, 16), (4, 36)):
        gp = winograd_gemm_problem(p, m)
        spec = WinogradFwdSpec(problem=p, m=m)
        assert (gp.N, gp.Hi, gp.Wi, gp.Y, gp.X) == (spec.num_tiles, 1, 1, 1, 1)
        assert gp.groups == n2 * 2
        assert gp.cpg == 16 and gp.kpg == 24
        assert gp.is_pointwise
        assert gp.M == spec.num_tiles


def test_tile_counts_dilation():
    p = ConvProblem(N=3, Hi=17, Wi=10, C=8, K=8, Y=3, X=3, pH=2, pW=2, dH=2, dW=2)
    s = WinogradFwdSpec(problem=p, m=4)
    # Ho = 17, Wo = 10 -> sub-grids of ceil(17/2)=9 x 5 -> 3 x 2 tiles each.
    assert (s.sub_h, s.sub_w, s.tiles_h, s.tiles_w) == (9, 5, 3, 2)
    assert s.num_tiles == 3 * 4 * 3 * 2


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        (dict(m=4, io_dtype="bf16", operand_dtype="bf16"), "bf16 operands"),
        (dict(m=2, io_dtype="fp16", operand_dtype="bf16"), "bf16 operands"),
        (dict(m=6), "m must be"),
        (dict(vec=3), "vec must be"),
    ],
)
def test_spec_rejects(kwargs, msg):
    p = ConvProblem(N=1, Hi=8, Wi=8, C=8, K=8, Y=3, X=3, pH=1, pW=1)
    with pytest.raises(ValueError, match=msg):
        WinogradFwdSpec(problem=p, **kwargs)


def test_spec_rejects_non_3x3_and_strided():
    with pytest.raises(ValueError, match="3x3"):
        WinogradFwdSpec(problem=ConvProblem(N=1, Hi=8, Wi=8, C=8, K=8, Y=5, X=5))
    with pytest.raises(ValueError, match="stride 1"):
        WinogradFwdSpec(
            problem=ConvProblem(N=1, Hi=8, Wi=8, C=8, K=8, Y=3, X=3, sH=2, sW=2)
        )


def test_vector_widths_follow_channel_counts():
    p = ConvProblem(N=1, Hi=8, Wi=8, C=12, K=10, Y=3, X=3, pH=1, pW=1, groups=2)
    s = WinogradFwdSpec(problem=p, m=2, vec=8)
    assert (s.vec_c, s.vec_k, s.vec_cg) == (4, 2, 2)


@pytest.mark.parametrize(
    "m, io, op, ps",
    [(2, "fp16", "fp16", False), (4, "bf16", "fp16", True), (2, "bf16", "bf16", False)],
)
def test_transform_kernels_build(m, io, op, ps):
    p = ConvProblem(N=1, Hi=9, Wi=7, C=12, K=20, Y=3, X=3, pH=1, pW=1, groups=2)
    s = WinogradFwdSpec(problem=p, m=m, io_dtype=io, operand_dtype=op, prescale=ps)
    for build in (
        build_winograd_filter_transform,
        build_winograd_input_transform,
        build_winograd_output_transform,
    ):
        k = build(s, arch="gfx1250")
        assert k.name.startswith("conv_winograd_fwd")


class _NumOps:
    """The ops the 1-D transforms use, on exact rationals."""

    @staticmethod
    def add(x, y):
        return x + y

    @staticmethod
    def sub(x, y):
        return x - y

    @staticmethod
    def fma(c, x, y):
        return Fraction(c) * x + y


@pytest.mark.parametrize("m", [2, 4])
def test_shared_subexpression_transforms_match_matrices(m):
    """``bt_1d`` / ``at_1d`` (the kernels' CSE forms) equal ``B^T d`` / ``A^T d``."""
    at, _, bt = toom_cook(m)
    n = m + 2
    rng = np.random.default_rng(m)
    for _ in range(8):
        d = [Fraction(int(v)) for v in rng.integers(-1000, 1000, n)]
        assert bt_1d(_NumOps, m, d) == [
            sum(bt[i][u] * d[u] for u in range(n)) for i in range(n)
        ]
        assert at_1d(_NumOps, m, d) == [
            sum(at[i][u] * d[u] for u in range(n)) for i in range(m)
        ]


_VARIANTS = [
    dict(),
    dict(tiles_per_thread=3),
    dict(tiles_per_thread=2, wg_tile_rows=2, block_size=128),
    dict(input_block_tiles=(2, 3)),
    dict(input_block_tiles=(4, 8), tiles_per_thread=4, lds_lanes=16),
    dict(prescale=True),
    dict(prescale=True, prescale_mode="split"),
    dict(prescale=True, input_block_tiles=(2, 2), tiles_per_thread=2),
    dict(nt_stores=False),
]


@pytest.mark.parametrize("m", [2, 4])
@pytest.mark.parametrize("kw", _VARIANTS, ids=[str(i) for i in range(len(_VARIANTS))])
def test_transform_variants_build(m, kw):
    """Every K2 variant (runs, workgroup rows, LDS regions, fused / split
    prescale) and the stage table build to IR, with W / H / C tails."""
    p = ConvProblem(N=2, Hi=13, Wi=11, C=40, K=24, Y=3, X=3, pH=1, pW=2)
    io = "bf16" if kw.get("prescale") else "fp16"
    s = WinogradFwdSpec(problem=p, m=m, io_dtype=io, operand_dtype="fp16", vec=4, **kw)
    stages = transform_stages(s)
    names = [st.name for st in stages]
    split = s.prescale and s.prescale_split
    assert names == ["K1"] + (["K2a"] if split else []) + ["K2", "K4"]
    for st in stages:
        assert min(st.grid) > 0 and 32 <= st.block <= 1024
        assert st.build(s, arch="gfx1250").name.startswith("conv_winograd_fwd")
    # Split prescale keeps the per-pixel maxima after e[T].
    extra = p.N * p.Hi * p.Wi * 4 if split else 0
    assert s.buffer_bytes()["e"] == s.num_tiles * 4 + extra


def test_prescale_mode_selection():
    p = ConvProblem(N=1, Hi=8, Wi=8, C=16, K=8, Y=3, X=3, pH=1, pW=1)
    kw = dict(problem=p, m=4, io_dtype="bf16", prescale=True, vec=4)
    assert not WinogradFwdSpec(**kw).prescale_split
    assert WinogradFwdSpec(prescale_mode="split", **kw).prescale_split
    # The LDS path only has the split prescale.
    assert WinogradFwdSpec(input_block_tiles=(2, 2), **kw).prescale_split
    # The fused max covers all C in one workgroup (<= 1024 lanes).
    big = ConvProblem(N=1, Hi=8, Wi=8, C=8192, K=8, Y=3, X=3, pH=1, pW=1)
    assert WinogradFwdSpec(**dict(kw, problem=big)).prescale_split
    with pytest.raises(ValueError, match="prescale_mode"):
        WinogradFwdSpec(prescale_mode="x", **kw)
    with pytest.raises(ValueError, match="input_block_tiles"):
        WinogradFwdSpec(input_block_tiles=(2, 0), **kw)


@pytest.mark.parametrize("m", [2, 4])
@pytest.mark.parametrize("ps", [False, True])
def test_tuned_options_are_valid(m, ps):
    p = ConvProblem(N=1, Hi=64, Wi=64, C=512, K=512, Y=3, X=3, pH=1, pW=1)
    s = WinogradFwdSpec(
        problem=p, m=m, io_dtype="bf16", prescale=ps, **tuned_transform_options(m, ps)
    )
    assert [st.name for st in transform_stages(s)] == ["K1", "K2", "K4"]
