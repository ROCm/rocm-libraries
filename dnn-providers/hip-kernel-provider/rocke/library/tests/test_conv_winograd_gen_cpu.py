# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the generalized Winograd transforms (conv_winograd_gen)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common import conv_winograd_gen as WG
from kernels.common.conv_winograd_fwd import toom_cook


def _direct(x, w, p):
    n, h, wd, c = x.shape
    y = np.zeros((n, p.Ho, p.Wo, p.K))
    xp = np.zeros((n, h + 2 * p.pH, wd + 2 * p.pW, c))
    xp[:, p.pH : p.pH + h, p.pW : p.pW + wd] = x
    cg, kg = p.cpg, p.kpg
    for ky in range(p.Y):
        for kx in range(p.X):
            xs = xp[:, ky * p.dH : ky * p.dH + p.Ho, kx * p.dW : kx * p.dW + p.Wo]
            for gi in range(p.groups):
                y[..., gi * kg : (gi + 1) * kg] += np.einsum(
                    "nhwc,kc->nhwk",
                    xs[..., gi * cg : (gi + 1) * cg],
                    w[gi * kg : (gi + 1) * kg, ky, kx],
                )
    return y


@pytest.mark.parametrize("mr", sorted(WG.POINTS))
def test_toom_cook_identity(mr):
    """A^T [(G g) * (B^T d)] equals the correlation, exactly (rationals)."""
    m, r = mr
    at, g, bt = WG.toom_cook_gen(m, r)
    n = m + r - 1
    rng = np.random.default_rng(m * 31 + r)
    gg = [Fraction(int(v)) for v in rng.integers(-5, 6, r)]
    dd = [Fraction(int(v)) for v in rng.integers(-5, 6, n)]
    u = [sum(g[j][k] * gg[k] for k in range(r)) for j in range(n)]
    v = [sum(bt[j][p] * dd[p] for p in range(n)) for j in range(n)]
    y = [sum(at[i][j] * u[j] * v[j] for j in range(n)) for i in range(m)]
    ref = [sum(gg[q] * dd[i + q] for q in range(r)) for i in range(m)]
    assert y == ref


def test_matches_3x3_module():
    """The default F(2,3) / F(4,3) points reproduce the 3x3 pipeline's matrices."""
    for m in (2, 4):
        assert WG.toom_cook_gen(m, 3) == toom_cook(m)


def test_custom_points_and_rejects():
    pts = (0, 1, -1, Fraction(1, 3), Fraction(-1, 3))
    at, _, _ = WG.toom_cook_gen(4, 3, pts)
    assert at[1][3] == Fraction(1, 3)
    with pytest.raises(ValueError):
        WG.toom_cook_gen(4, 3, (0, 1, -1, 2))  # too few points
    with pytest.raises(ValueError):
        WG.toom_cook_gen(4, 3, (0, 1, -1, 2, 2))  # repeated
    with pytest.raises(ValueError):
        WG.toom_cook_gen(9, 9)  # no default set


_CASES = [
    # (problem, ax_y, ax_x)
    (ConvProblem(N=2, Hi=1, Wi=37, C=8, K=16, Y=1, X=11, pW=5), (1, 1, 1), (4, 11, 3)),
    (
        ConvProblem(N=1, Hi=1, Wi=50, C=8, K=8, Y=1, X=11, pW=15, dW=3),
        (1, 1, 1),
        (6, 11, 2),
    ),
    (
        ConvProblem(N=1, Hi=1, Wi=60, C=8, K=8, Y=1, X=11, pW=25, dW=5),
        (1, 1, 1),
        (5, 11, 3),
    ),
    (
        ConvProblem(N=1, Hi=1, Wi=45, C=8, K=8, Y=1, X=7, pW=15, dW=5),
        (1, 1, 1),
        (4, 7, 1),
    ),
    (
        ConvProblem(N=1, Hi=1, Wi=33, C=8, K=8, Y=1, X=7, pW=9, dW=3),
        (1, 1, 1),
        (4, 7, 2),
    ),
    (ConvProblem(N=1, Hi=1, Wi=29, C=8, K=8, Y=1, X=3, pW=1), (1, 1, 1), (6, 3, 1)),
    (
        ConvProblem(N=1, Hi=1, Wi=31, C=8, K=8, Y=1, X=3, pW=5, dW=5),
        (1, 1, 1),
        (8, 3, 1),
    ),
    (
        ConvProblem(N=1, Hi=9, Wi=11, C=8, K=8, Y=5, X=5, pH=2, pW=2),
        (3, 5, 1),
        (3, 5, 1),
    ),
    (
        ConvProblem(N=1, Hi=8, Wi=7, C=8, K=8, Y=5, X=5, pH=2, pW=2),
        (2, 5, 1),
        (2, 5, 1),
    ),
    (ConvProblem(N=1, Hi=13, Wi=5, C=8, K=8, Y=7, X=1, pH=3), (4, 7, 1), (1, 1, 1)),
    (
        ConvProblem(N=1, Hi=1, Wi=30, C=8, K=8, Y=1, X=7, pW=3, groups=2),
        (1, 1, 1),
        (4, 7, 2),
    ),
]


@pytest.mark.parametrize("case", range(len(_CASES)))
def test_reference_matches_direct(case):
    p, ay, ax = _CASES[case]
    spec = WG.WinogradGenSpec(problem=p, ax_y=WG.WinoAxis(*ay), ax_x=WG.WinoAxis(*ax))
    rng = np.random.default_rng(case)
    x = rng.standard_normal((p.N, p.Hi, p.Wi, p.C))
    w = rng.standard_normal((p.K, p.Y, p.X, p.cpg))
    y, _, _, _ = WG.winograd_gen_reference_np(x, w, spec)
    ref = _direct(x, w, p)
    assert np.abs(y - ref).max() <= 1e-9 * max(1.0, np.abs(ref).max())


def test_fp16_rounded_workspaces_within_tolerance():
    """fp16 U/V/M (the kernels' operand dtype) stays well inside 5e-2."""
    p = ConvProblem(N=1, Hi=1, Wi=200, C=64, K=16, Y=1, X=11, pW=5)
    spec = WG.WinogradGenSpec(
        problem=p, ax_y=WG.WinoAxis(1, 1), ax_x=WG.WinoAxis(6, 11, 2)
    )
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 1, 200, 64)).astype(np.float16).astype(np.float64)
    w = (rng.standard_normal((16, 1, 11, 64)) / np.sqrt(64 * 11)).astype(np.float16)
    w = w.astype(np.float64)

    def rnd(_name, a):
        return a.astype(np.float16).astype(np.float64)

    y, *_ = WG.winograd_gen_reference_np(x, w, spec, round_fn=rnd)
    ref = _direct(x, w, p)
    err = np.abs(y - ref).max() / max(np.abs(ref).max(), 1.0)
    assert err < 2e-2


def test_spec_geometry_and_rejects():
    p = ConvProblem(N=2, Hi=1, Wi=100, C=32, K=32, Y=1, X=11, pW=5)
    spec = WG.WinogradGenSpec(
        problem=p, ax_y=WG.WinoAxis(1, 1), ax_x=WG.WinoAxis(4, 11, 3)
    )
    assert spec.bins == 7 and spec.S == 3
    assert spec.tiles_w == 25 and spec.tiles_w_in == 27
    assert spec.num_tiles == 2 * 27
    gp = spec.gemm_problem()
    assert (gp.T, gp.G, gp.Cg, gp.Kg) == (54, 7, 32, 32)
    assert spec.buffer_bytes()["V"] == (54 + 2) * 7 * 32 * 2
    with pytest.raises(ValueError):  # segments do not cover the filter
        WG.WinogradGenSpec(
            problem=p, ax_y=WG.WinoAxis(1, 1), ax_x=WG.WinoAxis(4, 11, 2)
        )
    with pytest.raises(ValueError):  # an all-zero segment
        WG.WinogradGenSpec(
            problem=p, ax_y=WG.WinoAxis(1, 1), ax_x=WG.WinoAxis(4, 11, 4)
        )
    with pytest.raises(ValueError):  # taps mismatch
        WG.WinogradGenSpec(problem=p, ax_y=WG.WinoAxis(1, 1), ax_x=WG.WinoAxis(4, 7))
    ps = ConvProblem(N=1, Hi=1, Wi=100, C=32, K=32, Y=1, X=11, sW=2)
    with pytest.raises(ValueError):  # stride 2
        WG.WinogradGenSpec(
            problem=ps, ax_y=WG.WinoAxis(1, 1), ax_x=WG.WinoAxis(4, 11, 3)
        )


@pytest.mark.parametrize("case", [0, 1, 5, 7, 9, 10])
def test_kernels_build(case):
    from rocke import compile_kernel

    p, ay, ax = _CASES[case]
    spec = WG.WinogradGenSpec(problem=p, ax_y=WG.WinoAxis(*ay), ax_x=WG.WinoAxis(*ax))
    for st in WG.transform_stages(spec):
        assert min(st.grid) > 0
        compile_kernel(st.build(spec, arch="gfx1250"), arch="gfx1250")


def test_k3_taps_spec():
    from kernels.common.conv_winograd_gemm_wmma import (
        WinogradGemmProblem,
        WinogradGemmWmmaSpec,
        build_winograd_gemm_wmma,
        winograd_gemm_reason,
    )

    s1 = WinogradGemmWmmaSpec()
    s3 = WinogradGemmWmmaSpec(taps=3)
    prob = WinogradGemmProblem(T=1000, G=7, Cg=128, Kg=128)
    assert s3.chunks(prob) == 3 * s1.chunks(prob)
    assert s1.kernel_name() != s3.kernel_name()
    assert winograd_gemm_reason(s3) is None
    assert winograd_gemm_reason(WinogradGemmWmmaSpec(taps=0)) is not None
    build_winograd_gemm_wmma(s3)


# --- fused 1-D kernel: spec validation and IR builds ------------------------


def _fspec(**kw):
    from kernels.common.conv_winograd_fused_1d_wmma import FusedWino1DSpec

    p = kw.pop(
        "problem", ConvProblem(N=2, Hi=1, Wi=203, C=32, K=32, Y=1, X=11, pW=15, dW=3)
    )
    ax = kw.pop("ax", WG.WinoAxis(4, 11, 3))
    return FusedWino1DSpec(problem=p, ax=ax, **kw)


def test_fused_spec_geometry_and_rejects():
    s = _fspec(q=2, fk=2, grid_blocks=4)
    assert (s.n, s.S, s.tb, s.kb, s.vt) == (7, 3, 32, 32, 34)
    assert s.raw_rows == 34 * 4 + 3
    assert s.tile_blocks == 2 * 3 * 1 and s.grid() == 4
    assert s.xf_tiles == 2 and s.xf_win == 16
    with pytest.raises(ValueError):  # 2-D filter
        _fspec(
            problem=ConvProblem(N=1, Hi=5, Wi=5, C=32, K=32, Y=3, X=3, pH=1, pW=1),
            ax=WG.WinoAxis(4, 3),
        )
    with pytest.raises(ValueError):  # C not a multiple of 32
        _fspec(problem=ConvProblem(N=1, Hi=1, Wi=50, C=48, K=32, Y=1, X=11, pW=5))
    with pytest.raises(ValueError):  # B^T of F(6,6) is not exact in fp16
        _fspec(ax=WG.WinoAxis(6, 11, 2), xf="wmma")
    with pytest.raises(ValueError):  # merge needs prefetch
        _fspec(merge=True)
    with pytest.raises(ValueError):  # LDS
        _fspec(
            problem=ConvProblem(N=1, Hi=1, Wi=5000, C=256, K=64, Y=1, X=11, pW=5), q=8
        )


@pytest.mark.parametrize(
    "kw",
    [
        dict(q=2, fk=2, grid_blocks=4),
        dict(q=2, fk=2, xf="wmma", prefetch=True, m_f16=True, merge=True, xf_batch=8),
        dict(q=1, fk=1, waves=4, prefetch=True, m_f16=True, merge=True),
        dict(
            q=2,
            fk=2,
            waves=4,
            waves_per_eu=2,
            xf="wmma",
            prefetch=True,
            m_f16=True,
            merge=True,
        ),
    ],
)
def test_fused_builds(kw):
    from rocke import compile_kernel
    from kernels.common.conv_winograd_fused_1d_wmma import build_fused_wino_1d

    s = _fspec(**kw)
    compile_kernel(build_fused_wino_1d(s), arch="gfx1250")


def _op_names(region):
    out = set()
    for op in region.ops:
        out.add(op.name)
        for r in op.regions:
            out |= _op_names(r)
    return out


def test_fused_f47_builds():
    """F(4,7): n = 10 bins with the VALU transform (B^T is not fp16-exact);
    prefetch moves the next raw block with async global->LDS copies."""
    from rocke import compile_kernel
    from kernels.common.conv_winograd_fused_1d_wmma import build_fused_wino_1d

    p = ConvProblem(N=2, Hi=1, Wi=517, C=64, K=64, Y=1, X=7, pW=15, dW=5)
    s = _fspec(
        problem=p,
        ax=WG.WinoAxis(4, 7, 1),
        q=2,
        fk=4,
        prefetch=True,
        m_f16=True,
        merge=True,
        waves=4,
    )
    k = build_fused_wino_1d(s)
    assert any("async" in nm for nm in _op_names(k.body))
    compile_kernel(k, arch="gfx1250")
