# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""f64 / cf64 GETT on the device against exact references.

Every output checked is compared with the correctly rounded exact result
(``gett_f64_host.reference``) under the f64 accumulation bound; the integer
case must match exactly. NaN / Inf propagation, the BLAS alpha / beta rules
and conjugation are checked bitwise on whole outputs. Runs on every target
whose catalog has the f64 atoms, through both engines when the native one is
built.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from rocke.instances.common.gett_f64 import (
    F64_GETT_ARCHS,
    GettF64Problem,
    arch_spec,
    gemm_problem,
    mixed_offset,
)
from rocke.instances.common.gett_f64_plan import packed_strides, plan_contraction
from rocke.runtime.hip_module import get_device_arch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gett_f64_host as H  # noqa: E402


def _arch():
    arch = (get_device_arch() or "").split(":")[0]
    if arch not in F64_GETT_ARCHS:
        pytest.skip(f"requires a target with the f64 atoms {F64_GETT_ARCHS}")
    return arch


def _engines():
    if importlib.util.find_spec("rocke_engine") is None:
        return ("python",)
    return ("python", "cpp")


def _samples(p: GettF64Problem, rng, n=1500):
    total = p.L * p.M * p.N
    if total <= 30000:
        return [(l, m, k) for l in range(p.L) for m in range(p.M) for k in range(p.N)]
    return [
        (int(rng.integers(p.L)), int(rng.integers(p.M)), int(rng.integers(p.N)))
        for _ in range(n)
    ]


def _run_check(
    spec, arch, *, alpha=1.25, beta=-0.5, seed=0, integer=False, c_nan=False
):
    p = spec.problem
    rng = np.random.default_rng(seed)
    A, B, C, D = H.random_buffers(p, rng, integer=integer, bilinear=spec.bilinear)
    if p.complex_:
        alpha, beta = complex(alpha, -0.375), complex(beta, 0.625)
    if c_nan and C is not None:
        C[:] = np.nan
        beta = 0j if p.complex_ else 0.0
    for engine in _engines():
        out = H.run(spec, arch, A, B, C, D.copy(), alpha, beta, engine=engine)
        if integer:
            Am, Bm = H.gather(p, "A", A), H.gather(p, "B", B)
            ref = alpha * np.einsum("lmk,lnk->lmn", Am, Bm)
            if spec.bilinear:
                ref = ref + beta * H.gather(p, "C", C)
            np.testing.assert_array_equal(H.gather(p, "D", out), ref)
            continue
        worst, bad = H.check(
            p,
            A,
            B,
            None if c_nan else C,
            out,
            D,
            alpha,
            beta if spec.bilinear else 0.0,
            _samples(p, rng),
        )
        assert bad is None, f"{engine}: {bad}"
        assert worst <= 1.0


# (M, N, K, A strides (m, k), B strides (n, k), D strides (m, n))
LAYOUTS = {
    "rcr": lambda M, N, K: ((K, 1), (K, 1), (N, 1)),
    "rrr": lambda M, N, K: ((K, 1), (1, N), (N, 1)),
    "ccc": lambda M, N, K: ((1, M), (K, 1), (1, M)),
    "crc": lambda M, N, K: ((1, M), (1, N), (1, M)),
}


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_gemm_layouts(layout):
    arch = _arch()
    M, N, K = 192, 160, 100  # K tail of the 16-row K tile
    sa, sb, sd = LAYOUTS[layout](M, N, K)
    p = gemm_problem(M, N, K, a_strides=sa, b_strides=sb, d_strides=sd)
    _run_check(arch_spec(p, arch), arch, seed=sorted(LAYOUTS).index(layout))


def test_tails_and_tiles():
    arch = _arch()
    p = gemm_problem(
        200, 136, 68, a_strides=(68, 1), b_strides=(68, 1), d_strides=(136, 1)
    )
    for spec in (
        arch_spec(p, arch),
        arch_spec(p, arch, tile_m=64, tile_n=64, tile_k=8),
        arch_spec(p, arch, tile_m=128, tile_n=64, warps_m=2, warps_n=1, group_m=1),
        arch_spec(p, arch, prefetch=False, xcds=8),
        arch_spec(p, arch, lds_swizzle=False, sched_fence=False),
        arch_spec(p, arch, tile_m=48, tile_n=64, warps_m=1, warps_n=2),
        arch_spec(
            p, arch, tile_m=96, tile_n=64, warps_m=2, warps_n=2, waves_per_eu=None
        ),
    ):
        _run_check(spec, arch, seed=3)


@pytest.mark.parametrize(
    "kw",
    [
        dict(lds_stages=1, tile_k=16),
        dict(lds_stages=1, tile_k=32, lds_swizzle_mode=1),
        dict(lds_swizzle_mode=2, tile_k=8),
        dict(prefetch_depth=2, lds_stages=1, tile_k=16),
        dict(prefetch_depth=2, tile_k=8),
        dict(unroll2=True, tile_k=16),
        dict(unroll2=True, tile_k=8, sched_fence=False),
        dict(k_stagger=1, lds_stages=1, tile_k=16),
        dict(k_stagger=4, unroll2=True, tile_k=8),
        dict(k_stagger=2, prefetch_depth=2, lds_stages=1, tile_k=8),
    ],
)
def test_schedule_knobs(kw):
    arch = _arch()
    # both operands K-contiguous (along-K loads) and an odd K-tile count
    p = gemm_problem(
        200, 136, 104, a_strides=(104, 1), b_strides=(104, 1), d_strides=(1, 200)
    )
    _run_check(arch_spec(p, arch, **kw), arch, seed=11)
    q = gemm_problem(
        192, 160, 72, a_strides=(1, 192), b_strides=(1, 160), d_strides=(160, 1)
    )
    _run_check(arch_spec(q, arch, **kw), arch, seed=12)


@pytest.mark.parametrize(
    "kw",
    [
        dict(complex_accs=4),
        dict(complex_accs=4, unroll2=True),
        dict(prefetch_depth=2, lds_stages=1, k_stagger=1),
    ],
)
@pytest.mark.parametrize(
    "conj", [(False, False), (True, False), (False, True), (True, True)]
)
def test_complex_schedule_knobs(kw, conj):
    arch = _arch()
    p = gemm_problem(
        96,
        64,
        40,
        a_strides=(40, 1),
        b_strides=(1, 64),
        d_strides=(1, 96),
        complex_=True,
        conj_a=conj[0],
        conj_b=conj[1],
    )
    _run_check(arch_spec(p, arch, tile_m=64, tile_n=64, tile_k=8, **kw), arch, seed=13)


def test_tuned_configurations():
    from rocke.instances.common.gett_f64 import TUNED_CONFIGS

    arch = _arch()
    for (a, kind, cls), cfg in TUNED_CONFIGS.items():
        if a != arch:
            continue
        p = gemm_problem(
            264,
            200,
            88,
            a_strides=(88, 1),
            b_strides=(1, 200),
            d_strides=(1, 264),
            complex_=kind == "complex",
        )
        _run_check(arch_spec(p, arch, **cfg), arch, seed=17)


def test_scale_and_beta_zero_ignores_c():
    arch = _arch()
    p = gemm_problem(
        96, 80, 40, a_strides=(40, 1), b_strides=(40, 1), d_strides=(80, 1)
    )
    _run_check(arch_spec(p, arch, bilinear=False), arch, seed=5)
    _run_check(arch_spec(p, arch), arch, seed=6, c_nan=True)


def test_c_has_its_own_strides_and_batch():
    arch = _arch()
    M, N, K, L = 64, 48, 32, 3
    p = gemm_problem(
        M,
        N,
        K,
        a_strides=(K, 1),
        b_strides=(1, N),
        d_strides=(N, 1),
        c_strides=(1, M),
        batch=L,
        batch_strides=(M * K, N * K, M * N),
        c_batch_stride=M * N,
    )
    _run_check(arch_spec(p, arch, tile_m=64, tile_n=64), arch, seed=7)


def test_integer_inputs_are_exact():
    arch = _arch()
    p = gemm_problem(
        160, 144, 512, a_strides=(512, 1), b_strides=(512, 1), d_strides=(144, 1)
    )
    _run_check(arch_spec(p, arch), arch, alpha=1.0, beta=1.0, integer=True, seed=9)


@pytest.mark.parametrize(
    "conj", [(False, False), (True, False), (False, True), (True, True)]
)
def test_complex_fused_4m(conj):
    arch = _arch()
    M, N, K = 96, 64, 36
    p = gemm_problem(
        M,
        N,
        K,
        a_strides=(K, 1),
        b_strides=(1, N),
        d_strides=(N, 1),
        complex_=True,
        conj_a=conj[0],
        conj_b=conj[1],
    )
    _run_check(arch_spec(p, arch, tile_m=64, tile_n=64), arch, seed=11)


def _contraction(modes, ext, *, swap_d=False):
    ma, mb, md = modes
    if swap_d:
        md = list(reversed(md))
    sa, sb, sd = (packed_strides(m, ext) for m in (ma, mb, md))
    return plan_contraction(ma, sa, mb, sb, md, sd, ext)


@pytest.mark.parametrize("swap_d", [False, True])
def test_p1_style_contraction(swap_d):
    """Interleaved K, permuted output (hipTensor P1), column-major packed."""
    arch = _arch()
    ext = {"M0": 12, "K0": 6, "M1": 10, "K1": 8, "N0": 14, "N1": 6}
    modes = (
        ["M0", "K0", "M1", "K1"],
        ["N0", "K0", "N1", "K1"],
        ["M0", "N0", "M1", "N1"],
    )
    plan = _contraction(modes, ext, swap_d=swap_d)
    assert len(plan.launches) == 1
    launch = plan.launches[0]
    assert (launch.problem.M, launch.problem.N, launch.problem.K) in (
        (120, 84, 48),
        (84, 120, 48),
    )
    _run_check(
        arch_spec(launch.problem, arch, tile_m=64, tile_n=64, tile_k=8), arch, seed=13
    )


def test_split_launches_cover_the_output_exactly_once():
    """A contraction planned into several launches (forced with a small
    address budget) writes every output once, through per-launch offsets."""
    arch = _arch()
    ext = {"m0": 12, "k0": 6, "m1": 10, "n0": 14, "k1": 8, "n1": 6}
    ma, mb, md = (
        ["m0", "k0", "m1", "k1"],
        ["n0", "k0", "n1", "k1"],
        ["m0", "n0", "m1", "n1"],
    )
    sa, sb, sd = (packed_strides(x, ext) for x in (ma, mb, md))
    plan = plan_contraction(ma, sa, mb, sb, md, sd, ext, max_span=8 * 4000)
    assert len(plan.launches) > 1
    rng = np.random.default_rng(21)
    n = lambda ms: int(np.prod([ext[m] for m in ms]))  # noqa: E731
    A, B, C = (
        rng.uniform(-1, 1, n(ma)),
        rng.uniform(-1, 1, n(mb)),
        rng.uniform(-1, 1, n(md)),
    )
    D = np.full(n(md), np.nan)
    alpha, beta = 1.25, -0.5
    out = H.run_plan(plan, arch, A, B, C, D, alpha, beta, tile_m=64, tile_n=64)

    # reference with numpy einsum over the mode letters (column-major packed views)
    def view(flat, modes):
        return flat.reshape([ext[m] for m in reversed(modes)])

    letters = {m: chr(ord("a") + i) for i, m in enumerate(sorted(ext))}
    eq = (
        "".join(letters[m] for m in reversed(ma))
        + ","
        + "".join(letters[m] for m in reversed(mb))
        + "->"
        + "".join(letters[m] for m in reversed(md))
    )
    ref = alpha * np.einsum(eq, view(A, ma), view(B, mb)) + beta * view(C, md)
    got = view(out, md)
    assert not np.isnan(got).any()
    assert np.max(np.abs(got - ref)) < 1e-12 * max(1.0, float(np.max(np.abs(ref))))


# ------------------------------------------------------ special values


def _gemm_nn(M, N, K, cx=False, **kw):
    return gemm_problem(
        M, N, K, a_strides=(K, 1), b_strides=(K, 1), d_strides=(N, 1), complex_=cx, **kw
    )


def _setup(p, seed, c_fill=None):
    rng = np.random.default_rng(seed)
    A, B, C, D = H.random_buffers(p, rng, bilinear=True)
    if c_fill is not None:
        C[:] = c_fill
    return A, B, C, D


def _flat_index(p, which, l, r, c):
    """Index of logical element (l, r, c) of ``which`` in its flat f64 buffer
    (complex: the real word)."""
    if which == "A":
        dig = (p.a_l, p.a_m, p.a_k)
    elif which == "B":
        dig = (p.b_l, p.b_n, p.b_k)
    else:
        dig = (p.d_l, p.d_m, p.d_n)
    off = sum(mixed_offset(i, d) for i, d in zip((l, r, c), dig))
    return off * (2 if p.complex_ else 1)


def _nonfinite(x):
    return ~np.isfinite(x)


@pytest.mark.parametrize("cx", [False, True])
def test_special_values(cx):
    """NaN / Inf propagation and the BLAS beta / alpha rules, bitwise on the
    whole output."""
    arch = _arch()
    M, N, K = 96, 80, 64
    p = _gemm_nn(M, N, K, cx)
    spec = arch_spec(p, arch, tile_m=64, tile_n=64)
    one = complex(1, 0) if cx else 1.0
    nan = complex(float("nan"), 0) if cx else float("nan")
    zero = complex(0, 0) if cx else 0.0
    for engine in _engines():
        compiled = H.compile_kernel(spec, arch, engine=engine)
        A, B, C, D = _setup(p, 31)
        A[_flat_index(p, "A", 0, 3, 17)] = np.nan  # D row 3 -> NaN
        B[_flat_index(p, "B", 0, 7, 11)] = np.inf  # D column 7 -> not finite
        for k in range(K):  # A row 5 all zero ...
            A[_flat_index(p, "A", 0, 5, k)] = 0.0
            if cx:
                A[_flat_index(p, "A", 0, 5, k) + 1] = 0.0
        B[_flat_index(p, "B", 0, 9, 0)] = np.inf  # ... times Inf: D(5, 9) NaN
        out = H.run(spec, arch, A, B, C, D.copy(), one, zero, compiled=compiled)
        Dm = H.gather(p, "D", out)[0]
        R = np.stack([Dm.real, Dm.imag], -1) if cx else Dm[..., None]
        assert np.isnan(R[3]).any(-1).all(), engine
        assert _nonfinite(R[:, 7]).any(-1).all(), engine
        assert np.isnan(R[5, 9]).any(), engine
        others = np.ones((M, N), bool)
        others[3, :] = others[:, 7] = others[:, 9] = False
        assert np.isfinite(R[others]).all(), engine

        A, B, C, D = _setup(p, 32, c_fill=np.nan)
        for beta in (zero, complex(-0.0, -0.0) if cx else -0.0):
            out = H.run(spec, arch, A, B, C, D.copy(), one, beta, compiled=compiled)
            assert np.isfinite(out).all(), f"{engine}: beta={beta} must not read C"
        A, B, C, D = _setup(p, 33, c_fill=1.0)
        for alpha, beta in ((one, nan), (nan, zero)):
            out = H.run(spec, arch, A, B, C, D.copy(), alpha, beta, compiled=compiled)
            Dm = H.gather(p, "D", out)
            R = np.stack([Dm.real, Dm.imag], -1) if cx else Dm[..., None]
            assert np.isnan(R).any(-1).all(), f"{engine}: alpha={alpha} beta={beta}"


@pytest.mark.parametrize("cx", [False, True])
def test_alpha_zero_does_not_read_a_or_b(cx):
    """BLAS: with alpha == 0, A and B are not used (NaN / Inf there do not
    reach D); D = beta * C, and D = 0 when beta is 0 too."""
    arch = _arch()
    p = _gemm_nn(96, 80, 40, cx)
    spec = arch_spec(p, arch, tile_m=64, tile_n=64)
    A, B, C, D = _setup(p, 41)
    A[::7] = np.nan
    B[::5] = np.inf
    zero = complex(0, 0) if cx else 0.0
    beta = complex(-0.75, 0.0) if cx else -0.75
    for engine in _engines():
        compiled = H.compile_kernel(spec, arch, engine=engine)
        out = H.run(spec, arch, A, B, C, D.copy(), zero, beta, compiled=compiled)
        got = H.gather(p, "D", out)
        want = H.gather(p, "C", C) * beta  # beta real: one rounding per part
        np.testing.assert_array_equal(got, want)
        out = H.run(spec, arch, A, B, C, D.copy(), zero, zero, compiled=compiled)
        assert not H.gather(p, "D", out).any()
        sc = arch_spec(p, arch, tile_m=64, tile_n=64, bilinear=False)
        out = H.run(sc, arch, A, B, None, D.copy(), zero, zero)
        assert not H.gather(p, "D", out).any()


def test_real_complex_scalars_keep_an_infinite_part_apart():
    """acc = (Inf, 0) times alpha = (2, 0) is (Inf, 0); a full complex
    multiply would give 0 * Inf = NaN in the imaginary part. Same for beta."""
    arch = _arch()
    M, N, K = 64, 64, 8
    p = _gemm_nn(M, N, K, True)
    spec = arch_spec(p, arch, tile_m=64, tile_n=64)
    for engine in _engines():
        A, B, C, D = _setup(p, 51)
        A[:] = 0.0
        B[:] = 0.0
        C[:] = 0.0
        ia = _flat_index(p, "A", 0, 2, 0)
        ib = _flat_index(p, "B", 0, 3, 0)
        A[ia], B[ib] = 1e300, 1e300  # overflows to Inf in the real accumulator
        out = H.run(spec, arch, A, B, C, D.copy(), complex(2, 0), complex(0, 0))
        d = H.gather(p, "D", out)[0, 2, 3]
        assert d.real == np.inf and d.imag == 0.0, (engine, d)
        ic = _flat_index(p, "D", 0, 4, 5)  # C has D's layout
        C[ic] = np.inf
        out = H.run(spec, arch, A, B, C, D.copy(), complex(1, 0), complex(3, 0))
        d = H.gather(p, "D", out)[0, 4, 5]
        assert d.real == np.inf and d.imag == 0.0, (engine, d)


@pytest.mark.parametrize("which", ["a", "b"])
def test_conjugation_equals_conjugated_data(which):
    """CONJ in the kernel is bitwise the kernel on host-conjugated data."""
    arch = _arch()
    M, N, K = 96, 64, 40
    kw = dict(a_strides=(K, 1), b_strides=(1, N), d_strides=(N, 1), complex_=True)
    pc = gemm_problem(M, N, K, conj_a=which == "a", conj_b=which == "b", **kw)
    pp = gemm_problem(M, N, K, **kw)
    rng = np.random.default_rng(61)
    A, B, C, D = H.random_buffers(pp, rng)
    alpha, beta = complex(1.25, -0.375), complex(-0.5, 0.625)
    Ac, Bc = A.copy(), B.copy()
    (Ac if which == "a" else Bc)[1::2] *= -1.0
    for accs in (2, 4):
        for engine in _engines():
            got = H.run(
                arch_spec(pc, arch, tile_m=64, tile_n=64, complex_accs=accs),
                arch,
                A,
                B,
                C,
                D.copy(),
                alpha,
                beta,
                engine=engine,
            )
            want = H.run(
                arch_spec(pp, arch, tile_m=64, tile_n=64, complex_accs=accs),
                arch,
                Ac,
                Bc,
                C,
                D.copy(),
                alpha,
                beta,
                engine=engine,
            )
            np.testing.assert_array_equal(got.view(np.int64), want.view(np.int64))


# ---------------------------------------------------- blocked K (k_block)


@pytest.mark.parametrize("cx", [False, True])
@pytest.mark.parametrize(
    "kw",
    [
        dict(k_block=4),
        dict(k_block=3, prefetch_depth=2, lds_stages=1),
        dict(k_block=2, unroll2=True),
        dict(k_block=5, k_stagger=2, lds_stages=1),
    ],
)
def test_k_block(kw, cx):
    arch = _arch()
    # long K, few tiles: the case blocked accumulation is for; odd tile count
    p = gemm_problem(
        96,
        80,
        1032,
        a_strides=(1032, 1),
        b_strides=(1, 80),
        d_strides=(80, 1),
        complex_=cx,
    )
    _run_check(arch_spec(p, arch, tile_m=64, tile_n=64, tile_k=8, **kw), arch, seed=71)


def test_k_block_shortens_the_rounding_chain():
    """On data whose sums cancel, blocked K accumulation is closer to the exact
    result than one sequential chain (the error grows with the chain length)."""
    arch = _arch()
    M, N, K = 64, 64, 16384
    p = gemm_problem(M, N, K, a_strides=(K, 1), b_strides=(K, 1), d_strides=(N, 1))
    rng = np.random.default_rng(73)
    A, B, C, D = H.random_buffers(p, rng, bilinear=False)
    A[:] = rng.uniform(0.5, 1.0, A.size)
    B[:] = rng.uniform(0.5, 1.0, B.size)  # all-positive: errors add up along K
    samples = [(0, m, n) for m in range(0, M, 7) for n in range(0, N, 5)]
    ref = H.reference(p, A, B, None, 1.0, 0.0, samples)

    def err(**kw):
        spec = arch_spec(p, arch, tile_m=64, tile_n=64, bilinear=False, **kw)
        out = H.gather(p, "D", H.run(spec, arch, A, B, None, D.copy(), 1.0, 0.0))
        return max(abs(out[l, m, n] - v) / abs(v) for (l, m, n), v, _ in ref)

    seq, blk = err(), err(k_block=16)
    assert blk < seq, (blk, seq)


# ------------------------------------------------------- runtime sizes


_RT_KNOBS = [
    dict(),
    dict(unroll2=True, k_stagger=1),
    dict(k_block=2, prefetch_depth=2, lds_stages=1, group_m=1),
    dict(xcds=8, group_m=4, lds_stages=1),
]


@pytest.mark.parametrize("kw", _RT_KNOBS)
def test_runtime_sizes_one_kernel_many_problems(kw):
    """One runtime-size kernel computes every problem of its layout class, and
    matches the compile-time kernel bitwise (up to the sign of zero)."""
    arch = _arch()
    probs = [
        gemm_problem(
            200, 136, 104, a_strides=(104, 1), b_strides=(1, 136), d_strides=(136, 1)
        ),
        gemm_problem(
            64, 48, 40, a_strides=(40, 1), b_strides=(1, 48), d_strides=(48, 1)
        ),
        gemm_problem(
            130,
            260,
            72,
            a_strides=(72, 1),
            b_strides=(1, 260),
            d_strides=(260, 1),
            c_strides=(260, 1),
            batch=3,
            batch_strides=(130 * 72, 260 * 72, 130 * 260),
            c_batch_stride=130 * 260 + 4,
        ),
    ]
    rt = None
    for i, p in enumerate(probs):
        spec = arch_spec(p, arch, runtime_sizes=True, tile_k=8, **kw)
        static = arch_spec(p, arch, tile_k=8, **kw)
        rng = np.random.default_rng(80 + i)
        A, B, C, D = H.random_buffers(p, rng)
        for engine in _engines():
            compiled = H.compile_kernel(spec, arch, engine=engine)
            if rt is None:
                rt = compiled[0]
            assert compiled[0] == rt  # one kernel for all
            got = H.run(
                spec, arch, A, B, C, D.copy(), 1.25, -0.5, problem=p, compiled=compiled
            )
            want = H.run(static, arch, A, B, C, D.copy(), 1.25, -0.5, engine=engine)
            g, w = H.gather(p, "D", got), H.gather(p, "D", want)
            np.testing.assert_array_equal(g, w)  # -0.0 == 0.0 compares equal
            worst, bad = H.check(
                p, A, B, C, got, D, 1.25, -0.5, _samples(p, rng, n=400)
            )
            assert bad is None and worst <= 1.0, engine


def test_runtime_sizes_complex():
    arch = _arch()
    for conj in ((True, False), (False, True)):
        p = gemm_problem(
            96,
            64,
            40,
            a_strides=(40, 1),
            b_strides=(1, 64),
            d_strides=(64, 1),
            complex_=True,
            conj_a=conj[0],
            conj_b=conj[1],
        )
        spec = arch_spec(p, arch, runtime_sizes=True, tile_m=64, tile_n=64, tile_k=8)
        rng = np.random.default_rng(91)
        A, B, C, D = H.random_buffers(p, rng)
        alpha, beta = complex(1.25, -0.375), complex(-0.5, 0.625)
        for engine in _engines():
            out = H.run(
                spec, arch, A, B, C, D.copy(), alpha, beta, engine=engine, problem=p
            )
            worst, bad = H.check(p, A, B, C, out, D, alpha, beta, _samples(p, rng))
            assert bad is None and worst <= 1.0, engine


def test_batches_beyond_the_grid_run_in_several_launches():
    arch = _arch()
    ext = {"m": 4, "n": 4, "k": 8, "l": 65537}
    ma, mb, md = ["m", "k", "l"], ["n", "k", "l"], ["m", "n", "l"]
    sa, sb, sd = (packed_strides(x, ext) for x in (ma, mb, md))
    plan = plan_contraction(ma, sa, mb, sb, md, sd, ext)
    assert len(plan.launches) == 2
    rng = np.random.default_rng(97)
    n = lambda ms: int(np.prod([ext[m] for m in ms]))  # noqa: E731
    A = rng.integers(-4, 5, n(ma)).astype(np.float64)
    B = rng.integers(-4, 5, n(mb)).astype(np.float64)
    C = rng.integers(-4, 5, n(md)).astype(np.float64)
    D = np.full(n(md), np.nan)
    out = H.run_plan(
        plan, arch, A, B, C, D, 2.0, -1.0, tile_m=16, tile_n=16, warps_m=1, warps_n=1
    )
    a = A.reshape(ext["l"], ext["k"], ext["m"])
    b = B.reshape(ext["l"], ext["k"], ext["n"])
    ref = 2.0 * np.einsum("lkm,lkn->lnm", a, b) - C.reshape(
        ext["l"], ext["n"], ext["m"]
    )
    np.testing.assert_array_equal(out.reshape(ext["l"], ext["n"], ext["m"]), ref)


# --------------------------------------------------------------- split_k


@pytest.mark.parametrize("cx", [False, True])
@pytest.mark.parametrize(
    "kw",
    [
        dict(split_k=4),
        dict(split_k=7, prefetch_depth=2, lds_stages=1),
        dict(split_k=3, k_block=4),
        dict(split_k=5, runtime_sizes=True),
    ],
)
def test_split_k(kw, cx):
    """Partial sums per K range, then the ordered reduce + epilogue: within the
    f64 bound, bitwise repeatable, and with C read through its own strides."""
    arch = _arch()
    M, N, K, L = 96, 80, 1032, 2
    p = gemm_problem(
        M,
        N,
        K,
        a_strides=(K, 1),
        b_strides=(1, N),
        d_strides=(N, 1),
        c_strides=(1, M),
        batch=L,
        batch_strides=(M * K, N * K, M * N),
        c_batch_stride=M * N,
        complex_=cx,
    )
    spec = arch_spec(p, arch, tile_m=64, tile_n=64, tile_k=8, **kw)
    rng = np.random.default_rng(101)
    A, B, C, D = H.random_buffers(p, rng)
    alpha, beta = (complex(1.25, -0.375), complex(-0.5, 0.625)) if cx else (1.25, -0.5)
    for engine in _engines():
        out = H.run(
            spec, arch, A, B, C, D.copy(), alpha, beta, engine=engine, problem=p
        )
        worst, bad = H.check(p, A, B, C, out, D, alpha, beta, _samples(p, rng))
        assert bad is None and worst <= 1.0, engine
        again = H.run(
            spec, arch, A, B, C, D.copy(), alpha, beta, engine=engine, problem=p
        )
        np.testing.assert_array_equal(out.view(np.int64), again.view(np.int64))


def test_split_k_special_values():
    """The reduce kernel keeps the BLAS rules: beta == 0 ignores NaN C, alpha
    == 0 ignores NaN A."""
    arch = _arch()
    p = gemm_problem(
        64, 48, 512, a_strides=(512, 1), b_strides=(512, 1), d_strides=(48, 1)
    )
    spec = arch_spec(p, arch, tile_m=64, tile_n=64, tile_k=8, split_k=4)
    rng = np.random.default_rng(103)
    A, B, C, D = H.random_buffers(p, rng)
    C[:] = np.nan
    out = H.run(spec, arch, A, B, C, D.copy(), 1.0, 0.0)
    assert np.isfinite(out).all()
    A[::3] = np.nan
    C[:] = 2.0
    out = H.run(spec, arch, A, B, C, D.copy(), 0.0, 0.5)
    np.testing.assert_array_equal(H.gather(p, "D", out), 1.0)


# ------------------------------------------- pipelined K loops / AGPR accumulators

# (Python engine levers of gett_f64: they keep the K order of every output, so
# the bits equal the default kernel's)
_PIPE_KNOBS = [
    dict(acc_agpr=True),
    dict(frag_pipe=True, tile_k=16),
    dict(frag_pipe=True, unroll2=True, tile_k=8),
    dict(frag_pipe=True, acc_agpr=True, unroll2=True, tile_k=16),
    dict(kmajor_lds=True, tile_k=16),
    dict(kmajor_lds=True, kmajor_pad=4, unroll2=True, tile_k=8),
    dict(ilv=4, prefetch_depth=2, tile_k=16),
    dict(
        ilv=8, prefetch_depth=2, unroll2=True, tile_k=8, kmajor_lds=True, acc_agpr=True
    ),
    dict(ilv=64, ilv_wgap=0, prefetch_depth=2, unroll2=True, tile_k=16),
    dict(ilv=4, prefetch_depth=2, unroll2=True, tile_k=12, warps_m=4, warps_n=1),
    dict(k_inner=2, mfma_fence=True, tile_k=16),
    dict(k_inner=4, frag_pipe=True, tile_k=16),
]

# (M, N, K, A strides, B strides, D strides): both operands K-contiguous with a
# K tail, both M/N-contiguous, mixed with an even K-tile count
_PIPE_PROBLEMS = [
    (200, 136, 104, (104, 1), (104, 1), (1, 200)),
    (192, 160, 72, (1, 192), (1, 160), (160, 1)),
    (136, 200, 96, (1, 136), (96, 1), (1, 136)),
]


@pytest.mark.parametrize("kw", _PIPE_KNOBS, ids=lambda kw: knobs_id(kw))
def test_pipeline_knobs_bitwise(kw):
    """Every lever passes the f64 bound and gives the default kernel's bits
    (same tiles: only the schedule / register file / LDS layout change)."""
    arch = _arch()
    for i, (M, N, K, sa, sb, sd) in enumerate(_PIPE_PROBLEMS):
        p = gemm_problem(M, N, K, a_strides=sa, b_strides=sb, d_strides=sd)
        tiles = dict(tile_m=64, tile_n=64)
        if "warps_m" in kw:
            tiles = dict(tile_m=128, tile_n=64)
        spec = arch_spec(p, arch, **tiles, **kw)
        base = arch_spec(
            p,
            arch,
            **tiles,
            tile_k=spec.tile_k,
            warps_m=spec.warps_m,
            warps_n=spec.warps_n,
        )
        _run_check(spec, arch, seed=40 + i)
        rng = np.random.default_rng(50 + i)
        A, B, C, D = H.random_buffers(p, rng)
        got = H.run(spec, arch, A, B, C, D.copy(), 1.25, -0.5)
        ref = H.run(base, arch, A, B, C, D.copy(), 1.25, -0.5)
        np.testing.assert_array_equal(got.view(np.int64), ref.view(np.int64))


def knobs_id(kw):
    return ",".join(f"{k}={v}" for k, v in kw.items())


@pytest.mark.parametrize("cx", [False, True])
@pytest.mark.parametrize(
    "conj", [(False, False), (True, False), (False, True), (True, True)]
)
def test_pipeline_knobs_complex(cx, conj):
    """frag_pipe, acc_agpr and ilv also carry the complex fragments (re, im):
    f64 bound, and the default kernel's bits."""
    arch = _arch()
    if not cx and conj != (False, False):
        pytest.skip("conjugation is complex only")
    p = gemm_problem(
        96,
        64,
        40,
        a_strides=(40, 1),
        b_strides=(1, 64),
        d_strides=(1, 96),
        complex_=cx,
        conj_a=conj[0],
        conj_b=conj[1],
    )
    for kw in (
        dict(frag_pipe=True),
        dict(frag_pipe=True, acc_agpr=True, unroll2=True),
        dict(ilv=8, prefetch_depth=2, unroll2=True, acc_agpr=True),
        dict(ilv=4, prefetch_depth=2, complex_accs=4),
    ):
        spec = arch_spec(p, arch, tile_m=64, tile_n=64, tile_k=8, **kw)
        _run_check(spec, arch, seed=61)
        base = arch_spec(
            p, arch, tile_m=64, tile_n=64, tile_k=8, complex_accs=spec.complex_accs
        )
        rng = np.random.default_rng(62)
        A, B, C, D = H.random_buffers(p, rng)
        al, be = (complex(1.25, -0.375), complex(-0.5, 0.625)) if cx else (1.25, -0.5)
        got = H.run(spec, arch, A, B, C, D.copy(), al, be)
        ref = H.run(base, arch, A, B, C, D.copy(), al, be)
        np.testing.assert_array_equal(got.view(np.int64), ref.view(np.int64))


@pytest.mark.parametrize("cx", [False, True])
@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("split", [2, 5])
def test_split_serial(mode, split, cx):
    """split_serial reduces the partials in the launch: f64 bound, bitwise
    equal to the two-kernel split_k (same z order; real data), repeatable,
    and every launch leaves the counters at zero."""
    arch = _arch()
    if mode == 2 and arch not in ("gfx942", "gfx950"):
        pytest.skip("write-through partials need gfx942 / gfx950")
    M, N, K, L = 96, 80, 1032, 2
    p = gemm_problem(
        M,
        N,
        K,
        a_strides=(K, 1),
        b_strides=(1, N),
        d_strides=(N, 1),
        c_strides=(1, M),
        batch=L,
        batch_strides=(M * K, N * K, M * N),
        c_batch_stride=M * N,
        complex_=cx,
    )
    spec = arch_spec(
        p, arch, tile_m=64, tile_n=64, tile_k=8, split_k=split, split_serial=mode
    )
    rng = np.random.default_rng(105)
    A, B, C, D = H.random_buffers(p, rng)
    alpha, beta = (complex(1.25, -0.375), complex(-0.5, 0.625)) if cx else (1.25, -0.5)
    out = H.run(spec, arch, A, B, C, D.copy(), alpha, beta, problem=p, launches=3)
    worst, bad = H.check(p, A, B, C, out, D, alpha, beta, _samples(p, rng))
    assert bad is None and worst <= 1.0
    if not cx:
        two = arch_spec(p, arch, tile_m=64, tile_n=64, tile_k=8, split_k=split)
        ref = H.run(two, arch, A, B, C, D.copy(), alpha, beta, problem=p)
        np.testing.assert_array_equal(out.view(np.int64), ref.view(np.int64))


def test_pipe_configurations():
    """Every pipelined dispatch configuration of the arch (PIPE_CONFIGS) on a
    problem with tails in M, N and K, both load directions."""
    from rocke.instances.common.gett_f64 import PIPE_CONFIGS

    arch = _arch()
    for (a, kind, cls), cfg in PIPE_CONFIGS.items():
        if a != arch:
            continue
        for sa, sb in (((136, 1), (1, 520)), ((1, 600), (136, 1))):
            p = gemm_problem(
                600,
                520,
                136,
                a_strides=sa,
                b_strides=sb,
                d_strides=(1, 600),
                complex_=kind == "complex",
            )
            _run_check(arch_spec(p, arch, **cfg), arch, seed=71)
