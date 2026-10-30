# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""f32 / complex-f32 GETT (``GettF64Problem.dtype = "f32"``) on the device
against exact references.

The f32 kernel is the f64 GETT with f32 words: f32 MFMA 16x16x4 products and
an f32 accumulator. Every output checked is compared with the correctly
rounded exact result (``gett_f64_host.reference``) under the f32 accumulation
bound (``u = 2^-24``); the integer case must match exactly. NaN / Inf
propagation, the BLAS alpha / beta rules and conjugation are checked on whole
outputs. Runs on every target whose catalog has the f32 atom, through both
engines when the native one is built.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rocke.instances.common.gett_f64 import (
    F32_GETT_ARCHS,
    _with_k_block,
    arch_spec,
    gemm_problem,
    mixed_offset,
    runtime_template,
    tuned_configs,
)
from rocke.runtime.hip_module import get_device_arch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gett_f64_host as H  # noqa: E402


def _arch():
    arch = (get_device_arch() or "").split(":")[0]
    if arch not in F32_GETT_ARCHS:
        pytest.skip(f"requires a target with the f32 atom {F32_GETT_ARCHS}")
    return arch


def _engines():
    if importlib.util.find_spec("rocke_engine") is None:
        return ("python",)
    return ("python", "cpp")


def _gp(*args, **kw):
    return gemm_problem(*args, dtype="f32", **kw)


def _samples(p, rng, n=1500):
    total = p.L * p.M * p.N
    if total <= 30000:
        return [(l, m, k) for l in range(p.L) for m in range(p.M) for k in range(p.N)]
    return [
        (int(rng.integers(p.L)), int(rng.integers(p.M)), int(rng.integers(p.N)))
        for _ in range(n)
    ]


def _run_check(
    spec,
    arch,
    *,
    alpha=1.25,
    beta=-0.5,
    seed=0,
    integer=False,
    c_nan=False,
    problem=None,
):
    p = problem if problem is not None else spec.problem
    rng = np.random.default_rng(seed)
    A, B, C, D = H.random_buffers(p, rng, integer=integer, bilinear=spec.bilinear)
    if p.complex_:
        alpha, beta = complex(alpha, -0.375), complex(beta, 0.625)
    if c_nan and C is not None:
        C[:] = np.nan
        beta = 0j if p.complex_ else 0.0
    for engine in _engines():
        out = H.run(
            spec, arch, A, B, C, D.copy(), alpha, beta, engine=engine, problem=problem
        )
        assert out.dtype == np.float32
        if integer:
            Am = H.gather(p, "A", A).astype(np.float64)
            Bm = H.gather(p, "B", B).astype(np.float64)
            ref = alpha * np.einsum("lmk,lnk->lmn", Am, Bm)
            if spec.bilinear:
                ref = ref + beta * H.gather(p, "C", C).astype(np.float64)
            np.testing.assert_array_equal(H.gather(p, "D", out).astype(np.float64), ref)
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


LAYOUTS = {
    "rcr": lambda M, N, K: ((K, 1), (K, 1), (N, 1)),
    "rrr": lambda M, N, K: ((K, 1), (1, N), (N, 1)),
    "ccc": lambda M, N, K: ((1, M), (K, 1), (1, M)),
    "crc": lambda M, N, K: ((1, M), (1, N), (1, M)),
}


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize(
    "K", [100, 98, 97]
)  # runs of 4k, 4k + 2 and odd: 16, 8, 4-byte loads
def test_gemm_layouts(layout, K):
    arch = _arch()
    M, N = 192, 160
    sa, sb, sd = LAYOUTS[layout](M, N, K)
    p = _gp(M, N, K, a_strides=sa, b_strides=sb, d_strides=sd)
    _run_check(arch_spec(p, arch), arch, seed=sorted(LAYOUTS).index(layout) + K)


def test_tails_and_tiles():
    arch = _arch()
    p = _gp(200, 136, 68, a_strides=(68, 1), b_strides=(68, 1), d_strides=(136, 1))
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
        dict(k_block=4, tile_k=8),
        dict(k_block=2, unroll2=True, tile_k=8),
    ],
)
def test_schedule_knobs(kw):
    arch = _arch()
    p = _gp(200, 136, 104, a_strides=(104, 1), b_strides=(104, 1), d_strides=(1, 200))
    _run_check(arch_spec(p, arch, **kw), arch, seed=11)
    q = _gp(192, 160, 72, a_strides=(1, 192), b_strides=(1, 160), d_strides=(160, 1))
    _run_check(arch_spec(q, arch, **kw), arch, seed=12)


@pytest.mark.parametrize(
    "kw",
    [dict(), dict(complex_accs=4), dict(complex_accs=4, unroll2=True), dict(k_block=3)],
)
@pytest.mark.parametrize(
    "conj", [(False, False), (True, False), (False, True), (True, True)]
)
def test_complex_fused_4m(kw, conj):
    arch = _arch()
    p = _gp(
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


@pytest.mark.parametrize("kind", ["real", "complex"])
def test_tuned_shapes(kind):
    """The f32 GETT's tuned configurations of this target (tuned_configs: its
    own classes, or the f64 classes with f32 words)."""
    arch = _arch()
    for cls, cfg in sorted(tuned_configs(arch, kind, "f32").items()):
        cfg = dict(cfg)
        cfg.pop("split_k", None)
        p = _gp(
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
    p = _gp(96, 80, 40, a_strides=(40, 1), b_strides=(40, 1), d_strides=(80, 1))
    _run_check(arch_spec(p, arch, bilinear=False), arch, seed=5)
    _run_check(arch_spec(p, arch), arch, seed=6, c_nan=True)


def test_c_has_its_own_strides_and_batch():
    arch = _arch()
    M, N, K, L = 64, 48, 32, 3
    p = _gp(
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
    # |sum| <= 64 * 512 < 2^24: every partial sum is an exact f32
    p = _gp(160, 144, 512, a_strides=(512, 1), b_strides=(512, 1), d_strides=(144, 1))
    _run_check(arch_spec(p, arch), arch, alpha=1.0, beta=1.0, integer=True, seed=9)


# ------------------------------------- the pack's runtime-size kernels


def _pack_spec(arch, cplx, cls, dclass, bilinear, kblock, K):
    """A pack descriptor (packed K-contiguous A and B, D class m / n / x) for
    a problem of ``K`` (padded the way the planner pads f32 operands)."""
    M, N, L = 64, 64, 2
    d_strides = {"m": (1, M), "n": (N, 1), "x": (2, 2 * M)}[dclass]
    prob = _gp(
        M,
        N,
        K,
        a_strides=(K, 1),
        b_strides=(K, 1),
        d_strides=d_strides,
        batch=L,
        batch_strides=(M * K, N * K, 4 * M * N),
        complex_=cplx,
    )
    kind = "complex" if cplx else "real"
    cfg = {} if cls == "default" else dict(tuned_configs(arch, kind, "f32")[cls])
    cfg.pop("split_k", None)
    spec = arch_spec(prob, arch, bilinear=bilinear, runtime_sizes=True, **cfg)
    spec = replace(spec, split_k=1, problem=runtime_template(prob))
    if kblock:
        kb = _with_k_block(spec, arch)
        if kb.k_block == spec.k_block:
            return None
        spec = kb
    return spec


@pytest.mark.parametrize("cplx", [False, True])
@pytest.mark.parametrize("dclass", ["m", "n", "x"])
def test_pack_kernels_serve_many_problems(cplx, dclass):
    """The pack's runtime-size f32 kernels (every class of the target, both K
    block forms) on problems of other sizes in the same layout class."""
    arch = _arch()
    kind = "complex" if cplx else "real"
    classes = sorted(tuned_configs(arch, kind, "f32"))
    for cls in classes + ["default"]:
        for kblock in (False, True):
            spec = _pack_spec(arch, cplx, cls, dclass, True, kblock, 64)
            if spec is None:
                continue
            for i, (M, N, K, L) in enumerate(((200, 136, 104, 2), (40, 72, 4100, 1))):
                if K > 1024 and not kblock:
                    continue
                d_strides = {"m": (1, M), "n": (N, 1), "x": (2, 2 * M)}[dclass]
                p = _gp(
                    M,
                    N,
                    K,
                    a_strides=(K, 1),
                    b_strides=(K, 1),
                    d_strides=d_strides,
                    batch=L,
                    batch_strides=(M * K, N * K, 4 * M * N),
                    complex_=cplx,
                )
                _run_check(spec, arch, seed=19 + i, problem=p)


# ------------------------------------------------------ special values


def _gemm_nn(M, N, K, cx=False, **kw):
    return _gp(
        M, N, K, a_strides=(K, 1), b_strides=(K, 1), d_strides=(N, 1), complex_=cx, **kw
    )


def _flat_index(p, which, l, r, c):
    if which == "A":
        dig = (p.a_l, p.a_m, p.a_k)
    elif which == "B":
        dig = (p.b_l, p.b_n, p.b_k)
    else:
        dig = (p.d_l, p.d_m, p.d_n)
    off = sum(mixed_offset(i, d) for i, d in zip((l, r, c), dig))
    return off * (2 if p.complex_ else 1)


@pytest.mark.parametrize("cx", [False, True])
def test_special_values(cx):
    """NaN / Inf propagation and the BLAS beta / alpha rules on the whole output."""
    arch = _arch()
    M, N, K = 96, 80, 64
    p = _gemm_nn(M, N, K, cx)
    spec = arch_spec(p, arch, tile_m=64, tile_n=64)
    one = complex(1, 0) if cx else 1.0
    nan = complex(float("nan"), 0) if cx else float("nan")
    zero = complex(0, 0) if cx else 0.0
    for engine in _engines():
        compiled = H.compile_kernel(spec, arch, engine=engine)
        rng = np.random.default_rng(31)
        A, B, C, D = H.random_buffers(p, rng)
        A[_flat_index(p, "A", 0, 3, 17)] = np.nan
        B[_flat_index(p, "B", 0, 7, 11)] = np.inf
        for k in range(K):
            A[_flat_index(p, "A", 0, 5, k)] = 0.0
            if cx:
                A[_flat_index(p, "A", 0, 5, k) + 1] = 0.0
        B[_flat_index(p, "B", 0, 9, 0)] = np.inf
        out = H.run(spec, arch, A, B, C, D.copy(), one, zero, compiled=compiled)
        Dm = H.gather(p, "D", out)[0]
        R = np.stack([Dm.real, Dm.imag], -1) if cx else Dm[..., None]
        assert np.isnan(R[3]).any(-1).all(), engine
        assert (~np.isfinite(R[:, 7])).any(-1).all(), engine
        assert np.isnan(R[5, 9]).any(), engine
        others = np.ones((M, N), bool)
        others[3, :] = others[:, 7] = others[:, 9] = False
        assert np.isfinite(R[others]).all(), engine

        A, B, C, D = H.random_buffers(p, np.random.default_rng(32))
        C[:] = np.nan
        for beta in (zero, complex(-0.0, -0.0) if cx else -0.0):
            out = H.run(spec, arch, A, B, C, D.copy(), one, beta, compiled=compiled)
            assert np.isfinite(out).all(), f"{engine}: beta={beta} must not read C"
        A, B, C, D = H.random_buffers(p, np.random.default_rng(33))
        C[:] = 1.0
        for alpha, beta in ((one, nan), (nan, zero)):
            out = H.run(spec, arch, A, B, C, D.copy(), alpha, beta, compiled=compiled)
            Dm = H.gather(p, "D", out)
            R = np.stack([Dm.real, Dm.imag], -1) if cx else Dm[..., None]
            assert np.isnan(R).any(-1).all(), f"{engine}: alpha={alpha} beta={beta}"


@pytest.mark.parametrize("cx", [False, True])
def test_alpha_zero_does_not_read_a_or_b(cx):
    arch = _arch()
    p = _gemm_nn(96, 80, 40, cx)
    spec = arch_spec(p, arch, tile_m=64, tile_n=64)
    A, B, C, D = H.random_buffers(p, np.random.default_rng(41))
    A[::7] = np.nan
    B[::5] = np.inf
    zero = complex(0, 0) if cx else 0.0
    beta = complex(-0.75, 0.0) if cx else -0.75
    for engine in _engines():
        out = H.run(spec, arch, A, B, C, D.copy(), zero, beta, engine=engine)
        got = H.gather(p, "D", out)
        want = (beta * H.gather(p, "C", C).astype(got.dtype)).astype(got.dtype)
        np.testing.assert_array_equal(got, want)
        out = H.run(spec, arch, A, B, C, D.copy(), zero, zero, engine=engine)
        assert not np.any(H.gather(p, "D", out)), engine


def test_f32_accumulation_is_not_emulated():
    """Products of f32 values with 24 significant bits keep their low bits:
    the result equals the exact f32-rounded value on data where a bf16 split
    with fewer products would lose them (one exact product per output)."""
    arch = _arch()
    M, N, K = 64, 64, 4
    p = _gp(M, N, K, a_strides=(K, 1), b_strides=(K, 1), d_strides=(N, 1))
    spec = arch_spec(p, arch, tile_m=64, tile_n=64, bilinear=False)
    rng = np.random.default_rng(5)
    A = np.zeros(M * K, np.float32)
    B = np.zeros(N * K, np.float32)
    # one non-zero K term per output: a * b with full 24-bit mantissas
    A[0::K] = (1.0 + rng.integers(1, 2**23, M) * 2.0**-23).astype(np.float32)
    B[0::K] = (1.0 + rng.integers(1, 2**23, N) * 2.0**-23).astype(np.float32)
    D = np.full(M * N, np.nan, np.float32)
    for engine in _engines():
        out = H.run(spec, arch, A, B, None, D.copy(), 1.0, 0.0, engine=engine)
        want = np.outer(A[0::K].astype(np.float64), B[0::K].astype(np.float64)).astype(
            np.float32
        )
        np.testing.assert_array_equal(out.reshape(M, N), want)


@pytest.mark.parametrize("K", [4, 12, 40])
def test_accumulation_is_an_f32_fma_chain_in_k_order(K):
    """The f32 MFMA accumulates like a chain of correctly rounded f32 fmas in K
    order (each product enters unrounded), on random full-mantissa operands
    with a wide exponent spread and both signs: bitwise on every output."""
    from fractions import Fraction

    arch = _arch()
    M = N = 32
    p = _gp(M, N, K, a_strides=(K, 1), b_strides=(K, 1), d_strides=(N, 1))
    spec = arch_spec(
        p, arch, tile_m=32, tile_n=32, warps_m=1, warps_n=1, tile_k=4, bilinear=False
    )
    rng = np.random.default_rng(K)

    def operand(n):
        mant = rng.integers(1 << 23, 1 << 24, n).astype(np.float64)
        sign = np.where(rng.integers(0, 2, n) == 1, 1.0, -1.0)
        return (sign * mant * 2.0 ** rng.integers(-36, -18, n)).astype(np.float32)

    def to_f32(x):
        """``x`` (a Fraction) rounded once to nearest even f32."""
        f = np.float32(float(x))
        best, gap = f, abs(Fraction(float(f)) - x)
        for g in (
            np.nextafter(f, np.float32(np.inf)),
            np.nextafter(f, np.float32(-np.inf)),
        ):
            d = abs(Fraction(float(g)) - x)
            if d < gap or (d == gap and int(g.view(np.int32)) % 2 == 0):
                best, gap = g, d
        return best

    A, B = operand(M * K), operand(N * K)
    D = np.full(M * N, np.nan, np.float32)
    want = np.empty((M, N), np.float32)
    for m in range(M):
        for n in range(N):
            acc = np.float32(0.0)
            for k in range(K):
                exact = Fraction(float(A[m * K + k])) * Fraction(float(B[n * K + k]))
                acc = to_f32(exact + Fraction(float(acc)))
            want[m, n] = acc
    for engine in _engines():
        out = H.run(spec, arch, A, B, None, D.copy(), 1.0, 0.0, engine=engine)
        np.testing.assert_array_equal(out.reshape(M, N), want, err_msg=engine)
