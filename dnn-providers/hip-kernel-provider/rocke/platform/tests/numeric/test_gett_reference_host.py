# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only: the vectorised GETT reference (``gett_f64_host.reference``) gives
exactly the values and bounds of the per-sample oracle (``reference_exact``:
``math.fsum`` / ``Fraction``), including rows whose sum lies on or next to a
rounding boundary (decided by ``math.fsum``) and operands outside the range
of the error-free transformations (the oracle's own path)."""

from __future__ import annotations

import itertools
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from rocke.instances.common.gett_f64 import gemm_problem

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gett_f64_host as H  # noqa: E402


def _bits(x):
    return struct.pack("<d", float(x))


def _same(fast, exact):
    assert len(fast) == len(exact)
    for (i1, v1, b1), (i2, v2, b2) in zip(fast, exact):
        assert i1 == i2
        if isinstance(v2, complex):
            assert _bits(v1.real) == _bits(v2.real) and _bits(v1.imag) == _bits(
                v2.imag
            ), (i1, v1, v2)
        else:
            assert _bits(v1) == _bits(v2), (i1, v1, v2)
        assert _bits(b1) == _bits(b2), (i1, b1, b2)


def _problem(M, N, K, *, cx=False, conj=(False, False), dtype="f64", batch=1):
    return gemm_problem(
        M,
        N,
        K,
        a_strides=(K, 1),
        b_strides=(K, 1),
        d_strides=(N, 1),
        batch=batch,
        batch_strides=(M * K, N * K, M * N),
        complex_=cx,
        conj_a=conj[0],
        conj_b=conj[1],
        dtype=dtype,
    )


def _all(p):
    return [(l, m, n) for l in range(p.L) for m in range(p.M) for n in range(p.N)]


@pytest.mark.parametrize("dtype", ["f64", "f32"])
@pytest.mark.parametrize("bilinear", [False, True])
@pytest.mark.parametrize("K", [1, 7, 96, 1030])
def test_real_matches_the_oracle(dtype, bilinear, K):
    p = _problem(9, 7, K, dtype=dtype, batch=2)
    rng = np.random.default_rng(K * 4 + bilinear)
    A, B, C, _ = H.random_buffers(p, rng, bilinear=bilinear)
    C = C if bilinear else None
    for alpha, beta in ((1.25, -0.5), (1.0, 0.0), (-3.0e-3, 7.5)):
        args = (p, A, B, C, alpha, beta, _all(p))
        _same(H.reference(*args), H.reference_exact(*args))


@pytest.mark.parametrize(
    "conj", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("dtype", ["f64", "f32"])
def test_complex_matches_the_oracle(conj, dtype):
    p = _problem(6, 5, 37, cx=True, conj=conj, dtype=dtype)
    rng = np.random.default_rng(sum(conj) + 3)
    A, B, C, _ = H.random_buffers(p, rng)
    for alpha, beta in ((complex(1.25, -0.375), complex(-0.5, 0.625)), (1.0, 0.0)):
        for c in (C, None):
            args = (p, A, B, c, alpha, beta, _all(p))
            _same(H.reference(*args), H.reference_exact(*args))


def _rows(rows, b_rows=None):
    """A real f64 problem whose rows of A are ``rows`` and B all ones (or b_rows)."""
    K = max(len(r) for r in rows)
    p = _problem(len(rows), 1, K)
    A = np.zeros(len(rows) * K)
    B = np.ones(K)
    for i, r in enumerate(rows):
        A[i * K : i * K + len(r)] = r
    if b_rows is not None:
        B[: len(b_rows)] = b_rows
    return p, A, B


def test_boundary_and_out_of_range_rows_match_the_oracle(monkeypatch):
    rows = [
        [1.0, 2.0**-53, 2.0**-106],  # just above a tie: RN is 1 + 2^-52
        [1.0, 2.0**-53],  # a tie: to even
        [1.0 + 2.0**-52, 2.0**-53],  # a tie: to even (upwards)
        [1e16, 1.0, -1e16, 3.0],  # cancellation
        [
            2.0**-600,
            2.0**-600,
        ],  # products below the error-free range: the oracle's path
        [2.0**600, -(2.0**600), 1.0],  # products above it
        [0.0, -0.0, 0.0],
        [0.1, 0.2, 0.3, -0.6],
    ]
    p, A, B = _rows(rows)
    samples = _all(p)
    exact = H.reference_exact(p, A, B, None, 1.0, 0.0, samples)
    calls = []
    real_fsum = math.fsum

    def counting_fsum(xs):
        calls.append(1)
        return real_fsum(xs)

    monkeypatch.setattr(H.math, "fsum", counting_fsum)
    fast = H.reference(p, A, B, None, 1.0, 0.0, samples)
    _same(fast, exact)
    assert fast[0][1] == 1.0 + 2.0**-52
    # the boundary rows are settled by fsum, not by the vectorised sum
    assert len(calls) >= 3


def test_well_separated_rows_need_no_fsum(monkeypatch):
    p = _problem(40, 30, 64)
    A, B, _, _ = H.random_buffers(p, np.random.default_rng(5), bilinear=False)

    def no_fsum(xs):
        raise AssertionError("fsum on a row the certificate should settle")

    exact = H.reference_exact(p, A, B, None, 1.25, 0.0, _all(p))
    monkeypatch.setattr(H.math, "fsum", no_fsum)
    _same(H.reference(p, A, B, None, 1.25, 0.0, _all(p)), exact)


def test_rn_row_sums_is_correctly_rounded():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((300, 50)) * np.exp2(rng.integers(-40, 40, (300, 50)))
    x[:, 1] = -x[:, 0] * (1 + 2.0**-40)  # heavy cancellation
    got = H._rn_row_sums(x)
    for i in range(x.shape[0]):
        assert _bits(got[i]) == _bits(math.fsum(x[i].tolist())), i


def test_rn_row_sums_on_rows_next_to_a_rounding_boundary():
    """1 + three or four terms of half-ulp scale and far below: every row is within a
    few error terms of a tie, so the certificate must leave some rows to fsum (rows
    on which a certificate without the error-sum bound rounds the wrong way)."""
    vals = [0.0]
    for e in (53, 54, 105, 106, 107, 159, 160):
        vals += [2.0**-e, -(2.0**-e)]
    grid = itertools.product(vals, repeat=4)
    x = np.array([[1.0, *r] for r in grid])
    got = H._rn_row_sums(x)
    want = np.array([math.fsum(row) for row in x.tolist()])
    assert np.array_equal(got, want)
