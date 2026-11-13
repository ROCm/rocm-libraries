# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only tests for the float64 SDPA backward oracle.

``rocke.numeric.sdpa_reference.sdpa_reference_bwd`` is checked four ways:

* against torch autograd in float64 on CPU (an explicit masked-softmax graph
  and ``torch.nn.functional.scaled_dot_product_attention``); torch is optional,
  so these tests skip without it;
* against central finite differences of ``sum(O * dO)`` (torch-free);
* against an independent scalar per-element loop with a different structure
  (``delta`` from ``sum_j P_ij * dP_ij`` instead of ``rowsum(dO * O)``);
* for fully masked rows, padded rows / keys and argument errors.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from rocke.numeric import references
from rocke.numeric.sdpa_reference import (
    SdpaBwdResult,
    resolve_diagonal_band,
    sdpa_reference,
    sdpa_reference_bwd,
)

# (name, left, right, top_left) masks used across the suites
MASKS = [
    ("none", -1, -1, True),
    ("causal_tl", -1, 0, True),
    ("causal_br", -1, 0, False),
    ("left_only_tl", 3, -1, True),
    ("right_only_br", -1, 2, False),
    ("two_sided_tl", 4, 2, True),
    ("window_br_w5", 5, 0, False),
    ("diag_only_br", 0, 0, False),
]


def _torch_available():
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


def _rand(rng, shape):
    return rng.standard_normal(shape)


def _band(sq, skv, left, right, tl):
    qi = np.arange(sq)[:, None]
    ki = np.arange(skv)[None, :]
    off = 0 if tl else skv - sq
    m = np.ones((sq, skv), dtype=bool)
    if right >= 0:
        m &= ki <= qi + off + right
    if left >= 0:
        m &= ki >= qi + off - left
    return m


def _problem(rng, B, Hq, Hk, Sq, Skv, D, Hv=None):
    Hv = Hk if Hv is None else Hv
    q = _rand(rng, (B, Hq, Sq, D))
    k = _rand(rng, (B, Hk, Skv, D))
    v = _rand(rng, (B, Hv, Skv, D))
    do = _rand(rng, (B, Hq, Sq, D))
    return q, k, v, do


def _run(
    q, k, v, do, *, scale=None, left=-1, right=-1, tl=True, lens_q=None, lens_kv=None
):
    kw = dict(scale=scale, left_bound=left, right_bound=right, top_left=tl)
    o, lse = sdpa_reference(q, k, v, seq_len_q=lens_q, seq_len_kv=lens_kv, **kw)
    res = sdpa_reference_bwd(
        q, k, v, o, do, lse, seq_len_q=lens_q, seq_len_kv=lens_kv, **kw
    )
    return o, lse, res


def _close(a, b, tol=1e-10):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    assert np.isfinite(a).all() and np.isfinite(b).all()
    scale = max(1.0, float(np.abs(b).max(initial=0.0)))
    err = float(np.abs(a - b).max(initial=0.0))
    assert err <= tol * scale, (err, scale)


# ---------------------------------------------------------------------------
# torch autograd (float64, CPU)
# ---------------------------------------------------------------------------


def _torch_grads(q, k, v, do, *, scale, left, right, tl, lens_q=None, lens_kv=None):
    """Autograd of ``sum(O * dO)`` through an explicit masked-softmax graph."""
    torch = pytest.importorskip("torch")
    tq, tk, tv = (
        torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in (q, k, v)
    )
    tdo = torch.tensor(do, dtype=torch.float64)
    B, Hq, Sq, _ = q.shape
    Hk, Skv, Hv = k.shape[1], k.shape[2], v.shape[1]
    loss = tq.new_zeros(())
    for b in range(B):
        sq = Sq if lens_q is None else lens_q[b]
        skv = Skv if lens_kv is None else lens_kv[b]
        if sq == 0:
            continue
        mask = torch.tensor(_band(sq, skv, left, right, tl))
        for h in range(Hq):
            qa = tq[b, h, :sq]
            ka = tk[b, h // (Hq // Hk), :skv]
            va = tv[b, h // (Hq // Hv), :skv]
            s = (qa @ ka.T) * scale
            sm = s.masked_fill(~mask, -math.inf)
            m = sm.max(dim=1, keepdim=True).values if skv else s.new_zeros((sq, 1))
            m = torch.where(torch.isfinite(m), m, torch.zeros_like(m)).detach()
            e = torch.where(
                mask, torch.exp(torch.where(mask, s - m, torch.zeros_like(s))), 0.0
            )
            den = e.sum(dim=1, keepdim=True)
            p = e / torch.where(den > 0, den, torch.ones_like(den))
            loss = loss + ((p @ va) * tdo[b, h, :sq]).sum()
    loss.backward()
    z = lambda t: np.zeros(t.shape) if t.grad is None else t.grad.numpy()  # noqa: E731
    return z(tq), z(tk), z(tv)


@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("heads", [(4, 4), (4, 2), (4, 1)], ids=["mha", "gqa2", "mqa"])
@pytest.mark.parametrize("mask", MASKS, ids=[m[0] for m in MASKS])
def test_against_torch_autograd_dense(d, heads, mask):
    _, left, right, tl = mask
    Hq, Hk = heads
    rng = np.random.default_rng(d * 31 + Hk * 7 + left * 3 + right + 10)
    for sq, skv in ((13, 13), (9, 21), (21, 9)):
        q, k, v, do = _problem(rng, 2, Hq, Hk, sq, skv, d)
        scale = 1.0 / math.sqrt(d) * 1.3
        _, _, res = _run(q, k, v, do, scale=scale, left=left, right=right, tl=tl)
        tq, tk, tv = _torch_grads(
            q, k, v, do, scale=scale, left=left, right=right, tl=tl
        )
        _close(res.dq, tq)
        _close(res.dk, tk)
        _close(res.dv, tv)


def test_against_torch_autograd_window_left_off_by_one():
    """``left = W`` keeps W + 1 keys; W - 1 / W + 1 must differ from W."""
    rng = np.random.default_rng(11)
    q, k, v, do = _problem(rng, 1, 2, 1, 24, 24, 32)
    grads = {}
    for w in (6, 7, 8):
        _, _, res = _run(q, k, v, do, left=w, right=0)
        tq, tk, tv = _torch_grads(
            q, k, v, do, scale=1 / math.sqrt(32), left=w, right=0, tl=True
        )
        _close(res.dq, tq)
        _close(res.dk, tk)
        _close(res.dv, tv)
        grads[w] = res
        # with dO nonzero on a single row r only, dV is nonzero exactly on the
        # W + 1 keys r - W .. r
        r = 15
        one = np.zeros_like(do)
        one[:, :, r] = do[:, :, r]
        _, _, single = _run(q, k, v, one, left=w, right=0)
        live = np.flatnonzero(np.abs(single.dv[0, 0]).sum(axis=1) > 0)
        assert list(live) == list(range(r - w, r + 1))
    assert not np.allclose(grads[6].dq, grads[7].dq)
    assert not np.allclose(grads[7].dk, grads[8].dk)


@pytest.mark.parametrize("mask", MASKS, ids=[m[0] for m in MASKS])
def test_against_torch_autograd_padded(mask):
    _, left, right, tl = mask
    rng = np.random.default_rng(50 + left + 10 * right)
    lens_q, lens_kv = [11, 4, 0, 17], [17, 9, 5, 3]
    q, k, v, do = _problem(rng, 4, 4, 2, 17, 17, 64)
    _, _, res = _run(
        q, k, v, do, left=left, right=right, tl=tl, lens_q=lens_q, lens_kv=lens_kv
    )
    tq, tk, tv = _torch_grads(
        q,
        k,
        v,
        do,
        scale=1 / 8.0,
        left=left,
        right=right,
        tl=tl,
        lens_q=lens_q,
        lens_kv=lens_kv,
    )
    _close(res.dq, tq)
    _close(res.dk, tk)
    _close(res.dv, tv)


def test_against_torch_sdpa_function():
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    rng = np.random.default_rng(3)
    for Hq, Hk, causal in ((4, 4, False), (4, 4, True), (8, 2, True), (8, 1, False)):
        q, k, v, do = _problem(rng, 2, Hq, Hk, 19, 19, 64)
        tq, tk, tv = (torch.tensor(a, requires_grad=True) for a in (q, k, v))
        out = F.scaled_dot_product_attention(
            tq, tk, tv, is_causal=causal, scale=0.2, enable_gqa=Hq != Hk
        )
        out.backward(torch.tensor(do))
        res = sdpa_reference_bwd(
            q,
            k,
            v,
            out.detach().numpy(),
            do,
            scale=0.2,
            causal=causal,
            recompute_lse=True,
        )
        _close(res.dq, tq.grad.numpy())
        _close(res.dk, tk.grad.numpy())
        _close(res.dv, tv.grad.numpy())


@pytest.mark.parametrize("mask", MASKS, ids=[m[0] for m in MASKS])
def test_ragged_matches_per_sequence_dense_and_torch(mask):
    _, left, right, tl = mask
    rng = np.random.default_rng(170 + left)
    Hq, Hk, D = 4, 2, 32
    lq, lk = [5, 1, 9, 3], [7, 4, 2, 9]
    qo = np.concatenate([[0], np.cumsum(lq)])
    ko = np.concatenate([[0], np.cumsum(lk)])
    q = _rand(rng, (qo[-1] + 2, Hq, D))  # two trailing rows outside every segment
    k = _rand(rng, (ko[-1], Hk, D))
    v = _rand(rng, (ko[-1], Hk, D))
    do = _rand(rng, (qo[-1] + 2, Hq, D))
    kw = dict(
        left_bound=left, right_bound=right, top_left=tl, q_offsets=qo, kv_offsets=ko
    )
    o, lse = sdpa_reference(q, k, v, **kw)
    res = sdpa_reference_bwd(q, k, v, o, do, lse, **kw)
    assert (
        res.dq.shape == q.shape
        and res.dk.shape == k.shape
        and res.delta.shape == lse.shape
    )
    assert np.all(res.dq[qo[-1] :] == 0) and np.all(res.delta[qo[-1] :] == 0)
    for b in range(len(lq)):
        qs, ks = slice(qo[b], qo[b + 1]), slice(ko[b], ko[b + 1])
        t = lambda a: a.transpose(1, 0, 2)[None]  # noqa: E731
        _, _, ref = _run(
            t(q[qs]), t(k[ks]), t(v[ks]), t(do[qs]), left=left, right=right, tl=tl
        )
        _close(res.dq[qs], ref.dq[0].transpose(1, 0, 2), 1e-12)
        _close(res.dk[ks], ref.dk[0].transpose(1, 0, 2), 1e-12)
        _close(res.dv[ks], ref.dv[0].transpose(1, 0, 2), 1e-12)
        _close(res.delta[qs], ref.delta[0].T, 1e-12)
        if b == 2 and _torch_available():
            tq, tk, tv = _torch_grads(
                t(q[qs]),
                t(k[ks]),
                t(v[ks]),
                t(do[qs]),
                scale=1 / math.sqrt(D),
                left=left,
                right=right,
                tl=tl,
            )
            _close(res.dq[qs], tq[0].transpose(1, 0, 2))
            _close(res.dk[ks], tk[0].transpose(1, 0, 2))


# ---------------------------------------------------------------------------
# finite differences (torch-free)
# ---------------------------------------------------------------------------


def _fd_grads(q, k, v, do, kw, eps=1e-6):
    def f(qq, kk, vv):
        o, _ = sdpa_reference(qq, kk, vv, **kw)
        return float((o * do).sum())

    out = []
    for which in range(3):
        args = [q.copy(), k.copy(), v.copy()]
        g = np.zeros(args[which].shape)
        it = np.nditer(g, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            plus = [a.copy() for a in args]
            minus = [a.copy() for a in args]
            plus[which][idx] += eps
            minus[which][idx] -= eps
            g[idx] = (f(*plus) - f(*minus)) / (2 * eps)
        out.append(g)
    return out


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(causal=True),
        dict(causal_bottom_right=True),
        dict(left_bound=1, right_bound=0, top_left=False),
        dict(left_bound=0, right_bound=1),
        dict(seq_len_q=[3, 1], seq_len_kv=[2, 4], causal_bottom_right=True),
        dict(scale=0.7, left_bound=2, right_bound=-1),
    ],
    ids=["none", "tl", "br", "win_br", "right_tl", "padded_br", "scale_left_only"],
)
def test_finite_differences(kw):
    rng = np.random.default_rng(99)
    q, k, v, do = _problem(rng, 2, 4, 2, 3, 4, 4)
    o, lse = sdpa_reference(q, k, v, **kw)
    res = sdpa_reference_bwd(q, k, v, o, do, lse, **kw)
    fq, fk, fv = _fd_grads(q, k, v, do, kw)
    for a, b in ((res.dq, fq), (res.dk, fk), (res.dv, fv)):
        np.testing.assert_allclose(a, b, atol=1e-7, rtol=1e-6)


def test_finite_differences_ragged_mqa():
    rng = np.random.default_rng(5)
    qo, ko = [0, 2, 5], [0, 3, 4]
    q = _rand(rng, (5, 2, 4))
    k = _rand(rng, (4, 1, 4))
    v = _rand(rng, (4, 1, 4))
    do = _rand(rng, (5, 2, 4))
    kw = dict(q_offsets=qo, kv_offsets=ko, causal_bottom_right=True)
    res = sdpa_reference_bwd(q, k, v, None, do, recompute_lse=True, **kw)
    fq, fk, fv = _fd_grads(q, k, v, do, kw)
    for a, b in ((res.dq, fq), (res.dk, fk), (res.dv, fv)):
        np.testing.assert_allclose(a, b, atol=1e-7, rtol=1e-6)


# ---------------------------------------------------------------------------
# independent scalar loop
# ---------------------------------------------------------------------------


def _scalar_bwd(q, k, v, do, lse, *, scale, left, right, tl, lens_q, lens_kv):
    """Per-element loops; delta from ``sum_j P_ij * dP_ij`` (no use of O)."""
    B, Hq, Sq, D = q.shape
    Hk, Hv, Dv = k.shape[1], v.shape[1], v.shape[3]
    dq = np.zeros(q.shape)
    dk = np.zeros(k.shape)
    dv = np.zeros(v.shape)
    for b in range(B):
        nq, nk = lens_q[b], lens_kv[b]
        off = 0 if tl else nk - nq
        for h in range(Hq):
            hk = h * Hk // Hq
            hv = h * Hv // Hq
            for i in range(nq):
                li = float(lse[b, h, i])
                if li == -math.inf:
                    continue
                cols = [
                    j
                    for j in range(nk)
                    if not (right >= 0 and j > i + off + right)
                    and not (left >= 0 and j < i + off - left)
                ]
                p, dp = {}, {}
                for j in cols:
                    dot = sum(
                        float(q[b, h, i, t]) * float(k[b, hk, j, t]) for t in range(D)
                    )
                    p[j] = math.exp(dot * scale - li)
                    dp[j] = sum(
                        float(do[b, h, i, t]) * float(v[b, hv, j, t]) for t in range(Dv)
                    )
                di = sum(p[j] * dp[j] for j in cols)
                for j in cols:
                    ds = p[j] * (dp[j] - di)
                    for t in range(Dv):
                        dv[b, hv, j, t] += p[j] * float(do[b, h, i, t])
                    for t in range(D):
                        dq[b, h, i, t] += scale * ds * float(k[b, hk, j, t])
                        dk[b, hk, j, t] += scale * ds * float(q[b, h, i, t])
    return dq, dk, dv


def test_random_sweep_against_scalar_loop():
    rng = np.random.default_rng(2024)
    for _ in range(40):
        B = int(rng.integers(1, 3))
        Hk = int(rng.choice([1, 2]))
        Hv = int(rng.choice([1, 2]))
        Hq = math.lcm(Hk, Hv) * int(rng.choice([1, 2]))
        Sq, Skv = int(rng.integers(1, 8)), int(rng.integers(1, 8))
        D = int(rng.choice([2, 4, 8]))
        left = int(rng.choice([-1, 0, 1, 3]))
        right = int(rng.choice([-1, 0, 2]))
        tl = bool(rng.integers(0, 2))
        scale = float(rng.choice([0.3, 1.0]))
        lq = [int(rng.integers(0, Sq + 1)) for _ in range(B)]
        lk = [int(rng.integers(0, Skv + 1)) for _ in range(B)]
        q, k, v, do = _problem(rng, B, Hq, Hk, Sq, Skv, D, Hv=Hv)
        _, lse, res = _run(
            q,
            k,
            v,
            do,
            scale=scale,
            left=left,
            right=right,
            tl=tl,
            lens_q=lq,
            lens_kv=lk,
        )
        sq_, sk_, sv_ = _scalar_bwd(
            q,
            k,
            v,
            do,
            lse,
            scale=scale,
            left=left,
            right=right,
            tl=tl,
            lens_q=lq,
            lens_kv=lk,
        )
        _close(res.dq, sq_, 1e-12)
        _close(res.dk, sk_, 1e-12)
        _close(res.dv, sv_, 1e-12)


def test_gqa_dk_dv_sum_over_query_group():
    rng = np.random.default_rng(8)
    Hq, Hk = 8, 2
    q, k, v, do = _problem(rng, 2, Hq, Hk, 7, 9, 8)
    _, _, res = _run(q, k, v, do, right=0, tl=False)
    g = Hq // Hk
    _, _, mha = _run(
        q, np.repeat(k, g, axis=1), np.repeat(v, g, axis=1), do, right=0, tl=False
    )
    _close(res.dq, mha.dq, 1e-12)
    _close(res.dk, mha.dk.reshape(2, Hk, g, 9, 8).sum(axis=2), 1e-12)
    _close(res.dv, mha.dv.reshape(2, Hk, g, 9, 8).sum(axis=2), 1e-12)


# ---------------------------------------------------------------------------
# fully masked rows, padding, stats handling, errors
# ---------------------------------------------------------------------------


def test_fully_masked_rows_get_zero_dq_and_no_nan():
    rng = np.random.default_rng(1)
    q, k, v, do = _problem(rng, 1, 4, 2, 9, 3, 16)
    for kw in (
        dict(causal_bottom_right=True),
        dict(left_bound=0, right_bound=0, top_left=False),
    ):
        o, lse = sdpa_reference(q, k, v, **kw)
        dead = np.isneginf(lse)
        assert dead.any()
        # LSE stored as fp32 (the kernel-facing dtype) keeps -inf
        res = sdpa_reference_bwd(q, k, v, o, do, lse.astype(np.float32), **kw)
        for a in res:
            assert np.isfinite(a).all()
        assert np.all(res.dq[dead] == 0)
        assert np.all(res.delta[dead] == 0)
    # bottom-right with s_q=9, s_kv=3: rows 0..5 see nothing
    o, lse = sdpa_reference(q, k, v, causal_bottom_right=True)
    assert np.isneginf(lse[0, :, :6]).all() and np.isfinite(lse[0, :, 6:]).all()


def test_padded_rows_and_keys():
    rng = np.random.default_rng(4)
    q, k, v, do = _problem(rng, 3, 4, 2, 8, 10, 8)
    lq, lk = [8, 3, 0], [10, 6, 4]
    o, lse = sdpa_reference(q, k, v, seq_len_q=lq, seq_len_kv=lk, causal=True)
    # garbage outside the valid region must not matter (incl. NaN in padded stats)
    q2, k2, v2, do2, o2, lse2 = (
        q.copy(),
        k.copy(),
        v.copy(),
        do.copy(),
        o.copy(),
        lse.copy(),
    )
    q2[1, :, 3:] = 1e3
    do2[1, :, 3:] = -7.0
    o2[1, :, 3:] = 5.0
    lse2[1, :, 3:] = np.nan
    lse2[2] = 3.0
    k2[1, :, 6:] = 1e3
    v2[2, :, 4:] = np.nan
    kw = dict(seq_len_q=lq, seq_len_kv=lk, causal=True)
    a = sdpa_reference_bwd(q, k, v, o, do, lse, **kw)
    b = sdpa_reference_bwd(q2, k2, v2, o2, do2, lse2, **kw)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    assert np.all(a.dq[1, :, 3:] == 0) and np.all(a.dq[2] == 0)
    assert np.all(a.dk[1, :, 6:] == 0) and np.all(a.dv[1, :, 6:] == 0)
    assert np.all(a.dk[2] == 0) and np.all(a.dv[2] == 0)  # batch 2 has no query rows
    assert np.all(a.delta[1, :, 3:] == 0)


def test_delta_is_rowsum_do_times_o_and_matches_forward_identity():
    rng = np.random.default_rng(6)
    q, k, v, do = _problem(rng, 1, 2, 2, 5, 7, 8)
    o, lse, res = _run(q, k, v, do, right=1)
    np.testing.assert_allclose(res.delta, (do * o).sum(-1), atol=1e-14)


def test_lse_and_stats_forms():
    rng = np.random.default_rng(12)
    q, k, v, do = _problem(rng, 2, 4, 1, 6, 6, 8)
    o, lse = sdpa_reference(q, k, v, causal=True)
    base = sdpa_reference_bwd(q, k, v, o, do, lse, causal=True)
    rank4 = sdpa_reference_bwd(q, k, v, o, do, lse[..., None], causal=True)
    rec = sdpa_reference_bwd(q, k, v, o, do, causal=True, recompute_lse=True)
    rec_o = sdpa_reference_bwd(q, k, v, None, do, causal=True, recompute_lse=True)
    for other in (rank4, rec, rec_o):
        for x, y in zip(base, other):
            np.testing.assert_array_equal(x, y)
    assert isinstance(base, SdpaBwdResult) and len(base) == 4
    dq, dk, dv, delta = base
    assert dq is base.dq and dk is base.dk and dv is base.dv and delta is base.delta


def test_consistency_with_fp32_stats_and_low_precision_output():
    """Stats stored as fp32 and O stored in a low precision stay consistent."""
    rng = np.random.default_rng(13)
    q, k, v, do = _problem(rng, 1, 4, 2, 33, 47, 64)
    o, lse = sdpa_reference(q, k, v, causal_bottom_right=True)
    ref = sdpa_reference_bwd(q, k, v, o, do, lse, causal_bottom_right=True)
    lo = sdpa_reference_bwd(
        q,
        k,
        v,
        o.astype(np.float16),
        do,
        lse.astype(np.float32),
        causal_bottom_right=True,
    )
    for x, y in zip(ref, lo):
        assert np.abs(x - y).max() <= 2e-3 * max(1.0, np.abs(x).max())


def test_argument_errors():
    rng = np.random.default_rng(0)
    q, k, v, do = _problem(rng, 1, 2, 2, 3, 4, 4)
    o, lse = sdpa_reference(q, k, v)
    with pytest.raises(ValueError, match="bias"):
        sdpa_reference_bwd(q, k, v, o, do, lse, bias=np.zeros((3, 4)))
    with pytest.raises(ValueError):
        sdpa_reference_bwd(q, k, v, o, do)  # no lse, no recompute
    with pytest.raises(ValueError):
        sdpa_reference_bwd(q, k, v, o, do, lse, recompute_lse=True)
    with pytest.raises(ValueError):
        sdpa_reference_bwd(q, k, v, None, do, lse)
    with pytest.raises(ValueError):
        sdpa_reference_bwd(q, k, v, o, do[..., :2], lse)
    with pytest.raises(ValueError):
        sdpa_reference_bwd(q, k, v, o, do, lse[..., :2])
    with pytest.raises(ValueError):
        sdpa_reference_bwd(q, k, v, o, do, np.full_like(lse, np.nan))
    with pytest.raises(ValueError, match="multiple"):
        q3h = np.repeat(q, 2, axis=1)[:, :3]  # Hq = 3 is not a multiple of Hk = 2
        sdpa_reference_bwd(q3h, k, v, o, do, lse)
    with pytest.raises(ValueError):
        sdpa_reference_bwd(q, k, v, o, do, lse, causal=True, causal_bottom_right=True)
    q3 = np.zeros((3, 2, 4))
    with pytest.raises(ValueError):
        sdpa_reference_bwd(q3, q3, q3, q3, q3, np.zeros((3, 2)), q_offsets=[0, 3])
    with pytest.raises(ValueError):
        sdpa_reference_bwd(
            q3, q3, q3, q3, q3, np.zeros((3, 2)), q_offsets=[0, 4], kv_offsets=[0, 3]
        )


def test_reexported_from_references():
    assert references.sdpa_reference_bwd is sdpa_reference_bwd
    assert references.SdpaBwdResult is SdpaBwdResult
    assert "sdpa_reference_bwd" in references.__all__
    assert resolve_diagonal_band(causal=True) == (-1, 0, True)
