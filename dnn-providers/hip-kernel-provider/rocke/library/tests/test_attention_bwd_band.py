# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Inverse band and tile classifier of the attention backward.

* Exhaustive inverse-band check (superset, tightness, empty band, keep
  agreement with ``rocke.numeric.sdpa_reference.keep``), evaluated with the
  typed integer builder so the code under test is the emitted formula.
* Exhaustive tile-classifier check against brute force over small grids,
  including ``s_q > s_kv``, windows, padded lengths and THD sequences.
* IR lowering of a probe kernel at three LLVM flavors and four archs, and the
  same probe on the local device (compared with the host evaluation).
"""

from __future__ import annotations

import itertools
import re

import numpy as np
import pytest
from kernels.common import _attention_bwd_band as BAND
from kernels.common._attention_bwd_addr import decode_batched, decode_thd
from rocke.core.ir import I32, IRBuilder, PtrType
from rocke.helpers.attention_band import AttnRuntimeBounds, BandEmitter
from rocke.numeric.sdpa_reference import keep as ref_keep

from ._attention_bwd_test_utils import device_arch
from ._attention_bwd_int_eval import IntEvalBuilder

ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
FLAVORS = ("llvm20", "llvm22", "llvm23")
BOUNDS = (-1, 0, 1, 3, 7)


def _mask(s_q, s_kv, bottom_right, left, right, len_q=None, len_kv=None):
    """Vectorised keep over a padded [s_q, s_kv] grid with valid lengths."""
    lq = s_q if len_q is None else len_q
    lk = s_kv if len_kv is None else len_kv
    qi = np.arange(s_q)[:, None]
    ki = np.arange(s_kv)[None, :]
    off = lk - lq if bottom_right else 0
    m = (qi < lq) & (ki < lk)
    if right >= 0:
        m &= ki <= qi + off + right
    if left >= 0:
        m &= ki >= qi + off - left
    return m


def test_vectorised_mask_matches_reference_keep():
    for s_q, s_kv in itertools.product((1, 3, 9, 17), (1, 4, 16, 21)):
        for br, left, right in itertools.product((False, True), BOUNDS, BOUNDS):
            m = _mask(s_q, s_kv, br, left, right)
            diag = "bottom_right" if br else "top_left"
            for q in range(s_q):
                for k in range(s_kv):
                    assert m[q, k] == ref_keep(q, k, s_q, s_kv, diag, left, right)


def _band(b, len_q, len_kv, off, left, right, runtime):
    """BandEmitter with runtime-like (typed) or static (int) bound spellings."""
    if runtime:
        bounds = AttnRuntimeBounds(
            b.const_i32(len_q),
            b.const_i32(len_kv),
            b.const_i32(off),
            b.const_i32(left),
            b.const_i32(right),
        )
    else:
        bounds = AttnRuntimeBounds(
            b.const_i32(len_q), b.const_i32(len_kv), off, left, right
        )
    return BandEmitter(b, bounds)


def _range(b, band, k0, k_n0, k_m0):
    qs, n = BAND.q_tile_range(b, band, b.const_i32(k0), k_n0=k_n0, k_m0=k_m0)
    return qs.v, n.v


def _check_ranges(b, s_q, s_kv, br, left, right, k_n0, k_m0, runtime, m=None):
    if m is None:
        m = _mask(s_q, s_kv, br, left, right)
    off = s_kv - s_q if br else 0
    band = _band(b, s_q, s_kv, off, left, right, runtime)
    n_kt = (s_kv + k_n0 - 1) // k_n0 + 1  # one tile past the end too
    for kt in range(n_kt):
        k0 = kt * k_n0
        qs, n = _range(b, band, k0, k_n0, k_m0)
        rows = np.nonzero(m[:, k0 : k0 + k_n0].any(axis=1))[0]
        if rows.size == 0:
            assert n == 0, (s_q, s_kv, br, left, right, k0, qs, n)
            continue
        # superset and tightness: exactly the tiles of the first and last row
        assert qs == rows[0] // k_m0, (s_q, s_kv, br, left, right, k0)
        assert qs + n - 1 == rows[-1] // k_m0, (s_q, s_kv, br, left, right, k0)
        # the rows inside the range are contiguous (no hole inside the band)
        assert rows[-1] - rows[0] + 1 == rows.size


def test_inverse_band_exhaustive():
    """s_q, s_kv in 1..40, bounds in {-1, 0, 1, 3, 7}, both alignments, 16x16."""
    b = IntEvalBuilder()
    n = 0
    for s_q in range(1, 41):
        for s_kv in range(1, 41):
            for br, left, right in itertools.product((False, True), BOUNDS, BOUNDS):
                _check_ranges(b, s_q, s_kv, br, left, right, 16, 16, runtime=True)
                n += 1
    assert n == 40 * 40 * 2 * 25


@pytest.mark.parametrize("k_m0,k_n0", [(16, 64), (32, 32), (64, 16), (64, 128)])
def test_inverse_band_other_tiles_and_static_spelling(k_m0, k_n0):
    b = IntEvalBuilder()
    sizes = (1, 2, 15, 16, 17, 31, 33, 64, 65, 97, 130)
    for s_q, s_kv in itertools.product(sizes, sizes):
        for br, left, right in itertools.product((False, True), BOUNDS, BOUNDS):
            for runtime in (False, True):
                _check_ranges(b, s_q, s_kv, br, left, right, k_n0, k_m0, runtime)


@pytest.mark.parametrize("runtime", [False, True])
def test_cell_keep_agrees_with_reference_keep(runtime):
    b = IntEvalBuilder()
    for s_q, s_kv in itertools.product((1, 5, 16, 23, 40), (1, 7, 16, 33, 40)):
        for br, left, right in itertools.product((False, True), BOUNDS, BOUNDS):
            off = s_kv - s_q if br else 0
            band = _band(b, s_q, s_kv, off, left, right, runtime)
            pol = BAND.MaskPolicy(b, band, k_m0=16, k_n0=16)
            diag = "bottom_right" if br else "top_left"
            for q in range(s_q + 2):
                row = pol.row(b, b.const_i32(q))
                for k in range(s_kv + 2):
                    got = pol.keep(b, row, b.const_i32(k)).v
                    want = (
                        q < s_q
                        and k < s_kv
                        and ref_keep(q, k, s_q, s_kv, diag, left, right)
                    )
                    assert got == want, (s_q, s_kv, br, left, right, q, k)


def _policy(b, s_q, s_kv, br, left, right, runtime, k_m0=16, k_n0=16):
    """RuntimeBand (runtime or static bound spellings) or, unmasked, NoMaskKTail."""
    if left == -1 and right == -1 and not runtime:
        return BAND.NoMaskKTail(
            b, b.const_i32(s_q), b.const_i32(s_kv), k_m0=k_m0, k_n0=k_n0
        )
    off = s_kv - s_q if br else 0
    if runtime:
        bounds = AttnRuntimeBounds(
            b.const_i32(s_q), b.const_i32(s_kv), b.const_i32(off),
            b.const_i32(left), b.const_i32(right),
        )  # fmt: skip
    else:
        bounds = AttnRuntimeBounds(
            b.const_i32(s_q), b.const_i32(s_kv), off, left, right
        )
    return BAND.RuntimeBand(b, bounds, k_m0=k_m0, k_n0=k_n0)


@pytest.mark.parametrize("runtime", [False, True])
def test_row_limits_and_cell_keep_agree_with_reference_keep(runtime):
    """The one-compare keep (per-row limits) and the per-CTA offset form of
    RuntimeBand equal the reference keep, including dead rows (``ok``)."""
    b = IntEvalBuilder()
    for s_q, s_kv in itertools.product((0, 1, 5, 16, 23, 40), (0, 1, 7, 16, 33, 40)):
        for br, left, right in itertools.product((False, True), BOUNDS, BOUNDS):
            pol = _policy(b, s_q, s_kv, br, left, right, runtime)
            diag = "bottom_right" if br else "top_left"
            for q in range(s_q + 2):
                qv = b.const_i32(q)
                lim = pol.row_limits(b, qv)
                dead = pol.row_limits(b, qv, b.cmp_lt(qv, b.const_i32(0)))
                for k in range(s_kv + 2):
                    kv = b.const_i32(k)
                    want = (
                        q < s_q
                        and k < s_kv
                        and ref_keep(q, k, s_q, s_kv, diag, left, right)
                    )
                    case = (s_q, s_kv, br, left, right, q, k)
                    assert pol.keep_in(b, lim, kv).v == want, case
                    assert pol.cell_keep(b, qv, kv).v == want, case
                    assert not pol.keep_in(b, dead, kv).v, case


def test_base_policy_has_no_row_limits_for_a_band():
    b = IntEvalBuilder()
    band = _band(b, 16, 16, 0, 3, 0, True)
    pol = BAND.MaskPolicy(b, band, k_m0=16, k_n0=16)
    with pytest.raises(NotImplementedError):
        pol.row_limits(b, b.const_i32(0))


@pytest.mark.parametrize("k_m0,k_n0", [(16, 16), (16, 64), (32, 32), (64, 16)])
@pytest.mark.parametrize("runtime", [False, True])
def test_interior_q_range_is_the_interior_tile_set(k_m0, k_n0, runtime):
    """``[i0, i1)`` is exactly the set of interior tiles of the visited range."""
    b = IntEvalBuilder()
    sizes = (0, 1, 8, 16, 31, 48, 70)
    n_interior = 0
    for s_q, s_kv in itertools.product(sizes, sizes):
        for br, left, right in itertools.product(
            (False, True), (-1, 0, 5, 40), (-1, 0, 5, 40)
        ):
            pol = _policy(b, s_q, s_kv, br, left, right, runtime, k_m0, k_n0)
            for kt in range((s_kv + k_n0 - 1) // k_n0 + 1):
                k0 = b.const_i32(kt * k_n0)
                qs, n = pol.q_tile_range(b, k0)
                i0, i1 = pol.interior_q_range(b, k0, qs, n)
                interior = {
                    qt
                    for qt in range(qs.v, qs.v + n.v)
                    if pol.tile_class(b, b.const_i32(qt), k0).v == BAND.TILE_INTERIOR
                }
                case = (s_q, s_kv, br, left, right, kt)
                assert qs.v <= i0.v <= i1.v <= qs.v + n.v, case
                assert set(range(i0.v, i1.v)) == interior, case
                n_interior += len(interior)
    assert n_interior > 0


def test_empty_problems_give_no_tiles():
    b = IntEvalBuilder()
    for len_q, len_kv in ((0, 0), (0, 20), (20, 0)):
        for left, right in itertools.product(BOUNDS, BOUNDS):
            band = _band(b, len_q, len_kv, 0, left, right, True)
            for k0 in (0, 16, 32):
                assert _range(b, band, k0, 16, 16)[1] == 0


# ---------------------------------------------------------------------------
# Tile classifier
# ---------------------------------------------------------------------------


def _classify_all(b, policy, s_q_max, s_kv_max, k_m0, k_n0, m, len_q, len_kv):
    """Check every (q tile, kv tile) of a padded grid against brute force."""
    n_qt_all = (s_q_max + k_m0 - 1) // k_m0 + 1
    n_kt_all = (s_kv_max + k_n0 - 1) // k_n0 + 1
    gm = np.zeros(((n_qt_all + 1) * k_m0, (n_kt_all + 1) * k_n0), dtype=bool)
    gm[: m.shape[0], : m.shape[1]] = m
    n_interior = n_edge = 0
    for kt in range(n_kt_all):
        k0 = kt * k_n0
        qs, n = policy.q_tile_range(b, b.const_i32(k0))
        visited = set(range(qs.v, qs.v + n.v))
        for qt in range(n_qt_all):
            q0 = qt * k_m0
            cells = gm[q0 : q0 + k_m0, k0 : k0 + k_n0]
            cls = policy.tile_class(b, b.const_i32(qt), b.const_i32(k0)).v
            full = cells.all() and q0 + k_m0 <= len_q and k0 + k_n0 <= len_kv
            assert (cls == BAND.TILE_INTERIOR) == bool(full), (qt, kt, len_q, len_kv)
            if cells.any():
                assert qt in visited, (qt, kt)
            if qt in visited:
                n_interior += cls == BAND.TILE_INTERIOR
                n_edge += cls == BAND.TILE_EDGE
    return n_interior, n_edge


@pytest.mark.parametrize("k_m0,k_n0", [(16, 16), (16, 64), (32, 32), (64, 16)])
def test_tile_classifier_exhaustive_masked(k_m0, k_n0):
    b = IntEvalBuilder()
    sizes = (1, 8, 16, 31, 48, 70)
    tot_int = tot_edge = 0
    for s_q, s_kv in itertools.product(sizes, sizes):
        for br, left, right in itertools.product(
            (False, True), (-1, 0, 5, 40), (-1, 0, 5, 40)
        ):
            m = _mask(s_q, s_kv, br, left, right)
            off = s_kv - s_q if br else 0
            pol = BAND.RuntimeBand(
                b,
                AttnRuntimeBounds(
                    b.const_i32(s_q),
                    b.const_i32(s_kv),
                    b.const_i32(off),
                    b.const_i32(left),
                    b.const_i32(right),
                ),
                k_m0=k_m0,
                k_n0=k_n0,
            )
            ni, ne = _classify_all(b, pol, s_q, s_kv, k_m0, k_n0, m, s_q, s_kv)
            tot_int += ni
            tot_edge += ne
    assert tot_int > 0 and tot_edge > 0


@pytest.mark.parametrize("k_m0,k_n0", [(16, 16), (32, 64)])
def test_tile_classifier_unmasked_tails(k_m0, k_n0):
    b = IntEvalBuilder()
    for len_q, len_kv in itertools.product((1, 16, 33, 64, 100), (1, 16, 47, 64, 128)):
        m = _mask(len_q, len_kv, False, -1, -1)
        pol = BAND.NoMaskKTail(
            b, b.const_i32(len_q), b.const_i32(len_kv), k_m0=k_m0, k_n0=k_n0
        )
        _classify_all(b, pol, len_q, len_kv, k_m0, k_n0, m, len_q, len_kv)


def test_tile_classifier_padded_lengths():
    """Per-batch lengths below the tensor extents (SEQ tensors), incl. 0."""
    b = IntEvalBuilder()
    s_q_max, s_kv_max = 70, 90
    seq_q = np.array([70, 33, 0, 1, 64, 200], np.int32)  # 200 clamps to S_max
    seq_kv = np.array([90, 17, 5, 0, 64, 7], np.int32)
    sq_ptr = b.add_buffer("SEQ_Q", seq_q)
    skv_ptr = b.add_buffer("SEQ_KV", seq_kv)
    for bi in range(len(seq_q)):
        bq = decode_batched(
            b, b.const_i32(bi), s_max=s_q_max, seq_ptr=sq_ptr,
            has_len=b.const_i32(1), len_stride=1,
        )  # fmt: skip
        bk = decode_batched(
            b, b.const_i32(bi), s_max=s_kv_max, seq_ptr=skv_ptr,
            has_len=b.const_i32(1), len_stride=1,
        )  # fmt: skip
        lq, lk = bq.length.v, bk.length.v
        assert lq == min(int(seq_q[bi]), s_q_max) and lk == min(
            int(seq_kv[bi]), s_kv_max
        )
        for br, left, right in itertools.product((False, True), (-1, 3), (-1, 0, 6)):
            m = _mask(s_q_max, s_kv_max, br, left, right, lq, lk)
            off = b.sub(bk.length, bq.length) if br else b.const_i32(0)
            pol = BAND.RuntimeBand(
                b,
                AttnRuntimeBounds(
                    bq.length, bk.length, off, b.const_i32(left), b.const_i32(right)
                ),
                k_m0=16,
                k_n0=32,
            )
            _classify_all(b, pol, s_q_max, s_kv_max, 16, 32, m, lq, lk)


@pytest.mark.parametrize("off64", [0, 1])
def test_tile_classifier_thd_sequences(off64):
    """THD lengths decoded from offsets (int32 / int64, element units)."""
    b = IntEvalBuilder()
    lens_q = [37, 0, 1, 64, 5]
    lens_kv = [50, 9, 0, 70, 5]
    mult = 1
    div = 8  # element offsets of a tensor with token stride 8

    def table(lens):
        offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64) * div
        return offs if off64 else offs.astype(np.int32)

    tq = table(lens_q)
    tk = table(lens_kv)
    if off64:
        tq, tk = tq.view(np.int32), tk.view(np.int32)
    pq = b.add_buffer("OFF_Q", tq.copy())
    pk = b.add_buffer("OFF_KV", tk.copy())
    s_max = 80
    for s in range(len(lens_q)):
        sq = decode_thd(
            b, b.const_i32(s), s_max=s_max, off_ptr=pq, off64=b.const_i32(off64),
            mult=b.const_i32(mult), div=b.const_i32(div),
        )  # fmt: skip
        sk = decode_thd(
            b, b.const_i32(s), s_max=s_max, off_ptr=pk, off64=b.const_i32(off64),
            mult=b.const_i32(mult), div=b.const_i32(div),
        )  # fmt: skip
        assert sq.length.v == lens_q[s] and sk.length.v == lens_kv[s]
        assert sq.tok0.v == sum(lens_q[:s])
        for br, left, right in itertools.product((False, True), (-1, 2), (-1, 0, 4)):
            lq, lk = lens_q[s], lens_kv[s]
            m = _mask(max(lq, 1), max(lk, 1), br, left, right, lq, lk)
            off = b.sub(sk.length, sq.length) if br else b.const_i32(0)
            pol = BAND.RuntimeBand(
                b,
                AttnRuntimeBounds(
                    sq.length, sk.length, off, b.const_i32(left), b.const_i32(right)
                ),
                k_m0=16,
                k_n0=16,
            )
            _classify_all(b, pol, max(lq, 1), max(lk, 1), 16, 16, m, lq, lk)


def test_interior_fast_path_equals_selected_values():
    """On interior tiles the select-free P / dS equal the selected ones."""
    rng = np.random.default_rng(0)
    b = IntEvalBuilder()
    s_q, s_kv, k_m0, k_n0 = 64, 64, 16, 16
    for br, left, right in ((False, -1, 0), (True, 8, 0), (False, 20, 20)):
        m = _mask(s_q, s_kv, br, left, right)
        off = s_kv - s_q if br else 0
        band = _band(b, s_q, s_kv, off, left, right, True)
        lse2 = rng.standard_normal(s_q).astype(np.float32)
        lse2[rng.random(s_q) < 0.2] = np.inf  # dead rows (sentinel)
        dsum = np.where(np.isinf(lse2), 0.0, rng.standard_normal(s_q)).astype(
            np.float32
        )
        for qt, kt in itertools.product(range(s_q // k_m0), range(s_kv // k_n0)):
            edge = BAND.is_edge_tile(
                b, band, b.const_i32(qt), b.const_i32(kt * k_n0), k_m0=k_m0, k_n0=k_n0
            ).v
            if edge:
                continue
            rs = slice(qt * k_m0, (qt + 1) * k_m0)
            cs = slice(kt * k_n0, (kt + 1) * k_n0)
            acc = rng.standard_normal((k_m0, k_n0)).astype(np.float32)
            dp = rng.standard_normal((k_m0, k_n0)).astype(np.float32)
            with np.errstate(invalid="ignore"):
                p_fast = np.exp2(acc * np.float32(0.3) - lse2[rs, None])
                ds_fast = p_fast * (dp - dsum[rs, None])
            keep = m[rs, cs]
            live = keep & ~np.isinf(lse2[rs, None])
            p_sel = np.where(keep, np.exp2(acc * np.float32(0.3) - lse2[rs, None]), 0)
            ds_sel = np.where(live, p_sel * (dp - dsum[rs, None]), 0)
            assert np.array_equal(p_fast, p_sel)
            assert np.array_equal(ds_fast, ds_sel)


def test_dkv_store_rule_and_empty_band_zero_fill():
    b = IntEvalBuilder()
    for s_kv_max, len_kv in ((64, 64), (64, 20), (64, 0)):
        for k in range(s_kv_max + 8):
            st, live = BAND.dkv_store_rule(
                b, b.const_i32(k), b.const_i32(len_kv), seq_mode="batched",
                s_kv_max=b.const_i32(s_kv_max),
            )  # fmt: skip
            assert st.v == (k < s_kv_max) and live.v == (k < len_kv)
            st, live = BAND.dkv_store_rule(
                b, b.const_i32(k), b.const_i32(len_kv), seq_mode="thd"
            )
            assert st.v == live.v == (k < len_kv)
    with pytest.raises(ValueError):
        BAND.dkv_store_rule(b, b.const_i32(0), b.const_i32(1), seq_mode="batched")
    with pytest.raises(ValueError):
        BAND.dkv_store_rule(b, b.const_i32(0), b.const_i32(1), seq_mode="paged")


def test_static_plan_matches_runtime_evaluation():
    b = IntEvalBuilder()
    for s_q, s_kv, br, left, right in (
        (128, 128, False, -1, 0),
        (100, 256, True, -1, 0),
        (256, 100, True, 32, 0),
        (64, 64, False, 16, 16),
        (48, 80, False, -1, -1),
    ):
        off = s_kv - s_q if br else 0
        band = _band(b, s_q, s_kv, off, left, right, True)
        for k0 in range(0, s_kv + 32, 32):
            plan = BAND.static_kv_tile_plan(
                s_q=s_q, s_kv=s_kv, k0=k0, k_m0=16, k_n0=32,
                left=left, right=right, bottom_right=br,
            )  # fmt: skip
            qs, n = _range(b, band, k0, 32, 16)
            assert (plan.qt_start, plan.n_qt) == (qs, n)
            for qt in range(qs, qs + n):
                e = BAND.is_edge_tile(
                    b, band, b.const_i32(qt), b.const_i32(k0), k_m0=16, k_n0=32
                ).v
                assert (qt in plan.edge_tiles) == e
            assert set(plan.interior_tiles) | set(plan.edge_tiles) == set(
                range(qs, qs + n)
            )


def test_policy_rejects_other_tile_sizes():
    b = IntEvalBuilder()
    pol = BAND.NoMaskKTail(b, b.const_i32(10), b.const_i32(10), k_m0=16, k_n0=32)
    with pytest.raises(ValueError):
        pol.q_tile_range(b, b.const_i32(0), 64, 16)
    with pytest.raises(ValueError):
        BAND.q_tile_range(b, pol.band, b.const_i32(0), k_n0=0, k_m0=16)
    with pytest.raises(ValueError):
        BAND.static_kv_tile_plan(s_q=1, s_kv=1, k0=0, k_m0=16, k_n0=16, left=-2)


def test_no_exported_helper_looks_like_a_builder():
    assert not [n for n in BAND.__all__ if n.startswith("build_")]


# ---------------------------------------------------------------------------
# IR probe kernel: lowering at three flavors, and on the local device
# ---------------------------------------------------------------------------

# Outputs per kv tile (one CTA per kv tile, lane 0 stores): qt_start, n_qt,
# then the tile class of q tiles 0 .. PROBE_QT - 1, then the keep bits of the
# probe cells (row PROBE_ROW, keys k0 .. k0 + 3).
PROBE_QT = 8
PROBE_OUT = 2 + PROBE_QT + 4
PROBE_ROW = 5


def _probe_kernel(name, *, k_m0, k_n0, masked):
    b = IRBuilder(name)
    out = b.param("OUT", PtrType(I32, "global"))
    len_q = b.param("len_q", I32)
    len_kv = b.param("len_kv", I32)
    off = b.param("off", I32)
    left = b.param("left", I32)
    right = b.param("right", I32)
    kt = b.block_id_x()
    k0 = b.mul(kt, b.const_i32(k_n0))
    if masked:
        pol = BAND.RuntimeBand(
            b, AttnRuntimeBounds(len_q, len_kv, off, left, right), k_m0=k_m0, k_n0=k_n0
        )
    else:
        pol = BAND.NoMaskKTail(b, len_q, len_kv, k_m0=k_m0, k_n0=k_n0)
    qs, n = pol.q_tile_range(b, k0)
    vals = [qs, n]
    for qt in range(PROBE_QT):
        vals.append(pol.tile_class(b, b.const_i32(qt), k0))
    row = pol.row(b, b.const_i32(PROBE_ROW))
    for j in range(4):
        keep = pol.keep(b, row, b.add(k0, b.const_i32(j)))
        vals.append(b.select(keep, b.const_i32(1), b.const_i32(0)))
    base = b.mul(kt, b.const_i32(PROBE_OUT))
    with b.scf_if(b.cmp_eq(b.thread_id_x(), b.const_i32(0))):
        for i, v in enumerate(vals):
            b.global_store(out, b.add(base, b.const_i32(i)), v, align=4)
    b.ret()
    b.kernel.attrs["max_workgroup_size"] = 64
    return b.kernel


def _lower(kernel, arch, flavor):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

    return lower(kernel, arch=arch, llvm_flavor=flavor)


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("masked", [True, False])
def test_probe_lowers(arch, flavor, masked):
    ir = _lower(
        _probe_kernel(f"bwd_band_probe_{int(masked)}", k_m0=16, k_n0=32, masked=masked),
        arch,
        flavor,
    )
    assert "alloca" not in ir
    # q-loop bounds and the tile class are CTA-uniform (SGPR)
    n_rfl = len(re.findall(r"@llvm\.amdgcn\.readfirstlane\.i32\(", ir))
    assert n_rfl >= 2 + PROBE_QT
    # all band arithmetic is i32 (bounds < 2**30)
    assert "mul nsw i64" not in ir and "add nsw i64" not in ir


def _host_probe(len_q, len_kv, off, left, right, k_m0, k_n0, n_kt, masked):
    b = IntEvalBuilder()
    rows = []
    for kt in range(n_kt):
        k0 = b.const_i32(kt * k_n0)
        if masked:
            pol = BAND.RuntimeBand(
                b,
                AttnRuntimeBounds(
                    b.const_i32(len_q),
                    b.const_i32(len_kv),
                    b.const_i32(off),
                    b.const_i32(left),
                    b.const_i32(right),
                ),
                k_m0=k_m0,
                k_n0=k_n0,
            )
        else:
            pol = BAND.NoMaskKTail(
                b, b.const_i32(len_q), b.const_i32(len_kv), k_m0=k_m0, k_n0=k_n0
            )
        qs, n = pol.q_tile_range(b, k0)
        vals = [qs.v, n.v]
        vals += [pol.tile_class(b, b.const_i32(qt), k0).v for qt in range(PROBE_QT)]
        row = pol.row(b, b.const_i32(PROBE_ROW))
        vals += [int(pol.keep(b, row, b.add(k0, b.const_i32(j))).v) for j in range(4)]
        rows.append(vals)
    return np.array(rows, np.int32)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "case",
    [
        {
            "len_q": 100,
            "len_kv": 70,
            "br": True,
            "left": -1,
            "right": 0,
            "masked": True,
        },
        {
            "len_q": 40,
            "len_kv": 120,
            "br": False,
            "left": 9,
            "right": 3,
            "masked": True,
        },
        {
            "len_q": 0,
            "len_kv": 50,
            "br": False,
            "left": -1,
            "right": -1,
            "masked": True,
        },
        {
            "len_q": 77,
            "len_kv": 77,
            "br": False,
            "left": -1,
            "right": -1,
            "masked": False,
        },
    ],
    ids=["causal_br", "window_tl", "empty_q", "unmasked_tails"],
)
def test_probe_on_device_matches_host(case):
    arch = device_arch()
    if arch not in ("gfx942", "gfx950", "gfx1151", "gfx1201"):
        pytest.skip(f"needs gfx942/gfx950/gfx1151/gfx1201; device is {arch}")
    import ctypes
    import struct

    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime

    k_m0, k_n0 = 16, 32
    masked = case["masked"]
    kern = _probe_kernel(
        f"bwd_band_dev_{int(masked)}", k_m0=k_m0, k_n0=k_n0, masked=masked
    )
    art = compile_kernel(kern, arch=arch, capture_ir_text=False, backend="python")
    lq, lk = case["len_q"], case["len_kv"]
    off = lk - lq if case["br"] else 0
    n_kt = (max(lk, 1) + k_n0 - 1) // k_n0 + 1
    out = np.full(n_kt * PROBE_OUT, -7, np.int32)
    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)
    dev = rt.alloc(out.nbytes)
    buf = (ctypes.c_uint8 * out.nbytes).from_buffer(out)
    rt.memcpy_h2d(dev, buf, out.nbytes)
    args = struct.pack("<Qiiiii", dev, lq, lk, off, case["left"], case["right"])
    rt.launch(fn, (n_kt, 1, 1), (64, 1, 1), args)
    rt.sync()
    rt.memcpy_d2h(buf, dev, out.nbytes)
    rt.free(dev)
    mod.unload()
    host = _host_probe(
        lq, lk, off, case["left"], case["right"], k_m0, k_n0, n_kt, masked
    )
    assert np.array_equal(out.reshape(n_kt, PROBE_OUT), host)
