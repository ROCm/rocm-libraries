# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Self-checks of the shared SDPA backward case table (host-only, no device)."""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import replace

import numpy as np
import pytest

from rocke.numeric.sdpa_reference import sdpa_reference_bwd
from tests.sdpa import bwd_cases as BC
from tests.sdpa import cases as C
from tests.sdpa import helpers as H

CASES = BC.backward_cases()
SMOKE = BC.select_bwd("smoke")
CHEAP_FULL = [
    c for c in BC.select_bwd("full") if c.cost <= 4e7 and "smoke" not in c.tags
]


def _ids(cases):
    return [c.id for c in cases]


# ---------------------------------------------------------------------------
# table structure
# ---------------------------------------------------------------------------


def test_ids_unique_clean_and_disjoint_from_forward():
    ids = _ids(CASES)
    assert len(ids) == len(set(ids))
    for i in ids:
        assert i.startswith("bwd_"), i
        assert re.fullmatch(r"[a-z0-9_]+", i), i
        assert not re.search(r"(^|_)st\d+(_|$)|(^|_)s[ab](_|$)|(^|_)q\d+(_|$)", i), i
    assert not set(ids) & set(_ids(C.all_cases()))


def test_forward_table_untouched():
    assert not [c for c in C.all_cases() if "bwd" in c.tags or c.id.startswith("bwd_")]
    assert C.backward_cases() is CASES or C.backward_cases() == CASES
    assert _ids(C.select_bwd("smoke")) == _ids(SMOKE)


def test_table_is_deterministic():
    rebuilt = tuple(c for fn in BC._BWD_BUILDERS for c in fn())
    assert rebuilt == CASES


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_every_case_is_valid(case):
    assert BC.validate_bwd_case(case) == []


def test_validator_rejects_out_of_scope_cases():
    base = BC.get_bwd_case("bwd_dtype_fp16_d128_mha")
    bad = {
        "d256": replace(base, d=256, archs=H.CDNA_ARCHS),
        "d32_not_optional": replace(base, d=32),
        "bias": replace(base, bias=H.BiasSpec((1, 1, 1, 128))),
        "no_item15": replace(base, reqs=(1, 9)),
        "no_bwd_tag": replace(base, tags=base.tags - {"bwd"}),
        "forward_id": replace(base, id="dtype_fp16_d128_mha"),
        "emit_lse": replace(base, emit_lse=True),
        "decode_untagged": replace(base, s_q=1, decode=True),
        "ragged_without_16": replace(
            BC.get_bwd_case("bwd_ragged_eq_tl"), reqs=(5, 8, 15)
        ),
    }
    for name, case in bad.items():
        assert BC.validate_bwd_case(case), name


def test_tiers_nested_and_smoke_budget():
    full = BC.select_bwd("full")
    assert set(_ids(SMOKE)) < set(_ids(full)) < set(_ids(CASES))
    assert 20 <= len(SMOKE) <= 45
    assert len(full) < 0.6 * len(CASES)
    for c in SMOKE:
        assert c.cost <= 3e8, c.id
        assert set(c.archs) >= set(H.CDNA_ARCHS)
        assert "decode_bwd" not in c.tags and "optional" not in c.tags
    s = BC.bwd_table_summary()
    assert s["total"] == len(CASES) and s["exhaustive"] == len(CASES)
    assert s["smoke"] == len(SMOKE) and s["full"] == len(full)


def test_every_item_covered_individually_and_in_smoke():
    for item in list(range(1, 12)) + [15, 16]:
        assert [c for c in BC.select_bwd("full") if item in c.reqs], item
        assert [c for c in SMOKE if item in c.reqs], item
    assert all(15 in c.reqs for c in CASES)
    prefixes = {
        1: "bwd_dtype_",
        2: "bwd_layout_",
        3: "bwd_gqa_",
        4: "bwd_scale_",
        5: "bwd_causal_tl_",
        6: "bwd_causal_br_",
        7: "bwd_lse_",
        8: "bwd_seqlen_",
        9: "bwd_headdim_",
        10: "bwd_padding_",
        11: "bwd_window_",
        16: "bwd_ragged_",
    }
    for item, p in prefixes.items():
        group = [c for c in CASES if c.id.startswith(p)]
        assert group and all(item in c.reqs for c in group), item


def test_scope_axes():
    assert {c.d for c in CASES} == {32, 64, 128}
    assert {c.d for c in CASES if "optional" not in c.tags} == {64, 128}
    assert all(c.bias is None for c in CASES)
    assert {c.dtype for c in CASES} == {"fp16", "bf16"}
    assert {c.layout for c in CASES} == set(H.LAYOUTS)
    assert {c.length_mode for c in CASES} == set(H.LENGTH_MODES)
    for arch in H.ALL_ARCHS:
        assert BC.select_bwd("full", arch=arch), arch
    # tails that are not a multiple of 16, on both sides and in per-batch lengths
    lens = set()
    for c in CASES:
        lq, lk = c.lens()
        lens |= set(lq) | set(lk)
    assert {17, 33, 72, 100} <= lens
    assert any(c.s_q % 16 and c.s_kv % 16 for c in SMOKE)


def test_decode_bwd_group():
    dec = [c for c in CASES if "decode_bwd" in c.tags]
    assert dec and all(c.decode and all(n == 1 for n in c.lens()[0]) for c in dec)
    assert {c.length_mode for c in dec} == set(H.LENGTH_MODES)
    no_dec = BC.select_bwd("exhaustive", include_decode=False)
    assert len(no_dec) == len(CASES) - len(dec)
    assert not [c for c in no_dec if c.decode]


def test_interesting_intersections_present():
    def has(pred, tag="full"):
        return any(pred(c) for c in BC.select_bwd(tag))

    br = lambda c: c.band[2] is False and c.band[1] >= 0  # noqa: E731
    win = lambda c: c.band[0] >= 0  # noqa: E731
    gqa = lambda c: c.h_q > c.h_k  # noqa: E731

    def quad(c):
        return gqa(c) and c.length_mode == "ragged" and br(c) and win(c)

    assert has(quad) and has(quad, "smoke")
    assert has(lambda c: gqa(c) and c.length_mode == "padded" and win(c))
    assert has(lambda c: c.fully_masked_rows and br(c) and c.s_q > c.s_kv, "smoke")
    assert has(lambda c: c.fully_masked_rows and win(c), "smoke")
    assert has(lambda c: c.fully_masked_rows and c.length_mode == "ragged")
    assert has(lambda c: c.fully_masked_rows and c.length_mode == "padded")
    assert has(
        lambda c: c.layout == "packed_qkv" and gqa(c) and c.length_mode == "padded"
    )
    assert has(lambda c: c.layout.startswith("strided") and c.length_mode == "padded")
    assert has(lambda c: c.h_q == 8 and c.h_k == 1)  # MQA
    lefts = {c.left_bound for c in BC.select_bwd("full")}
    assert {0, 1, 63, 64, 65} <= lefts
    assert has(lambda c: c.left_bound >= 0 and c.right_bound < 0)  # left-only
    assert has(lambda c: c.left_bound < 0 and c.right_bound > 0)  # right-only
    assert has(lambda c: c.left_bound > 0 and c.right_bound > 0)  # two-sided
    assert has(lambda c: c.h_k != c.h_v, "exhaustive")


_BX = re.compile(
    r"bwd_x_(mha|gqa4|mqa)_(fixed|padded|ragged)_(none|tl|br|tl_win|br_win)_(lt|eq|gt)$"
)


def test_cross_grid_is_exhaustive_and_pairwise_covered_in_full():
    grid = {c.id: _BX.match(c.id).groups() for c in CASES if _BX.match(c.id)}
    assert len(grid) == 3 * 3 * 5 * 3
    in_full = [grid[c.id] for c in BC.select_bwd("full") if c.id in grid]
    assert 0 < len(in_full) < len(grid) // 2
    for i, j in itertools.combinations(range(4), 2):
        want = {(g[i], g[j]) for g in grid.values()}
        got = {(g[i], g[j]) for g in in_full}
        assert got == want, (i, j)


# ---------------------------------------------------------------------------
# materialisation and oracle
# ---------------------------------------------------------------------------


def test_materialize_is_deterministic_and_distinct_from_forward():
    c = BC.get_bwd_case("bwd_padding_q_gt_kv_br")
    a, b = BC.materialize_bwd(c), BC.materialize_bwd(c)
    for n in a.buffers:
        assert np.array_equal(a.buffers[n], b.buffers[n], equal_nan=True)
    d = BC.materialize_bwd(c, seed=3)
    assert not np.array_equal(a.do, d.do)
    with pytest.raises(ValueError):
        BC.materialize_bwd(replace(c, bias=H.BiasSpec((1, 1, 1, c.s_kv))))


def _check_inputs(case, inp):
    for n in ("q", "k", "v", "o", "do"):
        dsc = inp.descs[n]
        view = H._view(inp.buffers[dsc.buffer], dsc).astype(np.float64)
        logical = getattr(inp, n)
        assert np.array_equal(view, logical), n
        assert np.array_equal(H.round_to(logical, case.dtype), logical), n
        assert dsc.offset + H._span(dsc.dims, dsc.strides) <= inp.sizes[dsc.buffer]
    for n in ("dq", "dk", "dv"):
        dsc = inp.descs[n]
        src = inp.descs[n[1:]]
        assert dsc.dims == src.dims and dsc.strides == src.strides
        assert dsc.offset + H._span(dsc.dims, dsc.strides) <= inp.sizes[dsc.buffer]
    if case.layout == "packed_qkv":
        assert {inp.descs[n].buffer for n in ("dq", "dk", "dv")} == {"dqkv"}
    assert inp.buffers["lse"].dtype == np.float32
    assert inp.lse.shape == H.lse_shape(case) == inp.valid_q.shape
    # padded rows hold finite garbage; valid rows hold the oracle forward
    fref = H.evaluate(inp.fwd)
    rows = inp.valid_q
    assert np.isfinite(inp.lse[~rows]).all()
    assert np.array_equal(
        inp.lse[rows], fref.lse.astype(np.float32).astype(np.float64)[rows]
    )
    np.testing.assert_array_equal(
        inp.o[rows], H.round_to(fref.o[rows], case.dtype).astype(np.float64)
    )
    assert np.isneginf(inp.lse[rows]).any() == case.fully_masked_rows


def _check_reference(case, inp, ref):
    for n in BC.GRADS:
        a = getattr(ref, n)
        assert a.shape == getattr(inp, n[1:]).shape
        assert np.isfinite(a).all(), n
    assert (ref.dq[ref.dead_q] == 0).all()
    assert ref.dead_q.any() == case.fully_masked_rows
    vq = ref.valid_q[..., None]
    assert (np.where(vq, 0.0, ref.dq) == 0).all()
    if case.length_mode == "padded":
        vk = ref.valid_kv[:, None, :, None]
        assert (np.where(vk, 0.0, ref.dk) == 0).all() and (
            np.where(vk, 0.0, ref.dv) == 0
        ).all()
    assert BC.compare_bwd(case, ref.dq, ref.dk, ref.dv, ref) == []


@pytest.mark.parametrize("case", SMOKE + CHEAP_FULL, ids=_ids(SMOKE + CHEAP_FULL))
def test_materialized_inputs_oracle_and_simulated_kernel(case):
    inp = BC.materialize_bwd(case)
    _check_inputs(case, inp)
    ref = BC.evaluate_bwd(inp)
    _check_reference(case, inp, ref)
    # an fp32-accumulate low-precision simulation passes the proposed tolerance
    dq, dk, dv = BC.simulate_lowp_bwd(inp)
    assert BC.compare_bwd(case, dq, dk, dv, ref) == []
    # ... also when per-query-head dK / dV partials are rounded before the GQA sum
    dq, dk, dv = BC.simulate_lowp_bwd(inp, round_head_partials=True)
    assert BC.compare_bwd(case, dq, dk, dv, ref) == []
    # a perturbation of one valid element of each gradient is caught
    for i, name in enumerate(BC.GRADS):
        grads = [ref.dq.copy(), ref.dk.copy(), ref.dv.copy()]
        g = grads[i]
        mx = np.abs(g).max()
        flat = g.reshape(-1)
        flat[int(np.argmax(np.abs(flat)))] += 0.1 * mx + 1.0
        assert BC.compare_bwd(case, *grads, ref), name
    # non-finite values in a valid row are caught
    bad = ref.dq.copy()
    bad[tuple(np.argwhere(ref.valid_q)[0])] = np.nan
    assert BC.compare_bwd(case, bad, ref.dk, ref.dv, ref)


def _keep_mask_bwd(inp, *, d_left=0, d_right=0, drop=None, extra_kv=0):
    """Float64 backward on the stored inputs with an editable keep-mask.

    Models a kernel that consumes the stored LSE but keeps a different set of
    cells: ``d_left`` / ``d_right`` move the band bounds, ``drop`` ("first" /
    "last") removes the first / last valid key of every sequence, and
    ``extra_kv`` (padded only) lets each row also see that many keys past
    ``seq_len_kv`` without moving the bottom-right diagonal.
    """
    case, f = inp.case, inp.fwd
    left, right, tl = case.band
    lq, lk = case.lens()
    ragged = case.length_mode == "ragged"
    qo = np.concatenate([[0], np.cumsum(lq)])
    ko = np.concatenate([[0], np.cumsum(lk)])
    gk, gv = case.h_q // case.h_k, case.h_q // case.h_v
    dq, dk, dv = np.zeros(f.q.shape), np.zeros(f.k.shape), np.zeros(f.v.shape)
    for b in range(case.b):
        sq, skv = lq[b], lk[b]
        width = min(skv + extra_kv, case.s_kv) if not ragged else skv
        qi = np.arange(sq)[:, None]
        ki = np.arange(width)[None, :]
        off = 0 if tl else skv - sq
        band = np.ones((sq, width), dtype=bool)
        if right >= 0:
            band &= ki <= qi + off + right + d_right
        if left >= 0:
            band &= ki >= qi + off - left - d_left
        if drop == "last":
            band[:, skv - 1] = False
        elif drop == "first":
            band[:, 0] = False
        for h in range(case.h_q):
            hk, hv = h // gk, h // gv
            if ragged:
                qs, ks = slice(qo[b], qo[b + 1]), slice(ko[b], ko[b + 1])
                q, k, v = f.q[qs, h], f.k[ks, hk], f.v[ks, hv]
                o, do, lse = inp.o[qs, h], inp.do[qs, h], inp.lse[qs, h]
            else:
                q, k, v = f.q[b, h, :sq], f.k[b, hk, :width], f.v[b, hv, :width]
                o, do, lse = inp.o[b, h, :sq], inp.do[b, h, :sq], inp.lse[b, h, :sq]
            live = np.isfinite(lse)
            keep = band & live[:, None]
            s = f.scale * (q @ k.T)
            p = np.where(
                keep,
                np.exp(np.where(keep, s - np.where(live, lse, 0.0)[:, None], 0)),
                0,
            )
            delta = (do * o).sum(axis=1)
            ds = p * (do @ v.T - delta[:, None])
            if ragged:
                dq[qs, h] = f.scale * (ds @ k)
                dk[ks, hk] += f.scale * (ds.T @ q)
                dv[ks, hv] += p.T @ do
            else:
                dq[b, h, :sq] = f.scale * (ds @ k)
                dk[b, hk, :width] += f.scale * (ds.T @ q)
                dv[b, hv, :width] += p.T @ do
    return dq, dk, dv


@pytest.mark.parametrize("case", SMOKE, ids=_ids(SMOKE))
def test_keep_mask_backward_matches_oracle_unmutated(case):
    inp = BC.materialize_bwd(case)
    ref = BC.evaluate_bwd(inp)
    for got, n in zip(_keep_mask_bwd(inp), BC.GRADS):
        np.testing.assert_allclose(
            got, getattr(ref, n), rtol=1e-10, atol=1e-10, err_msg=n
        )


def _mutations(case, inp):
    """Plausible kernel bugs, each evaluated on the stored inputs."""
    f = inp.fwd
    kw = BC._oracle_kwargs(inp)
    args = (f.q, f.k, f.v, inp.o, inp.do, inp.lse)
    left, right, tl = case.band
    out = {
        "drop_last_key": _keep_mask_bwd(inp, drop="last"),
        "drop_first_key": _keep_mask_bwd(inp, drop="first"),
    }
    if right >= 0:
        out["band_right_minus_one"] = _keep_mask_bwd(inp, d_right=-1)
    if left >= 0:
        out["band_left_minus_one"] = _keep_mask_bwd(inp, d_left=-1)
    if case.length_mode == "padded" and min(case.lens()[1]) < case.s_kv:
        out["one_padded_key_attended"] = _keep_mask_bwd(inp, extra_kv=1)
    if right >= 0 or left >= 0:
        shifted = dict(kw)
        if right >= 0:
            shifted["right_bound"] = right + 1
        else:
            shifted["left_bound"] = left + 1
        out["band_off_by_one"] = sdpa_reference_bwd(*args, **shifted)[:3]
    lq, lk = case.lens()
    if right >= 0 and any(a != b for a, b in zip(lq, lk)):
        out["alignment_flipped"] = sdpa_reference_bwd(
            *args, **dict(kw, top_left=not tl)
        )[:3]
    if case.h_q != case.h_k and case.h_k == case.h_v:
        g = case.h_q // case.h_k
        ax = 1
        reps = [1] * f.k.ndim
        reps[ax] = g
        r = sdpa_reference_bwd(
            f.q, np.tile(f.k, reps), np.tile(f.v, reps), *args[3:], **kw
        )
        shp = list(f.k.shape)
        shp[ax : ax + 1] = [g, case.h_k]
        out["gqa_interleaved"] = (
            r.dq,
            r.dk.reshape(shp).sum(axis=ax),
            r.dv.reshape(shp).sum(axis=ax),
        )
    ref = sdpa_reference_bwd(*args, **kw)
    out["dq_missing_scale"] = (ref.dq / kw["scale"], ref.dk, ref.dv)
    out["lse_log2_base"] = sdpa_reference_bwd(*args[:5], inp.lse / math.log(2.0), **kw)[
        :3
    ]
    if case.length_mode == "padded":
        dense = {k: v for k, v in kw.items() if k != "seq_len_kv"}
        out["padded_keys_attended"] = sdpa_reference_bwd(*args, **dense)[:3]
    return out


@pytest.mark.parametrize("case", SMOKE, ids=_ids(SMOKE))
def test_proposed_tolerance_catches_plausible_bugs(case):
    inp = BC.materialize_bwd(case)
    ref = BC.evaluate_bwd(inp)
    muts = _mutations(case, inp)
    active = 0
    for name, grads in muts.items():
        # a mutation that is a no-op on this case (MQA head mapping, padded keys
        # hidden by the causal band, ...) is not a bug here
        if all(
            np.allclose(g, getattr(ref, n), rtol=0, atol=1e-12)
            for g, n in zip(grads, BC.GRADS)
        ):
            continue
        active += 1
        assert BC.compare_bwd(case, *grads, ref), name
    assert active >= 2


def test_row_check_catches_dropped_tail_key_hidden_by_global_bound():
    """A dropped last key zeroes one small dK / dV row: below the tensor-wide
    bound, caught only by the per-row bound."""
    case = BC.get_bwd_case("bwd_dtype_bf16_d128_causal_tl")
    inp = BC.materialize_bwd(case)
    ref = BC.evaluate_bwd(inp)
    grads = _keep_mask_bwd(inp, drop="last")
    t = BC.TOLERANCE_BWD[case.dtype]
    ratios = BC.error_ratios(case, *grads, ref)
    assert ratios["dv"] < t["rtol"]  # the tensor-wide bound alone misses it
    assert ratios["dv_row"] > 4 * t["rtol"]
    probs = BC.compare_bwd(case, *grads, ref)
    assert any("dv row mismatch" in p for p in probs), probs


def test_fully_masked_rows_must_be_zero_in_dq():
    case = BC.get_bwd_case("bwd_lse_fully_masked_rows_neg_inf")
    inp = BC.materialize_bwd(case)
    ref = BC.evaluate_bwd(inp)
    assert ref.dead_q.any()
    dq = ref.dq.copy()
    dq[ref.dead_q] = 1e-2
    probs = BC.compare_bwd(case, dq, ref.dk, ref.dv, ref)
    assert any("fully masked" in p for p in probs)


def test_ragged_backward_matches_padded_dense_layout():
    """THD backward equals the padded dense backward on the same sequences."""
    rag = BC.get_bwd_case("bwd_ragged_q_lt_kv_br")
    inp = BC.materialize_bwd(rag)
    ref = BC.evaluate_bwd(inp)
    lq, lk = rag.lens()
    qo, ko = inp.fwd.q_offsets, inp.fwd.kv_offsets
    B, Sq, Skv = rag.b, max(lq), max(lk)

    def pad(x, offs, lens, S):
        out = np.zeros((B, x.shape[1], S, x.shape[2]))
        for b in range(B):
            out[b, :, : lens[b]] = x[offs[b] : offs[b + 1]].transpose(1, 0, 2)
        return out

    lse = np.full((B, rag.h_q, Sq), -np.inf)
    for b in range(B):
        lse[b, :, : lq[b]] = inp.lse[qo[b] : qo[b + 1]].T
    left, right, tl = rag.band
    dense = sdpa_reference_bwd(
        pad(inp.fwd.q, qo, lq, Sq),
        pad(inp.fwd.k, ko, lk, Skv),
        pad(inp.fwd.v, ko, lk, Skv),
        pad(inp.o, qo, lq, Sq),
        pad(inp.do, qo, lq, Sq),
        lse,
        scale=inp.fwd.scale,
        left_bound=left,
        right_bound=right,
        top_left=tl,
        seq_len_q=lq,
        seq_len_kv=lk,
    )
    np.testing.assert_allclose(pad(ref.dq, qo, lq, Sq), dense.dq, atol=1e-12)
    np.testing.assert_allclose(pad(ref.dk, ko, lk, Skv), dense.dk, atol=1e-12)
    np.testing.assert_allclose(pad(ref.dv, ko, lk, Skv), dense.dv, atol=1e-12)


def test_reference_bwd_for_by_id():
    inp, ref = BC.reference_bwd_for("bwd_gqa_mqa_causal_tl")
    assert ref.dk.shape == (2, 1, 96, 128)
    assert inp.case.id == "bwd_gqa_mqa_causal_tl"


def test_to_torch_bwd_roundtrip_if_torch_available():
    torch = pytest.importorskip("torch")
    for cid in (
        "bwd_layout_packed_qkv_mha_d128",
        "bwd_layout_strided_bshd_fp16_causal_tl",
        "bwd_ragged_q_lt_kv_br",
        "bwd_padding_q_lt_kv_tl",
    ):
        inp = BC.materialize_bwd(BC.get_bwd_case(cid))
        t = BC.to_torch_bwd(inp, device="cpu")
        for n in ("q", "k", "v", "o", "do"):
            assert np.array_equal(
                t[n].float().numpy().astype(np.float64), getattr(inp, n)
            ), n
        for n in ("dq", "dk", "dv"):
            assert tuple(t[n].shape) == tuple(inp.descs[n].dims)
            assert not t[n].float().abs().sum().item()
        assert (
            t["lse"].dtype == torch.float32 and tuple(t["lse"].shape) == inp.lse.shape
        )
        if inp.case.layout == "packed_qkv":
            assert (
                t["dq"].untyped_storage().data_ptr()
                == t["dk"].untyped_storage().data_ptr()
            )
