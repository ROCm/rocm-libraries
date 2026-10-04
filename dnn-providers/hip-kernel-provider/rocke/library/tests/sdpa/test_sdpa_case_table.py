# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Self-checks of the shared SDPA case table (host-only, no device needed)."""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest

from tests.sdpa import cases as C
from tests.sdpa import helpers as H

CASES = C.all_cases()
CHEAP = [c for c in CASES if c.cost <= 2e7 or "smoke" in c.tags]


def _ids(cases):
    return [c.id for c in cases]


def test_ids_unique_and_clean():
    ids = _ids(CASES)
    assert len(ids) == len(set(ids))
    for i in ids:
        assert re.fullmatch(r"[a-z0-9_]+", i), i
        assert not re.search(r"hackathon|(^|_)st\d+(_|$)", i), i


def test_table_is_deterministic():
    rebuilt = tuple(c for fn in C._BUILDERS for c in fn())
    assert rebuilt == CASES


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_every_case_is_valid(case):
    assert H.validate_case(case) == []


def test_validator_rejects_bad_cases():
    base = C.get_case("dtype_fp16_d128_mha")
    bad = {
        "gqa": replace(base, h_k=3, h_v=3),
        "d_multiple": replace(base, d=60),
        "dtype": replace(base, dtype="fp32"),
        "bound": replace(base, left_bound=-2),
        "arch": replace(base, archs=("gfx0000",)),
        "d256_rdna": replace(base, d=256, archs=("gfx1151",)),
        "bias_last": replace(base, bias=H.BiasSpec((1, 1, 1, 77))),
        "bias_rank": replace(base, bias=H.BiasSpec((1, 1, 1, 1, 128))),
        "bias_batch": replace(base, bias=H.BiasSpec((5, 1, 128, 128))),
        "lens_without_mode": replace(base, q_lens=(128, 128), kv_lens=(128, 128)),
        "ragged_layout": replace(
            base, length_mode="ragged", q_lens=(128, 128), kv_lens=(128, 128)
        ),
        "wrong_flag": replace(base, fully_masked_rows=True),
        "wrong_decode": replace(base, decode=True),
        "no_tier": replace(base, tags=frozenset()),
        "scale": replace(base, scale=-1.0),
    }
    for name, case in bad.items():
        assert H.validate_case(case), name
    long_len = replace(
        C.get_case("padding_eq_none"), q_lens=(99, 19, 31), kv_lens=(45, 19, 31)
    )
    assert H.validate_case(long_len)


def test_stride_overlap_helper():
    assert H.strides_overlap((2, 3, 4), (12, 4, 1)) is False
    assert H.strides_overlap((2, 3, 4), (4, 4, 1)) is True
    assert H.strides_overlap((1, 3, 4), (0, 4, 1)) is False


def test_tiers_nested_and_smoke_budget():
    smoke = C.select("smoke")
    full = C.select("full")
    assert set(_ids(smoke)) < set(_ids(full)) < set(_ids(CASES))
    assert 30 <= len(smoke) <= 60
    assert len(full) < 0.6 * len(CASES)
    for c in smoke:
        assert c.cost <= 6e8, c.id
        assert set(c.archs) >= set(H.CDNA_ARCHS)


def test_every_requirement_item_covered_individually_and_in_smoke():
    for item in range(1, 15):
        assert [c for c in C.select("full") if item in c.reqs], item
        assert [c for c in C.select("smoke") if item in c.reqs], item
    # the dedicated single-item groups exist per item (not only as side effects)
    prefixes = {
        1: "dtype_",
        2: "layout_",
        3: "gqa_",
        4: "scale_",
        5: "causal_tl_",
        6: "causal_br_",
        7: "lse_",
        8: "seqlen_",
        9: "headdim_",
        10: "padding_",
        11: "window_",
        12: "ragged_",
        13: "decode_",
        14: "bias_",
    }
    for item, p in prefixes.items():
        assert any(c.id.startswith(p) for c in CASES), item


def test_required_axis_values_present():
    assert {c.d for c in CASES} >= {64, 128, 256}
    assert {c.dtype for c in CASES} == {"fp16", "bf16"}
    assert {c.layout for c in CASES} == set(H.LAYOUTS)
    assert {c.length_mode for c in CASES} == set(H.LENGTH_MODES)
    for arch in H.ALL_ARCHS:
        assert C.select("full", arch=arch), arch
    assert not [c for c in CASES if c.d == 256 and set(c.archs) & set(H.RDNA_ARCHS)]
    # non-tile-multiple lengths on both sides
    assert any(c.s_q % 16 and c.s_kv % 16 for c in CASES)
    assert any(c.s_q == 1 and c.s_kv % 16 for c in CASES)


def test_interesting_intersections_present():
    def has(pred, tag="full"):
        return any(pred(c) for c in C.select(tag))

    br = lambda c: c.band[2] is False and c.band[:2] == (-1, 0)  # noqa: E731
    win = lambda c: c.band[0] >= 0  # noqa: E731
    gqa = lambda c: c.h_q > c.h_k  # noqa: E731
    # GQA x varlen x bottom-right x bias, in the smoke tier too
    quad = lambda c: (  # noqa: E731
        gqa(c)
        and c.length_mode == "ragged"
        and c.band[2] is False
        and c.bias is not None
    )
    assert has(quad) and has(quad, "smoke")
    assert has(
        lambda c: gqa(c) and c.length_mode == "padded" and win(c) and c.bias is not None
    )
    # window off-by-one around zero / one / tile edges
    lefts = {c.left_bound for c in C.select("full")}
    assert {0, 1, 63, 64, 65} <= lefts
    # fully masked rows from band, padding-shaped lengths and bias
    assert has(lambda c: c.fully_masked_rows and br(c) and c.s_q > c.s_kv)
    assert has(lambda c: c.fully_masked_rows and win(c))
    assert has(
        lambda c: c.fully_masked_rows and c.bias is not None and c.bias.kind != "dense"
    )
    assert has(lambda c: c.fully_masked_rows and c.length_mode == "ragged")
    assert has(lambda c: c.fully_masked_rows and c.length_mode == "padded")
    # decode in every length mode, with top-left (one visible key) and bottom-right
    for mode in H.LENGTH_MODES:
        assert has(lambda c, m=mode: c.decode and c.length_mode == m)
    assert has(lambda c: c.decode and c.band[2] and c.band[1] == 0)
    assert has(lambda c: c.decode and br(c))
    assert has(lambda c: c.decode and c.d == 256)
    # layouts
    for layout in (
        "bshd",
        "bhsd",
        "strided_bshd",
        "strided_bhsd",
        "packed_qkv",
        "mixed",
    ):
        assert has(lambda c, lay=layout: c.layout == lay)
    assert has(lambda c: c.layout == "packed_qkv" and gqa(c))
    assert has(lambda c: c.layout.startswith("strided") and c.length_mode == "padded")
    # bias broadcast shapes and layouts
    shapes = {c.bias.shape for c in CASES if c.bias is not None}
    assert (
        (1, 1, 1, 160) in shapes
        and (2, 8, 96, 160) in shapes
        and (1, 1, 96, 160) in shapes
    )
    assert {c.bias.layout for c in CASES if c.bias} == set(H.BIAS_LAYOUTS)
    # both lse settings, and unequal kv-head counts, and the deprecated form
    assert has(lambda c: c.emit_lse) and has(lambda c: not c.emit_lse)
    assert has(lambda c: c.h_k != c.h_v, "exhaustive")
    assert has(lambda c: c.causal_bool is not None, "exhaustive")


_X_ID = re.compile(
    r"x_(mha|gqa4|mqa)_(fixed|padded|ragged)_(none|tl|br|tl_win|br_win)"
    r"_(none|key_row|full)_(lt|eq|gt)$"
)


def test_cross_grid_is_exhaustive_and_pairwise_covered_in_full():
    grid = {c.id: _X_ID.match(c.id).groups() for c in CASES if _X_ID.match(c.id)}
    assert len(grid) == 3 * 3 * 5 * 3 * 3
    assert len(set(grid.values())) == len(grid)
    in_full = [grid[c.id] for c in C.select("full") if c.id in grid]
    assert 0 < len(in_full) < len(grid) // 3
    for i in range(5):
        for j in range(i + 1, 5):
            want = {(g[i], g[j]) for g in grid.values()}
            got = {(g[i], g[j]) for g in in_full}
            assert got == want, (i, j)


def test_materialize_is_deterministic_and_seeded_by_id():
    c = C.get_case("bias_full")
    a, b = H.materialize(c), H.materialize(c)
    for n in a.buffers:
        assert np.array_equal(a.buffers[n], b.buffers[n])
    other = H.materialize(C.get_case("bias_h_bcast"))
    assert not np.array_equal(a.q.reshape(-1)[:64], other.q.reshape(-1)[:64])
    assert H.case_seed(c) != H.case_seed(C.get_case("bias_h_bcast"))
    d = H.materialize(c, seed=1)
    assert not np.array_equal(a.q, d.q)


@pytest.mark.parametrize("case", CHEAP, ids=_ids(CHEAP))
def test_materialized_views_and_oracle(case):
    inp = H.materialize(case)
    # logical values are exactly what the strided descriptors address
    for n in "qkv":
        dsc = inp.descs[n]
        view = H._view(inp.buffers[dsc.buffer], dsc)
        assert np.array_equal(view.astype(np.float64), getattr(inp, n))
        assert np.array_equal(H.round_to(getattr(inp, n), case.dtype), getattr(inp, n))
        assert dsc.strides[-1] == 1
        assert dsc.offset + H._span(dsc.dims, dsc.strides) <= inp.sizes[dsc.buffer]
    if case.layout == "packed_qkv":
        assert {inp.descs[n].buffer for n in "qkv"} == {"qkv"}
    if case.bias is not None:
        bd = inp.bias_desc
        view = H._view(inp.buffers["bias"], bd).astype(np.float64)
        want = np.broadcast_to(
            inp.bias.reshape((1,) * (4 - inp.bias.ndim) + inp.bias.shape), view.shape
        )
        assert np.array_equal(view, want)
    if case.length_mode == "padded":
        assert (
            inp.seq_len_q.shape == (case.b, 1, 1, 1) and inp.seq_len_q.dtype == np.int32
        )
    if case.length_mode == "ragged":
        assert (
            inp.q_offsets[-1] == inp.q.shape[0] and inp.kv_offsets[-1] == inp.k.shape[0]
        )

    ref = H.evaluate(inp)
    assert ref.o.shape[:-1] == ref.lse.shape == ref.valid_rows.shape
    assert not np.isnan(ref.o).any() and np.isfinite(ref.o).all()
    assert not np.isnan(ref.lse).any()
    rows = ref.valid_rows
    dead = np.isneginf(ref.lse) & rows
    # the table flag agrees with an independent computation
    assert dead.any() == case.fully_masked_rows
    assert (ref.o[dead] == 0).all()
    assert np.isfinite(ref.lse[rows & ~dead]).all()
    if not rows.all():
        assert (ref.o[~rows] == 0).all() and np.isneginf(ref.lse[~rows]).all()
    # comparison helper: oracle matches itself, a perturbation is caught
    lse = ref.lse if case.emit_lse else None
    assert H.compare(case, ref.o, lse, ref) == []
    bumped = ref.o.copy()
    bumped.reshape(-1, bumped.shape[-1])[
        int(np.flatnonzero(rows.reshape(-1))[0])
    ] += 0.5
    assert H.compare(case, bumped, lse, ref)
    if case.emit_lse:
        if (rows & ~dead).any():
            wrong = ref.lse.copy()
            wrong[tuple(np.argwhere(rows & ~dead)[0])] += 0.5
            assert H.compare(case, ref.o, wrong, ref)
        if case.fully_masked_rows:
            alive = ref.lse.copy()
            alive[dead] = 0.0
            assert H.compare(case, ref.o, alive, ref)


def test_top_left_decode_sees_only_first_key():
    c = C.get_case("decode_skv17_tl")
    inp = H.materialize(c)
    ref = H.evaluate(inp)
    # one visible key: output equals V[0] of the mapped kv head
    group = c.h_q // c.h_k
    for h in range(c.h_q):
        assert np.allclose(ref.o[:, h, 0], inp.v[:, h // group, 0])


def test_window_left_w_keeps_w_plus_one_keys():
    c = C.get_case("window_tl_w1")
    inp = H.materialize(c)
    probe = replace(
        inp,
        v=np.broadcast_to(
            np.arange(c.s_kv, dtype=np.float64)[None, None, :, None], inp.v.shape
        ).copy(),
    )
    probe_ref = H.evaluate(probe)
    # uniform-ish scores are not guaranteed, so only check the support: the
    # output of a row is a convex combination of the values of its two keys
    q = c.s_q - 1
    o = probe_ref.o[0, 0, q, 0]
    assert q - 1 <= o <= q
    assert not c.fully_masked_rows


def test_to_torch_roundtrip_if_torch_available():
    torch = pytest.importorskip("torch")
    for cid in (
        "layout_packed_qkv_mha_d128",
        "layout_strided_bshd_fp16_causal_tl",
        "ragged_eq_tl",
    ):
        inp = H.materialize(C.get_case(cid))
        t = H.to_torch(inp, device="cpu")
        for n in "qkv":
            assert np.array_equal(
                t[n].float().numpy().astype(np.float64), getattr(inp, n)
            )
        assert t["o"].shape == tuple(inp.descs["o"].dims)
        assert (t["lse"] is not None) == inp.case.emit_lse
    del torch


def test_counts_documented_shape():
    s = C.table_summary()
    assert s["total"] == len(CASES)
    assert s["smoke"] < s["full"] < s["exhaustive"] == s["total"]
