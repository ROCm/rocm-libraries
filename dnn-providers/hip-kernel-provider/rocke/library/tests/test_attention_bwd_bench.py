# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests of the backward timing harness and census helpers.

CPU only: the committed census, performance classes and the
class-representative generator, competitor arm rules, the paired verdict and
cliff rules, planning, the results-directory guard, and the fp32 chunked
reference (checked against an fp64 autograd reference on CPU tensors).
"""

from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from benchmarks.common import attention_bwd_bench as bb

# Synthetic constants for the rule tests (the shipped values live in a private file).
CONSTS = {
    "BAND_K": 1.5,
    "TIE_EPS": 0.02,
    "RERUN_MAX": 3,
    "NOISY_SPREAD": 0.06,
    "ATTEMPTS_NOISY": 9,
    "CLIFF_BAND": 0.3,
}


@pytest.fixture(scope="module")
def census():
    return bb.load_census()


@pytest.fixture(scope="module")
def cells(census):
    return bb.census_cells(census)


# --------------------------------------------------------------------------- census
def test_census_composition(census, cells):
    assert census["schema"] == bb.CENSUS_SCHEMA
    assert len(cells) == census["n_configs"] == 84
    assert sum(c.core for c in cells) == census["n_core"] == 20
    cnt = collections.Counter
    assert cnt(c.d for c in cells) == {128: 59, 64: 25}
    assert cnt(c.mask for c in cells) == {
        "causal_tl": 49,
        "none": 19,
        "causal_br": 10,
        "swa": 6,
    }
    assert cnt(c.layout for c in cells) == {"bshd": 74, "bhsd": 4, "thd": 6}
    assert cnt(c.dtype for c in cells) == {"bf16": 71, "fp16": 13}
    assert len({(c.d, c.hq, c.hkv) for c in cells}) == 7
    cores = [c for c in cells if c.core]
    assert cnt(c.layout for c in cores) == {"bshd": 17, "bhsd": 1, "thd": 2}
    assert len({c.id for c in cells}) == 84


def test_census_has_no_measurements(census):
    text = json.dumps(census)
    for word in ("median", "_ms", "tflops", "spread", "wall_s", "latency"):
        assert word not in text
    allowed = {
        "id",
        "model",
        "d",
        "hq",
        "hkv",
        "b",
        "sq",
        "skv",
        "mask",
        "window",
        "dtype",
        "layout",
        "seqlens",
        "weight",
        "tag",
        "core",
    }
    assert all(set(c) <= allowed for c in census["configs"])


def test_census_class_counts_match_the_generator(census, cells):
    block = census["class_representatives"]
    counts = bb.class_counts(cells)
    assert set(block["counts_per_arch"]) == set(bb.CLASS_ARCHS)
    assert all(v == counts for v in block["counts_per_arch"].values())
    assert counts["class_representatives"] == len(bb.class_representatives(cells))
    assert counts["declared_classes"] == 3 * 2 * 4 * 3 * 3 * 3 * 2


def test_performance_class_of_census_cells(cells):
    by_id = {c.id: c for c in cells}
    thd = bb.performance_class(by_id["d128_h32_kv8_thd_n8_t16384_causal_tl_bf16"])
    assert (thd.length_mode, thd.head_mode, thd.mask, thd.length_relation) == (
        "thd",
        "gqa",
        "causal_tl",
        "square",
    )
    swa = bb.performance_class(by_id["d128_h32_kv8_b1_q2048_k8192_swa_bf16_bshd_w4096"])
    assert (swa.mask, swa.length_relation, swa.length_mode) == (
        "window",
        "unequal",
        "fixed",
    )
    mha = bb.performance_class(by_id["d64_h12_kv12_b8_q1024_k1024_none_bf16_bshd"])
    assert mha.head_mode == "mha" and mha.stage_vec == 8


def test_class_representatives_are_deterministic_and_land_in_their_class(cells):
    a = bb.class_representatives(cells)
    b = bb.class_representatives(cells)
    assert [bb.cell_to_dict(x) for x in a] == [bb.cell_to_dict(x) for x in b]
    covered = {bb.performance_class(c) for c in cells}
    core_ids = {c.id for c in cells if c.core}
    keys = set()
    for rep in a:
        cls = bb.performance_class(rep)
        assert rep.id == "rep_" + cls.key() and cls not in covered
        assert rep.parent in core_ids
        keys.add(cls)
        assert rep.hq % rep.hkv == 0 and rep.hq % rep.h_v == 0
        if rep.mask == "swa":
            assert 0 < rep.window <= rep.skv
        if rep.is_thd:
            assert len(rep.seqlens) == len(rep.kv_lengths) == rep.b
            assert max(rep.seqlens) == rep.sq and max(rep.kv_lengths) == rep.skv
        if rep.padded_kv is not None:
            assert max(rep.padded_kv) == rep.skv and len(rep.padded_kv) == rep.b
    assert len(keys) == len(a)


def test_representative_changes_only_class_axes():
    parent = bb.Cell(
        "p",
        128,
        32,
        8,
        4,
        2048,
        2048,
        "causal_tl",
        "bf16",
        "bshd",
        core=True,
        model="m",
    )
    cls = bb.PerfClass("general", 128, "fp16", "causal_tl", "gqa", "square", "fixed", 8)
    rep = bb.representative_for(cls, parent)
    same = ("d", "hq", "hkv", "b", "sq", "skv", "mask", "layout", "window", "hv")
    assert (
        all(getattr(rep, f) == getattr(parent, f) for f in same) and rep.dtype == "fp16"
    )
    rep1 = bb.representative_for(replace_cls(cls, length_relation="s_q_1"), parent)
    assert rep1.sq == 1 and rep1.skv == parent.skv


def replace_cls(cls, **kw):
    from dataclasses import replace

    return replace(cls, **kw)


def test_cliff_probes_move_one_axis(cells):
    core = next(c for c in cells if c.core and not c.is_thd and c.hq != c.hkv)
    probes = bb.cliff_probes(core)
    assert {p.parent for p in probes} == {core.id}
    for p in probes:
        diff = [
            f
            for f in ("hkv", "sq", "mask", "layout", "dtype")
            if getattr(p, f) != getattr(core, f)
        ]
        assert len(diff) in (1, 2)  # S +- 1 moves sq and skv together


# --------------------------------------------------------------------------- arms
def test_default_arm_rules():
    c = bb.Cell("x", 128, 32, 8, 1, 64, 64, "none", "bf16", "bshd")
    assert bb.default_arm(c) == "default|native"
    assert (
        bb.default_arm(bb.Cell("x", 64, 8, 8, 1, 64, 64, "none", "bf16", "bshd"))
        == "default|mha"
    )
    assert (
        bb.default_arm(bb.Cell("x", 64, 8, 4, 1, 64, 64, "none", "bf16", "bshd", hv=2))
        == "default|expand"
    )
    t = bb.Cell("x", 64, 8, 8, 2, 4, 4, "none", "bf16", "thd", seqlens=(3, 4))
    assert bb.default_arm(t) == "varlen_flash|-"
    for cell in (c, t):
        assert bb.default_arm(cell) in bb.arms_for_cell(cell)
    assert "math|native" not in bb.arms_for_cell(c, math_ok=False)
    assert set(bb.arms_for_cell(c)) >= {
        "default|expand",
        "flash|native",
        "efficient|expand",
    }


# --------------------------------------------------------------------------- verdicts
def test_paired_verdict_rules():
    base = [1.0, 1.01, 0.99, 1.0, 1.02]
    assert bb.paired_verdict([0.9] * 5, base, CONSTS) == "pass"
    assert bb.paired_verdict([1.5] * 5, base, CONSTS) == "fail"
    noisy = [1.0, 1.1, 0.95, 1.05, 1.08]  # median 1.05, relative IQR about 0.076
    assert bb.paired_verdict([1.08] * 5, noisy, CONSTS) == "ambiguous"
    assert bb.paired_verdict([1.08] * 5, noisy, CONSTS, reruns_done=3) == "fail"
    assert bb.paired_verdict([1.055] * 5, noisy, CONSTS, reruns_done=2) == "ambiguous"
    assert (
        bb.paired_verdict([1.055] * 5, noisy, CONSTS, reruns_done=3) == "pass"
    )  # tie rule
    assert (
        bb.paired_verdict([1.005] * 5, [1.0] * 5, CONSTS) == "fail"
    )  # no noise, no band
    with pytest.raises(ValueError):
        bb.paired_verdict([1.0] * 4, base, CONSTS)


def test_cliff_and_noise_rules():
    assert bb.is_cliff(1.35, 1.0, CONSTS) and not bb.is_cliff(1.25, 1.0, CONSTS)
    assert bb.needs_noisy_rerun([1.0, 1.2, 0.9, 1.1, 1.0], CONSTS)
    assert not bb.needs_noisy_rerun([1.0, 1.0, 1.01, 0.99, 1.0], CONSTS)


def test_gate_constants_are_read_by_name(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"constants": CONSTS}))
    assert bb.load_gate_constants(p) == CONSTS
    p.write_text(json.dumps({"BAND_K": 1}))
    with pytest.raises(ValueError):
        bb.load_gate_constants(p)


def _attempt(i, rocke, bar, status="ok"):
    return {
        "id": "c",
        "attempt": i,
        "arms": {
            "rocke|-": {"status": status, "median_ms": rocke},
            "default|mha": {"status": "ok", "median_ms": bar},
        },
    }


def test_summarize_cell_discards_the_first_attempt_and_forms_a_verdict():
    cell = bb.Cell("c", 64, 8, 8, 1, 64, 64, "none", "bf16", "bshd")
    atts = [_attempt(0, 100.0, 1.0)] + [_attempt(i, 0.8, 1.0) for i in range(1, 6)]
    s = bb.summarize_cell(cell, atts, CONSTS)
    assert s["arms"]["rocke|-"]["n"] == 5 and s["verdict"] == "pass"
    bad = [
        _attempt(i, 0.8, 1.0, status="correctness_fail" if i == 3 else "ok")
        for i in range(6)
    ]
    assert bb.summarize_cell(cell, bad, CONSTS)["verdict"] == "correctness_fail"


# --------------------------------------------------------------------------- planning
def test_plan_shards_and_wall_estimate(tmp_path):
    assert bb.plan_shards([500, 500, 500], job_cap_s=1200, margin_s=200) == 2
    assert bb.plan_shards([], job_cap_s=1200) == 0
    assert (
        bb.cell_wall_estimate({"a": 2.0, "b": 3.0}, attempts=6, reference_s=1.0) == 36.0
    )
    rec = {
        "id": "x",
        "arms": {
            "default|native|aotriton": {"wall_s": 4.0},
            "flash|native|aotriton": {"wall_s": 2.0},
        },
    }
    p = tmp_path / "w.jsonl"
    p.write_text(json.dumps(rec) + "\n" + "not json\n")
    w = bb._read_wall_times(p)
    assert w["x"]["arms"] == {"default|native": 4.0, "flash|native": 2.0}


def test_results_dir_guard(tmp_path, monkeypatch):
    inside = Path(bb.__file__).resolve().parent / "x"
    with pytest.raises(SystemExit):
        bb.resolve_results_dir(str(inside))
    monkeypatch.setenv(bb.RESULTS_ENV, str(tmp_path / "r"))
    assert bb.resolve_results_dir(None) == (tmp_path / "r").resolve()
    monkeypatch.delenv(bb.RESULTS_ENV)
    with pytest.raises(SystemExit):
        bb.resolve_results_dir(None)


def test_completed_cells_resume(tmp_path):
    p = tmp_path / "cells.jsonl"
    bb.append_jsonl(p, {"id": "a", "complete": True})
    bb.append_jsonl(p, {"id": "b"})
    assert bb.completed_cells(p) == {"a"}


def test_tolerance_matches_the_case_table():
    from tests.sdpa.bwd_cases import ROW_FLOOR_BWD, TOLERANCE_BWD

    assert bb.TOLERANCE_BWD == TOLERANCE_BWD and bb.ROW_FLOOR_BWD == ROW_FLOOR_BWD


def test_cli_count_and_list_arms_print_no_measurements(capsys):
    assert bb.main(["--count"]) == 0
    assert bb.main(["--list-arms", "--core-only"]) == 0
    out = capsys.readouterr().out
    assert "census cells=84 core=20" in out
    assert not re.search(r"\d+\.\d+", out)


# --------------------------------------------------------------------------- reference
def _fp64_grads(problem, o_io=None):
    """float64 gradients on exactly the given inputs: ``Dsum`` from ``o_io``
    (the I/O-dtype O a backward consumes; the stored-inputs convention of the
    test oracle), or from the fp64 O when ``o_io`` is None."""
    torch = pytest.importorskip("torch")
    c = problem.cell
    q, k, v, do = (
        t.detach().double() for t in (problem.q, problem.k, problem.v, problem.do)
    )
    gk, gv = c.hq // c.hkv, c.hq // c.h_v

    def one(qq, kk, vv, dd, oo, mask):
        # qq / dd / oo: [H, Lq, D]; kk: [Hk, Lk, D]; vv: [Hv, Lk, D]
        ke, ve = kk.repeat_interleave(gk, 0), vv.repeat_interleave(gv, 0)
        s = torch.matmul(qq, ke.transpose(-1, -2)) * problem.scale
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        p = torch.softmax(s, -1).nan_to_num(0.0)
        o = torch.matmul(p, ve) if oo is None else oo
        delta = (dd * o).sum(-1, keepdim=True)
        ds = p * (torch.matmul(dd, ve.transpose(-1, -2)) - delta)
        dq = torch.matmul(ds, ke) * problem.scale
        dke = torch.matmul(ds.transpose(-1, -2), qq) * problem.scale
        dve = torch.matmul(p.transpose(-1, -2), dd)
        dk = dke.view(kk.shape[0], gk, *dke.shape[1:]).sum(1)
        dv = dve.view(vv.shape[0], gv, *dve.shape[1:]).sum(1)
        return dq, dk, dv

    oi = None if o_io is None else o_io.detach().double()
    if c.is_thd:
        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        oq = ok = 0
        for lq, lk in zip(c.seqlens, c.kv_lengths):
            sq, sk = slice(oq, oq + lq), slice(ok, ok + lk)
            t = lambda x, sl: x[sl].transpose(0, 1)  # noqa: E731
            m = bb.allowed_mask(c.mask, lq, lk, c.window, q.device)
            g = one(t(q, sq), t(k, sk), t(v, sk), t(do, sq),
                    None if oi is None else t(oi, sq), m)  # fmt: skip
            dq[sq], dk[sk], dv[sk] = (x.transpose(0, 1) for x in g)
            oq, ok = oq + lq, ok + lk
        return dq, dk, dv
    m = bb._dense_attn_mask(problem)
    outs = [
        one(q[b], k[b], v[b], do[b], None if oi is None else oi[b],
            None if m is None else (m[b, 0] if m.dim() == 4 else m))
        for b in range(c.b)
    ]  # fmt: skip
    return tuple(torch.stack([o[i] for o in outs]) for i in range(3))


REF_CELLS = [
    bb.Cell("gqa_br", 32, 4, 2, 2, 24, 40, "causal_br", "bf16", "bshd"),
    bb.Cell("swa_bhsd", 32, 4, 4, 1, 40, 40, "swa", "fp16", "bhsd", window=7),
    bb.Cell("tl_unequal", 32, 4, 1, 1, 16, 48, "causal_tl", "bf16", "bshd"),
    bb.Cell("hkv_ne_hv", 32, 4, 2, 1, 20, 20, "none", "bf16", "bshd", hv=1),
    bb.Cell(
        "padded",
        32,
        2,
        2,
        3,
        20,
        28,
        "causal_br",
        "bf16",
        "bshd",
        padded_q=(20, 9, 1),
        padded_kv=(28, 13, 5),
    ),
    bb.Cell("thd", 32, 4, 2, 3, 17, 17, "causal_tl", "fp16", "thd", seqlens=(5, 17, 1)),
    bb.Cell("odd_stride", 32, 2, 2, 1, 16, 16, "none", "bf16", "bshd", stage_vec=1),
]


@pytest.mark.parametrize("cell", REF_CELLS, ids=lambda c: c.id)
def test_chunked_reference_matches_fp64_autograd(cell):
    pytest.importorskip("torch")
    problem = bb.make_problem(cell, seed=3, device="cpu")
    ref = bb.reference_backward(problem, q_block=7, head_block=3)
    want = _fp64_grads(problem, ref.o.to(problem.q.dtype))
    rows = {
        "dq": bb._valid_rows(problem, "dq"),
        "dk": bb._valid_rows(problem, "dk"),
        "dv": bb._valid_rows(problem, "dv"),
    }
    for name, got, w in zip(("dq", "dk", "dv"), (ref.dq, ref.dk, ref.dv), want):
        w = w.float()
        r = rows[name]
        if r is not None:
            assert float(got[~r].abs().max()) == 0.0  # padded rows are exact zeros
            got, w = got[r], w[r]
        err = float((got - w).abs().max())
        assert err <= 1e-4 * max(1.0, float(w.abs().max())), (name, err)
    assert bb.compare_to_reference(problem, (ref.dq, ref.dk, ref.dv), ref) == []
    bumped = ref.dk.clone()
    bumped[(0,) * bumped.dim()] += 10.0 * float(ref.dk.abs().max()) + 1.0
    assert bb.compare_to_reference(problem, (ref.dq, bumped, ref.dv), ref)
    # An error confined to the smallest valid dq row: only the per-row check sees it.
    t = bb.TOLERANCE_BWD[cell.dtype]
    g = bb.kv_growth(cell)
    mags = ref.dq.abs().amax(dim=-1)
    rows = bb._valid_rows(problem, "dq")
    mags = mags.masked_fill(~rows, float("inf")) if rows is not None else mags
    flat = int(mags.reshape(-1).argmin())
    idx = tuple(int(i) for i in np.unravel_index(flat, tuple(mags.shape)))
    mx = float(ref.dq.abs().max())
    row_bound = t["atol"] + t["rtol"] * g * max(float(mags[idx]), bb.ROW_FLOOR_BWD * mx)
    if 2 * row_bound >= t["atol"] + t["rtol"] * g * mx:
        pytest.skip("no row small enough to separate the two bounds")
    dq_row = ref.dq.clone()
    dq_row[idx + (0,)] += 2 * row_bound
    assert bb.compare_to_reference(problem, (dq_row, ref.dk, ref.dv), ref) == [
        "dq: row"
    ]
    assert (
        bb.compare_to_reference(problem, (dq_row, ref.dk, ref.dv), ref, row_check=False)
        == []
    )


def test_odd_stride_cells_are_not_eight_aligned():
    pytest.importorskip("torch")
    p = bb.make_problem(REF_CELLS[-1], seed=1, device="cpu")
    assert p.q.stride(-1) == 1 and any(s % 8 for s in p.q.stride()[:-1])


def test_kv_growth():
    assert (
        bb.kv_growth(bb.Cell("x", 64, 1, 1, 1, 1, 512, "none", "bf16", "bshd")) == 1.0
    )
    assert math.isclose(
        bb.kv_growth(bb.Cell("x", 64, 1, 1, 1, 1, 4096, "none", "bf16", "bshd")), 2.0
    )


def test_reference_dsum_uses_the_io_dtype_output():
    """The reference follows the stored-inputs convention: with the exact O it
    would differ on rows that attend two keys (near-cancelling dQ)."""
    torch = pytest.importorskip("torch")
    cell = bb.Cell("tl", 32, 2, 2, 1, 24, 24, "causal_tl", "bf16", "bshd")
    problem = bb.make_problem(cell, seed=7, device="cpu")
    ref = bb.reference_backward(problem)
    io = _fp64_grads(problem, ref.o.to(problem.q.dtype))
    exact = _fp64_grads(problem)
    err_io = float((ref.dq.double() - io[0]).abs().max())
    err_exact = float((ref.dq.double() - exact[0]).abs().max())
    assert err_io < 1e-4 * float(io[0].abs().max())
    assert err_exact > 10 * err_io
    assert torch.isfinite(ref.dq).all()


@pytest.mark.parametrize("cell", REF_CELLS, ids=lambda c: c.id)
def test_fp64_samples_match_the_fp64_oracle(cell):
    torch = pytest.importorskip("torch")
    problem = bb.make_problem(cell, seed=5, device="cpu")
    ref = bb.reference_backward(problem, q_block=5, head_block=2)
    o_io = ref.o.to(problem.q.dtype)
    smp = bb.fp64_samples(problem, o_io, n_rows=5, n_keys=4, row_block=7)
    want = _fp64_grads(problem, o_io)
    for name, w in zip(("dq", "dk", "dv"), want):
        idx = getattr(smp, f"{name}_idx")
        assert idx, name
        got = getattr(smp, name)
        assert torch.allclose(bb._gather(w, idx), got, rtol=1e-9, atol=1e-12), name
    assert bb.reference_self_check(ref, smp) == []
    grads = (ref.dq, ref.dk, ref.dv)
    assert bb.compare_to_samples(problem, grads, smp, ref) == []
    bad = ref.dk.clone()
    i = smp.dk_idx[0]
    bad[i] += 10.0 * float(ref.dk.abs().max()) + 1.0
    assert bb.compare_to_samples(problem, (ref.dq, bad, ref.dv), smp, ref) == [
        "dk: fp64_global"
    ]
    nan = ref.dv.clone()
    nan[smp.dv_idx[-1]] = float("nan")
    assert "dv: fp64_non_finite" in bb.compare_to_samples(
        problem, (ref.dq, ref.dk, nan), smp, ref
    )


def test_fp64_sample_threshold_and_spread():
    big = bb.Cell("b", 128, 32, 8, 1, 8192, 8192, "causal_tl", "bf16", "bshd")
    small = bb.Cell("s", 64, 12, 12, 8, 1024, 1024, "none", "bf16", "bshd")
    assert bb.needs_fp64_samples(big) and not bb.needs_fp64_samples(small)
    thd = bb.Cell("t", 64, 4, 4, 2, 5, 5, "none", "bf16", "thd", seqlens=(5, 3))
    assert bb.score_elements(thd) == 4 * (25 + 9)
    assert bb._spread(1, 16) == [0]
    s = bb._spread(1000, 16)
    assert s[:2] == [0, 1] and s[-1] == 999 and len(set(s)) == len(s) <= 16


def test_reference_cache_roundtrip_checks_the_inputs(tmp_path):
    pytest.importorskip("torch")
    cell = REF_CELLS[0]
    problem = bb.make_problem(cell, seed=3, device="cpu")
    ref = bb.reference_backward(problem)
    smp = bb.fp64_samples(problem, ref.o.to(problem.q.dtype), n_rows=3, n_keys=3)
    bb.save_reference(tmp_path, cell, problem, ref, smp, {"x": 1})
    again = bb.make_problem(cell, seed=3, device="cpu")
    r2, s2, extra = bb.load_reference(tmp_path, cell, again)
    assert extra == {"x": 1} and float((r2.dq - ref.dq).abs().max()) == 0.0
    assert s2.dk_idx == smp.dk_idx
    other = bb.make_problem(cell, seed=4, device="cpu")
    with pytest.raises(RuntimeError):
        bb.load_reference(tmp_path, cell, other)


# --------------------------------------------------------------------------- summaries
def _att(i, arms, block=0):
    return {"id": "c", "attempt": i, "block": block, "arms": arms}


def _ok(ms):
    return {"status": "ok", "median_ms": ms}


def test_summary_verdicts_per_rocke_arm_blocks_and_bar_substitution():
    cell = bb.Cell("c", 64, 8, 2, 1, 64, 64, "none", "bf16", "bshd")
    atts = []
    for blk in (0, 1):
        for i in range(6):
            atts.append(
                _att(
                    i,
                    {
                        "rocke|a": _ok(100.0 if i == 0 else 0.8),
                        "rocke|b": _ok(100.0 if i == 0 else 1.5),
                        "default|native": {"status": "correctness_fail"},
                        "default|expand": _ok(1.0),
                    },
                    blk,
                )
            )
    s = bb.summarize_cell(cell, atts, CONSTS, reruns_done=1)
    assert s["arms"]["rocke|a"]["n"] == 10  # first attempt of each block dropped
    assert s["bar_arm"] == "default|expand" and s["bar_substituted"]
    assert s["verdicts"] == {"rocke|a": "pass", "rocke|b": "fail"}
    assert "verdict" not in s and bb.verdict_token(s) == "verdicts=fail:1,pass:1"
    bare = bb.summarize_cell(cell, atts, None)
    assert bb.verdict_token(bare) == "unjudged"
    comp = {"arms": {"default|native": {"status": "ok"}}}
    assert bb.verdict_token(comp) == "competitor_only"
    none = [_att(i, {"rocke|a": _ok(1.0), "default|native": {"status": "error"},
                     "default|expand": {"status": "error"}}) for i in range(6)]  # fmt: skip
    assert bb.summarize_cell(cell, none, CONSTS)["verdict"] == "no_valid_competitor"
    uns = [_att(i, {"rocke|a": {"status": "unsupported"},
                    "default|native": _ok(1.0)}) for i in range(6)]  # fmt: skip
    assert bb.summarize_cell(cell, uns, CONSTS)["verdict"] == "unsupported"


def test_latest_summaries_and_bar_order(tmp_path):
    p = tmp_path / "cells.jsonl"
    bb.append_jsonl(p, {"id": "a", "complete": True, "n": 1})
    bb.append_jsonl(p, {"id": "a", "complete": True, "n": 2})
    bb.append_jsonl(p, {"id": "b"})
    assert bb.latest_summaries(p) == {"a": {"id": "a", "complete": True, "n": 2}}
    g = bb.Cell("x", 64, 8, 2, 1, 64, 64, "none", "bf16", "bshd")
    assert bb.bar_fallbacks(g) == ["default|native", "default|expand"]
    t = bb.Cell("x", 64, 8, 8, 2, 4, 4, "none", "bf16", "thd", seqlens=(3, 4))
    assert bb.bar_fallbacks(t) == ["varlen_flash|-", "njt|default"]
    args = bb.make_parser().parse_args(["--competitor-arms", "default"])
    assert bb.competitor_arms(g, args) == ["default|native", "default|expand"]
    assert bb.make_parser().parse_args([]).competitor_row_check == "off"


class _GateObj:
    def check(self, *a):
        return []


GATE_OBJ = _GateObj()


def _gate_factory():
    return _GateObj()


def test_load_gate_accepts_objects_and_factories():
    assert bb.load_gate(None) is None
    assert bb.load_gate(f"{__name__}:GATE_OBJ") is GATE_OBJ
    assert isinstance(bb.load_gate(f"{__name__}:_gate_factory"), _GateObj)


def test_cli_refuses_rocke_arms_without_the_gate(tmp_path):
    with pytest.raises(SystemExit):
        bb.main(["--ids", "x", "--out-dir", str(tmp_path), "--arch", "gfx942",
                 "--rocke-configs", "shipped"])  # fmt: skip


def test_plan_reads_cell_walls(tmp_path, capsys):
    p = tmp_path / "cells.jsonl"
    ids = [c.id for c in bb.census_cells(bb.load_census())[:3]]
    for cid in ids:
        bb.append_jsonl(p, {"id": cid, "complete": True, "cell_wall_s": 400.0,
                            "attempts": 6, "reference_s": 5.0, "arms": {}})  # fmt: skip
    assert bb.main(["--plan", "--wall-times", str(p), "--ids", ",".join(ids)]) == 0
    out = capsys.readouterr().out.strip()
    assert out == "plan cells=3 with_wall_times=3 missing=0 shards=2"


def test_main_reruns_only_ambiguous_cells_with_a_new_block(tmp_path, monkeypatch):
    cells = bb.census_cells(bb.load_census())[:3]
    ids = ",".join(c.id for c in cells)
    p = tmp_path / "cells.jsonl"
    verdicts = {cells[0].id: "ambiguous", cells[1].id: "pass", cells[2].id: "ambiguous"}
    for c in cells:
        bb.append_jsonl(p, {"id": c.id, "complete": True, "blocks": [0],
                            "reruns_done": 0, "arms": {},
                            "verdicts": {"rocke|shipped": verdicts[c.id]}})  # fmt: skip
    # the third cell already used every rerun
    bb.append_jsonl(p, {"id": cells[2].id, "complete": True, "blocks": [0, 1, 2],
                        "reruns_done": 2, "arms": {},
                        "verdicts": {"rocke|shipped": "ambiguous"}})  # fmt: skip
    seen = []

    def fake_run_cell(c, args, out_dir, constants, *, block, reruns_done):
        seen.append((c.id, block, reruns_done))
        return {"id": c.id, "complete": True, "arms": {}, "blocks": [0, block],
                "reruns_done": reruns_done, "cell_wall_s": 1.0,
                "verdicts": {"rocke|shipped": "pass"}}  # fmt: skip

    monkeypatch.setattr(bb, "_run_cell", fake_run_cell)
    monkeypatch.setattr(bb, "environment_record", lambda: {})
    monkeypatch.setattr(bb, "_clocks", lambda: None)
    monkeypatch.setattr(bb, "toolchain_problems", lambda a: [])
    g = tmp_path / "g.json"
    g.write_text(json.dumps({"constants": CONSTS}))
    argv = ["--ids", ids, "--out-dir", str(tmp_path / "o"), "--gate-constants", str(g),
            "--ref-dir", str(tmp_path / "ref")]  # fmt: skip
    (tmp_path / "o").mkdir()
    (tmp_path / "o" / "cells.jsonl").write_text(p.read_text())
    assert bb.main(argv) == 0 and seen == []  # nothing to do without the flag
    consts = dict(CONSTS, RERUN_MAX=2)
    g.write_text(json.dumps({"constants": consts}))
    assert bb.main(argv + ["--rerun-ambiguous"]) == 0
    assert seen == [(cells[0].id, 1, 1)]
    latest = bb.latest_summaries(tmp_path / "o" / "cells.jsonl")
    assert latest[cells[0].id]["verdicts"] == {"rocke|shipped": "pass"}


def test_main_deadline_stops_before_a_cell_that_would_not_fit(tmp_path, monkeypatch):
    cells = bb.census_cells(bb.load_census())[:3]
    seen = []
    clock = [0.0]

    def fake_run_cell(c, args, out_dir, constants, *, block, reruns_done):
        seen.append(c.id)
        clock[0] += 60.0
        return {"id": c.id, "complete": True, "arms": {}, "cell_wall_s": 60.0}

    monkeypatch.setattr(bb, "_run_cell", fake_run_cell)
    monkeypatch.setattr(bb.time, "time", lambda: clock[0])
    monkeypatch.setattr(bb, "environment_record", lambda: {})
    monkeypatch.setattr(bb, "_clocks", lambda: None)
    monkeypatch.setattr(bb, "toolchain_problems", lambda a: [])
    argv = ["--ids", ",".join(c.id for c in cells), "--out-dir", str(tmp_path),
            "--deadline-s", "100", "--ref-dir", str(tmp_path / "ref")]  # fmt: skip
    assert bb.main(argv) == 0
    assert seen == [cells[0].id]  # 60 s done + 60 s longest > 100 s
    assert bb.main(argv) == 0 and seen == [cells[0].id, cells[1].id]  # resumes


# --------------------------------------------------------------------------- arm order,
# same-kernel control and backend capture
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7])
def test_balanced_orders_balance_positions_and_neighbours(n):
    rows = bb.balanced_orders(n)
    assert len(rows) == (n if n % 2 == 0 or n == 1 else 2 * n)
    for row in rows:
        assert sorted(row) == list(range(n))
    for arm in range(n):
        per_pos = collections.Counter(row.index(arm) for row in rows)
        assert len(set(per_pos.values())) == 1 and len(per_pos) == n
    if n >= 2:
        pairs = collections.Counter(
            (row[i], row[i + 1]) for row in rows for i in range(n - 1)
        )
        assert len(pairs) == n * (n - 1)  # every ordered pair is adjacent
        assert len(set(pairs.values())) == 1  # equally often


def test_arm_order_is_not_a_rotation():
    arms = ["a", "b", "c"]
    orders = [tuple(bb.arm_order(arms, k)) for k in range(6)]
    assert len(set(orders)) == 6  # rotations alone give only 3
    adjacent = collections.Counter((o[i], o[i + 1]) for o in orders for i in range(2))
    assert adjacent[("a", "b")] == adjacent[("b", "a")]


def _summary_with(aa, base):
    return {
        "arms": {
            "rocke|x": {"status": "ok", "attempt_medians_ms": base},
            "rocke|x#aa": {"status": "ok", "attempt_medians_ms": aa},
        }
    }


def test_aa_control_flags_a_copy_that_differs():
    k = {"BAND_K": 2.0, "TIE_EPS": 0.005, "RERUN_MAX": 2}
    same = [1.0, 1.01, 0.99, 1.0, 1.02]
    assert bb.aa_control(_summary_with(same, same), k) == "ok"
    slow = [1.3, 1.31, 1.29, 1.3, 1.32]
    assert bb.aa_control(_summary_with(slow, same), k) == "fail"
    assert bb.aa_control(_summary_with(same, slow), k) == "fail"  # either way
    short = {"arms": {"rocke|x": {"status": "ok", "attempt_medians_ms": same}}}
    assert bb.aa_control(short, k) is None
    broken = _summary_with(same, same)
    broken["arms"]["rocke|x#aa"]["status"] = "error"
    assert bb.aa_control(broken, k) == "incomplete"


def test_aa_controls_judge_every_twin():
    """``--aa-control all``: a twin per configuration, a verdict per arm, and
    the cell's control fails when any twin fails (incomplete before ok)."""
    k = {"BAND_K": 2.0, "TIE_EPS": 0.005, "RERUN_MAX": 2}
    same = [1.0, 1.01, 0.99, 1.0, 1.02]
    slow = [1.3, 1.31, 1.29, 1.3, 1.32]
    s = _summary_with(same, same)
    s["arms"]["rocke|y"] = {"status": "ok", "attempt_medians_ms": same}
    s["arms"]["rocke|y#aa"] = {"status": "ok", "attempt_medians_ms": slow}
    assert bb.aa_controls(s, k) == {"rocke|x": "ok", "rocke|y": "fail"}
    assert bb.aa_control(s, k) == "fail"
    s["arms"]["rocke|y#aa"] = {"status": "error"}
    assert bb.aa_controls(s, k)["rocke|y"] == "incomplete"
    assert bb.aa_control(s, k) == "incomplete"
    s["arms"]["rocke|y#aa"] = {"status": "ok", "attempt_medians_ms": same}
    assert bb.aa_control(s, k) == "ok"


def test_rocke_arms_add_the_twins():
    cfgs = [type("C", (), {"label": n})() for n in ("x", "y")]
    arms, by = bb.rocke_arms(cfgs, "on")
    assert arms == ["rocke|x", "rocke|y", "rocke|x#aa"]
    assert by["rocke|x#aa"] is cfgs[0]
    arms, by = bb.rocke_arms(cfgs, "all")
    assert arms == ["rocke|x", "rocke|y", "rocke|x#aa", "rocke|y#aa"]
    assert by["rocke|y#aa"] is cfgs[1]
    assert bb.rocke_arms(cfgs, "off")[0] == ["rocke|x", "rocke|y"]
    assert bb.rocke_arms([], "all") == ([], {})
    with pytest.raises(ValueError):
        bb.rocke_arms(cfgs, "both")


def test_aa_arm_never_gets_a_verdict_and_flags_the_token():
    assert bb.is_aa_arm("rocke|x#aa") and not bb.is_aa_arm("rocke|x")
    s = {"verdict": "pass", "flagged": True}
    assert bb.verdict_token(s) == "aa_fail:pass"
    assert bb.verdict_token({"verdict": "pass"}) == "pass"


def test_order_imbalance_counts_positions():
    att = [{"arm_order": o} for o in (["a", "b"], ["b", "a"])]
    assert bb.order_imbalance(att) == 0
    assert bb.order_imbalance(att[:1] * 3) == 3
    assert bb.order_imbalance([]) == 0


@pytest.mark.parametrize(
    "names, label",
    [
        (["attn_bwd_kernel", "bwd_preprocess"], "flash"),
        (["aotriton::bwd_kernel_dk_dv"], "flash"),
        (["fmha_bwd_d128_bf16"], "efficient"),
        (["Cijk_Alik_Bljk_gemm", "softmax_warp_backward"], "math"),
        ([], "unknown"),
    ],
)
def test_selected_backend_labels(names, label):
    assert bb.selected_backend(names) == label


def test_parser_has_the_aa_control_flag_and_forwards_it():
    args = bb.make_parser().parse_args([])
    assert args.aa_control == "on"
    assert bb.make_parser().parse_args(["--aa-control", "off"]).aa_control == "off"
    assert bb.make_parser().parse_args(["--aa-control", "all"]).aa_control == "all"


# --------------------------------------------------------------------------- environment
def test_gate_has_no_environment_default(monkeypatch):
    """``--gate`` is explicit: an exported variable does not select a gate."""
    monkeypatch.setenv("ROCKE_BWD_PRUNE_GATE", f"{__name__}:GATE_OBJ")
    assert bb.make_parser().parse_args([]).gate is None
    args = bb.make_parser().parse_args(["--gate", f"{__name__}:GATE_OBJ"])
    assert args.gate == f"{__name__}:GATE_OBJ"


class _NoCudaTorch:
    """A torch stand-in without ``cuda``: a device query through it fails."""

    __version__ = "0.0-test"

    class version:  # noqa: N801 - mirrors torch.version
        hip = "0.0-test"


def test_environment_record_reads_the_device_through_the_rocke_runtime(monkeypatch):
    from rocke.runtime import hip_module as hm

    calls = []

    def fake(name, value):
        def f(device=0):
            calls.append((name, device))
            return value

        monkeypatch.setattr(hm, name, f)

    fake("get_device_target_id", "gfx950:sramecc+:xnack-")
    fake("get_device_arch", "gfx950")
    fake("get_device_num_cus", 7)

    def torch_stub():
        calls.append(("torch", None))
        return _NoCudaTorch

    monkeypatch.setattr(bb, "_torch", torch_stub)
    monkeypatch.setattr(bb, "_toolchain_record", lambda: {})
    monkeypatch.setattr(bb, "_clocks", lambda: None)
    rec = bb.environment_record()
    assert rec["gcn_arch"] == "gfx950:sramecc+:xnack-"
    assert rec["device_arch"] == "gfx950" and rec["cu_count"] == 7
    # the run's requested arch is recorded by the caller, never overwritten
    assert "arch" not in rec
    assert "device_name" not in rec  # the device is named by its gfx id only
    assert rec["torch"] == "0.0-test" and "torch_error" not in rec
    # torch is imported (versions only) before the rocKE runtime is queried,
    # so the runtime binds torch's HIP library
    assert calls[0] == ("torch", None)
    assert {d for n, d in calls[1:]} == {0} and len(calls) == 4


def test_bench_makes_no_torch_device_query():
    src = Path(bb.__file__).read_text(encoding="utf-8")
    for query in ("get_device_properties", "multi_processor_count", "gcnArchName"):
        assert query not in src, query
