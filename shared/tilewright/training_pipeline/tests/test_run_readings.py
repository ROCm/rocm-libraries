# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""lib/run_readings.py: the report-only readings of a bundle on held-out
GEMMs, on synthetic rows and with a stand-in evaluation."""

import math
from types import SimpleNamespace

import pytest

from lib import run_readings as rr


def row(winner, pick, screened=False, sid=1):
    return {
        "winner_us": winner,
        "pick_us": pick,
        "sel_eff": None if pick is None else winner / pick,
        "screened": screened,
        "pick_sol_idx": sid,
    }


def test_summary_reads_rw_the_long_decile_and_the_tails():
    rows = [row(10.0, 10.0) for _ in range(7)]
    rows += [row(100.0, 125.0), row(1.0, 1.4, screened=True), row(2.0, 4.5)]
    rows += [row(5.0, None), {"winner_us": 0.0, "pick_us": 1.0, "sel_eff": 0.0}]
    s = rr.summarize(rows)
    assert s["n_gemms"] == 12 and s["n_evaluated"] == 10
    effs = [1.0] * 7 + [0.8, 1 / 1.4, 2 / 4.5]
    assert s["geomean"] == pytest.approx(math.exp(sum(map(math.log, effs)) / 10))
    assert s["rw"] == pytest.approx((70 + 100 + 1 + 2) / (70 + 125 + 1.4 + 4.5))
    # the tenth with the longest winners: one GEMM of ten
    assert s["long_decile"] == {"n": 1, "winner_us_min": 100.0, "geomean": 0.8}
    # 0.8 is not below 0.8; skip-screened picks are counted apart
    assert s["below"] == {
        "0.8": {"timed": 1, "screened": 1},
        "0.5": {"timed": 1, "screened": 0},
    }
    empty = rr.summarize([])
    assert empty["n_evaluated"] == 0 and empty["geomean"] is None
    assert empty["rw"] is None and empty["long_decile"]["n"] == 0
    eleven = rr.summarize([row(float(w), float(w)) for w in range(1, 12)])
    assert eleven["long_decile"]["n"] == 2
    assert eleven["long_decile"]["winner_us_min"] == 10.0
    thirty = rr.summarize([row(float(w), float(w)) for w in range(1, 31)])
    assert thirty["long_decile"]["n"] == 3


def _result(m, pick, pick_us, winner=10.0):
    return SimpleNamespace(
        m=m,
        n=64,
        k=1000,
        batch=1,
        model_cell="c",
        pick_sol_idx=pick,
        pick_us=pick_us,
        served_by="model",
        miss=None,
        sel_eff=None if pick_us is None else winner / pick_us,
    )


def _gemm(m, skip=()):
    cands = [
        {"sol_idx_global": s, "us": us, "is_skip": s in skip}
        for s, us in ((1, 10.0), (2, 20.0), (3, 12.5))
    ]
    return {"m": m, "winner_us": 10.0, "candidates": cands, "cu_budget": 128}


def test_a_row_names_a_skip_screened_pick_and_the_context():
    r = rr.gemm_row(_result(8, 3, 12.5), _gemm(8, skip=(3,)))
    assert r["screened"] is True and r["sel_eff"] == pytest.approx(0.8)
    assert (r["m"], r["cu_budget"], r["schedule"], r["cell"]) == (
        8,
        128,
        "default",
        "c",
    )
    assert rr.gemm_row(_result(8, 3, 12.5), _gemm(8))["screened"] is False
    none = rr.gemm_row(None, _gemm(8))
    assert none["sel_eff"] is None and none["pick_sol_idx"] is None


def test_bundle_readings_add_the_veto_only_from_the_config():
    veto_key = {"deploy": {"alignment_veto": {"k_min": 512, "align_bytes": 128}}}
    gemms = [_gemm(m) for m in (8, 16, 32)]
    seen = []

    def evaluate(bundle):
        veto = bundle["serving"].get("alignment_veto")
        seen.append(veto)
        # the veto moves the second GEMM's pick to a slower kernel
        return [
            _result(8, 1, 10.0),
            _result(16, 2, 20.0) if veto else _result(16, 1, 10.0),
            _result(32, 3, 12.5),
        ]

    v3 = {"catalog": "v3", "serving": {"cascade_keys": 8}}
    plain = rr.bundle_readings(v3, gemms, {}, evaluate)
    assert seen == [None] and plain["veto"] is None and plain["with_veto"] is None
    assert plain["status"] == rr.STATUS and plain["n_changed"] is None
    assert len(plain["rows"]) == 3 and "veto_pick_sol_idx" not in plain["rows"][0]
    seen.clear()
    out = rr.bundle_readings(v3, gemms, veto_key, evaluate)
    assert seen == [None, {"k_min": 512, "align_bits": 1024}]
    assert out["veto"] == {"k_min": 512, "align_bits": 1024}
    assert out["n_changed"] == 1
    assert out["without_veto"]["geomean"] == pytest.approx((1 * 1 * 0.8) ** (1 / 3))
    assert out["with_veto"]["geomean"] == pytest.approx((1 * 0.5 * 0.8) ** (1 / 3))
    assert [r["veto_pick_sol_idx"] for r in out["rows"]] == [1, 2, 3]
    assert out["rows"][1]["veto_sel_eff"] == pytest.approx(0.5)
    # the decisions' own results are not evaluated again
    seen.clear()
    given = [_result(8, 1, 10.0), _result(16, 1, 10.0), _result(32, 3, 12.5)]
    out = rr.bundle_readings(v3, gemms, veto_key, evaluate, given)
    assert seen == [{"k_min": 512, "align_bits": 1024}]
    assert out["without_veto"] == plain["without_veto"]
    # a bundle the round carries with a veto is read without it; a v2 bundle
    # never with one
    seen.clear()
    carried = dict(v3, serving={"alignment_veto": {"k_min": 1, "align_bits": 8}})
    rr.bundle_readings(carried, gemms, {}, evaluate)
    assert seen == [None]
    seen.clear()
    v2 = rr.bundle_readings({"serving": {}}, gemms, veto_key, evaluate)
    assert seen == [None] and v2["with_veto"] is None
    assert "rows" not in rr.brief(out) and rr.brief(None) is None
    assert rr.failed("why") == {
        "schema": rr.SCHEMA,
        "status": rr.STATUS,
        "error": "why",
    }
