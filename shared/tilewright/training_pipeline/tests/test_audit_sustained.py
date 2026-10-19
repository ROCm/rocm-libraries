# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The sustained composite without a bench: its measurement's plan
(lib/audit_sustained.py) and the protocol a number, a paired change, the
ship decision and the per-cell gate take (lib/audit.py `preference`). The
bench and the scores on the fake world are in test_audit_v2.py."""

import json
import math

import pytest

from lib import audit as al
from lib import audit_sustained as asus
from lib import audit_v2 as v2
from lib.bench_yaml import line_solution_list

EX = al.EXCLUSIVE


def _line(shape, kernels=()):
    m, n, k, b = shape
    return (
        f"- {{M: {m}, N: {n}, K: {k}, batch_count: {b}, adaptive: true, "
        f"solution_index_list: [{', '.join(str(x) for x in kernels)}]}}"
    )


# ── the sustained measurement's plan ─────────────────────────────────────────

# by their v2 winners: not power-capped, long, very long, compute-bound
SMALL = (64, 64, 64, 1)
LONG = (1024, 1024, 128, 1)
XL = (4096, 4096, 64, 1)
CB = (8192, 8192, 8192, 1)


def _gemm(shape, rows):
    m, n, k, b = shape
    cands = [dict(sol_idx_global=s, us=u, is_skip=False) for s, u in rows.items()]
    return dict(m=m, n=n, k=k, batch_count=b, candidates=cands)


def _meas(median, gen=0):
    return v2.Measurement(gen, sorted(median), dict(median), False, False)


@pytest.fixture
def planned_v2(tmp_path):
    """A v2 measurement (planned, not benched) with one model registered,
    the exhaustive rows of four exclusive GEMMs and their v2 medians."""
    medians = {
        SMALL: {1: 50.0, 2: 52.0, 3: 90.0},
        LONG: {1: 150.0, 2: 160.0, 3: 165.0, 4: 300.0},
        XL: {1: 2000.0, 2: 2100.0, 3: 2400.0},
        CB: {1: 1000.0, 2: 1100.0, 3: 1140.0, 4: 1200.0},
    }
    gemms = {s: _gemm(s, {**m, 9: 999999.0}) for s, m in medians.items()}
    d = tmp_path / "v2"
    picks = {(EX, s): 3 for s in medians}
    picks[(EX, LONG)] = 9
    v2.plan(
        d,
        gemms={
            (EX, s): dict(g, winner_us=min(medians[s].values()))
            for s, g in gemms.items()
        },
        lines={s: _line(s) for s in medians},
        parts={EX: [list(medians)]},
        new_models=[v2.Model("m1", "1" * 64, "x", picks)],
    )
    return {
        "d": d,
        "gemms": gemms,
        "measured": {s: _meas(m) for s, m in medians.items()},
        "medians": medians,
    }


def _plan(w, **kw):
    sd = w["d"] / asus.DIRNAME
    return sd, asus.plan(
        sd,
        v2d=w["d"],
        gemms=w["gemms"],
        measured=w["measured"],
        config={},
        **kw,
    )


def test_no_sustained_set_holds_a_blocked_kernel(planned_v2):
    """A kernel the plan's blocklist names is in no set and no pick, even
    one a planned set held, and the plan names the blocklist; a blocked
    canary refuses it."""
    rec = {"file": "blocklists/plan_x.json", "sha256": "b" * 64, "kernels": 1}
    sd, gen = _plan(planned_v2, peak_tflops=2000.0, blocked={2}, blocklist=rec)
    assert gen == 0
    doc = json.loads((sd / "gen_000" / "plan.json").read_text())
    entries = [e for part in doc["groups"][EX] for e in part]
    assert entries and all(2 not in e["kernels"] for e in entries)
    assert all(2 not in e["candidates"] for e in entries)
    assert all(2 not in line_solution_list(e["line"]) for e in entries)
    assert doc["blocklist"] == rec and v2.manifest(sd)["blocklist"] == rec
    canary = v2.manifest(planned_v2["d"])["canary"]["kernel"]
    with pytest.raises(v2.V2Error, match="canary"):
        _plan(planned_v2, peak_tflops=2000.0, blocked={canary})


def test_the_strata_follow_the_v2_winner_and_the_devices_peak():
    st = {**asus.DEFAULTS, "compute_bound_tflops": 500.0}
    assert asus.stratum(SMALL, 50.0, st) == ""
    assert asus.stratum(LONG, 150.0, st) == "long"
    assert asus.stratum(XL, 2000.0, st) == "very_long"
    assert asus.stratum(CB, 1000.0, st) == "compute_bound"
    assert asus.stratum(CB, 99.0, st) == ""  # fast enough not to hold the limit
    rate, note = asus.compute_bound_rate(asus.DEFAULTS, 2000.0, "peak note")
    assert rate == pytest.approx(0.3 * 2000.0) and "peak note" in note
    rate, note = asus.compute_bound_rate(asus.DEFAULTS, None, "no MI entry")
    assert rate is None and "no GEMM is compute-bound" in note
    given = {**asus.DEFAULTS, "compute_bound_tflops": 42.0}
    assert asus.compute_bound_rate(given, 2000.0)[0] == 42.0
    assert asus.stratum(CB, 500.0, {**asus.DEFAULTS}) == "long"


def test_the_first_plan_sets_power_capped_gemms_near_winners_and_picks(planned_v2):
    sd, gen = _plan(
        planned_v2, peak_tflops=2000.0, settings={"part_gemms": 2}, peak_note="x"
    )
    assert gen == 0
    m = v2.manifest(sd)
    assert m["regime"] == "sustained" and "x" in m["compute_bound"]
    st = v2.settings_of(m)
    assert st["compute_bound_tflops"] == pytest.approx(600.0)
    assert (st["repetitions"], st["min_readings"]) == (3, 2)
    assert st["kernel_order"] == "shuffled" and st["sacrificial_kernel"] is False
    assert m["canary"] == v2.manifest(planned_v2["d"])["canary"]
    entries = asus.current_entries(sd)
    assert set(entries) == {LONG, XL, CB}
    assert {s: e["stratum"] for s, e in entries.items()} == {
        LONG: "long",
        XL: "very_long",
        CB: "compute_bound",
    }
    # 8% of the winner, then the pick; the model's pick on LONG is a kernel
    # the exhaustive pass benched (row 9), so it is in the set too
    assert entries[LONG]["kernels"] == [1, 2, 9]
    assert entries[XL]["kernels"] == [1, 2, 3]
    # compute-bound: 15% of the winner
    assert entries[CB]["kernels"] == [1, 2, 3]
    assert entries[CB]["picks"] == {"m1": 3}
    for s, e in entries.items():
        ln = e["line"]
        assert line_solution_list(ln) == e["kernels"]
        for key, val in (("warmup_time", 250), ("measure_time", 300)):
            assert f"{key}: {val}" in ln or f"{key}: {float(val)}" in ln
        assert "skip_slow_solution_ratio: 0" in ln and "adaptive: true" in ln
        assert e["v2_winner"] == [1, planned_v2["medians"][s][1]]
    plan = json.loads((v2.gen_dir(sd, 0) / "plan.json").read_text())
    parts = plan["groups"][EX]
    assert [len(p) for p in parts] == [2, 1]
    assert m["generations"][0]["strata"] == {
        "long": 1,
        "very_long": 1,
        "compute_bound": 1,
    }
    assert v2.pending(sd).keys() == {(EX, s) for s in entries}


def test_the_configs_set_floor_widens_the_first_sustained_plan(planned_v2):
    """scripts/audit_v2.py gives the sustained first plan the config's
    audit.v2_set_tolerance_us: the floor is recorded with the measurement
    and widens a long GEMM's set past its 8%, and not a compute-bound
    one's, whose 15% is wider."""
    import argparse

    import audit_v2 as audit_v2_script

    flags = audit_v2_script.SUSTAINED_FLAGS
    args = argparse.Namespace(**{f"sustained_{k}": None for k in flags})
    cfg = {"audit": {"v2_set_tolerance_us": 15.0}}
    settings = audit_v2_script.sustained_first_plan_settings(args, cfg)
    sd, gen = _plan(planned_v2, peak_tflops=2000.0, settings=settings)
    assert gen == 0
    assert v2.settings_of(v2.manifest(sd))["set_tolerance_us"] == 15.0
    entries = asus.current_entries(sd)
    assert entries[LONG]["candidates"] == [1, 2, 3]
    assert entries[LONG]["kernels"] == [1, 2, 3, 9]
    assert entries[CB]["candidates"] == [1, 2, 3]


def test_an_increment_plans_the_sets_that_lack_a_pick_and_new_power_capped_gemms(
    planned_v2,
):
    sd, gen = _plan(planned_v2, peak_tflops=2000.0)
    assert gen == 0
    assert _plan(planned_v2, peak_tflops=2000.0)[1] is None
    # a new model picks 4 on CB, outside its set; and SMALL's v2 winner now
    # takes long enough to be power-capped
    v2.plan(
        planned_v2["d"],
        gemms={},
        lines={},
        parts={EX: [[SMALL, LONG, XL, CB]]},
        new_models=[v2.Model("m2", "2" * 64, "x", {(EX, CB): 4, (EX, SMALL): 1})],
    )
    slow_small = {1: 120.0, 2: 125.0, 3: 200.0}
    planned_v2["measured"][SMALL] = _meas(slow_small, 1)
    planned_v2["measured"][XL] = _meas({1: 50.0, 3: 60.0}, 1)
    sd, gen = _plan(planned_v2, peak_tflops=2000.0)
    assert gen == 1
    p1 = json.loads((v2.gen_dir(sd, 1) / "plan.json").read_text())
    planned = {tuple(e["shape"]): e for part in p1["groups"][EX] for e in part}
    assert set(planned) == {SMALL, CB}
    assert planned[CB]["kernels"] == [1, 2, 3, 4]
    assert planned[CB]["picks"] == {"m1": 3, "m2": 4}
    assert planned[SMALL]["stratum"] == "long"
    assert planned[SMALL]["kernels"] == [1, 2, 3]
    # XL's v2 winner no longer makes it power-capped: planned once, it stays
    v2.plan(
        planned_v2["d"],
        gemms={},
        lines={},
        parts={EX: [[XL]]},
        new_models=[v2.Model("m3", "3" * 64, "x", {(EX, XL): 9})],
    )
    sd, gen = _plan(planned_v2, peak_tflops=2000.0)
    assert gen == 2
    p2 = json.loads((v2.gen_dir(sd, gen) / "plan.json").read_text())
    (e,) = [e for part in p2["groups"][EX] for e in part]
    assert tuple(e["shape"]) == XL and e["stratum"] == "very_long"
    assert e["kernels"] == [1, 2, 3, 9]
    assert asus.current_entries(sd)[XL]["kernels"] == [1, 2, 3, 9]


def test_a_plan_needs_the_v2_measurement(tmp_path):
    with pytest.raises(v2.V2Error, match="no v2 measurement"):
        asus.plan(
            tmp_path / "sus",
            v2d=tmp_path / "v2",
            gemms={},
            measured={},
            config={},
        )


def test_no_power_capped_gemm_writes_a_measurement_of_none(planned_v2):
    for s in list(planned_v2["measured"]):
        planned_v2["measured"][s] = _meas({1: 10.0, 2: 11.0})
    sd, gen = _plan(planned_v2, peak_tflops=None)
    assert gen is None
    m = v2.manifest(sd)
    assert m["generations"] == [] and m["settings"]["compute_bound_tflops"] is None
    assert "sustained: 0 of 0" in v2.status_line(sd, "sustained")


# ── which protocol a number takes ────────────────────────────────────────────

SHAPES = [(64, 64, 64, 1), (512, 512, 512, 1), (4096, 4096, 4096, 1), (8192, 64, 64, 1)]


def _grp(vals):
    se = math.exp(sum(math.log(v) for v in vals) / len(vals))
    return {
        "per_gemm": [[*s, v] for s, v in zip(SHAPES, vals)],
        "cell_mean": se,
        "cell_mean_ci": [se, se],
        "gemm_geomean": se,
        "n_gemms": len(vals),
        "n_cells": len(vals),
        "n_cells_below_target": 0,
        "min_cell": {"cell": "c", "sel_eff": min(vals)},
        "classes": {c: 0 for c in al.CLASSES},
    }


def _rec(v1, v2_=None, sus=None, ctx=None, what="r"):
    groups = {EX: _grp(v1)}
    rec = {"status": "ok", "what": what, "groups": groups, "target": 0.95}
    more = {"cu64": _grp(ctx)} if ctx else {}
    rec["groups"].update(more)
    if v2_ is not None:
        rec["v2"] = {"status": "ok", "groups": {EX: _grp(v2_), **more}}
    if sus is not None:
        rec["sustained"] = {"status": "ok", "groups": {EX: _grp(sus), **more}}
    return rec


def test_numbers_take_the_official_protocol_then_v2_then_v1():
    a = _rec([0.9] * 4, [0.91] * 4, [0.92] * 4, ctx=[0.8] * 4)
    b = _rec([0.9] * 4, [0.95] * 4, [0.85] * 4, ctx=[0.8] * 4)
    assert al.preference() == ["sustained", "v2", "v1"]
    assert al.preference("v2") == ["v2", "v1"]
    assert al.preference(label="cu64") == ["v2", "v1"]
    assert al.protocol_of([a, b]) == "sustained"
    assert al.protocol_of([a, b], official="v2") == "v2"
    assert al.protocol_of([a, b], "cu64") == "v2"
    no_sus = dict(b)
    del no_sus["sustained"]
    assert al.protocol_of([a, no_sus]) == "v2"
    assert al.protocol_of([_rec([0.9] * 4), b]) == "v1"
    assert al.view(a) == (al.protocol_record(a, "sustained"), "sustained")
    assert al.view(a, "v2")[1] == "v2"
    assert al.group_view(a, "cu64")[1] == "v2"
    assert al.group_view(a, EX)[0]["cell_mean"] == pytest.approx(0.92)
    # the composite says worse, v2 says better: the decision follows it
    ok, text, change, proto = al.not_worse(a, b)
    assert (ok, proto) == (False, "sustained")
    assert al.PROTOCOLS["sustained"] in text
    assert al.not_worse(a, b, official="v2")[0] is True


def test_the_per_cell_gate_judges_under_the_official_protocol():
    before = _rec([0.9] * 4, [0.9] * 4, [0.9, 0.9, 0.9, 0.9])
    after = _rec([0.9] * 4, [0.9] * 4, [0.9, 0.9, 0.9, 0.3])
    res = al.cell_gate(before, after, {"tail_floor": 0.5}, 200)
    g = res["groups"][EX]
    assert g["protocol"] == "sustained" and res["official"] == "sustained"
    # a collapse under the composite is a repeated measurement's: confirmed
    assert [t["confirmed"] for t in g["tail"]] == [True] and g["tail_refuses"]
    assert res["passed"] is False and "exclusive sustained" in res["reason"]
    res = al.cell_gate(before, after, {"tail_floor": 0.5}, 200, official="v2")
    assert res["groups"][EX]["protocol"] == "v2" and res["passed"] is True


def test_summary_lines_name_the_official_number():
    rec = _rec([0.9] * 4, [0.91] * 4, [0.93] * 4)
    rec["official"] = "sustained"
    lines = al.summary_lines(rec)
    assert any(
        ln.startswith("audit sustained exclusive: cell mean 93.00%") for ln in lines
    )
    (off,) = [ln for ln in lines if "official exclusive number" in ln]
    assert "93.00%" in off and al.PROTOCOLS["sustained"] in off
    del rec["sustained"]
    (off,) = [ln for ln in al.summary_lines(rec) if "official exclusive" in ln]
    assert "91.00%" in off and "has not evaluated" in off


def test_old_records_keep_their_protocol():
    """Records written before the composite carry no `sustained`: every
    choice is the one they had."""
    a, b = _rec([0.9] * 4, [0.91] * 4), _rec([0.92] * 4, [0.93] * 4)
    assert al.protocol_of([a, b]) == "v2" and al.view(a)[1] == "v2"
    assert al.not_worse(a, b)[3] == "v2"
