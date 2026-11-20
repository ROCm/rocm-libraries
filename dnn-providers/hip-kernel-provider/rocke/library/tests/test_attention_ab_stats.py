# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU unit tests of ``benchmarks/common/attention_ab_stats.py`` (synthetic data):
the statistics, the verdicts, the bar check and the readers. No torch, no GPU."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

_LIBROOT = Path(__file__).resolve().parents[1]
_STATS_FILE = _LIBROOT / "benchmarks" / "common" / "attention_ab_stats.py"
_SHAPES_FILE = (
    _LIBROOT / "benchmarks" / "common" / "attention_dense_strided_ab_shapes.yaml"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "attention_ab_stats_under_test", _STATS_FILE
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


stats = _load()

ARMS = ("A", "B", "A2")  # baseline, arm under test, control
LAYOUTS = ("bshd", "bhsd")


def _record(cid, layout, meas, body="grid"):
    ok = {a: {"pass": True} for a in ARMS}
    return {
        "case": f"{cid}/{layout}",
        "shape_id": cid,
        "layout": layout,
        "body": body,
        "variant": body,
        "dtype": "fp16",
        "status": "ok",
        "spec_identity_equal": True,
        "arms": {a: {"status": "ok", "reason": None} for a in ARMS},
        "measurements": meas,
        "check_pre": ok,
        "check_post": {a: dict(v) for a, v in ok.items()},
    }


def _case(
    cid,
    layout,
    base_us,
    ctrl_eff,
    b_eff,
    rounds=24,
    noise=0.002,
    rng=None,
    ctrl_noise=None,
    body="grid",
):
    """One synthetic case record with paired measurements (log-normal noise)."""
    rng = rng or random.Random(cid)
    ctrl_noise = noise if ctrl_noise is None else ctrl_noise
    meas = []
    for r in range(rounds):
        for pos, arm in enumerate(ARMS):
            eff = {"A": 0.0, "A2": ctrl_eff, "B": b_eff}[arm]
            sig = ctrl_noise if arm == "A2" else noise
            meas.append(
                {
                    "round": r,
                    "pos": pos,
                    "arm": arm,
                    "us": base_us * (1 + eff) * math.exp(rng.gauss(0, sig)),
                }
            )
    return _record(cid, layout, meas, body)


def _det_case(cid, layout, b_ratios, c_ratios=None, base_us=100.0, body="grid"):
    """A case whose per-round ratios B/A and A2/A are exactly the given lists."""
    c_ratios = c_ratios or [1.0] * len(b_ratios)
    meas = []
    for r, (rb, rc) in enumerate(zip(b_ratios, c_ratios)):
        meas += [
            {"round": r, "pos": 0, "arm": "A", "us": base_us},
            {"round": r, "pos": 1, "arm": "B", "us": base_us * rb},
            {"round": r, "pos": 2, "arm": "A2", "us": base_us * rc},
        ]
    return _record(cid, layout, meas, body)


def _det_result(cases):
    return {
        "meta": {
            "baseline": "A",
            "control": "A2",
            "timing_method": "synthetic",
            "arms": {"A": {"tree": "/t"}, "B": {"tree": "/u"}, "A2": {"tree": "/t"}},
        },
        "cases": cases,
    }


def _result(
    b_eff=0.0,
    ctrl_eff_fn=lambda i: 0.0,
    n=12,
    noise=0.002,
    ctrl_noise=None,
    seed=1,
    layouts=("bshd",),
):
    rng = random.Random(seed)
    cases = []
    for layout in layouts:
        for i in range(n):
            cases.append(
                _case(
                    f"s{i:02d}",
                    layout,
                    100.0 + 37 * i,
                    ctrl_eff_fn(i),
                    b_eff,
                    noise=noise,
                    rng=rng,
                    ctrl_noise=ctrl_noise,
                )
            )
    return {
        "meta": {
            "baseline": "A",
            "control": "A2",
            "timing_method": "synthetic",
            "arms": {"A": {"tree": "/t"}, "B": {"tree": "/u"}, "A2": {"tree": "/t"}},
        },
        "cases": cases,
    }


# ---- numeric helpers ---------------------------------------------------------- #


def test_percentile_matches_numpy_linear():
    xs = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert stats.percentile(xs, 0) == 1.0
    assert stats.percentile(xs, 100) == 5.0
    assert stats.percentile(xs, 50) == 3.0
    assert stats.percentile(xs, 95) == pytest.approx(4.8)
    np = pytest.importorskip("numpy")
    rng = random.Random(3)
    ys = [rng.random() for _ in range(37)]
    for q in (2.5, 50, 95, 97.5):
        assert stats.percentile(ys, q) == pytest.approx(float(np.percentile(ys, q)))


def test_median_and_geomean():
    assert stats.median([3, 1, 2]) == 2
    assert stats.median([4, 1, 3, 2]) == 2.5
    assert stats.geomean([1.0, 4.0]) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        stats.median([])


def test_paired_ratios_pairs_by_round_and_skips_missing():
    meas = [
        {"round": 0, "arm": "A", "us": 10.0},
        {"round": 0, "arm": "B", "us": 11.0},
        {"round": 1, "arm": "A", "us": 10.0},  # B missing in round 1
        {"round": 2, "arm": "B", "us": 9.0},
        {"round": 2, "arm": "A", "us": 10.0},
        {"round": 3, "arm": "B", "us": 9.0},  # A missing in round 3
        {"round": 4, "arm": "A", "us": 10.0},
        {"round": 4, "arm": "B", "us": float("nan")},
    ]
    assert stats.paired_ratios(meas, "B", "A") == pytest.approx([1.1, 0.9])
    assert stats.paired_ratios(meas, "A", "B") == pytest.approx([1 / 1.1, 1 / 0.9])


# ---- the cases the method names ----------------------------------------------- #


def test_identical_control_gives_band_floor():
    """Control == baseline exactly -> p95 = 0 -> band = 1% (the floor)."""
    an = stats.analyze(_result(noise=0.0, ctrl_noise=0.0), n_boot=300)
    c = an["strata"]["grid/bshd"]["control"]
    assert c["p95_abs_dev"] == 0.0
    assert c["band"] == 0.01


def test_b_equal_a_passes():
    an = stats.analyze(_result(), n_boot=2000)
    b = an["strata"]["grid/bshd"]["arms"]["B"]
    assert b["verdict"] == "pass", b
    lo, hi = b["geomean_ci95"]
    assert lo <= 1.0 <= hi
    assert hi < 1.01
    assert b["cases_wholly_above_band"] == []


def test_b_five_percent_slower_fails():
    an = stats.analyze(_result(b_eff=0.05), n_boot=2000)
    b = an["strata"]["grid/bshd"]["arms"]["B"]
    assert b["verdict"] == "fail", b
    assert b["geomean"] == pytest.approx(1.05, abs=0.003)
    assert b["geomean_ci95"][1] > 1.01
    assert len(b["cases_wholly_above_band"]) == 12  # every case


def test_b_five_percent_faster_passes():
    an = stats.analyze(_result(b_eff=-0.05), n_boot=2000)
    b = an["strata"]["grid/bshd"]["arms"]["B"]
    assert b["verdict"] == "pass", b
    assert b["geomean"] == pytest.approx(0.95, abs=0.003)


def test_small_steady_slowdown_is_wholly_above_band():
    """B +1.5% with low noise: each case's 95% interval sits above 1 + 1% band
    (the interval, not a multiple of the band, decides)."""
    an = stats.analyze(_result(b_eff=0.015, noise=0.001), n_boot=2000)
    e = an["strata"]["grid/bshd"]
    assert e["control"]["band"] == 0.01
    assert len(e["arms"]["B"]["cases_wholly_above_band"]) == 12, e["arms"]["B"]
    assert e["arms"]["B"]["verdict"] == "fail"


def test_noisy_control_is_too_noisy():
    """Control off by up to +-4% per case (code placement noise) -> p95 > 2%."""
    an = stats.analyze(
        _result(ctrl_eff_fn=lambda i: 0.04 * (-1) ** i * (i % 3) / 2), n_boot=500
    )
    e = an["strata"]["grid/bshd"]
    assert e["control"]["p95_abs_dev"] > 0.02
    assert e["arms"]["B"]["verdict"] == "too noisy"


def test_band_follows_control_p95_not_the_floor():
    """Control deviates by exactly 1.6% on every case, so the band is 1.6%: a case
    steadily 1.3% slower is NOT wholly above it (a fixed 1% band would flag it) and
    the geomean bound passes."""
    n = 24
    cases = []
    for i in range(12):
        ctrl = [1.016 if i % 2 else 0.984] * n
        b = [1.013 if i == 3 else 0.98] * n
        cases.append(_det_case(f"s{i:02d}", "bshd", b, ctrl))
    an = stats.analyze(_det_result(cases), n_boot=1500)
    e = an["strata"]["grid/bshd"]
    assert e["control"]["band"] == pytest.approx(0.016)
    b = e["arms"]["B"]
    assert b["cases_wholly_above_band"] == [], b
    assert b["geomean_ci95"][1] < 1.01
    assert b["verdict"] == "pass", b


def test_upper_bound_not_the_geomean_decides():
    """Every case has median ratio +0.7% but a wide spread of rounds: the geomean
    sits under +1% while its 95% upper bound is above it, so the verdict is fail
    (and no case is wholly above the band)."""
    ratios = [0.96] * 10 + [1.007] * 4 + [1.06] * 10
    cases = [_det_case(f"s{i:02d}", "bshd", ratios) for i in range(3)]
    an = stats.analyze(_det_result(cases), n_boot=3000)
    b = an["strata"]["grid/bshd"]["arms"]["B"]
    assert b["geomean"] == pytest.approx(1.007)
    assert b["geomean_ci95"][1] > 1.01, b
    assert b["cases_wholly_above_band"] == []
    assert b["verdict"] == "fail"


def test_one_case_wholly_above_band_fails_even_if_the_geomean_bound_passes():
    """Eleven cases 1.5% faster and one 7.5% slower: the geomean upper bound is under
    +1%, but the slow case is wholly above the band."""
    n = 24
    cases = [
        _det_case(f"s{i:02d}", "bshd", [1.075 if i == 2 else 0.985] * n)
        for i in range(12)
    ]
    an = stats.analyze(_det_result(cases), n_boot=2000)
    b = an["strata"]["grid/bshd"]["arms"]["B"]
    assert b["geomean_ci95"][1] < 1.01, b
    assert b["cases_wholly_above_band"] == ["s02/bshd"]
    assert b["verdict"] == "fail"


def test_decide_edges():
    assert stats.decide(p95=0.004, geo_hi=1.004, wholly_above=["g03"]) == "fail"
    assert stats.decide(p95=0.004, geo_hi=1.004, wholly_above=[]) == "pass"
    assert stats.decide(p95=0.004, geo_hi=1.0101, wholly_above=[]) == "fail"
    assert stats.decide(p95=0.021, geo_hi=1.0, wholly_above=[]) == "too noisy"
    assert stats.decide(p95=0.0, geo_hi=None, wholly_above=[], n_shapes=0) == "no data"
    assert (
        stats.decide(p95=0.0, geo_hi=None, wholly_above=[], n_shapes=0, n_refused=2)
        == "refused"
    )
    assert (
        stats.decide(p95=0.021, geo_hi=1.0, wholly_above=[], check_failures=["k01"])
        == "fail (check)"
    )
    assert (
        stats.decide(
            p95=0.0, geo_hi=None, wholly_above=[], check_failures=["k01"], n_shapes=0
        )
        == "fail (check)"
    )
    assert (
        stats.decide(p95=0.0, geo_hi=1.0, wholly_above=[], spec_mismatches=["k01"])
        == "fail (spec)"
    )
    assert (
        stats.decide(p95=0.0, geo_hi=1.0, wholly_above=[], errors=["k01"])
        == "fail (error)"
    )


def test_band_is_p95_when_above_floor():
    band, p95 = stats.noise_band([1.0, 1.015, 0.99, 1.0, 1.0])
    assert p95 == pytest.approx(stats.percentile([0, 0.015, 0.01, 0, 0], 95))
    assert band == p95 > 0.01
    assert stats.noise_band([1.0, 1.001]) == (0.01, pytest.approx(0.00095))
    assert stats.noise_band([]) == (None, None)


# ---- exclusions: check failures, refusals, spec identity, errors ---------------- #


def test_check_failure_excludes_the_case_and_fails_the_arm():
    res = _result()
    res["cases"][0]["check_post"]["B"]["pass"] = False
    an = stats.analyze(res, n_boot=300)
    assert an["all_output_checks_passed"] is False
    e = an["strata"]["grid/bshd"]
    assert e["excluded"] == {"s00/bshd": "check"}
    assert e["n_judged"] == 11
    assert e["arms"]["B"]["verdict"] == "fail (check)"
    assert e["arms"]["B"]["check_failures"] == ["s00/bshd"]
    assert e["arms"]["B"]["n_cases"] == 11  # the failed case never entered the geomean


def test_check_failure_of_the_baseline_fails_every_arm():
    res = _result()
    res["cases"][0]["check_pre"]["A"]["pass"] = False
    an = stats.analyze(res, n_boot=300)
    assert an["strata"]["grid/bshd"]["arms"]["B"]["verdict"] == "fail (check)"


def _refuse(case, arms):
    """Mark ``arms`` of one synthetic case refused (no measurement, no check)."""
    for arm in arms:
        case["arms"][arm] = {"status": "refused", "reason": "ValueError: not packed"}
        case["check_pre"].pop(arm, None)
        case["check_post"].pop(arm, None)
    case["measurements"] = [m for m in case["measurements"] if m["arm"] not in arms]
    if set(arms) >= set(case["arms"]):
        case["status"] = "refused"


def test_a_case_the_baseline_serves_and_the_arm_refuses_is_lost():
    """Excluded from that arm's ratios only, and it fails the arm: the change lost a
    case the baseline served (the bar must not PASS on what it dropped)."""
    res = _result(layouts=LAYOUTS, n=6)
    for c in res["cases"]:
        if c["layout"] == "bhsd" and c["shape_id"] in ("s01", "s02"):
            _refuse(c, ["B"])
    an = stats.analyze(res, n_boot=300)
    e = an["strata"]["grid/bhsd"]
    assert e["arms"]["B"]["n_cases"] == 4 and e["arms"]["B"]["n_refused"] == 2
    assert e["arms"]["B"]["refused_cases"] == ["s01/bhsd", "s02/bhsd"]
    assert e["arms"]["B"]["lost_cases"] == ["s01/bhsd", "s02/bhsd"]
    assert e["arms"]["B"]["verdict"] == "fail (refused)"
    assert e["control"]["n_cases"] == 6  # the control still measured every case
    assert an["strata"]["grid/bshd"]["arms"]["B"]["verdict"] == "pass"
    assert an["all_output_checks_passed"] is True
    bar = stats.bar_check(an)
    assert bar["verdict"] == "FAIL" and bar["exit_code"] == stats.EXIT_FAIL
    assert any(
        "lost (served by A): s01/bhsd, s02/bhsd" in line for line in bar["lines"]
    )


def test_an_arm_refusing_a_whole_stratum_the_baseline_serves_fails_it():
    """The arm drops a whole layout the baseline served: no ratio is left in that
    stratum, and the verdict is still a fail, not the neutral "refused"."""
    res = _result(layouts=LAYOUTS, n=4)
    for c in res["cases"]:
        if c["layout"] == "bhsd":
            _refuse(c, ["B"])
    an = stats.analyze(res, n_boot=300)
    b = an["strata"]["grid/bhsd"]["arms"]["B"]
    assert b["n_cases"] == 0 and b["n_refused"] == 4
    assert b["lost_cases"] == [f"s{i:02d}/bhsd" for i in range(4)]
    assert b["verdict"] == "fail (refused)"
    bar = stats.bar_check(an)
    assert bar["verdict"] == "FAIL" and bar["exit_code"] == stats.EXIT_FAIL
    assert any(
        "no judged case (refused in 4); lost (served by A): s00/bhsd" in line
        and line.endswith("-> fail (refused)")
        for line in bar["lines"]
    )


def test_a_case_every_arm_refuses_is_neutral():
    """A layout nobody serves (strided views on packed-only trees, say) is refused,
    not failed: the other strata alone decide the bar."""
    res = _result(layouts=LAYOUTS, n=4)
    for c in res["cases"]:
        if c["layout"] == "bhsd":
            _refuse(c, ["A", "B", "A2"])
    an = stats.analyze(res, n_boot=300)
    assert an["strata"]["grid/bhsd"]["arms"]["B"]["verdict"] == "refused"
    assert an["strata"]["grid/bhsd"]["arms"]["B"]["lost_cases"] == []
    assert an["strata"]["grid/bshd"]["arms"]["B"]["verdict"] == "pass"
    bar = stats.bar_check(an)
    assert bar["verdict"] == "PASS"


def test_an_arm_serving_what_the_baseline_refuses_is_judged_on_the_rest():
    """The reverse: the baseline refuses, the arm serves (strided layouts against a
    packed-only baseline). No lost case; the stratum with no ratio is not a fail."""
    res = _result(layouts=LAYOUTS, n=4)
    for c in res["cases"]:
        if c["layout"] == "bhsd":
            _refuse(c, ["A", "A2"])
    an = stats.analyze(res, n_boot=300)
    assert an["strata"]["grid/bhsd"]["arms"]["B"]["lost_cases"] == []
    assert an["strata"]["grid/bhsd"]["arms"]["B"]["verdict"] == "no data"
    assert stats.bar_check(an)["verdict"] == "PASS"


def test_a_case_the_baseline_raised_on_is_not_lost():
    """The arm refused, the baseline raised: an error, not a case the baseline served."""
    res = _result(n=4)
    c = res["cases"][1]
    _refuse(c, ["B"])
    c["arms"]["A"] = {"status": "error", "reason": "RuntimeError: boom"}
    c["status"] = "error"
    b = stats.analyze(res, n_boot=300)["strata"]["grid/bshd"]["arms"]["B"]
    assert b["verdict"] == "fail (error)" and b["lost_cases"] == []


def test_spec_mismatch_excludes_the_case_and_fails():
    res = _result()
    res["cases"][1]["spec_identity_equal"] = False
    an = stats.analyze(res, n_boot=300)
    e = an["strata"]["grid/bshd"]
    assert e["excluded"] == {"s01/bshd": "spec"}
    assert e["arms"]["B"]["verdict"] == "fail (spec)"


def test_error_case_fails_the_arm_that_raised():
    res = _result()
    res["cases"][1]["arms"]["B"] = {"status": "error", "reason": "RuntimeError: boom"}
    res["cases"][1]["status"] = "error"
    an = stats.analyze(res, n_boot=300)
    assert an["strata"]["grid/bshd"]["arms"]["B"]["verdict"] == "fail (error)"


def test_not_applicable_case_is_skipped_quietly():
    res = _result()
    res["cases"][0]["status"] = "not_applicable"
    res["cases"][0]["measurements"] = []
    an = stats.analyze(res, n_boot=300)
    e = an["strata"]["grid/bshd"]
    assert e["excluded"] == {"s00/bshd": "skipped"}
    assert e["arms"]["B"]["verdict"] == "pass"


def test_strata_group_by():
    res = _result(layouts=LAYOUTS, n=4)
    assert sorted(stats.analyze(res, n_boot=100)["strata"]) == [
        "grid/bhsd",
        "grid/bshd",
    ]
    assert sorted(stats.analyze(res, n_boot=100, group_by=("layout",))["strata"]) == [
        "bhsd",
        "bshd",
    ]
    assert list(stats.analyze(res, n_boot=100, group_by=())["strata"]) == ["all"]


def test_control_must_be_a_copy_of_the_baseline_tree():
    res = _result()
    res["meta"]["arms"]["A2"]["tree"] = "/other"
    with pytest.raises(ValueError, match="same tree"):
        stats.analyze(res, n_boot=100)
    res = _result()
    with pytest.raises(ValueError, match="same tree"):
        stats.analyze(
            res, n_boot=100, baseline="B", control="A2"
        )  # B's tree differs from A2's
    with pytest.raises(ValueError):
        stats.analyze(res, n_boot=100, baseline="A", control="A")
    with pytest.raises(ValueError):
        stats.analyze({"meta": {}, "cases": []}, n_boot=100)


def test_other_baseline_from_the_same_measurements():
    """Ratios can be taken against any arm that has a same-tree control: here the
    baseline moves to A2 with control A (both are the tree "/t")."""
    an = stats.analyze(_result(b_eff=0.0), n_boot=300, baseline="A2", control="A")
    assert an["baseline"] == "A2" and an["test_arms"] == ["B"]
    assert an["strata"]["grid/bshd"]["arms"]["B"]["verdict"] == "pass"


# ---- the bar check -------------------------------------------------------------- #


def test_bar_check_pass_fail_and_exit_codes():
    ok = stats.bar_check(stats.analyze(_result(), n_boot=500))
    assert ok["verdict"] == "PASS" and ok["exit_code"] == 0
    assert ok["lines"][-1] == "BAR: PASS"
    bad = stats.bar_check(stats.analyze(_result(b_eff=0.05), n_boot=500))
    assert bad["verdict"] == "FAIL" and bad["exit_code"] == 1
    good = stats.bar_check(stats.analyze(_result(b_eff=-0.05), n_boot=500))
    assert good["verdict"] == "PASS"
    noisy = stats.bar_check(
        stats.analyze(_result(ctrl_eff_fn=lambda i: 0.05 * (-1) ** i), n_boot=300)
    )
    assert noisy["verdict"].startswith("INCONCLUSIVE") and noisy["exit_code"] == 3
    empty = stats.bar_check(
        stats.analyze(
            {"meta": {"baseline": "A", "control": "A2"}, "cases": []}, n_boot=10
        )
    )
    assert empty["verdict"] == "NO DATA" and empty["exit_code"] == 2


def test_bar_one_failing_stratum_fails_the_whole_bar():
    res = _result(layouts=LAYOUTS, n=6)
    for c in res["cases"]:
        if c["layout"] == "bhsd":
            for m in c["measurements"]:
                if m["arm"] == "B":
                    m["us"] *= 1.06
    bar = stats.bar_check(stats.analyze(res, n_boot=500))
    verdicts = {r["stratum"]: r["verdict"] for r in bar["rows"]}
    assert verdicts == {"grid/bshd": "pass", "grid/bhsd": "fail"}
    assert bar["verdict"] == "FAIL"


def test_markdown_renders():
    an = stats.analyze(_result(b_eff=0.05, layouts=LAYOUTS, n=3), n_boot=200)
    md = stats.to_markdown(an)
    assert "| grid/bshd |" in md and "**fail**" in md and "s00/bshd" in md


# ---- readers and writers ---------------------------------------------------------- #


def _system_yaml(text: str):
    """``text`` parsed by a real PyYAML: this interpreter's when it has one, else a system
    Python's (the venv may not have PyYAML); the test skips when neither has it."""
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    if yaml is not None:
        return json.loads(json.dumps(yaml.safe_load(text), default=str))
    code = "import sys, yaml, json; print(json.dumps(yaml.safe_load(sys.stdin.read()), default=str))"
    for py in ("/usr/bin/python3", "/usr/local/bin/python3"):
        if not os.path.exists(py):
            continue  # run with -I from outside any venv: its site-packages are not seen
        run = subprocess.run(
            [py, "-I", "-c", code], input=text, capture_output=True, text=True
        )
        if run.returncode != 0 and "No module named 'yaml'" in run.stderr:
            continue
        assert run.returncode == 0, run.stderr
        return json.loads(run.stdout)
    pytest.skip("no PyYAML in this interpreter or a system Python")


def test_flow_yaml_is_json_valid_yaml_and_loads_without_pyyaml(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "yaml", None
    )  # `import yaml` raises: the reader must not need it
    obj = {
        "meta": {"x": float("nan"), "y": [1, 2.5], "s": "a b"},
        "cases": [{"case": "g01/bshd", "ok": True}],
    }
    text = stats.dump_flow_yaml(obj)
    assert json.loads(text)["meta"]["x"] is None  # non-finite floats become null
    assert text.isascii()
    path = tmp_path / "r.yaml"
    path.write_text(text)
    got = stats.load_structured(str(path))
    assert got["cases"] == obj["cases"] and got["meta"]["y"] == [1, 2.5]
    assert _system_yaml(text)["cases"] == obj["cases"]


def test_yaml_subset_reader_matches_pyyaml_on_a_shape_file_shaped_text():
    text = (
        "# header comment\n"
        "shapes:   # trailing comment\n"
        "  # a comment line between items\n"
        "  - {id: g01, model: Llama-3-8B TP2, batch: 1, hq: 32, d: 128, mask: none, select: auto,\n"
        "     layouts: [bshd, bhsd], note: 'it''s', quoted: \"a: b, c\",\n"
        "     expect: {candidate: attention_x, block_m: 256, kv_segments: 1, flag: true, ratio: 0.5, none_value: ~}}\n"
        "  - {id: g02, override: {kv_segments: rule}}\n"
        "extra:\n"
        "  - 1\n"
        "  - two\n"
        "  - [3, 4]\n"
        "key: value\n"
        "nested:\n"
        "  inner: 5\n"
        "  list: [a, b]\n"
    )
    got = stats.parse_yaml_subset(text)
    assert got["shapes"][0]["layouts"] == ["bshd", "bhsd"]
    assert (
        got["shapes"][0]["note"] == "it's" and got["shapes"][0]["quoted"] == "a: b, c"
    )
    assert (
        got["shapes"][0]["expect"]["none_value"] is None
        and got["shapes"][0]["expect"]["flag"] is True
    )
    assert (
        got["shapes"][0]["model"] == "Llama-3-8B TP2"
        and got["shapes"][0]["mask"] == "none"
    )
    assert got["extra"] == [1, "two", [3, 4]] and got["nested"] == {
        "inner": 5,
        "list": ["a", "b"],
    }
    assert got == _system_yaml(text)


def test_yaml_subset_reader_refuses_what_it_cannot_read():
    for bad in (
        "a: &anchor 1\nb: *anchor\n",
        "- a: 1\n  b: 2\n",
        "key\n",
        "x: {a: 1\n",
    ):
        with pytest.raises(ValueError):
            stats.parse_yaml_subset(bad)


def test_shipped_shape_file_reads_the_same_with_and_without_pyyaml():
    text = _SHAPES_FILE.read_text()
    assert text.isascii()
    assert stats.parse_yaml_subset(text) == _system_yaml(text)


def test_load_structured_reads_a_log(tmp_path):
    res = _result(n=2)
    log = tmp_path / "run.log"
    rows = "".join("noise AB_ROW " + json.dumps(c) + "\n" for c in res["cases"])
    log.write_text(
        "AB_META "
        + json.dumps(res["meta"])
        + "\n"
        + rows
        + 'AB_DONE {"exit_code": 0}\n'
    )
    got = stats.load_structured(str(log))
    assert got["meta"]["baseline"] == "A" and got["meta"]["exit_code"] == 0
    assert got["cases"] == res["cases"]
    empty = tmp_path / "empty.log"
    empty.write_text("nothing here\n")
    with pytest.raises(SystemExit):
        stats.load_structured(str(empty))


def test_cli_prints_the_verdict_and_exit_code(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text(stats.dump_flow_yaml(_result(b_eff=0.05, n=4)))
    run = subprocess.run(
        [sys.executable, str(_STATS_FILE), str(path), "--n-boot", "200"],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 1, run.stdout + run.stderr
    assert "BAR: FAIL" in run.stdout
    path.write_text(stats.dump_flow_yaml(_result(b_eff=0.0, n=4)))
    run = subprocess.run(
        [sys.executable, str(_STATS_FILE), str(path), "--n-boot", "200", "--markdown"],
        capture_output=True,
        text=True,
    )
    assert (
        run.returncode == 0 and "BAR: PASS" in run.stdout and "| case |" in run.stdout
    )
