# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU unit tests of ``benchmarks/common/attention_ab_stats.py`` (synthetic data):
the statistics, the verdicts, the bar check and the readers. No torch, no GPU."""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import random
import shutil
import subprocess
import sys
import types
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


def _meta(cases, **over):
    """The meta of a finished device run: the cases it listed, flagged complete."""
    meta = {
        "baseline": "A",
        "control": "A2",
        "timing_method": "graph, one process",
        "expected_cases": [c["case"] for c in cases],
        "complete": True,
        "arms": {"A": {"tree": "/t"}, "B": {"tree": "/u"}, "A2": {"tree": "/t"}},
    }
    meta.update(over)
    return meta


def _det_result(cases):
    return {"meta": _meta(cases), "cases": cases}


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
    return {"meta": _meta(cases), "cases": cases}


# ---- numeric helpers --------------------------------------------------------------- #


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


def _hand_median(xs):
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])


def _hand_geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def test_median_and_geomean():
    assert stats.median([3, 1, 2]) == 2
    assert stats.median([4, 1, 3, 2]) == 2.5
    assert stats.geomean([1.0, 4.0]) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="empty data"):
        stats.median([])


def test_median_and_geomean_edge_cases_and_the_hand_written_versions():
    """The standard-library helpers against the plain formulas: one value, an even and
    an odd count, and random lists. The median is the same float; the geomean is equal
    within rounding, because the standard library sums the logs with ``math.fsum`` and
    a plain ``sum`` before Python 3.12 adds them left to right (they differ in the
    last bit on Python 3.10)."""
    assert stats.median([7.5]) == 7.5 and stats.geomean([7.5]) == pytest.approx(7.5)
    assert stats.median([2.0, 4.0]) == 3.0
    assert stats.geomean([2.0, 8.0]) == pytest.approx(4.0)
    rng = random.Random(11)
    for n in (1, 2, 3, 6, 7, 24, 25):
        for _ in range(200):
            xs = [math.exp(rng.gauss(0, 0.05)) for _ in range(n)]
            assert stats.median(xs) == _hand_median(xs)
            assert stats.geomean(xs) == pytest.approx(
                _hand_geomean(xs), rel=1e-15, abs=0
            )


@pytest.mark.parametrize("xs", [[], [1.0, 0.0], [1.0, -2.0], [1.0, float("nan")]])
def test_geomean_refuses_what_is_not_a_positive_number_on_every_python(xs):
    """From Python 3.13 the standard library returns 0.0 for an input that holds a
    zero; the module's own check refuses it, and an empty, negative or NaN input,
    whatever the interpreter."""
    with pytest.raises(
        ValueError, match="geomean failed .* non-empty sequence of positive numbers"
    ):
        stats.geomean(xs)


def test_percentile_edge_cases():
    assert stats.percentile([4.0], 2.5) == 4.0 == stats.percentile([4.0], 97.5)
    assert stats.percentile([1.0, 3.0], 50) == 2.0  # interpolates between two values
    assert stats.percentile([1.0, 3.0], 25) == pytest.approx(1.5)
    assert stats.percentile([5.0, 1.0, 3.0], 100) == 5.0
    with pytest.raises(ValueError, match="non-empty sequence"):
        stats.percentile([], 95)


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


# ---- the cases the method names ---------------------------------------------------- #


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


def test_decide_pins_the_two_constants_and_the_precedence():
    """The perf bar is two numbers: the +1% bound of the geomean's upper end and the 2%
    noise cut. Each is pinned at the value itself, so a ``<`` that becomes ``<=`` (or
    the other way round) changes a verdict here."""
    pass_hi = 1.0 + stats.GEOMEAN_BOUND
    assert stats.decide(p95=0.0, geo_hi=pass_hi, wholly_above=[]) == "fail"
    assert stats.decide(
        p95=0.0, geo_hi=math.nextafter(pass_hi, 0), wholly_above=[]
    ) == ("pass")
    assert stats.decide(p95=stats.NOISY_P95, geo_hi=1.0, wholly_above=[]) == "pass"
    assert stats.decide(
        p95=math.nextafter(stats.NOISY_P95, 1), geo_hi=1.0, wholly_above=[]
    ) == ("too noisy")
    # a failure of the data wins over noise, and a failed check over a lost case
    assert (
        stats.decide(
            p95=0.5,
            geo_hi=1.0,
            wholly_above=[],
            check_failures=["k01"],
            lost_cases=["k02"],
        )
        == "fail (check)"
    )
    assert (
        stats.decide(p95=0.5, geo_hi=1.0, wholly_above=[], lost_cases=["k02"])
        == "fail (refused)"
    )
    assert (
        stats.decide(p95=0.0, geo_hi=1.0, wholly_above=[], short_cases=["k03"])
        == "fail (rounds)"
    )
    assert (
        stats.decide(
            p95=0.0,
            geo_hi=1.0,
            wholly_above=[],
            errors=["k01"],
            short_cases=["k03"],
        )
        == "fail (error)"
    )


def test_band_is_p95_when_above_floor():
    band, p95 = stats.noise_band([1.0, 1.015, 0.99, 1.0, 1.0])
    assert p95 == pytest.approx(stats.percentile([0, 0.015, 0.01, 0, 0], 95))
    assert band == p95 > 0.01
    assert stats.noise_band([1.0, 1.001]) == (0.01, pytest.approx(0.00095))
    assert stats.noise_band([]) == (None, None)


# ---- exclusions: check failures, refusals, spec identity, errors ------------------- #


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
    """The arm refused, the baseline raised: an error, not a case the baseline
    served."""
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


def _drop_checks(case):
    case["check_pre"], case["check_post"] = {}, {}


def _drop_post_check_of_b(case):
    del case["check_post"]["B"]


def _check_without_a_verdict(case):
    case["check_pre"]["A"] = {"max_abs": 0.0}


@pytest.mark.parametrize(
    "damage", [_drop_checks, _drop_post_check_of_b, _check_without_a_verdict]
)
def test_a_missing_output_check_is_not_a_passing_one(damage):
    """A case with no check on record (a hand-made file, a log of another format) was
    not checked: it never enters a geomean and the headline cannot say all passed."""
    res = _result(n=4)
    damage(res["cases"][0])
    assert stats.unchecked_arms(res["cases"][0])
    an = stats.analyze(res, n_boot=300)
    assert an["all_output_checks_passed"] is False
    e = an["strata"]["grid/bshd"]
    assert e["excluded"] == {"s00/bshd": "check"} and e["n_judged"] == 3
    assert e["arms"]["B"]["verdict"] == "fail (check)"
    assert stats.bar_check(an)["verdict"] == "FAIL"


def test_an_arm_that_never_ran_needs_no_check():
    res = _result(layouts=LAYOUTS, n=3)
    for c in res["cases"]:
        if c["layout"] == "bhsd":
            _refuse(c, ["B"])  # refused: no measurement, so no check either
    assert stats.unchecked_arms(res["cases"][-1]) == []
    assert stats.analyze(res, n_boot=300)["all_output_checks_passed"] is True


def _det_result_with_rounds(rounds_by_case):
    cases = [
        _det_case(f"s{i:02d}", "bshd", [0.99] * n) for i, n in enumerate(rounds_by_case)
    ]
    return _det_result(cases)


def test_a_case_with_too_few_paired_rounds_is_not_judged():
    """One paired round gives a zero-width interval that passes whatever it holds; the
    driver never writes fewer than MIN_ROUNDS, but ``bar`` reads any file."""
    res = _det_result_with_rounds([24, 24, 1, 24])
    an = stats.analyze(res, n_boot=300)
    e = an["strata"]["grid/bshd"]
    assert e["excluded"] == {"s02/bshd": "rounds"} and e["n_judged"] == 3
    assert e["arms"]["B"]["short_cases"] == ["s02/bshd"]
    assert e["arms"]["B"]["verdict"] == "fail (rounds)"
    assert stats.bar_check(an)["verdict"] == "FAIL"
    relaxed = stats.analyze(res, n_boot=300, min_rounds=1)
    assert relaxed["strata"]["grid/bshd"]["arms"]["B"]["verdict"] == "pass"
    assert an["method"]["min_rounds"] == stats.MIN_ROUNDS == 24


def test_a_missing_round_of_one_arm_shortens_the_pairing():
    """Rounds in which one arm has no time drop out of the ratios, so a case can fall
    under the minimum without any single record looking wrong."""
    res = _det_result_with_rounds([24, 24, 24, 24])
    case = res["cases"][1]
    case["measurements"] = [
        m for m in case["measurements"] if not (m["arm"] == "B" and m["round"] < 3)
    ]
    an = stats.analyze(res, n_boot=300)
    assert an["strata"]["grid/bshd"]["excluded"] == {"s01/bshd": "rounds"}
    assert an["cases"][1]["problem_arms"] == ["B"]


@pytest.mark.parametrize("bad", [None, float("nan"), 0.0, -1.0])
def test_an_arm_that_ran_with_no_usable_round_is_short_not_dropped(bad):
    """Arm B ran the case (status ok, both checks passed) but no time of it is usable:
    zero paired rounds is a shortfall like one, not an arm that quietly leaves the case
    while its geomean passes on the rest."""
    res = _det_result_with_rounds([24, 24, 24, 24])
    for m in res["cases"][1]["measurements"]:
        if m["arm"] == "B":
            m["us"] = bad
    an = stats.analyze(res, n_boot=300)
    e = an["strata"]["grid/bshd"]
    assert e["excluded"] == {"s01/bshd": "rounds"} and e["n_judged"] == 3
    assert an["cases"][1]["problem_arms"] == ["B"]
    assert e["arms"]["B"]["short_cases"] == ["s01/bshd"]
    assert e["arms"]["B"]["verdict"] == "fail (rounds)"
    assert stats.bar_check(an)["verdict"] == "FAIL"


def test_a_baseline_with_no_usable_round_leaves_every_arm_that_ran_short():
    """No ratio exists without the baseline's times, so every other arm that ran the
    case is short of rounds; a case the baseline refused stays a neutral one."""
    res = _det_result_with_rounds([24, 24, 24])
    for m in res["cases"][0]["measurements"]:
        if m["arm"] == "A":
            m["us"] = None
    an = stats.analyze(res, n_boot=300)
    assert an["cases"][0]["excluded"] == "rounds"
    assert an["cases"][0]["problem_arms"] == ["A2", "B"]
    assert an["strata"]["grid/bshd"]["arms"]["B"]["verdict"] == "fail (rounds)"


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
    with pytest.raises(ValueError, match="different arm labels"):
        stats.analyze(res, n_boot=100, baseline="A", control="A")
    with pytest.raises(ValueError, match="expected both labels"):
        stats.analyze({"meta": {}, "cases": []}, n_boot=100)


COMPILER = {"llvm_version": [22, 0, 0], "comgr_path": "/rocm/lib/libamd_comgr.so"}


def _with_compilers(res, **by_arm):
    for label, compiler in by_arm.items():
        res["meta"]["arms"][label]["compiler"] = compiler
    return res


def test_arms_compiled_by_different_compilers_are_refused():
    """A compiler change moves every ratio by more than the bar allows, and the
    control (the baseline's own code objects) cannot show it."""
    other_version = {**COMPILER, "llvm_version": [20, 1, 0]}
    other_library = {**COMPILER, "comgr_path": "/opt/torch/libamd_comgr.so"}
    same = {"A": COMPILER, "A2": COMPILER, "B": dict(COMPILER)}
    an = stats.analyze(_with_compilers(_result(n=3), **same), n_boot=100)
    assert an["meta"]["compiler"] == COMPILER  # the baseline arm's, for the report
    for odd in (other_version, other_library):
        res = _with_compilers(_result(n=3), **{**same, "B": odd})
        with pytest.raises(
            ValueError, match=r"arm 'B' was compiled with .*baseline 'A'"
        ):
            stats.analyze(res, n_boot=100)
    # one arm with no compiler on record, the others with one: not the same compiler
    res = _with_compilers(_result(n=3), **{**same, "B": None})
    with pytest.raises(ValueError, match="arm 'B' was compiled with None"):
        stats.analyze(res, n_boot=100)
    # nothing recorded anywhere (a dry run) is nothing to compare
    assert stats.analyze(_result(n=3), n_boot=100)["meta"]["compiler"] is None


def test_compiler_key_normalises_the_recorded_version():
    assert stats.compiler_key(None) is None and stats.compiler_key({}) is None
    assert stats.compiler_key(COMPILER) == ((22, 0, 0), COMPILER["comgr_path"])
    assert stats.compiler_key({"llvm_version": None, "comgr_path": "x"}) == (None, "x")


def _stamp_code(res, **hash_by_arm):
    for c in res["cases"]:
        for label, code in hash_by_arm.items():
            c["arms"][label]["code_sha256"] = code
    return res


def test_an_arm_that_ran_the_baselines_code_is_reported_not_judged():
    """``--arm a=T --arm b=T`` gives ratios near 1 and a pass; the bar says why. It is a
    note, not a fail: a change to the host code alone runs the same kernels."""
    res = _stamp_code(_result(n=3), A="aa", A2="aa", B="aa")
    an = stats.analyze(res, n_boot=300)
    assert an["same_code_arms"] == ["B"]  # never the control: it is the baseline's code
    bar = stats.bar_check(an)
    assert bar["verdict"] == "PASS"
    assert (
        "note: B loaded the same code objects as A in every case; their ratios "
        "measure noise, not a code change"
    ) in bar["lines"]
    res = _stamp_code(_result(n=3), A="aa", A2="aa", B="bb")
    assert stats.analyze(res, n_boot=300)["same_code_arms"] == []
    res["cases"][1]["arms"]["B"]["code_sha256"] = "aa"  # one case alone is not "every"
    assert stats.analyze(res, n_boot=300)["same_code_arms"] == []
    assert stats.analyze(_result(n=3), n_boot=300)["same_code_arms"] == []  # no hashes


def test_output_hashes_that_differ_are_reported_not_gated():
    res = _result(n=3)
    res["cases"][1]["o_hash_equal"] = {"pre": True, "post": False}
    res["cases"][2]["o_hash_equal"] = {"pre": True, "post": True}
    an = stats.analyze(res, n_boot=300)
    assert an["output_hash_differs"] == ["s01/bshd"]
    bar = stats.bar_check(an)
    assert bar["verdict"] == "PASS"
    assert (
        "note: output hashes differ between arms in s01/bshd (reported, not a gate)"
        in bar["lines"]
    )


def test_other_baseline_from_the_same_measurements():
    """Ratios can be taken against any arm that has a same-tree control: here the
    baseline moves to A2 with control A (both are the tree "/t")."""
    an = stats.analyze(_result(b_eff=0.0), n_boot=300, baseline="A2", control="A")
    assert an["baseline"] == "A2" and an["test_arms"] == ["B"]
    assert an["strata"]["grid/bshd"]["arms"]["B"]["verdict"] == "pass"


# ---- the bar check ----------------------------------------------------------------- #


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
            {"meta": _meta([]), "cases": []},
            n_boot=10,
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


# ---- the bar must not pass what it did not see ------------------------------------- #


def test_the_bar_of_a_crashed_run_is_incomplete_not_pass():
    """The driver rewrites the result after every case, so a stride bug that faults on
    case 7 of 12 leaves six passing cases and a file that says the run never ended."""
    res = _result(n=12)
    res["cases"] = res["cases"][:6]
    res["meta"]["complete"] = False
    bar = stats.bar_check(stats.analyze(res, n_boot=300))
    assert bar["verdict"] == "INCOMPLETE" and bar["exit_code"] == stats.EXIT_INCOMPLETE
    assert "run: 6 of 12 expected cases recorded; no record that it finished" in (
        bar["lines"]
    )
    assert any(
        line.startswith("  missing: s06/bshd, s07/bshd") for line in bar["lines"]
    )
    assert bar["lines"][-1] == "BAR: INCOMPLETE"


def test_the_bar_needs_the_finished_flag_and_every_expected_case():
    short = _result(n=4)
    short["cases"].pop()  # flagged complete, but one case never got recorded
    no_flag = _result(n=4)
    del no_flag["meta"]["complete"]
    no_list = _result(n=4)
    del no_list["meta"]["expected_cases"]  # a hand-made file: nothing was promised
    for res in (short, no_flag, no_list):
        bar = stats.bar_check(stats.analyze(res, n_boot=300))
        assert bar["verdict"] == "INCOMPLETE", res["meta"]
        assert bar["exit_code"] == stats.EXIT_INCOMPLETE
    assert stats.analyze(short, n_boot=300)["run"]["missing_cases"] == ["s03/bshd"]
    assert (
        "no expected list"
        in stats.bar_check(stats.analyze(no_list, n_boot=300))["lines"][1]
    )
    ok = stats.analyze(_result(n=4), n_boot=300)["run"]
    assert ok["complete"] is True and ok["missing_cases"] == []


def test_a_fail_verdict_wins_over_an_unfinished_run():
    res = _result(b_eff=0.05, n=4)
    res["meta"]["complete"] = False
    bar = stats.bar_check(stats.analyze(res, n_boot=300))
    assert bar["verdict"] == "FAIL" and bar["exit_code"] == stats.EXIT_FAIL


def test_the_bar_of_a_dry_run_is_not_a_measurement():
    """Synthetic times and stubbed checks cannot pass a speed bar, and the marker is
    in the bar's own lines, not only in the markdown header."""
    res = _result(n=4)
    res["meta"]["timing_method"] = stats.DRY_RUN_TIMING
    an = stats.analyze(res, n_boot=300)
    assert an["strata"]["grid/bshd"]["arms"]["B"]["verdict"] == "pass"
    bar = stats.bar_check(an)
    assert bar["verdict"] == "NOT A MEASUREMENT (dry run)"
    assert bar["exit_code"] == stats.EXIT_SYNTHETIC != stats.EXIT_PASS
    assert any("dry run (synthetic times" in line for line in bar["lines"])
    assert bar["lines"][-1] == "BAR: NOT A MEASUREMENT (dry run)"
    slow = _result(b_eff=0.05, n=4)
    slow["meta"]["timing_method"] = stats.DRY_RUN_TIMING
    assert stats.bar_check(stats.analyze(slow, n_boot=300))["verdict"] == "FAIL"


def test_a_log_without_its_done_line_reads_incomplete(tmp_path):
    """A run killed by a memory fault prints no AB_DONE: the log alone shows it."""
    res = _result(n=3)
    res["meta"]["complete"] = False
    rows = "".join("AB_ROW " + json.dumps(c) + "\n" for c in res["cases"][:2])
    head = "AB_META " + json.dumps(res["meta"]) + "\n"
    log = tmp_path / "run.log"
    log.write_text(head + rows)
    bar = stats.bar_check(stats.analyze(stats.load_structured(str(log)), n_boot=300))
    assert bar["verdict"] == "INCOMPLETE"
    log.write_text(head + rows + "AB_ROW " + json.dumps(res["cases"][2]) + "\n")
    assert (
        stats.bar_check(stats.analyze(stats.load_structured(str(log)), n_boot=300))[
            "verdict"
        ]
        == "INCOMPLETE"
    )  # every case is there, but the run never said it finished
    done = '{"finished_utc": "x", "exit_code": 0, "complete": true}'
    log.write_text(
        head + rows + "AB_ROW " + json.dumps(res["cases"][2]) + f"\nAB_DONE {done}\n"
    )
    bar = stats.bar_check(stats.analyze(stats.load_structured(str(log)), n_boot=300))
    assert bar["verdict"] == "PASS" and bar["exit_code"] == stats.EXIT_PASS


def test_bar_fails_a_verdict_it_does_not_know_and_reads_neutral_ones_as_no_data():
    an = stats.analyze(_result(n=4), n_boot=300)
    arm = an["strata"]["grid/bshd"]["arms"]["B"]
    for neutral in stats.NEUTRAL_VERDICTS:
        arm["verdict"] = neutral
        bar = stats.bar_check(an)
        assert bar["verdict"] == "NO DATA", neutral  # neutral, and nothing passed
    arm["verdict"] = "passed"  # a misspelt verdict is not a neutral one
    bar = stats.bar_check(an)
    assert bar["verdict"] == "FAIL" and bar["exit_code"] == stats.EXIT_FAIL
    assert "unknown verdicts (a bug in the analysis): passed" in bar["lines"]


def test_markdown_renders():
    an = stats.analyze(_result(b_eff=0.05, layouts=LAYOUTS, n=3), n_boot=200)
    md = stats.to_markdown(an)
    assert "| grid/bshd |" in md and "**fail**" in md and "s00/bshd" in md


@pytest.mark.parametrize("refused", [["s01"], ["s00", "s01", "s02"]])
def test_the_stratum_table_counts_the_cases_each_arm_was_judged_on(refused):
    """A refused case leaves arm B's geomean: the table says how many cases that
    geomean is over, of the stratum's, as the bar's own line does."""
    res = _result(n=3)
    for c in res["cases"]:
        if c["shape_id"] in refused:
            _refuse(c, ["B"])
    an = stats.analyze(res, n_boot=300)
    judged = an["strata"]["grid/bshd"]["arms"]["B"]["n_cases"]
    assert judged == 3 - len(refused)
    (row,) = [x for x in stats.to_markdown(an).splitlines() if x.startswith("| grid/")]
    assert row.startswith(f"| grid/bshd | {judged} of 3 |") and "| B |" in row
    assert "fail (refused)" in row


# ---- readers and writers ----------------------------------------------------------- #


def _other_pythons():
    """Interpreters other than this one (a venv's base Python, the ``python3`` on the
    PATH): where a PyYAML can be when the venv has none."""
    me = Path(sys.executable).absolute()
    found: list[str] = []
    for cand in (getattr(sys, "_base_executable", None), shutil.which("python3")):
        if cand and Path(cand).absolute() != me and cand not in found:
            found.append(cand)
    return found


def _real_yaml(text: str):
    """``text`` parsed by a real PyYAML: this interpreter's when it has one, else
    another Python's; the test skips when none has it."""
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    if yaml is not None:
        return json.loads(json.dumps(yaml.safe_load(text), default=str))
    code = (
        "import sys, yaml, json; "
        "print(json.dumps(yaml.safe_load(sys.stdin.read()), default=str))"
    )
    for py in _other_pythons():
        # -I: from outside any venv, whose site-packages are not seen
        run = subprocess.run(
            [py, "-I", "-c", code], input=text, capture_output=True, text=True
        )
        if run.returncode != 0 and "No module named 'yaml'" in run.stderr:
            continue
        assert run.returncode == 0, run.stderr
        return json.loads(run.stdout)
    pytest.skip("no PyYAML in this interpreter or another Python")


FLOW_OBJECT = {
    "meta": {"x": float("nan"), "y": [1, 2.5], "s": "a b"},
    "cases": [{"case": "g01/bshd", "ok": True}],
}


def test_flow_yaml_writer_output_loads_without_pyyaml(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "yaml", None
    )  # `import yaml` raises: the reader must not need it
    text = stats.dump_flow_yaml(FLOW_OBJECT)
    assert json.loads(text)["meta"]["x"] is None  # non-finite floats become null
    assert text.isascii()
    path = tmp_path / "r.yaml"
    path.write_text(text)
    got = stats.load_structured(str(path))
    assert got["cases"] == FLOW_OBJECT["cases"] and got["meta"]["y"] == [1, 2.5]


def test_flow_yaml_writer_output_is_valid_yaml_for_a_real_parser():
    text = stats.dump_flow_yaml(FLOW_OBJECT)
    assert _real_yaml(text)["cases"] == FLOW_OBJECT["cases"]


def test_a_json_result_is_read_as_json_even_when_pyyaml_is_installed(
    tmp_path, monkeypatch
):
    """PyYAML (YAML 1.1) reads ``1e-05`` as a string, and a string time is dropped
    from the ratios without a word; the driver writes JSON, so JSON goes first."""

    def refuse(_text):
        raise AssertionError("PyYAML must not read a JSON document")

    fake = types.SimpleNamespace(
        safe_load=refuse, YAMLError=type("YAMLError", (Exception,), {})
    )
    monkeypatch.setitem(sys.modules, "yaml", fake)
    path = tmp_path / "r.yaml"
    path.write_text(json.dumps({"us": 1e-05, "big": 1e20, "flag": True}))
    got = stats.load_structured(str(path))
    assert got == {"us": 1e-05, "big": 1e20, "flag": True}
    assert isinstance(got["us"], float)
    path.write_text("a: 1\n")  # not JSON: only now is PyYAML asked
    with pytest.raises(AssertionError, match="PyYAML must not read"):
        stats.load_structured(str(path))


SHAPE_LIKE_TEXT = (
    "# header comment\n"
    "shapes:   # trailing comment\n"
    "  # a comment line between items\n"
    "  - {id: g01, model: Llama-3-8B TP2, batch: 1, hq: 32, d: 128, mask: none,\n"
    "     select: auto, layouts: [bshd, bhsd], note: 'it''s', quoted: \"a: b, c\",\n"
    "     expect: {candidate: attention_x, block_m: 256, kv_segments: 1, flag: true,\n"
    "              ratio: 0.5, none_value: ~}}\n"
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


def test_yaml_subset_reader_reads_a_shape_file_shaped_text():
    got = stats.parse_yaml_subset(SHAPE_LIKE_TEXT)
    first = got["shapes"][0]
    assert first["layouts"] == ["bshd", "bhsd"]
    assert first["note"] == "it's" and first["quoted"] == "a: b, c"
    assert first["expect"]["none_value"] is None and first["expect"]["flag"] is True
    assert first["model"] == "Llama-3-8B TP2" and first["mask"] == "none"
    assert got["extra"] == [1, "two", [3, 4]]
    assert got["nested"] == {"inner": 5, "list": ["a", "b"]}


def test_yaml_subset_reader_matches_pyyaml_on_a_shape_file_shaped_text():
    assert stats.parse_yaml_subset(SHAPE_LIKE_TEXT) == _real_yaml(SHAPE_LIKE_TEXT)


@pytest.mark.parametrize(
    ("bad", "why"),
    [
        ("a: &anchor 1\nb: *anchor\n", "unsupported YAML"),
        ("- a: 1\n  b: 2\n", "unsupported YAML"),
        ("key\n", "expected 'key: value'"),
        ("x: {a: 1\n", "unterminated flow collection"),
    ],
)
def test_yaml_subset_reader_refuses_what_it_cannot_read(bad, why):
    with pytest.raises(ValueError, match=why):
        stats.parse_yaml_subset(bad)


def test_shipped_shape_file_reads_without_pyyaml():
    text = _SHAPES_FILE.read_text()
    assert text.isascii()
    shapes = stats.parse_yaml_subset(text)["shapes"]
    assert len(shapes) == 13 and all("id" in row for row in shapes)


def test_shipped_shape_file_reads_the_same_with_and_without_pyyaml():
    text = _SHAPES_FILE.read_text()
    assert stats.parse_yaml_subset(text) == _real_yaml(text)


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
    with pytest.raises(SystemExit, match="no AB_ROW lines"):
        stats.load_structured(str(empty))


def _cli(path, *flags):
    return subprocess.run(
        [sys.executable, str(_STATS_FILE), str(path), "--n-boot", "200", *flags],
        capture_output=True,
        text=True,
    )


def test_cli_prints_the_verdict_and_exit_code(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text(stats.dump_flow_yaml(_result(b_eff=0.05, n=4)))
    run = _cli(path)
    assert run.returncode == 1, run.stdout + run.stderr
    assert "BAR: FAIL" in run.stdout
    path.write_text(stats.dump_flow_yaml(_result(b_eff=0.0, n=4)))
    run = _cli(path, "--markdown")
    assert (
        run.returncode == 0 and "BAR: PASS" in run.stdout and "| case |" in run.stdout
    )


def test_cli_exits_nonzero_for_a_run_that_did_not_finish_or_did_not_measure(tmp_path):
    path = tmp_path / "r.yaml"
    unfinished = _result(n=4)
    unfinished["meta"]["complete"] = False
    path.write_text(stats.dump_flow_yaml(unfinished))
    run = _cli(path)
    assert run.returncode == stats.EXIT_INCOMPLETE, run.stdout + run.stderr
    assert "BAR: INCOMPLETE" in run.stdout and "PASS" not in run.stdout
    dry = _result(n=4)
    dry["meta"]["timing_method"] = stats.DRY_RUN_TIMING
    path.write_text(stats.dump_flow_yaml(dry))
    run = _cli(path)
    assert run.returncode == stats.EXIT_SYNTHETIC, run.stdout + run.stderr
    assert "BAR: NOT A MEASUREMENT (dry run)" in run.stdout
    path.write_text(stats.dump_flow_yaml(_det_result_with_rounds([24, 3])))
    assert _cli(path).returncode == 1  # a short case fails the bar ...
    assert _cli(path, "--min-rounds", "3").returncode == 0  # ... unless allowed


def test_cli_refuses_an_analysis_that_cannot_be_made(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text(stats.dump_flow_yaml(_result(n=3)))
    run = _cli(path, "--baseline", "A", "--control", "A")
    assert run.returncode == stats.EXIT_NO_DATA
    assert "refused: analyze failed - control='A' baseline='A'" in run.stderr


@pytest.mark.parametrize(
    ("name", "text", "why"),
    [
        ("missing.yaml", None, "No such file"),
        ("junk.log", "no rows here\n", "no AB_ROW lines"),
        ("bad.json", "{not json", "Expecting property name"),
        ("bad.log", "AB_ROW {broken\n", "Expecting property name"),
        ("list.json", "[1, 2]", "not a harness result"),
    ],
)
def test_cli_refuses_input_it_cannot_read_with_no_verdict_not_a_fail(
    tmp_path, name, text, why
):
    """Exit 1 says an arm failed; a file the bar cannot read gives no verdict (2), with
    the reason on stderr and no traceback."""
    path = tmp_path / name
    if text is not None:
        path.write_text(text)
    run = _cli(path)
    assert run.returncode == stats.EXIT_NO_DATA != stats.EXIT_FAIL, run.stderr
    assert "refused: " in run.stderr and why in run.stderr
    assert "Traceback" not in run.stderr


@pytest.mark.parametrize("pyyaml", [True, False])
def test_cli_refuses_a_malformed_yaml_result_with_or_without_pyyaml(
    tmp_path, monkeypatch, capsys, pyyaml
):
    if pyyaml:
        pytest.importorskip("yaml")
    else:
        monkeypatch.setitem(sys.modules, "yaml", None)  # `import yaml` raises
    path = tmp_path / "r.yaml"
    path.write_text("meta: {a: [1, 2}\n")
    assert stats.main([str(path)]) == stats.EXIT_NO_DATA
    err = capsys.readouterr().err
    assert err.startswith(f"refused: {path}: not readable"), err


def test_out_md_says_whether_the_run_is_complete_and_measured(tmp_path):
    """The markdown file is read on its own (attached to a review): its strata read
    ``pass``, so it carries the bar's lines, and an unfinished run or a dry run cannot
    read as a pass there."""
    path, md = tmp_path / "r.yaml", tmp_path / "r.md"
    unfinished = _result(n=4)
    unfinished["meta"]["complete"] = False
    unfinished["cases"].pop()
    path.write_text(stats.dump_flow_yaml(unfinished))
    run = _cli(path, "--out-md", str(md))
    assert run.returncode == stats.EXIT_INCOMPLETE, run.stdout + run.stderr
    text = md.read_text()
    assert "**pass**" in text  # the per-stratum verdicts alone would read as a pass
    assert "run: 3 of 4 expected cases recorded; no record that it finished" in text
    assert "  missing: s03/bshd" in text
    assert text.rstrip().endswith("BAR: INCOMPLETE")
    dry = _result(n=4)
    dry["meta"]["timing_method"] = stats.DRY_RUN_TIMING
    path.write_text(stats.dump_flow_yaml(dry))
    assert _cli(path, "--out-md", str(md)).returncode == stats.EXIT_SYNTHETIC
    text = md.read_text()
    assert "timing: dry run (synthetic times, stubbed output checks)" in text
    assert text.rstrip().endswith("BAR: NOT A MEASUREMENT (dry run)")
    an = stats.analyze(dry, n_boot=200)
    bar = stats.bar_check(an)
    assert stats.report_markdown(an, bar) == (
        stats.to_markdown(an) + "\n" + "\n".join(bar["lines"]) + "\n"
    )


def test_the_public_surface_is_listed_and_documented():
    """``__all__`` names every public function and class the module defines, each name
    resolves, and each callable carries a docstring."""
    defined = {
        name
        for name, obj in vars(stats).items()
        if not name.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and obj.__module__ == stats.__name__
    }
    assert defined <= set(stats.__all__), sorted(defined - set(stats.__all__))
    assert [n for n in stats.__all__ if not hasattr(stats, n)] == []
    undocumented = [
        n
        for n in stats.__all__
        if callable(getattr(stats, n)) and not inspect.getdoc(getattr(stats, n))
    ]
    assert undocumented == []
