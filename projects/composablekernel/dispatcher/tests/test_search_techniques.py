# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Opt-in tuning controls; deterministic CPU evidence, not GPU benchmarks."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from test_adaptive_search_coordinator import (
    ABCD, AS, TE_OPS, FakeAdapter, ck_search, events, json_pairs, rec, run, settings, strict,
)

sys.path.insert(0, str(TE_OPS / "gemm"))
import compare_search as compare  # noqa: E402
from gemm_search import read_ranked_csv  # noqa: E402


@pytest.mark.parametrize("extra, message", [
    ({"portfolio_members": ["tpe"]}, "strategy=portfolio"),
    ({"strategy": "portfolio", "portfolio_members": []}, "distinct"),
    ({"strategy": "portfolio", "portfolio_members": ["tpe", "tpe"]}, "distinct"),
    ({"strategy": "portfolio", "portfolio_members": [["tpe"]]}, "distinct"),
    ({"strategy": "portfolio", "portfolio_members": ["portfolio"]}, "distinct"),
    ({"build_aware": {}}, "requires TPE"),
    ({"strategy": "portfolio", "portfolio_members": ["hill", "de"],
      "build_aware": {}}, "requires TPE"),
    ({"strategy": "tpe", "build_aware": {"rank_fraction": 0}}, "in \\(0, 1]"),
    ({"strategy": "gp", "build_aware": {"explore_probability": 2}}, "in \\[0, 1]"),
    ({"confirm": {"mode": "adaptive"}}, "fixed or race"),
    ({"confirm": {"min_calls": 3}}, "require mode=race"),
    ({"confirm": {"mode": "race", "min_calls": 2}}, ">= 3"),
    ({"confirm": {"mode": "race", "fresh_calls": 2}}, ">= min_calls"),
    ({"confirm": {"mode": "race", "alpha": 0}}, "in \\(0, 1\\)"),
    ({"confirm": {"mode": "race", "margin": -1}}, ">= 0"),
    ({"strategy": "gp", "tpe_candidate_pool": "listed"}, "requires strategy=tpe"),
    ({"strategy": "tpe", "tpe_candidate_pool": "invalid"}, "sampled or listed"),
    ({"strategy": "tpe", "tpe_candidate_pool": True}, "expected type str"),
])
def test_experimental_settings_fail_explicitly(extra, message):
    with pytest.raises(AS.SettingsError, match=message):
        settings(**extra)


def test_opt_in_settings_round_trip_and_defaults_stay_disabled():
    plain = settings().to_json()
    assert "build_aware" not in plain and "portfolio_members" not in plain
    assert plain["confirm"] == {"finalists": 2, "fresh_calls": 3}
    assert "tpe_candidate_pool" not in plain
    new = settings(
        strategy="portfolio", portfolio_members=["hill", "gp", "de", "tpe"],
        build_aware={"rank_fraction": 0.25, "explore_probability": 0.4},
        confirm={"mode": "race", "fresh_calls": 7, "min_calls": 4,
                 "alpha": 0.02, "margin": 0.005},
    )
    assert AS.SearchSettings.from_json(new.to_json()) == new


class DeviceAdapter(FakeAdapter):
    def confirm(self, requests):
        return [replace(o, device=str(list(self.problems).index(o.request.problem_key)))
                for o in super().confirm(requests)]


def test_race_eliminates_clear_losers_after_minimum_and_learns_noise(tmp_path):
    lat = {("P1", "A"): 1.0, ("P1", "B"): 2.0, ("P1", "C"): 3.0}
    ad = DeviceAdapter({"P1": ABCD}, lat=lat)
    summary, _, co = run(
        tmp_path, ad, {"P1": [["A", "B"], ["C"]]},
        confirm={"mode": "race", "finalists": 3, "min_calls": 3, "fresh_calls": 9},
    )
    assert summary.winners == {"P1": "A"} and summary.counts["confirm_calls"] == 9
    meta = strict(tmp_path / "out" / "summary.json")
    assert meta["confirmation"]["P1"]["eliminated"] == ["B", "C"]
    assert meta["confirmation"]["P1"]["looks"] == 1
    assert meta["confirmation"]["P1"]["max_looks"] == 7
    assert meta["confirmation"]["P1"]["call_sigma"] == 0.003
    assert len(meta["confirmation"]["P1"]["noise_model"]["pairs"]) == 3
    assert meta["winners"]["P1"]["calls_ms"] == [1.0] * 3
    assert co.noise.call_sigma is None  # confirmation does not rewrite discovery
    assert [r.kernel_name for r in ad.confirmed[1]] == ["B", "C", "A"]


def test_race_ties_use_maximum_calls_and_report_unresolved_order(tmp_path):
    repeats = [1.0, 1.1, 0.9, 1.2, 0.8, 1.0]
    script = {("P1", k): [("ok", v) for v in repeats] for k in "AB"}
    ad = DeviceAdapter({"P1": ABCD}, confirm_script=script)
    summary, _, _ = run(tmp_path, ad, {"P1": [["A", "B"]]},
                        confirm={"mode": "race", "fresh_calls": 6})
    assert summary.counts["confirm_calls"] == 12
    meta = strict(tmp_path / "out" / "summary.json")
    winner = meta["winners"]["P1"]
    assert winner["selection"] == "budget_exhausted"
    assert winner["tied_kernels"] == ["A", "B"] and winner["calls_ms"] == repeats
    assert meta["confirmation"]["P1"]["call_sigma"] > 0.003
    assert len(meta["confirmation"]["P1"]["noise_model"]["pairs"]) == 6
    looks = events(tmp_path / "out", "race")
    assert len(looks) == 4
    assert all(e["alpha_per_look"] == 0.05 / 4 and not e["eliminated"] for e in looks)


@pytest.mark.parametrize("status, value", [
    ("numeric_fail", None), ("timeout", None), ("cancelled", None),
    ("host_memory", None), ("unattempted", None), ("ok", math.nan),
])
def test_race_never_reuses_failed_or_invalid_confirmation(tmp_path, status, value):
    ad = DeviceAdapter({"P1": ABCD},
                       confirm_script={("P1", "A"): [(status, value)]})
    summary, _, _ = run(tmp_path, ad, {"P1": [["A", "B"]]},
                        confirm={"mode": "race", "fresh_calls": 5})
    assert summary.winners == {"P1": "B"}
    assert summary.counts["confirm_calls"] == 4
    assert sum(r.kernel_name == "A" for calls in ad.confirmed for r in calls) == 1


def test_race_all_fail_has_no_discovery_fallback(tmp_path):
    ad = DeviceAdapter({"P1": ABCD},
                       confirm_script={("P1", k): [("numeric_fail", None)] for k in "AB"})
    summary, _, _ = run(tmp_path, ad, {"P1": [["A", "B"]]},
                        confirm={"mode": "race", "fresh_calls": 5})
    assert summary.winners == {} and summary.exit_code == 1
    assert "no finalist passed" in summary.unresolved["P1"]


@pytest.mark.parametrize("device", ["", "other"])
def test_race_requires_consistent_known_device(tmp_path, device):
    class ChangedDevice(DeviceAdapter):
        def confirm(self, requests):
            got = super().confirm(requests)
            return [replace(o, device=device) for o in got] if len(self.confirmed) == 2 else got

    summary, _, _ = run(tmp_path, ChangedDevice({"P1": ABCD}), {"P1": [["A", "B"]]},
                        confirm={"mode": "race", "fresh_calls": 5})
    assert not summary.winners and "requires one known device" in summary.error


def test_race_keeps_original_hypothesis_count_after_a_failure(tmp_path, monkeypatch):
    sizes = []
    real = ck_search.holm

    def holm(values):
        sizes.append(len(values))
        return real(values)

    monkeypatch.setattr(ck_search, "holm", holm)
    ad = DeviceAdapter({"P1": ABCD},
                       confirm_script={("P1", "C"): [("numeric_fail", None)]})
    summary, _, _ = run(tmp_path, ad, {"P1": [["A", "B"], ["C"]]},
                        confirm={"mode": "race", "finalists": 3, "fresh_calls": 5})
    assert sizes == [6, 6, 6] and summary.counts["confirm_calls"] == 11


def test_build_preference_only_chooses_reusable_near_ties_and_reserves_exploration():
    space = ck_search.SearchSpace.from_configs([{"x": i} for i in range(10)])
    genomes = [space.encode({"x": i}) for i in range(10)]
    pref = ck_search.BuildPreference(rank_fraction=0.2, explore_probability=0)
    pref.keys = {space.key(genomes[2])}
    scores = list(range(10, 0, -1))
    assert pref.choose(space, random.Random(0), genomes, scores, 0) == genomes[0]
    pref.keys.add(space.key(genomes[1]))
    assert pref.choose(space, random.Random(0), genomes, scores, 0) == genomes[1]
    assert pref.selections == 1
    pref.explore_probability = 1
    assert pref.choose(space, random.Random(0), genomes, scores, 0) == genomes[0]
    pref.explore_probability = 0
    pref.keys = {space.key(genomes[9])}
    assert pref.choose(space, random.Random(0), genomes, [1.0] * 10, 0) == genomes[9]


@pytest.mark.parametrize("name", ["tpe", "gp"])
def test_model_reuse_choice_happens_before_trial_ownership(name):
    space = ck_search.SearchSpace.from_configs([{"x": i} for i in range(30)])
    strategy = ck_search.make_strategy(name, space, options={"n_startup": 2})
    for t in strategy.ask(2):
        strategy.tell([ck_search.Observation(
            t, space.key(t.genome), ck_search.Status.OK, value=1 + t.genome[0])])
    picked = []

    class Preference(ck_search.BuildPreference):
        def choose(self, space, rng, genomes, scores, best):
            self.keys = {space.key(g) for g in genomes if space.key(g) != space.key(genomes[best])}
            result = super().choose(space, rng, genomes, scores, best)
            picked.append(result)
            return result

    pref = Preference(rank_fraction=1, explore_probability=0)
    assert ck_search.attach_build_preference(strategy, pref) == 1
    trials = strategy.ask(2)
    assert picked == [t.genome for t in trials] and pref.selections == 2
    assert len(strategy.pending) == 2 and len(strategy.history) == 2
    assert len({space.key(t.genome) for t in trials}) == 2
    assert not set(strategy.history) & {space.key(t.genome) for t in trials}


def test_coordinator_passes_planned_successful_not_failed_builds(tmp_path):
    path = tmp_path / "warm.json"
    path.write_text(json.dumps({p: ["A", "B"] for p in ("P1", "P2")}))
    captured = []

    def factory(name, space, seed, warm, prior):
        strategy = ck_search.make_strategy(name, space, seed, warm_records=warm,
                                            options={"n_startup": 2})
        ask = strategy.ask

        def observing(n):
            captured.append((space.name, set(strategy.build_preference.keys)))
            return ask(n)

        strategy.ask = observing
        return strategy

    ad = FakeAdapter({"P1": list("ABCDEFGH"), "P2": list("ABCDEFGH")}, fail_build={"A"})
    summary, _, co = run(tmp_path, ad, {}, factory=factory, strategy="tpe",
                        build_aware={}, pair_loader=json_pairs,
                        warm_start={"pairs_file": str(path), "top_k": 2})
    p1, p2 = co.problems
    key_a, key_b = (p2.space.key(p2.space.encode(rec(k))) for k in "AB")
    assert captured[0] == ("P1", set())
    assert captured[1] == ("P2", {key_a, key_b})
    assert all(key_a not in keys for _, keys in captured[2:])
    assert summary.counts["build_failed"] == 1 and summary.exit_code == 0
    assert not p1.strategy.pending and not p2.strategy.pending


def test_portfolio_members_and_build_context_reach_real_policies(tmp_path):
    co = AS.SearchCoordinator(
        FakeAdapter({"P1": list("ABCDEFGHIJKL")}),
        settings(strategy="portfolio", trials_per_problem=12,
                 portfolio_members=["hill", "gp", "de", "tpe"], build_aware={}),
        tmp_path / "out",
    )
    summary = co.run()
    members = co.problems[0].strategy.members
    assert [m.name for m in members] == ["hill", "gp", "de", "tpe"]
    assert members[1].build_preference is members[3].build_preference
    assert summary.exit_code == 0
    meta = strict(tmp_path / "out" / "summary.json")
    assert set(meta["problems"]["P1"]["portfolio_allocation"]) == {"hill", "gp", "de", "tpe"}
    assert set(meta["phase_s"]) == {"ask", "build_missing", "measure_pairs", "confirm"}


@pytest.fixture
def comparison(tmp_path):
    problems = {"128x128x128": list("ABCDEFGHIJKL"), "256x256x256": list("ABCDEFGHIJKL")}
    warm, diverse, ranked = (tmp_path / f for f in ("warm.json", "diverse.json", "ranked.csv"))
    warm.write_text(json.dumps({p: ["A", "B"] for p in problems}))
    diverse.write_text(json.dumps({p: ["A", "L"] for p in problems}))
    ranked.write_text("problem,rank,kernel,predicted_tflops\n" + "".join(
        f"{p},{i+1},{k},{12-i}\n" for p in problems for i, k in enumerate(problems[p])))
    baseline = settings(strategy="tpe", trials_per_problem=12,
                        warm_start={"pairs_file": str(warm), "top_k": 2})
    manifest = compare.prepare(baseline.to_json(), ranked, diverse,
                               tmp_path / "comparison", [0])

    class Adapter(DeviceAdapter):
        def context(self):
            return {**super().context(),
                    "domain_sha256": hashlib.sha256(json.dumps(problems).encode()).hexdigest()}

    for case in json.loads(manifest.read_text())["runs"]:
        summary = AS.SearchCoordinator(
            Adapter(problems), case["config"], case["output_dir"],
            pair_loader=json_pairs, prior_loader=read_ranked_csv,
        ).run()
        assert summary.exit_code == 0, summary.error
    return manifest


def test_comparison_runs_all_independent_and_combined_arms(comparison):
    cases = json.loads(comparison.read_text())["runs"]
    assert {c["arm"] for c in cases} == set(compare.ARMS)
    by_arm = {c["arm"]: c for c in cases}
    baseline = by_arm["baseline"]["settings"]
    assert set(baseline).isdisjoint({"build_aware", "portfolio_members"})
    assert "mode" not in baseline["confirm"]
    for arm, field in (("broad_prior", "warm_start"), ("diverse_warm", "warm_start"),
                       ("confirmation_control", "confirm"), ("race", "confirm"),
                       ("build_aware", "build_aware"), ("portfolio_control", "strategy"),
                       ("listed_tpe", "tpe_candidate_pool")):
        changed = {k for k in set(baseline) | set(by_arm[arm]["settings"])
                   if baseline.get(k) != by_arm[arm]["settings"].get(k)}
        assert changed == {field}
    output = io.StringIO()
    assert compare.summarize(comparison, output) == 0
    rows = list(csv.DictReader(io.StringIO(output.getvalue())))
    assert len(rows) == 10
    assert all(r["status"] == "complete" and r["paired_winners"] == "2" for r in rows)
    assert all(float(r["control_relative_quality"]) == 1 for r in rows)
    assert next(r for r in rows if r["arm"] == "race")["control_arm"] == "confirmation_control"
    assert next(r for r in rows if r["arm"] == "portfolio_tpe")["control_arm"] == "portfolio_control"
    combined = strict(Path(by_arm["combined"]["output_dir"]) / "run.json")
    assert all(p["prior_entries"] == 12 and p["warm_start"] == ["A", "L"]
               for p in combined["problems"].values())
    assert set(combined["warm_start_sha256"]) == {
        by_arm["combined"]["settings"]["warm_start"]["pairs_file"],
        by_arm["combined"]["settings"]["warm_start"]["ranked_csv"],
    }


def test_listed_tpe_is_wired_through_coordinator_and_settings(tmp_path):
    config = settings(strategy="tpe", trials_per_problem=12, tpe_candidate_pool="listed")
    assert AS.SearchSettings.from_json(config.to_json()).to_json() == config.to_json()
    co = AS.SearchCoordinator(FakeAdapter({"P1": list("ABCDEFGHIJKL")}), config, tmp_path / "run")
    summary = co.run()
    assert summary.exit_code == 0
    assert co.problems[0].strategy.name == "tpe_listed"
    assert len(co.problems[0].strategy.history) == 12
    assert strict(tmp_path / "run" / "run.json")["settings"]["tpe_candidate_pool"] == "listed"


def test_listed_tpe_cannot_be_silently_ignored_by_custom_factory(tmp_path):
    with pytest.raises(AS.SettingsError, match="tpe_candidate_pool requires the default"):
        AS.SearchCoordinator(FakeAdapter({"P1": list("ABCDEFGHIJKL")}),
                             settings(strategy="tpe", tpe_candidate_pool="listed"),
                             tmp_path / "run", strategy_factory=lambda *args: None)


def test_comparison_reports_missing_and_failed_runs_without_quality_bias(comparison):
    cases = json.loads(comparison.read_text())["runs"]
    missing, failed = (next(c for c in cases if c["arm"] == a)
                       for a in ("broad_prior", "diverse_warm"))
    (Path(missing["output_dir"]) / "summary.json").unlink()
    path = Path(failed["output_dir"]) / "summary.json"
    data = strict(path)
    data.update(status="partial", exit_code=1)
    data["winners"].pop("128x128x128")
    path.write_text(json.dumps(data))
    output = io.StringIO()
    assert compare.summarize(comparison, output) == 1
    rows = {r["arm"]: r for r in csv.DictReader(io.StringIO(output.getvalue()))}
    assert rows["broad_prior"]["status"] == "missing"
    assert rows["diverse_warm"]["paired_winners"] == "1"
    assert rows["diverse_warm"]["control_relative_quality"] == ""


@pytest.mark.parametrize("change", ["settings", "domain", "prior", "input", "source"])
def test_comparison_rejects_mismatched_experiments(comparison, change):
    case = next(c for c in json.loads(comparison.read_text())["runs"] if c["arm"] == "broad_prior")
    path = Path(case["output_dir"]) / "run.json"
    meta = strict(path)
    if change == "settings":
        meta["settings"]["seed"] += 1
    elif change == "domain":
        meta["context"]["domain_sha256"] = "different"
    elif change == "input":
        meta["warm_start_sha256"][next(iter(meta["warm_start_sha256"]))] = "changed"
    elif change == "source":
        meta["context"]["sources"] = {"ck_commit": "different"}
    else:
        meta["problems"]["128x128x128"]["prior_entries"] -= 1
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        compare.summarize(comparison, io.StringIO())


def test_comparison_refuses_invalid_baseline_and_existing_output(tmp_path):
    with pytest.raises(AS.SettingsError, match="baseline must"):
        compare.prepare(settings().to_json(), "absent", "absent", tmp_path, [0])
    warm = tmp_path / "warm.json"
    warm.write_text('{"P1": ["A"]}')
    base = settings(strategy="tpe", warm_start={"pairs_file": str(warm)}).to_json()
    with pytest.raises(AS.SettingsError, match="distinct"):
        compare.prepare(base, warm, warm, tmp_path / "new", [0, 0])
    with pytest.raises(FileExistsError):
        compare.prepare(base, warm, warm, tmp_path, [0])
    assert not (tmp_path / "configs").exists()
