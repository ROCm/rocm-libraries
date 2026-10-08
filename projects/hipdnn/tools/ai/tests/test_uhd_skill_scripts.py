# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the scripts the hipdnn-uhd skill bundles.

The scripts live in the skill directory rather than a package, so they are loaded by path.
The validation scripts run against a fake `hipdnn_bench` that answers like the real one.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
TOOLS_DIR = SKILLS_DIR.parent.parent
SCRIPTS = SKILLS_DIR / "hipdnn-uhd" / "scripts"
SLICE_SCRIPT = SCRIPTS / "eval_slice.py"

DECLARED_ID = "8b61d0c9-24af-4d17-9e52-3a7c06b8f145"


def _load(path: Path, name: str) -> ModuleType:
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"could not load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def slicing() -> ModuleType:
    return _load(SLICE_SCRIPT, "uhd_skill_eval_slice")


@pytest.fixture(scope="module")
def decide() -> ModuleType:
    return _load(SCRIPTS / "decide.py", "uhd_skill_decide")


@pytest.fixture(scope="module")
def tree() -> ModuleType:
    return _load(SCRIPTS / "candidate_tree.py", "uhd_skill_candidate_tree")


@pytest.fixture(scope="module")
def l2_job() -> ModuleType:
    return _load(SCRIPTS / "validate_l2.py", "uhd_skill_validate_l2")


def _fake_bench(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "bench.py"
    script.write_text(body)
    if os.name == "nt":
        bench = tmp_path / "bench.cmd"
        bench.write_text(f'@"{sys.executable}" "{script}" %*\n')
    else:
        bench = tmp_path / "bench"
        bench.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        bench.chmod(0o755)
    return bench


# --- evaluation slice ---------------------------------------------------------------


def _graphs(counts):
    return [{"benchmark": f"{regime}-{i}", "regime": regime}
            for regime, n in counts.items() for i in range(n)]


def test_every_regime_with_two_graphs_is_scored_and_none_is_held_out_whole(slicing):
    chosen = slicing.choose(_graphs({"big": 100, "pair": 2, "lone": 1}), 0.2, seed=0)
    assert chosen["by_regime"] == {"big": {"graphs": 100, "held_out": 20},
                                   "lone": {"graphs": 1, "held_out": 0},
                                   "pair": {"graphs": 2, "held_out": 1}}
    assert len(chosen["benchmarks"]) == 21


def test_the_slice_depends_on_the_set_and_seed_alone(slicing):
    graphs = _graphs({"a": 40, "b": 15})
    first = slicing.choose(graphs, 0.2, seed=3)
    assert slicing.choose(list(reversed(graphs)), 0.2, seed=3) == first, "order-free"
    assert slicing.choose(graphs, 0.2, seed=4)["benchmarks"] != first["benchmarks"]


def test_unlabelled_graphs_are_their_own_group_and_a_bad_fraction_is_refused(slicing):
    chosen = slicing.choose([{"benchmark": f"g{i}"} for i in range(10)], 0.2)
    assert chosen["by_regime"] == {"unlabelled": {"graphs": 10, "held_out": 2}}
    with pytest.raises(ValueError):
        slicing.choose(_graphs({"a": 5}), 1.0)
    with pytest.raises(ValueError, match="no graph ids"):
        slicing.choose([{"regime": "a"}])


def test_the_default_slice_is_a_tenth_of_each_regime(slicing):
    chosen = slicing.choose(_graphs({"big": 100, "small": 10}), slicing.DEFAULT_FRACTION)
    assert chosen["by_regime"] == {"big": {"graphs": 100, "held_out": 10},
                                   "small": {"graphs": 10, "held_out": 1}}


def test_a_sample_takes_every_regime_before_any_twice(slicing):
    graphs = _graphs({"big": 50, "mid": 3, "lone": 1})
    sampled = slicing.sample_evenly(graphs, 5, seed=0)
    regimes = [g["regime"] for g in sampled]
    assert sorted(regimes[:3]) == ["big", "lone", "mid"], "one of each regime first"
    assert regimes.count("big") == 2 and regimes.count("mid") == 2
    assert slicing.sample_evenly(list(reversed(graphs)), 5, seed=0) == sampled, "order-free"
    assert len(slicing.sample_evenly(graphs, 1000)) == len(graphs), "all of them, when fewer"


def _write_set(root: Path, operations: dict) -> list:
    manifests = []
    for operation, counts in operations.items():
        directory = root / operation
        (directory / "graphs").mkdir(parents=True)
        graphs = []
        for graph in _graphs(counts):
            name = f"{operation}_{graph['benchmark']}"
            (directory / "graphs" / f"{name}.fb").write_bytes(b"\x00")
            graphs.append({**graph, "benchmark": name, "file": f"graphs/{name}.fb"})
        (directory / "manifest.json").write_text(json.dumps({"graphs": graphs}))
        manifests.append(directory / "manifest.json")
    return manifests


def test_the_cli_slices_every_operation_once_and_resolves_graph_files(slicing, tmp_path):
    manifests = _write_set(tmp_path / "set", {"conv_fwd": {"a": 20}, "matmul": {"b": 10}})
    run = lambda *args: subprocess.run([sys.executable, str(SLICE_SCRIPT), *map(str, args)],
                                       capture_output=True, text=True)
    assert run("choose", *manifests).returncode == 0
    written = tmp_path / "set" / "eval_slice.json"
    document = json.loads(written.read_text())
    assert document["by_regime"] == {"a": {"graphs": 20, "held_out": 2},
                                     "b": {"graphs": 10, "held_out": 1}}
    again = run("choose", *manifests)
    assert again.returncode != 0 and "chosen once" in again.stderr

    assert run("holdout", written, "--out", tmp_path / "held.json").returncode == 0
    assert json.loads((tmp_path / "held.json").read_text()) == document["benchmarks"]

    assert run("sample", written, "--count", "2", "--out", tmp_path / "sample.json").returncode == 0
    sample = json.loads((tmp_path / "sample.json").read_text())["graphs"]
    assert len(sample) == 2 and all(Path(g["path"]).is_file() for g in sample)

    clean = {"models": [{"training_problem_keys": [["not-held-out", "gpu"]]}]}
    (tmp_path / "clean.json").write_text(json.dumps(clean))
    assert run("sample", written, "--trained", tmp_path / "clean.json",
               "--out", tmp_path / "s2.json").returncode == 0
    leaked = {"models": [{"training_problem_keys": [[document["benchmarks"][0], "gpu"]]}]}
    (tmp_path / "leaked.json").write_text(json.dumps(leaked))
    refused = run("sample", written, "--trained", tmp_path / "leaked.json",
                  "--out", tmp_path / "s3.json")
    assert refused.returncode != 0 and "training graphs" in refused.stderr

    exported = [{"benchmark": b, "tflops": 1.0} for b in document["benchmarks"]]
    exported.append({"benchmark": "trained-on", "tflops": 1.0})
    (tmp_path / "exported.json").write_text(json.dumps(exported))
    assert run("rows", tmp_path / "exported.json", written,
               "--out", tmp_path / "rows.json").returncode == 0
    kept = json.loads((tmp_path / "rows.json").read_text())
    assert {r["benchmark"] for r in kept} == set(document["benchmarks"])


# --- engine predictor (L1) verdict --------------------------------------------------


def _rows(n=10, *, candidate_status="available", revision="rev-A", candidate_bias=1.05,
          shipped_bias=1.30, measured=True, metric="tflops"):
    rows = []
    for i in range(n):
        truth = 100.0 + 10 * i
        row = {"graph": f"g{i}.fb",
               "shipped": {"status": "available", "value": truth * shipped_bias,
                           "metric": "tflops", "selector_revision": revision},
               "candidate": {"status": candidate_status, "value": truth * candidate_bias,
                             "metric": metric, "selector_revision": revision}}
        if measured:
            row["measured"] = {"ok": True, "value": truth, "metric": "tflops"}
        rows.append(row)
    return rows


def _document(rows, *, unique=True, byte_equal=True, measured=True):
    return {"measured": measured, "metric": "tflops", "rows": rows,
            "tree": {"unique": unique, "byte_equal": byte_equal, "candidate_id": DECLARED_ID,
                     "carrying_declared_id": ["candidate_8b61d0c9/heuristic.uhd.json"]}}


def _decide_l1(decide, document, **kwargs):
    kwargs.setdefault("candidate_selector_revision", "rev-A")
    return decide.decide(document, declared_id=DECLARED_ID, candidate_id=DECLARED_ID, **kwargs)


def test_a_good_candidate_passes_and_is_compared_with_the_shipped_model(decide):
    report = _decide_l1(decide, _document(_rows()),
                        max_regression=decide.DEFAULT_L1_MAX_REGRESSION)
    assert report["passed"], report["gates"]
    paired = report["metrics"]["paired"]
    assert paired["candidate"]["median_abs_pct_error"] == pytest.approx(5.0)
    assert paired["shipped"]["median_abs_pct_error"] == pytest.approx(30.0)
    assert paired["candidate"]["spearman"] == pytest.approx(1.0)


def test_a_tree_where_the_engine_could_read_another_model_fails_identity(decide):
    report = _decide_l1(decide, _document(_rows(), unique=False))
    assert not report["passed"] and not report["gates"]["identity"]["passed"]


def test_a_candidate_under_another_id_fails_identity(decide):
    report = decide.decide(_document(_rows()), declared_id=DECLARED_ID,
                           candidate_id="11111111-2222-3333-4444-555555555555",
                           candidate_selector_revision="rev-A")
    assert not report["gates"]["identity"]["passed"]


def test_an_unavailable_prediction_fails(decide):
    report = _decide_l1(decide, _document(_rows(candidate_status="unavailable")))
    assert not report["gates"]["available"]["passed"]
    assert report["gates"]["available"]["unavailable_sample"]


def test_graphs_the_engine_declined_do_not_count_against_availability(decide):
    rows = _rows()
    rows[0]["measured"] = {"ok": False, "error": "engine declined"}
    rows[0]["candidate"]["status"] = "unavailable"
    report = _decide_l1(decide, _document(rows))
    assert report["gates"]["available"]["passed"]
    assert report["gates"]["available"]["executed_graphs"] == 9


def test_a_stale_selector_revision_fails(decide):
    report = _decide_l1(decide, _document(_rows(revision="rev-B")))
    assert not report["gates"]["selector_revision"]["passed"]


def test_a_candidate_answering_in_another_metric_fails(decide):
    report = _decide_l1(decide, _document(_rows(metric="time")))
    assert not report["passed"] and not report["gates"]["metric"]["passed"]


def test_an_answer_that_names_no_metric_fails(decide):
    rows = _rows()
    rows[0]["candidate"]["metric"] = None
    gate = _decide_l1(decide, _document(rows))["gates"]["metric"]
    assert not gate["passed"] and "named no metric" in gate["problems"][0]
    assert gate["answered"] == ["tflops"]


def test_a_less_accurate_candidate_fails_the_default_tolerance(decide):
    worse = _document(_rows(candidate_bias=1.5, shipped_bias=1.1))
    assert _decide_l1(decide, worse)["passed"], "no tolerance, no accuracy gate"
    report = _decide_l1(decide, worse, max_regression=decide.DEFAULT_L1_MAX_REGRESSION)
    assert not report["passed"] and not report["gates"]["accuracy"]["passed"]


def test_with_nothing_shipped_the_candidate_needs_only_to_be_scored(decide):
    rows = _rows()
    for row in rows:
        row["shipped"] = {"status": "unavailable"}
    report = _decide_l1(decide, _document(rows), max_regression=0.05, shipped_baseline=False)
    assert report["passed"] and report["baseline"] == "none"
    assert not _decide_l1(decide, _document(rows), max_regression=0.05)["passed"], \
        "a shipped side that answers nothing is inconclusive unless nothing ships"


def test_each_l1_is_scored_against_the_selection_it_ships_beside(decide):
    def row(graph, measured, measured_shipped, candidate, shipped):
        return {"graph": graph, "measured": {"ok": True, "value": measured},
                "measured_shipped": {"ok": True, "value": measured_shipped},
                "candidate": {"status": "available", "value": candidate,
                              "selector_revision": "r", "metric": "tflops"},
                "shipped": {"status": "available", "value": shipped,
                            "selector_revision": "r", "metric": "tflops"}}
    rows = {"measured": True, "metric": "tflops", "tree": {"unique": True, "byte_equal": True},
            "rows": [row("a", 100, 50, 100, 50), row("b", 200, 80, 200, 80)]}
    report = decide.decide(rows, declared_id="c", candidate_id="c",
                           candidate_selector_revision="r")
    assert report["metrics"]["shipped"]["median_abs_pct_error"] == 0.0
    assert report["metrics"]["shipped_truth"] == "its own selection"


def test_too_few_graphs_is_not_a_pass(decide):
    assert not _decide_l1(decide, _document([]))["passed"]


# --- catalog ranker (L2) verdict ----------------------------------------------------

REFUSED = ("[error] uhd: engine 'e' ranks on [dtype], which its UED does not expose as knobs [];"
           " the model is not used and kernels rank by priority, then id")


def _l2_row(name, shipped, candidate, refusal=None):
    return {"graph": f"/g/{name}.fb", "shipped": {"ok": shipped, "refusals": []},
            "candidate": {"ok": candidate, "refusals": [refusal] if refusal else []}}


def _report(mean):
    return {"metrics": {"top1_regret": {"mean": mean}}}


def test_a_graph_the_shipped_tree_cannot_run_is_the_engines_failure(decide):
    rows = {"rows": [_l2_row("a", True, True), _l2_row("b", False, False)],
            "winner_records": {"shipped": 0, "candidate": 0}}
    report = decide.decide_catalog(rows, candidate_report=_report(0.01),
                                   shipped_report=_report(0.02))
    gate = report["gates"]["available"]
    assert gate["passed"] and (gate["runnable"], gate["engine_failures"]) == (1, 1)
    assert report["passed"]


def test_the_candidate_failing_where_the_shipped_tree_runs_fails_availability(decide):
    rows = {"rows": [_l2_row("a", True, True), _l2_row("b", True, False)], "winner_records": {}}
    gate = decide.decide_catalog(rows, candidate_report=_report(0.0),
                                 shipped_report=None)["gates"]["available"]
    assert not gate["passed"] and "1 of 2" in gate["problems"][0]


def test_a_ranker_the_runtime_declines_fails_even_though_every_plan_built(decide):
    rows = {"rows": [_l2_row("a", True, True, REFUSED)], "winner_records": {}}
    report = decide.decide_catalog(rows, candidate_report=_report(0.0), shipped_report=None)
    assert report["gates"]["available"]["passed"] and not report["gates"]["read"]["passed"]
    assert not report["passed"]


def test_a_selection_a_winner_record_made_is_inconclusive(decide):
    rows = {"rows": [_l2_row("a", True, True)], "winner_records": {"shipped": 0, "candidate": 1}}
    report = decide.decide_catalog(rows, candidate_report=_report(0.0), shipped_report=None)
    assert not report["gates"]["winner_records"]["passed"]


def test_regret_is_judged_against_the_shipped_ranker_with_slack(decide):
    rows = {"rows": [_l2_row("a", True, True)], "winner_records": {}}
    ok = decide.decide_catalog(rows, candidate_report=_report(0.024),
                               shipped_report=_report(0.02))
    worse = decide.decide_catalog(rows, candidate_report=_report(0.03),
                                  shipped_report=_report(0.02))
    assert ok["gates"]["regret"]["passed"] and not worse["gates"]["regret"]["passed"]


def test_a_native_ranker_is_compared_by_paired_gpu_time(decide):
    rows = {"rows": [_l2_row("a", True, True), _l2_row("b", True, True)], "winner_records": {}}
    for row in rows["rows"]:
        row["shipped"]["avgTimeMs"] = 1.0
        row["candidate"]["avgTimeMs"] = 100.0
    report = decide.decide_catalog(rows, candidate_report=_report(0.0), shipped_report=None,
                                   baseline_kind="native")
    assert not report["passed"] and report["gates"]["regret"]["mean_time_ratio"] == 100.0
    for row in rows["rows"]:
        row["candidate"]["avgTimeMs"] = 1.0
    assert decide.decide_catalog(rows, candidate_report=_report(0.0), shipped_report=None,
                                 baseline_kind="native")["passed"]


def test_a_first_ranker_needs_only_to_be_scored_when_nothing_ships(decide):
    rows = {"rows": [_l2_row("a", True, True)], "winner_records": {}}
    first = decide.decide_catalog(rows, candidate_report=_report(0.02), shipped_report=None,
                                  baseline_kind="none")
    assert first["passed"] and first["gates"]["regret"]["baseline"] == "none"
    assert not decide.decide_catalog(rows, candidate_report=None, shipped_report=None,
                                     baseline_kind="none")["passed"]


# --- the L2 run reader, against a fake engine ---------------------------------------


def test_a_run_records_refusals_and_record_ordered_catalogs(l2_job, tmp_path):
    bench = _fake_bench(tmp_path, (
        f"import sys\nprint({REFUSED!r}, file=sys.stderr)\n"
        "print('[info] ingestor: ordered 4 catalog entries from a benchmarked "
        "record; heuristic ranking skipped', file=sys.stderr)\n"
        "print('{\"avgTimeMs\": 1.5}')\n"))
    row = l2_job.run(str(bench), tmp_path / "g.fb", "/plugins", "7", dict(os.environ), None, 60)
    assert row["ok"] and row["avgTimeMs"] == 1.5
    assert row["from_record"] and "does not expose as knobs" in row["refusals"][0]


def test_a_failed_run_keeps_the_runtimes_error_lines_not_the_teardown(l2_job, tmp_path):
    bench = _fake_bench(tmp_path, (
        "import sys\nprint('[error] [hip_kernel_provider_plugin] kpack kernel source mismatch',"
        " file=sys.stderr)\nprint('[info] Destroying Container\\n' * 40, file=sys.stderr)\n"
        "sys.exit(1)\n"))
    row = l2_job.run(str(bench), tmp_path / "g.fb", "/p", "7", dict(os.environ), None, 60)
    assert not row["ok"] and "kpack kernel source mismatch" in row["error"]


def test_a_call_that_never_answers_is_recorded_not_waited_on(l2_job, tmp_path):
    bench = _fake_bench(tmp_path, "import time\ntime.sleep(30)\n")
    row = l2_job.run(str(bench), tmp_path / "g.fb", "/p", "7", dict(os.environ), None, 1)
    assert not row["ok"] and row["timed_out"]


# --- the candidate tree -------------------------------------------------------------


def _installed_tree(tmp_path: Path) -> tuple:
    root = tmp_path / "installed"
    shipped = root / "asm_sdpa" / "predict_engine" / "gfx950"
    shipped.mkdir(parents=True)
    document = {"version": "1.0", "id": DECLARED_ID, "name": "shipped", "adapter": "tree_data",
                "tree_data": {"artifact": "asm_sdpa_engine_tflops.bin"}}
    (shipped / "asm_sdpa_engine_tflops.uhd.json").write_text(json.dumps(document))
    (shipped / "asm_sdpa_engine_tflops.bin").write_bytes(b"shipped")
    (root / "other").mkdir()
    (root / "other" / "unrelated.uhd.json").write_text(json.dumps({"id": "other"}))
    (root / "kpack").mkdir()
    (root / "kpack" / "provider_gfx950.kpack").write_bytes(b"archive")
    candidate = tmp_path / "model"
    candidate.mkdir()
    (candidate / "heuristic.uhd.json").write_text(json.dumps(
        {**document, "name": "candidate", "tree_data": {"artifact": "model.bin"},
         "trained_against": {"selector_revision": "rev-A"}}))
    (candidate / "model.bin").write_bytes(b"candidate")
    return root, candidate


def test_a_candidate_tree_holds_everything_it_resolves(tree, tmp_path):
    root, _ = _installed_tree(tmp_path)
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "kpack").symlink_to(root / "kpack", target_is_directory=True)
    farm = tmp_path / "farm"
    tree.build_farm(linked, farm)
    files = list(farm.rglob("*"))
    assert not [p for p in files if p.is_symlink()]
    assert (farm / "kpack" / "provider_gfx950.kpack").read_bytes() == b"archive"


def test_the_candidate_is_the_only_document_carrying_the_declared_id(tree, tmp_path):
    root, candidate = _installed_tree(tmp_path)
    farm = tmp_path / "farm"
    tree.build_farm(root, farm)
    install = tree.install_opaque(farm, candidate, DECLARED_ID)
    assert tree.carrying(farm, DECLARED_ID) == [farm / install["installed"]]
    assert any(p.endswith("asm_sdpa_engine_tflops.bin") for p in install["removed"])
    assert (farm / "other" / "unrelated.uhd.json").is_file()
    assert (root / "asm_sdpa" / "predict_engine" / "gfx950" /
            "asm_sdpa_engine_tflops.bin").read_bytes() == b"shipped", "the build is untouched"
    with pytest.raises(SystemExit, match="built once"):
        tree.build_farm(root, farm)


# --- the L1 run, end to end against a fake engine -----------------------------------

_FAKE_L1_BENCH = r'''
import json, os, sys
args = sys.argv[1:]
graph = args[args.index("--graph") + 1]
index = int(os.path.basename(graph).split(".")[0][1:])
truth = 100.0 + 10 * index
binding = {"engine": "ASM_SDPA_ENGINE", "selector_revision": "rev-A", "arch": "gfx950",
           "role": "predict_engine", "trained_against": {"selector_revision": "rev-A"},
           "metric": "tflops"}
if "--collect-immediate" in args:
    flops = 1e12
    print(json.dumps({"engine_id": int(args[args.index("--engine-id") + 1]),
                      "engine_name": "ASM_SDPA_ENGINE", "graph_id": f"graph-{index}",
                      "device_id": "gpu#0", "arch": "gfx950", "selection_mode": "immediate",
                      "timing_statistic": "robustMeanMs", "is_valid": True, "binding": binding,
                      "features": {"graph.flops": flops, "device.cu_count": 256},
                      "avgTimeMs": flops / truth / 1e9, "robustMeanMs": flops / truth / 1e9,
                      "stddevMs": 0.0, "iters": 10, "constraints": {}}))
else:
    candidate = bool(os.environ.get("HIPDNN_DESCRIPTOR_DIR"))
    print(json.dumps({"status": "available", "model": "m", "reason": "", "metric": "tflops",
                      "value": truth * (1.02 if candidate else 1.25), "binding": binding}))
'''


def test_the_l1_run_and_its_verdict_end_to_end(tmp_path):
    pytest.importorskip("pandas")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(TOOLS_DIR), env.get("PYTHONPATH")]))
    probe = subprocess.run([sys.executable, "-c", "import uhd_gen.immediate"], env=env,
                           capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip(f"uhd_gen is not importable here: {probe.stderr.strip()[-200:]}")
    root, candidate = _installed_tree(tmp_path)
    bench = _fake_bench(tmp_path, _FAKE_L1_BENCH)
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    for i in range(8):
        (graphs / f"g{i}.fb").write_bytes(b"\x00graph")
    rows = tmp_path / "rows.json"
    run = subprocess.run(
        [sys.executable, str(SCRIPTS / "validate_l1.py"), "--bench", str(bench),
         "--plugin-dir", "/plugins", "--engine-id", "-7", "--descriptor-root", str(root),
         "--candidate-dir", str(candidate), "--declared-id", DECLARED_ID,
         "--graphs", str(graphs), "--work-dir", str(tmp_path / "work"), "--out", str(rows)],
        env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    document = json.loads(rows.read_text())
    assert document["tree"]["unique"] and document["tree"]["byte_equal"]
    assert len(document["rows"]) == 8

    report_path = tmp_path / "report.json"
    verdict = subprocess.run(
        [sys.executable, str(SCRIPTS / "decide.py"), "l1", "--rows", str(rows),
         "--candidate-dir", str(candidate), "--out", str(report_path)],
        capture_output=True, text=True)
    report = json.loads(report_path.read_text())
    assert verdict.returncode == 0, report["gates"]
    paired = report["metrics"]["paired"]
    assert paired["candidate"]["median_abs_pct_error"] == pytest.approx(2.0)
    assert paired["shipped"]["median_abs_pct_error"] == pytest.approx(25.0)
