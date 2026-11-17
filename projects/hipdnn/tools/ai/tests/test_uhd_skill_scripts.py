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


@pytest.fixture(scope="module")
def engine_run() -> ModuleType:
    return _load(SCRIPTS / "engine_run.py", "uhd_skill_engine_run")


#: What a caller's environment may hold that must not reach an isolated run.
LEAKY_ENV = {
    "HIPDNN_DESCRIPTOR_DIR": "/user/descriptors",
    "HIPDNN_DESCRIPTOR_RUNTIME_DIR": "/user/runtime_descriptors",
    "HIPDNN_DESCRIPTOR_PATH": "/user/a:/user/b",
    "HIPDNN_CACHE_DIR": "/user/cache",
}

#: A fake bench's first lines: exit 3 unless the run reads one named tree at most and starts
#: from an empty cache of its own; then leave a winner record there, as a real run does.
_ISOLATION_GUARD = r"""
import os, sys
_leaked = [v for v in ("HIPDNN_DESCRIPTOR_RUNTIME_DIR", "HIPDNN_DESCRIPTOR_PATH")
           if v in os.environ]
_tree = os.environ.get("HIPDNN_DESCRIPTOR_DIR", "")
_cache = os.environ.get("HIPDNN_CACHE_DIR", "")
if (_leaked or _tree.startswith("/user") or not _cache or _cache.startswith("/user")
        or os.listdir(_cache)):
    print(f"not isolated: {_leaked} tree={_tree!r} cache={_cache!r}", file=sys.stderr)
    sys.exit(3)
open(os.path.join(_cache, "winner.record"), "w").write("record")
"""


def _scripts_env(**extra) -> dict:
    """This environment with the tools importable (uhd_gen) and `extra` set."""
    env = dict(os.environ, **extra)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(TOOLS_DIR), env.get("PYTHONPATH")])
    )
    return env


def _require_uhd_gen(env: dict) -> None:
    probe = subprocess.run(
        [sys.executable, "-c", "import uhd_gen.immediate"],
        env=env,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip(f"uhd_gen is not importable here: {probe.stderr.strip()[-200:]}")


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
    return [
        {"benchmark": f"{regime}-{i}", "regime": regime}
        for regime, n in counts.items()
        for i in range(n)
    ]


def test_every_regime_with_two_graphs_is_scored_and_none_is_held_out_whole(slicing):
    chosen = slicing.choose(_graphs({"big": 100, "pair": 2, "lone": 1}), 0.2, seed=0)
    assert chosen["by_regime"] == {
        "big": {"graphs": 100, "held_out": 20},
        "lone": {"graphs": 1, "held_out": 0},
        "pair": {"graphs": 2, "held_out": 1},
    }
    assert len(chosen["benchmarks"]) == 21


def test_the_slice_depends_on_the_set_and_seed_alone(slicing):
    graphs = _graphs({"a": 40, "b": 15})
    first = slicing.choose(graphs, 0.2, seed=3)
    assert slicing.choose(list(reversed(graphs)), 0.2, seed=3) == first, "order-free"
    assert slicing.choose(graphs, 0.2, seed=4)["benchmarks"] != first["benchmarks"]


def test_unlabelled_graphs_are_their_own_group_and_a_bad_fraction_is_refused(slicing):
    chosen = slicing.choose([{"benchmark": f"g{i}"} for i in range(10)], 0.2)
    assert chosen["by_regime"] == {"unlabelled": {"graphs": 10, "held_out": 2}}
    with pytest.raises(ValueError, match="0 < f < 1"):
        slicing.choose(_graphs({"a": 5}), 1.0)
    with pytest.raises(ValueError, match="no graph ids"):
        slicing.choose([{"regime": "a"}])


def test_the_default_slice_is_a_tenth_of_each_regime(slicing):
    chosen = slicing.choose(
        _graphs({"big": 100, "small": 10}), slicing.DEFAULT_FRACTION
    )
    assert chosen["by_regime"] == {
        "big": {"graphs": 100, "held_out": 10},
        "small": {"graphs": 10, "held_out": 1},
    }


def test_a_sample_takes_every_regime_before_any_twice(slicing):
    graphs = _graphs({"big": 50, "mid": 3, "lone": 1})
    sampled = slicing.sample_evenly(graphs, 5, seed=0)
    regimes = [g["regime"] for g in sampled]
    assert sorted(regimes[:3]) == ["big", "lone", "mid"], "one of each regime first"
    assert regimes.count("big") == 2 and regimes.count("mid") == 2
    assert (
        slicing.sample_evenly(list(reversed(graphs)), 5, seed=0) == sampled
    ), "order-free"
    assert len(slicing.sample_evenly(graphs, 1000)) == len(
        graphs
    ), "all of them, when fewer"


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


def test_the_cli_slices_every_operation_once_and_resolves_graph_files(
    slicing, tmp_path
):
    manifests = _write_set(
        tmp_path / "set", {"conv_fwd": {"a": 20}, "matmul": {"b": 10}}
    )
    run = lambda *args: subprocess.run(
        [sys.executable, str(SLICE_SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
    )
    assert run("choose", *manifests).returncode == 0
    written = tmp_path / "set" / "eval_slice.json"
    document = json.loads(written.read_text())
    assert document["by_regime"] == {
        "a": {"graphs": 20, "held_out": 2},
        "b": {"graphs": 10, "held_out": 1},
    }
    again = run("choose", *manifests)
    assert again.returncode != 0 and "chosen once" in again.stderr

    assert run("holdout", written, "--out", tmp_path / "held.json").returncode == 0
    assert json.loads((tmp_path / "held.json").read_text()) == document["benchmarks"]

    assert (
        run(
            "sample", written, "--count", "2", "--out", tmp_path / "sample.json"
        ).returncode
        == 0
    )
    sample = json.loads((tmp_path / "sample.json").read_text())["graphs"]
    assert len(sample) == 2 and all(Path(g["path"]).is_file() for g in sample)

    clean = {"models": [{"training_problem_keys": [["not-held-out", "gpu"]]}]}
    (tmp_path / "clean.json").write_text(json.dumps(clean))
    assert (
        run(
            "sample",
            written,
            "--trained",
            tmp_path / "clean.json",
            "--out",
            tmp_path / "s2.json",
        ).returncode
        == 0
    )
    leaked = {
        "models": [{"training_problem_keys": [[document["benchmarks"][0], "gpu"]]}]
    }
    (tmp_path / "leaked.json").write_text(json.dumps(leaked))
    refused = run(
        "sample",
        written,
        "--trained",
        tmp_path / "leaked.json",
        "--out",
        tmp_path / "s3.json",
    )
    assert refused.returncode != 0 and "training graphs" in refused.stderr

    exported = [{"benchmark": b, "tflops": 1.0} for b in document["benchmarks"]]
    exported.append({"benchmark": "trained-on", "tflops": 1.0})
    (tmp_path / "exported.json").write_text(json.dumps(exported))
    assert (
        run(
            "rows", tmp_path / "exported.json", written, "--out", tmp_path / "rows.json"
        ).returncode
        == 0
    )
    kept = json.loads((tmp_path / "rows.json").read_text())
    assert {r["benchmark"] for r in kept} == set(document["benchmarks"])


# --- engine predictor (L1) verdict --------------------------------------------------


def _rows(
    n=10,
    *,
    candidate_status="available",
    revision="rev-A",
    candidate_bias=1.05,
    incumbent_bias=1.30,
    measured=True,
    metric="tflops",
):
    rows = []
    for i in range(n):
        truth = 100.0 + 10 * i
        row = {
            "graph": f"g{i}.fb",
            "incumbent": {
                "status": "available",
                "value": truth * incumbent_bias,
                "metric": "tflops",
                "selector_revision": revision,
            },
            "candidate": {
                "status": candidate_status,
                "value": truth * candidate_bias,
                "metric": metric,
                "selector_revision": revision,
            },
        }
        if measured:
            row["measured"] = {"ok": True, "value": truth, "metric": "tflops"}
        rows.append(row)
    return rows


def _document(rows, *, unique=True, byte_equal=True, measured=True):
    return {
        "measured": measured,
        "metric": "tflops",
        "rows": rows,
        "tree": {
            "unique": unique,
            "byte_equal": byte_equal,
            "candidate_id": DECLARED_ID,
            "carrying_declared_id": ["candidate_8b61d0c9/heuristic.uhd.json"],
        },
    }


def _decide_l1(decide, document, **kwargs):
    kwargs.setdefault("candidate_selector_revision", "rev-A")
    return decide.decide(
        document, declared_id=DECLARED_ID, candidate_id=DECLARED_ID, **kwargs
    )


def test_a_candidate_that_cannot_be_scored_fails_the_accuracy_gate(decide):
    """Every prediction NaN: the candidate has no error to compare, which is a failure, not
    a pass by default."""
    rows = _rows()
    for row in rows:
        row["candidate"]["value"] = float("nan")
    report = _decide_l1(
        decide,
        _document(rows),
        min_available=0.0,
        max_regression=decide.DEFAULT_L1_MAX_REGRESSION,
    )
    assert not report["gates"]["accuracy"]["passed"]
    assert not report["passed"]


def test_fewer_executed_graphs_than_asked_fail_the_available_gate(decide):
    report = _decide_l1(decide, _document(_rows(n=10)), min_graphs=11)
    assert not report["gates"]["available"]["passed"]
    assert _decide_l1(decide, _document(_rows(n=10)), min_graphs=10)["gates"][
        "available"
    ]["passed"]


def test_a_good_candidate_passes_and_is_compared_with_the_incumbent_model(decide):
    report = _decide_l1(
        decide, _document(_rows()), max_regression=decide.DEFAULT_L1_MAX_REGRESSION
    )
    assert report["passed"], report["gates"]
    paired = report["metrics"]["paired"]
    assert paired["candidate"]["median_abs_pct_error"] == pytest.approx(5.0)
    assert paired["incumbent"]["median_abs_pct_error"] == pytest.approx(30.0)
    assert paired["candidate"]["spearman"] == pytest.approx(1.0)


def test_a_tree_where_the_engine_could_read_another_model_fails_identity(decide):
    report = _decide_l1(decide, _document(_rows(), unique=False))
    assert not report["passed"] and not report["gates"]["identity"]["passed"]


def test_a_candidate_under_another_id_fails_identity(decide):
    report = decide.decide(
        _document(_rows()),
        declared_id=DECLARED_ID,
        candidate_id="11111111-2222-3333-4444-555555555555",
        candidate_selector_revision="rev-A",
    )
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


def test_a_candidate_model_of_another_metric_fails(decide):
    """The bench echoes the requested metric; the descriptor says what the model predicts."""
    gate = _decide_l1(decide, _document(_rows()), candidate_metric="time")["gates"][
        "metric"
    ]
    assert not gate["passed"] and "predicts time" in gate["problems"][0]
    assert _decide_l1(decide, _document(_rows()), candidate_metric="tflops")["gates"][
        "metric"
    ]["passed"]


def test_an_answer_that_names_no_metric_fails(decide):
    rows = _rows()
    rows[0]["candidate"]["metric"] = None
    gate = _decide_l1(decide, _document(rows))["gates"]["metric"]
    assert not gate["passed"] and "named no metric" in gate["problems"][0]
    assert gate["answered"] == ["tflops"]


def test_a_less_accurate_candidate_fails_the_default_tolerance(decide):
    worse = _document(_rows(candidate_bias=1.5, incumbent_bias=1.1))
    assert _decide_l1(decide, worse)["passed"], "no tolerance, no accuracy gate"
    report = _decide_l1(decide, worse, max_regression=decide.DEFAULT_L1_MAX_REGRESSION)
    assert not report["passed"] and not report["gates"]["accuracy"]["passed"]


def test_with_no_incumbent_the_candidate_needs_only_to_be_scored(decide):
    rows = _rows()
    for row in rows:
        row["incumbent"] = {"status": "unavailable"}
    report = _decide_l1(
        decide, _document(rows), max_regression=0.05, incumbent_baseline=False
    )
    assert report["passed"] and report["baseline"] == "none"
    assert not _decide_l1(decide, _document(rows), max_regression=0.05)[
        "passed"
    ], "an incumbent that answers nothing is inconclusive unless there is none"


def test_each_l1_is_scored_against_the_selection_it_is_installed_beside(decide):
    def row(graph, measured, measured_incumbent, candidate, incumbent):
        return {
            "graph": graph,
            "measured": {"ok": True, "value": measured},
            "measured_incumbent": {"ok": True, "value": measured_incumbent},
            "candidate": {
                "status": "available",
                "value": candidate,
                "selector_revision": "r",
                "metric": "tflops",
            },
            "incumbent": {
                "status": "available",
                "value": incumbent,
                "selector_revision": "r",
                "metric": "tflops",
            },
        }

    rows = {
        "measured": True,
        "metric": "tflops",
        "tree": {"unique": True, "byte_equal": True},
        "rows": [row("a", 100, 50, 100, 50), row("b", 200, 80, 200, 80)],
    }
    report = decide.decide(
        rows, declared_id="c", candidate_id="c", candidate_selector_revision="r"
    )
    assert report["metrics"]["incumbent"]["median_abs_pct_error"] == 0.0
    assert report["metrics"]["incumbent_truth"] == "its own selection"


def test_too_few_graphs_is_not_a_pass(decide):
    assert not _decide_l1(decide, _document([]))["passed"]


# --- catalog ranker (L2) verdict ----------------------------------------------------

REFUSED = (
    "[error] uhd: engine 'e' ranks on [dtype], which its UED does not expose as knobs [];"
    " the model is not used and kernels rank by priority, then id"
)


def _l2_row(name, incumbent, candidate, refusal=None):
    return {
        "graph": f"/g/{name}.fb",
        "incumbent": {"ok": incumbent, "refusals": []},
        "candidate": {"ok": candidate, "refusals": [refusal] if refusal else []},
    }


def _report(mean):
    return {"metrics": {"top1_regret": {"mean": mean}}}


def test_a_refusal_both_runs_report_is_another_models_not_the_candidates(decide):
    """The candidate tree copies the whole root, so another engine's refused model is
    refused in both runs; only what the candidate run alone reports fails `read`."""
    other = (
        "descriptor loader: engine 'other' model=1 disabled: x; declared-order fallback"
    )
    shared = _l2_row("a", True, True, refusal=other)
    shared["incumbent"]["refusals"] = [other]
    rows = {"rows": [shared], "winner_records": {}}
    gate = decide.decide_catalog(
        rows, candidate_report=_report(0.0), incumbent_report=None
    )["gates"]["read"]
    assert gate["passed"], gate

    rows = {"rows": [_l2_row("b", True, True, refusal=REFUSED)], "winner_records": {}}
    gate = decide.decide_catalog(
        rows, candidate_report=_report(0.0), incumbent_report=None
    )["gates"]["read"]
    assert not gate["passed"] and "not used" in gate["problems"][0]


def test_a_graph_the_build_tree_cannot_run_is_the_engines_failure(decide):
    rows = {
        "rows": [_l2_row("a", True, True), _l2_row("b", False, False)],
        "winner_records": {"incumbent": 0, "candidate": 0},
    }
    report = decide.decide_catalog(
        rows, candidate_report=_report(0.01), incumbent_report=_report(0.02)
    )
    gate = report["gates"]["available"]
    assert gate["passed"] and (gate["runnable"], gate["engine_failures"]) == (1, 1)
    assert report["passed"]


def test_the_candidate_failing_where_the_build_tree_runs_fails_availability(decide):
    rows = {
        "rows": [_l2_row("a", True, True), _l2_row("b", True, False)],
        "winner_records": {},
    }
    gate = decide.decide_catalog(
        rows, candidate_report=_report(0.0), incumbent_report=None
    )["gates"]["available"]
    assert not gate["passed"] and "1 of 2" in gate["problems"][0]


def test_a_ranker_the_runtime_declines_fails_even_though_every_plan_built(decide):
    rows = {"rows": [_l2_row("a", True, True, REFUSED)], "winner_records": {}}
    report = decide.decide_catalog(
        rows, candidate_report=_report(0.0), incumbent_report=None
    )
    assert (
        report["gates"]["available"]["passed"] and not report["gates"]["read"]["passed"]
    )
    assert not report["passed"]


def test_a_selection_a_winner_record_made_is_inconclusive(decide):
    rows = {
        "rows": [_l2_row("a", True, True)],
        "winner_records": {"incumbent": 0, "candidate": 1},
    }
    report = decide.decide_catalog(
        rows, candidate_report=_report(0.0), incumbent_report=None
    )
    assert not report["gates"]["winner_records"]["passed"]


def test_regret_is_judged_against_the_incumbent_ranker_with_slack(decide):
    rows = {"rows": [_l2_row("a", True, True)], "winner_records": {}}
    ok = decide.decide_catalog(
        rows, candidate_report=_report(0.024), incumbent_report=_report(0.02)
    )
    worse = decide.decide_catalog(
        rows, candidate_report=_report(0.03), incumbent_report=_report(0.02)
    )
    assert ok["gates"]["regret"]["passed"] and not worse["gates"]["regret"]["passed"]


def test_a_native_ranker_is_compared_by_paired_gpu_time(decide):
    rows = {
        "rows": [_l2_row("a", True, True), _l2_row("b", True, True)],
        "winner_records": {},
    }
    for row in rows["rows"]:
        row["incumbent"]["avgTimeMs"] = 1.0
        row["candidate"]["avgTimeMs"] = 100.0
    report = decide.decide_catalog(
        rows,
        candidate_report=_report(0.0),
        incumbent_report=None,
        baseline_kind="native",
    )
    assert (
        not report["passed"] and report["gates"]["regret"]["mean_time_ratio"] == 100.0
    )
    for row in rows["rows"]:
        row["candidate"]["avgTimeMs"] = 1.0
    assert decide.decide_catalog(
        rows,
        candidate_report=_report(0.0),
        incumbent_report=None,
        baseline_kind="native",
    )["passed"]


def test_a_first_ranker_needs_only_to_be_scored_when_there_is_no_incumbent(decide):
    rows = {"rows": [_l2_row("a", True, True)], "winner_records": {}}
    first = decide.decide_catalog(
        rows,
        candidate_report=_report(0.02),
        incumbent_report=None,
        baseline_kind="none",
    )
    assert first["passed"] and first["gates"]["regret"]["baseline"] == "none"
    assert not decide.decide_catalog(
        rows, candidate_report=None, incumbent_report=None, baseline_kind="none"
    )["passed"]


# --- the L2 run reader, against a fake engine ---------------------------------------


def test_a_run_records_refusals_and_record_ordered_catalogs(l2_job, tmp_path):
    bench = _fake_bench(
        tmp_path,
        (
            f"import sys\nprint({REFUSED!r}, file=sys.stderr)\n"
            "print('[info] ingestor: ordered 4 catalog entries from a benchmarked "
            "record; heuristic ranking skipped', file=sys.stderr)\n"
            "print('{\"avgTimeMs\": 1.5}')\n"
        ),
    )
    row = l2_job.run(
        str(bench),
        tmp_path / "g.fb",
        "/plugins",
        "7",
        dict(os.environ),
        None,
        60,
        tmp_path / "caches",
    )
    assert row["ok"] and row["avgTimeMs"] == 1.5
    assert row["from_record"] and "does not expose as knobs" in row["refusals"][0]


@pytest.mark.parametrize(
    "refusal",
    [
        # DescriptorLoader.hpp, a model its provenance or arch does not admit.
        "[error] descriptor loader: engine 'hipkernel:X' role=sort_kernel_catalog "
        "arch='gfx942' model=1 disabled: arch-relevant UMD is missing from training "
        "provenance: 2; engine remains available with declared-order fallback",
        # DescriptorLoader.hpp, an unregistered symbol or an artifact outside the tree.
        "[error] descriptor loader: engine 'hipkernel:X' arch='gfx942' model=1 names "
        "unregistered score symbol 's'; disabling model, preserving engine",
        # The custom-library adapter.
        "[error] CustomLibraryAdapter: no hash declared for library lib.so; an unverified "
        "library is not loaded",
        # UhdKernelHeuristic.hpp, no model for the device's arch.
        "[info] uhd trace: hipkernel:X decided_by=declared_order reason=no_model_for_arch "
        "metric=tflops arch=gfx942 candidates=4",
    ],
)
def test_every_way_the_runtime_declines_a_model_is_a_refusal(l2_job, tmp_path, refusal):
    """The L2 gate passes only when the candidate was used; each of the runtime's refusal
    messages means it was not."""
    bench = _fake_bench(
        tmp_path,
        (
            f"import sys\nprint({refusal!r}, file=sys.stderr)\n"
            "print('{\"avgTimeMs\": 1.5}')\n"
        ),
    )
    row = l2_job.run(
        str(bench),
        tmp_path / "g.fb",
        "/plugins",
        "7",
        dict(os.environ),
        None,
        60,
        tmp_path / "caches",
    )
    # Recorded without the log prefix, so the two runs' lines compare equal.
    assert row["refusals"] == [l2_job.LOG_PREFIX.sub("", refusal).strip()]


def test_a_failed_run_keeps_the_runtimes_error_lines_not_the_teardown(l2_job, tmp_path):
    bench = _fake_bench(
        tmp_path,
        (
            "import sys\nprint('[error] [hip_kernel_provider_plugin] kpack kernel source mismatch',"
            " file=sys.stderr)\nprint('[info] Destroying Container\\n' * 40, file=sys.stderr)\n"
            "sys.exit(1)\n"
        ),
    )
    row = l2_job.run(
        str(bench),
        tmp_path / "g.fb",
        "/p",
        "7",
        dict(os.environ),
        None,
        60,
        tmp_path / "caches",
    )
    assert not row["ok"] and "kpack kernel source mismatch" in row["error"]


def test_a_call_that_never_answers_is_recorded_not_waited_on(l2_job, tmp_path):
    bench = _fake_bench(tmp_path, "import time\ntime.sleep(30)\n")
    row = l2_job.run(
        str(bench),
        tmp_path / "g.fb",
        "/p",
        "7",
        dict(os.environ),
        None,
        1,
        tmp_path / "caches",
    )
    assert not row["ok"] and row["timed_out"]


# --- the candidate tree -------------------------------------------------------------


def _installed_tree(tmp_path: Path) -> tuple:
    root = tmp_path / "installed"
    incumbent = root / "opaque_engine" / "predict_engine" / "gfx950"
    incumbent.mkdir(parents=True)
    document = {
        "version": "1.0",
        "id": DECLARED_ID,
        "name": "incumbent",
        "adapter": "tree_data",
        "tree_data": {"artifact": "engine_tflops.bin"},
    }
    (incumbent / "engine_tflops.uhd.json").write_text(json.dumps(document))
    (incumbent / "engine_tflops.bin").write_bytes(b"incumbent")
    (root / "other").mkdir()
    (root / "other" / "unrelated.uhd.json").write_text(json.dumps({"id": "other"}))
    (root / "kpack").mkdir()
    (root / "kpack" / "provider_gfx950.kpack").write_bytes(b"archive")
    candidate = tmp_path / "model"
    candidate.mkdir()
    (candidate / "heuristic.uhd.json").write_text(
        json.dumps(
            {
                **document,
                "name": "candidate",
                "tree_data": {"artifact": "model.bin"},
                "trained_against": {"selector_revision": "rev-A"},
            }
        )
    )
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
    assert any(p.endswith("engine_tflops.bin") for p in install["removed"])
    assert (farm / "other" / "unrelated.uhd.json").is_file()
    assert (
        root / "opaque_engine" / "predict_engine" / "gfx950" / "engine_tflops.bin"
    ).read_bytes() == b"incumbent", "the build is untouched"
    with pytest.raises(SystemExit, match="built once"):
        tree.build_farm(root, farm)


# --- the L1 run, end to end against a fake engine -----------------------------------

#: The engine under test is staged only in the root the validation names, as a test descriptor
#: set or an engine staged outside the build's default tree is: with no tree named, the
#: runtime does not know it, and a side that reads the build's default tree fails every graph.
_STAGED_ONLY_IN_ROOT = r"""
if not _tree:
    print("[error] Engine id is not in a valid range of engine IDs", file=sys.stderr)
    sys.exit(4)
"""


_FAKE_L1_BENCH = (
    _ISOLATION_GUARD
    + _STAGED_ONLY_IN_ROOT
    + r"""
import json, os, sys
args = sys.argv[1:]
graph = args[args.index("--graph") + 1]
index = int(os.path.basename(graph).split(".")[0][1:])
truth = 100.0 + 10 * index
binding = {"engine": "OPAQUE_ENGINE", "selector_revision": "rev-A", "arch": "gfx950",
           "role": "predict_engine", "trained_against": {"selector_revision": "rev-A"},
           "metric": "tflops"}
if "--collect-immediate" in args:
    flops = 1e12
    print(json.dumps({"engine_id": int(args[args.index("--engine-id") + 1]),
                      "engine_name": "OPAQUE_ENGINE", "graph_id": f"graph-{index}",
                      "graph_key": f"key-{index}",
                      "device_id": "gpu#0", "arch": "gfx950", "selection_mode": "immediate",
                      "timing_statistic": "robustMeanMs", "is_valid": True, "binding": binding,
                      "features": {"graph.flops": flops, "device.cu_count": 256},
                      "avgTimeMs": flops / truth / 1e9, "robustMeanMs": flops / truth / 1e9,
                      "stddevMs": 0.0, "iters": 10, "constraints": {}}))
else:
    candidate = "descriptor_tree" in os.environ.get("HIPDNN_DESCRIPTOR_DIR", "")
    print(json.dumps({"status": "available", "model": "m", "reason": "", "metric": "tflops",
                      "value": truth * (1.02 if candidate else 1.25), "binding": binding}))
"""
)


def test_the_l1_run_and_its_verdict_end_to_end(tmp_path):
    """The caller's descriptor roots and cache reach no run (the fake bench exits 3 if they
    do), the incumbent side reads the root it is given (the engine exists nowhere else), and
    the candidate is scored against the incumbent on the same graphs."""
    pytest.importorskip("pandas")
    env = _scripts_env(**LEAKY_ENV)
    _require_uhd_gen(env)
    root, candidate = _installed_tree(tmp_path)
    bench = _fake_bench(tmp_path, _FAKE_L1_BENCH)
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    for i in range(8):
        (graphs / f"g{i}.fb").write_bytes(b"\x00graph")
    rows = tmp_path / "rows.json"
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "validate_l1.py"),
            "--bench",
            str(bench),
            "--plugin-dir",
            "/plugins",
            "--engine-id",
            "-7",
            "--descriptor-root",
            str(root),
            "--candidate-dir",
            str(candidate),
            "--declared-id",
            DECLARED_ID,
            "--graphs",
            str(graphs),
            "--work-dir",
            str(tmp_path / "work"),
            "--out",
            str(rows),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    document = json.loads(rows.read_text())
    assert document["tree"]["unique"] and document["tree"]["byte_equal"]
    assert len(document["rows"]) == 8
    assert all(
        row[side].get("status") == "available" and row["measured"]["ok"]
        for row in document["rows"]
        for side in ("incumbent", "candidate")
    ), document["rows"][0]
    assert not list(
        (tmp_path / "work" / "caches").iterdir()
    ), "every run's cache is removed"

    report_path = tmp_path / "report.json"
    verdict = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "decide.py"),
            "l1",
            "--rows",
            str(rows),
            "--candidate-dir",
            str(candidate),
            "--out",
            str(report_path),
        ],
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text())
    assert verdict.returncode == 0, report["gates"]
    paired = report["metrics"]["paired"]
    assert paired["candidate"]["median_abs_pct_error"] == pytest.approx(2.0)
    assert paired["incumbent"]["median_abs_pct_error"] == pytest.approx(25.0)


# --- one isolated engine run --------------------------------------------------------


def test_an_isolated_run_reads_only_the_tree_it_names(engine_run, tmp_path):
    # The caller's roots, cache and log file are not inherited, and nothing autotunes.
    build = engine_run.isolated_env(
        base=dict(LEAKY_ENV, HIPDNN_LOG_FILE="/user/log", HIPDNN_FORCE_BENCHMARKING="1")
    )
    assert build == {"HIPDNN_FORCE_BENCHMARKING": "0"}
    named = engine_run.isolated_env(tmp_path, base=LEAKY_ENV)
    assert named["HIPDNN_DESCRIPTOR_DIR"] == str(tmp_path)
    assert not {"HIPDNN_DESCRIPTOR_RUNTIME_DIR", "HIPDNN_DESCRIPTOR_PATH"} & set(named)


def test_every_call_starts_from_an_empty_cache_of_its_own(engine_run, tmp_path):
    bench = _fake_bench(tmp_path, _ISOLATION_GUARD + "print('{\"avgTimeMs\": 2.0}')\n")
    env = dict(os.environ, **LEAKY_ENV)
    env.pop("HIPDNN_DESCRIPTOR_DIR")
    env = engine_run.isolated_env(base=env)
    scratch = tmp_path / "caches"
    for _ in range(2):  # the second would find the first's record in a shared cache
        result = engine_run.call([str(bench)], env, 60, scratch)
        assert result["ok"], result["stderr"]
        assert result["response"] == {"avgTimeMs": 2.0}
    assert not list(scratch.iterdir())


def test_the_answer_is_the_first_json_object_whatever_surrounds_it(
    engine_run, tmp_path
):
    printed = (
        "[info] loading plugins {not json}\n"
        '{"status": "available", "value": 3.5}\n'
        "[info] Destroying Container {}\n"
    )
    bench = _fake_bench(tmp_path, f"print({printed!r})\n")
    result = engine_run.call([str(bench)], dict(os.environ), 60, tmp_path / "c")
    assert result["ok"] and result["response"] == {"status": "available", "value": 3.5}


@pytest.mark.parametrize(
    "body, error",
    [
        (
            "import sys\nprint('kpack mismatch', file=sys.stderr)\nsys.exit(2)\n",
            "kpack mismatch",
        ),
        ("print('[info] no answer here')\n", "no JSON object"),
    ],
)
def test_a_failed_or_silent_call_is_not_an_answer(engine_run, tmp_path, body, error):
    bench = _fake_bench(tmp_path, body)
    result = engine_run.call([str(bench)], dict(os.environ), 60, tmp_path / "c")
    assert not result["ok"] and error in result["error"] and "response" not in result


def test_graphs_come_from_a_sample_or_from_directories_without_manifests(
    engine_run, tmp_path
):
    (tmp_path / "set" / "graphs").mkdir(parents=True)
    for name in ("b.fb", "a.json", "manifest.json"):
        (tmp_path / "set" / "graphs" / name).write_text("{}")
    lone = tmp_path / "lone.fb"
    lone.write_text("")
    found = engine_run.graphs_of(None, [str(tmp_path / "set"), str(lone)], None)
    assert found == sorted(
        [lone, *(tmp_path / "set" / "graphs" / n for n in ("a.json", "b.fb"))]
    )
    assert engine_run.graphs_of(None, [str(tmp_path / "set")], 1) == [
        tmp_path / "set" / "graphs" / "a.json"
    ]
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({"graphs": [{"path": str(lone)}]}))
    assert engine_run.graphs_of(str(sample), [], None) == [lone]


# --- candidate_tree.py as a command -------------------------------------------------

_PRINT_ENV = (
    "import json, os; print(json.dumps({k: v for k, v in os.environ.items() "
    "if k.startswith(('HIPDNN_DESCRIPTOR', 'HIPDNN_CACHE'))}))"
)


def test_the_command_runs_against_the_copy_and_nothing_else(tmp_path):
    root, candidate = _installed_tree(tmp_path)
    farm, record = tmp_path / "farm", tmp_path / "installed.json"
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "candidate_tree.py"),
            "--root",
            str(root),
            "--farm",
            str(farm),
            "--opaque",
            DECLARED_ID.upper(),
            str(candidate),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--record",
            str(record),
            "--",
            sys.executable,
            "-c",
            _PRINT_ENV,
        ],
        env=dict(os.environ, **LEAKY_ENV),
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    seen = json.loads(run.stdout.strip().splitlines()[-1])
    assert seen == {
        "HIPDNN_DESCRIPTOR_DIR": str(farm),
        "HIPDNN_CACHE_DIR": str(tmp_path / "cache"),
    }
    installed = json.loads(record.read_text())["installed"]
    assert [i["declared_id"] for i in installed] == [
        DECLARED_ID
    ], "ids compare lower-case"


@pytest.fixture
def promotable(tmp_path):
    """A descriptor engine's tree and a model `uhd_gen promote` installs into it, from
    uhd_gen's own promotion tests; skips where those cannot run."""
    env = _scripts_env()
    _require_uhd_gen(env)
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    fixtures = pytest.importorskip("uhd_gen.tests.test_promote")
    from uhd_gen.features import resolve_feature_evaluator

    try:
        resolve_feature_evaluator()
    except ValueError as error:
        pytest.skip(str(error))
    tree = fixtures._tree(tmp_path / "build" / "gfx942")
    model = fixtures._model(tmp_path / "model", tree)
    return {"root": tree, "model": model, "new_id": fixtures.NEW, "env": env}


def test_a_promoted_candidate_is_bound_in_the_copy_and_the_build_is_untouched(
    tree, promotable, tmp_path, monkeypatch
):
    monkeypatch.setenv("PYTHONPATH", promotable["env"]["PYTHONPATH"])
    before = (promotable["root"] / "engine.ued.json").read_text()
    farm = tmp_path / "farm"
    tree.build_farm(promotable["root"], farm)
    install = tree.install_promoted(
        farm, promotable["model"], "test:engine", "sort_kernel_catalog", "gfx942"
    )
    assert install["roles_after"]["sort_kernel_catalog"]["gfx942"] == [
        promotable["new_id"]
    ]
    assert tree.carrying(farm, promotable["new_id"])
    assert (promotable["root"] / "engine.ued.json").read_text() == before
    with pytest.raises(SystemExit, match="UED"):
        tree.install_promoted(
            farm, promotable["model"], "no:such", "sort_kernel_catalog", "gfx942"
        )


def test_a_promote_runs_the_evaluator_it_is_given(
    tree, promotable, tmp_path, monkeypatch
):
    """promote is handed the build's evaluator: one that does not exist is refused by its
    name, whatever promote could find on its own, and the real one installs."""
    from uhd_gen.features import resolve_feature_evaluator

    evaluator = resolve_feature_evaluator()
    monkeypatch.setenv("PYTHONPATH", promotable["env"]["PYTHONPATH"])
    absent = str(tmp_path / "no-such-build" / "hipdnn_uhd_features")
    refused = tmp_path / "refused"
    tree.build_farm(promotable["root"], refused)
    # The refusal quotes the path as a repr, which doubles Windows separators, so the
    # match is on the directory only this test names.
    with pytest.raises(SystemExit, match="no-such-build"):
        tree.install_promoted(
            refused,
            promotable["model"],
            "test:engine",
            "sort_kernel_catalog",
            "gfx942",
            absent,
        )
    named = tmp_path / "named"
    tree.build_farm(promotable["root"], named)
    install = tree.install_promoted(
        named,
        promotable["model"],
        "test:engine",
        "sort_kernel_catalog",
        "gfx942",
        evaluator,
    )
    assert install["roles_after"]["sort_kernel_catalog"]["gfx942"] == [
        promotable["new_id"]
    ]


_FAKE_L2_BENCH = (
    _ISOLATION_GUARD
    + _STAGED_ONLY_IN_ROOT
    + r"""
import json
candidate = "descriptor_tree" in _tree
if candidate:
    print("[error] uhd: the model is not used and kernels rank by priority, then id",
          file=sys.stderr)
print(json.dumps({"avgTimeMs": 2.0 if candidate else 1.0, "graph_key": "g", "device_id": 0}))
"""
)


def test_the_l2_run_and_its_verdict_end_to_end(promotable, tmp_path):
    """Both sides run the engine, staged only in the root, so the verdict rests on whether the
    runtime read the candidate: here it declines it, and the ranker is never installed.
    """
    bench = _fake_bench(tmp_path, _FAKE_L2_BENCH)
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    for i in range(3):
        (graphs / f"g{i}.fb").write_bytes(b"\x00")
    rows = tmp_path / "rows.json"
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "validate_l2.py"),
            "--bench",
            str(bench),
            "--plugin-dir",
            "/plugins",
            "--engine-id",
            "7",
            "--engine",
            "test:engine",
            "--root",
            str(promotable["root"]),
            "--role",
            "sort_kernel_catalog",
            "--arch-key",
            "gfx942",
            "--candidate-dir",
            str(promotable["model"]),
            "--graphs",
            str(graphs),
            "--work-dir",
            str(tmp_path / "work"),
            "--out",
            str(rows),
        ],
        env=dict(promotable["env"], **LEAKY_ENV),
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    document = json.loads(rows.read_text())
    assert document["graphs"] == 3 and document["winner_records"] == {
        "incumbent": 0,
        "candidate": 0,
    }
    first = document["rows"][0]
    assert first["incumbent"] == {
        "ok": True,
        "refusals": [],
        "from_record": False,
        "avgTimeMs": 1.0,
        "benchmark": "g",
        "device": 0,
    }
    assert first["candidate"]["ok"] and "not used" in first["candidate"]["refusals"][0]

    (tmp_path / "candidate.json").write_text(json.dumps(_report(0.01)))
    report_path = tmp_path / "report.json"
    verdict = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "decide.py"),
            "l2",
            "--rows",
            str(rows),
            "--candidate-report",
            str(tmp_path / "candidate.json"),
            "--baseline",
            "none",
            "--out",
            str(report_path),
        ],
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text())
    assert verdict.returncode == 1, "a ranker the runtime declines is never installed"
    assert (
        not report["gates"]["read"]["passed"] and report["gates"]["available"]["passed"]
    )
    assert report["comparison"] == (
        "not compared: the checkout binds no model of this role " "for this arch yet"
    )


# --- what an installing user is told ------------------------------------------------


@pytest.mark.parametrize(
    "role, metrics, said",
    [
        (
            "predict_engine",
            {
                "candidate": {"median_abs_pct_error": 4.0},
                "incumbent": {"median_abs_pct_error": 3.0},
            },
            "less accurate",
        ),
        (
            "predict_engine",
            {
                "candidate": {"median_abs_pct_error": 2.0},
                "incumbent": {"median_abs_pct_error": 3.0},
            },
            None,
        ),
        (
            "predict_engine",
            {"candidate": {"median_abs_pct_error": 2.0}, "incumbent": None},
            "answered none",
        ),
        (
            "sort_kernel_catalog",
            {"candidate": {"mean": 0.03}, "incumbent": {"mean": 0.02}},
            "worse than the incumbent ranker",
        ),
        (
            "sort_kernel_catalog",
            {"candidate": {"mean": 0.01}, "incumbent": {"mean": None}},
            "there was none to score",
        ),
        (
            "sort_kernel_catalog",
            {"baseline": "native", "paired": {"mean_time_ratio": 1.2}},
            "slower than native",
        ),
        (
            "sort_kernel_catalog",
            {"baseline": "native", "paired": {"mean_time_ratio": 0.9}},
            None,
        ),
        ("sort_kernel_catalog", {"baseline": "none"}, "binds no model"),
    ],
)
def test_worse_at_all_or_not_compared_is_said(decide, role, metrics, said):
    note = decide.comparison_warning(role, metrics)
    assert (note is None) if said is None else (said in note)


def test_a_descriptor_engines_l1_is_measured_under_the_tree_it_is_promoted_into(
    promotable, tmp_path
):
    bench = _fake_bench(tmp_path, _FAKE_L1_BENCH)
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    for i in range(2):
        (graphs / f"g{i}.fb").write_bytes(b"\x00")
    rows = tmp_path / "rows.json"
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "validate_l1.py"),
            "--bench",
            str(bench),
            "--plugin-dir",
            "/plugins",
            "--engine-id",
            "-7",
            "--descriptor-root",
            str(promotable["root"]),
            "--candidate-dir",
            str(promotable["model"]),
            "--declared-id",
            promotable["new_id"],
            "--graphs",
            str(graphs),
            "--promote",
            "test:engine",
            "sort_kernel_catalog",
            "gfx942",
            str(promotable["model"]),
            "--work-dir",
            str(tmp_path / "work"),
            "--out",
            str(rows),
        ],
        env=dict(promotable["env"], **LEAKY_ENV),
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    document = json.loads(rows.read_text())
    assert document["tree"]["unique"] and document["tree"]["byte_equal"], document[
        "tree"
    ]
    assert [i["kind"] for i in document["tree"]["installs"]] == ["descriptor"]
    assert all(
        row["measured"]["ok"] and row["measured_incumbent"]["ok"]
        for row in document["rows"]
    ), "truth under each tree"


# --- definitions shared with uhd_gen ----------------------------------------------------


def test_the_slice_hash_is_the_one_uhd_gen_splits_by(slicing):
    """eval_slice ranks a graph by the digest evaluate's split uses for a shape, so a slice
    chosen here and a split drawn there agree on the same seed."""
    pytest.importorskip("pandas")
    from uhd_gen.evaluate import _problem_digest

    for benchmark, seed in (("1102948572918374651", 0), ("g-7", 42)):
        assert slicing._rank(benchmark, seed) == _problem_digest((benchmark,), seed)


def test_stage_v_accuracy_and_the_sizing_target_measure_error_alike(decide, tmp_path):
    """The sizing target (within X%, median |error|) is read against Stage V's accuracy
    gate: both summarise the same relative errors the same way."""
    pytest.importorskip("pandas")
    from uhd_gen.sizing import _score

    pairs = [(110.0, 100.0), (95.0, 100.0), (130.0, 100.0), (100.0, 100.0)]
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "per_row": [
                    {"key": [f"g{i}"], "signed_relative_error": (p - m) / m}
                    for i, (p, m) in enumerate(pairs)
                ]
            }
        ),
        encoding="utf-8",
    )
    stage_v = decide.accuracy(pairs)
    sizing = _score(report, 0.10)
    assert sizing["median_abs_pct"] == pytest.approx(stage_v["median_abs_pct_error"])
    assert sizing["within"] == pytest.approx(stage_v["within_10pct"])
