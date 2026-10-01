# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Collection must never associate a timing with an unaddressed candidate."""
import copy
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
from uhd_gen.generate import (
    _catalog_label,
    collect_graph,
    feature_recipe,
    withheld_kernel_fields,
)
from uhd_gen.provenance import snapshot_provenance


def _page():
    return {
        "engine_id": 7,
        "graph_id": "graph",
        "device_id": "board",
        "device_arch": "gfx942",
        "problem_features": {"attention.query.dims[2]": 128, "graph.flops": 4e12},
        "device_features": {"device.cu_count": 120},
        "engine_descriptor_id": "13ab344f-4818-4772-bb8e-8e1441fec82c",
        "engine_name": "hipkernel:test",
        "candidates": [
            {
                "id": "kernel-a",
                "knob_settings": {"tile_m": 32},
                "kernel_features": {"kernel.tile_m": 32},
            }
        ],
        "total_count": 1,
        "next_offset": None,
    }


def _timing(page):
    candidate = page["candidates"][0]
    return {
        key: copy.deepcopy(page[key])
        for key in (
            "engine_id",
            "graph_id",
            "device_id",
            "device_arch",
            "engine_descriptor_id",
            "engine_name",
            "problem_features",
            "device_features",
        )
    } | {
        "results": [
            {
                "candidate_id": candidate["id"],
                "knob_settings": candidate["knob_settings"],
                "kernel_features": candidate["kernel_features"],
                "succeeded": True,
                "is_valid": True,
                "numerically_valid": True,
                "validation": "agrees_with_catalog: 2 of 2 cross-checked candidates produced this output",
                "robust_time_ms": 2.5,
                "min_time_ms": 2.0,
                "avg_time_ms": 2.6,
                "stddev_ms": 0.01,
                "iterations": 40,
            }
        ]
    }


def _sweep(page, *results):
    """The single `--sweep --json` response: the catalog it enumerated AND what it timed.

    One process per graph enumerates and times, so both halves arrive together; the old
    protocol paid a second bench startup per graph to get the first half.
    """
    response = copy.deepcopy(page)
    response["results"] = list(results) if results else [_timing(page)["results"][0]]
    return response


def _collect(monkeypatch, tmp_path, responses, calls=None):
    iterator = iter(responses)

    def _record(command, *args):
        if calls is not None:
            calls.append(command)
        return next(iterator)

    monkeypatch.setattr("uhd_gen.generate._run_json", _record)
    return collect_graph(
        ["hipdnn_bench", "--graph", "graph.json", "--engine-id", "7"],
        {},
        tmp_path,
        [],
        engine_descriptor_id="13ab344f-4818-4772-bb8e-8e1441fec82c",
    )


@pytest.mark.parametrize(
    "mutation", ["candidate", "knobs", "kernel_features", "provenance"]
)
def test_timing_must_resolve_exact_enrolled_identity(monkeypatch, tmp_path, mutation):
    page = _page()
    timed = _timing(page)
    if mutation == "provenance":
        page["engine_descriptor_id"] = "not-the-trained-engine"
    elif mutation == "candidate":
        timed["results"][0]["candidate_id"] = "kernel-b"
    elif mutation == "knobs":
        timed["results"][0]["knob_settings"] = {"tile_m": 64}
    else:
        timed["results"][0]["kernel_features"] = {"kernel.tile_m": 64}
    with pytest.raises(ValueError):
        _collect(monkeypatch, tmp_path, [_sweep(page, timed["results"][0])])


def test_incomplete_enumeration_is_not_a_training_corpus(monkeypatch, tmp_path):
    page = _page()
    page["total_count"] = 2
    with pytest.raises(ValueError, match="truncation"):
        _collect(monkeypatch, tmp_path, [_sweep(page)])


def test_ambiguous_enrolled_tuple_is_rejected_before_timing(monkeypatch, tmp_path):
    page = _page()
    duplicate = copy.deepcopy(page["candidates"][0])
    duplicate["id"] = "kernel-b"
    page["candidates"].append(duplicate)
    page["total_count"] = 2
    with pytest.raises(ValueError, match="unique"):
        _collect(monkeypatch, tmp_path, [_sweep(page)])


def test_feature_suffixed_device_uses_bare_training_architecture(monkeypatch, tmp_path):
    page = _page()
    page["device_arch"] = "gfx942:sramecc+:xnack-"
    rows, _ = _collect(monkeypatch, tmp_path, [_sweep(page)])
    assert {row["arch"] for row in rows} == {"gfx942"}


def test_collected_rows_carry_the_noise_columns_and_the_calibrated_rate(
    monkeypatch, tmp_path
):
    """§8.3's envelope, and §11.1's cross-engine score off §11.2's statistic."""
    page = _page()
    row = _collect(monkeypatch, tmp_path, [_sweep(page)])[0][0]
    assert (row["stddevMs"], row["iters"]) == (0.01, 40)
    # graph.flops / (avgTimeMs * 1e9): the mean, not the robust mean, because §11.2
    # (:2003) pins a calibrated score to `avgTimeMs`. Off robust_time_ms=2.5 this
    # would read 1600.0 instead.
    assert row["tflops"] == pytest.approx(4e12 / (2.6 * 1e9))


def test_an_engine_that_publishes_no_work_count_gets_no_fabricated_throughput(
    monkeypatch, tmp_path
):
    """§8.3: `tflops` is derived from a declared FLOP count or it is absent."""
    page = _page()
    page["problem_features"].pop("graph.flops")
    row = _collect(monkeypatch, tmp_path, [_sweep(page)])[0][0]
    assert "tflops" not in row


def test_a_collected_corpus_activates_the_evaluate_noise_band(monkeypatch, tmp_path):
    """§8.5 records the spread "so it can be used": the band keys on these exact names.

    `evaluate` looks for `stddevMs`/`iters`. Collection wrote neither, so the band was
    unreachable from a generated corpus however the run was configured -- and the report
    blamed the target's units for it.
    """
    import pandas as pd
    from uhd_gen.evaluate import evaluate_corpus

    page = _page()
    second = {
        "id": "kernel-b",
        "knob_settings": {"tile_m": 64},
        "kernel_features": {"kernel.tile_m": 64},
    }
    first_result = _timing(page)["results"][0]
    second_result = copy.deepcopy(first_result)
    second_result.update(
        candidate_id="kernel-b",
        knob_settings={"tile_m": 64},
        kernel_features={"kernel.tile_m": 64},
        robust_time_ms=2.51,
        min_time_ms=2.01,
        avg_time_ms=2.61,
    )
    page["candidates"].append(second)
    page["total_count"] = 2
    rows, _ = _collect(
        monkeypatch, tmp_path, [_sweep(page, first_result, second_result)]
    )

    report = evaluate_corpus(
        pd.DataFrame(rows),
        lambda frame: frame["avgTimeMs"].to_numpy(dtype=float),
        target="avgTimeMs",
        objective="min",
        eval_fraction=1.0,
    ).report
    assert report["ties"]["noise_band_applied"] is True
    assert "standard errors" in report["ties"]["policy"]


def test_a_numerically_invalid_candidate_keeps_its_row_but_loses_its_timing(
    monkeypatch, tmp_path
):
    """RFC 0019 §13.2: recorded with its measurement suppressed and an invalid marker.

    The two halves are one rule. Dropping the row loses the failure surface the section
    wants the model to learn; keeping the timing hands the ranker the group's best time,
    because a kernel that does not compute the answer is the fastest one in it.
    """
    page = _page()
    timed = _timing(page)
    timed["results"][0].update(
        numerically_valid=False,
        validation="output_mismatch: tensor 'Y' element 2 is 9.9e+01",
    )
    row = _collect(monkeypatch, tmp_path, [_sweep(page, timed["results"][0])])[0][0]

    assert row["numerically_valid"] is False
    assert row["validation"].startswith("output_mismatch")
    # Suppressed where the row is built, so `corpus.json`, `corpus.csv` and the `evaluate`
    # regret pass that reads the corpus back all see the same absence.
    assert [
        row[column] for column in ("robustMeanMs", "minTimeMs", "avgTimeMs", "stddevMs")
    ] == [None] * 4
    # No derived rate either: a throughput computed from a suppressed time would put the
    # measurement back under a different column name.
    assert "tflops" not in row
    # The candidate stays visible: which kernel was wrong, on which problem, and why.
    assert (row["kernel"], row["benchmark"], row["succeeded"], row["is_valid"]) == (
        "kernel-a",
        "graph",
        True,
        True,
    )


def test_an_undecided_verdict_is_carried_rather_than_treated_as_correct(
    monkeypatch, tmp_path
):
    """A null verdict keeps its measurement but never claims the candidate was checked.

    Open Question 19(a) has not settled what reference each op validates against, so a
    single-candidate problem is genuinely undecidable. Suppressing those timings would
    train on nothing; recording them as `True` would be the silent pass §13.2 forbids.
    """
    page = _page()
    timed = _timing(page)
    timed["results"][0].update(
        numerically_valid=None, validation="no_reference: one candidate ran"
    )
    row = _collect(monkeypatch, tmp_path, [_sweep(page, timed["results"][0])])[0][0]

    assert row["numerically_valid"] is None
    assert row["robustMeanMs"] == 2.5


@pytest.mark.parametrize("missing", ["numerically_valid", "validation"])
def test_a_benchmark_that_records_no_verdict_is_refused(monkeypatch, tmp_path, missing):
    """§13.2 records the verdict on the row, so an absent one is a tool that did not check.

    Refused rather than defaulted. Any default is wrong: `True` inverts the oracle, and
    `None` would let a benchmark silently regress out of validating while the corpus still
    looks well-formed.
    """
    page = _page()
    timed = _timing(page)
    timed["results"][0].pop(missing)
    with pytest.raises(ValueError, match="numerical-validation verdict"):
        _collect(monkeypatch, tmp_path, [_sweep(page, timed["results"][0])])


def test_every_candidate_is_timed_by_one_invocation_per_graph(monkeypatch, tmp_path):
    """RFC 0019 §13.2: sweeping inside one process amortises load, build and compilation.

    A process per candidate paid all three once per row. The row content is unchanged --
    this pins the cost, which is the whole reason the sweep exists: two candidates, two
    invocations (enumerate, then sweep), not three.
    """
    page = _page()
    second = {
        "id": "kernel-b",
        "knob_settings": {"tile_m": 64},
        "kernel_features": {"kernel.tile_m": 64},
    }
    page["candidates"].append(second)
    page["total_count"] = 2
    first_result = _timing(page)["results"][0]
    second_result = copy.deepcopy(first_result)
    second_result.update(
        candidate_id="kernel-b",
        knob_settings={"tile_m": 64},
        kernel_features={"kernel.tile_m": 64},
    )

    calls = []
    rows, _ = _collect(
        monkeypatch, tmp_path, [_sweep(page, first_result, second_result)], calls
    )

    # ONE process for a two-candidate graph: the sweep enumerates and times together.
    # It was two -- an `enumerate` run that built the catalog and discarded its timings,
    # then the sweep that rebuilt the same catalog to time it.
    assert len(calls) == 1
    assert [row["kernel"] for row in rows] == ["kernel-a", "kernel-b"]
    assert "--sweep" in calls[0] and "--json" in calls[0]
    # The collection pins restrict the sweep; they are not replaced by one candidate's tuple,
    # which is what made the old protocol need one process per row.
    assert "--knob" not in calls[0]


def test_a_sweep_that_times_fewer_candidates_than_it_enumerated_is_not_a_corpus(
    monkeypatch, tmp_path
):
    """A subset is silent data loss: the rows are simply absent from the corpus.

    Reachable now in a way it was not before -- one process holds every candidate, so a
    crash or an early return takes the rest of the graph with it.
    """
    page = _page()
    page["candidates"].append(
        {
            "id": "kernel-b",
            "knob_settings": {"tile_m": 64},
            "kernel_features": {"kernel.tile_m": 64},
        }
    )
    page["total_count"] = 2
    only_one = _timing(page)["results"][0]

    with pytest.raises(
        ValueError, match="whole catalog|exactly the catalog it reported"
    ):
        _collect(monkeypatch, tmp_path, [_sweep(page, only_one)])


def _catalog(**columns):
    return pd.DataFrame(
        {"robustMeanMs": [2.5, 3.0], "avgTimeMs": [2.6, 3.1], **columns}
    )


def test_each_catalog_metric_takes_its_own_label_and_is_calibrated_on_the_mean():
    """RFC 0019 §13.4: `tflops` from FLOPs over avgTimeMs, `time` from avgTimeMs itself."""
    usable = _catalog(tflops=[1.5, 1.3])
    assert _catalog_label("tflops", usable, False, "sort_kernel_catalog") == (
        "tflops",
        "tflops",
        True,
        "avgTimeMs",
    )
    assert _catalog_label("time", usable, False, "sort_kernel_catalog") == (
        "time",
        "avgTimeMs",
        True,
        "avgTimeMs",
    )


def test_an_unpublished_work_count_falls_back_only_when_no_metric_was_asked_for():
    """The default run keeps today's metric-less robustMeanMs ranker; a run that NAMED
    tflops gets an error rather than a model that estimates something else."""
    usable = _catalog()
    assert _catalog_label("tflops", usable, True, "sort_kernel_catalog") == (
        None,
        "robustMeanMs",
        False,
        "robustMeanMs",
    )
    with pytest.raises(ValueError, match="graph.flops"):
        _catalog_label("tflops", usable, False, "sort_kernel_catalog")


UED = "6d2b90f4-8c15-4a37-9e58-04b7c3fa1d62"
KMD = "3f8a1c07-52d9-4e61-b0a4-9c7d61e2830f"


def _immediate_bench(
    monkeypatch,
    provenance,
    *,
    flops=True,
    declared=None,
    broken=(),
    wrong=(),
    engine="provider:engine7",
):
    """A `--collect-immediate` stand-in; returns the metrics it was asked in, in order.

    `declared` maps metric -> the id the engine's description reports (binding.uhd_id);
    graphs whose id is in `broken` make the bench fail the way a crashing one does, and
    those in `wrong` come back with a failed correctness verdict.
    """
    requested = []

    def bench(command, environment, log_dir, ordinal, commands):
        commands.append({"argv": command})
        metric = command[command.index("--ranking-metric") + 1]
        requested.append(metric)
        graph = json.loads(
            Path(command[command.index("--graph") + 1]).read_text(encoding="utf-8")
        )
        if graph["id"] in broken:
            raise ValueError("hipdnn_bench failed (-11): segmentation fault")
        # A `time` request lets the engine's time ranker pick a faster kernel.
        average = (1.0 + graph["size"] / 10) * (0.8 if metric == "time" else 1.0)
        # As EnginePredictor/GenericEngine emit it: the selector revision, plus the descriptor
        # set a descriptor-backed engine loaded from.
        binding = {
            "engine": engine,
            "role": "predict_engine",
            "arch": "gfx942",
            "metric": metric,
            "selector_revision": "provider-1",
            "trained_against": {"selector_revision": "provider-1", **provenance},
        }
        if declared and metric in declared:
            binding["uhd_id"] = declared[metric]
        features = {
            "graph.nodes[0].dy.dims[0]": graph["size"] + 1,
            "device.cu_count": 120,
        }
        if flops:
            features["graph.flops"] = 2e9 * (graph["size"] + 1)
        response = {
            "engine_id": 7,
            "engine_name": engine,
            "graph_id": graph["id"],
            "device_id": "board",
            "arch": "gfx942",
            "metric": metric,
            "binding": binding,
            "features": features,
            "avgTimeMs": average,
            "robustMeanMs": average * 0.9,
            "stddevMs": 0.01,
            "iters": 30,
            "is_valid": True,
            "selection_mode": "immediate",
            "timing_statistic": "robustMeanMs",
        }
        if graph["id"] in wrong:
            response.update(
                numerically_valid=False, validation="output_mismatch: wrong output"
            )
        return response

    monkeypatch.setattr("uhd_gen.generate._run_json", bench)
    monkeypatch.setattr("uhd_gen.generate.shutil.which", lambda name: name)
    return requested


def _graphs(root, count=16):
    root.mkdir()
    for index in range(count):
        (root / f"{index}.json").write_text(
            json.dumps({"id": f"graph-{index}", "size": index}), encoding="utf-8"
        )
    return root


def _ued_tree(root):
    root.mkdir()
    (root / "engine.ued.json").write_text(
        json.dumps(
            {"version": "1.0", "id": UED, "name": "provider:engine7", "metadata": KMD}
        ),
        encoding="utf-8",
    )
    (root / "metadata.kmd.json").write_text(
        json.dumps({"version": "1.0", "id": KMD}), encoding="utf-8"
    )
    return root


def _l1_args(graphs, tree, output, evaluator, *extra, features="graph.flops"):
    return [
        "generate",
        "--graphs",
        str(graphs),
        "--descriptor-tree",
        str(tree),
        "--engine-id",
        "7",
        "--role",
        "predict_engine",
        "--features",
        features,
        "--num-boost-round",
        "4",
        "--early-stopping",
        "2",
        "--eval-fraction",
        "0.25",
        "--arch",
        "gfx942",
        "--feature-evaluator",
        evaluator,
        "--output-dir",
        str(output),
        *extra,
    ]


def test_one_generate_run_emits_and_installs_one_l1_model_per_metric(
    monkeypatch, tmp_path, evaluator
):
    """`--metric tflops time`: each metric's labels are measured under that metric (the
    engine's kernel choice follows it), each trains and evaluates its own UHD, and both
    land in the arch's role list rather than the second replacing the first."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("flatbuffers")
    from uhd_gen.__main__ import main

    tree = _ued_tree(tmp_path / "descriptors")
    requested = _immediate_bench(monkeypatch, snapshot_provenance(tree))
    output = tmp_path / "out"
    common = _l1_args(_graphs(tmp_path / "graphs"), tree, output, evaluator)
    # One bare id cannot name two UHDs.
    assert main([*common, "--metric", "tflops", "time", "--uhd-id", UED]) == 1
    assert not requested
    assert main([*common, "--metric", "tflops", "--metric", "time"]) == 0

    assert requested.count("tflops") == requested.count("time") == 16
    tflops_corpus = json.loads(
        (output / "corpus_tflops.json").read_text(encoding="utf-8")
    )
    time_corpus = json.loads((output / "corpus_time.json").read_text(encoding="utf-8"))
    assert time_corpus[0]["avgTimeMs"] == pytest.approx(
        0.8 * tflops_corpus[0]["avgTimeMs"]
    )
    emitted = {}
    for metric, objective in (("tflops", "max"), ("time", "min")):
        uhd = json.loads(
            (output / f"model_{metric}" / "heuristic.uhd.json").read_text(
                encoding="utf-8"
            )
        )
        report = json.loads(
            (output / f"model_{metric}" / "eval_report.json").read_text(
                encoding="utf-8"
            )
        )
        assert (
            uhd["score"]["metric"],
            uhd["score"]["calibrated"],
            uhd["objective"],
        ) == (metric, True, objective)
        assert (report["metric"], report["objective"]) == (metric, objective)
        emitted[metric] = uhd["id"]
    manifest = json.loads(
        (output / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert [model["metric"] for model in manifest["models"]] == ["tflops", "time"]
    ued = json.loads((tree / "engine.ued.json").read_text(encoding="utf-8"))
    assert ued["predict_engine"] == {"gfx942": [emitted["tflops"], emitted["time"]]}


TFLOPS_ID = "c47e1b3a-8f60-4a92-b5d4-1e08c9a27f63"
TIME_ID = "30284ebe-6e15-4f8e-968d-09f92d8a9480"


def test_an_opaque_time_model_over_a_corpus_without_flops_skips_one_bad_graph(
    monkeypatch, tmp_path, evaluator
):
    """T6 + 0.2: L1 `time` needs no work count (a conv-bwd corpus publishes none), one
    crashing graph is recorded and skipped rather than ending the run, and an engine with
    no UED trains and installs under the id its provider declares for the metric."""
    pytest.importorskip("lightgbm")
    from uhd_gen.__main__ import main

    tree = tmp_path / "descriptors"
    tree.mkdir()
    _immediate_bench(
        monkeypatch,
        {"selector_revision": "provider-1"},
        flops=False,
        declared={"time": TIME_ID},
        broken={"graph-3"},
        engine="MIOPEN_ENGINE",
    )
    output = tmp_path / "out"
    assert (
        main(
            _l1_args(
                _graphs(tmp_path / "graphs", 20),
                tree,
                output,
                evaluator,
                "--metric",
                "time",
                "--engine",
                "MIOPEN_ENGINE",
                "--max-graph-failures",
                "0.1",
                features="graph.nodes[0].dy.dims[0]",
            )
        )
        == 0
    )

    manifest = json.loads(
        (output / "generation_manifest.json").read_text(encoding="utf-8")
    )
    [failure] = manifest["failed_graphs"]
    assert (
        failure["source"].endswith("3.json")
        and "segmentation fault" in failure["error"]
    )
    assert "graph-3" not in {
        row["benchmark"]
        for row in json.loads((output / "corpus.json").read_text(encoding="utf-8"))
    }
    assert (
        json.loads(
            (output / "model" / "heuristic.uhd.json").read_text(encoding="utf-8")
        )["id"]
        == TIME_ID
    )
    [installed] = tree.rglob("*.uhd.json")
    assert json.loads(installed.read_text(encoding="utf-8"))["id"] == TIME_ID


def test_graph_failures_over_the_budget_fail_the_run_with_the_list(
    monkeypatch, tmp_path, evaluator, caplog
):
    tree = _ued_tree(tmp_path / "descriptors")
    _immediate_bench(
        monkeypatch, snapshot_provenance(tree), broken={"graph-3", "graph-4"}
    )
    from uhd_gen.__main__ import main

    assert (
        main(
            _l1_args(
                _graphs(tmp_path / "graphs"),
                tree,
                tmp_path / "out",
                evaluator,
                "--max-graph-failures",
                "0.1",
            )
        )
        == 1
    )
    assert (
        "2 of 16 graph(s) failed" in caplog.text
        and "3.json" in caplog.text
        and "4.json" in caplog.text
    )


@pytest.mark.parametrize(
    "requested,declared,message",
    [
        ([], None, "--uhd-id time=<uuid>"),
        ([f"time={TFLOPS_ID}"], {"time": TIME_ID}, "contradicts"),
    ],
)
def test_an_opaque_engine_is_never_trained_under_an_undeclared_id(
    monkeypatch, tmp_path, evaluator, caplog, requested, declared, message
):
    """Refused after the first graph, not after the whole corpus has been measured."""
    from uhd_gen.__main__ import main

    tree = tmp_path / "descriptors"
    tree.mkdir()
    calls = _immediate_bench(
        monkeypatch,
        {"selector_revision": "provider-1"},
        declared=declared,
        engine="MIOPEN_ENGINE",
    )
    uhd_ids = [argument for value in requested for argument in ("--uhd-id", value)]
    assert (
        main(
            _l1_args(
                _graphs(tmp_path / "graphs"),
                tree,
                tmp_path / "out",
                evaluator,
                "--metric",
                "time",
                *uhd_ids,
            )
        )
        == 1
    )
    assert message in caplog.text
    assert len(calls) == 1


def test_l1_generation_never_trains_on_picks_checked_wrong(
    monkeypatch, tmp_path, evaluator, caplog
):
    """§13.2 at the L1 entrance: every immediate pick came back wrong, so there is no label
    at all -- refused, with each row's verdict and reason kept in the preserved corpus.
    Import used to erase the verdict and this run trained and evaluated on 13 rows."""
    from uhd_gen.__main__ import main

    tree = _ued_tree(tmp_path / "descriptors")
    _immediate_bench(
        monkeypatch,
        snapshot_provenance(tree),
        wrong={f"graph-{index}" for index in range(16)},
    )
    output = tmp_path / "out"
    assert (
        main(
            _l1_args(
                _graphs(tmp_path / "graphs"), tree, output, evaluator, "--no-promote"
            )
        )
        == 1
    )
    assert "no successful valid timings" in caplog.text
    assert not output.exists()
    [stage] = tmp_path.glob(".uhd-generate-*")
    corpus = json.loads((stage / "corpus.json").read_text(encoding="utf-8"))
    assert len(corpus) == 16
    assert all(
        row["numerically_valid"] is False
        and row["validation"] == "output_mismatch: wrong output"
        and row["avgTimeMs"] is None
        for row in corpus
    )


@pytest.mark.parametrize("role", ["predict_engine", "sort_kernel_catalog"])
def test_the_measured_tree_is_the_only_descriptor_root_the_bench_sees(
    monkeypatch, tmp_path, role
):
    """The loader keeps the FIRST definition of an id across roots, so an inherited
    replacement or additive root ahead of the tree being generated against supplies its
    own selector, and the labels describe a tree this run never installs into. Both roles
    run with exactly one root: the given tree (L1) or its collection copy (L2)."""
    from uhd_gen.__main__ import main

    tree = _ued_tree(tmp_path / "descriptors")
    kmd = tree / "metadata.kmd.json"
    kmd.write_text(
        json.dumps(
            {"version": "1.0", "id": KMD, "fields": [{"name": "tile_m", "type": "int"}]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HIPDNN_DESCRIPTOR_DIR", str(tmp_path / "earlier"))
    monkeypatch.setenv("HIPDNN_DESCRIPTOR_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("HIPDNN_DESCRIPTOR_PATH", str(tmp_path / "additive"))
    seen = []

    def bench(command, environment, *args):
        seen.append(dict(environment))
        raise ValueError("stop after observing the child environment")

    monkeypatch.setattr("uhd_gen.generate._run_json", bench)
    monkeypatch.setattr("uhd_gen.generate.shutil.which", lambda name: name)
    arguments = [
        "generate",
        "--graphs",
        str(_graphs(tmp_path / "graphs", 1)),
        "--descriptor-tree",
        str(tree),
        "--engine-id",
        "7",
        "--role",
        role,
        "--arch",
        "gfx942",
        "--output-dir",
        str(tmp_path / "out"),
    ]
    assert main(arguments) == 1

    [environment] = seen
    assert "HIPDNN_DESCRIPTOR_PATH" not in environment
    assert "HIPDNN_DESCRIPTOR_RUNTIME_DIR" not in environment
    root = Path(environment["HIPDNN_DESCRIPTOR_DIR"])
    if role == "predict_engine":
        assert root == tree.resolve()
    else:
        assert root.name == "collection_descriptors" and root.parent.name.startswith(
            ".uhd-generate-"
        )


def _corpus_root(root):
    """What `hipdnn_corpus_gen --output` writes: graphs/ beside manifest.json and .csv."""
    (root / "graphs").mkdir(parents=True)
    rows = []
    for index in range(3):
        (root / "graphs" / f"conv_{index}.json").write_text(
            json.dumps({"id": f"g{index}"}), encoding="utf-8"
        )
        rows.append(
            {
                "benchmark": f"g{index}",
                "name": f"conv_{index}",
                "file": f"graphs/conv_{index}.json",
            }
        )
    (root / "graphs" / "stray.json").write_text("{}", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"tool": "hipdnn_corpus_gen", "graphs": rows}), encoding="utf-8"
    )
    return root


def test_a_corpus_root_is_read_through_its_manifest_graph_list(tmp_path):
    """T6: the manifest is never collected as a graph, and it -- not a directory walk --
    decides which graphs the corpus holds."""
    from uhd_gen.generate import discover_graphs

    root = _corpus_root(tmp_path / "corpus")
    expected = [
        (root / "graphs" / f"conv_{index}.json").resolve() for index in range(3)
    ]
    assert discover_graphs([str(root)]) == expected
    assert discover_graphs([str(root / "manifest.json")]) == expected
    # A directory above corpus roots is walked, and their manifests are skipped.
    assert all(
        path.name != "manifest.json" for path in discover_graphs([str(tmp_path)])
    )


def test_a_manifest_listing_a_missing_graph_is_refused(tmp_path):
    from uhd_gen.generate import discover_graphs

    root = _corpus_root(tmp_path / "corpus")
    (root / "graphs" / "conv_1.json").unlink()
    with pytest.raises(ValueError, match="conv_1.json"):
        discover_graphs([str(root)])


#: The gfx950 attention shape: the collection UED exposed every KMD field, the shipping UED
#: one knob, and the matcher binds `causal` from the graph.
_KMD_FIELDS = ["block_m", "causal", "ragged"]
_SHIPPING_KNOBS = ["block_m"]


def _attention_frame():
    return pd.DataFrame(
        {
            "kernel.block_m": [64, 128, 64, 128],
            "kernel.causal": [1, 1, 0, 0],
            "kernel.ragged": [0, 1, 0, 1],
            "gfx950_attention_dense.causal": [1, 1, 0, 0],
            "graph.flops": [1e12, 1e12, 2e12, 2e12],
            "device.cu_count": [256] * 4,
        }
    )


def test_generation_offers_kernel_features_only_for_the_shipping_knobs():
    """The runtime admits a ranker only if each `$kernel.*` axis is a knob of the UED that
    ships it; the collection UED exposes every KMD field, and proposing from it is how the
    gfx950 rankers came to read `$kernel.causal` and were never used. A graph-bound field is
    still learned from -- through its problem-side twin."""
    frame = _attention_frame()
    signature, _ = feature_recipe(
        frame, set(frame.columns), None, _KMD_FIELDS, _SHIPPING_KNOBS, []
    )

    assert (
        "$kernel.block_m" in signature and "$gfx950_attention_dense.causal" in signature
    )
    assert not {"$kernel.causal", "$kernel.ragged"} & set(signature)
    assert [
        (entry["field"], entry["reason"], entry.get("read_instead"))
        for entry in withheld_kernel_fields(
            _KMD_FIELDS, _SHIPPING_KNOBS, set(frame.columns)
        )
    ] == [
        ("causal", "graph_bound", "gfx950_attention_dense.causal"),
        ("ragged", "not_a_shipping_knob", None),
    ]


@pytest.mark.parametrize(
    "authored",
    [
        ["$kernel.causal", "$graph.flops"],
        [{"ceil_div": ["$graph.flops", "$kernel.ragged[0]"]}],
    ],
)
def test_an_authored_recipe_reading_an_unexposed_kernel_field_is_refused_before_training(
    authored,
):
    frame = _attention_frame()
    with pytest.raises(ValueError, match="does not expose as knobs"):
        feature_recipe(
            frame, set(frame.columns), authored, _KMD_FIELDS, _SHIPPING_KNOBS, []
        )
