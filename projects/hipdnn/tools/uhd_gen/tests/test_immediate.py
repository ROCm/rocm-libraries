# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""L1 labels describe an engine's immediate execution, not a tuned configuration."""
import copy
import json
from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from uhd_gen.features import compute_features_hash
from uhd_gen.immediate import (
    ROLE,
    evaluate_immediate,
    normalize_corpus,
    normalize_row,
    prediction_scorer,
    validate_model,
    validate_signature,
)
from uhd_gen.promote import _apply, build_plan


UHD = "727e5401-3b99-49ff-a2fc-68fd4eedbb54"
UED = "6d2b90f4-8c15-4a37-9e58-04b7c3fa1d62"
KMD = "3f8a1c07-52d9-4e61-b0a4-9c7d61e2830f"


PROVENANCE = {
    "ued": {"id": UED, "revision": "1.0"},
    "kmd": {"id": KMD, "revision": "1.0"},
    "umd": [],
}


def measurement(*, engine=7, graph="graph", elapsed=2.0, robust=2.5, metric="tflops"):
    name = f"provider:engine{engine}"
    selector = "provider-1/immediate-2/library-3"
    # trained_against as GenericEngine emits it: the selector revision beside the descriptor
    # set the engine loaded from.
    return {
        "engine_id": engine,
        "engine_name": name,
        "graph_id": graph,
        "device_id": "board",
        "arch": "gfx942",
        "binding": {
            "engine": name,
            "role": ROLE,
            "arch": "gfx942",
            "metric": metric,
            "selector_revision": selector,
            "trained_against": {
                **copy.deepcopy(PROVENANCE),
                "selector_revision": selector,
            },
        },
        "features": {"graph.flops": 2e12, "graph.nodes": 1, "device.cu_count": 120},
        # RFC 0019.13 §11.2 (:2003) pins a calibrated score to `avgTimeMs`, so that
        # is the label. `robustMeanMs` rides along as §8.5's informational statistic
        # and is deliberately a DIFFERENT number here: a label read from the wrong
        # column produces a wrong TFLOPS rather than the same one.
        "avgTimeMs": elapsed,
        "robustMeanMs": robust,
        "stddevMs": 0.05,
        "iters": 30,
        "is_valid": True,
        "selection_mode": "immediate",
        "timing_statistic": "robustMeanMs",
    }


# Callers take the `evaluator` fixture: the digest below is the runtime's, computed by the
# shared binary, because RFC 0019 §6.3 leaves features_hash with exactly one definition.
def descriptor(row, metric="tflops", signature=("$graph.flops",)):
    return {
        "version": "1.0",
        "id": UHD,
        "name": "immediate model",
        "adapter": "tree_data",
        "trained_against": row["binding"]["trained_against"],
        "objective": {"tflops": "max", "time": "min"}[metric],
        "score": {"metric": metric, "calibrated": True, "transform": "log1p"},
        "features_signature": list(signature),
        "features_hash": compute_features_hash(list(signature)),
        "tree_data": {"artifact": "model.bin"},
    }


def bundle(
    row, prediction, training_keys=(), metric="tflops", signature=("$graph.flops",)
):
    # The owning engine is recorded by the training manifest's binding, never by the
    # descriptor: RFC 0019 Section 3.1 leaves that binding to the UED role map.
    scorer = (
        prediction
        if callable(prediction)
        else lambda frame: np.full(len(frame), prediction)
    )
    return SimpleNamespace(
        descriptor=descriptor(row, metric, signature),
        manifest={
            "training_problem_keys": list(training_keys),
            "binding": copy.deepcopy(row["binding"]),
            "arch": row["arch"],
        },
        scorer=scorer,
    )


def test_import_derives_physical_throughput_from_full_graph_and_mean_timing():
    """§11.2 (:2003)/§10.6.2 (:1914-1916): a calibrated score trains on `avgTimeMs`."""
    row = normalize_row(measurement())
    # 2e12 flops / (2.0 ms * 1e9). Off robustMeanMs=2.5 this would read 800.
    assert row["tflops"] == 1000.0
    assert row["timing_statistic"] == "avgTimeMs"
    assert row["robustMeanMs"] == 2.5, "§8.5's statistic stays, informationally"
    assert (row["stddevMs"], row["iters"]) == (0.05, 30), "§8.3's noise columns survive"
    assert normalize_row(row)["tflops"] == row["tflops"]


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True])
def test_invalid_timing_cannot_become_a_training_label(bad):
    row = measurement(elapsed=bad)
    with pytest.raises(ValueError):
        normalize_row(row)


def test_unknown_work_count_and_supplied_rate_are_not_substitutes_for_flops():
    row = measurement()
    row["tflops"] = 999.0
    with pytest.raises(ValueError):
        normalize_row(row)
    row["features"].pop("graph.flops")
    with pytest.raises(ValueError):
        normalize_row(row)


@pytest.mark.parametrize(
    "where", ["envelope", "feature", "expression", "sweep", "label"]
)
def test_candidate_information_cannot_enter_an_l1_corpus_or_recipe(where):
    row = measurement()
    if where == "expression":
        with pytest.raises(ValueError):
            validate_signature([{"/": ["$graph.flops", "$kernel.tile_m"]}])
    elif where == "label":
        # The label is envelope, never input: a feature that reads it would be fitting
        # the answer. `_LEAKED_FIELDS` is exempted for the row, never for the recipe.
        with pytest.raises(ValueError):
            validate_signature(["$graph.avgTimeMs"])
    elif where == "sweep":
        with pytest.raises(ValueError):
            normalize_corpus(pd.DataFrame([row, copy.deepcopy(row)]))
    else:
        if where == "envelope":
            row["candidate_id"] = "winner"
        else:
            row["features"]["kernel.tile_m"] = 64
        with pytest.raises(ValueError):
            normalize_row(row)


def test_engine_selector_provenance_is_the_descriptor_set_and_cannot_drift_mid_corpus():
    row = measurement()
    row["binding"]["trained_against"] = {
        "engine": {"name": row["engine_name"], "version": "v1"}
    }
    with pytest.raises(ValueError):
        normalize_row(row)

    first, second = measurement(), measurement(graph="other")
    second["binding"]["selector_revision"] = "provider-1/immediate-9/library-3"
    with pytest.raises(ValueError):
        normalize_corpus(pd.DataFrame([first, second]))

    third = measurement(graph="third")
    third["binding"]["trained_against"]["ued"]["revision"] = "2.0"
    with pytest.raises(ValueError):
        normalize_corpus(pd.DataFrame([first, third]))


def test_cross_engine_labels_require_the_same_full_graph_work_count():
    first, second = measurement(), measurement(engine=8)
    second["features"]["graph.flops"] *= 2
    with pytest.raises(ValueError):
        normalize_corpus(pd.DataFrame([first, second]))


def test_a_failed_correctness_verdict_survives_import_without_its_label():
    """RFC 0019 §13.2: the row is kept with its verdict and reason, its timing is not.

    Import used to return a row with neither field and the wrong pick's `avgTimeMs` as an
    ordinary label, so nothing after it -- training, evaluation, promotion -- could tell.
    """
    wrong = measurement()
    wrong.update(numerically_valid=False, validation="output_mismatch: wrong output")
    row = normalize_row(wrong)
    assert row["numerically_valid"] is False
    assert row["validation"] == "output_mismatch: wrong output"
    for label in ("avgTimeMs", "robustMeanMs", "stddevMs", "tflops"):
        assert row[label] is None, label
    # Read back as `corpus.json` is, the verdict and the absence both hold.
    [again] = normalize_corpus(pd.DataFrame([row])).to_dict(orient="records")
    assert (again["numerically_valid"], again["validation"]) == (
        False,
        row["validation"],
    )
    assert pd.isna(again["avgTimeMs"])

    # Undecided is carried as undecided, measurement intact -- never promoted to True.
    unknown = measurement(graph="other")
    unknown.update(numerically_valid=None, validation="no_reference: one engine ran")
    row = normalize_row(unknown)
    assert row["numerically_valid"] is None and row["validation"].startswith("no_ref")
    assert row["avgTimeMs"] == 2.0


def test_a_pick_checked_wrong_is_not_the_oracle_of_selection_regret(evaluator):
    """The wrong engine ran fastest; it is out of the held-out rows and the oracle, so the
    valid engine is compared against nothing rather than charged regret against a wrong
    answer."""
    wrong, valid = measurement(elapsed=1), measurement(engine=8, elapsed=4)
    wrong.update(numerically_valid=False, validation="output_mismatch: wrong output")
    report = evaluate_immediate(
        pd.DataFrame([wrong, valid]),
        [bundle(wrong, 2000), bundle(valid, 500)],
        eval_fraction=1,
        seed=0,
        include_per_problem=True,
    )
    assert report["metrics"]["prediction_coverage"]["rows"] == 1
    [problem] = report["per_problem"]
    assert (problem["engines"], problem["picked_engine"]) == (1, 8)
    assert report["metrics"]["immediate_selection"]["problems_compared"] == 0


@pytest.mark.parametrize("transform", [None, ""])
def test_an_omitted_transform_is_the_runtimes_identity(evaluator, transform):
    """`score_transform::SUPPORTED_TRANSFORMS` lists "" and `applyInverse` returns the raw
    score for it, so the engine admits an L1 model that declares no transform."""
    model = descriptor(measurement())
    if transform is None:
        del model["score"]["transform"]
    else:
        model["score"]["transform"] = transform
    assert validate_model(model).name == "tflops"
    model["score"]["transform"] = "sqrt"
    with pytest.raises(ValueError, match="transform"):
        validate_model(model)


def test_single_engine_predictions_have_signed_errors_without_fake_ranking_regret(
    evaluator,
):
    row = measurement()
    report = evaluate_immediate(
        pd.DataFrame([row]), [bundle(row, 1200)], eval_fraction=1, seed=0
    )
    calibration = report["metrics"]["calibration"]
    assert calibration["signed_bias_tflops"] == 200
    assert calibration["signed_relative_bias"] == pytest.approx(0.2)
    assert report["metrics"]["immediate_selection"]["problems_compared"] == 0
    assert report["metrics"]["immediate_selection"]["regret"]["mean"] is None


def test_selection_regret_compares_other_immediate_engines_not_tuned_candidates(
    evaluator,
):
    fast, slow = measurement(), measurement(engine=8, elapsed=4)
    report = evaluate_immediate(
        pd.DataFrame([fast, slow]),
        [bundle(fast, 400), bundle(slow, 600)],
        eval_fraction=1,
        seed=0,
        include_per_problem=True,
    )
    assert report["metrics"]["immediate_selection"]["regret"]["mean"] == pytest.approx(
        0.5
    )
    assert report["per_problem"][0]["picked_engine"] == 8
    assert (
        report["metrics"]["per_engine"][fast["engine_name"]]["signed_bias_tflops"]
        == -600
    )


def test_holdout_is_checked_by_graph_device_keys_not_input_filename(evaluator):
    row = measurement()
    report = evaluate_immediate(
        pd.DataFrame([row]),
        [bundle(row, 1000, [("graph", "board")])],
        eval_fraction=1,
        seed=0,
    )
    assert report["holdout_integrity"]["status"] == "COMPROMISED"


def test_runtime_prediction_must_match_measured_request_but_can_use_new_l1_model(
    evaluator,
):
    row = measurement()
    row["binding"]["uhd_id"] = "old-model"
    response = copy.deepcopy(row)
    response.update(model=UHD, status="available", metric="tflops", value=1234)
    response["binding"]["uhd_id"] = UHD
    model = descriptor(row)
    scorer = prediction_scorer(model, [response])
    assert scorer(normalize_corpus(pd.DataFrame([row]))).tolist() == [1234]
    changed = copy.deepcopy(row)
    changed["features"]["constraint.workspace_limit"] = 0
    with pytest.raises(ValueError):
        scorer(normalize_corpus(pd.DataFrame([changed])))
    # The answer names its metric, and one in another metric is refused, never converted.
    wrong_metric = copy.deepcopy(response)
    wrong_metric["metric"] = "time"
    with pytest.raises(ValueError, match="metric"):
        prediction_scorer(model, [wrong_metric])


def test_time_predictions_rank_lower_first_and_report_in_milliseconds(evaluator):
    """RFC 0019 §4.4: `time` is avgTimeMs directly and lower wins, so the engine with the
    lower predicted time is picked and the report is keyed by `time`, not throughput."""
    fast, slow = measurement(metric="time"), measurement(
        engine=8, elapsed=4, metric="time"
    )
    report = evaluate_immediate(
        pd.DataFrame([fast, slow]),
        [bundle(fast, 3.0, metric="time"), bundle(slow, 2.5, metric="time")],
        eval_fraction=1,
        seed=0,
        include_per_problem=True,
    )
    assert (report["metric"], report["target"], report["objective"]) == (
        "time",
        "avgTimeMs",
        "min",
    )
    assert report["per_problem"][0]["picked_engine"] == 8
    assert report["per_problem"][0]["best_immediate_time"] == 2.0
    assert report["metrics"]["immediate_selection"]["regret"]["mean"] == pytest.approx(
        1.0
    )
    assert report["metrics"]["per_engine"][fast["engine_name"]][
        "signed_bias_time"
    ] == pytest.approx(1.0)


def test_engines_are_only_compared_in_one_metric(evaluator):
    first, second = measurement(), measurement(engine=8)
    with pytest.raises(ValueError, match="same metric"):
        evaluate_immediate(
            pd.DataFrame([first, second]),
            [bundle(first, 1000), bundle(second, 2.0, metric="time")],
            eval_fraction=1,
            seed=0,
        )


def test_real_immediate_training_and_promotion_loads_standard_calibrated_artifact(
    tmp_path, evaluator
):
    pytest.importorskip("lightgbm")
    pytest.importorskip("flatbuffers")
    from uhd_gen.__main__ import main
    from uhd_gen.evaluate import load_model
    from uhd_gen.immediate import read_corpus

    root = tmp_path / "descriptors"
    root.mkdir()
    rows = []
    for index in range(24):
        row = measurement(graph=f"graph-{index}")
        row["features"]["graph.flops"] = float(2e9 * (index + 1))
        row["avgTimeMs"] = (index + 1) / (1 + index / 240)
        rows.append(row)
    (root / "engine.ued.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "id": UED,
                "name": rows[0]["engine_name"],
                "metadata": KMD,
            }
        ),
        encoding="utf-8",
    )
    (root / "metadata.kmd.json").write_text(
        json.dumps({"version": "1.0", "id": KMD}), encoding="utf-8"
    )
    corpus = tmp_path / "immediate.json"
    corpus.write_text(json.dumps(rows), encoding="utf-8")
    model_dir = tmp_path / "trained"
    assert (
        main(
            [
                "train",
                "--role",
                ROLE,
                "--input",
                str(corpus),
                "--features",
                "graph.flops",
                "--num-boost-round",
                "4",
                "--early-stopping",
                "2",
                "--output-dir",
                str(model_dir),
            ]
        )
        == 0
    )
    plan = build_plan(model_dir, root, role=ROLE, arch="gfx942")
    _apply(plan)
    installed = load_model(plan.destination_descriptor.parent)
    predictions = installed.scorer(read_corpus(corpus))
    assert installed.role == ROLE
    assert np.all(
        np.abs(predictions - np.asarray([2 * (1 + index / 240) for index in range(24)]))
        < 0.3
    )
    ued = json.loads((root / "engine.ued.json").read_text(encoding="utf-8"))
    assert ued[ROLE]["gfx942"] == [installed.descriptor["id"]]


def test_time_l1_training_declares_the_metric_its_label_and_direction(
    tmp_path, evaluator
):
    """`--metric time` trains on avgTimeMs directly and declares the direction the metric
    fixes; a label or objective that contradicts the metric is refused before any output.
    """
    pytest.importorskip("lightgbm")
    pytest.importorskip("flatbuffers")
    from uhd_gen.__main__ import main

    rows = []
    for index in range(24):
        row = measurement(graph=f"graph-{index}", metric="time")
        row["features"]["graph.flops"] = float(2e9 * (index + 1))
        row["avgTimeMs"] = 1.0 + index / 10
        rows.append(row)
    corpus = tmp_path / "immediate.json"
    corpus.write_text(json.dumps(rows), encoding="utf-8")
    model_dir = tmp_path / "trained"
    common = [
        "train",
        "--role",
        ROLE,
        "--input",
        str(corpus),
        "--features",
        "graph.flops",
        "--num-boost-round",
        "4",
        "--early-stopping",
        "2",
    ]
    assert (
        main(
            [
                *common,
                "--metric",
                "time",
                "--target",
                "tflops",
                "--output-dir",
                str(model_dir),
            ]
        )
        == 1
    )
    assert (
        main(
            [
                *common,
                "--metric",
                "time",
                "--objective",
                "max",
                "--output-dir",
                str(model_dir),
            ]
        )
        == 1
    )
    assert not model_dir.exists()
    assert main([*common, "--metric", "time", "--output-dir", str(model_dir)]) == 0
    uhd = json.loads((model_dir / "heuristic.uhd.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (model_dir / "train_manifest.json").read_text(encoding="utf-8")
    )
    assert uhd["objective"] == "min"
    assert uhd["score"] == {"metric": "time", "calibrated": True, "transform": "log"}
    assert (manifest["target"], manifest["score_metric"], manifest["timing_statistic"]) == (
        "avgTimeMs", "time", "avgTimeMs")


def test_a_prediction_the_runtime_would_refuse_is_a_decline_not_a_broken_artifact(
    evaluator,
):
    """A prediction its metric cannot take is one the engine reports as INVALID, leaving
    the engine unscored for that graph. The model is fitted on log1p
    and inverted with expm1, so a log-space prediction below zero lands in (-1, 0) -- a few
    rows at the bottom of the range. Failing the artifact discarded a trained AITER model
    over 4 rows in 495 (run 67929709); the rows are now reported and skipped."""
    good, bad = measurement(), measurement(engine=8, graph="other")
    report = evaluate_immediate(
        pd.DataFrame([good, bad]),
        [bundle(good, 1200), bundle(bad, -5)],
        eval_fraction=1,
        seed=0,
    )
    assert report["metrics"]["unscored_rows"]["total"] == 1
    assert report["metrics"]["unscored_rows"]["per_engine"][bad["engine_name"]] == 1
    assert report["metrics"]["calibration"]["rows"] == 1
    assert report["metrics"]["calibration"]["signed_bias_tflops"] == 200


def test_a_model_that_can_score_nothing_is_still_a_failure(evaluator):
    """Reporting every row as a decline is not a model; it is an artifact that would leave
    the engine on static ordering everywhere, which the generator must not publish."""
    row = measurement()
    with pytest.raises(ValueError, match="scored no evaluation row"):
        evaluate_immediate(
            pd.DataFrame([row]), [bundle(row, -1)], eval_fraction=1, seed=0
        )


def test_rows_collected_for_one_metric_cannot_score_a_model_of_another(evaluator):
    """T2: the engine's selector answers in the requested metric, so rows described under
    the tflops selector are not what a `time` model is about. Refused, naming both."""
    row = measurement()
    with pytest.raises(
        ValueError, match="collected for metric 'tflops'.*predicts 'time'"
    ):
        evaluate_immediate(
            pd.DataFrame([row]),
            [bundle(row, 2.0, metric="time")],
            eval_fraction=1,
            seed=0,
        )
    # The selector revision the model was trained against is held to the rows' too.
    model = bundle(row, 1000)
    model.descriptor["trained_against"] = {
        **PROVENANCE,
        "selector_revision": "provider-1/immediate-9/library-3",
    }
    with pytest.raises(ValueError, match="selector revision"):
        evaluate_immediate(pd.DataFrame([row]), [model], eval_fraction=1, seed=0)


def test_a_measurement_must_name_the_metric_it_was_described_under():
    row = measurement()
    row["binding"].pop("metric")
    with pytest.raises(ValueError, match="binding.metric"):
        normalize_row(row)
    # The bench's own record of what it asked for must agree with the description.
    row = measurement()
    row["metric"] = "time"
    with pytest.raises(ValueError, match="collected for metric 'tflops'"):
        normalize_row(row)


def test_a_time_label_needs_no_flop_count_but_a_throughput_label_does():
    """0.9: a provider that publishes no FLOP count (conv backward) still yields `time`
    labels; a corpus mixing graphs with and without a count is not inconsistent."""
    counted, uncounted = measurement(metric="time"), measurement(
        graph="bwd", metric="time"
    )
    uncounted["features"].pop("graph.flops")
    frame = normalize_corpus(pd.DataFrame([counted, uncounted]))
    assert frame.set_index("benchmark").loc["bwd", "avgTimeMs"] == 2.0
    assert pd.isna(frame.set_index("benchmark").loc["bwd", "tflops"])
    throughput = measurement(graph="bwd")
    throughput["features"].pop("graph.flops")
    with pytest.raises(ValueError, match="graph.flops"):
        normalize_row(throughput)


DY = "graph.nodes[0].dy.dims[0]"


def test_a_mixed_operation_corpus_scores_with_a_signature_that_defaults_absent_inputs(
    evaluator,
):
    """T1: a conv-fwd row publishes no `dy`. The runtime evaluates `value_or_default` over
    it, so requiring every referenced name to be published rejected a model the engine
    loads and scores. A bare reference to the same name is what the runtime refuses."""
    forward, backward = measurement(), measurement(graph="bwd")
    backward["features"][DY] = 4
    frame = pd.DataFrame([forward, backward])
    defaulted = [{"value_or_default": [f"${DY}", 0]}]
    report = evaluate_immediate(
        frame, [bundle(forward, 1000, signature=defaulted)], eval_fraction=1, seed=0
    )
    assert report["metrics"]["calibration"]["rows"] == 2
    with pytest.raises(ValueError, match="does not evaluate"):
        evaluate_immediate(
            frame,
            [bundle(forward, 1000, signature=[f"${DY}"])],
            eval_fraction=1,
            seed=0,
        )


def test_a_declined_fastest_engine_is_still_the_oracle_and_its_loss_is_reported(
    evaluator,
):
    """T3: A runs at 1000 TFLOPS but its model declines; B runs at 500 and is scored. The
    runtime picks B, so the problem loses half its throughput. Dropping A before building
    the oracle left B alone and reported nothing to compare."""
    fast, slow = measurement(elapsed=2.0), measurement(engine=8, elapsed=4.0)
    report = evaluate_immediate(
        pd.DataFrame([fast, slow]),
        [bundle(fast, -1), bundle(slow, 500)],
        eval_fraction=1,
        seed=0,
        include_per_problem=True,
    )
    problem = report["per_problem"][0]
    assert (problem["picked_engine"], problem["picked_by"]) == (8, "prediction")
    assert problem["best_immediate_tflops"] == 1000
    assert problem["immediate_selection_regret"] == pytest.approx(0.5)
    assert report["metrics"]["immediate_selection"]["regret"]["mean"] == pytest.approx(
        0.5
    )
    coverage = report["metrics"]["prediction_coverage"]
    assert (
        coverage["rows"],
        coverage["scored_rows"],
        coverage["problems_fully_scored"],
    ) == (2, 1, 0)
    assert (
        report["metrics"]["calibration"]["rows"] == 1
    ), "accuracy is over scored rows only"


def test_a_problem_no_engine_scores_is_picked_by_the_static_rules(evaluator):
    """With nothing scored the runtime falls to `sortEngineIds`, which puts the
    deterministic MIOpen engine last whatever its ID. Its FNV-1a ID is negative, so an
    ID-ordered fallback would pick it first."""
    deterministic = (
        -6748551569128940061
    )  # engineNameToId("MIOPEN_ENGINE_DETERMINISTIC")
    rows = [
        measurement(engine=engine, graph=graph)
        for graph in ("graph", "declined")
        for engine in (7, deterministic)
    ]
    frame = pd.DataFrame(rows)

    def declines_one_graph(prediction):
        return lambda frame: np.where(
            frame["benchmark"].eq("declined"), -1.0, prediction
        )

    report = evaluate_immediate(
        frame,
        [
            bundle(rows[0], declines_one_graph(1000)),
            bundle(rows[1], declines_one_graph(900)),
        ],
        eval_fraction=1,
        seed=0,
        include_per_problem=True,
    )
    picked = {
        tuple(item["key"]): (item["picked_engine"], item["picked_by"])
        for item in report["per_problem"]
    }
    assert picked[("declined", "board")] == (7, "static_order")
    assert picked[("graph", "board")] == (7, "prediction")


def test_a_declined_runtime_prediction_is_unscored_not_an_error(evaluator):
    """An INVALID or UNAVAILABLE answer is the engine declining that graph; the runtime then
    orders it statically, so the evaluator does too instead of refusing the whole run.
    """
    row = measurement()
    response = copy.deepcopy(row)
    response.update(model=UHD, status="INVALID", metric="tflops", value=0)
    scorer = prediction_scorer(descriptor(row), [response])
    assert np.isnan(scorer(normalize_corpus(pd.DataFrame([row])))).all()
    response["status"] = "SOMETHING"
    with pytest.raises(ValueError, match="unknown status"):
        prediction_scorer(descriptor(row), [response])


def test_a_mixed_forward_backward_corpus_trains_and_evaluates_end_to_end(
    tmp_path, evaluator
):
    """0.4 acceptance: forward rows publish `x` and a FLOP count, backward rows publish
    `dy` and no count. A `time` model over `value_or_default` of both trains, and the
    evaluator scores it, without NaN reaching the feature pipe or a name-membership check
    refusing what the runtime evaluates."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("flatbuffers")
    from uhd_gen.__main__ import main

    rows = []
    for index in range(24):
        row = measurement(graph=f"graph-{index}", metric="time")
        if index % 2:
            row["features"].pop("graph.flops")
            row["features"][DY] = index + 1
        else:
            row["features"]["graph.nodes[0].x.dims[0]"] = index + 1
        row["avgTimeMs"] = 1.0 + index / 10
        rows.append(row)
    corpus, recipe = tmp_path / "mixed.json", tmp_path / "features.json"
    corpus.write_text(json.dumps(rows), encoding="utf-8")
    recipe.write_text(
        json.dumps(
            [
                {"value_or_default": [f"${DY}", 0]},
                {"value_or_default": ["$graph.nodes[0].x.dims[0]", 0]},
            ]
        ),
        encoding="utf-8",
    )
    model_dir, report = tmp_path / "model", tmp_path / "report.json"
    assert (
        main(
            [
                "train",
                "--role",
                ROLE,
                "--input",
                str(corpus),
                "--feature-signature",
                str(recipe),
                "--metric",
                "time",
                "--feature-evaluator",
                evaluator,
                "--num-boost-round",
                "4",
                "--early-stopping",
                "2",
                "--output-dir",
                str(model_dir),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "evaluate",
                "--input",
                str(corpus),
                "--model-dir",
                str(model_dir),
                "--eval-fraction",
                "1",
                "--feature-evaluator",
                evaluator,
                "--output",
                str(report),
            ]
        )
        == 0
    )
    scored = json.loads(report.read_text(encoding="utf-8"))["metrics"][
        "prediction_coverage"
    ]
    assert scored["scored_rows"] == scored["rows"] == 24
