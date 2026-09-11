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
    ROLE, evaluate_immediate, normalize_corpus, normalize_row, prediction_scorer,
    validate_signature,
)
from uhd_gen.promote import _apply, build_plan


UHD = "727e5401-3b99-49ff-a2fc-68fd4eedbb54"
UED = "6d2b90f4-8c15-4a37-9e58-04b7c3fa1d62"
KMD = "3f8a1c07-52d9-4e61-b0a4-9c7d61e2830f"


PROVENANCE = {"ued": {"id": UED, "revision": "1.0"},
              "kmd": {"id": KMD, "revision": "1.0"}, "umd": []}


def measurement(*, engine=7, graph="graph", elapsed=2.0, robust=2.5):
    name = f"provider:engine{engine}"
    return {"engine_id": engine, "engine_name": name, "graph_id": graph, "device_id": "board",
            "arch": "gfx942", "binding": {"engine": name, "role": ROLE, "arch": "gfx942",
            "selector_revision": "provider-1/immediate-2/library-3",
            "trained_against": copy.deepcopy(PROVENANCE)},
            "features": {"graph.flops": 2e12, "graph.nodes": 1, "device.cu_count": 120},
            # RFC 0019.13 §11.2 (:2003) pins a calibrated score to `avgTimeMs`, so that
            # is the label. `robustMeanMs` rides along as §8.5's informational statistic
            # and is deliberately a DIFFERENT number here: a label read from the wrong
            # column produces a wrong TFLOPS rather than the same one.
            "avgTimeMs": elapsed, "robustMeanMs": robust,
            "stddevMs": 0.05, "iters": 30, "is_valid": True,
            "selection_mode": "immediate", "timing_statistic": "robustMeanMs"}


def descriptor(row):
    return {"version": "1.0", "id": UHD, "name": "immediate model", "adapter": "tree_data",
            "trained_against": row["binding"]["trained_against"], "objective": "max",
            "score": {"units": "tflops", "calibrated": True, "transform": "log1p"},
            "features_signature": ["$graph.flops"],
            "features_hash": compute_features_hash(["$graph.flops"]),
            "tree_data": {"artifact": "model.bin"}}


def bundle(row, prediction, training_keys=()):
    # The owning engine is recorded by the training manifest's binding, never by the
    # descriptor: RFC 0019 Section 3.1 leaves that binding to the UED role map.
    return SimpleNamespace(descriptor=descriptor(row),
                           manifest={"training_problem_keys": list(training_keys),
                                     "binding": copy.deepcopy(row["binding"]),
                                     "arch": row["arch"]},
                           scorer=lambda frame: np.full(len(frame), prediction))


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


@pytest.mark.parametrize("where", ["envelope", "feature", "expression", "sweep", "label"])
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
    row["binding"]["trained_against"] = {"engine": {"name": row["engine_name"], "version": "v1"}}
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


def test_single_engine_predictions_have_signed_errors_without_fake_ranking_regret():
    row = measurement()
    report = evaluate_immediate(pd.DataFrame([row]), [bundle(row, 1200)], eval_fraction=1, seed=0)
    calibration = report["metrics"]["calibration"]
    assert calibration["signed_bias_tflops"] == 200
    assert calibration["signed_relative_bias"] == pytest.approx(0.2)
    assert report["metrics"]["immediate_selection"]["problems_compared"] == 0
    assert report["metrics"]["immediate_selection"]["regret"]["mean"] is None


def test_selection_regret_compares_other_immediate_engines_not_tuned_candidates():
    fast, slow = measurement(), measurement(engine=8, elapsed=4)
    report = evaluate_immediate(pd.DataFrame([fast, slow]), [bundle(fast, 400), bundle(slow, 600)],
                                eval_fraction=1, seed=0, include_per_problem=True)
    assert report["metrics"]["immediate_selection"]["regret"]["mean"] == pytest.approx(0.5)
    assert report["per_problem"][0]["picked_engine"] == 8
    assert report["metrics"]["per_engine"][fast["engine_name"]]["signed_bias_tflops"] == -600


def test_holdout_is_checked_by_graph_device_keys_not_input_filename():
    row = measurement()
    report = evaluate_immediate(pd.DataFrame([row]), [bundle(row, 1000, [("graph", "board")])],
                                eval_fraction=1, seed=0)
    assert report["holdout_integrity"]["status"] == "COMPROMISED"


def test_runtime_prediction_must_match_measured_request_but_can_use_new_l1_model():
    row = measurement()
    row["binding"]["uhd_id"] = "old-model"
    response = copy.deepcopy(row)
    response.update(model=UHD, status="available", tflops=1234)
    response["binding"]["uhd_id"] = UHD
    model = descriptor(row)
    scorer = prediction_scorer(model, [response])
    assert scorer(normalize_corpus(pd.DataFrame([row]))).tolist() == [1234]
    changed = copy.deepcopy(row)
    changed["features"]["constraint.workspace_limit"] = 0
    with pytest.raises(ValueError):
        scorer(normalize_corpus(pd.DataFrame([changed])))


def test_real_immediate_training_and_promotion_loads_standard_calibrated_artifact(tmp_path):
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
    (root / "engine.ued.json").write_text(json.dumps({"version": "1.0", "id": UED,
        "name": rows[0]["engine_name"], "metadata": KMD}), encoding="utf-8")
    (root / "metadata.kmd.json").write_text(json.dumps({"version": "1.0", "id": KMD}), encoding="utf-8")
    corpus = tmp_path / "immediate.json"
    corpus.write_text(json.dumps(rows), encoding="utf-8")
    model_dir = tmp_path / "trained"
    assert main(["train", "--role", ROLE, "--input", str(corpus), "--features", "graph.flops",
                 "--num-boost-round", "4", "--early-stopping", "2", "--output-dir", str(model_dir)]) == 0
    plan = build_plan(model_dir, root, role=ROLE, arch="gfx942")
    _apply(plan)
    installed = load_model(plan.destination_descriptor.parent)
    predictions = installed.scorer(read_corpus(corpus))
    assert installed.role == ROLE
    assert np.all(np.abs(predictions - np.asarray([2 * (1 + index / 240) for index in range(24)])) < 0.3)
    ued = json.loads((root / "engine.ued.json").read_text(encoding="utf-8"))
    assert ued[ROLE]["gfx942"] == installed.descriptor["id"]
