# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`train` fits only what the runtime will score the way it was fitted."""
import json

import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("flatbuffers")
pytest.importorskip("pandas")

from uhd_gen.__main__ import main  # noqa: E402

UED = "6d2b90f4-8c15-4a37-9e58-04b7c3fa1d62"
KMD = "3f8a1c07-52d9-4e61-b0a4-9c7d61e2830f"


def _l1_corpus(path, metric="tflops"):
    rows = []
    for index in range(10):
        binding = {
            "engine": "probe:A",
            "role": "predict_engine",
            "metric": metric,
            "arch": "gfx942",
            "selector_revision": "probe-1",
            "trained_against": {"selector_revision": "probe-1"},
        }
        rows.append(
            {
                "engine_id": 1,
                "engine_name": "probe:A",
                "graph_id": f"g{index}",
                "device_id": "board",
                "arch": "gfx942",
                "binding": binding,
                "metric": metric,
                "features": {
                    "graph.flops": 1e12 + index * 1e10,
                    "graph.nodes[0].x.dims[0]": index + 1,
                },
                "avgTimeMs": 1 + index * 0.1,
                "robustMeanMs": 1 + index * 0.1,
                "is_valid": True,
                "selection_mode": "immediate",
                "timing_statistic": "robustMeanMs",
            }
        )
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _l1_train(corpus, output, *extra):
    return main(
        [
            "train",
            "--role",
            "predict_engine",
            "--engine",
            "probe:A",
            "--input",
            str(corpus),
            "--output-dir",
            str(output),
            "--features",
            "graph.flops",
            "graph.nodes[0].x.dims[0]",
            "--num-boost-round",
            "2",
            "--early-stopping",
            "1",
            *extra,
        ]
    )


def test_rows_collected_under_another_metric_cannot_train_a_model(
    tmp_path, evaluator, caplog
):
    """Rows describe what their own metric's selector picked; the refusal names both."""
    corpus = _l1_corpus(tmp_path / "corpus.json", metric="tflops")
    assert (
        _l1_train(
            corpus,
            tmp_path / "model",
            "--metric",
            "time",
            "--feature-evaluator",
            evaluator,
        )
        == 1
    )
    assert "'tflops'" in caplog.text and "'time'" in caplog.text
    assert not (tmp_path / "model").exists()
    assert (
        _l1_train(
            corpus,
            tmp_path / "model",
            "--metric",
            "tflops",
            "--feature-evaluator",
            evaluator,
        )
        == 0
    )


def test_a_grouped_engine_prediction_model_is_refused_before_fitting(tmp_path, caplog):
    """The runtime scores only the root ensemble of an L1 model."""
    corpus = _l1_corpus(tmp_path / "corpus.json")
    assert (
        _l1_train(
            corpus,
            tmp_path / "model",
            "--metric",
            "tflops",
            "--group-by-feature",
            "graph.nodes[0].x.dims[0]",
        )
        == 1
    )
    assert "--group-by-feature is not supported for predict_engine" in caplog.text
    assert not (tmp_path / "model").exists()


@pytest.mark.parametrize("categories", [("1", "2"), ("A", "B")])
def test_grouped_export_keys_each_group_by_the_code_the_feature_row_carries(
    tmp_path, evaluator, categories
):
    """The runtime routes by the string feature's categorical code, not float(raw)."""
    from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModelT

    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "ued": {"id": UED, "revision": "1.0"},
                "kmd": {"id": KMD, "revision": "1.0"},
                "umd": [],
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "benchmark": f"p{problem}",
            "device": "board",
            "kernel": f"{raw}-k{variant}",
            "kernel.solver": raw,
            "kernel.variant": variant,
            "graph.size": problem + 1,
            "tflops": 20 + 60 * code + problem + variant,
        }
        for problem in range(10)
        for code, raw in enumerate(categories)
        for variant in range(2)
    ]
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "model"
    assert (
        main(
            [
                "train",
                "--input",
                str(corpus),
                "--output-dir",
                str(output),
                "--provenance",
                str(provenance),
                "--features",
                "kernel.solver",
                "kernel.variant",
                "graph.size",
                "--group-by-feature",
                "kernel.solver",
                "--metric",
                "tflops",
                "--num-boost-round",
                "2",
                "--early-stopping",
                "1",
                "--feature-evaluator",
                evaluator,
            ]
        )
        == 0
    )

    encoding = json.loads((output / "heuristic.uhd.json").read_text(encoding="utf-8"))[
        "categorical_encoding"
    ]
    model = GbdtModelT.InitFromPackedBuf(
        bytearray((output / "model.bin").read_bytes()), 0
    )
    assert sorted(group.value for group in model.groups) == sorted(
        float(code) for code in encoding["$kernel.solver"].values()
    )
    assert (
        json.loads((output / "train_manifest.json").read_text(encoding="utf-8"))[
            "group_models"
        ]
        == 2
    )


def _manifest(output):
    return json.loads((output / "train_manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("suffix", [".json", ".csv", ".parquet"])
def test_a_catalog_row_checked_numerically_wrong_never_trains(
    tmp_path, evaluator, suffix
):
    """RFC 0019 §13.2: a wrong candidate is never a label; null (unchecked) trains."""
    pd = pytest.importorskip("pandas")
    if suffix == ".parquet":
        pytest.importorskip("pyarrow")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "ued": {"id": UED, "revision": "1.0"},
                "kmd": {"id": KMD, "revision": "1.0"},
                "umd": [],
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "benchmark": f"p{problem}",
            "device": "board",
            "kernel": f"k{variant}",
            "kernel.variant": variant,
            "graph.size": problem + 1,
            "is_valid": True,
            "numerically_valid": None if problem % 2 else True,
            "validation": "no_reference" if problem % 2 else "agrees_with_catalog",
            "tflops": 20.0 + problem + variant,
        }
        for problem in range(10)
        for variant in range(2)
    ]
    rows[0].update(
        numerically_valid=False, validation="output_mismatch: tensor 'Y'", tflops=195.2
    )
    corpus = tmp_path / f"corpus{suffix}"
    if suffix == ".json":
        corpus.write_text(json.dumps(rows), encoding="utf-8")
    elif suffix == ".csv":
        pd.DataFrame(rows).to_csv(corpus, index=False)
    else:
        pd.DataFrame(rows).to_parquet(corpus, index=False)
    output = tmp_path / "model"
    assert (
        main(
            [
                "train",
                "--input",
                str(corpus),
                "--output-dir",
                str(output),
                "--provenance",
                str(provenance),
                "--features",
                "kernel.variant",
                "graph.size",
                "--metric",
                "tflops",
                "--num-boost-round",
                "2",
                "--early-stopping",
                "1",
                "--feature-evaluator",
                evaluator,
            ]
        )
        == 0
    )
    assert _manifest(output)["num_samples"] == len(rows) - 1


def test_an_immediate_pick_checked_numerically_wrong_never_trains(tmp_path, evaluator):
    """The L1 verdict survives import and the row is excluded from the labels."""
    corpus = _l1_corpus(tmp_path / "corpus.json")
    rows = json.loads(corpus.read_text(encoding="utf-8"))
    rows[0].update(numerically_valid=False, validation="output_mismatch: wrong output")
    corpus.write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "model"
    assert (
        _l1_train(
            corpus, output, "--metric", "tflops", "--feature-evaluator", evaluator
        )
        == 0
    )
    assert _manifest(output)["num_samples"] == len(rows) - 1
