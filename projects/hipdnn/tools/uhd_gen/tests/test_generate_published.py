# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""generate from a published dataset: §8.3 rows from any producer, trained as `train` trains them.

A published dataset (a sweep run elsewhere, the MIOpen perf DB's translation) carries no engine
binding and no UED -- an opaque catalog. generate trains it with the label and features stated,
holds problems out as it does for every dataset, and refuses to promote what has nowhere to go.
"""

from __future__ import annotations

import json

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("lightgbm")
pytest.importorskip("flatbuffers")

PROBLEMS = 40


@pytest.fixture
def published(tmp_path):
    """A MIOpen-shaped catalog: per problem, two solvers whose configuration strings differ in
    shape, every candidate timed. Solver 2 wins large problems, solver 1 small ones."""
    from uhd_gen.dataset import store

    rows = []
    for p in range(PROBLEMS):
        m = 64 * (p + 1)
        for solver, descriptor, scale in (
            (1, "8,2", 1.0 + p / 20),
            (2, "4,4,1", 2.5 - p / 40),
        ):
            for variant in range(2):
                time = scale * (1.0 + 0.1 * variant)
                rows.append(
                    {
                        "benchmark": f"conv_problem_{p}",
                        "device": "board",
                        "q.M": m,
                        "q.flops": 2.0 * m**3,
                        "q.bytes": 3.0 * m * m * 4,
                        "kernel.solver_id": solver,
                        "kernel.descriptor": descriptor + f",{variant}",
                        "device.cu_count": 80,
                        "minTimeMs": time * 0.95,
                        "avgTimeMs": time,
                        "stddevMs": 0.01,
                        "iters": 10,
                    }
                )
    csv = tmp_path / "perfdb.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    dataset = tmp_path / "ds"
    store.add(
        dataset,
        *store.from_csv(
            [csv],
            engine="MIOPEN",
            engine_id=11,
            role="sort_kernel_catalog",
            arch="gfx942",
            selector_revision="miopen/abc",
            collected_at="2026-10-02T00:00:00+00:00",
            contribution_id="perfdb-1",
            expand_descriptor=["kernel.descriptor"],
            scope_by="kernel.solver_id",
        ),
    )
    return {"dataset": dataset, "root": tmp_path, "tree": tmp_path / "empty_tree"}


def _generate(published, evaluator, name, *extra):
    from uhd_gen.__main__ import main

    published["tree"].mkdir(exist_ok=True)
    output = published["root"] / name
    code = main(
        [
            "generate",
            "--dataset",
            str(published["dataset"]),
            "--descriptor-tree",
            str(published["tree"]),
            "--role",
            "sort_kernel_catalog",
            "--engine",
            "MIOPEN",
            "--arch",
            "gfx942",
            "--no-promote",
            "--num-boost-round",
            "20",
            "--early-stopping",
            "5",
            "--eval-fraction",
            "0.25",
            "--feature-evaluator",
            evaluator,
            "--output-dir",
            str(output),
            *extra,
        ]
    )
    return code, output


FEATURES = [
    "--features",
    "q.M",
    "kernel.solver_id",
    "kernel.descriptor.s1_f0",
    "kernel.descriptor.s2_f2",
]


def test_a_published_dataset_trains_a_grouped_model_scored_on_held_out_problems(
    published, evaluator
):
    code, output = _generate(
        published,
        evaluator,
        "grouped",
        *FEATURES,
        "--target",
        "tflops",
        "--objective",
        "max",
        "--group-by-feature",
        "kernel.solver_id",
        "--descriptor-name",
        "miopen_conv",
        "--model-version",
        "0.1",
    )
    assert code == 0
    manifest = json.loads((output / "generation_manifest.json").read_text())
    assert manifest["published_dataset"] is True and manifest["evaluation"] == "holdout"
    assert manifest["training_options"] == [
        {"expand_descriptor": ["kernel.descriptor"], "scope_by": "kernel.solver_id"}
    ]
    model = manifest["models"][0]
    trained = {tuple(k) for k in model["training_problem_keys"]}
    scored = {tuple(k) for k in model["eval_problem_keys"]}
    assert trained and scored and not trained & scored
    assert len(trained) + len(scored) == PROBLEMS
    arguments = model["training_arguments"]
    assert arguments[arguments.index("--group-by-feature") + 1] == "kernel.solver_id"
    assert (output / "model" / "miopen_conv.uhd.json").is_file()
    report = json.loads((output / "model" / "eval_report.json").read_text())
    assert report["holdout_integrity"]["status"] == "held_out"


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--target", "tflops", "--objective", "max"], "names no feature recipe"),
        (FEATURES, "pass --target and --objective"),
        ([*FEATURES, "--target", "nope", "--objective", "max"], "no 'nope' column"),
    ],
)
def test_what_a_published_dataset_cannot_say_must_be_stated(
    published, evaluator, caplog, extra, message
):
    code, _ = _generate(published, evaluator, "refused", *extra)
    assert code == 1
    assert message in caplog.text


def test_a_published_dataset_has_no_ued_to_promote_into(published, evaluator, caplog):
    from uhd_gen.__main__ import main

    published["tree"].mkdir(exist_ok=True)
    code = main(
        [
            "generate",
            "--dataset",
            str(published["dataset"]),
            "--descriptor-tree",
            str(published["tree"]),
            "--role",
            "sort_kernel_catalog",
            *FEATURES,
            "--target",
            "tflops",
            "--objective",
            "max",
            "--feature-evaluator",
            evaluator,
            "--output-dir",
            str(published["root"] / "promoted"),
        ]
    )
    assert code == 1
    assert "no UED to install into" in caplog.text


def test_trained_on_every_row_it_is_the_model_train_makes_from_the_same_rows(
    published, evaluator
):
    """What uhd-train did, generate does: with nothing held out (--recall), the model is the
    one the old path made -- `publish` to a Parquet file, then `train` on it with the same
    options -- byte for byte."""
    from uhd_gen.__main__ import main
    from uhd_gen.dataset.publish import publish_frame, write_parquet

    code, output = _generate(
        published,
        evaluator,
        "recall",
        *FEATURES,
        "--target",
        "tflops",
        "--objective",
        "max",
        "--group-by-feature",
        "kernel.solver_id",
        "--recall",
    )
    assert code == 0
    direct = published["root"] / "direct"
    arguments = json.loads((output / "generation_manifest.json").read_text())["models"][
        0
    ]["training_arguments"]
    # The command generate ran, on the old path's input: the CSV published straight to Parquet.
    parquet = published["root"] / "published.parquet"
    write_parquet(
        publish_frame(
            [published["root"] / "perfdb.csv"],
            expand_descriptor=["kernel.descriptor"],
            scope_by="kernel.solver_id",
        ),
        parquet,
    )
    arguments = [str(parquet) if a.endswith("train.json") else a for a in arguments]
    arguments[arguments.index("--output-dir") + 1] = str(direct)
    arguments[arguments.index("--provenance") + 1] = str(output / "provenance.json")
    assert main(arguments) == 0
    assert (direct / "model.bin").read_bytes() == (
        output / "model" / "model.bin"
    ).read_bytes()
    rows = json.loads((output / "train.json").read_text())
    assert len(rows) == PROBLEMS * 4, "recall trains on every row"
