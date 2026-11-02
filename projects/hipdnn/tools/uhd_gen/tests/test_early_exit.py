# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Early-exit schedules: the runtime's checks, the staged ranking offline evaluation
reproduces, writing and verifying the schedule, and the trainer's choice of one."""
from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
lgb = pytest.importorskip("lightgbm")
pytest.importorskip("flatbuffers")

import uhd_gen  # noqa: E402,F401  puts _generated/ on sys.path
from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModelT  # noqa: E402
from uhd_gen.artifact import verify_tree_artifact  # noqa: E402
from uhd_gen.early_exit import (  # noqa: E402
    check_schedule,
    choose_schedule,
    out_of_fold_catalogs,
    parse_schedule,
    staged_pick,
    staged_scores,
    work_fraction,
)
from uhd_gen.evaluate import (
    StagedPredictions,
    _flatbuffer_scorer,
    evaluate_corpus,
)  # noqa: E402
from uhd_gen.lgbm_to_flatbuffer import build_gbdt_model  # noqa: E402
from uhd_gen.train_uhd import train_model  # noqa: E402

from uhd_gen.tests.model_bin import write_model  # noqa: E402


@pytest.mark.parametrize(
    "schedule, reason",
    [
        ([(0, 2)], "trees must be at least 1 and fewer than the model's trees"),
        ([(10, 2)], "trees must be at least 1 and fewer than the model's trees"),
        ([(2, 0)], "keep must be at least 1"),
        ([(4, 4), (3, 2)], "stages must increase in trees and decrease in keep"),
        ([(2, 4), (4, 4)], "stages must increase in trees and decrease in keep"),
    ],
)
def test_malformed_schedules_are_refused_with_the_runtimes_reason(schedule, reason):
    with pytest.raises(ValueError, match=reason):
        check_schedule(schedule, 10)


def test_schedules_parse_from_the_command_line_form():
    assert parse_schedule("30:16") == [(30, 16)]
    assert parse_schedule("10:32, 40:8") == [(10, 32), (40, 8)]
    with pytest.raises(ValueError, match="TREES:KEEP"):
        parse_schedule("30")


def _tree_outputs(seed: int, rows: int, trees: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(rows, trees))


def _adder(outputs: np.ndarray):
    def add(acc, rows, begin, end):
        for t in range(begin, end):
            acc[rows] += outputs[rows, t]

    return add


def test_staged_scores_finish_survivors_and_keep_pruned_partials():
    outputs = _tree_outputs(1, 30, 12)
    schedule = [(3, 10), (7, 4)]
    staged = staged_scores(
        _adder(outputs), 30, 12, schedule, 0.5, lower_is_better=False
    )
    assert (staged.stage == 2).sum() == 4
    for row in range(30):
        trees = 12 if staged.stage[row] == 2 else schedule[staged.stage[row]][0]
        total = 0.0
        for t in range(trees):
            total += outputs[row, t]
        expected = 0.5 + total if staged.stage[row] == 2 else total
        assert staged.raw[row] == expected
    # Every row that went past a stage beat, on that prefix, every row pruned there.
    for index, (trees, _) in enumerate(schedule):
        prefix = outputs[:, :trees].cumsum(axis=1)[:, -1]
        kept = staged.stage > index
        pruned = staged.stage == index
        assert prefix[kept].min() >= prefix[pruned].max()


def test_ties_keep_the_earlier_row():
    outputs = np.ones((5, 4))
    staged = staged_scores(_adder(outputs), 5, 4, [(1, 2)], 0.0, lower_is_better=True)
    assert staged.stage.tolist() == [1, 1, 0, 0, 0]


def test_the_pick_with_a_schedule_is_the_best_survivor():
    full = np.array([1.0, 9.0, 5.0, 8.0])
    prefix = {1: np.array([4.0, 0.0, 3.0, 2.0])}
    # Max: after one tree rows 0 and 2 survive, so the true best (row 1) is lost.
    assert staged_pick(prefix, full, [(1, 2)], lower_is_better=False) == 2
    assert staged_pick(prefix, full, [], lower_is_better=False) == 1


def test_work_counts_tree_walks_per_catalog():
    # 10 rows x 10 trees: 10x2 walks, then 3 rows x 8 trees.
    assert work_fraction([10], [(2, 3)], 10) == pytest.approx((20 + 24) / 100)
    assert work_fraction([2], [(2, 3)], 10) == pytest.approx(1.0)


def _catalogs(misleading: bool, count: int = 40):
    """Catalogs of 20 where the first 10 trees rank as the whole model does, or misrank
    the eventual winner last."""
    rng = np.random.default_rng(7)
    catalogs = []
    for _ in range(count):
        measured = rng.uniform(1.0, 2.0, 20)
        full = measured.copy()  # a perfect model under max
        prefix = {p: full * p / 100 for p in (5, 10, 20, 30, 40, 60, 100)}
        if misleading:
            winner = int(np.argmax(full))
            for values in prefix.values():
                values[winner] = -1.0
        catalogs.append((prefix, full, measured))
    return catalogs


def test_a_schedule_that_costs_regret_is_not_chosen():
    choice = choose_schedule(_catalogs(misleading=True), 200, "max")
    assert choice.schedule == []
    assert choice.full["mean_regret"] == 0.0


def test_the_cheapest_schedule_within_tolerance_is_chosen():
    choice = choose_schedule(_catalogs(misleading=False), 200, "max")
    assert choice.schedule == [(5, 8)]
    manifest = choice.to_manifest()
    assert manifest["schedule"] == [[5, 8]]
    assert manifest["candidates"]


def _ranking_frame(rows: int = 60, problems: int = 6, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "benchmark": np.repeat(
                [f"p{i}" for i in range(problems)], rows // problems
            ),
            "device": "d",
            "kernel": [f"k{i}" for i in range(rows)],
            "pack": "p",
            "dispatch": "d",
            "q.size": np.repeat(rng.uniform(1, 4, problems), rows // problems),
            "kernel.tile": rng.integers(1, 9, rows).astype(float),
            "is_valid": "True",
        }
    )
    frame["minTimeMs"] = frame["q.size"] / frame["kernel.tile"] + rng.uniform(
        0, 0.1, rows
    )
    return frame


def _model_with_schedule(
    tmp_path: Path, schedule
) -> tuple[Path, lgb.Booster, np.ndarray]:
    frame = _ranking_frame()
    X = frame[["q.size", "kernel.tile"]].to_numpy()
    booster = lgb.train(
        {
            "objective": "regression",
            "num_leaves": 7,
            "verbose": -1,
            "min_data_in_leaf": 2,
        },
        lgb.Dataset(X, label=np.log(frame["minTimeMs"].to_numpy())),
        num_boost_round=20,
    )
    model = GbdtModelT.InitFromPackedBuf(
        bytearray(
            build_gbdt_model(booster.dump_model(), "sha256:test", early_exit=schedule)
        ),
        0,
    )
    return write_model(model, tmp_path / "model.bin"), booster, X


def test_the_converter_writes_a_schedule_the_verifier_accepts(tmp_path):
    path, _, _ = _model_with_schedule(tmp_path, [(5, 4), (10, 2)])
    model = GbdtModelT.InitFromPackedBuf(bytearray(verify_tree_artifact(path, None)), 0)
    assert [(s.trees, s.keep) for s in model.earlyExit] == [(5, 4), (10, 2)]


def test_the_converter_refuses_a_schedule_the_runtime_would_refuse():
    booster = lgb.train(
        {"objective": "regression", "verbose": -1},
        lgb.Dataset(np.random.default_rng(0).random((50, 2)), label=np.arange(50.0)),
        num_boost_round=3,
    )
    with pytest.raises(ValueError, match="fewer than the model's trees"):
        build_gbdt_model(booster.dump_model(), "sha256:test", early_exit=[(3, 2)])


def test_the_verifier_refuses_a_malformed_stored_schedule(tmp_path):
    path, _, _ = _model_with_schedule(tmp_path, [(5, 4)])
    model = GbdtModelT.InitFromPackedBuf(bytearray(path.read_bytes()), 0)
    model.earlyExit[0].keep = 0
    write_model(model, path)
    with pytest.raises(ValueError, match="keep must be at least 1"):
        verify_tree_artifact(path, None)


def test_offline_scoring_applies_the_schedule_as_the_runtime_does(tmp_path):
    path, booster, X = _model_with_schedule(tmp_path, [(4, 3)])
    scorer = _flatbuffer_scorer(
        path, ["q.size", "kernel.tile"], objective="min", physical=False
    )
    frame = _ranking_frame().iloc[:10]
    scored = scorer(frame)
    assert isinstance(scored, StagedPredictions)
    survivors = scored.stage == scored.stages
    assert survivors.sum() == 3
    assert np.isneginf(scored.predictions[~survivors]).all()
    raw = booster.predict(X[:10], raw_score=True)
    prefix = booster.predict(X[:10], num_iteration=4, raw_score=True)
    # Survivors are the three lowest prefix scores (objective min), scored in full.
    assert set(np.flatnonzero(survivors)) == set(np.argsort(prefix, kind="stable")[:3])
    np.testing.assert_allclose(
        np.log(scored.predictions[survivors]), raw[survivors], rtol=1e-6
    )


def test_evaluation_ranks_pruned_rows_after_usable_survivors_and_before_unusable_ones():
    frame = pd.DataFrame(
        {
            "benchmark": "g",
            "device": "d",
            "kernel": ["a", "b", "c", "d"],
            "pack": "p",
            "dispatch": "d",
            "minTimeMs": [4.0, 1.0, 2.0, 3.0],
            "is_valid": "True",
        }
    )

    def staged(predictions, stage, partial):
        return lambda _: StagedPredictions(
            predictions=np.asarray(predictions, dtype=float),
            stage=np.asarray(stage),
            partial=np.asarray(partial, dtype=float),
            stages=2,
        )

    def pick(scorer) -> float:
        result = evaluate_corpus(
            frame, scorer, target="minTimeMs", objective="min", eval_fraction=1.0
        )
        return result.problems[0].picked_value

    # Row a survived and is usable: it is picked although pruned b is measured best.
    assert (
        pick(staged([4.0, -np.inf, -np.inf, -np.inf], [2, 1, 0, 0], [0, 1, 2, 3]))
        == 4.0
    )
    # Survivor a is unusable (-inf): the latest-stage pruned row (b) comes next.
    assert pick(staged([-np.inf] * 4, [2, 1, 0, 0], [0, 9.0, 0.0, 0.0])) == 1.0
    # Same stage: the lower partial sum wins under min.
    assert pick(staged([-np.inf] * 4, [2, 0, 0, 0], [0, 5.0, 3.0, 4.0])) == 2.0


def test_out_of_fold_catalogs_feed_the_choice():
    frame = _ranking_frame(rows=200, problems=20)
    record: list = []
    train_model(
        frame,
        ["q.size", "kernel.tile"],
        "minTimeMs",
        group_cols=["benchmark"],
        params={
            "objective": "regression",
            "num_leaves": 7,
            "verbose": -1,
            "min_data_in_leaf": 2,
        },
        num_boost_round=30,
        early_stopping_rounds=5,
        n_splits=4,
        cross_validation=record,
    )
    catalogs = out_of_fold_catalogs(
        record[0], frame["benchmark"].to_numpy(), frame["minTimeMs"].to_numpy()
    )
    assert len(catalogs) == 20
    choice = choose_schedule(catalogs, record[0]["trees"], "min")
    assert choice.full["mean_regret"] >= 0.0
    assert all(stage[0] < record[0]["trees"] for stage in choice.schedule)


_PROVENANCE = {
    "ued": {"id": "13ab344f-4818-4772-bb8e-8e1441fec82c", "revision": "1.0"},
    "kmd": {"id": "46d64d06-18eb-483d-9bb4-94472d32b78d", "revision": "1.0"},
    "umd": [],
}


def _train(tmp_path: Path, *extra: str) -> tuple[int, Path]:
    import json

    from uhd_gen.__main__ import main

    corpus = tmp_path / "corpus.csv"
    _ranking_frame(rows=400, problems=40).to_csv(corpus, index=False)
    snapshot = tmp_path / "provenance.json"
    snapshot.write_text(json.dumps(_PROVENANCE), encoding="utf-8")
    output = tmp_path / "model"
    code = main(
        [
            "train",
            "--input",
            str(corpus),
            "--provenance",
            str(snapshot),
            "--features",
            "q.size",
            "kernel.tile",
            "--target",
            "minTimeMs",
            "--objective",
            "min",
            "--output-dir",
            str(output),
            "--num-boost-round",
            "40",
            "--early-stopping",
            "10",
            *extra,
        ]
    )
    return code, output


def _stored_schedule(output: Path) -> list[tuple[int, int]]:
    model = GbdtModelT.InitFromPackedBuf(
        bytearray(verify_tree_artifact(output / "model.bin", None)), 0
    )
    return [(s.trees, s.keep) for s in model.earlyExit or []]


def _manifest(output: Path) -> dict:
    import json

    return json.loads((output / "train_manifest.json").read_text(encoding="utf-8"))


def test_train_writes_a_requested_schedule(tmp_path, evaluator):
    code, output = _train(tmp_path, "--early-exit", "3:4")
    assert code == 0
    assert _stored_schedule(output) == [(3, 4)]
    assert _manifest(output)["early_exit"] == {"schedule": [[3, 4]], "chosen": False}


def test_train_chooses_a_schedule_and_records_the_candidates(tmp_path, evaluator):
    code, output = _train(tmp_path)
    assert code == 0
    record = _manifest(output)["early_exit"]
    assert record["chosen"] is True
    assert record["candidates"]
    assert _stored_schedule(output) == [tuple(stage) for stage in record["schedule"]]


def test_train_writes_no_schedule_when_asked_for_none(tmp_path, evaluator):
    code, output = _train(tmp_path, "--early-exit", "none")
    assert code == 0
    assert _stored_schedule(output) == []
