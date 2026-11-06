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
from hipdnn_flatbuffers_sdk.data_objects.EarlyExitStage import (  # noqa: E402
    EarlyExitStageT,
)
from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModelT  # noqa: E402
from uhd_gen.artifact import verify_tree_artifact  # noqa: E402
from uhd_gen.early_exit import (  # noqa: E402
    PREFIXES,
    candidate_schedules,
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


def _stages(count: int) -> list[tuple[int, int]]:
    """`count` valid stages: trees 1..count, keep count..1."""
    return [(index, count + 1 - index) for index in range(1, count + 1)]


def test_a_schedule_may_have_255_stages_and_no_more():
    # The runtime reports a row's stage in a uint8, so 256 stages would wrap.
    check_schedule(_stages(255), 300)
    with pytest.raises(ValueError, match="more than 255 stages"):
        check_schedule(_stages(256), 300)


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


def _candidate(choice, schedule):
    (found,) = [
        record
        for record in choice.candidates
        if record["schedule"] == [list(stage) for stage in schedule]
    ]
    return found


def _lossy_catalogs(regret: float, lossy: int, count: int = 40, size: int = 20):
    """Catalogs under max that full scoring ranks perfectly. In `lossy` of them every prefix
    ranks the winner last, so any schedule keeping fewer than `size` rows picks a candidate
    `regret` short of it."""
    rng = np.random.default_rng(11)
    catalogs = []
    for index in range(count):
        measured = rng.permutation(
            np.concatenate(([1.0, 1.0 - regret], rng.uniform(0.2, 0.9, size - 2)))
        )
        prefix = measured.copy()
        if index < lossy:
            prefix[np.argmax(measured)] = -1.0
        catalogs.append(({p: prefix for p in PREFIXES}, measured.copy(), measured))
    return catalogs


# Of 200 trees every schedule keeping 16 or 32 of 20 rows walks more than 0.8 of the
# trees, so a schedule is either (5, 8) -- the cheapest -- or none.
@pytest.mark.parametrize("regret, schedule", [(0.00019, [(5, 8)]), (0.00021, [])])
def test_a_schedule_is_chosen_only_within_the_mean_regret_tolerance(regret, schedule):
    choice = choose_schedule(_lossy_catalogs(regret, lossy=40), 200, "max")
    assert choice.full["mean_regret"] == 0.0
    assert _candidate(choice, [(5, 8)])["mean_regret"] == pytest.approx(regret)
    assert choice.schedule == schedule


@pytest.mark.parametrize("regret, schedule", [(0.0019, [(5, 8)]), (0.0021, [])])
def test_a_schedule_is_chosen_only_within_the_p95_regret_tolerance(regret, schedule):
    # Three catalogs of 40 lose `regret`: the mean stays inside its own tolerance and the
    # 95th percentile is `regret`.
    choice = choose_schedule(_lossy_catalogs(regret, lossy=3), 200, "max")
    record = _candidate(choice, [(5, 8)])
    assert record["mean_regret"] < 0.0002
    assert record["p95_regret"] == pytest.approx(regret)
    assert choice.schedule == schedule


def _perfect_catalogs(size: int, count: int = 20):
    rng = np.random.default_rng(5)
    catalogs = []
    for _ in range(count):
        measured = rng.uniform(1.0, 2.0, size)
        catalogs.append(({5: measured}, measured.copy(), measured))
    return catalogs


# A model of 10 trees has one prefix, 5: (5, 8) walks 5 trees on every row and 5 more on 8
# of them, and the other two schedules keep every row of these catalogs.
@pytest.mark.parametrize(
    "size, work, schedule",
    [(14, 110 / 140, [(5, 8)]), (13, 105 / 130, [])],
)
def test_a_schedule_is_chosen_only_when_it_saves_enough_work(size, work, schedule):
    choice = choose_schedule(_perfect_catalogs(size), 10, "max")
    assert _candidate(choice, [(5, 8)])["work_fraction"] == pytest.approx(work)
    assert choice.schedule == schedule


def _min_catalogs(prune_the_picks: bool, count: int = 40, size: int = 20):
    """Catalogs of run times (min): the best 1.0, then 1.25, then 2.0, 2.1, .... Full scoring
    swaps the best two, so it picks the 1.25 and ends 0.25 over the best. Prefixes rank as the
    full model does, or (`prune_the_picks`) put those two rows last."""
    rng = np.random.default_rng(13)
    catalogs = []
    for _ in range(count):
        measured = np.concatenate(([1.0, 1.25], 2.0 + 0.1 * np.arange(size - 2)))
        order = rng.permutation(size)
        measured = measured[order]
        full = measured.copy()
        full[measured == 1.0], full[measured == 1.25] = 1.25, 1.0
        prefix = full.copy()
        if prune_the_picks:
            prefix[measured <= 1.25] = 99.0
        catalogs.append(({p: prefix for p in PREFIXES}, full, measured))
    return catalogs


def test_a_min_objective_catalog_is_scored_in_the_direction_of_lower_is_better():
    choice = choose_schedule(_min_catalogs(prune_the_picks=False), 200, "min")
    assert choice.full == pytest.approx(
        {"mean_regret": 0.25, "p95_regret": 0.25, "work_fraction": 1.0}
    )
    assert choice.schedule == [(5, 8)]
    assert _candidate(choice, [(5, 8)])["mean_regret"] == pytest.approx(0.25)


def test_a_min_schedule_that_prunes_the_pick_costs_the_regret_of_the_next_candidate():
    choice = choose_schedule(_min_catalogs(prune_the_picks=True), 200, "min")
    assert choice.full["mean_regret"] == pytest.approx(0.25)
    # The survivors' best is the 2.0 against a best of 1.0.
    assert _candidate(choice, [(5, 8)])["mean_regret"] == pytest.approx(1.0)
    assert choice.schedule == []


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


def _scores(path: Path, **options):
    scorer = _flatbuffer_scorer(
        path, ["q.size", "kernel.tile"], objective="min", physical=False, **options
    )
    return scorer(_ranking_frame().iloc[:10])


def test_a_schedule_is_ignored_when_the_model_scores_one_row(tmp_path):
    (tmp_path / "scheduled").mkdir()
    (tmp_path / "plain").mkdir()
    scheduled, _, _ = _model_with_schedule(tmp_path / "scheduled", [(4, 3)])
    plain, _, _ = _model_with_schedule(tmp_path / "plain", [])
    # EnginePredictor scores a row with every tree, whatever the artifact stores.
    assert isinstance(_scores(scheduled), StagedPredictions)
    scores = _scores(scheduled, engine_prediction=True)
    assert not isinstance(scores, StagedPredictions)
    np.testing.assert_array_equal(scores, _scores(plain, engine_prediction=True))


def _grouped_model_with_schedule(directory: Path, schedule) -> Path:
    """A two-layer model (groups by kernel.tile) that also stores `schedule`, which the
    converter would refuse to write."""
    directory.mkdir()
    frame = _ranking_frame(rows=120, problems=6)
    params = {
        "objective": "regression",
        "num_leaves": 7,
        "verbose": -1,
        "min_data_in_leaf": 2,
    }

    def booster(rows) -> lgb.Booster:
        return lgb.train(
            params,
            lgb.Dataset(
                rows[["q.size", "kernel.tile"]].to_numpy(),
                label=np.log(rows["minTimeMs"].to_numpy()),
            ),
            num_boost_round=12,
        )

    groups = [
        (tile, booster(frame[frame["kernel.tile"] == tile]).dump_model())
        for tile in (2.0, 6.0)
    ]
    model = GbdtModelT.InitFromPackedBuf(
        bytearray(
            build_gbdt_model(
                booster(frame).dump_model(),
                "sha256:test",
                group_by_feature_index=1,
                groups=groups,
            )
        ),
        0,
    )
    model.earlyExit = []
    for trees, keep in schedule:
        stage = EarlyExitStageT()
        stage.trees, stage.keep = trees, keep
        model.earlyExit.append(stage)
    return write_model(model, directory / "model.bin")


def test_a_schedule_is_ignored_when_the_model_is_grouped(tmp_path):
    scheduled = _grouped_model_with_schedule(tmp_path / "scheduled", [(4, 3)])
    plain = _grouped_model_with_schedule(tmp_path / "plain", [])
    frame = pd.DataFrame(
        [
            {"q.size": size, "kernel.tile": tile}
            for size in (1.5, 3.0)
            for tile in (2.0, 6.0, 2.0, 6.0)
        ]
    )

    def scores(path: Path):
        return _flatbuffer_scorer(
            path, ["q.size", "kernel.tile"], objective="min", physical=False
        )(frame)

    result = scores(scheduled)
    assert not isinstance(result, StagedPredictions)
    # The group decision is per catalog: this frame's rows outside it score -inf.
    assert np.isneginf(result).any() and np.isfinite(result).any()
    np.testing.assert_array_equal(result, scores(plain))


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


def test_evaluation_breaks_a_tie_between_pruned_rows_toward_the_earlier_row_under_max():
    frame = pd.DataFrame(
        {
            "benchmark": "g",
            "device": "d",
            "kernel": ["a", "b", "c", "d"],
            "pack": "p",
            "dispatch": "d",
            "tflops": [4.0, 1.0, 2.0, 3.0],
            "is_valid": "True",
        }
    )

    def pick(partial) -> float:
        scorer = lambda _: StagedPredictions(  # noqa: E731
            predictions=np.full(4, -np.inf),
            stage=np.zeros(4, dtype=int),
            partial=np.asarray(partial, dtype=float),
            stages=1,
        )
        result = evaluate_corpus(
            frame, scorer, target="tflops", objective="max", eval_fraction=1.0
        )
        return result.problems[0].picked_value

    # Nothing survived: the largest partial sum leads under max, and of the two rows tied
    # for it (b, d) the earlier one in the corpus is picked.
    assert pick([1.0, 5.0, 3.0, 5.0]) == 1.0
    assert pick([1.0, 5.0, 3.0, 6.0]) == 3.0
    assert pick([7.0, 5.0, 3.0, 6.0]) == 4.0


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
    trees = record[0]["trees"]
    choice = choose_schedule(catalogs, trees, "min")
    # Full scoring's top-1 regret under min, from the catalogs alone.
    regrets = [
        float(measured[np.argmin(full)] / measured.min() - 1.0)
        for _, full, measured in catalogs
    ]
    assert choice.full["mean_regret"] > 0.0
    assert choice.full["mean_regret"] == pytest.approx(np.mean(regrets))
    assert choice.full["p95_regret"] == pytest.approx(np.quantile(regrets, 0.95))
    assert len(choice.candidates) == len(candidate_schedules(trees))
    for stage in choice.schedule:
        assert stage[0] < trees


_PROVENANCE = {
    "ued": {"id": "13ab344f-4818-4772-bb8e-8e1441fec82c", "revision": "1.0"},
    "kmd": {"id": "46d64d06-18eb-483d-9bb4-94472d32b78d", "revision": "1.0"},
    "umd": [],
}


def _train(
    tmp_path: Path, *extra: str, rows: int = 1200, problems: int = 40
) -> tuple[int, Path]:
    import json

    from uhd_gen.__main__ import main

    corpus = tmp_path / "corpus.csv"
    frame = _ranking_frame(rows=rows, problems=problems)
    frame["avgTimeMs"] = frame["minTimeMs"]
    frame.to_csv(corpus, index=False)
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
    # 30 candidates per problem: enough that a prefix of the trees saves work.
    code, output = _train(tmp_path)
    assert code == 0
    record = _manifest(output)["early_exit"]
    assert record["chosen"] is True
    assert record["candidates"]
    assert record["schedule"]
    assert _stored_schedule(output) == [tuple(stage) for stage in record["schedule"]]


def test_train_ranks_the_held_out_catalogs_in_the_corpus_direction(
    tmp_path, evaluator, monkeypatch
):
    import uhd_gen.early_exit as early_exit

    objectives = []
    choose = early_exit.choose_schedule

    def spy(catalogs, tree_count, objective):
        objectives.append(objective)
        return choose(catalogs, tree_count, objective)

    monkeypatch.setattr(early_exit, "choose_schedule", spy)
    code, _ = _train(tmp_path)  # `--objective min`
    assert code == 0
    assert objectives == ["min"]


def test_train_writes_no_schedule_when_asked_for_none(tmp_path, evaluator):
    # The corpus on which auto writes a schedule.
    code, output = _train(tmp_path, "--early-exit", "none")
    assert code == 0
    assert _stored_schedule(output) == []
    assert _manifest(output)["early_exit"] == {"schedule": [], "chosen": False}


def test_train_chooses_no_schedule_for_a_calibrated_model(tmp_path, evaluator):
    # A calibrated ranking reads every candidate's estimate, which a pruned one lacks.
    code, output = _train(
        tmp_path,
        "--metric",
        "time",
        "--calibrated",
        "--timing-statistic",
        "avgTimeMs",
        "--target",
        "avgTimeMs",
    )
    assert code == 0
    assert _stored_schedule(output) == []
    assert _manifest(output)["early_exit"] == {"schedule": [], "chosen": False}


@pytest.mark.parametrize(
    "unsupported, reason",
    [
        ({"calibrated": True}, "--calibrated"),
        ({"role": "predict_engine"}, "single-layer sort_kernel_catalog"),
        ({"group_by_feature": "size"}, "single-layer sort_kernel_catalog"),
    ],
)
def test_an_explicit_schedule_the_runtime_would_not_apply_is_refused(
    unsupported, reason
):
    import argparse

    from uhd_gen.__main__ import _early_exit_request

    args = argparse.Namespace(
        early_exit="30:16",
        role="sort_kernel_catalog",
        group_by_feature=None,
        calibrated=False,
    )
    assert _early_exit_request(args, groups=[0]) == [(30, 16)]
    # A pruned candidate carries no estimate, a calibrated ranking reads every one,
    # predict_engine scores one row, and a grouped model's runtime ignores the schedule.
    vars(args).update(unsupported)
    with pytest.raises(ValueError, match=reason):
        _early_exit_request(args, groups=[0])
