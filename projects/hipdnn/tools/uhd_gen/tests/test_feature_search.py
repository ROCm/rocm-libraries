# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for `uhd_gen feature-search`: the candidates, the search, and its recipe."""
import json

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("lightgbm")
from uhd_gen import feature_search
from uhd_gen.__main__ import main
from uhd_gen.features import evaluate_feature_rows, resolve_feature_evaluator
from uhd_gen.feature_search import (
    build_candidates,
    catalog_values_from_tree,
    search_features,
    verify_recipe,
)

SEEDS = ["0", "1"]


@pytest.fixture(scope="module", autouse=True)
def _short_boosting():
    """Enough trees for a synthetic frame; the search itself keeps its production count."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(feature_search, "BOOST_ROUNDS", 50)
        yield


def _evaluator_path() -> str:
    try:
        return resolve_feature_evaluator()
    except ValueError as error:
        pytest.skip(str(error))


def _rows(count=800, seed=7) -> pd.DataFrame:
    """Time depends on the grid `B*H*ceil_div(S, 64)` and on how full the padded head dim
    `ceil_div(D, 8) * 8` is; `noise` explains nothing and `q_copy` copies B."""
    rng = np.random.default_rng(seed)
    batch = rng.choice([1, 2, 4, 8], count)
    heads = rng.choice([2, 4, 8, 16], count)
    seq = rng.integers(16, 1025, count)
    dim = rng.integers(1, 129, count)
    fill = dim / (np.ceil(dim / 8) * 8)
    grid = batch * heads * np.ceil(seq / 64)
    log_time = 0.7 * np.log(grid) + 1.2 * fill + 0.03 * rng.standard_normal(count)
    published = {
        "g.q.dims[0]": batch,
        "g.q.dims[1]": heads,
        "g.q.dims[2]": seq,
        "g.q.dims[3]": dim,
        "g.noise": rng.random(count),
        "g.q_copy.dims[0]": batch,
        # Each of these is a column an exclusion rule names.
        "graph.tensors[0].dims[0]": batch,
        "g.q.numel": batch * heads * seq * dim,
        "g.scale": rng.integers(1, 5, count),
        "device.total_global_mem": np.full(count, 2**36),
    }
    frame = pd.DataFrame(published)
    frame.insert(0, "benchmark", [str(i) for i in range(count)])
    frame.insert(1, "device", "0")
    frame["features"] = [
        json.dumps(
            {
                name: (float(column[i]) if name == "g.noise" else int(column[i]))
                for name, column in published.items()
            }
        )
        for i in range(count)
    ]
    frame["tflops"] = np.exp(-log_time) * 100
    return frame


@pytest.fixture(scope="module")
def searched(tmp_path_factory):
    """The CLI run every assertion about the search reads: rows, output directory, report."""
    _evaluator_path()
    directory = tmp_path_factory.mktemp("search")
    frame = _rows()
    frame.to_csv(directory / "train.csv", index=False)
    output = directory / "out"
    code = main(
        [
            "feature-search",
            "--input",
            str(directory / "train.csv"),
            "--output-dir",
            str(output),
            "--seed",
            *SEEDS,
        ]
    )
    assert code == 0
    report = json.loads((output / "feature_search_report.json").read_text("utf-8"))
    return {"frame": frame, "directory": directory, "output": output, "report": report}


def test_the_search_keeps_the_pad_and_grid_features_and_drops_the_noise(searched):
    report = searched["report"]
    kept = report["final"]["features"]
    names = {entry["feature"] for entry in kept}
    # The padded head dim (or what it implies: the fill or the remainder) and a grid over
    # the sequence length are what time depends on.
    assert any(
        entry["template"] == "pad" and "$g.q.dims[3]" in entry["feature"]
        for entry in kept
    )
    assert any(
        entry["template"] == "grid"
        and "ceil_div" in entry["feature"]
        and "$g.q.dims[2]" in entry["feature"]
        for entry in kept
    )
    assert "$g.noise" not in names
    # Every kept feature says what removing it would cost.
    assert all(entry["removal_cost"] is not None for entry in kept)
    assert json.loads((searched["output"] / "recipe.json").read_text("utf-8")) == (
        report["recipe"]
    )
    assert (searched["output"] / "feature_search_report.md").read_text("utf-8")
    dropped = {entry["feature"]: entry["reason"] for entry in report["dropped"]}
    assert "$g.noise" in dropped and dropped["$g.noise"]


def test_a_duplicate_column_collapses_into_the_first_by_name(searched):
    candidates = searched["report"]["candidates"]
    assert candidates["duplicate_groups"] == [
        {"kept": "$g.q.dims[0]", "collapsed": ["$g.q_copy.dims[0]"]}
    ]
    assert "$g.q_copy.dims[0]" not in {
        entry["feature"] for entry in searched["report"]["final"]["features"]
    }


def test_excluded_columns_are_recorded_and_never_searched(searched):
    report = searched["report"]
    assert {(e["expression"], e["reason"]) for e in report["exclusions"]} == {
        ("$graph.tensors[0].dims[0]", "positional_tensor"),
        ("$g.q.numel", "derived_numel"),
        ("$g.scale", "caller_value"),
        ("$device.total_global_mem", "board_identity"),
    }
    # The positional copy of B would otherwise collapse into a duplicate group of its own.
    assert not any(
        "graph.tensors" in name
        for group in report["candidates"]["duplicate_groups"]
        for name in [group["kept"], *group["collapsed"]]
    )
    assert "$g.q.numel" not in json.dumps(report["recipe"])


def test_every_round_drops_a_correlated_cluster_whole(searched):
    report = searched["report"]
    clusters = report["candidates"]["correlated_clusters"]
    assert clusters, "the frame must hold near-duplicates for this to test anything"
    for round_ in report["rounds"]:
        kept = set(round_["kept"])
        for cluster in clusters:
            assert len(kept & set(cluster)) in (0, len(cluster)), round_["round"]


def test_a_later_round_never_holds_a_feature_an_earlier_one_dropped(searched):
    rounds = searched["report"]["rounds"]
    assert len(rounds) > 2
    for before, after in zip(rounds, rounds[1:]):
        assert set(after["kept"]) < set(before["kept"])


def test_the_same_rows_and_seeds_give_the_same_recipe(searched, tmp_path):
    again = tmp_path / "again"
    assert (
        main(
            [
                "feature-search",
                "--input",
                str(searched["directory"] / "train.csv"),
                "--output-dir",
                str(again),
                "--seed",
                *SEEDS,
            ]
        )
        == 0
    )
    assert (again / "recipe.json").read_bytes() == (
        searched["output"] / "recipe.json"
    ).read_bytes()
    # Not only the recipe: every round scored the same.
    repeat = json.loads((again / "feature_search_report.json").read_text("utf-8"))
    assert repeat["rounds"] == searched["report"]["rounds"]
    assert repeat["final"] == searched["report"]["final"]


def test_the_recipe_evaluates_through_the_shared_evaluator_for_every_row(searched):
    frame, recipe = searched["frame"], searched["report"]["recipe"]
    _, values = evaluate_feature_rows(frame, recipe, executable=_evaluator_path())
    matrix = np.asarray(values)
    assert matrix.shape == (len(frame), len(recipe))
    assert np.isfinite(matrix).all()


def _catalog_frame(count=200) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "t.q.dims[0]": rng.integers(1, 9, count),
            "t.q.dims[1]": rng.integers(1, 33, count),
            "t.q.dims[2]": rng.integers(1, 1500, count),
            "t.q.dims[3]": rng.choice([16, 40, 64, 100, 128, 200], count),
            "t.k.dims[1]": rng.integers(1, 9, count),
            "t.k.dims[2]": rng.integers(1, 1500, count),
            "t.bias.uid": rng.choice([-1, 12], count),
        }
    )
    # Strides of q in head-major or token-major layout.
    batch, heads, seq, dim = (frame[f"t.q.dims[{i}]"].to_numpy() for i in range(4))
    major = np.arange(count) % 3 == 0
    frame["t.q.strides[0]"] = batch * 0 + heads * seq * dim
    frame["t.q.strides[1]"] = np.where(major, seq * dim, dim)
    frame["t.q.strides[2]"] = np.where(major, dim, heads * dim)
    frame["t.q.strides[3]"] = 1
    frame["device"] = np.where(np.arange(count) % 2, "0", "1")
    frame["device.cu_count"] = np.where(frame["device"] == "0", 20, 40)
    return frame


def test_every_template_computes_what_the_shared_evaluator_computes():
    """The search scores numpy arithmetic; the runtime evaluates the same expressions in
    C++. Each candidate of each template must agree row by row."""
    frame = _catalog_frame()
    columns = [name for name in frame.columns if name.startswith(("t.", "device."))]
    columns.remove("t.bias.uid")
    pool, details = build_candidates(
        frame, columns, ["t.bias.uid"], {"head_dim_max": [64.0, 128.0, 256.0]}, 2000
    )
    assert {"base", "presence", "pad", "ratio", "stride_order", "grid", "tier"} == set(
        pool.stats
    )
    assert all(stat["kept"] for stat in pool.stats.values())
    assert any(
        "device.cu_count" in candidate.name
        for candidate in pool.candidates
        if candidate.template == "grid"
    )
    verify_recipe(
        [candidate.entry for candidate in pool.candidates],
        pool.candidates,
        frame,
        _evaluator_path(),
    )


def test_stride_order_flags_compare_adjacent_strides_before_the_innermost():
    frame = _catalog_frame()
    columns = [name for name in frame.columns if "strides" in name]
    pool, _ = build_candidates(frame, columns, [], None, 2000)
    # Adjacent pairs among strides[0..2]; the innermost stride is never compared.
    flags = [c.entry[">"] for c in pool.candidates if c.template == "stride_order"]
    assert ["$t.q.strides[1]", "$t.q.strides[2]"] in flags
    assert all("$t.q.strides[3]" not in flag for flag in flags)
    assert len(flags) == 2


def test_the_search_keeps_a_stride_order_flag_when_time_depends_on_layout():
    """A model sees raw strides, not which axis is outer; the order flag says it outright."""
    rng = np.random.default_rng(11)
    count = 700
    heads = rng.choice([2, 4, 8, 16], count)
    # Short sequences, so neither raw stride alone tells the layouts apart.
    seq = rng.integers(1, 25, count)
    dim = rng.choice([16, 32, 64, 128], count)
    head_major = rng.random(count) < 0.5
    frame = pd.DataFrame(
        {
            "t.q.dims[0]": rng.choice([1, 2, 4], count),
            "t.q.dims[1]": heads,
            "t.q.dims[2]": seq,
            "t.q.dims[3]": dim,
            "t.q.strides[0]": heads * seq * dim,
            "t.q.strides[1]": np.where(head_major, seq * dim, dim),
            "t.q.strides[2]": np.where(head_major, dim, heads * dim),
            "t.q.strides[3]": 1,
        }
    )
    frame["benchmark"] = [str(i) for i in range(count)]
    log_time = (
        0.8 * head_major
        + 0.3 * np.log(frame["t.q.dims[2]"])
        + 0.03 * rng.standard_normal(count)
    )
    frame["tflops"] = np.exp(-log_time)
    report = search_features(
        frame,
        columns=[name for name in frame.columns if name.startswith("t.")],
        target="tflops",
        objective="max",
        groups=frame["benchmark"],
        seeds=[0, 1],
        feature_evaluator=_evaluator_path(),
    )
    assert report["candidates"]["by_template"]["stride_order"]["kept"] == 1
    assert {"feature": '{">":["$t.q.strides[1]","$t.q.strides[2]"]}'} in [
        {"feature": entry["feature"]} for entry in report["final"]["features"]
    ]


def _signal_search(**options):
    """One column time depends on and five it does not; no dim columns, so no templates."""
    rng = np.random.default_rng(5)
    count = 400
    frame = pd.DataFrame({"s.x": rng.random(count)})
    for index in range(5):
        frame[f"s.n{index}"] = rng.random(count)
    frame["benchmark"] = [str(i) for i in range(count)]
    frame["tflops"] = np.exp(-3 * frame["s.x"])
    return search_features(
        frame,
        columns=[name for name in frame.columns if name.startswith("s.")],
        target="tflops",
        objective="max",
        groups=frame["benchmark"],
        seeds=[0],
        feature_evaluator=_evaluator_path(),
        **options,
    )


def test_the_final_pass_removes_what_the_score_does_not_need():
    report = _signal_search()
    # Six features is under the floor, so the rounds stop at once; the one-at-a-time pass
    # is what leaves the signal alone.
    assert len(report["rounds"]) == 1
    assert report["recipe"] == ["$s.x"]
    assert sum(step["removed"] for step in report["final_pass"]) == 5


def test_a_round_drops_the_least_important_features_first():
    report = _signal_search(floor=1)
    assert len(report["rounds"]) > 2
    for round_ in report["rounds"][:-1]:
        assert "$s.x" in round_["kept"], round_["round"]


def test_the_rounds_stop_after_two_worse_than_the_best():
    rng = np.random.default_rng(9)
    count = 500
    frame = pd.DataFrame({f"s.x{i}": rng.random(count) for i in range(4)})
    for index in range(4):
        frame[f"s.n{index}"] = rng.random(count)
    frame["benchmark"] = [str(i) for i in range(count)]
    frame["tflops"] = np.exp(-sum(frame[f"s.x{i}"] for i in range(4)))
    report = search_features(
        frame,
        columns=[name for name in frame.columns if name.startswith("s.")],
        target="tflops",
        objective="max",
        groups=frame["benchmark"],
        seeds=[0],
        floor=1,
        feature_evaluator=_evaluator_path(),
    )
    # 15% of eight is two, then one a round. Three left and two left are each worse than
    # four, so the search stops there instead of going on to the floor of one.
    assert [round_["features"] for round_ in report["rounds"]] == [8, 6, 5, 4, 3, 2]


def test_a_pad_multiple_at_or_above_the_dim_is_not_a_candidate():
    """Heads reach 32: padding them to 8 or 16 can differ, to 32 or more cannot."""
    pool, _ = build_candidates(_catalog_frame(), ["t.q.dims[1]"], [], None, 2000)
    names = {candidate.name for candidate in pool.candidates}
    assert '{"*":[{"ceil_div":["$t.q.dims[1]",16]},16]}' in names
    assert not any(
        '"$t.q.dims[1]",32]' in name or '"$t.q.dims[1]",64]' in name for name in names
    )


def test_a_candidate_the_evaluator_reads_differently_is_refused():
    frame = _catalog_frame()
    pool, _ = build_candidates(frame, ["t.q.dims[2]", "t.q.dims[3]"], [], None, 2000)
    wrong = pool.candidates[-1]
    pool.candidates[-1] = feature_search.Candidate(
        wrong.name, wrong.entry, wrong.template, wrong.values + 1.0
    )
    with pytest.raises(ValueError, match="disagrees"):
        verify_recipe(
            [candidate.entry for candidate in pool.candidates],
            pool.candidates,
            frame,
            _evaluator_path(),
        )


def test_the_tier_template_needs_a_catalog_and_says_so_when_it_has_none(tmp_path):
    frame = _catalog_frame()
    columns = [name for name in frame.columns if name.startswith("t.")]
    without, _ = build_candidates(frame, columns, [], None, 2000)
    assert "tier" not in without.stats
    with_tiers, _ = build_candidates(frame, columns, [], {"tile": [64.0, 128.0]}, 2000)
    chain = next(c for c in with_tiers.candidates if c.template == "tier")
    # Rounded up to the next catalog value; past the last one it stays as it is.
    assert chain.entry["/"][1]["if"][-1] == chain.entry["/"][0]
    assert set(with_tiers.stats["tier"]) >= {"generated", "kept"}


def test_the_catalog_values_come_from_the_kdp_kernels_of_one_engine(tmp_path):
    ued, other = (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )

    def kdp(identity, engine, tiles):
        return {
            "version": "1.0",
            "id": identity,
            "name": "kernels",
            "engine": engine,
            "kernelDescriptors": [
                {
                    "id": f"{identity[:-1]}{i}",
                    "metadata": {"tile": tile, "dtype": "bf16"},
                }
                for i, tile in enumerate(tiles)
            ],
        }

    (tmp_path / "a.ued.json").write_text(
        json.dumps({"version": "1.0", "id": ued, "name": "provider:a"}), "utf-8"
    )
    (tmp_path / "a.kdp.json").write_text(
        json.dumps(kdp("33333333-3333-4333-8333-333333333330", ued, [64, 128, 64])),
        "utf-8",
    )
    (tmp_path / "b.kdp.json").write_text(
        json.dumps(kdp("44444444-4444-4444-8444-444444444440", other, [32])), "utf-8"
    )
    assert catalog_values_from_tree(tmp_path, "provider:a") == {"tile": [64, 128]}
    assert catalog_values_from_tree(tmp_path, None) == {"tile": [32, 64, 128]}


def test_more_candidates_than_the_bound_are_refused_not_sampled():
    frame = _catalog_frame()
    columns = [name for name in frame.columns if name.startswith("t.")]
    with pytest.raises(ValueError, match="refusing rather than sampling"):
        build_candidates(frame, columns, [], None, 10)


def test_a_constant_column_is_dropped_and_the_first_duplicate_by_name_is_kept():
    frame = pd.DataFrame(
        {"b": [1, 2, 3, 4], "a": [1, 2, 3, 4], "c": [5, 5, 5, 5], "d": [4, 3, 1, 2]}
    )
    pool, details = build_candidates(frame, ["b", "a", "c", "d"], [], None, 2000)
    assert [candidate.name for candidate in pool.candidates] == ["$a", "$d"]
    assert pool.duplicates == {"$a": ["$b"]}
    assert details["constant_columns"] == ["c"]


def test_the_search_needs_enough_problems_for_its_folds():
    frame = _rows(count=40)
    with pytest.raises(ValueError, match="at least 5 training problems"):
        search_features(
            frame,
            columns=["g.q.dims[2]"],
            target="tflops",
            objective="max",
            groups=pd.Series(["x"] * 20 + ["y"] * 20, index=frame.index),
        )
