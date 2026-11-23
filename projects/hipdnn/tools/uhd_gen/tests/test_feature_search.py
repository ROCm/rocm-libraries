# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for `uhd_gen feature-search`: the candidates, the search, and its recipe."""
import json
import types
import warnings

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
from uhd_gen.tests.kdp_stub import kdp


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


def _run_search(train, output, *options):
    """`uhd_gen feature-search` over the rows in `train`, writing `output`: its exit code and
    the report (None when the search wrote none)."""
    code = main(
        [
            "feature-search",
            "--input",
            str(train),
            "--output-dir",
            str(output),
            *map(str, options),
        ]
    )
    report = output / "feature_search_report.json"
    return code, json.loads(report.read_text("utf-8")) if report.exists() else None


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
    code, report = _run_search(directory / "train.csv", output)
    assert code == 0
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
    # Every kept feature costs more than the noise to remove.
    noise = max(report["final"]["spread"], 0.002)
    assert all(entry["removal_cost"] > noise for entry in kept), kept
    assert json.loads((searched["output"] / "recipe.json").read_text("utf-8")) == (
        report["recipe"]
    )
    assert (searched["output"] / "feature_search_report.md").read_text("utf-8")
    dropped = {entry["feature"]: entry["reason"] for entry in report["dropped"]}
    assert "$g.noise" in dropped and dropped["$g.noise"]


def test_columns_equal_on_the_rows_are_all_searched_and_listed_and_the_search_keeps_one(
    searched,
):
    report = searched["report"]
    copy, original = "$g.q_copy.dims[0]", "$g.q.dims[0]"
    assert [original, copy] in report["candidates"]["equal_candidates"]
    assert {original, copy} <= set(report["rounds"][0]["kept"])
    # Nothing in the rows tells the two apart, so the elimination keeps no more than one.
    final = {entry["feature"] for entry in report["final"]["features"]}
    assert len({original, copy} & final) <= 1


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
        for group in report["candidates"]["equal_candidates"]
        for name in group
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
    code, repeat = _run_search(searched["directory"] / "train.csv", again)
    assert code == 0
    assert (again / "recipe.json").read_bytes() == (
        searched["output"] / "recipe.json"
    ).read_bytes()
    # Not only the recipe: every round scored the same.
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


def _signal_columns(count=400):
    """One column time depends on and five it does not; no dim columns, so no templates."""
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"s.x": rng.random(count)})
    for index in range(5):
        frame[f"s.n{index}"] = rng.random(count)
    return frame


def _signal_search(**options):
    options.setdefault("seeds", [0])
    frame = _signal_columns()
    count = len(frame)
    frame["benchmark"] = [str(i) for i in range(count)]
    frame["tflops"] = np.exp(-3 * frame["s.x"])
    return search_features(
        frame,
        columns=[name for name in frame.columns if name.startswith("s.")],
        target="tflops",
        objective="max",
        groups=frame["benchmark"],
        feature_evaluator=_evaluator_path(),
        **options,
    )


def test_the_prune_removes_a_feature_that_survives_elimination_but_costs_nothing():
    report = _signal_search()
    # Six features is under the floor, so the rounds stop at once with the noise columns
    # still in; the prune is what leaves the signal alone.
    assert len(report["rounds"]) == 1
    assert report["recipe"] == ["$s.x"]
    assert sum(step["removed"] for step in report["prune"]) == 5
    assert all(step["cost"] <= 0.002 for step in report["prune"] if step["removed"])


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


def test_grid_tiles_come_from_the_catalogs_tile_fields_else_64_and_128():
    from uhd_gen.feature_search import tile_sizes

    catalog = {
        "block_m": [64, 128],
        "block_n": [32],
        "tile_k": [1, 16],
        "head_dim_max": [96, 256],
        "has_bias": [0, 1],
    }
    assert tile_sizes(catalog) == ([16, 32, 64, 128], "catalog")
    assert tile_sizes({"head_dim_max": [96, 256]}) == ([64, 128], "fallback")


def test_a_grid_tiles_one_non_innermost_axis_and_multiplies_all_the_others():
    frame = _catalog_frame()
    columns = [name for name in frame.columns if name.startswith("t.q.")]
    pool, details = build_candidates(frame, columns, [], None, 2000, tiles=[16, 32])
    grids = [
        c.entry["*"] for c in pool.candidates if c.template == "grid" and "*" in c.entry
    ]
    assert details["grid_tiles"] == [16, 32]
    assert grids
    for factors in grids:
        # B, H and S are the non-innermost axes (D has stride 1): one tiled, two whole.
        text = json.dumps(factors)
        assert len(factors) == 3 and text.count("ceil_div") == 1
        assert "dims[3]" not in text
    tiles = {
        factor["ceil_div"][1]
        for factors in grids
        for factor in factors
        if isinstance(factor, dict)
    }
    assert tiles == {16, 32}


def test_the_contiguous_axis_is_the_one_with_stride_one_not_the_last_index():
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(
        {
            "t.q.dims[0]": rng.integers(1, 9, 100),
            "t.q.dims[1]": rng.integers(100, 900, 100),
            "t.q.dims[2]": rng.integers(1, 33, 100),
            "t.q.dims[3]": rng.integers(100, 900, 100),
            "t.q.strides[1]": 1,
        }
    )
    pool, _ = build_candidates(frame, list(frame.columns), [], None, 2000)
    grids = [json.dumps(c.entry) for c in pool.candidates if c.template == "grid"]
    assert grids and not any("dims[1]" in grid for grid in grids)
    assert any('{"ceil_div": ["$t.q.dims[3]"' in grid for grid in grids)


def _adapter_frame(count=300):
    """Time depends on `x` alone; the head dim only has a remainder an MLP might want."""
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {"s.x": rng.random(count), "g.q.dims[3]": rng.integers(1, 129, count)}
    )
    frame["benchmark"] = [str(i) for i in range(count)]
    frame["tflops"] = np.exp(-3 * frame["s.x"])
    return frame


def _adapter_search(**options):
    options.setdefault("seeds", [0, 1])
    frame = _adapter_frame()
    return frame, search_features(
        frame,
        columns=["s.x", "g.q.dims[3]"],
        target="tflops",
        objective="max",
        groups=frame["benchmark"],
        feature_evaluator=_evaluator_path(),
        **options,
    )


def _noisy_cells(monkeypatch, wanted, gain):
    """Stub `_MlpScorer.cells` with a known paired difference per (seed, fold) cell.

    Cells are shaped (seeds, folds) as the real method's are: `0.5 + 1e-4 * len(columns)`,
    less `gain(seed_index, fold)` when a column equals `wanted`. A set that adds a column
    pays 1e-4 for it; every other difference between two sets is `gain` on the cells it
    names, so a gain can differ by seed and fold. Returns the (columns, folds) of each
    call."""
    calls = []

    def cells(self, columns, folds):
        calls.append((tuple(columns), tuple(folds)))
        useful = any(np.array_equal(self.scorer.matrix[:, c], wanted) for c in columns)
        return np.array(
            [
                [
                    0.5 + 1e-4 * len(columns) - (gain(seed, fold) if useful else 0.0)
                    for fold in folds
                ]
                for seed in range(len(self.scorer.seeds))
            ]
        )

    monkeypatch.setattr(feature_search._MlpScorer, "cells", cells)
    return calls


def _remainder_values():
    """`dim % 8` of the adapter frame, the column the stub MLPs want."""
    return (_adapter_frame()["g.q.dims[3]"] % 8).to_numpy(float)


def _mlp_that_wants_the_remainder(monkeypatch):
    """A constant stub: the MLP scores 0.1 better with `dim % 8` among its columns."""
    _noisy_cells(monkeypatch, _remainder_values(), lambda seed, fold: 0.1)


def test_the_adapter_pass_adds_what_the_mlp_scores_better_and_records_each_delta(
    monkeypatch,
):
    _mlp_that_wants_the_remainder(monkeypatch)
    _, tree = _adapter_search()
    _, mlp = _adapter_search(adapter="mlp_data", adapter_pass_budget=40)
    assert tree["adapter_pass"] is None and tree["recipe"] == ["$s.x"]
    remainder_entry = {"%": ["$g.q.dims[3]", 8]}
    assert mlp["recipe"] == ["$s.x", remainder_entry]
    assert mlp["adapter_pass"]["added"] == [
        json.dumps(remainder_entry, separators=(",", ":"))
    ]
    tried = {entry["feature"]: entry for entry in mlp["adapter_pass"]["candidates"]}
    assert tried[json.dumps(remainder_entry, separators=(",", ":"))]["mlp_delta"] > 0.09
    # Every other candidate was tried, scored, and left out.
    assert all(
        not entry["added"] and entry["mlp_delta"] < 0.002
        for name, entry in tried.items()
        if name != json.dumps(remainder_entry, separators=(",", ":"))
    )
    [added] = [f for f in mlp["final"]["features"] if f["mlp_removal_cost"]]
    assert added["mlp_removal_cost"] > 0.09 and added["removal_cost"] is None


def test_the_mlp_keeps_the_pre_prune_set_when_it_scores_better(monkeypatch, tmp_path):
    """LightGBM sees `n0` as noise and the prune removes it; the stub MLP uses it."""
    n0 = _signal_columns()["s.n0"].to_numpy()
    _noisy_cells(monkeypatch, n0, lambda seed, fold: 0.1)
    report = _signal_search(adapter="mlp_data")
    choice = report["adapter_pass"]["set_choice"]
    assert report["prune"] and sum(step["removed"] for step in report["prune"]) == 5
    assert choice["chosen"] == "pre_prune"
    assert choice["pre_prune_mlp_score"] < choice["pruned_mlp_score"]
    assert choice["pruned_features"] == 1 and choice["pre_prune_features"] == 6
    assert "$s.n0" in report["recipe"] and len(report["recipe"]) == 6
    # The manifest summary describes the set that shipped, not the pruned one.
    summary = feature_search.search_summary(
        report, feature_search.write_search(report, tmp_path / "out")
    )
    assert summary["set_choice"]["chosen"] == "pre_prune"
    assert summary["mlp_score"] == report["adapter_pass"]["final_score"]


@pytest.mark.parametrize("budget", [0, 1, 6])
def test_the_adapter_pass_tries_no_more_candidates_than_its_budget(monkeypatch, budget):
    """Four templates compete for the budget: a budget of 6 is a round of four and a round
    of two, not two rounds of four."""
    frame = _catalog_frame()
    frame["benchmark"] = [str(i) for i in range(len(frame))]
    frame["tflops"] = np.exp(-0.3 * np.log(frame["t.q.dims[2]"]))
    _noisy_cells(monkeypatch, np.full(len(frame), -1.0), lambda seed, fold: 0.0)
    report = search_features(
        frame,
        columns=[name for name in frame.columns if ".dims[" in name],
        target="tflops",
        objective="max",
        groups=frame["benchmark"],
        catalog_values={"tile": [64.0, 128.0, 256.0]},
        seeds=[0],
        adapter="mlp_data",
        adapter_pass_budget=budget,
        feature_evaluator=_evaluator_path(),
    )
    tried = report["adapter_pass"]["candidates"]
    assert report["adapter_pass"]["budget"] == budget
    assert len(tried) == budget
    # Dealt out across the templates in turn: no template has the whole of a short budget.
    templates = [entry["template"] for entry in tried]
    assert templates == list(feature_search.ADAPTER_TEMPLATES)[:budget] + templates[4:]


def test_the_mlp_keeps_the_pruned_set_when_it_scores_better(monkeypatch):
    def mlp_that_prefers_fewer_features(self, columns, folds):
        return np.full((len(self.scorer.seeds), len(folds)), 0.5 + 0.01 * len(columns))

    monkeypatch.setattr(
        feature_search._MlpScorer, "cells", mlp_that_prefers_fewer_features
    )
    report = _signal_search(adapter="mlp_data")
    choice = report["adapter_pass"]["set_choice"]
    assert choice["chosen"] == "pruned"
    assert choice["pruned_mlp_score"] < choice["pre_prune_mlp_score"]
    assert report["recipe"] == ["$s.x"]
    assert report["final"]["features"][0]["removal_cost"] is None


def test_a_one_seed_adapter_pass_keeps_a_feature_the_removal_fold_says_is_needed(
    monkeypatch,
):
    """One seed leaves the removal fold a single cell, which has no spread to estimate: its
    standard error is 0 (the margin is the floor), not NaN, which kept nothing."""
    _mlp_that_wants_the_remainder(monkeypatch)
    _, report = _adapter_search(adapter="mlp_data", adapter_pass_budget=40, seeds=[0])
    remainder = {"%": ["$g.q.dims[3]", 8]}
    assert report["recipe"] == ["$s.x", remainder]
    [check] = report["adapter_pass"]["removal_check"]
    assert check["kept"] and np.isfinite(check["mlp_removal_cost"])
    [added] = [f for f in report["final"]["features"] if f["mlp_removal_cost"]]
    assert np.isfinite(added["mlp_removal_cost"]) and added["mlp_removal_cost"] > 0.09


def test_paired_with_one_fold_has_zero_standard_error():
    with warnings.catch_warnings():
        # A standard error of one fold is not computed: no "degrees of freedom" warning.
        warnings.simplefilter("error")
        mean, error = feature_search._paired(
            np.array([[0.5], [0.5]]), np.array([[0.25], [0.35]])
        )
    assert mean == pytest.approx(0.2) and error == 0.0


def test_paired_takes_the_standard_error_over_the_folds_not_the_cells():
    worse = np.full((3, 2), 0.5)
    # The same fall of 0.01 on one fold and 0.03 on the other, whatever the seed.
    mean, error = feature_search._paired(worse, worse - [[0.01, 0.03]] * 3)
    assert mean == pytest.approx(0.02)
    # Six independent cells would give 0.0045, a third of this.
    assert error == pytest.approx(np.std([0.01, 0.03], ddof=1) / np.sqrt(2))


def test_paired_averages_the_seeds_of_a_fold_before_the_error():
    worse = np.full((3, 2), 0.5)
    # The seeds disagree on each fold and agree on average: nothing is left to the error.
    falls = np.array([[0.01, 0.03], [0.03, 0.01], [0.02, 0.02]])
    mean, error = feature_search._paired(worse, worse - falls)
    assert mean == pytest.approx(0.02)
    assert error == pytest.approx(0.0, abs=1e-12)


def _gain_by_fold(discovery=0.0, confirm=0.0, removal=0.0):
    """`gain` for `_noisy_cells`: a level per stage (the folds 0-1, 2-3 and 4), or
    (level, wobble) for a gain that swings by +-wobble from one fold to the next, the same
    for every seed."""
    stages = {0: discovery, 1: discovery, 2: confirm, 3: confirm, 4: removal}

    def gain(seed, fold):
        level, wobble = (
            stages[fold] if isinstance(stages[fold], tuple) else (stages[fold], 0)
        )
        return level + wobble * (-1) ** fold

    return gain


def _fold_gains(gain, folds):
    """The stub's gain on each of `folds`, averaged over the three seeds."""
    return [np.mean([gain(seed, fold) for seed in range(3)]) for fold in folds]


def _standard_error(values):
    return np.std(values, ddof=1) / np.sqrt(len(values))


_REMAINDER = json.dumps({"%": ["$g.q.dims[3]", 8]}, separators=(",", ":"))


def _remainder_pass(monkeypatch, gain):
    """The adapter pass over `_adapter_frame` with three seeds, the stub MLP gaining `gain`
    from `dim % 8`: the report and the remainder's entry."""
    calls = _noisy_cells(monkeypatch, _remainder_values(), gain)
    _, report = _adapter_search(
        adapter="mlp_data", adapter_pass_budget=40, seeds=[0, 1, 2]
    )
    [entry] = [
        e for e in report["adapter_pass"]["candidates"] if e["feature"] == _REMAINDER
    ]
    return report, entry, calls


def test_a_gain_inside_two_standard_errors_is_not_picked(monkeypatch):
    # The folds swing between 0.05 and 0.01: the mean 0.03 clears the floor and one
    # standard error (0.02), not two.
    noisy = _gain_by_fold(discovery=(0.03, 0.02))
    report, entry, _ = _remainder_pass(monkeypatch, noisy)
    folds = _fold_gains(noisy, feature_search.ADAPTER_DISCOVERY)
    assert entry["mlp_delta"] == pytest.approx(np.mean(folds) - 1e-4)
    assert entry["mlp_delta_se"] == pytest.approx(_standard_error(folds))
    assert entry["mlp_delta_se"] < entry["mlp_delta"] < 2 * entry["mlp_delta_se"]
    assert not entry["picked"] and not entry["added"]
    assert entry["confirm_delta"] is None
    assert report["recipe"] == ["$s.x"]


def test_a_gain_past_two_standard_errors_and_inside_three_is_picked(monkeypatch):
    # The folds swing between 0.07 and 0.03: the mean 0.05 is 2.5 standard errors.
    _, entry, _ = _remainder_pass(monkeypatch, _gain_by_fold(discovery=(0.05, 0.02)))
    assert 2 * entry["mlp_delta_se"] < entry["mlp_delta"] < 3 * entry["mlp_delta_se"]
    assert entry["picked"]


def test_a_gain_past_two_standard_errors_is_picked_confirmed_and_kept(monkeypatch):
    steady = (0.05, 0.001)
    report, entry, _ = _remainder_pass(
        monkeypatch, _gain_by_fold(discovery=steady, confirm=steady, removal=steady)
    )
    assert entry["mlp_delta"] > 2 * entry["mlp_delta_se"] > 0
    assert entry["picked"] and entry["added"]
    assert report["adapter_pass"]["removal_check"][0]["kept"]
    assert json.dumps(report["recipe"][-1], separators=(",", ":")) == _REMAINDER


def test_a_pick_the_confirm_folds_do_not_support_is_not_added(monkeypatch):
    steady = (0.05, 0.001)
    report, entry, calls = _remainder_pass(
        monkeypatch, _gain_by_fold(discovery=steady, removal=steady)
    )
    assert entry["picked"] and not entry["added"]
    assert entry["confirm_delta"] < feature_search.MIN_TOLERANCE
    assert report["adapter_pass"]["removal_check"] == []
    assert report["recipe"] == ["$s.x"]
    # The confirmation reads the folds discovery did not.
    assert feature_search.ADAPTER_CONFIRM in {folds for _, folds in calls}


def test_a_pick_the_confirm_folds_support_inside_one_standard_error_is_not_added(
    monkeypatch,
):
    steady = (0.05, 0.001)
    # The folds swing between 0.04 and -0.02: the mean 0.01 clears the floor, not one error.
    report, entry, _ = _remainder_pass(
        monkeypatch,
        _gain_by_fold(discovery=steady, confirm=(0.01, 0.03), removal=steady),
    )
    assert entry["picked"] and not entry["added"]
    assert entry["confirm_delta"] > feature_search.MIN_TOLERANCE
    assert report["recipe"] == ["$s.x"]


def test_a_pick_the_confirm_folds_support_past_one_standard_error_is_added(monkeypatch):
    steady = (0.05, 0.001)
    # The folds swing between 0.025 and 0.005: the mean 0.015 is 1.5 standard errors.
    confirm = (0.015, 0.01)
    report, entry, _ = _remainder_pass(
        monkeypatch, _gain_by_fold(discovery=steady, confirm=confirm, removal=steady)
    )
    error = _standard_error(_fold_gains(_gain_by_fold(confirm=confirm), (2, 3)))
    assert error < entry["confirm_delta"] < 2 * error
    assert entry["picked"] and entry["added"]


def test_a_gain_only_on_the_confirm_folds_is_never_picked(monkeypatch):
    steady = (0.05, 0.001)
    report, entry, _ = _remainder_pass(
        monkeypatch, _gain_by_fold(confirm=steady, removal=steady)
    )
    assert not entry["picked"] and not entry["added"]
    assert entry["confirm_delta"] is None
    assert report["recipe"] == ["$s.x"]


def test_an_added_feature_the_removal_fold_does_not_need_is_dropped_again(monkeypatch):
    steady = (0.05, 0.001)
    report, entry, _ = _remainder_pass(
        monkeypatch,
        _gain_by_fold(discovery=steady, confirm=steady, removal=(0.0, 0.001)),
    )
    assert entry["picked"] and entry["added"]
    [check] = report["adapter_pass"]["removal_check"]
    assert check["feature"] == _REMAINDER and not check["kept"]
    assert report["recipe"] == ["$s.x"]
    assert report["adapter_pass"]["added"] == []


#: A stage's gain with no spread, so its margin is the floor alone: a paired difference is
#: the gain less the 1e-4 the extra column costs, 0.0014 or 0.0024 for these.
BELOW_THE_FLOOR, ABOVE_THE_FLOOR = 0.0015, 0.0025


@pytest.mark.parametrize(
    "stage, level, passes",
    [
        ("discovery", BELOW_THE_FLOOR, False),
        ("discovery", ABOVE_THE_FLOOR, True),
        ("confirm", BELOW_THE_FLOOR, False),
        ("confirm", ABOVE_THE_FLOOR, True),
        ("removal", BELOW_THE_FLOOR, False),
        ("removal", ABOVE_THE_FLOOR, True),
    ],
)
def test_each_stage_needs_the_floor_when_its_gain_has_no_spread(
    monkeypatch, stage, level, passes
):
    levels = {"discovery": 0.05, "confirm": 0.05, "removal": 0.05, stage: level}
    report, entry, _ = _remainder_pass(monkeypatch, _gain_by_fold(**levels))
    if stage == "removal":
        [check] = report["adapter_pass"]["removal_check"]
        assert check["kept"] == passes
    else:
        assert {"discovery": entry["picked"], "confirm": entry["added"]}[
            stage
        ] == passes


def _pre_prune_choice(monkeypatch, gain):
    """The set choice when the stub MLP gains `gain` from `n0`, which the prune removed."""
    n0 = _signal_columns()["s.n0"].to_numpy()
    _noisy_cells(monkeypatch, n0, gain)
    report = _signal_search(adapter="mlp_data", seeds=[0, 1, 2])
    assert sum(step["removed"] for step in report["prune"]) == 5
    return report, report["adapter_pass"]["set_choice"]


def test_the_pre_prune_set_needs_a_gain_past_two_standard_errors_to_replace_the_pruned(
    monkeypatch,
):
    swinging = (0.02, 0.03)
    report, choice = _pre_prune_choice(
        monkeypatch, _gain_by_fold(confirm=swinging, removal=swinging)
    )
    # Ahead on average, by more than one standard error and by less than 2.
    gain, error = choice["pre_prune_gain"], choice["pre_prune_gain_se"]
    assert error < gain < 2 * error
    assert choice["chosen"] == "pruned"
    assert report["recipe"] == ["$s.x"]


def test_the_pre_prune_set_replaces_the_pruned_with_a_gain_inside_three_errors(
    monkeypatch,
):
    swinging = (0.015, 0.01)
    report, choice = _pre_prune_choice(
        monkeypatch, _gain_by_fold(confirm=swinging, removal=swinging)
    )
    gain, error = choice["pre_prune_gain"], choice["pre_prune_gain_se"]
    assert 2 * error < gain < 3 * error
    assert choice["chosen"] == "pre_prune"


@pytest.mark.parametrize(
    "level, chosen",
    [
        # The five extra columns cost 5e-4: the gain is 0.0019 and 0.003.
        (0.0024, "pruned"),
        (0.0035, "pre_prune"),
    ],
)
def test_the_pre_prune_set_needs_the_floor_when_its_gain_has_no_spread(
    monkeypatch, level, chosen
):
    _, choice = _pre_prune_choice(
        monkeypatch, _gain_by_fold(confirm=level, removal=level)
    )
    assert choice["chosen"] == chosen


def test_a_clear_gain_on_the_held_folds_chooses_the_pre_prune_set(monkeypatch):
    steady = (0.05, 0.001)
    # Folds 2-4 held: the stub's gain there, not on the discovery folds, decides.
    report, choice = _pre_prune_choice(
        monkeypatch, _gain_by_fold(confirm=steady, removal=steady)
    )
    assert choice["pre_prune_gain"] > 2 * choice["pre_prune_gain_se"] > 0
    assert choice["chosen"] == "pre_prune"
    assert "$s.n0" in report["recipe"]


def test_a_gain_only_on_the_discovery_folds_does_not_choose_the_pre_prune_set(
    monkeypatch,
):
    report, choice = _pre_prune_choice(
        monkeypatch, _gain_by_fold(discovery=(0.05, 0.001))
    )
    assert choice["chosen"] == "pruned"
    assert report["recipe"] == ["$s.x"]


def _scripted_rounds(monkeypatch, errors, spread):
    """Stub `_Scorer.score`: the held-out error of a set is `errors[its size]`, whatever it
    holds, with `spread` over seeds and no importance to rank by."""

    def score(self, columns, clusters=None):
        importance = None if clusters is None else np.zeros(len(clusters))
        return errors[len(columns)], spread, importance

    monkeypatch.setattr(feature_search._Scorer, "score", score)


def test_the_rounds_and_the_prune_tolerate_the_spread_over_seeds_when_it_is_wide(
    monkeypatch,
):
    # Best at four features; three are 0.009 worse, inside the spread 0.01, not the floor.
    _scripted_rounds(
        monkeypatch,
        {6: 0.507, 5: 0.504, 4: 0.500, 3: 0.509, 2: 0.512, 1: 0.515},
        spread=0.01,
    )
    report = _signal_search(floor=1)
    assert [r["features"] for r in report["rounds"]] == [6, 5, 4, 3, 2, 1]
    assert report["selected"]["best_round"] == 2
    assert report["selected"]["features"] == 3
    # Removing one more costs 0.003: past the floor, within the spread.
    assert [step["removed"] for step in report["prune"]] == [True, True]
    assert all(0.002 < step["cost"] < 0.01 for step in report["prune"])


def test_the_rounds_tolerate_the_floor_when_the_spread_over_seeds_is_narrower(
    monkeypatch,
):
    # Three features are 0.0015 worse than four: no spread to cover it, the floor does.
    _scripted_rounds(
        monkeypatch,
        {6: 0.510, 5: 0.505, 4: 0.500, 3: 0.5015, 2: 0.520, 1: 0.530},
        spread=0.0,
    )
    report = _signal_search(floor=1)
    assert [r["features"] for r in report["rounds"]] == [6, 5, 4, 3, 2, 1]
    assert report["selected"]["best_round"] == 2
    assert report["selected"]["features"] == 3


def _scorer_over(columns, groups=None, count=400):
    rng = np.random.default_rng(6)
    matrix = rng.random((count, 3))
    target = -3 * matrix[:, 0]
    groups = np.arange(count) if groups is None else groups
    return feature_search._Scorer(matrix, target, groups, [0, 1], 5), matrix


def test_the_real_mlp_scores_the_column_time_depends_on_far_better():
    scorer, _ = _scorer_over(None)
    mlp = feature_search._MlpScorer(scorer)
    signal, noise = mlp.cells([0], (0,)), mlp.cells([1], (0,))
    # One cell per seed on the one fold.
    assert signal.shape == noise.shape == (2, 1)
    assert (noise - signal).min() > 0.1
    delta, error = feature_search._paired(noise, signal)
    assert delta > 0.1 and error >= 0


def test_the_mlp_pass_fits_with_the_callers_options_and_whole_graphs(monkeypatch):
    seen = []

    def fit(matrix, target, **options):
        seen.append(options)
        return types.SimpleNamespace(predict=lambda rows: np.zeros(len(rows))), {}

    monkeypatch.setattr("uhd_gen.mlp.fit_mlp", fit)
    groups = np.arange(400) // 4
    scorer, _ = _scorer_over(None, groups)
    feature_search._MlpScorer(scorer, {"hidden": (8,), "epochs": 7}).cells([0], (0, 1))
    assert len(seen) == 4
    for options in seen:
        assert options["hidden"] == (8,) and options["epochs"] == 7
        assert options["members"] == 1
    # Early stopping holds out whole graphs: the groups are the training rows' graphs.
    assert np.array_equal(seen[0]["groups"], groups[scorer.adapter_fold_of != 0])


def test_no_graph_falls_in_two_folds_even_across_devices(monkeypatch):
    """Two boards measure each graph: the search groups by graph, so both rows are on one
    side of every split (grouping by graph and device would split 'one graph's rows').
    """
    recorded = []
    real = feature_search._folds

    def spy(group_ids, seed, folds):
        result = real(group_ids, seed, folds)
        recorded.append((np.asarray(group_ids), result))
        return result

    monkeypatch.setattr(feature_search, "_folds", spy)
    rng = np.random.default_rng(8)
    count = 400
    frame = pd.DataFrame({"s.x": np.repeat(rng.random(count // 2), 2)})
    frame["s.n0"] = rng.random(count)
    frame["benchmark"] = [str(i // 2) for i in range(count)]
    frame["device"] = [str(i % 2) for i in range(count)]
    frame["tflops"] = np.exp(-3 * frame["s.x"])
    search_features(
        frame,
        columns=["s.x", "s.n0"],
        target="tflops",
        objective="max",
        groups=frame["benchmark"],
        seeds=[0, 1],
        feature_evaluator=_evaluator_path(),
    )
    assert len(recorded) >= 2
    for group_ids, folds in recorded:
        assert len(set(folds)) == 5
        for graph in np.unique(group_ids):
            assert len(set(folds[group_ids == graph])) == 1


def test_the_adapter_pass_scores_the_same_held_out_graphs_whatever_the_seed(
    monkeypatch,
):
    """The discovery, confirm and removal folds are disjoint sets of graphs across all the
    seeds, not only within one: a seed varies the model, not which graphs are held out.
    """
    count = 200
    matrix = np.column_stack([np.arange(count), np.random.default_rng(3).random(count)])
    held = []

    def fit(train, target, **options):
        held.append([])
        return (
            types.SimpleNamespace(
                predict=lambda rows: (
                    held[-1].extend(rows[:, 0].astype(int)),
                    np.zeros(len(rows)),
                )[1]
            ),
            {},
        )

    monkeypatch.setattr("uhd_gen.mlp.fit_mlp", fit)
    scorer = feature_search._Scorer(
        matrix, np.zeros(count), np.arange(count), [0, 1, 2], 5
    )
    mlp = feature_search._MlpScorer(scorer)
    stages = {}
    for name, folds in (
        ("discovery", feature_search.ADAPTER_DISCOVERY),
        ("confirm", feature_search.ADAPTER_CONFIRM),
        ("removal", feature_search.ADAPTER_REMOVAL),
    ):
        held.clear()
        mlp.cells([0, 1], folds)
        # One fit per (seed, fold), seed-major.
        assert len(held) == 3 * len(folds)
        per_fold = [
            {tuple(sorted(held[seed * len(folds) + k])) for seed in range(3)}
            for k in range(len(folds))
        ]
        assert all(len(sets) == 1 for sets in per_fold), name
        stages[name] = set().union(*(set(*sets) for sets in per_fold))
    assert not stages["discovery"] & stages["confirm"]
    assert not stages["discovery"] & stages["removal"]
    assert not stages["confirm"] & stages["removal"]
    assert len(stages["discovery"] | stages["confirm"] | stages["removal"]) == count


def test_the_stride_order_flag_reads_the_stride_even_when_it_equals_a_dim():
    rng = np.random.default_rng(1)
    count = 60
    batch = rng.integers(2, 9, count)
    frame = pd.DataFrame(
        {
            "t.q.dims[0]": batch,
            "t.q.strides[0]": rng.integers(1000, 2000, count),
            # Equal to the batch dim on every row.
            "t.q.strides[1]": batch,
            "t.q.strides[2]": rng.integers(1, 9, count),
            "t.q.strides[3]": 1,
        }
    )
    pool, _ = build_candidates(frame, list(frame.columns), [], None, 2000)
    flags = [c.entry for c in pool.candidates if c.template == "stride_order"]
    assert {">": ["$t.q.strides[1]", "$t.q.strides[2]"]} in flags


def test_a_grid_reads_its_own_tensors_heads_when_another_tensors_are_equal_on_the_rows():
    """Every training graph here has as many key heads as query heads, so `k.dims[1]` equals
    `q.dims[1]` on every row; the grid over q's launch must still read q's."""
    rng = np.random.default_rng(12)
    count = 200
    heads = rng.choice([2, 4, 8], count)
    frame = pd.DataFrame(
        {
            "t.k.dims[0]": rng.integers(1, 9, count),
            "t.k.dims[1]": heads,
            "t.k.dims[2]": rng.integers(1, 1500, count),
            "t.q.dims[1]": heads,
            "t.q.dims[2]": rng.integers(1, 1500, count),
        }
    )
    frame["t.q.dims[0]"] = frame["t.k.dims[0]"]
    frame["t.k.dims[3]"] = frame["t.q.dims[3]"] = rng.choice([64, 128], count)
    pool, _ = build_candidates(frame, list(frame.columns), [], None, 2000)
    assert ["$t.k.dims[1]", "$t.q.dims[1]"] in pool.equal_candidates()
    grids = {
        c.name: json.dumps(c.entry["*"])
        for c in pool.candidates
        if c.template == "grid"
    }
    q_grids = [text for text in grids.values() if "$t.q.dims[1]" in text]
    assert q_grids and all("$t.k." not in text for text in q_grids)
    k_grids = [text for text in grids.values() if "$t.k.dims[1]" in text]
    assert k_grids and all("$t.q." not in text for text in k_grids)
    # Ratios across roles are offered for the equal pair too: kv heads over query heads
    # would be constant, but the query sequence over the key sequence is not.
    assert any(
        c.entry == {"/": ["$t.k.dims[2]", "$t.q.dims[2]"]}
        or c.entry == {"/": ["$t.q.dims[2]", "$t.k.dims[2]"]}
        for c in pool.candidates
    )


def test_a_fill_is_offered_for_an_optional_dim_and_reads_one_where_it_is_unbound():
    from uhd_gen.optional import lint

    frame = _optional_rows()
    columns, identifiers, guards, _ = _optional_inputs(frame)
    pool, _ = build_candidates(
        frame,
        columns,
        identifiers,
        None,
        2000,
        feature_search.FALLBACK_TILES,
        guards,
        _evaluator_path(),
    )
    top = frame[BIAS_DIM].max()
    multiples = [m for m in feature_search.PAD_MULTIPLES if m < top]
    fills = [
        c
        for c in pool.candidates
        if c.template == "pad"
        and BIAS_DIM in json.dumps(c.entry)
        and isinstance(c.entry, dict)
        and "if" in c.entry
    ]
    assert len(fills) == len(multiples) > 1
    unbound = frame[BIAS_DIM].isna().to_numpy()
    assert unbound.any() and (~unbound).any()
    for fill in fills:
        # No padding to fill where there is no dim; the fraction of the padded dim where
        # there is one.
        assert (fill.values[unbound] == 1).all()
        assert ((fill.values[~unbound] > 0) & (fill.values[~unbound] <= 1)).all()
    entries = [c.entry for c in pool.candidates]
    verify_recipe(entries, pool.candidates, frame, _evaluator_path())
    assert lint(entries, "tflops", _evaluator_path()) == []


def test_a_tier_ladder_of_flags_or_out_of_range_values_is_not_crossed_with_the_dims():
    frame = _catalog_frame()
    columns = [name for name in frame.columns if name.startswith("t.")]
    pool, _ = build_candidates(
        frame,
        columns,
        [],
        {"has_bias": [0.0, 1.0], "far": [5000.0, 6000.0], "one_big": [1.0, 64.0]},
        2000,
    )
    assert "tier" not in pool.stats


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
    chain = next(
        c for c in with_tiers.candidates if c.template == "tier" and "/" in c.entry
    )
    # Rounded up to the next catalog value; past the last one it stays as it is.
    assert chain.entry["/"][1]["if"][-1] == chain.entry["/"][0]
    assert set(with_tiers.stats["tier"]) >= {"generated", "kept"}


def test_the_catalog_values_come_from_the_kdp_kernels_of_one_engine(tmp_path):
    ued, other = (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )

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


def test_a_constant_column_is_dropped_and_columns_with_equal_values_are_all_kept():
    frame = pd.DataFrame(
        {"b": [1, 2, 3, 4], "a": [1, 2, 3, 4], "c": [5, 5, 5, 5], "d": [4, 3, 1, 2]}
    )
    pool, details = build_candidates(frame, ["b", "a", "c", "d"], [], None, 2000)
    assert [candidate.name for candidate in pool.candidates] == ["$a", "$b", "$d"]
    assert pool.equal_candidates() == [["$a", "$b"]]
    assert details["constant_columns"] == ["c"]


def test_an_expression_offered_twice_is_one_candidate():
    """Two catalog fields with the same ladder round a dim up to the same tiers."""
    frame = _catalog_frame()
    columns = [name for name in frame.columns if name.startswith("t.q.")]
    ladder = [64.0, 128.0, 256.0]
    once, _ = build_candidates(frame, columns, [], {"a": ladder}, 2000)
    twice, _ = build_candidates(frame, columns, [], {"a": ladder, "b": ladder}, 2000)
    assert [c.name for c in twice.candidates] == [c.name for c in once.candidates]
    assert twice.stats["tier"]["duplicate"] == once.stats["tier"]["kept"] > 0
    assert twice.equal_candidates() == once.equal_candidates()


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


BIAS_UID = "graph.nodes[0].bias.uid"
BIAS_DIM = "graph.nodes[0].bias.dims[1]"
DROPOUT = "graph.nodes[0].dropout_probability"


def _optional_rows(count=600, seed=11) -> pd.DataFrame:
    """RMSNorm-like rows whose optional bias tensor and dropout attribute are unbound on
    some of them (NaN: the row does not publish them). Time depends on whether the bias is
    bound, on how full its padded dim is, and on the dropout probability."""
    rng = np.random.default_rng(seed)
    batch = rng.choice([1, 2, 4, 8], count)
    seq = rng.integers(16, 1025, count)
    bound = rng.random(count) < 0.6
    bias_dim = rng.integers(1, 129, count)
    dropout = rng.choice([0.1, 0.2, 0.5], count)
    with_dropout = rng.random(count) < 0.5
    fill = np.where(bound, bias_dim / (np.ceil(bias_dim / 8) * 8), 0.0)
    log_time = (
        0.5 * np.log(batch * seq)
        + 0.9 * bound
        + 1.2 * fill
        + 2.0 * np.where(with_dropout, dropout, 0.0)
        + 0.03 * rng.standard_normal(count)
    )
    frame = pd.DataFrame(
        {
            "benchmark": [str(i) for i in range(count)],
            "device": "0",
            "graph.nodes[0].x.dims[0]": batch,
            "graph.nodes[0].x.dims[1]": seq,
            BIAS_UID: np.where(bound, 12.0, np.nan),
            BIAS_DIM: np.where(bound, bias_dim, np.nan),
            DROPOUT: np.where(with_dropout, dropout, np.nan),
        }
    )
    frame["tflops"] = np.exp(-log_time) * 100
    return frame


def _unguarded_reads(node, names: set) -> list:
    """The reads of `names` in a signature entry outside value_or_default's first argument."""
    if isinstance(node, str):
        return [node] if node.startswith("$") and node[1:] in names else []
    if isinstance(node, list):
        return [read for item in node for read in _unguarded_reads(item, names)]
    if not isinstance(node, dict):
        return []
    ((op, arguments),) = node.items()
    if op == "value_or_default":
        return _unguarded_reads(arguments[1:], names)
    return _unguarded_reads(arguments, names)


def _optional_inputs(frame):
    from uhd_gen.proposal import search_inputs

    return search_inputs(
        frame,
        set(frame.columns) - {"benchmark", "device", "tflops"},
        None,
        [],
        "tflops",
        _evaluator_path(),
    )


def test_a_search_guards_every_optional_read_and_a_uid_stays_a_presence_flag():
    from uhd_gen.proposal import presence_flag
    from uhd_gen.optional import lint

    frame = _optional_rows()
    columns, identifiers, guards, omitted = _optional_inputs(frame)
    # The optional columns some rows leave unbound are searched, each with the default a
    # model reads; the uid is only ever a presence flag.
    assert guards == {BIAS_UID: -1, BIAS_DIM: -1, DROPOUT: 0.0}
    assert sorted(columns) == sorted(
        [BIAS_DIM, DROPOUT, "graph.nodes[0].x.dims[0]", "graph.nodes[0].x.dims[1]"]
    )
    assert identifiers == [BIAS_UID]
    assert {
        "expression": "$" + BIAS_UID,
        "reason": "tensor_identifier",
        "read_instead": presence_flag(BIAS_UID, -1),
    } in omitted

    report = search_features(
        frame,
        columns=columns,
        presence_columns=identifiers,
        guards=guards,
        target="tflops",
        objective="max",
        groups=frame["benchmark"],
        seeds=[0, 1],
        feature_evaluator=_evaluator_path(),
    )
    recipe = report["recipe"]
    # The unbound rows stayed in the frame and the evaluator scores every one of them.
    assert report["rows"] == len(frame)
    assert lint(recipe, "tflops", _evaluator_path()) == []
    text = json.dumps(recipe)
    assert "value_or_default" in text
    # Each optional name is searched, and every read of one sits inside value_or_default,
    # whatever template it is under.
    assert {name for name in guards if name in text} >= {DROPOUT, BIAS_DIM}
    assert all(_unguarded_reads(entry, set(guards)) == [] for entry in recipe), recipe
    # The uid is never read by value: the only entry that names it is the presence flag.
    uid_entries = [entry for entry in recipe if BIAS_UID in json.dumps(entry)]
    assert uid_entries in ([], [presence_flag(BIAS_UID, -1)])
    assert BIAS_UID not in [entry[1:] for entry in recipe if isinstance(entry, str)]


def test_a_template_over_an_optional_dim_reads_it_guarded_and_matches_the_evaluator():
    from uhd_gen.proposal import presence_flag
    from uhd_gen.optional import lint

    frame = _optional_rows()
    columns, identifiers, guards, _ = _optional_inputs(frame)
    pool, _ = build_candidates(
        frame,
        columns,
        identifiers,
        {"head_dim_max": [64.0, 128.0, 256.0]},
        2000,
        feature_search.FALLBACK_TILES,
        guards,
        _evaluator_path(),
    )
    over_dim = [c for c in pool.candidates if BIAS_DIM in json.dumps(c.entry)]
    assert {c.template for c in over_dim} >= {"base", "pad", "tier"}
    assert all(_unguarded_reads(c.entry, {BIAS_DIM}) == [] for c in over_dim)
    presence = [c for c in pool.candidates if c.template == "presence"]
    assert [c.entry for c in presence] == [presence_flag(BIAS_UID, -1)]
    # A guarded identifier is never a value column.
    assert not any(
        c.template == "base" and BIAS_UID in json.dumps(c.entry)
        for c in pool.candidates
    )
    # Rows that leave the dim unbound read the default: the flag is 0 exactly there, and
    # every candidate's numpy values are the shared evaluator's on every row.
    unbound = frame[BIAS_DIM].isna().to_numpy()
    assert unbound.any() and (~unbound).any()
    np.testing.assert_array_equal(presence[0].values, (~unbound).astype(float))
    verify_recipe(
        [c.entry for c in pool.candidates], pool.candidates, frame, _evaluator_path()
    )
    assert lint([c.entry for c in pool.candidates], "tflops", _evaluator_path()) == []


def test_the_standalone_search_publishes_the_columns_any_row_binds(tmp_path):
    frame = _optional_rows()
    # The first row leaves the optional bias and dropout unbound; other rows bind them.
    frame.loc[0, [BIAS_UID, BIAS_DIM, DROPOUT]] = np.nan
    assert frame[BIAS_DIM].notna().any() and frame[DROPOUT].notna().any()
    names = [c for c in frame.columns if c not in ("benchmark", "device", "tflops")]
    # A row's `features` map lists only the keys it binds.
    frame["features"] = [
        json.dumps({name: row[name] for name in names if pd.notna(row[name])})
        for _, row in frame.iterrows()
    ]
    assert set(json.loads(frame["features"].iloc[0])) == {
        "graph.nodes[0].x.dims[0]",
        "graph.nodes[0].x.dims[1]",
    }
    frame.to_csv(tmp_path / "train.csv", index=False)
    output = tmp_path / "out"
    code, report = _run_search(
        tmp_path / "train.csv", output, "--feature-evaluator", _evaluator_path()
    )
    assert code == 0
    # The columns only some rows bind reach the search, each read through value_or_default:
    # two bound columns and the bias dim and dropout probability.
    assert report["candidates"]["by_template"]["base"]["generated"] == 4
    searched_text = json.dumps(report["rounds"][0]["kept"])
    assert BIAS_DIM in searched_text and DROPOUT in searched_text
    assert all(
        _unguarded_reads(entry, {BIAS_DIM, DROPOUT}) == [] for entry in report["recipe"]
    ), report["recipe"]


def test_an_optional_column_that_holds_strings_cannot_take_a_default_and_is_left_out():
    frame = _optional_rows()
    # An optional attribute published as a string: its default is a number, which the
    # encoding of a categorical cannot take.
    name = "graph.nodes[0].swish_beta"
    frame[name] = np.where(frame[DROPOUT].isna(), None, "yes")
    columns, _, guards, omitted = _optional_inputs(frame)
    assert name not in columns and name not in guards
    assert {"expression": "$" + name, "reason": "optional_categorical"} in omitted


def _csv(directory, frame, name="train.csv"):
    frame.to_csv(directory / name, index=False)
    return directory / name


def test_the_target_objective_columns_and_seed_options_reach_the_report(tmp_path):
    frame = _rows(count=300)
    frame["score"] = frame["tflops"]
    code, report = _run_search(
        _csv(tmp_path, frame),
        tmp_path / "out",
        *("--target", "score", "--objective", "min", "--seed", 5),
        *("--columns", "g.q.dims[1]", "g.q.dims[2]", "g.noise"),
    )
    assert code == 0
    assert (report["target"], report["objective"]) == ("score", "min")
    assert report["seeds"] == [5, 6, 7]
    # Only the named columns are read; the other published ones are not searched.
    assert {n for n in report["rounds"][0]["kept"] if n.startswith("$")} == {
        "$g.q.dims[1]",
        "$g.q.dims[2]",
        "$g.noise",
    }
    assert report["candidates"]["by_template"]["base"]["generated"] == 3


def _two_device_rows(count=100):
    """`count` graphs, each measured on two boards: the same published features."""
    frame = pd.concat([_rows(count=count), _rows(count=count)], ignore_index=True)
    frame["device"] = ["0"] * count + ["1"] * count
    frame.loc[count:, "tflops"] *= 1.1
    return frame


def test_group_by_names_the_columns_that_together_identify_a_graph(tmp_path):
    train = _csv(tmp_path, _two_device_rows())
    reports = {}
    for name, columns in (
        ("default", []),
        ("both", ["--group-by", "benchmark", "device"]),
        ("reversed", ["--group-by", "device", "benchmark"]),
    ):
        code, reports[name] = _run_search(train, tmp_path / name, *columns)
        assert code == 0, name
    # By default a graph is its benchmark; with both columns listed, a benchmark on each
    # board is a graph of its own, in whichever order they are listed.
    assert reports["default"]["problems"] == 100
    assert reports["both"]["problems"] == reports["reversed"]["problems"] == 200
    assert reports["reversed"]["rounds"] == reports["both"]["rounds"]
    assert reports["reversed"]["recipe"] == reports["both"]["recipe"]


def test_a_group_by_column_the_rows_lack_is_refused(tmp_path, caplog):
    code, report = _run_search(
        _csv(tmp_path, _rows(count=100)),
        tmp_path / "out",
        *("--group-by", "benchmark", "board"),
    )
    assert code == 1 and report is None
    assert "board" in caplog.text


def test_max_candidates_refuses_a_larger_search(tmp_path, caplog):
    train = _csv(tmp_path, _rows(count=300))
    code, report = _run_search(train, tmp_path / "refused", "--max-candidates", 5)
    assert code == 1 and report is None
    assert "refusing rather than sampling" in caplog.text
    assert _run_search(train, tmp_path / "allowed", "--max-candidates", 2000)[0] == 0


def test_catalog_values_feed_the_tier_template(tmp_path):
    train = _csv(tmp_path, _rows(count=300))
    values = tmp_path / "catalog.json"
    values.write_text(json.dumps({"tile": [64, 128, 256]}), "utf-8")
    code, report = _run_search(train, tmp_path / "out", "--catalog-values", values)
    assert code == 0
    assert report["catalog"]["tier_fields"] == {"tile": [64.0, 128.0, 256.0]}
    assert report["catalog"]["source"] == str(values.resolve())
    assert report["candidates"]["by_template"]["tier"]["generated"] > 0
    assert report["candidates"]["grid_tiles"] == {
        "tiles": [64, 128, 256],
        "source": "catalog",
    }


def test_a_descriptor_tree_feeds_the_tier_template_with_the_kernels_of_one_engine(
    tmp_path,
):
    ued, other = (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "a.ued.json").write_text(
        json.dumps({"version": "1.0", "id": ued, "name": "provider:a"}), "utf-8"
    )
    (tree / "a.kdp.json").write_text(
        json.dumps(kdp("33333333-3333-4333-8333-333333333330", ued, [64, 128])), "utf-8"
    )
    (tree / "b.kdp.json").write_text(
        json.dumps(kdp("44444444-4444-4444-8444-444444444440", other, [32, 256])),
        "utf-8",
    )
    train = _csv(tmp_path, _rows(count=300))
    code, report = _run_search(
        train,
        tmp_path / "one",
        *("--descriptor-tree", tree, "--engine", "provider:a"),
    )
    assert code == 0
    assert report["catalog"]["tier_fields"] == {"tile": [64.0, 128.0]}
    assert report["catalog"]["source"] == str(tree.resolve())
    code, report = _run_search(train, tmp_path / "all", "--descriptor-tree", tree)
    assert code == 0
    assert report["catalog"]["tier_fields"] == {"tile": [32.0, 64.0, 128.0, 256.0]}


def test_two_catalogs_are_refused(tmp_path):
    values = tmp_path / "catalog.json"
    values.write_text("{}", "utf-8")
    code, report = _run_search(
        _csv(tmp_path, _rows(count=100)),
        tmp_path / "out",
        *("--catalog-values", values, "--descriptor-tree", tmp_path),
    )
    assert code == 1 and report is None


def test_the_adapter_and_its_budget_reach_the_search(tmp_path, monkeypatch):
    _noisy_cells(monkeypatch, np.full(300, -1.0), lambda seed, fold: 0.0)
    train = _csv(tmp_path, _rows(count=300))
    code, tree = _run_search(train, tmp_path / "tree")
    assert code == 0 and tree["adapter_pass"] is None
    code, mlp = _run_search(
        train,
        tmp_path / "mlp",
        *("--adapter", "mlp_data", "--adapter-pass-budget", 2),
    )
    assert code == 0
    assert mlp["adapter_pass"]["adapter"] == "mlp_data"
    assert mlp["adapter_pass"]["budget"] == 2
    assert len(mlp["adapter_pass"]["candidates"]) == 2
    code, none = _run_search(
        train,
        tmp_path / "none",
        *("--adapter", "mlp_data", "--adapter-pass-budget", 0),
    )
    assert code == 0 and none["adapter_pass"]["candidates"] == []


def test_a_negative_adapter_pass_budget_is_refused(tmp_path, capsys):
    with pytest.raises(SystemExit):
        _run_search(
            _csv(tmp_path, _rows(count=100)),
            tmp_path / "out",
            *("--adapter", "mlp_data", "--adapter-pass-budget", -1),
        )
    assert "must not be negative" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_an_existing_output_directory_is_refused(tmp_path):
    (tmp_path / "out").mkdir()
    code, report = _run_search(_csv(tmp_path, _rows(count=100)), tmp_path / "out")
    assert code == 1 and report is None
