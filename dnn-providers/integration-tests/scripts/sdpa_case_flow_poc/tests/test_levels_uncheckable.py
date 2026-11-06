# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Uncheckable singles of the level generator, on a small model written to a temporary folder."""

import csv
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdpa_case_flow_poc import check_levels, levels  # noqa: E402

# Axes the size and cost step reads, each with one value.
FIXED = {
    "batch_size": "one",
    "q_heads": "one",
    "seq_q": "one",
    "seq_kv": "one",
    "seq_q_vs_seq_kv": "equal",
    "k_head_sharing": "mha",
    "v_head_sharing": "mha",
    "k_heads_vs_v_heads": "equal",
    "head_dim_qk": "d64",
    "head_dim_v": "d64",
    "dtype_q": "bf16",
    "dtype_o": "bf16",
    "bias": "absent",
}

# axis -> [(value, usage, answer key)]; the first value is the base value.
TEST_AXES = {
    "p": [("p0", "common", "yes"), ("p1", "common", "no"), ("p2", "common", "yes")],
    "q": [("q0", "common", "yes"), ("q1", "common", "no"), ("q2", "common", "yes")],
    "r": [("r0", "common", "yes"), ("r1", "common", "yes"), ("r2", "common", "yes")],
    "s": [("s0", "common", "yes"), ("s1", "common", "no")],
    "t": [("t0", "common", "yes"), ("t1", "fringe", "yes")],
}

RULES = [
    # p1 needs a q other than q0. A completion in model order would take q1, a second "no" value.
    {
        "id": "R-P1",
        "kind": "definition",
        "reason": "test",
        "forbid": {"p": ["p1"], "q": ["q0"]},
    },
    # No valid row holds s1.
    {"id": "R-S1", "kind": "definition", "reason": "test", "forbid": {"s": ["s1"]}},
]

ANSWER_KEY_NEEDS = [
    {
        "id": "AK-T",
        "why": "test",
        "forbid_any": [{"r": ["r1"]}, {"r": ["r2"], "p": ["p2"]}],
    },
]


def axis(aid, values):
    return {
        "id": aid,
        "kind": "categorical",
        "values": [{"id": v} for v, _, _ in values],
        "group": "G1",
        "size_sensitive": ["no", "test"],
        "labels": {v: [usage, "yes", key] for v, usage, key in values},
    }


def write_model(folder: Path) -> Path:
    axes = [axis(a, [(v, "common", "yes")]) for a, v in FIXED.items()]
    axes += [axis(a, vals) for a, vals in TEST_AXES.items()]
    base = dict(FIXED)
    base.update({a: vals[0][0] for a, vals in TEST_AXES.items()})
    level = {
        "budget_min": None,
        "ceiling_mflop": None,
        "shapes": ["S", "H"],
        "changes_numbers": "any",
        "answer_key": ["yes", "partial"],
        "answerable": True,
        "pairs": "all",
        "max_changes": {"start": 2, "escalate": True},
        "bundle_first": True,
        "one_at_a_time": True,
        "targeted": True,
    }
    model = {
        "axes": axes,
        "base_cases": {"shared": {"values": base}},
        "level_labels": {
            "bridges": [],
            "answer_key_needs": ANSWER_KEY_NEEDS,
            "targeted": [
                {"id": "TG-R1", "why": "test", "changes": {"r": "r1", "p": "p2"}}
            ],
            "seeds": [{"id": "SEED-T1", "why": "test", "changes": {"t": "t1"}}],
            "shape_tiers": {
                "S": {"B": 2, "H_q": 8, "S": 64, "S_short": 32},
                "H": {"B": 3, "H_q": 8, "S": 127, "S_short": 65},
            },
            "cost": {
                "basis": "gfx942",
                "model": {"gfx942": {"a": 0.01, "b": 0.0, "c": 0.0}},
                "fitted_range": {"max_mflop": 1e9, "max_tensor_mb": 1e9},
            },
            "over_budget": "report",
        },
        "levels": [
            {**level, "name": "first", "usage": ["common"], "real_world": True},
            {**level, "name": "last", "usage": "any", "uncheckable_singles": "S"},
        ],
    }
    (folder / "model.yaml").write_text(yaml.safe_dump(model, sort_keys=False))
    (folder / "rules.yaml").write_text(yaml.safe_dump({"rules": RULES}))
    (folder / "injections.yaml").write_text("{}\n")
    return folder


def generate(model_dir: Path, out_dir: Path) -> list:
    assert (
        levels.main(
            ["--model-dir", str(model_dir), "--out-dir", str(out_dir), "--seed", "7"]
        )
        == 0
    )
    with open(out_dir / "levels.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def no_values(row):
    return {
        f"{a}={row[a]}"
        for a, vals in TEST_AXES.items()
        for v, _, key in vals
        if key == "no" and row[a] == v
    }


def ak_matches(row):
    return [
        i
        for i, c in enumerate(ANSWER_KEY_NEEDS[0]["forbid_any"])
        if all(row[a] in vs for a, vs in c.items())
    ]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return write_model(tmp_path_factory.mktemp("model"))


@pytest.fixture(scope="module")
def rows(model_dir, tmp_path_factory):
    return generate(model_dir, tmp_path_factory.mktemp("out"))


def singles(rows):
    return [r for r in rows if r["detail"].startswith("uncheckable ")]


def test_one_single_per_target_at_the_tier(rows):
    got = sorted(r["detail"] for r in singles(rows))
    assert got == [
        "uncheckable AK-T#0",
        "uncheckable AK-T#1",
        "uncheckable p=p1",
        "uncheckable q=q1",
    ]
    for r in singles(rows):
        assert (r["level"], r["shape_tier"], r["origin"]) == (
            "last",
            "S",
            "one-at-a-time",
        )


def test_each_single_holds_only_its_own_reason(rows):
    by_target = {r["detail"].split(" ", 1)[1]: r for r in singles(rows)}
    assert no_values(by_target["p=p1"]) == {"p=p1"} and not ak_matches(
        by_target["p=p1"]
    )
    assert by_target["p=p1"]["q"] == "q2"
    assert no_values(by_target["q=q1"]) == {"q=q1"} and not ak_matches(
        by_target["q=q1"]
    )
    for i in (0, 1):
        row = by_target[f"AK-T#{i}"]
        assert i in ak_matches(row) and not no_values(row)


def test_no_uncheckable_row_outside_the_singles(rows):
    for r in rows:
        if r["detail"].startswith("uncheckable ") or r["origin"] in (
            "targeted",
            "real_world",
        ):
            continue
        assert not no_values(r) and not ak_matches(r), r["case_id"]


def test_infeasible_value_has_no_case(rows):
    assert all(r["s"] == "s0" for r in rows)


def test_seed_runs_where_the_answerable_levels_without_singles_put_it(rows):
    seeds = [r for r in rows if "seed:SEED-T1" in r["detail"].split("; ")[-1]]
    assert {(r["level"], r["t"]) for r in seeds} == {("last", "t1")}


def test_targeted_case_an_answer_key_rule_matches_runs_in_the_last_level(rows):
    got = [(r["level"], r["origin"]) for r in rows if r["detail"] == "TG-R1"]
    assert got == [("last", "targeted")]


def test_two_runs_give_the_same_csv(model_dir, rows, tmp_path):
    assert generate(model_dir, tmp_path / "again") == rows


def test_checker_reads_every_target_of_a_merged_single():
    detail = "uncheckable AK-X#3; one-at-a-time:uncheckable AK-X#4; real_world:seed:S1"
    assert check_levels.single_targets({"detail": detail}) == ["AK-X#3", "AK-X#4"]
    assert check_levels.single_targets(
        {"detail": "X3c; one-at-a-time:uncheckable p=p1"}
    ) == ["p=p1"]
    assert check_levels.single_targets({"detail": "p=p1"}) == []


def test_checker_passes_and_fails_on_a_missing_or_extra_single(
    model_dir, rows, tmp_path
):
    def run(table):
        path = tmp_path / "levels.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        return check_levels.main(
            ["--model-dir", str(model_dir), "--only", "uncheckable-singles", str(path)]
        )

    assert run(rows) == 0
    missing = [r for r in rows if r["detail"] != "uncheckable AK-T#1"]
    assert run(missing) == 1
    paired = next(r for r in rows if r["origin"] == "low-change")
    extra = dict(paired, case_id="lv-extra", case_key="extra", r="r1")
    assert run(rows + [extra]) == 1
    own = next(r for r in rows if r["detail"] == "uncheckable p=p1")
    assert run([dict(r, q="q1") if r is own else r for r in rows]) == 1
