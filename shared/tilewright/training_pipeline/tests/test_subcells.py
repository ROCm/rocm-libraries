# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import random

import pytest

from lib import grid, mlrec
from lib import subcells as sc
from lib.hardware import arch_constants

from test_mlrec import random_bundle

TREE_LABELS = [
    "Large|Large|LargeK|Bnone#M<=4096#K<=2048",
    "Large|Large|LargeK|Bnone#M<=4096#K>2048",
    "Large|Large|LargeK|Bnone#M>4096",
    "Large|Mid|LargeK|Bany#N<=300",
    "Mid|Large|MidK|Bnone#N<=1000#N<=700",
    "Mid|Large|MidK|Bnone",
    "Small|Small|TinyK|Bnone",
]


def test_split_tree_from_labels():
    tree = sc.split_tree_from_labels(TREE_LABELS)
    assert set(tree) == {
        "Large|Large|LargeK|Bnone",
        "Large|Large|LargeK|Bnone#M<=4096",
        "Large|Mid|LargeK|Bany",
        "Mid|Large|MidK|Bnone",
        "Mid|Large|MidK|Bnone#N<=1000",
    }
    r = tree["Large|Large|LargeK|Bnone#M<=4096"]
    assert (r.axis, r.threshold) == ("K", 2048)
    assert r.lo_label == "Large|Large|LargeK|Bnone#M<=4096#K<=2048"
    assert r.hi_label == "Large|Large|LargeK|Bnone#M<=4096#K>2048"
    assert sc.split_tree_from_labels(["Tiny|Tiny|TinyK|Bnone"]) == {}


def test_split_tree_from_labels_rejects_bad_labels():
    with pytest.raises(ValueError):
        sc.split_tree_from_labels(["A#M<=10", "A#N>3"])
    with pytest.raises(ValueError):
        sc.split_tree_from_labels(["A#Q<=10"])
    with pytest.raises(ValueError):
        sc.split_tree_from_labels(["A#M<10"])


def test_routing_and_ancestor_fallback():
    tree = sc.split_tree_from_labels(TREE_LABELS)
    models = set(TREE_LABELS)
    assert sc.route(5000, 5000, 4096, 1, tree, models) == (
        "Large|Large|LargeK|Bnone#M>4096",
        "Large|Large|LargeK|Bnone#M>4096",
    )
    assert sc.route(4096, 600, 2048, 1, tree, models)[0] == (
        "Large|Large|LargeK|Bnone#M<=4096#K<=2048"
    )
    leaf, cell = sc.route(400, 800, 100, 1, tree, models)
    assert leaf == "Mid|Large|MidK|Bnone#N<=1000#N>700"
    assert cell == "Mid|Large|MidK|Bnone"
    assert sc.route(400, 1200, 100, 1, tree, models)[1] == "Mid|Large|MidK|Bnone"
    assert sc.route(1000, 400, 1000, 3, tree, models) == (
        "Large|Mid|LargeK|Bany#N>300",
        None,
    )
    assert sc.resolve_model_cell("X#M<=1#N>2", {"X"}) == "X"
    assert sc.resolve_model_cell("X#M<=1", set()) is None


def test_routing_matches_engine():
    tw = pytest.importorskip("tilewright")
    from lib import evaluate as ev

    data = mlrec.write_model(
        random_bundle(TREE_LABELS), None, "gfx950", arch_constants("gfx950"), "fp32"
    )
    model = tw.load_model_from_memory(data)
    tree = sc.split_tree_from_labels(TREE_LABELS)
    edges = [1, 31, 32, 33, 127, 128, 129, 300, 301, 511, 512, 513]
    edges += [699, 700, 701, 1000, 1001, 2047, 2048, 2049, 4095, 4096, 4097, 9000]
    rng = random.Random(5)
    for _ in range(3000):
        m, n, k = rng.choice(edges), rng.choice(edges), rng.choice(edges)
        b = rng.choice([1, 1, 2, 7])
        p = dict(
            m=m,
            n=n,
            k=k,
            batch=b,
            a_transpose="T",
            b_transpose="N",
            a_dtype="bfloat16",
            b_dtype="bfloat16",
            c_dtype="bfloat16",
            d_dtype="bfloat16",
            mi_dtype="bfloat16",
        )
        expect = sc.route(m, n, k, b, tree, set(TREE_LABELS))[1]
        assert ev.model_cell_label(tw, model, ev.make_problem(tw, p)) == expect, p


def gemms_for(values, axis="M"):
    out = []
    for v in values:
        g = {"m": 600, "n": 600, "k": 600, "batch_count": 1}
        g[{"M": "m", "N": "n", "K": "k"}[axis]] = v
        out.append(g)
    return out


def test_choose_split_for_cell():
    label = "Large|Mid|MidK|Bnone"
    vals = [600 + 37 * i for i in range(30)] + [20000 + 997 * i for i in range(30)]
    ratios = [0.0] * 30 + [0.5] * 30
    axis, thr = sc.choose_split_for_cell(
        label, gemms_for(vals), ratios, min_split_count=10
    )
    assert axis == "M" and 600 + 37 * 29 <= thr < 20000
    assert (
        sc.choose_split_for_cell(label, gemms_for(vals), ratios, min_split_count=31)
        is None
    )
    assert (
        sc.choose_split_for_cell("Mid|Mid|MidK|Bnone", gemms_for(vals), ratios) is None
    )
    assert sc.split_eligible_axes("Large|Large|LargeK|Bany", 6) == ["M", "N", "K"]
    assert sc.split_eligible_axes("Large|Mid|MidK|Bnone#M<=1#M<=0", 2) == []
    assert sc.split_eligible_axes("Small|Mid|TinyK|Bnone", 6) == []


def test_split_json_round_trip(tmp_path):
    rule = sc.build_split_rule("Large|Large|LargeK|Bnone", "K", 4096, reason="r")
    sc.write_splits_json(tmp_path / "round_1" / "stage04b" / "splits.json", 1, [rule])
    tree = sc.load_cumulative_split_tree([tmp_path / "round_0", tmp_path / "round_1"])
    assert tree == {rule.cell: rule}
    assert set(sc.leaves_under(rule.cell, tree)) == {rule.lo_label, rule.hi_label}
    assert grid.cell_key(4097, 4097, 4097, 1) == rule.cell
