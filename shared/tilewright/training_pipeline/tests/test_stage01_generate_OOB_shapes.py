# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import sys

import pytest
import yaml

import stage01_generate_OOB_shapes as s01
from lib import grid, shapes
from lib.bench_yaml import parse_shapes_yaml
from lib.shapes import ShapeFilter

CONFIG = {
    "arch": "gfx1250v0",
    "hipblaslt": {
        "transA": "T",
        "transB": "N",
        "a_type": "f8_r",
        "b_type": "f8_r",
        "c_type": "bf16_r",
        "d_type": "bf16_r",
        "scale_type": "f32_r",
        "compute_type": "f32_r",
        "scaleA": 3,
        "scaleB": 3,
    },
    "bench": {"device_memory_gib": 64},
    "seed": {"exclude": [{"max_mn": 8}, {"min_k": 65536}]},
}
TINY_BOX = "Tiny|Tiny|TinyK|Bnone#M<=2#N<=2#K<=2"


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(CONFIG))
    return path


def _filter():
    return ShapeFilter.from_config(CONFIG)


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["stage01", *map(str, argv)])
    return s01.main()


def test_derive_seed_is_stable_and_round_specific():
    assert s01.derive_seed(42, 0) == s01.derive_seed(42, 0)
    assert s01.derive_seed(42, 0) != s01.derive_seed(42, 1)
    assert s01.derive_seed(42, 1) != s01.derive_seed(43, 1)
    assert 0 <= s01.derive_seed("x") < 2**64


def test_sampling_is_deterministic_and_avoids_history():
    plan = {"Mid|Mid|MidK|Bnone": 30, "Small|Large|LargeK|Bany": 20}
    first = s01.sample_shapes(plan, 1, set(), _filter())
    assert first.shapes == s01.sample_shapes(plan, 1, set(), _filter()).shapes
    assert first.actual_per_cell == plan
    second = s01.sample_shapes(plan, 2, set(first.shapes), _filter())
    assert not set(second.shapes) & set(first.shapes)
    assert second.reused == []
    assert second.actual_per_cell == plan


def test_a_cells_shapes_do_not_depend_on_other_cells():
    alone = s01.sample_shapes({"Mid|Mid|MidK|Bnone": 25}, 5, set(), _filter())
    both = s01.sample_shapes(
        {"Mid|Mid|MidK|Bnone": 25, "Large|Mid|LargeK|Bnone": 40}, 5, set(), _filter()
    )
    in_cell = [s for s in both.shapes if grid.cell_key(*s) == "Mid|Mid|MidK|Bnone"]
    assert sorted(in_cell) == sorted(alone.shapes)


def test_top_up_reuses_history_shapes_and_reports_them():
    box = {(m, n, k, 1) for m in (1, 2) for n in (1, 2) for k in (1, 2)}
    loose = ShapeFilter(memory_budget_bytes=64 * 1024**3)
    res = s01.sample_shapes({TINY_BOX: 5}, 3, box, loose, min_per_cell=4)
    assert res.actual_per_cell[TINY_BOX] == 4
    assert len(res.reused) == 4 and set(res.reused) <= box
    assert res.reused_per_cell[TINY_BOX] == 4
    none = s01.sample_shapes({TINY_BOX: 5}, 3, box, loose)
    assert none.shapes == [] and none.n_history_dropped > 0


def test_initial_round_end_to_end(tmp_path, monkeypatch, config_path):
    out = tmp_path / "round_0" / "stage01"
    args = ["--mode", "initial", "--out-dir", out, "--config-yaml", config_path]
    args += ["--target-per-cell", 3, "--n-strata", 2, "--quiet"]
    assert _run(monkeypatch, *args) == 0
    cat = json.loads((out / "categorization.json").read_text())
    shapes_list = parse_shapes_yaml(out / "shapes.yaml")
    assert len(shapes_list) == cat["n_final"] > 150
    assert cat["round_index"] == 0
    assert cat["memory_budget"]["source"] == "config"
    assert cat["exclude"] == [{"max_mn": 8}, {"min_k": 65536}]
    f = _filter()
    for m, n, k, b in shapes_list:
        assert f.accepts(m, n, k, b)
        assert min(m, n) > 8 and k < 65536 and k % 32 == 0
    assert any(min(m, n) <= 32 for m, n, _, _ in shapes_list)
    assert any(k > 8192 for _, _, k, _ in shapes_list)
    assert shapes.read_reused_shapes(out) == set()
    first = (out / "shapes.yaml").read_text()
    assert _run(monkeypatch, *args) == 0
    assert (out / "shapes.yaml").read_text() == first
    other = tmp_path / "other"
    assert _run(monkeypatch, *args[:3], other, *args[4:], "--round-index", 1) == 0
    overlap = set(parse_shapes_yaml(other / "shapes.yaml")) & set(shapes_list)
    assert len(overlap) < len(shapes_list) // 4


def test_active_round_end_to_end(tmp_path, monkeypatch, config_path):
    prior = tmp_path / "round_0"
    args = ["--mode", "initial", "--out-dir", prior / "stage01"]
    args += ["--config-yaml", config_path, "--target-per-cell", 4, "--quiet"]
    assert _run(monkeypatch, *args) == 0
    (prior / "stage05").mkdir()
    planned = ["Mid|Mid|MidK|Bnone", "Large|Small|LargeK|Bnone", TINY_BOX.split("#")[0]]
    (prior / "stage05" / "cells.json").write_text(
        json.dumps({"cells": [{"label": c, "best_sel_eff": 0.9} for c in planned]})
    )
    out = tmp_path / "round_1" / "stage01"
    args = ["--mode", "active", "--out-dir", out, "--config-yaml", config_path]
    args += ["--prior-round-dir", prior, "--shapes-per-cell", 6, "--quiet"]
    assert _run(monkeypatch, *args) == 0
    cat = json.loads((out / "categorization.json").read_text())
    assert cat["round_index"] == 1
    new = parse_shapes_yaml(out / "shapes.yaml")
    old = set(parse_shapes_yaml(prior / "stage01" / "shapes.yaml"))
    assert not set(new) & old
    assert {grid.cell_key(*s) for s in new} <= set(planned)
    plan = json.loads((out / "per_cell_plan.json").read_text())
    assert plan["cells"]["Mid|Mid|MidK|Bnone"]["n_actual"] == 6
    assert shapes.read_reused_shapes(out) == set()


def test_active_plan_covers_carried_and_fallback_cells(tmp_path):
    from lib import subcells as sc

    parent, carried = "Large|Large|LargeK|Bnone", "Mid|Mid|MidK|Bnone"
    rule = sc.build_split_rule(parent, "M", 4096)
    tree = {rule.cell: rule}
    cj = tmp_path / "cells.json"
    cj.write_text(
        json.dumps(
            {
                "model_labels": [parent, rule.lo_label, carried],
                "cells": [{"label": rule.lo_label, "best_sel_eff": 0.9}],
            }
        )
    )
    plan = s01._build_active_plan(cj, 7, tree)
    assert plan[carried] == {
        "sel_eff_prev_round": None,
        "n_target": 7,
        "has_prior_model": True,
        "parent_cell": None,
    }
    assert plan[rule.lo_label]["sel_eff_prev_round"] == 0.9
    assert plan[rule.lo_label]["parent_cell"] is None
    assert plan[rule.hi_label] == {
        "sel_eff_prev_round": None,
        "n_target": 7,
        "has_prior_model": True,
        "parent_cell": parent,
    }
    assert parent not in plan
    assert plan["Tiny|Tiny|TinyK|Bnone"]["n_target"] == 0
    cj.write_text(json.dumps({"cells": [{"label": rule.lo_label}]}))
    plan = s01._build_active_plan(cj, 7, tree)
    assert plan[carried]["n_target"] == 0 and rule.hi_label not in plan


def test_bad_inputs_are_rejected(tmp_path, monkeypatch, config_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({**CONFIG, "seed": {"exclude": [{"max_q": 1}]}}))
    args = ["--mode", "initial", "--out-dir", tmp_path / "o", "--quiet"]
    assert _run(monkeypatch, *args, "--config-yaml", bad) == 1
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps({"cells": [{"label": "Mid|Mid|MidK|Bnone"}]}))
    args = ["--mode", "active", "--out-dir", tmp_path / "a", "--config-yaml"]
    args += [config_path, "--prior-cells", cells, "--quiet"]
    assert _run(monkeypatch, *args) == 1
    assert _run(monkeypatch, *args, "--round-index", 2) == 0
    cat = json.loads((tmp_path / "a" / "categorization.json").read_text())
    assert cat["round_index"] == 2
