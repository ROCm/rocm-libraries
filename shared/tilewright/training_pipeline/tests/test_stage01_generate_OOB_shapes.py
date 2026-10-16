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
    "arch": "gfx1250-strict",
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


def test_a_round_on_a_repaired_final_retrain_samples_its_leaves(
    tmp_path, monkeypatch, config_path
):
    """round_0 split a cell, and the final retrain the round builds on
    carried that cell from a bundle without the split and revoked it: with
    the final retrain as --base-bundle-dir the round samples the cell whole
    (the final retrain's labels, the split left out of the tree), not its
    two halves."""
    from lib import subcells as sc

    parent, other = "Large|Large|LargeK|Bnone", "Mid|Mid|MidK|Bnone"
    rule = sc.build_split_rule(parent, "M", 4096)
    prior = tmp_path / "round_0"
    args = ["--mode", "initial", "--out-dir", prior / "stage01"]
    args += ["--config-yaml", config_path, "--target-per-cell", 4, "--quiet"]
    assert _run(monkeypatch, *args) == 0
    sc.write_splits_json(prior / "stage04b" / "splits.json", 0, [rule])
    halves = [rule.lo_label, rule.hi_label]
    (prior / "stage05").mkdir()
    (prior / "stage05" / "cells.json").write_text(
        json.dumps({"cells": [{"label": c} for c in [other, *halves]]})
    )
    final = tmp_path / "final_retrain"
    (final / "stage05").mkdir(parents=True)
    (final / "stage05" / "cells.json").write_text(
        json.dumps({"cells": [{"label": c} for c in [other, parent]]})
    )
    sc.write_revocations(final / "stage05" / sc.REVOKED_SPLITS, [rule], [parent])
    base = ["--mode", "active", "--config-yaml", config_path, "--quiet"]
    base += ["--prior-round-dir", prior, "--shapes-per-cell", 6]

    def plan(name, *more):
        out = tmp_path / name / "stage01"
        assert _run(monkeypatch, *base, "--out-dir", out, *more) == 0
        doc = json.loads((out / "per_cell_plan.json").read_text())
        planned = {c for c, v in doc["cells"].items() if v["n_target"]}
        return doc, planned, parse_shapes_yaml(out / "shapes.yaml")

    doc, planned, _ = plan("without")
    assert planned == {other, *halves}
    assert doc["base_bundle_dir"] is None and doc["revoked_by_base_bundle"] == []
    doc, planned, drawn = plan("with", "--base-bundle-dir", final)
    assert planned == {other, parent}
    assert doc["prior_cells_json"] == str(final / "stage05" / "cells.json")
    assert doc["base_bundle_dir"] == str(final)
    assert doc["revoked_by_base_bundle"] == [parent]
    assert doc["cells"][parent]["n_actual"] == 6
    assert {grid.cell_key(*s) for s in drawn} == {other, parent}
    # a base bundle without cells.json, or with a record that is no
    # revocation record, stops the round
    (final / "stage05" / sc.REVOKED_SPLITS).write_text("{}")
    out = tmp_path / "bad" / "stage01"
    assert _run(monkeypatch, *base, "--out-dir", out, "--base-bundle-dir", final) == 1
    (final / "stage05" / "cells.json").unlink()
    assert _run(monkeypatch, *base, "--out-dir", out, "--base-bundle-dir", final) == 1


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


def test_only_cells_restricts_the_plan_to_the_requested_leaves():
    from lib import subcells as sc

    parent, other = "Large|Large|LargeK|Bnone", "Mid|Mid|MidK|Bnone"
    rule = sc.build_split_rule(parent, "M", 4096)
    tree = {rule.cell: rule}
    plan = {
        rule.lo_label: {"n_target": 7, "has_prior_model": True},
        rule.hi_label: {"n_target": 7, "has_prior_model": True},
        other: {"n_target": 7, "has_prior_model": True},
    }
    fresh = "Small|Small|MidK|Bnone"
    out, unknown = s01.restrict_plan(plan, [parent, fresh], 11, tree)
    assert unknown == []
    assert out[other]["n_target"] == 0 and not out[other]["requested"]
    for leaf in (rule.lo_label, rule.hi_label, fresh):
        assert out[leaf]["n_target"] == 11 and out[leaf]["requested"]
    assert out[fresh]["has_prior_model"] is False
    assert parent not in out
    _, unknown = s01.restrict_plan(plan, ["Nope|Mid|MidK|Bnone"], 11, tree)
    assert unknown == ["Nope|Mid|MidK|Bnone"]


def test_only_cells_end_to_end(tmp_path, monkeypatch, config_path):
    prior = tmp_path / "round_0"
    args = ["--mode", "initial", "--out-dir", prior / "stage01"]
    args += ["--config-yaml", config_path, "--target-per-cell", 4, "--quiet"]
    assert _run(monkeypatch, *args) == 0
    (prior / "stage05").mkdir()
    planned = ["Mid|Mid|MidK|Bnone", "Large|Small|LargeK|Bnone"]
    (prior / "stage05" / "cells.json").write_text(
        json.dumps({"cells": [{"label": c, "best_sel_eff": 0.9} for c in planned]})
    )
    out = tmp_path / "round_1" / "stage01"
    args = ["--mode", "active", "--out-dir", out, "--config-yaml", config_path]
    args += ["--prior-round-dir", prior, "--shapes-per-cell", 6, "--quiet"]
    wanted = "Mid|Mid|MidK|Bnone,Small|Mid|MidK|Bnone"
    assert _run(monkeypatch, *args, "--only-cells", wanted) == 0
    new = parse_shapes_yaml(out / "shapes.yaml")
    assert {grid.cell_key(*s) for s in new} == set(wanted.split(","))
    plan = json.loads((out / "per_cell_plan.json").read_text())
    assert plan["only_cells"] == wanted
    assert plan["cells"]["Large|Small|LargeK|Bnone"]["n_target"] == 0
    assert _run(monkeypatch, *args, "--only-cells", "Nope|Mid|MidK|Bnone") == 1
    assert _run(monkeypatch, *args, "--only-cells", ",") == 1


def _tail(m, n, k, b, se, cu=0):
    return {"m": m, "n": n, "k": k, "batch": b, "cu_budget": cu,
            "schedule": "default", "sel_eff": se}  # fmt: skip


def test_tail_gemms_of_the_previous_round_get_shapes_near_them(
    tmp_path, monkeypatch, config_path
):
    prior = tmp_path / "round_0"
    args = ["--mode", "initial", "--out-dir", prior / "stage01"]
    args += ["--config-yaml", config_path, "--target-per-cell", 4, "--quiet"]
    assert _run(monkeypatch, *args) == 0
    (prior / "stage05").mkdir()
    planned = ["Mid|Mid|MidK|Bnone", "Large|Small|LargeK|Bnone"]
    (prior / "stage05" / "cells.json").write_text(
        json.dumps({"cells": [{"label": c, "best_sel_eff": 0.97} for c in planned]})
    )
    tails = [_tail(300, 300, 288, 1, 0.42, cu=128), _tail(260, 200, 96, 1, 0.6)]
    unplanned = _tail(4000, 4000, 4000, 1, 0.3)
    (prior / "stage04b").mkdir()
    (prior / "stage04b" / "decisions.json").write_text(
        json.dumps(
            {
                "per_cell": [
                    {"cell": planned[0], "decision": "pass", "tail_gemms": tails},
                    {"cell": "x", "decision": "pass", "tail_gemms": [unplanned]},
                ]
            }
        )
    )
    audit = tmp_path / "audit.yaml"
    out = tmp_path / "round_1" / "stage01"
    base = ["--mode", "active", "--out-dir", out, "--config-yaml", config_path]
    base += ["--prior-round-dir", prior, "--shapes-per-cell", 6, "--quiet"]
    assert _run(monkeypatch, *base, "--targeted-per-gemm", 3) == 0
    doc = json.loads((out / s01.TARGETED_FILE).read_text())
    (entries,) = doc["leaves"].values()
    assert [e["tail"]["sel_eff"] for e in entries] == [0.42, 0.6]
    near = [tuple(s) for e in entries for s in e["shapes"]]
    assert len(near) == 6 and doc["n_shapes"] == 6
    written = parse_shapes_yaml(out / "shapes.yaml")
    old = set(parse_shapes_yaml(prior / "stage01" / "shapes.yaml"))
    assert set(near) <= set(written) and not set(near) & old
    assert len(written) == len(set(written))
    for m, n, k, b in near:
        assert grid.cell_key(m, n, k, b) == planned[0] and b == 1
    cat = json.loads((out / "categorization.json").read_text())
    assert cat["targeted"]["n_tail_gemms"] == 3 and cat["targeted"]["n_shapes"] == 6
    assert cat["n_final"] == len(written)
    plan = json.loads((out / "per_cell_plan.json").read_text())
    assert plan["cells"][planned[0]]["n_targeted"] == 6
    # the cap per leaf, and never a shape of an excluded (audit) list
    first = near[0]
    audit.write_text(
        "- {rotating_buffer: 512, transA: T, transB: N, "
        f"M: {first[0]}, N: {first[1]}, K: {first[2]}, batch_count: 1}}\n"
    )
    capped = tmp_path / "capped"
    capped_args = [*base[:3], capped, *base[4:], "--targeted-max-per-leaf", 4]
    assert _run(monkeypatch, *capped_args, "--exclude-shapes-yaml", audit) == 0
    doc = json.loads((capped / s01.TARGETED_FILE).read_text())
    got = [tuple(s) for es in doc["leaves"].values() for e in es for s in e["shapes"]]
    assert len(got) == 4 and first not in got
    assert first not in parse_shapes_yaml(capped / "shapes.yaml")
    # off, or nothing listed: no targeted shapes and no file
    off = tmp_path / "off"
    assert _run(monkeypatch, *base[:3], off, *base[4:], "--targeted-per-gemm", 0) == 0
    assert not (off / s01.TARGETED_FILE).exists()
    assert (
        json.loads((off / "categorization.json").read_text())["targeted"]["n_shapes"]
        == 0
    )
    (prior / "stage04b" / "decisions.json").unlink()
    none = tmp_path / "none"
    assert _run(monkeypatch, *base[:3], none, *base[4:]) == 0
    assert not (none / s01.TARGETED_FILE).exists()


def test_a_held_out_targeted_shape_stops_the_round(
    tmp_path, monkeypatch, config_path, capsys
):
    """The targeted shapes are checked against the held-out lists as the
    round's draws are: one that is held out stops the round before it
    writes anything."""
    prior = tmp_path / "round_0"
    args = ["--mode", "initial", "--out-dir", prior / "stage01"]
    args += ["--config-yaml", config_path, "--target-per-cell", 4, "--quiet"]
    assert _run(monkeypatch, *args) == 0
    cell = "Mid|Mid|MidK|Bnone"
    (prior / "stage05").mkdir()
    (prior / "stage05" / "cells.json").write_text(
        json.dumps({"cells": [{"label": cell, "best_sel_eff": 0.97}]})
    )
    (prior / "stage04b").mkdir()
    (prior / "stage04b" / "decisions.json").write_text(
        json.dumps(
            {
                "per_cell": [
                    {
                        "cell": cell,
                        "decision": "pass",
                        "tail_gemms": [_tail(300, 300, 288, 1, 0.42)],
                    }
                ]
            }
        )
    )
    held = (320, 320, 256, 1)
    audit = tmp_path / "audit.yaml"
    audit.write_text(
        "- {rotating_buffer: 512, transA: T, transB: N, "
        f"M: {held[0]}, N: {held[1]}, K: {held[2]}, batch_count: 1}}\n"
    )
    out = tmp_path / "round_1" / "stage01"
    base = ["--mode", "active", "--out-dir", out, "--config-yaml", config_path]
    base += ["--prior-round-dir", prior, "--shapes-per-cell", 6, "--quiet"]
    base += ["--targeted-per-gemm", 3, "--exclude-shapes-yaml", audit]
    assert _run(monkeypatch, *base) == 0
    monkeypatch.setattr(
        s01,
        "targeted_shapes",
        lambda *a, **kw: {cell: [{"tail": {}, "shapes": [list(held)]}]},
    )
    for name in ("shapes.yaml", s01.TARGETED_FILE):
        (out / name).unlink(missing_ok=True)
    capsys.readouterr()
    assert _run(monkeypatch, *base) == 1
    assert "1 targeted shapes are held out" in capsys.readouterr().err
    assert not (out / "shapes.yaml").exists()
    assert not (out / s01.TARGETED_FILE).exists()


def test_a_tail_gemm_goes_to_the_leaf_of_this_rounds_tree(config_path):
    from lib import subcells as sc

    parent = "Large|Small|LargeK|Bnone"
    rule = sc.build_split_rule(parent, "M", 4096)
    f = _filter()
    tails = [_tail(9000, 100, 1000, 1, 0.3), _tail(2000, 100, 1000, 1, 0.4)]
    out = s01.targeted_shapes(
        tails, {rule.hi_label}, {rule.cell: rule}, 7, f,
        per_gemm=2, max_per_leaf=8, taken=set(),
    )  # fmt: skip
    (entries,) = out.values()
    assert list(out) == [rule.hi_label] and len(entries) == 1
    assert all(m > 4096 for m, _, _, _ in entries[0]["shapes"])
    assert s01.leaf_box(rule.hi_label)[0] == (4097, grid.TIER_RANGES_MN["Large"][1])
