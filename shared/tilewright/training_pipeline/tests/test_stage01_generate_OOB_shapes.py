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
    # a leaf's new shapes, which the very-large-K stratum counts, are its
    # fresh draws: re-used history shapes are none of them
    assert res.fresh_per_cell[TINY_BOX] == []
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


# ── weak-cell budget ─────────────────────────────────────────────────────────


def _row(cell, decision="pass", se=0.97, n_eval=40, **kw):
    return dict(cell=cell, decision=decision, sel_eff_new=se, n_eval=n_eval, **kw)


def _plan(*cells, spc=10):
    return {c: {"n_target": spc, "has_prior_model": True} for c in cells}


def _decisions(*rows, threshold=0.95):
    return {"sel_eff_threshold": threshold, "per_cell": list(rows)}


def test_weak_leaves_get_the_factor_and_the_others_keep_the_round():
    """A leaf stage04b retrained, for its mean or for a held-out collapse,
    gets the factor; a pass, a row that split and a leaf without a row
    keep shapes_per_cell; a leaf without a model stays at none."""
    plan = _plan("A", "B", "C", "D", "E")
    plan["F"] = {"n_target": 0, "has_prior_model": False}
    rec = s01.need_budget(
        plan,
        _decisions(
            _row("A", "retrain", 0.91),
            _row("B", "pass", 0.99, collapse_retrain=True),
            _row("C", "pass", 0.99),
            _row("D", "split", 0.90),
            _row("F", "retrain", 0.5),
        ),
        10,
        weak_factor=2.0,
    )
    assert {c: v["n_target"] for c, v in plan.items()} == {
        "A": 20, "B": 20, "C": 10, "D": 10, "E": 10, "F": 0,
    }  # fmt: skip
    assert {c: v["need"] for c, v in plan.items()} == {
        "A": "weak", "B": "weak", "C": "strong", "D": None, "E": None, "F": None,
    }  # fmt: skip
    assert rec["applied"] and rec["n_weak"] == 2 and rec["n_strong"] == 1
    assert (rec["uniform_total"], rec["weak_extra"], rec["total_target"]) == (
        50,
        20,
        70,
    )
    assert rec["strong_cut_wanted"] == rec["strong_cut"] == 0
    assert rec["sel_eff_threshold"] == 0.95
    # rounded half up
    half = _plan("A", spc=5)
    s01.need_budget(half, _decisions(_row("A", "retrain")), 5, weak_factor=2.5)
    assert half["A"]["n_target"] == 13


def test_a_leaf_reads_the_rows_under_it_and_none_above_it():
    """The leaves of a split the decisions' round made (the parent's row
    only) read none; a leaf whose split a repair revoked reads its former
    children's rows: weak when one was retrained, strong only when all
    passed far above the bar."""
    parent = "Large|Large|LargeK|Bnone"
    plan = _plan(parent + "#M<=4096", parent + "#M>4096", "Mid|Mid|MidK|Bnone")
    s01.need_budget(
        plan,
        _decisions(
            _row(parent, "split", 0.9), _row("Mid|Mid|MidK|Bnone#K<=99", "retrain")
        ),
        10,
        weak_factor=3.0,
    )
    assert [plan[c]["n_target"] for c in sorted(plan)] == [10, 10, 30]
    # whatever the row above them says
    s01.need_budget(plan, _decisions(_row(parent, "retrain", 0.9)), 10, weak_factor=3.0)
    assert [plan[c]["need"] for c in sorted(plan)] == [None, None, None]
    strong = _plan(parent)
    rows = [_row(parent + "#M<=4096", se=0.99), _row(parent + "#M>4096", se=0.99)]
    s01.need_budget(strong, _decisions(*rows), 10)
    assert strong[parent]["need"] == "strong"
    rows[1]["sel_eff_new"] = 0.96
    s01.need_budget(strong, _decisions(*rows), 10)
    assert strong[parent]["need"] is None
    # one retrained former leaf makes it weak, whatever the others read
    rows[1].update(decision="retrain", sel_eff_new=0.9)
    s01.need_budget(strong, _decisions(*rows), 10, weak_factor=2.0)
    assert strong[parent]["need"] == "weak" and strong[parent]["n_target"] == 20
    # a leaf's rows are its own and those under it, not a label it prefixes
    near = _plan(parent + "#M<=409")
    s01.need_budget(near, _decisions(*rows), 10, weak_factor=2.0)
    assert near[parent + "#M<=409"]["need"] is None


def test_strong_needs_a_fresh_reading_far_above_the_threshold_stage04b_applied():
    plan = _plan("A", "B", "C", "D", "E")
    rows = [
        _row("A", se=0.98),
        _row("B", se=0.979),
        _row("C", se=0.99, n_eval=0, reason="carried_forward_no_new_data"),
        _row("D", se=float("nan")),
        _row("E", se=1.0),
    ]
    s01.need_budget(plan, _decisions(*rows), 10)
    assert [plan[c]["need"] for c in "ABCDE"] == ["strong", None, None, None, "strong"]
    s01.need_budget(plan, _decisions(*rows, threshold=0.97), 10, strong_margin=0.0)
    assert [plan[c]["need"] for c in "ABCDE"] == [
        "strong",
        "strong",
        None,
        None,
        "strong",
    ]
    s01.need_budget(plan, {"per_cell": rows}, 10)
    assert all(plan[c]["need"] is None for c in "ABCDE")
    # a threshold that is no number is none
    rec = s01.need_budget(
        plan, {"sel_eff_threshold": True, "per_cell": rows}, 10, strong_margin=0.0
    )
    assert rec["sel_eff_threshold"] is None
    assert all(plan[c]["need"] is None for c in "ABCDE")


def test_strong_leaves_give_up_at_most_what_the_weak_leaves_gain():
    """The round's target never falls below shapes_per_cell per leaf: the
    strong leaves' cut pays for the weak leaves' extra and no more, shared
    in proportion, a remainder shape at a time from the highest reading."""
    rows = [_row("W", "retrain", 0.9)] + [
        _row(c, se=se) for c, se in (("S1", 0.99), ("S2", 0.995), ("S3", 0.985))
    ]

    def run(weak, strong, min_per_cell=0):
        plan = _plan("W", "S1", "S2", "S3", "U")
        rec = s01.need_budget(
            plan,
            _decisions(*rows, _row("U", se=0.96)),
            10,
            weak_factor=weak,
            strong_factor=strong,
            min_per_cell=min_per_cell,
        )
        targets = {c: v["n_target"] for c, v in plan.items()}
        assert rec["total_target"] == sum(targets.values())
        assert rec["uniform_total"] <= rec["total_target"]
        assert rec["total_target"] <= rec["uniform_total"] + rec["weak_extra"]
        return targets, rec

    targets, rec = run(1.5, 0.5)
    assert targets == {"W": 15, "S1": 8, "S2": 8, "S3": 9, "U": 10}
    assert (rec["weak_extra"], rec["strong_cut_wanted"], rec["strong_cut"]) == (
        5,
        15,
        5,
    )
    assert rec["total_target"] == rec["uniform_total"] == 50
    targets, rec = run(3.0, 0.5)
    assert targets == {"W": 30, "S1": 5, "S2": 5, "S3": 5, "U": 10}
    assert rec["total_target"] == 55 == rec["uniform_total"] + 20 - 15
    targets, rec = run(1.0, 0.5)
    assert set(targets.values()) == {10} and rec["strong_cut"] == 0
    targets, _ = run(3.0, 0.5, min_per_cell=8)
    assert targets == {"W": 30, "S1": 8, "S2": 8, "S3": 8, "U": 10}
    targets, rec = run(3.0, 0.5, min_per_cell=12)
    assert targets == {"W": 30, "S1": 10, "S2": 10, "S3": 10, "U": 10}
    assert rec["strong_cut_wanted"] == rec["strong_cut"] == 0


def test_every_factor_at_one_keeps_every_leaf_at_shapes_per_cell():
    plan = _plan("A", "B", "C", spc=7)
    rows = [_row("A", "retrain"), _row("B", se=0.999), _row("C", collapse_retrain=True)]
    rec = s01.need_budget(plan, _decisions(*rows), 7)
    assert {v["n_target"] for v in plan.values()} == {7}
    assert rec["total_target"] == rec["uniform_total"] == 21
    none = _plan("A", spc=7)
    rec = s01.need_budget(none, None, 7, weak_factor=2.0)
    assert none["A"] == {"n_target": 7, "has_prior_model": True, "need": None}
    assert not rec["applied"] and rec["total_target"] == 7


def _active_round(tmp_path, monkeypatch, config_path, decisions, planned):
    prior = tmp_path / "round_0"
    args = ["--mode", "initial", "--out-dir", prior / "stage01"]
    args += ["--config-yaml", config_path, "--target-per-cell", 4, "--quiet"]
    assert _run(monkeypatch, *args) == 0
    (prior / "stage05").mkdir()
    (prior / "stage05" / "cells.json").write_text(
        json.dumps({"cells": [{"label": c, "best_sel_eff": 0.9} for c in planned]})
    )
    if decisions is not None:
        (prior / "stage04b").mkdir()
        (prior / "stage04b" / "decisions.json").write_text(json.dumps(decisions))
    base = ["--mode", "active", "--config-yaml", config_path, "--quiet"]
    base += ["--prior-round-dir", prior, "--shapes-per-cell", 6]

    def run(name, *more, rc=0):
        out = tmp_path / name / "stage01"
        assert _run(monkeypatch, *base, "--out-dir", out, *more) == rc
        return out

    return run


def _by_cell(path):
    counts = {}
    for s in parse_shapes_yaml(path / "shapes.yaml"):
        counts[grid.cell_key(*s)] = counts.get(grid.cell_key(*s), 0) + 1
    return counts


def test_a_round_draws_the_factor_for_the_leaves_the_last_round_retrained(
    tmp_path, monkeypatch, config_path
):
    weak, strong, kept = (
        "Mid|Mid|MidK|Bnone",
        "Small|Mid|MidK|Bnone",
        "Mid|Small|MidK|Bnone",
    )
    decisions = _decisions(
        _row(weak, "retrain", 0.9), _row(strong, se=0.99), _row(kept, se=0.96)
    )
    run = _active_round(
        tmp_path, monkeypatch, config_path, decisions, [weak, strong, kept]
    )
    out = run("weak", "--weak-cell-factor", 2.5, "--strong-cell-factor", 0.5)
    assert _by_cell(out) == {weak: 15, strong: 3, kept: 6}
    plan = json.loads((out / "per_cell_plan.json").read_text())
    budget = plan["budget"]
    assert budget["applied"] and budget["not_applied"] is None
    assert budget["decisions"] == str(
        tmp_path / "round_0" / "stage04b" / "decisions.json"
    )
    assert (budget["weak_extra"], budget["strong_cut"], budget["total_target"]) == (
        9,
        3,
        24,
    )
    assert (
        plan["cells"][weak]["need"] == "weak" and plan["cells"][weak]["n_actual"] == 15
    )
    assert plan["cells"][strong]["need"] == "strong"
    cat = json.loads((out / "categorization.json").read_text())
    targets = {c: n for c, n in cat["target_per_cell"].items() if n}
    assert targets == {weak: 15, strong: 3, kept: 6}
    assert cat["n_final"] == 24
    # a strong leaf's target is never below the top-up's floor
    floored = run(
        "floored", "--weak-cell-factor", 2.5, "--strong-cell-factor", 0.5,
        "--min-cell-gemms", 5,
    )  # fmt: skip
    assert _by_cell(floored) == {weak: 15, strong: 5, kept: 6}
    budget = json.loads((floored / "per_cell_plan.json").read_text())["budget"]
    assert (budget["strong_cut"], budget["total_target"]) == (1, 26)
    # a strong factor alone reads the decisions but cuts nothing: no weak
    # leaf gains a shape for a strong one to pay for
    alone = run("alone", "--strong-cell-factor", 0.5)
    budget = json.loads((alone / "per_cell_plan.json").read_text())["budget"]
    assert budget["applied"] and budget["n_strong"] == 1
    assert (budget["strong_cut_wanted"], budget["strong_cut"]) == (3, 0)
    # the defaults, and every factor at 1, draw the round as it always was
    plain = run("plain")
    neutral = run(
        "neutral",
        "--weak-cell-factor", 1, "--strong-cell-factor", 1, "--large-k-factor", 1,
    )  # fmt: skip
    (tmp_path / "round_0" / "stage04b" / "decisions.json").unlink()
    without = run("without", "--weak-cell-factor", 2.5, "--strong-cell-factor", 0.5)
    first = (plain / "shapes.yaml").read_text()
    for other in (neutral, alone, without):
        assert (other / "shapes.yaml").read_text() == first
    assert _by_cell(plain) == {weak: 6, strong: 6, kept: 6}
    budget = json.loads((without / "per_cell_plan.json").read_text())["budget"]
    assert not budget["applied"] and budget["not_applied"] == "no stage04b decisions"
    assert budget["decisions"] is None


def test_an_enhance_round_keeps_its_shapes_per_cell(tmp_path, monkeypatch, config_path):
    """--only-cells draws --shapes-per-cell for each chosen leaf, weak or
    not, and no very-large-K shapes."""
    weak, other = "Mid|Small|LargeK|Bnone", "Small|Mid|MidK|Bnone"
    decisions = _decisions(_row(weak, "retrain", 0.9), _row(other, se=0.99))
    run = _active_round(tmp_path, monkeypatch, config_path, decisions, [weak, other])
    more = ["--weak-cell-factor", 3, "--large-k-factor", 2, "--shapes-per-cell", 40]
    out = run("enhance", "--only-cells", weak, *more)
    assert _by_cell(out) == {weak: 40}
    drawn = parse_shapes_yaml(out / "shapes.yaml")
    assert any(s[2] > s01.LARGE_K_ABOVE for s in drawn)
    budget = json.loads((out / "per_cell_plan.json").read_text())["budget"]
    assert budget["not_applied"] == "--only-cells" and budget["n_weak"] == 0
    cat = json.loads((out / "categorization.json").read_text())
    assert cat["large_k"]["factor"] == 1.0 and cat["large_k"]["n_shapes"] == 0
    full = run("full", *more)
    assert _by_cell(full)[weak] > 3 * 40
    stratum = json.loads((full / "categorization.json").read_text())["large_k"]
    assert stratum["factor"] == 2.0 and stratum["n_shapes"] > 0


def test_bad_budget_options_and_decisions_stop_the_round(
    tmp_path, monkeypatch, config_path, capsys
):
    cell = "Mid|Mid|MidK|Bnone"
    run = _active_round(tmp_path, monkeypatch, config_path, _decisions(), [cell])
    for flag, value in (
        ("--weak-cell-factor", 0.5),
        ("--weak-cell-factor", "nan"),
        ("--strong-cell-factor", 0),
        ("--strong-cell-factor", 1.5),
        ("--strong-cell-margin", -0.1),
        ("--large-k-factor", 0.9),
        ("--large-k-factor", "inf"),
        ("--decisions-from", tmp_path / "nowhere.json"),
    ):
        out = run(f"bad_{flag}_{value}", flag, value, rc=1)
        assert not (out / "shapes.yaml").exists(), flag
    bad = tmp_path / "bad.json"
    for text in ("{", '{"per_cell": {}}', "[]"):
        bad.write_text(text)
        capsys.readouterr()
        run("bad_doc", "--decisions-from", bad, "--weak-cell-factor", 2, rc=1)
        assert "stage04b decisions" in capsys.readouterr().err
        # factors at 1 read no decisions, as stage01 never did
        out = run("neutral_doc", "--decisions-from", bad)
        budget = json.loads((out / "per_cell_plan.json").read_text())["budget"]
        assert budget["not_applied"] == "every factor at 1"
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_decisions(_row(cell, "retrain"))))
    out = run("good", "--decisions-from", good, "--weak-cell-factor", 2)
    assert _by_cell(out) == {cell: 12}


# ── very-large-K stratum ─────────────────────────────────────────────────────


def test_the_very_large_k_stratum_doubles_a_leafs_draws_above_32768():
    f = _filter()
    leaf, batched = "Mid|Mid|LargeK|Bnone", "Mid|Mid|LargeK|Bany"
    plan = {leaf: 120, batched: 60, "Mid|Mid|MidK|Bnone": 30}
    res = s01.sample_shapes(plan, 9, set(), f, n_strata=2)
    big = [s for s in res.fresh_per_cell[leaf] if s[2] > s01.LARGE_K_ABOVE]
    assert big and not [s for s in res.fresh_per_cell[batched] if s[2] > 32768]
    taken = set(res.shapes)
    out = s01.large_k_shapes(
        res.fresh_per_cell, 2.0, 9, f, n_strata=2, taken=set(taken)
    )
    assert list(out) == [leaf] and len(out[leaf]) == len(big)
    for s in out[leaf]:
        assert s[2] > s01.LARGE_K_ABOVE and f.accepts(*s) and s not in taken
        assert grid.cell_key(*s) == leaf
    again = s01.large_k_shapes(
        res.fresh_per_cell, 2.0, 9, f, n_strata=2, taken=set(taken)
    )
    assert again == out
    half = s01.large_k_shapes(
        res.fresh_per_cell, 1.5, 9, f, n_strata=2, taken=set(taken)
    )
    assert len(half[leaf]) == int(len(big) * 0.5 + 0.5)
    assert (
        s01.large_k_shapes(res.fresh_per_cell, 1.0, 9, f, n_strata=2, taken=set()) == {}
    )


def test_a_split_leaf_draws_its_stratum_inside_its_box():
    from lib import subcells as sc

    rule = sc.build_split_rule("Mid|Mid|LargeK|Bnone", "K", 40000)
    f = _filter()
    fresh = {
        rule.lo_label: [(300, 300, 33000, 1)],
        rule.hi_label: [(300, 300, 50000, 1)],
    }
    out = s01.large_k_shapes(fresh, 3.0, 4, f, n_strata=1, taken=set())
    assert len(out[rule.lo_label]) == 2 and len(out[rule.hi_label]) == 2
    assert all(32768 < s[2] <= 40000 for s in out[rule.lo_label])
    assert all(40000 < s[2] for s in out[rule.hi_label])


def test_each_leaf_draws_its_stratum_from_a_stream_of_its_own(monkeypatch):
    """The stream a leaf's stratum draws from depends on the round and the
    leaf: no two leaves, and no two rounds, draw the same sequence."""
    from lib import subcells as sc

    rule = sc.build_split_rule("Mid|Mid|LargeK|Bnone", "K", 40000)
    fresh = {
        rule.lo_label: [(300, 300, 33000, 1)],
        rule.hi_label: [(300, 300, 50000, 1)],
    }
    states = []
    real = s01.synthesize_cell_shapes

    def spy(base, n, rng, f, **kw):
        states.append(rng.getstate())
        return real(base, n, rng, f, **kw)

    monkeypatch.setattr(s01, "synthesize_cell_shapes", spy)
    for round_seed in (4, 5):
        s01.large_k_shapes(fresh, 2.0, round_seed, _filter(), n_strata=1, taken=set())
    assert len(states) == len(set(states)) == 4


def test_targeted_shapes_never_repeat_a_very_large_k_shape(
    tmp_path, monkeypatch, config_path
):
    """The stratum's shapes join the shapes the targeted draw must avoid:
    a shape near a tail GEMM that the stratum drew is not drawn again."""
    leaf = "Mid|Small|LargeK|Bnone"
    tail = _tail(300, 100, 40000, 1, 0.4)
    decisions = _decisions(_row(leaf, se=0.97, tail_gemms=[tail]))
    run = _active_round(tmp_path, monkeypatch, config_path, decisions, [leaf])
    drawn = {}
    real_large_k, real_near = s01.large_k_shapes, s01.synthesize_near

    def large_k(*a, **kw):
        drawn.update(real_large_k(*a, **kw))
        return drawn

    def near(shape, n, rng, f, box, **kw):
        return [s for v in drawn.values() for s in v] + real_near(
            shape, n, rng, f, box, **kw
        )

    monkeypatch.setattr(s01, "large_k_shapes", large_k)
    monkeypatch.setattr(s01, "synthesize_near", near)
    out = run(
        "both", "--shapes-per-cell", 40, "--large-k-factor", 2, "--targeted-per-gemm", 3
    )
    stratum = {s for v in drawn.values() for s in v}
    assert stratum
    targeted = json.loads((out / s01.TARGETED_FILE).read_text())["leaves"][leaf]
    near_shapes = {tuple(s) for e in targeted for s in e["shapes"]}
    assert len(near_shapes) == 3 and not near_shapes & stratum
    written = parse_shapes_yaml(out / "shapes.yaml")
    assert len(written) == len(set(written))
    assert stratum | near_shapes <= set(written)


def test_a_round_draws_the_stratum_end_to_end(tmp_path, monkeypatch, config_path):
    leaf, other = "Mid|Small|LargeK|Bnone", "Mid|Mid|MidK|Bnone"
    run = _active_round(tmp_path, monkeypatch, config_path, None, [leaf, other])
    plain = run("plain", "--shapes-per-cell", 40)
    out = run("large_k", "--shapes-per-cell", 40, "--large-k-factor", 2)
    before = [s for s in parse_shapes_yaml(plain / "shapes.yaml")]
    after = parse_shapes_yaml(out / "shapes.yaml")
    assert set(before) < set(after)
    added = set(after) - set(before)
    n_big = sum(1 for s in before if grid.cell_key(*s) == leaf and s[2] > 32768)
    assert n_big and len(added) == n_big
    assert all(s[2] > 32768 and grid.cell_key(*s) == leaf for s in added)
    cat = json.loads((out / "categorization.json").read_text())
    assert cat["large_k"] == {
        "factor": 2.0, "above": 32768, "n_shapes": n_big, "per_cell": {leaf: n_big},
    }  # fmt: skip
    assert cat["n_final"] == len(after) and cat["actual_per_cell"][leaf] == 40
    plan = json.loads((out / "per_cell_plan.json").read_text())
    assert plan["cells"][leaf]["n_large_k"] == n_big
    assert plan["cells"][other]["n_large_k"] == 0
    assert json.loads((plain / "categorization.json").read_text())["large_k"] == {
        "factor": 1.0, "above": 32768, "n_shapes": 0, "per_cell": {},
    }  # fmt: skip


def _bench_yaml(path, shapes):
    path.write_text(
        "".join(
            "- {rotating_buffer: 512, transA: T, transB: N, "
            f"M: {m}, N: {n}, K: {k}, batch_count: {b}}}\n"
            for m, n, k, b in shapes
        )
    )
    return path


def test_the_stratum_never_draws_a_held_out_shape(tmp_path, monkeypatch, config_path):
    """A held-out list's GEMMs are no stratum shapes: with every shape the
    stratum drew held out, it draws as many others, and the round's other
    draws stay as they were."""
    leaf, other = "Mid|Small|LargeK|Bnone", "Mid|Mid|MidK|Bnone"
    run = _active_round(tmp_path, monkeypatch, config_path, None, [leaf, other])
    more = ["--shapes-per-cell", 40, "--large-k-factor", 2]
    plain = run("plain", "--shapes-per-cell", 40)
    regular = set(parse_shapes_yaml(plain / "shapes.yaml"))
    first = parse_shapes_yaml(run("first", *more) / "shapes.yaml")
    stratum = set(first) - regular
    assert stratum
    held = _bench_yaml(tmp_path / "held.yaml", sorted(stratum))
    out = run("held", *more, "--exclude-shapes-yaml", held)
    again = parse_shapes_yaml(out / "shapes.yaml")
    assert not set(again) & stratum
    assert regular < set(again) and len(again) == len(set(again)) == len(first)
    cat = json.loads((out / "categorization.json").read_text())
    assert cat["large_k"]["n_shapes"] == len(stratum)
    for s in set(again) - regular:
        assert s[2] > s01.LARGE_K_ABOVE and grid.cell_key(*s) == leaf


def test_a_held_out_stratum_shape_stops_the_round(
    tmp_path, monkeypatch, config_path, capsys
):
    """The stratum's shapes are checked against the held-out lists as the
    round's other draws are: one that is held out stops the round before
    it writes anything."""
    leaf = "Mid|Small|LargeK|Bnone"
    run = _active_round(tmp_path, monkeypatch, config_path, None, [leaf])
    held = (300, 100, 40000, 1)
    more = ["--shapes-per-cell", 40, "--large-k-factor", 2]
    more += ["--exclude-shapes-yaml", _bench_yaml(tmp_path / "held.yaml", [held])]
    assert held not in parse_shapes_yaml(run("ok", *more) / "shapes.yaml")
    monkeypatch.setattr(s01, "large_k_shapes", lambda *a, **kw: {leaf: [held]})
    capsys.readouterr()
    out = run("leaked", *more, rc=1)
    assert "1 very-large-K shapes are held out" in capsys.readouterr().err
    assert not (out / "shapes.yaml").exists()
    assert not (out / "categorization.json").exists()


def test_the_stratum_skips_history_held_out_and_this_rounds_shapes(
    tmp_path, monkeypatch, config_path
):
    """The stratum draws under the round's history dedup and held-out lists
    and never repeats a shape the round drew: offered a shape of an earlier
    round, a held-out shape and one of the round's own draws before its
    own candidates, it skips all three and still draws its count."""
    leaf = "Mid|Small|LargeK|Bnone"
    run = _active_round(tmp_path, monkeypatch, config_path, None, [leaf])
    regular = parse_shapes_yaml(run("plain", "--shapes-per-cell", 40) / "shapes.yaml")
    n_big = sum(1 for s in regular if s[2] > s01.LARGE_K_ABOVE)
    own = next(s for s in regular if s[2] > s01.LARGE_K_ABOVE)
    old, held = (300, 100, 40000, 1), (320, 100, 40000, 1)
    assert n_big and not {old, held} & set(regular)
    real = s01.synthesize_cell_shapes

    def offer(base, n, rng, f, axis_bounds=None, **kw):
        drawn = real(base, n, rng, f, axis_bounds=axis_bounds, **kw)
        if axis_bounds and axis_bounds["K"][0] > s01.LARGE_K_ABOVE:
            return [old, held, own] + drawn
        return drawn

    monkeypatch.setattr(s01, "synthesize_cell_shapes", offer)
    out = run(
        "offered", "--shapes-per-cell", 40, "--large-k-factor", 2,
        "--prior-shapes-yaml", _bench_yaml(tmp_path / "old.yaml", [old]),
        "--exclude-shapes-yaml", _bench_yaml(tmp_path / "held.yaml", [held]),
    )  # fmt: skip
    written = parse_shapes_yaml(out / "shapes.yaml")
    assert len(written) == len(set(written)) and not {old, held} & set(written)
    assert set(regular) < set(written)
    assert len(set(written) - set(regular)) == n_big
    cat = json.loads((out / "categorization.json").read_text())
    assert cat["large_k"]["n_shapes"] == n_big


# ── seeded targets ───────────────────────────────────────────────────────────

SEEDED_HEAD = "library,m,n,k,batch,cu_budget,bench_contexts,cell,source\n"


def _seeded_csv(path, rows):
    path.write_text(SEEDED_HEAD + "".join(rows))
    return path


def test_seeded_targets_are_read_for_one_library(tmp_path):
    p = _seeded_csv(
        tmp_path / "t.csv",
        [
            'a,300,300,288,1,0,exclusive,Mid|Mid|MidK|Bnone,"x, y"\n',
            'a,260,200,96,4,48,"cu32_cot,cu64_cot",Mid|Mid|MidK|Bany,z\n',
            "b,64,64,64,1,32,cu32_cot,Small|Small|MidK|Bnone,w\n",
        ],
    )
    got = shapes.read_seeded_targets(p, "a")
    assert [(a["m"], a["n"], a["k"], a["batch"], a["row"]) for a in got] == [
        (300, 300, 288, 1, 2),
        (260, 200, 96, 4, 3),
    ]
    assert got[0]["bench_contexts"] == [] and got[0]["source"] == "x, y"
    assert got[1]["bench_contexts"] == ["cu32_cot", "cu64_cot"]
    assert shapes.read_seeded_targets(p, "c") == []
    with pytest.raises(ValueError, match="several libraries"):
        shapes.read_seeded_targets(p)
    plain = tmp_path / "plain.csv"
    plain.write_text("m,n,k,batch\n300,300,288,1\n")
    assert len(shapes.read_seeded_targets(plain)) == 1
    with pytest.raises(ValueError, match="no library column"):
        shapes.read_seeded_targets(plain, "a")
    for bad, why in (
        ("m,n,k\n1,1,1\n", "no column batch"),
        ("m,n,k,batch\n1,1,x,1\n", "whole numbers"),
        ("m,n,k,batch\n1,0,1,1\n", "at least 1"),
    ):
        plain.write_text(bad)
        with pytest.raises(ValueError, match=why):
            shapes.read_seeded_targets(plain)


def _anchor(m, n, k, b, contexts=()):
    return {"m": m, "n": n, "k": k, "batch": b, "bench_contexts": list(contexts),
            "row": 2}  # fmt: skip


def test_seeded_shapes_are_drawn_near_their_anchors_deterministically():
    leaf = "Mid|Mid|MidK|Bnone"
    anchors = [_anchor(300, 300, 288, 1), _anchor(300, 300, 288, 1, ["cu32_cot"])]
    other = _anchor(4000, 4000, 4000, 1)

    def draw(seed, taken=()):
        return s01.seeded_shapes(
            anchors + [other], {leaf}, {}, seed, _filter(), per_target=4,
            taken=set(taken),
        )  # fmt: skip

    out, skipped = draw(11)
    (entries,) = out.values()
    assert [e["anchor"]["bench_contexts"] for e in entries] == [[], ["cu32_cot"]]
    near = [tuple(s) for e in entries for s in e["shapes"]]
    assert len(near) == 8 and len(set(near)) == 8
    assert (300, 300, 288, 1) not in near
    for m, n, k, b in near:
        assert grid.cell_key(m, n, k, b) == leaf and b == 1
        assert 150 <= m <= 600 and 150 <= n <= 600 and 144 <= k <= 512
        assert shapes.alignment(m) == shapes.alignment(300)
    assert skipped == [dict(other, leaf=grid.cell_key(4000, 4000, 4000, 1),
                            why="the round samples no shape in its leaf")]  # fmt: skip
    # The same round seed draws the same shapes, another round others; the
    # anchors with other contexts draw from streams of their own.
    assert draw(11)[0] == out
    assert draw(12)[0] != out
    assert entries[0]["shapes"] != entries[1]["shapes"]
    # Alone, an anchor draws what it draws beside the others, and the anchor
    # with other contexts still draws other shapes.
    alone = [
        s01.seeded_shapes([a], {leaf}, {}, 11, _filter(), per_target=4,
                          taken=set())[0][leaf][0]["shapes"]
        for a in anchors
    ]  # fmt: skip
    assert alone[0] == entries[0]["shapes"] and alone[1] != alone[0]
    # Shapes already taken are not drawn again, and no anchor ever is.
    taken = {tuple(s) for s in entries[0]["shapes"]}
    again, _ = draw(11, taken)
    assert not taken & {tuple(s) for e in again[leaf] for s in e["shapes"]}
    close = [_anchor(300, 300, 288, 1), _anchor(304, 300, 288, 1)]
    out, _ = s01.seeded_shapes(close, {leaf}, {}, 5, _filter(), per_target=40,
                               taken=set())  # fmt: skip
    drawn = {tuple(s) for e in out[leaf] for s in e["shapes"]}
    assert not drawn & {(300, 300, 288, 1), (304, 300, 288, 1)}


def test_a_round_draws_its_seeded_targets_after_its_tails(
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
    tails = [_tail(300, 300, 288, 1, 0.42, cu=128)]
    (prior / "stage04b").mkdir()
    (prior / "stage04b" / "decisions.json").write_text(
        json.dumps({"per_cell": [{"cell": planned[0], "tail_gemms": tails}]})
    )
    csv_path = _seeded_csv(
        tmp_path / "targets.csv",
        [
            'lib,260,200,96,1,48,"cu32_cot,cu64_cot",Mid|Mid|MidK|Bnone,a\n',
            "lib,4000,4000,4000,1,0,exclusive,Large|Large|LargeK|Bnone,b\n",
            "other,300,300,96,1,0,exclusive,Mid|Mid|MidK|Bnone,c\n",
        ],
    )
    base = ["--mode", "active", "--config-yaml", config_path, "--prior-round-dir"]
    base += [prior, "--shapes-per-cell", 6, "--targeted-per-gemm", 3, "--quiet"]
    plain = tmp_path / "plain"
    assert _run(monkeypatch, *base, "--out-dir", plain) == 0
    out = tmp_path / "seeded"
    seeded = ["--targeted-seed", csv_path, "--targeted-seed-library", "lib"]
    assert _run(monkeypatch, *base, "--out-dir", out, *seeded) == 0
    doc = json.loads((out / s01.TARGETED_FILE).read_text())
    (entries,) = doc["leaves"].values()
    assert ["tail" in e for e in entries] == [True, False]
    tail_doc = json.loads((plain / s01.TARGETED_FILE).read_text())
    (tail_entries,) = tail_doc["leaves"].values()
    assert entries[0] == tail_entries[0], "the tails draw as without the key"
    anchor = entries[1]["anchor"]
    assert (anchor["m"], anchor["bench_contexts"]) == (260, ["cu32_cot", "cu64_cot"])
    assert len(entries[1]["shapes"]) == 4
    sha = __import__("hashlib").sha256(csv_path.read_bytes()).hexdigest()
    rec = doc["seeded"]
    assert (rec["list"], rec["sha256"], rec["library"]) == (str(csv_path), sha, "lib")
    assert (rec["per_target"], rec["n_anchors"], rec["n_anchors_drawn"]) == (4, 2, 1)
    assert rec["n_shapes"] == 4 and rec["per_cell"] == {planned[0]: 4}
    assert [s["row"] for s in rec["skipped"]] == [3]
    cat = json.loads((out / "categorization.json").read_text())
    assert cat["targeted"]["seeded"] == rec
    assert cat["targeted"]["n_shapes"] == 3 + 4
    assert (
        cat["n_final"]
        == json.loads((plain / "categorization.json").read_text())["n_final"] + 4
    )
    plan = json.loads((out / "per_cell_plan.json").read_text())
    assert plan["cells"][planned[0]]["n_seeded"] == 4
    written = set(parse_shapes_yaml(out / "shapes.yaml"))
    assert {tuple(s) for s in entries[1]["shapes"]} <= written
    assert written - {tuple(s) for s in entries[1]["shapes"]} == set(
        parse_shapes_yaml(plain / "shapes.yaml")
    )
    # Deterministic: the same round draws the same anchors' shapes.
    again = tmp_path / "again"
    assert _run(monkeypatch, *base, "--out-dir", again, *seeded) == 0
    assert (
        json.loads((again / s01.TARGETED_FILE).read_text())["leaves"] == doc["leaves"]
    )
    # The key without tail GEMMs still writes its draws; a bad list stops.
    (prior / "stage04b" / "decisions.json").unlink()
    alone = tmp_path / "alone"
    assert _run(monkeypatch, *base, "--out-dir", alone, *seeded) == 0
    (entries,) = json.loads((alone / s01.TARGETED_FILE).read_text())["leaves"].values()
    assert [list(e) for e in entries] == [["anchor", "shapes"]]
    bad = ["--targeted-seed", csv_path]
    assert _run(monkeypatch, *base, "--out-dir", tmp_path / "bad", *bad) == 1
    zero = [*seeded, "--targeted-seed-per-target", 0]
    assert _run(monkeypatch, *base, "--out-dir", tmp_path / "zero", *zero) == 1


def test_no_anchor_is_drawn_near_another(monkeypatch):
    """Every anchor joins the shapes the draws avoid before any is drawn, so
    an anchor next to another is never one of its draws."""
    leaf = "Mid|Mid|MidK|Bnone"
    a, b = (300, 300, 288, 1), (320, 300, 288, 1)
    offered = [b, (336, 300, 288, 1), a]

    def near(shape, n, rng, f, box, **kw):
        return [s for s in offered if s != tuple(shape)]

    monkeypatch.setattr(s01, "synthesize_near", near)
    out, skipped = s01.seeded_shapes(
        [_anchor(*a), _anchor(*b)], {leaf}, {}, 3, _filter(), per_target=4, taken=set()
    )
    assert [e["shapes"] for e in out[leaf]] == [[[336, 300, 288, 1]]]
    assert [s["why"] for s in skipped] == ["no new shape near it"]


def test_a_held_out_gemm_near_an_anchor_is_drawn_around(
    tmp_path, monkeypatch, config_path
):
    """The draws near an anchor avoid the held-out lists' GEMMs as the
    round's other draws do: with the first shape the anchor's stream draws
    held out, the round draws the next one in its place and is written."""
    leaf = "Mid|Mid|MidK|Bnone"
    run = _active_round(tmp_path, monkeypatch, config_path, None, [leaf])
    targets = _seeded_csv(
        tmp_path / "targets.csv",
        ['lib,260,200,96,1,48,"cu32_cot,cu64_cot",Mid|Mid|MidK|Bnone,a\n'],
    )
    seeded = ["--targeted-seed", targets, "--targeted-seed-library", "lib"]

    def near(out):
        doc = json.loads((out / s01.TARGETED_FILE).read_text())
        (entry,) = [e for es in doc["leaves"].values() for e in es if "anchor" in e]
        return [tuple(s) for s in entry["shapes"]]

    first = run("first", *seeded)
    drawn = near(first)
    assert len(drawn) == 4
    held = _bench_yaml(tmp_path / "held.yaml", [drawn[0]])
    out = run("held", *seeded, "--exclude-shapes-yaml", held)
    again = near(out)
    assert drawn[0] not in again and again[:3] == drawn[1:] and len(again) == 4
    written = set(parse_shapes_yaml(out / "shapes.yaml"))
    assert drawn[0] not in written and set(again) <= written
    assert written - set(again) == set(parse_shapes_yaml(first / "shapes.yaml")) - {
        *drawn
    }
