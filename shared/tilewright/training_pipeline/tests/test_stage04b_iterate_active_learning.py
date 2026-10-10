# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import math
import random
import sys

import pytest

from test_stage05_train import (
    CELLS3,
    load_bundle,
    run_stage,
    shapes_for,
    train,
    write_config,
    write_round,
)

pytest.importorskip("tilewright")

ORCHESTRATOR_KEYS = {
    "cell",
    "decision",
    "reason",
    "n_eval",
    "sel_eff_new",
    "origami_sel_eff_new",
    "origami_n_eval",
}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """round_0 trained on two cells; round_1 data with new shapes, three
    re-used round_0 shapes and two more round_0 shapes."""
    tmp = tmp_path_factory.mktemp("run")
    r0, r1 = tmp / "round_0", tmp / "round_1"
    old = shapes_for(10, 30, CELLS3[:2])
    write_round(r0 / "stage04", old)
    train(tmp, [r0 / "stage04"], r0 / "stage05", "--weight-dtype", "bf16")
    write_round(r1 / "stage04", shapes_for(11, 45, CELLS3) + old[:5])
    (r1 / "stage01").mkdir(parents=True)
    (r1 / "stage01" / "reused_shapes.yaml").write_text(
        "round_index: 1\nshapes:\n"
        + "".join(f"- [{m}, {n}, {k}, {b}]\n" for m, n, k, b in old[:3])
    )
    (r1 / "stage01" / "per_cell_plan.json").write_text('{"shapes_per_cell": 30}')
    write_config(tmp / "config.yaml")
    return tmp


def stage04b_args(tmp, out, *extra):
    return [
        "--models-dir",
        tmp / "round_0" / "stage05",
        "--enriched-csv-dir",
        tmp / "round_1" / "stage04",
        "--out-dir",
        out,
        "--arch",
        "gfx950",
        "--config-yaml",
        tmp / "config.yaml",
        "--weight-dtype",
        "bf16",
        "--round-idx",
        1,
        *extra,
    ]


def stage04b(tmp, out, *extra):
    run_stage(
        "stage04b_iterate_active_learning.py",
        *stage04b_args(tmp, out, "--quiet", *extra),
    )
    return json.loads((out / "decisions.json").read_text())


def test_held_out_data_and_outputs(run):
    out = run / "round_1" / "stage04b"
    d = stage04b(run, out, "--min-cell-gemms", 20)
    assert d["aggregate"]["n_gemms_excluded_not_held_out"] == 5
    assert d["split_budget_check"] == {
        "per_leaf_budget": 30,
        "gemms_needed_for_split": 40,
        "feasible": False,
    }
    rows = {r["cell"]: r for r in d["per_cell"]}
    assert set(rows) == set(CELLS3)
    for r in rows.values():
        assert ORCHESTRATOR_KEYS <= set(r)
        assert r["n_eval"] == 45
    assert rows[CELLS3[2]]["decision"] == "no_model"
    assert rows[CELLS3[2]]["reason"] == "no_prior_model__too_few_training_gemms(45<50)"
    assert rows[CELLS3[2]]["sel_eff_new"] is not None
    assert [rows[c]["n_training_gemms"] for c in CELLS3] == [75, 75, 45]
    assert rows[CELLS3[0]]["n_excluded_not_held_out"] == 5
    assert (
        rows[CELLS3[0]]["n_model_served"] + rows[CELLS3[0]]["n_origami_fallback"] == 45
    )
    assert d["aggregate"]["all_held_out"]["paired"]["n"] == 135
    assert d["cells_no_model"] == [CELLS3[2]] and d["cells_train"] == []
    retrain = (out / "retrain_cells.txt").read_text().split(",")
    assert set(filter(None, retrain)) == set(d["cells_retrain"])
    assert json.loads((out / "splits.json").read_text())["splits"] == {}


def _wgeo(pairs):
    pairs = [(v, n) for v, n in pairs if v is not None and n > 0]
    return math.exp(sum(math.log(v) * n for v, n in pairs) / sum(n for _, n in pairs))


def test_paired_numbers_and_misses(run):
    d = stage04b(
        run,
        run / "round_1" / "paired",
        "--min-cell-gemms",
        20,
        "--pick-measurement",
        "timed",
    )
    agg, rows = d["aggregate"], d["per_cell"]
    for r in rows:
        if r.get("origami_n_eval"):
            assert r["paired_sel_eff_new"] is not None
    assert agg["n_gemms_paired"] == agg["all_held_out"]["paired"]["n"] > 0
    assert agg["global_paired_sel_eff_new"] == pytest.approx(
        _wgeo((r.get("paired_sel_eff_new"), r.get("origami_n_eval") or 0) for r in rows)
    )
    assert agg["pick_measurement"] == "timed"
    assert agg["n_miss_total"] == agg["all_held_out"]["n_miss"] > 0
    assert agg["n_miss_total"] == sum(r.get("n_miss") or 0 for r in rows)
    for rule, out in (("warmup", "default"), ("any", "warm")):
        args = () if rule == "warmup" else ("--pick-measurement", rule)
        lenient = stage04b(run, run / "round_1" / out, "--min-cell-gemms", 20, *args)
        assert lenient["aggregate"]["pick_measurement"] == rule
        assert lenient["aggregate"]["n_miss_total"] == 0


def test_thresholds_are_numbers(run, tmp_path):
    for flag in ("--sel-eff-threshold", "--split-floor"):
        proc = run_stage(
            "stage04b_iterate_active_learning.py",
            *stage04b_args(run, tmp_path / "o", "--quiet", flag, "origami"),
            expect_rc=2,
        )
        assert flag in proc.stderr
    assert not (tmp_path / "o" / "decisions.json").exists()


def test_origami_numbers_print_only_as_a_reference(run):
    base = run / "round_1"
    plain = run_stage(
        "stage04b_iterate_active_learning.py",
        *stage04b_args(run, base / "plain", "--min-cell-gemms", 20),
    )
    assert "origami" not in plain.stdout.lower()
    ref = run_stage(
        "stage04b_iterate_active_learning.py",
        *stage04b_args(
            run, base / "ref", "--min-cell-gemms", 20, "--origami-reference"
        ),
    )
    assert "[reference] Origami, same GEMMs" in ref.stdout
    a, b = (
        json.loads((base / d / "decisions.json").read_text())["per_cell"]
        for d in ("plain", "ref")
    )
    assert [(r["cell"], r["decision"]) for r in a] == [
        (r["cell"], r["decision"]) for r in b
    ]


def test_plan_lists_the_benched_kernels(run, tmp_path):
    shapes = shapes_for(11, 45, CELLS3)[:7]
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "shapes": [
                    {"m": m, "n": n, "k": k, "batch": b, "kernels": [100, 101]}
                    for m, n, k, b in shapes
                ]
            }
        )
    )
    d = stage04b(run, tmp_path / "planned", "--min-cell-gemms", 20, "--plan", plan)
    assert d["aggregate"]["n_gemms_listed"] == 7
    plan.write_text("{not json")
    run_stage(
        "stage04b_iterate_active_learning.py",
        *stage04b_args(run, tmp_path / "bad", "--quiet", "--plan", plan),
        expect_rc=1,
    )


def _keys(folder):
    import stage05_train as s5
    from lib import evaluate as ev

    cells, _ = s5.load_enriched_chunks(str(folder), True)
    return {ev.gemm_key(g) for gs in cells.values() for g in gs}


def _rows(folder):
    return sum(
        len(p.read_text().splitlines()) - 1 for p in sorted(folder.glob("chunk_*.csv"))
    )


KERNELS = '{"library_stem": null, "attribute_names": ["wave_num"], "kernels": {}}\n'
SUMMARY = {
    "library_stem": "stem",
    "gemms": 140,
    "rows": 9999,
    "blocklist": {"digest": "sha256:" + "a" * 64, "kernels": 2},
    "launch": {
        "heuristics": ["sha256:" + "b" * 64],
        "overrides": {},
        "workspace_bytes": [1024],
        "unknown": 0,
    },
}


def test_gate_gemms_stay_out_of_this_rounds_training(run):
    src = run / "round_1" / "stage04"
    (src / "kernels.json").write_text(KERNELS)
    (src / "enrich_summary.json").write_text(json.dumps(SUMMARY))
    try:
        _check_gate(run, src)
    finally:
        (src / "kernels.json").unlink()
        (src / "enrich_summary.json").unlink(missing_ok=True)


def _check_gate(run, src):
    out = run / "round_1" / "gated"
    d = stage04b(run, out, "--min-cell-gemms", 20, "--gate-frac", 0.2, "--seed", 5)
    gate = d["gate"]
    assert gate["frac"] == 0.2 and gate["per_cell"] == {c: 9 for c in CELLS3}
    assert gate["n_gemms"] == 27 and gate["n_cells"] == 3
    rows = {r["cell"]: r for r in d["per_cell"]}
    assert [rows[c]["n_training_gemms"] for c in CELLS3] == [66, 66, 36]
    assert all(rows[c]["n_gate_gemms"] == 9 and rows[c]["n_eval"] == 45 for c in CELLS3)
    gate_keys, train_keys = _keys(out / "gate"), _keys(out / "train")
    assert len(gate_keys) == 27 and not gate_keys & train_keys
    assert gate_keys | train_keys == _keys(src)
    assert _rows(out / "gate") == gate["n_rows"]
    assert _rows(out / "train") == gate["n_training_rows"]
    assert gate["n_rows"] + gate["n_training_rows"] == _rows(src)
    for sub in ("gate", "train"):
        assert (out / sub / "kernels.json").read_text() == KERNELS
        part = json.loads((out / sub / "enrich_summary.json").read_text())
        assert part == {
            "library_stem": "stem",
            "blocklist": SUMMARY["blocklist"],
            "launch": SUMMARY["launch"],
            "slice": sub,
            "slice_of": str(src),
            "rows": _rows(out / sub),
        }
    import stage05_train as s5

    assert s5.bench_launch([out / "train"])["unknown"] == 0
    assert s5.labels_blocklist(None, [out / "train"])["enriched"] == [
        SUMMARY["blocklist"]["digest"]
    ]
    (src / "enrich_summary.json").unlink()
    again = stage04b(
        run, run / "round_1" / "gated2", "--min-cell-gemms", 20, "--gate-frac", 0.2
    )
    assert again["gate"]["n_gemms"] == 27
    assert not (run / "round_1" / "gated2" / "train" / "enrich_summary.json").exists()
    same = stage04b(run, out, "--min-cell-gemms", 20, "--gate-frac", 0.2, "--seed", 5)
    assert _keys(out / "gate") == gate_keys and same["gate"] == gate
    off = stage04b(run, out, "--min-cell-gemms", 20)
    assert off["gate"] is None and not (out / "gate").exists()
    assert not (out / "train").exists()


def test_a_leaf_without_a_model_is_listed_once_trainable(run):
    out = run / "train_new"
    d = stage04b(run, out, "--min-cell-gemms", 20, "--train-min-cell-gemms", 40)
    row = next(r for r in d["per_cell"] if r["cell"] == CELLS3[2])
    assert (row["decision"], row["reason"]) == ("train", "no_prior_model")
    assert d["cells_train"] == [CELLS3[2]] and d["cells_no_model"] == [CELLS3[2]]
    retrain = (out / "retrain_cells.txt").read_text().split(",")
    assert set(filter(None, retrain)) == set(d["cells_retrain"]) | {CELLS3[2]}


def test_a_split_needs_trainable_children(run):
    d = stage04b(
        run,
        run / "untrainable_split",
        "--min-cell-gemms",
        20,
        "--sel-eff-threshold",
        1.01,
        "--split-floor",
        1.01,
        "--split-after-attempts",
        0,
    )
    assert d["cells_split"] == [] and d["train_min_cell_gemms"] == 50
    assert d["cells_split_blocked"] == CELLS3[:2]
    for cell in CELLS3[:2]:
        r = next(r for r in d["per_cell"] if r["cell"] == cell)
        assert r["decision"] == "retrain"
        assert "__children_need_50_training_gemms(lo=" in r["reason"]
    retrain = (run / "untrainable_split" / "retrain_cells.txt").read_text()
    assert set(retrain.split(",")) == set(CELLS3[:2])


def test_round_one_has_no_improvement_trigger(run):
    d = stage04b(
        run,
        run / "no_baseline",
        "--min-cell-gemms",
        20,
        "--sel-eff-threshold",
        1.01,
        "--split-floor",
        0.01,
        "--split-after-attempts",
        0,
    )
    for cell in CELLS3[:2]:
        r = next(r for r in d["per_cell"] if r["cell"] == cell)
        assert r["decision"] == "retrain"
        assert r["reason"] == "below_threshold__no_split_trigger(no_baseline)"


def test_baseline_from_prior_decisions(run, tmp_path):
    prior = tmp_path / "decisions.json"
    rows = [
        {"cell": CELLS3[0], "decision": "retrain", "sel_eff_new": 2.0, "n_eval": 5},
        {"cell": CELLS3[1], "decision": "retrain", "sel_eff_new": 0.01, "n_eval": 5},
    ]
    prior.write_text(json.dumps({"per_cell": rows}))
    d = stage04b(
        run,
        run / "baseline",
        "--min-cell-gemms",
        20,
        "--train-min-cell-gemms",
        20,
        "--sel-eff-threshold",
        1.01,
        "--split-floor",
        0.01,
        "--split-after-attempts",
        1,
        "--prior-decisions",
        prior,
    )
    r0 = next(r for r in d["per_cell"] if r["cell"] == CELLS3[0])
    r1 = next(r for r in d["per_cell"] if r["cell"] == CELLS3[1])
    assert r0["decision"] == "split" and "cross_round_prior_decisions" in r0["reason"]
    assert r0["prior_retrain_attempts"] == 1
    assert r1["decision"] == "retrain" and "no_split_trigger(delta=" in r1["reason"]
    tree = json.loads((run / "baseline" / "splits.json").read_text())["splits"]
    assert list(tree) == [CELLS3[0]]
    retrain = (run / "baseline" / "retrain_cells.txt").read_text().split(",")
    assert (
        tree[CELLS3[0]]["lo_label"] in retrain
        and tree[CELLS3[0]]["hi_label"] in retrain
    )


def test_split_needing_more_gemms_is_reported(run):
    d = stage04b(
        run,
        run / "blocked",
        "--min-cell-gemms",
        25,
        "--sel-eff-threshold",
        1.01,
        "--split-floor",
        1.01,
        "--split-after-attempts",
        0,
    )
    assert d["cells_split"] == []
    assert d["cells_split_blocked"] == CELLS3[:2]
    for cell in CELLS3[:2]:
        r = next(r for r in d["per_cell"] if r["cell"] == cell)
        assert r["reason"].endswith("__split_needs_50_gemms_has_45")


def test_a_fallback_parent_serves_its_untrained_leaf_until_it_trains(tmp_path):
    import torch

    from lib import subcells as sc

    r0, r1, r2 = (tmp_path / f"round_{i}" for i in range(3))
    write_round(r0 / "stage04", shapes_for(2, 30, CELLS3[:2]))
    train(tmp_path, [r0 / "stage04"], r0 / "stage05", "--weight-dtype", "bf16")
    rule = sc.build_split_rule(CELLS3[0], "M", 5800)
    sc.write_splits_json(r1 / "stage04b" / "splits.json", 1, [rule])
    prior = load_bundle(r0 / "stage05" / "models.pt")
    kept = {
        CELLS3[0]: prior["models"][CELLS3[0]],
        rule.lo_label: prior["models"][CELLS3[0]],
        CELLS3[1]: prior["models"][CELLS3[1]],
    }
    (r1 / "stage05").mkdir(parents=True)
    torch.save(dict(prior, models=kept), r1 / "stage05" / "models.pt")
    rng = random.Random(4)
    hi = [
        (rng.randint(5801, 6000), rng.randint(600, 6000), rng.randint(20, 187) * 32, 1)
        for _ in range(24)
    ]
    write_round(r2 / "stage04", hi + shapes_for(5, 20, CELLS3[:1]))
    out = r2 / "stage04b"
    run_stage(
        "stage04b_iterate_active_learning.py",
        "--models-dir",
        r1 / "stage05",
        "--enriched-csv-dir",
        r2 / "stage04",
        "--out-dir",
        out,
        "--arch",
        "gfx950",
        "--config-yaml",
        write_config(tmp_path / "config.yaml"),
        "--weight-dtype",
        "bf16",
        "--round-idx",
        2,
        "--prior-round-dir",
        r0,
        "--prior-round-dir",
        r1,
        "--min-cell-gemms",
        10,
        "--train-min-cell-gemms",
        20,
        "--sel-eff-threshold",
        0.0,
        "--quiet",
    )
    d = json.loads((out / "decisions.json").read_text())
    assert d["cells_fallback_parents"] == [CELLS3[0]]
    assert CELLS3[0] not in {r["cell"] for r in d["per_cell"]}
    row = next(r for r in d["per_cell"] if r["cell"] == rule.hi_label)
    assert row["decision"] == "train" and row["model_cells"] == [CELLS3[0]]
    assert row["n_model_served"] + row["n_origami_fallback"] == row["n_eval"] > 0
    retrain = (out / "retrain_cells.txt").read_text().split(",")
    assert retrain == [rule.hi_label]

    train(
        tmp_path,
        [r0 / "stage04", r2 / "stage04"],
        r2 / "stage05",
        "--prior-round-dir",
        r0,
        "--prior-round-dir",
        r1,
        "--current-round-dir",
        r2,
        "--only-cells",
        ",".join(retrain),
        "--min-cell-gemms",
        20,
    )
    healed = load_bundle(r2 / "stage05" / "models.pt")
    assert sorted(healed["models"]) == sorted([rule.lo_label, rule.hi_label, CELLS3[1]])
    assert healed["fallback_parents"] == []


def test_a_routing_mismatch_fails_without_outputs(run, tmp_path, monkeypatch):
    import stage04b_iterate_active_learning as s4b

    from lib import evaluate as ev

    real = ev.evaluate_model

    def skewed(*a, **kw):
        out = real(*a, **kw)
        out.n_routing_mismatch = 3
        return out

    monkeypatch.setattr(ev, "evaluate_model", skewed)
    out = tmp_path / "mismatch"
    argv = ["stage04b", "--models-dir", run / "round_0" / "stage05"]
    argv += ["--enriched-csv-dir", run / "round_1" / "stage04", "--out-dir", out]
    argv += ["--arch", "gfx950", "--config-yaml", run / "config.yaml", "--quiet"]
    monkeypatch.setattr(sys, "argv", [str(a) for a in argv])
    assert s4b.main() == 1
    assert not (out / "decisions.json").exists()
    assert not (out / "retrain_cells.txt").exists()


def test_final_round_never_splits(run):
    d = stage04b(
        run,
        run / "final",
        "--min-cell-gemms",
        20,
        "--sel-eff-threshold",
        1.01,
        "--split-floor",
        1.01,
        "--split-after-attempts",
        0,
        "--final-round",
    )
    assert d["cells_split"] == []
    assert any("final_round_no_split" in r["reason"] for r in d["per_cell"])


def test_held_out_contexts_are_reported_per_context(run, tmp_path):
    from test_stage05_train import write_context_round

    shapes = shapes_for(21, 6, CELLS3[:2])
    held = write_context_round(tmp_path / "holdout", shapes, 32, "default")
    data = tmp_path / "round_1c"
    write_round(data, shapes_for(11, 45, CELLS3))
    ctx = write_context_round(tmp_path / "ctx", shapes_for(12, 5, CELLS3), 64, "auto")
    for i, path in enumerate(sorted(ctx.glob("chunk_*.csv"))):
        path.rename(data / f"chunk_9{i:03d}.csv")
    d = run_stage(
        "stage04b_iterate_active_learning.py",
        "--models-dir",
        run / "round_0" / "stage05",
        "--enriched-csv-dir",
        data,
        "--out-dir",
        tmp_path / "out",
        "--arch",
        "gfx950",
        "--config-yaml",
        run / "config.yaml",
        "--weight-dtype",
        "bf16",
        "--round-idx",
        1,
        "--quiet",
        "--context-holdout-dir",
        held,
    )
    d = json.loads((tmp_path / "out" / "decisions.json").read_text())
    agg = d["aggregate"]
    assert set(agg["context_holdout"]) == {"32/default"}
    assert agg["context_holdout"]["32/default"]["n_gemms"] == len(set(shapes))
    assert set(agg["by_context"]) == {"0/default", "64/auto"}
    assert agg["by_context"]["64/auto"]["n_gemms"] == 15
    assert sum(r["n_eval"] for r in d["per_cell"]) == 135 + 15
    plain = stage04b(run, tmp_path / "plain", "--min-cell-gemms", 20)
    assert "context_holdout" not in plain["aggregate"]
    assert "by_context" not in plain["aggregate"]


def test_rows_of_another_build_are_read_at_their_kernels(run, tmp_path):
    """round_1's rows recorded as enriched with build A, scored against
    build B, which numbers every kernel otherwise, decide as build A's rows
    against build A."""
    import shutil

    from test_kernel_identity import pool_library, record_rows
    from test_stage05_train import kernel_pool

    stem = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx950"
    ident_a = pool_library(tmp_path / "a", kernel_pool(), stem=stem)
    pool_library(tmp_path / "b", kernel_pool(), offset=1000, stem=stem)
    moved = record_rows(
        shutil.copytree(run / "round_1" / "stage04", tmp_path / "rows"), ident_a, stem
    )
    got = {}
    for name, rows in (("a", run / "round_1" / "stage04"), ("b", moved)):
        args = stage04b_args(run, tmp_path / f"out_{name}", "--quiet")
        args[args.index("--enriched-csv-dir") + 1] = rows
        args += ["--library-dir", tmp_path / name, "--library-stem", stem]
        run_stage("stage04b_iterate_active_learning.py", *args)
        d = json.loads((tmp_path / f"out_{name}" / "decisions.json").read_text())
        got[name] = {
            r["cell"]: (r["decision"], r["n_eval"], r["sel_eff_new"])
            for r in d["per_cell"]
        }
    assert got["b"] == got["a"] and got["a"]
