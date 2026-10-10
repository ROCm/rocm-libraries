# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The audit set and the final retrain inside one `--mode full` command:
audit.build (before: the run builds the audit set it points at before its
first round; parallel: beside the rounds, joined at the end; off),
audit.v2 (each round's bundle measured under protocol v2 and the sustained
composite before stage06 scores it; what is missing at the end),
audit.official (the protocol the ship decision and the per-cell gate take),
and full.final_retrain (every cell on every round's data, shipped when the
audit set finds it not worse, then validated). With stub stages and stub
audit scripts, and end to end on the fake bench."""

import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest
import yaml

import bench_fakes as bf
import fake_hipblaslt as fh
import run_orchestrator as ro
from lib import audit as al
from lib import audit_sustained as asus
from lib import audit_v2 as v2
from lib import procs
from lib import run_bundles as rb
from lib import run_layout
from lib.bench_yaml import parse_shapes_yaml
import test_run_orchestrator as tro
from test_run_orchestrator import (
    LARGE,
    TINY,
    arg,
    by_stage,
    calls,
    layout_of_the_past,
    make_config,
    run,
)

stubs = tro.stubs
env = tro.env

AUDIT_EVAL_STUB = r"""
import hashlib, json, math, os, sys
from pathlib import Path

argv = sys.argv[1:]


def values(name):
    return [argv[i + 1] for i, a in enumerate(argv) if a == name]


if os.environ.get("STUB_LOG"):
    with open(os.environ["STUB_LOG"], "a") as f:
        f.write(json.dumps({"stage": "audit_eval", "argv": argv}) + "\n")
scores = json.loads(os.environ.get("STUB_AUDIT", "{}"))
shapes = [[64, 64, 64, 1], [512, 512, 512, 1], [4096, 4096, 4096, 1],
          [8192, 64, 8192, 1]]


def sha256(model):
    data = Path(model).read_bytes() if Path(model).is_file() else model.encode()
    return hashlib.sha256(data).hexdigest()


def record(what, model):
    v = scores.get(what, scores.get("default", 0.9))
    vals = [float(x) for x in v] if isinstance(v, list) else [float(v)] * len(shapes)
    se = math.exp(sum(math.log(x) for x in vals) / len(vals))
    group = {"per_gemm": [s + [x] for s, x in zip(shapes, vals)], "cell_mean": se,
             "cell_mean_ci": [se, se], "n_gemms": 4, "n_cells": 4,
             "n_cells_below_target": 0}
    rec = {"status": "ok", "status_text": "complete", "what": what,
           "model": model, "groups": {"exclusive": group},
           "audit_dir": values("--audit-dir")[0],
           "model_sha256": sha256(model),
           "official": (values("--official") or [None])[0]}
    if os.environ.get("STUB_AUDIT_V2"):
        rec["v2"] = {"status": "ok", "protocol": "v2",
                     "groups": {"exclusive": group}}
        if not os.environ.get("STUB_AUDIT_NO_SUSTAINED"):
            rec["sustained"] = {"status": "ok", "protocol": "sustained",
                                "groups": {"exclusive": group}}
    return rec


recs = [record(Path(rd).name, str(Path(rd) / "stage05" / "models.pt"))
        for rd in values("--round-dir")]
recs += [record(Path(f).name, f) for f in values("--model-file")]
out = Path(values("--out")[0])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(recs[0] if len(recs) == 1 else {"models": recs}))
"""

AUDIT_V2_STUB = r"""
import json, os, sys

if os.environ.get("STUB_LOG"):
    with open(os.environ["STUB_LOG"], "a") as f:
        f.write(json.dumps({"stage": "audit_v2", "argv": sys.argv[1:]}) + "\n")
sys.exit(int(os.environ.get("STUB_RC_audit_v2", "0")))
"""


STAGE06_AUDIT = r"""
if "--audit-dir" in argv:
    import hashlib
    model = str(Path(opt("--train-dir")) / "models.pt")
    import math
    scores = json.loads(os.environ.get("STUB_AUDIT", "{}"))
    v = scores.get(Path(model).parent.parent.name, scores.get("default", 0.9))
    vals = [float(x) for x in v] if isinstance(v, list) else [float(v)]
    se = math.exp(sum(math.log(x) for x in vals) / len(vals))
    group = {"per_gemm": [[64, 64, 64, 1, se]], "cell_mean": se,
             "cell_mean_ci": [se, se], "gemm_geomean": se, "n_gemms": 1,
             "n_cells": 1, "n_cells_below_target": 0,
             "per_cell": {"Tiny|Tiny|TinyK|Bnone": {"n": 1, "n_miss": 0,
                                                   "sel_eff": se}}}
    rec = {"status": "ok", "what": "stage06", "model": model,
           "groups": {"exclusive": group}, "audit_dir": opt("--audit-dir"),
           "model_sha256": hashlib.sha256(Path(model).read_bytes()).hexdigest()}
    if os.environ.get("STUB_AUDIT_V2"):
        rec["v2"] = {"status": "ok", "groups": {"exclusive": group}}
        if not os.environ.get("STUB_AUDIT_NO_SUSTAINED"):
            rec["sustained"] = {"status": "ok", "groups": {"exclusive": group}}
    (o / "audit.json").write_text(json.dumps(rec))
    deployed = json.loads((o / "deploy_record.json").read_text())
    deployed["sha256"] = rec["model_sha256"]
    if ("--cell-gate-baseline" in argv and "--cell-gate-record-only" not in argv
            and os.environ.get("STUB_CELL_GATE") == "fail"):
        deployed.update(deployed=False, not_deployed="the per-cell gate keeps it out")
    (o / "deploy_record.json").write_text(json.dumps(deployed))
"""


@pytest.fixture
def audit_stubs(tmp_path, monkeypatch, stubs):
    """Stub audit_eval.py / audit_v2.py scripts, a stage06 stub that also
    writes its audit record, and an audit set the stub runs see as
    complete."""
    s06 = stubs / ro.STAGE_DEFS["stage06"].script
    s06.write_text(s06.read_text() + STAGE06_AUDIT)
    d = tmp_path / "audit_scripts"
    d.mkdir()
    (d / "audit_eval.py").write_text(AUDIT_EVAL_STUB)
    (d / "audit_v2.py").write_text(AUDIT_V2_STUB)
    monkeypatch.setattr(ro, "AUDIT_EVAL_SCRIPT", d / "audit_eval.py")
    monkeypatch.setattr(ro, "AUDIT_V2_SCRIPT", d / "audit_v2.py")
    monkeypatch.setattr(ro, "audit_status", lambda cfg, paths: ("ok", None))
    for k in (
        "STUB_AUDIT",
        "STUB_AUDIT_V2",
        "STUB_AUDIT_NO_SUSTAINED",
        "STUB_RC_audit_v2",
        "STUB_CELL_GATE",
    ):
        monkeypatch.delenv(k, raising=False)
    return stubs


def audited_config(tmp_path, **overrides):
    return make_config(
        tmp_path,
        **{"audit.dir": str(tmp_path / "audit"), "audit.v2": "off", **overrides},
    )


def values(entry, name):
    a = entry["argv"]
    return [a[i + 1] for i, x in enumerate(a) if x == name]


# ── config ───────────────────────────────────────────────────────────────────


def _cfg(tmp_path, **sections):
    cfg = yaml.safe_load(make_config(tmp_path).read_text())
    for dotted, value in sections.items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return cfg


def test_audit_build_keys_validate(tmp_path):
    cfg = _cfg(tmp_path)
    assert ro.cfg_get(cfg, "audit.build") == "before"
    assert ro.cfg_get(cfg, "audit.v2") == "rounds"
    assert ro.audit_devices(cfg) == [0, 1]
    a = str(tmp_path / "audit")
    for sections, needle in (
        ({"audit.build": "parallel"}, "list its devices in audit.devices"),
        (
            {"audit.build": "parallel", "audit.devices": [1, 2]},
            "audit.devices [1] are runtime.devices too",
        ),
        ({"audit.devices": [3, 3]}, "audit.devices: duplicate device indices"),
        ({"audit.build": "later"}, "audit.build: 'later' is not one of"),
        ({"audit.v2": "some"}, "audit.v2: 'some' is not one of"),
        ({"audit.official": "v1"}, "audit.official: 'v1' is not one of"),
        (
            {"audit.sustained": False},
            "audit.official sustained takes the official exclusive number",
        ),
    ):
        errors, _ = ro.validate_config(_cfg(tmp_path, **{"audit.dir": a, **sections}))
        assert any(needle in e for e in errors), (sections, errors)
    ok = _cfg(tmp_path, **{"audit.dir": a, "audit.build": "parallel"})
    ok["audit"]["devices"] = [4, 5]
    assert ro.validate_config(ok)[0] == []
    assert ro.audit_devices(ok) == [4, 5]
    assert ro.validate_config(_cfg(tmp_path, **{"audit.build": "parallel"}))[0] == []


def test_a_final_retrain_gated_on_the_audit_set_needs_one(tmp_path):
    errors, _ = ro.validate_config(
        _cfg(tmp_path, **{"full.final_retrain.enabled": True})
    )
    assert any("set audit.dir, or gate: none" in e for e in errors)
    cfg = _cfg(
        tmp_path,
        **{"full.final_retrain.enabled": True, "full.final_retrain.gate": "none"},
    )
    errors, warnings = ro.validate_config(cfg)
    assert errors == []
    assert any("without a comparison on held-out GEMMs" in w for w in warnings)


def test_the_recipe_overrides_train_keys_and_adds_stage05_options(tmp_path):
    recipe = {
        "unmeasured_samples": 0,
        "served_candidates": True,
        "lam": 0.4,
        "no_teacher": True,
        "n_estimators": 50,
        "tie_tolerance": None,
        "T": 3.0,
    }
    cfg = _cfg(tmp_path, **{"train.catalog": "v3", "full.final_retrain.recipe": recipe})
    assert ro.validate_config(cfg)[0] == []
    overrides, argv, errors = ro.final_retrain_recipe(cfg)
    assert errors == []
    assert overrides == {"unmeasured_samples": 0, "served_candidates": True}
    assert argv == [
        "--lam",
        "0.4",
        "--no-teacher",
        "--n-estimators",
        "50",
        "--T",
        "3.0",
    ]


@pytest.mark.parametrize(
    "recipe, needle",
    [
        ({"catalog": "v3"}, "the final retrain keeps train.catalog"),
        ({"attribute_spec": []}, "the final retrain keeps train.attribute_spec"),
        # stage06 deploys every bundle with the run's trained-key gate, which
        # the final retrain's gate must therefore score it with too
        ({"trained_key_gate": False}, "the final retrain keeps train.trained_key_gate"),
        (
            {"no_trained_key_gate": True},
            "the final retrain keeps train.trained_key_gate",
        ),
        ({"output_dir": "x"}, "the orchestrator sets this stage05 option"),
        ({"only_cells": "x"}, "the orchestrator sets this stage05 option"),
        (
            {"pooled_pretrain": True},
            "neither a train key nor an option of stage05 (--pooled-pretrain)",
        ),
        ({"no_teacher": 3}, "a flag of stage05 takes true or false"),
        ({"lam": [0.1, 0.2]}, "stage05 takes one value for this option"),
        ({"lam": "half"}, "stage05 refuses it"),
        ({"unmeasured_samples": -1}, "recipe.unmeasured_samples: must be >= 0"),
        ({"lam-x": 1}, "'lam-x' is not an option name"),
        ({"Lam": 1}, "neither a train key nor an option of stage05 (--Lam)"),
    ],
)
def test_the_recipe_is_checked_before_anything_runs(tmp_path, recipe, needle):
    cfg = _cfg(tmp_path, **{"full.final_retrain.recipe": recipe})
    errors, _ = ro.validate_config(cfg)
    assert any(needle in e for e in errors), errors


@pytest.mark.parametrize(
    "train, recipe, needle",
    [
        ({"qat_frac": 0.1}, {"recipe": "pooled_ft"}, "train.qat_frac with"),
        ({"boundary_margin": 0.5}, {"recipe": "pooled_ft"}, "train.boundary_margin"),
        ({}, {"recipe": "pooled_ft", "validation_frac": 0}, "train.validation_frac"),
        ({}, {"pseudo_negatives": "targeted"}, "targeted draws keys of a v3 model"),
        (
            {"pseudo_negatives": "random", "unmeasured_samples": 8},
            {"unmeasured_samples": 0},
            "train.pseudo_negatives random draws at least one",
        ),
        ({"unmeasured_samples": 8}, {"pseudo_negatives": "none"}, "none draws none"),
        ({}, {"launch_keys": True}, "launch keys need train.catalog v3"),
    ],
)
def test_the_final_retrains_train_section_is_checked_as_the_runs(
    tmp_path, train, recipe, needle
):
    sections = {f"train.{k}": v for k, v in train.items()}
    cfg = _cfg(tmp_path, **sections, **{"full.final_retrain.recipe": recipe})
    (error,) = ro.validate_config(cfg)[0]
    assert error.startswith("full.final_retrain.recipe: ") and needle in error
    assert ro.validate_config(_cfg(tmp_path, **sections))[0] == []


def test_an_error_of_the_runs_train_section_is_named_once(tmp_path):
    bad = {"train.recipe": "pooled_ft", "train.qat_frac": 0.1}
    for recipe in ({"lam": 0.3}, {"fine_tune_lr": 0.002}):
        errors, _ = ro.validate_config(
            _cfg(tmp_path, **bad, **{"full.final_retrain.recipe": recipe})
        )
        assert errors == ["train.qat_frac with train.recipe pooled_ft: per_leaf only"]


def test_a_pooled_recipe_warns_of_the_per_leaf_schedule_it_ignores(tmp_path):
    schedule = {"train.patience": 8, "train.lr": 0.001, "train.epochs": 200}
    _, warnings = ro.validate_config(_cfg(tmp_path, **schedule))
    assert not [w for w in warnings if "per-leaf schedule" in w]
    _, warnings = ro.validate_config(
        _cfg(tmp_path, **schedule, **{"train.recipe": "pooled_ft"})
    )
    (w,) = [w for w in warnings if "per-leaf schedule" in w]
    assert w.startswith("train.lr, train.patience: train.recipe pooled_ft")
    final = {
        **schedule,
        "train.epochs": 50,
        "full.final_retrain.recipe": {"recipe": "pooled_ft"},
    }
    _, warnings = ro.validate_config(_cfg(tmp_path, **final))
    (w,) = [w for w in warnings if "per-leaf schedule" in w]
    assert w.startswith(
        "full.final_retrain.recipe: train.epochs, train.lr, train.patience: "
    )


# ── stub runs: protocol v2 and the final retrain ─────────────────────────────


def test_each_round_measures_its_bundle_under_v2_before_stage06(
    tmp_path, audit_stubs, env, monkeypatch
):
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    monkeypatch.setenv("STUB_AUDIT_V2", "1")
    cfg = audited_config(tmp_path, **{"audit.v2": "rounds"})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    log = calls(env)
    order = [c["stage"] for c in log]
    v2_calls = [c for c in log if c["stage"] == "audit_v2"]
    assert len(v2_calls) == 3
    for i, c in enumerate(v2_calls):
        rd = run_dir / f"round_{i}"
        assert c["argv"][0] == "run" and "--name-with-sha" in c["argv"]
        assert arg(c, "--devices") == "0,1"
        assert arg(c, "--config").startswith(str(run_dir / "configs"))
        assert values(c, "--model") == [f"run.round_{i}={rd / 'stage05' / 'models.pt'}"]
    # round_0 has no stage06: the audit evaluation of its bundle follows v2
    first = order.index("audit_v2")
    assert order[first + 1] == "audit_eval"
    s06 = [i for i, s in enumerate(order) if s == "stage06"]
    v2s = [i for i, s in enumerate(order) if s == "audit_v2"]
    assert v2s[1] < s06[0] < v2s[2] < s06[1]
    # every round's own record has v2: the end of the run adds nothing
    assert [c for c in log if "rescore" in (arg(c, "--out") or "")] == []
    assert rb.audit_result_files(run_dir) == []
    n = len(log)
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    assert [c["stage"] for c in calls(env)[n:]] == []


def test_v2_final_measures_the_final_and_the_shipped_bundle_at_the_end(
    tmp_path, audit_stubs, env, monkeypatch
):
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.5")
    cfg = audited_config(tmp_path, **{"audit.v2": "final"})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "initial", "--deploy") == 0
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    log = calls(env)
    (c,) = [c for c in log if c["stage"] == "audit_v2"]
    assert log.index(c) > max(i for i, x in enumerate(log) if x["stage"] == "stage06")
    shipped = run_dir / "round_0"
    assert [m.split("=")[0] for m in values(c, "--model")] == [
        "run.round_0",
        "run.round_2",
    ]
    assert values(c, "--model")[0].endswith(str(shipped / "stage05" / "models.pt"))
    # their records lack v2: both are scored again for the report
    (rescore,) = [c for c in log if "rescore" in (arg(c, "--out") or "")]
    assert values(rescore, "--round-dir") == [str(shipped), str(run_dir / "round_2")]
    for r in (0, 2):
        rec = json.loads((run_dir / "audit_results" / f"round_{r}.json").read_text())
        assert rec["what"] == f"round_{r}"
    assert not (run_dir / "audit_results" / "rescore.json").exists()
    assert rb.audit_result_files(run_dir) == [
        run_dir / "audit_results" / "round_0.json",
        run_dir / "audit_results" / "round_2.json",
    ]


def test_a_failed_v2_bench_never_stops_the_run(tmp_path, audit_stubs, env, monkeypatch):
    monkeypatch.setenv("STUB_RC_audit_v2", "1")
    cfg = audited_config(tmp_path, **{"audit.v2": "rounds", "full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    st = by_stage(calls(env))
    assert len(st["audit_v2"]) == 3
    assert [values(c, "--model")[0].split("=")[0] for c in st["audit_v2"][2:]] == [
        "run.round_0"
    ]


def _final_config(tmp_path, **overrides):
    return audited_config(
        tmp_path,
        **{
            "full.final_retrain.enabled": True,
            "full.final_retrain.recipe": {"unmeasured_samples": 0, "lam": 0.4},
            "train.unmeasured_samples": 128,
            "stage07.parity_sample_trained": 0,
            **overrides,
        },
    )


@pytest.mark.parametrize("planner", ["stage02", "stage01b"])
def test_the_final_retrain_reads_the_plans_where_its_run_wrote_them(
    tmp_path, audit_stubs, env, monkeypatch, planner
):
    """A run written before the planner became stage02 resumes into its
    final retrain, which trains with the plans of every round in stage01b/,
    as one of this pipeline trains with those in stage02/."""
    cfg = _final_config(
        tmp_path, **{"planner.enabled": True, "planner.mode": "planned"}
    )
    run_dir = tmp_path / "run"
    rounds = [run_dir / f"round_{i}" for i in range(3)]
    with layout_of_the_past(planner):
        assert run(audit_stubs, cfg, run_dir, "--mode", "initial") == 0
        assert run(audit_stubs, cfg, run_dir, "--mode", "active") == 0
    for rd in rounds[:2]:
        for plan in (rd / planner / "plan.json", rd / planner / "seed" / "plan.json"):
            plan.parent.mkdir(parents=True, exist_ok=True)
            plan.write_text("{}")
    env.write_text("")
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    st = by_stage(calls(env))
    assert [arg(c, "--out-dir") for c in st["stage02"]] == [str(rounds[2] / planner)]
    final = st["stage05"][-1]
    assert arg(final, "--output-dir") == str(run_dir / ro.FINAL_RETRAIN_DIR / "stage05")
    assert values(final, "--plan") == [
        str(rd / planner / name)
        for rd in rounds[:2]
        for name in ("plan.json", "seed/plan.json")
    ]
    assert rb.final_retrain(run_dir) == run_dir / ro.FINAL_RETRAIN_DIR
    assert run_layout.of(run_dir).planner == planner
    assert not list(run_dir.rglob("stage01b" if planner == "stage02" else "stage02"))


def test_the_final_retrain_trains_every_cell_on_every_rounds_data_and_ships(
    tmp_path, audit_stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    monkeypatch.setenv(
        "STUB_AUDIT",
        json.dumps({"models.pt": 0.95, ro.FINAL_RETRAIN_DIR: 0.96, "default": 0.9}),
    )
    cfg = _final_config(tmp_path)
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    rounds = [run_dir / f"round_{i}" for i in range(3)]
    final = run_dir / ro.FINAL_RETRAIN_DIR
    st = by_stage(calls(env))
    s05 = st["stage05"][-1]
    assert arg(s05, "--output-dir") == str(final / "stage05")
    assert values(s05, "--enriched-csv-dir") == [str(rd / "stage04") for rd in rounds]
    assert values(s05, "--prior-round-dir") == [str(rd) for rd in rounds]
    assert (
        "--only-cells" not in s05["argv"] and "--current-round-dir" not in s05["argv"]
    )
    assert values(s05, "--unmeasured-samples") == ["0"]
    assert s05["argv"][-2:] == ["--lam", "0.4"]
    assert arg(s05, "--off-list-rebench") is None
    rec = json.loads((final / ro.FinalRetrain.RECORD).read_text())
    assert list(rec["rounds"]) == ["round_0", "round_1", "round_2"]
    assert rec["recipe"] == {"unmeasured_samples": 0, "lam": 0.4}
    ev = [c for c in st["audit_eval"] if arg(c, "--out") == str(final / "audit.json")]
    (ev,) = ev
    assert values(ev, "--round-dir") == [str(rounds[2])]
    assert values(ev, "--model-file") == [str(final / "stage05" / "models.pt")]
    assert values(ev, "--trained-round-dir") == [str(rd) for rd in rounds]
    dec = json.loads((final / ro.FinalRetrain.DECISION).read_text())
    assert dec["accepted"] and dec["gate"] == "not-worse" and dec["protocol"] == "v1"
    assert dec["baseline"]["round"] == "round_2" and dec["baseline"]["deployed"]
    assert dec["change"]["cell_mean_ratio"] == pytest.approx(0.95 / 0.9)
    assert dec["change"]["after"] == ro.FINAL_RETRAIN_DIR
    assert dec["selected_by_audit"] and "optimistic bias" in dec["selection_note"]
    d06 = st["stage06"][-1]
    assert arg(d06, "--train-dir") == str(final / "stage05")
    assert arg(d06, "--out-dir") == str(final / "stage06")
    assert "--gate-csv-dir" not in d06["argv"]
    assert values(d06, "--audit-trained-round-dir") == [str(rd) for rd in rounds]
    assert rb.final_retrain(run_dir) == final
    assert ro.run_bundles(run_dir) == rounds + [final]
    s07, s08 = st["stage07"][-1], st["stage08"][-1]
    assert arg(s07, "--deploy-record") == str(final / "stage06" / "deploy_record.json")
    assert arg(s08, "--train-dir") == str(final / "stage05")
    breakdown = json.loads((run_dir / "held_out_sel_eff_breakdown.json").read_text())
    assert breakdown["shipped"]["round"] == ro.FINAL_RETRAIN_DIR
    assert breakdown["shipped"]["train_dir"] == str(final / "stage05")
    assert breakdown["source"] == "audit" and breakdown["final_bundle"] is None
    assert breakdown["held_out_sel_eff"] == pytest.approx(0.96)
    (head,) = breakdown["measurements"]
    assert (head["cell_mean"], head["protocol"]) == (pytest.approx(0.96), "v1")
    retrain = breakdown["final_retrain"]
    assert retrain["ships"] is True and retrain["not_shipped"] is None
    assert retrain["trained_on"] == [rd.name for rd in rounds]
    assert retrain["rounds_after"] == [] and retrain["selected_by_audit"]
    out = capsys.readouterr().out
    assert (
        "shipped bundle final_retrain (every cell retrained on the data of 3 "
        "rounds): audit cell mean 96.00%"
    ) in out
    assert "the audit set chose which of the final retrain and round_2" in out
    assert "shipped bundle round_2" not in out
    files = rb.audit_result_files(run_dir)
    assert files == [final / "stage06" / "audit.json", final / "audit.json"]

    n = len(calls(env))
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    assert [c["stage"] for c in calls(env)[n:]] == []
    n = len(calls(env))
    assert run(audit_stubs, cfg, run_dir, "--mode", "validate") == 0
    st = by_stage(calls(env)[n:])
    assert arg(st["stage08"][0], "--train-dir") == str(final / "stage05")
    assert arg(st["stage07"][0], "--deploy-record") == str(
        final / "stage06" / "deploy_record.json"
    )


def test_every_reader_describes_the_shipped_final_retrain(
    tmp_path, audit_stubs, env, monkeypatch
):
    import make_report as mr
    import report_sections as rs
    from lib import run_analysis as ra

    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    monkeypatch.setenv("STUB_CELLS", json.dumps([TINY]))
    monkeypatch.setenv(
        "STUB_AUDIT",
        json.dumps({"models.pt": 0.95, ro.FINAL_RETRAIN_DIR: 0.96, "default": 0.9}),
    )
    cfg = _final_config(tmp_path)
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    rounds = [run_dir / f"round_{i}" for i in range(3)]
    final = run_dir / ro.FINAL_RETRAIN_DIR

    model = ra.final_model(run_dir, want_bytes=False)
    assert (model.train_round, model.deploy_round) == (final.name, final.name)
    assert model.deployed and model.trained_rounds == [rd.name for rd in rounds]
    sets = ra.model_benches(run_dir, model, with_plans=False)
    assert sets["training"].dirs == [rd / "stage04" for rd in rounds]
    assert not sets["held_out"] and not sets["gate"]
    assert [rd for rd, _ in ra.deploy_records(run_dir)] == [*rounds[1:], final]

    recs = rs.audit_records(run_dir)
    assert [(r, role) for r, role, _ in recs][-1] == (final.name, "candidate")
    assert recs[-1][2]["source"] == str(final / "stage06" / "audit.json")

    sb = mr.subject_bundle(str(run_dir))
    assert (sb["kind"], sb["bundle"]["round"], sb["source"]) == (
        "deployed",
        final.name,
        "audit",
    )
    assert mr._bundle_name(sb) == "deployed bundle (final_retrain)"
    assert mr._train_round(sb) == final.name
    cells = mr.final_cells(str(run_dir), {}, upto=final.name)
    assert [c["cell"] for c in cells] == [TINY]
    reasons = mr.missing_reasons(str(run_dir), yaml.safe_load(cfg.read_text()), sb)
    assert reasons["gate"].startswith("the final retrain has no gate GEMMs")
    gates = mr.deploy_gates(str(run_dir))
    assert gates[-1]["round"] == final.name and gates[-1]["deployed"]
    assert gates[-1]["reason"].startswith("decided on the audit set: not worse")
    state = mr.audit_state(str(run_dir), yaml.safe_load(cfg.read_text()), sb, recs)
    assert state["head"][0] == final.name
    assert state["change"][0] == "round_2"

    out = tmp_path / "report.html"
    mr.build(str(run_dir), str(out), engine=False, bench=False)
    page = out.read_text()
    assert "deployed bundle (final_retrain)" in page
    assert "deployed bundle (round_2)" not in page
    assert "extra model final_retrain" not in page
    assert "the audit set chose which of the final retrain and round_2" in page
    assert "<b>audit</b> = the final retrain's audit record" in page


def test_a_worse_final_retrain_does_not_ship(tmp_path, audit_stubs, env, monkeypatch):
    monkeypatch.setenv("STUB_AUDIT", json.dumps({"models.pt": 0.85, "default": 0.9}))
    monkeypatch.setenv("STUB_AUDIT_V2", "1")
    cfg = _final_config(tmp_path, **{"audit.v2": "rounds"})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    final = run_dir / ro.FINAL_RETRAIN_DIR
    dec = json.loads((final / ro.FinalRetrain.DECISION).read_text())
    assert dec["accepted"] is False and dec["protocol"] == "sustained"
    assert dec["official"] == "sustained"
    assert dec["final"]["v2"]["cell_mean"] == dec["final"]["cell_mean"]
    assert dec["reason"].startswith("worse than round_2's bundle on the audit set")
    assert not (final / "stage06").exists()
    st = by_stage(calls(env))
    assert all(arg(c, "--out-dir") != str(final / "stage06") for c in st["stage06"])
    assert arg(st["stage08"][-1], "--train-dir") == str(run_dir / "round_2" / "stage05")
    assert ro.run_bundles(run_dir)[-1] == run_dir / "round_2"
    (v2_final,) = [
        c
        for c in st["audit_v2"]
        if values(c, "--model")[0].startswith("run.final_retrain=")
    ]
    assert values(v2_final, "--model") == [
        f"run.final_retrain={final / 'stage05' / 'models.pt'}"
    ]
    breakdown = json.loads((run_dir / "held_out_sel_eff_breakdown.json").read_text())
    assert breakdown["shipped"]["round"] == "round_2"
    retrain = breakdown["final_retrain"]
    assert retrain["ships"] is False and retrain["selected_by_audit"]
    assert retrain["not_shipped"].startswith("worse than round_2's bundle")


def test_the_official_protocol_decides_and_v2_stands_alongside(
    tmp_path, audit_stubs, env, monkeypatch, capsys
):
    """The final retrain's decision and its per-cell gate judge under
    audit.official (the sustained composite by default), which every
    audit_eval and audited stage06 is told, with v2's numbers beside it; a
    decision made under another official protocol is made again."""
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    monkeypatch.setenv("STUB_AUDIT_V2", "1")
    monkeypatch.setenv("STUB_AUDIT", json.dumps({"models.pt": 0.95, "default": 0.9}))
    cfg = _final_config(tmp_path, **{"audit.v2": "rounds"})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    st = by_stage(calls(env))
    assert all(arg(c, "--official") == "sustained" for c in st["audit_eval"])
    audited = [c for c in st["stage06"] if "--audit-dir" in c["argv"]]
    assert audited
    assert all(arg(c, "--audit-official") == "sustained" for c in audited)
    assert all("--no-sustained" not in c["argv"] for c in st["audit_v2"])
    final = run_dir / ro.FINAL_RETRAIN_DIR
    dec = json.loads((final / ro.FinalRetrain.DECISION).read_text())
    assert (dec["official"], dec["protocol"]) == ("sustained", "sustained")
    assert dec["accepted"] and dec["cell_gate"]["official"] == "sustained"
    assert dec["cell_gate"]["groups"]["exclusive"]["protocol"] == "sustained"
    assert dec["final"]["v2"]["cell_mean"] == pytest.approx(0.95)
    assert dec["baseline"]["v2"]["cell_mean"] == pytest.approx(0.9)
    assert "v2 alongside 95.00%" in capsys.readouterr().out
    cfg2 = _final_config(tmp_path, **{"audit.v2": "rounds", "audit.official": "v2"})
    n = len(calls(env))
    assert run(audit_stubs, cfg2, run_dir, "--mode", "full") == 0
    assert "audit_eval" in by_stage(calls(env)[n:])
    dec = json.loads((final / ro.FinalRetrain.DECISION).read_text())
    assert (dec["official"], dec["protocol"]) == ("v2", "v2") and dec["accepted"]
    assert "v2" not in dec["final"]
    n = len(calls(env))
    assert run(audit_stubs, cfg2, run_dir, "--mode", "full") == 0
    assert "audit_eval" not in by_stage(calls(env)[n:])


def test_a_round_record_lacking_the_composite_is_measured_again_unless_off(
    tmp_path, audit_stubs, env, monkeypatch
):
    """With audit.sustained (default) a bundle's record is complete only
    with the composite: one whose measurement has none is benched and
    scored again at the end of the run. With it off the benches leave the
    composite alone and v2 completes a record."""
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    monkeypatch.setenv("STUB_AUDIT_V2", "1")
    monkeypatch.setenv("STUB_AUDIT_NO_SUSTAINED", "1")
    cfg = audited_config(tmp_path, **{"audit.v2": "rounds"})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    log = calls(env)
    v2_calls = [c for c in log if c["stage"] == "audit_v2"]
    assert len(v2_calls) == 4
    assert [m.split("=")[0] for m in values(v2_calls[-1], "--model")] == [
        f"run.round_{i}" for i in range(3)
    ]
    assert [c for c in log if "rescore" in (arg(c, "--out") or "")]

    off = tmp_path / "off"
    off.mkdir()
    cfg = audited_config(
        off, **{"audit.v2": "rounds", "audit.sustained": False, "audit.official": "v2"}
    )
    n = len(calls(env))
    assert run(audit_stubs, cfg, off / "run", "--mode", "full") == 0
    log = calls(env)[n:]
    v2_calls = [c for c in log if c["stage"] == "audit_v2"]
    assert len(v2_calls) == 3
    assert all("--no-sustained" in c["argv"] for c in v2_calls)
    assert [c for c in log if "rescore" in (arg(c, "--out") or "")] == []
    assert all(arg(c, "--official") == "v2" for c in log if c["stage"] == "audit_eval")


def test_with_nothing_deployed_the_final_retrain_is_compared_with_the_final_round(
    tmp_path, audit_stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_AUDIT", json.dumps({"models.pt": 0.5, "default": 0.9}))
    cfg = _final_config(
        tmp_path, **{"deploy.on_final_round": False, "full.total_rounds": 2}
    )
    run_dir = tmp_path / "run"
    # Nothing is deployed and the final retrain does not ship, so stage 7 has
    # no bundle to validate: the run ends with 2 (validation has not passed).
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 2
    final = run_dir / ro.FINAL_RETRAIN_DIR
    (ev,) = [
        c
        for c in by_stage(calls(env))["audit_eval"]
        if arg(c, "--out") == str(final / "audit.json")
    ]
    assert values(ev, "--round-dir") == [str(run_dir / "round_1")]
    dec = json.loads((final / ro.FinalRetrain.DECISION).read_text())
    assert dec["accepted"] is False
    assert dec["reason"].startswith(
        "worse than round_1's bundle (not deployed) on the audit set"
    )
    assert dec["baseline"]["round"] == "round_1"
    assert dec["baseline"]["deployed"] is False
    assert dec["final"]["cell_mean"] == pytest.approx(0.5)
    assert rb.final_retrain(run_dir) is None and not (final / "stage06").exists()
    assert "this run deployed no bundle" in capsys.readouterr().err
    status = json.loads((run_dir / ro.RUN_STATUS).read_text())
    assert status["final_retrain"]["status"] == "rejected"
    assert status["validation"]["stage07"]["status"] == "not_run"
    assert status["deployed"] is None and not status["complete"]

    monkeypatch.setenv("STUB_AUDIT", json.dumps({"models.pt": 0.95, "default": 0.9}))
    run2 = tmp_path / "run2"
    assert run(audit_stubs, cfg, run2, "--mode", "full") == 0
    final2 = run2 / ro.FINAL_RETRAIN_DIR
    status = json.loads((run2 / ro.RUN_STATUS).read_text())
    assert status["final_retrain"]["status"] == "shipped" and status["complete"]
    assert status["deployed"]["round"] == ro.FINAL_RETRAIN_DIR
    dec = json.loads((final2 / ro.FinalRetrain.DECISION).read_text())
    assert dec["accepted"] and dec["reason"].startswith(
        "not worse than round_1's bundle (not deployed) on the audit set"
    )
    assert rb.final_retrain(run2) == final2
    assert "shipped bundle final_retrain" in capsys.readouterr().out


def test_a_final_retrain_that_adds_a_collapsed_pick_does_not_ship(
    tmp_path, audit_stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    monkeypatch.setenv("STUB_AUDIT_V2", "1")
    scores = {"models.pt": [1.0, 1.0, 1.0, 0.3], "default": [0.6, 0.6, 0.6, 0.9]}
    monkeypatch.setenv("STUB_AUDIT", json.dumps(scores))
    cfg = _final_config(tmp_path)
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    final = run_dir / ro.FINAL_RETRAIN_DIR
    dec = json.loads((final / ro.FinalRetrain.DECISION).read_text())
    assert dec["change"]["cell_mean_ratio"] > 1 and dec["accepted"] is False
    assert dec["cell_gate_settings"] == al.CELL_GATE_DEFAULTS
    cells = dec["cell_gate"]
    assert cells["passed"] is False and cells["n_tail"] == 1
    (tail,) = cells["groups"]["exclusive"]["tail"]
    assert tail["shape"] == [8192, 64, 8192, 1]
    assert (tail["before"], tail["after"]) == (0.9, 0.3)
    assert tail["confirmed"] and tail["decides"]
    assert dec["reason"].startswith("not worse than round_2's bundle on the audit set")
    assert (
        "the per-cell gate keeps it out: 1 group(s) with more new picks collapsing "
        "below 50.0% than it lifted: exclusive (1 new, 0 lifted: 8192x64x8192 "
        "(Large|Small|LargeK|Bnone) 30.00% (was 90.00%))"
    ) in dec["reason"]
    assert not (final / "stage06").exists()
    assert ro.run_bundles(run_dir)[-1] == run_dir / "round_2"
    assert "the per-cell gate keeps it out" in capsys.readouterr().err

    import make_report as mr

    gates = mr.deploy_gates(str(run_dir))
    assert gates[-1]["round"] == ro.FINAL_RETRAIN_DIR
    assert gates[-1]["cell_gate"] == "kept it out"
    (f,) = gates[-1]["findings"]
    assert (f["kind"], f["gemm"], f["deciding"]) == ("tail", "8192x64x8192", True)
    page_part = mr.deploy_gates_html(gates)
    assert "<h3>Per-cell gate findings</h3>" in page_part
    assert "pick collapsed below the tail floor" in page_part

    # a changed gate decides again, with the settings now
    cfg2 = _final_config(tmp_path, **{"gate.tail_floor": 0})
    assert run(audit_stubs, cfg2, run_dir, "--mode", "full") == 0
    dec = json.loads((final / ro.FinalRetrain.DECISION).read_text())
    assert dec["accepted"] and dec["cell_gate"]["passed"] is True
    assert dec["cell_gate_settings"]["tail_floor"] == 0.0
    assert rb.final_retrain(run_dir) == final
    out = capsys.readouterr().out
    assert "per-cell gate: no cell significantly worse beyond 1.0%" in out

    # a decision the per-cell gate took no part in stands, whatever its
    # settings now (made before the gate existed, or with it off)
    dec.pop("cell_gate_settings")
    dec.pop("cell_gate")
    (final / ro.FinalRetrain.DECISION).write_text(json.dumps(dec))
    n = len(calls(env))
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    assert "audit_eval" not in [c["stage"] for c in calls(env)[n:]]
    assert rb.final_retrain(run_dir) == final


def test_with_the_per_cell_gate_off_the_final_retrain_is_judged_as_before(
    tmp_path, audit_stubs, env, monkeypatch
):
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    scores = {"models.pt": [1.0, 1.0, 1.0, 0.45], "default": [0.6, 0.6, 0.6, 0.9]}
    monkeypatch.setenv("STUB_AUDIT", json.dumps(scores))
    cfg = _final_config(tmp_path, **{"gate.per_cell": False})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    dec = json.loads((run_dir / ro.FINAL_RETRAIN_DIR / "decision.json").read_text())
    assert dec["accepted"] and "cell_gate" not in dec
    assert "cell_gate_settings" not in dec
    # turning the gate on later leaves a decision it took no part in
    n = len(calls(env))
    on = _final_config(tmp_path, **{"gate.tail_floor": 0.6})
    assert run(audit_stubs, on, run_dir, "--mode", "full") == 0
    assert "audit_eval" not in [c["stage"] for c in calls(env)[n:]]


def test_each_round_gates_per_cell_against_the_bundle_that_ships(
    tmp_path, audit_stubs, env, monkeypatch
):
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    cfg = audited_config(tmp_path, **{"gate.cell_floor": 0.8})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    s06 = {
        Path(arg(c, "--out-dir")).parent.name: c
        for c in by_stage(calls(env))["stage06"]
    }
    assert "--cell-gate-baseline" not in s06["round_1"]["argv"]
    assert arg(s06["round_2"], "--cell-gate-baseline") == str(run_dir / "round_1")
    assert (
        arg(s06["round_2"], "--cell-tolerance"),
        arg(s06["round_2"], "--cell-floor"),
        arg(s06["round_2"], "--cell-gate-alpha"),
        arg(s06["round_2"], "--tail-floor"),
        arg(s06["round_2"], "--tail-collapse"),
        arg(s06["round_2"], "--tail-budget"),
        arg(s06["round_2"], "--cell-gate-groups"),
    ) == ("0.01", "0.8", "0.05", "0.5", "0.5", "0", "exclusive")
    assert "--cell-gate-record-only" not in s06["round_2"]["argv"]
    assert arg(s06["round_2"], "--audit-dir") == str(tmp_path / "audit")

    forced = tmp_path / "forced"
    n = len(calls(env))
    assert run(audit_stubs, cfg, forced, "--mode", "full", "--deploy") == 0
    s06 = {
        Path(arg(c, "--out-dir")).parent.name: c
        for c in by_stage(calls(env)[n:])["stage06"]
    }
    assert "--cell-gate-baseline" not in s06["round_0"]["argv"]
    assert arg(s06["round_1"], "--cell-gate-baseline") == str(forced / "round_0")
    assert "--cell-gate-record-only" in s06["round_1"]["argv"]

    for other in (
        make_config(tmp_path),
        audited_config(tmp_path, **{"gate.per_cell": False}),
    ):
        n = len(calls(env))
        plain = tmp_path / f"plain{n}"
        assert run(audit_stubs, other, plain, "--mode", "full") == 0
        st = by_stage(calls(env)[n:])["stage06"]
        assert all("--cell-gate-baseline" not in c["argv"] for c in st)


def test_a_bundle_the_per_cell_gate_keeps_out_never_stops_the_run_early(
    tmp_path, audit_stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    monkeypatch.setenv("STUB_AUDIT", json.dumps({"models.pt": 0.95, "default": 0.9}))
    run_dir = tmp_path / "run"
    stop = {"deploy.stop_when_target_reached": True}
    first = _final_config(tmp_path, **{"full.total_rounds": 2, **stop})
    assert run(audit_stubs, first, run_dir, "--mode", "full") == 0
    assert rb.final_retrain(run_dir) == run_dir / ro.FINAL_RETRAIN_DIR
    monkeypatch.setenv("STUB_CELL_GATE", "fail")
    more = _final_config(tmp_path, **{"full.total_rounds": 4, **stop})
    n = len(calls(env))
    assert run(audit_stubs, more, run_dir, "--mode", "full") == 0
    s06 = {
        Path(arg(c, "--out-dir")).parent.name: c
        for c in by_stage(calls(env)[n:])["stage06"]
    }
    assert arg(s06["round_2"], "--cell-gate-baseline") == str(
        run_dir / ro.FINAL_RETRAIN_DIR
    )
    rec = json.loads(
        (run_dir / "round_2" / "stage06" / "deploy_record.json").read_text()
    )
    assert rec["deployed"] is False
    assert len(ro.complete_prefix(run_dir)) == 4
    assert "deploy target reached; stopping early" not in capsys.readouterr().out


def test_a_gate_of_none_ships_the_final_retrain_without_a_comparison(
    tmp_path, audit_stubs, env, monkeypatch
):
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    monkeypatch.setenv("STUB_AUDIT", json.dumps({"models.pt": 0.5, "default": 0.9}))
    cfg = _final_config(tmp_path, **{"full.final_retrain.gate": "none"})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    dec = json.loads((run_dir / ro.FINAL_RETRAIN_DIR / "decision.json").read_text())
    assert dec["accepted"] and dec["reason"] == "full.final_retrain.gate is none"
    assert dec["selected_by_audit"] is False and "selection_note" not in dec
    assert dec["baseline"]["round"] == "round_2" and dec["baseline"]["deployed"]
    # it compared nothing: another official protocol leaves it as it is
    cfg = _final_config(
        tmp_path, **{"full.final_retrain.gate": "none", "audit.official": "v2"}
    )
    n = len(calls(env))
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    assert "stage06" not in by_stage(calls(env)[n:])
    again = json.loads((run_dir / ro.FINAL_RETRAIN_DIR / "decision.json").read_text())
    assert again == dec


def test_the_final_retrain_waits_for_an_audit_set_that_can_decide(
    tmp_path, audit_stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    cfg = _final_config(tmp_path, **{"full.total_rounds": 2, "audit.build": "off"})
    run_dir = tmp_path / "run"
    state = {"audit": ("incomplete", "the audit set is still being benched")}
    monkeypatch.setattr(ro, "audit_status", lambda cfg, paths: state["audit"])
    assert run(audit_stubs, cfg, run_dir, "--mode", "full", "--no-report") == 1
    final = run_dir / ro.FINAL_RETRAIN_DIR
    assert ro.stage_complete(final / "stage05")
    assert not (final / ro.FinalRetrain.DECISION).exists()
    out = capsys.readouterr()
    assert "the same command decides once it is" in out.err
    assert "the end of the run is incomplete" in out.out
    state["audit"] = ("ok", None)
    n = len(calls(env))
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    st = by_stage(calls(env)[n:])
    assert "stage05" not in st and len(st["stage06"]) == 1
    assert (final / ro.FinalRetrain.DECISION).is_file()


def test_the_final_retrain_trains_again_when_its_rounds_or_options_change(
    tmp_path, audit_stubs, env, monkeypatch
):
    monkeypatch.setenv("STUB_STAGES", str(Path(ro.__file__).parent / "stages"))
    cfg = _final_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    final = run_dir / ro.FINAL_RETRAIN_DIR
    first = (final / ro.FinalRetrain.RECORD).read_text()
    assert run(audit_stubs, cfg, run_dir, "--mode", "full", "--from-round", "1") == 0
    assert (final / ro.FinalRetrain.RECORD).read_text() != first
    assert len(list((run_dir / "stale").glob(f"*/{ro.FINAL_RETRAIN_DIR}"))) == 1
    n = len(calls(env))
    other = _final_config(
        tmp_path,
        **{"full.total_rounds": 2, "full.final_retrain.recipe": {"lam": 0.3}},
    )
    assert run(audit_stubs, other, run_dir, "--mode", "full") == 0
    s05 = [c for c in calls(env)[n:] if c["stage"] == "stage05"]
    assert len(s05) == 1 and s05[0]["argv"][-2:] == ["--lam", "0.3"]
    assert len(list((run_dir / "stale").glob(f"*/{ro.FINAL_RETRAIN_DIR}"))) == 2
    n = len(calls(env))
    gate = _final_config(
        tmp_path,
        **{
            "full.total_rounds": 2,
            "full.final_retrain.recipe": {"lam": 0.3},
            "full.final_retrain.gate": "none",
        },
    )
    assert run(audit_stubs, gate, run_dir, "--mode", "full") == 0
    # The same final retrain is deployed again, so the passes of stages 7 and
    # 8, which vouch for it, stand.
    assert [c["stage"] for c in calls(env)[n:]] == ["audit_eval", "stage06"]
    assert list((final / "stale").glob(f"*/{ro.FinalRetrain.DECISION}"))
    vdir = run_dir / "validate"
    s07 = ro.validated_by(vdir / "stage07")
    assert s07["deploy_record"] == str(final / "stage06" / "deploy_record.json")
    assert ro.validated_by(vdir / "stage08")["train_dir"] == str(final / "stage05")


def test_a_changed_training_budget_trains_the_final_retrain_again(
    tmp_path, audit_stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_STAGES", str(Path(ro.__file__).parent / "stages"))
    cfg = _final_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    final = run_dir / ro.FINAL_RETRAIN_DIR
    rec = json.loads((final / ro.FinalRetrain.RECORD).read_text())
    assert rec["options"]["lam"] == 0.4 and "output_dir" not in rec["options"]
    recipe = {"unmeasured_samples": 0, "lam": 0.4}
    budget = _final_config(
        tmp_path,
        **{
            "full.total_rounds": 2,
            "full.final_retrain.recipe": {**recipe, "epochs": 7, "lr": 0.001},
        },
    )
    n = len(calls(env))
    capsys.readouterr()
    assert run(audit_stubs, budget, run_dir, "--mode", "full") == 0
    s05 = [c for c in calls(env)[n:] if c["stage"] == "stage05"]
    assert len(s05) == 1
    assert values(s05[0], "--epochs") == ["7"] and values(s05[0], "--lr") == ["0.001"]
    assert "trained with other stage05 options (epochs" in capsys.readouterr().err
    rec = json.loads((final / ro.FinalRetrain.RECORD).read_text())
    assert rec["recipe"] == {**recipe, "epochs": 7, "lr": 0.001}
    assert len(list((run_dir / "stale").glob(f"*/{ro.FINAL_RETRAIN_DIR}"))) == 1
    placed = _final_config(
        tmp_path,
        **{
            "full.total_rounds": 2,
            "full.final_retrain.recipe": {**recipe, "epochs": 7, "lr": 0.001},
            "train.workers": 3,
            "train.pooled_threads": 3,
            "train.pooled_teacher_threads": 2,
            "runtime.devices": [1, 0],
        },
    )
    n = len(calls(env))
    assert run(audit_stubs, placed, run_dir, "--mode", "full") == 0
    assert [c for c in calls(env)[n:] if c["stage"] == "stage05"] == []


def test_a_round_deployed_after_the_final_retrain_replaces_it(
    tmp_path, audit_stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    monkeypatch.setenv("STUB_AUDIT", json.dumps({"models.pt": 0.95, "default": 0.9}))
    cfg = _final_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    r0, r1, r2, r3 = (run_dir / f"round_{i}" for i in range(4))
    final = run_dir / ro.FINAL_RETRAIN_DIR
    assert ro.run_bundles(run_dir) == [r0, r1, final]

    # an enhance round its gate keeps out: the final retrain stays the run's
    # bundle, the baseline of that gate, and what validation checks
    monkeypatch.setenv("STUB_GATE_SE", "0.5")
    n = len(calls(env))
    capsys.readouterr()
    assert run(audit_stubs, cfg, run_dir, "--mode", "enhance", "--cells", TINY) == 0
    st = by_stage(calls(env)[n:])
    (d06,) = st["stage06"]
    assert arg(d06, "--gate-baseline-train-dir") == str(final / "stage05")
    assert ro.run_bundles(run_dir) == [r0, r1, final, r2]
    assert ro.shipped_round(ro.run_bundles(run_dir)) == final
    # stages 7 and 8 passed for the final retrain, which still ships
    assert "stage07" not in st and "stage08" not in st
    vdir = run_dir / "validate"
    assert ro.validated_by(vdir / "stage07")["deploy_record"] == str(
        final / "stage06" / "deploy_record.json"
    )
    assert ro.validated_by(vdir / "stage08")["train_dir"] == str(final / "stage05")
    breakdown = json.loads((run_dir / "held_out_sel_eff_breakdown.json").read_text())
    assert breakdown["shipped"]["round"] == ro.FINAL_RETRAIN_DIR
    assert breakdown["final_bundle"]["round"] == "round_2"
    assert breakdown["final_retrain"]["rounds_after"] == ["round_2"]
    assert "did not train on round_2, which completed after it" in (
        capsys.readouterr().err
    )

    # an enhance round that deploys replaces it
    n = len(calls(env))
    argv = ["--mode", "enhance", "--cells", TINY, "--deploy"]
    assert run(audit_stubs, cfg, run_dir, *argv) == 0
    st = by_stage(calls(env)[n:])
    assert ro.run_bundles(run_dir) == [r0, r1, final, r2, r3]
    assert ro.shipped_round(ro.run_bundles(run_dir)) == r3
    assert arg(st["stage07"][0], "--deploy-record") == str(
        r3 / "stage06" / "deploy_record.json"
    )
    assert arg(st["stage08"][0], "--train-dir") == str(r3 / "stage05")
    breakdown = json.loads((run_dir / "held_out_sel_eff_breakdown.json").read_text())
    assert breakdown["shipped"]["round"] == "round_3"
    retrain = breakdown["final_retrain"]
    assert retrain["ships"] is False
    assert retrain["not_shipped"] == "round_3's bundle deployed after it"
    import make_report as mr

    out = tmp_path / "report.html"
    mr.build(str(run_dir), str(out), engine=False, bench=False)
    page = out.read_text()
    assert "deployed bundle (round_3)" in page
    # the audit set chose between the final retrain and round_1, not round_3
    assert "the audit set chose which of the final retrain" not in page
    assert "<b>audit</b> = the final retrain's audit record" not in page
    n = len(calls(env))
    assert run(audit_stubs, cfg, run_dir, "--mode", "validate") == 0
    st = by_stage(calls(env)[n:])
    assert arg(st["stage07"][0], "--deploy-record") == str(
        r3 / "stage06" / "deploy_record.json"
    )
    assert arg(st["stage08"][0], "--train-dir") == str(r3 / "stage05")


def test_rounds_after_a_shipped_final_retrain_build_on_it(
    tmp_path, audit_stubs, env, monkeypatch, capsys
):
    """While the final retrain ships, a later round measures its models,
    carries every cell it does not retrain from it and is gated against it,
    so that the cells its gate does not measure stay the final retrain's;
    the round after that builds on that round, with the final retrain as
    the bundle before it."""
    monkeypatch.setenv("STUB_STAGES", str(Path(ro.__file__).parent / "stages"))
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    monkeypatch.setenv("STUB_AUDIT", json.dumps({"models.pt": 0.95, "default": 0.9}))
    monkeypatch.setenv("STUB_CELLS", json.dumps([TINY, LARGE]))
    cfg = _final_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    r0, r1, r2, r3 = (run_dir / f"round_{i}" for i in range(4))
    final = run_dir / ro.FINAL_RETRAIN_DIR
    assert ro.shipped_round(ro.run_bundles(run_dir)) == final

    n = len(calls(env))
    capsys.readouterr()
    assert run(audit_stubs, cfg, run_dir, "--mode", "enhance", "--cells", TINY) == 0
    st = by_stage(calls(env)[n:])
    (s04b,) = st["stage04b"]
    assert arg(s04b, "--models-dir") == str(final / "stage05")
    assert arg(s04b, "--prior-models") == str(r1 / "stage05" / "models.pt")
    assert values(s04b, "--prior-round-dir") == [str(r0), str(r1)]
    (s05,) = st["stage05"]
    assert values(s05, "--prior-round-dir") == [str(r0), str(r1), str(final)]
    # The final retrain trained its cells with its recipe, which the round
    # does not train with: they are carried as they are, not retrained.
    assert arg(s05, "--only-cells") == TINY
    (s06,) = st["stage06"]
    assert arg(s06, "--gate-baseline-train-dir") == str(final / "stage05")
    assert "round_2 builds on the shipped final_retrain" in capsys.readouterr().out
    labels = json.loads((r2 / "stage05" / "cells.json").read_text())["model_labels"]
    assert sorted(labels) == sorted([TINY, LARGE])

    n = len(calls(env))
    more = _final_config(tmp_path, **{"full.total_rounds": 4})
    assert run(audit_stubs, more, run_dir, "--mode", "active") == 0
    st = by_stage(calls(env)[n:])
    (s04b,) = st["stage04b"]
    assert arg(s04b, "--models-dir") == str(r2 / "stage05")
    assert arg(s04b, "--prior-models") == str(final / "stage05" / "models.pt")
    (s05,) = st["stage05"]
    assert values(s05, "--prior-round-dir") == [str(r0), str(r1), str(r2)]
    assert ro.round_complete(r3)


def test_a_round_carries_the_final_retrains_cells_while_its_options_hold(
    tmp_path, audit_stubs, env, monkeypatch
):
    """`option_changes` compares a cell the final retrain trained with the
    options the final retrain trains with now: kept while they hold,
    retrained once they change."""
    monkeypatch.setenv("STUB_STAGES", str(Path(ro.__file__).parent / "stages"))
    monkeypatch.setenv("STUB_AUDIT", json.dumps({"models.pt": 0.95, "default": 0.9}))
    monkeypatch.setenv("STUB_CELLS", json.dumps([TINY, LARGE]))
    cfg = _final_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    final = run_dir / ro.FINAL_RETRAIN_DIR
    loaded, _ = ro.load_config(cfg)
    rounds = ro.complete_prefix(run_dir)
    pipeline = ro.Pipeline(
        cfg=loaded,
        snapshot=sorted((run_dir / "configs").glob("*_full.yaml"))[-1],
        run_dir=run_dir,
        paths=ro.derive_paths(loaded),
        stages_dir=audit_stubs,
        timing=ro.Timing(),
        force_deploy=False,
    )
    step = ro.RoundStep(2, "active", rounds, None, True)
    lineage = pipeline.bundle_lineage(step)
    assert lineage == [*rounds, final]
    s5 = ro.stage05_module()
    options = s5.command_model_options(
        pipeline.args_stage05(run_dir / "round_2", step, TINY)
    )
    assert ro.option_changes(lineage, [LARGE], options)[LARGE]
    now = pipeline.final_retrain_model_options(rounds)
    assert ro.option_changes(lineage, [LARGE], options, now) == {}
    changed = ro.load_config(
        _final_config(
            tmp_path,
            **{"full.total_rounds": 2, "full.final_retrain.recipe": {"lam": 0.3}},
        )
    )[0]
    pipeline.cfg = changed
    now = pipeline.final_retrain_model_options(rounds)
    assert set(ro.option_changes(lineage, [LARGE], options, now)[LARGE]) == {
        "lam",
        "unmeasured_samples",
    }


def test_a_carried_cell_keeps_its_origin_after_the_final_retrain_leaves(
    tmp_path, audit_stubs, env, monkeypatch
):
    """A round that carries cells from the shipped final retrain records
    where they were trained (stage05/cell_origins.json): their options are
    still judged as the final retrain's, and the report still describes
    them by the final retrain's cells.json entries, once the final retrain
    is no bundle of the run any more."""
    import make_report as mr

    monkeypatch.setenv("STUB_STAGES", str(Path(ro.__file__).parent / "stages"))
    monkeypatch.setenv("STUB_AUDIT", json.dumps({"models.pt": 0.95, "default": 0.9}))
    monkeypatch.setenv("STUB_CELLS", json.dumps([TINY, LARGE]))
    cfg = _final_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    r0, r1, r2 = (run_dir / f"round_{i}" for i in range(3))
    final = run_dir / ro.FINAL_RETRAIN_DIR
    assert rb.final_retrain(run_dir) == final
    # the final retrain's entry of LARGE differs from the rounds'
    cj = json.loads((final / "stage05" / "cells.json").read_text())
    for c in cj["cells"]:
        c["best_sel_eff"] = 0.77
    (final / "stage05" / "cells.json").write_text(json.dumps(cj))
    argv = ["--mode", "enhance", "--cells", TINY, "--deploy"]
    assert run(audit_stubs, cfg, run_dir, *argv) == 0
    assert ro.shipped_round(ro.run_bundles(run_dir)) == r2
    options = json.loads((final / "stage05" / "model_options.json").read_text())
    rec = json.loads((r2 / "stage05" / rb.CELL_ORIGINS).read_text())
    assert rec["source"] == ro.FINAL_RETRAIN_DIR and list(rec["cells"]) == [LARGE]
    large = rec["cells"][LARGE]
    assert large["entry"]["best_sel_eff"] == 0.77
    assert rec["origins"][large["origin"]] == {
        "bundle": ro.FINAL_RETRAIN_DIR,
        "finished": rb.stage05_finished(final),
        "model_options": options,
    }

    # the report describes LARGE in round_2's bundle by the final retrain's
    # entry: from the final retrain itself while it is one of the run's
    # bundles, and from round_2's record
    (r2 / "stage05" / rb.CELL_ORIGINS).rename(tmp_path / "origins.json")
    model = {c["cell"]: c["model"] for c in mr.final_cells(str(run_dir), {}, "round_2")}
    assert model == {LARGE: 77.0, TINY: None}
    (tmp_path / "origins.json").rename(r2 / "stage05" / rb.CELL_ORIGINS)

    # the final retrain leaves the run (as a later one replacing it would)
    final.rename(tmp_path / "gone")
    assert rb.final_retrain(run_dir) is None
    model = {c["cell"]: c["model"] for c in mr.final_cells(str(run_dir), {}, "round_2")}
    assert model == {LARGE: 77.0, TINY: None}
    origin = rb.cell_origins([r0, r1, r2])[LARGE]
    assert (origin["trained_in"], origin["bundle"]) == (ro.FINAL_RETRAIN_DIR, None)
    assert origin["model_options"] == options
    loaded, _ = ro.load_config(cfg)
    pipeline = ro.Pipeline(
        cfg=loaded,
        snapshot=sorted((run_dir / "configs").glob("*_full.yaml"))[-1],
        run_dir=run_dir,
        paths=ro.derive_paths(loaded),
        stages_dir=audit_stubs,
        timing=ro.Timing(),
        force_deploy=False,
    )
    step = ro.RoundStep(3, "active", [r0, r1, r2], None, True)
    s5 = ro.stage05_module()
    round_options = s5.command_model_options(
        pipeline.args_stage05(run_dir / "round_3", step, TINY)
    )
    # trained with the recipe, not with the round's options (round_1 trained
    # LARGE with those): kept while a final retrain trains with them now
    lineage = pipeline.bundle_lineage(step)
    assert lineage == [r0, r1, r2]
    assert set(ro.option_changes(lineage, [LARGE], round_options)[LARGE]) == {
        "lam",
        "unmeasured_samples",
    }
    now = pipeline.final_retrain_model_options([r0, r1])
    assert ro.option_changes(lineage, [LARGE], round_options, now) == {}
    other = ro.load_config(
        _final_config(
            tmp_path,
            **{"full.total_rounds": 2, "full.final_retrain.recipe": {"lam": 0.3}},
        )
    )[0]
    pipeline.cfg = other
    now = pipeline.final_retrain_model_options([r0, r1])
    assert set(ro.option_changes(lineage, [LARGE], round_options, now)[LARGE]) == {
        "lam",
        "unmeasured_samples",
    }


def test_a_round_that_runs_again_ends_the_final_retrains_place(
    tmp_path, audit_stubs, env, monkeypatch
):
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    monkeypatch.setenv("STUB_AUDIT", json.dumps({"models.pt": 0.95, "default": 0.9}))
    cfg = _final_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    final = run_dir / ro.FINAL_RETRAIN_DIR
    marker = run_dir / "round_1" / ro.ROUND_MARKER
    rec = json.loads(marker.read_text())
    assert rb.final_retrain(run_dir) == final
    marker.write_text(json.dumps(dict(rec, finished="2099-01-01T00:00:00")))
    assert rb.final_retrain(run_dir) is None
    assert rb.changed_rounds(run_dir, rb.trained_rounds(final)) == ["round_1"]
    numbers = ro.shipped_bundle_numbers(run_dir)
    assert numbers["shipped"]["round"] == "round_1"
    assert numbers["final_retrain"]["not_shipped"] == (
        "round_1 changed since it trained on them"
    )
    marker.write_text(json.dumps(rec))
    assert rb.final_retrain(run_dir) == final


def test_run_status_names_the_shipped_final_retrain(
    tmp_path, audit_stubs, env, monkeypatch
):
    """run_status.json: the final retrain as the deployed bundle, whose
    stage 7 and 8 passes the run status checks, and the final retrain's own
    state; the run is complete only once it is decided (and deployed when it
    ships)."""
    monkeypatch.setenv("STUB_AUDIT", json.dumps({"models.pt": 0.95, "default": 0.9}))
    cfg = _final_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    final = run_dir / ro.FINAL_RETRAIN_DIR
    status = json.loads((run_dir / ro.RUN_STATUS).read_text())
    assert status["complete"]
    assert status["deployed"]["round"] == ro.FINAL_RETRAIN_DIR
    assert status["deployed"]["deploy_record"] == str(
        final / "stage06" / "deploy_record.json"
    )
    retrain = status["final_retrain"]
    assert retrain["status"] == "shipped" and retrain["trained_on"] == [
        "round_0",
        "round_1",
    ]
    for key in ("stage07", "stage08"):
        assert status["validation"][key]["status"] == "passed"
    assert status["validation"]["stage08"]["validated"]["train_dir"] == str(
        final / "stage05"
    )

    def state():
        return ro.final_retrain_status(run_dir, True)["status"]

    marker = final / "stage06" / ro.STAGE_MARKER
    kept = marker.read_text()
    marker.unlink()
    assert state() == "accepted"
    marker.write_text(kept)
    decision = final / ro.FinalRetrain.DECISION
    kept = decision.read_text()
    decision.unlink()
    assert state() == "undecided"
    decision.write_text(json.dumps(dict(json.loads(kept), accepted=False)))
    assert state() == "rejected"
    decision.write_text(kept)
    round_marker = run_dir / "round_1" / ro.ROUND_MARKER
    rec = json.loads(round_marker.read_text())
    round_marker.write_text(json.dumps(dict(rec, finished="2099-01-01T00:00:00")))
    entry = ro.final_retrain_status(run_dir, True)
    assert entry["status"] == "pending" and "round_1" in entry["reason"]
    round_marker.write_text(json.dumps(rec))
    assert state() == "shipped"
    assert ro.final_retrain_status(tmp_path / "other", False) == {
        "enabled": False,
        "status": "off",
    }
    assert ro.final_retrain_status(tmp_path / "other", True)["status"] == "pending"


def test_a_final_retrain_that_may_ship_gives_stage7_a_bundle(tmp_path, capsys):
    """Full-mode preflight counts the final retrain among what deploys the
    bundle stage 7 validates: with gate none it always deploys, with
    not-worse it may."""

    def errors(**overrides):
        cfg, _ = ro.load_config(_final_config(tmp_path, **overrides))
        total = ro.cfg_get(cfg, "full.total_rounds")
        steps = ro.plan_training("full", tmp_path / "fresh", total, None, None)
        return ro.deploy_errors(cfg, tmp_path / "fresh", steps, False)

    one = {"full.total_rounds": 1}
    capsys.readouterr()
    assert errors(**one) == []
    assert (
        "the final retrain deploys when the audit set finds it not worse "
        "(full.final_retrain), so the run ends with stage 7 not run (exit 2) when "
        "neither a round reaches it nor the final retrain ships"
    ) in capsys.readouterr().err
    assert errors(**one, **{"full.final_retrain.gate": "none"}) == []
    assert capsys.readouterr().err == ""
    (err,) = errors(**one, **{"full.final_retrain.enabled": False})
    assert "a run of one round deploys only with --deploy" in err


def test_a_copied_run_keeps_its_final_retrain_trained(
    tmp_path, audit_stubs, env, monkeypatch
):
    """A copy that does not keep the round markers' mtimes still ships the
    final retrain, and the next --mode full does not train it again."""
    import shutil

    monkeypatch.setenv("STUB_AUDIT", json.dumps({"models.pt": 0.95, "default": 0.9}))
    cfg = _final_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    copy = tmp_path / "copy"
    shutil.copytree(
        run_dir, copy, copy_function=shutil.copy, ignore=shutil.ignore_patterns(".lock")
    )
    final = copy / ro.FINAL_RETRAIN_DIR
    assert rb.final_retrain(copy) == final
    n = len(calls(env))
    assert run(audit_stubs, cfg, copy, "--mode", "full") == 0
    assert "stage05" not in by_stage(calls(env)[n:])
    assert not (copy / "stale").exists() and rb.final_retrain(copy) == final


def test_an_option_the_record_predates_keeps_the_final_retrain(
    tmp_path, audit_stubs, env, monkeypatch, capsys
):
    """final_retrain.json without an option stage05 gained later reads as
    that option at its default, or at its value from before it was an
    option (stage05 PREDATING_OPTIONS); an option stage05 no longer has is
    not compared."""
    from lib import repeats

    monkeypatch.setenv("STUB_STAGES", str(Path(ro.__file__).parent / "stages"))
    monkeypatch.setenv("STUB_AUDIT", json.dumps({"models.pt": 0.95, "default": 0.9}))
    cfg = _final_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    path = run_dir / ro.FINAL_RETRAIN_DIR / ro.FinalRetrain.RECORD
    rec = json.loads(path.read_text())
    assert rec["options"]["qat_frac"] == 0
    del rec["options"]["qat_frac"]
    rec["options"]["an_option_stage05_dropped"] = 1
    path.write_text(json.dumps(rec))
    n = len(calls(env))
    capsys.readouterr()
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    assert "stage05" not in by_stage(calls(env)[n:])
    assert "training the final retrain again" not in capsys.readouterr().err

    # the record predates --repeat-aggregation: it was trained with the
    # merge from before the option, which today's default is not
    assert repeats.PREDATING_AGGREGATION != rec["options"]["repeat_aggregation"]
    del rec["options"]["repeat_aggregation"]
    path.write_text(json.dumps(rec))
    n = len(calls(env))
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    assert len(by_stage(calls(env)[n:])["stage05"]) == 1
    assert (
        f"trained with other stage05 options (repeat_aggregation "
        f"{repeats.PREDATING_AGGREGATION} -> {repeats.DEFAULT_AGGREGATION})"
        in capsys.readouterr().err
    )


def test_a_record_from_before_the_draws_keeps_the_final_retrain(
    tmp_path, audit_stubs, env, monkeypatch, capsys
):
    """final_retrain.json names the pseudo-negatives' options as stage05
    resolves them (`recorded_options`): a record from before
    --pseudo-negatives, which names no draw and the margin and weight that
    stage05 then defaulted to, keeps the final retrain, and so does a draw
    named as the one the sample count implies."""
    monkeypatch.setenv("STUB_STAGES", str(Path(ro.__file__).parent / "stages"))
    monkeypatch.setenv("STUB_AUDIT", json.dumps({"models.pt": 0.95, "default": 0.9}))
    cfg = _final_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    path = run_dir / ro.FINAL_RETRAIN_DIR / ro.FinalRetrain.RECORD
    rec = json.loads(path.read_text())
    del rec["options"]["pseudo_negatives"]
    rec["options"].update(unmeasured_margin=0.05, unmeasured_weight=0.3)
    path.write_text(json.dumps(rec))
    n = len(calls(env))
    capsys.readouterr()
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    assert "stage05" not in by_stage(calls(env)[n:])
    assert "training the final retrain again" not in capsys.readouterr().err

    named = _final_config(
        tmp_path,
        **{
            "full.total_rounds": 2,
            "full.final_retrain.recipe": {
                "unmeasured_samples": 0,
                "lam": 0.4,
                "pseudo_negatives": "none",
            },
        },
    )
    n = len(calls(env))
    assert run(audit_stubs, named, run_dir, "--mode", "full") == 0
    assert "stage05" not in by_stage(calls(env)[n:])
    assert "training the final retrain again" not in capsys.readouterr().err


def test_an_early_stop_says_what_waits_for_the_planned_rounds(
    tmp_path, audit_stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.99")
    cfg = _final_config(
        tmp_path,
        **{
            "full.total_rounds": 3,
            "deploy.stop_when_target_reached": True,
            "deploy.target_sel_eff": 0.95,
        },
    )
    run_dir = tmp_path / "run"
    capsys.readouterr()
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    err = capsys.readouterr().err
    assert len(ro.complete_prefix(run_dir)) == 2
    assert not (run_dir / ro.FINAL_RETRAIN_DIR).exists()
    assert (
        "the final retrain (full.final_retrain) and the rounds' audit records "
        "completed at the end of the run wait for the planned rounds (2 of 3 "
        "complete): set full.total_rounds to 2" in err
    )
    pipeline = ro.Pipeline(
        cfg=ro.load_config(cfg)[0],
        snapshot=cfg,
        run_dir=run_dir,
        paths=ro.derive_paths(ro.load_config(cfg)[0]),
        stages_dir=audit_stubs,
        timing=ro.Timing(),
        force_deploy=False,
    )
    pipeline.cfg = dict(pipeline.cfg, audit={**pipeline.cfg["audit"], "dir": None})
    (only,) = pipeline.end_of_run_waits(2, 3)
    assert only.startswith(
        "the final retrain (full.final_retrain) waits for the planned rounds"
    )
    n = len(calls(env))
    fewer = _final_config(
        tmp_path,
        **{
            "full.total_rounds": 2,
            "deploy.stop_when_target_reached": True,
            "deploy.target_sel_eff": 0.95,
        },
    )
    assert run(audit_stubs, fewer, run_dir, "--mode", "full") == 0
    assert by_stage(calls(env)[n:])["stage05"]
    assert (run_dir / ro.FINAL_RETRAIN_DIR / ro.FinalRetrain.DECISION).is_file()


def test_a_shipped_final_retrain_gets_the_v2_its_decision_missed(
    tmp_path, audit_stubs, env, monkeypatch
):
    """A protocol v2 bench that failed while the final retrain was decided
    is run again at the end of the next full run, and the shipped final
    retrain scored again into audit_results/final_retrain.json, which its
    readers then prefer."""
    monkeypatch.setenv("STUB_AUDIT", json.dumps({"models.pt": 0.95, "default": 0.9}))
    cfg = _final_config(tmp_path, **{"full.total_rounds": 2, "audit.v2": "final"})
    run_dir = tmp_path / "run"
    monkeypatch.setenv("STUB_RC_audit_v2", "1")
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    final = run_dir / ro.FINAL_RETRAIN_DIR
    assert rb.final_retrain(run_dir) == final
    own = json.loads((final / "stage06" / al.RESULT_FILE).read_text())
    assert "v2" not in own
    assert ro.final_audit_measurement(final)["protocol"] == "v1"

    monkeypatch.delenv("STUB_RC_audit_v2")
    monkeypatch.setenv("STUB_AUDIT_V2", "1")
    n = len(calls(env))
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    st = by_stage(calls(env)[n:])
    names = [m.split("=")[0] for c in st["audit_v2"] for m in values(c, "--model")]
    assert "run.final_retrain" in names
    (rescore,) = [c for c in st["audit_eval"] if "rescore" in (arg(c, "--out") or "")]
    assert values(rescore, "--model-file") == [str(final / "stage05" / "models.pt")]
    assert values(rescore, "--trained-round-dir") == [
        str(run_dir / "round_0"),
        str(run_dir / "round_1"),
    ]
    later = json.loads(
        (run_dir / ro.AUDIT_RESULTS_DIR / "final_retrain.json").read_text()
    )
    assert later["what"] == ro.FINAL_RETRAIN_DIR and later["v2"]["status"] == "ok"
    measured = ro.final_audit_measurement(final)
    assert measured["protocol"] == "sustained"
    assert measured["v2"]["cell_mean"] == measured["cell_mean"]
    n = len(calls(env))
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    assert "audit_v2" not in by_stage(calls(env)[n:])


def test_a_gate_change_that_keeps_out_an_installed_final_retrain_says_so(
    tmp_path, audit_stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_AUDIT", json.dumps({"models.pt": 0.5, "default": 0.9}))
    cfg = _final_config(
        tmp_path, **{"full.total_rounds": 2, "full.final_retrain.gate": "none"}
    )
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    final = run_dir / ro.FINAL_RETRAIN_DIR
    record = final / "stage06" / "deploy_record.json"
    rec = json.loads(record.read_text())
    record.write_text(json.dumps(dict(rec, applied=True)))
    capsys.readouterr()
    gated = _final_config(tmp_path, **{"full.total_rounds": 2})
    run(audit_stubs, gated, run_dir, "--mode", "full")
    err = capsys.readouterr().err
    assert rb.final_retrain(run_dir) is None
    assert (
        "the final retrain no longer ships (full.final_retrain.gate not-worse), "
        "but its earlier stage06 installed its weights into" in err
    )
    assert "round_1's bundle" in err


# ── stub runs: a final retrain trained again while the run ships one ─────────


def _shipped_final_retrain(tmp_path, audit_stubs, env, monkeypatch):
    """A finished full run of two rounds whose final retrain shipped (its
    audit cell mean 0.95 against round_1's 0.9): (run dir, final_retrain/,
    its decision.json text)."""
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    monkeypatch.setenv("STUB_AUDIT", json.dumps({"models.pt": 0.95, "default": 0.9}))
    cfg = _final_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    final = run_dir / ro.FINAL_RETRAIN_DIR
    assert ro.shipped_round(ro.run_bundles(run_dir)) == final
    return run_dir, final, (final / ro.FinalRetrain.DECISION).read_text()


# The audit set scores the shipped final retrain (scored as its directory)
# 0.95, a new final retrain's models.pt `new`, and every round 0.9.
def _scores(new):
    return json.dumps({ro.FINAL_RETRAIN_DIR: 0.95, "models.pt": new, "default": 0.9})


def _recipe_change(tmp_path, monkeypatch, **more):
    """The recipe changes (it is not pinned): the final retrain is trained
    again on the same rounds."""
    recipe = {"unmeasured_samples": 0, "lam": 0.3}
    cfg = _final_config(
        tmp_path,
        **{"full.total_rounds": 2, "full.final_retrain.recipe": recipe, **more},
    )
    return cfg, ["round_0", "round_1"]


def _round_kept_out(tmp_path, monkeypatch):
    """full.total_rounds is raised: round_2 builds on the shipped final
    retrain, its not-worse gate keeps it out, and the final retrain is
    trained again on the three rounds."""
    monkeypatch.setenv("STUB_GATE_SE", "0.5")
    return _final_config(tmp_path, **{"full.total_rounds": 3}), [
        "round_0",
        "round_1",
        "round_2",
    ]


@pytest.mark.parametrize("change", [_recipe_change, _round_kept_out])
def test_a_new_final_retrain_is_gated_against_the_shipped_one(
    tmp_path, audit_stubs, env, monkeypatch, capsys, change
):
    """A final retrain trained again while the run ships the one before
    trains beside it (final_retrain_next/) and is compared with the shipped
    one, not with the last round's bundle: worse than the shipped one,
    though better than round_1's, it does not ship, and the shipped one
    stays the run's bundle, untouched."""
    import make_report as mr

    run_dir, final, decided = _shipped_final_retrain(
        tmp_path, audit_stubs, env, monkeypatch
    )
    cfg, names = change(tmp_path, monkeypatch)
    monkeypatch.setenv("STUB_AUDIT", _scores(0.92))
    nxt = run_dir / rb.FINAL_RETRAIN_NEXT_DIR
    n = len(calls(env))
    capsys.readouterr()
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    out = capsys.readouterr()
    st = by_stage(calls(env)[n:])
    if "round_2" in names:
        (s06,) = [c for c in st["stage06"] if "round_2" in arg(c, "--out-dir")]
        assert arg(s06, "--gate-baseline-train-dir") == str(final / "stage05")
        rec = json.loads(
            (run_dir / "round_2" / "stage06" / "deploy_record.json").read_text()
        )
        assert rec["deployed"] is False
    (s05,) = st["stage05"][-1:]
    assert arg(s05, "--output-dir") == str(nxt / "stage05")
    (ev,) = [c for c in st["audit_eval"] if arg(c, "--out") == str(nxt / "audit.json")]
    assert values(ev, "--round-dir") == [str(final)]
    assert values(ev, "--model-file") == [str(nxt / "stage05" / "models.pt")]
    assert values(ev, "--trained-round-dir") == [str(run_dir / r) for r in names]
    dec = json.loads((nxt / ro.FinalRetrain.DECISION).read_text())
    assert dec["accepted"] is False and dec["rounds"] == names
    assert dec["reason"].startswith(
        "worse than the shipped final retrain (trained on round_0, round_1) on "
        "the audit set: cell mean x0.9684"
    )
    base = dec["baseline"]
    assert (base["round"], base["deployed"], base["trained_on"]) == (
        ro.FINAL_RETRAIN_DIR,
        True,
        ["round_0", "round_1"],
    )
    assert base["cell_mean"] == pytest.approx(0.95)
    assert dec["final"]["cell_mean"] == pytest.approx(0.92)
    assert all(
        arg(c, "--out-dir") != str(nxt / "stage06") for c in st.get("stage06", [])
    )
    # the shipped final retrain is left as it was, and stays the run's bundle
    assert (final / ro.FinalRetrain.DECISION).read_text() == decided
    assert rb.final_retrain(run_dir) == final
    assert ro.shipped_round(ro.run_bundles(run_dir)) == final
    assert not (run_dir / "stale").exists()
    assert "stage07" not in st and "stage08" not in st
    status = json.loads((run_dir / ro.RUN_STATUS).read_text())
    assert status["complete"] and status["deployed"]["round"] == ro.FINAL_RETRAIN_DIR
    entry = status["final_retrain"]
    assert (entry["status"], entry["dir"], entry["trained_on"]) == (
        "rejected",
        str(nxt),
        names,
    )
    assert entry["replaces"] == {
        "dir": str(final),
        "trained_on": ["round_0", "round_1"],
    }
    breakdown = json.loads((run_dir / "held_out_sel_eff_breakdown.json").read_text())
    assert breakdown["shipped"]["round"] == ro.FINAL_RETRAIN_DIR
    retrain = breakdown["final_retrain"]
    assert retrain["ships"] and retrain["candidate"]["status"] == "rejected"
    assert (
        f"a new final retrain (trained on {', '.join(names)}; {nxt}) does not "
        f"ship: worse than the shipped final retrain"
    ) in out.err
    assert "the next --mode full trains it again" not in out.err
    assert f"training the final retrain again beside the one in {final}" in out.err
    gates = mr.deploy_gates(str(run_dir))
    assert [(g["round"], g["deployed"]) for g in gates[-2:]] == [
        (ro.FINAL_RETRAIN_DIR, True),
        (rb.FINAL_RETRAIN_NEXT_DIR, False),
    ]
    # the same command again finds nothing to do
    n = len(calls(env))
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    assert [c["stage"] for c in calls(env)[n:]] == []
    assert json.loads((nxt / ro.FinalRetrain.DECISION).read_text()) == dec


def test_a_new_final_retrain_not_worse_than_the_shipped_one_replaces_it(
    tmp_path, audit_stubs, env, monkeypatch
):
    """Accepted, the new final retrain is deployed by stage06 in
    final_retrain_next/ and then takes final_retrain/'s place, its records
    naming final_retrain/; the one before moves to stale/."""
    run_dir, final, _ = _shipped_final_retrain(tmp_path, audit_stubs, env, monkeypatch)
    cfg, names = _recipe_change(tmp_path, monkeypatch)
    monkeypatch.setenv("STUB_AUDIT", _scores(0.97))
    nxt = run_dir / rb.FINAL_RETRAIN_NEXT_DIR
    n = len(calls(env))
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    st = by_stage(calls(env)[n:])
    s06 = st["stage06"][-1]
    assert arg(s06, "--train-dir") == str(nxt / "stage05")
    assert arg(s06, "--out-dir") == str(nxt / "stage06")
    assert values(s06, "--audit-trained-round-dir") == [str(run_dir / r) for r in names]
    assert not nxt.exists()
    rec = json.loads((final / ro.FinalRetrain.RECORD).read_text())
    assert rec["recipe"]["lam"] == 0.3
    dec = json.loads((final / ro.FinalRetrain.DECISION).read_text())
    assert dec["accepted"] and dec["reason"].startswith(
        "not worse than the shipped final retrain (trained on round_0, round_1)"
    )
    assert dec["models"] == str(final / "stage05" / "models.pt")
    deployed = json.loads((final / "stage06" / "deploy_record.json").read_text())
    assert deployed["train_dir"] == str(final / "stage05")
    assert not any(str(nxt) in p.read_text() for p in final.rglob("*.json"))
    assert not (final / ro.RELOCATED).exists()
    (old,) = (run_dir / "stale").glob(f"*/{ro.FINAL_RETRAIN_DIR}")
    assert (
        json.loads((old / ro.FinalRetrain.RECORD).read_text())["recipe"]["lam"] == 0.4
    )
    assert rb.final_retrain(run_dir) == final
    assert ro.shipped_round(ro.run_bundles(run_dir)) == final
    # stages 7 and 8 validate the bundle that ships now
    assert arg(st["stage07"][0], "--deploy-record") == str(
        final / "stage06" / "deploy_record.json"
    )
    assert arg(st["stage08"][0], "--train-dir") == str(final / "stage05")
    breakdown = json.loads((run_dir / "held_out_sel_eff_breakdown.json").read_text())
    assert breakdown["final_retrain"]["ships"]
    assert breakdown["final_retrain"]["candidate"] is None
    status = json.loads((run_dir / ro.RUN_STATUS).read_text())
    assert status["complete"] and status["final_retrain"]["status"] == "shipped"
    n = len(calls(env))
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    assert [c["stage"] for c in calls(env)[n:]] == []


@pytest.mark.parametrize("how", ["undecided", "untrained", "undeployed"])
def test_the_shipped_final_retrain_stays_while_a_new_one_cannot_replace_it(
    tmp_path, audit_stubs, env, monkeypatch, capsys, how
):
    """A new final retrain that the audit set cannot decide yet, whose
    stage05 fails, or whose stage06 fails after its gate accepted it leaves
    the shipped final retrain in place; the same command then goes on."""
    run_dir, final, decided = _shipped_final_retrain(
        tmp_path, audit_stubs, env, monkeypatch
    )
    cfg, _ = _recipe_change(tmp_path, monkeypatch, **{"audit.build": "off"})
    monkeypatch.setenv("STUB_AUDIT", _scores(0.97))
    nxt = run_dir / rb.FINAL_RETRAIN_NEXT_DIR
    audit = {"state": ("incomplete", "the audit set is still being benched")}
    if how == "undecided":
        monkeypatch.setattr(ro, "audit_status", lambda cfg, paths: audit["state"])
    else:
        monkeypatch.setenv(
            "STUB_FAIL",
            "stage05@validate" if how == "untrained" else "stage06@validate",
        )
    capsys.readouterr()
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 1
    err = capsys.readouterr().err
    assert (final / ro.FinalRetrain.DECISION).read_text() == decided
    assert rb.final_retrain(run_dir) == final
    assert ro.shipped_round(ro.run_bundles(run_dir)) == final
    assert not (run_dir / "stale").exists()
    ships = f"the final retrain in {final} (trained on round_0, round_1)"
    if how == "undecided":
        assert not (nxt / ro.FinalRetrain.DECISION).exists()
        assert (
            "the new final retrain is compared with the shipped final retrain "
            "(trained on round_0, round_1) on the audit set, which is not complete"
        ) in err
    elif how == "untrained":
        assert "the new final retrain's stage05 failed" in err
        assert f"the run ships {ships} meanwhile" in err
    else:
        assert json.loads((nxt / ro.FinalRetrain.DECISION).read_text())["accepted"]
        assert not ro.stage_complete(nxt / "stage06")
        assert "the new final retrain passed its gate, but its stage06 failed" in err
        assert f"the run ships {ships} meanwhile" in err
        assert f"may have written into {run_dir.parent / 'deploy'}" in err
    status = json.loads((run_dir / ro.RUN_STATUS).read_text())
    assert not status["complete"]
    assert status["deployed"]["round"] == ro.FINAL_RETRAIN_DIR
    assert (
        status["final_retrain"]["status"]
        == {
            "undecided": "undecided",
            "untrained": "pending",
            "undeployed": "accepted",
        }[how]
    )
    assert status["final_retrain"]["replaces"]["dir"] == str(final)

    audit["state"] = ("ok", None)
    monkeypatch.delenv("STUB_FAIL", raising=False)
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    assert not nxt.exists()
    assert (
        json.loads((final / ro.FinalRetrain.RECORD).read_text())["recipe"]["lam"] == 0.3
    )
    assert rb.final_retrain(run_dir) == final
    assert len(list((run_dir / "stale").glob(f"*/{ro.FINAL_RETRAIN_DIR}"))) == 1


def test_an_interrupted_promotion_is_finished_by_the_next_invocation(
    tmp_path, audit_stubs, env, monkeypatch
):
    """A new final retrain that stage06 deployed, which an invocation did
    not get to put in final_retrain/'s place, takes it when the next
    invocation starts (--mode validate here, which then validates it),
    without being trained or deployed again."""
    run_dir, final, _ = _shipped_final_retrain(tmp_path, audit_stubs, env, monkeypatch)
    cfg, _ = _recipe_change(tmp_path, monkeypatch)
    monkeypatch.setenv("STUB_AUDIT", _scores(0.97))
    nxt = run_dir / rb.FINAL_RETRAIN_NEXT_DIR
    with monkeypatch.context() as m:
        m.setattr(ro.FinalRetrain, "promote", lambda self: None)
        assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    assert ro.stage_complete(nxt / "stage06") and rb.final_retrain(run_dir) == final
    status = json.loads((run_dir / ro.RUN_STATUS).read_text())
    assert status["final_retrain"]["status"] == "accepted"
    n = len(calls(env))
    assert run(audit_stubs, cfg, run_dir, "--mode", "validate") == 0
    st = by_stage(calls(env)[n:])
    assert sorted(st) == ["stage07", "stage08"]
    assert not nxt.exists()
    assert (
        json.loads((final / ro.FinalRetrain.RECORD).read_text())["recipe"]["lam"] == 0.3
    )
    rec = json.loads((final / "stage06" / "deploy_record.json").read_text())
    assert rec["train_dir"] == str(final / "stage05")
    assert arg(st["stage07"][0], "--deploy-record") == str(
        final / "stage06" / "deploy_record.json"
    )
    assert arg(st["stage08"][0], "--train-dir") == str(final / "stage05")
    n = len(calls(env))
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    assert [c["stage"] for c in calls(env)[n:]] == []


def test_a_relocated_directory_names_its_new_place(tmp_path):
    """`relocate` moves a directory and rewrites the paths its JSON records
    hold, as given and resolved; `finish_relocation` finishes one that was
    interrupted after the move."""
    real = tmp_path / "real"
    (real / "run").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real)
    src, dst = link / "run" / "next", link / "run" / "final"
    (src / "stage06").mkdir(parents=True)
    rec = {
        "train_dir": str(src / "stage05"),
        "staged_model": str(real / "run" / "next" / "stage06" / "m.bin"),
        "note": f"from {src}",
        "other": str(link / "run" / "next2" / "x"),
        "same": str(src),
    }
    (src / "stage06" / "deploy_record.json").write_text(json.dumps(rec, indent=1))
    (src / "stage06" / ro.STAGE_MARKER).write_text(json.dumps({"cmd": [str(src)]}))
    (src / "log.txt").write_text(str(src))
    ro.relocate(src, dst)
    assert not src.exists() and not (dst / ro.RELOCATED).exists()
    got = json.loads((dst / "stage06" / "deploy_record.json").read_text())
    assert got == {
        "train_dir": str(dst / "stage05"),
        "staged_model": str(real / "run" / "final" / "stage06" / "m.bin"),
        "note": f"from {dst}",
        "other": str(link / "run" / "next2" / "x"),
        "same": str(dst),
    }
    assert json.loads((dst / "stage06" / ro.STAGE_MARKER).read_text()) == {
        "cmd": [str(dst)]
    }
    assert (dst / "log.txt").read_text() == str(src)

    # interrupted after the move: the marker is still there
    (dst / "stage06" / "deploy_record.json").write_text(json.dumps(rec))
    (dst / ro.RELOCATED).write_text(
        json.dumps({"paths": {str(src): str(dst)}}, indent=2)
    )
    ro.finish_relocation(dst)
    got = json.loads((dst / "stage06" / "deploy_record.json").read_text())
    assert got["train_dir"] == str(dst / "stage05")
    assert not (dst / ro.RELOCATED).exists()
    ro.finish_relocation(dst)


# ── stub runs: the audit set built beside the rounds ─────────────────────────

FAKE_AUDIT_BUILD = r"""
import json, os, sys, time
from pathlib import Path

audit = Path(sys.argv[1])
(audit / "stage01").mkdir(parents=True, exist_ok=True)
(audit / "stage01" / "shapes.yaml").write_text("- {M: 8, N: 8, K: 8, batch_count: 1}\n")
(audit / "stage01" / ".stage_complete.json").write_text("{}")
Path(os.environ["STUB_BUILD_PID"]).write_text(str(os.getpid()))
gate = Path(os.environ["STUB_BUILD_GATE"])
while not gate.exists():
    time.sleep(0.1)
rc = int(gate.read_text() or 0)
if rc == 0:
    (audit / ".audit_complete.json").write_text("{}")
sys.exit(rc)
"""


@pytest.fixture
def parallel(tmp_path, audit_stubs, monkeypatch):
    """A parallel audit build that is a fake child: it draws the audit
    shapes at once, then waits for the gate file (its exit code) and
    completes the set; the audit status follows its marker."""
    script = tmp_path / "fake_audit_build.py"
    script.write_text(FAKE_AUDIT_BUILD)
    audit = tmp_path / "audit"
    gate = tmp_path / "build_gate"
    pid = tmp_path / "build.pid"
    monkeypatch.setenv("STUB_BUILD_GATE", str(gate))
    monkeypatch.setenv("STUB_BUILD_PID", str(pid))

    def command(self):
        return [sys.executable, str(script), str(self.dir)]

    def status(cfg, paths):
        if (audit / al.COMPLETE_MARKER).is_file():
            return "ok", None
        return "incomplete", "the audit set is still being benched"

    monkeypatch.setattr(ro.AuditBuild, "command", command)
    monkeypatch.setattr(ro, "audit_status", status)
    return {"gate": gate, "pid": pid, "audit": audit}


def _parallel_config(tmp_path, **overrides):
    return audited_config(
        tmp_path,
        **{
            "audit.build": "parallel",
            "audit.devices": [5],
            "audit.v2": "rounds",
            "full.total_rounds": 2,
            **overrides,
        },
    )


def test_a_parallel_build_runs_beside_the_rounds_and_joins_at_the_end(
    tmp_path, audit_stubs, parallel, env, monkeypatch
):
    cfg = _parallel_config(tmp_path)
    run_dir = tmp_path / "run"
    seen = {}
    real = ro.Pipeline.finish_run

    def finish(self, build):
        seen["running"] = build.proc is not None and build.proc.poll() is None
        parallel["gate"].write_text("0")
        return real(self, build)

    monkeypatch.setattr(ro.Pipeline, "finish_run", finish)
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    assert seen["running"]
    log = calls(env)
    s01 = [c for c in log if c["stage"] == "stage01"]
    assert all(
        arg(c, "--exclude-shapes-yaml")
        == str(parallel["audit"] / "stage01" / "shapes.yaml")
        for c in s01
    )
    st = by_stage(log)
    (v2_end,) = st["audit_v2"]
    assert log.index(v2_end) > log.index(st["stage05"][-1])
    assert [m.split("=")[0] for m in values(v2_end, "--model")] == [
        "run.round_0",
        "run.round_1",
    ]
    rescored = [c for c in st["audit_eval"] if "rescore" in (arg(c, "--out") or "")]
    assert [values(c, "--round-dir") for c in rescored] == [
        [str(run_dir / "round_0"), str(run_dir / "round_1")]
    ]
    timing = json.loads(sorted((run_dir / "timing").glob("*.json"))[-1].read_text())
    assert any(r["stage"] == "audit set" and r["rc"] == 0 for r in timing["rows"])


PARALLEL_PARENT = """
import sys, time
from pathlib import Path

sys.path.insert(0, {root!r})
import run_orchestrator as ro
from lib import attempts

run_dir = Path({run_dir!r})
started = ro.start_logged(
    [sys.executable, "-c", "import time; time.sleep(600)"],
    run_dir / "audit_build.log",
    running=attempts.Running(run_dir),
    label=ro.AuditBuild.LABEL,
)
print("CHILD", started.proc.pid, started.token, flush=True)
time.sleep(600)
"""


@pytest.mark.skipif(procs.die_with_parent() is None, reason="needs prctl")
def test_a_parallel_build_ends_with_an_orchestrator_killed_by_sigkill(tmp_path):
    """The parallel audit set build is started as the stages are: it gets
    SIGTERM when its orchestrator dies and carries an attempt token that
    .running.json names, so that the next invocation stops what is left
    before it checks the audit directory's lock."""
    import subprocess
    import textwrap

    from gpu_fakes import wait_for
    from lib import attempts
    from test_attempts import _alive

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    script = textwrap.dedent(
        PARALLEL_PARENT.format(root=str(Path(ro.__file__).parent), run_dir=str(run_dir))
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    try:
        line = ""
        while not line.startswith("CHILD"):
            line = proc.stdout.readline()
            assert line, "the parent ended before it started the build"
        _, pid, token = line.split()
        child = int(pid)
        assert attempts.processes_of(token) == [child]
        (rec,) = attempts.Running(run_dir).records().values()
        assert rec["label"] == ro.AuditBuild.LABEL and rec["pid"] == child
        proc.kill()
        proc.wait()
        assert wait_for(lambda: not _alive(child), 10)
        assert ro.stop_survivors(run_dir) == []
        assert attempts.Running(run_dir).records() == {}
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_a_joined_parallel_build_leaves_no_record(
    tmp_path, audit_stubs, parallel, env, monkeypatch
):
    from lib import attempts

    cfg = _parallel_config(tmp_path)
    run_dir = tmp_path / "run"
    seen = {}
    real = ro.Pipeline.finish_run

    def finish(self, build):
        seen["records"] = attempts.Running(run_dir).records()
        parallel["gate"].write_text("0")
        return real(self, build)

    monkeypatch.setattr(ro.Pipeline, "finish_run", finish)
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    (rec,) = seen["records"].values()
    assert rec["label"] == ro.AuditBuild.LABEL
    assert attempts.Running(run_dir).records() == {}


def test_a_failed_parallel_build_fails_the_end_of_the_run(
    tmp_path, audit_stubs, parallel, env, capsys
):
    parallel["gate"].write_text("3")
    cfg = _parallel_config(tmp_path, **{"full.total_rounds": 1})
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 3
    assert ro.round_complete(run_dir / "round_0")
    err = capsys.readouterr().err
    assert "building the audit set failed (rc=3)" in err
    assert "the same command resumes it" in err
    assert "the rounds' audit records stay as they are" in err


@pytest.mark.parametrize("when", ["before", "parallel"])
def test_a_failed_final_retrain_still_completes_the_end_of_the_run(
    tmp_path, audit_stubs, parallel, env, monkeypatch, capsys, when
):
    monkeypatch.setenv("STUB_FAIL", "stage05@validate")
    seen = {}
    if when == "before":
        parallel["gate"].write_text("0")
    else:
        real = ro.FinalRetrain.train

        def train(self):
            rc = real(self)
            build = parallel["audit"] / al.COMPLETE_MARKER
            seen["building"] = not build.exists()
            parallel["gate"].write_text("0")
            return rc

        monkeypatch.setattr(ro.FinalRetrain, "train", train)
    cfg = _parallel_config(
        tmp_path,
        **{
            "audit.build": when,
            "audit.v2": "final",
            "full.final_retrain.enabled": True,
        },
    )
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 1
    final = run_dir / ro.FINAL_RETRAIN_DIR
    assert seen.get("building", True)
    assert (parallel["audit"] / al.COMPLETE_MARKER).is_file()
    log = calls(env)
    st = by_stage(log)
    failed = st["stage05"][-1]
    assert arg(failed, "--output-dir") == str(final / "stage05")
    (v2_end,) = st["audit_v2"]
    assert log.index(v2_end) > log.index(failed)
    assert [m.split("=")[0] for m in values(v2_end, "--model")] == ["run.round_1"]
    rescored = [c for c in st["audit_eval"] if "rescore" in (arg(c, "--out") or "")]
    assert rescored and log.index(rescored[0]) > log.index(failed)
    assert all(arg(c, "--out") != str(final / "audit.json") for c in st["audit_eval"])
    assert not (final / ro.FinalRetrain.DECISION).exists()
    assert all(arg(c, "--out-dir") != str(final / "stage06") for c in st["stage06"])
    timing = json.loads(sorted((run_dir / "timing").glob("*.json"))[-1].read_text())
    assert any(r["stage"] == "audit set" and r["rc"] == 0 for r in timing["rows"])
    out = capsys.readouterr()
    assert "the final retrain's stage05 failed (rc=1" in out.err
    assert "run the same command to train it again" in out.err
    assert "--from-stage" not in out.err
    assert "the end of the run is incomplete" in out.out
    monkeypatch.delenv("STUB_FAIL")
    n = len(calls(env))
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 0
    st = by_stage(calls(env)[n:])
    assert [arg(c, "--output-dir") for c in st["stage05"]] == [str(final / "stage05")]
    assert (final / ro.FinalRetrain.DECISION).is_file()


def test_an_interrupt_stops_the_parallel_build(
    tmp_path, audit_stubs, parallel, env, monkeypatch
):
    cfg = _parallel_config(tmp_path)
    run_dir = tmp_path / "run"
    real = ro.Pipeline.run_stage

    def interrupting(self, owner_dir, key, args, round_idx, out_dir=None, **kw):
        if key == "stage03":
            raise ro.Interrupted(signal.SIGTERM)
        return real(self, owner_dir, key, args, round_idx, out_dir, **kw)

    monkeypatch.setattr(ro.Pipeline, "run_stage", interrupting)
    rc = run(audit_stubs, cfg, run_dir, "--mode", "full")
    assert rc == 128 + signal.SIGTERM
    pid = int(parallel["pid"].read_text())
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the parallel audit build outlived the run")


# ── end to end on the fake bench ─────────────────────────────────────────────

DATASET = ((1024, 256, 2048), (2048, 512, 64), (4096, 1024, 1024))


@pytest.fixture(scope="module")
def one_command(tmp_path_factory):
    """`--mode full` on the fake bench, with real stages, whose config points
    at an audit set that does not exist yet, measures every round under
    protocol v2 and retrains every cell at the end; then the same command
    again."""
    pytest.importorskip("tilewright")
    root = tmp_path_factory.mktemp("onecmd")
    mp = pytest.MonkeyPatch()
    try:
        yield _one_command(root, mp)
    finally:
        mp.undo()


def _files(root):
    return {
        str(p.relative_to(root)): p.stat().st_mtime_ns
        for p in Path(root).rglob("*")
        if p.is_file() and p.name != ro.LOCK_NAME
    }


def _one_command(root, mp):
    import test_audit as ta

    fh.unhide_devices(mp)
    state = root / "gpu_state"
    (state / "sys").mkdir(parents=True)
    mp.setenv("TILEWRIGHT_GPU_LEASE_DIR", str(state / "leases"))
    mp.setenv("TILEWRIGHT_SYSFS_ROOT", str(state / "sys"))
    mp.setenv("TILEWRIGHT_AMD_SMI", "")
    build = root / "bench" / "projects" / "hipblaslt" / "build" / "release"
    kernels = fh.kernels_grid()
    fh.write_fake_build(build, ta.ARCH, bf.LIBRARY_STEM, kernels)
    mp.setenv("FAKE_LIBRARY_DIR", str(build / "Tensile" / "library" / ta.ARCH))
    mp.setenv("FAKE_LIBRARY_STEM", bf.LIBRARY_STEM)
    audit = root / "audit_set"
    cfg = ta.base_config(root / "bench", audit)
    cfg["runtime"]["devices"] = [0, 1, 2]
    cfg["audit"]["v2"] = "rounds"
    cfg["full"]["final_retrain"] = {"enabled": True, "recipe": {"lam": 0.4}}
    dataset = root / "dataset.yaml"
    dataset.write_text(
        "".join(
            f"- {{M: {m}, N: {n}, K: {k}, batch_count: 1}}\n" for m, n, k in DATASET
        )
    )
    cfg["stage08"] = {"datasets": [{"name": "ds", "path": str(dataset)}]}
    assert ro.validate_config(cfg) == ([], [])
    cfg_path = root / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    run_dir = root / "run"
    cmd = ["--config", str(cfg_path), "--mode", "full", "--run-dir", str(run_dir)]
    rc = ro.main(cmd)
    before = {"audit": _files(audit), "run": _files(run_dir)}
    rc_again = ro.main(cmd)
    return {
        "rc": rc,
        "rc_again": rc_again,
        "before": before,
        "audit": audit,
        "run": run_dir,
        "cfg": cfg,
        "cfg_path": cfg_path,
        "kernels": kernels,
        "libdir": build / "Tensile" / "library" / ta.ARCH,
        "enhanced": _enhance_a_copy(root, run_dir, cfg),
    }


def _enhance_a_copy(root, run_dir, cfg):
    """`--mode enhance` of one cell on a copy of the finished run (protocol
    v2 off, so that the shared audit set is left as it is): (its exit
    status, the cell, the copy)."""
    import shutil

    import torch

    copy = root / "enhanced"
    shutil.copytree(run_dir, copy, symlinks=True)
    final = torch.load(
        copy / ro.FINAL_RETRAIN_DIR / "stage05" / "models.pt", weights_only=True
    )
    cell = sorted(final["models"])[0]
    other = json.loads(json.dumps(cfg))
    other["audit"]["v2"] = "off"
    cfg_path = root / "enhance.yaml"
    cfg_path.write_text(yaml.safe_dump(other))
    argv = ["--config", str(cfg_path), "--mode", "enhance", "--cells", cell]
    rc = ro.main(argv + ["--run-dir", str(copy), "--no-report"])
    return rc, cell, copy


def _marker_cmd(stage_dir):
    return json.loads((stage_dir / ro.STAGE_MARKER).read_text())["cmd"]


def test_one_command_builds_the_audit_set_before_its_first_round(one_command):
    w = one_command
    audit, run_dir = w["audit"], w["run"]
    assert w["rc"] == 0
    pool = al.pool_digests(
        ro.bank_pool.config_pool(w["cfg"], w["libdir"], bf.LIBRARY_STEM, 0).kernels
    )
    assert al.status(audit, pool) == ("ok", None)
    m = al.manifest(audit)
    assert m["source_runs"] == [str(run_dir)]
    assert "mode=audit" in (audit / "orchestrator.log").read_text()
    assert "audit set complete" in (run_dir / "audit_build.log").read_text()
    shapes = al.shapes_yaml(audit)
    built = (audit / al.COMPLETE_MARKER).stat().st_mtime_ns
    audit_shapes = set(parse_shapes_yaml(shapes))
    for rd in (run_dir / "round_0", run_dir / "round_1"):
        cmd = _marker_cmd(rd / "stage01")
        assert cmd[cmd.index("--exclude-shapes-yaml") + 1] == str(shapes)
        assert (rd / "stage01" / ro.STAGE_MARKER).stat().st_mtime_ns > built
        assert not set(parse_shapes_yaml(rd / "stage01" / "shapes.yaml")) & audit_shapes
    first = sorted((run_dir / "timing").glob("*.json"))[0]
    rows = json.loads(first.read_text())["rows"]
    assert rows[0]["stage"] == "audit set" and rows[0]["rc"] == 0


def test_one_command_measures_every_round_under_v2(one_command):
    w = one_command
    run_dir = w["run"]
    d = v2.v2_dir(w["audit"])
    names = sorted(v2.models(d))
    assert [n.rsplit(".", 1)[0] for n in names] == [
        "run.final_retrain",
        "run.round_0",
        "run.round_1",
    ]
    assert all(len(n.rsplit(".", 1)[1]) == 12 for n in names)
    for rec_path in (
        run_dir / "round_0" / "stage05" / al.RESULT_FILE,
        run_dir / "round_1" / "stage04b" / al.RESULT_FILE,
        run_dir / "round_1" / "stage06" / al.RESULT_FILE,
    ):
        rec = json.loads(rec_path.read_text())
        assert rec["status"] == "ok" and rec["v2"]["status"] == "ok", rec_path
        assert rec["sustained"]["status"] == "ok", rec_path
        assert rec["official"] == "sustained"
    m = v2.manifest(asus.sustained_dir(w["audit"]))
    assert m["regime"] == "sustained" and m["generations"]
    assert not (run_dir / ro.AUDIT_RESULTS_DIR).exists()
    deployed = json.loads(
        (run_dir / "round_1" / "stage06" / "deploy_record.json").read_text()
    )
    official = deployed["audit"]["official"]
    assert official["protocol"] == "sustained"
    assert official["v2"]["cell_mean"] == rec["v2"]["headline"]["cell_mean"]
    page = (run_dir / "report.html").read_text()
    assert "v2 (exclusive)" in page and al.PROTOCOLS["sustained"] in page
    assert "audit set: the same under protocol v2, alongside" in page


def test_one_command_retrains_every_cell_and_gates_it_on_the_audit_set(one_command):
    w = one_command
    run_dir = w["run"]
    final = run_dir / ro.FINAL_RETRAIN_DIR
    cmd = _marker_cmd(final / "stage05")
    data = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--enriched-csv-dir"]
    assert data == [str(run_dir / f"round_{i}" / "stage04") for i in (0, 1)]
    assert "--only-cells" not in cmd and cmd[-2:] == ["--lam", "0.4"]
    dec = json.loads((final / ro.FinalRetrain.DECISION).read_text())
    assert dec["gate"] == "not-worse" and dec["protocol"] == "sustained"
    assert dec["official"] == "sustained"
    assert dec["final"]["v2"]["n_gemms"] == dec["final"]["n_gemms"]
    assert dec["baseline"]["round"] == "round_1"
    ratio = dec["change"]["cell_mean_ratio"]
    assert dec["cell_gate_settings"] == al.CELL_GATE_DEFAULTS
    assert dec["baseline"]["deployed"]
    cells = dec["cell_gate"]
    assert cells["groups"]["exclusive"]["protocol"] == "sustained"
    assert cells["n_groups"] == 1
    assert dec["change"]["n_gemms"] > 0
    assert dec["accepted"] == (ratio >= 1 and cells["passed"] is True)
    paired = dec["change"]["n_gemms"]
    assert paired == dec["final"]["n_gemms"] == dec["baseline"]["n_gemms"]
    s08 = _marker_cmd(run_dir / "validate" / "stage08")
    shipped = final if dec["accepted"] else run_dir / "round_1"
    assert s08[s08.index("--train-dir") + 1] == str(shipped / "stage05")
    assert (final / "stage06").is_dir() == dec["accepted"]
    if dec["accepted"]:
        rec = json.loads((final / "stage06" / al.RESULT_FILE).read_text())
        assert rec["n_excluded_trained"] == 0 and rec["v2"]["status"] == "ok"
        # its gate scored the bytes stage06 deployed
        deployed = json.loads((final / "stage06" / "deploy_record.json").read_text())
        assert dec["final"]["model_sha256"] == deployed["sha256"]
        assert rec["model_sha256"] == deployed["sha256"]
    breakdown = json.loads((run_dir / "held_out_sel_eff_breakdown.json").read_text())
    assert breakdown["final_retrain"]["ships"] == dec["accepted"]
    assert breakdown["final_retrain"]["selected_by_audit"]
    page = (run_dir / "report.html").read_text()
    if dec["accepted"]:
        assert breakdown["shipped"]["round"] == ro.FINAL_RETRAIN_DIR
        assert "deployed bundle (final_retrain)" in page
        assert "extra model final_retrain/stage05" not in page
    else:
        assert "extra model final_retrain/stage05" in page
    assert '<td class="c">run</td>' in page
    for column in ("audit set build", "final retrain stage05", "audit v2"):
        assert f"<th>{column}</th>" in page


def test_one_command_writes_the_weak_spots_with_its_report(one_command):
    from lib import weak_spots as ws

    w = one_command
    d = w["run"] / ro.WEAK_SPOTS_DIR
    doc = json.loads((d / ws.POOL_GAPS_JSON).read_text())
    # the run plans no seed bench, so no training-side GEMM gives gaps; the
    # held-out audit set gives the cells' roofline, never a shape
    assert doc["schema"] == ws.SCHEMA and doc["source"] is None
    assert doc["held_out"] is False and doc["gaps"] == [] and doc["n_gemms"] == 0
    assert "no exhaustive training bench" in doc["why"]
    assert doc["threshold"] == ws.POOL_GAP_FRACTION and doc["pool_digest"]
    assert doc["layout"]["arch"] and doc["peaks"]["latency_floor_us"] > 0
    header = (d / ws.POOL_GAPS_CSV).read_text().splitlines()
    assert header == [",".join(ws.CSV_FIELDS)]
    weak = json.loads((d / ws.WEAK_SPOTS_JSON).read_text())
    assert weak["summary"][0]["group"] == "exclusive" and weak["cells"]
    assert (weak["roofline_source"], weak["roofline_held_out"]) == (
        ws.HELD_OUT_SOURCE,
        True,
    )
    assert any(c["roofline"] is not None for c in weak["cells"])
    page = (w["run"] / "report.html").read_text()
    section = page[page.index('<h2 id="weak">Weak spots</h2>') :]
    assert "<h3>Weak cells (" in section[: section.index("<h2", 1)]


def test_the_same_command_again_finds_nothing_to_do(one_command):
    w = one_command
    assert w["rc_again"] == 0
    assert _files(w["audit"]) == w["before"]["audit"]
    after = _files(w["run"])
    changed = sorted(
        p
        for p, t in after.items()
        if w["before"]["run"].get(p) != t
        and not p.startswith(("configs/", "timing/", f"{ro.WEAK_SPOTS_DIR}/"))
        and p
        not in (
            "orchestrator.log",
            "report.html",
            "held_out_sel_eff_breakdown.json",
            ro.RUN_STATUS,
        )
    )
    assert changed == []
    status = json.loads((w["run"] / ro.RUN_STATUS).read_text())
    assert status["complete"] and status["invocation"]["exit_status"] == 0
    assert status["final_retrain"]["status"] in ("shipped", "rejected")
    last = sorted((w["run"] / "timing").glob("*.json"))[-1]
    assert json.loads(last.read_text())["rows"] == []


def _benching_copy(src, dst):
    """A copy of the complete audit set `src` as it was while it was
    benched: no completion marker."""
    import shutil

    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(ro.LOCK_NAME))
    (dst / al.COMPLETE_MARKER).unlink()
    return dst


def test_preflight_refuses_an_audit_set_the_run_could_not_build(
    one_command, tmp_path, capsys
):
    w = one_command
    paths = ro.derive_paths(w["cfg"])

    def errors(**audit):
        cfg = json.loads(json.dumps(w["cfg"]))
        cfg["audit"].update(audit)
        return ro.preflight(cfg, paths, "full")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ro, "_tilewright_errors", lambda catalog="v2": [])
        assert errors() == []
        benching = _benching_copy(w["audit"], tmp_path / "benching")
        assert errors(dir=str(benching)) == []
        assert any(
            "was built with shapes_per_cell=4" in e
            for e in errors(dir=str(benching), shapes_per_cell=5)
        )
        fd = ro.lock_run_dir(benching)
        try:
            assert any(
                "another process is building the audit set" in e
                for e in errors(dir=str(benching))
            )
        finally:
            os.close(fd)
        inside = tmp_path / "train_run"
        (inside / "configs").mkdir(parents=True)
        (inside / "orchestrator.log").write_text("")
        nested = _benching_copy(w["audit"], inside / "audit")
        assert any("lies inside the training run" in e for e in errors(dir=str(nested)))
        replanned = _benching_copy(w["audit"], tmp_path / "replanned")
        m = al.manifest(replanned)
        m["generations"] = []
        m["planned_kernels"] = {k: "0" * 16 for k in m["planned_kernels"]}
        (replanned / al.MANIFEST).write_text(json.dumps(m))
        got = errors(dir=str(replanned))
        assert any(
            "cannot be built further with this hipBLASLt build: its shapes were "
            "planned with another hipBLASLt build" in e
            for e in got
        ), got
        cfg = json.loads(json.dumps(w["cfg"]))
        cfg["audit"].update(dir=str(benching), build="off")
        capsys.readouterr()
        assert ro.preflight(cfg, paths, "full") == []
        assert "full.final_retrain.gate not-worse decides once --mode audit" in (
            capsys.readouterr().err
        )
        cfg["full"]["final_retrain"]["enabled"] = False
        assert ro.preflight(cfg, paths, "full") == []
        assert "decides once" not in capsys.readouterr().err
        common = paths.bench_binary.parent / "hipblaslt_common.yaml"
        saved = common.read_text()
        common.write_text("Arguments: []\n")
        try:
            got = errors(dir=str(benching))
        finally:
            common.write_text(saved)
        assert got and all(e.startswith("audit set: planner.enabled") for e in got)


def _same(a, b):
    import torch

    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(_same(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)):
        return (
            isinstance(b, (list, tuple))
            and len(a) == len(b)
            and all(_same(x, y) for x, y in zip(a, b))
        )
    return a == b


def test_a_round_after_the_shipped_final_retrain_keeps_its_cells(one_command):
    """An enhance round after the final retrain shipped retrains the cell it
    names and carries every other cell of the final retrain unchanged, not
    the final round's models."""
    import torch

    rc, cell, run_dir = one_command["enhanced"]
    assert rc == 0
    final = run_dir / ro.FINAL_RETRAIN_DIR
    dec = json.loads((final / ro.FinalRetrain.DECISION).read_text())
    assert dec["accepted"], "the fake run's final retrain is expected to ship"

    def models(d):
        path = d / "stage05" / "models.pt"
        return torch.load(path, weights_only=True)["models"]

    retrain, round_1, round_2 = (
        models(d) for d in (final, run_dir / "round_1", run_dir / "round_2")
    )
    assert json.loads((run_dir / "round_2" / ro.ENHANCE_RECORD).read_text())[
        "retrain"
    ] == [cell]
    assert sorted(round_2) == sorted(retrain)
    assert not _same(round_2[cell], retrain[cell])
    others = [c for c in retrain if c != cell]
    assert others
    for c in others:
        assert _same(round_2[c], retrain[c]), c
        assert not _same(round_2[c], round_1[c]), c
    cmd = _marker_cmd(run_dir / "round_2" / "stage04b")
    assert cmd[cmd.index("--models-dir") + 1] == str(final / "stage05")
    # round_2 records that the final retrain trained every cell it carried,
    # with the final retrain's cells.json entry and model options
    origins = json.loads(
        (run_dir / "round_2" / "stage05" / rb.CELL_ORIGINS).read_text()
    )
    trained = {c["label"]: c for c in rb.trained_entries(final)}
    options = json.loads((final / "stage05" / "model_options.json").read_text())
    assert origins["source"] == ro.FINAL_RETRAIN_DIR
    assert sorted(origins["cells"]) == sorted(others)
    (key,) = origins["origins"]
    assert origins["origins"][key] == {
        "bundle": ro.FINAL_RETRAIN_DIR,
        "finished": rb.stage05_finished(final),
        "model_options": options,
    }
    for c, o in origins["cells"].items():
        assert o == {"origin": key, "entry": trained[c]}


def test_an_audit_set_inside_the_run_directory_is_refused(one_command, tmp_path):
    w = one_command
    cfg = json.loads(json.dumps(w["cfg"]))
    run_dir = tmp_path / "run"
    cfg["audit"]["dir"] = str(run_dir / "audit")
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    argv = ["--config", str(cfg_path), "--mode", "full", "--run-dir", str(run_dir)]
    assert ro.main(argv) == 2
    assert not run_dir.exists()


def test_a_failed_build_before_the_rounds_stops_the_run(
    tmp_path, audit_stubs, env, monkeypatch, capsys
):
    failing = tmp_path / "fail.py"
    failing.write_text("import sys\nsys.exit(4)\n")
    monkeypatch.setattr(
        ro.AuditBuild, "command", lambda self: [sys.executable, str(failing)]
    )
    monkeypatch.setattr(
        ro, "audit_status", lambda cfg, paths: ("missing", "no audit set yet")
    )
    cfg = audited_config(tmp_path)
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 4
    assert calls(env) == [] and not ro.round_dirs(run_dir)
    assert "building the audit set failed (rc=4)" in capsys.readouterr().err
    build = ro.AuditBuild(
        ro.Pipeline(
            cfg=yaml.safe_load(cfg.read_text()),
            snapshot=cfg,
            run_dir=run_dir,
            paths=ro.derive_paths(yaml.safe_load(cfg.read_text())),
            stages_dir=audit_stubs,
            timing=ro.Timing(),
            force_deploy=False,
        ),
        ro.parse_args(["--config", str(cfg), "--mode", "full"]),
    )
    monkeypatch.undo()
    cmd = build.command()
    assert cmd[cmd.index("--mode") + 1] == "audit"
    assert cmd[cmd.index("--run-dir") + 1] == str(tmp_path / "audit")
    assert cmd[cmd.index("--source-run-dir") + 1] == str(run_dir)
    assert cmd[cmd.index("--devices") + 1] == "0,1"
    assert cmd[cmd.index("--config") + 1] == str(cfg)


def test_a_build_refused_on_its_pin_names_the_change_and_takes_the_flag(
    tmp_path, audit_stubs, env, monkeypatch, capsys
):
    """An audit set produced with other values of the keys its benches read
    refuses the build (`settle_run_pin`): the run names the change and the
    remedy, and the same command with --allow-code-change hands the flag to
    the build."""
    audit = tmp_path / "audit"
    audit.mkdir()
    old = {"config": ro.pinning.config_record({"bench.rotating_mb": 256})}
    ro.pinning.write_run_pin(audit, old, first_stage="stage01")
    new = {"config": ro.pinning.config_record({"bench.rotating_mb": 512})}
    errs = ro.settle_run_pin(audit, new, False, ro.pin_parts("audit"))
    refusing = tmp_path / "refusing.py"
    refusing.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ro.THIS_DIR)!r})\n"
        "import run_orchestrator as ro\n"
        f"ro._print_problems('pin', {errs!r}, [])\n"
        "sys.exit(2)\n"
    )
    seen = []

    def command(self):
        seen.append(self.allow_change)
        return [sys.executable, str(refusing)]

    monkeypatch.setattr(ro.AuditBuild, "command", command)
    monkeypatch.setattr(
        ro, "audit_status", lambda cfg, paths: ("incomplete", "1 kernel to bench")
    )
    monkeypatch.setattr(ro, "audit_build_refusal", lambda d, current: None)
    cfg = audited_config(tmp_path)
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 2
    err = capsys.readouterr().err
    assert "config bench.rotating_mb: 256 -> 512" in err
    assert "The same command with --allow-code-change extends it" in err
    assert "the same command resumes it" not in err
    assert ro.pin_refusal(run_dir / "audit_build.log") == [
        "config bench.rotating_mb: 256 -> 512"
    ]
    assert run(audit_stubs, cfg, run_dir, "--mode", "full", "--allow-code-change") == 2
    assert seen == [False, True]
    monkeypatch.undo()
    args = ro.parse_args(["--config", str(cfg), "--mode", "full"])
    pipeline = ro.Pipeline(
        cfg=yaml.safe_load(cfg.read_text()),
        snapshot=cfg,
        run_dir=run_dir,
        paths=ro.derive_paths(yaml.safe_load(cfg.read_text())),
        stages_dir=audit_stubs,
        timing=ro.Timing(),
        force_deploy=False,
    )
    assert "--allow-code-change" not in ro.AuditBuild(pipeline, args).command()
    args.allow_code_change = True
    assert "--allow-code-change" in ro.AuditBuild(pipeline, args).command()


@pytest.mark.parametrize("planner", ["stage02", "stage01b"])
def test_a_build_refused_for_another_hipblaslt_build_is_named_so(
    tmp_path, audit_stubs, env, monkeypatch, capsys, planner
):
    """Of an audit directory in either layout (stage01b/ when written before
    the planner became stage02)."""
    audit = tmp_path / "audit"
    if planner == "stage02":
        run_layout.settle(audit)
    (audit / planner).mkdir(parents=True)
    (audit / planner / ro.STAGE_MARKER).write_text("{}")
    m = {
        "schema": al.SCHEMA_VERSION,
        "generations": [],
        "planned_kernels": {"3": "aaa", "4": "bbb"},
    }
    (audit / al.MANIFEST).write_text(json.dumps(m))
    planned = "its shapes were planned with another hipBLASLt build"
    assert ro.audit_build_refusal(audit, {3: "aaa", 4: "bbb"}) is None
    assert ro.audit_build_refusal(audit, {3: "aaa", 4: "ccc"}) == planned
    (audit / planner / ro.STAGE_MARKER).unlink()
    assert ro.audit_build_refusal(audit, {3: "aaa", 4: "ccc"}) is None
    m["generations"] = [{"index": 0, "kernels": {"3": "aaa", "4": "bbb"}}]
    (audit / al.MANIFEST).write_text(json.dumps(m))
    assert ro.audit_build_refusal(audit, {3: "aaa", 4: "bbb"}) is None
    inc = audit / "increments" / "inc_001"
    inc.mkdir(parents=True)
    (inc / "increment.json").write_text(json.dumps({"kernels": {"5": "ddd"}}))
    assert ro.audit_build_refusal(audit, {3: "aaa", 4: "bbb", 5: "ddd"}) is None
    got = ro.audit_build_refusal(audit, {3: "aaa", 4: "bbb", 5: "eee"})
    assert got == (
        f"its increment 1 ({inc / 'increment.json'}) was planned with another "
        f"hipBLASLt build"
    )

    failing = tmp_path / "fail.py"
    failing.write_text("import sys\nsys.exit(2)\n")
    monkeypatch.setattr(
        ro.AuditBuild, "command", lambda self: [sys.executable, str(failing)]
    )
    monkeypatch.setattr(
        ro, "audit_status", lambda cfg, paths: ("incomplete", "1 kernel to bench")
    )
    current = {3: "aaa", 4: "bbb", 5: "eee"}
    monkeypatch.setattr(ro.Pipeline, "pool_digests", lambda self: current)
    cfg = audited_config(tmp_path)
    run_dir = tmp_path / "run"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 2
    err = capsys.readouterr().err
    assert "hipBLASLt build, so the same command cannot resume it" in err
    assert "the same command resumes it" not in err
    current[5] = "ddd"
    assert run(audit_stubs, cfg, run_dir, "--mode", "full") == 2
    assert "the same command resumes it" in capsys.readouterr().err
