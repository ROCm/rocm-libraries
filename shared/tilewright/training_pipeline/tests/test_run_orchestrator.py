# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Orchestrator behavior with stub stage scripts (no bench, no training)."""

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

import run_orchestrator as ro

STEM = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_ID75a0_gfx950"

STUB = r"""
import json, os, signal, sys
from pathlib import Path

KEY = __KEY__
argv = sys.argv[1:]


def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


if os.environ.get("STUB_LOG"):
    with open(os.environ["STUB_LOG"], "a") as f:
        f.write(json.dumps({"stage": KEY, "argv": argv}) + "\n")
if os.environ.get("STUB_SLEEP") == KEY:
    import time
    Path(os.environ["STUB_PID_FILE"]).write_text(str(os.getpid()))
    time.sleep(600)
out = opt("--out-dir") or opt("--output-dir")
rnd = next((p for p in Path(out).parts if p.startswith("round_")), "validate")
for spec in filter(None, os.environ.get("STUB_FAIL", "").split(",")):
    name, _, how = spec.partition(":")
    if name not in (KEY, KEY + "@" + rnd):
        continue
    once = os.environ.get("STUB_FAIL_ONCE_DIR")
    if once:
        marker = Path(once) / (KEY + rnd + ".failed")
        if marker.exists():
            continue
        marker.write_text("1")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "partial.txt").write_text("partial")
    if KEY == "stage05":
        (Path(out) / "metrics.json").write_text(json.dumps({"cells_failed": [],
            "cells_not_trained": [{"cell": "X#M>1", "reason": "too few gemms (2)"}]}))
    if how == "signal":
        os.kill(os.getpid(), signal.SIGKILL)
    sys.exit(int(how or 1))
o = Path(out)
o.mkdir(parents=True, exist_ok=True)
if KEY == "stage01":
    (o / "shapes.yaml").write_text("- {M: 64, N: 64, K: 64, batch_count: 1}\n")
elif KEY == "stage02":
    (o / "shapes_probed.yaml").write_text("- {M: 64, N: 64, K: 64, batch_count: 1}\n")
elif KEY == "stage03":
    (o / "logs").mkdir()
    (o / "logs" / "block_0000.log").write_text("log\n")
elif KEY == "stage04":
    (o / "chunk_0000.csv").write_text("m,n,k,batch_count\n64,64,64,1\n")
elif KEY == "stage04b":
    retrain = json.loads(os.environ.get("STUB_RETRAIN", "{}")).get(rnd, "Tiny|Tiny|TinyK|Bnone")
    se = float(os.environ.get("STUB_SEL_EFF", "0.9"))
    (o / "decisions.json").write_text(json.dumps({
        "models_dir": opt("--models-dir"),
        "aggregate": {"global_sel_eff_new": se, "n_gemms_evaluated": 40},
        "per_cell": [
        {"cell": "Tiny|Tiny|TinyK|Bnone", "sel_eff_new": se, "n_eval": 10,
         "n_training_gemms": 40},
        {"cell": "Large|Large|LargeK|Bnone", "sel_eff_new": se, "n_eval": 30,
         "n_training_gemms": 40}]}))
    (o / "retrain_cells.txt").write_text(retrain)
    (o / "splits.json").write_text(json.dumps({"splits": {}}))
    if os.environ.get("STUB_GATE"):
        for sub in ("gate", "train"):
            (o / sub).mkdir()
            (o / sub / "chunk_0000.csv").write_text("m,n,k,batch_count\n64,64,64,1\n")
elif KEY == "stage05":
    (o / "models.pt").write_text("models of " + rnd)
    (o / "cells.json").write_text(json.dumps({"cells": []}))
elif KEY == "stage06":
    rec = {"train_dir": opt("--train-dir")}
    if "--gate-csv-dir" in argv:
        se = float(os.environ.get("STUB_GATE_SE", "0.9"))
        target = opt("--gate-target")
        passed = (
            target is None
            or se >= float(target)
            or opt("--gate-fallback") == "not-worse"
            and (opt("--gate-baseline-train-dir") is None
                 or os.environ.get("STUB_NOT_WORSE") == "1")
        )
        rec.update(deployed=passed, gate={"sel_eff": se, "passed": passed})
        if not passed:
            rec["not_deployed"] = "below the target"
        (o / "gate.json").write_text(json.dumps({
            "n_evaluated": 10, "sel_eff": se, "n_miss": 1,
            "per_cell": {"Tiny|Tiny|TinyK|Bnone": {
                "sel_eff_new": se, "n_eval": 10, "paired_sel_eff_new": se,
                "origami_sel_eff_new": 0.5, "origami_n_eval": 10}}}))
    (o / "deploy_record.json").write_text(json.dumps(rec))
elif KEY in ("stage07", "stage08"):
    (o / "summary.json").write_text(json.dumps({"ok": True}))
    sys.exit(int(os.environ.get("STUB_RC_" + KEY, "0")))
"""


@pytest.fixture
def stubs(tmp_path):
    d = tmp_path / "stages"
    d.mkdir()
    for key, sdef in ro.STAGE_DEFS.items():
        (d / sdef.script).write_text(STUB.replace("__KEY__", repr(key)))
    return d


def make_config(tmp_path, **overrides):
    cfg = {
        "config_id": "orch_test",
        "arch": "gfx950",
        "hipblaslt": {
            "a_type": "bf16_r",
            "b_type": "bf16_r",
            "c_type": "bf16_r",
            "d_type": "bf16_r",
            "scale_type": "f32_r",
            "compute_type": "c_f32_r",
            "transA": "T",
            "transB": "N",
            "library_stem": STEM,
        },
        "runtime": {"devices": [0, 1], "rocm_libraries_root": str(tmp_path / "bench")},
        "seed": {"target_per_cell": 40, "shapes_per_cell": 40, "seed": 7},
        "bench": {"adaptive": {"enabled": True}},
        "train": {"min_cell_gemms": 20},
        "validate": {"min_cell_gemms": 10, "split_after_attempts": 0},
        "full": {"total_rounds": 3},
        "deploy": {
            "weight_dtype": "int4",
            "rocm_libraries_to_deploy": str(tmp_path / "deploy"),
            "apply_patch": True,
            "build_after_apply": True,
        },
        "stage07": {
            "parity_sample_trained": 0,
            "bench_yamls": [str(tmp_path / "w.yaml")],
        },
        "stage08": {"datasets": [{"name": "ds", "path": str(tmp_path / "ds.yaml")}]},
    }
    for dotted, value in overrides.items():
        cur = cfg
        parts = dotted.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    path = tmp_path / "orch_test.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert ro.validate_config(cfg)[0] == []
    return path


def run(stubs, cfg_path, run_dir, *extra):
    return ro.main(
        [
            "--config",
            str(cfg_path),
            "--run-dir",
            str(run_dir),
            "--skip-preflight",
            "--no-report",
            "--stages-dir",
            str(stubs),
            *extra,
        ]
    )


def calls(log):
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def arg(entry, name):
    a = entry["argv"]
    return a[a.index(name) + 1] if name in a else None


@pytest.fixture
def env(monkeypatch, tmp_path):
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("STUB_LOG", str(log))
    for k in (
        "STUB_FAIL",
        "STUB_RETRAIN",
        "STUB_FAIL_ONCE_DIR",
        "STUB_SEL_EFF",
        "STUB_GATE",
        "STUB_GATE_SE",
        "STUB_NOT_WORSE",
    ):
        monkeypatch.delenv(k, raising=False)
    return log


def test_full_run_carries_models_through_an_all_pass_round(
    tmp_path, stubs, env, monkeypatch
):
    monkeypatch.setenv("STUB_RETRAIN", json.dumps({"round_2": ""}))
    cfg = make_config(tmp_path)
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    log = calls(env)
    for r in range(3):
        assert (run_dir / f"round_{r}" / ro.ROUND_MARKER).is_file()
    trained = [arg(c, "--output-dir") for c in log if c["stage"] == "stage05"]
    assert [Path(p).parent.name for p in trained] == ["round_0", "round_1"]
    carried = run_dir / "round_2" / "stage05"
    assert (carried / "models.pt").read_text() == "models of round_1"
    assert (carried / "carried_forward.json").is_file()
    assert (carried / ro.STAGE_MARKER).is_file()
    deploys = [c for c in log if c["stage"] == "stage06"]
    assert len(deploys) == 1
    assert arg(deploys[0], "--train-dir") == str(carried)
    assert (
        "--apply-patch" in deploys[0]["argv"]
        and "--build-after-apply" in deploys[0]["argv"]
    )
    assert arg(deploys[0], "--library-stem") == STEM

    by_stage = {}
    for c in log:
        by_stage.setdefault(c["stage"], []).append(c)
    assert [arg(c, "--round-index") for c in by_stage["stage01"]] == ["0", "1", "2"]
    assert "stage02" not in by_stage
    assert arg(by_stage["stage03"][0], "--in-dir").endswith("round_0/stage01")
    assert arg(by_stage["stage04"][0], "--library-stem") == STEM
    for key in ("stage04b", "stage05"):
        for c in by_stage[key]:
            assert arg(c, "--weight-dtype") == "int4"
            assert arg(c, "--library-stem") == STEM
    snaps = sorted((run_dir / "configs").glob("*_full.yaml"))
    assert len(snaps) == 1
    for c in log:
        if "--config-yaml" in c["argv"]:
            assert arg(c, "--config-yaml") == str(snaps[0])
    s04b = by_stage["stage04b"][1]
    assert arg(s04b, "--models-dir").endswith("round_1/stage05")
    assert arg(s04b, "--train-min-cell-gemms") == "20"
    assert "--final-round" in s04b["argv"]
    assert arg(s04b, "--prior-models").endswith("round_0/stage05/models.pt")
    validate = [c["stage"] for c in log if c["stage"] in ("stage07", "stage08")]
    assert validate == ["stage07", "stage08"]
    s07 = by_stage["stage07"][0]
    assert arg(s07, "--deploy-record").endswith("round_2/stage06/deploy_record.json")
    timing_dirs = [
        s07["argv"][i + 1]
        for i, a in enumerate(s07["argv"])
        if a == "--timings-csv-dir"
    ]
    assert timing_dirs == [str(run_dir / f"round_{r}" / "stage04") for r in range(3)]
    assert arg(by_stage["stage08"][0], "--train-dir") == str(carried)
    assert arg(by_stage["stage08"][0], "--pick-measurement") == "warmup"
    assert "--origami-reference" not in by_stage["stage08"][0]["argv"]
    assert arg(s04b, "--gate-frac") == "0.2" and arg(s04b, "--seed") == "7"
    assert arg(s04b, "--pick-measurement") == "warmup"
    assert all(arg(c, "--pick-measurement") == "warmup" for c in by_stage["stage05"])
    assert "--gate-csv-dir" not in deploys[0]["argv"]
    breakdown = json.loads((run_dir / "held_out_sel_eff_breakdown.json").read_text())
    assert breakdown["shipped"]["round"] == "round_2"
    assert breakdown["source"] == "stage04b" and breakdown["n_cells"] == 2
    assert [m["round"] for m in breakdown["measurements"]] == ["round_2"]


def by_stage(log):
    out = {}
    for c in log:
        out.setdefault(c["stage"], []).append(c)
    return out


def test_each_round_ships_the_bundle_its_gate_measured(
    tmp_path, stubs, env, monkeypatch
):
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.97")
    cfg = make_config(tmp_path)
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    st = by_stage(calls(env))
    r1, r2 = run_dir / "round_1", run_dir / "round_2"
    trained = [arg(c, "--output-dir") for c in st["stage05"]]
    assert [Path(p).parent.name for p in trained] == ["round_0", "round_1", "round_2"]
    for c, rd in zip(st["stage05"][1:], (r1, r2)):
        dirs = [
            c["argv"][i + 1]
            for i, a in enumerate(c["argv"])
            if a == "--enriched-csv-dir"
        ]
        assert dirs[-1] == str(rd / "stage04b" / "train")
        assert str(run_dir / "round_0" / "stage04") in dirs
    d1, d2 = st["stage06"]
    assert arg(d1, "--train-dir") == str(r1 / "stage05")
    assert arg(d1, "--gate-csv-dir") == str(r1 / "stage04b" / "gate")
    assert arg(d1, "--gate-target") == "0.95" and "--gate-fallback" not in d1["argv"]
    assert arg(d1, "--library-dir").endswith("library/gfx950")
    assert arg(d2, "--gate-fallback") == "not-worse"
    assert arg(d2, "--gate-baseline-train-dir") == str(r1 / "stage05")
    for rd in (r1, r2):
        marker = json.loads((rd / ro.ROUND_MARKER).read_text())
        assert marker["deployed"] and marker["gate_sel_eff"] == pytest.approx(0.97)
        assert marker["held_out_sel_eff"] == pytest.approx(0.9)
    breakdown = json.loads((run_dir / "held_out_sel_eff_breakdown.json").read_text())
    assert breakdown["source"] == "deploy_gate"
    assert breakdown["shipped"]["round"] == "round_2"
    assert breakdown["held_out_sel_eff"] == pytest.approx(0.97)
    assert breakdown["per_cell"]["Tiny|Tiny|TinyK|Bnone"]["n_eval"] == 10
    assert breakdown["paired_origami"]["n"] == 10 and breakdown["final_bundle"] is None


def test_a_round_below_its_gate_target_does_not_deploy(
    tmp_path, stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.5")
    cfg = make_config(tmp_path)
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    st = by_stage(calls(env))
    assert [Path(arg(c, "--out-dir")).parent.name for c in st["stage06"]] == [
        "round_1",
        "round_2",
    ]
    m1 = json.loads((run_dir / "round_1" / ro.ROUND_MARKER).read_text())
    assert m1["deployed"] is False
    assert "--gate-baseline-train-dir" not in st["stage06"][1]["argv"]
    s07 = st["stage07"][0]
    assert arg(s07, "--deploy-record").endswith("round_2/stage06/deploy_record.json")
    out = capsys.readouterr()
    assert "round_1 not deployed: below the target" in out.err
    assert "origami" not in out.out.lower()


def test_the_final_round_keeps_a_worse_bundle_out(tmp_path, stubs, env, monkeypatch):
    cfg = make_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial", "--deploy") == 0
    monkeypatch.setenv("STUB_GATE", "1")
    monkeypatch.setenv("STUB_GATE_SE", "0.5")
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    st = by_stage(calls(env)[n:])
    (d1,) = st["stage06"]
    assert arg(d1, "--gate-baseline-train-dir") == str(run_dir / "round_0" / "stage05")
    rec = ro.latest_deploy_record(ro.complete_prefix(run_dir))
    assert rec == run_dir / "round_0" / "stage06" / "deploy_record.json"
    assert arg(st["stage07"][0], "--deploy-record") == str(rec)
    assert arg(st["stage08"][0], "--train-dir") == str(run_dir / "round_0" / "stage05")
    breakdown = json.loads((run_dir / "held_out_sel_eff_breakdown.json").read_text())
    assert breakdown["shipped"]["round"] == "round_0"
    assert breakdown["final_bundle"]["round"] == "round_1"
    assert breakdown["final_bundle"]["gate_sel_eff"] == pytest.approx(0.5)
    monkeypatch.setenv("STUB_NOT_WORSE", "1")
    assert run(stubs, cfg, run_dir, "--mode", "full", "--from-round", "1") == 0
    rec = ro.latest_deploy_record(ro.complete_prefix(run_dir))
    assert rec == run_dir / "round_1" / "stage06" / "deploy_record.json"


def test_the_reference_flag_reaches_the_stages_and_the_report(
    tmp_path, stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_GATE", "1")
    cfg = make_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full", "--origami-reference") == 0
    assert "[reference] Origami on the same 10 GEMMs" in capsys.readouterr().out
    st = by_stage(calls(env))
    for key in ("stage04b", "stage06", "stage07", "stage08"):
        assert all("--origami-reference" in c["argv"] for c in st[key]), key
    seen = []
    monkeypatch.setattr(ro.subprocess, "run", lambda cmd, **kw: seen.append(cmd))
    ro.emit_report(run_dir, origami_reference=True)
    ro.emit_report(run_dir)
    assert seen[0][-1] == "--origami-reference" and "--origami-reference" not in seen[1]


def test_shipped_numbers_follow_the_deployed_bundle(tmp_path):
    run_dir = tmp_path / "run"
    rows = [{"cell": "A", "sel_eff_new": 0.8, "n_eval": 20, "n_miss": 2}]
    for r in range(4):
        rd = run_dir / f"round_{r}"
        (rd / "stage05").mkdir(parents=True)
        (rd / "stage05" / "models.pt").write_text(f"bundle {min(r, 2)}")
        if r:
            (rd / "stage04b").mkdir()
            prior = run_dir / f"round_{r - 1}" / "stage05"
            (rd / "stage04b" / "decisions.json").write_text(
                json.dumps(
                    {
                        "models_dir": str(prior),
                        "aggregate": {
                            "global_sel_eff_new": 0.1 * r,
                            "n_gemms_evaluated": 10 * r,
                        },
                        "per_cell": rows,
                    }
                )
            )
    s6 = run_dir / "round_2" / "stage06"
    s6.mkdir()
    (s6 / "deploy_record.json").write_text(
        json.dumps({"train_dir": str(run_dir / "round_2" / "stage05")})
    )
    ro.mark_stage_complete(s6, None)
    got = ro.shipped_bundle_numbers(run_dir)
    assert got["shipped"]["round"] == "round_2" and got["source"] == "stage04b"
    assert [m["round"] for m in got["measurements"]] == ["round_3"]
    assert got["held_out_sel_eff"] == pytest.approx(0.3) and got["n_gemms"] == 30
    assert got["per_cell"]["A"]["n_miss"] == 2
    (s6 / "gate.json").write_text(
        json.dumps({"n_evaluated": 5, "sel_eff": 0.7, "per_cell": {"B": {}}})
    )
    got = ro.shipped_bundle_numbers(run_dir)
    assert got["source"] == "deploy_gate" and got["held_out_sel_eff"] == 0.7
    assert [m["source"] for m in got["measurements"]] == ["deploy_gate", "stage04b"]
    s6b = run_dir / "round_3" / "stage06"
    s6b.mkdir()
    (s6b / "deploy_record.json").write_text(
        json.dumps(
            {"deployed": False, "not_deployed": "below", "gate": {"sel_eff": 0.2}}
        )
    )
    ro.mark_stage_complete(s6b, None)
    got = ro.shipped_bundle_numbers(run_dir)
    assert got["shipped"]["round"] == "round_2"
    assert got["final_bundle"] == {
        "round": "round_3",
        "train_dir": None,
        "deployed": False,
        "reason": "below",
        "gate_sel_eff": 0.2,
    }
    assert ro.shipped_bundle_numbers(tmp_path / "empty")["shipped"] is None


def test_rerun_skips_complete_rounds(tmp_path, stubs, env):
    cfg = make_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    assert len(calls(env)) == n
    assert len(list((run_dir / "configs").glob("*_full*.source.yaml"))) == 2


def test_failed_stage_resumes_with_fresh_output(tmp_path, stubs, env, monkeypatch):
    monkeypatch.setenv("STUB_FAIL", "stage04")
    monkeypatch.setenv("STUB_FAIL_ONCE_DIR", str(tmp_path))
    cfg = make_config(tmp_path, **{"full.total_rounds": 1})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 1
    assert (run_dir / "round_0" / "stage04" / "partial.txt").is_file()
    assert not (run_dir / "round_0" / ro.ROUND_MARKER).exists()
    first = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    second = [c["stage"] for c in calls(env)[first:]]
    assert second[:2] == ["stage04", "stage05"]
    assert not (run_dir / "round_0" / "stage04" / "partial.txt").exists()
    stale = list((run_dir / "round_0" / "stale").glob("*/stage04/partial.txt"))
    assert len(stale) == 1
    assert list((run_dir / "round_0" / "stale").glob("*/stage04.log"))


def test_from_stage_needs_completed_earlier_stages(tmp_path, stubs, env):
    cfg = make_config(tmp_path)
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial", "--from-stage", "3") == 2
    assert calls(env) == []


def test_explicit_from_stage_reruns_from_there(tmp_path, stubs, env):
    cfg = make_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    n = len(calls(env))
    assert (
        run(
            stubs,
            cfg,
            run_dir,
            "--mode",
            "full",
            "--from-round",
            "1",
            "--from-stage",
            "4.5",
        )
        == 0
    )
    rerun = [c["stage"] for c in calls(env)[n:]]
    assert rerun[:3] == ["stage04b", "stage05", "stage06"]


def test_mode_preconditions(tmp_path, stubs, env):
    cfg = make_config(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert run(stubs, cfg, run_dir, "--mode", "active") == 2
    assert run(stubs, cfg, run_dir, "--mode", "validate") == 2
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 2
    assert run(stubs, cfg, run_dir, "--mode", "active") == 0
    assert (run_dir / "round_1" / ro.ROUND_MARKER).is_file()


@pytest.mark.parametrize(
    "fail,rc",
    [
        ({}, 0),
        ({"STUB_RC_stage07": "3"}, 3),
        ({"STUB_FAIL": "stage08:signal"}, 137),
        ({"STUB_RC_stage07": "1", "STUB_RC_stage08": "4"}, 4),
    ],
)
def test_validate_propagates_stage_exit_codes(
    tmp_path, stubs, env, monkeypatch, fail, rc
):
    cfg = make_config(tmp_path, **{"full.total_rounds": 1})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full", "--deploy") == 0
    for k, v in fail.items():
        monkeypatch.setenv(k, v)
    assert run(stubs, cfg, run_dir, "--mode", "validate") == rc


def test_validate_without_anything_configured(tmp_path, stubs, env):
    cfg = make_config(tmp_path, **{"full.total_rounds": 1})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full", "--deploy") == 0
    bare = make_config(tmp_path, **{"stage07.bench_yamls": [], "stage08.datasets": []})
    assert run(stubs, bare, run_dir, "--mode", "validate") == 2


def test_stage07_needs_a_deployed_model(tmp_path, stubs, env):
    cfg = make_config(tmp_path, **{"full.total_rounds": 1, "stage08.datasets": []})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    assert run(stubs, cfg, run_dir, "--mode", "validate") == 2
    assert not [c for c in calls(env) if c["stage"] == "stage07"]


def test_failed_deploy_fails_the_round(tmp_path, stubs, env, monkeypatch):
    monkeypatch.setenv("STUB_FAIL", "stage06:5")
    cfg = make_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 5
    assert (run_dir / "round_0" / ro.ROUND_MARKER).is_file()
    assert not (run_dir / "round_1" / ro.ROUND_MARKER).exists()


def test_each_invocation_gets_its_own_snapshot(tmp_path, stubs, env):
    cfg = make_config(tmp_path, **{"train.epochs": 11})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    cfg = make_config(tmp_path, **{"train.epochs": 22})
    assert run(stubs, cfg, run_dir, "--mode", "active") == 0
    snaps = {}
    for c in calls(env):
        if c["stage"] == "stage05":
            snap = Path(arg(c, "--config-yaml"))
            snaps[snap] = yaml.safe_load(snap.read_text())["train"]["epochs"]
            assert arg(c, "--epochs") == str(snaps[snap])
    assert sorted(snaps.values()) == [11, 22]
    assert len(list((run_dir / "configs").glob("*.source.yaml"))) == 2


def test_probe_runs_without_adaptive_timing(tmp_path, stubs, env):
    probe = {
        "first_line_iters": 2,
        "first_line_cold_iters": 2,
        "other_lines_iters": 1,
        "other_lines_cold_iters": 0,
        "duration_us": 100,
        "skip_ratio_mode": "sigmoid",
    }
    cfg = make_config(tmp_path, **{"bench.adaptive.enabled": False, "probe": probe})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    s03 = [c for c in calls(env) if c["stage"] == "stage03"][0]
    assert arg(s03, "--in-dir").endswith("round_0/stage02")


def test_untrained_cells_stop_the_round_and_it_resumes(
    tmp_path, stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_FAIL", "stage05@round_1:3")
    monkeypatch.setenv("STUB_FAIL_ONCE_DIR", str(tmp_path))
    cfg = make_config(tmp_path, **{"full.total_rounds": 2})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == ro.STAGE05_CELLS_NOT_TRAINED
    err = capsys.readouterr().err
    assert "stage05 did not train every requested cell" in err
    assert "X#M>1" in err and "round_1 stops before deployment" in err
    assert not (run_dir / "round_1" / ro.ROUND_MARKER).exists()
    assert not [c for c in calls(env) if c["stage"] in ("stage06", "stage07")]
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    rerun = [c["stage"] for c in calls(env)[n:]]
    assert rerun[:2] == ["stage05", "stage06"]
    assert (run_dir / "round_1" / ro.ROUND_MARKER).is_file()


def test_sigterm_stops_the_stage_and_the_round_resumes(tmp_path, stubs, env):
    cfg = make_config(tmp_path, **{"full.total_rounds": 1})
    run_dir = tmp_path / "run"
    pid_file = tmp_path / "stage.pid"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(Path(ro.__file__)),
            "--config",
            str(cfg),
            "--run-dir",
            str(run_dir),
            "--skip-preflight",
            "--no-report",
            "--stages-dir",
            str(stubs),
            "--mode",
            "full",
        ],
        env=dict(os.environ, STUB_SLEEP="stage03", STUB_PID_FILE=str(pid_file)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 120
        while not pid_file.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        stage_pid = int(pid_file.read_text())
        assert os.getpgid(stage_pid) == stage_pid != os.getpgid(0)
        proc.send_signal(signal.SIGTERM)
        _out, err = proc.communicate(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 128 + signal.SIGTERM
    assert "SIGTERM: stopped" in err
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            os.kill(stage_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the interrupted stage is still running")
    assert not ro.stage_complete(run_dir / "round_0" / "stage03")
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    assert [c["stage"] for c in calls(env)[n:]][:2] == ["stage03", "stage04"]


def test_a_run_dir_takes_one_orchestrator_at_a_time(tmp_path, stubs, env, capsys):
    cfg = make_config(tmp_path, **{"full.total_rounds": 1})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fd = ro.lock_run_dir(run_dir)
    assert fd is not None
    try:
        assert run(stubs, cfg, run_dir, "--mode", "full") == 2
        assert "another orchestrator is using" in capsys.readouterr().err
        assert calls(env) == []
    finally:
        os.close(fd)
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0


def test_stage08_datasets_use_the_bench_build_dir(tmp_path):
    other = tmp_path / "other"
    cfg = make_config(
        tmp_path,
        **{
            "runtime.build_dir": str(tmp_path / "custom_build"),
            "stage08.datasets": [
                {
                    "name": "same",
                    "path": "/d.yaml",
                    "rocm_libraries_used": str(tmp_path / "bench"),
                },
                {"name": "other", "path": "/e", "rocm_libraries_used": str(other)},
            ],
        },
    )
    p = _pipeline(tmp_path, cfg)
    args = p.args_stage08(tmp_path / "run", tmp_path / "run" / "round_0")
    specs = [args[i + 1] for i, a in enumerate(args) if a == "--dataset-build-dir"]
    default = other / "projects" / "hipblaslt" / "build" / "release"
    assert specs == [f"same={tmp_path / 'custom_build'}", f"other={default}"]


def _pipeline(tmp_path, cfg_path):
    cfg, _ = ro.load_config(cfg_path)
    return ro.Pipeline(
        cfg=cfg,
        snapshot=cfg_path,
        run_dir=tmp_path / "run",
        paths=ro.derive_paths(cfg),
        stages_dir=Path(ro.__file__).resolve().parent / "stages",
        timing=ro.Timing(),
        force_deploy=False,
    )


def defined_options(help_text):
    """Option strings argparse lists in `--help`: the invocation part of the
    lines indented by exactly two spaces (help text is indented further)."""
    opts = set()
    for line in help_text.splitlines():
        if not re.match(r"^  -", line):
            continue
        invocation = re.split(r"\s{2,}", line.strip(), maxsplit=1)[0]
        opts.update(re.findall(r"(?:^|, )(--?[A-Za-z0-9][\w-]*)", invocation))
    return opts


def test_defined_options_ignores_mentions_in_help_text():
    text = (
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --smart-k-max SMART_K_MAX\n"
        "                        cut at the knee in [--smart-k-min, --smart-k-max]\n"
        "  --bench-yaml BENCH_YAML, --yaml BENCH_YAML\n"
        "  --weight-dtype {bf16,fp32}  dtype; see --library-stem\n"
    )
    assert defined_options(text) == {
        "-h",
        "--help",
        "--smart-k-max",
        "--bench-yaml",
        "--yaml",
        "--weight-dtype",
    }


def test_serving_options_reach_stage05(tmp_path):
    rd = tmp_path / "run" / "round_0"
    step = ro.RoundStep(0, "initial", [], None, False)
    plain = _pipeline(tmp_path, make_config(tmp_path)).args_stage05(rd, step, None)
    assert "--no-non-temporal-rule" not in plain and "--cascade-k" not in plain
    cfg = make_config(
        tmp_path,
        **{"train.smart_k": 0, "train.non_temporal_rule": False, "train.cascade_k": 32},
    )
    args = _pipeline(tmp_path, cfg).args_stage05(rd, step, None)
    assert "--no-non-temporal-rule" in args
    assert args[args.index("--cascade-k") + 1] == "32"
    assert args[args.index("--smart-k") + 1] == "0"


def test_stage_cli_contract(tmp_path, pipeline_dir):
    """Every flag the orchestrator passes is one the real stage accepts."""
    cfg = make_config(
        tmp_path,
        **{
            "train.smart_k_max": 25,
            "train.validation_frac": 0.2,
            "train.non_temporal_rule": False,
            "train.cascade_k": 32,
            "stage08.tie_tolerance": 0.01,
            "stage08.bootstrap": 10,
            "stage08.datasets": [
                {"name": "ds", "path": "/d.yaml", "rocm_libraries_used": "/r"}
            ],
        },
    )
    p = _pipeline(tmp_path, cfg)
    p.origami_reference = True
    rd = tmp_path / "run" / "round_1"
    priors = [tmp_path / "run" / "round_0", tmp_path / "run" / "round_00"]
    (priors[0] / "stage04").mkdir(parents=True)
    initial = ro.RoundStep(0, "initial", [], None, False)
    active = ro.RoundStep(1, "active", priors, None, True)
    enhance = ro.RoundStep(1, "active", priors, None, True, targets=["a"])
    built = {
        "stage01": p.args_stage01(rd, initial)
        + p.args_stage01(rd, active)
        + p.args_stage01(rd, enhance),
        "stage02": p.args_stage02(rd),
        "stage03": p.args_stage03(rd),
        "stage04": p.args_stage04(rd),
        "stage04b": p.args_stage04b(rd, active, priors[0] / "stage05"),
        "stage05": p.args_stage05(rd, initial, None)
        + p.args_stage05(rd, active, "a,b")
        + p.args_stage05(rd, enhance, "a"),
        "stage06": p.args_stage06(rd)
        + p.args_stage06(rd, ro.GatePlan(rd / "g", 0.95, "not-worse", rd / "b")),
        "stage07": p.args_stage07(rd, rd / "r.json", [rd / "w.yaml"], priors[:1]),
        "stage08": p.args_stage08(rd, rd),
    }
    for key, args in built.items():
        script = pipeline_dir / "stages" / ro.STAGE_DEFS[key].script
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            env=dict(os.environ, COLUMNS="80"),
        )
        assert proc.returncode == 0, proc.stderr
        accepted = defined_options(proc.stdout)
        for flag in sorted({a for a in args if a.startswith("--")}):
            assert flag in accepted, f"{key} does not accept {flag}"


def _after(args, flag):
    return [args[i + 1] for i, a in enumerate(args) if a == flag]


def test_misses_are_rebenched_by_the_deciding_stages(tmp_path):
    cfg = make_config(
        tmp_path,
        **{
            "validate.rebench_runs": 4,
            "bench.adaptive.enabled": False,
            "probe": PLANNER_PROBE,
            "contexts.list": [{"cu_budget": 64, "cotenant": True}],
            "train.catalog": "v3",
        },
    )
    p = _pipeline(tmp_path, cfg)
    rd = tmp_path / "run" / "round_1"
    active = ro.RoundStep(1, "active", [tmp_path / "run" / "round_0"], None, False)
    gate = ro.GatePlan(rd / "g", 0.95)
    s04b = p.args_stage04b(rd, active, rd / "m")
    s06 = p.args_stage06(rd, gate)
    s08 = p.args_stage08(rd, rd)
    for args in (s04b, s06, s08):
        assert "--rebench-misses" in args and _after(args, "--rebench-runs") == ["4"]
        assert _after(args, "--bench-binary") == [str(p.paths.bench_binary)]
        assert _after(args, "--devices") == ["0,1"]
        assert _after(args, "--cotenant-binary") == [
            str(ro.context_cotenant(p.cfg, p.paths))
        ]
    for args in (s04b, s06):
        assert _after(args, "--rebench-lines") == [
            str(rd / "stage02" / "shapes_probed.yaml"),
            str(rd / "stage01" / "shapes.yaml"),
        ]
    assert "--rebench-lines" not in s08
    assert "--rebench-misses" not in p.args_stage06(rd)
    pipeline_dir = Path(ro.__file__).resolve().parent
    for key, args in (("stage04b", s04b), ("stage06", s06), ("stage08", s08)):
        script = pipeline_dir / "stages" / ro.STAGE_DEFS[key].script
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            env=dict(os.environ, COLUMNS="80"),
        )
        accepted = defined_options(proc.stdout)
        for flag in sorted({a for a in args if a.startswith("--")}):
            assert flag in accepted, f"{key} does not accept {flag}"
    off = _pipeline(
        tmp_path, make_config(tmp_path, **{"validate.rebench_misses": False})
    )
    for args in (
        off.args_stage04b(rd, active, rd / "m"),
        off.args_stage06(rd, gate),
        off.args_stage08(rd, rd),
    ):
        assert not any(a.startswith("--rebench") for a in args)
    assert ro.cfg_get({}, "validate.rebench_misses") is True
    assert ro.cfg_get({}, "validate.rebench_runs") == 3
    errors, _ = ro.validate_config({"validate": {"rebench_runs": 0}})
    assert any(e.startswith("validate.rebench_runs: must be >= 1") for e in errors)


def test_preflight_wants_a_module_of_this_engine(monkeypatch):
    import types

    from lib import evaluate, features

    class OldConfig:
        def __init__(self, **kwargs):
            if "attributes" in kwargs:
                raise TypeError("unexpected keyword argument 'attributes'")

    def broken(**kwargs):
        raise RuntimeError("no engine")

    def module(config):
        return types.SimpleNamespace(
            Config=config, feature_catalog_hash=features.feature_names_hash
        )

    for config, expected in (
        (lambda **kw: None, []),
        (OldConfig, ["older than this checkout's engine"]),
        (broken, ["cannot build kernel configs: RuntimeError('no engine')"]),
    ):
        tw = module(config)
        monkeypatch.setattr(evaluate, "tilewright_module", lambda m=tw: m)
        errors = ro._tilewright_errors()
        assert len(errors) == len(expected)
        assert all(want in got for want, got in zip(expected, errors))


def test_exit_code_mapping():
    assert ro.exit_code(0) == 0
    assert ro.exit_code(3) == 3
    assert ro.exit_code(-9) == 137


def test_move_aside_refuses_foreign_paths(tmp_path):
    owner = tmp_path / "round_0"
    owner.mkdir()
    other = tmp_path / "elsewhere"
    other.mkdir()
    with pytest.raises(RuntimeError):
        ro.move_aside(owner, other)


def test_v3_options_reach_stage05(tmp_path, pipeline_dir):
    rd = tmp_path / "run" / "round_0"
    step = ro.RoundStep(0, "initial", [], None, False)
    plain = _pipeline(tmp_path, make_config(tmp_path)).args_stage05(rd, step, None)
    assert "--catalog" not in plain
    cfg = make_config(
        tmp_path,
        **{
            "train.catalog": "v3",
            "train.cascade_k": 0,
            "train.dot2_rule": True,
            "train.non_temporal_rule": True,
            "train.launch_keys": True,
            "train.clip_features": False,
            "train.whitelist": "view-key",
            "train.min_cu_fraction": 0.25,
            "train.schedules": ["default", "dynamic"],
            "train.attribute_spec": [
                {"name": "param_a", "transform": "log2p1"},
                {"name": "param_b", "transform": "onehot", "values": [0, 2]},
            ],
        },
    )
    args = _pipeline(tmp_path, cfg).args_stage05(rd, step, None)
    assert args[args.index("--catalog") + 1] == "v3"
    assert args[args.index("--cascade-k") + 1] == "0"
    assert args[args.index("--whitelist") + 1] == "view-key"
    assert args[args.index("--schedules") + 1] == "default,dynamic"
    assert args[args.index("--min-cu-fraction") + 1] == "0.25"
    assert {"--dot2-rule", "--non-temporal-rule", "--no-clip-features"} <= set(args)
    assert "--launch-keys" in args
    assert "--small-batched-rule" not in args and "--no-non-temporal-rule" not in args
    v3_default = make_config(tmp_path, **{"train.catalog": "v3"})
    args = _pipeline(tmp_path, v3_default).args_stage05(rd, step, None)
    assert "--cascade-k" not in args and "--non-temporal-rule" not in args
    assert "--launch-keys" not in args
    proc = subprocess.run(
        [sys.executable, str(pipeline_dir / "stages" / "stage05_train.py"), "--help"],
        capture_output=True,
        text=True,
        env=dict(os.environ, COLUMNS="80"),
    )
    accepted = defined_options(proc.stdout)
    full = _pipeline(tmp_path, cfg).args_stage05(rd, step, None)
    assert {a for a in full if a.startswith("--")} <= accepted


STAGE05_OPTION_VALUES = {
    "validation_frac": 0.2,
    "batch_gemms": 8,
    "lr": 0.001,
    "lr_schedule": "step",
    "warmup_frac": 0.1,
    "lr_min_frac": 0.05,
    "eval_points": 20,
    "patience": 4,
    "student_temperature": "fixed",
    "qat_frac": 0.25,
    "qat_lr_frac": 0.2,
    "boundary_margin": 0.5,
    "boundary_weight": 0.3,
    "off_list_sel_eff": 0.8,
    "unmeasured_samples": 32,
    "unmeasured_margin": 0.05,
    "unmeasured_weight": 0.3,
}


def test_stage05_sees_the_plans_its_training_rounds_were_benched_from(
    tmp_path, pipeline_dir
):
    run = tmp_path / "run"
    r0, r1 = run / "round_0", run / "round_1"
    for path in (
        r0 / "stage01b" / "plan.json",
        r0 / "stage01b" / "seed" / "plan.json",
        r1 / "stage01b" / "plan.json",
        r1 / "stage01b" / "contexts" / "cu64" / "plan.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    step = ro.RoundStep(1, "active", [r0], None, False)
    plain = _pipeline(tmp_path, make_config(tmp_path)).args_stage05(r1, step, None)
    assert "--plan" not in plain and "--context-plan" not in plain
    cfg = make_config(
        tmp_path,
        **{
            "planner.enabled": True,
            "train.catalog": "v3",
            "contexts.list": [{"cu_budget": 64}],
        },
    )
    p = _pipeline(tmp_path, cfg)
    args = p.args_stage05(r1, step, None)
    plans = [args[i + 1] for i, a in enumerate(args) if a == "--plan"]
    assert plans == [
        str(r0 / "stage01b" / "plan.json"),
        str(r0 / "stage01b" / "seed" / "plan.json"),
        str(r1 / "stage01b" / "plan.json"),
    ]
    assert "--context-plan" not in args
    replan = make_config(
        tmp_path,
        **{
            "planner.enabled": True,
            "train.catalog": "v3",
            "contexts.list": [{"cu_budget": 64}],
            "contexts.replan": True,
        },
    )
    args = _pipeline(tmp_path, replan).args_stage05(r1, step, None)
    i = args.index("--context-plan")
    assert args[i + 1 : i + 4] == [
        "64",
        "default",
        str(r1 / "stage01b" / "contexts" / "cu64" / "plan.json"),
    ]
    proc = subprocess.run(
        [sys.executable, str(pipeline_dir / "stages" / "stage05_train.py"), "--help"],
        capture_output=True,
        text=True,
        env=dict(os.environ, COLUMNS="80"),
    )
    assert {a for a in args if a.startswith("--")} <= defined_options(proc.stdout)


def test_training_options_reach_stage05(tmp_path, pipeline_dir):
    assert set(STAGE05_OPTION_VALUES) == set(ro.STAGE05_OPTION_KEYS)
    rd = tmp_path / "run" / "round_0"
    step = ro.RoundStep(0, "initial", [], None, False)
    plain = _pipeline(tmp_path, make_config(tmp_path)).args_stage05(rd, step, None)
    flags = {"--" + k.replace("_", "-"): v for k, v in STAGE05_OPTION_VALUES.items()}
    assert not set(flags) & set(plain)
    cfg = make_config(
        tmp_path, **{f"train.{k}": v for k, v in STAGE05_OPTION_VALUES.items()}
    )
    assert not ro.validate_config(ro.load_config(cfg)[0])[0]
    args = _pipeline(tmp_path, cfg).args_stage05(rd, step, None)
    for flag, value in flags.items():
        assert args[args.index(flag) + 1] == str(value)
    proc = subprocess.run(
        [sys.executable, str(pipeline_dir / "stages" / "stage05_train.py"), "--help"],
        capture_output=True,
        text=True,
        env=dict(os.environ, COLUMNS="80"),
    )
    assert set(flags) <= defined_options(proc.stdout)
    for value in STAGE05_OPTION_VALUES.values():
        if isinstance(value, str):
            assert re.search(rf"\{{[^}}]*\b{value}\b[^}}]*\}}", proc.stdout), value


def test_training_option_validation():
    assert not any("train" in e for e in ro.validate_config({"train": {}})[0])
    for bad in (
        {"batch_gemms": -1},
        {"lr": 0},
        {"lr_schedule": "linear"},
        {"warmup_frac": 1.0},
        {"lr_min_frac": -0.1},
        {"eval_points": 0},
        {"patience": -1},
        {"student_temperature": "auto"},
        {"qat_frac": -0.5},
        {"qat_lr_frac": 0},
        {"batch_gemms": 1.5},
        {"off_list_sel_eff": 0},
        {"off_list_sel_eff": 1.5},
        {"unmeasured_samples": -1},
        {"unmeasured_margin": -0.1},
        {"unmeasured_weight": 0},
    ):
        errors = ro.validate_config({"train": bad})[0]
        assert any(f"train.{next(iter(bad))}" in e for e in errors), bad


def test_v3_config_validation():
    base = {"train": {"catalog": "v3"}}
    assert not any("train" in e for e in ro.validate_config(base)[0])
    for bad in (
        {"catalog": "v4"},
        {"attribute_spec": [{"transform": "identity"}]},
        {"attribute_spec": [{"name": "x", "transform": "cube"}]},
        {"schedules": ["fast"]},
        {"min_cu_fraction": 0.0},
        {"whitelist": "both"},
        {"launch_keys": "yes"},
    ):
        errors = ro.validate_config({"train": dict(base["train"], **bad)})[0]
        assert any("train" in e for e in errors), bad


def test_launch_keys_are_a_v3_serving_option(tmp_path):
    cfg = ro.load_config(make_config(tmp_path))[0]
    assert ro.cfg_get(cfg, "train.launch_keys") is False
    cfg.setdefault("train", {})["launch_keys"] = True
    errors = ro.validate_config(cfg)[0]
    assert any("train.launch_keys" in e and "v3" in e for e in errors)
    cfg["train"]["catalog"] = "v3"
    assert ro.validate_config(cfg)[0] == []


def test_preflight_checks_the_v3_catalog(monkeypatch):
    import types

    from lib import evaluate, features

    tw = types.SimpleNamespace(
        Config=lambda **kw: None,
        feature_catalog_hash=features.feature_names_hash,
        feature_catalog_hashes=lambda: [features.feature_names_hash()],
    )
    monkeypatch.setattr(evaluate, "tilewright_module", lambda: tw)
    assert ro._tilewright_errors("v2") == []
    (err,) = ro._tilewright_errors("v3")
    assert "does not compute the v3 catalog" in err
    tw.feature_catalog_hashes = lambda: [
        features.feature_names_hash(),
        features.v3_catalog_hash(),
    ]
    assert ro._tilewright_errors("v3") == []


# ── planner ──────────────────────────────────────────────────────────────────

PLANNER_PROBE = {
    "first_line_iters": 2,
    "first_line_cold_iters": 2,
    "other_lines_iters": 1,
    "other_lines_cold_iters": 0,
    "duration_us": 100,
    "skip_ratio_mode": "sigmoid",
}


def test_planner_is_off_by_default(tmp_path, stubs, env):
    run_dir = tmp_path / "run"
    assert run(stubs, make_config(tmp_path), run_dir, "--mode", "initial") == 0
    stages = [c["stage"] for c in calls(env)]
    assert "stage01b" not in stages
    s04 = [c for c in calls(env) if c["stage"] == "stage04"][0]
    assert "--plan" not in s04["argv"] and s04["argv"].count("--log-dir") == 1


def test_planner_defaults_match_the_stage():
    import stage01b_plan_benchmarks as s01b

    for key, default in s01b.DEFAULTS.items():
        assert ro.cfg_get({}, f"planner.{key}") == default, key


def test_planner_config_is_validated(tmp_path):
    cfg_path = make_config(tmp_path, **{"planner.enabled": True})
    cfg, _ = ro.load_config(cfg_path)
    cfg["planner"]["origami_top_k"] = 4
    cfg["planner"]["max_list"] = 4097
    cfg["planner"]["mode"] = "heuristic"
    errors, _ = ro.validate_config(cfg)
    assert "planner.origami_top_k: unknown key" in errors
    assert any(e.startswith("planner.max_list: must be <= 4096") for e in errors)
    assert any(e.startswith("planner.mode: 'heuristic'") for e in errors)


def test_planned_round_wiring(tmp_path, stubs, env):
    cfg = make_config(
        tmp_path,
        **{
            "bench.adaptive.enabled": False,
            "probe": PLANNER_PROBE,
            "planner.enabled": True,
            "planner.mode": "planned",
            "planner.bank_library_stems": [STEM.replace("ID75a0", "CU256_ID75a0")],
        },
    )
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    log = calls(env)
    rd0 = run_dir / "round_0"
    stages0 = [c["stage"] for c in log if "round_0" in (arg(c, "--out-dir") or "")]
    assert stages0[:5] == ["stage01", "stage01b", "stage02", "stage03", "stage04"]
    s01b = [c for c in log if c["stage"] == "stage01b"]
    assert arg(s01b[0], "--in-dir") == str(rd0 / "stage01")
    seed = json.loads(arg(s01b[0], "--seed-commands"))
    assert "--model-file" not in s01b[0]["argv"]
    assert arg({"argv": seed["stage02"]}, "--in-dir") == str(rd0 / "stage01b" / "seed")
    assert arg({"argv": seed["stage02"]}, "--plan").endswith("seed/plan.json")
    assert arg({"argv": seed["stage03"]}, "--in-dir").endswith("seed/stage02")
    assert arg({"argv": seed["stage04"]}, "--log-dir").endswith("seed/stage03/logs")
    assert "stage05" not in seed
    s02 = [c for c in log if c["stage"] == "stage02"][0]
    assert arg(s02, "--in-dir") == str(rd0 / "stage01b")
    assert arg(s02, "--plan") == str(rd0 / "stage01b" / "plan.json")
    s04 = [c for c in log if c["stage"] == "stage04"][0]
    assert arg(s04, "--plan") == str(rd0 / "stage01b" / "plan.json")
    assert arg(s04, "--extra-library-stem").startswith("TensileLibrary_")
    # Later rounds rank with the latest trained model instead of seeds.
    assert arg(s01b[1], "--model-file") == str(rd0 / "stage05" / "models.pt")
    assert "--seed-commands" not in s01b[1]["argv"]


def test_planner_from_stage_and_stage05_seed_ranker(tmp_path, stubs, env):
    cfg = make_config(
        tmp_path,
        **{
            "planner.enabled": True,
            "planner.mode": "planned",
            "planner.ranker": "stage05",
            "planner.seed_per_cell": 4,
        },
    )
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    s01b = [c for c in calls(env) if c["stage"] == "stage01b"][0]
    seed = json.loads(arg(s01b, "--seed-commands"))
    assert seed["stage02"] is None
    assert arg({"argv": seed["stage03"]}, "--in-dir").endswith("stage01b/seed")
    s05 = {"argv": seed["stage05"]}
    assert arg(s05, "--enriched-csv-dir").endswith("stage01b/seed/stage04")
    assert arg(s05, "--output-dir").endswith("stage01b/seed/stage05")
    assert arg(s05, "--min-cell-gemms") == "4"
    s03 = [c for c in calls(env) if c["stage"] == "stage03"][-1]
    assert arg(s03, "--in-dir").endswith("round_0/stage01b")
    env.write_text("")
    assert run(stubs, cfg, run_dir, "--mode", "initial", "--from-stage", "1.5") == 0
    assert [c["stage"] for c in calls(env)][:2] == ["stage01b", "stage03"]


def test_explicit_model_file_skips_the_seed_bootstrap(tmp_path, stubs, env):
    model = tmp_path / "given.pt"
    cfg = make_config(
        tmp_path,
        **{"planner.enabled": True, "planner.model_file": str(model)},
    )
    assert run(stubs, cfg, tmp_path / "run", "--mode", "initial") == 0
    s01b = [c for c in calls(env) if c["stage"] == "stage01b"][0]
    assert arg(s01b, "--model-file") == str(model)


def test_planner_preflight_needs_bench_support(tmp_path):
    cfg_path = make_config(tmp_path, **{"planner.enabled": True})
    cfg, _ = ro.load_config(cfg_path)
    bench = tmp_path / "clients" / "hipblaslt-bench"
    bench.parent.mkdir(parents=True)
    assert "cannot read" in ro._planner_bench_errors(bench)[0]
    (bench.parent / "hipblaslt_common.yaml").write_text("  - solution_index_list: x\n")
    assert "skip_screen_prepass" in ro._planner_bench_errors(bench)[0]
    (bench.parent / "hipblaslt_common.yaml").write_text(
        "  - solution_index_list: c_int32*512\n  - skip_screen_prepass: c_bool\n"
    )
    assert "fewer than 4096" in ro._planner_bench_errors(bench)[0]
    (bench.parent / "hipblaslt_common.yaml").write_text(
        "  - solution_index_list: c_int32*4096\n  - skip_screen_prepass: c_bool\n"
    )
    assert ro._planner_bench_errors(bench) == []


def _bank_library(library_dir):
    """STEM (Prediction table [1, 2]) plus a `_CU256_` file whose Equality
    row the 0x75a0 / 256-CU device accepts and an `_ID75a8` one it does
    not."""
    import msgpack

    from bench_fakes import solution

    def write(stem, indices, hw, library):
        sols = [dict(solution(i, i), hardwarePredicate=hw) for i in indices]
        (library_dir / f"{stem}.dat").write_bytes(
            msgpack.packb({"solutions": sols, "library": library})
        )

    def equality(indices):
        table = [{"key": [i], "index": i} for i in indices]
        return {
            "type": "Problem",
            "rows": [
                {
                    "predicate": {"type": "EqualityMatching"},
                    "library": {"type": "Matching", "table": table},
                }
            ],
        }

    def gpu(*terms):
        return {"type": "AMDGPU", "value": {"type": "And", "value": list(terms)}}

    chip = {"type": "PciChipId", "value": 0x75A0}
    library_dir.mkdir(parents=True)
    prediction = {"type": "Prediction", "table": [1, 2]}
    write(STEM, [1, 2], gpu(chip), prediction)
    cu256 = STEM.replace("ID75a0", "CU256_ID75a0")
    write(cu256, [5, 4], gpu(chip, {"type": "CUCount", "value": 256}), equality([5, 4]))
    a8 = STEM.replace("ID75a0", "ID75a8")
    write(a8, [7], gpu({"type": "PciChipId", "value": 0x75A8}), equality([7]))
    return cu256


def test_planner_bank_auto_wiring(tmp_path, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "-1")
    hardware = {"n_cu": 256, "lds_bytes": 65536, "l2_bytes": 4 << 20}
    cfg_path = make_config(
        tmp_path,
        **{
            "planner.enabled": True,
            "planner.bank": "auto",
            "bench.pool_only_library": True,
            "hardware": dict(hardware, pci_chip_id=0x75A0),
        },
    )
    p = _pipeline(tmp_path, cfg_path)
    cu256 = _bank_library(p.paths.library_dir)
    pool_only = p._pool_only_args()
    stems = [
        pool_only[i + 1] for i, a in enumerate(pool_only) if a == "--pool-only-library"
    ]
    assert stems == [STEM, cu256]
    s04 = p.args_stage04(tmp_path / "run" / "round_0")
    assert [s04[i + 1] for i, a in enumerate(s04) if a == "--extra-library-stem"] == [
        cu256
    ]
    assert ro._bank_errors(p.cfg, p.paths) == []
    no_chip = dict(p.cfg, hardware=hardware)
    (err,) = ro._bank_errors(no_chip, p.paths)
    assert err.startswith("planner.bank: auto: no PCI chip id")
    explicit = dict(p.cfg, planner=dict(p.cfg["planner"], bank_library_stems=[cu256]))
    assert ro._bank_errors(explicit, p.paths) == []
    errors, _ = ro.validate_config(dict(p.cfg, planner={"bank": "all"}))
    assert any(e.startswith("planner.bank: 'all'") for e in errors)


def test_planner_stage_cli_contract(tmp_path, pipeline_dir):
    """Every flag the planned round passes is one the real stage accepts."""
    cfg = make_config(
        tmp_path,
        **{
            "bench.adaptive.enabled": False,
            "probe": PLANNER_PROBE,
            "planner.enabled": True,
            "planner.ranker": "stage05",
            "planner.bank_library_stems": [STEM],
        },
    )
    p = _pipeline(tmp_path, cfg)
    rd = tmp_path / "run" / "round_0"
    step = ro.RoundStep(0, "initial", [], None, False)
    seed = p.seed_commands(rd, step)
    built = {
        "stage01b": p.args_stage01b(rd, step),
        "stage02": p.args_stage02(rd) + seed["stage02"][1:],
        "stage03": seed["stage03"][1:],
        "stage04": p.args_stage04(rd) + seed["stage04"][1:],
        "stage05": seed["stage05"][1:],
    }
    for key, args in built.items():
        script = pipeline_dir / "stages" / ro.STAGE_DEFS[key].script
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            env=dict(os.environ, COLUMNS="80"),
        )
        assert proc.returncode == 0, proc.stderr
        accepted = defined_options(proc.stdout)
        for flag in sorted({a for a in args if a.startswith("--")}):
            assert flag in accepted, f"{key} does not accept {flag}"


def test_rebench_runs_between_bench_and_enrich(tmp_path, stubs, env, pipeline_dir):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert run(stubs, make_config(plain), plain / "run", "--mode", "initial") == 0
    assert "stage03b" not in [c["stage"] for c in calls(env)]
    s04 = [c for c in calls(env) if c["stage"] == "stage04"][0]
    assert arg(s04, "--log-dir") == str(plain / "run" / "round_0" / "stage03" / "logs")

    env.write_text("")
    cfg = make_config(tmp_path, **{"bench.rebench": {"enabled": True, "runs": 5}})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    log = calls(env)
    stages = [c["stage"] for c in log]
    assert stages[stages.index("stage03") : stages.index("stage03") + 3] == [
        "stage03",
        "stage03b",
        "stage04",
    ]
    rd = run_dir / "round_0"
    s03 = [c for c in log if c["stage"] == "stage03"][0]
    s03b = [c for c in log if c["stage"] == "stage03b"][0]
    assert arg(s03b, "--in-dir") == arg(s03, "--in-dir")
    assert arg(s03b, "--stage03-dir") == str(rd / "stage03")
    assert arg(s03b, "--out-dir") == str(rd / "stage03b")
    assert arg(s03b, "--devices") == arg(s03, "--devices")
    s04 = [c for c in log if c["stage"] == "stage04"][0]
    assert arg(s04, "--log-dir") == str(rd / "stage03b" / "logs")
    env.write_text("")
    assert run(stubs, cfg, run_dir, "--mode", "initial", "--from-stage", "3.2") == 0
    assert [c["stage"] for c in calls(env)][:2] == ["stage03b", "stage04"]

    proc = subprocess.run(
        [
            sys.executable,
            str(pipeline_dir / "stages" / ro.STAGE_DEFS["stage03b"].script),
            "--help",
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ, COLUMNS="80"),
    )
    assert proc.returncode == 0, proc.stderr
    built = _pipeline(tmp_path, cfg).args_stage03b(rd)
    assert {a for a in built if a.startswith("--")} <= defined_options(proc.stdout)


def test_rebench_config_is_validated():
    for bad in ({"tolerance": 0}, {"runs": 0}, {"iters_factor": 0.5}, {"window": 2}):
        errors = ro.validate_config({"bench": {"rebench": {"enabled": True, **bad}}})[0]
        assert any(e.startswith("bench.rebench.") for e in errors), bad


def test_rebench_defaults_match_the_stage():
    import stage03b_rebench_near_ties as s03b

    assert set(ro.SCHEMA.children["bench"].children["rebench"].children) == {
        "enabled",
        *s03b.CONFIG_KEYS,
    }


def test_planner_keys_without_enabled_warn(tmp_path):
    cfg, _ = ro.load_config(make_config(tmp_path))
    cfg["planner"] = {"mode": "planned"}
    errors, warnings = ro.validate_config(cfg)
    assert errors == []
    assert "planner: mode ignored (planner.enabled is false)" in warnings


TINY = "Tiny|Tiny|TinyK|Bnone"
LARGE = "Large|Large|LargeK|Bnone"


def initial_run(stubs, tmp_path, **overrides):
    cfg = make_config(tmp_path, **{"full.total_rounds": 1, **overrides})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    return cfg, run_dir


def write_cells(run_dir, rnd, rows):
    """round_<rnd>/stage05/cells.json with (label, validation sel_eff,
    training GEMMs) entries."""
    cells = [
        {
            "label": label,
            "n_gemms": n,
            "deployed": {"sel_eff": min(1.0, se + 0.02), "n_eval": n},
            "deployed_val": {"sel_eff": se, "n_eval": n // 4},
        }
        for label, se, n in rows
    ]
    path = run_dir / f"round_{rnd}" / "stage05" / "cells.json"
    path.write_text(json.dumps({"cells": cells}))


def test_enhance_round_retrains_only_the_chosen_cells(
    tmp_path, stubs, env, monkeypatch
):
    cfg, run_dir = initial_run(stubs, tmp_path)
    monkeypatch.setenv("STUB_RETRAIN", json.dumps({"round_1": f"{TINY},{LARGE}"}))
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "enhance", "--cells", TINY) == 0
    by_stage = {}
    for c in calls(env)[n:]:
        by_stage.setdefault(c["stage"], []).append(c)
    assert arg(by_stage["stage01"][0], "--only-cells") == TINY
    assert arg(by_stage["stage01"][0], "--mode") == "active"
    assert "--final-round" in by_stage["stage04b"][0]["argv"]
    assert arg(by_stage["stage05"][0], "--only-cells") == TINY
    assert len(by_stage["stage06"]) == 1
    rd = run_dir / "round_1"
    marker = json.loads((rd / ro.ROUND_MARKER).read_text())
    assert marker["mode"] == "enhance" and marker["deployed"]
    record = json.loads((rd / ro.ENHANCE_RECORD).read_text())
    assert record["targets"] == [TINY]
    assert record["retrain"] == [TINY]
    assert record["deferred"] == [LARGE]
    assert run(stubs, cfg, run_dir, "--mode", "active") == 0
    assert (run_dir / "round_2" / ro.ROUND_MARKER).is_file()


def test_enhance_dry_run_writes_nothing(tmp_path, stubs, env, capsys):
    cfg, run_dir = initial_run(stubs, tmp_path)
    write_cells(run_dir, 0, [(TINY, 0.80, 40), (LARGE, 0.97, 40)])
    before = sorted(str(p) for p in run_dir.rglob("*"))
    n = len(calls(env))
    capsys.readouterr()
    rc = run(
        stubs,
        cfg,
        run_dir,
        "--mode",
        "enhance",
        "--weakest",
        "3",
        "--shapes-per-cell",
        "60",
        "--enhance-rounds",
        "2",
        "--dry-run",
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert sorted(str(p) for p in run_dir.rglob("*")) == before
    assert len(calls(env)) == n
    assert out.count("--only-cells") == 4
    assert f"--only-cells '{TINY}'" in out
    assert LARGE not in out
    assert "--shapes-per-cell 60" in out
    assert "round_1" in out and "round_2" in out and "(final round)" in out
    assert out.count("[stage06] ") == 2 and out.count("stage06_") == 1


def test_enhance_selects_weak_cells_from_training_evidence(tmp_path, stubs, env):
    _, run_dir = initial_run(stubs, tmp_path)
    mid = "Mid|Mid|MidK|Bnone"
    write_cells(run_dir, 0, [(TINY, 0.80, 40), (LARGE, 0.90, 400), (mid, 0.97, 40)])
    rounds = ro.complete_prefix(run_dir)

    def pick(**kw):
        opts = dict(cells=[], weakest=None, below=None, rank_by="sel_eff")
        opts.update(kw)
        return ro.select_enhance_cells(run_dir, rounds, target=0.95, **opts)

    assert pick(weakest=1)["targets"] == [TINY]
    assert pick(weakest=1, rank_by="gain")["targets"] == [LARGE]
    assert pick(weakest=5)["targets"] == [TINY, LARGE]
    assert pick(below=0.85)["targets"] == [TINY]
    assert pick(below=0.99)["targets"] == [TINY, LARGE, mid]
    assert pick(cells=[mid], weakest=1)["targets"] == [mid, TINY]
    sel = pick(weakest=1)["selection"]
    assert sel["picked"][0]["source"] == "validation"
    assert sel["evidence_rounds"] == ["round_0"]
    with pytest.raises(ro.ConfigError):
        pick(below=0.5)
    with pytest.raises(ro.ConfigError):
        pick(cells=["Nope|Tiny|TinyK|Bnone"])

    s8 = run_dir / "validate" / "stage08"
    s8.mkdir(parents=True, exist_ok=True)
    (s8 / "summary.json").write_text(
        json.dumps({"per_cell": {mid: {"sel_eff": 0.1}, LARGE: {"sel_eff": 0.1}}})
    )
    (s8 / "per_gemm.csv").write_text(f"cell,sel_eff\n{mid},0.1\n")
    incomplete = run_dir / "round_1" / "stage04b"
    incomplete.mkdir(parents=True)
    (incomplete / "decisions.json").write_text(
        json.dumps({"per_cell": [{"cell": mid, "sel_eff_new": 0.1, "n_eval": 50}]})
    )
    assert pick(weakest=1)["targets"] == [TINY]
    assert pick(weakest=1, rank_by="gain")["targets"] == [LARGE]


def test_enhance_resumes_with_the_recorded_selection(tmp_path, stubs, env, monkeypatch):
    cfg, run_dir = initial_run(stubs, tmp_path)
    monkeypatch.setenv("STUB_FAIL", "stage05@round_1")
    monkeypatch.setenv("STUB_FAIL_ONCE_DIR", str(tmp_path))
    assert run(stubs, cfg, run_dir, "--mode", "enhance", "--cells", TINY) == 1
    assert run(stubs, cfg, run_dir, "--mode", "active") == 2
    n = len(calls(env))
    assert run(stubs, cfg, run_dir, "--mode", "enhance", "--cells", LARGE) == 0
    rerun = calls(env)[n:]
    assert [c["stage"] for c in rerun][:2] == ["stage05", "stage06"]
    assert arg(rerun[0], "--only-cells") == TINY


def test_enhance_splits_hand_their_children_to_stage05(tmp_path):
    from lib import subcells as sc

    p = _pipeline(tmp_path, make_config(tmp_path, **{"train.min_cell_gemms": 20}))
    run_dir = tmp_path / "run"
    rd = run_dir / "round_1"
    (rd / "stage04b").mkdir(parents=True)
    rule = sc.build_split_rule(LARGE, "M", 4096)
    sc.write_splits_json(rd / "stage04b" / "splits.json", 1, [rule])
    rows = [
        {"cell": LARGE, "n_training_gemms": 80},
        {"cell": rule.lo_label, "n_training_gemms": 40},
        {"cell": rule.hi_label, "n_training_gemms": 40},
        {"cell": TINY, "n_training_gemms": 5},
    ]
    (rd / "stage04b" / "decisions.json").write_text(json.dumps({"per_cell": rows}))
    step = ro.RoundStep(
        1, "active", [run_dir / "round_0"], None, True, targets=[LARGE, TINY]
    )
    listed = f"{rule.lo_label},{rule.hi_label},Mid|Mid|MidK|Bnone"
    got = p._enhance_retrain(rd, step, listed)
    assert got.split(",") == [rule.lo_label, rule.hi_label]
    record = json.loads((rd / ro.ENHANCE_RECORD).read_text())
    assert record["too_few_gemms"] == [TINY]
    assert record["deferred"] == ["Mid|Mid|MidK|Bnone"]


def test_enhance_preconditions(tmp_path, stubs, env, capsys):
    cfg = make_config(tmp_path, **{"full.total_rounds": 1})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for argv in (
        ["--mode", "active", "--cells", TINY],
        ["--mode", "full", "--dry-run"],
        ["--mode", "enhance"],
        ["--mode", "enhance", "--weakest", "0"],
        ["--mode", "enhance", "--below", "95"],
        ["--mode", "enhance", "--cells", TINY, "--from-round", "1"],
    ):
        with pytest.raises(SystemExit) as e:
            run(stubs, cfg, run_dir, *argv)
        assert e.value.code == 2, argv
    assert run(stubs, cfg, run_dir, "--mode", "enhance", "--cells", TINY) == 2
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    bad = "Huge|Tiny|TinyK|Bnone"
    assert run(stubs, cfg, run_dir, "--mode", "enhance", "--cells", bad) == 2
    assert not (run_dir / "round_1").exists()
    assert run(stubs, cfg, run_dir, "--mode", "enhance", "--below", "0.5") == 2
    assert not (run_dir / "round_1").exists()
    assert [c["stage"] for c in calls(env)].count("stage01") == 1
