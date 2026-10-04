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
    (o / "decisions.json").write_text(json.dumps({"per_cell": [
        {"cell": "Tiny|Tiny|TinyK|Bnone", "sel_eff_new": se, "n_eval": 10},
        {"cell": "Large|Large|LargeK|Bnone", "sel_eff_new": se, "n_eval": 30}]}))
    (o / "retrain_cells.txt").write_text(retrain)
    (o / "splits.json").write_text(json.dumps({"splits": {}}))
elif KEY == "stage05":
    (o / "models.pt").write_text("models of " + rnd)
    (o / "cells.json").write_text(json.dumps({"cells": []}))
elif KEY == "stage06":
    (o / "deploy_record.json").write_text(json.dumps({"train_dir": opt("--train-dir")}))
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
    for k in ("STUB_FAIL", "STUB_RETRAIN", "STUB_FAIL_ONCE_DIR", "STUB_SEL_EFF"):
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
    breakdown = json.loads((run_dir / "held_out_sel_eff_breakdown.json").read_text())
    assert breakdown["n_cells"] == 2


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


def test_aggregate_drops_split_parents(tmp_path):
    run_dir = tmp_path / "run"
    r1 = run_dir / "round_1" / "stage04b"
    r2 = run_dir / "round_2" / "stage04b"
    for d in (r1, r2):
        d.mkdir(parents=True)
    parent = "Large|Large|LargeK|Bnone"
    (r1 / "decisions.json").write_text(
        json.dumps(
            {
                "per_cell": [
                    {"cell": parent, "sel_eff_new": 0.5, "n_eval": 40},
                    {"cell": "Tiny|Tiny|TinyK|Bnone", "sel_eff_new": 0.9, "n_eval": 10},
                ]
            }
        )
    )
    (r1 / "splits.json").write_text(json.dumps({"splits": {parent: {}}}))
    (r2 / "decisions.json").write_text(
        json.dumps(
            {
                "per_cell": [
                    {"cell": parent + "#M<=4096", "sel_eff_new": 0.8, "n_eval": 20},
                    {"cell": "Tiny|Tiny|TinyK|Bnone", "sel_eff_new": 0.95, "n_eval": 0},
                    {"sel_eff_new": 0.1, "n_eval": 5},
                ]
            }
        )
    )
    agg = ro.aggregate_held_out(run_dir)
    assert set(agg["per_cell"]) == {"Tiny|Tiny|TinyK|Bnone", parent + "#M<=4096"}
    assert agg["per_cell"]["Tiny|Tiny|TinyK|Bnone"]["round"] == "round_1"
    assert agg["held_out_sel_eff"] == pytest.approx((0.9**10 * 0.8**20) ** (1 / 30))


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


def test_stage_cli_contract(tmp_path, pipeline_dir):
    """Every flag the orchestrator passes is one the real stage accepts."""
    cfg = make_config(
        tmp_path,
        **{
            "train.smart_k_max": 25,
            "train.validation_frac": 0.2,
            "stage08.tie_tolerance": 0.01,
            "stage08.bootstrap": 10,
            "stage08.datasets": [
                {"name": "ds", "path": "/d.yaml", "rocm_libraries_used": "/r"}
            ],
        },
    )
    p = _pipeline(tmp_path, cfg)
    rd = tmp_path / "run" / "round_1"
    priors = [tmp_path / "run" / "round_0", tmp_path / "run" / "round_00"]
    initial = ro.RoundStep(0, "initial", [], None, False)
    active = ro.RoundStep(1, "active", priors, None, True)
    built = {
        "stage01": p.args_stage01(rd, initial) + p.args_stage01(rd, active),
        "stage02": p.args_stage02(rd),
        "stage03": p.args_stage03(rd),
        "stage04": p.args_stage04(rd),
        "stage04b": p.args_stage04b(rd, active, priors[0] / "stage05"),
        "stage05": p.args_stage05(rd, initial, None)
        + p.args_stage05(rd, active, "a,b"),
        "stage06": p.args_stage06(rd),
        "stage07": p.args_stage07(rd, rd / "r.json", [rd / "w.yaml"]),
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
