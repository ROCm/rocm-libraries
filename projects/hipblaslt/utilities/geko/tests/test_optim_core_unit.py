# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from geko.optim import optim as ocore
from geko.schemas import GemmConfig, GemmType


def _mk_gemm_config() -> GemmConfig:
    gt = GemmType.from_hipblaslt("N", "N", "f16_r", "f16_r", "f16_r", "f32_r")
    return GemmConfig(gt, [[16, 16, 1, 16]])


def test_configure_builds_config_and_calls_generator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    out_dir = tmp_path / "out"
    gc = _mk_gemm_config()

    seen = {}

    def _defaults(cfg):
        cfg["_defaults_applied"] = True

    def _run(cfg, hip_path, target, write_shell_scripts=False):
        seen["cfg"] = cfg
        seen["hip"] = hip_path
        seen["target"] = target
        seen["write_shell_scripts"] = write_shell_scripts

    monkeypatch.setattr(ocore, "apply_input_config_defaults", _defaults)
    monkeypatch.setattr(ocore.cg, "run", _run)

    cfg = ocore.configure(hip, gc, out_dir, arch="gfx950", backend="ductile")
    assert cfg["ARCH"] == "gfx950"
    assert cfg["backend"] == "ductile"
    assert cfg["search_space"] is None
    assert cfg["_defaults_applied"] is True
    assert seen["write_shell_scripts"] is False
    assert seen["target"] == out_dir
    assert len(cfg["GemmProblems"]) == 1


def test_run_validates_inputs_and_empty_configs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="hipBLASLt path not found"):
        ocore.run(tmp_path / "missing", tmp_path)

    hip = tmp_path / "hip"
    hip.mkdir()
    with pytest.raises(ValueError, match="n_slots must be >= 1"):
        ocore.run(hip, tmp_path, n_slots=0)

    monkeypatch.setattr(ocore, "list_optimization_configs", lambda _p: [])
    called = {"build": False}
    monkeypatch.setattr(ocore, "build_tensilelite_client", lambda *_a, **_k: called.__setitem__("build", True))
    ocore.run(hip, tmp_path)
    assert called["build"] is False


def test_run_worker_flow_updates_config_and_timing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    (hip / "tensilelite/Tensile/bin").mkdir(parents=True)
    tuning = tmp_path / "tuning"
    tuning.mkdir()
    cfg = tuning / "job_1.yaml"
    cfg.write_text("Device: 0\n", encoding="utf-8")

    monkeypatch.setattr(ocore, "list_optimization_configs", lambda _p: [str(cfg)])
    monkeypatch.setattr(ocore, "build_tensilelite_client", lambda *_a, **_k: None)
    monkeypatch.setattr(ocore, "parse_devices", lambda d: list(d))
    state = {"n": 0}

    def _get_state(_p):
        state["n"] += 1
        # setup path sees failed; teardown re-check sees completed.
        return "failed" if state["n"] == 1 else "completed"

    monkeypatch.setattr(ocore, "get_build_state", _get_state)
    monkeypatch.setattr(ocore, "clean_failed_build", lambda _p: None)
    monkeypatch.setattr(ocore, "wait_process_or_stop", lambda proc, *_a, **_k: setattr(proc, "returncode", 0))
    monkeypatch.setattr(ocore, "estimate_workload", lambda _p: 1.0)
    monkeypatch.setattr(ocore, "_log_work", lambda *_a, **_k: None)

    class _Proc:
        def __init__(self):
            self.returncode = 0

    monkeypatch.setattr(ocore.subprocess, "Popen", lambda *a, **k: _Proc())

    class _Runner:
        def __init__(self, items, worker_impl, devices, n_slots, estimate_workload_fn, job_logger_fn):
            self.items = items
            self.worker_impl = worker_impl
            self.device = devices[0]

        def __call__(self, _workdir):
            for item in self.items:
                w = self.worker_impl(item, self.device, 0, type("Stop", (), {"is_set": lambda self: False})(), None)
                w.setup()
                assert w.run() is True
                w.teardown()
            return self.items

    monkeypatch.setattr(ocore, "Runner", _Runner)
    ocore.run(hip, tuning, devices=[0], n_slots=1, retry=True)

    content = cfg.read_text(encoding="utf-8")
    assert "Device: 0" in content
    assert (tuning / "timing.log").is_file()


def test_analyze_no_kept_results_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "transA": "N",
                "transB": "N",
                "batch_count": 1,
                "m": 16,
                "n": 16,
                "k": 16,
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "c_f32_r",
                "kernel_reference": "k",
                "kernel_tuned": "k",
                "us_reference": 10.0,
                "us_tuned": 9.0,
                "ratio": 1.11,
                "lib": "x.yaml",
            }
        ]
    )
    monkeypatch.setattr(ocore, "parse_devices", lambda d: list(d))
    monkeypatch.setattr(ocore.bench, "compare", lambda *_a, **_k: df)
    monkeypatch.setattr(ocore._metrics, "enrich", lambda d, summary_csv=None: d.assign(uplift_pct=11.0))
    monkeypatch.setattr(
        ocore._metrics,
        "summarize",
        lambda **_k: {"uplift_kept": {}, "counts": {"n_kept": 0}, "e2e": {}},
    )
    monkeypatch.setattr(ocore._metrics, "write_metrics_json", lambda *_a, **_k: None)

    out, full_df = ocore.analyze(tmp_path / "hip", tmp_path / "libs", tmp_path / "o", verify=False)
    assert out is None
    assert full_df is not None
    assert (tmp_path / "o/raw_results.csv").is_file()


def test_analyze_kept_results_writes_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "transA": "N",
                "transB": "N",
                "batch_count": 1,
                "m": 16,
                "n": 16,
                "k": 16,
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "c_f32_r",
                "kernel_reference": "k0",
                "kernel_tuned": "k1",
                "us_reference": 10.0,
                "us_tuned": 8.0,
                "ratio": 1.25,
                "lib": "x.yaml",
                "error_tuned": 0.0,
            }
        ]
    )

    monkeypatch.setattr(ocore, "parse_devices", lambda d: list(d))
    monkeypatch.setattr(ocore.bench, "compare", lambda *_a, **_k: df)
    monkeypatch.setattr(ocore._metrics, "enrich", lambda d, summary_csv=None: d.assign(uplift_pct=25.0))
    monkeypatch.setattr(
        ocore._metrics,
        "summarize",
        lambda **_k: {
            "uplift_kept": {"mean_uplift_pct": 25.0, "geomean_uplift_pct": 24.0},
            "counts": {"n_kept": 1},
            "e2e": {"e2e_uplift_pct": 12.0},
        },
    )
    monkeypatch.setattr(ocore._metrics, "write_metrics_json", lambda *_a, **_k: None)

    dashboard_called = {"n": 0}

    class _Dash:
        def to_csv(self, *_a, **_k):
            dashboard_called["n"] += 1

    monkeypatch.setattr(ocore.bench.utils, "as_dashboard_format", lambda _d: _Dash())

    out, full_df = ocore.analyze(tmp_path / "hip", tmp_path / "libs", tmp_path / "o", verify=True, device=2)
    assert out is not None
    assert len(out) == 1
    assert full_df is not None
    assert (tmp_path / "o/final_results.csv").is_file()
    assert dashboard_called["n"] == 1


def test_log_work_collects_running_jobs(tmp_path: Path) -> None:
    tuning = tmp_path / "tuning"
    build = tuning / "build_a"
    build.mkdir(parents=True)
    (build / ".running").write_text("device=1\nslot=2\n", encoding="utf-8")
    (build / "a-tensilelite.log").write_text("  1 | 10 | x\n  3 | 30 | y\n", encoding="utf-8")

    ocore._log_work(tuning, "running_jobs")
    out = (tuning / "running_jobs.log").read_text(encoding="utf-8")
    assert "device=1 | slot=2 | n_gen=03" in out


def test_run_worker_resumable_no_retry_skips_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    tuning = tmp_path / "tuning"
    tuning.mkdir()
    cfg = tuning / "job_2.yaml"
    cfg.write_text("Device: 0\n", encoding="utf-8")

    monkeypatch.setattr(ocore, "list_optimization_configs", lambda _p: [str(cfg)])
    monkeypatch.setattr(ocore, "build_tensilelite_client", lambda *_a, **_k: None)
    monkeypatch.setattr(ocore, "parse_devices", lambda d: list(d))
    monkeypatch.setattr(ocore, "get_build_state", lambda _p: "resumable")
    monkeypatch.setattr(ocore, "estimate_workload", lambda _p: 1.0)
    monkeypatch.setattr(ocore, "_log_work", lambda *_a, **_k: None)

    called = {"clean": 0, "popen": 0}
    monkeypatch.setattr(ocore, "clean_failed_build", lambda _p: called.__setitem__("clean", called["clean"] + 1))
    monkeypatch.setattr(ocore.subprocess, "Popen", lambda *a, **k: called.__setitem__("popen", called["popen"] + 1))

    class _Runner:
        def __init__(self, items, worker_impl, devices, n_slots, estimate_workload_fn, job_logger_fn):
            self.items = items
            self.worker_impl = worker_impl
            self.device = devices[0]

        def __call__(self, _workdir):
            for item in self.items:
                w = self.worker_impl(item, self.device, 0, type("Stop", (), {"is_set": lambda self: False})(), None)
                w.setup()
                assert w.run() is False
                w.teardown()
            return self.items

    monkeypatch.setattr(ocore, "Runner", _Runner)
    ocore.run(hip, tuning, devices=[0], n_slots=1, retry=False)
    assert called["clean"] == 0
    assert called["popen"] == 0


def test_run_worker_running_state_and_nonzero_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hip = tmp_path / "hip"
    (hip / "tensilelite/Tensile/bin").mkdir(parents=True)
    tuning = tmp_path / "tuning"
    tuning.mkdir()
    cfg = tuning / "job_3.yaml"
    cfg.write_text("Device: 0\n", encoding="utf-8")
    build_dir = tuning / "build_job_3"
    build_dir.mkdir()
    (build_dir / ".running").write_text("device=0\nslot=0\n", encoding="utf-8")

    monkeypatch.setattr(ocore, "list_optimization_configs", lambda _p: [str(cfg)])
    monkeypatch.setattr(ocore, "build_tensilelite_client", lambda *_a, **_k: None)
    monkeypatch.setattr(ocore, "parse_devices", lambda d: list(d))

    states = iter(["running", "failed", "failed"])
    monkeypatch.setattr(ocore, "get_build_state", lambda _p: next(states))
    monkeypatch.setattr(ocore, "clean_failed_build", lambda _p: None)
    monkeypatch.setattr(ocore, "wait_process_or_stop", lambda proc, *_a, **_k: setattr(proc, "returncode", 1))
    monkeypatch.setattr(ocore, "estimate_workload", lambda _p: 1.0)
    monkeypatch.setattr(ocore, "_log_work", lambda *_a, **_k: None)

    class _Proc:
        def __init__(self):
            self.returncode = 1

    monkeypatch.setattr(ocore.subprocess, "Popen", lambda *a, **k: _Proc())

    class _Runner:
        def __init__(self, items, worker_impl, devices, n_slots, estimate_workload_fn, job_logger_fn):
            self.items = items
            self.worker_impl = worker_impl
            self.device = devices[0]

        def __call__(self, _workdir):
            for item in self.items:
                w = self.worker_impl(item, self.device, 0, type("Stop", (), {"is_set": lambda self: False})(), None)
                w.setup()
                assert w.run() is False
                w.teardown()
            return self.items

    monkeypatch.setattr(ocore, "Runner", _Runner)
    ocore.run(hip, tuning, devices=[0], n_slots=1, retry=True)
    # .running should always be removed by teardown.
    assert not (build_dir / ".running").exists()
