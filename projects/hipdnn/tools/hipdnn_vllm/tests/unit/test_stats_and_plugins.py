# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import json
import multiprocessing
import os
import threading

import pytest

from hipdnn_vllm import _bootstrap, stats
from hipdnn_vllm.config import load_config


@pytest.fixture(autouse=True)
def _clean_stats(monkeypatch):
    stats.reset()
    monkeypatch.setattr(stats, "_config", None)
    yield
    stats.reset()


def test_report_lists_fallback_reasons(tmp_path):
    stats.record("attention.mixed", stats.ROUTE_FALLBACK, "UNSUPPORTED_FEATURE:paged", 3)
    stats.record("attention.uniform_decode", stats.ROUTE_FALLBACK, "UNSUPPORTED_FEATURE:paged")
    stats.record("attention.mixed", stats.ROUTE_FALLBACK, "UNSUPPORTED_FEATURE:softcap")
    path = stats.write_report(str(tmp_path))
    assert os.path.basename(path) == f"report-{os.getpid()}.json"
    data = json.loads(open(path).read())
    assert data["fallback_reasons"] == {
        "UNSUPPORTED_FEATURE:paged": 4,
        "UNSUPPORTED_FEATURE:softcap": 1,
    }
    rows = {(r["op"], r["reason"]): r for r in data["counters"]}
    assert rows[("attention.mixed", "UNSUPPORTED_FEATURE:paged")]["count"] == 3
    assert all(r["phase"] == "eager" for r in data["counters"])


def test_no_report_without_data_or_dir(tmp_path):
    assert stats.write_report(str(tmp_path)) is None
    stats.record("x", stats.ROUTE_FALLBACK, "DISABLED")
    assert stats.write_report(None) is None
    assert os.listdir(tmp_path) == []


def test_summary_line():
    stats.record("attention.mixed", stats.ROUTE_FALLBACK, "UNSUPPORTED_FEATURE:paged", 2)
    line = stats._summary_line(stats.snapshot())
    assert "attention.mixed:fallback=2" in line


def test_concurrent_report_writes(tmp_path):
    """The summary thread and exit hooks may write the report at the same
    time; the writes are serialized and leave one valid JSON file."""
    stats.record("attention.mixed", stats.ROUTE_FALLBACK, "UNSUPPORTED_FEATURE:paged", 2)
    errors = []

    def writer():
        try:
            for _ in range(50):
                stats.write_report(str(tmp_path))
        except Exception as exc:  # noqa: BLE001  (reported by the assert below)
            errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert os.listdir(tmp_path) == [f"report-{os.getpid()}.json"]
    data = json.loads((tmp_path / f"report-{os.getpid()}.json").read_text())
    assert data["fallback_reasons"] == {"UNSUPPORTED_FEATURE:paged": 2}


def _child_records_and_exits(report_dir):
    stats.configure(load_config({"HIPDNN_VLLM_REPORT_DIR": report_dir, "HIPDNN_VLLM_REPORT_INTERVAL": "0"}))
    stats.record("attention.mixed", stats.ROUTE_FALLBACK, "UNSUPPORTED_FEATURE:paged", 2)
    raise SystemExit(0)


@pytest.mark.parametrize("method", [m for m in ("fork", "spawn") if m in multiprocessing.get_all_start_methods()])
def test_multiprocessing_child_writes_report_at_exit(tmp_path, method):
    """vLLM's EngineCore and workers are multiprocessing children, which skip
    atexit; the report must still be written when they exit."""
    proc = multiprocessing.get_context(method).Process(target=_child_records_and_exits, args=(str(tmp_path),))
    proc.start()
    proc.join(60)
    assert proc.exitcode == 0
    reports = [p for p in os.listdir(tmp_path) if p == f"report-{proc.pid}.json"]
    assert len(reports) == 1, os.listdir(tmp_path)
    data = json.loads((tmp_path / reports[0]).read_text())
    assert data["fallback_reasons"] == {"UNSUPPORTED_FEATURE:paged": 2}


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()
    return path


@pytest.fixture
def dirs(tmp_path):
    bundled = tmp_path / "hipdnn_vllm" / "providers" / "gfx94X"
    b_blt = _touch(str(bundled / "libhipblaslt_plugin.so"))
    b_hkp = _touch(str(bundled / "libhip_kernel_provider.so"))
    b_mio = _touch(str(bundled / "libmiopen_plugin.so"))
    os.makedirs(bundled / "hip_kernel_provider", exist_ok=True)  # data dir, not a plugin
    return {"bundled": str(bundled), "b_blt": b_blt, "b_hkp": b_hkp, "b_mio": b_mio}


# Plans list provider files in sorted file-name order ("libhip_" < "libhipb").
def test_auto_is_absolute_bundled_only_without_miopen(dirs):
    plan = _bootstrap.plugin_plan(load_config({}), dirs["bundled"])
    assert plan.mode == "absolute"
    assert plan.paths == (dirs["b_hkp"], dirs["b_blt"])


def test_conv_keeps_bundled_miopen(dirs):
    plan = _bootstrap.plugin_plan(load_config({"HIPDNN_VLLM_OPS": "rms_norm,conv"}), dirs["bundled"])
    assert plan.mode == "absolute"
    assert plan.paths == (dirs["b_hkp"], dirs["b_blt"], dirs["b_mio"])


def test_no_bundle_loads_nothing_absolutely(dirs):
    plan = _bootstrap.plugin_plan(load_config({}), None)
    assert plan == _bootstrap.PluginPlan("absolute", (), plan.why)


def test_forced_additive_passes_bundled(dirs):
    plan = _bootstrap.plugin_plan(load_config({"HIPDNN_VLLM_PLUGIN_MODE": "additive"}), dirs["bundled"])
    assert plan.mode == "additive" and plan.paths == (dirs["b_hkp"], dirs["b_blt"])


def test_absolute_with_provider_dirs_replaces_everything(dirs):
    cfg = load_config({"HIPDNN_VLLM_PLUGIN_MODE": "absolute", "HIPDNN_VLLM_PROVIDER_DIRS": "/x/y"})
    plan = _bootstrap.plugin_plan(cfg, dirs["bundled"])
    assert plan.mode == "absolute" and plan.paths == ("/x/y",)


def test_auto_appends_provider_dirs(dirs):
    cfg = load_config({"HIPDNN_VLLM_PROVIDER_DIRS": "/extra"})
    plan = _bootstrap.plugin_plan(cfg, dirs["bundled"])
    assert plan.mode == "absolute" and plan.paths[-1] == "/extra"


def test_bundled_dirs_fall_back_to_prefix(monkeypatch, tmp_path):
    monkeypatch.setattr(_bootstrap, "_PKG_DIR", str(tmp_path / "pkg"))
    monkeypatch.setenv("HIPDNN_VLLM_HIPDNN_PREFIX", str(tmp_path / "stage"))
    assert _bootstrap.bundled_backend_dir() == str(tmp_path / "stage" / "lib")
    assert _bootstrap.bundled_provider_dir() == str(tmp_path / "stage" / "lib" / "hipdnn_plugins" / "engines")


def test_wheel_layout_wins_over_prefix(monkeypatch, tmp_path):
    pkg = tmp_path / "pkg"
    os.makedirs(pkg / "backend" / "gfx94X")
    os.makedirs(pkg / "providers" / "gfx94X")
    monkeypatch.setattr(_bootstrap, "_PKG_DIR", str(pkg))
    monkeypatch.setenv("HIPDNN_VLLM_HIPDNN_PREFIX", str(tmp_path / "stage"))
    assert _bootstrap.bundled_backend_dir() == str(pkg / "backend" / "gfx94X")
    assert _bootstrap.bundled_provider_dir() == str(pkg / "providers" / "gfx94X")
