# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Platform records and the session check, on fake sysfs trees
(tests/gpu_fakes.py)."""

import json
import os
import stat

import pytest

import gpu_fakes as gf
from lib import fingerprint as fp
from lib import kfd


@pytest.fixture
def visible(monkeypatch):
    for var in kfd.HIP_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(kfd.ROCR_VAR, "0,1,2")


def _sysfs(tmp_path, monkeypatch, **kw):
    root = gf.write_sysfs(tmp_path / "sys", **kw)
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(root))
    return root


def test_a_session_records_every_gpu(tmp_path, monkeypatch, visible):
    _sysfs(tmp_path, monkeypatch)
    doc = fp.start_session([0, 1], tmp_path)
    assert json.loads((tmp_path / fp.PLATFORM_FILE).read_text()) == doc
    assert doc["problems"] == [] and doc["amd_smi_check"] is None
    assert set(doc["host"]) == {"amdgpu_driver", "kernel", "rocm"}
    g0, g1 = doc["gpus"]
    assert (g0["device"], g0["gpu"], g0["kfd_node"], g0["gpu_id"]) == (0, 0, 1, 1001)
    assert (g1["bdf"], g1["unique_id"]) == ("0000:0a:00.0", "GPU-0000000000002222")
    assert (g1["perf_level"], g1["power_cap_w"], g1["power_cap_default_w"]) == (
        "auto",
        100.0,
        100.0,
    )
    assert g1["vbios"] == "fake-vbios" and g1["amd_smi_gpu"] is None
    assert (g1["vram_used_bytes"], g1["unattributed"]) == (gf.IDLE_CARD_BYTES, 0)
    assert doc["unattributed_allowance_bytes"] == kfd.UNATTRIBUTED_ALLOWANCE_BYTES
    assert "cpu" not in doc


def test_a_session_records_the_cus_partition_mode_and_xccs(
    tmp_path, monkeypatch, visible
):
    _sysfs(
        tmp_path, monkeypatch, xcc=[(3, "SPX"), (3, "SPX"), None], cus=[32, 32, None]
    )
    doc = fp.start_session([0, 1, 2], tmp_path, environment={"X": "1", "Y": None})
    got = [(g["n_cu"], g["num_xcc"], g["compute_partition"]) for g in doc["gpus"]]
    assert got == [(32, 3, "SPX"), (32, 3, "SPX"), (None, None, None)]
    assert doc["problems"] == []
    assert doc["environment"] == {"X": "1", "Y": None}
    assert "environment" not in fp.start_session([0], tmp_path)
    warm = {"enabled": True, "line": "- {M: 512}", "kernels": []}
    assert fp.start_session([0], tmp_path, warmup_line=warm)["warmup_line"] == warm
    assert "warmup_line" not in fp.start_session([0], tmp_path)


def test_a_session_records_vram_no_listed_process_holds(tmp_path, monkeypatch, visible):
    root = _sysfs(tmp_path, monkeypatch)
    gf.set_memory(root, 4242, {1002: 1 << 30})
    gf.set_card_used(root, gf.GPUS[1], 6 << 30)
    doc = fp.start_session([0, 1], tmp_path)
    g0, g1 = doc["gpus"]
    assert (g1["vram_used_bytes"], g1["vram_attributed_bytes"]) == (6 << 30, 1 << 30)
    assert (g0["unattributed"], g1["unattributed"]) == (0, 5 << 30)
    assert doc["problems"] == []
    (root / "bus" / "pci" / "devices" / gf.GPUS[0].bdf / "mem_info_vram_used").unlink()
    g0 = fp.start_session([0], tmp_path)["gpus"][0]
    assert (g0["vram_used_bytes"], g0["unattributed"]) == (None, None)


@pytest.mark.parametrize(
    "kw, label, values",
    [
        ({"perf_level": ["auto", "manual", "auto"]}, "performance level", "manual"),
        ({"power_cap_w": [100.0, 80.0, 100.0]}, "power cap", "80.0"),
        ({"cus": [32, 16, 32]}, "CU count", "16"),
        (
            {"xcc": [(3, "SPX"), (3, "DPX"), (3, "SPX")]},
            "compute partition mode",
            "DPX",
        ),
        ({"xcc": [(3, "SPX"), (1, "SPX"), (3, "SPX")]}, "XCC count", "1"),
    ],
)
def test_gpus_that_disagree_refuse_the_session(
    tmp_path, monkeypatch, visible, kw, label, values
):
    _sysfs(tmp_path, monkeypatch, **kw)
    with pytest.raises(fp.PlatformError) as e:
        fp.start_session([0, 1, 2], tmp_path)
    msg = str(e.value)
    assert f"the session's GPUs disagree on the {label}" in msg
    assert f"GPU 1 (0000:0a:00.0) {values}" in msg and "GPU 0 (0000:0b:00.0)" in msg
    doc = json.loads((tmp_path / fp.PLATFORM_FILE).read_text())
    assert doc["problems"] == [msg]


def test_unknown_values_are_null_and_never_fail(tmp_path, monkeypatch, visible):
    _sysfs(tmp_path, monkeypatch, perf_level=[None, "auto", "auto"])
    doc = fp.start_session([0, 1], tmp_path)
    assert [g["perf_level"] for g in doc["gpus"]] == [None, "auto"]
    monkeypatch.setenv(kfd.ROCR_VAR, "-1")
    hidden = fp.start_session([0], tmp_path)["gpus"][0]
    assert hidden["gpu"] is None and hidden["perf_level"] is None


def test_amd_smi_that_disagrees_with_the_topology_refuses(
    tmp_path, monkeypatch, visible
):
    _sysfs(tmp_path, monkeypatch)
    exe = tmp_path / "fake-amd-smi"
    rows = [{"gpu": 0, "bdf": "0000:0a:00.0", "kfd_id": 1002, "node_id": 2}]
    exe.write_text(f"#!/bin/sh\necho '{json.dumps(rows)}'\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(kfd.AMD_SMI_VAR, str(exe))
    doc = fp.start_session([1], tmp_path)
    assert doc["amd_smi_check"] == "agrees" and doc["gpus"][0]["amd_smi_gpu"] == 0
    rows[0]["node_id"] = 3
    exe.write_text(f"#!/bin/sh\necho '{json.dumps(rows)}'\n")
    with pytest.raises(fp.PlatformError, match="GPU numbering: amd-smi lists GPU 0"):
        fp.start_session([1], tmp_path)


def test_cpu_settings_name_pinned_cpus_and_their_siblings(tmp_path, monkeypatch):
    gf.write_cpus(tmp_path / "sys", n=4)
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(tmp_path / "sys"))
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1})
    cpu = fp.cpu_settings()
    assert (cpu["affinity"], cpu["pinned"]) == ("0-1", True)
    assert cpu["governors"] == [{"governor": "performance", "cpus": "0-1"}]
    assert (cpu["boost"], cpu["smt_active"]) == (True, True)
    assert cpu["smt_siblings_outside_affinity"] == "2-3"
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})
    unpinned = fp.cpu_settings()
    assert unpinned["pinned"] is False
    assert unpinned["smt_siblings_outside_affinity"] is None


def test_the_cpu_window_measures_the_busy_share(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(kfd, "PROC", proc)
    (proc / "loadavg").write_text("1.00 2.00 3.00 1/100 42\n")
    stat_line = "cpu{} {} 0 0 {} 0 0 0 0 0 0\n"
    (proc / "stat").write_text(
        "cpu 0 0 0 0 0 0 0 0 0 0\n"
        + stat_line.format(2, 20, 80)
        + stat_line.format(3, 0, 100)
    )
    window = fp.CpuWindow([2, 3])
    (proc / "stat").write_text(
        stat_line.format(2, 70, 130) + stat_line.format(3, 10, 190)
    )
    (proc / "loadavg").write_text("4.00 2.00 3.00 1/100 42\n")
    out = window.stop()
    assert out["cpus"] == "2-3"
    assert out["busy_max"] == pytest.approx(0.5)
    assert out["busy_mean"] == pytest.approx(0.3)
    assert (out["loadavg_start"], out["loadavg_end"]) == (
        [1.0, 2.0, 3.0],
        [4.0, 2.0, 3.0],
    )
    none = fp.CpuWindow(None).stop()
    assert (none["cpus"], none["busy_max"]) == (None, None)


def test_cpu_lists_round_trip():
    assert fp.parse_cpulist("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert fp.format_cpulist([11, 0, 1, 2, 3, 8, 10]) == "0-3,8,10-11"
    assert fp.parse_cpulist("x") is None and fp.parse_cpulist(None) is None


def test_a_stage_adds_to_its_sessions_record(tmp_path, monkeypatch, visible):
    _sysfs(tmp_path, monkeypatch)
    assert not fp.add_to_record(tmp_path, {"host_load": {}})
    fp.start_session([0], tmp_path)
    assert fp.add_to_record(tmp_path, {"host_load": {"samples": 3}})
    doc = json.loads((tmp_path / fp.PLATFORM_FILE).read_text())
    assert doc["host_load"] == {"samples": 3} and doc["gpus"][0]["gpu"] == 0
