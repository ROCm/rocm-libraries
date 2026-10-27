# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Preflight's check that the KFD node and HIP report the bench device alike
(lib/device_values.py), on fake KFD trees and a fake probe. Every value is
made up."""

import json
import stat
import subprocess
import sys

import pytest

import gpu_fakes as gf
from lib import device_values as dv
from lib import kfd

GPU = gf.GPUS[0]
KFD = {"n_cu": 12, "lds_bytes": 48 * 1024, "l2_bytes": 3 * 1024 * 1024, "wave_size": 32}


@pytest.fixture
def sysfs(tmp_path, monkeypatch):
    for var in (*kfd.HIP_VARS, kfd.ROCR_VAR):
        monkeypatch.delenv(var, raising=False)
    root = gf.write_sysfs(
        tmp_path / "sys", gf.GPUS[:2], isa=[("gfx950", 0)] * 2, cus=[12, 12]
    )
    for g in gf.GPUS[:2]:
        node = gf.kfd_root(root) / "topology" / "nodes" / str(g.node)
        with (node / "properties").open("a") as f:
            f.write("lds_size_in_kb 48\nwave_front_size 32\n")
        for i, (level, size) in enumerate(((1, 16), (2, 3072), (2, 3072))):
            c = node / "caches" / str(i)
            c.mkdir(parents=True)
            (c / "properties").write_text(f"level {level}\nsize {size}\n")
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(root))
    return root


def probe(tmp_path, monkeypatch, **values):
    """A fake probe that prints HIP's values (KFD's unless given), and the
    GPU its environment shows."""
    doc = {
        "arch": "gfx950:sramecc+:xnack-",
        "multi_processor_count": KFD["n_cu"],
        "physical_multi_processor_count": KFD["n_cu"],
        "shared_mem_per_block": KFD["lds_bytes"],
        "l2_cache_size": KFD["l2_bytes"],
        "warp_size": KFD["wave_size"],
    }
    doc.update(values)
    exe = tmp_path / "probe"
    exe.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        f"doc = {doc!r}\n"
        "doc['seen'] = os.environ.get('ROCR_VISIBLE_DEVICES')\n"
        "if doc.get('fail'):\n    sys.exit(1)\n"
        "print('banner')\nprint(json.dumps(doc))\n"
    )
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(dv.PROBE_VAR, str(exe))
    return exe


def _cfg(**hardware):
    cfg = {"arch": "gfx950", "runtime": {"devices": [1, 0]}}
    if hardware:
        cfg["hardware"] = hardware
    return cfg


def test_the_kfd_node_gives_its_values(sysfs):
    (g,) = [x for x in kfd.gpus() if x.index == 1]
    assert dv.kfd_values(g) == dict(KFD, arch="gfx950")


def test_agreeing_values_pass_and_the_probe_sees_one_gpu(sysfs, tmp_path, monkeypatch):
    probe(tmp_path, monkeypatch)
    c = dv.check(_cfg(), tmp_path / "cache")
    assert (c.device, c.gpu, c.errors, c.warnings) == (1, 1, [], [])
    assert c.hip["n_cu"] == KFD["n_cu"]
    assert {k: c.used[k] for k in dv.KEYS} == KFD
    assert c.record()["kfd"]["lds_bytes"] == KFD["lds_bytes"]


@pytest.mark.parametrize(
    "hip, label",
    [
        ({"physical_multi_processor_count": 6}, "CU count"),
        ({"shared_mem_per_block": 1024}, "LDS bytes"),
        ({"l2_cache_size": 1024}, "L2 bytes"),
    ],
)
def test_hip_values_other_than_the_runs_are_refused(
    sysfs, tmp_path, monkeypatch, hip, label
):
    probe(tmp_path, monkeypatch, **hip)
    (err,) = dv.check(_cfg(), tmp_path / "cache").errors
    assert err.startswith("device 1 (GPU 1): the run uses ")
    assert f"as the {label}" in err
    assert "from the KFD node" in err and "hardware." in err


def test_a_pinned_value_hip_agrees_with_is_a_note(sysfs, tmp_path, monkeypatch):
    probe(tmp_path, monkeypatch, physical_multi_processor_count=6)
    c = dv.check(_cfg(n_cu=6), tmp_path / "cache")
    assert c.errors == [] and len(c.warnings) == 1
    assert "the KFD node gives 12 as the CU count, HIP 6" in c.warnings[0]
    pinned = dv.check(_cfg(n_cu=5), tmp_path / "cache")
    (err,) = pinned.errors
    assert "the run uses 5 as the CU count (hardware.n_cu pins it)" in err


def test_the_wave_size_and_the_device_must_agree(sysfs, tmp_path, monkeypatch):
    probe(tmp_path, monkeypatch, warp_size=64, arch="gfx942")
    errs = dv.check(_cfg(), tmp_path / "cache").errors
    assert any(
        "the KFD node is a gfx950 and HIP's device 0 a gfx942" in e for e in errs
    )
    assert any("disagree on the wave size: KFD 32, HIP 64" in e for e in errs)


def test_no_probe_no_check(sysfs, tmp_path, monkeypatch):
    monkeypatch.setenv(dv.PROBE_VAR, "")
    c = dv.check(_cfg(), tmp_path / "cache")
    assert c.hip is None and c.errors == [] and "empty" in c.skipped
    monkeypatch.delenv(dv.PROBE_VAR)
    assert dv.check(_cfg(), tmp_path / "cache").skipped
    monkeypatch.setenv(kfd.ROCR_VAR, "-1")
    assert "hides" in dv.check(_cfg(), tmp_path / "cache").skipped
    monkeypatch.setenv(kfd.ROCR_VAR, "0")
    assert "is not one of the 1 GPU(s)" in dv.check(_cfg(), tmp_path / "cache").skipped


def test_a_probe_that_fails_is_refused(sysfs, tmp_path, monkeypatch):
    probe(tmp_path, monkeypatch, fail=True)
    (err,) = dv.check(_cfg(), tmp_path / "cache").errors
    assert "cannot read HIP's device values" in err
    monkeypatch.setenv(dv.PROBE_VAR, str(tmp_path / "absent"))
    (err,) = dv.check(_cfg(), tmp_path / "cache").errors
    assert "is not an executable" in err and dv.PROBE_VAR in err


def test_values_the_stages_cannot_read_are_refused(sysfs, tmp_path, monkeypatch):
    """The comparison takes the values the stages read (lib/hardware.py),
    so a KFD node without one is refused unless the config pins it."""
    node = gf.kfd_root(sysfs) / "topology" / "nodes" / str(gf.GPUS[1].node)
    props = node / "properties"
    props.write_text(props.read_text().replace("lds_size_in_kb 48\n", ""))
    probe(tmp_path, monkeypatch)
    c = dv.check(_cfg(), tmp_path / "cache")
    (err,) = c.errors
    assert "the run's stages cannot read its device values" in err
    assert "lds_size_in_kb" in err and f"LDS bytes {KFD['lds_bytes']}" in err
    assert c.used["lds_bytes"] is None and c.record()["used_error"]
    pinned = dv.check(_cfg(lds_bytes=KFD["lds_bytes"]), tmp_path / "cache")
    assert pinned.errors == [] and pinned.used["lds_bytes"] == KFD["lds_bytes"]


def test_a_probe_that_does_not_end_is_refused(sysfs, tmp_path, monkeypatch):
    exe = tmp_path / "slow"
    exe.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(dv.PROBE_VAR, str(exe))
    monkeypatch.setattr(dv, "PROBE_TIMEOUT_S", 0.5)
    c = dv.check(_cfg(), tmp_path / "cache")
    (err,) = c.errors
    assert "GPU 1 does not answer" in err and "it was killed" in err
    assert c.no_answer and c.record()["no_answer"]


def test_a_probe_stuck_in_the_driver_does_not_hold_up_preflight(tmp_path, monkeypatch):
    class Stuck:
        def __init__(self, *args, **kwargs):
            self.killed = False

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired("probe", timeout)

        def kill(self):
            self.killed = True

    monkeypatch.setattr(subprocess, "Popen", Stuck)
    with pytest.raises(dv.NoAnswer, match="does not end after SIGKILL either"):
        dv.hip_values(tmp_path / "probe", 3, {})


def test_the_probe_is_built_once_from_its_source(tmp_path, monkeypatch):
    monkeypatch.delenv(dv.PROBE_VAR, raising=False)
    monkeypatch.delenv(kfd.SYSFS_ROOT_VAR, raising=False)
    compiler = tmp_path / "cc"
    log = tmp_path / "cc.log"
    compiler.write_text(
        f"#!{sys.executable}\nimport sys\n"
        f"open({str(log)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
        "out = sys.argv[sys.argv.index('-o') + 1]\n"
        "open(out, 'w').write('#!/bin/sh\\n')\n"
    )
    compiler.chmod(compiler.stat().st_mode | stat.S_IXUSR)
    first = dv.probe_executable(tmp_path / "cache", compiler=str(compiler))
    again = dv.probe_executable(tmp_path / "cache", compiler=str(compiler))
    assert first == again and first.is_file()
    (args,) = log.read_text().splitlines()
    assert str(dv.PROBE_SOURCE) in args and "-lamdhip64" in args
    assert "hipblaslt" not in args


def test_preflight_records_both_and_refuses_a_disagreement(
    sysfs, tmp_path, monkeypatch
):
    import run_orchestrator as ro
    from test_run_orchestrator import make_config

    cfg, _ = ro.load_config(make_config(tmp_path, **{"runtime.devices": [1]}))
    paths = ro.derive_paths(cfg)
    record = tmp_path / "run" / "configs" / "x_full.platform.json"
    record.parent.mkdir(parents=True)
    probe(tmp_path, monkeypatch, shared_mem_per_block=1024)
    errs = ro.preflight(cfg, paths, "full", tmp_path / "run", record)
    assert sum("as the LDS bytes" in e for e in errs) == 1
    doc = json.loads(record.read_text())
    seen = doc["device_values"]
    assert (
        seen["kfd"]["lds_bytes"] == KFD["lds_bytes"]
        and seen["hip"]["lds_bytes"] == 1024
    )
    assert seen["problems"] and set(doc["host"]) == {"amdgpu_driver", "kernel", "rocm"}
    probe(tmp_path, monkeypatch)
    errs = ro.preflight(cfg, paths, "full", tmp_path / "run", record)
    assert not any("as the LDS bytes" in e for e in errs)
    assert json.loads(record.read_text())["device_values"]["problems"] == []
    assert ro.preflight(cfg, paths, "full") == [
        e for e in errs if "device values" not in e
    ]


def test_a_run_with_data_and_an_audit_set_only_warn(
    sysfs, tmp_path, monkeypatch, capsys
):
    """A run that already holds data was made with the values it uses, and
    an audit set serves runs that check their own device: a difference is
    named there, not refused."""
    import run_orchestrator as ro
    from lib import pinning
    from test_run_orchestrator import make_config

    cfg, _ = ro.load_config(make_config(tmp_path, **{"runtime.devices": [1]}))
    probe(tmp_path, monkeypatch, shared_mem_per_block=1024)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (err,) = ro._device_value_errors(cfg, run_dir)
    assert "as the LDS bytes" in err
    assert ro._device_value_errors(cfg, run_dir, mode="audit") == []
    assert "(not refused for an audit set)" in capsys.readouterr().err
    pinning.write_run_pin(run_dir, {"config": {}}, first_stage="stage01")
    assert ro._device_value_errors(cfg, run_dir) == []
    assert "(not refused for a run that already holds data)" in (
        capsys.readouterr().err
    )
    doc = json.loads((run_dir / "platform.json").read_text())
    assert doc["device_values"]["problems"] == [err]
    exe = tmp_path / "slow"
    exe.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(dv.PROBE_VAR, str(exe))
    monkeypatch.setattr(dv, "PROBE_TIMEOUT_S", 0.5)
    (err,) = ro._device_value_errors(cfg, run_dir, mode="audit")
    assert "does not answer" in err
