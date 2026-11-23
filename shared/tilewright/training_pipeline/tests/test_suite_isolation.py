# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The suite's own host (tests/conftest.py): a module-scoped fixture that
builds an audit set, as tests/test_audit.py builds its world before any
test's own fixtures, starts on the session's lease directory, empty fake
sysfs and readings, without amd-smi or a HIP read; like the audit worlds,
it then points the sysfs at a fake KFD tree of its own, whose GPUs load
the set's arch (test_audit.py `strict_host`), and builds the set, reading
no other sysfs and building and running no device probe. Every test then
gets a host of its own, and a test that would build the device probe, or
run a compiled one, fails unless it allows it."""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import bench_fakes as bf
import fake_hipblaslt as fh
import run_orchestrator as ro
import test_audit as ta
from lib import device_values as dv
from lib import fingerprint, kfd

GUARD_VARS = (
    "TILEWRIGHT_GUARD_DIR",
    "TILEWRIGHT_GUARD_SESSION",
    "TILEWRIGHT_GUARD_ROOT",
)
DEVICE_LIST_VARS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)
HOST_VARS = (
    "TILEWRIGHT_GPU_LEASE_DIR",
    "TILEWRIGHT_SYSFS_ROOT",
    "TILEWRIGHT_PROC_ROOT",
    "TILEWRIGHT_AMD_SMI",
    "TILEWRIGHT_DEVICE_PROBE",
    *GUARD_VARS,
    *DEVICE_LIST_VARS,
)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """An audit-mode invocation of the orchestrator on the fake bench that
    builds the set, with the host variables it started with, its own fake
    sysfs and every sysfs root it read."""
    pytest.importorskip("tilewright")
    root = tmp_path_factory.mktemp("isolation")
    mp = pytest.MonkeyPatch()
    try:
        yield _build(root, mp)
    finally:
        mp.undo()


def _build(root: Path, mp: pytest.MonkeyPatch) -> dict:
    env = {var: os.environ.get(var) for var in HOST_VARS}
    roots = []
    sysfs_root = kfd.sysfs_root

    def recorded():
        roots.append(sysfs_root())
        return roots[-1]

    mp.setattr(kfd, "sysfs_root", recorded)
    fh.unhide_devices(mp)
    sys_root = ta.strict_host(root, mp)
    bench_root = root / "bench"
    build = bench_root / "projects" / "hipblaslt" / "build" / "release"
    fh.write_fake_build(build, ta.ARCH, bf.LIBRARY_STEM, fh.kernels_grid())
    mp.setenv("FAKE_LIBRARY_DIR", str(build / "Tensile" / "library" / ta.ARCH))
    mp.setenv("FAKE_LIBRARY_STEM", bf.LIBRARY_STEM)
    audit = root / "audit_set"
    cfg_path = root / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(ta.base_config(bench_root, audit)))
    argv = ["--config", str(cfg_path), "--mode", "audit", "--run-dir", str(audit)]
    assert ro.main(argv) == 0
    platform = [
        json.loads(p.read_text())
        for p in sorted(audit.glob(f"configs/*.{fingerprint.PLATFORM_FILE}"))
    ]
    return {
        "root": root,
        "env": env,
        "sys": sys_root,
        "roots": roots,
        "platform": platform,
    }


def test_a_module_world_starts_on_the_sessions_host(world, private_host):
    env = world["env"]
    assert env["TILEWRIGHT_GPU_LEASE_DIR"] == str(private_host / "leases")
    assert env["TILEWRIGHT_SYSFS_ROOT"] == str(private_host / "sys")
    assert env["TILEWRIGHT_PROC_ROOT"] == str(private_host / "proc")
    assert env["TILEWRIGHT_AMD_SMI"] == env["TILEWRIGHT_DEVICE_PROBE"] == ""
    assert [env[var] for var in GUARD_VARS] == [None] * len(GUARD_VARS)
    assert [env[var] for var in DEVICE_LIST_VARS] == ["-1"] * len(DEVICE_LIST_VARS)


def test_a_module_world_reads_no_sysfs_but_its_fake_tree(world):
    assert world["roots"] and set(world["roots"]) == {world["sys"]}
    assert world["platform"]
    for doc in world["platform"]:
        assert doc["host"]["amdgpu_driver"] is None
        values = doc["device_values"]
        assert values["gpu"] == 0 and values["kfd"]["arch"] == "gfx1250"


def test_a_module_world_builds_and_runs_no_device_probe(world):
    assert not list(world["root"].rglob("device_props"))
    assert world["platform"]
    for doc in world["platform"]:
        assert doc["device_values"]["hip"] is None


def test_every_test_gets_a_host_of_its_own(private_host):
    for var, name in (
        ("TILEWRIGHT_GPU_LEASE_DIR", "leases"),
        ("TILEWRIGHT_SYSFS_ROOT", "sys"),
        ("TILEWRIGHT_PROC_ROOT", "proc"),
    ):
        assert os.environ[var] != str(private_host / name)
    assert not list(Path(os.environ["TILEWRIGHT_SYSFS_ROOT"]).iterdir())
    assert os.environ["TILEWRIGHT_AMD_SMI"] == ""
    assert os.environ["TILEWRIGHT_DEVICE_PROBE"] == ""


def test_a_test_may_take_a_host_that_loads_the_strict_arch(strict_sysfs, private_host):
    from lib.arch import STRICT_SWITCH

    assert os.environ["TILEWRIGHT_SYSFS_ROOT"] == str(strict_sysfs)
    assert strict_sysfs != private_host / "sys"
    gpus = kfd.gpus()
    assert gpus and all(kfd.gpu_isa(g) == ("gfx1250", 0) for g in gpus)
    devices = list(range(len(gpus)))
    cfg = {"arch": ta.ARCH, "runtime": {"devices": devices}}
    assert ro._bench_device_errors(cfg, {STRICT_SWITCH: "0"}) == []
    assert ro._bench_device_errors(cfg, {}) != []


def _compiler(tmp_path):
    """A compiler stand-in that logs each run."""
    log = tmp_path / "cc.log"
    cc = tmp_path / "cc"
    cc.write_text(f"#!{sys.executable}\nopen({str(log)!r}, 'a').write('ran\\n')\n")
    cc.chmod(cc.stat().st_mode | stat.S_IXUSR)
    return cc, log


def _compiled_probe(tmp_path):
    """An executable that starts as a compiled program does and runs
    nowhere."""
    probe = tmp_path / "device_props"
    probe.write_bytes(b"\x7fELF" + bytes(60))
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)
    return probe


def test_a_test_that_would_build_the_device_probe_fails(tmp_path, monkeypatch):
    monkeypatch.delenv(dv.PROBE_VAR)
    monkeypatch.delenv(kfd.SYSFS_ROOT_VAR)
    cc, log = _compiler(tmp_path)
    with pytest.raises(pytest.fail.Exception, match="would build the device probe"):
        dv.probe_executable(tmp_path / "cache", compiler=str(cc))
    assert not log.exists()


def test_a_test_that_would_run_a_compiled_probe_fails(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: started.append(a))
    with pytest.raises(pytest.fail.Exception, match="would run the compiled device"):
        dv.hip_values(_compiled_probe(tmp_path), 0, {})
    assert started == []


@pytest.mark.usefixtures("allow_device_probe")
def test_a_test_may_allow_the_device_probe(tmp_path, monkeypatch):
    class Done:
        returncode = 0

        def __init__(self, *args, **kwargs):
            pass

        def communicate(self, timeout=None):
            return json.dumps({"warp_size": 32}) + "\n", ""

    monkeypatch.setattr(subprocess, "Popen", Done)
    assert dv.hip_values(_compiled_probe(tmp_path), 0, {})["wave_size"] == 32
