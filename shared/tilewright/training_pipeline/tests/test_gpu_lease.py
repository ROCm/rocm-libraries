# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-GPU leases and scripts/gpu_run.py, with lease files in the test's
own directory (conftest) and commands that use no GPU."""

import json
import os
import pwd
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

import gpu_fakes as gf
from lib import gpu_lease, kfd

PIPELINE_DIR = Path(__file__).resolve().parent.parent
GPU_RUN = PIPELINE_DIR / "scripts" / "gpu_run.py"

HOLDER = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {root!r})
    from lib import gpu_lease
    with gpu_lease.hold_gpus({gpus!r}, "experiment", run_dir="/runs/x"):
        print("held", flush=True)
        time.sleep(float(sys.argv[1]))
    """
)


@pytest.fixture
def visible(monkeypatch):
    for var in kfd.HIP_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(kfd.ROCR_VAR, "0,1,2")


def _holder(gpus, seconds=60.0):
    proc = subprocess.Popen(
        [sys.executable, "-c", HOLDER.format(root=str(PIPELINE_DIR), gpus=gpus)]
        + [str(seconds)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.readline().strip() == "held"
    return proc


def test_the_lease_directory(monkeypatch, tmp_path):
    monkeypatch.setenv(gpu_lease.LEASE_DIR_VAR, str(tmp_path / "explicit"))
    assert gpu_lease.lease_dir() == tmp_path / "explicit"
    monkeypatch.delenv(gpu_lease.LEASE_DIR_VAR)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert gpu_lease.lease_dir() == tmp_path / "tmp" / gpu_lease.LEASE_DIR_NAME
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    assert gpu_lease.lease_dir() == tmp_path / "tmp" / gpu_lease.LEASE_DIR_NAME


def test_a_gpu_another_process_tree_holds_is_refused_naming_the_holder():
    holder = _holder([1])
    try:
        with pytest.raises(gpu_lease.LeaseBusy) as e:
            with gpu_lease.hold_gpus([0, 1], "stage03 bench"):
                pass
        msg = str(e.value)
        assert "refused (one process tree per GPU): GPU 1 is leased by" in msg
        assert f"pid {holder.pid} (user {pwd.getpwuid(os.getuid()).pw_name})" in msg
        assert "for 'experiment' in /runs/x since " in msg
        assert e.value.holder.purpose == "experiment"
        with gpu_lease.hold_gpus([0], "only gpu 0") as held:
            assert held.acquired == [0]
    finally:
        holder.kill()
        holder.wait()
    with gpu_lease.hold_gpus([1], "after") as held:
        assert held.acquired == [1]


def test_a_held_lease_records_its_holder_and_release_clears_it():
    path = gpu_lease.lock_path(2)
    with gpu_lease.hold_gpus([2], "probe", run_dir="/runs/y") as held:
        rec = gpu_lease.read_holder(path)
        assert (rec.gpu, rec.pid, rec.purpose, rec.run_dir) == (
            2,
            os.getpid(),
            "probe",
            "/runs/y",
        )
        assert held.gpus == [2] and held.fds
        status = gpu_lease.status()
        assert [(s["gpu"], s["held"]) for s in status] == [(2, True)]
        assert status[0]["holder"]["purpose"] == "probe"
    assert path.read_text() == ""
    assert [(s["gpu"], s["held"], s["holder"]) for s in gpu_lease.status()] == [
        (2, False, None)
    ]


def test_the_holder_and_its_descendants_use_its_lease():
    child = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(PIPELINE_DIR)!r})
        from lib import gpu_lease
        with gpu_lease.hold_gpus([0, 1], "nested") as held:
            print(held.acquired, held.inherited)
        """
    )
    with gpu_lease.hold_gpus([0], "outer"):
        with gpu_lease.hold_gpus([0], "same process") as again:
            assert (again.acquired, again.inherited) == ([], [0])
        out = subprocess.run(
            [sys.executable, "-c", child], capture_output=True, text=True, check=True
        )
        assert out.stdout.split("\n")[0] == "[1] [0]"
        assert gpu_lease.read_holder(gpu_lease.lock_path(0)).purpose == "outer"


def test_wait_s_waits_for_the_holder_to_finish():
    holder = _holder([0], seconds=1.0)
    try:
        t0 = time.monotonic()
        with gpu_lease.hold_gpus([0], "patient", wait_s=60) as held:
            assert held.acquired == [0]
        assert time.monotonic() - t0 < 60
    finally:
        holder.kill()
        holder.wait()


def test_a_wait_for_a_lease_is_reported_until_it_is_taken(monkeypatch):
    monkeypatch.setattr(gpu_lease, "WAIT_REPORT_S", 0.5)
    holder = _holder([0], seconds=2.0)
    heard = []
    try:
        with gpu_lease.hold_gpus([0], "patient", wait_s=60, report=heard.append):
            pass
    finally:
        holder.kill()
        holder.wait()
    assert heard[0].startswith(f"GPU 0 is leased by pid {holder.pid}")
    assert heard[0].endswith("; waiting up to 1m00s for it")
    assert heard[1].startswith("still waiting for GPU 0 (") and "of 1m00s" in heard[1]
    assert heard[-1].startswith("GPU 0 leased after ")
    quiet = []
    with gpu_lease.hold_gpus([0], "free", wait_s=60, report=quiet.append):
        pass
    assert quiet == []


def test_hold_leases_the_gpus_of_the_devices_only(visible, monkeypatch):
    with gpu_lease.hold([2], "stage03 bench") as held:
        assert held.acquired == [2]
    monkeypatch.setenv(kfd.ROCR_VAR, "-1")
    with gpu_lease.hold([0, 1], "hidden") as held:
        assert held.gpus == []
    monkeypatch.setenv(kfd.ROCR_VAR, "0,1")
    with pytest.raises(gpu_lease.LeaseError, match="one of the 2 GPU"):
        with gpu_lease.hold([3], "missing"):
            pass


# ── gpu_run.py ───────────────────────────────────────────────────────────────


def _gpu_run(*argv, env=None, **kw):
    return subprocess.run(
        [sys.executable, str(GPU_RUN), *argv],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        **kw,
    )


SHOW = textwrap.dedent(
    """
    import json, os, sys
    from pathlib import Path
    lease = Path(os.environ["TILEWRIGHT_GPU_LEASE_DIR"]) / "gpu2.lock"
    print(json.dumps({
        "env": {k: os.environ.get(k) for k in
                ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")},
        "lease": json.loads(lease.read_text()),
    }))
    sys.exit(3)
    """
)


def test_gpu_run_runs_the_command_on_its_gpu_under_the_lease():
    env = dict(os.environ, HIP_VISIBLE_DEVICES="5", CUDA_VISIBLE_DEVICES="5")
    out = _gpu_run(
        "--gpu", "2", "--purpose", "sweep", "--", sys.executable, "-c", SHOW, env=env
    )
    assert out.returncode == 3, out.stderr
    seen = json.loads(out.stdout)
    assert seen["env"] == {
        "ROCR_VISIBLE_DEVICES": "2",
        "HIP_VISIBLE_DEVICES": None,
        "CUDA_VISIBLE_DEVICES": None,
    }
    assert (seen["lease"]["gpu"], seen["lease"]["purpose"]) == (2, "sweep")
    assert [(s["gpu"], s["held"]) for s in gpu_lease.status()] == [(2, False)]


def test_gpu_run_refuses_a_gpu_another_process_tree_holds():
    holder = _holder([1])
    try:
        out = _gpu_run("--gpu", "1", "--", "true")
        assert out.returncode == 2
        assert f"GPU 1 is leased by pid {holder.pid}" in out.stderr
        assert "for 'experiment'" in out.stderr
        status = _gpu_run("--status")
        assert f"GPU 1: held by pid {holder.pid}" in status.stdout
    finally:
        holder.kill()
        holder.wait()


def test_gpu_run_lease_lasts_as_long_as_the_command(tmp_path):
    pid_file = tmp_path / "pid"
    cmd = f"echo $$ > {pid_file}; exec sleep 60"
    run = subprocess.Popen(
        [sys.executable, str(GPU_RUN), "--gpu", "0", "--", "sh", "-c", cmd]
    )
    try:
        assert gf.wait_for(lambda: pid_file.is_file() and pid_file.read_text())
        command = int(pid_file.read_text())
        run.kill()
        run.wait()
        with pytest.raises(gpu_lease.LeaseBusy, match="that pid has exited"):
            with gpu_lease.hold_gpus([0], "too early"):
                pass
    finally:
        os.kill(command, signal.SIGKILL)
    assert gf.wait_for(lambda: kfd.proc_info(command).gone)
    with gpu_lease.hold_gpus([0], "after the command") as held:
        assert held.acquired == [0]


def test_gpu_run_passes_sigterm_on_and_returns_the_commands_status(tmp_path):
    marker = tmp_path / "started"
    run = subprocess.Popen(
        [sys.executable, str(GPU_RUN), "--gpu", "1", "--"]
        + ["sh", "-c", f"touch {marker}; exec sleep 60"]
    )
    assert gf.wait_for(marker.exists)
    run.send_signal(signal.SIGTERM)
    assert run.wait(timeout=60) == 128 + signal.SIGTERM


def test_gpu_run_checks_its_arguments_and_names_other_users_of_the_gpu(
    tmp_path, monkeypatch
):
    assert _gpu_run("--gpu", "1").returncode == 2
    sys_root = gf.write_sysfs(tmp_path / "sys")
    env = dict(os.environ, TILEWRIGHT_SYSFS_ROOT=str(sys_root))
    missing = _gpu_run("--gpu", "7", "--", "true", env=env)
    assert missing.returncode == 2 and "there is no GPU 7" in missing.stderr
    with gf.outside_process() as pid:
        gf.set_memory(sys_root, pid, {1002: 4096})
        out = _gpu_run("--gpu", "1", "--", "true", env=env)
    assert out.returncode == 0
    assert f"warning: GPU 1 is in use: pid {pid} ('sleep'" in out.stderr
    gf.set_card_used(sys_root, gf.GPUS[2], 2 << 30)
    out = _gpu_run("--gpu", "2", "--", "true", env=env)
    assert out.returncode == 0
    assert "GPU 2 is in use: 2.00 GiB of VRAM no listed process holds" in out.stderr
