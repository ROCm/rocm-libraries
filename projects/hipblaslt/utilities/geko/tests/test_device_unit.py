# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
import sys

import pytest

from geko import device


# --- _count_gpus -------------------------------------------------------


def test_count_gpus_plain_lines() -> None:
    # amdgpu-arch / rocm_agent_enumerator style: one bare gfx token per line.
    output = "gfx950\ngfx950\n"
    assert device._count_gpus(output) == 2


def test_count_gpus_labelled_lines() -> None:
    # hipinfo / rocminfo style: gfx token embedded in a "key: value" line.
    output = "gcnArchName:                gfx1100\n"
    assert device._count_gpus(output) == 1


def test_count_gpus_excludes_placeholder_arch() -> None:
    # amdgpu-arch reports gfx000 on a GPU-less host; must not count as a device.
    assert device._count_gpus("gfx000\n") == 0


def test_count_gpus_mixed_and_empty_output() -> None:
    assert device._count_gpus("") == 0
    output = "gfx000\ngfx1100\n  Name:  gfx1250-strict\nsome unrelated line\n"
    assert device._count_gpus(output) == 2


# --- _candidate_enumerators / _find_enumerator --------------------------


def test_candidate_enumerators_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device.os, "name", "nt")
    assert device._candidate_enumerators() == ("hipinfo",)


def test_candidate_enumerators_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device.os, "name", "posix")
    assert device._candidate_enumerators() == (
        "amdgpu-arch",
        "rocm_agent_enumerator",
        "rocminfo",
    )


def test_find_enumerator_returns_first_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "_candidate_enumerators", lambda: ("a", "b", "c"))

    def _fake_which(name):
        return "/usr/bin/b" if name == "b" else None

    monkeypatch.setattr(device.shutil, "which", _fake_which)
    assert device._find_enumerator() == "/usr/bin/b"


def test_find_enumerator_returns_none_when_nothing_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "_candidate_enumerators", lambda: ("a", "b"))
    monkeypatch.setattr(device.shutil, "which", lambda _name: None)
    assert device._find_enumerator() is None


# --- _probe_device_count -------------------------------------------------


def test_probe_device_count_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "_find_enumerator", lambda: "/usr/bin/amdgpu-arch")

    def _fake_run(_cmd, **_kwargs):
        return subprocess.CompletedProcess(args=_cmd, returncode=0, stdout="gfx950\ngfx950\n", stderr="")

    monkeypatch.setattr(device.subprocess, "run", _fake_run)
    count, error = device._probe_device_count()
    assert error is None
    assert count == 2


def test_probe_device_count_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "_find_enumerator", lambda: "/usr/bin/amdgpu-arch")

    def _fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1.0))

    monkeypatch.setattr(device.subprocess, "run", _fake_run)
    count, error = device._probe_device_count()
    assert count is None
    assert error == "timeout"


def test_probe_device_count_no_enumerator_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "_find_enumerator", lambda: None)

    def _unexpected_run(*_args, **_kwargs):
        raise AssertionError("subprocess.run must not be called when no enumerator tool is found")

    monkeypatch.setattr(device.subprocess, "run", _unexpected_run)
    count, error = device._probe_device_count()
    assert count is None
    assert error is not None


def test_probe_device_count_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "_find_enumerator", lambda: "/usr/bin/amdgpu-arch")

    def _fake_run(_cmd, **_kwargs):
        return subprocess.CompletedProcess(args=_cmd, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(device.subprocess, "run", _fake_run)
    count, error = device._probe_device_count()
    assert count is None
    assert error is not None


# --- main ----------------------------------------------------------------


def test_main_returns_zero_when_sufficient_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "_probe_device_count", lambda **_kwargs: (4, None))
    assert device.main(2) == 0


def test_main_returns_zero_with_warning_when_enumerator_unavailable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(device, "_probe_device_count", lambda **_kwargs: (None, "no tool found"))
    with caplog.at_level("WARNING", logger="GEKO"):
        rc = device.main(2)
    assert rc == 0
    assert any("no tool found" in rec.message for rec in caplog.records)


def test_main_returns_one_on_timeout(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(device, "_probe_device_count", lambda **_kwargs: (None, "timeout"))
    with caplog.at_level("WARNING", logger="GEKO"):
        rc = device.main(2)
    assert rc == 1
    assert any("timed out" in rec.message for rec in caplog.records)


def test_main_returns_two_on_insufficient_devices(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(device, "_probe_device_count", lambda **_kwargs: (1, None))
    with caplog.at_level("WARNING", logger="GEKO"):
        rc = device.main(2)
    assert rc == 2
    assert any("fewer than" in rec.message for rec in caplog.records)


# --- __main__ smoke test (real subprocess, no mocking) -------------------


def test_module_invocation_zero_devices_always_passes() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "geko.device", "0"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
