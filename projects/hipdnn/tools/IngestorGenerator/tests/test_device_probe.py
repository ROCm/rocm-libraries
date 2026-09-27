# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""`device_probe`'s exit statuses, which a caller branches on.

The probe shells out to ``rocminfo``. Exit 1 means observed and negative, exit 3 means
not observed: a missing ``rocminfo`` raises ``FileNotFoundError`` (an ``OSError``) and
says nothing about the host's GPUs, so it must not halt an unattended run at its first
gate.
"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import device_probe  # noqa: E402


def _args(tmp_path: Path) -> list:
    return [
        "--mode",
        "early",
        "--arch",
        "gfx942",
        "--sweep-root",
        str(tmp_path),
    ]


def _fake_run(stdout: str = "", returncode: int = 0):
    def run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else [],
            returncode=returncode,
            stdout=stdout,
            stderr="",
        )

    return run


class TestExitStatusDistinguishesUnobservedFromNegative:
    def test_missing_utility_is_unobserved_not_device_absent(
        self, tmp_path, monkeypatch, capsys
    ):
        """No rocminfo on PATH must not read as no GPU."""

        def raise_missing(*args, **kwargs):
            raise FileNotFoundError(2, "No such file or directory: 'rocminfo'")

        monkeypatch.setattr(device_probe.subprocess, "run", raise_missing)
        rc = device_probe.main(_args(tmp_path))
        assert rc == 3, (
            "a host without rocminfo reported a device verdict; "
            "probe-unavailable and device-absent must not share an exit status"
        )
        err = capsys.readouterr().err
        assert "UNOBSERVED" in err
        assert "FAIL" not in err, "an unobserved condition was reported as a failure"

    def test_utility_present_and_arch_absent_is_a_failure(
        self, tmp_path, monkeypatch, capsys
    ):
        """rocminfo ran and the arch is not there: a real negative, still 1."""
        monkeypatch.setattr(
            device_probe.subprocess, "run", _fake_run(stdout="Name: gfx90a\n")
        )
        rc = device_probe.main(_args(tmp_path))
        assert rc == 1
        err = capsys.readouterr().err
        assert "FAIL" in err
        assert "gfx90a" in err, "the negative should name what was found instead"

    def test_utility_present_and_arch_present_succeeds(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            device_probe.subprocess, "run", _fake_run(stdout="Name: gfx942\n")
        )
        rc = device_probe.main(_args(tmp_path))
        assert rc == 0
        assert "UNOBSERVED" not in capsys.readouterr().err

    def test_nonzero_rocminfo_is_a_failure_not_unobserved(self, tmp_path, monkeypatch):
        """A utility that ran and errored HAS reported; it is not unobserved."""
        monkeypatch.setattr(device_probe.subprocess, "run", _fake_run(returncode=1))
        assert device_probe.main(_args(tmp_path)) == 1

    def test_second_utility_observes_what_the_first_could_not(
        self, tmp_path, monkeypatch, capsys
    ):
        """A host missing only the reference tool is observable: the Windows ROCm wheels
        ship hipInfo and no rocminfo, so the fallthrough is behaviour rather than
        convenience."""

        def run(args, **kwargs):
            if args[0] == "rocminfo":
                raise FileNotFoundError(2, "No such file or directory: 'rocminfo'")
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="gcnArchName: gfx942\n", stderr=""
            )

        monkeypatch.setattr(device_probe.subprocess, "run", run)
        rc = device_probe.main(_args(tmp_path))
        assert rc == 0
        err = capsys.readouterr().err
        assert "UNOBSERVED" not in err
        assert "FAIL" not in err

    def test_every_utility_missing_is_still_unobserved(self, tmp_path, monkeypatch):
        """The fallthrough must not turn an unobservable host into a verdict."""

        def raise_missing(*args, **kwargs):
            raise FileNotFoundError(2, "No such file or directory")

        monkeypatch.setattr(device_probe.subprocess, "run", raise_missing)
        assert device_probe.main(_args(tmp_path)) == 3


class TestDeviceInfoRaisesTheDistinctType:
    def test_oserror_becomes_probe_unavailable(self, monkeypatch):
        def raise_perm(*args, **kwargs):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(device_probe.subprocess, "run", raise_perm)
        with pytest.raises(device_probe.ProbeUnavailable):
            device_probe.device_info("gfx942")

    def test_wrong_arch_stays_a_valueerror(self, monkeypatch):
        monkeypatch.setattr(
            device_probe.subprocess, "run", _fake_run(stdout="Name: gfx1100\n")
        )
        with pytest.raises(ValueError):
            device_probe.device_info("gfx942")
