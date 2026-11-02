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


def _args(tmp_path: Path, arch: str = "gfx942") -> list:
    return [
        "--mode",
        "early",
        "--arch",
        arch,
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

    def test_strict_host_is_not_read_as_its_base_arch(self, tmp_path, monkeypatch):
        """gfx1250-strict is a distinct target with its own code-object identity."""
        monkeypatch.setattr(
            device_probe.subprocess, "run", _fake_run(stdout="Name: gfx1250-strict\n")
        )
        assert device_probe.main(_args(tmp_path, "gfx1250")) == 1
        assert device_probe.main(_args(tmp_path, "gfx1250-strict")) == 0

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


@pytest.mark.parametrize(
    "text",
    ["gfx90a", "gfx942", "gfx1100", "gfx1250", "gfx1250-strict", "gfx1250-strictall"],
)
def test_concrete_name_is_an_arch_token(text):
    assert device_probe.is_arch_token(text)


@pytest.mark.parametrize(
    "text",
    [
        # TheRock families and variants, and generic targets: no device reports them.
        "gfx94X-dcgpu",
        "gfx950-dcgpu",
        "gfx950-dcgpu-asan",
        "gfx900-dgpu",
        "gfx90c-igpu",
        "gfx950-all",
        "gfx1250-all-strict",
        "gfx11-generic",
        "gfx9-4-generic",
        # Shapes that are not target names.
        "gfx9",
        "gfx1250-Strict",
        "gfx1250--strict",
        "gfx1250-",
        "gfx1250-4",
        "GFX942",
        "gfx942:xnack-",
        "native",
        "",
    ],
)
def test_non_concrete_name_is_not_an_arch_token(text):
    assert not device_probe.is_arch_token(text)


@pytest.mark.parametrize("arch", ["gfx950-dcgpu", "gfx11-generic", "gfx1250-Strict"])
def test_cli_rejects_a_family_or_malformed_arch(tmp_path, arch):
    with pytest.raises(SystemExit) as excinfo:
        device_probe.main(_args(tmp_path, arch))
    assert excinfo.value.code == 2


class TestTokensInDeviceOutput:
    """A token is a whole concrete name; neighbouring text must not split or fake one."""

    @staticmethod
    def _probe(tmp_path, monkeypatch, stdout, arch):
        monkeypatch.setattr(device_probe.subprocess, "run", _fake_run(stdout=stdout))
        return device_probe.main(_args(tmp_path, arch))

    def test_isa_line_with_feature_suffix_names_the_base_arch(
        self, tmp_path, monkeypatch
    ):
        out = "Name: amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-\n"
        assert self._probe(tmp_path, monkeypatch, out, "gfx942") == 0

    def test_strict_isa_line_names_the_strict_target(self, tmp_path, monkeypatch):
        out = "Name: amdgcn-amd-amdhsa--gfx1250-strict:sramecc+\n"
        assert self._probe(tmp_path, monkeypatch, out, "gfx1250-strict") == 0
        assert self._probe(tmp_path, monkeypatch, out, "gfx1250") == 1

    def test_a_family_name_in_the_output_is_not_a_device(self, tmp_path, monkeypatch):
        out = "Name: gfx950-dcgpu\n"
        assert self._probe(tmp_path, monkeypatch, out, "gfx950") == 1

    def test_a_malformed_hyphen_suffix_does_not_read_as_the_base_arch(
        self, tmp_path, monkeypatch
    ):
        # An uppercase or empty suffix is not a target name, and not a device either.
        for out in ("Name: gfx942-Foo\n", "Name: gfx942-\n"):
            assert self._probe(tmp_path, monkeypatch, out, "gfx942") == 1
