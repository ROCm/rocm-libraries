# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""release_metadata.py: the version and commit CMake gives the native client."""

import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import pytest

from tensilelite import GENERATOR_VERSION


pytestmark = pytest.mark.unit

_SOURCE_ROOT = Path(__file__).resolve().parents[4]
_SHA = "0123456789abcdef0123456789abcdef01234567"


def _copy_metadata(dest, version="5.0.0"):
    """The files release_metadata.py needs, outside any git repository."""
    (dest / "tensilelite").mkdir(parents=True)
    shutil.copyfile(_SOURCE_ROOT / "release_metadata.py", dest / "release_metadata.py")
    shutil.copyfile(
        _SOURCE_ROOT / "tensilelite" / "_source_revision.py",
        dest / "tensilelite" / "_source_revision.py",
    )
    (dest / "VERSION").write_text(f"{version}\n", encoding="utf-8")
    return dest / "release_metadata.py"


def _run(script, *args, **env):
    base = {k: v for k, v in os.environ.items()
            if k not in ("TENSILELITE_SOURCE_REVISION", "ROCM_PATH", "ROCM_VERSION")}
    return subprocess.run(
        [sys.executable, str(script), *args],
        env={**base, **env},
        capture_output=True,
        text=True,
    )


def test_component_version_is_the_generator_version():
    metadata = runpy.run_path(str(_SOURCE_ROOT / "release_metadata.py"))
    component_version = (_SOURCE_ROOT / "VERSION").read_text(encoding="utf-8").strip()

    assert GENERATOR_VERSION == component_version
    assert metadata["component_version"]() == component_version


def test_cli_reports_the_dev_version_and_commit(tmp_path):
    script = _copy_metadata(tmp_path / "src")

    version = _run(script, "version", TENSILELITE_SOURCE_REVISION=_SHA)
    revision = _run(script, "source-revision", TENSILELITE_SOURCE_REVISION=_SHA)

    assert version.returncode == 0, version.stderr
    assert version.stdout == f"5.0.0+g{_SHA}\n"
    assert revision.stdout == f"{_SHA}\n"


def test_cli_reports_none_without_a_commit(tmp_path):
    script = _copy_metadata(tmp_path / "src")

    assert _run(script, "version").stdout == "5.0.0\n"
    assert _run(script, "source-revision").stdout == "none\n"


def test_cli_ignores_the_rocm_environment(tmp_path):
    script = _copy_metadata(tmp_path / "src")
    rocm = tmp_path / "rocm"
    (rocm / ".info").mkdir(parents=True)
    (rocm / ".info" / "version").write_text("7.2.4\n", encoding="utf-8")

    result = _run(script, "version", ROCM_PATH=str(rocm), ROCM_VERSION="7.2.4")

    assert result.stdout == "5.0.0\n"


@pytest.mark.parametrize(
    ("identity", "label"),
    [
        ("7.12.0", "rocm7.12.0"),
        ("7.12.0rc1", "rocm7.12.0rc1"),
        ("7.12.0a20261110", "rocm7.12.0a20261110"),
        ("7.12.0+bkc.20261110", "rocm7.12.0.bkc.20261110"),
        (f"7.12.0.dev0+{_SHA}", f"devrocm7.12.0.dev0.{_SHA}"),
        (f"7.12.0.dev0+{_SHA}.asan", f"devrocm7.12.0.dev0.{_SHA}.asan"),
    ],
)
def test_rocm_identity_replaces_the_commit_label(tmp_path, identity, label):
    script = _copy_metadata(tmp_path / "src")
    metadata = runpy.run_path(str(script))

    assert metadata["distribution_version"](_SHA, identity) == f"5.0.0+{label}"
    result = _run(script, "version", "--rocm-identity", identity, TENSILELITE_SOURCE_REVISION=_SHA)
    assert result.stdout == f"5.0.0+{label}\n", result.stderr


@pytest.mark.parametrize("identity", ["git", "seven", "7.12", ""])
def test_rocm_identity_must_be_a_version(tmp_path, identity):
    metadata = runpy.run_path(str(_copy_metadata(tmp_path / "src")))

    if identity:
        with pytest.raises(RuntimeError, match="ROCm identity must look like"):
            metadata["distribution_version"](_SHA, identity)
    else:
        assert metadata["distribution_version"](_SHA, identity) == f"5.0.0+g{_SHA}"


@pytest.mark.parametrize(
    "args",
    [[], ["rocm-root"], ["version", "extra"], ["/opt/rocm"], ["source-revision", "--rocm-identity", "7.2.4"]],
)
def test_cli_rejects_other_arguments(tmp_path, args):
    script = _copy_metadata(tmp_path / "src")

    result = _run(script, *args)

    assert result.returncode == 2
    assert "usage:" in result.stderr


@pytest.mark.parametrize("value", ["5.0", "5.0.0.dev1", "v5.0.0", ""])
def test_component_version_rejects_non_release_values(tmp_path, value):
    script = _copy_metadata(tmp_path / "src", version=value)
    metadata = runpy.run_path(str(script))

    with pytest.raises(RuntimeError, match="VERSION must contain"):
        metadata["component_version"]()
