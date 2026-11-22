# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""_build_rocm.py: the ROCm identity CMake bakes into ``tensilelite-client --build-rocm``."""

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import types

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[3] / "_build_rocm.py"
_BUILD_ROCM = runpy.run_path(str(_SCRIPT))
_resolve = _BUILD_ROCM["resolve"]


def _root(tmp_path, info=None, manifest=None, name="rocm"):
    root = tmp_path / name
    root.mkdir()
    if info is not None:
        (root / ".info").mkdir()
        (root / ".info" / "version").write_text(info, encoding="utf-8")
    if manifest is not None:
        path = root / "share" / "therock" / "therock_manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(manifest if isinstance(manifest, str) else json.dumps(manifest))
    return root


def _no_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "rocm_sdk_core", None)


def test_pre_therock_root_records_the_base_version(tmp_path, monkeypatch):
    _no_sdk(monkeypatch)
    root = _root(tmp_path, info="7.2.4-43\n")

    assert str(_resolve(root)) == "7.2.4 base"


@pytest.mark.parametrize(
    "identity",
    ["7.12.0", "7.12.0rc1", "7.12.0a20261110", "7.12.0.dev0+" + "a" * 40,
     "7.12.0.dev0+" + "a" * 40 + ".asan", "7.12.0+bkc.20261110"],
)
def test_therock_manifest_records_the_full_identity(tmp_path, monkeypatch, identity):
    _no_sdk(monkeypatch)
    root = _root(tmp_path, info="7.12.0\n", manifest={"rocm_package_version": identity})

    assert str(_resolve(root)) == f"{identity} full"


@pytest.mark.parametrize(
    "manifest",
    [{"rocm_package_version": "git"}, {"rocm_package_version": 7}, {}, [], "{not json"],
)
def test_unusable_manifest_falls_back_to_info_version(tmp_path, monkeypatch, manifest):
    _no_sdk(monkeypatch)
    root = _root(tmp_path, info="7.12.0\n", manifest=manifest)

    assert str(_resolve(root)) == "7.12.0 base"


def test_manifest_without_info_version_is_enough(tmp_path, monkeypatch):
    _no_sdk(monkeypatch)
    root = _root(tmp_path, manifest={"rocm_package_version": "7.12.0rc1"})

    assert str(_resolve(root)) == "7.12.0rc1 full"


def test_inconsistent_installation_is_unknown_with_a_warning(tmp_path, monkeypatch):
    _no_sdk(monkeypatch)
    root = _root(tmp_path, info="7.11.0\n", manifest={"rocm_package_version": "7.12.0"})
    warnings = []

    assert str(_resolve(root, warn=warnings.append)) == "unknown unknown"
    assert "inconsistent ROCm installation" in warnings[0]


@pytest.mark.parametrize("info", [None, "garbage\n"])
def test_root_without_a_version_is_unknown(tmp_path, monkeypatch, info):
    _no_sdk(monkeypatch)
    root = _root(tmp_path, info=info)

    assert str(_resolve(root)) == "unknown unknown"


def _fake_sdk(monkeypatch, sdk_root, version="7.12.0rc2"):
    monkeypatch.setitem(sys.modules, "rocm_sdk_core", types.SimpleNamespace(__version__=version))

    def fake_run(cmd, **kwargs):
        assert cmd[1:] == ["-m", "rocm_sdk", "path", "--root"]
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{sdk_root}\n", stderr="")

    monkeypatch.setattr(_BUILD_ROCM["subprocess"], "run", fake_run)


def test_python_sdk_root_records_rocm_sdk_core_version(tmp_path, monkeypatch):
    root = _root(tmp_path, info="7.12.0\n", manifest={"rocm_package_version": "7.12.0rc1"})
    _fake_sdk(monkeypatch, root)

    assert str(_resolve(root)) == "7.12.0rc2 full"


def test_python_sdk_is_ignored_for_a_different_root(tmp_path, monkeypatch):
    root = _root(tmp_path, info="7.2.4\n")
    _fake_sdk(monkeypatch, _root(tmp_path, name="sdk"))

    assert str(_resolve(root)) == "7.2.4 base"


def test_command_line_prints_one_line(tmp_path):
    root = _root(tmp_path, info="7.2.4\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-S", str(_SCRIPT), str(root)],
        env=env, capture_output=True, text=True, check=True,
    )

    assert result.stdout == "7.2.4 base\n"
