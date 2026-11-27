# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for windows_rocm_setup.py delegation to resolve_quartz_rocm.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

_SCRIPT_PATH = Path(__file__).resolve().parent / "windows_rocm_setup.py"
_SPEC = importlib.util.spec_from_file_location("windows_rocm_setup", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
windows_rocm_setup = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = windows_rocm_setup
_SPEC.loader.exec_module(windows_rocm_setup)


def test_find_quartz_resolver_locates_shared_script_in_tree(tmp_path: Path) -> None:
    path = windows_rocm_setup._find_quartz_resolver()
    assert path.name == "resolve_quartz_rocm.py"
    assert path.is_file()

    override_script = (
        tmp_path / "projects" / "hipdnn" / "scripts" / "resolve_quartz_rocm.py"
    )
    override_script.parent.mkdir(parents=True)
    override_script.write_text("# stub\n", encoding="utf-8")
    assert (
        windows_rocm_setup._find_quartz_resolver(repo_root=tmp_path) == override_script
    )


def test_resolve_quartz_windows_rocm_delegates_contract_to_shared_resolver(
    monkeypatch,
) -> None:
    seen_kwargs: dict[str, object] = {}
    seen_repo_root: list[object] = []

    def fake_resolve(**kwargs):
        seen_kwargs.update(kwargs)
        return SimpleNamespace(
            rocm_version="10.3.0a20261120",
            wheels_url="https://nightly.repo.amd.com/rocm/whl-next/",
            tarballs_url="https://nightly.repo.amd.com/rocm/tarballs/",
            test_status="success",
            fallback_used=False,
        )

    def fake_load(repo_root=None):
        seen_repo_root.append(repo_root)
        return SimpleNamespace(resolve_quartz_rocm=fake_resolve)

    monkeypatch.setattr(windows_rocm_setup, "_load_quartz_resolver", fake_load)
    result = windows_rocm_setup.resolve_quartz_windows_rocm(
        "gfx1151", repo_root=Path("/fake/repo")
    )
    assert seen_repo_root == [Path("/fake/repo")]
    assert seen_kwargs == {
        "platform_name": "windows",
        "gpu_target": "gfx1151",
        "artifact": "wheels",
        "allow_build_only_fallback": True,
    }
    assert result == (
        "10.3.0a20261120",
        "https://nightly.repo.amd.com/rocm/whl-next/",
        "success",
        False,
    )


def test_pip_install_command_pins_quartz_version_and_supports_latest_and_sha() -> None:
    py = Path("D:/venv/Scripts/python.exe")
    pinned = windows_rocm_setup.pip_install_command(
        py,
        "gfx1151",
        None,
        rocm_version="10.3.0a20261120",
        index_url="https://nightly.repo.amd.com/rocm/whl-next/",
    )
    assert pinned[-3:] == [
        "--index-url",
        "https://nightly.repo.amd.com/rocm/whl-next/",
        "rocm[libraries,devel,device-gfx1151]==10.3.0a20261120",
    ]

    floating = windows_rocm_setup.pip_install_command(
        py, "gfx1100-all", None, rocm_version="LATEST"
    )
    assert floating[-1] == "rocm[libraries,devel,device-gfx1100]"

    staging = windows_rocm_setup.pip_install_command(py, "gfx1151", "deadbeef")
    assert any("deadbeef" in arg for arg in staging)


def test_provision_passes_resolved_quartz_version_and_index_to_pip(
    tmp_path: Path, monkeypatch
) -> None:
    venv = tmp_path / "venv"
    commands: list[list[str]] = []
    seen_targets: list[str] = []

    def fake_resolve(target, **_kw):
        seen_targets.append(target)
        return (
            "10.3.0a20261120",
            "https://rocm.nightlies.amd.com/whl-next/",
            "success",
            False,
        )

    monkeypatch.setattr(windows_rocm_setup, "resolve_quartz_windows_rocm", fake_resolve)
    monkeypatch.setattr(
        windows_rocm_setup, "run", lambda cmd: (commands.append(list(cmd)), 0)[1]
    )

    rc = windows_rocm_setup.provision(
        venv, "gfx906-all", None, force=False, rocm_version="Quartz"
    )
    assert rc == 0
    assert seen_targets == ["gfx906"]
    pip_cmd = next(cmd for cmd in commands if "pip" in cmd)
    assert "--index-url" in pip_cmd
    assert "https://rocm.nightlies.amd.com/whl-next/" in pip_cmd
    assert "rocm[libraries,devel,device-gfx906]==10.3.0a20261120" in pip_cmd


def test_provision_explicit_version_and_latest_bypass_quartz(
    tmp_path: Path, monkeypatch
) -> None:
    def fail_if_called(_target, **_kw):
        raise AssertionError("Quartz should not be called")

    monkeypatch.setattr(
        windows_rocm_setup, "resolve_quartz_windows_rocm", fail_if_called
    )

    commands: list[list[str]] = []
    monkeypatch.setattr(
        windows_rocm_setup, "run", lambda cmd: (commands.append(list(cmd)), 0)[1]
    )
    assert (
        windows_rocm_setup.provision(
            tmp_path / "v1",
            "gfx1151",
            None,
            force=False,
            rocm_version="10.3.0a20261119",
        )
        == 0
    )
    pip_explicit = next(cmd for cmd in commands if "pip" in cmd)
    assert pip_explicit[-1] == "rocm[libraries,devel,device-gfx1151]==10.3.0a20261119"

    commands.clear()
    assert (
        windows_rocm_setup.provision(
            tmp_path / "v2", "gfx1151", None, force=False, rocm_version="LATEST"
        )
        == 0
    )
    pip_latest = next(cmd for cmd in commands if "pip" in cmd)
    assert pip_latest[-1] == "rocm[libraries,devel,device-gfx1151]"


def test_provision_aborts_before_removing_venv_when_quartz_fails(
    tmp_path: Path, monkeypatch
) -> None:
    venv = tmp_path / "venv"
    venv.mkdir()
    sentinel = venv / "keep.txt"
    sentinel.write_text("ok")

    def boom(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(windows_rocm_setup, "resolve_quartz_windows_rocm", boom)
    rc = windows_rocm_setup.provision(venv, "gfx1151", None, force=True)
    assert rc == 1
    assert sentinel.exists()
