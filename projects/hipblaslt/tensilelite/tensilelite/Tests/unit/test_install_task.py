# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

import tasks


pytestmark = pytest.mark.unit

_SOURCE_ROOT = Path(__file__).resolve().parents[3]


def test_invoke_install_is_a_discoverable_developer_workflow():
    result = subprocess.run(
        ["invoke", "--help", "install"],
        cwd=_SOURCE_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--build-dir" in result.stdout
    assert "--gpu-targets" in result.stdout
    assert "--rocm-path" in result.stdout


def test_install_binds_the_actual_cmake_client_output(tmp_path):
    executable = "tensilelite-client.exe" if sys.platform == "win32" else "tensilelite-client"
    expected = tmp_path / "build/tensilelite/client" / executable
    assert tasks._built_client_path(tmp_path / "build") == expected


@pytest.mark.skipif(sys.platform != "linux", reason="invoke install is Linux-only")
def test_install_uses_one_selected_rocm_root_and_binds_the_built_client(
    tmp_path, monkeypatch
):
    rocm_root = (tmp_path / "rocm").resolve()
    rocm_root.mkdir()
    build_dir = tmp_path / "build"
    client = tasks._built_client_path(build_dir)
    client.parent.mkdir(parents=True)
    client.write_text("client", encoding="utf-8")
    client.chmod(0o755)

    calls = []

    def record_rocisa(context, **kwargs):
        calls.append(("rocisa", kwargs))

    def record_build_client(context, **kwargs):
        calls.append(("build-client", kwargs))

    class Context:
        def run(self, command, **kwargs):
            calls.append(("run", command, kwargs))

    monkeypatch.setenv("ROCM_PATH", str(rocm_root))
    monkeypatch.setattr(tasks, "rocisa", SimpleNamespace(body=record_rocisa))
    monkeypatch.setattr(
        tasks, "build_client", SimpleNamespace(body=record_build_client)
    )

    tasks.install.body(Context(), build_dir=str(build_dir))

    requirements_command = shlex.split(calls[0][1])
    assert requirements_command == [
        str(Path(sys.executable).absolute()),
        "-m",
        "pip",
        "install",
        "-r",
        str(_SOURCE_ROOT / "requirements-dev-common.txt"),
    ]
    assert calls[0][2] == {}
    assert calls[1] == ("rocisa", {"rocm_path": str(rocm_root)})
    assert calls[2][0] == "build-client"
    assert calls[2][1]["rocm_path"] == str(rocm_root)
    editable_command = shlex.split(calls[3][1])
    assert editable_command[-4:] == [
        "--no-build-isolation",
        "--no-deps",
        "--editable",
        str(_SOURCE_ROOT),
    ]
    assert calls[3][2]["env"]["ROCM_PATH"] == str(rocm_root)
    assert shlex.split(calls[4][1]) == [
        str(Path(sys.executable).absolute()),
        "-m",
        "tensilelite_configure_client",
        "--client",
        str(client),
    ]


@pytest.mark.skipif(sys.platform != "linux", reason="invoke install is Linux-only")
@pytest.mark.parametrize("create_non_executable", [False, True])
def test_install_rejects_an_invalid_built_client(
    tmp_path, monkeypatch, create_non_executable
):
    rocm_root = (tmp_path / "rocm").resolve()
    rocm_root.mkdir()
    if create_non_executable:
        client = tasks._built_client_path(tmp_path / "build")
        client.parent.mkdir(parents=True)
        client.write_text("client", encoding="utf-8")

    class Context:
        def run(self, command, **kwargs):
            pass

    monkeypatch.setenv("ROCM_PATH", str(rocm_root))
    monkeypatch.setattr(tasks, "rocisa", SimpleNamespace(body=lambda *args, **kwargs: None))
    monkeypatch.setattr(
        tasks, "build_client", SimpleNamespace(body=lambda *args, **kwargs: None)
    )

    with pytest.raises(tasks.Exit, match="Built tensilelite-client is missing or not executable"):
        tasks.install.body(Context(), build_dir=str(tmp_path / "build"))
