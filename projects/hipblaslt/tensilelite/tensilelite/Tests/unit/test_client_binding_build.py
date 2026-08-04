# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import csv
import io
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import tarfile
import zipfile

import pytest


pytestmark = pytest.mark.unit

_SOURCE_ROOT = Path(__file__).resolve().parents[3]
_BINDING_FILE = "tensilelite-client-path.json"


def _rocm_root(tmp_path: Path) -> Path:
    root = tmp_path / "rocm"
    (root / ".info").mkdir(parents=True)
    (root / ".info" / "version").write_text("7.2.4\n", encoding="utf-8")
    return root


def _custom_client(tmp_path: Path) -> Path:
    executable = tmp_path / "built-client"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    link = tmp_path / "client-link"
    link.symlink_to(executable)
    return link.absolute()


def test_source_wheel_records_custom_client_path(tmp_path):
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()
    client = _custom_client(tmp_path)
    env = dict(os.environ, ROCM_PATH=str(_rocm_root(tmp_path)))

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            f"--config-settings=tensilelite.client-path={client}",
            str(_SOURCE_ROOT),
        ],
        check=True,
        cwd=_SOURCE_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    wheel = next(wheel_dir.glob("tensilelite-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        binding_names = [
            name for name in archive.namelist() if name.endswith(f".dist-info/{_BINDING_FILE}")
        ]
        assert len(binding_names) == 1
        binding_name = binding_names[0]
        assert json.loads(archive.read(binding_name)) == str(client)

        record_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/RECORD")
        )
        recorded_paths = {row[0] for row in csv.reader(io.TextIOWrapper(archive.open(record_name)))}
        assert binding_name in recorded_paths


def test_installed_custom_binding_is_used_without_rocm_client(tmp_path):
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()
    client = _custom_client(tmp_path)
    rocm_root = _rocm_root(tmp_path)
    build_env = dict(os.environ, ROCM_PATH=str(rocm_root))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            f"--config-settings=tensilelite.client-path={client}",
            str(_SOURCE_ROOT),
        ],
        check=True,
        cwd=_SOURCE_ROOT,
        env=build_env,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_dir.glob("tensilelite-*.whl"))
    target = tmp_path / "site-packages"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--target",
            str(target),
            str(wheel),
        ],
        check=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    runtime_env = dict(os.environ, PYTHONPATH=str(target), ROCM_PATH=str(rocm_root))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tensilelite",
        ],
        cwd=tmp_path,
        env=runtime_env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_sdist_carries_backend_but_not_custom_client_path(tmp_path):
    client = _custom_client(tmp_path)
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    env = dict(os.environ, ROCM_PATH=str(_rocm_root(tmp_path)))
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import build_backend, sys; "
            "build_backend.build_sdist(sys.argv[1], "
            "{'tensilelite.client-path': sys.argv[2]})",
            str(dist_dir),
            str(client),
        ],
        check=True,
        cwd=_SOURCE_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    sdist = next(dist_dir.glob("tensilelite-*.tar.gz"))
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
        assert any(name.endswith("/build_backend.py") for name in names)
        assert str(client).encode() not in b"".join(
            archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        )


def test_release_checker_rejects_custom_bound_wheel(tmp_path):
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()
    client = _custom_client(tmp_path)
    env = dict(os.environ, ROCM_PATH=str(_rocm_root(tmp_path)))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            f"--config-settings=tensilelite.client-path={client}",
            str(_SOURCE_ROOT),
        ],
        check=True,
        cwd=_SOURCE_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_dir.glob("tensilelite-*.whl"))

    result = subprocess.run(
        [
            sys.executable,
            str(_SOURCE_ROOT / "scripts" / "check_wheel_contents.py"),
            str(wheel),
            "--source-root",
            str(_SOURCE_ROOT),
        ],
        cwd=_SOURCE_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "custom client binding" in result.stderr


def test_release_wheel_has_no_custom_binding(tmp_path):
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()
    rocm_root = _rocm_root(tmp_path)
    client = rocm_root / "libexec" / "hipblaslt" / "tensilelite" / "tensilelite-client"
    client.parent.mkdir(parents=True)
    client.write_text("", encoding="utf-8")
    client.chmod(0o755)
    env = dict(os.environ, ROCM_PATH=str(rocm_root))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(_SOURCE_ROOT),
        ],
        check=True,
        cwd=_SOURCE_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_dir.glob("tensilelite-*.whl"))

    with zipfile.ZipFile(wheel) as archive:
        assert not any(name.endswith(_BINDING_FILE) for name in archive.namelist())

    result = subprocess.run(
        [
            sys.executable,
            str(_SOURCE_ROOT / "scripts" / "check_wheel_contents.py"),
            str(wheel),
            "--source-root",
            str(_SOURCE_ROOT),
        ],
        cwd=_SOURCE_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_relative_custom_client_path_is_rejected(tmp_path):
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()
    env = dict(os.environ, ROCM_PATH=str(_rocm_root(tmp_path)))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            "--config-settings=tensilelite.client-path=relative/client",
            str(_SOURCE_ROOT),
        ],
        cwd=_SOURCE_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "must be an absolute executable path" in result.stderr


def test_editable_environments_keep_independent_client_bindings(tmp_path):
    rocm_root = _rocm_root(tmp_path)
    support = tmp_path / "support"
    support.mkdir()
    (support / "rocisa.py").write_text("", encoding="utf-8")
    clients = []
    interpreters = []

    for name in ("first", "second"):
        client_dir = tmp_path / name
        client_dir.mkdir()
        client = _custom_client(client_dir)
        clients.append(client)
        environment = tmp_path / f"venv-{name}"
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(environment)],
            check=True,
            capture_output=True,
            text=True,
        )
        python = environment / "bin" / "python"
        interpreters.append(python)
        install_env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(site.getsitepackages()),
            ROCM_PATH=str(rocm_root),
        )
        install_result = subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-build-isolation",
                "--no-deps",
                "--editable",
                str(_SOURCE_ROOT),
                f"--config-settings=tensilelite.client-path={client}",
            ],
            cwd=tmp_path,
            env=install_env,
            capture_output=True,
            text=True,
        )
        assert install_result.returncode == 0, install_result.stderr

    for python, expected in zip(interpreters, clients):
        runtime_env = dict(
            os.environ,
            PYTHONPATH=str(support),
            ROCM_PATH=str(rocm_root),
        )
        result = subprocess.run(
            [
                str(python),
                "-c",
                "import tensilelite; from tensilelite import _runtime; "
                "print(_runtime.client_executable())",
            ],
            check=True,
            cwd=_SOURCE_ROOT,
            env=runtime_env,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == str(expected)
