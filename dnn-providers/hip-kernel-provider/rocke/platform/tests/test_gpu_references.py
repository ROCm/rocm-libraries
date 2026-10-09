# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Configure/install the published reference path without a compiler or GPU."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

_PLATFORM = Path(__file__).resolve().parents[1]
_LIBRARY = _PLATFORM.parent / "library"
_CMAKE = shutil.which("cmake")
pytestmark = pytest.mark.skipif(_CMAKE is None, reason="CMake is required")


@pytest.fixture
def project(tmp_path):
    library = tmp_path / "library"
    common = library / "tests/reference_common"
    common.mkdir(parents=True)
    shutil.copyfile(
        _LIBRARY / "tests/reference_common/artifact.py", common / "artifact.py"
    )
    # No qualification registries: installation needs only publication metadata.
    cmake = tmp_path / "cmake"
    cmake.mkdir()
    shutil.copyfile(
        _PLATFORM / "cmake/InstallGpuReferences.cmake",
        cmake / "InstallGpuReferences.cmake",
    )
    (cmake / "PublishedGpuReferences.cmake").write_text(
        "set(ROCKE_PUBLISHED_SDPA_ARCHITECTURES gfx942)\n"
        "set(ROCKE_PUBLISHED_CONV_ARCHITECTURES gfx942)\n"
    )
    for operation in ("sdpa", "conv"):
        bundle = tmp_path / operation
        (bundle / "payload").mkdir(parents=True)
        payload = operation.encode()
        (bundle / "payload/kernel.hsaco").write_bytes(payload)
        manifest = {
            "schema": 2,
            "baseline_revision": "a" * 40,
            "operation": "conv-fwd" if operation == "conv" else "sdpa",
            "files": {"kernel.hsaco": hashlib.sha256(payload).hexdigest()},
        }
        encoded = json.dumps(manifest).encode()
        (bundle / "manifest.json").write_bytes(encoded)
        lock = (
            library
            / f"tests/{operation}_reference/architectures/gfx942/baseline_lock.json"
        )
        lock.parent.mkdir(parents=True)
        lock.write_text(
            json.dumps(
                {
                    "schema": 2,
                    "baseline_revision": manifest["baseline_revision"],
                    "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
        )
        archive = library / f"tests/reference_bundles/{operation}/gfx942.tar.gz"
        archive.parent.mkdir(parents=True)
        subprocess.run(
            [
                sys.executable,
                "-I",
                str(common / "artifact.py"),
                "pack",
                "--operation",
                operation,
                "--bundle",
                str(bundle),
                "--lock",
                str(lock),
                "--archive",
                str(archive),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    (tmp_path / "CMakeLists.txt").write_text(
        f"""cmake_minimum_required(VERSION 3.25)
project(reference_install NONE)
set(_ROCKE_LIBRARY_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/library")
set(ROCKE_TEST_INSTALL_DIR "tests")
set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}")
include("${{CMAKE_CURRENT_SOURCE_DIR}}/cmake/InstallGpuReferences.cmake")
file(WRITE "${{CMAKE_CURRENT_BINARY_DIR}}/installed.txt"
     "sdpa=${{_ROCKE_SDPA_INSTALLED_ARCHITECTURES}}\\nconv=${{_ROCKE_CONV_INSTALLED_ARCHITECTURES}}\\n")
"""
    )
    return tmp_path


def configure(project, *, enabled=True, python=True):
    return subprocess.run(
        [
            _CMAKE,
            "-S",
            str(project),
            "-B",
            str(project / "build"),
            f"-DROCKE_INSTALL_TEST_GPU_REFERENCES={'ON' if enabled else 'OFF'}",
            f"-DPython3_Interpreter_FOUND={'TRUE' if python else 'FALSE'}",
        ],
        capture_output=True,
        text=True,
    )


def test_published_operations_install_separately(project):
    result = configure(project)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (project / "build/installed.txt").read_text() == "sdpa=gfx942\nconv=gfx942\n"
    prefix = project / "installed"
    subprocess.run(
        [
            _CMAKE,
            "--install",
            str(project / "build"),
            "--prefix",
            str(prefix),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for operation in ("sdpa", "conv"):
        payload = (
            prefix
            / f"tests/engines/test_arch_content/rocke/{operation}/gfx942/payload/kernel.hsaco"
        )
        assert payload.read_bytes() == operation.encode()


def test_disabled_references_need_neither_python_nor_bundles(project):
    shutil.rmtree(project / "library")
    result = configure(project, enabled=False, python=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (project / "build/installed.txt").read_text() == "sdpa=\nconv=\n"


@pytest.mark.parametrize("problem", ["archive", "lock", "integrity", "python"])
def test_required_reference_prerequisites_fail_configuration(project, problem):
    library = project / "library/tests"
    lock = library / "conv_reference/architectures/gfx942/baseline_lock.json"
    if problem == "archive":
        (library / "reference_bundles/conv/gfx942.tar.gz").unlink()
    elif problem == "lock":
        lock.unlink()
    elif problem == "integrity":
        data = json.loads(lock.read_text())
        data["manifest_sha256"] = "0" * 64
        lock.write_text(json.dumps(data))
    result = configure(project, python=problem != "python")
    assert result.returncode != 0
    expected = {
        "archive": "reference archive missing",
        "lock": "No qualified conv baseline lock for gfx942",
        "integrity": "does not match the pinned lock",
        "python": "requires a Python3 interpreter",
    }
    assert expected[problem] in result.stdout + result.stderr
    if problem == "lock":
        assert "baseline_lock.json" in result.stdout + result.stderr
        assert "Traceback" not in result.stdout + result.stderr


def test_published_bundles_have_supported_cohorts_and_committed_locks(tmp_path):
    """Evaluate the real CMake manifest and check its qualification prerequisites."""
    script = tmp_path / "publication.cmake"
    script.write_text(
        f'include("{(_PLATFORM / "cmake/PublishedGpuReferences.cmake").as_posix()}")\n'
        "foreach(operation sdpa conv)\n"
        '  string(TOUPPER "${operation}" upper)\n'
        f'  file(WRITE "{tmp_path.as_posix()}/${{operation}}.txt" '
        '"${ROCKE_PUBLISHED_${upper}_ARCHITECTURES}")\n'
        "endforeach()\n"
    )
    subprocess.run(
        [_CMAKE, "-P", str(script)], check=True, capture_output=True, text=True
    )
    published = {
        operation: list(
            filter(None, (tmp_path / f"{operation}.txt").read_text().split(";"))
        )
        for operation in ("sdpa", "conv")
    }
    tests = _LIBRARY / "tests"
    for operation, architectures in published.items():
        assert len(architectures) == len(set(architectures))
        root = tests / f"{operation}_reference/architectures"
        supported = json.loads((root / "registry.json").read_text())
        assert set(architectures) <= set(supported)
        for architecture in architectures:
            lock = json.loads((root / architecture / "baseline_lock.json").read_text())
            assert lock["schema"] == 2
            assert len(bytes.fromhex(lock["baseline_revision"])) == 20
            assert len(bytes.fromhex(lock["manifest_sha256"])) == 32
