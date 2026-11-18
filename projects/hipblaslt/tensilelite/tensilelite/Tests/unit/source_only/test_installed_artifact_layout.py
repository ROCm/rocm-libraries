# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The test artifact ships the tensilelite wheel, not an unpacked copy of it.

TheRock runs the installed tests with the artifact root on PYTHONPATH. An
unpacked tensilelite package there would shadow the installed wheel, so the
tests would never exercise what ships.
"""

import os
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_HIPBLASLT_ROOT = Path(__file__).resolve().parents[5]


def _artifact_install_block() -> str:
    cmake = (_HIPBLASLT_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    return cmake.split(
        "if(HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS AND NOT WIN32)", 1
    )[1].split("\nendif()", 1)[0]


def test_artifact_install_uses_latest_manifest_without_unpacking(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    build = tmp_path / "build"
    publication = tmp_path / "publication"
    publication.mkdir()
    manifest = publication / ".published"
    (source / "probe.c").write_text("int probe(void) { return 0; }\n")
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(artifact_install C)\ninclude(GNUInstallDirs)\n"
        "add_library(_rocisa SHARED probe.c)\n"
        f'set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}")\n'
        f'set(_tensilelite_src "{(_HIPBLASLT_ROOT / "tensilelite").as_posix()}")\n'
        f'set(TENSILELITE_CANONICAL_WHEEL_MANIFEST "{manifest.as_posix()}")\n'
        'set(_tensilelite_install_dest "share/hipblaslt/tensilelite")\n'
        + _artifact_install_block()
    )
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build)], check=True, capture_output=True
    )
    subprocess.run(["cmake", "--build", str(build)], check=True, capture_output=True)
    install = tmp_path / "install"
    artifact = install / "share/hipblaslt/tensilelite"
    for version in ("5.0.0+gfirst", "5.0.0+gsecond"):
        wheel = publication / f"tensilelite-{version}-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("tensilelite/__init__.py", "# must stay in wheel\n")
            archive.writestr("_tensilelite_client_binding.py", "# must stay in wheel\n")
        manifest.write_text(
            json.dumps(
                {"wheel": wheel.name, "metadata": {"distribution_version": version}}
            )
        )
        subprocess.run(
            [
                "cmake",
                "--install",
                str(build),
                "--prefix",
                str(install),
                "--component",
                "tests",
            ],
            check=True,
            capture_output=True,
        )
        assert list((artifact / "wheels").glob("*.whl")) == [
            artifact / "wheels" / wheel.name
        ]
        assert (artifact / "wheels" / wheel.name).read_bytes() == wheel.read_bytes()
        assert not (artifact / "tensilelite/__init__.py").exists()
        assert not (artifact / "_tensilelite_client_binding.py").exists()
        assert (artifact / "tensilelite/Tests/conftest.py").is_file()
        assert not (artifact / "tensilelite/Tests/unit/source_only").exists()


@pytest.mark.skipif(
    not os.environ.get("TENSILELITE_INSTALLED_ARTIFACT_ROOT"),
    reason="set TENSILELITE_INSTALLED_ARTIFACT_ROOT to an installed share/hipblaslt/tensilelite",
)
def test_installed_artifact_root_has_no_tensilelite_package():
    root = Path(os.environ["TENSILELITE_INSTALLED_ARTIFACT_ROOT"])

    assert not (root / "tensilelite" / "__init__.py").exists()
    assert not list(root.glob("tensilelite-*.dist-info"))
    assert not (root / "_tensilelite_client_binding.py").exists()
    assert len(list((root / "wheels").glob("tensilelite-*-py3-none-any.whl"))) == 1
    assert (root / "tensilelite" / "Tests" / "conftest.py").is_file()
    assert not (root / "tensilelite" / "Tests" / "unit" / "source_only").exists()
