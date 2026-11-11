# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The test artifact ships the tensilelite wheel, not an unpacked copy of it.

TheRock runs the installed tests with the artifact root on PYTHONPATH. An
unpacked tensilelite package there would shadow the installed wheel, so the
tests would never exercise what ships.
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_HIPBLASLT_ROOT = Path(__file__).resolve().parents[5]


def _artifact_install_block() -> str:
    cmake = (_HIPBLASLT_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    return cmake.split("if(HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS AND NOT WIN32)", 1)[1].split(
        "\nendif()", 1
    )[0]


def test_artifact_installs_the_wheel_and_tests_but_does_not_unpack_the_wheel():
    block = _artifact_install_block()

    assert 'FILES "${_tensilelite_canonical_wheel}"' in block
    assert 'DESTINATION "${_tensilelite_install_dest}/wheels"' in block
    assert 'DESTINATION "${_tensilelite_install_dest}/tensilelite/Tests"' in block
    assert "install_release_wheels.py" not in block
    # rocisa, its stinkytofu adaptor and stinkytofu stay where TheRock's runner
    # expects them.
    assert 'DESTINATION "${_tensilelite_install_dest}/rocisa"' in block
    assert 'DESTINATION "${_tensilelite_install_dest}/rocisa_stinkytofu_adaptor"' in block
    assert 'DESTINATION "${_tensilelite_install_dest}/stinkytofu"' in block


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
