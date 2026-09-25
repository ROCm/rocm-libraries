# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

_TENSILELITE_ROOT = Path(__file__).resolve().parents[4]
_REPO_ROOT = _TENSILELITE_ROOT.parents[2]


@pytest.mark.parametrize("relative_path", ["tox.ini", "test_categories.yaml"])
def test_test_commands_do_not_forward_removed_prebuilt_client_option(relative_path):
    contents = (_TENSILELITE_ROOT / relative_path).read_text(encoding="utf-8")

    assert "--prebuilt-client" not in contents


def test_conftest_does_not_expose_removed_source_client_helpers():
    contents = (
        _TENSILELITE_ROOT / "tensilelite" / "Tests" / "conftest.py"
    ).read_text(encoding="utf-8")

    assert "--no-common-build" not in contents
    assert "tensile_script_path" not in contents


def test_rocisa_only_environment_does_not_build_the_tensilelite_client():
    contents = (_TENSILELITE_ROOT / "tox.ini").read_text(encoding="utf-8")
    rocisa_environment = contents.split("[testenv:rocisa]", 1)[1].split(
        "\n[testenv:", 1
    )[0]

    assert "invoke build-client" not in rocisa_environment


def test_workflows_and_docs_do_not_use_retired_source_entrypoints():
    paths = (
        _REPO_ROOT / ".github/workflows/gfx1250_hipblaslt_tensilelite_test.yml",
        _TENSILELITE_ROOT / "README.md",
        _TENSILELITE_ROOT / "AGENTS.md",
        _TENSILELITE_ROOT / "AGENTS_reference.md",
        _TENSILELITE_ROOT / "tensilelite/Utilities/tensile_generator/README.md",
        _TENSILELITE_ROOT / "tests/configs/SolutionLibraries/readme",
    )

    for path in paths:
        contents = path.read_text(encoding="utf-8")
        assert "tensilelite/bin/Tensile" not in contents, path
        assert "--prebuilt-client" not in contents, path

    workflow = paths[0].read_text(encoding="utf-8")
    assert "python -m tensilelite_configure_client" in workflow
    assert "--ensure-client" in workflow
