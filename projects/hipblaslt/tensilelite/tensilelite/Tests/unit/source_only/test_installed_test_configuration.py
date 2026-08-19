# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

_TENSILELITE_ROOT = Path(__file__).resolve().parents[4]


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
