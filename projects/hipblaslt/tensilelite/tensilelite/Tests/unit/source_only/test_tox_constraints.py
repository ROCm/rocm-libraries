# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Sanity-check installed dependencies against the generated requirements."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
from packaging.requirements import Requirement

pytestmark = pytest.mark.unit


def test_installed_dependencies_match_constraints():
    project = Path(__file__).resolve().parents[4]
    for line in (project / "requirements-dev-common.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            installed = version(requirement.name)
        except PackageNotFoundError:
            # Each test environment installs a subset of the development requirements.
            continue
        assert installed in requirement.specifier, (
            f"Installed {requirement.name}=={installed}; expected {requirement}"
        )
