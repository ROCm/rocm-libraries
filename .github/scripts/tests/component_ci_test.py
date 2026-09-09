# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import sys

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from component_ci import COMPONENTS, detect_changed_components


def test_stinkytofu_source_triggers_stinkytofu_and_rocisa() -> None:
    changed = detect_changed_components({"shared/stinkytofu/src/Pipeline.cpp"})

    assert changed["stinkytofu"] is True
    assert changed["rocisa"] is True
    assert changed["geko"] is False
    assert changed["miopen"] is False
    assert changed["tensilelite_coverage"] is False


def test_stinkytofu_workflow_triggers_only_stinkytofu() -> None:
    changed = detect_changed_components(
        {".github/workflows/component-ci-stinkytofu.yml"}
    )

    assert changed["stinkytofu"] is True
    assert changed["rocisa"] is False
    assert changed["geko"] is False
    assert changed["miopen"] is False
    assert changed["tensilelite_coverage"] is False


def test_rocisa_workflow_triggers_only_rocisa() -> None:
    changed = detect_changed_components({".github/workflows/component-ci-rocisa.yml"})

    assert changed["stinkytofu"] is False
    assert changed["rocisa"] is True
    assert changed["geko"] is False
    assert changed["miopen"] is False
    assert changed["tensilelite_coverage"] is False


def test_shared_component_ci_infrastructure_triggers_everything() -> None:
    changed = detect_changed_components({".github/workflows/component-ci.yml"})

    assert changed == {component: True for component in COMPONENTS}
