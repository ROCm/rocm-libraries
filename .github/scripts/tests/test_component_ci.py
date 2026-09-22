"""Tests for component-CI path selection."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from component_ci import COMPONENTS, detect_changed_components


def test_component_source_change_is_scoped():
    changed = detect_changed_components({"shared/stinkytofu/lib/example.cpp"})

    assert changed == {
        "stinkytofu": True,
        "rocisa": True,
        "geko": False,
        "miopen": False,
        "tensilelite_coverage": False,
    }


def test_ci_environment_change_runs_every_component():
    changed = detect_changed_components({".github/actions/ci-env/action.yml"})

    assert changed == {name: True for name in COMPONENTS}


def test_shared_action_changes_run_their_consumers():
    expected = {
        ".github/actions/pip-install-test/action.yml": {"stinkytofu", "rocisa", "geko"},
        ".github/actions/setup-rocm-linux/action.yml": {
            "stinkytofu",
            "rocisa",
            "geko",
            "tensilelite_coverage",
        },
        ".github/actions/setup-rocm-windows/action.yml": {
            "stinkytofu",
            "rocisa",
            "geko",
        },
        ".github/actions/setup-llvm/action.yml": {"miopen"},
    }

    for action, consumers in expected.items():
        changed = detect_changed_components({action})
        assert {name for name, selected in changed.items() if selected} == consumers
