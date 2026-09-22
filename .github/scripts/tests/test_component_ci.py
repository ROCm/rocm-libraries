"""Tests for component-CI path selection."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from component_ci import COMPONENTS, detect_changed_components


def test_component_source_change_runs_transitive_consumers():
    changed = detect_changed_components({"shared/stinkytofu/lib/example.cpp"})

    assert changed == {
        "stinkytofu": True,
        "rocisa": True,
        "geko": False,
        "miopen": False,
        "tensilelite_coverage": True,
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


def test_component_workflow_changes_run_their_component():
    expected = {
        ".github/workflows/component-ci-stinkytofu.yml": "stinkytofu",
        ".github/workflows/component-ci-rocisa.yml": "rocisa",
        ".github/workflows/component-ci-geko.yml": "geko",
        ".github/workflows/component-ci-miopen.yml": "miopen",
        ".github/workflows/component-ci-tensilelite-coverage.yml": "tensilelite_coverage",
    }

    for workflow, component in expected.items():
        changed = detect_changed_components({workflow})
        assert {name for name, selected in changed.items() if selected} == {component}


def test_in_tree_dependency_changes_run_their_consumers():
    expected = {
        "cmake/modules/default_amdclang.cmake": {
            "stinkytofu",
            "rocisa",
            "tensilelite_coverage",
        },
        "projects/hipblaslt/tensilelite/rocisa/rocisa/include/enum.hpp": {
            "stinkytofu",
            "rocisa",
            "tensilelite_coverage",
        },
        "shared/origami/include/origami/origami.hpp": {
            "rocisa",
            "tensilelite_coverage",
        },
        "shared/stinkytofu/include/stinkytofu/ir/DumpStinkyModulePass.hpp": {
            "stinkytofu",
            "rocisa",
            "tensilelite_coverage",
        },
        "projects/hipblaslt/tasks.py": {"tensilelite_coverage"},
        "projects/hipblaslt/clients/scripts/performance/specs.py": {
            "tensilelite_coverage"
        },
        ".dvc/config": {"miopen"},
        ".dvcignore": {"miopen"},
        ".gitmodules": {"miopen"},
        "shared/ctest/TestCategories.cmake": {"miopen"},
    }

    for dependency, consumers in expected.items():
        changed = detect_changed_components({dependency})
        assert {name for name, selected in changed.items() if selected} == consumers
