# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The CPU-only pip SDK smoke check is wired as documented."""

import configparser
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_TENSILELITE_ROOT = Path(__file__).resolve().parents[4]
_REPO_ROOT = _TENSILELITE_ROOT.parents[2]
_WORKFLOW = _REPO_ROOT / ".github/workflows/component-ci-tensilelite-pip-sdk-smoke.yml"


def test_tox_env_installs_only_the_newest_nightly_core_and_this_checkout():
    tox = configparser.ConfigParser(interpolation=None)
    tox.read(_TENSILELITE_ROOT / "tox.ini")
    env = tox["testenv:pip-sdk-smoke"]
    commands = env["commands"]

    assert env["skip_install"].strip() == "true"
    assert env["deps"].strip() == ""
    # Unpinned and core only: no rocm[...] meta-package, no device or libraries.
    assert (
        "--pre --index-url https://nightly.repo.amd.com/rocm/whl-next/ rocm-sdk-core" in commands
    )
    assert "rocm-sdk-core==" not in commands and "rocm[" not in commands
    assert "pip install {toxinidir}\n" in commands + "\n"
    assert "scripts/pip_sdk_smoke.py" in commands


def test_workflow_runs_on_cpu_daily_and_from_component_ci():
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    component_ci = (_REPO_ROOT / ".github/workflows/component-ci.yml").read_text(encoding="utf-8")
    codeowners = (_REPO_ROOT / ".github/CODEOWNERS").read_text(encoding="utf-8")

    assert "runs-on: ubuntu-24.04" in workflow
    assert "schedule:" in workflow and "workflow_call:" in workflow
    assert "tox -e pip-sdk-smoke" in workflow
    assert "uses: ./.github/workflows/component-ci-tensilelite-pip-sdk-smoke.yml" in component_ci
    assert "/.github/workflows/component-ci-tensilelite-pip-sdk-smoke.yml" in codeowners
