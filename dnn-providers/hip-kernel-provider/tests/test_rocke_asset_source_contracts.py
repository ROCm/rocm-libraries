# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The source asset CI entrypoint fails if its cohort is skipped or uncollected."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

PROVIDER = Path(__file__).resolve().parents[1]
RUNNER = PROVIDER / "tests/run_rocke_asset_source_contracts.py"


def test_both_source_superbuilds_execute_the_contracts_before_configuration():
    workflow = PROVIDER.parents[1] / ".github/workflows/hipdnn-superbuild-ci.yml"
    jobs = yaml.safe_load(workflow.read_text())["jobs"]
    checked = 0
    for name, job in jobs.items():
        steps = job.get("steps", [])
        names = [step.get("name") for step in steps]
        if "Configure superbuild" not in names:
            continue
        contract = names.index("Check rocKE source asset contracts")
        assert (
            names.index("Install Python dependencies")
            < contract
            < names.index("Configure superbuild")
        )
        assert steps[contract]["run"].strip() == (
            '"$VENV_PYTHON" -I dnn-providers/hip-kernel-provider/tests/run_rocke_asset_source_contracts.py'
        )
        if name == "hipdnn-linux-superbuild":
            assert steps[contract]["env"]["CMAKE_GENERATOR"] == "Ninja"
        checked += 1
    assert checked == 2


@pytest.mark.parametrize("case", ["pass", "skip", "empty", "missing", "failure"])
def test_source_asset_runner_does_not_hide_reduced_coverage(tmp_path, case):
    provider = tmp_path / "provider"
    scripts = provider / "tests"
    scripts.mkdir(parents=True)
    shutil.copy2(RUNNER, scripts / RUNNER.name)
    asset_test = provider / "rocke/platform/tests/tools/test_sdpa_asset_staging.py"
    asset_test.parent.mkdir(parents=True)
    asset_test.write_text("def test_asset():\n    assert True\n")
    (scripts / "test_rocke_test_assets.py").write_text(
        "def test_provider():\n    assert True\n"
    )
    (scripts / "test_rocke_asset_source_contracts.py").write_text(
        "def test_runner():\n    assert True\n"
    )
    if case == "skip":
        asset_test.write_text(
            "import pytest\ndef test_asset():\n    pytest.skip('missing sources')\n"
        )
    elif case == "empty":
        asset_test.write_text("# No tests collected.\n")
    elif case == "missing":
        asset_test.unlink()
    elif case == "failure":
        asset_test.write_text("def test_asset():\n    assert False\n")
    result = subprocess.run(
        [sys.executable, "-I", str(scripts / RUNNER.name)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (case == "pass"), result.stdout + result.stderr
    if case in ("skip", "empty"):
        assert "Source asset contract coverage failed:" in result.stdout
