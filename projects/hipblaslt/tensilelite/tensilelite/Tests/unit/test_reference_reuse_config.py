# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import importlib.util
import io
import os
from pathlib import Path
import sys
import tarfile
import types

import pytest
import yaml

pytestmark = pytest.mark.unit


@pytest.fixture
def config_runner(monkeypatch):
    common = Path(__file__).resolve().parents[1] / "common"
    monkeypatch.syspath_prepend(str(common))
    # The runner's scheduling/forwarding behavior does not need the generator.
    package = types.ModuleType("tensilelite")
    package.run = types.SimpleNamespace(main=lambda args: None)
    monkeypatch.setitem(sys.modules, "tensilelite", package)
    spec = importlib.util.spec_from_file_location(
        "reference_reuse_config_runner", common / "test_config_run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("scheduled", [False, True])
def test_selected_config_forwards_verification_to_child_and_restores_parent(
    config_runner, monkeypatch, tmp_path, scheduled
):
    # A real subprocess verifies the environment reaching the client runner.
    package = tmp_path / "tensilelite"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "run.py").write_text(
        "import os\n"
        "def main(args):\n"
        "    assert os.environ['TENSILELITE_VERIFY_REFERENCE_REUSE'] == '1'\n"
        "    print('REFERENCE_REUSE_VERIFIED:2')\n"
        "    if 'TENSILE_PERSISTENT_FIXED_GRID' in os.environ:\n"
        "        print(\"TensileLite::StreamK LAUNCH SUMMARY kernel='test'\\n\"\n"
        "              ' physicalPartialSlots = 0\\n'\n"
        "              ' fallback = insufficient-workspace')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TENSILELITE_VERIFY_REFERENCE_REUSE", "0")
    parameters = {"verify_reference_reuse": True}
    if scheduled:
        parameters["streamk_schedules"] = [{"reduction": "whole", "grid": 1}]
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"TestParameters": parameters}))
    artifact = tmp_path / (config_runner.artifact_name_for_config(str(config)) + ".tar.gz")
    with tarfile.open(artifact, "w:gz") as tar:
        directory = tarfile.TarInfo("output")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        tar.addfile(directory)
        tar.addfile(tarfile.TarInfo("output/caches/test/source/library/test.co"))
    output = tmp_path / "output"
    config_runner._run(str(config), str(output), str(tmp_path), [])
    assert os.environ["TENSILELITE_VERIFY_REFERENCE_REUSE"] == "0"
    log = output / ("streamk-schedule-0.log" if scheduled else "reference-reuse.log")
    assert "REFERENCE_REUSE_VERIFIED:2" in log.read_text()


@pytest.mark.parametrize("output", ["", "REFERENCE_REUSE_VERIFIED:0\n", "No verification\n"])
def test_selected_config_rejects_absent_or_zero_verification(config_runner, output):
    with pytest.raises(AssertionError, match="No CPU reference reuse verification"):
        config_runner._require_verified_reuse(io.StringIO(output))


def test_verification_counts_multiple_client_runs(config_runner):
    config_runner._require_verified_reuse(
        io.StringIO("REFERENCE_REUSE_VERIFIED:0\nREFERENCE_REUSE_VERIFIED:3\n")
    )
