# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import importlib.util
import io
import os
import re
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


@pytest.mark.parametrize("verify_reuse,verify_upload", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("scheduled", [False, True])
def test_selected_config_forwards_verification_to_child_and_restores_parent(
    config_runner, monkeypatch, tmp_path, scheduled, verify_reuse, verify_upload
):
    # A real subprocess verifies the environment reaching the client runner.
    package = tmp_path / "tensilelite"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "run.py").write_text(
        "import os\n"
        "def main(args):\n"
        f"    assert os.environ['TENSILELITE_VERIFY_REFERENCE_REUSE'] == '{int(verify_reuse)}'\n"
        f"    assert os.environ['TENSILELITE_VERIFY_UPLOAD_SKIP'] == '{int(verify_upload)}'\n"
        "    print('REFERENCE_REUSE_VERIFIED:2')\n"
        "    print('Verified 3 skipped GPU input uploads against the host data.')\n"
        "    print('Verified 4 skipped GPU batched-pointer uploads.')\n"
        "    if 'TENSILE_PERSISTENT_FIXED_GRID' in os.environ:\n"
        "        print(\"TensileLite::StreamK LAUNCH SUMMARY kernel='test'\\n\"\n"
        "              ' physicalPartialSlots = 0\\n'\n"
        "              ' fallback = insufficient-workspace')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TENSILELITE_VERIFY_REFERENCE_REUSE", "0")
    monkeypatch.setenv("TENSILELITE_VERIFY_UPLOAD_SKIP", "0")
    parameters = {"verify_reference_reuse": verify_reuse, "verify_upload_skip": verify_upload}
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
    assert os.environ["TENSILELITE_VERIFY_UPLOAD_SKIP"] == "0"
    log = output / ("streamk-schedule-0.log" if scheduled else "reuse-verification.log")
    assert "REFERENCE_REUSE_VERIFIED:2" in log.read_text()


def _write_client_artifact(config_runner, config, tmp_path):
    artifact = tmp_path / (config_runner.artifact_name_for_config(str(config)) + ".tar.gz")
    with tarfile.open(artifact, "w:gz") as tar:
        directory = tarfile.TarInfo("output")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        tar.addfile(directory)
        tar.addfile(tarfile.TarInfo("output/caches/test/source/library/test.co"))


@pytest.mark.parametrize("scheduled", [False, True])
def test_client_env_reaches_the_client_and_not_the_parent(
    config_runner, monkeypatch, tmp_path, scheduled
):
    package = tmp_path / "tensilelite"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "run.py").write_text(
        "import os\n"
        "def main(args):\n"
        "    assert os.environ['TENSILE_STREAMK_TILES'] == '3'\n"
        "    assert os.environ['TENSILE_STREAMK_SPLIT'] == '4'\n"
        "    print('CLIENT_ENV_SEEN')\n"
        "    if 'TENSILE_PERSISTENT_FIXED_GRID' in os.environ:\n"
        "        print(\"TensileLite::StreamK LAUNCH SUMMARY kernel='test'\\n\"\n"
        "              ' physicalPartialSlots = 0\\n'\n"
        "              ' fallback = insufficient-workspace')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSILE_STREAMK_TILES", raising=False)
    monkeypatch.delenv("TENSILE_STREAMK_SPLIT", raising=False)
    parameters = {"client_env": {"TENSILE_STREAMK_TILES": 3, "TENSILE_STREAMK_SPLIT": "4"}}
    if scheduled:
        parameters["streamk_schedules"] = [{"reduction": "whole", "grid": 1}]
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"TestParameters": parameters}))
    _write_client_artifact(config_runner, config, tmp_path)
    output = tmp_path / "output"
    config_runner._run(str(config), str(output), str(tmp_path), [])
    assert "TENSILE_STREAMK_TILES" not in os.environ
    assert "TENSILE_STREAMK_SPLIT" not in os.environ
    log = output / ("streamk-schedule-0.log" if scheduled else "client.log")
    assert "CLIENT_ENV_SEEN" in log.read_text()


def test_streamk_launches_enable_the_summaries_in_the_client(config_runner, monkeypatch, tmp_path):
    package = tmp_path / "tensilelite"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "run.py").write_text(
        "import os\n"
        "def main(args):\n"
        "    if os.environ.get('TENSILE_DB') == '0x200000':\n"
        "        print(\"TensileLite::StreamK LAUNCH SUMMARY  kernel='test'\\n\"\n"
        "              '  tiles:\\n'\n"
        "              '    tiles      = 12\\n'\n"
        "              '    skTiles    = 12\\n'\n"
        "              '    skSplit    = 8')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSILE_DB", raising=False)
    parameters = {"streamk_launches": [{"tiles": 12, "skTiles": 12, "skSplit": ">1"}]}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"TestParameters": parameters}))
    _write_client_artifact(config_runner, config, tmp_path)
    config_runner._run(str(config), str(tmp_path / "output"), str(tmp_path), [])
    assert "TENSILE_DB" not in os.environ


_SUMMARIES = (
    "TensileLite::StreamK LAUNCH SUMMARY  kernel='a'\n"
    "  tiles:\n    tiles      = 256\n    skTiles    = 100\n    skSplit    = 16\n"
    "noise from the client\n"
    "TensileLite::StreamK LAUNCH SUMMARY  kernel='b'\n"
    "  tiles:\n    tiles      = 12\n    skTiles    = 12\n    skSplit    = 16\n"
)


def _client_log(text):
    log = io.StringIO(text)
    log.name = "client.log"
    return log


def test_streamk_launches_accept_matching_summaries(config_runner):
    config_runner._require_streamk_launches(_client_log(_SUMMARIES), [
        {"tiles": 256, "skTiles": 100, "skSplit": 16},
        {"tiles": 12, "skTiles": 12, "skSplit": ">1"},
    ])


@pytest.mark.parametrize("expectation,message", [
    ({"tiles": 256, "skTiles": 0}, "a: 256 tiles launched with skTiles = 100, expected 0"),
    ({"tiles": 12, "skSplit": ">16"}, "b: 12 tiles launched with skSplit = 16, expected >16"),
    ({"tiles": 12, "parallelSplits": 2}, "parallelSplits = None"),
    ({"tiles": 64, "skTiles": 0}, "No StreamK launch of 64 tiles"),
])
def test_streamk_launches_reject_other_summaries(config_runner, expectation, message):
    with pytest.raises(AssertionError, match=re.escape(message)):
        config_runner._require_streamk_launches(_client_log(_SUMMARIES), [expectation])


def test_streamk_launches_reject_a_mismatch_in_any_launch(config_runner):
    other = _SUMMARIES.replace("kernel='b'", "kernel='c'").replace("= 12\n    skSplit", "= 0\n    skSplit")
    with pytest.raises(AssertionError, match="c: 12 tiles launched with skTiles = 0"):
        config_runner._require_streamk_launches(_client_log(_SUMMARIES + other),
                                                [{"tiles": 12, "skTiles": 12}])


@pytest.mark.parametrize("output", ["", "REFERENCE_REUSE_VERIFIED:0\n", "No verification\n"])
def test_selected_config_rejects_absent_or_zero_verification(config_runner, output):
    with pytest.raises(AssertionError, match="No CPU reference reuse verification"):
        config_runner._require_verification(io.StringIO(output), True, False)


def test_verification_counts_multiple_client_runs(config_runner):
    config_runner._require_verification(
        io.StringIO("REFERENCE_REUSE_VERIFIED:0\nREFERENCE_REUSE_VERIFIED:3\n"), True, False
    )


@pytest.mark.parametrize("kind", ["input", "batched-pointer"])
@pytest.mark.parametrize("count", [None, 0])
def test_selected_config_requires_each_upload_verification(config_runner, kind, count):
    reports = {
        "input": "Verified 2 skipped GPU input uploads against the host data.\n",
        "batched-pointer": "Verified 3 skipped GPU batched-pointer uploads.\n",
    }
    reports[kind] = "" if count is None else f"Verified 0 skipped GPU {kind} uploads.\n"
    with pytest.raises(AssertionError, match=f"No GPU {kind} upload-skip verification"):
        config_runner._require_verification(io.StringIO("".join(reports.values())), False, True)


def test_upload_verification_counts_multiple_client_runs(config_runner):
    config_runner._require_verification(io.StringIO(
        "Verified 0 skipped GPU input uploads against the host data.\n"
        "Verified 2 skipped GPU input uploads against the host data.\n"
        "Verified 0 skipped GPU batched-pointer uploads.\n"
        "Verified 3 skipped GPU batched-pointer uploads.\n"), False, True)
