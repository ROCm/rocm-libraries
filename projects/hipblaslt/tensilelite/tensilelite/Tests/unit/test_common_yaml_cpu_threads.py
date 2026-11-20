# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
import yaml

from tensilelite import LibraryIO


pytestmark = pytest.mark.unit


def _cpu_threads_violations(common_dir):
    yaml_paths = sorted(
        path for pattern in ("*.yaml", "*.yml") for path in common_dir.rglob(pattern)
    )
    assert yaml_paths, f"No common test YAMLs found in {common_dir}"

    offenders = []
    for path in yaml_paths:
        config = LibraryIO.readYAML(path)
        if not isinstance(config, dict):
            continue
        global_parameters = config.get("GlobalParameters") or {}
        cpu_threads = global_parameters.get("CpuThreads")
        if type(cpu_threads) is not int:
            continue
        test_parameters = config.get("TestParameters") or {}
        # Zero and values below -1 behave differently when progress reporting is off.
        if cpu_threads == 0 or cpu_threads < -1 or (
            cpu_threads == 1 and test_parameters.get("allow_serial_codegen") is not True
        ):
            offenders.append((path.relative_to(common_dir).as_posix(), cpu_threads))
    return offenders


def test_common_yaml_files_require_opt_in_for_serial_codegen():
    common_dir = Path(__file__).resolve().parents[1] / "common"
    offenders = _cpu_threads_violations(common_dir)
    if offenders:
        paths_and_values = "\n".join(f"  {path} (CpuThreads: {value})" for path, value in offenders)
        pytest.fail(
            "Invalid GlobalParameters.CpuThreads values in common test YAMLs.\n"
            "Use -1 or omit CpuThreads for the default codegen pool. "
            "For intentional serial codegen, use CpuThreads: 1 with "
            f"TestParameters.allow_serial_codegen: true.\n{paths_and_values}",
            pytrace=False,
        )


@pytest.mark.parametrize(
    ("cpu_threads", "opt_in", "is_violation"),
    [
        pytest.param(None, None, False, id="omitted"),
        pytest.param(-1, None, False, id="default-pool"),
        pytest.param(2, None, False, id="parallel"),
        pytest.param(1, None, True, id="serial-without-opt-in"),
        pytest.param(1, True, False, id="serial-with-opt-in"),
        pytest.param(1, False, True, id="serial-with-false-flag"),
        pytest.param(1, "true", True, id="serial-with-string-flag"),
        pytest.param(0, None, True, id="zero"),
        pytest.param(0, True, True, id="zero-with-opt-in"),
        pytest.param(-2, None, True, id="below-minus-one"),
        pytest.param(-2, True, True, id="below-minus-one-with-opt-in"),
    ],
)
def test_common_yaml_cpu_threads_policy(tmp_path, cpu_threads, opt_in, is_violation):
    config = {"GlobalParameters": {}}
    if cpu_threads is not None:
        config["GlobalParameters"]["CpuThreads"] = cpu_threads
    if opt_in is not None:
        config["TestParameters"] = {"allow_serial_codegen": opt_in}
    yaml_path = tmp_path / "gemm" / "example.yml"
    yaml_path.parent.mkdir()
    yaml_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    expected = [("gemm/example.yml", cpu_threads)] if is_violation else []
    assert _cpu_threads_violations(tmp_path) == expected
