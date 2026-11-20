# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from tensilelite import LibraryIO


@pytest.mark.unit
def test_common_yaml_files_require_opt_in_for_serial_codegen():
    common_dir = Path(__file__).resolve().parents[1] / "common"
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
        test_parameters = config.get("TestParameters") or {}
        if (
            global_parameters.get("CpuThreads") == 1
            and test_parameters.get("allow_serial_codegen") is not True
        ):
            offenders.append(path.relative_to(common_dir).as_posix())

    if offenders:
        pytest.fail(
            "Remove GlobalParameters.CpuThreads: 1 to use the default codegen pool, "
            "or set TestParameters.allow_serial_codegen: true if serial codegen "
            "is required. Affected common test YAMLs:\n  " + "\n  ".join(offenders),
            pytrace=False,
        )
