# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Subtile beta epilogue must mask the lanes past N around every C load."""

from pathlib import Path

import pytest
import yaml

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit


def _emit(tmp_path, group_load_store, beta):
    seed = (Path(__file__).parent / "characterization/_codegen/data/test_data/"
            "_designed/gfx950/mx_bias_act_gsu.yaml")
    config = yaml.safe_load(seed.read_text())
    problem, params = config["BenchmarkProblems"][0]
    problem.update(UseBeta=beta)
    for option in params["ForkParameters"]:
        if "MatrixInstruction" in option:
            option["MatrixInstruction"] = [[16, 16, 128, 1, 1, 2, 2, 2, 2]]
        if "GlobalSplitU" in option:
            option["GlobalSplitU"] = [1]
        if "StreamK" in option:
            option["StreamK"] = [0]
    params["ForkParameters"].append({"GroupLoadStore": [group_load_store]})
    config["GlobalParameters"]["CpuThreads"] = 1
    path = tmp_path / "c_load.yaml"
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(path, limit=1, arch="gfx950")
    assert len(results) == 1
    _, assembly, error = results[0]
    assert error == 0
    return assembly.splitlines()


def _is_edge_load(lines, index):
    # The edge path clips OOB lanes through the address VGPR instead of exec.
    return any("LDC clip" in line for line in lines[max(0, index - 4):index])


def _masked_sites(lines):
    return [i for i, line in enumerate(lines) if "mask C load lanes" in line]


@pytest.mark.parametrize("group_load_store", [False, True])
def test_masked_c_loads_are_wrapped(group_load_store, tmp_path):
    lines = _emit(tmp_path, group_load_store, True)
    sites = _masked_sites(lines)
    assert sites
    for site in sites:
        load = next(i for i in range(site + 1, len(lines)) if "// load C" in lines[i])
        # Only scalar row-advance code may sit between the mask and the load.
        assert all(not line.startswith(("v_", "ds_", "buffer_")) for line in lines[site + 1:load])
        assert lines[load + 1].startswith("s_mov_b64 exec, -1"), lines[load]


def test_no_unmasked_subtile_c_load(tmp_path):
    lines = _emit(tmp_path, False, True)
    for index, line in enumerate(lines):
        if "// load C" not in line or _is_edge_load(lines, index):
            continue
        previous = next(l for l in reversed(lines[:index]) if "exec" in l)
        assert "mask C load lanes" in previous, line


def test_no_exec_mask_without_beta(tmp_path):
    lines = _emit(tmp_path, False, False)
    assert not any("mask C load lanes" in line for line in lines)
