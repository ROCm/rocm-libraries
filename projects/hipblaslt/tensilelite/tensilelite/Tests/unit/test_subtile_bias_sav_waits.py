# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Subtile epilogue LDS results must be ready at their arithmetic consumers."""

from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import yaml

from config_harness import emit_kernels_from_config
from Tensile.Common import DataDirection
from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("bias,sav", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("multi_du", [False, True])
def test_interleaved_wait_preserves_unconsumed_loads(bias, sav, multi_du):
    writer = object.__new__(GlobalWriteBatchWriter)
    writer.kernel = {
        "DepthU": 256, "_DepthUA": 128 if multi_du else 256,
        "UseSubtileImpl": True, "GlobalSplitU": 1, "StreamK": 0,
        "_GlobalAccumulation": None, "GroupLoadStore": False,
        "ProblemType": {"UseScaleAlphaVec": sav, "UseScaleAB": ""},
    }
    writer.parentWriter = SimpleNamespace(states=SimpleNamespace(
        useBias=DataDirection.READ if bias else DataDirection.NONE,
        useGateResidual=False,
        asmCaps={"SeparateVscnt": False, "SeparateVMcnt": False},
    ))
    writer.beta = writer.loadE = False
    writer.biasLoadIssued = writer.scaleAlphaVecLoadIssued = [1, 2]
    writer.storesIssued = 0
    # Two elements, one independent LDS read per enabled feature per element.
    per_element = int(bias) + int(sav)
    total = 2 * per_element
    counter = [0, total]
    first = writer.globalStoreWait(0, counter, 0, total, True)
    second = writer.globalStoreWait(1, counter, 0, total, True)
    if multi_du or not per_element:
        # Multi-DU has already drained these reads before _emitAdd.
        assert first is None and second is None
    else:
        assert first.dscnt == per_element  # retain overlap with element 1
        assert second.dscnt == 0


def _registers(text):
    result = set()
    for lo, hi, single in re.findall(r"\bv(?:\[(\d+):(\d+)\]|(\d+)\b)", text):
        result.update(range(int(lo), int(hi) + 1) if lo else [int(single)])
    return result


def _check_lds_consumers(assembly):
    pending = []
    features = set()
    for line in assembly.splitlines():
        if line.startswith("ds_"):
            feature = re.search(r"// load (Bias|scaleAlpha)\b", line)
            registers = _registers(line.split(",", 1)[0]) if feature else set()
            pending.append(registers)
            if feature:
                assert registers, line
                features.add(feature[1])
        wait = re.match(r"s_waitcnt .*lgkmcnt\((\d+)\)", line)
        if re.match(r"s_waitcnt 0\b", line):
            pending = []
        elif wait:
            count = int(wait[1])
            pending = pending[-count:] if count else []
        if line.startswith("v_") and "," in line:
            sources = _registers(line.split(",", 1)[1].split("//", 1)[0])
            assert not any(sources & registers for registers in pending), line
    return features


@pytest.mark.parametrize("bias,sav", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("group_load_store", [False, True])
def test_single_du_bias_sav_consumers(bias, sav, group_load_store, tmp_path):
    seed = (Path(__file__).parent / "characterization/_codegen/data/test_data/"
            "_designed/gfx950/mx_bias_act_gsu.yaml")
    config = yaml.safe_load(seed.read_text())
    problem, params = config["BenchmarkProblems"][0]
    problem.update(UseBias=int(bias), UseScaleAlphaVec=int(sav))
    for option in params["ForkParameters"]:
        if "MatrixInstruction" in option:
            option["MatrixInstruction"] = [[16, 16, 128, 1, 1, 2, 2, 2, 2]]
        if "GlobalSplitU" in option:
            option["GlobalSplitU"] = [1]
        if "StreamK" in option:
            option["StreamK"] = [0]
    params["ForkParameters"].append({"GroupLoadStore": [group_load_store]})
    config["GlobalParameters"]["CpuThreads"] = 1
    path = tmp_path / "bias_sav.yaml"
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(path, limit=1, arch="gfx950")
    assert len(results) == 1
    _, assembly, error = results[0]
    assert error == 0
    expected = ({"Bias"} if bias else set()) | ({"scaleAlpha"} if sav else set())
    assert _check_lds_consumers(assembly) == expected
