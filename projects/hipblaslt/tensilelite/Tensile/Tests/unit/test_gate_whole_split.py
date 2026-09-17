# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Gate presence must select a complete store tree, including CLS resources."""

from pathlib import Path
import re

import pytest
import yaml

from config_harness import assert_assembles, emit_kernels_from_config, solutions_from_config

pytestmark = pytest.mark.unit

_CONFIG = Path(__file__).parents[1] / "common/sparse/gfx1250/spmm_gate.yaml"
_TREE_BOUNDARY = re.compile(
    r"^label_(?:GW_G([01])_.+?_Tree|GW_End)(?:_\d+)?:.*$", re.MULTILINE
)


@pytest.mark.parametrize("problem_index", [0, 1], ids=["fp8_multi_gate", "fp16_gate"])
@pytest.mark.parametrize("compact", [False, True], ids=["cls0", "cls1"])
def test_gate_whole_split(tmp_path, problem_index, compact):
    config = yaml.safe_load(_CONFIG.read_text())
    config["BenchmarkProblems"] = [config["BenchmarkProblems"][problem_index]]
    fork = config["BenchmarkProblems"][0][1]["ForkParameters"]
    for parameter in fork:
        if "CompactLoopStore" in parameter:
            parameter["CompactLoopStore"] = [compact]
        for hint in ("NonTemporalGate", "TemporalHintGate", "NonVolatileGate"):
            if hint in parameter:
                parameter[hint] = [1]
    config_path = tmp_path / "gate.yaml"
    config_path.write_text(yaml.safe_dump(config))

    solutions = solutions_from_config(config_path, arch="gfx1250", limit_solutions=1)
    assert len(solutions) == 1
    assert solutions[0]["CompactLoopStore"] == compact

    results = emit_kernels_from_config(config_path, arch="gfx1250", limit=1, canonical=False)
    assert len(results) == 1
    for name, source, error in results:
        assert error == 0
        (tmp_path / f"{name}.s").write_text(source)
        assert_assembles(source, name)
        assert "GateNullOne" not in source
        boundaries = list(_TREE_BOUNDARY.finditer(source))
        gate_paths = set()
        for index, boundary in enumerate(boundaries):
            gate = boundary.group(1)
            if gate is None:
                continue
            gate_paths.add(gate)
            assert index + 1 < len(boundaries), "store tree must end before GW_End"
            body = source[boundary.end():boundaries[index + 1].start()]
            if gate == "0":
                for forbidden in (
                    "sgprSrdGate", "sgprGateType", "sgprCLSGateRowInc",
                    "coutRowPtrGate", "load Gate", "GateResidual", "GateCvt",
                ):
                    assert forbidden not in body, f"G0 references {forbidden}"
            else:
                assert "sgprSrdGate" in body
                assert "v_pk_fma_f32" in body
                assert ("sgprCLSGateRowInc" in body) == compact
        assert gate_paths == {"0", "1"}
