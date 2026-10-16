# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SIA4 must synchronize the one-tile PGR escape without disturbing PGR2."""

from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import yaml

from config_harness import emit_kernels_from_config
from Tensile.KernelWriter import KernelWriter


pytestmark = pytest.mark.unit

_CONFIG = Path(__file__).parent / "test_data" / "fp4_SourceSwap_nonsquare_gfx1250.yaml"


def test_sia4_single_loop_sync_is_path_local(tmp_path):
    config = yaml.safe_load(_CONFIG.read_text())
    params = config["BenchmarkProblems"][0][1]["ForkParameters"]
    for parameter in params:
        if "MatrixInstruction" in parameter:
            parameter["MatrixInstruction"] = [parameter["MatrixInstruction"][0]]
        elif "ScheduleIterAlg" in parameter:
            parameter["ScheduleIterAlg"] = [4]
        elif "AssertSummationElementMultiple" in parameter:
            parameter["AssertSummationElementMultiple"] = [1]

    path = tmp_path / "sia4_single_loop.yaml"
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(path, limit=1, arch="gfx1250")

    assert len(results) == 1
    _, assembly, error = results[0]
    assert error == 0

    direct_edge = re.search(
        r"s_cbranch_scc0 label_AfterSingleLoopPGRSync_\d+.*\n"
        r"(?:.*\n)*?s_wait_tensorcnt 0.*\n"
        r"(?:.*\n)*?s_barrier_wait -1.*single-loop PGR bypass sync.*\n"
        r"s_branch label_toPGR1.*\n"
        r"label_AfterSingleLoopPGRSync_\d+:",
        assembly,
    )
    assert direct_edge, "the one-tile toPGR1 edge is not synchronized locally"

    normal_path = assembly[direct_edge.end():assembly.index("label_toPGR1:")]
    assert "auto token transition barrier" in normal_path, \
        "the one-tile escape disabled token planning on the normal PGR2 path"

    params.append({"PrefetchGL2": [1]})
    pgl_path = tmp_path / "sia4_single_loop_pgl1.yaml"
    pgl_path.write_text(yaml.safe_dump(config))
    pgl_results = emit_kernels_from_config(pgl_path, limit=1, arch="gfx1250")

    assert len(pgl_results) == 1
    _, pgl_assembly, pgl_error = pgl_results[0]
    assert pgl_error == 0
    assert "AfterSingleLoopPGRSync" not in pgl_assembly
    assert re.search(
        r"s_cbranch_scc1 label_toPGR1.*PGR=2 but only 1 loop, toPGR1",
        pgl_assembly,
    ), "PGL1 already fences the one-tile path and should retain the original branch"


def test_sia4_single_loop_sync_not_needed_after_prefetched_local_reads():
    writer = SimpleNamespace(
        states=SimpleNamespace(numItersPLR=1),
        _usesMainLoopTokenBarrierPlanner=lambda kernel: True,
    )
    kernel = {"PrefetchGL2": 0}

    assert not KernelWriter._needsSingleLoopPGRSync(writer, kernel)
