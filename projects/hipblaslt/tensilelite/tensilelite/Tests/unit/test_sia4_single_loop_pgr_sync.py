# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""TDM main-loop PGR synchronization on gfx1250.

SIA4 must synchronize the one-tile PGR escape without disturbing PGR2.

A TDM main loop with PrefetchLocalRead=0 must wait for its tensor loads: F8
with MI K=64 and DepthU=64 has one iteration per unroll, so PLR collapses to 0
and each loop body reads LDS right after its opening barrier, from a tile a
tensor_load_to_lds issued in an earlier body. Without s_wait_tensorcnt before
that barrier, from the second main-loop iteration on, ds_load reads LDS the
TDM has not finished writing.
"""

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


def _tdm_plr0_config(prefetch_global_read):
    fork_parameters = [
        {"MatrixInstruction": [[16, 16, 64, 1, 1, 2, 2, 2, 2]]},
        {"DepthU": [64]},
        {"WavefrontSize": [32]},
        {"TDMInst": [3]},
        {"LDSTrInst": [True]},
        {"PrefetchGlobalRead": [prefetch_global_read]},
        {"ScheduleIterAlg": [3]},
    ]
    if prefetch_global_read == 1:
        # LDSTrInst with PGR1 is only supported with one LDS buffer and no pointer swap.
        fork_parameters += [{"1LDSBuffer": [1]}, {"ExpandPointerSwap": [False]}]
    return {
        "GlobalParameters": {"Architecture": "gfx1250", "PrintLevel": 0},
        "BenchmarkProblems": [[
            {
                "OperationType": "GEMM",
                "DataType": "F8",
                "DestDataType": "s",
                "ComputeDataType": "s",
                "HighPrecisionAccumulate": True,
                "TransposeA": False,
                "TransposeB": False,
                "UseBeta": True,
                "Batched": True,
            },
            {
                "InitialSolutionParameters": None,
                "BenchmarkCommonParameters": [{"KernelLanguage": ["Assembly"]}],
                "ForkParameters": fork_parameters,
                "BenchmarkFinalParameters": [{"ProblemSizes": [{"Exact": [64, 64, 1, 256]}]}],
            },
        ]],
    }


@pytest.mark.parametrize("prefetch_global_read", [1, 2], ids=["pgr1", "pgr2"])
def test_tdm_plr0_loop_body_waits_for_tensor_load(tmp_path, prefetch_global_read):
    path = tmp_path / ("tdm_plr0_pgr%u.yaml" % prefetch_global_read)
    path.write_text(yaml.safe_dump(_tdm_plr0_config(prefetch_global_read)))
    results = emit_kernels_from_config(path, limit=1, arch="gfx1250")

    assert len(results) == 1
    _, assembly, error = results[0]
    assert error == 0
    iters = re.findall(r"^/\* iter (\d+)", assembly, re.M)
    assert iters and set(iters) == {"0"}, \
        "config no longer has LoopIters=1, so PrefetchLocalRead is no longer 0"
    assert "tensor_load_to_lds" in assembly

    loop_tops = re.findall(
        r"/\* Unrolled Loop \d+/\d+ - Begin\s+\*/\n"
        r"/\*+/\n"
        r"((?:.*\n)*?)"
        r"/\* Begin Each Unroll",
        assembly,
    )
    assert len(loop_tops) == 2, "expected the first-iteration and label_LoopBeginL loop copies"
    for loop_top in loop_tops:
        assert "s_barrier_signal" in loop_top
        assert re.search(r"s_wait_tensorcnt 0\b(?:.*\n)*?s_barrier_signal", loop_top), \
            "main-loop body syncs before its tensor loads have landed in LDS:\n" + loop_top
