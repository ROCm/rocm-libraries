# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Single-wave TDM main loops must wait for the previous trip's tensor loads."""

import re
from types import SimpleNamespace

import pytest
import yaml

from config_harness import emit_kernels_from_config
from Tensile.KernelWriter import KernelWriter

pytestmark = pytest.mark.unit

_SPARSE_FORKS = {"TransposeLDS": [-1], "LDSTrInst": [False], "DirectToVgprSparseMetadata": [False],
                 "SourceSwap": [False], "LocalReadVectorWidth": [8]}


def _config(problem, mi, depth_u, prefetch_global_read, extra_forks):
    forks = {
        "MatrixInstruction": [mi],
        "DepthU": [depth_u],
        "TileProcessingStrategy": ["DataParallel"],
        "WorkAssignment": ["StaticGrid"],
        "ClusterDim": [[1, 1]],
        "TDMInst": [3],
        "WavefrontSize": [32],
        "WorkGroupMapping": [1],
        "GlobalSplitU": [0],
        "StaggerU": [0],
        "PrefetchGlobalRead": [prefetch_global_read],
        "PrefetchLocalRead": [1],
        "ScheduleIterAlg": [4],
        "1LDSBuffer": [0],
        "ExpandPointerSwap": [False],
        "StoreRemapVectorWidth": [0],
        "PrefetchAcrossPersistent": [0],
        **extra_forks,
    }
    problem_type = {"OperationType": "GEMM", "DataType": "h", "DestDataType": "s", "ComputeDataType": "s",
                    "HighPrecisionAccumulate": True, "TransposeA": True, "TransposeB": False,
                    "UseBeta": True, "Batched": True, **problem}
    return {
        "GlobalParameters": {"Architecture": "gfx1250", "PruneSparseMode": 0},
        "BenchmarkProblems": [[
            problem_type,
            {"BenchmarkCommonParameters": [{"KernelLanguage": ["Assembly"]}],
             "ForkParameters": [{key: value} for key, value in forks.items()],
             "BenchmarkFinalParameters": [{"ProblemSizes": [{"Exact": [32, 32, 1, 320]}]}]},
        ]],
    }


def _emit(tmp_path, config):
    path = tmp_path / "single_wave_tdm.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    kernels = emit_kernels_from_config(path, arch="gfx1250", limit=1)
    assert len(kernels) == 1
    _, source, error = kernels[0]
    (tmp_path / "single_wave_tdm.s").write_text(source)
    assert error == 0
    return source


# One loop iteration per unroll (DepthU == MatrixInstK) leaves no local-read
# prefetch, so the loop-top reads consume the previous trip's tensor load.
# PGR2 and expanded PGR1 loops need only the wait; a single PGR1 loop copy
# keeps the loop barrier.
@pytest.mark.parametrize("problem,mi,depth_u,pgr,forks,barrier", [
    ({"SupportUserArgs": True, "UseBias": 0, "Activation": False, "Sparse": 1},
     [16, 16, 64, 1, 1, 1, 1, 1, 1], 64, 2, _SPARSE_FORKS, False),
    ({"SupportUserArgs": True, "UseBias": 0, "Activation": False, "Sparse": 2, "MetadataLayout": 1},
     [16, 16, 64, 1, 1, 1, 1, 1, 1], 64, 2, _SPARSE_FORKS, False),
    ({"SupportUserArgs": True, "UseBias": 0, "Activation": False, "Sparse": 1},
     [16, 16, 64, 1, 1, 1, 1, 1, 1], 64, 1, _SPARSE_FORKS, True),
    ({}, [16, 16, 32, 1, 1, 1, 1, 1, 1], 32, 2, {}, False),
    ({}, [16, 16, 32, 1, 1, 1, 1, 1, 1], 32, 1, {}, True),
    ({}, [16, 16, 32, 1, 1, 1, 1, 1, 1], 32, 1, {"ExpandPointerSwap": [True]}, False),
], ids=["sparse1", "sparse2-metadata-layout1", "sparse1-pgr1", "dense", "dense-pgr1", "dense-pgr1-expanded"])
def test_persistent_single_wave_tdm_waits_before_loop_top_reads(tmp_path, problem, mi, depth_u, pgr, forks, barrier):
    source = _emit(tmp_path, _config(problem, mi, depth_u, pgr, forks))
    loop = source.split("label_LoopBeginL:", 1)[1].split("label_LoopEndL:", 1)[0]
    assert ("s_barrier" in loop) == barrier
    loop_top = loop.split("ds_load", 1)[0]
    assert re.search(r"\bs_wait_tensorcnt\s+\d", loop_top)
    if barrier:
        assert loop_top.index("s_wait_tensorcnt") < loop_top.index("tensor_load_to_lds")


def _writer(plr):
    writer = SimpleNamespace(states=SimpleNamespace(numItersPLR=plr))
    writer._needsLoopCarriedTensorTokenDeps = \
        lambda kernel: KernelWriter._needsLoopCarriedTensorTokenDeps(writer, kernel)
    return writer


@pytest.mark.parametrize("num_threads,tdm,plr,expected", [
    (32, True, 0, True),
    (32, True, 1, False),
    (32, False, 0, False),
    (128, True, 0, False),
])
def test_loop_carried_tensor_token_deps_only_for_single_wave_tdm_without_prefetch(num_threads, tdm, plr, expected):
    kernel = {"NumThreads": num_threads, "WavefrontSize": 32, "enableTDMA": tdm, "enableTDMB": False}
    assert KernelWriter._needsLoopCarriedTensorTokenDeps(_writer(plr), kernel) is expected


@pytest.mark.parametrize("num_threads,plr,pgr,expand,expected", [
    (32, 0, 1, False, True),
    (32, 0, 2, False, False),
    (32, 0, 1, True, False),
    (32, 1, 1, False, False),
    (128, 0, 1, False, False),
])
def test_single_wave_loop_barrier_only_for_unexpanded_pgr1(num_threads, plr, pgr, expand, expected):
    kernel = {"NumThreads": num_threads, "WavefrontSize": 32, "enableTDMA": True, "enableTDMB": True,
              "PrefetchGlobalRead": pgr, "ExpandPointerSwap": expand}
    assert KernelWriter._needsSingleWaveTensorLoopBarrier(_writer(plr), kernel) is expected
