# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Single-wave sparse TDM StreamK kernels must fit the SGPR budget."""

import os
import re
import shutil
import subprocess

import pytest
import yaml

from config_harness import derive_states, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ASSEMBLER = shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++"


def _config(sparse, waves, strategy="StreamK", cluster=(1, 1), staggerU=0):
    problem_type = {"OperationType": "GEMM", "DataType": "h", "DestDataType": "s", "ComputeDataType": "s",
                    "HighPrecisionAccumulate": True, "TransposeA": True, "TransposeB": False,
                    "UseBeta": True, "Batched": True, "SupportUserArgs": True, "UseBias": 0,
                    "Activation": False, **sparse}
    forks = {
        "MatrixInstruction": [[16, 16, 64, 1, 1, 1, 1, waves[0], waves[1]]],
        "DepthU": [64],
        "TileProcessingStrategy": [strategy],
        "WorkAssignment": ["StaticGrid"],
        "ClusterDim": [list(cluster)],
        "TDMInst": [3],
        "WavefrontSize": [32],
        "WorkGroupMapping": [1],
        "GlobalSplitU": [0],
        "StaggerU": [staggerU],
        "PrefetchGlobalRead": [2],
        "PrefetchLocalRead": [1],
        "ScheduleIterAlg": [4],
        "1LDSBuffer": [0],
        "ExpandPointerSwap": [False],
        "StoreRemapVectorWidth": [0],
        "StreamKFixupTreeReduction": [int(strategy == "StreamK")],
        "PrefetchAcrossPersistent": [0],
        "TransposeLDS": [-1],
        "LDSTrInst": [False],
        "DirectToVgprSparseMetadata": [False],
        "SourceSwap": [False],
        "LocalReadVectorWidth": [8],
    }
    return {
        "GlobalParameters": {"Architecture": "gfx1250", "PruneSparseMode": 0},
        "BenchmarkProblems": [[
            problem_type,
            {"BenchmarkCommonParameters": [{"KernelLanguage": ["Assembly"]}],
             "ForkParameters": [{key: value} for key, value in forks.items()],
             "BenchmarkFinalParameters": [{"ProblemSizes": [{"Exact": [32, 32, 1, 320]}]}]},
        ]],
    }


def _write(tmp_path, config):
    path = tmp_path / "sparse_streamk.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _emit(tmp_path, config):
    path = _write(tmp_path, config)
    kernels = emit_kernels_from_config(path, arch="gfx1250", limit=1)
    assert len(kernels) == 1
    _, source, error = kernels[0]
    (tmp_path / "sparse_streamk.s").write_text(source)
    return source, error


def _assemble(tmp_path):
    return subprocess.run(
        [_ASSEMBLER, "-x", "assembler", "--target=amdgcn-amd-amdhsa", "-mcode-object-version=4",
         "-Xclangas", "-target-feature", "-Xclangas", "+real-true16",
         "-mcpu=gfx1250", "-mno-wavefrontsize64", "-c",
         str(tmp_path / "sparse_streamk.s"), "-o", str(tmp_path / "sparse_streamk.o")],
        capture_output=True, text=True)


def _gfx1250_assembler_available():
    if not os.path.exists(_ASSEMBLER):
        return False
    probe = subprocess.run([_ASSEMBLER, "-x", "assembler", "--target=amdgcn-amd-amdhsa", "-mcpu=gfx1250",
                            "-c", "-", "-o", os.devnull], input="s_nop 0\n", capture_output=True, text=True)
    return probe.returncode == 0


_SPARSE = [{"Sparse": 1}, {"Sparse": 2, "MetadataLayout": 1}]


# One wave keeps separate A, B and metadata TDM descriptors; with the stagger
# registers on top the kernel needed 110 SGPRs and was replaced by an
# s_endpgm stub (error 2). Without stagger the persistent kernel folds the
# A/B increments into assembler constants, which the single-wave TDM
# increment must read as constants, not as SGPRs.
@pytest.mark.skipif(not _gfx1250_assembler_available(), reason="assembler does not support gfx1250")
@pytest.mark.parametrize("sparse", _SPARSE, ids=["sparse1", "sparse2-metadata-layout1"])
def test_single_wave_sparse_tdm_streamk_fits_sgprs_and_assembles(tmp_path, sparse):
    source, error = _emit(tmp_path, _config(sparse, [1, 1]))
    assert error == 0
    assert "WrapU" not in source
    assert int(re.search(r"\.amdhsa_next_free_sgpr\s+(\d+)", source).group(1)) <= 106
    assert re.search(r"s_add_u32 s\[sgprtdmAGroup0\+2\], s\[sgprtdmAGroup0\+2\], GlobalReadIncsA\b", source)
    result = _assemble(tmp_path)
    assert result.returncode == 0, result.stderr


# A workgroup cluster also drops stagger, so the single-wave increments are
# constants there too.
@pytest.mark.skipif(not _gfx1250_assembler_available(), reason="assembler does not support gfx1250")
def test_single_wave_sparse_tdm_cluster_assembles(tmp_path):
    source, error = _emit(tmp_path, _config(_SPARSE[0], [1, 1], strategy="DataParallel", cluster=(2, 2)))
    assert error == 0
    assert re.search(r"s_add_u32 s\[sgprtdmAGroup0\+2\], s\[sgprtdmAGroup0\+2\], GlobalReadIncsA\b", source)
    result = _assemble(tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("sparse", _SPARSE[:1], ids=["sparse1"])
def test_multi_wave_sparse_tdm_streamk_keeps_stagger(tmp_path, sparse):
    source, error = _emit(tmp_path, _config(sparse, [2, 2]))
    assert error == 0
    assert "sgprWrapUA" in source


# The kernel has no stagger code, so the solution must not advertise runtime
# StaggerU either.
@pytest.mark.parametrize("waves, disabled", [([1, 1], True), ([2, 2], False)], ids=["one-wave", "four-wave"])
def test_single_wave_sparse_tdm_streamk_drops_runtime_stagger(tmp_path, waves, disabled):
    path = _write(tmp_path, _config(_SPARSE[0], waves, staggerU=32))
    (state,) = derive_states(path, arch="gfx1250", limit_solutions=1)
    assert (state["StaggerU"] == 0) is disabled
    assert state["InternalSupportParams"]["SupportCustomStaggerU"] is (not disabled)
