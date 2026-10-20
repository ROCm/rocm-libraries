# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Assemble input-bias gradients and check completion ordering on gfx1250."""

from pathlib import Path
import re

import pytest
import yaml

from config_harness import assert_assembles, emit_kernels_from_config
from Tensile import __file__ as tensile_file

pytestmark = pytest.mark.unit


def _input_bias_config(strategy, source, feature=None):
    root = Path(tensile_file).parent
    config = yaml.safe_load((root / "Tests/common/gradient/gfx1250/hhs_bgrada_gfx1250.yaml").read_text())
    config["BenchmarkProblems"] = config["BenchmarkProblems"][:1]
    problem, group = config["BenchmarkProblems"][0]
    problem["BiasSrc"] = source
    values = {key: value[:1] for entry in group["ForkParameters"] for key, value in entry.items()}
    values.update(TileProcessingStrategy=[strategy], WorkAssignment=["StaticGrid"],
                  StreamKFixupTreeReduction=[int(strategy == "StreamK")], GlobalSplitU=[0],
                  PrefetchAcrossPersistent=[0], ClusterDim=[[2, 2]], StaggerU=[0],
                  ForceDisableShadowInit=[True], ScheduleIterAlg=[0], DepthU=[64])
    if feature == "sav":
        problem["UseScaleAlphaVec"] = 1
    elif feature == "scale_ab":
        problem["UseScaleAB"] = "Vector"
    elif feature == "remap":
        values["StoreRemapVectorWidth"] = [4]
    group["ForkParameters"] = [{key: value} for key, value in values.items()]
    group["BenchmarkFinalParameters"] = [
        {"ProblemSizes": [{"Exact": [16, 16, 1, 1024]}, {"Exact": [17, 17, 1, 1025]}]},
        {"BiasTypeArgs": ["s"]}]
    return config


@pytest.mark.parametrize("strategy", ["DataParallel", "StreamK"])
@pytest.mark.parametrize("source", ["A", "B"])
def test_input_bias_gradient_completion_assembles(tmp_path, strategy, source):
    path = tmp_path / "input_bias.yaml"
    path.write_text(yaml.safe_dump(_input_bias_config(strategy, source)))
    results = emit_kernels_from_config(str(path), limit=1, arch="gfx1250", canonical=False)
    assert len(results) == 1
    for name, assembly, error in results:
        assert error == 0
        assert_assembles(assembly, name)
        bias_store = assembly.index("global store bias")
        # A/B bias sums use raw inputs even when GEMM alpha is zero. Neither
        # the prologue nor the StreamK activation/main-loop setup may skip K.
        compute_setup = assembly[:assembly.index("tensor_load_to_lds")]
        assert "SKAlphaCheck" not in compute_setup
        assert "PersistentAlphaCheck" not in compute_setup
        assert "Set summation dim=0 if Alpha == 0" not in compute_setup
        assert "Skip iterations" not in compute_setup
        if strategy == "DataParallel":
            assert "publish input bias partial" not in assembly
            continue
        reduction = assembly.index("accumulate peer input bias partial")
        assert assembly.index("phantom tiles skip the store") < reduction < bias_store
        owner_gate = assembly.rfind("does wg start tile?", reduction, bias_store)
        assert owner_gate != -1
        assert "label_SK_Partials" in assembly[owner_gate:bias_store]
        publication = assembly.index("publish input bias partial before ready flag")
        flag = assembly.index("// set flag", publication)
        assert "global_wb" in assembly[publication:flag]
        assert "wait for data store" in assembly[publication:flag]
        assert "input bias partial slot offset (high word)" in assembly
        assert "s[sgprGSUSumIdx]" not in assembly


@pytest.mark.parametrize("source", ["A", "B"])
@pytest.mark.parametrize("feature", ["sav", "scale_ab", "remap"])
def test_input_bias_partials_survive_other_epilogue_lds_users(tmp_path, source, feature):
    path = tmp_path / "input_bias_combined.yaml"
    path.write_text(yaml.safe_dump(_input_bias_config("StreamK", source, feature)))
    results = emit_kernels_from_config(str(path), limit=1, arch="gfx1250", canonical=False)
    assert len(results) == 1
    for name, assembly, error in results:
        assert error == 0
        assert_assembles(assembly, name)
        bias_store = assembly.index("global store bias")
        assert assembly.index("accumulate peer input bias partial") < bias_store
        if feature == "remap":
            # The raw SumUnroll sums retain their LDS0 layout; remap uses its
            # separate final D-store path only after the bias is consumed.
            assert "add bias lds offset" not in assembly
            continue
        marker = "store scaleAlpha" if feature == "sav" else "store scale"
        stores = [line for line in assembly.splitlines() if "ds_store" in line and marker in line]
        assert stores
        # These kernels have MT16, maxKId2, float accumulation: LDS[0:128]
        # contains the raw input-bias reduction and must survive vector staging.
        offsets = [int(re.search(r"offset:(\d+)", line).group(1)) for line in stores]
        assert min(offsets) >= 128


@pytest.mark.parametrize("strategy", ["DataParallel", "StreamK"])
def test_output_bias_gradient_intermediate_and_reduction_helper(tmp_path, strategy):
    from config_harness import solutions_from_config
    from Tensile.KernelHelperNaming import initReductionKernelObjects

    path = tmp_path / "output_bias.yaml"
    path.write_text(yaml.safe_dump(_input_bias_config(strategy, "D")))
    solutions = solutions_from_config(str(path), arch="gfx1250", limit_solutions=1)
    assert len(solutions) == 1
    helpers = initReductionKernelObjects(solutions[0])
    assert len(helpers) == len(solutions[0]["ProblemType"]["BiasDataTypeList"])
    assert all(h.state["ProblemType"]["ComputeDataType"].isSingle() for h in helpers)
    helper = next(h for h in helpers if h.state["ProblemType"]["BiasDataType"].isSingle())
    assert helper.state["ProblemType"]["BiasSrc"] == "D"
    assert helper.state["ProblemType"]["ComputeDataType"].isSingle()
    assert helper.state["ProblemType"]["BiasDataType"].isSingle()
    assert helper.getKernelName().endswith("_SS_Reduction")
    assert "float *in, float* out" in helper.getHeaderFileString()

    results = emit_kernels_from_config(str(path), limit=1, arch="gfx1250", canonical=False)
    assert len(results) == 1
    for name, assembly, error in results:
        assert error == 0
        assert_assembles(assembly, name)
        # D-gradient reduces the completed GEMM result through an intermediate
        # FP32 bias matrix, including when the final output is half precision.
        stores = [line for line in assembly.splitlines()
                  if "buffer_store_b32" in line and "// store Bias" in line]
        assert stores and all("sgprSrdBias" in line for line in stores)
        first_store = assembly.index(stores[0])
        assert assembly.index("phantom tiles skip the store") < first_store
        if strategy == "StreamK":
            assert assembly.index("flag offset based on physical partial slot") < first_store
            assert "publish input bias partial" not in assembly


@pytest.mark.parametrize("source", ["A", "B"])
def test_alpha_zero_hardware_config_assembles_all_variants(tmp_path, source):
    root = Path(tensile_file).parent
    config = yaml.safe_load((root / "Tests/common/streamk/gfx1250/persistent_input_bias_alpha0.yaml").read_text())
    config["BenchmarkProblems"] = [entry for entry in config["BenchmarkProblems"]
                                   if entry[0]["BiasSrc"] == source]
    path = tmp_path / "alpha_zero.yaml"
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(str(path), limit=4, arch="gfx1250", canonical=False)
    assert len(results) == 4, "both persistent strategies and prefetch depths must generate"
    for name, assembly, error in results:
        assert error == 0
        assert_assembles(assembly, name)
        assert "global store bias" in assembly
        compute_setup = assembly[:assembly.index("tensor_load_to_lds")]
        assert "SKAlphaCheck" not in compute_setup
        assert "PersistentAlphaCheck" not in compute_setup
        assert "Set summation dim=0 if Alpha == 0" not in compute_setup
        assert "Skip iterations" not in compute_setup
