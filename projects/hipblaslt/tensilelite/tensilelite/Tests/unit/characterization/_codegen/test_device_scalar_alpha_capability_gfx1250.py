# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Legacy SAV kernels must not consume an unsupported device-scalar selector."""

from pathlib import Path
import re

import pytest
import yaml

from config_harness import emit_kernels_from_config, solutions_from_config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("capability", [None, False, True], ids=["omitted", "disabled", "enabled"])
@pytest.mark.parametrize("runtime_scalar", [False, True])
def test_f32_sav_capability_requires_explicit_opt_in(tmp_path, capability, runtime_scalar):
    from Tensile.Contractions import InternalArgsSupport
    from Tensile.KernelWriter import KernelWriter

    source_config = Path(__file__).parent / "data/test_data/_designed/gfx1250/rich_gemm_sb.yaml"
    config = yaml.safe_load(source_config.read_text())
    benchmark = config["BenchmarkProblems"][0][1]
    benchmark["BenchmarkFinalParameters"].append({"DeviceScalarAlpha": [runtime_scalar]})
    if capability is not None:
        benchmark["InternalSupportParams"] = {"SupportDeviceScalarAlpha": capability}
    benchmark["ForkParameters"] = [
        {key: values[:1] for key, values in fork.items()}
        for fork in benchmark["ForkParameters"]
    ]
    config_path = tmp_path / "capability.yaml"
    config_path.write_text(yaml.safe_dump(config))

    solutions = solutions_from_config(config_path, arch="gfx1250", expected_fork_count=1)
    assert len(solutions) == 1
    solution = solutions[0]
    enabled = capability is True
    assert solution["InternalSupportParams"]["SupportDeviceScalarAlpha"] is enabled
    assert InternalArgsSupport.FromOriginalState(solution._state).deviceScalarAlpha is enabled
    assert KernelWriter.gsuMaskHex(None, solution) == ("0x7ff" if enabled else "0xfff")

    results = emit_kernels_from_config(config_path, arch="gfx1250", expected_fork_count=1)
    assert len(results) == 1
    _, source, error = results[0]
    assert error == 0
    assert ("replace inline alpha with ScaleAlphaVec[0]" in source) is enabled
    assert ("device scalar uses ScaleAlphaVec identity" in source) is enabled
    assert bool(re.search(r"s_bitcmp[01]_b32\s+s\[sgprArgType\],\s*9\b", source)) is enabled
    assert "Load ScaleAlphaVec" in source
    assert "*= ScaleAlphaVec" in source


def test_generated_kernel_rejects_other_internal_overrides(tmp_path):
    source_config = Path(__file__).parent / "data/test_data/_designed/gfx1250/rich_gemm_sb.yaml"
    config = yaml.safe_load(source_config.read_text())
    config["BenchmarkProblems"][0][1]["InternalSupportParams"] = {
        "SupportDeviceScalarAlpha": True,
        "KernArgsVersion": 2,
    }
    config_path = tmp_path / "unsupported_override.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(SystemExit):
        solutions_from_config(config_path, arch="gfx1250", limit_solutions=1)


@pytest.mark.parametrize("sav", [False, True], ids=["missing-sav", "i32-compute"])
def test_scalar_capability_rejects_unsupported_problem(tmp_path, capsys, monkeypatch, sav):
    from Tensile.Common.GlobalParameters import globalParameters

    # Keep rejection output in this process instead of a reused joblib worker.
    monkeypatch.setitem(globalParameters, "CpuThreads", 1)
    source_config = (
        Path(__file__).resolve().parents[3] / "common/gemm/gfx12/i8ii_gfx1250.yaml"
        if sav else
        Path(__file__).parent / "data/test_data/_designed/gfx1250/rich_gemm_sb.yaml"
    )
    config = yaml.safe_load(source_config.read_text())
    config["GlobalParameters"]["PrintSolutionRejectionReason"] = True
    config["BenchmarkProblems"][0][0]["UseScaleAlphaVec"] = int(sav)
    config["BenchmarkProblems"][0][1]["InternalSupportParams"] = {
        "SupportDeviceScalarAlpha": True
    }
    config_path = tmp_path / "unsupported_scalar.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert not solutions_from_config(config_path, arch="gfx1250", limit_solutions=1)
    assert "SupportDeviceScalarAlpha requires" in capsys.readouterr().out


@pytest.mark.parametrize("problem_index", range(4))
def test_scalar_runtime_yaml_explicitly_enables_all_kernels(problem_index):
    source_config = (Path(__file__).resolve().parents[3]
                     / "common/gemm/gfx12/mxf4_device_scalar_alpha_gfx1250.yaml")
    solutions = solutions_from_config(source_config, arch="gfx1250", problem_index=problem_index)
    assert solutions
    assert all(solution["InternalSupportParams"]["SupportDeviceScalarAlpha"]
               for solution in solutions)


@pytest.mark.parametrize("support_user_args", [False, True])
def test_i32_sav_does_not_read_device_scalar_selector(tmp_path, support_user_args):
    # I32 compute supports legacy vector alpha but not device scalar alpha.
    # Without user args, ArgType bit 9 is not even initialized in the prologue.
    source_config = Path(__file__).resolve().parents[3] / "common/gemm/gfx12/i8ii_gfx1250.yaml"
    config = yaml.safe_load(source_config.read_text())
    config["BenchmarkProblems"] = config["BenchmarkProblems"][:1]
    problem, benchmark = config["BenchmarkProblems"][0]
    problem["SupportUserArgs"] = support_user_args
    benchmark["ForkParameters"] = [
        {key: values[:1] for key, values in fork.items()}
        for fork in benchmark["ForkParameters"]
    ]
    config_path = tmp_path / "i32_legacy_sav.yaml"
    config_path.write_text(yaml.safe_dump(config))

    solutions = solutions_from_config(
        config_path, arch="gfx1250", expected_fork_count=1
    )
    assert len(solutions) == 1
    solution = solutions[0]
    assert solution["ProblemType"]["ComputeDataType"].isInt32()
    assert solution["ProblemType"]["UseScaleAlphaVec"] == 1
    assert solution["ProblemType"]["SupportUserArgs"] is support_user_args
    assert solution["BufferLoad"]
    assert not solution["InternalSupportParams"]["SupportDeviceScalarAlpha"]

    results = emit_kernels_from_config(
        config_path, arch="gfx1250", expected_fork_count=1
    )
    assert len(results) == 1
    base, source, error = results[0]
    assert error == 0, f"Kernel {base!r} emitted with err={error}"
    assert not re.search(
        r"s_bitcmp[01]_b32\s+s\[sgprArgType\],\s*9\b", source
    ), "A legacy-only SAV kernel must not read the device-scalar selector"
    assert not re.search(
        r"s_cselect_b32\s+s\[sgprSrdScaleAlphaVec\+2\],\s*0,\s*"
        r"s\[sgprSrdScaleAlphaVec\+2\]", source
    ), "A legacy-only SAV kernel must not suppress the vector SRD length"
    assert "Load ScaleAlphaVec" in source
    assert "store scaleAlpha" in source
    assert "load scaleAlpha" in source
    assert "*= ScaleAlphaVec" in source


@pytest.mark.parametrize("support_user_args", [False, True])
def test_grouped_scalar_alpha_selects_record_layout(tmp_path, support_user_args):
    source_config = (Path(__file__).parent / "data/test_data/_designed/gfx1250/rich_gemm_sb.yaml")
    config = yaml.safe_load(source_config.read_text())
    config["BenchmarkProblems"] = config["BenchmarkProblems"][:1]
    problem, benchmark = config["BenchmarkProblems"][0]
    problem["GroupedGemm"] = True
    problem["SupportUserArgs"] = support_user_args
    benchmark["InternalSupportParams"] = {"SupportDeviceScalarAlpha": True}
    benchmark["ForkParameters"] = [
        {key: values[:1] for key, values in fork.items()}
        for fork in benchmark["ForkParameters"]
    ]
    config_path = tmp_path / "grouped_scalar.yaml"
    config_path.write_text(yaml.safe_dump(config))
    solutions = solutions_from_config(config_path, arch="gfx1250", expected_fork_count=1)
    assert len(solutions) == 1
    assert solutions[0]["InternalSupportParams"]["SupportDeviceScalarAlpha"]
    results = emit_kernels_from_config(config_path, arch="gfx1250", expected_fork_count=1)
    assert len(results) == 1
    _, source, error = results[0]
    assert error == 0
    assert "replace inline alpha with ScaleAlphaVec[0]" in source
    assert "device scalar uses ScaleAlphaVec identity" in source
    if support_user_args:
        assert "fixed device-user-args layout" in source
        assert re.search(r"s_cselect_b32[^\n]*0x90", source)
    else:
        assert "fixed device-user-args layout" not in source


@pytest.mark.parametrize("algorithm", ["SingleBuffer", "MultipleBuffer"])
def test_conversion_applies_device_scalar_alpha_once(tmp_path, algorithm):
    """SingleBuffer workspace already includes alpha*A*B and beta*C."""
    from tensilelite.KernelHelperNaming import initConversionKernelObjects
    from tensilelite.TensileCreateLibrary.Run import generateKernelObjectsFromSolutions
    from config_harness import _toolchain_for

    source_config = (Path(__file__).resolve().parents[3]
                     / "common/gemm/gfx12/mxf4_device_scalar_alpha_gfx1250.yaml")
    config = yaml.safe_load(source_config.read_text())
    config["BenchmarkProblems"] = config["BenchmarkProblems"][:1]
    problem, benchmark = config["BenchmarkProblems"][0]
    # FP16 output forces SingleBuffer to use the FP32 workspace + conversion.
    problem["DestDataType"] = "h"
    problem["UseBias"] = 0
    forks = {key: values[:1] for fork in benchmark["ForkParameters"]
             for key, values in fork.items()}
    forks.update(GlobalSplitU=[2], GlobalSplitUAlgorithm=[algorithm], StoreRemapVectorWidth=[0])
    benchmark["ForkParameters"] = [{key: values} for key, values in forks.items()]
    config_path = tmp_path / "conversion_alpha.yaml"
    config_path.write_text(yaml.safe_dump(config))

    solutions = solutions_from_config(config_path, arch="gfx1250", expected_fork_count=1)
    assert len(solutions) == 1
    kernel = generateKernelObjectsFromSolutions(solutions)[0]
    assert kernel["_GlobalAccumulation"] == algorithm
    assert kernel["InternalSupportParams"]["SupportDeviceScalarAlpha"]
    _, isa_info = _toolchain_for("gfx1250")
    conversions = initConversionKernelObjects(kernel, isa_info)
    assert {conversion.num_elements_load for conversion in conversions} == {1, 2, 4}
    for conversion in conversions:
        error, source = conversion.getSourceFileString()
        assert error == 0
        assert "_GG_" in source  # Both grouped and regular helper variants.
        assert ("arg.alpha = arg.ScaleAlphaVec[0];" in source) == (algorithm == "MultipleBuffer")
        # Scalar mode must still suppress per-element scaling, even when the
        # workspace is already scaled; legacy vector mode keeps its multiply.
        assert "if(arg.ScaleAlphaVec != nullptr && !arg.deviceScalarAlpha){" in source
        assert "arg.ScaleAlphaVec[id0+0]" in source
