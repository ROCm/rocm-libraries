# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Output-amax validity and packed-store accumulation regressions."""

from pathlib import Path
import re

import pytest
import yaml

from config_harness import emit_kernels_from_config, solutions_from_config

pytestmark = pytest.mark.unit
CONFIG = Path(__file__).parent / "test_data" / "output_amax.yaml"


@pytest.mark.parametrize("destination", ["h", "b"])
@pytest.mark.parametrize("output_amax", [False, True])
def test_amax_activation_uses_accumulator_precision(destination, output_amax):
    from Tensile.SolutionStructs import ProblemType

    config = yaml.safe_load(CONFIG.read_text())["BenchmarkProblems"][0][0]
    config.update(
        DataType=destination,
        DestDataType=destination,
        Activation=True,
        ActivationType="all",
        ActivationComputeDataType=destination,
        OutputAmaxD=output_amax,
    )
    problem = ProblemType(config, False)
    assert problem["ActivationType"] == "all"
    expected = problem["ComputeDataType"] if output_amax else problem["DestDataType"]
    assert problem["ActivationComputeDataType"] == expected


def _assert_scalar_scales_ready_before_use(assembly):
    # Read emitted instructions: every scale load must complete before its
    # destination register is consumed or reused. This also catches ScaleD-only
    # reads, which must not depend on a nonzero beta or an unrelated bias load.
    loads = list(re.finditer(
        r"^\s*s_load_(?:dword|b32)\s+(s(?:\[[^\]]+\]|\d+)),[^\n]*// load scale[CD]\s*$",
        assembly, re.MULTILINE,
    ))
    assert loads, "scaled kernels must load the requested scalar values"
    for load in loads:
        register = load.group(1)
        remaining = assembly[load.end():]
        use = re.search(re.escape(register) + r"(?!\d)", remaining)
        assert use, f"loaded scale register {register} must be consumed"
        before_use = remaining[:use.start()]
        assert re.search(
            r"\bs_waitcnt\b[^\n]*\blgkmcnt\(0\)|\bs_wait_kmcnt\b[^\n]*\b(?:0|0x0)\b",
            before_use,
        ), f"scalar scale {register} is used before its memory load completes"


@pytest.mark.parametrize("stream_k", [1, 2, 3, 4, 5])
def test_streamk_amax_combination_rejected_before_derivation(stream_k, capsys):
    from Tensile.SolutionStructs import Solution

    state = {"StreamK": stream_k, "ProblemType": {"OutputAmaxD": True}, "Valid": True}
    Solution.assignDerivedParameters(state, False, True, False, None, None)
    assert state["Valid"] is False
    assert "one final-output tile per workgroup" in capsys.readouterr().out


@pytest.mark.parametrize("gsu", [-1, 0, 2, 4])
def test_split_reduction_amax_rejected_before_derivation(gsu, capsys):
    from Tensile.SolutionStructs import Solution

    state = {"GlobalSplitU": gsu, "ProblemType": {"OutputAmaxD": True}, "Valid": True}
    Solution.assignDerivedParameters(state, False, True, False, None, None)
    assert state["Valid"] is False
    assert "split-reduction helpers do not reduce amax" in capsys.readouterr().out


@pytest.mark.parametrize("arch", ["gfx90a", "gfx942", "gfx950"])
@pytest.mark.parametrize("scale_cd", [False, True])
def test_packed_outputs_contribute_to_amax_before_scaling(arch, scale_cd, tmp_path):
    config = yaml.safe_load(CONFIG.read_text())
    config["BenchmarkProblems"][0][0]["UseScaleCD"] = scale_cd
    path = tmp_path / "output_amax.yaml"
    path.write_text(yaml.safe_dump(config))
    solutions = solutions_from_config(path, arch=arch, limit_solutions=1)
    assert len(solutions) == 1
    assert solutions[0]["BatchSizeEqual"] == 1
    results = emit_kernels_from_config(path, limit=1, arch=arch)
    assert len(results) == 1
    _, assembly, error = results[0]
    assert error == 0
    (tmp_path / "output_amax.s").write_text(assembly)
    # Packed stores produce consecutive scalar FP32 accumulations before any
    # ScaleD multiplication or FP16 conversion. Each vi must contribute.
    runs = list(re.finditer(r"(?:[^\n]*v_max_f32[^\n]*absmax[^\n]*\n){8}", assembly))
    assert runs, "eight-element vector stores must accumulate all eight values"
    for run in runs:
        inputs = re.findall(r"abs\(([^)]*)\)", run.group())
        assert len(set(inputs)) == 8, run.group()
        start = assembly.rfind(
            "/* apply mask, calc new C and issue writes */", 0, run.start()
        )
        assert start >= 0
        conversion = assembly.index("convert C to fp16", start)
        assert run.end() <= conversion, "amax must use FP32 values before packing"
        if scale_cd:
            scaling = assembly.index("result *= ScaleD", start)
            assert run.end() <= scaling, "amax must be independent of ScaleD"
    if scale_cd:
        _assert_scalar_scales_ready_before_use(assembly)
        assert "result *= ScaleD" in assembly
    else:
        assert "result *= ScaleD" not in assembly


@pytest.mark.parametrize("destination", ["h", "b"])
def test_non_amax_retains_packed_activation(destination, tmp_path):
    config = yaml.safe_load(CONFIG.read_text())
    problem, parameters = config["BenchmarkProblems"][0]
    problem.update(
        DataType=destination,
        DestDataType=destination,
        Activation=True,
        ActivationType="all",
        ActivationComputeDataType="s",
        OutputAmaxD=False,
        UseScaleCD=False,
    )
    parameters["ForkParameters"].append({"ActivationFuncCall": [False]})
    parameters["BenchmarkFinalParameters"].append(
        {"ActivationArgs": [[{"Enum": "relu"}], [{"Enum": "abs"}]]}
    )
    path = tmp_path / "packed_activation.yaml"
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(path, limit=1, arch="gfx950")
    assert len(results) == 1
    _, assembly, error = results[0]
    assert error == 0
    assert "absmax" not in assembly
    (tmp_path / "packed_activation.s").write_text(assembly)
    names = ["Abs", "Relu"] if destination == "h" else ["Abs"]
    conversion = "convert C to fp16" if destination == "h" else "convert C to bf16"
    for name in names:
        blocks = re.findall(
            rf"^label_Activation_{name}(?:_\d+)?:\n(.*?)(?=^label_Activation_|\Z)",
            assembly, re.MULTILINE | re.DOTALL,
        )
        assert blocks, f"missing {name} branch"
        packed = "0x7fff7fff" if name == "Abs" else "v_pk_max_f16"
        assert any(packed in block for block in blocks)
        instruction = "Remove sign bit" if name == "Abs" else "x = max(0, x)"
        for block in blocks:
            # Edge stores can use one half-word instead of a packed pair.
            assert block.index(conversion) < block.index(instruction)


@pytest.mark.parametrize("arch", ["gfx90a", "gfx942", "gfx950"])
@pytest.mark.parametrize("destination", ["h", "b"])
@pytest.mark.parametrize("function_call", [False, True])
@pytest.mark.parametrize("scale_cd", [False, True])
def test_activated_amax_precedes_destination_conversion(
    arch, destination, function_call, scale_cd, tmp_path
):
    config = yaml.safe_load(CONFIG.read_text())
    problem, parameters = config["BenchmarkProblems"][0]
    problem.update(
        DataType=destination,
        DestDataType=destination,
        Activation=True,
        # Explicit relu/abs ProblemTypes are normalized to none. The all
        # dispatcher exercises their real inline and function-call branches.
        ActivationType="all",
        ActivationComputeDataType=destination if function_call else "s",
        UseScaleCD=scale_cd,
    )
    parameters["ForkParameters"].append({"ActivationFuncCall": [function_call]})
    parameters["BenchmarkFinalParameters"].append(
        {"ActivationArgs": [[{"Enum": "relu"}], [{"Enum": "abs"}]]}
    )
    path = tmp_path / "activated_amax.yaml"
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(path, limit=1, arch=arch)
    assert len(results) == 1
    _, assembly, error = results[0]
    assert error == 0
    (tmp_path / "activated_amax.s").write_text(assembly)

    # Every eight-element store must reduce the FP32 values, including when
    # activation would otherwise be optimized to run on packed destinations.
    runs = list(re.finditer(r"(?:[^\n]*v_max_f32[^\n]*absmax[^\n]*\n){8}", assembly))
    assert runs
    conversion_comment = "convert C to fp16" if destination == "h" else "convert C to bf16"
    for run in runs:
        start = assembly.rfind("/* apply mask, calc new C and issue writes */", 0, run.start())
        assert start >= 0
        conversion = assembly.index(conversion_comment, start)
        assert run.end() <= conversion, "amax must precede destination conversion"
        if function_call:
            assert "s_swappc_b64" in assembly[start:run.start()], "activate before amax"
        if scale_cd:
            scaling = assembly.index("result *= ScaleD", start)
            assert run.end() <= scaling < conversion

    # Verify the actual activation instructions, not just the reduction order:
    # ReLU must discard negative FP32 values and abs must clear the FP32 sign bit.
    assert re.search(r"v_max_f32[^\n]*x = max\(0, x\)", assembly)
    assert re.search(r"v_and_b32[^\n]*0x7fffffff[^\n]*Remove sign bit", assembly)
    assert not re.search(r"v_(?:pk_)?max_f16[^\n]*x = max\(0, x\)", assembly)
    assert "0x7fff7fff" not in assembly
    if not function_call:
        for name, instruction in [("Relu", "x = max(0, x)"), ("Abs", "Remove sign bit")]:
            blocks = re.findall(
                rf"^label_Activation_{name}(?:_\d+)?:\n(.*?)(?=^label_Activation_|\Z)",
                assembly, re.MULTILINE | re.DOTALL,
            )
            assert blocks, f"missing {name} branch"
            for block in blocks:
                assert block.index(instruction) < block.index("absmax")
    if scale_cd:
        _assert_scalar_scales_ready_before_use(assembly)
