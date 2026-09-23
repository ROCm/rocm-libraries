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
