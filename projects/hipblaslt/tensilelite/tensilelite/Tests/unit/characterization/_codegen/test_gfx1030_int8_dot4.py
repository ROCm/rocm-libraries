# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from config_harness import emit_kernels_from_config, solutions_from_config


@pytest.mark.parametrize("transpose_a,transpose_b", [(False, False), (False, True),
                                                      (True, False), (True, True)])
def test_gfx1030_int8_dot4_codegen(tmp_path, transpose_a, transpose_b):
    source = Path(__file__).parent / "data/test_data/_designed/gfx90a/mac.yaml"
    config = yaml.safe_load(source.read_text())
    config["BenchmarkProblems"][0][0].update(
        DataType="i8", DestDataType="i", ComputeDataType="i",
        HighPrecisionAccumulate=True, TransposeA=transpose_a, TransposeB=transpose_b)
    params = config["BenchmarkProblems"][0][1]["ForkParameters"]
    for entry in params:
        if "LocalReadVectorWidth" in entry:
            entry["LocalReadVectorWidth"] = [4]
    params.append({"LocalDotLayout": [4]})
    path = tmp_path / "int8.yaml"
    path.write_text(yaml.safe_dump(config))
    kernels = emit_kernels_from_config(path, limit=8, arch="gfx1030", canonical=False)
    assert kernels
    assembler = shutil.which("amdclang++")
    if not assembler:
        pytest.skip("amdclang++ is unavailable")
    for index, (_, assembly, err) in enumerate(kernels):
        assert err == 0 and "v_dot4c_i32_i8" in assembly
        asm = tmp_path / f"kernel-{index}.s"
        asm.write_text(assembly)
        subprocess.run(
            [assembler, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
             "-mcpu=gfx1030", "-c", str(asm), "-o", str(asm.with_suffix(".o"))],
            check=True, capture_output=True)


@pytest.mark.parametrize("name,value,reason", [
    ("InnerUnroll", 4, "int8 dot4 requires InnerUnroll = 1"),
    ("GlobalReadVectorWidthA", 4, "int8 dot4 requires GlobalReadVectorWidthA/B = 1"),
    ("LocalReadVectorWidthA", 8, "int8 dot4 requires LocalReadVectorWidthA = 4"),
])
def test_gfx1030_int8_dot4_rejects_unsupported_geometry(
        tmp_path, monkeypatch, capsys, name, value, reason):
    import tensilelite.BenchmarkProblems as benchmark_problems

    monkeypatch.setattr(
        benchmark_problems, "ParallelMap2",
        lambda function, objects, *_args, **_kwargs: [function(*args) for args in objects])
    source = Path(__file__).parent / "data/test_data/_designed/gfx90a/mac.yaml"
    config = yaml.safe_load(source.read_text())
    config["GlobalParameters"]["PrintSolutionRejectionReason"] = True
    config["BenchmarkProblems"][0][0].update(
        DataType="i8", DestDataType="i", ComputeDataType="i",
        HighPrecisionAccumulate=True)
    params = config["BenchmarkProblems"][0][1]["ForkParameters"]
    for entry in params:
        if "ThreadTile" in entry:
            entry["ThreadTile"] = [[4, 4]]
        if "WorkGroup" in entry:
            entry["WorkGroup"] = [[8, 8, 1]]
        if "LocalReadVectorWidth" in entry:
            entry["LocalReadVectorWidth"] = [4]
        if name in entry:
            entry[name] = [value]
    if not any(name in entry for entry in params):
        params.append({name: [value]})
    params.append({"LocalDotLayout": [4]})
    path = tmp_path / "int8-invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    assert not solutions_from_config(path, arch="gfx1030")
    assert reason in capsys.readouterr().out
