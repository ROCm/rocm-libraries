# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate a small gfx942 library and execute one regular GEMM per solution."""

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from artifact_helpers import compress_output, extract_artifact
from tensilelite import Tensile
from tensilelite.Common import state
from tensilelite.SolutionStructs.Problem import ProblemType
from tensilelite.Tests.gpu_detection import get_available_archs
from tensilelite.Utilities.gemm_problem_bench import bench_yaml
from tensilelite.Utilities.gemm_problem_for_solution import load_library_metadata

pytestmark = [pytest.mark.gfx94x, pytest.mark.emu_fast, pytest.mark.gpu, pytest.mark.gemm]


def _logic(dtype, split):
    # One wave / 16x16 output tile keeps eight reduction blocks cheap to emulate.
    mi_k = 4 if dtype in {"S", "D"} else 16
    compute_type = "D" if dtype == "D" else "S"
    logic = {
        "MinimumRequiredVersion": "5.0.0",
        "ScheduleName": "gfx942",
        "ArchitectureName": "gfx942",
        "DeviceNames": ["Device 74a0"],
        "LibraryType": "Prediction",
        "ProblemType": {
            "OperationType": "GEMM",
            "DataType": dtype,
            "DestDataType": compute_type,
            "ComputeDataType": compute_type,
            "HighPrecisionAccumulate": dtype in {"H", "B"},
            "TransposeA": False,
            "TransposeB": False,
            "Batched": True,
            "UseBeta": True,
        },
        "Solutions": [
            {
                "SolutionIndex": 0,
                "KernelLanguage": "Assembly",
                "MatrixInstruction": [16, 16, mi_k, 1],
                "EnableMatrixInstruction": True,
                "MIBlock": [16, 16, mi_k, 1, 1, 1],
                "MIWaveGroup": [1, 1],
                "MIWaveTile": [1, 1],
                "WorkGroup": [16, 4, 1],
                "MIInputPerThread": mi_k // 4,
                "MIInputPerThreadA": mi_k // 4,
                "MIInputPerThreadB": mi_k // 4,
                "DepthU": 32,
                "TileProcessingStrategy": "None",
                "MFMA_BF16_1K": False,
                "GlobalSplitU": split,
                "GlobalSplitUAlgorithm": "MultipleBuffer",
                "PrefetchGlobalRead": 1,
                "PrefetchLocalRead": 1,
                "ScheduleIterAlg": 3,
                "StaggerU": 0,
                "WorkGroupMapping": 1,
            }
        ],
    }
    logic["ProblemType"] = state(ProblemType(logic["ProblemType"], False).state)
    return logic


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("S", marks=pytest.mark.Float),
        pytest.param("D", marks=pytest.mark.Double),
        pytest.param("H", marks=pytest.mark.Half),
        pytest.param("B", marks=pytest.mark.BFloat16),
    ],
)
@pytest.mark.parametrize("split", [1, 4])
def test_gemm_problem_for_solution(
    dtype, split, tmp_path, pytestconfig, tensile_args, worker_lock_instance
):
    build_only = pytestconfig.getoption("--build-only")
    use_cache = pytestconfig.getoption("--use-cache")
    targets = pytestconfig.getoption("--gpu-targets")
    archs = targets.split(";") if build_only else get_available_archs()
    if "gfx942" not in archs:
        pytest.skip("requires gfx942 hardware or emulator")
    bench = pytestconfig.getoption("--hipblaslt-bench") or shutil.which("hipblaslt-bench")
    if not build_only and not bench:
        pytest.skip("requires hipblaslt-bench on PATH or --hipblaslt-bench")

    parser = argparse.ArgumentParser()
    Tensile.addCommonArguments(parser)
    parser.add_argument("--gpu-targets")
    options = parser.parse_args(tensile_args)
    artifact_name = f"gemm_problem_{dtype}_gsu{split}_gfx942"
    output = tmp_path / artifact_name
    artifact_dir = pytestconfig.getoption("--artifact-dir")
    if use_cache:
        extract_artifact(Path(artifact_dir) / f"{artifact_name}.tar.gz", output)
    else:
        logic_dir = tmp_path / "logic"
        logic_dir.mkdir()
        (logic_dir / "gfx942.yaml").write_text(yaml.safe_dump(_logic(dtype, split)))
        command = [
            sys.executable,
            str(Path(Tensile.__file__).parent / "bin/TensileCreateLibrary"),
            str(logic_dir),
            str(output),
            "HIP",
            "--architecture",
            "gfx942",
            "--library-format",
            "msgpack",
            "--jobs",
            "1",
            "--no-enumerate",
            "--no-generate-solution-table",
        ]
        if options.CxxCompiler:
            command += ["--cxx-compiler", options.CxxCompiler]
        subprocess.run(
            command,
            check=True,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
            timeout=120,
        )
        if build_only:
            compress_output(str(output), dest_dir=artifact_dir, name=artifact_name)
            return

    library = load_library_metadata(output / "library", target="gfx942")
    assert len(library.indices) == 1
    problem = library.get_problem(library.indices[0])
    assert (problem.m, problem.n, problem.k) == (16, 16, 8 * 32 * split)
    config = tmp_path / "regular.yaml"
    config.write_text(bench_yaml([problem]))
    # Bench's --yaml converter has its own build-selected interpreter, which
    # can lack PyYAML in artifact-based CI. Use the pytest interpreter that
    # already has our YAML dependency, with the matching installed template.
    bench_dir = Path(bench).resolve().parent
    data = tmp_path / "regular.data"
    subprocess.run(
        [
            sys.executable,
            str(bench_dir / "hipblaslt_gentest.py"),
            "--template",
            str(bench_dir / "hipblaslt_template.yaml"),
            "-o",
            str(data),
            str(config),
        ],
        check=True,
        timeout=30,
    )
    with worker_lock_instance:
        result = subprocess.run(
            [bench, "--data", str(data)],
            env={**os.environ, "HIPBLASLT_TENSILE_LIBPATH": str(library.directory)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=90,
        )
    text = result.stdout
    (tmp_path / "bench.log").write_text(text)
    assert result.returncode == 0, text
    assert "\nRACE " not in "\n" + text, text

    rows, header = [], None
    for line in text.splitlines():
        if re.match(r"^\[\d+\]:", line):
            header = next(csv.reader([line.split(":", 1)[1]]))
        elif header:
            row = next(csv.reader([line.strip()]))
            if len(row) == len(header):
                rows.append(dict(zip(header, row)))
                header = None
    assert len(rows) == 1, text
    row = rows[0]
    assert tuple(int(row[k]) for k in ("m", "n", "k", "batch_count")) == (
        problem.m,
        problem.n,
        problem.k,
        1,
    )
    for label, expected in (
        ("Solution index", str(problem.solution_index)),
        ("Solution name", problem.solution_name),
        ("kernel name", problem.kernel_name),
    ):
        assert re.findall(rf"--{label}:\s*(.*)", text) == [expected], text
    for key in ("norm_error", "atol", "rtol"):
        assert math.isfinite(float(row[key])) and float(row[key]) >= 0, text
    # When the logging plugin is active, prove that the target kernel ran once.
    # The split reduction and initialization helpers can launch separately.
    if "[rocjitsu] Kernel dispatch:" in text:
        kernels = re.findall(r'\[rocjitsu\] Kernel dispatch: "([^"]+)"', text)
        assert kernels.count(problem.kernel_name) == 1, text
