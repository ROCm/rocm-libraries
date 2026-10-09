# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate MX candidates and execute them against the client's CPU reference.

Runs in the gfx950 common-test lane, or under a gfx950 rocJITsu configuration.
Only one small kernel is built per input type; no production device library is
needed. Build-only/use-cache runs share the normal common-test artifact helpers.
"""

import argparse
import csv
import os
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
from tensilelite.Utilities.gemm_candidates import load_library_metadata, tensile_configs

pytestmark = [pytest.mark.gfx950, pytest.mark.emu_fast, pytest.mark.gpu, pytest.mark.gemm]


def _logic(dtype):
    """A 64x64 tile with DepthU=256: small enough for CPU emulation."""
    logic = {
        "MinimumRequiredVersion": "5.0.0",
        "ScheduleName": "gfx950",
        "ArchitectureName": "gfx950",
        "DeviceNames": ["Device 75a0"],
        "LibraryType": "Prediction",
        "ProblemType": {
            "OperationType": "GEMM",
            "DataType": dtype,
            "DestDataType": "S",
            "ComputeDataType": "S",
            "HighPrecisionAccumulate": True,
            "TransposeA": True,
            "TransposeB": False,
            "Batched": True,
            "UseBeta": True,
            "UseBias": 1 if dtype == "F8" else 0,
            "ActivationType": "hipblaslt_all" if dtype == "F8" else "none",
            "MXBlockA": 32,
            "MXBlockB": 32,
        },
        "Solutions": [
            {
                "SolutionIndex": 0,
                "KernelLanguage": "Assembly",
                # Library logic uses the expanded matrix-instruction fields.
                "MatrixInstruction": [16, 16, 128, 1],
                "EnableMatrixInstruction": True,
                "MIBlock": [16, 16, 128, 1, 1, 1],
                "MIWaveGroup": [2, 2],
                "MIWaveTile": [2, 2],
                "WorkGroup": [32, 8, 1],
                "MIInputPerThread": 32,
                "MIInputPerThreadA": 32,
                "MIInputPerThreadB": 32,
                "MIInputPerThreadMXSA": 1,
                "MIInputPerThreadMXSB": 1,
                "MIInputPerThreadMetadata": 32,
                "DepthU": 256,
                "PrefetchGlobalRead": 1,
                "PrefetchLocalRead": 1,
                "ScheduleIterAlg": 3,
                "DirectToLds": 1,
                "TransposeLDS": 1,
                "1LDSBuffer": 0,
                "SourceSwap": False,
                "TileProcessingStrategy": "StreamK",
                "WorkAssignment": "StaticGrid",
                "StaggerU": 0,
                "UseSubtileImpl": True,
                "WorkGroupMapping": 1,
                "AssertSummationElementMultiple": 32,
            }
        ],
    }
    logic["ProblemType"] = state(ProblemType(logic["ProblemType"], False).state)
    return logic


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("F4", marks=pytest.mark.Float4),
        pytest.param("F8", marks=pytest.mark.Float8),
    ],
)
def test_mx_candidates(dtype, tmp_path, pytestconfig, tensile_args, worker_lock_instance):
    build_only = pytestconfig.getoption("--build-only")
    use_cache = pytestconfig.getoption("--use-cache")
    targets = pytestconfig.getoption("--gpu-targets")
    archs = targets.split(";") if build_only else get_available_archs()
    if "gfx950" not in archs:
        pytest.skip("requires gfx950 (hardware or emulator)")

    parser = argparse.ArgumentParser()
    Tensile.addCommonArguments(parser)
    parser.add_argument("--gpu-targets")
    options = parser.parse_args(tensile_args)
    artifact_name = f"gemm_candidates_mx_{dtype}_gfx950"
    output = tmp_path / artifact_name
    artifact_dir = pytestconfig.getoption("--artifact-dir")
    if use_cache:
        extract_artifact(Path(artifact_dir) / f"{artifact_name}.tar.gz", output)
    else:
        logic_dir = tmp_path / "logic"
        logic_dir.mkdir()
        (logic_dir / f"gfx950_{dtype}.yaml").write_text(yaml.safe_dump(_logic(dtype)))
        command = [
            sys.executable,
            str(Path(Tensile.__file__).parent / "bin/TensileCreateLibrary"),
            str(logic_dir),
            str(output),
            "HIP",
            "--architecture",
            "gfx950",
            "--library-format",
            "msgpack",
            "--jobs",
            "1",
            "--no-enumerate",
            "--no-generate-solution-table",
        ]
        if options.CxxCompiler:
            command += ["--cxx-compiler", options.CxxCompiler]
        # The installed artifact and the source checkout both work in children.
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
        subprocess.run(command, check=True, env=env, timeout=120)
        if build_only:
            compress_output(str(output), dest_dir=artifact_dir, name=artifact_name)
            return

    library = load_library_metadata(output / "library", target="gfx950")
    assert len(library.indices) == 1
    cases = library.get_problems(library.indices[0], 10)
    assert len(cases) == 10
    assert {(c.alpha, c.beta) for c in cases} == {(1, 0), (2, 0), (1, 1), (2, 1)}
    if dtype == "F8":
        assert any(c.profile.use_bias or c.activation != "None" for c in cases)
    records, logs = [], []
    with worker_lock_instance:
        for i, ini in enumerate(tensile_configs(cases, results_file=tmp_path / "results.csv")):
            config = tmp_path / f"candidates-{i}.ini"
            config.write_text(ini)
            with (tmp_path / f"client-{i}.log").open("w+") as log:
                result = subprocess.run(
                    [options.prebuilt_client, "--config-file", str(config)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=90,
                )
                log.seek(0)
                text = log.read()
            assert result.returncode == 0, text
            assert "\nRACE " not in "\n" + text, text
            logs.append(text)
            rows = list(csv.reader(text.splitlines()))
            header = next(row for row in rows if row[:2] == ["run", "problem-progress"])
            records.extend(dict(zip(header, row)) for row in rows if row and row[0] == "0")
    text = "\n".join(logs)
    expected = {case.shape.tensile_sizes: case for case in cases}
    assert len(records) == len(expected), text
    observed = set()
    for record in records:
        shape = tuple(map(int, record["problem-sizes"].strip("()").split(",")))
        case = expected[shape]
        assert shape not in observed, text
        assert int(record["solution-progress"].split("/")[0]) == case.solution_index, text
        assert record["solution"] == case.solution_name, text
        assert record["validation"] == "PASSED", text
        observed.add(shape)
