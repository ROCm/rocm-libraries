# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Exercise numerical failure and continuation through the native bench client."""

import argparse
import copy
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
from tensilelite.Utilities.BinaryLibraryIO import iterSolutionRecords

pytestmark = [pytest.mark.emu_fast, pytest.mark.gpu, pytest.mark.gemm]


def _logic():
    # One wave / 16x16 output tile keeps eight reduction blocks cheap to emulate.
    logic = {
        "MinimumRequiredVersion": "5.0.0",
        "ScheduleName": "gfx950",
        "ArchitectureName": "gfx950",
        "DeviceNames": ["Device 75a0"],
        "LibraryType": "Prediction",
        "ProblemType": {
            "OperationType": "GEMM",
            "DataType": "S",
            "DestDataType": "S",
            "ComputeDataType": "S",
            "HighPrecisionAccumulate": False,
            "TransposeA": False,
            "TransposeB": False,
            "Batched": True,
            "UseBeta": True,
        },
        "Solutions": [
            {
                "SolutionIndex": 0,
                "KernelLanguage": "Assembly",
                "MatrixInstruction": [16, 16, 4, 1],
                "EnableMatrixInstruction": True,
                "MIBlock": [16, 16, 4, 1, 1, 1],
                "MIWaveGroup": [1, 1],
                "MIWaveTile": [1, 1],
                "WorkGroup": [16, 4, 1],
                "MIInputPerThread": 1,
                "MIInputPerThreadA": 1,
                "MIInputPerThreadB": 1,
                "DepthU": 32,
                "TileProcessingStrategy": "None",
                "MFMA_BF16_1K": False,
                "GlobalSplitU": 1,
                "GlobalSplitUAlgorithm": "MultipleBuffer",
                "PrefetchGlobalRead": 1,
                "PrefetchLocalRead": 1,
                "ScheduleIterAlg": 3,
                "StaggerU": 0,
                "WorkGroupMapping": 1,
            }
        ],
    }
    logic["ProblemType"] = state(
        ProblemType(logic["ProblemType"], printIndexAssignmentInfo=False).state
    )
    return logic


def _replace_kernel_entry_with_endpgm(library_dir, kernel_name, target, compiler, tmp_path):
    """Make one generated kernel return without writing D, leaving its peer intact."""
    llvm_dir = Path(shutil.which(str(compiler)) or compiler).resolve().parent
    bundler = llvm_dir / "clang-offload-bundler"
    readobj = llvm_dir / "llvm-readobj"
    for number, code_object in enumerate(sorted(library_dir.glob("*.co"))):
        payload = tmp_path / f"device-{number}.hsaco"
        if code_object.read_bytes().startswith(b"\x7fELF"):
            payload.write_bytes(code_object.read_bytes())
        else:
            listed = subprocess.check_output(
                [str(bundler), "--list", "--type=o", f"--input={code_object}"],
                text=True,
            ).splitlines()
            bundle_target = next(
                name for name in listed if name.startswith("hip") and f"--{target}" in name
            )
            subprocess.run(
                [
                    str(bundler),
                    "--unbundle",
                    "--type=o",
                    f"--targets={bundle_target}",
                    f"--input={code_object}",
                    f"--output={payload}",
                ],
                check=True,
            )
        symbols = subprocess.check_output([str(readobj), "--symbols", str(payload)], text=True)
        symbol = next(
            (
                block
                for block in re.findall(r"  Symbol \{(.*?)\n  \}", symbols, re.S)
                if f"Name: {kernel_name} (" in block and "Type: Function" in block
            ),
            None,
        )
        if symbol is None:
            continue
        sections = subprocess.check_output([str(readobj), "--sections", str(payload)], text=True)
        text_section = next(
            block
            for block in re.findall(r"  Section \{(.*?)\n  \}", sections, re.S)
            if "Name: .text (" in block
        )
        address = int(re.search(r"Address: (0x[0-9A-Fa-f]+)", text_section)[1], 16)
        offset = int(re.search(r"Offset: (0x[0-9A-Fa-f]+)", text_section)[1], 16)
        size = int(re.search(r"Size: (0x[0-9A-Fa-f]+|\d+)", text_section)[1], 0)
        entry = int(re.search(r"Value: (0x[0-9A-Fa-f]+)", symbol)[1], 16)
        assert address <= entry < address + size
        location = offset + entry - address
        device = bytearray(payload.read_bytes())
        endpgm = bytes.fromhex("000081bf")  # s_endpgm on gfx950
        assert device[location : location + 4] != endpgm
        device[location : location + 4] = endpgm
        code_object.write_bytes(device)
        return
    pytest.fail(f"Generated code object does not define {kernel_name}")


@pytest.mark.gfx950
def test_bench_continues_after_numerical_failure(
    tmp_path, pytestconfig, tensile_args, worker_lock_instance
):
    """A failed generated kernel must not prevent the next index from running."""
    build_only = pytestconfig.getoption("--build-only")
    use_cache = pytestconfig.getoption("--use-cache")
    targets = pytestconfig.getoption("--gpu-targets")
    archs = targets.split(";") if targets else get_available_archs()
    if "gfx950" not in archs:
        pytest.skip("requires gfx950 hardware or emulator")
    bench = pytestconfig.getoption("--hipblaslt-bench") or shutil.which("hipblaslt-bench")
    if not build_only and not bench:
        pytest.skip("requires hipblaslt-bench on PATH or --hipblaslt-bench")

    parser = argparse.ArgumentParser()
    Tensile.addCommonArguments(parser)
    parser.add_argument("--gpu-targets")
    options = parser.parse_args(tensile_args)
    artifact_name = "bench_continue_s_gfx950"
    output = tmp_path / artifact_name
    artifact_dir = pytestconfig.getoption("--artifact-dir")
    if use_cache:
        extract_artifact(Path(artifact_dir) / f"{artifact_name}.tar.gz", output)
    else:
        logic = _logic()
        second = copy.deepcopy(logic["Solutions"][0])
        second.update(SolutionIndex=1, DepthU=64)
        logic["Solutions"].append(second)
        logic_dir = tmp_path / "logic"
        logic_dir.mkdir()
        (logic_dir / "gfx950.yaml").write_text(yaml.safe_dump(logic))
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
        subprocess.run(
            command,
            check=True,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
            timeout=120,
        )
        if build_only:
            compress_output(str(output), dest_dir=artifact_dir, name=artifact_name)
            return

    library_dir = output / "library" / "gfx950"
    kernels = {
        index: record["kernelName"]
        for shard in sorted(library_dir.glob("*_gfx950.dat*"))
        for index, record in iterSolutionRecords(shard)
    }
    assert set(kernels) == {0, 1}
    assert kernels[0] != kernels[1]
    broken_dir = tmp_path / "broken-library" / "gfx950"
    shutil.copytree(library_dir, broken_dir)
    compiler = options.CxxCompiler or shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++"
    _replace_kernel_entry_with_endpgm(broken_dir, kernels[0], "gfx950", compiler, tmp_path)

    bench_dir = Path(bench).resolve().parent
    config = tmp_path / "continuation.yaml"
    data = tmp_path / "continuation.data"
    # Each fixture kernel has a 16x16 tile; eight DepthU blocks give K=256/512.
    records = []
    for index, k in enumerate((256, 512)):
        records.append(
            {
                "function": "matmul",
                "algo_method": 2,
                "solution_index": index,
                "requested_solution_num": 1,
                "M": 16,
                "N": 16,
                "K": k,
                "batch_count": 1,
                "transA": "N",
                "transB": "N",
                "a_type": "f32_r",
                "b_type": "f32_r",
                "c_type": "f32_r",
                "d_type": "f32_r",
                "compute_type": "c_f32_r",
                "scale_type": "f32_r",
                "alpha": 1,
                "beta": 0,
                "initialization": "rand_int",
                "norm_check": 1,
                "norm_check_assert": True,
                "allclose_check": 1,
                "iters": 0,
                "cold_iters": 0,
                "adaptive": False,
                "use_gpu_timer": False,
                "print_kernel_info": True,
                "lda": 16,
                "ldb": k,
                "ldc": 16,
                "ldd": 16,
                "stride_a": 16 * k,
                "stride_b": 16 * k,
                "stride_c": 256,
                "stride_d": 256,
            }
        )
    config.write_text(yaml.safe_dump(records))
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
    for continue_on_failure in (False, True):
        with worker_lock_instance:
            result = subprocess.run(
                [
                    bench,
                    "--host_side_fill_kernel",
                    *(["--continue-on-failure"] if continue_on_failure else []),
                    "--data",
                    str(data),
                ],
                env={**os.environ, "HIPBLASLT_TENSILE_LIBPATH": str(broken_dir)},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=90,
            )
        text = result.stdout
        (tmp_path / f"continuation-{continue_on_failure}.log").write_text(text)
        assert result.returncode == 1, text
        expected = [
            "[hipblaslt-case-v1] BEGIN 0 0 16 16 1 256",
            "[hipblaslt-case-v1] RESULT 0 FAILED",
            "[hipblaslt-case-v1] END 0 0 FAILED",
        ]
        if continue_on_failure:
            expected += [
                "[hipblaslt-case-v1] BEGIN 1 1 16 16 1 512",
                "[hipblaslt-case-v1] RESULT 1 PASSED",
                "[hipblaslt-case-v1] END 1 1 PASSED",
            ]
        executed = 2 if continue_on_failure else 1
        expected.append(f"[hipblaslt-case-v1] DONE {executed} 1")
        assert re.findall(r"^\[hipblaslt-case-v1\].*$", text, re.M) == expected, text
        assert re.findall(r"--Solution index:\s*(\d+)", text) == [
            str(i) for i in range(executed)
        ], text
        assert re.findall(r"--kernel name:\s*(.*)", text) == [
            kernels[i] for i in range(executed)
        ], text
        assert "\nRACE " not in "\n" + text, text
        if "[rocjitsu] Kernel dispatch:" in text:
            dispatches = re.findall(r'\[rocjitsu\] Kernel dispatch: "([^"]+)"', text)
            assert [dispatches.count(kernels[i]) for i in (0, 1)] == [1, executed - 1], text
