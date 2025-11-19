################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import os
import subprocess
import yaml
import shutil
import argparse
from io import StringIO

yaml_data = "- {function: matmul, M: 160, N: 64, K: 6144, lda: 6144, ldb: 6144, ldc: 160, ldd: 160, stride_a: 0, stride_b: 0, stride_c: 0, stride_d: 0, alpha: 1.0, beta: 0.0, transA: T, transB: N, batch_count: 1, scaleA: 0, scaleB: 0, scaleC: 0, scaleD: 0, swizzleA: false, swizzleB: false, scaleAlpha_vector: false, gradient: false, use_e: false, bias_vector: false, bias_source: d, a_type: bf16_r, b_type: bf16_r, c_type: bf16_r, d_type: bf16_r, scale_type: f32_r, bias_type: f32_r, compute_type: c_f32_r, activation_type: none, flush: false, any_stride: true, rotating: 512, cold_iters: 0, iters: 1, print_kernel_info: true}"


def main(hipblaslt_path, device_id, workspace, arch):
    hipblaslt_bench = os.path.join(
        hipblaslt_path, "build/release/clients/hipblaslt-bench"
    )
    match_table = os.path.join(
        hipblaslt_path, "build/release/device-library/MatchTable.yaml"
    )
    tensilelite_path = os.path.join(hipblaslt_path, "tensilelite")
    tensile_bin = os.path.join(tensilelite_path, "Tensile/bin/")

    # run test gemm
    gemm_log = os.path.join(workspace, "gemm.log")
    with open(f"{gemm_log}", "w") as f:
        subprocess.run(
            [f"{hipblaslt_bench}", "--device", f"{device_id}", "--yaml", "-"],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
            input=yaml_data.encode("utf-8"),
        )

    # extract solutin index and name
    blocks = open(f"{gemm_log}").read().split("[0]:")[1:]
    if not blocks:
        raise ValueError("The benchmark output file may be corrupted.")

    solution_index = (
        blocks[0].split("\n")[2].strip("--Solution index").strip(":")[1:].strip()
    )
    solution_name = (
        blocks[0].split("\n")[3].strip("--Solution name").strip("Cijk")[1:].strip()
    )

    if not solution_name or not solution_index:
        raise ValueError("The benchmark output file may be corrupted.")

    # find matching library from solution index in gemm.log
    table = open(match_table, "r").read().split(f"{solution_index}:")[1:]
    line = table[0].strip(":").split("\n")[0:2]
    lib_name = line[0].strip("[")[2:-1]
    internal_solution_index = line[1].strip("]").strip()
    print(internal_solution_index)

    # generate config yaml from library
    config_yaml = os.path.join(workspace, "config.yaml")
    subprocess.run(
        [
            f"{tensile_bin}/TensileLibLogicToYaml",
            "--input",
            f"{lib_name}",
            "-d",
            f"{internal_solution_index}",
            "--output",
            f"{config_yaml}",
        ],
        stderr=subprocess.STDOUT,
        check=True,
    )

    # if library was origami, we need to replace exact solution
    data = yaml.safe_load(StringIO(yaml_data))

    old_pattern = r"Exact: \[1, 1, 1, 1\]"
    new_pattern = rf"Exact: \[{data[0]['M']}, {data[0]['N']}, 1, {data[0]['K']}\]"

    config_yaml = os.path.join(workspace, f"config_{internal_solution_index}.yaml")
    subprocess.run(
        [
            "sed",
            "-i",
            f"s/{old_pattern}/{new_pattern}/",
            f"{config_yaml}",
        ],
        stderr=subprocess.STDOUT,
        check=True,
    )

    # if tensile-client does not exist, build it
    build_dir = os.path.join(workspace, "build_tmp")
    client_path = os.path.join(build_dir, "tensilelite/client/tensilelite-client")

    if not os.path.isfile(client_path):
        print("Building tensilelite client...")
        subprocess.call(
            ["invoke", "build-client", "--build-dir", build_dir], cwd=tensilelite_path
        )

    # call Tensile to generate liblogic
    work_dir = os.path.join(workspace, "WDirDevice")
    shutil.rmtree(work_dir, ignore_errors=True)
    subprocess.run(
        [
            f"{tensile_bin}/Tensile",
            "--prebuilt-client",
            f"{client_path}",
            f"{config_yaml}",
            f"{work_dir}",
        ],
        stderr=subprocess.STDOUT,
        check=True,
    )

    # create a library from the new library logic
    new_liblogic_path = os.path.join(work_dir, "3_LibraryLogic")
    new_tensile_path = os.path.join(workspace, "tensile")
    subprocess.run(
        [
            f"{tensile_bin}/TensileCreateLibrary",
            "--code-object-version",
            "4",
            "--library-format",
            "msgpack",
            "--architecture",
            arch,
            os.path.abspath(new_liblogic_path),
            new_tensile_path,
            "HIP",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        check=True,
        text=True,
    )

    # run hblt again
    os.environ["HIPBLASLT_TENSILE_LIBPATH"] = f"{new_tensile_path}/library"
    gemm_out_log = os.path.join(workspace, "gemm_out.log")
    with open(f"{gemm_out_log}", "w") as f:
        subprocess.run(
            [f"{hipblaslt_bench}", "--device", f"{device_id}", "--yaml", "-"],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
            input=yaml_data.encode("utf-8")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Test utility for testing TensileLibLogicToYaml script."
    )

    parser.add_argument("--hipblaslt", help="Path to hipBLASLt", type=str)
    parser.add_argument(
        "--workspace",
        help="Path to the working space, all files will be saved here. Default is current dir.",
        type=str,
        default="",
    )
    parser.add_argument(
        "--device", help="Device to run the initial benchmarks on.", type=int, default=0
    )
    parser.add_argument(
        "--arch",
        help="Architecture. Support only gfx950 currently. Default is gfx950",
        type=str,
        default="gfx950",
    )

    args = parser.parse_args()

    hipblaslt_path = args.hipblaslt
    arch = args.arch
    device_id = args.device
    workspace = os.path.abspath(args.workspace)
    os.makedirs(workspace, exist_ok=True)

    main(hipblaslt_path, device_id, workspace, arch)
