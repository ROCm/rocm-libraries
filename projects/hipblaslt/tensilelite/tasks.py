# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

from invoke.tasks import task
import os

def detect_gpu_arch():
    import subprocess
    import sys
    try:
        result = subprocess.run(["rocm_agent_enumerator", "-v"], capture_output=True, text=True, timeout=5, check=True)
        if result.returncode == 0:
            target = next((line.strip() for line in result.stdout.splitlines() if line.startswith("gfx") and line.strip() != "gfx000"), None)
            if target:
                return target
    except FileNotFoundError:
        print("Error: 'rocm_agent_enumerator' command not found. Please install ROCm.", file=sys.stderr)

    except subprocess.TimeoutExpired:
        print("Error: GPU detection timed out. Hardware might be unresponsive.", file=sys.stderr)

    except Exception as e:
        print(f"An unexpected error occurred during GPU detection: {e}", file=sys.stderr)

    print(f"Failed to detect a valid GPU architecture (gfx target not found).", file=sys.stderr)
    return None

@task
def get_gpu_arch(c):
    print(detect_gpu_arch())

@task(
    help={
        "clean": "Remove the client build directory before building.",
        "configure": "Run CMake configuration for the client.",
        "build": "Build the tensilelite-client executable.",
        "build_dir": "Path to client build dir.",
        "build_type": "CMake build type (e.g. Release, Debug).",
        "gpu_targets": "Comma-separated list of GPU targets (e.g. gfx90a,gfx1101).",
        "enable_rocprof": "Build tensilelite-client with rocprof.",
    }
)
def build_client(c, clean=False, configure=True, build=True, build_dir="build_tmp", build_type="Release", gpu_targets=None, enable_rocprof=False):

    if gpu_targets is None:
        gpu_targets = detect_gpu_arch()
        if gpu_targets == "None":
            print("Error: No GPU detected and no gpu_targets provided. Skipping build.")
            return
        print(f"warning: No GPU targets specified. Detected and using: {gpu_targets}")

    if clean and os.path.exists(build_dir):
        c.run(f"rm -rf {build_dir}")

    if configure:
        os.makedirs(build_dir, exist_ok=True)

        cmake_cmd = [
            "cmake",
            "--preset",
            "tensilelite",
            "-S", "../",
            "-B", build_dir,
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DGPU_TARGETS={gpu_targets}",
            f"-DTENSILELITE_CLIENT_ENABLE_ROCPROFSDK={enable_rocprof}",
        ]

        c.run(" ".join(cmake_cmd))

    if build:
        c.run(f"cmake --build {build_dir} --parallel")
