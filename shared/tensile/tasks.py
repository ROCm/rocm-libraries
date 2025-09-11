from invoke.tasks import task
import os

dir = "build_hostlibtest"

def cmake_configure(c, coverage):
    cov = "ON" if coverage else "OFF"
    command = (
            "cmake "
            f"-B `pwd`/{dir} "
            "-S `pwd`/HostLibraryTests "
            "-DCMAKE_BUILD_TYPE=Debug "
            "-DCMAKE_CXX_COMPILER=amdclang++ "
            '-DCMAKE_CXX_FLAGS="-D__HIP_HCC_COMPAT_MODE__=1" '
            "-DTensile_CPU_THREADS=16 "
            "-DTensile_ROOT=`pwd`/Tensile "
            "-DTensile_VERBOSE=1 "
            f"-DTENSILE_ENABLE_COVERAGE={cov}"
    )
    c.run(command, pty=True)

def cmake_build(c):
    c.run(f"cmake --build `pwd`/{dir} -j4", pty=True)

def run_tests(c, coverage):
    if coverage:
        c.run(f"cmake --build `pwd`/{dir} --target coverage --parallel", pty=True)
    else:
        c.run("./{dir}/TensileTests")

def clean_build(c):
    c.run(f"rm -rf {dir}")

@task(
    help={
        "clean": "Remove the build directory before building.",
        "configure": "Run CMake configuration step.",
        "build": "Compile the Tensile HostLib tests.",
        "run": "Run tests or generate coverage depending on the flag.",
        "coverage": "Enable code coverage and reporting.",
    }
)
def hostlibtest(c, clean=False, configure=False, build=False, run=False, coverage=False):
    if clean:
        clean_build(c)
    if configure:
        cmake_configure(c, coverage)
    if build:
        cmake_build(c)
    if run:
        run_tests(c, coverage)

@task(
    help={
        "clean": "Remove the client build directory before building.",
        "configure": "Run CMake configuration for the client.",
        "build": "Compile the tensile-client executable.",
        "build_type": "CMake build type (e.g. Release, Debug).",
        "gpu_targets": "Comma-separated list of GPU targets (e.g. gfx90a,gfx1101)."
    }
)
def build_client(c, clean=False, configure=True, build=True, build_type="Release", gpu_targets="all"):
    client_build_dir = "build/client"

    if clean and os.path.exists(client_build_dir):
        c.run(f"rm -rf {client_build_dir}")

    if configure:
        os.makedirs(client_build_dir, exist_ok=True)

        cmake_cmd = [
            "cmake",
            "-S", "next-cmake",
            "-B", client_build_dir,
            "-DCMAKE_PREFIX_PATH=/opt/rocm",
            "-DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            "-DTENSILE_ENABLE_CLIENT=ON",
            "-DTENSILE_ENABLE_HOST=ON",
            "-DTENSILE_ENABLE_DEVICE=OFF",
            f"-DGPU_TARGETS={gpu_targets}"
        ]

        c.run(" ".join(cmake_cmd))

    if build:
        c.run(f"cmake --build {client_build_dir} --parallel")
