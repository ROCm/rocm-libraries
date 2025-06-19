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
        "arch": "Specify the GPU architecture to build for (e.g., gfx90a)."
    }
)
def buildclient(c, clean=True, configure=True, build=True, install=False, arch=None):
    client_build_dir = "build/client"

    if clean and os.path.exists(client_build_dir):
        print("Cleaning previous client build directory...")
        c.run(f"rm -rf {client_build_dir}")

    if configure:
        print("Configuring tensile-client...")
        os.makedirs(client_build_dir, exist_ok=True)
        
        rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        
        cmake_cmd = [
            "cmake",
            "-S", "next-cmake",
            "-B", client_build_dir,
            f"-DCMAKE_PREFIX_PATH={rocm_path}",
            f"-DCMAKE_CXX_COMPILER={rocm_path}/bin/amdclang++",
            "-DCMAKE_BUILD_TYPE=Release",
            # "-DTENSILE_ENABLE_CLIENT=ON",
            # "-DTENSILE_ENABLE_HOST=ON",
            # "-DTENSILE_ENABLE_DEVICE=OFF",
            # "-DTENSILE_BUILD_TESTING=OFF",
            "-DGPU_TARGETS=gfx90a"
        ]
        
        # if arch:
        #     cmake_cmd.append(f"-DGPU_TARGETS={arch}")

        c.run(" ".join(cmake_cmd))

    if build:
        print("Building tensile-client...")
        c.run(f"cmake --build {client_build_dir} --parallel")

    if install:
        print("Installing tensile-client...")
        c.run(f"cmake --install {client_build_dir}")

    # print(f"Build complete. Executable at: {client_build_dir}/bin/tensile-client")
