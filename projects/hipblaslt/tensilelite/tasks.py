from invoke.tasks import task
import os

@task(
    help={
        "clean": "Remove the client build directory before building.",
        "configure": "Run CMake configuration for the client.",
        "build": "Build the tensilelite-client executable.",
        "build_type": "CMake build type (e.g. Release, Debug).",
        "gpu_targets": "Comma-separated list of GPU targets (e.g. gfx90a,gfx1101)."
    }
)
def build_client(c, clean=False, configure=True, build=True, build_dir=None, build_type="Release", gpu_targets="gfx90a"):
    build_dir = "build_tmp"

    if clean and os.path.exists(client_build_dir):
        c.run(f"rm -rf {build_dir}")

    if configure:
        os.makedirs(build_dir, exist_ok=True)

        cmake_cmd = [
            "cmake",
            "-S", ".",
            "-B", build_dir,
            "-DCMAKE_PREFIX_PATH=/opt/rocm",
            "-DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            "-DHIPBLASLT_ENABLE_CLIENT=OFF", 
            "-DHIPBLASLT_ENABLE_HOST=OFF", 
            "-DHIPBLASLT_ENABLE_DEVICE=OFF",
            "-DTENSILELITE_ENABLE_HOST=ON",
            "-DTENSILELITE_ENABLE_CLIENT=ON",
            "-DHIPBLASLT_ENABLE_LLVM=ON", 
            f"-DGPU_TARGETS={gpu_targets}"
        ]

        c.run(" ".join(cmake_cmd))

    if build:
        c.run(f"cmake --build {build_dir} --parallel")
