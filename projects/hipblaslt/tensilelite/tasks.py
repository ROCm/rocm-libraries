from invoke.tasks import task
import os

@task(
    help={
        "clean": "Remove the client build directory before building.",
        "configure": "Run CMake configuration for the client.",
        "build": "Build the tensilelite-client executable.",
        "build_dir": "Path to client build dir.",
        "build_type": "CMake build type (e.g. Release, Debug).",
        "gpu_targets": "Comma-separated list of GPU targets (e.g. gfx90a,gfx1101)."
    }
)
def build_client(c, clean=False, configure=True, build=True, build_dir="build_tmp", build_type="Release", gpu_targets="gfx90a"):

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
            f"-DGPU_TARGETS={gpu_targets}"
        ]

        c.run(" ".join(cmake_cmd))

    if build:
        c.run(f"cmake --build {build_dir} --parallel")
