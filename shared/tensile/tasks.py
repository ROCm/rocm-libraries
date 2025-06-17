from invoke.tasks import task

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
            "-DTensile_CPU_THREADS=8 "
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
