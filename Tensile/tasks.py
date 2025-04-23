from invoke.tasks import task


@task
def hostlibtest(c, clean=False, configure=False, build=False, run=False):
    dir = "build_hostlibtest"

    if clean:
        c.run(f"rm -rf {dir}")
    if configure:
        c.run(
            "cmake "
            f"-B `pwd`/{dir} "
            "-S `pwd`/HostLibraryTests "
            "-DCMAKE_BUILD_TYPE=Debug "
            "-DCMAKE_CXX_COMPILER=amdclang++ "
            '-DCMAKE_CXX_FLAGS="-D__HIP_HCC_COMPAT_MODE__=1" '
            "-DTensile_CPU_THREADS=8 "
            "-DTensile_ROOT=`pwd`/Tensile "
            "-DTensile_VERBOSE=1", pty=True
        )
    if build:
        c.run(f"cmake --build `pwd`/{dir} -j4", pty=True)
    if run:
        c.run("./{dir}/TensileTest")
