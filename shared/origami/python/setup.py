# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import nanobind
import glob
import sys
import argparse
import os
from pathlib import Path

# Get ROCM_PATH from environment or use default
ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")
HIPCC_PATH = os.path.join(ROCM_PATH, "bin", "hipcc")

def parse_args():
    parser = argparse.ArgumentParser(description="Origami Python Bindings Setup Script")
    parser.add_argument(
        "--source", "-s",
        type=str,
        default=Path(__file__).parent.parent.resolve() / "src",
        help="Path to origami source directory.",
    )
    args, unknown = parser.parse_known_args()
    return args, unknown

def existing_dirs(paths):
    return [p for p in paths if p and os.path.isdir(p)]

class HIPCCBuildExt(build_ext):
    def build_extensions(self):
        # Use hipcc for compile AND link; force shared link so Python symbols
        # can remain undefined and be resolved at import time.
        if hasattr(self.compiler, "compiler_so"):
            self.compiler.set_executable("compiler_so", HIPCC_PATH)
        if hasattr(self.compiler, "compiler_cxx"):
            self.compiler.set_executable("compiler_cxx", HIPCC_PATH)
        # self.compiler.set_executable("linker_so", HIPCC_PATH) # optional
        super().build_extensions()

        # Distutils/setuptools normally passes -shared, but since we override the linker,
        # make it explicit for hipcc/clang-lld.
        if hasattr(self.compiler, "linker_so"):
            # linker_so can be a list of argv
            self.compiler.linker_so = [HIPCC_PATH, "-shared"]

        super().build_extensions()

if __name__ == "__main__":
    # Preserve unrecognized args for setuptools
    sys.argv = [sys.argv[0]] + unknown

    project_root = Path(__file__).parent.parent.resolve()
    source_dir = Path(args.source)
    include_dir = project_root / "include"
    cpp_files = sorted(glob.glob(str(source_dir / "origami" / "*.cpp")))

    # nanobind runtime source (required)
    nb_base = os.path.dirname(nanobind.include_dir())
    nb_runtime = os.path.join(nb_base, "src", "nb_combined.cpp")

    # Build list: your module first, then nanobind runtime, then project sources
    cpp_files = ["origami_module.cpp", nb_runtime] + cpp_files

    # Likely TBB locations
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    tbb_dirs = existing_dirs([
        "/usr/lib/x86_64-linux-gnu",            # Debian/Ubuntu oneTBB (libtbb.so.12)
        "/usr/local/lib",
        os.path.join(conda_prefix, "lib") if conda_prefix else "",
        os.path.join(ROCM_PATH, "lib"),
    ])

    # RPATHs so import can find libs without LD_LIBRARY_PATH
    rpaths = ["-Wl,-rpath,'$ORIGIN'"]
    rpaths += [f"-Wl,-rpath,'{d}'" for d in tbb_dirs]
    rpaths.append(f"-Wl,-rpath,'{os.path.join(ROCM_PATH, 'lib')}'")

    print("Include dir:", include_dir)
    print("ROCm include:", os.path.join(ROCM_PATH, "include"))
    if conda_prefix:
        print("Conda prefix:", conda_prefix)
    print("TBB lib dirs considered:", tbb_dirs)

    ext_modules = [
        Extension(
            "origami",
            sources=cpp_files,
            include_dirs=[
                nanobind.include_dir(),
                str(include_dir),
                os.path.join(ROCM_PATH, "include"),
                os.path.join(nb_base, "ext", "robin_map", "include"), # Add nanobind's external dependencies
            ],
            language="c++",
            extra_compile_args=[
                "-D__HIP_PLATFORM_AMD__",
                "-fPIC",
                "-std=c++17",
                "-O3", 
                "-Wall",
                "-DNB_SHARED",
                "-fvisibility=hidden",
            ],
            # List tbb here, but also force it via extra_link_args for hipcc
            libraries=["tbb"],
            library_dirs=tbb_dirs + [os.path.join(ROCM_PATH, "lib")],
            extra_link_args=[
                f"-L{os.path.join(ROCM_PATH, 'lib')}",
                "-pthread",
                "-Wl,--no-as-needed",
                "-ltbb",                 # ensure TBB is pulled in
                *rpaths,
            ],
        ),
    ]
    
    setup(
        name="origami",
        ext_modules=ext_modules,
        cmdclass={"build_ext": HIPCCBuildExt},
        setup_requires=["nanobind>=2.0.0"],
        install_requires=["nanobind>=2.0.0"],
    )
