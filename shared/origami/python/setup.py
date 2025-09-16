# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# setup.py
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
    parser.add_argument("--source", "-s", type=str, default=Path(__file__).parent.parent.resolve() / "src", help="Path to origami source directory.")
    args, unknown = parser.parse_known_args()
    return args, unknown


class HIPCCBuildExt(build_ext):
    def build_extensions(self):
        if hasattr(self.compiler, 'compiler_so'):
            self.compiler.set_executable("compiler_so", HIPCC_PATH)
        if hasattr(self.compiler, 'compiler_cxx'):
            self.compiler.set_executable("compiler_cxx", HIPCC_PATH)
        # self.compiler.set_executable("linker_so", HIPCC_PATH) # optional
        super().build_extensions()


if __name__ == "__main__":
    args, unknown = parse_args()
    sys.argv = [sys.argv[0]] + unknown  # Preserve unrecognized arguments for setuptools

    project_root = Path(__file__).parent.parent.resolve()

    source_dir = Path(args.source)
    include_dir = project_root / "include"
    cpp_path = source_dir / "origami" / "*.cpp"
    cpp_files = sorted(glob.glob(str(cpp_path)))

    nanobind_base = os.path.dirname(nanobind.include_dir()) # Get nanobind base path
    
    nanobind_src = os.path.join(nanobind_base, "src", "nb_combined.cpp") # Add nanobind's runtime support source file
    cpp_files = ["origami_module.cpp", nanobind_src] + cpp_files

    print(include_dir)
    print(os.path.join(ROCM_PATH, "include"))

    ext_modules = [
        Extension(
            "origami",
            cpp_files,
            include_dirs=[
                nanobind.include_dir(),
                str(include_dir),
                os.path.join(ROCM_PATH, "include"),
                os.path.join(nanobind_base, "ext", "robin_map", "include"), # Add nanobind's external dependencies
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
            extra_link_args=[f"-L{os.path.join(ROCM_PATH, 'lib')}"],
        ),
    ]
    
    setup(
        name="origami",
        ext_modules=ext_modules,
        cmdclass={"build_ext": HIPCCBuildExt},
        setup_requires=["nanobind>=2.0.0"],
        install_requires=["nanobind>=2.0.0"],
    )
