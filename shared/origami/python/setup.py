################################################################################
#
# MIT License
#
# Copyright 2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

# setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import nanobind
import glob
import sys
import argparse
import os
from pathlib import Path

# Find ROCM_PATH and determine compiler
def find_rocm_and_compiler():
    """Find ROCM_PATH and determine appropriate compiler for Python bindings."""

    common_paths = [
        os.environ.get("ROCM_PATH", ""),
        "/opt/rocm",
        "/usr/local/rocm",
    ]
    
    rocm_path = None
    for path in common_paths:
        if not path:
            continue
        hip_header = os.path.join(path, "include", "hip", "hip_runtime.h")
        if os.path.exists(hip_header):
            rocm_path = path
            break
    
    if not rocm_path:
        # Fallback to environment variable or default, even if headers not found
        rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        print(f"Warning: Could not verify HIP headers in {rocm_path}, using anyway")
    
    # For Python bindings (host-only code), use regular C++ compiler
    # Check for clang++ or g++ - prefer clang++ as it's more compatible with ROCm headers
    import shutil
    compiler = shutil.which("clang++")
    if not compiler:
        compiler = shutil.which("g++")
    if not compiler:
        raise RuntimeError("Neither clang++ nor g++ found. Please install a C++ compiler.")
    
    return compiler, rocm_path

COMPILER_PATH, ROCM_PATH = find_rocm_and_compiler()


def parse_args():
    parser = argparse.ArgumentParser(description="Origami Python Bindings Setup Script")
    parser.add_argument("--source", "-s", type=str, default=None, help="Path to origami source directory.")
    args, unknown = parser.parse_known_args()
    return args, unknown


class HIPCCBuildExt(build_ext):
    def build_extensions(self):
        if not os.path.exists(COMPILER_PATH) or not os.access(COMPILER_PATH, os.X_OK):
            raise RuntimeError(
                f"C++ compiler not found at {COMPILER_PATH}. "
                f"Please install clang++ or g++."
            )
        if hasattr(self.compiler, 'compiler_so'):
            self.compiler.set_executable("compiler_so", COMPILER_PATH)
        if hasattr(self.compiler, 'compiler_cxx'):
            self.compiler.set_executable("compiler_cxx", COMPILER_PATH)
        super().build_extensions()


if __name__ == "__main__":
    args, unknown = parse_args()
    sys.argv = [sys.argv[0]] + unknown

    # Get the directory where this setup.py is located (python/)
    python_dir = Path(__file__).parent.resolve()
    # Get the parent directory (shared/origami root)
    project_root = python_dir.parent.resolve()
    
    if args.source:
        source_dir = Path(args.source)
    else:
        source_dir = project_root / "src"
    
    include_dir = project_root / "include"
    
    # Get all C++ source files from src/origami/
    cpp_path = source_dir / "origami" / "*.cpp"
    cpp_files = sorted(glob.glob(str(cpp_path)))
    
    # Add the Python module C++ file (in src/origami/)
    origami_module_cpp = python_dir / "src" / "origami" / "bindings.cpp"
    if not origami_module_cpp.exists():
        raise FileNotFoundError(f"Python module file not found: {origami_module_cpp}")
    
    nanobind_base = os.path.dirname(nanobind.include_dir())
    
    nanobind_src = os.path.join(nanobind_base, "src", "nb_combined.cpp")
    cpp_files = [str(origami_module_cpp), nanobind_src] + cpp_files

    print(f"Using C++ compiler: {COMPILER_PATH}")
    print(f"Using ROCM_PATH: {ROCM_PATH}")
    print(f"Include directory: {include_dir}")
    print(f"ROCM include directory: {os.path.join(ROCM_PATH, 'include')}")

    ext_modules = [
        Extension(
            "origami.origami",  # Extension creates origami/origami.so, matches NB_MODULE(origami, m)
            cpp_files,
            include_dirs=[
                nanobind.include_dir(),
                str(include_dir),
                os.path.join(ROCM_PATH, "include"),
                os.path.join(nanobind_base, "ext", "robin_map", "include"),
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
            extra_link_args=[
                f"-L{os.path.join(ROCM_PATH, 'lib')}",
                "-lamdhip64",
            ],
        ),
    ]
    
    readme_path = project_root / "README.md"
    long_description = ""
    if readme_path.exists():
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                long_description = f.read()
        except Exception:
            pass
    
    setup(
        name="origami",
        version="0.1.0",
        description="Analytical GEMM Solution Selection",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Advanced Micro Devices, Inc.",
        license="MIT",
        package_dir={"": "src"},
        packages=["origami"],
        ext_modules=ext_modules,
        cmdclass={"build_ext": HIPCCBuildExt},
        setup_requires=["nanobind>=2.0.0"],
        install_requires=["nanobind>=2.0.0"],
        python_requires=">=3.7",
        zip_safe=False,
    )
