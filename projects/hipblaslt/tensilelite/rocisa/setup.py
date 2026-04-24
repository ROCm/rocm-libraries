# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import glob
import os
import shutil
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeBuild(build_ext):
    def run(self):
        try:
            os.makedirs(self.build_temp, exist_ok=True)
            os.makedirs(self.build_lib, exist_ok=True)
            # Build rocisa/ itself as a standalone CMake project
            source_dir = os.path.abspath(os.path.dirname(__file__))
            rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
            compilerpath = os.path.join(rocm_path, 'bin/amdclang++')
            cmakeargs = [
                "cmake",
                f"-S{source_dir}",
                f"-B{self.build_temp}",
                f"-DCMAKE_CXX_COMPILER={compilerpath}",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-DPython3_EXECUTABLE={sys.executable}",
                "-DHIPBLASLT_BUNDLE_PYTHON_DEPS=ON",
                "-DHIPBLASLT_ENABLE_YAML=ON",
                "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            ]
            if sys.platform == "win32":
                cmakeargs.extend(["-A", "x64"])
            subprocess.check_call(cmakeargs)
            subprocess.check_call([
                "cmake", "--build", self.build_temp,
                "--config", "Release", "-j", "8",
            ])

            # _rocisa lands in {build_temp}/rocisa/_rocisa*.so
            ext_suffix = ".so" if sys.platform != "win32" else ".pyd"
            matches = glob.glob(
                os.path.join(self.build_temp, 'rocisa', f"_rocisa*{ext_suffix}")
            )
            if not matches:
                raise FileNotFoundError(
                    f"Could not find _rocisa extension in {self.build_temp}/rocisa/. "
                    f"Files present: {os.listdir(os.path.join(self.build_temp, 'rocisa'))}"
                )
            build_module = matches[0]

            # Install into the rocisa package directory inside build_lib
            rocisa_dir = os.path.join(self.build_lib, 'rocisa')
            os.makedirs(rocisa_dir, exist_ok=True)
            shutil.copy2(build_module, os.path.join(rocisa_dir, os.path.basename(build_module)))

            # Copy __init__.py alongside the extension
            init_src = os.path.join(source_dir, 'rocisa', '__init__.py')
            shutil.copy2(init_src, os.path.join(rocisa_dir, '__init__.py'))

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while building with CMake: {e}")
            assert 0

setup(
    name='rocisa',
    version="0.1.0",
    packages=["rocisa"],
    ext_modules=[Extension("rocisa._rocisa", sources=[])],
    install_requires=["nanobind"],
    cmdclass={"build_ext": CMakeBuild},
    author='YangWen Huang',
    author_email='yangwen.huang@amd.com',
    description='A C++ code generator for ROCm ISA',
    long_description=open('README.md').read(),
    url='https://github.com/ROCm/hipBLASLt',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
