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
            source_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
            rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
            compilerpath = os.path.join(rocm_path, 'bin/amdclang++')
            cmakeargs = [
                "cmake",
                "--preset rocisa",
                f"-S{source_dir}",
                f"-B{self.build_temp}",
                f"-DCMAKE_CXX_COMPILER={compilerpath}",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DPython_EXECUTABLE={sys.executable}",
            ]
            if sys.platform == "win32":
                cmakeargs.extend(["-A", "x64"])
            subprocess.check_call(cmakeargs)
            make_args = [
                "cmake",
                "--build",
                self.build_temp,
                "--config",
                "Release",
                "-j",
                "8",
            ]
            subprocess.check_call(make_args)
            ext_suffix = ".so" if sys.platform != "win32" else ".pyd"
            build_module = glob.glob(os.path.join(self.build_temp, 'tensilelite', 'rocisa', "lib", f"rocisa*{ext_suffix}"))[0]
            target_module = os.path.join(self.build_lib, os.path.basename(build_module))
            shutil.copy2(build_module, target_module)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while building with CMake: {e}")
            assert 0

setup(
    name='rocisa',
    version="0.1.0",
    packages=["rocisa"],
    ext_modules=[Extension("rocisa", sources=[])],
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
    python_requires='>=3.8',
)
