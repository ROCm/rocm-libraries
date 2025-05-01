################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

################################################################################
# Install Tensile
# - installs python scripts, c++ source files and a few configs
# - creates executables for running benchmarking
# - installs TensileConfig.cmake so one call find_package(Tensile)
################################################################################
from setuptools import setup
import os.path
import re

def readRequirementsFromTxt():
  requirements = []
  with open("requirements.txt") as req_file:
    for line in req_file.read().splitlines():
      if not line.strip().startswith("#"):
        requirements.append(line)
  return requirements

def readVersionFromInit():
    import Tensile
    return Tensile.__version__

setup(
  name="Tensile",
  version=readVersionFromInit(),
  description="An auto-tuning tool for GEMMs and higher-dimensional tensor contractions on GPUs.",
  url="https://github.com/RadeonOpenCompute/Tensile",
  author="Advanced Micro Devices",
  license="MIT",
  install_requires=readRequirementsFromTxt(),
  python_requires='>=3.5',
  packages=["Tensile"],
  package_data={ "Tensile": ["Tensile/cmake/*"] },
  data_files=[ ("cmake", ["Tensile/cmake/TensileConfig.cmake", "Tensile/cmake/TensileConfigVersion.cmake"]) ],
  include_package_data=True,
  entry_points={"console_scripts": [
    # user runs a benchmark
    "Tensile = Tensile.Tensile:main",
    "tensileBenchmarkLibraryClient = Tensile.TensileBenchmarkLibraryClient:main",
    # CMake calls this to create Tensile.lib
    "TensileCreateLibrary = Tensile.TensileCreateLibrary:TensileCreateLibrary",

    "TensileGetPath = Tensile:PrintTensileRoot",
    # automatic benchmarking for rocblas
    "tensile_rocblas_sgemm = Tensile.Tensile:TensileROCBLASSGEMM",
    "tensile_rocblas_dgemm = Tensile.Tensile:TensileROCBLASDGEMM",
    "tensile_rocblas_cgemm = Tensile.Tensile:TensileROCBLASCGEMM",
    "tensile_rocblas_zgemm = Tensile.Tensile:TensileROCBLASZGEMM",
    # automatically find fastest sgemm exhaustive search
    "tensile_sgemm = Tensile.Tensile:TensileSGEMM5760",
    # Run tensile benchmark from cluster
    "TensileBenchmarkCluster = Tensile.TensileBenchmarkCluster:main",
    # Retune library logic file
    "TensileRetuneLibrary = Tensile.TensileRetuneLibrary:main"
    ]}
  )
