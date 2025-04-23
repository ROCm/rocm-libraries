# Copyright (c) 2018 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from setuptools import setup, find_packages, Extension
from setuptools.dist import Distribution
import sys
import os

if sys.version_info < (3, 0):
    sys.exit('rocal Python Package requires Python > 3.0')

ROCM_PATH = '/opt/rocm'
if "ROCM_PATH" in os.environ:
    ROCM_PATH = os.environ.get('ROCM_PATH')
print("\nROCm PATH set to -- "+ROCM_PATH+"\n")

class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    @classmethod
    def has_ext_modules(self):
        return True

# Version updates - IMP: Change in version requires to match top level CMakeLists.txt
## * getrocALWheelname.py
## * setup.py
setup(
    name='amd-rocal',
    description='AMD ROCm Augmentation Library Python Bindings',
    url='https://github.com/ROCm/rocAL',
    version='2.3.0',
    author='AMD',
    license='MIT',
    packages=find_packages(where='@TARGET_NAME@'),
    package_dir={'amd': '@TARGET_NAME@/amd'},
    include_package_data=True,
    ext_modules=[Extension('rocal_pybind',
                    sources=['rocal_pybind.cpp'], 
                    include_dirs=['@pybind11_INCLUDE_DIRS@', ROCM_PATH+'/include', '@PROJECT_SOURCE_DIR@/../rocAL/include/api'])],
    distclass=BinaryDistribution
)
