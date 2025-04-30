################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

# Even though we don't support python 2, this is still packaged sometimes with python 2.
from __future__ import print_function
from os import path

# Hardcoded tensilelite version, also in Tensile/Source/TensileConfigVersion.cmake
__version__ = "4.33.0"

ROOT_PATH: str = path.dirname(__file__)
SOURCE_PATH: str = path.join(ROOT_PATH, "Source")
CUSTOM_KERNEL_PATH: str = path.join(ROOT_PATH, "CustomKernels")

def PrintTensileRoot():
    print(ROOT_PATH, end='')

__all__ = ["__version__", "ROOT_PATH", "SOURCE_PATH", "CUSTOM_KERNEL_PATH"]
