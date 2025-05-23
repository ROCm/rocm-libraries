# ########################################################################
# Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

if (DEFINED ENV{ROCM_PATH})
  set(rocm_bin "$ENV{ROCM_PATH}/bin")
else()
  set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to the ROCm installation.")
  set(rocm_bin "/opt/rocm/bin")
endif()

if (NOT DEFINED ENV{CXX})
  set(CMAKE_CXX_COMPILER "${rocm_bin}/amdclang++")
else()
  set(CMAKE_CXX_COMPILER "$ENV{CXX}")
endif()

if (NOT DEFINED ENV{CC})
  set(CMAKE_C_COMPILER "${rocm_bin}/amdclang")
else()
  set(CMAKE_C_COMPILER "$ENV{CC}")
endif()
