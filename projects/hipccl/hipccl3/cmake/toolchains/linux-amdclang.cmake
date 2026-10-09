# MIT License
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# ########################################################################

set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
set(CMAKE_PREFIX_PATH "${ROCM_PATH}" CACHE PATH "Search path for ROCm packages")

set(ROCM_LLVM_PATH "${ROCM_PATH}/lib/llvm")
set(CMAKE_C_COMPILER "${ROCM_LLVM_PATH}/bin/amdclang" CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "${ROCM_LLVM_PATH}/bin/amdclang++" CACHE FILEPATH "C++/HIP compiler")
set(CMAKE_Fortran_COMPILER "${ROCM_LLVM_PATH}/bin/flang" CACHE FILEPATH "Fortran compiler")

set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Enable position independent code")
