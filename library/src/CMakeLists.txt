# ########################################################################
# Copyright (C) 2018 Advanced Micro Devices, Inc. All rights Reserved.
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

# hipSPARSE source
if(NOT USE_CUDA)
  # hipSPARSE source
  FILE(GLOB_RECURSE AMD_DETAIL_SOURCE src/amd_detail/*.cpp)
  set(hipsparse_source ${AMD_DETAIL_SOURCE})
else()
  # hipSPARSE CUDA source
  FILE(GLOB_RECURSE NVIDIA_DETAIL_SOURCE src/nvidia_detail/*.cpp)
  set(hipsparse_source ${NVIDIA_DETAIL_SOURCE})
endif()

# hipSPARSE Fortran source
set(hipsparse_fortran_source src/hipsparse.f90 src/hipsparse_enums.f90)
