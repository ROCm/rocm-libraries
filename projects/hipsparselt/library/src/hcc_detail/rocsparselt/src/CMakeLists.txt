# ########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
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

set(DL_LIB dl)

if( NOT BUILD_WITH_TENSILE)
  include(src/hcc_detail/rocsparselt/src/spmm/hip/CMakeLists.txt)
  include(src/hcc_detail/rocsparselt/src/spmm/kernels/CMakeLists.txt)
endif()

# rocSPARSELt source
set(rocsparselt_source
  src/hcc_detail/rocsparselt/src/handle.cpp
  src/hcc_detail/rocsparselt/src/Debug.cpp
  src/hcc_detail/rocsparselt/src/status.cpp
  src/hcc_detail/rocsparselt/src/utility.cpp
  src/hcc_detail/rocsparselt/src/rocsparselt_auxiliary.cpp

# spmm
  src/hcc_detail/rocsparselt/src/spmm/rocsparselt_compress.cpp
  src/hcc_detail/rocsparselt/src/spmm/rocsparselt_prune.cpp
  src/hcc_detail/rocsparselt/src/spmm/rocsparselt_spmm.cpp
  ${SPMM_KERNELS_SRC}
  ${KERNEL_LAUNCHER_SRC}
  ${Tensile_SRC}
)
