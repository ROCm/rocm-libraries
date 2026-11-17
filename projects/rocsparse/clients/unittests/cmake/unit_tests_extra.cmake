# ########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

# Unit tests for the extra area (SpMM, SpGEMM, SpGEAM, SDDMM, SpSM, SpTRSM,
# bsrsm, BELL, gebsrmm, gemmi, spmat_scale).
# Included from clients/unittests/CMakeLists.txt.

list(APPEND ROCSPARSE_UNIT_TEST_DEVICE_SOURCES
  unit_test_internal_level3_sddmm_branches.cpp
  unit_test_device_extra_bell.cpp
  unit_test_device_extra_bsrgeam.cpp
  unit_test_device_extra_bsrsm.cpp
  unit_test_device_extra_csrgemm_bsrgemm.cpp
  unit_test_device_extra_geam.cpp
  unit_test_device_extra_gebsrmm.cpp
  unit_test_device_extra_gemm_geam.cpp
  unit_test_device_extra_gemmi.cpp
  unit_test_device_extra_sddmm.cpp
  unit_test_device_extra_spgeam.cpp
  unit_test_device_extra_spgemm.cpp
  unit_test_device_extra_spmm.cpp
  unit_test_device_extra_spsm.cpp
)
if(ROCSPARSE_ENABLE_SPMAT_SCALE)
  list(APPEND ROCSPARSE_UNIT_TEST_DEVICE_SOURCES unit_test_device_extra_spmat_scale.cpp)
endif()
if(BUILD_WITH_TRSM_REFACTORING)
  list(APPEND ROCSPARSE_UNIT_TEST_DEVICE_SOURCES unit_test_device_extra_sptrsm.cpp)
endif()
