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

# Unit tests for the host area (bfloat16, BLAS wrappers, auxiliary and
# library-internal host branches).
# Included from clients/unittests/CMakeLists.txt.

# rocsparse_bfloat16.h only defines the full rocsparse_bfloat16 struct when
# compiled as HIP. Build this TU as HIP host code only: no device code object,
# so the binary stays GPU-independent.
target_sources(rocsparse-unit-test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/unit_test_bfloat16.cpp
)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/unit_test_bfloat16.cpp PROPERTIES
  COMPILE_OPTIONS "-x;hip;--offload-host-only"
)

list(APPEND ROCSPARSE_UNIT_TEST_DEVICE_SOURCES
  unit_test_internal_common_blas.cpp
  unit_test_host_auxiliary_branches.cpp
  unit_test_host_internal_branches.cpp
)
