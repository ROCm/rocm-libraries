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

# Unit tests for the level2 area: SpMV, SpSV, BSR products and solves, iterative
# triangular solves, plus the sparse vector rotations and dot products and the
# csrcolor reordering. unit_test_device_level2_common.hpp is shared by these files.
# Included from clients/unittests/CMakeLists.txt.

list(APPEND ROCSPARSE_UNIT_TEST_DEVICE_SOURCES
  unit_test_device_level2_bsr.cpp
  unit_test_device_level2_bsrsv.cpp
  unit_test_device_level2_csrcolor.cpp
  unit_test_device_level2_csrmv.cpp
  unit_test_device_level2_dispatch.cpp
  unit_test_device_level2_formats.cpp
  unit_test_device_level2_itsv.cpp
  unit_test_device_level2_legacy.cpp
  unit_test_device_level2_level1.cpp
  unit_test_device_level2_mv.cpp
  unit_test_device_level2_spsv.cpp
)
