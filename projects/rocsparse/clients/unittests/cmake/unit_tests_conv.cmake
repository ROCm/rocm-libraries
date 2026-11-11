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

# Unit tests for the conv area (conversion, pruning, sorting, sparse-matrix descriptors).
# Included from clients/unittests/CMakeLists.txt.

list(APPEND ROCSPARSE_UNIT_TEST_DEVICE_SOURCES
  unit_test_device_conv_internal.cpp
  unit_test_device_conv_extract.cpp
  unit_test_device_conv_sort.cpp
  unit_test_device_conv_spmat.cpp
  unit_test_device_conv_prune.cpp
  unit_test_device_conv_gebsr.cpp
  unit_test_device_conv_descr.cpp
  unit_test_device_conv_hyb.cpp
  unit_test_device_conv_edges.cpp
  unit_test_device_conv_check.cpp
  unit_test_device_conv_dense.cpp
)
