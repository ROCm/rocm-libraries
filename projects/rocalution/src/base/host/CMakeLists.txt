# ########################################################################
# Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

set(HOST_SOURCES
  base/host/host_matrix_csr.cpp
  base/host/host_matrix_mcsr.cpp
  base/host/host_matrix_bcsr.cpp
  base/host/host_matrix_coo.cpp
  base/host/host_matrix_dia.cpp
  base/host/host_matrix_ell.cpp
  base/host/host_matrix_hyb.cpp
  base/host/host_matrix_dense.cpp
  base/host/host_vector.cpp
  base/host/host_conversion.cpp
  base/host/host_affinity.cpp
  base/host/host_io.cpp
  base/host/host_stencil_laplace2d.cpp
  base/host/host_sparse.cpp
  base/host/host_ilut_driver_csr.cpp
)
