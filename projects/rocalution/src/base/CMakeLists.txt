# ########################################################################
# Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights Reserved.
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

set(BASE_SOURCES
  base/base_rocalution.cpp
  base/operator.cpp
  base/vector.cpp
  base/local_matrix.cpp
  base/global_matrix.cpp
  base/local_vector.cpp
  base/global_vector.cpp
  base/base_matrix.cpp
  base/base_vector.cpp
  base/backend_manager.cpp
  base/parallel_manager.cpp
  base/local_stencil.cpp
  base/base_stencil.cpp
)

set(BASE_PUBLIC_HEADERS
  base/matrix_formats.hpp
  base/matrix_formats_ind.hpp
  base/base_rocalution.hpp
  base/operator.hpp
  base/vector.hpp
  base/local_matrix.hpp
  base/global_matrix.hpp
  base/local_vector.hpp
  base/global_vector.hpp
  base/backend_manager.hpp
  base/parallel_manager.hpp
  base/local_stencil.hpp
  base/stencil_types.hpp
)
