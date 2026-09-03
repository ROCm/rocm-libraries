# MIT License
#
# Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

include(dependencies/rocm-cmake)
include(dependencies/monorepo)

include(FetchContent)
if (NOT BUILD_WITH_LIB STREQUAL "CUDA")
  fetch_monorepo_dep(
    PACKAGE rocrand
  )
  get_target_property(
    ROCRAND_LINK_LIBRARIES roc::rocrand INTERFACE_LINK_LIBRARIES
  )
  string(FIND "${ROCRAND_LINK_LIBRARIES}" "TBB::tbb" ROCRAND_REQUIRES_TBB)
  if (ROCRAND_REQUIRES_TBB GREATER_EQUAL 0)
    message(
      STATUS
      "The found version of rocRAND requires TBB (Thread Building Blocks)"
    )
    find_package(TBB REQUIRED)
  endif()
endif()

if(BUILD_FORTRAN_WRAPPER)
    enable_language(Fortran)
endif()

if(BUILD_TEST)
  include(dependencies/googletest)
endif()

