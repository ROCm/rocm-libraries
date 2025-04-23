#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#####################################################################################

include(CheckCXXSourceCompiles)

function(find_parallel_stl_check RESULT)
    set(CMAKE_REQUIRED_LIBRARIES ${ARGN})
    set(CMAKE_REQUIRED_FLAGS)
    if(NOT MSVC)
        set(CMAKE_REQUIRED_FLAGS "-std=c++17")
    endif()
    string(MD5 _flags_hash "${CMAKE_REQUIRED_FLAGS} ${CMAKE_REQUIRED_LIBRARIES}")
    set(_source "
#include <execution>

#ifdef _PSTL_PAR_BACKEND_SERIAL
#error \"Using serial backend\"
#endif

int main() {
    int* i = nullptr;
    std::sort(std::execution::par, i, i);
}
")
    check_cxx_source_compiles("${_source}" _has_execution_${_flags_hash})
    set(${RESULT} ${_has_execution_${_flags_hash}} PARENT_SCOPE)
endfunction()

set(ParallelSTL_FOUND Off)
set(ParallelSTL_LIBRARIES)
set(ParallelSTL_USES_TBB Off)
find_parallel_stl_check(ParallelSTL_HAS_EXECUTION_PAR)
if(ParallelSTL_HAS_EXECUTION_PAR)
    set(ParallelSTL_FOUND On)
else()
    find_package(TBB QUIET)
    if(TARGET TBB::tbb)
        find_parallel_stl_check(ParallelSTL_TBB_HAS_EXECUTION_PAR TBB::tbb)
        if(ParallelSTL_TBB_HAS_EXECUTION_PAR)
            set(ParallelSTL_USES_TBB On)
            set(ParallelSTL_LIBRARIES TBB::tbb)
            message(STATUS "Using TBB for parallel execution")
        endif()
    endif()
endif()

foreach(VAR ParallelSTL_FOUND ParallelSTL_LIBRARIES ParallelSTL_USES_TBB)
    string(TOUPPER ${VAR} ParallelSTL_VAR)
    set(${ParallelSTL_VAR} ${${VAR}})
endforeach()

