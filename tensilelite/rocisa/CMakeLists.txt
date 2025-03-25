# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
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
cmake_minimum_required(VERSION 3.15)
project(rocisa LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# find nanobind
execute_process(
    COMMAND python3 -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    ERROR_VARIABLE error
    RESULT_VARIABLE result
)
if(result)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env python3 -m pip install nanobind
        ERROR_VARIABLE error
        RESULT_VARIABLE result
    )
    if(result)
        message(FATAL_ERROR "Failed to install nanobind via pip: ${error}")
    endif()
    execute_process(
        COMMAND python3 -m nanobind --cmake_dir
        OUTPUT_VARIABLE nanobind_DIR
        ERROR_VARIABLE error
        RESULT_VARIABLE result
    )
endif()
message(STATUS "Nanobind path: ${nanobind_DIR}")
string(REPLACE "\n" "" nanobind_DIR "${nanobind_DIR}")

if(DEFINED PYTHON_VERSION AND PYTHON_VERSION)
    message(STATUS "Manually set Python version to ${PYTHON_VERSION}")
else()
    set(PYTHON_VERSION 3)
endif()

# Use cmake -Dpybind11_DIR=$(python3 -m nanobind --cmakedir) .. to set pybind dir
# Use cmake -DCMAKE_PREFIX_PATH=$(python3 -m nanobind --cmake_dir) .. to set nanobind

find_package(Python ${PYTHON_VERSION} COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED HINTS ${nanobind_DIR})

set(ROCISAINST_SOURCE src/instruction/instruction.cpp
                      src/instruction/common.cpp
                      src/instruction/branch.cpp
                      src/instruction/cmp.cpp
                      src/instruction/cvt.cpp
                      src/instruction/mem.cpp
                      src/instruction/mfma.cpp)

include_directories(${CMAKE_SOURCE_DIR}/include)
nanobind_add_module(rocisa NOMINSIZE src/main.cpp
                                     src/base.cpp
                                     src/code.cpp
                                     src/container.cpp
                                     src/count.cpp
                                     src/enum.cpp
                                     src/label.cpp
                                     ${ROCISAINST_SOURCE})
