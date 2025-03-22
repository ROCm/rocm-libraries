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
project(rocisa LANGUAGES HIP CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(DEFINED Python_EXECUTABLE AND Python_EXECUTABLE)
    message(STATUS "Manually set Python_EXECUTABLE to ${Python_EXECUTABLE}")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# find nanobind
if(NOT nanobind_FOUND)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
        OUTPUT_VARIABLE nanobind_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE error
        RESULT_VARIABLE result
    )
    if(result)
        message(FATAL_ERROR "Did you forget to install nanobind?")
    endif()
    
    # Use cmake -Dpybind11_DIR=$(python3 -m nanobind --cmakedir) .. to set pybind dir
    # Use cmake -DCMAKE_PREFIX_PATH=$(python3 -m nanobind --cmake_dir) .. to set nanobind
    find_package(nanobind CONFIG REQUIRED HINTS ${nanobind_DIR})
endif()
message(STATUS "Nanobind path: ${nanobind_DIR}")


set(ROCISAINST_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/instruction.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/common.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/branch.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/cmp.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/cvt.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/mem.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/mfma.cpp)

nanobind_add_module(rocisa NOMINSIZE NB_SUPPRESS_WARNINGS 
                        ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/main.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/base.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/code.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/container.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/count.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/enum.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/label.cpp
                        ${ROCISAINST_SOURCE})
target_include_directories(rocisa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/include)
