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

if(DEFINED Python_EXECUTABLE AND Python_EXECUTABLE)
    message(STATUS "Manually set Python_EXECUTABLE to ${Python_EXECUTABLE}")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

if(Python_VERSION VERSION_GREATER_EQUAL "3.8")
    message(STATUS "Found Python version: ${Python_VERSION}")
else()
    message(FATAL_ERROR "Python version 3.8 or higher is required, but found version ${Python_VERSION}")
endif()

include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        9b3afa9dbdc23641daf26fadef7743e7127ff92f # v2.6.1
)
FetchContent_MakeAvailable(nanobind)

set(ROCISAINST_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/instruction.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/common.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/branch.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/cmp.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/cvt.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/mem.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/mfma.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/extension.cpp)

set(ROCISAFUNC_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/functions.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/argument.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/f_math.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/f_branch.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/f_cast.cpp)

set(ROCISAPASS_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/pass.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/graph.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/composite.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/cycle.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/remove.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/insert_delay_alu.cpp)

nanobind_add_module(rocisa NOMINSIZE NB_SUPPRESS_WARNINGS
                        ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/main.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/base.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/code.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/container.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/count.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/enum.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/helper.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/label.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/macro.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/register.cpp
                        ${ROCISAINST_SOURCE}
                        ${ROCISAPASS_SOURCE}
                        ${ROCISAFUNC_SOURCE})
target_include_directories(rocisa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/rocisa/include)
set_target_properties(rocisa PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
