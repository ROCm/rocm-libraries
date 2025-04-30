################################################################################
#
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

cmake_minimum_required(VERSION 3.25.2)

# Override all paths arguments as they do not work properly
file(TO_CMAKE_PATH "$ENV{ROCM_PATH}" ROCM_PATH_ENV_VALUE)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH_ENV_VALUE} /opt/rocm /opt/rocm/lib/llvm)

project(Tensile)

option(Tensile_ENABLE_MARKER "ENable roctx marker in Tensile" OFF)

if(Tensile_ENABLE_MARKER)
    find_library(rocTracer roctx64)
    if(NOT rocTracer)
        message(FATAL_ERROR "roctracer not found, but Tensile_ENABLE_MARKER is enabled")
    endif()
    add_definitions(-DTensile_ENABLE_MARKER)
endif()

set(TENSILE_USE_HIP      ON CACHE BOOL "Use the Hip runtime.")
if(Tensile_LIBRARY_FORMAT MATCHES "msgpack")
    set(TENSILE_USE_MSGPACK  ON CACHE BOOL "Use message pack for parsing config files.")
else()
    set(TENSILE_USE_LLVM     ON CACHE BOOL "Use LLVM for parsing config files.")
endif()
set(TENSILE_USE_OPENMP   ON CACHE BOOL "Use OpenMP to improve performance.")
set(TENSILE_STATIC_ONLY  ON CACHE BOOL "Disable exposing Tensile symbols in a shared library.")

if(NOT DEFINED CXX_VERSION_STRING)
    if(CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" OR CMAKE_CXX_COMPILER MATCHES ".*clang\\+\\+")
      # Determine if CXX Compiler is hip-clang or nvcc
      execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--version" OUTPUT_VARIABLE CXX_OUTPUT
              OUTPUT_STRIP_TRAILING_WHITESPACE
              ERROR_STRIP_TRAILING_WHITESPACE)
      string(REGEX MATCH "[A-Za-z]* ?clang version" TMP_CXX_VERSION ${CXX_OUTPUT})
      string(REGEX MATCH "[A-Za-z]+" CXX_VERSION_STRING ${TMP_CXX_VERSION})
    endif()
endif()

if(CMAKE_CXX_COMPILER STREQUAL "hipcc")
  set(TENSILE_GPU_ARCHS gfx803 gfx900 gfx906:xnack- gfx908:xnack- gfx90a:xnack- gfx1010 gfx1011 gfx1012 gfx1030 gfx1100 gfx1101 gfx1102 gfx1200 gfx1201 CACHE STRING "GPU architectures")
else()
  set(TENSILE_GPU_ARCHS gfx803 gfx900 gfx906 gfx908 gfx90a gfx1010 gfx1011 gfx1012 gfx1030 gfx1100 gfx1101 gfx1102 gfx1200 gfx1201 CACHE STRING "GPU architectures")
endif()

include(CMakeDependentOption)
CMAKE_DEPENDENT_OPTION(TENSILE_BUILD_CLIENT "Build the benchmarking client" ON
                        "TENSILE_USE_HIP" OFF)

if(TENSILE_USE_HIP)
    find_package(HIP REQUIRED CONFIG PATHS ${ROCM_PATH_ENV_VALUE} /opt/rocm)
endif()

if(TENSILE_USE_OPENMP)
    # Workaround for https://gitlab.kitware.com/cmake/cmake/-/issues/21787
    # ensures we link to HIP's libomp and get an rpath to it.
    add_library(custom_openmp_cxx INTERFACE)

    if(TENSILE_USE_HIP)
        target_compile_options(custom_openmp_cxx INTERFACE "-fopenmp")
        target_link_options(custom_openmp_cxx INTERFACE "-fopenmp")
    else ()
        find_package(OpenMP REQUIRED)
        target_link_libraries(custom_openmp_cxx INTERFACE OpenMP::OpenMP_CXX)
    endif ()
endif()

add_subdirectory(lib)

if(TENSILE_BUILD_CLIENT)
    add_subdirectory(client)
endif()

export(TARGETS TensileHost NAMESPACE Tensile:: FILE TensileExports.cmake)

