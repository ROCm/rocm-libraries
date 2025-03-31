###############################################################################
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

cmake_minimum_required(VERSION 3.16.3 FATAL_ERROR)

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

###############################################################################
# GLOBAL COMPILE FLAGS
###############################################################################
if (NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
endif()
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")

###############################################################################
# DEFAULT BUILD TYPE
###############################################################################
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "CMAKE_BUILD_TYPE unspecified: generating Release build")

  set(
    CMAKE_BUILD_TYPE
    "Release"
    CACHE
      STRING
        "build type: Release, Debug, RelWithDebInfo, MinSizeRel"
    FORCE
  )
endif()

###############################################################################
# PROJECT
###############################################################################
project(rocshmem_unit_tests VERSION 1.0.0 LANGUAGES CXX)

###############################################################################
# SOURCES
###############################################################################
add_executable(${PROJECT_NAME} "")

target_include_directories(
  ${PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/../../library
)

target_sources(
  ${PROJECT_NAME}
  PRIVATE
    shmem_gtest.cpp
    heap_memory_gtest.cpp
    hipmalloc_gtest.cpp
    bin_gtest.cpp
    binner_gtest.cpp
    #bitwise_gtest.cpp # Test is disabled becasue of compilation errors
    address_record_gtest.cpp
    index_strategy_gtest.cpp
    single_heap_gtest.cpp
    #slab_heap_gtest.cpp # Test is disabled because class unused
    symmetric_heap_gtest.cpp
    pow2_bins_gtest.cpp
    remote_heap_info_gtest.cpp
    mpi_init_singleton_gtest.cpp
    #spin_ebo_block_mutex_gtest.cpp
    abql_block_mutex_gtest.cpp
    notifier_gtest.cpp
    #forward_list_gtest.cpp
    free_list_gtest.cpp
    #context_ipc_gtest.cpp
    wavefront_size_gtest.cpp
    atomic_wf_queue_gtest.cpp
)

if (USE_IPC)
  target_sources(
    ${PROJECT_NAME}
    PRIVATE
      ipc_impl_simple_coarse_gtest.cpp
      ipc_impl_simple_fine_gtest.cpp
      ipc_impl_tiled_fine_gtest.cpp
  )
endif()

###############################################################################
# ROCSHMEM DEPENDENCY
###############################################################################
find_package(hip REQUIRED)

if (BUILD_TESTS_ONLY)
  find_package(MPI REQUIRED)

  target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
      $<BUILD_INTERFACE:${ROCSHMEM_HOME}/include>
      $<BUILD_INTERFACE:${MPI_CXX_HEADER_DIR}>
  )

  foreach (target ${DEFAULT_GPUS})
    list(APPEND static_link_flags --offload-arch=${target})
  endforeach()
  list(JOIN static_link_flags " " flags_str)

  target_compile_options(
    ${PROJECT_NAME}
    PRIVATE
      ${flags_str}
       -fgpu-rdc
  )

  target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
      ${MPI_mpi_LIBRARY}
      ${MPI_mpicxx_LIBRARY}
      -L${ROCSHMEM_HOME}/lib
      -lamdhip64
      -lhsa-runtime64
      -lrocshmem
      -fgpu-rdc
  )
else()
  target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
      roc::rocshmem
  )

  target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
      roc::rocshmem
      hip::host
      -fgpu-rdc
  )
endif()

###############################################################################
# GTEST DEPENDENCY
###############################################################################
# These packages are required for the unit tests
find_package(rocprim REQUIRED CONFIG PATHS "${ROCM_PATH}/include/rocprim")
find_package(rocthrust REQUIRED CONFIG PATHS "${ROCM_PATH}/include/rocthrust")

include(FetchContent)

FetchContent_Declare(
    googletest
    PREFIX extern/googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE
    gtest
    gtest_main
    roc::rocthrust
)
