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

set(TESTS_NAME rocshmem_example_driver)

###############################################################################
# SOURCES
###############################################################################
add_executable(${TESTS_NAME} "")

target_include_directories(
  ${TESTS_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(
  ${TESTS_NAME}
  PRIVATE
    barrier_all_tester.cpp
    sync_tester.cpp
    test_driver.cpp
    tester.cpp
    tester_arguments.cpp
    ping_pong_tester.cpp
    ping_all_tester.cpp
    primitive_tester.cpp
    primitive_mr_tester.cpp
    default_ctx_primitive_tester.cpp
    team_ctx_primitive_tester.cpp
    team_ctx_infra_tester.cpp
    amo_bitwise_tester.cpp
    amo_extended_tester.cpp
    amo_standard_tester.cpp
    random_access_tester.cpp
    shmem_ptr_tester.cpp
    signaling_operations_tester.cpp
    signaling_operations_tester.hpp
    workgroup_primitives.cpp
    empty_tester.cpp
    wavefront_primitives.cpp
)

###############################################################################
# ROCSHMEM
###############################################################################
if (BUILD_TESTS_ONLY)
  find_package(MPI REQUIRED)

  target_include_directories(
    ${TESTS_NAME}
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
    ${TESTS_NAME}
    PRIVATE
      ${flags_str}
       -fgpu-rdc
  )

  target_link_libraries(
    ${TESTS_NAME}
    PRIVATE
      ${MPI_mpi_LIBRARY}
      ${MPI_mpicxx_LIBRARY}
      ${flags_str}
      -L${ROCSHMEM_HOME}/lib
      -lamdhip64
      -lhsa-runtime64
      -lrocshmem
      -fgpu-rdc
  )
else()
  target_include_directories(
    ${TESTS_NAME}
    PRIVATE
      roc::rocshmem
  )

  target_link_libraries(
    ${TESTS_NAME}
    PRIVATE
      roc::rocshmem
      -fgpu-rdc
  )
endif()
