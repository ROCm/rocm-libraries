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

###############################################################################
# SOURCES
###############################################################################
set(EXAMPLE_SOURCES
  rocshmem_allreduce_test.cc
  rocshmem_alltoall_test.cc
  rocshmem_broadcast_test.cc
  rocshmem_getmem_test.cc
  rocshmem_put_signal_test.cc
  rocshmem_init_attr_test.cc
)

foreach(SOURCE_FILE IN LISTS EXAMPLE_SOURCES)
  get_filename_component(EXECUTABLE_NAME ${SOURCE_FILE} NAME_WE)

  add_executable(${EXECUTABLE_NAME} ${SOURCE_FILE})

  if (BUILD_TESTS_ONLY)
    find_package(MPI REQUIRED)

    target_include_directories(
      ${EXECUTABLE_NAME}
      PRIVATE
        $<BUILD_INTERFACE:${ROCSHMEM_HOME}/include>
        $<BUILD_INTERFACE:${MPI_CXX_HEADER_DIR}>
    )

    foreach (target ${DEFAULT_GPUS})
      list(APPEND static_link_flags --offload-arch=${target})
    endforeach()
    list(JOIN static_link_flags " " flags_str)

    target_compile_options(
      ${EXECUTABLE_NAME}
      PRIVATE
        ${flags_str}
         -fgpu-rdc
    )

    target_link_libraries(
      ${EXECUTABLE_NAME}
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
      ${EXECUTABLE_NAME}
      PRIVATE
        roc::rocshmem
    )

    target_link_libraries(
      ${EXECUTABLE_NAME}
      PRIVATE
        roc::rocshmem
        -fgpu-rdc
    )
  endif()
endforeach()
