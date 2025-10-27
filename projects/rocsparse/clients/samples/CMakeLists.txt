# ########################################################################
# Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

# Function to add rocsparse examples
function(add_rocsparse_example EXAMPLE_SOURCE)
  get_filename_component(EXAMPLE_TARGET ${EXAMPLE_SOURCE} NAME_WE)
  add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE})

  # Target compile options
  target_compile_options(${EXAMPLE_TARGET} PRIVATE -Wno-unused-command-line-argument)

  # Include common client headers
  target_include_directories(${EXAMPLE_TARGET} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
  if (BUILD_CLIENTS_ONLY)
    target_include_directories(${EXAMPLE_TARGET} PRIVATE ${rocsparse_INCLUDE_DIRS}/rocsparse)
  endif()

  # Linker dependencies
  target_link_libraries(${EXAMPLE_TARGET} PRIVATE roc::rocsparse hip::host hip::device)
  # Target properties
  set_target_properties(${EXAMPLE_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
endfunction()

# Examples

# Empty C program for checking rocsparse headers are C compatible.
# weird compiler flags on Windows.
if(NOT WIN32)
  add_rocsparse_example(example_c_headers.c)
endif()

# Auxiliary
add_rocsparse_example(example_handle.cpp)

# Level 1
add_rocsparse_example(example_axpyi.cpp)
add_rocsparse_example(example_doti.cpp)
add_rocsparse_example(example_dotci.cpp)
add_rocsparse_example(example_gthr.cpp)
add_rocsparse_example(example_gthrz.cpp)
add_rocsparse_example(example_roti.cpp)
add_rocsparse_example(example_sctr.cpp)

# Level 2
add_rocsparse_example(example_bsrmv.cpp)
add_rocsparse_example(example_bsrsv.cpp)
add_rocsparse_example(example_coomv.cpp)
add_rocsparse_example(example_csrmv.cpp)
add_rocsparse_example(example_ellmv.cpp)
add_rocsparse_example(example_hybmv.cpp)
add_rocsparse_example(example_csrsv.cpp)
add_rocsparse_example(example_csritsv.cpp)
add_rocsparse_example(example_spmv_coo.cpp)
add_rocsparse_example(example_spmv_csr.cpp)
add_rocsparse_example(example_spmv_ell.cpp)
add_rocsparse_example(example_gebsrmv.cpp)
add_rocsparse_example(example_gemvi.cpp)
add_rocsparse_example(example_v2_spmv_csr.cpp)
add_rocsparse_example(example_sptrsv.cpp)

set_source_files_properties(
  example_spmv_coo.cpp
  PROPERTIES
  COMPILE_FLAGS "-Wno-deprecated-declarations"
)

set_source_files_properties(
  example_spmv_csr.cpp
  PROPERTIES
  COMPILE_FLAGS "-Wno-deprecated-declarations"
)

set_source_files_properties(
  example_spmv_ell.cpp
  PROPERTIES
  COMPILE_FLAGS "-Wno-deprecated-declarations"
)

# Level 3
add_rocsparse_example(example_bsrmm.cpp)
add_rocsparse_example(example_csrmm.cpp)
add_rocsparse_example(example_spmm_csr.cpp)
add_rocsparse_example(example_spmm_coo.cpp)
add_rocsparse_example(example_spmm_batched_csr.cpp)
add_rocsparse_example(example_spmm_batched_coo.cpp)
add_rocsparse_example(example_bsrsm.cpp)
add_rocsparse_example(example_csrsm.cpp)
add_rocsparse_example(example_gemmi.cpp)
add_rocsparse_example(example_gtsv.cpp)

# Precond
add_rocsparse_example(example_gpsv.cpp)

# Extra
add_rocsparse_example(example_csrgeam.cpp)
add_rocsparse_example(example_csrgemm.cpp)
add_rocsparse_example(example_spgemm.cpp)
add_rocsparse_example(example_spgeam.cpp)
add_rocsparse_example(example_spgeam_2.cpp)
add_rocsparse_example(example_spgeam_3.cpp)

# Managed
add_rocsparse_example(example_csrmv_managed.cpp)
add_rocsparse_example(example_csrsv_managed.cpp)

# Fortran examples
if(TARGET rocsparse)
  # Compile Fortran examples only if built directly with package
  # else the Fortran module file is not generated

  if (NOT WIN32 AND BUILD_FORTRAN_CLIENTS)

    function(add_rocsparse_fortran_example EXAMPLE_SOURCE)
      set( fortran_language "Fortran" )
      get_filename_component(EXAMPLE_TARGET ${EXAMPLE_SOURCE} NAME_WE)
      add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE})

      # Target compile options
      target_compile_options(${EXAMPLE_TARGET} PRIVATE -Wno-unused-command-line-argument)

      # Include common client headers
      target_include_directories(${EXAMPLE_TARGET} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
      if (BUILD_CLIENTS_ONLY)
        target_include_directories(${EXAMPLE_TARGET} PRIVATE ${rocsparse_INCLUDE_DIRS}/rocsparse)
      endif()

      # Linker dependencies
      target_link_libraries(${EXAMPLE_TARGET} PRIVATE roc::rocsparse hip::host)
      if (BUILD_CODE_COVERAGE)
	      target_link_libraries(${EXAMPLE_TARGET} PRIVATE -lgcov)
      endif()

      target_link_libraries( ${EXAMPLE_TARGET} PRIVATE rocsparse_fortran )

      # Target properties
      set_target_properties(${EXAMPLE_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    endfunction()

    add_rocsparse_fortran_example(example_fortran_gtsv.f90)
    add_rocsparse_fortran_example(example_fortran_csric0.f90)
    add_rocsparse_fortran_example(example_fortran_csrilu0.f90)
    add_rocsparse_fortran_example(example_fortran_bsrmm.f90)
    add_rocsparse_fortran_example(example_fortran_bsric0.f90)
    add_rocsparse_fortran_example(example_fortran_bsrilu0.f90)
    add_rocsparse_fortran_example(example_fortran_bsrsv.f90)
    add_rocsparse_fortran_example(example_fortran_csrsv.f90)
    add_rocsparse_fortran_example(example_fortran_spmv.f90)
    add_rocsparse_fortran_example(example_fortran_csrsm.f90)
    add_rocsparse_fortran_example(example_fortran_gemmi.f90)
    add_rocsparse_fortran_example(example_fortran_auxiliary.f90)
    add_rocsparse_fortran_example(example_fortran_dotci.f90)
    add_rocsparse_fortran_example(example_fortran_roti.f90)
  endif()
endif()

# Build documentation examples
FILE(GLOB_RECURSE DOCUMENTATION_EXAMPLES documentation_examples/*.cpp)

foreach(file ${DOCUMENTATION_EXAMPLES})
    message(STATUS "Found .cpp file: ${file}")
    add_rocsparse_example(${file})
endforeach()
