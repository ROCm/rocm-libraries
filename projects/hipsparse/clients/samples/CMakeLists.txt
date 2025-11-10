# ########################################################################
# Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

set(HIPSPARSE_CLIENTS_COMMON ../common/utility.cpp)

# Function to add hipsparse examples
function(add_hipsparse_example EXAMPLE_SOURCE)
  get_filename_component(EXAMPLE_TARGET ${EXAMPLE_SOURCE} NAME_WE)
  add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE} ${HIPSPARSE_CLIENTS_COMMON})

  get_filename_component(EXAMPLE_EXTENSION ${EXAMPLE_SOURCE} LAST_EXT)

  if("${EXAMPLE_EXTENSION}" STREQUAL ".f90")
    add_dependencies(${EXAMPLE_TARGET} hipsparse_fortran)
  endif()

  # Include common client headers
  target_include_directories(${EXAMPLE_TARGET} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

  if (BUILD_CLIENTS_ONLY)
    target_include_directories(${EXAMPLE_TARGET} PRIVATE ${hipsparse_INCLUDE_DIRS}/hipsparse)
  endif()

  # Linker dependencies
  target_link_libraries(${EXAMPLE_TARGET} PRIVATE roc::hipsparse)

  if(NOT USE_CUDA)
    target_link_libraries(${EXAMPLE_TARGET} PRIVATE hip::host)
  else()
    target_compile_definitions(${EXAMPLE_TARGET} PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_include_directories(${EXAMPLE_TARGET} PRIVATE ${HIP_INCLUDE_DIRS})
    target_link_libraries(${EXAMPLE_TARGET} PRIVATE ${CUDA_LIBRARIES})
  endif()

  set_target_properties(${EXAMPLE_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
endfunction()

# Examples
if (NOT WIN32)
  # C program for checking hipsparse headers are C compatible.
  add_hipsparse_example(example_c_headers.c)
  if(NOT USE_CUDA)
    add_hipsparse_example(example_axpyi.c)
  endif()
endif()

add_hipsparse_example(example_handle.cpp)

if(NOT USE_CUDA OR (USE_CUDA AND CUDA_VERSION LESS 11))
add_hipsparse_example(example_csrmv.cpp)
add_hipsparse_example(example_hybmv.cpp)
endif()

if (NOT WIN32)
  # Fortran examples
  if(TARGET hipsparse AND NOT USE_CUDA)
    # Compile Fortran examples only if built directly with package
    # else the Fortran module file is not generated

    add_hipsparse_example(example_fortran_auxiliary.f90)
    add_hipsparse_example(example_fortran_csrsv2.f90)
    add_hipsparse_example(example_fortran_bsric02.f90)
    add_hipsparse_example(example_fortran_csric02.f90)
    add_hipsparse_example(example_fortran_bsrilu02.f90)
    add_hipsparse_example(example_fortran_csrilu02.f90)
    add_hipsparse_example(example_fortran_csrsm2.f90)
    add_hipsparse_example(example_fortran_dotci.f90)
    add_hipsparse_example(example_fortran_roti.f90)
    add_hipsparse_example(example_fortran_spmv.f90)
    add_hipsparse_example(example_fortran_bsrmm.f90)

  endif()
endif()

if(NOT USE_CUDA)
  # Build documentation examples
  FILE(GLOB_RECURSE DOCUMENTATION_EXAMPLES documentation_examples/*.cpp)

  foreach(file ${DOCUMENTATION_EXAMPLES})
      message(STATUS "Found .cpp file: ${file}")
      add_hipsparse_example(${file})
  endforeach()
endif()
