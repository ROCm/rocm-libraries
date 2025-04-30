# ########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
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

set(hipsparselt_bench_source
  client.cpp
  )

add_executable( hipsparselt-bench ${hipsparselt_bench_source} ${hipsparselt_test_bench_common} )

# Internal header includes
target_include_directories( hipsparselt-bench
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/spmm>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
)

# External header includes included as system files
target_include_directories( hipsparselt-bench
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # may be blank if not used
)

target_link_libraries( hipsparselt-bench PRIVATE ${BLAS_LIBRARY} roc::hipsparselt)

if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # GCC or hip-clang needs specific flags to turn on f16c intrinsics
  target_compile_options( hipsparselt-bench PRIVATE -mf16c )
endif( )

target_compile_definitions( hipsparselt-bench PRIVATE HIPSPARSELT_BENCH ROCM_USE_FLOAT16 HIPSPARSELT_INTERNAL_API ${TENSILE_DEFINES} )
if ( NOT BUILD_FORTRAN_CLIENTS )
  target_compile_definitions( hipsparselt-bench PRIVATE CLIENTS_NO_FORTRAN )
endif()

target_compile_options(hipsparselt-bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}>)

if( NOT BUILD_CUDA )
  target_link_libraries( hipsparselt-bench PRIVATE hip::host hip::device )
else()
  target_compile_definitions( hipsparselt-bench PRIVATE __HIP_PLATFORM_NVIDIA__ )
  target_include_directories( hipsparselt-bench
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>

  )
  target_link_libraries( hipsparselt-bench PRIVATE ${CUDA_LIBRARIES} )
endif()

# target_compile_options does not go to linker like CMAKE_CXX_FLAGS does, so manually add
if (NOT WIN32)
  if (BUILD_CUDA)
    target_link_libraries( hipsparselt-bench PRIVATE -llapack -lcblas )
  else()
    target_link_libraries( hipsparselt-bench PRIVATE lapack cblas )
  endif()
  list( APPEND COMMON_LINK_LIBS "-lm -lstdc++fs")
  if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    list( APPEND COMMON_LINK_LIBS "-lgfortran") # for lapack
  else()
    list( APPEND COMMON_LINK_LIBS "-lflang -lflangrti") # for lapack
  endif()
else()
  list( APPEND COMMON_LINK_LIBS "libomp")
endif()
target_link_libraries( hipsparselt-bench PRIVATE ${COMMON_LINK_LIBS} )

set_target_properties( hipsparselt-bench PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

add_dependencies( hipsparselt-bench hipsparselt-common )

rocm_install(TARGETS hipsparselt-bench COMPONENT benchmarks)
