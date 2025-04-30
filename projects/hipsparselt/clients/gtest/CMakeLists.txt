# ########################################################################
# Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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

#enable_testing()
find_package( GTest REQUIRED )


set(hipsparselt_test_source
    hipsparselt_gtest_main.cpp
    hipsparselt_test.cpp
    prune_gtest.cpp
    compress_gtest.cpp
    spmm_gtest.cpp
    auxiliary_gtest.cpp
  )

add_executable( hipsparselt-test ${hipsparselt_test_source} ${hipsparselt_test_bench_common} )

target_compile_definitions( hipsparselt-test PRIVATE GOOGLE_TEST )

# Internal header includes
target_include_directories( hipsparselt-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
)

# External header includes included as system files
target_include_directories( hipsparselt-test
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # may be blank if not used
    $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
)
message("BLIS_INCLUDE_DIR=" ${BLIS_INCLUDE_DIR})
target_link_libraries( hipsparselt-test PRIVATE ${BLAS_LIBRARY} ${GTEST_BOTH_LIBRARIES} roc::hipsparselt )

if( NOT BUILD_CUDA )
  target_link_libraries( hipsparselt-test PRIVATE hip::host hip::device )
else()
  target_compile_definitions( hipsparselt-test PRIVATE __HIP_PLATFORM_NVIDIA__ )
  target_include_directories( hipsparselt-test
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>

  )
  target_link_libraries( hipsparselt-test PRIVATE ${CUDA_LIBRARIES} )
endif()

if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # GCC or hip-clang needs specific flag to turn on f16c intrinsics
  target_compile_options( hipsparselt-test PRIVATE -mf16c )
endif( )

target_compile_definitions( hipsparselt-test PRIVATE ROCM_USE_FLOAT16 HIPSPARSELT_INTERNAL_API )

target_compile_options(hipsparselt-test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}>)
# target_compile_options does not go to linker like CMAKE_CXX_FLAGS does, so manually add

if (BUILD_CUDA)
  target_link_libraries( hipsparselt-test PRIVATE -llapack -lcblas )
else()
  target_link_libraries( hipsparselt-test PRIVATE lapack cblas )
endif()

list( APPEND COMMON_LINK_LIBS "-lm -lstdc++fs")

if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  list( APPEND COMMON_LINK_LIBS "-lgfortran") # for lapack
else()
  list( APPEND COMMON_LINK_LIBS "-lflang -lflangrti") # for lapack
endif()

#if (NOT WIN32)
#  target_link_libraries( hipsparselt-test PRIVATE lapack cblas )
#  list( APPEND COMMON_LINK_LIBS "-lm -lstdc++fs")
#  if (NOT BUILD_FORTRAN_CLIENTS)
#    list( APPEND COMMON_LINK_LIBS "-lgfortran") # for lapack
#  endif()
#else()
#  list( APPEND COMMON_LINK_LIBS "libomp")
#endif()

target_link_libraries( hipsparselt-test PRIVATE ${COMMON_LINK_LIBS} )

set_target_properties( hipsparselt-test PROPERTIES
  IMPORT_PREFIX ""
  IMPORT_SUFFIX ".lib"
  LINKER_LANGUAGE CXX
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

set( HIPSPARSELT_TEST_DATA "${PROJECT_BINARY_DIR}/staging/hipsparselt_gtest.data")
add_custom_command( OUTPUT "${HIPSPARSELT_TEST_DATA}"
                    COMMAND ${python} ../common/hipsparselt_gentest.py -I ../include hipsparselt_gtest.yaml -o "${HIPSPARSELT_TEST_DATA}"
                    DEPENDS ../common/hipsparselt_gentest.py ../include/hipsparselt_common.yaml
                            spmm_common.yaml spmm_batched_common.yaml spmm_strided_batched_common.yaml
                            spmm_common_16.yaml spmm_batched_common_16.yaml spmm_strided_batched_common_16.yaml
                            spmm_common_row.yaml spmm_batched_common_row.yaml spmm_strided_batched_common_row.yaml
                            spmm_common_16_row.yaml spmm_batched_common_16_row.yaml spmm_strided_batched_common_16_row.yaml
                            prune_gtest.yaml prune_batched_gtest.yaml prune_strided_batched_gtest.yaml
                            prune_gtest_1b.yaml prune_batched_gtest_1b.yaml prune_strided_batched_gtest_1b.yaml
                            prune_gtest_row.yaml prune_batched_gtest_row.yaml prune_strided_batched_gtest_row.yaml
                            prune_gtest_1b_row.yaml prune_batched_gtest_1b_row.yaml prune_strided_batched_gtest_1b_row.yaml
                            compress_gtest.yaml compress_batched_gtest.yaml compress_strided_batched_gtest.yaml
                            compress_gtest_1b.yaml compress_batched_gtest_1b.yaml compress_strided_batched_gtest_1b.yaml
                            compress_gtest_row.yaml compress_batched_gtest_row.yaml compress_strided_batched_gtest_row.yaml
                            compress_gtest_1b_row.yaml compress_batched_gtest_1b_row.yaml compress_strided_batched_gtest_1b_row.yaml
                            spmm_gtest.yaml spmm_batched_gtest.yaml spmm_strided_batched_gtest.yaml
                            spmm_gtest_1b.yaml spmm_batched_gtest_1b.yaml spmm_strided_batched_gtest_1b.yaml
                            spmm_gtest_row.yaml spmm_batched_gtest_row.yaml spmm_strided_batched_gtest_row.yaml
                            spmm_gtest_1b_row.yaml spmm_batched_gtest_1b_row.yaml spmm_strided_batched_gtest_1b_row.yaml
                            auxiliary_gtest.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
add_custom_target( hipsparselt-test-data
                   DEPENDS "${HIPSPARSELT_TEST_DATA}" )
add_dependencies( hipsparselt-test hipsparselt-test-data hipsparselt-common )

rocm_install(TARGETS hipsparselt-test COMPONENT tests)
rocm_install(FILES ${HIPSPARSELT_TEST_DATA} DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT tests)
