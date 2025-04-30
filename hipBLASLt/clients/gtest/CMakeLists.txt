# ########################################################################
# Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
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
find_package (Python3 REQUIRED COMPONENTS Interpreter)


set(hipblaslt_test_source
    hipblaslt_gtest_main.cpp
    hipblaslt_test.cpp
    matmul_gtest.cpp
    auxiliary_gtest.cpp
    matrix_transform_gtest.cpp
    hipblaslt_gtest_ext_op.cpp
  )

add_executable( hipblaslt-test ${hipblaslt_test_source} ${hipblaslt_test_bench_common} )

target_compile_definitions( hipblaslt-test PRIVATE GOOGLE_TEST )

if( LEGACY_HIPBLAS_DIRECT )
  target_compile_definitions(hipblaslt-test PUBLIC LEGACY_HIPBLAS_DIRECT )
endif()

# Internal header includes
target_include_directories( hipblaslt-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
)

# External header includes included as system files
target_include_directories( hipblaslt-test
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # may be blank if not used
    $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
)
message("BLIS_INCLUDE_DIR=" ${BLIS_INCLUDE_DIR})
target_link_libraries( hipblaslt-test PRIVATE ${BLAS_LIBRARY} ${GTEST_BOTH_LIBRARIES} roc::hipblaslt )

if( NOT BUILD_CUDA )
  target_link_libraries( hipblaslt-test PRIVATE hip::host hip::device )
else()
  target_compile_definitions( hipblaslt-test PRIVATE __HIP_PLATFORM_NVIDIA__ )
  target_include_directories( hipblaslt-test
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>

  )
  target_link_libraries( hipblaslt-test PRIVATE ${CUDA_LIBRARIES} )
endif()

if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # GCC or hip-clang needs specific flag to turn on f16c intrinsics
  target_compile_options( hipblaslt-test PRIVATE -mf16c )
endif( )

target_compile_definitions( hipblaslt-test PRIVATE ROCM_USE_FLOAT16 HIPBLASLT_INTERNAL_API )

target_compile_options(hipblaslt-test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}>)
# target_compile_options does not go to linker like CMAKE_CXX_FLAGS does, so manually add

if (BUILD_CUDA)
  target_link_libraries( hipblaslt-test PRIVATE -llapack -lcblas )
else()
  target_link_libraries( hipblaslt-test PRIVATE lapack cblas )
endif()

list( APPEND COMMON_LINK_LIBS "-lm -lstdc++fs")

if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  list( APPEND COMMON_LINK_LIBS "-lgfortran") # for lapack
else()
  list( APPEND COMMON_LINK_LIBS "-lflang -lflangrti") # for lapack
endif()

#if (NOT WIN32)
#  target_link_libraries( hipblaslt-test PRIVATE lapack cblas )
#  list( APPEND COMMON_LINK_LIBS "-lm -lstdc++fs")
#  if (NOT BUILD_FORTRAN_CLIENTS)
#    list( APPEND COMMON_LINK_LIBS "-lgfortran") # for lapack
#  endif()
#else()
#  list( APPEND COMMON_LINK_LIBS "libomp")
#endif()

target_link_libraries( hipblaslt-test PRIVATE ${COMMON_LINK_LIBS} )

set_target_properties( hipblaslt-test PROPERTIES
  IMPORT_PREFIX ""
  IMPORT_SUFFIX ".lib"
  LINKER_LANGUAGE CXX
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

set( HIPBLASLT_TEST_DATA "${PROJECT_BINARY_DIR}/staging/hipblaslt_gtest.data")
add_custom_command( OUTPUT "${HIPBLASLT_TEST_DATA}"
                    COMMAND "${Python3_EXECUTABLE}" ../common/hipblaslt_gentest.py -I ../include hipblaslt_gtest.yaml -o "${HIPBLASLT_TEST_DATA}"
                    DEPENDS ../common/hipblaslt_gentest.py ../include/hipblaslt_common.yaml
                            matmul_common.yaml
                            matmul_gtest.yaml
                            auxiliary_gtest.yaml
                            smoke_gtest.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
add_custom_target( hipblaslt-test-data
                   DEPENDS "${HIPBLASLT_TEST_DATA}" )
add_dependencies( hipblaslt-test hipblaslt-test-data hipblaslt-common )

# Not suitable because loading takes too much time
# include(GoogleTest)
# gtest_discover_tests(hipblaslt-test TEST_PREFIX gtest:)
add_test(NAME hipblaslt-test COMMAND hipblaslt-test --gtest_output=xml --gtest_color=yes '--gtest_filter=*')
set_tests_properties(hipblaslt-test
                     PROPERTIES ENVIRONMENT GTEST_LISTENER=NO_PASS_LINE_IN_LOG)

rocm_install(TARGETS hipblaslt-test COMPONENT tests)
rocm_install(FILES ${HIPBLASLT_TEST_DATA} DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT tests)
