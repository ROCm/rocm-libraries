# ########################################################################
# Copyright (C) 2018-2021 Advanced Micro Devices, Inc. All rights Reserved.
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

find_package(GTest REQUIRED)


#
# Client matrices.
#
if(NOT EXISTS "${CMAKE_MATRICES_DIR}")
  #
  # Download.
  #
  set(CMAKE_MATRICES_DIR ${PROJECT_BINARY_DIR}/matrices CACHE STRING "Matrices directory.")
  file(MAKE_DIRECTORY ${CMAKE_MATRICES_DIR})

  if(NOT TARGET hipsparse)
    set(CONVERT_SOURCE ${CMAKE_SOURCE_DIR}/../deps/convert.cpp CACHE STRING "Convert tool mtx2csr.")
    include(${CMAKE_SOURCE_DIR}/../cmake/ClientMatrices.cmake)
  else()
   set(CONVERT_SOURCE ${CMAKE_SOURCE_DIR}/deps/convert.cpp CACHE STRING "Convert tool mtx2csr.")
    include(${CMAKE_SOURCE_DIR}/cmake/ClientMatrices.cmake)
  endif()

else()
  #
  # Copy.
  #
  if(NOT CMAKE_MATRICES_DIR STREQUAL "${PROJECT_BINARY_DIR}/matrices")
    message("Copy matrix files from ${CMAKE_MATRICES_DIR} to ${PROJECT_BINARY_DIR}/matrices")

    execute_process(COMMAND cp -r ${CMAKE_MATRICES_DIR} ${PROJECT_BINARY_DIR}/matrices RESULT_VARIABLE STATUS WORKING_DIRECTORY ${CMAKE_MATRICES_DIR})

    if(STATUS AND NOT STATUS EQUAL 0)
      message(FATAL_ERROR "Failed to copy matrix .bin files, aborting.")
    endif()
  endif()

endif()

set(HIPSPARSE_TEST_SOURCES
  hipsparse_gtest_main.cpp
  test_axpyi.cpp
  test_gthr.cpp
  test_gthrz.cpp
  test_roti.cpp
  test_sctr.cpp
  test_bsrmv.cpp
  test_bsrxmv.cpp
  test_bsrsv2.cpp
  test_csrsv2.cpp
  test_bsrmm.cpp
  test_bsrsm2.cpp
  test_csrsm2.cpp
  test_gemmi.cpp
  test_gemvi.cpp
  test_csrgeam2.cpp
  test_csrgemm2_a.cpp
  test_csrgemm2_b.cpp
  test_bsrilu02.cpp
  test_csrilu02.cpp
  test_bsric02.cpp
  test_csric02.cpp
  test_nnz.cpp
  test_csr2dense.cpp
  test_csc2dense.cpp
  test_dense2csr.cpp
  test_prune_dense2csr.cpp
  test_prune_dense2csr_by_percentage.cpp
  test_dense2csc.cpp
  test_csr2coo.cpp
  test_csr2bsr.cpp
  test_bsr2csr.cpp
  test_gebsr2csr.cpp
  test_csr2csr_compress.cpp
  test_prune_csr2csr.cpp
  test_prune_csr2csr_by_percentage.cpp
  test_coo2csr.cpp
  test_identity.cpp
  test_csrsort.cpp
  test_cscsort.cpp
  test_coosort.cpp
  test_csru2csr.cpp
  test_csrilusv.cpp
  test_gebsr2gebsr.cpp
  test_csr2gebsr.cpp
  test_gebsr2gebsc.cpp
  test_spmat_descr.cpp
  test_spvec_descr.cpp
  test_dnmat_descr.cpp
  test_dnvec_descr.cpp
  test_const_spmat_descr.cpp
  test_const_spvec_descr.cpp
  test_const_dnmat_descr.cpp
  test_const_dnvec_descr.cpp
  test_spmv_coo.cpp
  test_spmv_coo_aos.cpp
  test_spmv_csr.cpp
  test_axpby.cpp
  test_gather.cpp
  test_scatter.cpp
  test_rot.cpp
  test_spvv.cpp
  test_dense_to_sparse_csr.cpp
  test_dense_to_sparse_csc.cpp
  test_dense_to_sparse_coo.cpp
  test_sparse_to_dense_csr.cpp
  test_sparse_to_dense_csc.cpp
  test_sparse_to_dense_coo.cpp
  test_spmm_csr.cpp
  test_spmm_batched_csr.cpp
  test_spmm_csc.cpp
  test_spmm_batched_csc.cpp
  test_spmm_coo.cpp
  test_spmm_batched_coo.cpp
  test_spmm_bell.cpp
  test_spgemm_csr.cpp
  test_spgemmreuse_csr.cpp
  test_sddmm_csr.cpp
  test_sddmm_csc.cpp
  test_sddmm_coo.cpp
  test_sddmm_coo_aos.cpp
  test_gpsv_interleaved_batch.cpp
  test_gtsv2_strided_batch.cpp
  test_gtsv.cpp
  test_gtsv2_nopivot.cpp
  test_gtsv_interleaved_batch.cpp
  test_csrcolor.cpp
  test_spsv_csr.cpp
  test_spsv_coo.cpp
  test_spsm_csr.cpp
  test_spsm_coo.cpp
)


if(NOT USE_CUDA)
    list(APPEND HIPSPARSE_TEST_SOURCES
        test_doti.cpp
        test_dotci.cpp
        test_csr2csc.cpp
        test_csr2csc_ex2.cpp
        test_csrgemm.cpp
        test_csrgeam.cpp
        test_csrmv.cpp
        test_csrmm.cpp
        test_hybmv.cpp
        test_csr2hyb.cpp
        test_hyb2csr.cpp
    )
endif()

set(HIPSPARSE_CLIENTS_COMMON
  ../common/arg_check.cpp
  ../common/unit.cpp
  ../common/utility.cpp
  ../common/hipsparse_parse_data.cpp
  ../common/hipsparse_template_specialization.cpp
)

add_executable(hipsparse-test ${HIPSPARSE_TEST_SOURCES} ${HIPSPARSE_CLIENTS_COMMON})

# Set GOOGLE_TEST definition
target_compile_definitions(hipsparse-test PRIVATE GOOGLE_TEST)

# Target compile options
target_compile_options(hipsparse-test PRIVATE -Wno-unused-command-line-argument -Wall)

# Internal common header
target_include_directories(hipsparse-test PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

if (BUILD_CLIENTS_ONLY)
  target_include_directories(hipsparse-test PRIVATE ${hipsparse_INCLUDE_DIRS}/hipsparse)
endif()

# Target link libraries
target_link_libraries(hipsparse-test PRIVATE GTest::gtest roc::hipsparse)

# Add OpenMP if available
if(OPENMP_FOUND AND THREADS_FOUND)
  target_link_libraries(hipsparse-test PRIVATE OpenMP::OpenMP_CXX ${OpenMP_CXX_FLAGS})
endif()

if(NOT USE_CUDA)
  target_link_libraries(hipsparse-test PRIVATE hip::host)
else()
  target_compile_definitions(hipsparse-test PRIVATE __HIP_PLATFORM_NVIDIA__)
  target_include_directories(hipsparse-test PRIVATE ${HIP_INCLUDE_DIRS})
  target_link_libraries(hipsparse-test PRIVATE ${CUDA_LIBRARIES})
endif()

set_target_properties(hipsparse-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

rocm_install(TARGETS hipsparse-test COMPONENT tests)

set(HIPSPARSE_TEST_YAMLS test_axpyi.yaml 
                         test_doti.yaml 
                         test_dotci.yaml
                         test_sctr.yaml
                         test_roti.yaml
                         test_gthr.yaml
                         test_gthrz.yaml)

# Prepare testing data
set(HIPSPARSE_TEST_DATA "${PROJECT_BINARY_DIR}/staging/hipsparse_test.data")
add_custom_command(OUTPUT "${HIPSPARSE_TEST_DATA}"
                   COMMAND ${python} ../common/hipsparse_gentest.py -m ${PROJECT_BINARY_DIR}/matrices -I ../include hipsparse_test.yaml -o "${HIPSPARSE_TEST_DATA}"
                   DEPENDS ../common/hipsparse_gentest.py hipsparse_test.yaml ../include/hipsparse_common.yaml ${HIPSPARSE_TEST_YAMLS}
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_custom_target(hipsparse-test-data
  DEPENDS "${HIPSPARSE_TEST_DATA}" )

add_dependencies(hipsparse-test  hipsparse-test-data hipsparse-common)

rocm_install(FILES ${HIPSPARSE_TEST_DATA} DESTINATION "${CMAKE_INSTALL_DATADIR}/hipsparse/test" COMPONENT tests)

if (WIN32)
  # for now adding in all .dll as dependency chain is not cmake based on win32
  file( GLOB third_party_dlls
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    $ENV{HIP_DIR}/bin/*.dll
    $ENV{HIP_DIR}/bin/hipinfo.exe
    ${ROCSPARSE_PATH}/bin/rocsparse.dll
    ${CMAKE_SOURCE_DIR}/rtest.*
    C:/Windows/System32/libomp140*.dll
  )
  foreach( file_i ${third_party_dlls})
    add_custom_command( TARGET hipsparse-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/ )
  endforeach( file_i )

endif()

add_test(hipsparse-test hipsparse-test)
