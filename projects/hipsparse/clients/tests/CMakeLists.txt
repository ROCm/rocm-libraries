# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Optional: CTest test categorization when building in rocm-libraries with shared/ctest.
# Skipped when shared/ is not present (e.g. standalone hipsparse or sparse checkout in CI).
if(DEFINED ROCM_LIBRARIES_ROOT AND EXISTS "${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
  include("${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
  set(HIPSPARSE_HAS_CTEST_CATEGORIES ON)
endif()

find_package(GTest REQUIRED)

# Download or copy client matrices
if(NOT EXISTS "${CMAKE_MATRICES_DIR}")
    # Download matrices
    set(CMAKE_MATRICES_DIR "${PROJECT_BINARY_DIR}/clients/matrices" CACHE STRING "Matrices directory.")
    file(MAKE_DIRECTORY "${CMAKE_MATRICES_DIR}")
    
    set(CONVERT_SOURCE "${PROJECT_SOURCE_DIR}/deps/convert.cpp" CACHE STRING "Convert tool mtx2csr.")
    include("${PROJECT_SOURCE_DIR}/cmake/ClientMatrices.cmake")
else()
    # Copy matrices
    if(NOT CMAKE_MATRICES_DIR STREQUAL "${PROJECT_BINARY_DIR}/clients/matrices")
        message("Copy matrix files from ${CMAKE_MATRICES_DIR} to ${PROJECT_BINARY_DIR}/clients/matrices")

        file(COPY ${CMAKE_MATRICES_DIR}/ DESTINATION ${PROJECT_BINARY_DIR}/clients/matrices)

        if(STATUS AND NOT STATUS EQUAL 0)
            message(FATAL_ERROR "Failed to copy matrix .bin files, aborting.")
        endif()
    endif()
endif()

add_executable(hipsparse-test)

target_sources(hipsparse-test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_gtest_main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_axpyi.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_gthr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_gthrz.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_roti.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_sctr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_bsrmv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_bsrxmv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_bsrsv2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_csrsv2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_bsrmm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_bsrsm2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_csrsm2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_gemmi.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_gemvi.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_csrgeam2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_csrgemm2_a.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_csrgemm2_b.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_bsrilu02.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_csrilu02.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_bsric02.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_csric02.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_nnz.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_csr2dense.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_csc2dense.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_dense2csr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_prune_dense2csr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_prune_dense2csr_by_percentage.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_dense2csc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_csr2coo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_csr2bsr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_bsr2csr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_gebsr2csr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_csr2csr_compress.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_prune_csr2csr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_prune_csr2csr_by_percentage.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_coo2csr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_identity.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_csrsort.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_cscsort.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_coosort.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_csru2csr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_csrilusv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_gebsr2gebsr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_csr2gebsr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_gebsr2gebsc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spmat_descr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spvec_descr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_dnmat_descr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_dnvec_descr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_const_spmat_descr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_const_spvec_descr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_const_dnmat_descr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_const_dnvec_descr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spmv_coo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spmv_coo_aos.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spmv_csr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spmv_sell.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_axpby.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_gather.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_scatter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rot.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spvv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_dense_to_sparse_csr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_dense_to_sparse_csc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_dense_to_sparse_coo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_sparse_to_dense_csr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_sparse_to_dense_csc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_sparse_to_dense_coo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spmm_csr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spmm_batched_csr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spmm_csc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spmm_batched_csc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spmm_coo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spmm_batched_coo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spmm_bell.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spgemm_csr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spgemmreuse_csr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_sddmm_csr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_sddmm_csc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_sddmm_coo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_sddmm_coo_aos.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_gpsv_interleaved_batch.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_gtsv2_strided_batch.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_gtsv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_gtsv2_nopivot.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_gtsv_interleaved_batch.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_csrcolor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spsv_csr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spsv_coo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spsm_csr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_spsm_coo.cpp"
)

if(HIPSPARSE_ENABLE_HIP)
    target_sources(hipsparse-test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/test_doti.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_dotci.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_csr2csc.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_csr2csc_ex2.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_csrgemm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_csrgeam.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_csrmv.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_csrmm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_hybmv.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_csr2hyb.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_hyb2csr.cpp"
    )
endif()

# Include sources
if(HIPSPARSE_ENABLE_HIP)
  include(rocm/CMakeLists.txt)
elseif(HIPSPARSE_ENABLE_CUDA)
  if(CUDAToolkit_VERSION MATCHES "12.8")
    include(cuda/12.8/CMakeLists.txt)
    message("Building hipsparse-test executable using CUDAToolkit_VERSION ${CUDAToolkit_VERSION}")
  else()
    message("ERROR: Testing on CUDAToolkit_VERSION ${CUDAToolkit_VERSION} not supported.")
  endif()
endif()

target_compile_definitions(hipsparse-test PRIVATE GOOGLE_TEST)

target_link_libraries(hipsparse-test PRIVATE
    hipsparse::clients-common
    GTest::gtest
)

# GTest requires pthread on Unix platforms
if(NOT WIN32)
    target_link_libraries(hipsparse-test PRIVATE pthread)
endif()

if(HIPSPARSE_ENABLE_OPENMP)
    target_link_libraries(hipsparse-test PRIVATE OpenMP::OpenMP_CXX)
endif()

if(HIPSPARSE_ENABLE_CUDA)
    target_compile_definitions(hipsparse-test PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_link_libraries(hipsparse-test PRIVATE CUDA::cudart CUDA::cuda_driver)
endif()

set_target_properties(hipsparse-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
)

# Build mtx2csr converter
if(NOT CONVERT_SOURCE)
    set(CONVERT_SOURCE "${PROJECT_SOURCE_DIR}/deps/convert.cpp")
endif()

add_executable(mtx2csr)
target_sources(mtx2csr PRIVATE "${CONVERT_SOURCE}")
target_compile_options(mtx2csr PRIVATE -O3)
if(NOT WIN32)
    target_link_options(mtx2csr PRIVATE -Wl,--build-id=sha1)
endif()
set_target_properties(mtx2csr PROPERTIES
    OUTPUT_NAME "mtx2csr.exe"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients"
)

set(HIPSPARSE_CLIENTMATRICES "${PROJECT_SOURCE_DIR}/cmake/hipsparse_clientmatrices.cmake")

set(HIPSPARSE_CONVERT "${PROJECT_BINARY_DIR}/clients/hipsparse_mtx2csr")
add_custom_command(
    OUTPUT "${HIPSPARSE_CONVERT}"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mtx2csr> "${HIPSPARSE_CONVERT}"
    DEPENDS mtx2csr
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

set(HIPSPARSE_COMMON "${PROJECT_BINARY_DIR}/clients/staging/hipsparse_common.yaml")
add_custom_command(
    OUTPUT "${HIPSPARSE_COMMON}"
    COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/clients/include/hipsparse_common.yaml" "${HIPSPARSE_COMMON}"
    DEPENDS "${PROJECT_SOURCE_DIR}/clients/include/hipsparse_common.yaml"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

set(HIPSPARSE_TEMPLATE "${PROJECT_BINARY_DIR}/clients/staging/hipsparse_template.yaml")
add_custom_command(
    OUTPUT "${HIPSPARSE_TEMPLATE}"
    COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/clients/include/hipsparse_template.yaml" "${HIPSPARSE_TEMPLATE}"
    DEPENDS "${PROJECT_SOURCE_DIR}/clients/include/hipsparse_template.yaml"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

set(HIPSPARSE_GENTEST "${PROJECT_BINARY_DIR}/clients/staging/hipsparse_gentest.py")
add_custom_command(
    OUTPUT "${HIPSPARSE_GENTEST}"
    COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/clients/common/hipsparse_gentest.py" "${HIPSPARSE_GENTEST}"
    DEPENDS "${PROJECT_SOURCE_DIR}/clients/common/hipsparse_gentest.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_custom_target(hipsparse-common
    DEPENDS "${HIPSPARSE_COMMON}" "${HIPSPARSE_TEMPLATE}" "${HIPSPARSE_CLIENTMATRICES}" "${HIPSPARSE_GENTEST}" "${HIPSPARSE_CONVERT}"
)

# Prepare testing data
set(HIPSPARSE_TEST_DATA "${PROJECT_BINARY_DIR}/clients/staging/hipsparse_test.data")
add_custom_command(OUTPUT "${HIPSPARSE_TEST_DATA}"
                   COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/clients/common/hipsparse_gentest.py" -m ${PROJECT_BINARY_DIR}/clients/matrices -I "${PROJECT_SOURCE_DIR}/clients/include" "${HIPSPARSE_TEST_YAML_INCLUDE}" -o "${HIPSPARSE_TEST_DATA}"
                   DEPENDS "${PROJECT_SOURCE_DIR}/clients/common/hipsparse_gentest.py" "${HIPSPARSE_TEST_YAML_INCLUDE}" "${PROJECT_SOURCE_DIR}/clients/include/hipsparse_common.yaml" ${HIPSPARSE_TEST_YAMLS}
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_custom_target(hipsparse-test-data
  DEPENDS "${HIPSPARSE_TEST_DATA}" )

add_dependencies(hipsparse-test hipsparse-test-data hipsparse-common)


rocm_install(TARGETS hipsparse-test COMPONENT tests)
rocm_install(FILES ${HIPSPARSE_TEST_DATA} DESTINATION "${CMAKE_INSTALL_DATADIR}/hipsparse/test" COMPONENT tests)



# Only when both test_categories.yaml and shared/ctest exist (full rocm-libraries tree).
set(HIPSPARSE_CTEST_INSTALL_FILE "")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml")
  if(HIPSPARSE_HAS_CTEST_CATEGORIES)
    message(STATUS "hipSPARSE: YAML-based test categorization")
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    # Install-time CTestTestfile for TheRock: relative paths so ctest can run from installed location
    set(HIPSPARSE_CTEST_INSTALL_FILE "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
    file(WRITE "${HIPSPARSE_CTEST_INSTALL_FILE}"
[=[
# This is a test file generated by hipSPARSE for install time.
# Tests are defined with relative paths to work in the installed location.
]=]
    )
    apply_test_category_labels(
      hipsparse-test
      "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml"
      "${PROJECT_BINARY_DIR}/staging"
      "${HIPSPARSE_CTEST_INSTALL_FILE}"
    )
  else()
    message(STATUS "hipSPARSE: shared/ctest not found (e.g. standalone checkout), skipping CTest categories")
  endif()
else()
  message(STATUS "hipSPARSE: No test_categories.yaml (CTest categories not used)")
endif()

if(HIPSPARSE_CTEST_INSTALL_FILE)
    rocm_install(
      FILES "${HIPSPARSE_CTEST_INSTALL_FILE}"
      DESTINATION "${CMAKE_INSTALL_BINDIR}/hipsparse"
      COMPONENT tests
      RENAME "CTestTestfile.cmake"
  )
endif()

# Windows-specific DLL copying
if(WIN32)
    # For now adding in all .dll as dependency chain is not cmake based on win32
    file(GLOB third_party_dlls
        LIST_DIRECTORIES OFF
        CONFIGURE_DEPENDS
        $ENV{HIP_DIR}/bin/*.dll
        $ENV{HIP_DIR}/bin/hipinfo.exe
        ${ROCSPARSE_PATH}/bin/rocsparse.dll
        ${PROJECT_SOURCE_DIR}/rtest.*
        C:/Windows/System32/libomp140*.dll
    )
    foreach(file_i ${third_party_dlls})
        add_custom_command(TARGET hipsparse-test POST_BUILD 
            COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/clients/staging/)
    endforeach(file_i)
endif()
