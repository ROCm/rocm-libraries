# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(rocsparse-bench
    client.cpp
    rocsparse_bench_app.cpp
    rocsparse_arguments_config.cpp
    rocsparse_bench.cpp
    rocsparse_bench_cmdlines.cpp
    rocsparse_routine.cpp
    # Testing implementations (reused from tests)
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_axpyi.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_doti.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_dotci.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_gthr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_gthrz.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_roti.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_sctr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_bsrmv.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_bsrxmv.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_bsrsv.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_coomv.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csrmv.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csrmv_managed.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csrsv.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csritsv.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_ellmv.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_hybmv.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_gebsrmv.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_bsrmm.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_gebsrmm.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csrmm.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spmm_bsr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spmm_csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spmm_csc.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spmm_coo.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spmm_bell.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spmm_batched_csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spmm_batched_csc.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spmm_batched_coo.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spmm_batched_bell.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csrsm.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_bsrsm.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_gemmi.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_bsrgemm.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_bsrgeam.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csrgeam.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csrgemm.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csrgemm_reuse.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_bsric0.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_bsrilu0.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csric0.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csrilu0.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csritilu0.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csritilu0_ex.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_gpsv_interleaved_batch.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_gtsv.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_gtsv_no_pivot.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_gtsv_no_pivot_strided_batch.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_gtsv_interleaved_batch.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csr2coo.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csr2csc.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_gebsr2gebsc.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_gebsr2gebsr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csr2ell.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csr2hyb.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csr2bsr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csr2gebsr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_coo2csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_ell2csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_hyb2csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_bsr2csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_gebsr2csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csr2csr_compress.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_prune_csr2csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_prune_csr2csr_by_percentage.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_identity.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_inverse_permutation.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csrsort.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_cscsort.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_coosort.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csricsv.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csrilusv.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_nnz.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_dense2csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_dense2coo.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_prune_dense2csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_prune_dense2csr_by_percentage.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_dense2csc.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csr2dense.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csc2dense.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_coo2dense.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spmv_bsr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spmv_coo.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spmv_coo_aos.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spmv_csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spmv_csc.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spmv_ell.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_v2_spmv_bsr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_v2_spmv_coo.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_v2_spmv_coo_aos.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_v2_spmv_csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_v2_spmv_csc.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_v2_spmv_ell.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_v2_spmv_sell.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spsv_csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spsv_coo.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_sptrsv_csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_sptrsv_coo.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spitsv_csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spsm_csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spsm_coo.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_sptrsm_csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_sptrsm_coo.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_sparse_to_dense_coo.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_sparse_to_dense_csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_sparse_to_dense_csc.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_dense_to_sparse_coo.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_dense_to_sparse_csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_dense_to_sparse_csc.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_sparse_to_sparse.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_extract.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spgemm_bsr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spgemm_csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spgemm_reuse_csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spgeam_csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spgeam_reuse_csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spgeam_csr_2.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_spgeam_reuse_csr_2.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_gemvi.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_sddmm.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_csrcolor.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_check_matrix_csr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_check_matrix_csc.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_check_matrix_coo.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_check_matrix_gebsr.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_check_matrix_gebsc.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_check_matrix_ell.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_check_matrix_hyb.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_check_spmat.cpp
    ${PROJECT_SOURCE_DIR}/clients/testings/testing_bsrpad_value.cpp
    # Common utilities
    ${PROJECT_SOURCE_DIR}/clients/common/utility.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_reproducibility.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_reproducibility_test.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_reproducibility_test_data.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_reproducibility_results.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_random.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_check.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_enum.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_enum_name.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_enum_from_name.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_init.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_host.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_vector_utils.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_matrix_factory.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_matrix_factory_laplace2d.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_matrix_factory_laplace3d.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_matrix_factory_zero.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_matrix_factory_random.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_matrix_factory_tridiagonal.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_matrix_factory_pentadiagonal.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_matrix_factory_file.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_exporter_rocsparseio.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_exporter_rocalution.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_exporter_matrixmarket.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_exporter_ascii.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_exporter_format_t.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_type_conversion.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_importer_rocalution.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_importer_rocsparseio.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_importer_matrixmarket.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_importer_mlbsr.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_importer_mlcsr.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_importer_format_t.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_clients_envariables.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparse_clients_matrices_dir.cpp
    ${PROJECT_SOURCE_DIR}/clients/common/rocsparseio.cpp
)

if(ROCSPARSE_ENABLE_MEMSTAT)
    target_compile_definitions(rocsparse-bench PRIVATE ROCSPARSE_WITH_MEMSTAT)
endif()

if(ROCSPARSE_ENABLE_ILP64)
    target_compile_definitions(rocsparse-bench PRIVATE rocsparse_ILP64)
endif()

target_compile_options(rocsparse-bench PRIVATE
    -Wno-deprecated
    -Wno-unused-command-line-argument
    -Wall
)

target_compile_features(rocsparse-bench PRIVATE cxx_std_17)

target_include_directories(rocsparse-bench PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/clients/include>
)

# OpenMP RPATH handling for hip-clang
if(NOT WIN32 AND OPENMP_FOUND)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        if(TARGET OpenMP::OpenMP_CXX)
            set_target_properties(rocsparse-bench PROPERTIES
                BUILD_RPATH "${HIP_CLANG_ROOT}/lib"
                INSTALL_RPATH "$ORIGIN/../llvm/lib"
            )
        elseif(TARGET OpenMP::omp)
            if(DEFINED openmp_LIB_DIR)
                set_target_properties(rocsparse-bench PROPERTIES
                    BUILD_RPATH "${HIP_CLANG_ROOT}/${openmp_LIB_DIR}"
                    INSTALL_RPATH "$ORIGIN/../llvm/${openmp_LIB_DIR}"
                )
            endif()
        endif()
    endif()
endif()

target_link_libraries(rocsparse-bench PRIVATE
    roc::rocsparse
    hip::host
    hip::device
)

if(OPENMP_FOUND)
    if(TARGET OpenMP::omp)
        target_link_libraries(rocsparse-bench PRIVATE OpenMP::omp)
    elseif(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(rocsparse-bench PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

if(rocsparseio_FOUND)
    target_link_libraries(rocsparse-bench PRIVATE roc::rocsparseio)
endif()

if(NOT WIN32 AND ROCSPARSE_ENABLE_MARKER)
    if(ROCTRACER_INCLUDE_DIR AND ROCTX_LIBRARY)
        target_include_directories(rocsparse-bench PRIVATE ${ROCTRACER_INCLUDE_DIR})
        target_link_libraries(rocsparse-bench PRIVATE ${ROCTX_LIBRARY})
        target_compile_definitions(rocsparse-bench PRIVATE ROCSPARSE_CLIENTS_WITH_ROCTX)
    endif()
endif()

rocm_install(TARGETS rocsparse-bench COMPONENT benchmarks DESTINATION "${CMAKE_INSTALL_BINDIR}")
