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

# MEMSTAT ?
if(BUILD_MEMSTAT)
  add_compile_options(-DROCSPARSE_WITH_MEMSTAT)
endif()

if(BUILD_ROCSPARSE_ILP64)
  add_compile_options(-Drocsparse_ILP64)
endif()

set(ROCSPARSE_BENCHMARK_SOURCES
  client.cpp
  rocsparse_bench_app.cpp
  rocsparse_arguments_config.cpp
  rocsparse_bench.cpp
  rocsparse_bench_cmdlines.cpp
  rocsparse_routine.cpp
)

set(ROCSPARSE_CLIENTS_COMMON
  ../common/utility.cpp
  ../common/rocsparse_reproducibility.cpp
  ../common/rocsparse_reproducibility_test.cpp
  ../common/rocsparse_reproducibility_test_data.cpp
  ../common/rocsparse_reproducibility_results.cpp
  ../common/rocsparse_random.cpp
  ../common/rocsparse_check.cpp
  ../common/rocsparse_enum.cpp
  ../common/rocsparse_enum_name.cpp
  ../common/rocsparse_enum_from_name.cpp
  ../common/rocsparse_init.cpp
  ../common/rocsparse_host.cpp
  ../common/rocsparse_vector_utils.cpp
  ../common/rocsparse_matrix_factory.cpp
  ../common/rocsparse_matrix_factory_laplace2d.cpp
  ../common/rocsparse_matrix_factory_laplace3d.cpp
  ../common/rocsparse_matrix_factory_zero.cpp
  ../common/rocsparse_matrix_factory_random.cpp
  ../common/rocsparse_matrix_factory_tridiagonal.cpp
  ../common/rocsparse_matrix_factory_pentadiagonal.cpp
  ../common/rocsparse_matrix_factory_file.cpp
  ../common/rocsparse_exporter_rocsparseio.cpp
  ../common/rocsparse_exporter_rocalution.cpp
  ../common/rocsparse_exporter_matrixmarket.cpp
  ../common/rocsparse_exporter_ascii.cpp
  ../common/rocsparse_exporter_format_t.cpp
  ../common/rocsparse_type_conversion.cpp
  ../common/rocsparse_importer_rocalution.cpp
  ../common/rocsparse_importer_rocsparseio.cpp
  ../common/rocsparse_importer_matrixmarket.cpp
  ../common/rocsparse_importer_mlbsr.cpp
  ../common/rocsparse_importer_mlcsr.cpp
  ../common/rocsparse_importer_format_t.cpp
  ../common/rocsparse_clients_envariables.cpp
  ../common/rocsparse_clients_matrices_dir.cpp
  ../common/rocsparseio.cpp
)


set(ROCSPARSE_CLIENTS_TESTINGS
../testings/testing_axpyi.cpp
../testings/testing_doti.cpp
../testings/testing_dotci.cpp
../testings/testing_gthr.cpp
../testings/testing_gthrz.cpp
../testings/testing_roti.cpp
../testings/testing_sctr.cpp
../testings/testing_bsrmv.cpp
../testings/testing_bsrxmv.cpp
../testings/testing_bsrsv.cpp
../testings/testing_coomv.cpp
../testings/testing_csrmv.cpp
../testings/testing_csrmv_managed.cpp
../testings/testing_csrsv.cpp
../testings/testing_csritsv.cpp
../testings/testing_ellmv.cpp
../testings/testing_hybmv.cpp
../testings/testing_gebsrmv.cpp
../testings/testing_bsrmm.cpp
../testings/testing_gebsrmm.cpp
../testings/testing_csrmm.cpp
../testings/testing_spmm_bsr.cpp
../testings/testing_spmm_csr.cpp
../testings/testing_spmm_csc.cpp
../testings/testing_spmm_coo.cpp
../testings/testing_spmm_bell.cpp
../testings/testing_spmm_batched_csr.cpp
../testings/testing_spmm_batched_csc.cpp
../testings/testing_spmm_batched_coo.cpp
../testings/testing_spmm_batched_bell.cpp
../testings/testing_csrsm.cpp
../testings/testing_bsrsm.cpp
../testings/testing_gemmi.cpp
../testings/testing_bsrgemm.cpp
../testings/testing_bsrgeam.cpp
../testings/testing_csrgeam.cpp
../testings/testing_csrgemm.cpp
../testings/testing_csrgemm_reuse.cpp
../testings/testing_bsric0.cpp
../testings/testing_bsrilu0.cpp
../testings/testing_csric0.cpp
../testings/testing_csrilu0.cpp
../testings/testing_csritilu0.cpp
../testings/testing_csritilu0_ex.cpp
../testings/testing_gpsv_interleaved_batch.cpp
../testings/testing_gtsv.cpp
../testings/testing_gtsv_no_pivot.cpp
../testings/testing_gtsv_no_pivot_strided_batch.cpp
../testings/testing_gtsv_interleaved_batch.cpp
../testings/testing_csr2coo.cpp
../testings/testing_csr2csc.cpp
../testings/testing_gebsr2gebsc.cpp
../testings/testing_gebsr2gebsr.cpp
../testings/testing_csr2ell.cpp
../testings/testing_csr2hyb.cpp
../testings/testing_csr2bsr.cpp
../testings/testing_csr2gebsr.cpp
../testings/testing_coo2csr.cpp
../testings/testing_ell2csr.cpp
../testings/testing_hyb2csr.cpp
../testings/testing_bsr2csr.cpp
../testings/testing_gebsr2csr.cpp
../testings/testing_csr2csr_compress.cpp
../testings/testing_prune_csr2csr.cpp
../testings/testing_prune_csr2csr_by_percentage.cpp
../testings/testing_identity.cpp
../testings/testing_inverse_permutation.cpp
../testings/testing_csrsort.cpp
../testings/testing_cscsort.cpp
../testings/testing_coosort.cpp
../testings/testing_csricsv.cpp
../testings/testing_csrilusv.cpp
../testings/testing_nnz.cpp
../testings/testing_dense2csr.cpp
../testings/testing_dense2coo.cpp
../testings/testing_prune_dense2csr.cpp
../testings/testing_prune_dense2csr_by_percentage.cpp
../testings/testing_dense2csc.cpp
../testings/testing_csr2dense.cpp
../testings/testing_csc2dense.cpp
../testings/testing_coo2dense.cpp
../testings/testing_spmv_bsr.cpp
../testings/testing_spmv_coo.cpp
../testings/testing_spmv_coo_aos.cpp
../testings/testing_spmv_csr.cpp
../testings/testing_spmv_csc.cpp
../testings/testing_spmv_ell.cpp
../testings/testing_v2_spmv_bsr.cpp
../testings/testing_v2_spmv_coo.cpp
../testings/testing_v2_spmv_coo_aos.cpp
../testings/testing_v2_spmv_csr.cpp
../testings/testing_v2_spmv_csc.cpp
../testings/testing_v2_spmv_ell.cpp
../testings/testing_spsv_csr.cpp
../testings/testing_spsv_coo.cpp
../testings/testing_sptrsv_csr.cpp
../testings/testing_sptrsv_coo.cpp
../testings/testing_spitsv_csr.cpp
../testings/testing_spsm_csr.cpp
../testings/testing_spsm_coo.cpp
../testings/testing_sptrsm_csr.cpp
../testings/testing_sptrsm_coo.cpp
../testings/testing_sparse_to_dense_coo.cpp
../testings/testing_sparse_to_dense_csr.cpp
../testings/testing_sparse_to_dense_csc.cpp
../testings/testing_dense_to_sparse_coo.cpp
../testings/testing_dense_to_sparse_csr.cpp
../testings/testing_dense_to_sparse_csc.cpp
../testings/testing_sparse_to_sparse.cpp
../testings/testing_extract.cpp
../testings/testing_spgemm_bsr.cpp
../testings/testing_spgemm_csr.cpp
../testings/testing_spgemm_reuse_csr.cpp
../testings/testing_spgeam_csr.cpp
../testings/testing_spgeam_reuse_csr.cpp
../testings/testing_spgeam_csr_2.cpp
../testings/testing_spgeam_reuse_csr_2.cpp
../testings/testing_gemvi.cpp
../testings/testing_sddmm.cpp
../testings/testing_csrcolor.cpp
../testings/testing_check_matrix_csr.cpp
../testings/testing_check_matrix_csc.cpp
../testings/testing_check_matrix_coo.cpp
../testings/testing_check_matrix_gebsr.cpp
../testings/testing_check_matrix_gebsc.cpp
../testings/testing_check_matrix_ell.cpp
../testings/testing_check_matrix_hyb.cpp
../testings/testing_check_spmat.cpp
../testings/testing_bsrpad_value.cpp
)

add_executable(rocsparse-bench ${ROCSPARSE_BENCHMARK_SOURCES} ${ROCSPARSE_CLIENTS_COMMON} ${ROCSPARSE_CLIENTS_TESTINGS})

# Target compile options
target_compile_options(rocsparse-bench PRIVATE -Wno-deprecated -Wno-unused-command-line-argument -Wall)

# Internal common header
target_include_directories(rocsparse-bench PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
if (BUILD_CLIENTS_ONLY)
  target_include_directories(rocsparse-bench PRIVATE ${rocsparse_INCLUDE_DIRS}/rocsparse)
endif()

# Add OpenMP if available
apply_omp_settings(rocsparse-bench)

# Target link libraries
target_link_libraries(rocsparse-bench PRIVATE roc::rocsparse hip::host hip::device ${ROCSPARSE_CLIENT_LIBS})
if (rocsparseio_FOUND)
  target_link_libraries(rocsparse-bench PRIVATE roc::rocsparseio)
endif()

if (NOT WIN32)
  if (BUILD_WITH_ROCTX)

    if(ROCTRACER_INCLUDE_DIR AND ROCTX_LIBRARY)
      # Add the include directory
      target_include_directories(rocsparse-bench PRIVATE ${ROCTRACER_INCLUDE_DIR})

      # Link the rocTx lib
      target_link_libraries(rocsparse-bench PRIVATE ${ROCTX_LIBRARY})
      target_compile_definitions( rocsparse-bench PRIVATE ROCSPARSE_CLIENTS_WITH_ROCTX )
    endif()
  endif()
endif()

# Set benchmark output directory
set_target_properties(rocsparse-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

rocm_install(TARGETS rocsparse-bench COMPONENT benchmarks)
