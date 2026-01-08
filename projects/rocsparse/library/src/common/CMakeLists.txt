# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(rocsparse PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_argdescr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_assign_async.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_blas.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_blas_rocblas.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_conjugate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_csric0_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_csrilu0_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_datatype_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_debug.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_dense_transpose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_dense_transpose_back.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_determine_indextype.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_envariables.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_handle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_indextype_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_mat_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_memstat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_message.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_one.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_pivot_info_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_position_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_roctx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_singular_info_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_sorted_coo2csr_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_trm_data_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_trm_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_trm_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_valset.cpp
)
