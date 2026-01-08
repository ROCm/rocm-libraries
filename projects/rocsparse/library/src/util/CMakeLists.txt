# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(rocsparse PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_check_matrix_coo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_check_matrix_coo_buffer_size.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_check_matrix_csc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_check_matrix_csc_buffer_size.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_check_matrix_csr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_check_matrix_csr_buffer_size.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_check_matrix_ell.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_check_matrix_ell_buffer_size.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_check_matrix_gebsc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_check_matrix_gebsc_buffer_size.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_check_matrix_gebsr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_check_matrix_gebsr_buffer_size.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_check_matrix_hyb.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_check_matrix_hyb_buffer_size.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_check_spmat.cpp
)
