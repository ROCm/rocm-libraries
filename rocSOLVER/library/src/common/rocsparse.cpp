/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.
 * ************************************************************************ */

#include "rocsparse.hpp"

ROCSOLVER_BEGIN_NAMESPACE

fp_rocsparse_create_handle g_sparse_create_handle;
fp_rocsparse_destroy_handle g_sparse_destroy_handle;
fp_rocsparse_set_stream g_sparse_set_stream;
fp_rocsparse_create_mat_descr g_sparse_create_mat_descr;
fp_rocsparse_destroy_mat_descr g_sparse_destroy_mat_descr;
fp_rocsparse_set_mat_type g_sparse_set_mat_type;
fp_rocsparse_set_mat_index_base g_sparse_set_mat_index_base;
fp_rocsparse_set_mat_fill_mode g_sparse_set_mat_fill_mode;
fp_rocsparse_set_mat_diag_type g_sparse_set_mat_diag_type;
fp_rocsparse_create_mat_info g_sparse_create_mat_info;
fp_rocsparse_destroy_mat_info g_sparse_destroy_mat_info;
fp_rocsparse_scsrilu0_buffer_size g_sparse_scsrilu0_buffer_size;
fp_rocsparse_dcsrilu0_buffer_size g_sparse_dcsrilu0_buffer_size;
fp_rocsparse_ccsrilu0_buffer_size g_sparse_ccsrilu0_buffer_size;
fp_rocsparse_zcsrilu0_buffer_size g_sparse_zcsrilu0_buffer_size;
fp_rocsparse_scsric0_buffer_size g_sparse_scsric0_buffer_size;
fp_rocsparse_dcsric0_buffer_size g_sparse_dcsric0_buffer_size;
fp_rocsparse_ccsric0_buffer_size g_sparse_ccsric0_buffer_size;
fp_rocsparse_zcsric0_buffer_size g_sparse_zcsric0_buffer_size;
fp_rocsparse_scsric0_analysis g_sparse_scsric0_analysis;
fp_rocsparse_dcsric0_analysis g_sparse_dcsric0_analysis;
fp_rocsparse_ccsric0_analysis g_sparse_ccsric0_analysis;
fp_rocsparse_zcsric0_analysis g_sparse_zcsric0_analysis;
fp_rocsparse_scsrsm_analysis g_sparse_scsrsm_analysis;
fp_rocsparse_dcsrsm_analysis g_sparse_dcsrsm_analysis;
fp_rocsparse_ccsrsm_analysis g_sparse_ccsrsm_analysis;
fp_rocsparse_zcsrsm_analysis g_sparse_zcsrsm_analysis;
fp_rocsparse_scsrsm_buffer_size g_sparse_scsrsm_buffer_size;
fp_rocsparse_dcsrsm_buffer_size g_sparse_dcsrsm_buffer_size;
fp_rocsparse_ccsrsm_buffer_size g_sparse_ccsrsm_buffer_size;
fp_rocsparse_zcsrsm_buffer_size g_sparse_zcsrsm_buffer_size;
fp_rocsparse_scsrsm_solve g_sparse_scsrsm_solve;
fp_rocsparse_dcsrsm_solve g_sparse_dcsrsm_solve;
fp_rocsparse_ccsrsm_solve g_sparse_ccsrsm_solve;
fp_rocsparse_zcsrsm_solve g_sparse_zcsrsm_solve;
fp_rocsparse_scsrilu0_analysis g_sparse_scsrilu0_analysis;
fp_rocsparse_dcsrilu0_analysis g_sparse_dcsrilu0_analysis;
fp_rocsparse_ccsrilu0_analysis g_sparse_ccsrilu0_analysis;
fp_rocsparse_zcsrilu0_analysis g_sparse_zcsrilu0_analysis;
fp_rocsparse_scsrilu0 g_sparse_scsrilu0;
fp_rocsparse_dcsrilu0 g_sparse_dcsrilu0;
fp_rocsparse_ccsrilu0 g_sparse_ccsrilu0;
fp_rocsparse_zcsrilu0 g_sparse_zcsrilu0;
fp_rocsparse_scsric0 g_sparse_scsric0;
fp_rocsparse_dcsric0 g_sparse_dcsric0;
fp_rocsparse_ccsric0 g_sparse_ccsric0;
fp_rocsparse_zcsric0 g_sparse_zcsric0;

ROCSOLVER_END_NAMESPACE
