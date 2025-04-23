!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! ==============================================================================
! hipfort: FORTRAN Interfaces for GPU kernels
! ==============================================================================
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
! [MITx11 License]
! 
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
! 
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
! 
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          
           
module hipfort_rocsparse_enums
  implicit none

  enum, bind(c)
    enumerator :: rocsparse_operation_none = 111
    enumerator :: rocsparse_operation_transpose = 112
    enumerator :: rocsparse_operation_conjugate_transpose = 113
  end enum

  enum, bind(c)
    enumerator :: rocsparse_index_base_zero = 0
    enumerator :: rocsparse_index_base_one = 1
  end enum

  enum, bind(c)
    enumerator :: rocsparse_matrix_type_general = 0
    enumerator :: rocsparse_matrix_type_symmetric = 1
    enumerator :: rocsparse_matrix_type_hermitian = 2
    enumerator :: rocsparse_matrix_type_triangular = 3
  end enum

  enum, bind(c)
    enumerator :: rocsparse_diag_type_non_unit = 0
    enumerator :: rocsparse_diag_type_unit = 1
  end enum

  enum, bind(c)
    enumerator :: rocsparse_fill_mode_lower = 0
    enumerator :: rocsparse_fill_mode_upper = 1
  end enum

  enum, bind(c)
    enumerator :: rocsparse_action_symbolic = 0
    enumerator :: rocsparse_action_numeric = 1
  end enum

  enum, bind(c)
    enumerator :: rocsparse_direction_row = 0
    enumerator :: rocsparse_direction_column = 1
  end enum

  enum, bind(c)
    enumerator :: rocsparse_hyb_partition_auto = 0
    enumerator :: rocsparse_hyb_partition_user = 1
    enumerator :: rocsparse_hyb_partition_max = 2
  end enum

  enum, bind(c)
    enumerator :: rocsparse_analysis_policy_reuse = 0
    enumerator :: rocsparse_analysis_policy_force = 1
  end enum

  enum, bind(c)
    enumerator :: rocsparse_solve_policy_auto = 0
  end enum

  enum, bind(c)
    enumerator :: rocsparse_pointer_mode_host = 0
    enumerator :: rocsparse_pointer_mode_device = 1
  end enum

  enum, bind(c)
    enumerator :: rocsparse_layer_mode_none = 0
    enumerator :: rocsparse_layer_mode_log_trace = 1
    enumerator :: rocsparse_layer_mode_log_bench = 2
  end enum

  enum, bind(c)
    enumerator :: rocsparse_status_success = 0
    enumerator :: rocsparse_status_invalid_handle = 1
    enumerator :: rocsparse_status_not_implemented = 2
    enumerator :: rocsparse_status_invalid_pointer = 3
    enumerator :: rocsparse_status_invalid_size = 4
    enumerator :: rocsparse_status_memory_error = 5
    enumerator :: rocsparse_status_internal_error = 6
    enumerator :: rocsparse_status_invalid_value = 7
    enumerator :: rocsparse_status_arch_mismatch = 8
    enumerator :: rocsparse_status_zero_pivot = 9
    enumerator :: rocsparse_status_not_initialized = 10
    enumerator :: rocsparse_status_type_mismatch = 11
  end enum

  enum, bind(c)
    enumerator :: rocsparse_indextype_u16 = 1
    enumerator :: rocsparse_indextype_i32 = 2
    enumerator :: rocsparse_indextype_i64 = 3
  end enum

  enum, bind(c)
    enumerator :: rocsparse_datatype_f32_r = 151
    enumerator :: rocsparse_datatype_f64_r = 152
    enumerator :: rocsparse_datatype_f32_c = 154
    enumerator :: rocsparse_datatype_f64_c = 155
  end enum

  enum, bind(c)
    enumerator :: rocsparse_format_coo = 0
    enumerator :: rocsparse_format_coo_aos = 1
    enumerator :: rocsparse_format_csr = 2
    enumerator :: rocsparse_format_csc = 3
    enumerator :: rocsparse_format_ell = 4
    enumerator :: rocsparse_format_bell = 5
  end enum

  enum, bind(c)
    enumerator :: rocsparse_order_row = 0
    enumerator :: rocsparse_order_column = 1
  end enum

  enum, bind(c)
    enumerator :: rocsparse_spmat_fill_mode = 0
    enumerator :: rocsparse_spmat_diag_type = 1
    enumerator :: rocsparse_spmat_matrix_type = 2
  end enum

  enum, bind(c)
    enumerator :: rocsparse_spmv_alg_default = 0
    enumerator :: rocsparse_spmv_alg_coo = 1
    enumerator :: rocsparse_spmv_alg_csr_adaptive = 2
    enumerator :: rocsparse_spmv_alg_csr_stream = 3
    enumerator :: rocsparse_spmv_alg_ell = 4
  end enum

  enum, bind(c)
    enumerator :: rocsparse_spsv_alg_default = 0
  end enum

  enum, bind(c)
    enumerator :: rocsparse_spsv_stage_auto = 0
    enumerator :: rocsparse_spsv_stage_buffer_size = 1
    enumerator :: rocsparse_spsv_stage_preprocess = 2
    enumerator :: rocsparse_spsv_stage_compute = 3
  end enum

  enum, bind(c)
    enumerator :: rocsparse_spsm_alg_default = 0
  end enum

  enum, bind(c)
    enumerator :: rocsparse_spsm_stage_auto = 0
    enumerator :: rocsparse_spsm_stage_buffer_size = 1
    enumerator :: rocsparse_spsm_stage_preprocess = 2
    enumerator :: rocsparse_spsm_stage_compute = 3
  end enum

  enum, bind(c)
    enumerator :: rocsparse_spmm_alg_default = 0
    enumerator :: rocsparse_spmm_alg_csr
    enumerator :: rocsparse_spmm_alg_coo_segmented
    enumerator :: rocsparse_spmm_alg_coo_atomic
    enumerator :: rocsparse_spmm_alg_csr_row_split
    enumerator :: rocsparse_spmm_alg_csr_merge
    enumerator :: rocsparse_spmm_alg_coo_segmented_atomic
    enumerator :: rocsparse_spmm_alg_bell
  end enum

  enum, bind(c)
    enumerator :: rocsparse_sddmm_alg_default = 0
  end enum

  enum, bind(c)
    enumerator :: rocsparse_sparse_to_dense_alg_default = 0
  end enum

  enum, bind(c)
    enumerator :: rocsparse_dense_to_sparse_alg_default = 0
  end enum

  enum, bind(c)
    enumerator :: rocsparse_spmm_stage_auto = 0
    enumerator :: rocsparse_spmm_stage_buffer_size = 1
    enumerator :: rocsparse_spmm_stage_preprocess = 2
    enumerator :: rocsparse_spmm_stage_compute = 3
  end enum

  enum, bind(c)
    enumerator :: rocsparse_spgemm_stage_auto = 0
    enumerator :: rocsparse_spgemm_stage_buffer_size = 1
    enumerator :: rocsparse_spgemm_stage_nnz = 2
    enumerator :: rocsparse_spgemm_stage_compute = 3
  end enum

  enum, bind(c)
    enumerator :: rocsparse_spgemm_alg_default = 0
  end enum

 

#ifdef USE_FPOINTER_INTERFACES

  
#endif
end module hipfort_rocsparse_enums