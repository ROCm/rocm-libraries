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
          
           
module hipfort_rocblas_enums
  implicit none

  enum, bind(c)
    enumerator :: rocblas_operation_none = 111
    enumerator :: rocblas_operation_transpose = 112
    enumerator :: rocblas_operation_conjugate_transpose = 113
  end enum

  enum, bind(c)
    enumerator :: rocblas_fill_upper = 121
    enumerator :: rocblas_fill_lower = 122
    enumerator :: rocblas_fill_full = 123
  end enum

  enum, bind(c)
    enumerator :: rocblas_diagonal_non_unit = 131
    enumerator :: rocblas_diagonal_unit = 132
  end enum

  enum, bind(c)
    enumerator :: rocblas_side_left = 141
    enumerator :: rocblas_side_right = 142
    enumerator :: rocblas_side_both = 143
  end enum

  enum, bind(c)
    enumerator :: rocblas_status_success = 0
    enumerator :: rocblas_status_invalid_handle = 1
    enumerator :: rocblas_status_not_implemented = 2
    enumerator :: rocblas_status_invalid_pointer = 3
    enumerator :: rocblas_status_invalid_size = 4
    enumerator :: rocblas_status_memory_error = 5
    enumerator :: rocblas_status_internal_error = 6
    enumerator :: rocblas_status_perf_degraded = 7
    enumerator :: rocblas_status_size_query_mismatch = 8
    enumerator :: rocblas_status_size_increased = 9
    enumerator :: rocblas_status_size_unchanged = 10
    enumerator :: rocblas_status_invalid_value = 11
    enumerator :: rocblas_status_continue = 12
    enumerator :: rocblas_status_check_numerics_fail = 13
  end enum

  enum, bind(c)
    enumerator :: rocblas_datatype_f16_r = 150
    enumerator :: rocblas_datatype_f32_r = 151
    enumerator :: rocblas_datatype_f64_r = 152
    enumerator :: rocblas_datatype_f16_c = 153
    enumerator :: rocblas_datatype_f32_c = 154
    enumerator :: rocblas_datatype_f64_c = 155
    enumerator :: rocblas_datatype_i8_r = 160
    enumerator :: rocblas_datatype_u8_r = 161
    enumerator :: rocblas_datatype_i32_r = 162
    enumerator :: rocblas_datatype_u32_r = 163
    enumerator :: rocblas_datatype_i8_c = 164
    enumerator :: rocblas_datatype_u8_c = 165
    enumerator :: rocblas_datatype_i32_c = 166
    enumerator :: rocblas_datatype_u32_c = 167
    enumerator :: rocblas_datatype_bf16_r = 168
    enumerator :: rocblas_datatype_bf16_c = 169
  end enum

  enum, bind(c)
    enumerator :: rocblas_pointer_mode_host = 0
    enumerator :: rocblas_pointer_mode_device = 1
  end enum

  enum, bind(c)
    enumerator :: rocblas_atomics_not_allowed = 0
    enumerator :: rocblas_atomics_allowed = 1
  end enum

  enum, bind(c)
    enumerator :: rocblas_default_performance_metric = 0
    enumerator :: rocblas_device_efficiency_performance_metric = 1
    enumerator :: rocblas_cu_efficiency_performance_metric = 2
  end enum

  enum, bind(c)
    enumerator :: rocblas_layer_mode_none = 0
    enumerator :: rocblas_layer_mode_log_trace = 1
    enumerator :: rocblas_layer_mode_log_bench = 2
    enumerator :: rocblas_layer_mode_log_profile = 4
  end enum

  enum, bind(c)
    enumerator :: rocblas_gemm_algo_standard = 0
  end enum

  enum, bind(c)
    enumerator :: rocblas_gemm_flags_none = 0
    enumerator :: rocblas_gemm_flags_pack_int8x4 = 1
    enumerator :: rocblas_gemm_flags_use_cu_efficiency = 2
    enumerator :: rocblas_gemm_flags_fp16_alt_impl = 4
  end enum

  enum, bind(c)
    enumerator :: rocblas_check_numerics_mode_no_check = 0
    enumerator :: rocblas_check_numerics_mode_info = 1
    enumerator :: rocblas_check_numerics_mode_warn = 2
    enumerator :: rocblas_check_numerics_mode_fail = 4
  end enum

 

#ifdef USE_FPOINTER_INTERFACES

  
#endif
end module hipfort_rocblas_enums