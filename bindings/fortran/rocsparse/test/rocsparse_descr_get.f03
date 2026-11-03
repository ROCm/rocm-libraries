!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
!
! SPDX-License-Identifier: MIT
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
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!
! rocSPARSE generic descriptor getters (Fortran 2003)
! see: https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/
!
! Builds four descriptors over the same device buffers -- a zero-based CSR
! matrix, a one-based COO matrix, a dense vector and a sparse vector -- and
! then reads every attribute back with the "get" family:
!   rocsparse_spmat_get_format, rocsparse_spmat_get_index_base,
!   rocsparse_csr_get, rocsparse_coo_get, rocsparse_dnvec_get,
!   rocsparse_spvec_get.
! The matrix is deliberately non-square (m=4, n=5, nnz=7) so that a getter
! swapping rows and cols cannot go unnoticed, and the COO copy uses a
! one-based index base so that the base assertion is not trivially the
! default. Every output variable is poisoned before the call and compared
! against the value handed to the corresponding create routine, including
! the device pointers themselves (c_associated).
!
! The validity of the CSR data is then confirmed on the GPU twice, through
! the generic rocsparse_check_spmat and through the typed
! rocsparse_dcheck_matrix_csr (whose workspace comes from
! rocsparse_dcheck_matrix_csr_buffer_size); both must report
! rocsparse_data_status_success.
!
! The "get" routines return their outputs through Fortran scalars, passed
! directly rather than through c_loc.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program rocsparse_descr_get
  use iso_c_binding
  use hip
  use rocsparse
  implicit none

  integer(c_int), parameter :: m = 4, n = 5, nnz = 7
  integer(c_int64_t), parameter :: spv_size = 8, spv_nnz = 4

  integer(c_int), target :: h_row_ptr(5) = (/0, 1, 3, 5, 7/)
  integer(c_int), target :: h_col_ind(7) = (/0, 0, 1, 1, 2, 2, 3/)
  integer(c_int), target :: h_row_ind(7) = (/1, 2, 2, 3, 3, 4, 4/)
  real(c_double), target :: h_val(7) = (/4.0d0, 2.0d0, 8.0d0, 1.0d0, 8.0d0, 2.0d0, 4.0d0/)

  type(c_ptr) :: d_row_ptr, d_row_ind, d_col_ind, d_val
  type(c_ptr) :: handle, matA, matC, vecX, spv, dBuf

  integer(c_int64_t) :: g_rows, g_cols, g_nnz
  integer(c_int64_t) :: s_rows, s_cols, s_nnz
  integer(c_int64_t) :: g_size
  type(c_ptr) :: g_a, g_b, g_c
  integer(kind(rocsparse_indextype_i32)) :: g_rpt, g_cit, g_it
  integer(kind(rocsparse_index_base_zero)) :: g_base
  integer(kind(rocsparse_datatype_f16_r)) :: g_dt
  integer(kind(rocsparse_format_coo)) :: g_fmt
  integer(kind(rocsparse_data_status_success)) :: dstat
  integer(c_size_t) :: bsz

  write(*,"(a)",advance="no") "-- Running test 'rocsparse_descr_get' (Fortran 2003 interfaces) - "

  call hipCheck(hipMalloc(d_row_ptr, int(m+1,c_size_t) * 4))
  call hipCheck(hipMalloc(d_row_ind, int(nnz,c_size_t) * 4))
  call hipCheck(hipMalloc(d_col_ind, int(nnz,c_size_t) * 4))
  call hipCheck(hipMalloc(d_val,     int(nnz,c_size_t) * 8))
  call hipCheck(hipMemcpy(d_row_ptr, c_loc(h_row_ptr(1)), int(m+1,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_row_ind, c_loc(h_row_ind(1)), int(nnz,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_col_ind, c_loc(h_col_ind(1)), int(nnz,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_val,     c_loc(h_val(1)),     int(nnz,c_size_t) * 8, hipMemcpyHostToDevice))

  call rocsparseCheck(rocsparse_create_handle(handle))

  call rocsparseCheck(rocsparse_create_csr_descr(matA, int(m,c_int64_t), int(n,c_int64_t), int(nnz,c_int64_t), &
       d_row_ptr, d_col_ind, d_val, &
       rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_datatype_f64_r))
  call rocsparseCheck(rocsparse_create_coo_descr(matC, int(m,c_int64_t), int(n,c_int64_t), int(nnz,c_int64_t), &
       d_row_ind, d_col_ind, d_val, &
       rocsparse_indextype_i32, rocsparse_index_base_one, rocsparse_datatype_f64_r))
  call rocsparseCheck(rocsparse_create_dnvec_descr(vecX, int(n,c_int64_t), d_val, rocsparse_datatype_f64_r))
  call rocsparseCheck(rocsparse_create_spvec_descr(spv, spv_size, spv_nnz, d_col_ind, d_val, &
       rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_datatype_f64_r))

  ! ---- format ---------------------------------------------------------------
  g_fmt = -1
  call rocsparseCheck(rocsparse_spmat_get_format(matA, g_fmt))
  if (g_fmt /= rocsparse_format_csr) then
     write(*,*) "FAILED! spmat_get_format(matA) expected ", rocsparse_format_csr, " got ", g_fmt
     STOP 1
  end if
  g_fmt = -1
  call rocsparseCheck(rocsparse_spmat_get_format(matC, g_fmt))
  if (g_fmt /= rocsparse_format_coo) then
     write(*,*) "FAILED! spmat_get_format(matC) expected ", rocsparse_format_coo, " got ", g_fmt
     STOP 1
  end if

  ! ---- index base -----------------------------------------------------------
  g_base = -1
  call rocsparseCheck(rocsparse_spmat_get_index_base(matA, g_base))
  if (g_base /= rocsparse_index_base_zero) then
     write(*,*) "FAILED! spmat_get_index_base(matA) expected ", rocsparse_index_base_zero, " got ", g_base
     STOP 1
  end if
  g_base = -1
  call rocsparseCheck(rocsparse_spmat_get_index_base(matC, g_base))
  if (g_base /= rocsparse_index_base_one) then
     write(*,*) "FAILED! spmat_get_index_base(matC) expected ", rocsparse_index_base_one, " got ", g_base
     STOP 1
  end if

  ! ---- csr_get --------------------------------------------------------------
  g_rows = -1; g_cols = -1; g_nnz = -1
  g_rpt = -1; g_cit = -1; g_base = -1; g_dt = -1
  g_a = c_null_ptr; g_b = c_null_ptr; g_c = c_null_ptr
  call rocsparseCheck(rocsparse_csr_get(matA, g_rows, g_cols, g_nnz, g_a, g_b, g_c, &
       g_rpt, g_cit, g_base, g_dt))
  if (g_rows /= int(m,c_int64_t) .or. g_cols /= int(n,c_int64_t) .or. g_nnz /= int(nnz,c_int64_t)) then
     write(*,*) "FAILED! csr_get dimensions expected ", m, n, nnz, " got ", g_rows, g_cols, g_nnz
     STOP 1
  end if
  if (g_rpt /= rocsparse_indextype_i32 .or. g_cit /= rocsparse_indextype_i32) then
     write(*,*) "FAILED! csr_get index types expected ", rocsparse_indextype_i32, " got ", g_rpt, g_cit
     STOP 1
  end if
  if (g_base /= rocsparse_index_base_zero .or. g_dt /= rocsparse_datatype_f64_r) then
     write(*,*) "FAILED! csr_get base/datatype expected ", rocsparse_index_base_zero, rocsparse_datatype_f64_r, &
          " got ", g_base, g_dt
     STOP 1
  end if
  if (.not. (c_associated(g_a, d_row_ptr) .and. c_associated(g_b, d_col_ind) .and. c_associated(g_c, d_val))) then
     write(*,*) "FAILED! csr_get expected the device pointers handed to create_csr_descr, got different ones"
     STOP 1
  end if
  s_rows = g_rows; s_cols = g_cols; s_nnz = g_nnz

  ! ---- coo_get --------------------------------------------------------------
  g_rows = -1; g_cols = -1; g_nnz = -1
  g_it = -1; g_base = -1; g_dt = -1
  g_a = c_null_ptr; g_b = c_null_ptr; g_c = c_null_ptr
  call rocsparseCheck(rocsparse_coo_get(matC, g_rows, g_cols, g_nnz, g_a, g_b, g_c, &
       g_it, g_base, g_dt))
  if (g_rows /= int(m,c_int64_t) .or. g_cols /= int(n,c_int64_t) .or. g_nnz /= int(nnz,c_int64_t)) then
     write(*,*) "FAILED! coo_get dimensions expected ", m, n, nnz, " got ", g_rows, g_cols, g_nnz
     STOP 1
  end if
  if (g_it /= rocsparse_indextype_i32 .or. g_base /= rocsparse_index_base_one .or. &
      g_dt /= rocsparse_datatype_f64_r) then
     write(*,*) "FAILED! coo_get idxtype/base/datatype expected ", rocsparse_indextype_i32, &
          rocsparse_index_base_one, rocsparse_datatype_f64_r, " got ", g_it, g_base, g_dt
     STOP 1
  end if
  if (.not. (c_associated(g_a, d_row_ind) .and. c_associated(g_b, d_col_ind) .and. c_associated(g_c, d_val))) then
     write(*,*) "FAILED! coo_get expected the device pointers handed to create_coo_descr, got different ones"
     STOP 1
  end if

  ! ---- dnvec_get ------------------------------------------------------------
  g_size = -1; g_dt = -1; g_c = c_null_ptr
  call rocsparseCheck(rocsparse_dnvec_get(vecX, g_size, g_c, g_dt))
  if (g_size /= int(n,c_int64_t) .or. g_dt /= rocsparse_datatype_f64_r) then
     write(*,*) "FAILED! dnvec_get expected size ", n, " datatype ", rocsparse_datatype_f64_r, &
          " got ", g_size, g_dt
     STOP 1
  end if
  if (.not. c_associated(g_c, d_val)) then
     write(*,*) "FAILED! dnvec_get expected the device pointer handed to create_dnvec_descr"
     STOP 1
  end if

  ! ---- spvec_get ------------------------------------------------------------
  g_size = -1; g_nnz = -1; g_it = -1; g_base = -1; g_dt = -1
  g_b = c_null_ptr; g_c = c_null_ptr
  call rocsparseCheck(rocsparse_spvec_get(spv, g_size, g_nnz, g_b, g_c, g_it, g_base, g_dt))
  if (g_size /= spv_size .or. g_nnz /= spv_nnz) then
     write(*,*) "FAILED! spvec_get expected size/nnz ", spv_size, spv_nnz, " got ", g_size, g_nnz
     STOP 1
  end if
  if (g_it /= rocsparse_indextype_i32 .or. g_base /= rocsparse_index_base_zero .or. &
      g_dt /= rocsparse_datatype_f64_r) then
     write(*,*) "FAILED! spvec_get idxtype/base/datatype expected ", rocsparse_indextype_i32, &
          rocsparse_index_base_zero, rocsparse_datatype_f64_r, " got ", g_it, g_base, g_dt
     STOP 1
  end if
  if (.not. (c_associated(g_b, d_col_ind) .and. c_associated(g_c, d_val))) then
     write(*,*) "FAILED! spvec_get expected the device pointers handed to create_spvec_descr"
     STOP 1
  end if

  ! ---- generic check_spmat on the CSR descriptor ----------------------------
  bsz = 0; dstat = -1
  call rocsparseCheck(rocsparse_check_spmat(handle, matA, dstat, &
       rocsparse_check_spmat_stage_buffer_size, bsz, c_null_ptr))
  if (bsz <= 0) then
     write(*,*) "FAILED! check_spmat buffer size expected > 0, got ", bsz
     STOP 1
  end if
  call hipCheck(hipMalloc(dBuf, max(bsz, 1_c_size_t)))
  call rocsparseCheck(rocsparse_check_spmat(handle, matA, dstat, &
       rocsparse_check_spmat_stage_compute, bsz, dBuf))
  call hipCheck(hipDeviceSynchronize())
  if (dstat /= rocsparse_data_status_success) then
     write(*,*) "FAILED! check_spmat data status expected ", rocsparse_data_status_success, " got ", dstat
     STOP 1
  end if
  call hipCheck(hipFree(dBuf))

  ! ---- typed check_matrix_csr on the raw arrays -----------------------------
  bsz = 0; dstat = -1
  call rocsparseCheck(rocsparse_dcheck_matrix_csr_buffer_size(handle, m, n, nnz, d_val, d_row_ptr, d_col_ind, &
       rocsparse_index_base_zero, rocsparse_matrix_type_general, rocsparse_fill_mode_lower, &
       rocsparse_storage_mode_sorted, bsz))
  if (bsz <= 0) then
     write(*,*) "FAILED! dcheck_matrix_csr_buffer_size expected > 0, got ", bsz
     STOP 1
  end if
  call hipCheck(hipMalloc(dBuf, max(bsz, 1_c_size_t)))
  call rocsparseCheck(rocsparse_dcheck_matrix_csr(handle, m, n, nnz, d_val, d_row_ptr, d_col_ind, &
       rocsparse_index_base_zero, rocsparse_matrix_type_general, rocsparse_fill_mode_lower, &
       rocsparse_storage_mode_sorted, dstat, dBuf))
  call hipCheck(hipDeviceSynchronize())
  if (dstat /= rocsparse_data_status_success) then
     write(*,*) "FAILED! dcheck_matrix_csr data status expected ", rocsparse_data_status_success, " got ", dstat
     STOP 1
  end if

  call rocsparseCheck(rocsparse_destroy_spvec_descr(spv))
  call rocsparseCheck(rocsparse_destroy_dnvec_descr(vecX))
  call rocsparseCheck(rocsparse_destroy_spmat_descr(matC))
  call rocsparseCheck(rocsparse_destroy_spmat_descr(matA))
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(dBuf))
  call hipCheck(hipFree(d_row_ptr)); call hipCheck(hipFree(d_row_ind))
  call hipCheck(hipFree(d_col_ind)); call hipCheck(hipFree(d_val))

  write(*,*) "PASSED! CSR/COO/dnvec/spvec descriptors round-trip m=", s_rows, " n=", s_cols, " nnz=", s_nnz

end program rocsparse_descr_get
