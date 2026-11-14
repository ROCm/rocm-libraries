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
! rocsparse spildlt0 example (incomplete LDL^H factorization, level 0)
! see: https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/
!
! Exercises the generic staged SpILDLT0 preconditioner API: create the config
! descriptor, set inputs (algorithm, compute datatype, analysis policy), then run
! the analysis and compute stages (each with its own workspace queried by
! spildlt0_buffer_size). The matrix to factorize (A) is also passed as the
! preconditioner (P), as required. Success is verified by the post-compute
! singularity output being rocsparse_singularity_none.
!
! The descriptor/spmat/config arguments are c_ptr-only, so device buffers are
! passed via c_loc and scalar host inputs via c_loc(...).
!!!!!!!!!!!!!!
!
program spildlt0
  use iso_c_binding
  use hip
  use rocsparse
  implicit none
  integer(c_int), parameter :: m = 4, nnz = 7
  ! SPD lower-CSR (from the rocSPARSE example): factorizable without singularity.
  integer(c_int), target :: h_row_ptr(5) = (/0, 1, 3, 5, 7/)
  integer(c_int), target :: h_col_ind(7) = (/0, 0, 1, 1, 2, 2, 3/)
  real(c_double), target :: h_val(7) = (/4.0d0, 2.0d0, 8.0d0, 1.0d0, 8.0d0, 2.0d0, 4.0d0/)

  integer(c_int), pointer :: d_row_ptr(:), d_col_ind(:)
  real(c_double), pointer :: d_val(:)
  type(c_ptr) :: handle, matA, descr, dBuf
  integer(kind(rocsparse_spildlt0_alg_default)), target :: alg = rocsparse_spildlt0_alg_default
  integer(kind(rocsparse_analysis_policy_reuse)), target :: apol = rocsparse_analysis_policy_reuse
  integer(kind(rocsparse_datatype_f64_r)), target :: cdt = rocsparse_datatype_f64_r
  integer(kind(rocsparse_singularity_none)), target :: sing
  integer(c_size_t) :: bufSize
  write(*,"(a)",advance="no") "-- Running test 'rocsparse_spildlt0' (Fortran 2008 interfaces) - "

  call hipCheck(hipMalloc(d_row_ptr, source=h_row_ptr))
  call hipCheck(hipMalloc(d_col_ind, source=h_col_ind))
  call hipCheck(hipMalloc(d_val,     source=h_val))

  call rocsparseCheck(rocsparse_create_handle(handle))
  call rocsparseCheck(rocsparse_create_csr_descr(matA, int(m,c_int64_t), int(m,c_int64_t), int(nnz,c_int64_t), &
       c_loc(d_row_ptr), c_loc(d_col_ind), c_loc(d_val), &
       rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_datatype_f64_r))
  call rocsparseCheck(rocsparse_spildlt0_descr_create(handle, descr, c_null_ptr))

  call rocsparseCheck(rocsparse_spildlt0_set_input(handle, descr, rocsparse_spildlt0_input_alg, &
       c_loc(alg), int(c_sizeof(alg),c_size_t), c_null_ptr))
  call rocsparseCheck(rocsparse_spildlt0_set_input(handle, descr, rocsparse_spildlt0_input_compute_datatype, &
       c_loc(cdt), int(c_sizeof(cdt),c_size_t), c_null_ptr))
  call rocsparseCheck(rocsparse_spildlt0_set_input(handle, descr, rocsparse_spildlt0_input_analysis_policy, &
       c_loc(apol), int(c_sizeof(apol),c_size_t), c_null_ptr))

  ! Analysis stage.
  call rocsparseCheck(rocsparse_spildlt0_buffer_size(handle, descr, matA, matA, &
       rocsparse_spildlt0_stage_analysis, bufSize, c_null_ptr))
  call hipCheck(hipMalloc(dBuf, max(bufSize, 1_c_size_t)))
  call rocsparseCheck(rocsparse_spildlt0(handle, descr, matA, matA, &
       rocsparse_spildlt0_stage_analysis, bufSize, dBuf, c_null_ptr))
  call hipCheck(hipFree(dBuf))

  ! Compute stage.
  call rocsparseCheck(rocsparse_spildlt0_buffer_size(handle, descr, matA, matA, &
       rocsparse_spildlt0_stage_compute, bufSize, c_null_ptr))
  call hipCheck(hipMalloc(dBuf, max(bufSize, 1_c_size_t)))
  call rocsparseCheck(rocsparse_spildlt0(handle, descr, matA, matA, &
       rocsparse_spildlt0_stage_compute, bufSize, dBuf, c_null_ptr))
  call hipCheck(hipDeviceSynchronize())

  call rocsparseCheck(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host))
  sing = -1   ! sentinel: must be overwritten with rocsparse_singularity_none
  call rocsparseCheck(rocsparse_spildlt0_get_output(handle, descr, rocsparse_spildlt0_output_singularity, &
       c_loc(sing), int(c_sizeof(sing),c_size_t), c_null_ptr))

  if (sing /= rocsparse_singularity_none) then
     write(*,*) "FAILED! singularity = ", sing, " (expected none=0)"; call exit(1)
  end if

  call rocsparseCheck(rocsparse_spildlt0_descr_destroy(handle, descr, c_null_ptr))
  call rocsparseCheck(rocsparse_destroy_spmat_descr(matA))
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(d_row_ptr)); call hipCheck(hipFree(d_col_ind)); call hipCheck(hipFree(d_val))
  call hipCheck(hipFree(dBuf))
  write(*,*) "PASSED!"
end program spildlt0
