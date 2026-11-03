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
! hipSPARSE generic-API descriptor getters (Fortran 2003)
! see: https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Builds the four generic-API descriptors from known data and reads every
! attribute back out:
!   * CSR 3x4, nnz=5, 0-based  -> hipsparseSpMatGetFormat / GetIndexBase /
!                                 GetSize / hipsparseCsrGet
!   * COO 3x4, nnz=5, 1-based  -> hipsparseSpMatGetFormat / hipsparseCooGet
!     (deliberately 1-based so the index-base assertion cannot pass by
!      accident on a zeroed output variable)
!   * dense vector, size 4     -> hipsparseDnVecGet
!   * sparse vector, size 8, nnz 3 -> hipsparseSpVecGet
!
! Pass/fail: every recovered dimension, index type, index base and value type
! must equal what the descriptor was built with, and every recovered device
! pointer must be the exact address that was handed to the create call
! (c_associated). All scalar outputs are preset to -1 first, so a getter that
! writes nothing fails instead of silently matching a zero.
!
! No hipsparse handle is required: the descriptor API is handle-free.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program hipsparse_descr_get
  use iso_c_binding
  use hip
  use hipsparse
  implicit none

  ! A (3x4), nnz = 5:  A = [[1,0,2,0],[0,3,0,0],[4,0,0,5]]
  integer(c_int), parameter :: M = 3, N = 4, nnz = 5
  integer(c_int), target :: h_rowptr(4) = (/0, 2, 3, 5/)          ! CSR, 0-based
  integer(c_int), target :: h_colind(5) = (/0, 2, 1, 0, 3/)
  integer(c_int), target :: h_rowind(5) = (/1, 1, 2, 3, 3/)       ! COO, 1-based
  real(c_double), target :: h_val(5)    = (/1, 2, 3, 4, 5/)
  real(c_double), target :: h_x(4)      = (/1, 2, 3, 4/)          ! dense vector
  integer(c_int), target :: h_idx(3)    = (/0, 2, 3/)             ! sparse vector
  real(c_double), target :: h_sv(3)     = (/7, 8, 9/)

  type(c_ptr) :: d_rowptr, d_colind, d_rowind, d_val, d_x, d_idx, d_sv
  type(c_ptr) :: csr, coo, dnv, spv
  type(c_ptr) :: p_rowptr, p_colind, p_val, p_x, p_idx, p_sv

  integer(c_int64_t) :: r_out, c_out, nnz_out
  integer(c_int64_t) :: rec_r, rec_c, rec_nnz
  integer(c_int64_t) :: sz_out
  integer(kind(HIPSPARSE_FORMAT_CSR)) :: fmt
  integer(kind(HIPSPARSE_INDEX_BASE_ZERO)) :: base
  integer(kind(HIPSPARSE_INDEX_16U)) :: it1, it2
  integer(kind(HIP_R_32F)) :: vt

  write(*,"(a)",advance="no") "-- Running test 'hipsparse_descr_get' (Fortran 2003 interfaces) - "

  call hipCheck(hipMalloc(d_rowptr, int(M+1,c_size_t) * 4))
  call hipCheck(hipMalloc(d_colind, int(nnz,c_size_t) * 4))
  call hipCheck(hipMalloc(d_rowind, int(nnz,c_size_t) * 4))
  call hipCheck(hipMalloc(d_val,    int(nnz,c_size_t) * 8))
  call hipCheck(hipMalloc(d_x,      int(N,c_size_t)   * 8))
  call hipCheck(hipMalloc(d_idx,    int(3,c_size_t)   * 4))
  call hipCheck(hipMalloc(d_sv,     int(3,c_size_t)   * 8))
  call hipCheck(hipMemcpy(d_rowptr, c_loc(h_rowptr(1)), int(M+1,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_colind, c_loc(h_colind(1)), int(nnz,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_rowind, c_loc(h_rowind(1)), int(nnz,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_val,    c_loc(h_val(1)),    int(nnz,c_size_t) * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_x,      c_loc(h_x(1)),      int(N,c_size_t)   * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_idx,    c_loc(h_idx(1)),    int(3,c_size_t)   * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_sv,     c_loc(h_sv(1)),     int(3,c_size_t)   * 8, hipMemcpyHostToDevice))

  !!!!!!!!!!!!!! Part A: CSR, 0-based !!!!!!!!!!!!!!
  call hipsparseCheck(hipsparseCreateCsr(csr, int(M,c_int64_t), int(N,c_int64_t), int(nnz,c_int64_t), &
       d_rowptr, d_colind, d_val, &
       HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F))

  fmt = -1
  call hipsparseCheck(hipsparseSpMatGetFormat(csr, fmt))
  if (fmt /= HIPSPARSE_FORMAT_CSR) then
    write(*,*) "FAILED! hipsparseSpMatGetFormat(CSR) returned ", fmt, " expected ", HIPSPARSE_FORMAT_CSR
    STOP 1
  end if

  base = -1
  call hipsparseCheck(hipsparseSpMatGetIndexBase(csr, base))
  if (base /= HIPSPARSE_INDEX_BASE_ZERO) then
    write(*,*) "FAILED! hipsparseSpMatGetIndexBase(CSR) returned ", base, " expected ", HIPSPARSE_INDEX_BASE_ZERO
    STOP 1
  end if

  r_out = -1; c_out = -1; nnz_out = -1
  call hipsparseCheck(hipsparseSpMatGetSize(csr, r_out, c_out, nnz_out))
  if (r_out /= M .or. c_out /= N .or. nnz_out /= nnz) then
    write(*,*) "FAILED! hipsparseSpMatGetSize returned ", r_out, c_out, nnz_out, " expected ", M, N, nnz
    STOP 1
  end if
  rec_r = r_out; rec_c = c_out; rec_nnz = nnz_out

  r_out = -1; c_out = -1; nnz_out = -1
  it1 = -1; it2 = -1; base = -1; vt = -1
  p_rowptr = c_null_ptr; p_colind = c_null_ptr; p_val = c_null_ptr
  call hipsparseCheck(hipsparseCsrGet(csr, r_out, c_out, nnz_out, &
       p_rowptr, p_colind, p_val, it1, it2, base, vt))
  if (r_out /= M .or. c_out /= N .or. nnz_out /= nnz) then
    write(*,*) "FAILED! hipsparseCsrGet dimensions ", r_out, c_out, nnz_out, " expected ", M, N, nnz
    STOP 1
  end if
  if (it1 /= HIPSPARSE_INDEX_32I .or. it2 /= HIPSPARSE_INDEX_32I) then
    write(*,*) "FAILED! hipsparseCsrGet index types ", it1, it2, " expected ", HIPSPARSE_INDEX_32I
    STOP 1
  end if
  if (base /= HIPSPARSE_INDEX_BASE_ZERO .or. vt /= HIP_R_64F) then
    write(*,*) "FAILED! hipsparseCsrGet base/value type ", base, vt, " expected ", &
               HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F
    STOP 1
  end if
  if (.not. (c_associated(p_rowptr, d_rowptr) .and. c_associated(p_colind, d_colind) &
             .and. c_associated(p_val, d_val))) then
    write(*,*) "FAILED! hipsparseCsrGet did not return the device pointers it was given"
    STOP 1
  end if

  call hipsparseCheck(hipsparseDestroySpMat(csr))

  !!!!!!!!!!!!!! Part B: COO, 1-based !!!!!!!!!!!!!!
  call hipsparseCheck(hipsparseCreateCoo(coo, int(M,c_int64_t), int(N,c_int64_t), int(nnz,c_int64_t), &
       d_rowind, d_colind, d_val, &
       HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ONE, HIP_R_64F))

  fmt = -1
  call hipsparseCheck(hipsparseSpMatGetFormat(coo, fmt))
  if (fmt /= HIPSPARSE_FORMAT_COO) then
    write(*,*) "FAILED! hipsparseSpMatGetFormat(COO) returned ", fmt, " expected ", HIPSPARSE_FORMAT_COO
    STOP 1
  end if

  r_out = -1; c_out = -1; nnz_out = -1
  it1 = -1; base = -1; vt = -1
  p_rowptr = c_null_ptr; p_colind = c_null_ptr; p_val = c_null_ptr
  call hipsparseCheck(hipsparseCooGet(coo, r_out, c_out, nnz_out, &
       p_rowptr, p_colind, p_val, it1, base, vt))
  if (r_out /= M .or. c_out /= N .or. nnz_out /= nnz) then
    write(*,*) "FAILED! hipsparseCooGet dimensions ", r_out, c_out, nnz_out, " expected ", M, N, nnz
    STOP 1
  end if
  if (it1 /= HIPSPARSE_INDEX_32I .or. base /= HIPSPARSE_INDEX_BASE_ONE .or. vt /= HIP_R_64F) then
    write(*,*) "FAILED! hipsparseCooGet index type/base/value type ", it1, base, vt, " expected ", &
               HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ONE, HIP_R_64F
    STOP 1
  end if
  if (.not. (c_associated(p_rowptr, d_rowind) .and. c_associated(p_colind, d_colind) &
             .and. c_associated(p_val, d_val))) then
    write(*,*) "FAILED! hipsparseCooGet did not return the device pointers it was given"
    STOP 1
  end if

  call hipsparseCheck(hipsparseDestroySpMat(coo))

  !!!!!!!!!!!!!! Part C: dense vector !!!!!!!!!!!!!!
  call hipsparseCheck(hipsparseCreateDnVec(dnv, int(N,c_int64_t), d_x, HIP_R_64F))

  sz_out = -1; vt = -1; p_x = c_null_ptr
  call hipsparseCheck(hipsparseDnVecGet(dnv, sz_out, p_x, vt))
  if (sz_out /= N .or. vt /= HIP_R_64F) then
    write(*,*) "FAILED! hipsparseDnVecGet size/value type ", sz_out, vt, " expected ", N, HIP_R_64F
    STOP 1
  end if
  if (.not. c_associated(p_x, d_x)) then
    write(*,*) "FAILED! hipsparseDnVecGet did not return the device pointer it was given"
    STOP 1
  end if

  call hipsparseCheck(hipsparseDestroyDnVec(dnv))

  !!!!!!!!!!!!!! Part D: sparse vector !!!!!!!!!!!!!!
  call hipsparseCheck(hipsparseCreateSpVec(spv, 8_c_int64_t, 3_c_int64_t, d_idx, d_sv, &
       HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F))

  sz_out = -1; nnz_out = -1; it1 = -1; base = -1; vt = -1
  p_idx = c_null_ptr; p_sv = c_null_ptr
  call hipsparseCheck(hipsparseSpVecGet(spv, sz_out, nnz_out, p_idx, p_sv, it1, base, vt))
  if (sz_out /= 8 .or. nnz_out /= 3) then
    write(*,*) "FAILED! hipsparseSpVecGet size/nnz ", sz_out, nnz_out, " expected ", 8, 3
    STOP 1
  end if
  if (it1 /= HIPSPARSE_INDEX_32I .or. base /= HIPSPARSE_INDEX_BASE_ZERO .or. vt /= HIP_R_64F) then
    write(*,*) "FAILED! hipsparseSpVecGet index type/base/value type ", it1, base, vt, " expected ", &
               HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F
    STOP 1
  end if
  if (.not. (c_associated(p_idx, d_idx) .and. c_associated(p_sv, d_sv))) then
    write(*,*) "FAILED! hipsparseSpVecGet did not return the device pointers it was given"
    STOP 1
  end if

  call hipsparseCheck(hipsparseDestroySpVec(spv))

  call hipCheck(hipFree(d_rowptr)); call hipCheck(hipFree(d_colind))
  call hipCheck(hipFree(d_rowind)); call hipCheck(hipFree(d_val))
  call hipCheck(hipFree(d_x)); call hipCheck(hipFree(d_idx)); call hipCheck(hipFree(d_sv))

  write(*,"(a,i0,a,i0,a,i0,a)") " PASSED! CSR/COO/DnVec/SpVec descriptors round-trip (recovered ", &
       rec_r, "x", rec_c, " nnz=", rec_nnz, ")"

end program hipsparse_descr_get
