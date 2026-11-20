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
! hipsparse SpMV example, device pointer mode (single, y = alpha*A*x + beta*y)
! see: https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Device-pointer-mode variant of hipsparse_sspmv.f03. The handle is switched to
! HIPSPARSE_POINTER_MODE_DEVICE and alpha/beta are staged in device memory, so
! hipSPARSE reads the scalars with a device load rather than a host one. That is
! what exercises the type(c_ptr),value scalar binding end to end; the host-mode
! sibling only proves it compiles.
!
! The scalars go to both SpMV_bufferSize and SpMV, which is the case that proves
! the query stage does not dereference them on the host.
!
! h_expected is still the host-side expected result, so a library that read the
! scalars from host memory would produce a wrong answer rather than pass.
!!!!!!!!!!!!!!
!
program hipsparse_sspmv_devptr
  use iso_c_binding
  use hip
  use hipsparse
  implicit none
  integer :: i

  ! Sparse A (3x3) in CSR (0-based):  A = [[1,0,2],[0,3,0],[4,0,5]]
  integer(c_int), parameter :: M = 3, N = 3, nnz = 5
  integer(c_int), target :: h_csr_row_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int), target :: h_csr_col_ind(5) = (/0, 2, 1, 0, 2/)
  real(c_float), target :: h_csr_val(5)     = (/1, 2, 3, 4, 5/)
  real(c_float), target :: h_x(3) = (/1, 2, 3/)
  real(c_float), target :: h_y(3)
  real(c_float) :: h_expected(3) = (/7, 6, 19/)   ! A*x
  real(c_float), target :: alpha = 1.0_c_float, beta = 0.0_c_float

  ! Device-resident copies of the dual-mode scalars; the library reads them
  ! from device memory because the handle is in device pointer mode.
  type(c_ptr) :: d_alpha = c_null_ptr
  type(c_ptr) :: d_beta = c_null_ptr

  type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val, d_x, d_y
  type(c_ptr) :: handle, matA, vecX, vecY, d_buffer
  integer(c_size_t) :: buffer_size

  real(c_float) :: error
  real(c_float), parameter :: error_max = 10 * epsilon(error_max)

  write(*,"(a)",advance="no") "-- Running test 'hipsparse_sspmv_devptr' (Fortran 2003 interfaces) - "

  call hipCheck(hipMalloc(d_csr_row_ptr, int(M+1,c_size_t) * 4))
  call hipCheck(hipMalloc(d_csr_col_ind, int(nnz,c_size_t) * 4))
  call hipCheck(hipMalloc(d_csr_val,     int(nnz,c_size_t) * 4))
  call hipCheck(hipMalloc(d_x,           int(N,c_size_t)   * 4))
  call hipCheck(hipMalloc(d_y,           int(M,c_size_t)   * 4))
  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), int(M+1,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind(1)), int(nnz,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_val,     c_loc(h_csr_val(1)),     int(nnz,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_x,           c_loc(h_x(1)),           int(N,c_size_t)   * 4, hipMemcpyHostToDevice))

  call hipsparseCheck(hipsparseCreate(handle))

  ! Switch to device pointer mode and stage the scalars in device memory
  call hipsparseCheck(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE))
  call hipCheck(hipMalloc(d_alpha, c_sizeof(alpha)))
  call hipCheck(hipMalloc(d_beta,  c_sizeof(beta)))
  call hipCheck(hipMemcpy(d_alpha, c_loc(alpha), c_sizeof(alpha), hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_beta,  c_loc(beta),  c_sizeof(beta),  hipMemcpyHostToDevice))

  call hipsparseCheck(hipsparseCreateCsr(matA, int(M,c_int64_t), int(N,c_int64_t), int(nnz,c_int64_t), &
       d_csr_row_ptr, d_csr_col_ind, d_csr_val, &
       HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F))
  call hipsparseCheck(hipsparseCreateDnVec(vecX, int(N,c_int64_t), d_x, HIP_R_32F))
  call hipsparseCheck(hipsparseCreateDnVec(vecY, int(M,c_int64_t), d_y, HIP_R_32F))

  call hipsparseCheck(hipsparseSpMV_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
       d_alpha, matA, vecX, d_beta, vecY, HIP_R_32F, HIPSPARSE_SPMV_ALG_DEFAULT, buffer_size))
  ! hipSPARSE requires a null buffer when the queried size is 0; a non-null
  ! (dummy) pointer makes SpMV return HIPSPARSE_STATUS_INVALID_VALUE.
  d_buffer = c_null_ptr
  if (buffer_size > 0) call hipCheck(hipMalloc(d_buffer, buffer_size))
  call hipsparseCheck(hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
       d_alpha, matA, vecX, d_beta, vecY, HIP_R_32F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(h_y(1)), d_y, int(M,c_size_t) * 4, hipMemcpyDeviceToHost))

  do i = 1, M
    error = abs(h_y(i) - h_expected(i)) / max(abs(h_expected(i)), 1.0_c_float)
    if(.not. (error .le. error_max)) then
        write(*,*) "FAILED! y(", i, ") = ", h_y(i), " expected ", h_expected(i); call exit(1)
    end if
  end do

  call hipsparseCheck(hipsparseDestroyDnVec(vecX))
  call hipsparseCheck(hipsparseDestroyDnVec(vecY))
  call hipsparseCheck(hipsparseDestroySpMat(matA))
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(d_csr_row_ptr)); call hipCheck(hipFree(d_csr_col_ind))
  call hipCheck(hipFree(d_csr_val)); call hipCheck(hipFree(d_x)); call hipCheck(hipFree(d_y))
  if (c_associated(d_buffer)) call hipCheck(hipFree(d_buffer))
  call hipCheck(hipFree(d_alpha)); call hipCheck(hipFree(d_beta))
  write(*,*) "PASSED!"
end program hipsparse_sspmv_devptr
