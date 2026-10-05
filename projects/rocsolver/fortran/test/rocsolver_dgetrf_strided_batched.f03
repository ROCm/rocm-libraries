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

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! rocsolver_dgetrf_strided_batched example / unit test
!
! Computes the LU factorization of a batch of matrices on the GPU and checks,
! for every batch entry, info, the pivots against a hand-computed result, and
! that P*A = L*U. info is an array of batch_count integers in device memory,
! passed as type(c_ptr).
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
program dgetrf_strided_batched
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none

  integer(c_int), parameter :: M = 3, N = 3, lda = 3
  integer(c_int), parameter :: batch_count = 2
  integer(c_int), parameter :: sz_piv = min(M, N)
  integer(c_int64_t), parameter :: strideA = int(lda, c_int64_t) * int(N, c_int64_t)
  integer(c_int64_t), parameter :: strideP = int(sz_piv, c_int64_t)

  ! Two invertible 3x3 matrices stored back-to-back (column-major).
  real(c_double), target :: hA(lda, N*batch_count)
  real(c_double) :: hA0(lda, N*batch_count), PA(M,N), L(M,N), U(N,N), tmp(N)
  integer(c_int), target :: hInfo(batch_count)
  integer(c_int), target :: hIpiv(sz_piv*batch_count)
  ! Partial pivoting, worked by hand: |-51| leads column 1 of the first matrix,
  ! then row 3 leads column 2; the second matrix is diagonal.
  integer(c_int), parameter :: hIpiv_ref(sz_piv*batch_count) = (/ 2, 3, 3, 1, 2, 3 /)

  integer(c_size_t) :: size_A    = lda*N*batch_count
  integer(c_size_t) :: size_Ipiv = sz_piv*batch_count
  integer(c_size_t) :: size_Info = batch_count

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: dA     ! GPU buffer for the strided matrices
  type(c_ptr) :: dIpiv  ! GPU buffer for the pivot indices
  type(c_ptr) :: dInfo  ! GPU buffer for the info array

  integer :: b, i, j, c0
  real(c_double) :: error
  real(c_double), parameter :: rtol = 1.0d-12

  write(*,"(a)",advance="no") &
    "-- Running test 'rocsolver_dgetrf_strided_batched' (Fortran 2003 interfaces) - "

  hA(:, 1:N)       = reshape((/ 12.d0, -51.d0,   4.d0, &
                                 6.d0, 167.d0, -68.d0, &
                                -4.d0,  24.d0, -41.d0/), (/lda, N/))
  hA(:, N+1:2*N)   = reshape((/  2.d0,   0.d0,   0.d0, &
                                 0.d0,   3.d0,   0.d0, &
                                 0.d0,   0.d0,   5.d0/), (/lda, N/))

  hA0 = hA
  hInfo = -1

  ! Create rocBLAS handle
  call rocblasCheck(rocblas_create_handle(handle))

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA,    size_A    * 8))  ! c_double
  call hipCheck(hipMalloc(dIpiv, size_Ipiv * 4))  ! c_int
  call hipCheck(hipMalloc(dInfo, size_Info * 4))  ! c_int

  ! Copy the input matrices to the device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dInfo, c_loc(hInfo(1)), size_Info * 4, hipMemcpyHostToDevice))

  ! Compute the batched LU factorization on the device.
  call rocsolverCheck(rocsolver_dgetrf_strided_batched(handle, M, N, dA, lda, strideA, &
                                                 dIpiv, strideP, dInfo, batch_count))

  ! Copy the results back to the host
  call hipCheck(hipMemcpy(c_loc(hInfo(1)), dInfo, size_Info * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, size_A * 8, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hIpiv(1)), dIpiv, size_Ipiv * 4, hipMemcpyDeviceToHost))

  ! For invertible inputs rocSOLVER sets info(b) == 0 for every batch entry.
  do b = 1, batch_count
    if (hInfo(b) /= 0) then
      write(*,*) "FAILED! info(", b, ") = ", hInfo(b), " (expected 0)"
      call exit(1)
    end if
  end do

  do i = 1, sz_piv*batch_count
    if (hIpiv(i) /= hIpiv_ref(i)) then
      write(*,*) "FAILED! ipiv(", i, ") = ", hIpiv(i), " expected ", hIpiv_ref(i)
      call exit(1)
    end if
  end do

  ! Reconstruct each matrix: apply the row interchanges to A, compare with L*U.
  do b = 1, batch_count
    c0 = (b-1)*N
    PA = hA0(1:M, c0+1:c0+N)
    do i = 1, sz_piv
      j = hIpiv((b-1)*sz_piv + i)
      tmp = PA(i,:); PA(i,:) = PA(j,:); PA(j,:) = tmp
    end do
    L = 0.0d0; U = 0.0d0
    do j = 1, N
      do i = 1, M
        if (i > j) then
          L(i,j) = hA(i, c0+j)
        else
          U(i,j) = hA(i, c0+j)
          if (i == j) L(i,j) = 1.0d0
        end if
      end do
    end do
    ! Frobenius norms rather than maxval, which can skip a NaN.
    error = sqrt(sum((PA - matmul(L, U))**2)) / sqrt(sum(hA0(1:M, c0+1:c0+N)**2))
    if (.not. (error <= rtol)) then
      write(*,*) "FAILED! batch ", b, ": ||P*A - L*U||_F / ||A||_F = ", error
      call exit(1)
    end if
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dIpiv))
  call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dgetrf_strided_batched
