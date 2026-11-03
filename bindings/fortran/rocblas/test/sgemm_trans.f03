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

program rocblas_sgemm_trans_test

  ! rocblas_sgemm with every transa/transb combination. The matrices are rectangular,
  ! padded (ld > rows) and filled with varying values, so that a swapped dimension,
  ! a wrong leading dimension, a wrong operation constant or a missed conjugation
  ! all change the result. The padding rows of C must come back untouched.

  use iso_c_binding
  use hip
  use rocblas

  implicit none

  integer(kind(rocblas_operation_none)), parameter :: ops(3) = [rocblas_operation_none, &
      rocblas_operation_transpose, rocblas_operation_conjugate_transpose]
  character(len=1), parameter :: op_names(3) = ['N', 'T', 'C']
  integer, parameter :: m = 37, n = 23, k = 19, pad = 3
  real(c_float), target :: alpha = 1.5_c_float, beta = 0.5_c_float

  real(c_float), allocatable, target :: hA(:,:), hB(:,:), hC(:,:)
  real(c_float), allocatable :: hC0(:,:), opA(:,:), opB(:,:), hC_ref(:,:)
  type(c_ptr) :: dA = c_null_ptr, dB = c_null_ptr, dC = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr
  integer(c_size_t), parameter :: elem = 4

  integer :: ia, ib, i, j, rows_a, cols_a, rows_b, cols_b, lda, ldb, ldc
  real(c_float) :: error, error_max

  write(*,"(a)",advance="no") "-- Running test 'SGEMM transposes' (Fortran 2003 interfaces) - "

  call rocblasCheck(rocblas_create_handle(handle))
  call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host))

  error_max = 100 * epsilon(error) * (abs(alpha) * k + abs(beta))

  do ia = 1, 3
    do ib = 1, 3
      ! op(A) is m x k and op(B) is k x n
      if (ops(ia) == rocblas_operation_none) then
        rows_a = m; cols_a = k
      else
        rows_a = k; cols_a = m
      end if
      if (ops(ib) == rocblas_operation_none) then
        rows_b = k; cols_b = n
      else
        rows_b = n; cols_b = k
      end if
      lda = rows_a + pad; ldb = rows_b + pad; ldc = m + pad

      allocate(hA(lda, cols_a), hB(ldb, cols_b), hC(ldc, n))
      do j = 1, cols_a
        do i = 1, lda
          hA(i,j) = fill(i, j, 1)
        end do
      end do
      do j = 1, cols_b
        do i = 1, ldb
          hB(i,j) = fill(i, j, 2)
        end do
      end do
      do j = 1, n
        do i = 1, ldc
          hC(i,j) = fill(i, j, 3)
        end do
      end do
      hC0 = hC

      call hipCheck(hipMalloc(dA, size(hA, kind=c_size_t) * elem))
      call hipCheck(hipMalloc(dB, size(hB, kind=c_size_t) * elem))
      call hipCheck(hipMalloc(dC, size(hC, kind=c_size_t) * elem))
      call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size(hA, kind=c_size_t) * elem, hipMemcpyHostToDevice))
      call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), size(hB, kind=c_size_t) * elem, hipMemcpyHostToDevice))
      call hipCheck(hipMemcpy(dC, c_loc(hC(1,1)), size(hC, kind=c_size_t) * elem, hipMemcpyHostToDevice))

      call rocblasCheck(rocblas_sgemm(handle, ops(ia), ops(ib), m, n, k, c_loc(alpha), dA, lda, dB, ldb, &
                                    c_loc(beta), dC, ldc))
      call hipCheck(hipDeviceSynchronize())
      call hipCheck(hipMemcpy(c_loc(hC(1,1)), dC, size(hC, kind=c_size_t) * elem, hipMemcpyDeviceToHost))

      if (ops(ia) == rocblas_operation_none) then
        opA = hA(1:m, 1:k)
      else
        opA = transpose(hA(1:k, 1:m))
      end if
      if (ops(ib) == rocblas_operation_none) then
        opB = hB(1:k, 1:n)
      else
        opB = transpose(hB(1:n, 1:k))
      end if
      hC_ref = alpha * matmul(opA, opB) + beta * hC0(1:m, 1:n)

      error = maxval(abs(hC(1:m, 1:n) - hC_ref))
      if (error > error_max) then
        write(*,*) "FAILED! transa = ", op_names(ia), " transb = ", op_names(ib), " error = ", error
        call exit(1)
      end if
      if (any(hC(m+1:ldc, :) /= hC0(m+1:ldc, :))) then
        write(*,*) "FAILED! transa = ", op_names(ia), " transb = ", op_names(ib), " wrote past m in C"
        call exit(1)
      end if

      call hipCheck(hipFree(dA))
      call hipCheck(hipFree(dB))
      call hipCheck(hipFree(dC))
      deallocate(hA, hB, hC, hC0, opA, opB, hC_ref)
    end do
  end do

  call rocblasCheck(rocblas_destroy_handle(handle))

  write(*,*) "PASSED!"

contains

  ! Varying small multiples of 1/8, so every product and sum below is exact.
  real(c_float) function fill(i, j, s)
    integer, intent(in) :: i, j, s
    fill = real(mod(7*i + 3*j + s, 13) - 6, c_float) / 8
  end function fill

end program rocblas_sgemm_trans_test
