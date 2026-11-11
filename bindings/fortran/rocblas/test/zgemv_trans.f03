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

program rocblas_zgemv_trans_test

  ! rocblas_zgemv with every trans value. A is rectangular and padded (lda > m), x and y
  ! use different non-unit increments and the values vary, so that a
  ! swapped dimension or increment, a wrong leading dimension, a wrong operation
  ! constant or a missed conjugation all change the result. The entries of y
  ! between the strided ones must come back untouched.

  use iso_c_binding
  use hip
  use rocblas

  implicit none

  integer(kind(rocblas_operation_none)), parameter :: ops(3) = [rocblas_operation_none, &
      rocblas_operation_transpose, rocblas_operation_conjugate_transpose]
  character(len=1), parameter :: op_names(3) = ['N', 'T', 'C']
  integer, parameter :: m = 37, n = 23, pad = 3, incx = 2, incy = 3
  integer, parameter :: lda = m + pad
  complex(c_double_complex), target :: alpha = (1.5_c_double, -0.5_c_double), beta = (0.5_c_double, 0.25_c_double)

  complex(c_double_complex), allocatable, target :: hA(:,:), hx(:), hy(:)
  complex(c_double_complex), allocatable :: hy0(:), opA(:,:), y_ref(:), mask_ref(:)
  type(c_ptr) :: dA = c_null_ptr, dx = c_null_ptr, dy = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr
  integer(c_size_t), parameter :: elem = 16

  integer :: it, i, j, lx, ly
  real(c_double) :: error, error_max

  write(*,"(a)",advance="no") "-- Running test 'ZGEMV transposes' (Fortran 2003 interfaces) - "

  call rocblasCheck(rocblas_create_handle(handle))
  call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host))

  error_max = 100 * epsilon(error) * (abs(alpha) * max(m, n) + abs(beta))

  allocate(hA(lda, n))
  do j = 1, n
    do i = 1, lda
      hA(i,j) = fill(i, j, 1)
    end do
  end do

  do it = 1, 3
    ! op(A) is ly x lx
    if (ops(it) == rocblas_operation_none) then
      lx = n; ly = m
    else
      lx = m; ly = n
    end if

    allocate(hx(1 + (lx-1)*incx), hy(1 + (ly-1)*incy))
    do i = 1, size(hx)
      hx(i) = fill(i, 1, 2)
    end do
    do i = 1, size(hy)
      hy(i) = fill(i, 1, 3)
    end do
    hy0 = hy

    call hipCheck(hipMalloc(dA, size(hA, kind=c_size_t) * elem))
    call hipCheck(hipMalloc(dx, size(hx, kind=c_size_t) * elem))
    call hipCheck(hipMalloc(dy, size(hy, kind=c_size_t) * elem))
    call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size(hA, kind=c_size_t) * elem, hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(dx, c_loc(hx(1)), size(hx, kind=c_size_t) * elem, hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(dy, c_loc(hy(1)), size(hy, kind=c_size_t) * elem, hipMemcpyHostToDevice))

    call rocblasCheck(rocblas_zgemv(handle, ops(it), m, n, c_loc(alpha), dA, lda, dx, incx, &
                                  c_loc(beta), dy, incy))
    call hipCheck(hipDeviceSynchronize())
    call hipCheck(hipMemcpy(c_loc(hy(1)), dy, size(hy, kind=c_size_t) * elem, hipMemcpyDeviceToHost))

    if (ops(it) == rocblas_operation_none) then
      opA = hA(1:m, 1:n)
    else
      opA = transpose(hA(1:m, 1:n))
      if (ops(it) == rocblas_operation_conjugate_transpose) opA = conjg(opA)
    end if
    y_ref = alpha * matmul(opA, hx(1::incx)) + beta * hy0(1::incy)

    error = maxval(abs(hy(1::incy) - y_ref))
    if (error > error_max) then
      write(*,*) "FAILED! trans = ", op_names(it), " error = ", error
      call exit(1)
    end if
    ! the strided writes must leave the entries in between alone
    mask_ref = hy0
    mask_ref(1::incy) = hy(1::incy)
    if (any(hy /= mask_ref)) then
      write(*,*) "FAILED! trans = ", op_names(it), " wrote between the strided entries of y"
      call exit(1)
    end if

    call hipCheck(hipFree(dA))
    call hipCheck(hipFree(dx))
    call hipCheck(hipFree(dy))
    deallocate(hx, hy, hy0, opA, y_ref, mask_ref)
  end do

  deallocate(hA)

  call rocblasCheck(rocblas_destroy_handle(handle))

  write(*,*) "PASSED!"

contains

  ! Varying small multiples of 1/8, so every product and sum below is exact.
  complex(c_double_complex) function fill(i, j, s)
    integer, intent(in) :: i, j, s
    fill = cmplx(real(mod(7*i + 3*j + s, 13) - 6, c_double) / 8, &
                 real(mod(5*i + 11*j + s, 17) - 8, c_double) / 8, c_double)
  end function fill

end program rocblas_zgemv_trans_test
