!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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


program rocblas_cherk_test

  use iso_c_binding
  use hip
  use rocblas

  implicit none

  ! C := alpha*A*A**H + beta*C, upper triangle, with real alpha = 1, beta = 0.
  ! A has non-zero imaginary parts, so A*A**H differs from A*A**T and a binding
  ! that called the symmetric 'syrk' form instead would be caught. The expected
  ! upper triangle is computed on the host.
  integer, parameter :: n = 2, k = 2
  real(c_float), target :: alpha = 1.0, beta = 0.0

  complex(c_float_complex) :: hA(n,k) = reshape([(1.0, 1.0), (2.0, -1.0), (0.0, 2.0), (3.0, 1.0)], [n,k])
  complex(c_float_complex) :: hC(n,n)
  complex(c_float_complex) :: expected(n,n)

  complex(c_float_complex), pointer, dimension(:,:) :: dA, dC
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j
  real(c_float) :: error
  real(c_float), parameter :: error_max = 10 * epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'cherk' (Fortran 2008 interfaces) - "

  call rocblasCheck(rocblas_create_handle(handle))
  call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host)) ! host pointer mode

  hC = (0.0, 0.0)
  expected = alpha * matmul(hA, conjg(transpose(hA))) + beta * hC

  call hipCheck(hipMalloc(dA, source=hA))
  call hipCheck(hipMalloc(dC, source=hC))

  call rocblasCheck(rocblas_cherk(handle, rocblas_fill_upper, rocblas_operation_none, &
       n, k, alpha, dA, size(dA,1), beta, dC, size(dC,1)))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(hC, dC, hipMemcpyDeviceToHost))

  ! only the upper triangle is referenced/written
  do j = 1, n
    do i = 1, j
      error = abs(expected(i,j) - hC(i,j))
      if (error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " at ", i, j
        call exit(1)
      end if
    end do
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dC))
  call rocblasCheck(rocblas_destroy_handle(handle))

  write(*,*) "PASSED!"

end program rocblas_cherk_test
