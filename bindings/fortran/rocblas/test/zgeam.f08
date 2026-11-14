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


program rocblas_zgeam_test

  use iso_c_binding
  use hip
  use rocblas

  implicit none

  ! C := alpha*op(A) + beta*op(B), both non-transposed, alpha = 2, beta = -1,
  ! so C = 2*A - B; distinct alpha and beta catch an alpha/beta or A/B swap.
  integer, parameter :: m = 2, n = 2
  complex(c_double_complex), target :: alpha = (2.0, 0.0), beta = (-1.0, 0.0)

  complex(c_double_complex) :: hA(m,n) = reshape([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)], [m,n])
  complex(c_double_complex) :: hB(m,n) = reshape([(10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (40.0, 0.0)], [m,n])
  complex(c_double_complex) :: hC(m,n)
  complex(c_double_complex) :: expected(m,n) = reshape([(-8.0, 0.0), (-16.0, 0.0), (-24.0, 0.0), (-32.0, 0.0)], [m,n])

  complex(c_double_complex), pointer, dimension(:,:) :: dA, dB, dC
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j
  real(c_double) :: error
  real(c_double), parameter :: error_max = 10 * epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'zgeam' (Fortran 2008 interfaces) - "

  call rocblasCheck(rocblas_create_handle(handle))
  call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host)) ! host pointer mode

  hC = (0.0, 0.0)

  call hipCheck(hipMalloc(dA, source=hA))
  call hipCheck(hipMalloc(dB, source=hB))
  call hipCheck(hipMalloc(dC, source=hC))

  call rocblasCheck(rocblas_zgeam(handle, rocblas_operation_none, rocblas_operation_none, m, n, &
       alpha, dA, size(dA,1), beta, dB, size(dB,1), dC, size(dC,1)))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(hC, dC, hipMemcpyDeviceToHost))

  do j = 1, n
    do i = 1, m
      error = abs(expected(i,j) - hC(i,j))
      if (error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " at ", i, j
        call exit(1)
      end if
    end do
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dB))
  call hipCheck(hipFree(dC))
  call rocblasCheck(rocblas_destroy_handle(handle))

  write(*,*) "PASSED!"

end program rocblas_zgeam_test
