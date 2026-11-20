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


program rocblas_zhemm_test

  use iso_c_binding
  use hip
  use rocblas

  implicit none

  ! C := alpha*A*B + beta*C with A Hermitian, side left, upper triangle referenced.
  ! A is m x m and B is m x n with m /= n and B not the identity, so a side or
  ! m/n swap is caught. The strictly lower triangle of A holds junk that must not
  ! be read, so a fill swap is caught. A has complex off-diagonal entries, so a
  ! symm call in place of hemm is caught.
  ! The expected result is computed on the host from the upper triangle.
  integer, parameter :: m = 3, n = 2
  complex(c_double_complex), target :: alpha = (1.0, 0.0), beta = (0.0, 0.0)

  complex(c_double_complex), target :: hA(m,m)
  complex(c_double_complex), target :: hB(m,n) = reshape([(1.0, 1.0), (2.0, 0.0), (0.0, -1.0), &
      (-1.0, 0.0), (3.0, 2.0), (1.0, 1.0)], [m,n])
  complex(c_double_complex), target :: hC(m,n)
  complex(c_double_complex) :: expected(m,n), Afull(m,m)

  type(c_ptr) :: dA, dB, dC
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j
  real(c_double) :: error
  real(c_double), parameter :: error_max = 10 * epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'zhemm' (Fortran 2003 interfaces) - "

  call rocblasCheck(rocblas_create_handle(handle))
  call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host)) ! host pointer mode

  hC = (0.0, 0.0)

  ! Referenced upper triangle (real diagonal); the strictly lower triangle is junk.
  hA = (99.0, 99.0)
  hA(1,1) = (1.0, 0.0)
  hA(1,2) = (2.0, 1.0)
  hA(2,2) = (4.0, 0.0)
  hA(1,3) = (3.0, -1.0)
  hA(2,3) = (5.0, 2.0)
  hA(3,3) = (6.0, 0.0)

  ! Full Hermitian A rebuilt from the upper triangle, for the host reference
  do j = 1, m
    do i = 1, m
      if (i .le. j) then
        Afull(i,j) = hA(i,j)
      else
        Afull(i,j) = conjg(hA(j,i))
      end if
    end do
  end do
  expected = alpha * matmul(Afull, hB) + beta * hC

  call hipCheck(hipMalloc(dA, int(m*m,c_size_t) * 16))
  call hipCheck(hipMalloc(dB, int(m*n,c_size_t) * 16))
  call hipCheck(hipMalloc(dC, int(m*n,c_size_t) * 16))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), int(m*m,c_size_t) * 16, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), int(m*n,c_size_t) * 16, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dC, c_loc(hC(1,1)), int(m*n,c_size_t) * 16, hipMemcpyHostToDevice))

  call rocblasCheck(rocblas_zhemm(handle, rocblas_side_left, rocblas_fill_upper, m, n, &
       c_loc(alpha), dA, m, dB, m, c_loc(beta), dC, m))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hC(1,1)), dC, int(m*n,c_size_t) * 16, hipMemcpyDeviceToHost))

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

end program rocblas_zhemm_test
