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

program hip_csymm
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  ! C := alpha*A*B + beta*C with A symmetric (upper triangle referenced) on the left.
  ! B is m x n with m /= n and A(2,1) holds junk, so swapping A and B, the side
  ! or the triangle changes the result.
  integer, parameter :: m = 2, n = 3
  complex(c_float_complex), target :: alpha = (1.0, 0.0), beta = (0.0, 0.0)

  complex(c_float_complex) :: hA(m,m) = reshape([(1.0, 1.0), (7.0, -7.0), (2.0, -1.0), (3.0, 0.5)], [m,m])
  complex(c_float_complex) :: hB(m,n) = reshape([(1.0, 0.0), (4.0, 1.0), (2.0, -2.0), &
       (5.0, 0.0), (3.0, 1.0), (-6.0, 0.0)], [m,n])
  complex(c_float_complex) :: hC(m,n)
  complex(c_float_complex) :: expected(m,n), Aref(m,m)
  complex(c_float_complex), pointer, dimension(:,:) :: dA, dB, dC
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j
  real(c_float) :: error
  real(c_float), parameter :: error_max = 10*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'csymm' (Fortran 2008 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  hC = (0.0, 0.0)

  ! Host reference built from the upper triangle only, A(2,1) is junk
  do j = 1, m
    do i = 1, m
      Aref(i,j) = hA(min(i,j), max(i,j))
    end do
  end do
  expected = alpha*matmul(Aref, hB)

  call hipCheck(hipMalloc(dA, shape(hA)))
  call hipCheck(hipMemcpy(dA, hA, hipMemcpyHostToDevice))
  call hipCheck(hipMalloc(dB, shape(hB)))
  call hipCheck(hipMemcpy(dB, hB, hipMemcpyHostToDevice))
  call hipCheck(hipMalloc(dC, shape(hC)))
  call hipCheck(hipMemcpy(dC, hC, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasCsymm(handle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_UPPER, m, n, &
       alpha, dA, size(dA,1), dB, size(dB,1), beta, dC, size(dC,1)))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(hC, dC, hipMemcpyDeviceToHost))

  do j = 1, n
    do i = 1, m
      error = abs(expected(i,j) - hC(i,j))
      if (error > error_max) then
        write(*,*) "FAILED! error = ", error, " at ", i, j
        call exit(1)
      end if
    end do
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dB))
  call hipCheck(hipFree(dC))
  call hipblasCheck(hipblasDestroy(handle))

  write(*,*) "PASSED!"

end program hip_csymm
