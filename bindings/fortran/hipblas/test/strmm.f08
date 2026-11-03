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

program hip_strmm
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  ! C := alpha*op(A)*B with A upper triangular on the left (out-of-place trmm).
  ! B is m x n with m /= n and A(2,1) holds junk, so swapping A and B, the side
  ! or the triangle changes the result.
  integer, parameter :: m = 2, n = 3
  real(c_float), target :: alpha = 1.0

  real(c_float) :: hA(m,m) = reshape([1.0, 7.0, 2.0, 3.0], [m,m])
  real(c_float) :: hB(m,n) = reshape([1.0, 4.0, 2.0, 5.0, 3.0, -6.0], [m,n])
  real(c_float) :: hC(m,n)
  real(c_float) :: expected(m,n), Aref(m,m)
  real(c_float), pointer, dimension(:,:) :: dA, dB, dC
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j
  real(c_float) :: error
  real(c_float), parameter :: error_max = 10*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'strmm' (Fortran 2008 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  hC = 0.0

  ! Host reference built from the upper triangle only, A(2,1) is junk
  Aref = 0.0
  do j = 1, m
    do i = 1, j
      Aref(i,j) = hA(i,j)
    end do
  end do
  expected = alpha*matmul(Aref, hB)

  call hipCheck(hipMalloc(dA, shape(hA)))
  call hipCheck(hipMemcpy(dA, hA, hipMemcpyHostToDevice))
  call hipCheck(hipMalloc(dB, shape(hB)))
  call hipCheck(hipMemcpy(dB, hB, hipMemcpyHostToDevice))
  call hipCheck(hipMalloc(dC, shape(hC)))
  call hipCheck(hipMemcpy(dC, hC, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasStrmm(handle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, &
       HIPBLAS_DIAG_NON_UNIT, m, n, alpha, dA, size(dA,1), dB, size(dB,1), dC, size(dC,1)))
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

end program hip_strmm
