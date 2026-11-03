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

program hip_sgeam
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  ! C := alpha*op(A) + beta*op(B) with alpha = 2, beta = -1 and no
  ! transposition, so C = 2*A - B (a swap of alpha and beta, or of A and B, shows).
  integer, parameter :: ld = 2
  real(c_float), target :: alpha = 2.0, beta = -1.0

  real(c_float) :: hA(ld,ld) = reshape([1.0, 2.0, 3.0, 4.0], [ld,ld])
  real(c_float) :: hB(ld,ld) = reshape([10.0, 20.0, 30.0, 40.0], [ld,ld])
  real(c_float) :: hC(ld,ld)
  real(c_float) :: expected(ld,ld) = reshape([-8.0, -16.0, -24.0, -32.0], [ld,ld])
  real(c_float), pointer, dimension(:,:) :: dA, dB, dC
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j
  real(c_float) :: error
  real(c_float), parameter :: error_max = 10*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'sgeam' (Fortran 2008 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  hC = 0.0

  call hipCheck(hipMalloc(dA, shape(hA)))
  call hipCheck(hipMemcpy(dA, hA, hipMemcpyHostToDevice))
  call hipCheck(hipMalloc(dB, shape(hB)))
  call hipCheck(hipMemcpy(dB, hB, hipMemcpyHostToDevice))
  call hipCheck(hipMalloc(dC, shape(hC)))
  call hipCheck(hipMemcpy(dC, hC, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasSgeam(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, ld, ld, &
       alpha, dA, size(dA,1), beta, dB, size(dB,1), dC, size(dC,1)))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(hC, dC, hipMemcpyDeviceToHost))

  do j = 1, ld
    do i = 1, ld
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

end program hip_sgeam
