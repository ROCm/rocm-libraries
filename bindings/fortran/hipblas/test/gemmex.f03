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

program hip_gemmex
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  ! C := alpha*op(A)*op(B) + beta*C through the extended-precision entry point,
  ! with all buffers f32 and an f32 compute type. A is m x k and B is k x n with
  ! m, n, k distinct, so swapping A and B or the dimensions changes the result.
  integer, parameter :: m = 2, n = 3, k = 4
  integer(c_size_t) :: Nbytes
  real(c_float), target :: alpha = 1.0, beta = 0.0

  real(c_float), target :: hA(m,k) = reshape([1.0, -1.0, 2.0, 0.0, 3.0, 1.0, -2.0, 4.0], [m,k])
  real(c_float), target :: hB(k,n) = reshape([1.0, 2.0, 0.0, -1.0, 3.0, -2.0, 1.0, 0.0, 0.0, 1.0, 2.0, -3.0], [k,n])
  real(c_float), target :: hC(m,n)
  real(c_float) :: expected(m,n)
  type(c_ptr) :: dA = c_null_ptr, dB = c_null_ptr, dC = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j
  real(c_float) :: error
  real(c_float), parameter :: error_max = 10*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'gemmex' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  hC = 0.0
  expected = alpha*matmul(hA, hB)

  Nbytes = int(m*k, c_size_t) * 4
  call hipCheck(hipMalloc(dA, Nbytes))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), Nbytes, hipMemcpyHostToDevice))
  Nbytes = int(k*n, c_size_t) * 4
  call hipCheck(hipMalloc(dB, Nbytes))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), Nbytes, hipMemcpyHostToDevice))
  Nbytes = int(m*n, c_size_t) * 4
  call hipCheck(hipMalloc(dC, Nbytes))
  call hipCheck(hipMemcpy(dC, c_loc(hC(1,1)), Nbytes, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &
       c_loc(alpha), dA, HIP_R_32F, m, &
       dB, HIP_R_32F, k, c_loc(beta), &
       dC, HIP_R_32F, m, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hC(1,1)), dC, Nbytes, hipMemcpyDeviceToHost))

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

end program hip_gemmex
