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


program rocblas_gemm_ex_test

  use iso_c_binding
  use hip
  use rocblas

  implicit none

  ! D := alpha*op(A)*op(B) + beta*C through the extended-precision entry point.
  ! All buffers are f32_r and the compute type is f32_r. A and B are not the
  ! identity and m, n, k all differ, so an A/B or dimension swap is caught; every
  ! buffer is sized from its own dimensions. The expected result is computed on
  ! the host.
  integer, parameter :: m = 3, n = 2, k = 4
  real(c_float), target :: alpha = 1.0, beta = 0.0

  real(c_float), target :: hA(m,k) = reshape([1.0, -2.0, 3.0, 0.5, 1.0, -1.0, &
      2.0, 0.0, 4.0, -3.0, 1.0, 2.0], [m,k])
  real(c_float), target :: hB(k,n) = reshape([2.0, 1.0, 0.0, -1.0, 1.0, 3.0, -2.0, 0.5], [k,n])
  real(c_float), target :: hC(m,n)
  real(c_float), target :: hD(m,n)
  real(c_float) :: expected(m,n)

  type(c_ptr) :: dA, dB, dC, dD
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j
  real(c_float) :: error
  real(c_float), parameter :: error_max = 10 * epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'gemm_ex' (Fortran 2003 interfaces) - "

  call rocblasCheck(rocblas_create_handle(handle))
  call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host)) ! host pointer mode

  hC = 0.0
  hD = 0.0
  expected = alpha * matmul(hA, hB) + beta * hC

  call hipCheck(hipMalloc(dA, int(m*k,c_size_t) * 4))
  call hipCheck(hipMalloc(dB, int(k*n,c_size_t) * 4))
  call hipCheck(hipMalloc(dC, int(m*n,c_size_t) * 4))
  call hipCheck(hipMalloc(dD, int(m*n,c_size_t) * 4))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), int(m*k,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), int(k*n,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dC, c_loc(hC(1,1)), int(m*n,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dD, c_loc(hD(1,1)), int(m*n,c_size_t) * 4, hipMemcpyHostToDevice))

  call rocblasCheck(rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none, &
       m, n, k, c_loc(alpha), dA, rocblas_datatype_f32_r, m, &
       dB, rocblas_datatype_f32_r, k, c_loc(beta), &
       dC, rocblas_datatype_f32_r, m, &
       dD, rocblas_datatype_f32_r, m, &
       rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hD(1,1)), dD, int(m*n,c_size_t) * 4, hipMemcpyDeviceToHost))

  do j = 1, n
    do i = 1, m
      error = abs(expected(i,j) - hD(i,j))
      if (error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " at ", i, j
        call exit(1)
      end if
    end do
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dB))
  call hipCheck(hipFree(dC))
  call hipCheck(hipFree(dD))
  call rocblasCheck(rocblas_destroy_handle(handle))

  write(*,*) "PASSED!"

end program rocblas_gemm_ex_test
