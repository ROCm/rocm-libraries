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

!!!!!!!!!!!!!/
! zgetrs example (double-precision complex LU solve)
! see: https://www.netlib.org/lapack/explore-html/d6/d49/group__double_g_ecomputational_gafa35ce1d7865b80563bbed6317050ad7.html
!
! Self-verifying: pick a known solution x, form b = A*x, factorize A with
! getrf, solve A*X = b with getrs, and confirm X recovers x.
!
! NOTE: getrf writes `info` to DEVICE memory, so info is backed by a device
! allocation and passed as c_loc(dInfo); passing a host scalar faults on the GPU.
!!!!!!!!!!!!!!/
!
program zgetrs
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i ! index for iterating over results

  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: ldb = 3
  integer(c_int), parameter :: nrhs = 1

  ! Nonsingular input matrix (column-major)
  complex(c_double_complex) :: hA(3,3) = reshape((/ &
    (12.,1.),(6.,-2.),(-4.,3.), (-51.,2.),(167.,-1.),(24.,4.), (4.,-3.),(-68.,2.),(-41.,1.) /), (/3, 3/))
  complex(c_double_complex) :: hX(3,1) = reshape((/ (1.,1.), (2.,-1.), (3.,2.) /), (/3, 1/))  ! known solution
  complex(c_double_complex) :: hB(3,1)                                                          ! RHS = A*x

  integer(c_int) :: hIpiv(3)

  complex(c_double_complex), pointer :: dA(:,:)
  complex(c_double_complex), pointer :: dB(:,:)
  integer(c_int), pointer :: dIpiv(:)
  integer(c_int), pointer :: dInfo(:)
  integer(c_int) :: hInfo(1)

  type(c_ptr) :: handle ! rocblas_handle

  real(c_double) :: error
  real(c_double), parameter :: error_max = 100 * epsilon(error_max)
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_zgetrs' (Fortran 2008 interfaces) - "

  ! Build a consistent RHS so that A*x = b
  hB = matmul(hA, hX)

  ! Allocate device-side memory & copy memory from host to device
  call hipCheck(hipMalloc(dA,    source=hA))
  call hipCheck(hipMalloc(dB,    source=hB))
  call hipCheck(hipMalloc(dIpiv, mold=hIpiv))
  call hipCheck(hipMalloc(dInfo, 1))

  ! Create rocBLAS handle
  call rocblasCheck(rocblas_create_handle(handle))

  ! Factorize, then solve A*X = B in place (B is overwritten with the solution)
  call rocsolverCheck(rocsolver_zgetrf(handle, N, N, dA, lda, dIpiv, c_loc(dInfo)))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  if (hInfo(1) /= 0) then
    write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"; call exit(1)
  end if
  call rocsolverCheck(rocsolver_zgetrs(handle, rocblas_operation_none, N, nrhs, dA, lda, dIpiv, dB, ldb))

  ! Copy result from device to host
  call hipCheck(hipMemcpy(hB, dB, hipMemcpyDeviceToHost))

  ! Verify the recovered solution matches x
  do i = 1,N
    error = abs(hB(i,1) - hX(i,1)) / max(abs(hX(i,1)), 1.0_c_double)
    if(.not. (error .le. error_max)) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " X(", i, ") = ", hB(i,1)
        call exit(1)
    end if
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dB))
  call hipCheck(hipFree(dIpiv))
  call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program zgetrs
