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

!!!!!!!!!!!!!!
! dsytrs example (solve A*X = B for a symmetric matrix, double)
! see: https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/
!
! Self-verifying: factorize a symmetric A with sytrf (Bunch-Kaufman), then solve
! A*x = b with sytrs for a right-hand side built from a known solution, and check
! that the recovered x matches. rocSOLVER writes info to DEVICE memory.
!
! sytrf has array overloads (exercised here via array pointers); sytrs is
! c_ptr-only, so the same device arrays are passed to it with c_loc.
!!!!!!!!!!!!!!
!
program dsytrs
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver
  implicit none
  integer :: i
  integer(c_int), parameter :: N = 4, nrhs = 1, lda = 4, ldb = 4
  real(c_double) :: hA(N,N) = reshape((/ &
      10.0d0,  2.0d0,  3.0d0,  6.0d0, &
       2.0d0, 11.0d0,  1.0d0,  0.0d0, &
       3.0d0,  1.0d0, 12.0d0,  2.0d0, &
       6.0d0,  0.0d0,  2.0d0, 13.0d0/), (/N,N/))
  real(c_double) :: hX(N) = (/1.0d0, 2.0d0, 3.0d0, 4.0d0/)   ! known solution
  real(c_double) :: hB(N)                                     ! RHS = A*x
  integer(c_int), target :: hInfo(1)
  integer(c_int) :: hIpiv(N)
  real(c_double), pointer :: dA(:,:), dB(:)
  integer(c_int), pointer :: dIpiv(:)
  type(c_ptr) :: dInfo
  type(c_ptr) :: handle
  real(c_double), parameter :: error_max = 1.0d-9
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dsytrs' (Fortran 2008 interfaces) - "

  hB = matmul(hA, hX)

  call hipCheck(hipMalloc(dA, source=hA))
  call hipCheck(hipMalloc(dB, source=hB))
  call hipCheck(hipMalloc(dIpiv, mold=hIpiv))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  call rocblasCheck(rocblas_create_handle(handle))
  ! sytrf: array overload (dA, dIpiv passed directly)
  call rocsolverCheck(rocsolver_dsytrf(handle, rocblas_fill_upper, N, dA, lda, dIpiv, dInfo))
  ! sytrs: c_ptr-only, pass the same device arrays via c_loc
  call rocsolverCheck(rocsolver_dsytrs(handle, rocblas_fill_upper, N, nrhs, &
       c_loc(dA), lda, c_loc(dIpiv), c_loc(dB), ldb))
  call hipCheck(hipMemcpy(c_loc(hInfo(1)), dInfo, 4_c_size_t, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(hB, dB, hipMemcpyDeviceToHost))

  if (hInfo(1) /= 0) then
     write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"; call exit(1)
  end if
  do i = 1, N
     if (.not. (abs(hB(i) - hX(i)) <= error_max)) then
        write(*,*) "FAILED! x(", i, ") = ", hB(i), " expected ", hX(i); call exit(1)
     end if
  end do

  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dB))
  call hipCheck(hipFree(dIpiv)); call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle)); call hipCheck(hipDeviceReset())
  write(*,*) "PASSED!"
end program dsytrs
