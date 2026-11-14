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
! dsteqr example (eigenvalues of a symmetric tridiagonal matrix)
! The sum of the eigenvalues equals the trace (sum of the input diagonal D), and the sum of their
! squares equals the squared Frobenius norm of the matrix, sum(D**2) + 2*sum(E**2).
! rocSOLVER writes info to DEVICE memory (passed as c_loc(dInfo)).
!!!!!!!!!!!!!!
!
program dsteqr
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver
  implicit none
  integer(c_int), parameter :: N = 4, ldc = 4
  real(c_double) :: hD(N)   = (/2.0d0, 2.0d0, 2.0d0, 2.0d0/)   ! diagonal
  real(c_double) :: hE(N-1) = (/1.0d0, 1.0d0, 1.0d0/)          ! off-diagonal
  real(c_double), pointer :: dD(:), dE(:), dC(:,:)
  integer(c_int), pointer :: dInfo(:)
  integer(c_int) :: hInfo(1)
  type(c_ptr) :: handle
  real(c_double) :: trace, frob2, error
  real(c_double), parameter :: rtol = 1.0d-9
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dsteqr' (Fortran 2008 interfaces) - "
  trace = sum(hD)   ! = 8
  frob2 = sum(hD**2) + 2.0d0*sum(hE**2)   ! ||T||_F**2, from the input arrays
  call hipCheck(hipMalloc(dD, source=hD))
  call hipCheck(hipMalloc(dE, source=hE))
  call hipCheck(hipMalloc(dC, int(N,c_size_t), int(N,c_size_t)))
  call hipCheck(hipMalloc(dInfo, 1))
  call rocblasCheck(rocblas_create_handle(handle))
  call rocsolverCheck(rocsolver_dsteqr(handle, rocblas_evect_none, N, dD, dE, dC, ldc, c_loc(dInfo)))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  if (hInfo(1) /= 0) then
    write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"; call exit(1)
  end if
  call hipCheck(hipMemcpy(hD, dD, hipMemcpyDeviceToHost))
  error = abs(sum(hD) - trace) / abs(trace)
  if (.not. (error <= rtol)) then
     write(*,*) "FAILED! sum(eigenvalues) = ", sum(hD), " expected trace = ", trace
     call exit(1)
  end if
  error = abs(sum(hD**2) - frob2) / abs(frob2)
  if (.not. (error <= rtol)) then
     write(*,*) "FAILED! sum(eigenvalues**2) = ", sum(hD**2), " expected ||T||_F**2 = ", frob2
     call exit(1)
  end if
  call hipCheck(hipFree(dD)); call hipCheck(hipFree(dE)); call hipCheck(hipFree(dC)); call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle)); call hipCheck(hipDeviceReset())
  write(*,*) "PASSED!"
end program dsteqr
