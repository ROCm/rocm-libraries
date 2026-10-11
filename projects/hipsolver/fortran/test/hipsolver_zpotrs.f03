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
! hipsolverZpotrs example (double-complex Cholesky solve, Fortran 2003
! interfaces)
! see: https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Self-verifying: pick a known solution x, form b = A*x for a Hermitian positive
! definite A, factorize with potrf, solve A*X = b with potrs, and confirm X
! recovers x.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. devInfo is a type(c_ptr) passed by
! value, so its device allocation is handed over directly.
!!!!!!!!!!!!!!/
!
program zpotrs
  use iso_c_binding
  use hip
  use hipsolver

  implicit none
  integer :: i ! index for iterating over results

  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: nrhs = 1
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: ldb = 3

  ! Hermitian positive definite input (column-major)
  complex(c_double_complex), target :: hA(3,3) = reshape((/ &
    (4.0d0,0.0d0), (1.0d0,1.0d0), (0.0d0,0.0d0), &
    (1.0d0,-1.0d0), (5.0d0,0.0d0), (0.0d0,-1.0d0), &
    (0.0d0,0.0d0), (0.0d0,1.0d0), (6.0d0,0.0d0)/), (/3, 3/))
  complex(c_double_complex), target :: hX(3,1) = reshape((/ &
    (1.0d0,1.0d0), (2.0d0,-1.0d0), (3.0d0,2.0d0)/), (/3, 1/))  ! known solution
  complex(c_double_complex), target :: hB(3,1)                 ! RHS = A*x

  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_B = size(hB)

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: dA, dB, dInfo, dWork
  integer(c_int) :: lwork_f, lwork_s, lwork

  real(c_double) :: error
  real(c_double), parameter :: error_max = 100 * epsilon(error_max)
  !
  write(*,"(a)",advance="no") "-- Running test 'hipsolverZpotrs' (Fortran 2003 interfaces) - "

  ! Build a consistent RHS so that A*x = b
  hB = matmul(hA, hX)

  call hipsolverCheck(hipsolverCreate(handle))

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, size_A * 16))
  call hipCheck(hipMalloc(dB, size_B * 16))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 16, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), size_B * 16, hipMemcpyHostToDevice))

  ! Workspace big enough for both potrf and potrs
  call hipsolverCheck(hipsolverZpotrf_bufferSize(handle, HIPSOLVER_FILL_MODE_LOWER, N, dA, lda, lwork_f))
  call hipsolverCheck(hipsolverZpotrs_bufferSize(handle, HIPSOLVER_FILL_MODE_LOWER, N, nrhs, dA, lda, dB, ldb, lwork_s))
  lwork = max(lwork_f, lwork_s)
  call hipCheck(hipMalloc(dWork, int(lwork,c_size_t) * 16))

  ! Factorize A = L*L**H, then solve A*X = B in place
  call hipsolverCheck(hipsolverZpotrf(handle, HIPSOLVER_FILL_MODE_LOWER, N, dA, lda, dWork, lwork, dInfo))
  call hipsolverCheck(hipsolverZpotrs(handle, HIPSOLVER_FILL_MODE_LOWER, N, nrhs, dA, lda, dB, ldb, dWork, lwork, dInfo))

  ! Copy result from device to host
  call hipCheck(hipMemcpy(c_loc(hB(1,1)), dB, size_B * 16, hipMemcpyDeviceToHost))

  ! Verify the recovered solution matches x
  do i = 1,N
    error = abs(hB(i,1) - hX(i,1)) / max(abs(hX(i,1)), 1.0_c_double)
    if(.not. (error .le. error_max)) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " X(", i, ") = ", hB(i,1)
        call exit(1)
    end if
  end do

  ! Clean up
  call hipCheck(hipFree(dWork))
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dB))
  call hipCheck(hipFree(dInfo))
  call hipsolverCheck(hipsolverDestroy(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program zpotrs
