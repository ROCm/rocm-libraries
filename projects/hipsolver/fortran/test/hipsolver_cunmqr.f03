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
! hipsolverCunmqr example (single-complex multiply by Q from a QR
! factorization, Fortran 2003 interfaces)
! see: https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Self-verifying: factorize A with geqrf to obtain Q (as Householder vectors),
! then form Q*C with unmqr. Q is unitary, so it preserves the Frobenius norm:
! ||Q*C||_F = ||C||_F. Norm preservation is phase-convention independent, so no
! reference matrix is needed.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. tau and devInfo are type(c_ptr) passed
! by value, so their device allocations are handed over directly.
!!!!!!!!!!!!!!/
!
program cunmqr
  use iso_c_binding
  use hip
  use hipsolver

  implicit none
  integer :: i, j

  integer(c_int), parameter :: M = 3
  integer(c_int), parameter :: N = 2
  integer(c_int), parameter :: K = 3
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: ldc = 3

  complex(c_float_complex), target :: hA(3,3) = reshape((/ &
    (1.0,1.0), (4.0,0.0), (7.0,0.0), &
    (2.0,0.0), (5.0,1.0), (8.0,0.0), &
    (3.0,0.0), (6.0,0.0), (10.0,1.0)/), (/3, 3/))
  complex(c_float_complex), target :: hC(3,2) = reshape((/ &
    (1.0,1.0), (2.0,0.0), (3.0,-1.0), (4.0,0.0), (5.0,2.0), (6.0,0.0)/), (/3, 2/))
  real(c_float) :: norm_in, norm_out
  complex(c_float_complex) :: hC0(3,2)

  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_C = size(hC)
  integer(c_size_t) :: size_tau = 3

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: dA, dC, dTau, dInfo, dWork
  integer(c_int) :: lwork_qr, lwork_mq, lwork

  real(c_float) :: error
  real(c_float), parameter :: error_max = 1.0e-5
  !
  write(*,"(a)",advance="no") "-- Running test 'hipsolverCunmqr' (Fortran 2003 interfaces) - "

  norm_in = sqrt(sum(abs(hC)**2))
  hC0 = hC

  call hipsolverCheck(hipsolverCreate(handle))

  call hipCheck(hipMalloc(dA, size_A * 8))
  call hipCheck(hipMalloc(dC, size_C * 8))
  call hipCheck(hipMalloc(dTau, size_tau * 8))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dC, c_loc(hC(1,1)), size_C * 8, hipMemcpyHostToDevice))

  call hipsolverCheck(hipsolverCgeqrf_bufferSize(handle, M, K, dA, lda, lwork_qr))
  call hipsolverCheck(hipsolverCunmqr_bufferSize(handle, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, &
       M, N, K, dA, lda, dTau, dC, ldc, lwork_mq))
  lwork = max(lwork_qr, lwork_mq)
  call hipCheck(hipMalloc(dWork, int(lwork,c_size_t) * 8))

  call hipsolverCheck(hipsolverCgeqrf(handle, M, K, dA, lda, dTau, dWork, lwork, dInfo))
  call hipsolverCheck(hipsolverCunmqr(handle, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, &
       M, N, K, dA, lda, dTau, dC, ldc, dWork, lwork, dInfo))

  call hipCheck(hipMemcpy(c_loc(hC(1,1)), dC, size_C * 8, hipMemcpyDeviceToHost))

  norm_out = sqrt(sum(abs(hC)**2))
  error = abs(norm_out - norm_in) / max(norm_in, 1.0_c_float)
  if(.not. (error .le. error_max)) then
      write(*,*) "FAILED! Norm not preserved! ||C|| = ", norm_in, " ||Q*C|| = ", norm_out
      call exit(1)
  end if
  ! Q is not the identity, so Q*C must differ from C.
  if(.not. (sqrt(sum(abs(hC - hC0)**2)) .gt. error_max)) then
      write(*,*) "FAILED! Q*C = C: the multiplication had no effect"
      call exit(1)
  end if

  call hipCheck(hipFree(dWork))
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dC))
  call hipCheck(hipFree(dTau))
  call hipCheck(hipFree(dInfo))
  call hipsolverCheck(hipsolverDestroy(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program cunmqr
