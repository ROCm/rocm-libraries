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
! dgetf2_npvt example (double-precision unblocked LU without pivoting)
! Reconstructs L*U and checks it equals the original A (which must have an LU
! without pivoting - here an SPD, diagonally dominant matrix). rocSOLVER writes
! info to DEVICE memory (passed as c_loc(dInfo)).
!!!!!!!!!!!!!!
!
program dgetf2_npvt
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver
  implicit none
  integer(c_int), parameter :: M = 3, N = 3, lda = 3
  real(c_double) :: hA(3,3)  = reshape((/4, 1, 1,  1, 4, 1,  1, 1, 4/), (/3,3/))  ! SPD, no pivoting
  real(c_double) :: hLU(3,3)
  real(c_double), pointer :: dA(:,:)
  integer(c_int), pointer :: dInfo(:)
  integer(c_int) :: hInfo(1)
  type(c_ptr) :: handle
  integer :: i, j, k
  real(c_double) :: lij, ukj, recon, error
  real(c_double), parameter :: error_max = 1.0d-10
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dgetf2_npvt' (Fortran 2008 interfaces) - "
  call hipCheck(hipMalloc(dA, source=hA))
  call hipCheck(hipMalloc(dInfo, 1))
  call rocblasCheck(rocblas_create_handle(handle))
  call rocsolverCheck(rocsolver_dgetf2_npvt(handle, M, N, dA, lda, c_loc(dInfo)))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  if (hInfo(1) /= 0) then
    write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"; call exit(1)
  end if
  call hipCheck(hipMemcpy(hLU, dA, hipMemcpyDeviceToHost))
  ! Reconstruct A = L*U (L unit-lower from strictly-below, U on/above diagonal).
  do j = 1, N
     do i = 1, M
        recon = 0.0d0
        do k = 1, min(i,j)
           if (k == i) then
              lij = 1.0d0
           else
              lij = hLU(i,k)
           end if
           ukj = hLU(k,j)
           recon = recon + lij * ukj
        end do
        error = abs(recon - hA(i,j))
        if (.not. (error <= error_max)) then
           write(*,*) "FAILED! (L*U)(", i, ",", j, ") = ", recon, " expected ", hA(i,j)
           call exit(1)
        end if
     end do
  end do
  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle)); call hipCheck(hipDeviceReset())
  write(*,*) "PASSED!"
end program dgetf2_npvt
