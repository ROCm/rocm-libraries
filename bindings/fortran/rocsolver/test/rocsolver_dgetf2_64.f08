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
! dgetf2_64 example (unblocked LU with the 64-bit integer API)
! Same math as dgetrf; dimensions, pivots and info are 64-bit. Device buffers
! (int64 ipiv/info included) are passed as type(c_ptr) - the _64 routines have
! no native-array overloads. info lives on the DEVICE. The input is diagonally
! dominant so no pivoting occurs and L*U == A is checked by reconstruction
! (layout-agnostic).
!!!!!!!!!!!!!!
!
program dgetf2_64
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver
  implicit none
  integer(c_int64_t), parameter :: M = 3, N = 3, lda = 3
  real(c_double), target :: hA(3,3)  = reshape((/4, 1, 1,  1, 4, 1,  1, 1, 4/), (/3, 3/))
  real(c_double), target :: hLU(3,3)
  integer(c_size_t) :: size_A = 9
  type(c_ptr) :: dA, dIpiv, dInfo, handle
  integer(c_int64_t), target :: hInfo(1)
  integer :: i, j, k
  real(c_double) :: lij, recon, error
  real(c_double), parameter :: error_max = 1.0d-10
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dgetf2_64' (Fortran 2008 interfaces) - "
  call hipCheck(hipMalloc(dA,    size_A * 8))
  call hipCheck(hipMalloc(dIpiv, 3_c_size_t * 8))   ! int64 pivots
  call hipCheck(hipMalloc(dInfo, 8_c_size_t))       ! int64 info
  call rocblasCheck(rocblas_create_handle(handle))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))
  call rocsolverCheck(rocsolver_dgetf2_64(handle, M, N, dA, lda, dIpiv, dInfo))
  call hipCheck(hipMemcpy(c_loc(hInfo(1)), dInfo, 8_c_size_t, hipMemcpyDeviceToHost))
  if (hInfo(1) /= 0) then
    write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"; call exit(1)
  end if
  call hipCheck(hipMemcpy(c_loc(hLU(1,1)), dA, size_A * 8, hipMemcpyDeviceToHost))
  ! Reconstruct A = L*U (unit-lower L below the diagonal, U on/above).
  do j = 1, 3
     do i = 1, 3
        recon = 0.0d0
        do k = 1, min(i,j)
           if (k == i) then; lij = 1.0d0; else; lij = hLU(i,k); end if
           recon = recon + lij * hLU(k,j)
        end do
        error = abs(recon - hA(i,j))
        if (.not. (error <= error_max)) then
           write(*,*) "FAILED! (L*U)(", i, ",", j, ") = ", recon, " expected ", hA(i,j)
           call exit(1)
        end if
     end do
  end do
  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dIpiv)); call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle)); call hipCheck(hipDeviceReset())
  write(*,*) "PASSED!"
end program dgetf2_64
