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
! zgesvd example (double-precision complex singular value decomposition)
! see: https://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga84fdf22a62b12ff364621e4713ce02f2.html
!
! Self-verifying: compute A = U * S * V (rocSOLVER returns V as V**H), then
! confirm the reconstruction matches the original A.
!
! NOTE: gesvd writes `info` to DEVICE memory, so info is backed by a device
! allocation and passed as c_loc(dInfo); passing a host scalar faults on the GPU.
! For complex matrices A, U, V are complex but S and E are REAL.
!!!!!!!!!!!!!!/
!
program zgesvd
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i, j ! indices for iterating over results

  integer(c_int), parameter :: M = 3
  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: ldu = 3
  integer(c_int), parameter :: ldv = 3

  ! Input matrix (column-major) and a copy kept for verification
  complex(c_double_complex) :: hA(3,3) = reshape((/ &
    (12.,1.),(6.,-2.),(-4.,3.), (-51.,2.),(167.,-1.),(24.,4.), (4.,-3.),(-68.,2.),(-41.,1.) /), (/3, 3/))
  complex(c_double_complex) :: hA0(3,3)
  real(c_double) :: hS(3)                    ! singular values (real)
  complex(c_double_complex) :: hU(3,3)       ! left singular vectors
  complex(c_double_complex) :: hV(3,3)       ! right singular vectors (stored as V**H)
  real(c_double) :: hE(2)                     ! superdiagonal workspace (real)
  complex(c_double_complex) :: recon(3,3), Sd(3,3)

  complex(c_double_complex), pointer :: dA(:,:)
  real(c_double), pointer :: dS(:)
  complex(c_double_complex), pointer :: dU(:,:)
  complex(c_double_complex), pointer :: dV(:,:)
  real(c_double), pointer :: dE(:)
  integer(c_int), pointer :: dInfo(:)
  integer(c_int) :: hInfo(1)

  type(c_ptr) :: handle ! rocblas_handle

  real(c_double) :: error
  real(c_double), parameter :: error_max = 1.0d-9
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_zgesvd' (Fortran 2008 interfaces) - "

  hA0 = hA ! keep the original for the reconstruction check

  ! Allocate device-side memory & copy memory from host to device
  call hipCheck(hipMalloc(dA, source=hA))
  call hipCheck(hipMalloc(dS, mold=hS))
  call hipCheck(hipMalloc(dU, mold=hU))
  call hipCheck(hipMalloc(dV, mold=hV))
  call hipCheck(hipMalloc(dE, mold=hE))
  call hipCheck(hipMalloc(dInfo, 1))

  ! Create rocBLAS handle
  call rocblasCheck(rocblas_create_handle(handle))

  ! Compute the full SVD on the device
  call rocsolverCheck(rocsolver_zgesvd(handle, rocblas_svect_all, rocblas_svect_all, M, N, dA, lda, &
                                 dS, dU, ldu, dV, ldv, dE, rocblas_outofplace, c_loc(dInfo)))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  if (hInfo(1) /= 0) then
    write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"; call exit(1)
  end if

  ! Copy factors back to host
  call hipCheck(hipMemcpy(hS, dS, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(hU, dU, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(hV, dV, hipMemcpyDeviceToHost))

  ! Reconstruct A = U * diag(S) * V (V is already V**H from rocSOLVER)
  Sd = (0.0_c_double, 0.0_c_double)
  do i = 1,N
    Sd(i,i) = cmplx(hS(i), 0.0_c_double, c_double_complex)
  end do
  recon = matmul(hU, matmul(Sd, hV))

  ! Verify reconstruction matches the original
  do j = 1,N
    do i = 1,M
        error = abs(recon(i,j) - hA0(i,j)) / max(abs(hA0(i,j)), 1.0_c_double)
        if(.not. (error .le. error_max)) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " at (", i, ",", j, ")"
            call exit(1)
        end if
    end do
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dS))
  call hipCheck(hipFree(dU))
  call hipCheck(hipFree(dV))
  call hipCheck(hipFree(dE))
  call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program zgesvd
