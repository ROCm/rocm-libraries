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
! hipsolver cgetrf example (single-precision complex LU factorization, Fortran 2008 interfaces)
! see: https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! f2008 style: device buffers are native Fortran array pointers allocated with
! hipMalloc(source=/mold=); A and the pivots are passed as typed arrays. The
! workspace stays a bare type(c_ptr); devInfo is a device-backed scalar pointer.
!!!!!!!!!!!!!!
!
program hipsolver_cgetrf
  use iso_c_binding
  use hip
  use hipsolver

  implicit none
  integer :: i, j

  integer(c_int), parameter :: M = 3, N = 3, lda = 3

  ! Input (column-major); expected packed LU from hipSOLVER (== rocSOLVER cgetrf)
  complex(c_float_complex) :: hA(3,3) = reshape((/ &
    (12.,1.),(6.,-2.),(-4.,3.), (-51.,2.),(167.,-1.),(24.,4.), (4.,-3.),(-68.,2.),(-41.,1.) /), (/3, 3/))
  complex(c_float_complex) :: hResult(3,3) = reshape((/ &
    ( 0.1200000000000000E+02,  0.1000000000000000E+01), &
    ( 0.4827586114406586E+00, -0.2068965435028076E+00), &
    (-0.3103448152542114E+00,  0.2758620679378510E+00), &
    (-0.5100000000000000E+02,  0.2000000000000000E+01), &
    ( 0.1912068939208984E+03, -0.1251724052429199E+02), &
    ( 0.3906043618917465E-01,  0.1003027930855751E+00), &
    ( 0.4000000000000000E+01, -0.3000000000000000E+01), &
    (-0.6931034851074219E+02,  0.4275861740112305E+01), &
    (-0.3745003509521484E+02,  0.5750521659851074E+01) /), (/3, 3/))
  integer(c_int) :: hIpiv_ref(3) = (/1, 2, 3/)

  integer(c_int) :: hIpiv(3), hInfo

  complex(c_float_complex), pointer :: dA(:,:)   ! GPU buffer for A (holds packed LU on output)
  integer(c_int),           pointer :: dIpiv(:)  ! GPU buffer for pivots
  integer(c_int),           pointer :: dInfo     ! GPU scalar for devInfo (written on device)
  type(c_ptr) :: dWork                           ! opaque workspace
  type(c_ptr) :: handle = c_null_ptr
  integer(c_int) :: lwork

  real :: error
  real, parameter :: error_max = 1.0e-4

  write(*,"(a)",advance="no") "-- Running test 'hipsolver_cgetrf' (Fortran 2008 interfaces) - "

  call hipsolverCheck(hipsolverCreate(handle))

  ! Allocate device memory as native arrays and copy A to device
  call hipCheck(hipMalloc(dA,    source=hA))
  call hipCheck(hipMalloc(dIpiv, mold=hIpiv))
  call hipCheck(hipMalloc(dInfo))

  ! Query and allocate the workspace
  call hipsolverCheck(hipsolverCgetrf_bufferSize(handle, M, N, dA, lda, lwork))
  call hipCheck(hipMalloc(dWork, max(int(lwork,c_size_t) * 8, 1_c_size_t)))

  ! Compute the LU factorization. A and the pivots are native arrays, and devInfo is
  ! the device-resident integer itself, like them.
  call hipsolverCheck(hipsolverCgetrf(handle, M, N, dA, lda, dWork, lwork, dIpiv, dInfo))

  ! Copy results back to host
  call hipCheck(hipMemcpy(hA,    dA,    hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(hIpiv, dIpiv, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))

  ! Check info
  if(hInfo /= 0) then
    write(*,*) "FAILED! info = ", hInfo, " (expected 0)"
    call exit(1)
  end if

  ! Check factor values
  do j = 1,size(hA,2)
    do i = 1,size(hA,1)
        error = abs(hA(i,j) - hResult(i,j)) / max(abs(hResult(i,j)), 1.0)
        if(.not. (error .le. error_max)) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " hA(", i, ",", j, ") = ", hA(i,j)
            call exit(1)
        end if
    end do
  end do

  ! Check pivots
  do i = 1,3
    if(hIpiv(i) .ne. hIpiv_ref(i)) then
        write(*,*) "FAILED! Pivot mismatch at ", i, " got ", hIpiv(i), " expected ", hIpiv_ref(i)
        call exit(1)
    end if
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dIpiv))
  call hipCheck(hipFree(dInfo))
  call hipCheck(hipFree(dWork))
  call hipsolverCheck(hipsolverDestroy(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program hipsolver_cgetrf
