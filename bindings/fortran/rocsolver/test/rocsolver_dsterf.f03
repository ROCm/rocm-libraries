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
! dsterf example
!!!!!!!!!!!!!!/
!
program dsterf
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i ! index for iterating over results

  ! Define our input data
  real(c_double), target :: hD(3) = (/ 2, 2, 2 /)
  real(c_double), target :: hE(2) = (/ -1, -1 /)
  real(c_double), target :: hResult(3) = (/ 0.58578643762690485d0, 2.0d0, 3.41421356237309515d0 /)
  integer(c_int), target :: hInfo = -1

  integer(c_int), parameter :: n = 3

  integer(c_size_t) :: size_D = size(hD)
  integer(c_size_t) :: size_E = size(hE)
  integer(c_size_t) :: size_Info = 1

  type(c_ptr) :: dD     ! GPU buffer for D
  type(c_ptr) :: dE     ! GPU buffer for E
  type(c_ptr) :: dInfo  ! GPU buffer for myInfo

  type(c_ptr) :: handle ! rocblas_handle

  real(c_double) :: error
  real(c_double), parameter :: error_max = 100 * epsilon(error_max)
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dsterf' (Fortran 2003 interfaces) - "

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dD, size_D * 8))
  call hipCheck(hipMalloc(dE, size_E * 8))
  call hipCheck(hipMalloc(dInfo, size_Info * 4))

  ! Create rocBLAS handle
  call rocblasCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dD, c_loc(hD(1)), size_D * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dE, c_loc(hE(1)), size_E * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dInfo, c_loc(hInfo), size_Info * 4, hipMemcpyHostToDevice))

  ! Compute eigenvalues
  call rocsolverCheck(rocsolver_dsterf(handle, n, dD, dE, dInfo))

  ! Copy result from device to host
  call hipCheck(hipMemcpy(c_loc(hD(1)), dD,    size_D * 8,    hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hInfo), dInfo, size_Info * 4, hipMemcpyDeviceToHost))

  ! Check and output results
  if(hInfo /= 0) then
    write(*,*) "FAILED! info = ", hInfo, " (expected 0)"
    call exit(1)
  else
    do i = 1,n
      error = abs(hD(i) - hResult(i)) / abs(hResult(i))
        if(.not. (error .le. error_max)) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " hD(", i, ") = ", hD(i)
            call exit(1)
        end if
    end do
  end if

  ! Clean up
  call hipCheck(hipFree(dD))
  call hipCheck(hipFree(dE))
  call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"
  write(*,*) "Eigenvalues in increasing order:"
  do i = 1,n
    write(*,*) hD(i)
  end do

end program dsterf
