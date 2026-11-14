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

program hip_strsv
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer(kind(HIPBLAS_FILL_MODE_UPPER)), parameter :: uplo   = HIPBLAS_FILL_MODE_LOWER
  integer(kind(HIPBLAS_OP_N)),            parameter :: transA = HIPBLAS_OP_N
  integer(kind(HIPBLAS_DIAG_NON_UNIT)),   parameter :: diag   = HIPBLAS_DIAG_NON_UNIT
  integer, parameter :: m = 1024
  integer, parameter :: bytes_per_element = 4
  integer(c_size_t) :: NAbytes, Nxbytes
  real(kind=4), allocatable, target, dimension(:) :: hA, hx
  real(kind=4), parameter :: x_exact = 1.0
  real(kind=4) :: error
  real(kind=4), parameter :: error_max = 10*epsilon(error)
  type(c_ptr) :: dA = c_null_ptr, dx = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j

  write(*,"(a)",advance="no") "-- Running test 'STRSV' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  NAbytes = int(m, c_size_t) * int(m, c_size_t) * bytes_per_element
  Nxbytes = int(m, c_size_t) * bytes_per_element
  allocate(hA(m*m), hx(m))

  hA = 0.0
  do j = 1, m
    do i = j, m
      hA(i + (j-1)*m) = 1.0
    end do
  end do
  do i = 1, m
    hx(i) = real(i)
  end do

  call hipCheck(hipMalloc(dA, NAbytes))
  call hipCheck(hipMalloc(dx, Nxbytes))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1)), NAbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nxbytes, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasStrsv(handle, uplo, transA, diag, m, dA, m, dx, 1))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hx(1)), dx, Nxbytes, hipMemcpyDeviceToHost))

  do i = 1, m
    error = abs((x_exact - hx(i)) / x_exact)
    if (error > error_max) then
      write(*,*) "FAILED! Error bigger than max! Error = ", error, " hx(", i, ") = ", hx(i)
      call exit(1)
    end if
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dx))
  call hipblasCheck(hipblasDestroy(handle))
  deallocate(hA, hx)
  write(*,*) "PASSED!"

end program hip_strsv
