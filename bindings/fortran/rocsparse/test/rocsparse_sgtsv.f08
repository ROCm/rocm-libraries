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
! sgtsv example (single-precision tridiagonal solve, rocSPARSE)
! see: https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/precond.html
!
! Solves A x = b for a non-symmetric, diagonally dominant tridiagonal system
! (d = 4, dl = -1, du = 2; dl(1) = du(m) = 0), with b = A*x_known and
! x_known(i) = i, and checks the solution against x_known. dl/d/du/B are
! passed as Fortran arrays, exercising the generic array form of the diagonal
! arguments (they used to be declared type(c_ptr), SWDEV-485451).
!!!!!!!!!!!!!!
!
program rocsparse_sgtsv_test

  use iso_c_binding
  use hip
  use rocsparse

  implicit none

  integer, parameter :: m = 512   ! tridiagonal system size
  integer, parameter :: n = 1     ! single right-hand side

  ! Non-symmetric diagonally dominant tridiagonal; b = A*x_known with x_known(i) = i.
  real(c_float), allocatable, dimension(:) :: hdl, hd, hdu, hB, hx
  real(c_float), pointer, dimension(:) :: ddl => null(), dd => null(), &
                                          ddu => null(), dB => null()
  type(c_ptr) :: handle  = c_null_ptr
  type(c_ptr) :: dbuffer = c_null_ptr
  integer(c_size_t) :: buffer_size

  integer :: i
  real(c_float) :: error
  real(c_float), parameter :: error_max = 100*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'SGTSV' (Fortran 2008 interfaces) - "

  call rocsparseCheck(rocsparse_create_handle(handle))

  allocate(hdl(m), hd(m), hdu(m), hB(m), hx(m))

  hdl(:) = -1.0
  hd(:)  = 4.0
  hdu(:) = 2.0
  hdl(1) = 0.0   ! gtsv convention: dl(1) and du(m) are unused and must be 0
  hdu(m) = 0.0
  do i = 1, m
    hx(i) = real(i)   ! known solution x_known(i) = i
  end do
  ! Right-hand side b = A*x_known
  do i = 1, m
    hB(i) = hd(i)*hx(i)
    if (i > 1) hB(i) = hB(i) + hdl(i)*hx(i-1)
    if (i < m) hB(i) = hB(i) + hdu(i)*hx(i+1)
  end do

  call hipCheck(hipMalloc(ddl, source=hdl))
  call hipCheck(hipMalloc(dd,  source=hd))
  call hipCheck(hipMalloc(ddu, source=hdu))
  call hipCheck(hipMalloc(dB,  source=hB))

  ! Query the temporary buffer size. dl/d/du/B are passed as Fortran arrays -
  ! this is the generic array form that used to fail to compile because the
  ! diagonals were declared type(c_ptr) (SWDEV-485451).
  call rocsparseCheck(rocsparse_sgtsv_buffer_size(handle, m, n, ddl, dd, ddu, dB, m, buffer_size))

  call hipCheck(hipMalloc(dbuffer, buffer_size))

  ! Tridiagonal solve A x = b, in place; dl/d/du/B passed as Fortran arrays.
  call rocsparseCheck(rocsparse_sgtsv(handle, m, n, ddl, dd, ddu, dB, m, dbuffer))

  call hipCheck(hipDeviceSynchronize())

  call hipCheck(hipMemcpy(hB, dB, hipMemcpyDeviceToHost))

  do i = 1, m
    error = abs((hx(i) - hB(i))/hx(i))
    if(.not. (error <= error_max)) then
      write(*,*) "FAILED! Error bigger than max! Error = ", error, " hB(", i, ") = ", hB(i)
      call exit(1)
    end if
  end do

  call hipCheck(hipFree(ddl))
  call hipCheck(hipFree(dd))
  call hipCheck(hipFree(ddu))
  call hipCheck(hipFree(dB))
  call hipCheck(hipFree(dbuffer))

  call rocsparseCheck(rocsparse_destroy_handle(handle))

  deallocate(hdl, hd, hdu, hB, hx)

  write(*,*) "PASSED!"

end program rocsparse_sgtsv_test
