!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2020-2026 Advanced Micro Devices, Inc.
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

program rocblas_sgemv_test

  use iso_c_binding
  use hip
  use rocblas

  implicit none

  integer(kind(rocblas_operation_none)), parameter :: trans = rocblas_operation_none
  real(c_float), parameter :: alpha = 1.1, beta = 0.9

  integer, parameter :: m = 1024, n = 1024

  real(c_float), allocatable, target, dimension(:) :: hA, hx, hy
  real(c_float) :: y_exact

  type(c_ptr) :: dA = c_null_ptr, dx = c_null_ptr, dy = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr

  integer, parameter :: bytes_per_element = 4 ! single precision
  integer(c_size_t) :: NAbytes, Nxbytes, Nybytes

  integer :: i
  real(c_float) :: error
  real(c_float), parameter :: error_max = 10*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'SGEMV' (Fortran 2003 interfaces) - "

  ! Create rocblas handle and set host pointer mode for host alpha/beta
  call rocblasCheck(rocblas_create_handle(handle))
  call rocblasCheck(rocblas_set_pointer_mode(handle, 0))

  NAbytes = int(m*n, c_size_t) * bytes_per_element
  Nxbytes = int(n,   c_size_t) * bytes_per_element
  Nybytes = int(m,   c_size_t) * bytes_per_element

  allocate(hA(m*n), hx(n), hy(m))

  ! Use these constant matrix/vectors so the exact answer is also a
  ! constant vector and therefore easy to check
  hA(:) = 1.
  hx(:) = 1.
  hy(:) = 1.
  y_exact = alpha * n + beta   ! = 1.1*1024 + 0.9 = 1127.3

  ! Allocate device memory
  call hipCheck(hipMalloc(dA, NAbytes))
  call hipCheck(hipMalloc(dx, Nxbytes))
  call hipCheck(hipMalloc(dy, Nybytes))

  ! Transfer from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1)), NAbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nxbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, c_loc(hy(1)), Nybytes, hipMemcpyHostToDevice))

  call rocblasCheck(rocblas_sgemv(handle, trans, m, n, alpha, dA, m, dx, 1, beta, dy, 1))

  call hipCheck(hipDeviceSynchronize())

  ! Transfer data back to host memory
  call hipCheck(hipMemcpy(c_loc(hy(1)), dy, Nybytes, hipMemcpyDeviceToHost))

  do i = 1,m
     error = abs((y_exact - hy(i))/y_exact)
     if( error > error_max )then
        write(*,*) "FAILED! Error bigger than max! Error = ", error
        call exit(1)
     end if
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))

  call rocblasCheck(rocblas_destroy_handle(handle))

  deallocate(hA, hx, hy)

  write(*,*) "PASSED!"

end program rocblas_sgemv_test
