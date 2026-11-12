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

program hip_dtrsm
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer(kind(HIPBLAS_SIDE_LEFT)),       parameter :: side   = HIPBLAS_SIDE_LEFT
  integer(kind(HIPBLAS_FILL_MODE_LOWER)), parameter :: uplo   = HIPBLAS_FILL_MODE_LOWER
  integer(kind(HIPBLAS_OP_N)),            parameter :: transA = HIPBLAS_OP_N
  integer(kind(HIPBLAS_DIAG_NON_UNIT)),   parameter :: diag   = HIPBLAS_DIAG_NON_UNIT
  integer, parameter :: m = 1024, n = 1024
  integer, parameter :: bytes_per_element = 8
  double precision, target :: alpha = 2.d0
  integer(c_size_t) :: NAbytes, NBbytes
  double precision, allocatable, target, dimension(:) :: hA, hB
  double precision, parameter :: x_exact = 1.d0
  double precision :: error
  double precision, parameter :: error_max = 10*epsilon(error)
  type(c_ptr) :: dA = c_null_ptr, dB = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j

  write(*,"(a)",advance="no") "-- Running test 'DTRSM' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  NAbytes = int(m, c_size_t) * int(m, c_size_t) * bytes_per_element
  NBbytes = int(m, c_size_t) * int(n, c_size_t) * bytes_per_element
  allocate(hA(m*m), hB(m*n))

  hA = 0.d0
  do j = 1, m
    do i = j, m
      hA(i + (j-1)*m) = 1.d0
    end do
  end do
  do j = 1, n
    do i = 1, m
      hB(i + (j-1)*m) = dble(i) / 2.d0
    end do
  end do

  call hipCheck(hipMalloc(dA, NAbytes))
  call hipCheck(hipMalloc(dB, NBbytes))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1)), NAbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1)), NBbytes, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasDtrsm(handle, side, uplo, transA, diag, m, n, c_loc(alpha), dA, m, dB, m))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hB(1)), dB, NBbytes, hipMemcpyDeviceToHost))

  do i = 1, m*n
    error = abs((x_exact - hB(i)) / x_exact)
    if (error > error_max) then
      write(*,*) "FAILED! Error bigger than max! Error = ", error, " hB(", i, ") = ", hB(i)
      call exit(1)
    end if
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dB))
  call hipblasCheck(hipblasDestroy(handle))
  deallocate(hA, hB)
  write(*,*) "PASSED!"

end program hip_dtrsm
