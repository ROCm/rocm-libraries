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

program hip_zdotc
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer, parameter :: n = 10240
  integer, parameter :: bytes_per_element = 16
  integer(c_size_t) :: Nbytes
  complex(kind=8), allocatable, target, dimension(:) :: hx, hy
  complex(kind=8), allocatable, target :: res
  complex(kind=8) :: res_exact
  double precision :: error
  double precision, parameter :: error_max = 10*epsilon(error)
  type(c_ptr) :: dx = c_null_ptr, dy = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr
  ! On the heap, so 16-byte aligned: the CUDA backend accesses a
  ! cuDoubleComplex scalar with an aligned load or store.
  allocate(res)

  write(*,"(a)",advance="no") "-- Running test 'ZDOTC' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  Nbytes = int(n, c_size_t) * bytes_per_element
  allocate(hx(n), hy(n))
  hx = (1.d0, 1.d0)
  hy = (2.d0, 0.d0)
  res_exact = cmplx(2.d0*n, -2.d0*n, kind=8)

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMalloc(dy, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, c_loc(hy(1)), Nbytes, hipMemcpyHostToDevice))

  res = (0.d0, 0.d0)
  call hipblasCheck(hipblasZdotc(handle, n, dx, 1, dy, 1, c_loc(res)))
  call hipCheck(hipDeviceSynchronize())

  error = abs((res_exact - res) / res_exact)
  if (error > error_max) then
    write(*,*) "FAILED! error = ", error, " result = ", res
    call exit(1)
  end if

  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))
  call hipblasCheck(hipblasDestroy(handle))
  deallocate(hx, hy, res)
  write(*,*) "PASSED!"

end program hip_zdotc
