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

program hipfftw_c2c_test
  use iso_c_binding
  use hip
  use hipfftw

  implicit none

  integer(c_int), parameter :: N = 16
  integer(c_size_t), parameter :: Nbytes = N * 16  ! sizeof(double complex) = 16
  double precision, parameter :: pi = 4.0d0 * atan(1.0d0)
  double precision, parameter :: tol = 1.0d-12

  complex(c_double_complex), allocatable, target, dimension(:) :: hx, hresult
  type(c_ptr) :: dx = c_null_ptr, dy = c_null_ptr
  type(c_ptr) :: plan = c_null_ptr
  integer :: j
  double precision :: error, max_error
  complex(c_double_complex) :: w

  write(*,"(a)",advance="no") "-- Running test 'hipfftw_c2c' (Fortran 2003) - "

  ! Signal: x[j] = exp(2*pi*i*1*j/N) + 2*exp(2*pi*i*5*j/N)
  ! Expected DFT: X[1]=N, X[5]=2*N, all other bins zero (0-indexed)
  w = cmplx(0.0d0, 2.0d0 * pi / dble(N), kind=c_double_complex)
  allocate(hx(N), hresult(N))
  do j = 0, N-1
    hx(j+1) = exp(w * dble(j)) + 2.0d0 * exp(5.0d0 * w * dble(j))
  end do

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMalloc(dy, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  ! Device pointers pass directly to the FFTW C API (in/out are void*).
  plan = fftw_plan_dft_1d(N, dx, dy, FFTW_FORWARD, FFTW_ESTIMATE)
  call fftw_execute_dft(plan, dx, dy)
  call fftw_destroy_plan(plan)

  call hipCheck(hipMemcpy(c_loc(hresult(1)), dy, Nbytes, hipMemcpyDeviceToHost))

  max_error = 0.0d0
  do j = 0, N-1
    if (j == 1) then
      error = abs(hresult(j+1) - cmplx(dble(N), 0.0d0, kind=c_double_complex))
    else if (j == 5) then
      error = abs(hresult(j+1) - cmplx(2.0d0 * N, 0.0d0, kind=c_double_complex))
    else
      error = abs(hresult(j+1))
    end if
    if (max_error == max_error .and. .not. (error <= max_error)) max_error = error  ! keeps a NaN
  end do

  if (.not. (max_error <= tol)) then
    write(*,*) "FAILED! max error = ", max_error
    call exit(1)
  end if

  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))
  deallocate(hx, hresult)

  write(*,*) "PASSED!"
end program hipfftw_c2c_test
