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

program hipfftw_r2c_c2r_test
  use iso_c_binding
  use hip
  use hipfftw

  implicit none

  integer(c_int), parameter :: N = 16
  integer(c_int), parameter :: Nc = N / 2 + 1
  integer(c_size_t), parameter :: Nbytes_r = N * 8    ! sizeof(double) = 8
  integer(c_size_t), parameter :: Nbytes_c = Nc * 16  ! sizeof(double complex) = 16
  double precision, parameter :: pi = 4.0d0 * atan(1.0d0)
  double precision, parameter :: tol = 1.0d-12

  real(c_double), allocatable, target, dimension(:) :: hx, hresult_r
  complex(c_double_complex), allocatable, target, dimension(:) :: hresult_c
  type(c_ptr) :: dx = c_null_ptr, dy = c_null_ptr, dz = c_null_ptr
  type(c_ptr) :: plan = c_null_ptr
  integer :: j
  double precision :: error, max_error

  write(*,"(a)",advance="no") "-- Running test 'hipfftw_r2c_c2r' (Fortran 2003) - "

  ! Real signal: x[j] = 1 + 2*cos(2*pi*j/N) + 3*cos(2*pi*3*j/N)
  ! R2C output (Nc = N/2+1 complex values, 0-indexed):
  !   X[0] = N, X[1] = N, X[3] = 3*N/2, rest zero
  allocate(hx(N), hresult_c(Nc), hresult_r(N))
  do j = 0, N-1
    hx(j+1) = 1.0d0 + 2.0d0 * cos(2.0d0 * pi * j / dble(N)) &
                     + 3.0d0 * cos(6.0d0 * pi * j / dble(N))
  end do

  call hipCheck(hipMalloc(dx, Nbytes_r))
  call hipCheck(hipMalloc(dy, Nbytes_c))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes_r, hipMemcpyHostToDevice))

  ! Forward R2C. Device pointers pass directly to the FFTW C API.
  plan = fftw_plan_dft_r2c_1d(N, dx, dy, FFTW_ESTIMATE)
  call fftw_execute_dft_r2c(plan, dx, dy)
  call fftw_destroy_plan(plan)

  call hipCheck(hipMemcpy(c_loc(hresult_c(1)), dy, Nbytes_c, hipMemcpyDeviceToHost))

  max_error = 0.0d0
  do j = 0, Nc-1
    if (j == 0 .or. j == 1) then
      error = abs(hresult_c(j+1) - cmplx(dble(N), 0.0d0, kind=c_double_complex))
    else if (j == 3) then
      error = abs(hresult_c(j+1) - cmplx(1.5d0 * N, 0.0d0, kind=c_double_complex))
    else
      error = abs(hresult_c(j+1))
    end if
    if (max_error == max_error .and. .not. (error <= max_error)) max_error = error  ! keeps a NaN
  end do

  if (.not. (max_error <= tol)) then
    write(*,*) "FAILED! R2C: max error = ", max_error
    call exit(1)
  end if

  ! Backward C2R: c2r(r2c(x)) should equal N * x
  call hipCheck(hipMemcpy(dy, c_loc(hresult_c(1)), Nbytes_c, hipMemcpyHostToDevice))
  call hipCheck(hipMalloc(dz, Nbytes_r))

  plan = fftw_plan_dft_c2r_1d(N, dy, dz, FFTW_ESTIMATE)
  call fftw_execute_dft_c2r(plan, dy, dz)
  call fftw_destroy_plan(plan)

  call hipCheck(hipMemcpy(c_loc(hresult_r(1)), dz, Nbytes_r, hipMemcpyDeviceToHost))

  max_error = 0.0d0
  do j = 1, N
    error = abs(hresult_r(j) - dble(N) * hx(j))
    if (max_error == max_error .and. .not. (error <= max_error)) max_error = error  ! keeps a NaN
  end do

  if (.not. (max_error <= tol)) then
    write(*,*) "FAILED! C2R round-trip: max error = ", max_error
    call exit(1)
  end if

  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))
  call hipCheck(hipFree(dz))
  deallocate(hx, hresult_c, hresult_r)

  write(*,*) "PASSED!"
end program hipfftw_r2c_c2r_test
