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

! hipFFTW allocation API: fftw_alloc_real / fftw_alloc_complex and their
! single-precision counterparts return host-accessible buffers that can be
! handed straight to a plan, mirroring the allocation tests in
! clients/tests/hipfftw_test.cpp.
program hipfftw_alloc_test
  use iso_c_binding
  use hip
  use hipfftw

  implicit none

  integer(c_int), parameter :: N = 16
  integer(c_int), parameter :: Nc = N / 2 + 1
  double precision, parameter :: pi = 4.0d0 * atan(1.0d0)
  double precision, parameter :: tol = 1.0d-12
  real, parameter :: tol_s = 1.0e-4

  type(c_ptr) :: pr = c_null_ptr, pc = c_null_ptr, plan = c_null_ptr
  real(c_double), pointer :: x(:) => null()
  complex(c_double_complex), pointer :: y(:) => null()
  real(c_float), pointer :: xs(:) => null()
  complex(c_float_complex), pointer :: ys(:) => null()
  integer :: j
  double precision :: error, max_error

  write(*,"(a)",advance="no") "-- Running test 'hipfftw_alloc' (Fortran 2003) - "

  ! Real signal: x[j] = 1 + 2*cos(2*pi*j/N) + 3*cos(2*pi*3*j/N)
  ! R2C output (0-indexed): X[0] = N, X[1] = N, X[3] = 3*N/2, rest zero.
  pr = fftw_alloc_real(int(N, c_size_t))
  pc = fftw_alloc_complex(int(Nc, c_size_t))
  if (.not. c_associated(pr) .or. .not. c_associated(pc)) then
    write(*,*) "FAILED! fftw_alloc_real/fftw_alloc_complex returned a null pointer"
    call exit(1)
  end if

  call c_f_pointer(pr, x, [N])
  call c_f_pointer(pc, y, [Nc])
  do j = 0, N-1
    x(j+1) = 1.0d0 + 2.0d0 * cos(2.0d0 * pi * j / dble(N)) &
                   + 3.0d0 * cos(6.0d0 * pi * j / dble(N))
  end do

  plan = fftw_plan_dft_r2c_1d(N, pr, pc, FFTW_ESTIMATE)
  call fftw_execute_dft_r2c(plan, pr, pc)
  call fftw_destroy_plan(plan)

  max_error = 0.0d0
  do j = 0, Nc-1
    if (j == 0 .or. j == 1) then
      error = abs(y(j+1) - cmplx(dble(N), 0.0d0, kind=c_double_complex))
    else if (j == 3) then
      error = abs(y(j+1) - cmplx(1.5d0 * N, 0.0d0, kind=c_double_complex))
    else
      error = abs(y(j+1))
    end if
    if (max_error == max_error .and. .not. (error <= max_error)) max_error = error  ! keeps a NaN
  end do

  call fftw_free(pr)
  call fftw_free(pc)

  if (.not. (max_error <= tol)) then
    write(*,*) "FAILED! double precision: max error = ", max_error
    call exit(1)
  end if

  pr = fftwf_alloc_real(int(N, c_size_t))
  pc = fftwf_alloc_complex(int(Nc, c_size_t))
  if (.not. c_associated(pr) .or. .not. c_associated(pc)) then
    write(*,*) "FAILED! fftwf_alloc_real/fftwf_alloc_complex returned a null pointer"
    call exit(1)
  end if

  call c_f_pointer(pr, xs, [N])
  call c_f_pointer(pc, ys, [Nc])
  do j = 0, N-1
    xs(j+1) = real(1.0d0 + 2.0d0 * cos(2.0d0 * pi * j / dble(N)) &
                         + 3.0d0 * cos(6.0d0 * pi * j / dble(N)), c_float)
  end do

  plan = fftwf_plan_dft_r2c_1d(N, pr, pc, FFTW_ESTIMATE)
  call fftwf_execute_dft_r2c(plan, pr, pc)
  call fftwf_destroy_plan(plan)

  max_error = 0.0d0
  do j = 0, Nc-1
    if (j == 0 .or. j == 1) then
      error = abs(ys(j+1) - cmplx(real(N), 0.0, kind=c_float_complex))
    else if (j == 3) then
      error = abs(ys(j+1) - cmplx(1.5 * N, 0.0, kind=c_float_complex))
    else
      error = abs(ys(j+1))
    end if
    if (max_error == max_error .and. .not. (error <= max_error)) max_error = error  ! keeps a NaN
  end do

  call fftwf_free(pr)
  call fftwf_free(pc)

  if (.not. (max_error <= tol_s)) then
    write(*,*) "FAILED! single precision: max error = ", max_error
    call exit(1)
  end if

  write(*,*) "PASSED!"
end program hipfftw_alloc_test
