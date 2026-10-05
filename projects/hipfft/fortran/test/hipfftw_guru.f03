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

program hipfftw_guru_test
  use iso_c_binding
  use hip
  use hipfftw

  implicit none

  integer(c_int), parameter :: N = 16, HOWMANY = 3
  integer(c_size_t), parameter :: Nbytes = N * HOWMANY * 16  ! sizeof(double complex) = 16
  double precision, parameter :: pi = 4.0d0 * atan(1.0d0)
  double precision, parameter :: tol = 1.0d-10, tol_s = 1.0d-5

  complex(c_double_complex), allocatable, target, dimension(:) :: hx, hresult
  complex(c_float_complex), allocatable, target, dimension(:) :: hxs, hresults
  type(c_ptr) :: dx = c_null_ptr, dy = c_null_ptr
  type(c_ptr) :: plan = c_null_ptr
  type(fftw_iodim) :: dims(1), howmany_dims(1)
  integer :: b, j, k
  double precision :: error, max_error
  complex(c_double_complex) :: w, expected

  write(*,"(a)",advance="no") "-- Running test 'hipfftw_guru' (Fortran 2003) - "

  ! 1D batched C2C: HOWMANY batches of length N, contiguous (stride=1, dist=N).
  ! Batch b (0-indexed): x[j] = exp(2*pi*i*(b+1)*j/N).
  ! Expected forward DFT: output bin b+1 of batch b = N, all others zero.
  allocate(hx(N*HOWMANY), hresult(N*HOWMANY))
  do b = 0, HOWMANY-1
    w = cmplx(0.0d0, 2.0d0 * pi * dble(b+1) / dble(N), kind=c_double_complex)
    do j = 0, N-1
      hx(b*N + j + 1) = exp(w * dble(j))
    end do
  end do

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMalloc(dy, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  dims(1) = fftw_iodim(N, 1, 1)         ! n=N, is=1, os=1
  howmany_dims(1) = fftw_iodim(HOWMANY, N, N)  ! n=HOWMANY, is=N, os=N
  ! dims and howmany_dims are arrays of rank and howmany_rank fftw_iodim; the
  ! dummies are assumed-size type(fftw_iodim) :: dims(*), so pass the arrays.
  ! The fftwf_ planner below takes the same arrays (fftwf_iodim is fftw_iodim).
  plan = fftw_plan_guru_dft(1, dims, 1, howmany_dims, &
      dx, dy, FFTW_FORWARD, FFTW_ESTIMATE)
  call fftw_execute_dft(plan, dx, dy)
  call fftw_destroy_plan(plan)

  call hipCheck(hipMemcpy(c_loc(hresult(1)), dy, Nbytes, hipMemcpyDeviceToHost))

  ! Verify: for batch b, output bin b+1 should equal N; all others zero.
  max_error = 0.0d0
  do b = 0, HOWMANY-1
    do k = 0, N-1
      if (k == b+1) then
        expected = cmplx(dble(N), 0.0d0, kind=c_double_complex)
      else
        expected = cmplx(0.0d0, 0.0d0, kind=c_double_complex)
      end if
      error = abs(hresult(b*N + k + 1) - expected)
      if (max_error == max_error .and. .not. (error <= max_error)) max_error = error  ! keeps a NaN
    end do
  end do

  if (.not. (max_error <= tol * dble(N))) then
    write(*,*) "FAILED! max error = ", max_error
    call exit(1)
  end if

  ! Same transform in single precision through fftwf_plan_guru_dft. The buffers
  ! are reused: the single-precision data takes half the bytes.
  allocate(hxs(N*HOWMANY), hresults(N*HOWMANY))
  hxs = cmplx(hx, kind=c_float_complex)
  hresults = cmplx(-1.0, -1.0, kind=c_float_complex)
  call hipCheck(hipMemcpy(dx, c_loc(hxs(1)), Nbytes / 2, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, c_loc(hresults(1)), Nbytes / 2, hipMemcpyHostToDevice))

  plan = fftwf_plan_guru_dft(1, dims, 1, howmany_dims, &
      dx, dy, FFTW_FORWARD, FFTW_ESTIMATE)
  if (.not. c_associated(plan)) then
    write(*,*) "FAILED! fftwf_plan_guru_dft returned a null plan"
    call exit(1)
  end if
  call fftwf_execute_dft(plan, dx, dy)
  call fftwf_destroy_plan(plan)

  call hipCheck(hipMemcpy(c_loc(hresults(1)), dy, Nbytes / 2, hipMemcpyDeviceToHost))

  max_error = 0.0d0
  do b = 0, HOWMANY-1
    do k = 0, N-1
      if (k == b+1) then
        expected = cmplx(dble(N), 0.0d0, kind=c_double_complex)
      else
        expected = cmplx(0.0d0, 0.0d0, kind=c_double_complex)
      end if
      error = abs(cmplx(hresults(b*N + k + 1), kind=c_double_complex) - expected)
      if (max_error == max_error .and. .not. (error <= max_error)) max_error = error  ! keeps a NaN
    end do
  end do

  if (.not. (max_error <= tol_s * dble(N))) then
    write(*,*) "FAILED! single-precision max error = ", max_error
    call exit(1)
  end if

  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))
  deallocate(hx, hresult, hxs, hresults)

  write(*,*) "PASSED!"
end program hipfftw_guru_test
