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

program hipfft_xt_makeplanmany_c
  use iso_c_binding
  use hip
  use hipfft

  implicit none

  integer(c_int), parameter :: N = 32, batch = 3
  integer(c_size_t), parameter :: Nbytes = int(N,c_size_t) * int(batch,c_size_t) * 8
  real, parameter :: error_max = 1.0e-5

  complex(c_float_complex), target :: hx(N, batch), hx_input(N, batch)
  integer(c_int64_t), target :: nn(1)
  type(c_ptr) :: plan = c_null_ptr
  type(c_ptr) :: dx = c_null_ptr
  integer(c_size_t) :: work, work_query
  integer :: i, b
  real :: error

  write(*,"(a)",advance="no") "-- Running test 'hipFFT XtMakePlanMany C2C' (Fortran 2003 interfaces) - "

  do b = 1, batch
     do i = 1, N
        hx(i, b) = cmplx(real(i + b), real(N - i), c_float_complex)
     end do
  end do
  hx_input = hx

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx), Nbytes, hipMemcpyHostToDevice))

  ! The Xt plan names its data types (hipDataType) instead of an hipfftType.
  nn(1) = N
  call hipfftCheck(hipfftCreate(plan))
  call hipfftCheck(hipfftXtGetSizeMany(plan, 1, c_loc(nn), c_null_ptr, 1_c_int64_t, &
       int(N,c_int64_t), HIP_C_32F, c_null_ptr, 1_c_int64_t, int(N,c_int64_t), HIP_C_32F, &
       int(batch,c_int64_t), work_query, HIP_C_32F))
  call hipfftCheck(hipfftXtMakePlanMany(plan, 1, c_loc(nn), c_null_ptr, 1_c_int64_t, &
       int(N,c_int64_t), HIP_C_32F, c_null_ptr, 1_c_int64_t, int(N,c_int64_t), HIP_C_32F, &
       int(batch,c_int64_t), work, HIP_C_32F))

  call hipfftCheck(hipfftXtExec(plan, dx, dx, HIPFFT_FORWARD))
  call hipfftCheck(hipfftXtExec(plan, dx, dx, HIPFFT_BACKWARD))
  call hipCheck(hipDeviceSynchronize())
  call hipfftCheck(hipfftDestroy(plan))

  call hipCheck(hipMemcpy(c_loc(hx), dx, Nbytes, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  ! hipFFT is unnormalized, so forward+inverse yields N times the original input.
  do b = 1, batch
     do i = 1, N
        error = abs(hx(i, b) - N * hx_input(i, b)) / (N * max(1.0, abs(hx_input(i, b))))
        if (.not. (error <= error_max)) then
           write(*,*) "FAILED! i=", i, " b=", b, " error=", error
           call exit(1)
        end if
     end do
  end do

  write(*,*) "PASSED!"

end program hipfft_xt_makeplanmany_c
