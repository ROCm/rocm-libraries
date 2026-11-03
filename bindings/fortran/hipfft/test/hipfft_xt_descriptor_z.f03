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

program hipfft_xt_descriptor_z
  use iso_c_binding
  use hip
  use hipfft

  implicit none

  integer(c_int), parameter :: NX = 16, NY = 16
  double precision, parameter :: error_max = 1.0d-10

  complex(c_double_complex), target :: hx(NX, NY), hx_input(NX, NY)
  integer(c_int), target :: gpus(2)
  type(c_ptr) :: plan = c_null_ptr
  type(c_ptr) :: d = c_null_ptr
  type(hipLibXtDesc), pointer :: desc
  type(hipXtDesc), pointer :: xt
  integer(c_size_t), target :: work(2)   ! one work size per GPU
  integer(c_int) :: ndev
  integer :: i, j
  double precision :: error

  write(*,"(a)",advance="no") "-- Running test 'hipFFT Xt descriptor 2D Z2Z' (Fortran 2003 interfaces) - "

  ! The descriptor API needs a plan spread over at least two GPUs.
  call hipCheck(hipGetDeviceCount(ndev))
  ! Exit 77, which CTest reports as skipped: nothing was checked.
  if (ndev < 2) then
     write(*,*) "SKIPPED (needs two GPUs, found ", ndev, ")"
     stop 77
  end if

  do j = 1, NY
     do i = 1, NX
        hx(i, j) = cmplx(dble(i + j), dble(NX - i), c_double_complex)
     end do
  end do
  hx_input = hx

  gpus = [0, 1]
  call hipfftCheck(hipfftCreate(plan))
  call hipfftCheck(hipfftXtSetGPUs(plan, 2, c_loc(gpus)))
  ! A multi-GPU plan writes one work size per GPU: pass c_loc of an array with
  ! one slot per GPU to the type(c_ptr) form.
  call hipfftCheck(hipfftMakePlan2d(plan, NY, NX, HIPFFT_Z2Z, c_loc(work)))

  ! hipFFT allocates and owns the descriptor: keep its address, and pass the
  ! pointer that c_f_pointer associates with it, never a copy of the type.
  call hipfftCheck(hipfftXtMalloc(plan, d, HIPFFT_XT_FORMAT_INPLACE))
  call c_f_pointer(d, desc)
  call c_f_pointer(desc%descriptor, xt)
  if (xt%nGPUs /= 2) then
     write(*,*) "FAILED! descriptor spans ", xt%nGPUs, " GPUs (expected 2)"
     call exit(1)
  end if

  call hipfftCheck(hipfftXtMemcpy(plan, d, c_loc(hx), HIPFFT_COPY_HOST_TO_DEVICE))
  call hipfftCheck(hipfftXtExecDescriptorZ2Z(plan, desc, desc, HIPFFT_FORWARD))
  call hipfftCheck(hipfftXtExecDescriptorZ2Z(plan, desc, desc, HIPFFT_BACKWARD))
  call hipfftCheck(hipfftXtMemcpy(plan, c_loc(hx), d, HIPFFT_COPY_DEVICE_TO_HOST))
  call hipfftCheck(hipfftXtFree(desc))
  call hipfftCheck(hipfftDestroy(plan))

  ! hipFFT is unnormalized, so forward+inverse yields NX*NY times the input.
  do j = 1, NY
     do i = 1, NX
        error = abs(hx(i, j) - NX * NY * hx_input(i, j)) / (NX * NY * max(1.0d0, abs(hx_input(i, j))))
        if (.not. (error <= error_max)) then
           write(*,*) "FAILED! i=", i, " j=", j, " error=", error
           call exit(1)
        end if
     end do
  end do

  write(*,*) "PASSED!"

end program hipfft_xt_descriptor_z
