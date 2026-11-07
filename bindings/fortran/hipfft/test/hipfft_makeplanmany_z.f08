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

program hipfft_makeplanmany_z
  use iso_c_binding
  use hip
  use hipfft

  implicit none

  ! hipfftCreate / hipfftSetAutoAllocation / hipfftMakePlanMany /
  ! hipfftGetSizeMany / explicit work buffer / hipfftExecZ2Z
  ! (Fortran 2008 interfaces)
  integer(c_int), parameter :: N = 16
  integer(c_int), parameter :: batch_count = 3
  integer(c_int), parameter :: Ntot = N * batch_count

  ! nlen, inembed_1d, onembed_1d are target so the rank_1 wrapper can c_loc them.
  integer(c_int), target :: nlen(1), inembed_1d(1), onembed_1d(1)
  ! Non-VALUE scalars for the rank_1 wrappers.
  integer(c_int) :: fft_rank, istride_v, idist_v, ostride_v, odist_v, batch_v
  integer(kind(HIPFFT_Z2Z)) :: fft_type

  double precision, parameter :: pi = 4.0d0 * atan(1.0d0)
  double precision, parameter :: tol = 1.0d-8

  complex(8), allocatable, target, dimension(:) :: hx
  complex(8), pointer, dimension(:) :: dx => null()
  type(c_ptr) :: plan = c_null_ptr
  type(c_ptr) :: workbuf = c_null_ptr
  integer(c_size_t) :: workSzMake, gSzMany
  integer(c_int) :: direction
  integer :: b, j, k_exp, pos
  complex(8) :: expected
  double precision :: error, max_error

  write(*,"(a)",advance="no") &
    "-- Running test 'hipFFT MakePlanMany Z2Z 1D batched (z)' (Fortran 2008 interfaces) - "

  nlen(1)      = N
  inembed_1d(1) = N
  onembed_1d(1) = N
  fft_rank  = 1
  istride_v = 1
  idist_v   = N
  ostride_v = 1
  odist_v   = N
  batch_v   = batch_count
  fft_type  = HIPFFT_Z2Z

  allocate(hx(Ntot))

  ! Batch b (0-indexed) holds exp(2*pi*i*(b+1)*j/N), j=0..N-1.
  ! Forward transform must place N in bin b+1 (0-indexed) and zero elsewhere.
  do b = 0, batch_count - 1
     do j = 0, N - 1
        hx(b * N + j + 1) = exp(cmplx(0.0d0, &
             2.0d0 * pi * dble((b + 1) * j) / dble(N), kind=8))
     end do
  end do

  call hipCheck(hipMalloc(dx, source=hx))

  call hipfftCheck(hipfftCreate(plan))
  call hipfftCheck(hipfftSetAutoAllocation(plan, 0_c_int))

  ! MakePlanMany configures the plan and returns the required work area size.
  call hipfftCheck(hipfftMakePlanMany(plan, fft_rank, nlen, inembed_1d, &
       istride_v, idist_v, onembed_1d, ostride_v, odist_v, &
       fft_type, batch_v, workSzMake))

  ! GetSizeMany queries the configured plan; must agree with MakePlanMany.
  call hipfftCheck(hipfftGetSizeMany(plan, fft_rank, nlen, inembed_1d, &
       istride_v, idist_v, onembed_1d, ostride_v, odist_v, &
       fft_type, batch_v, gSzMany))

  if (gSzMany /= workSzMake) then
     write(*,*) "FAILED! GetSizeMany=", gSzMany, " MakePlanMany=", workSzMake
     STOP 1
  end if

  ! Provide the work buffer explicitly (auto-alloc was disabled above).
  if (workSzMake > 0_c_size_t) then
     call hipCheck(hipMalloc(workbuf, workSzMake))
     call hipfftCheck(hipfftSetWorkArea(plan, workbuf))
  end if

  direction = HIPFFT_FORWARD
  call hipfftCheck(hipfftExecZ2Z(plan, dx, dx, direction))
  call hipCheck(hipDeviceSynchronize())

  call hipfftCheck(hipfftDestroy(plan))

  if (c_associated(workbuf)) call hipCheck(hipFree(workbuf))

  call hipCheck(hipMemcpy(hx, dx, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  max_error = 0.0d0
  do b = 0, batch_count - 1
     k_exp = b + 1
     do j = 0, N - 1
        pos = b * N + j
        if (j == k_exp) then
           expected = cmplx(dble(N), 0.0d0, kind=8)
        else
           expected = (0.0d0, 0.0d0)
        end if
        error = abs(hx(pos + 1) - expected)
        if (max_error == max_error .and. .not. (error <= max_error)) max_error = error  ! keeps a NaN
     end do
  end do

  if (.not. (max_error <= tol * N)) then
     write(*,*) "FAILED! max error = ", max_error
     STOP 1
  end if

  deallocate(hx)

  write(*,*) "PASSED!"

end program hipfft_makeplanmany_z
