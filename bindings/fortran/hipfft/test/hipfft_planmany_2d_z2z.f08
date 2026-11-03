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

program hipfft_planmany_2d_z2z
  use iso_c_binding
  use hip
  use hipfft

  implicit none

  ! Advanced-interface 2D batched C2C transform (Fortran 2008 interfaces).
  ! Last dimension (Ny) is contiguous, matching C row-major order.
  integer(c_int), parameter :: Nx = 4, Ny = 5
  integer(c_int), parameter :: howmany = 3
  integer(c_int), parameter :: Ntot = Nx * Ny

  ! nlen, inembed_a, onembed_a: target arrays passed directly to the rank_1 wrapper.
  integer(c_int), target :: nlen(2)
  integer(c_int), target :: inembed_a(2)
  integer(c_int), target :: onembed_a(2)
  ! Scalar plan-many arguments: non-VALUE in the rank_1 wrapper, so use variables.
  integer(c_int) :: fft_rank, istride_v, idist_v, ostride_v, odist_v, howmany_v

  double precision, parameter :: pi = 4.0d0 * atan(1.0d0)
  double precision, parameter :: tol = 1.0d-8

  complex(8), allocatable, target, dimension(:) :: hx
  complex(8), pointer, dimension(:) :: dx => null()
  type(c_ptr) :: plan = c_null_ptr
  integer(c_int) :: direction
  integer :: b, i, j, kx, ky, pos
  complex(8) :: expected
  double precision :: error, max_error

  write(*,"(a)",advance="no") &
    "-- Running test 'hipFFT PlanMany C2C 2D batched (z)' (Fortran 2008 interfaces) - "

  nlen      = [Nx, Ny]
  inembed_a = [Nx, Ny]
  onembed_a = [Nx, Ny]
  fft_rank  = 2
  istride_v = 1
  ostride_v = 1
  idist_v   = Nx * Ny
  odist_v   = Nx * Ny
  howmany_v = howmany

  allocate(hx(Ntot * howmany))

  ! Batch b holds exp(2*pi*i*(kx*ii/Nx + ky*jj/Ny)) with kx=b mod Nx, ky=(b+1) mod Ny.
  ! Forward transform must place Nx*Ny in bin (kx,ky) and zero everywhere else.
  ! A wrong stride, embed or batch distance moves energy into other bins and fails.
  do b = 0, howmany - 1
     kx = mod(b, Nx)
     ky = mod(b + 1, Ny)
     do i = 0, Nx - 1
        do j = 0, Ny - 1
           pos = b * idist_v + i * inembed_a(2) + j
           hx(pos + 1) = exp(cmplx(0.0d0, &
                2.0d0 * pi * (dble(kx * i) / dble(Nx) + dble(ky * j) / dble(Ny)), &
                kind=8))
        end do
     end do
  end do

  call hipCheck(hipMalloc(dx, source=hx))

  ! The rank_1 wrapper calls c_loc(nlen), c_loc(inembed_a), c_loc(onembed_a) internally;
  ! no c_loc here. Non-VALUE scalars must be variables (not literals).
  call hipfftCheck(hipfftPlanMany(plan, fft_rank, nlen, inembed_a, &
       istride_v, idist_v, onembed_a, ostride_v, odist_v, &
       HIPFFT_Z2Z, howmany_v))

  direction = HIPFFT_FORWARD
  call hipfftCheck(hipfftExecZ2Z(plan, dx, dx, direction))
  call hipCheck(hipDeviceSynchronize())

  call hipfftCheck(hipfftDestroy(plan))

  call hipCheck(hipMemcpy(hx, dx, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  max_error = 0.0d0
  do b = 0, howmany - 1
     kx = mod(b, Nx)
     ky = mod(b + 1, Ny)
     do i = 0, Nx - 1
        do j = 0, Ny - 1
           pos = b * odist_v + i * onembed_a(2) + j
           if (i == kx .and. j == ky) then
              expected = cmplx(dble(Ntot), 0.0d0, kind=8)
           else
              expected = (0.0d0, 0.0d0)
           end if
           error = abs(hx(pos + 1) - expected)
           if (max_error == max_error .and. .not. (error <= max_error)) max_error = error  ! keeps a NaN
        end do
     end do
  end do

  if (.not. (max_error <= tol * Ntot)) then
     write(*,*) "FAILED! max error = ", max_error
     STOP 1
  end if

  deallocate(hx)

  write(*,*) "PASSED!"

end program hipfft_planmany_2d_z2z
