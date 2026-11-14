!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights reserved.
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

program hip_dscal

  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer, parameter :: N = 10240;
  integer, parameter :: bytes_per_element = 8 !double precision
  integer(c_size_t), parameter :: Nbytes = N*bytes_per_element

  double precision, target :: alpha = 10.d0

  ! Device-resident copies of the dual-mode scalars; the library reads them
  ! from device memory because the handle is in device pointer mode.
  type(c_ptr) :: d_alpha = c_null_ptr

  type(c_ptr) :: dx = c_null_ptr

  double precision,allocatable,target,dimension(:) :: hx
  double precision,allocatable,dimension(:) ::hx_scaled

  double precision :: error
  double precision, parameter :: error_max = 10*epsilon(error_max)

  type(c_ptr) :: hip_blas_handle = c_null_ptr

  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'dscal_devptr' (Fortran 2003 interfaces) - "
  allocate(hx(N))
  allocate(hx_scaled(N))

  hx(:) = 10.d0

  hx_scaled = alpha*hx

  call hipblasCheck(hipblasCreate(hip_blas_handle))

  ! Switch to device pointer mode and stage the scalars in device memory
  call hipblasCheck(hipblasSetPointerMode(hip_blas_handle, HIPBLAS_POINTER_MODE_DEVICE))
  call hipCheck(hipMalloc(d_alpha, c_sizeof(alpha)))
  call hipCheck(hipMemcpy(d_alpha, c_loc(alpha), c_sizeof(alpha), hipMemcpyHostToDevice))

  call hipCheck(hipMalloc(dx,Nbytes))

   ! Transfer data from host to device memory
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasDscal(hip_blas_handle, N, d_alpha, dx, 1))

  call hipCheck(hipDeviceSynchronize())

  ! Transfer data back to host memory
  call hipCheck(hipMemcpy(c_loc(hx(1)), dx, Nbytes, hipMemcpyDeviceToHost))

  call hipCheck(hipFree(dx))

  ! Verification
  do i = 1,N
     error = abs(hx(i) - hx_scaled(i) )
     if( error .gt. error_max ) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " hx(i) = ", hx(i)
        call exit(1)
     endif
  end do

  call hipblasCheck(hipblasDestroy(hip_blas_handle))

  deallocate(hx_scaled)
  deallocate(hx)

  call hipCheck(hipFree(d_alpha))
  write(*,*) "PASSED!"

end program hip_dscal
