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

program hip_sgemm

  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer(kind(HIPBLAS_OP_N)), parameter :: transa = HIPBLAS_OP_N, transb = HIPBLAS_OP_N;
  real(c_float), target :: alpha = 1.1, beta = 0.9;

  integer, parameter ::  m = 1024, n = 1024, k = 1024;

  real(c_float), allocatable, dimension(:,:) :: ha, hb, hc, hc_exact

  real(c_float), pointer, dimension(:,:) :: da, db, dc
  type(c_ptr) :: handle = c_null_ptr

  integer :: i,j
  real :: error
  real, parameter :: error_max = 10*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'SGEMM' (Fortran 2008 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  allocate(ha(m,k))
  allocate(hb(k,n))
  allocate(hc(m,n))
  allocate(hc_exact(m,n))

  ! Use these constant matrices so the exact answer is also a
  ! constant matrix and therefore easy to check
  ha(:,:) = 1.0
  hb(:,:) = 2.0
  hc(:,:) = 3.0
  hc_exact = alpha*k*2.0 + beta*3.0

  ! Allocate device memory
  call hipCheck(hipMalloc(da,source=ha)) ! implies (blocking) memcpy
  call hipCheck(hipMalloc(db,source=hb))
  call hipCheck(hipMalloc(dc,source=hc))

  call hipblasCheck(hipblasSgemm(handle,transa,transb,m,n,k,alpha,da,size(da,1),db,size(db,1),beta,dc,size(dc,1)))

  call hipCheck(hipDeviceSynchronize())

  ! Transfer data back to host memory
  call hipCheck(hipMemcpy(hc, dc, hipMemcpyDeviceToHost))

  do j = 1,n
    do i = 1,m
       error = abs((hc_exact(i,j) - hc(i,j))/hc_exact(i,j))
       if( error > error_max )then
          write(*,*) "FAILED! Error bigger than max! Error = ", error
          call exit(1)
       end if
    end do
  end do

  call hipCheck(hipFree(da))
  call hipCheck(hipFree(db))
  call hipCheck(hipFree(dc))

  call hipblasCheck(hipblasDestroy(handle))

  deallocate(ha,hb,hc,hc_exact)

  write(*,*) "PASSED!"

end program hip_sgemm
