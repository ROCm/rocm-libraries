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

program hip_zgemm
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer(kind(HIPBLAS_OP_N)), parameter :: transa = HIPBLAS_OP_N, transb = HIPBLAS_OP_N;
  complex(kind=8), allocatable, target :: alpha, beta

  integer, parameter :: m = 512, n = 512, k = 512;

  complex(kind=8), allocatable, dimension(:,:) :: ha, hb, hc
  complex(kind=8), allocatable, dimension(:,:) :: hc_exact

  complex(kind=8), pointer, dimension(:,:) :: da, db, dc
  type(c_ptr) :: handle = c_null_ptr

  integer :: i,j
  double precision :: error
  double precision, parameter :: error_max = 10*epsilon(error)
  ! On the heap, so 16-byte aligned: the CUDA backend accesses a
  ! cuDoubleComplex scalar with an aligned load or store.
  allocate(alpha, beta)
  alpha = 1.1d0
  beta = 0.9d0;

  write(*,"(a)",advance="no") "-- Running test 'ZGEMM' (Fortran 2008 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  ! C = A_MxK * B_KxN

  allocate(ha(m,k))
  allocate(hb(k,n))
  allocate(hc(m,n))
  allocate(hc_exact(m,n))

  ! Use these constant matrices so the exact answer is also a
  ! constant matrix and therefore easy to check
  ha(:,:) = 1.d0
  hb(:,:) = 2.d0
  hc(:,:) = 3.d0
  hc_exact(:,:) = alpha*k*2.d0 + beta*3.d0

  ! Allocate device memory
  call hipCheck(hipMalloc(da,mold=ha))
  call hipCheck(hipMalloc(db,mold=hb))
  call hipCheck(hipMalloc(dc,mold=hc))

  !Transfer from host to device
  call hipCheck(hipMemcpy(da, ha, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(db, hb, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dc, hc, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasZgemm(handle,transa,transb,m,n,k,alpha,da,size(da,1),db,size(db,1),beta,dc,size(dc,1)))

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

  deallocate(ha,hb,hc,hc_exact, alpha, beta)

  write(*,*) "PASSED!"

end program hip_zgemm
