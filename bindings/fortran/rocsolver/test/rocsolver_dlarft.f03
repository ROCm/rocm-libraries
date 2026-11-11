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

!!!!!!!!!!!!!!
! dlarft example (rocSOLVER)
! see: https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/auxiliary.html
!
! Forms the triangular factor T of a block Householder reflector
! H = I - V T V**T from the reflectors V and their scalar factors tau. The
! device buffers, tau included, are passed as type(c_ptr).
!!!!!!!!!!!!!!
!
program dlarft
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none

  integer(c_int), parameter :: order = 4, k = 3, ldv = 4, ldt = 3

  ! V holds the Householder vectors column-wise (unit diagonal implicit).
  real(c_double), target :: hV(order,k) = reshape([ &
      1.0d0,  0.3d0,  0.2d0,  0.1d0, &
      0.0d0,  1.0d0,  0.4d0,  0.2d0, &
      0.0d0,  0.0d0,  1.0d0,  0.5d0], [order,k])
  real(c_double), target :: htau(k) = [1.5d0, 1.2d0, 1.8d0]
  real(c_double), target :: hT(k,k) = 0.0d0
  real(c_double) :: hTref(k,k) = 0.0d0

  integer(c_size_t) :: size_V   = order*k
  integer(c_size_t) :: size_tau = k
  integer(c_size_t) :: size_T   = k*k

  type(c_ptr) :: dV, dtau, dT
  type(c_ptr) :: handle ! rocblas_handle

  integer :: i, j
  real(c_double) :: error
  real(c_double), parameter :: rtol = 1.0d-12

  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dlarft' (Fortran 2003 interfaces) - "

  call hipCheck(hipMalloc(dV,   size_V   * 8))
  call hipCheck(hipMalloc(dtau, size_tau * 8))
  call hipCheck(hipMalloc(dT,   size_T   * 8))

  call rocblasCheck(rocblas_create_handle(handle))

  call hipCheck(hipMemcpy(dV,   c_loc(hV(1,1)),  size_V   * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dtau, c_loc(htau(1)),  size_tau * 8, hipMemcpyHostToDevice))

  call rocsolverCheck(rocsolver_dlarft(handle, rocblas_forward_direction, rocblas_column_wise, &
                                       order, k, dV, ldv, dtau, dT, ldt))

  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hT(1,1)), dT, size_T * 8, hipMemcpyDeviceToHost))

  ! Forward, column-wise reference (LAPACK dlarft): T(i,i) = tau(i) and
  ! T(1:i-1,i) = -tau(i) * T(1:i-1,1:i-1) * V(:,1:i-1)**T * V(:,i).
  do i = 1, k
     hTref(i,i) = htau(i)
     if (i > 1) then
        hTref(1:i-1,i) = -htau(i) * matmul(hTref(1:i-1,1:i-1), &
                                          matmul(transpose(hV(:,1:i-1)), hV(:,i)))
     end if
  end do
  do j = 1, k
     do i = 1, j
        error = abs(hT(i,j) - hTref(i,j)) / max(abs(hTref(i,j)), 1.0d0)
        if (.not. (error <= rtol)) then
           write(*,*) "FAILED! T(", i, ",", j, ") = ", hT(i,j), " expected ", hTref(i,j)
           call exit(1)
        end if
     end do
  end do

  call hipCheck(hipFree(dV))
  call hipCheck(hipFree(dtau))
  call hipCheck(hipFree(dT))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dlarft
