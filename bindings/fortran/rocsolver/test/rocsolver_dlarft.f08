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
! H = I - V T V**T from the reflectors V and their scalar factors tau. This test
! passes V (matrix), tau (vector) and T (matrix) as native Fortran device
! arrays; tau used to be declared as a scalar, blocking this typed-array call.
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
  real(c_double) :: hV(order,k) = reshape([ &
      1.0d0,  0.3d0,  0.2d0,  0.1d0, &
      0.0d0,  1.0d0,  0.4d0,  0.2d0, &
      0.0d0,  0.0d0,  1.0d0,  0.5d0], [order,k])
  real(c_double) :: htau(k) = [1.5d0, 1.2d0, 1.8d0]
  real(c_double) :: hT(k,k) = 0.0d0

  real(c_double), pointer :: dV(:,:)   ! GPU buffer for V
  real(c_double), pointer :: dtau(:)   ! GPU buffer for the Householder scalars
  real(c_double), pointer :: dT(:,:)   ! GPU buffer for the triangular factor T

  type(c_ptr) :: handle ! rocblas_handle

  integer :: i
  real(c_double) :: error
  real(c_double), parameter :: rtol = 1.0d-12

  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dlarft' (Fortran 2008 interfaces) - "

  call hipCheck(hipMalloc(dV,   source=hV))
  call hipCheck(hipMalloc(dtau, source=htau))
  call hipCheck(hipMalloc(dT,   source=hT))

  call rocblasCheck(rocblas_create_handle(handle))

  ! V/tau/T passed as native Fortran device arrays (resolves to _full_rank).
  call rocsolverCheck(rocsolver_dlarft(handle, rocblas_forward_direction, rocblas_column_wise, &
                                       order, k, dV, ldv, dtau, dT, ldt))

  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(hT, dT, hipMemcpyDeviceToHost))

  ! For the forward, column-wise variant T is upper triangular with T(i,i)=tau(i).
  do i = 1, k
     error = abs(hT(i,i) - htau(i))
     if (.not. (error <= rtol)) then
        write(*,*) "FAILED! T(", i, ",", i, ") = ", hT(i,i), " expected tau(", i, ") = ", htau(i)
        call exit(1)
     end if
  end do

  call hipCheck(hipFree(dV))
  call hipCheck(hipFree(dtau))
  call hipCheck(hipFree(dT))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dlarft
