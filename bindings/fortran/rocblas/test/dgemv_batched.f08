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
! dgemv_batched example (batched matrix-vector, array-of-pointers form)
! see: https://rocm.docs.amd.com/projects/rocBLAS/en/latest/
!
! Exercises the "array of device pointers" argument class for rocBLAS: A, x and
! y are each a device array of pointers, one per batch. Computes y = A*x for two
! batches (alpha=1, beta=0) and checks the result.
!!!!!!!!!!!!!!
!
program rocblas_dgemv_batched_test
  use iso_c_binding
  use hip
  use rocblas
  implicit none
  integer :: b, i
  integer(c_int), parameter :: M = 2, N = 2, lda = 2, incx = 1, incy = 1, batch = 2
  real(c_double), target :: alpha = 1.0d0, beta = 0.0d0
  ! A = [[2,0],[0,3]] (column-major), x = [1,1] -> y = [2,3]
  real(c_double), target :: hA(2,2) = reshape((/2.0d0, 0.0d0, 0.0d0, 3.0d0/), (/2,2/))
  real(c_double), target :: hx(2)   = (/1.0d0, 1.0d0/)
  real(c_double), target :: hy(2)
  real(c_double) :: hy_ref(2) = (/2.0d0, 3.0d0/)
  type(c_ptr) :: dA1=c_null_ptr, dA2=c_null_ptr, dx1=c_null_ptr, dx2=c_null_ptr, dy1=c_null_ptr, dy2=c_null_ptr
  type(c_ptr), target :: hAp(batch), hxp(batch), hyp(batch)
  type(c_ptr) :: dAp=c_null_ptr, dxp=c_null_ptr, dyp=c_null_ptr, handle=c_null_ptr
  integer(c_size_t) :: mbytes, vbytes, psize
  real(c_double) :: error
  real(c_double), parameter :: error_max = 10 * epsilon(error_max)

  write(*,"(a)",advance="no") "-- Running test 'DGEMV_BATCHED' (Fortran 2008 interfaces) - "

  mbytes = int(M,c_size_t) * int(N,c_size_t) * 8
  vbytes = int(M,c_size_t) * 8
  psize  = c_sizeof(dA1)

  call hipCheck(hipMalloc(dA1, mbytes)); call hipCheck(hipMalloc(dA2, mbytes))
  call hipCheck(hipMalloc(dx1, vbytes)); call hipCheck(hipMalloc(dx2, vbytes))
  call hipCheck(hipMalloc(dy1, vbytes)); call hipCheck(hipMalloc(dy2, vbytes))
  call hipCheck(hipMemcpy(dA1, c_loc(hA(1,1)), mbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dA2, c_loc(hA(1,1)), mbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dx1, c_loc(hx(1)),   vbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dx2, c_loc(hx(1)),   vbytes, hipMemcpyHostToDevice))

  hAp(1) = dA1; hAp(2) = dA2
  hxp(1) = dx1; hxp(2) = dx2
  hyp(1) = dy1; hyp(2) = dy2
  call hipCheck(hipMalloc(dAp, int(batch,c_size_t)*psize))
  call hipCheck(hipMemcpy(dAp, c_loc(hAp(1)), int(batch,c_size_t)*psize, hipMemcpyHostToDevice))
  call hipCheck(hipMalloc(dxp, int(batch,c_size_t)*psize))
  call hipCheck(hipMemcpy(dxp, c_loc(hxp(1)), int(batch,c_size_t)*psize, hipMemcpyHostToDevice))
  call hipCheck(hipMalloc(dyp, int(batch,c_size_t)*psize))
  call hipCheck(hipMemcpy(dyp, c_loc(hyp(1)), int(batch,c_size_t)*psize, hipMemcpyHostToDevice))

  call rocblasCheck(rocblas_create_handle(handle))
  call rocblasCheck(rocblas_dgemv_batched(handle, rocblas_operation_none, M, N, c_loc(alpha), dAp, lda, &
                                      dxp, incx, c_loc(beta), dyp, incy, batch))
  call hipCheck(hipDeviceSynchronize())

  do b = 1, batch
     hy = 0.0d0
     if (b == 1) then
        call hipCheck(hipMemcpy(c_loc(hy(1)), dy1, vbytes, hipMemcpyDeviceToHost))
     else
        call hipCheck(hipMemcpy(c_loc(hy(1)), dy2, vbytes, hipMemcpyDeviceToHost))
     end if
     do i = 1, M
        error = abs(hy(i) - hy_ref(i)) / max(abs(hy_ref(i)), 1.0d0)
        if (error > error_max) then
           write(*,*) "FAILED! batch ", b, " y(", i, ") = ", hy(i), " expected ", hy_ref(i)
           call exit(1)
        end if
     end do
  end do

  call hipCheck(hipFree(dA1)); call hipCheck(hipFree(dA2))
  call hipCheck(hipFree(dx1)); call hipCheck(hipFree(dx2))
  call hipCheck(hipFree(dy1)); call hipCheck(hipFree(dy2))
  call hipCheck(hipFree(dAp)); call hipCheck(hipFree(dxp)); call hipCheck(hipFree(dyp))
  call rocblasCheck(rocblas_destroy_handle(handle)); call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program rocblas_dgemv_batched_test
