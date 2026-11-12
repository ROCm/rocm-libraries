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
! dgetrf_batched example (batched LU, array-of-pointers form)
! Exercises the "array of device pointers" argument class: A is a device array
! of per-batch matrix pointers (not the strided form). The input is diagonally
! dominant (no pivoting), so each batch's L*U == A is checked by reconstruction,
! plus info == 0 per batch. rocSOLVER writes info to DEVICE memory.
!!!!!!!!!!!!!!
!
program dgetrf_batched
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver
  implicit none
  integer :: i, j, k, b
  integer(c_int), parameter :: M = 3, N = 3, lda = 3, batch = 2
  real(c_double), target :: hA(3,3)  = reshape((/4, 1, 1,  1, 4, 1,  1, 1, 4/), (/3,3/))
  real(c_double), target :: hLU(3,3)
  integer(c_int) :: hInfo(batch)
  integer(c_int), target :: hInfo_t(batch)
  type(c_ptr) :: d1 = c_null_ptr, d2 = c_null_ptr
  type(c_ptr), target :: hptrs(batch)
  type(c_ptr) :: dA_ptrs = c_null_ptr, dIpiv = c_null_ptr, dInfo = c_null_ptr, handle = c_null_ptr
  integer(c_size_t) :: matbytes, psize
  real(c_double) :: lij, recon, error
  real(c_double), parameter :: error_max = 1.0d-10

  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dgetrf_batched' (Fortran 2008 interfaces) - "

  matbytes = int(M,c_size_t) * int(N,c_size_t) * 8
  psize    = c_sizeof(d1)

  call hipCheck(hipMalloc(d1, matbytes))
  call hipCheck(hipMalloc(d2, matbytes))
  call hipCheck(hipMemcpy(d1, c_loc(hA(1,1)), matbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d2, c_loc(hA(1,1)), matbytes, hipMemcpyHostToDevice))

  hptrs(1) = d1
  hptrs(2) = d2
  call hipCheck(hipMalloc(dA_ptrs, int(batch,c_size_t) * psize))
  call hipCheck(hipMemcpy(dA_ptrs, c_loc(hptrs(1)), int(batch,c_size_t) * psize, hipMemcpyHostToDevice))

  call hipCheck(hipMalloc(dIpiv, int(N,c_size_t) * int(batch,c_size_t) * 4))
  call hipCheck(hipMalloc(dInfo, int(batch,c_size_t) * 4))

  call rocblasCheck(rocblas_create_handle(handle))
  call rocsolverCheck(rocsolver_dgetrf_batched(handle, M, N, dA_ptrs, lda, dIpiv, int(N,c_int64_t), dInfo, batch))

  call hipCheck(hipMemcpy(c_loc(hInfo_t(1)), dInfo, int(batch,c_size_t)*4, hipMemcpyDeviceToHost))
  hInfo = hInfo_t
  do b = 1, batch
     if (hInfo(b) /= 0) then
        write(*,*) "FAILED! info(", b, ") = ", hInfo(b)
        call exit(1)
     end if
  end do

  ! Reconstruct L*U == A for each batch matrix.
  do b = 1, batch
     if (b == 1) then
        call hipCheck(hipMemcpy(c_loc(hLU(1,1)), d1, matbytes, hipMemcpyDeviceToHost))
     else
        call hipCheck(hipMemcpy(c_loc(hLU(1,1)), d2, matbytes, hipMemcpyDeviceToHost))
     end if
     do j = 1, N
        do i = 1, M
           recon = 0.0d0
           do k = 1, min(i,j)
              if (k == i) then; lij = 1.0d0; else; lij = hLU(i,k); end if
              recon = recon + lij * hLU(k,j)
           end do
           error = abs(recon - hA(i,j))
           if (.not. (error <= error_max)) then
              write(*,*) "FAILED! batch ", b, " (L*U)(", i, ",", j, ") = ", recon
              call exit(1)
           end if
        end do
     end do
  end do

  call hipCheck(hipFree(d1)); call hipCheck(hipFree(d2)); call hipCheck(hipFree(dA_ptrs))
  call hipCheck(hipFree(dIpiv)); call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle)); call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"
end program dgetrf_batched
