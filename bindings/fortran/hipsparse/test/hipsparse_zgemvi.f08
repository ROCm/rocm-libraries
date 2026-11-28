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
! hipsparse Zgemvi example (dense matrix * sparse vector, double complex)
! see: https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Computes y = alpha * A * x + beta * y, where A is a dense m-by-n matrix and x
! is a sparse vector (nnz values xVal at indices xInd). The result is checked
! against a dense host reference (alpha * matmul(A, x_dense) + beta * y).
!!!!!!!!!!!!!!
!
program hipsparse_zgemvi
  use iso_c_binding
  use hip
  use hipsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: m = 3, n = 4, lda = 3, nnz = 2
  complex(c_double_complex) :: hA(3,4) = reshape((/ &
    (1.0d0,1.0d0), (2.0d0,0.0d0), (3.0d0,1.0d0), (4.0d0,0.0d0), (5.0d0,1.0d0), (6.0d0,0.0d0), &
    (7.0d0,1.0d0), (8.0d0,0.0d0), (9.0d0,1.0d0), (10.0d0,0.0d0), (11.0d0,1.0d0), (12.0d0,0.0d0)/), (/3,4/))
  complex(c_double_complex) :: hXval(2) = (/(2.0d0,1.0d0), (3.0d0,-1.0d0)/)
  integer(c_int) :: hXind(2) = (/0, 2/)
  complex(c_double_complex) :: hY(3) = (/(1.0d0,0.0d0), (1.0d0,0.0d0), (1.0d0,0.0d0)/)
  complex(c_double_complex), target :: alpha = (2.0d0,1.0d0), beta = (3.0d0,0.0d0)
  complex(c_double_complex) :: xDense(4), hRef(3)
  type(c_ptr) :: handle = c_null_ptr
  complex(c_double_complex), pointer :: dA(:,:), dXval(:), dY(:)
  integer(c_int), pointer :: dXind(:)
  type(c_ptr) :: dBuf
  integer(c_int) :: bufSize

  write(*,"(a)",advance="no") "-- Running test 'hipsparse_zgemvi' (Fortran 2008 interfaces) - "

  ! Dense host reference.
  xDense = (0.0d0,0.0d0)
  do i = 1, nnz
     xDense(hXind(i) + 1) = hXval(i)
  end do
  hRef = alpha * matmul(hA, xDense) + beta * hY

  call hipCheck(hipMalloc(dA,    source=hA))
  call hipCheck(hipMalloc(dXval, source=hXval))
  call hipCheck(hipMalloc(dXind, source=hXind))
  call hipCheck(hipMalloc(dY,    source=hY))

  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseZgemvi_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
       m, n, nnz, bufSize))
  call hipCheck(hipMalloc(dBuf, int(max(bufSize,1),c_size_t)))
  call hipsparseCheck(hipsparseZgemvi(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, m, n, alpha, &
       dA, lda, nnz, dXval, dXind, beta, dY, HIPSPARSE_INDEX_BASE_ZERO, dBuf))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(hY, dY, hipMemcpyDeviceToHost))

  do i = 1, m
     if (.not. (abs(hY(i) - hRef(i)) <= 1.0d-10)) then
        write(*,*) "FAILED! y(", i, ") = ", hY(i), " expected ", hRef(i); call exit(1)
     end if
  end do
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dXval)); call hipCheck(hipFree(dXind))
  call hipCheck(hipFree(dY)); call hipCheck(hipFree(dBuf))
  write(*,*) "PASSED!"
end program hipsparse_zgemvi
