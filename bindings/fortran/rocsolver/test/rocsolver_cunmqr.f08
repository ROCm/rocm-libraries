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

!!!!!!!!!!!!!/
! cunmqr example (single-complex multiplication by Q from a QR factorization)
! see: https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! Self-verifying: factorize A with geqrf to obtain Q (as Householder vectors),
! then apply Q and Q**H successively to a matrix C. Since Q**H * Q = I, the
! result must recover the original C, so no reference matrix is needed.
!!!!!!!!!!!!!!/
!
program cunmqr
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i, j ! indices for iterating over results

  integer(c_int), parameter :: M = 3
  integer(c_int), parameter :: N = 2
  integer(c_int), parameter :: K = 3
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: ldc = 3

  ! Matrix to factorize (column-major) and a separate C to multiply
  complex(c_float_complex) :: hA(3,3) = reshape((/ &
    (1.0,1.0), (4.0,0.0), (7.0,0.0), &
    (2.0,0.0), (5.0,1.0), (8.0,0.0), &
    (3.0,0.0), (6.0,0.0), (10.0,1.0)/), (/3, 3/))
  complex(c_float_complex) :: hC(3,2) = reshape((/ &
    (1.0,1.0), (2.0,0.0), (3.0,-1.0), (4.0,0.0), (5.0,2.0), (6.0,0.0)/), (/3, 2/))
  complex(c_float_complex) :: hC0(3,2)   ! original C kept for verification
  complex(c_float_complex) :: hIpiv(3)

  complex(c_float_complex), pointer :: dA(:,:)
  complex(c_float_complex), pointer :: dC(:,:)
  complex(c_float_complex), pointer :: dIpiv(:)

  type(c_ptr) :: handle ! rocblas_handle

  real(c_float) :: error
  real(c_float), parameter :: error_max = 1.0e-5
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_cunmqr' (Fortran 2008 interfaces) - "

  hC0 = hC ! keep original C for the round-trip check

  ! Allocate device-side memory & copy memory from host to device
  call hipCheck(hipMalloc(dA,    source=hA))
  call hipCheck(hipMalloc(dC,    source=hC))
  call hipCheck(hipMalloc(dIpiv, mold=hIpiv))

  ! Create rocBLAS handle
  call rocblasCheck(rocblas_create_handle(handle))

  ! Factorize to get Q, then apply C <- Q*C followed by C <- Q**H*C
  call rocsolverCheck(rocsolver_cgeqrf(handle, M, K, dA, lda, dIpiv))
  call rocsolverCheck(rocsolver_cunmqr(handle, rocblas_side_left, rocblas_operation_none, &
       M, N, K, dA, lda, dIpiv, dC, ldc))
  call rocsolverCheck(rocsolver_cunmqr(handle, rocblas_side_left, rocblas_operation_conjugate_transpose, &
       M, N, K, dA, lda, dIpiv, dC, ldc))

  ! Copy result back to host
  call hipCheck(hipMemcpy(hC, dC, hipMemcpyDeviceToHost))

  ! Verify Q**H * Q * C = C
  do j = 1,N
    do i = 1,M
      error = abs(hC(i,j) - hC0(i,j))
      if(.not. (error .le. error_max)) then
          write(*,*) "FAILED! Round trip mismatch! Error = ", error, " (", i, ",", j, ")"
          call exit(1)
      end if
    end do
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dC))
  call hipCheck(hipFree(dIpiv))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program cunmqr
