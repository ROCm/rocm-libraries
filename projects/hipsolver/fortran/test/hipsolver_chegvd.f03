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
! chegvd example (complex generalized Hermitian-definite eigensolver, Fortran 2003)
! see: https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! hegvd solves A*x = lambda*B*x (itype=1). f2003 style: device buffers are
! type(c_ptr). Self-verifying: A0*v_k = lambda_k * B0*v_k.
!!!!!!!!!!!!!!
!
program hipsolver_chegvd
  use iso_c_binding
  use hip
  use hipsolver
  implicit none
  integer(c_int), target :: hInfo
  integer :: i, k

  integer(c_int), parameter :: N = 3, lda = 3, ldb = 3

  complex(c_float_complex), target :: hA(3,3) = reshape((/ &
       (2.,0.),(-1.,0.),(0.,0.), (-1.,0.),(2.,0.),(-1.,0.), (0.,0.),(-1.,0.),(2.,0.) /), (/3,3/))
  complex(c_float_complex), target :: hB(3,3) = reshape((/ &
       (2.,0.),(0.,0.),(0.,0.), (0.,0.),(2.,0.),(0.,0.), (0.,0.),(0.,0.),(2.,0.) /), (/3,3/))
  complex(c_float_complex) :: hA0(3,3), hB0(3,3)
  real(c_float), target :: hW(3)
  complex(c_float_complex) :: lhs(3), rhs(3)

  integer(c_size_t) :: sizeA = 9, sizeB = 9, sizeW = 3

  type(c_ptr) :: dA, dB, dW
  type(c_ptr) :: dInfo
  type(c_ptr) :: dWork, handle = c_null_ptr
  integer(c_int) :: lwork

  real(c_float) :: error
  real(c_float), parameter :: error_max = 1.0e-5

  write(*,"(a)",advance="no") "-- Running test 'hipsolver_chegvd' (Fortran 2003 interfaces) - "

  hA0 = hA
  hB0 = hB

  call hipCheck(hipMalloc(dA, sizeA * 8))
  call hipCheck(hipMalloc(dB, sizeB * 8))
  call hipCheck(hipMalloc(dW, sizeW * 4))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), sizeA * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), sizeB * 8, hipMemcpyHostToDevice))

  call hipsolverCheck(hipsolverCreate(handle))

  call hipsolverCheck(hipsolverChegvd_bufferSize(handle, HIPSOLVER_EIG_TYPE_1, &
       HIPSOLVER_EIG_MODE_VECTOR, HIPSOLVER_FILL_MODE_UPPER, N, dA, lda, dB, ldb, dW, lwork))
  call hipCheck(hipMalloc(dWork, max(int(lwork,c_size_t) * 8, 1_c_size_t)))

  hInfo = -1
  call hipCheck(hipMemcpy(dInfo, c_loc(hInfo), 4_c_size_t, hipMemcpyHostToDevice))
  call hipsolverCheck(hipsolverChegvd(handle, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_VECTOR, &
       HIPSOLVER_FILL_MODE_UPPER, N, dA, lda, dB, ldb, dW, dWork, lwork, dInfo))
  call hipCheck(hipMemcpy(c_loc(hInfo), dInfo, 4_c_size_t, hipMemcpyDeviceToHost))
  if (hInfo /= 0) then
    write(*,*) "FAILED! info = ", hInfo, " (expected 0)"; call exit(1)
  end if

  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, sizeA * 8, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hW(1)), dW, sizeW * 4, hipMemcpyDeviceToHost))

  do k = 1,N
    lhs = matmul(hA0, hA(:,k))
    rhs = hW(k) * matmul(hB0, hA(:,k))
    error = abs(abs(dot_product(hA(:,k), matmul(hB0, hA(:,k)))) - 1)
    if(.not. (error .le. error_max)) then
        write(*,*) "FAILED! eigenvector ", k, " B-norm error = ", error; call exit(1)
    end if
    do i = 1,N
        error = abs(lhs(i) - rhs(i))
        if(.not. (error .le. error_max)) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " eigenpair ", k; call exit(1)
        end if
    end do
  end do

  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dB)); call hipCheck(hipFree(dW))
  call hipCheck(hipFree(dInfo)); call hipCheck(hipFree(dWork))
  call hipsolverCheck(hipsolverDestroy(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program hipsolver_chegvd
