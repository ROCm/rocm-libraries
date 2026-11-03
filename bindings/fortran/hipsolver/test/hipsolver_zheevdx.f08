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
! zheevdx example (complex partial Hermitian eigensolver, Fortran 2008 interfaces)
! see: https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! heevdx computes a selected subset of eigenvalues/eigenvectors. range=all requests
! the full spectrum, so nev returns N and A is overwritten with the eigenvectors.
! Workspace via bufferSize; nev is a host integer, devInfo is device-backed.
! Self-verifying: A0*v_k = lambda_k*v_k and ||v_k|| = 1.
!!!!!!!!!!!!!!
!
program hipsolver_zheevdx
  use iso_c_binding
  use hip
  use hipsolver
  implicit none
  integer :: i, k

  integer(c_int), parameter :: N = 3, lda = 3

  ! Hermitian with complex off-diagonals, both triangles stored (A(j,i) = conj(A(i,j)))
  complex(c_double_complex) :: hA(3,3) = reshape((/ &
       (2.,0.),(-1.,-1.),(0.,1.), (-1.,1.),(2.,0.),(-1.,-1.), (0.,-1.),(-1.,1.),(2.,0.) /), (/3,3/))
  complex(c_double_complex) :: hA0(3,3)
  real(c_double) :: hW(3)
  integer(c_int) :: hNev
  integer(c_int) :: nevBuf
  complex(c_double_complex) :: lhs(3), rhs(3)

  complex(c_double_complex), pointer :: dA(:,:)
  real(c_double), pointer :: dW(:)
  integer(c_int), pointer :: dInfo(:)
  integer(c_int) :: hInfo(1)
  type(c_ptr) :: dWork, handle = c_null_ptr
  integer(c_int) :: lwork

  real(c_double) :: error
  real(c_double), parameter :: error_max = 1.0d-10

  write(*,"(a)",advance="no") "-- Running test 'hipsolver_zheevdx' (Fortran 2008 interfaces) - "

  hA0 = hA

  call hipCheck(hipMalloc(dA, source=hA))
  call hipCheck(hipMalloc(dW, mold=hW))
  call hipCheck(hipMalloc(dInfo, 1))

  call hipsolverCheck(hipsolverCreate(handle))

  call hipsolverCheck(hipsolverZheevdx_bufferSize(handle, HIPSOLVER_EIG_MODE_VECTOR, &
       HIPSOLVER_EIG_RANGE_ALL, HIPSOLVER_FILL_MODE_UPPER, N, c_loc(dA(1,1)), lda, &
       0.0d0, 0.0d0, 1, N, nevBuf, c_loc(dW(1)), lwork))
  call hipCheck(hipMalloc(dWork, max(int(lwork,c_size_t) * 16, 1_c_size_t)))

  call hipsolverCheck(hipsolverZheevdx(handle, HIPSOLVER_EIG_MODE_VECTOR, &
       HIPSOLVER_EIG_RANGE_ALL, HIPSOLVER_FILL_MODE_UPPER, N, c_loc(dA(1,1)), lda, &
       0.0d0, 0.0d0, 1, N, hNev, c_loc(dW(1)), dWork, lwork, c_loc(dInfo(1))))

  call hipCheck(hipMemcpy(hA, dA, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(hW, dW, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  if (hInfo(1) /= 0) then
     write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"
     call exit(1)
  end if

  if (hNev /= N) then
    write(*,*) "FAILED! nev = ", hNev, " expected ", N; call exit(1)
  end if

  do k = 1,N
    lhs = matmul(hA0, hA(:,k))
    rhs = hW(k) * hA(:,k)
    do i = 1,N
        error = abs(lhs(i) - rhs(i))
        if(.not. (error .le. error_max)) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " eigenpair ", k; call exit(1)
        end if
    end do
  end do

  ! Each returned eigenvector must have unit 2-norm
  do k = 1,N
    error = abs(sqrt(sum(abs(hA(:,k))**2)) - 1)
    if(.not. (error .le. error_max)) then
        write(*,*) "FAILED! eigenvector ", k, " has norm ", sqrt(sum(abs(hA(:,k))**2)), " expected 1"; call exit(1)
    end if
  end do

  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dW))
  call hipCheck(hipFree(dInfo)); call hipCheck(hipFree(dWork))
  call hipsolverCheck(hipsolverDestroy(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program hipsolver_zheevdx
