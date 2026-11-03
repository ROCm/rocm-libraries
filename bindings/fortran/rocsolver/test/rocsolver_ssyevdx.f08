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
! ssyevdx example (partial symmetric eigensolver)
! see: https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! syevdx computes a selected subset of eigenvalues/eigenvectors. erange=index with
! il=1, iu=N requests the full spectrum, so nev returns N; eigenvectors go to Z.
! Self-verifying: A0*v_k = lambda_k*v_k. info and nev live on the DEVICE.
!!!!!!!!!!!!!!/
!
program ssyevdx
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i, k

  integer(c_int), parameter :: N = 3, lda = 3, ldz = 3

  real(c_float) :: hA(3,3) = reshape((/2, -1, 0, -1, 2, -1, 0, -1, 2/), (/3, 3/))
  real(c_float) :: hA0(3,3)
  real(c_float) :: hW(3)
  real(c_float) :: hZ(3,3)
  integer(c_int) :: hNev
  real(c_float) :: lhs(3), rhs(3)

  real(c_float), pointer :: dA(:,:), dZ(:,:)
  real(c_float), pointer :: dW(:)
  integer(c_int), pointer :: dNev(:), dInfo(:)
  integer(c_int) :: hInfo(1)

  type(c_ptr) :: handle

  real(c_float) :: error
  real(c_float), parameter :: error_max = 1.0e-5
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_ssyevdx' (Fortran 2008 interfaces) - "

  hA0 = hA

  call hipCheck(hipMalloc(dA, source=hA))
  call hipCheck(hipMalloc(dW, mold=hW))
  call hipCheck(hipMalloc(dZ, mold=hZ))
  call hipCheck(hipMalloc(dNev, 1))
  call hipCheck(hipMalloc(dInfo, 1))

  call rocblasCheck(rocblas_create_handle(handle))

  call rocsolverCheck(rocsolver_ssyevdx(handle, rocblas_evect_original, rocblas_erange_index, &
       rocblas_fill_upper, N, c_loc(dA(1,1)), lda, 0.0, 0.0, 1, N, c_loc(dNev(1)), &
       c_loc(dW(1)), c_loc(dZ(1,1)), ldz, c_loc(dInfo(1))))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  if (hInfo(1) /= 0) then
    write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"; call exit(1)
  end if

  call hipCheck(hipMemcpy(hNev, dNev(1), hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(hW, dW, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(hZ, dZ, hipMemcpyDeviceToHost))

  if (hNev /= N) then
    write(*,*) "FAILED! nev = ", hNev, " expected ", N; call exit(1)
  end if

  do k = 1,N
    lhs = matmul(hA0, hZ(:,k))
    rhs = hW(k) * hZ(:,k)
    do i = 1,N
        error = abs(lhs(i) - rhs(i))
        if(.not. (error .le. error_max)) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " eigenpair ", k; call exit(1)
        end if
    end do
    ! Eigenvectors are normalised: ||v_k|| = 1
    error = abs(sqrt(sum(abs(hZ(:,k))**2)) - 1)
    if(.not. (error .le. error_max)) then
        write(*,*) "FAILED! ||v|| - 1 = ", error, " eigenvector ", k; call exit(1)
    end if
  end do

  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dW)); call hipCheck(hipFree(dZ))
  call hipCheck(hipFree(dNev)); call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program ssyevdx
