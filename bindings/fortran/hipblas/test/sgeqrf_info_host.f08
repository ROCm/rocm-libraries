!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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

! Checks hipblasSgeqrf's `info`, a host integer that hipBLAS writes on the host
! (argument validation), through the array overload (X_full_rank: A(lda, n) and
! tau on the device, info a Fortran variable):
!   1. a valid call gives info = 0, and R and tau are consistent with A:
!      |R(1,1)| = ||A(:,1)||, R(1,2) = A(:,1).A(:,2) / R(1,1), tau(1) in [1, 2];
!   2. an invalid argument gives info = -(its position, counted after the
!      handle), written to the host variable: m < 0 gives -1, lda < m gives -4.
program hipblas_sgeqrf_info_host_test

    use iso_c_binding
    use hip
    use hipblas

    implicit none

    integer(c_int), parameter :: m = 6, n = 4, lda = m
    real(c_float), parameter :: tol = 1.e-4

    real(c_float), target :: hA(lda,n), hR(lda,n), htau(n)
    real(c_float), pointer :: dA(:,:) => null(), dtau(:) => null()
    integer(c_int), target :: info
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: status
    real(c_float) :: r11, r12
    type(c_ptr) :: handle
    integer :: i, j

    write(*,"(a)",advance="no") "-- Running test 'sgeqrf_info_host' (Fortran 2008 interfaces) - "

    call hipblasCheck(hipblasCreate(handle))

    do j = 1, n
        do i = 1, m
            hA(i,j) = real(mod(i*j + 3*i, 7) + 1, c_float) / real(j, c_float)
        end do
    end do
    call hipCheck(hipMalloc(dA, source=hA))
    call hipCheck(hipMalloc(dtau, n))

    ! 1. valid arguments
    info = -99
    call hipblasCheck(hipblasSgeqrf(handle, m, n, dA, lda, dtau, info))
    call hipCheck(hipDeviceSynchronize())
    if (info /= 0) then
        write(*,*) "FAILED! valid call returned info = ", info
        call exit(1)
    end if
    call hipCheck(hipMemcpy(hR, dA, hipMemcpyDeviceToHost))
    call hipCheck(hipMemcpy(htau, dtau, hipMemcpyDeviceToHost))
    r11 = norm2(hA(:,1))
    r12 = dot_product(hA(:,1), hA(:,2)) / hR(1,1)
    if (abs(abs(hR(1,1)) - r11) > tol * r11 .or. abs(hR(1,2) - r12) > tol * abs(r11) &
        .or. htau(1) < 1.0 - tol .or. htau(1) > 2.0 + tol) then
        write(*,*) "FAILED! R(1,1) = ", hR(1,1), " expected +-", r11, " R(1,2) = ", hR(1,2), &
            " expected ", r12, " tau(1) = ", htau(1)
        call exit(1)
    end if

    ! 2. invalid arguments: hipBLAS reports them in info (and in the status)
    info = 0
    status = hipblasSgeqrf(handle, -1, n, dA, lda, dtau, info)
    if (info /= -1) then
        write(*,*) "FAILED! m < 0 gave info = ", info, " status = ", status
        call exit(1)
    end if
    info = 0
    status = hipblasSgeqrf(handle, m, n, dA, m - 1, dtau, info)
    if (info /= -4) then
        write(*,*) "FAILED! lda < m gave info = ", info, " status = ", status
        call exit(1)
    end if

    call hipCheck(hipFree(dA))
    call hipCheck(hipFree(dtau))
    call hipblasCheck(hipblasDestroy(handle))
    call hipCheck(hipDeviceReset())

    write(*,*) "PASSED!"

end program hipblas_sgeqrf_info_host_test
