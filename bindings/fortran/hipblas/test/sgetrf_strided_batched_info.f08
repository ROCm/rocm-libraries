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

! Checks hipblasSgetrfStridedBatched's per-batch `info`, an array of batchCount
! integers on the device, with A(lda, n, batch) and ipiv(n, batch):
!   1. info a device integer array (integer(c_int), pointer :: dinfo(:))
!                                                  -> X_strided_rank
!   2. info a type(c_ptr) to the same device array -> X_strided_rank_devptr
! Batch entry 2 has a zero third column, so it alone must give info = 3 (U(3,3)
! is the first zero pivot); the others give info = 0 and P*A = L*U.
program hipblas_sgetrf_strided_batched_info_test

    use iso_c_binding
    use hip
    use hipblas

    implicit none

    integer(c_int), parameter :: n = 5, lda = n + 1, batch = 3, singular = 2
    integer(c_int64_t), parameter :: strideA = lda*n, strideP = n
    real(c_float), parameter :: tol = 1.e-4

    real(c_float), target :: hA(lda,n,batch), hLU(lda,n,batch)
    integer(c_int), target :: hipiv(n,batch), hinfo(batch), expected(batch)
    real(c_float), pointer :: dA(:,:,:) => null()
    integer(c_int), pointer :: dipiv(:,:) => null(), dinfo(:) => null()
    type(c_ptr) :: handle
    integer :: i, j, b, form

    write(*,"(a)",advance="no") "-- Running test 'sgetrf_strided_batched_info' (Fortran 2008 interfaces) - "

    call hipblasCheck(hipblasCreate(handle))

    do b = 1, batch
        do j = 1, n
            do i = 1, lda
                hA(i,j,b) = real(mod(3*i + 5*j*b, 11), c_float) - 4.0
            end do
            hA(j,j,b) = hA(j,j,b) + 10.0 ! keep the regular entries well conditioned
        end do
    end do
    hA(:,3,singular) = 0.0
    expected = 0
    expected(singular) = 3

    call hipCheck(hipMalloc(dA, lda, n, batch))
    call hipCheck(hipMalloc(dipiv, n, batch))
    call hipCheck(hipMalloc(dinfo, batch))

    do form = 1, 2
        call hipCheck(hipMemcpy(dA, hA, hipMemcpyHostToDevice))
        hinfo = -1
        call hipCheck(hipMemcpy(dinfo, hinfo, hipMemcpyHostToDevice))
        if (form == 1) then
            call hipblasCheck(hipblasSgetrfStridedBatched(handle, n, dA, lda, strideA, dipiv, strideP, &
                dinfo, batch))
        else
            call hipblasCheck(hipblasSgetrfStridedBatched(handle, n, dA, lda, strideA, dipiv, strideP, &
                c_loc(dinfo), batch))
        end if
        call hipCheck(hipDeviceSynchronize())
        call hipCheck(hipMemcpy(hinfo, dinfo, hipMemcpyDeviceToHost))
        call hipCheck(hipMemcpy(hLU, dA, hipMemcpyDeviceToHost))
        call hipCheck(hipMemcpy(hipiv, dipiv, hipMemcpyDeviceToHost))

        if (any(hinfo /= expected)) then
            write(*,*) "FAILED! ", merge("device array", "type(c_ptr) ", form == 1), &
                " info = ", hinfo, " expected ", expected
            call exit(1)
        end if
        do b = 1, batch
            if (b /= singular) call check_lu(b, form)
        end do
    end do

    call hipCheck(hipFree(dA))
    call hipCheck(hipFree(dipiv))
    call hipCheck(hipFree(dinfo))
    call hipblasCheck(hipblasDestroy(handle))
    call hipCheck(hipDeviceReset())

    write(*,*) "PASSED!"

contains

    ! P*A = L*U for batch entry b, with the 1-based row interchanges of ipiv
    subroutine check_lu(b, form)
        integer, intent(in) :: b, form
        real(c_float) :: PA(n,n), L(n,n), U(n,n), row(n)
        integer :: k
        PA = hA(1:n,1:n,b)
        do k = 1, n
            row = PA(k,:)
            PA(k,:) = PA(hipiv(k,b),:)
            PA(hipiv(k,b),:) = row
        end do
        L = 0.0
        U = 0.0
        do k = 1, n
            L(k,k) = 1.0
            L(k+1:n,k) = hLU(k+1:n,k,b)
            U(1:k,k) = hLU(1:k,k,b)
        end do
        if (maxval(abs(matmul(L, U) - PA)) > tol * maxval(abs(PA))) then
            write(*,*) "FAILED! ", merge("device array", "type(c_ptr) ", form == 1), &
                " P*A /= L*U for batch entry ", b
            call exit(1)
        end if
    end subroutine

end program hipblas_sgetrf_strided_batched_info_test
