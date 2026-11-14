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

! Checks the two array overloads of rocblas_dgemm_strided_batched that take
! the batch in a different shape, on the same inputs:
!   * X_full_rank:    A(lda, k*batch), B(ldb, n*batch), C(ldc, n*batch), the
!                     batch packed in the last dimension of a 2-D array;
!   * X_strided_rank: A(lda, k, batch), B(ldb, n, batch), C(ldc, n, batch),
!                     the batch as a third dimension.
! Both must give C = alpha*A*B + beta*C per batch entry, as a host reference.
! Leading dimensions larger than the matrices check that lda/ldb/ldc and the
! strides are passed through unchanged.
program rocblas_dgemm_strided_batched_ranks_test

    use iso_c_binding
    use hip
    use rocblas

    implicit none

    integer(c_int), parameter :: m = 3, n = 4, k = 5, batch = 3
    integer(c_int), parameter :: lda = m + 1, ldb = k + 2, ldc = m + 2
    integer(c_int64_t), parameter :: stride_a = lda*k, stride_b = ldb*n, stride_c = ldc*n
    real(c_double), parameter :: alpha = 1.5d0, beta = 0.5d0
    real(c_double), parameter :: error_max = 100 * epsilon(1.d0)

    real(c_double), allocatable, target :: hA3(:,:,:), hB3(:,:,:), hC3(:,:,:), ref(:,:,:)
    real(c_double), allocatable, target :: hA2(:,:), hB2(:,:), hC2(:,:)
    real(c_double), pointer :: dA2(:,:) => null(), dB2(:,:) => null(), dC2(:,:) => null()
    real(c_double), pointer :: dA3(:,:,:) => null(), dB3(:,:,:) => null(), dC3(:,:,:) => null()
    real(c_double) :: error
    type(c_ptr) :: handle
    integer :: i, j, l, b

    write(*,"(a)",advance="no") "-- Running test 'dgemm_strided_batched_ranks' (Fortran 2008 interfaces) - "

    call rocblasCheck(rocblas_create_handle(handle))
    call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host))

    allocate(hA3(lda,k,batch), hB3(ldb,n,batch), hC3(ldc,n,batch), ref(ldc,n,batch))
    do b = 1, batch
        do j = 1, k
            do i = 1, lda
                hA3(i,j,b) = dble(i + 2*j - b) / 7.d0
            end do
        end do
        do j = 1, n
            do i = 1, ldb
                hB3(i,j,b) = dble(i*j + b) / 11.d0
            end do
            do i = 1, ldc
                hC3(i,j,b) = dble(i - j + 3*b)
            end do
        end do
    end do

    ! Host reference; the padding rows of C (i > m) must stay untouched
    ref = hC3
    do b = 1, batch
        do j = 1, n
            do i = 1, m
                ref(i,j,b) = beta * hC3(i,j,b)
                do l = 1, k
                    ref(i,j,b) = ref(i,j,b) + alpha * hA3(i,l,b) * hB3(l,j,b)
                end do
            end do
        end do
    end do

    ! The same data, batch packed in the last dimension (hipfort's 2-D shape)
    hA2 = reshape(hA3, [lda, k*batch])
    hB2 = reshape(hB3, [ldb, n*batch])
    hC2 = reshape(hC3, [ldc, n*batch])

    ! 1. X_full_rank: 2-D arrays, alpha/beta as host variables
    call hipCheck(hipMalloc(dA2, source=hA2))
    call hipCheck(hipMalloc(dB2, source=hB2))
    call hipCheck(hipMalloc(dC2, source=hC2))
    call rocblasCheck(rocblas_dgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none, &
        m, n, k, alpha, dA2, lda, stride_a, dB2, ldb, stride_b, beta, dC2, ldc, stride_c, batch))
    call hipCheck(hipDeviceSynchronize())
    call hipCheck(hipMemcpy(hC2, dC2, hipMemcpyDeviceToHost))
    error = maxval(abs(reshape(hC2, [ldc, n, batch]) - ref)) / maxval(abs(ref))
    if (error > error_max) then
        write(*,*) "FAILED! full_rank (2-D packed) result differs from reference, error = ", error
        call exit(1)
    end if

    ! 2. X_strided_rank: 3-D arrays, batch as the last dimension
    call hipCheck(hipMalloc(dA3, source=hA3))
    call hipCheck(hipMalloc(dB3, source=hB3))
    call hipCheck(hipMalloc(dC3, source=hC3))
    call rocblasCheck(rocblas_dgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none, &
        m, n, k, alpha, dA3, lda, stride_a, dB3, ldb, stride_b, beta, dC3, ldc, stride_c, batch))
    call hipCheck(hipDeviceSynchronize())
    call hipCheck(hipMemcpy(hC3, dC3, hipMemcpyDeviceToHost))
    error = maxval(abs(hC3 - ref)) / maxval(abs(ref))
    if (error > error_max) then
        write(*,*) "FAILED! strided_rank (3-D) result differs from reference, error = ", error
        call exit(1)
    end if

    call hipCheck(hipFree(dA2))
    call hipCheck(hipFree(dB2))
    call hipCheck(hipFree(dC2))
    call hipCheck(hipFree(dA3))
    call hipCheck(hipFree(dB3))
    call hipCheck(hipFree(dC3))
    deallocate(hA3, hB3, hC3, ref, hA2, hB2, hC2)
    call rocblasCheck(rocblas_destroy_handle(handle))
    call hipCheck(hipDeviceReset())

    write(*,*) "PASSED!"

end program rocblas_dgemm_strided_batched_ranks_test
