!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

program rocblas_dtrmm_test

    ! Exercises the out-of-place (14-argument) rocblas_?trmm interface:
    !
    !     C := alpha * op(A) * B      (side = left)
    !
    ! With A an m-by-m lower-triangular matrix whose referenced entries are
    ! all 1, B all 2 and alpha = 1, the exact result is C(i,j) = 2*i, which
    ! is integer-valued in double precision and therefore checkable exactly.

    use iso_c_binding
    use hip
    use rocblas

    implicit none

    integer, parameter :: m = 512, n = 512

    real(c_double) :: alpha = 1.0d0

    real(c_double), allocatable, target, dimension(:,:) :: hA, hB, hC, hC_exact
    real(c_double), pointer, dimension(:,:) :: dA, dB, dC

    type(c_ptr) :: handle = c_null_ptr

    integer :: i, j
    real(c_double) :: error
    real(c_double), parameter :: error_max = 10 * epsilon(error)

    write(*,"(a)",advance="no") "-- Running test 'dtrmm' (Fortran 2008 interfaces) - "

    call rocblasCheck(rocblas_create_handle(handle))

    allocate(hA(m,m))
    allocate(hB(m,n))
    allocate(hC(m,n))
    allocate(hC_exact(m,n))

    ! Lower-triangular A (strictly upper part is not referenced by rocBLAS)
    hA(:,:) = 0.0d0
    do j = 1, m
        do i = j, m
            hA(i,j) = 1.0d0
        end do
    end do
    hB(:,:) = 2.0d0
    hC(:,:) = -1.0d0   ! poison: must be fully overwritten by the output

    do j = 1, n
        do i = 1, m
            hC_exact(i,j) = alpha * 2.0d0 * i
        end do
    end do

    ! Allocate device memory (source= implies a blocking memcpy)
    call hipCheck(hipMalloc(dA, source=hA))
    call hipCheck(hipMalloc(dB, source=hB))
    call hipCheck(hipMalloc(dC, source=hC))

    call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host))
    call rocblasCheck(rocblas_dtrmm(handle, rocblas_side_left, rocblas_fill_lower, &
        rocblas_operation_none, rocblas_diagonal_non_unit, m, n, alpha, &
        dA, size(dA,1), dB, size(dB,1), dC, size(dC,1)))

    call hipCheck(hipDeviceSynchronize())

    ! Transfer data back to host memory
    call hipCheck(hipMemcpy(hC, dC, hipMemcpyDeviceToHost))

    ! Verification
    do j = 1, n
        do i = 1, m
            error = abs((hC_exact(i,j) - hC(i,j)) / hC_exact(i,j))
            if (error > error_max) then
                write(*,*) "FAILED! Error bigger than max! Error = ", error, &
                    " hC(", i, ",", j, ") = ", hC(i,j)
                call exit(1)
            end if
        end do
    end do

    ! Cleanup
    call hipCheck(hipFree(dA))
    call hipCheck(hipFree(dB))
    call hipCheck(hipFree(dC))
    deallocate(hA, hB, hC, hC_exact)
    call rocblasCheck(rocblas_destroy_handle(handle))

    write(*,*) "PASSED!"

end program rocblas_dtrmm_test
