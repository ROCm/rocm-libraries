!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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
    integer, parameter :: lda = m, ldb = m, ldc = m

    real(c_double), target :: alpha = 1.0d0

    real(c_double), allocatable, target, dimension(:) :: hA, hB, hC
    real(c_double), allocatable, dimension(:) :: hC_exact

    type(c_ptr) :: dA = c_null_ptr, dB = c_null_ptr, dC = c_null_ptr
    type(c_ptr) :: handle = c_null_ptr

    integer, parameter :: bytes_per_element = 8 ! double precision
    integer(c_size_t) :: NAbytes, NBbytes, NCbytes

    integer :: i, j
    real(c_double) :: error
    real(c_double), parameter :: error_max = 10 * epsilon(error)

    write(*,"(a)",advance="no") "-- Running test 'dtrmm' (Fortran 2003 interfaces) - "

    call rocblasCheck(rocblas_create_handle(handle))

    NAbytes = m * lda * bytes_per_element
    NBbytes = n * ldb * bytes_per_element
    NCbytes = n * ldc * bytes_per_element

    allocate(hA(m*lda))
    allocate(hB(n*ldb))
    allocate(hC(n*ldc))
    allocate(hC_exact(n*ldc))

    ! Lower-triangular A (strictly upper part is not referenced by rocBLAS).
    ! Column-major storage: A(i,j) lives at index (j-1)*lda + i.
    hA(:) = 0.0d0
    do j = 1, m
        do i = j, m
            hA((j-1)*lda + i) = 1.0d0
        end do
    end do
    hB(:) = 2.0d0
    hC(:) = -1.0d0   ! poison: must be fully overwritten by the output

    do j = 1, n
        do i = 1, m
            hC_exact((j-1)*ldc + i) = alpha * 2.0d0 * i
        end do
    end do

    ! Allocate device memory
    call hipCheck(hipMalloc(dA, NAbytes))
    call hipCheck(hipMalloc(dB, NBbytes))
    call hipCheck(hipMalloc(dC, NCbytes))

    ! Transfer from host to device
    call hipCheck(hipMemcpy(dA, c_loc(hA(1)), NAbytes, hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(dB, c_loc(hB(1)), NBbytes, hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(dC, c_loc(hC(1)), NCbytes, hipMemcpyHostToDevice))

    call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host))
    call rocblasCheck(rocblas_dtrmm(handle, rocblas_side_left, rocblas_fill_lower, &
        rocblas_operation_none, rocblas_diagonal_non_unit, m, n, alpha, &
        dA, lda, dB, ldb, dC, ldc))

    call hipCheck(hipDeviceSynchronize())

    ! Transfer data back to host memory
    call hipCheck(hipMemcpy(c_loc(hC(1)), dC, NCbytes, hipMemcpyDeviceToHost))

    ! Verification
    do i = 1, n*ldc
        error = abs((hC_exact(i) - hC(i)) / hC_exact(i))
        if (error > error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, &
                " hC(", i, ") = ", hC(i)
            call exit(1)
        end if
    end do

    ! Cleanup
    call hipCheck(hipFree(dA))
    call hipCheck(hipFree(dB))
    call hipCheck(hipFree(dC))
    deallocate(hA, hB, hC, hC_exact)
    call rocblasCheck(rocblas_destroy_handle(handle))

    write(*,*) "PASSED!"

end program rocblas_dtrmm_test
