!> Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
!>
!> Permission is hereby granted, free of charge, to any person obtaining a copy
!> of this software and associated documentation files (the "Software"), to deal
!> in the Software without restriction, including without limitation the rights
!> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
!> copies of the Software, and to permit persons to whom the Software is
!> furnished to do so, subject to the following conditions:
!>
!> The above copyright notice and this permission notice shall be included in
!> all copies or substantial portions of the Software.
!>
!> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
!> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
!> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
!> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
!> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
!> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
!> THE SOFTWARE.

! [doc example start]
program example_hipsparse_csrmm
    use iso_c_binding
    implicit none

    ! HIP
    interface
        function hipMalloc(ptr, size) &
                bind(c, name = 'hipMalloc')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipMalloc
            type(c_ptr) :: ptr
            integer(c_size_t), value :: size
        end function hipMalloc

        function hipFree(ptr) &
                bind(c, name = 'hipFree')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipFree
            type(c_ptr), value :: ptr
        end function hipFree

        function hipMemcpy(dst, src, size, kind) &
                bind(c, name = 'hipMemcpy')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipMemcpy
            type(c_ptr), value :: dst
            type(c_ptr), intent(in), value :: src
            integer(c_size_t), value :: size
            integer(c_int), value :: kind
        end function hipMemcpy

        function hipMemset(dst, val, size) &
                bind(c, name = 'hipMemset')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipMemset
            type(c_ptr), value :: dst
            integer(c_int), value :: val
            integer(c_size_t), value :: size
        end function hipMemset

        function hipDeviceSynchronize() &
                bind(c, name = 'hipDeviceSynchronize')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipDeviceSynchronize
        end function hipDeviceSynchronize
    end interface

    integer, parameter :: hipMemcpyHostToDevice = 1
    integer, parameter :: hipMemcpyDeviceToHost = 2

    ! hipSPARSE
    interface
        function hipsparseCreate(handle) &
                bind(c, name = 'hipsparseCreate')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseCreate
            type(c_ptr) :: handle
        end function hipsparseCreate

        function hipsparseDestroy(handle) &
                bind(c, name = 'hipsparseDestroy')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseDestroy
            type(c_ptr), value :: handle
        end function hipsparseDestroy

        function hipsparseCreateMatDescr(descr) &
                bind(c, name = 'hipsparseCreateMatDescr')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseCreateMatDescr
            type(c_ptr) :: descr
        end function hipsparseCreateMatDescr

        function hipsparseDestroyMatDescr(descr) &
                bind(c, name = 'hipsparseDestroyMatDescr')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseDestroyMatDescr
            type(c_ptr), value :: descr
        end function hipsparseDestroyMatDescr

        function hipsparseScsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrSortedValA, &
                                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc) &
                bind(c, name = 'hipsparseScsrmm')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseScsrmm
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), value :: descrA
            type(c_ptr), intent(in), value :: csrSortedValA
            type(c_ptr), intent(in), value :: csrSortedRowPtrA
            type(c_ptr), intent(in), value :: csrSortedColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseScsrmm
    end interface

    integer, parameter :: HIPSPARSE_OPERATION_NON_TRANSPOSE = 0

    ! Variables
    type(c_ptr) :: handle
    type(c_ptr) :: descr
    integer :: i, j, stat
    
    ! Matrix A (4x6) in CSR format
    !     1 2 0 3 0 0
    ! A = 0 4 5 0 0 0
    !     0 0 0 7 8 0
    !     0 0 1 2 4 1
    integer, parameter :: m = 4
    integer, parameter :: k = 6
    integer, parameter :: nnz = 11
    
    integer, dimension(m+1), target :: hcsrRowPtr = (/0, 3, 5, 7, 11/)
    integer, dimension(nnz), target :: hcsrColInd = (/0, 1, 3, 1, 2, 3, 4, 2, 3, 4, 5/)
    real(c_float), dimension(nnz), target :: hcsrVal = (/1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 1.0, 2.0, 4.0, 1.0/)
    
    ! Dense matrix B (6x3) - column major
    integer, parameter :: n = 3
    real(c_float), dimension(k*n), target :: hB = (/ &
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, &
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, &
        13.0, 14.0, 15.0, 16.0, 17.0, 18.0 /)
    
    ! Result matrix C (4x3)
    real(c_float), dimension(m*n), target :: hC
    
    ! Scalar values
    real(c_float), target :: alpha = 1.0
    real(c_float), target :: beta = 0.0
    
    ! Device pointers
    type(c_ptr) :: dcsrRowPtr
    type(c_ptr) :: dcsrColInd
    type(c_ptr) :: dcsrVal
    type(c_ptr) :: dB
    type(c_ptr) :: dC

    ! Create hipSPARSE handle
    stat = hipsparseCreate(handle)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseCreate failed'
        stop
    end if

    ! Create matrix descriptor
    stat = hipsparseCreateMatDescr(descr)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseCreateMatDescr failed'
        stop
    end if

    ! Allocate device memory for CSR matrix A
    stat = hipMalloc(dcsrRowPtr, int((m + 1) * 4, c_size_t))
    if (stat /= 0) then
        write(*,*) 'Error: hipMalloc dcsrRowPtr failed'
        stop
    end if

    stat = hipMalloc(dcsrColInd, int(nnz * 4, c_size_t))
    if (stat /= 0) then
        write(*,*) 'Error: hipMalloc dcsrColInd failed'
        stop
    end if

    stat = hipMalloc(dcsrVal, int(nnz * 4, c_size_t))
    if (stat /= 0) then
        write(*,*) 'Error: hipMalloc dcsrVal failed'
        stop
    end if

    ! Allocate device memory for dense matrices B and C
    stat = hipMalloc(dB, int(k * n * 4, c_size_t))
    if (stat /= 0) then
        write(*,*) 'Error: hipMalloc dB failed'
        stop
    end if

    stat = hipMalloc(dC, int(m * n * 4, c_size_t))
    if (stat /= 0) then
        write(*,*) 'Error: hipMalloc dC failed'
        stop
    end if

    ! Copy data to device
    stat = hipMemcpy(dcsrRowPtr, c_loc(hcsrRowPtr), int((m + 1) * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) then
        write(*,*) 'Error: hipMemcpy dcsrRowPtr failed'
        stop
    end if

    stat = hipMemcpy(dcsrColInd, c_loc(hcsrColInd), int(nnz * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) then
        write(*,*) 'Error: hipMemcpy dcsrColInd failed'
        stop
    end if

    stat = hipMemcpy(dcsrVal, c_loc(hcsrVal), int(nnz * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) then
        write(*,*) 'Error: hipMemcpy dcsrVal failed'
        stop
    end if

    stat = hipMemcpy(dB, c_loc(hB), int(k * n * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) then
        write(*,*) 'Error: hipMemcpy dB failed'
        stop
    end if

    ! Perform matrix-matrix multiplication: C = alpha * A * B + beta * C
    stat = hipsparseScsrmm(handle, &
                           HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                           m, &
                           n, &
                           k, &
                           nnz, &
                           c_loc(alpha), &
                           descr, &
                           dcsrVal, &
                           dcsrRowPtr, &
                           dcsrColInd, &
                           dB, &
                           k, &
                           c_loc(beta), &
                           dC, &
                           m)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseScsrmm failed'
        stop
    end if

    ! Copy result back to host
    stat = hipMemcpy(c_loc(hC), dC, int(m * n * 4, c_size_t), hipMemcpyDeviceToHost)
    if (stat /= 0) then
        write(*,*) 'Error: hipMemcpy hC failed'
        stop
    end if

    ! Print result
    write(*,*) 'hC:'
    do i = 1, m * n
        write(*,*) hC(i)
    end do

    ! Clean up
    stat = hipFree(dcsrRowPtr)
    stat = hipFree(dcsrColInd)
    stat = hipFree(dcsrVal)
    stat = hipFree(dB)
    stat = hipFree(dC)

    stat = hipsparseDestroyMatDescr(descr)
    stat = hipsparseDestroy(handle)

end program example_hipsparse_csrmm
! [doc example end]
