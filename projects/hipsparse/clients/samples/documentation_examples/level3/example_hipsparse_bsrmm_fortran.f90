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
program example_hipsparse_bsrmm
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

        function hipsparseSbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, &
                                 bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, &
                                 B, ldb, beta, C, ldc) &
                bind(c, name = 'hipsparseSbsrmm')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseSbsrmm
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: mb
            integer(c_int), value :: n
            integer(c_int), value :: kb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), value :: descrA
            type(c_ptr), intent(in), value :: bsrSortedValA
            type(c_ptr), intent(in), value :: bsrSortedRowPtrA
            type(c_ptr), intent(in), value :: bsrSortedColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseSbsrmm
    end interface

    integer, parameter :: HIPSPARSE_DIRECTION_ROW = 0
    integer, parameter :: HIPSPARSE_OPERATION_NON_TRANSPOSE = 0

    ! Variables
    type(c_ptr) :: handle
    type(c_ptr) :: descr
    integer :: i, stat
    
    ! Block sparse matrix A (2x3 blocks of size 2x2)
    integer, parameter :: blockDim = 2
    integer, parameter :: mb = 2
    integer, parameter :: kb = 3
    integer, parameter :: nnzb = 4
    integer, parameter :: m = mb * blockDim  ! 4
    integer, parameter :: k = kb * blockDim  ! 6
    
    integer, dimension(mb+1), target :: hbsrRowPtr = (/0, 2, 4/)
    integer, dimension(nnzb), target :: hbsrColInd = (/0, 1, 1, 2/)
    real(c_float), dimension(nnzb*blockDim*blockDim), target :: hbsrVal = (/ &
        1.0, 2.0, 0.0, 4.0, &  ! block (0,0)
        0.0, 3.0, 5.0, 0.0, &  ! block (0,1)
        0.0, 7.0, 1.0, 2.0, &  ! block (1,1)
        8.0, 0.0, 4.0, 1.0 /)  ! block (1,2)
    
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
    type(c_ptr) :: dbsrRowPtr
    type(c_ptr) :: dbsrColInd
    type(c_ptr) :: dbsrVal
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

    ! Allocate device memory for BSR matrix A
    stat = hipMalloc(dbsrRowPtr, int((mb + 1) * 4, c_size_t))
    if (stat /= 0) then
        write(*,*) 'Error: hipMalloc dbsrRowPtr failed'
        stop
    end if

    stat = hipMalloc(dbsrColInd, int(nnzb * 4, c_size_t))
    if (stat /= 0) then
        write(*,*) 'Error: hipMalloc dbsrColInd failed'
        stop
    end if

    stat = hipMalloc(dbsrVal, int(nnzb * blockDim * blockDim * 4, c_size_t))
    if (stat /= 0) then
        write(*,*) 'Error: hipMalloc dbsrVal failed'
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
    stat = hipMemcpy(dbsrRowPtr, c_loc(hbsrRowPtr), int((mb + 1) * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) then
        write(*,*) 'Error: hipMemcpy dbsrRowPtr failed'
        stop
    end if

    stat = hipMemcpy(dbsrColInd, c_loc(hbsrColInd), int(nnzb * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) then
        write(*,*) 'Error: hipMemcpy dbsrColInd failed'
        stop
    end if

    stat = hipMemcpy(dbsrVal, c_loc(hbsrVal), int(nnzb * blockDim * blockDim * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) then
        write(*,*) 'Error: hipMemcpy dbsrVal failed'
        stop
    end if

    stat = hipMemcpy(dB, c_loc(hB), int(k * n * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) then
        write(*,*) 'Error: hipMemcpy dB failed'
        stop
    end if

    ! Perform block sparse matrix-matrix multiplication: C = alpha * A * B + beta * C
    stat = hipsparseSbsrmm(handle, &
                           HIPSPARSE_DIRECTION_ROW, &
                           HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                           HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                           mb, &
                           n, &
                           kb, &
                           nnzb, &
                           c_loc(alpha), &
                           descr, &
                           dbsrVal, &
                           dbsrRowPtr, &
                           dbsrColInd, &
                           blockDim, &
                           dB, &
                           k, &
                           c_loc(beta), &
                           dC, &
                           m)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseSbsrmm failed'
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
    stat = hipFree(dbsrRowPtr)
    stat = hipFree(dbsrColInd)
    stat = hipFree(dbsrVal)
    stat = hipFree(dB)
    stat = hipFree(dC)

    stat = hipsparseDestroyMatDescr(descr)
    stat = hipsparseDestroy(handle)

end program example_hipsparse_bsrmm
! [doc example end]
