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
program example_hipsparse_bsrilu02
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

        function hipsparseCreateBsrilu02Info(info) &
                bind(c, name = 'hipsparseCreateBsrilu02Info')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseCreateBsrilu02Info
            type(c_ptr) :: info
        end function hipsparseCreateBsrilu02Info

        function hipsparseDestroyBsrilu02Info(info) &
                bind(c, name = 'hipsparseDestroyBsrilu02Info')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseDestroyBsrilu02Info
            type(c_ptr), value :: info
        end function hipsparseDestroyBsrilu02Info

        function hipsparseSbsrilu02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedValA, &
                                               bsrSortedRowPtrA, bsrSortedColIndA, blockDim, &
                                               info, pBufferSizeInBytes) &
                bind(c, name = 'hipsparseSbsrilu02_bufferSize')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseSbsrilu02_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), value :: descrA
            type(c_ptr), intent(in), value :: bsrSortedValA
            type(c_ptr), intent(in), value :: bsrSortedRowPtrA
            type(c_ptr), intent(in), value :: bsrSortedColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            type(c_ptr), value :: pBufferSizeInBytes
        end function hipsparseSbsrilu02_bufferSize

        function hipsparseSbsrilu02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedValA, &
                                             bsrSortedRowPtrA, bsrSortedColIndA, blockDim, &
                                             info, policy, pBuffer) &
                bind(c, name = 'hipsparseSbsrilu02_analysis')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseSbsrilu02_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), value :: descrA
            type(c_ptr), intent(in), value :: bsrSortedValA
            type(c_ptr), intent(in), value :: bsrSortedRowPtrA
            type(c_ptr), intent(in), value :: bsrSortedColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: pBuffer
        end function hipsparseSbsrilu02_analysis

        function hipsparseSbsrilu02(handle, dirA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, &
                                    bsrSortedColIndA, blockDim, info, policy, pBuffer) &
                bind(c, name = 'hipsparseSbsrilu02')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseSbsrilu02
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), value :: descrA
            type(c_ptr), value :: bsrSortedValA
            type(c_ptr), intent(in), value :: bsrSortedRowPtrA
            type(c_ptr), intent(in), value :: bsrSortedColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: pBuffer
        end function hipsparseSbsrilu02

        function hipsparseXbsrilu02_zeroPivot(handle, info, position) &
                bind(c, name = 'hipsparseXbsrilu02_zeroPivot')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseXbsrilu02_zeroPivot
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function hipsparseXbsrilu02_zeroPivot
    end interface

    integer, parameter :: HIPSPARSE_DIRECTION_COLUMN = 1
    integer, parameter :: HIPSPARSE_SOLVE_POLICY_USE_LEVEL = 1

    ! Variables
    type(c_ptr) :: handle
    type(c_ptr) :: descr
    type(c_ptr) :: info
    integer :: i, stat
    integer, target :: zeroPivot
    integer(c_int), target :: bufferSize
    
    ! Block sparse matrix A (4x4 with block size 1)
    integer, parameter :: m = 4
    integer, parameter :: n = 4
    integer, parameter :: bs = 1
    integer, parameter :: mb = m / bs
    integer, parameter :: nb = n / bs
    integer, parameter :: nnzb = 10
    
    integer, dimension(mb+1), target :: hbsrRowPtr = (/0, 2, 5, 8, 10/)
    integer, dimension(nnzb), target :: hbsrColInd = (/0, 1, 0, 1, 2, 1, 2, 3, 2, 3/)
    real(c_float), dimension(nnzb*bs*bs), target :: hbsrVal = (/1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0/)
    
    ! Result array
    real(c_float), dimension(nnzb*bs*bs), target :: hbsrVal_result
    
    ! Device pointers
    type(c_ptr) :: dbsrRowPtr
    type(c_ptr) :: dbsrColInd
    type(c_ptr) :: dbsrVal
    type(c_ptr) :: dbuffer

    ! Create hipSPARSE handle
    stat = hipsparseCreate(handle)
    if (stat /= 0) stop

    ! Create matrix descriptor
    stat = hipsparseCreateMatDescr(descr)
    if (stat /= 0) stop

    ! Create bsrilu02 info
    stat = hipsparseCreateBsrilu02Info(info)
    if (stat /= 0) stop

    ! Allocate device memory
    stat = hipMalloc(dbsrRowPtr, int((mb + 1) * 4, c_size_t))
    if (stat /= 0) stop
    stat = hipMalloc(dbsrColInd, int(nnzb * 4, c_size_t))
    if (stat /= 0) stop
    stat = hipMalloc(dbsrVal, int(nnzb * bs * bs * 4, c_size_t))
    if (stat /= 0) stop

    ! Copy data to device
    stat = hipMemcpy(dbsrRowPtr, c_loc(hbsrRowPtr), int((mb + 1) * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop
    stat = hipMemcpy(dbsrColInd, c_loc(hbsrColInd), int(nnzb * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop
    stat = hipMemcpy(dbsrVal, c_loc(hbsrVal), int(nnzb * bs * bs * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop

    ! Get buffer size
    stat = hipsparseSbsrilu02_bufferSize(handle, &
                                         HIPSPARSE_DIRECTION_COLUMN, &
                                         mb, &
                                         nnzb, &
                                         descr, &
                                         dbsrVal, &
                                         dbsrRowPtr, &
                                         dbsrColInd, &
                                         bs, &
                                         info, &
                                         c_loc(bufferSize))
    if (stat /= 0) stop

    ! Allocate temporary buffer
    stat = hipMalloc(dbuffer, int(bufferSize, c_size_t))
    if (stat /= 0) stop

    ! Perform analysis step
    stat = hipsparseSbsrilu02_analysis(handle, &
                                       HIPSPARSE_DIRECTION_COLUMN, &
                                       mb, &
                                       nnzb, &
                                       descr, &
                                       dbsrVal, &
                                       dbsrRowPtr, &
                                       dbsrColInd, &
                                       bs, &
                                       info, &
                                       HIPSPARSE_SOLVE_POLICY_USE_LEVEL, &
                                       dbuffer)
    if (stat /= 0) stop

    ! Perform factorization
    stat = hipsparseSbsrilu02(handle, &
                              HIPSPARSE_DIRECTION_COLUMN, &
                              mb, &
                              nnzb, &
                              descr, &
                              dbsrVal, &
                              dbsrRowPtr, &
                              dbsrColInd, &
                              bs, &
                              info, &
                              HIPSPARSE_SOLVE_POLICY_USE_LEVEL, &
                              dbuffer)
    if (stat /= 0) stop

    ! Check for zero pivots
    stat = hipsparseXbsrilu02_zeroPivot(handle, info, c_loc(zeroPivot))
    if (zeroPivot /= -1) then
        write(*,*) 'Error: Zero pivot detected at row index', zeroPivot
    else
        write(*,*) 'BSRILU02 factorization completed successfully'
    end if

    ! Copy result back to host
    stat = hipMemcpy(c_loc(hbsrVal_result), dbsrVal, int(nnzb * bs * bs * 4, c_size_t), hipMemcpyDeviceToHost)
    if (stat /= 0) stop

    ! Print result
    write(*,*) 'Factorized BSR values (L and U combined):'
    do i = 1, nnzb * bs * bs
        write(*,*) 'val[', i-1, '] =', hbsrVal_result(i)
    end do

    ! Clean up
    stat = hipFree(dbsrRowPtr)
    stat = hipFree(dbsrColInd)
    stat = hipFree(dbsrVal)
    stat = hipFree(dbuffer)

    stat = hipsparseDestroyBsrilu02Info(info)
    stat = hipsparseDestroyMatDescr(descr)
    stat = hipsparseDestroy(handle)

end program example_hipsparse_bsrilu02
! [doc example end]
