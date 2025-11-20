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
program example_hipsparse_csrilu02
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

        function hipsparseCreateCsrilu02Info(info) &
                bind(c, name = 'hipsparseCreateCsrilu02Info')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseCreateCsrilu02Info
            type(c_ptr) :: info
        end function hipsparseCreateCsrilu02Info

        function hipsparseDestroyCsrilu02Info(info) &
                bind(c, name = 'hipsparseDestroyCsrilu02Info')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseDestroyCsrilu02Info
            type(c_ptr), value :: info
        end function hipsparseDestroyCsrilu02Info

        function hipsparseScsrilu02_bufferSize(handle, m, nnz, descr, csrSortedValA, csrSortedRowPtrA, &
                                               csrSortedColIndA, info, pBufferSizeInBytes) &
                bind(c, name = 'hipsparseScsrilu02_bufferSize')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseScsrilu02_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), value :: descr
            type(c_ptr), intent(in), value :: csrSortedValA
            type(c_ptr), intent(in), value :: csrSortedRowPtrA
            type(c_ptr), intent(in), value :: csrSortedColIndA
            type(c_ptr), value :: info
            type(c_ptr), value :: pBufferSizeInBytes
        end function hipsparseScsrilu02_bufferSize

        function hipsparseScsrilu02_analysis(handle, m, nnz, descr, csrSortedValA, csrSortedRowPtrA, &
                                             csrSortedColIndA, info, policy, pBuffer) &
                bind(c, name = 'hipsparseScsrilu02_analysis')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseScsrilu02_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), value :: descr
            type(c_ptr), intent(in), value :: csrSortedValA
            type(c_ptr), intent(in), value :: csrSortedRowPtrA
            type(c_ptr), intent(in), value :: csrSortedColIndA
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: pBuffer
        end function hipsparseScsrilu02_analysis

        function hipsparseScsrilu02(handle, m, nnz, descr, csrSortedValA_valM, csrSortedRowPtrA, &
                                    csrSortedColIndA, info, policy, pBuffer) &
                bind(c, name = 'hipsparseScsrilu02')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseScsrilu02
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), value :: descr
            type(c_ptr), value :: csrSortedValA_valM
            type(c_ptr), intent(in), value :: csrSortedRowPtrA
            type(c_ptr), intent(in), value :: csrSortedColIndA
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: pBuffer
        end function hipsparseScsrilu02

        function hipsparseXcsrilu02_zeroPivot(handle, info, position) &
                bind(c, name = 'hipsparseXcsrilu02_zeroPivot')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseXcsrilu02_zeroPivot
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function hipsparseXcsrilu02_zeroPivot
    end interface

    integer, parameter :: HIPSPARSE_SOLVE_POLICY_USE_LEVEL = 1

    ! Variables
    type(c_ptr) :: handle
    type(c_ptr) :: descr
    type(c_ptr) :: info
    integer :: i, stat
    integer, target :: zeroPivot
    integer(c_int), target :: bufferSize
    
    ! Matrix A (4x4) in CSR format
    integer, parameter :: m = 4
    integer, parameter :: n = 4
    integer, parameter :: nnz = 10
    
    integer, dimension(m+1), target :: hcsrRowPtr = (/0, 2, 5, 8, 10/)
    integer, dimension(nnz), target :: hcsrColInd = (/0, 1, 0, 1, 2, 1, 2, 3, 2, 3/)
    real(c_float), dimension(nnz), target :: hcsrVal = (/1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0/)
    
    ! Result array
    real(c_float), dimension(nnz), target :: hcsrVal_result
    
    ! Device pointers
    type(c_ptr) :: dcsrRowPtr
    type(c_ptr) :: dcsrColInd
    type(c_ptr) :: dcsrVal
    type(c_ptr) :: dbuffer

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

    ! Create csrilu02 info
    stat = hipsparseCreateCsrilu02Info(info)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseCreateCsrilu02Info failed'
        stop
    end if

    ! Allocate device memory
    stat = hipMalloc(dcsrRowPtr, int((m + 1) * 4, c_size_t))
    if (stat /= 0) stop
    stat = hipMalloc(dcsrColInd, int(nnz * 4, c_size_t))
    if (stat /= 0) stop
    stat = hipMalloc(dcsrVal, int(nnz * 4, c_size_t))
    if (stat /= 0) stop

    ! Copy data to device
    stat = hipMemcpy(dcsrRowPtr, c_loc(hcsrRowPtr), int((m + 1) * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop
    stat = hipMemcpy(dcsrColInd, c_loc(hcsrColInd), int(nnz * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop
    stat = hipMemcpy(dcsrVal, c_loc(hcsrVal), int(nnz * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop

    ! Get buffer size
    stat = hipsparseScsrilu02_bufferSize(handle, &
                                         m, &
                                         nnz, &
                                         descr, &
                                         dcsrVal, &
                                         dcsrRowPtr, &
                                         dcsrColInd, &
                                         info, &
                                         c_loc(bufferSize))
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseScsrilu02_bufferSize failed'
        stop
    end if

    ! Allocate temporary buffer
    stat = hipMalloc(dbuffer, int(bufferSize, c_size_t))
    if (stat /= 0) stop

    ! Perform analysis step
    stat = hipsparseScsrilu02_analysis(handle, &
                                       m, &
                                       nnz, &
                                       descr, &
                                       dcsrVal, &
                                       dcsrRowPtr, &
                                       dcsrColInd, &
                                       info, &
                                       HIPSPARSE_SOLVE_POLICY_USE_LEVEL, &
                                       dbuffer)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseScsrilu02_analysis failed'
        stop
    end if

    ! Perform factorization
    stat = hipsparseScsrilu02(handle, &
                              m, &
                              nnz, &
                              descr, &
                              dcsrVal, &
                              dcsrRowPtr, &
                              dcsrColInd, &
                              info, &
                              HIPSPARSE_SOLVE_POLICY_USE_LEVEL, &
                              dbuffer)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseScsrilu02 failed'
        stop
    end if

    ! Check for zero pivots
    stat = hipsparseXcsrilu02_zeroPivot(handle, info, c_loc(zeroPivot))
    if (zeroPivot /= -1) then
        write(*,*) 'Error: Zero pivot detected at row index', zeroPivot
    else
        write(*,*) 'CSRILU02 factorization completed successfully'
    end if

    ! Copy result back to host
    stat = hipMemcpy(c_loc(hcsrVal_result), dcsrVal, int(nnz * 4, c_size_t), hipMemcpyDeviceToHost)
    if (stat /= 0) stop

    ! Print result
    write(*,*) 'Factorized CSR values (L and U combined):'
    do i = 1, nnz
        write(*,*) 'val[', i-1, '] =', hcsrVal_result(i)
    end do

    ! Clean up
    stat = hipFree(dcsrRowPtr)
    stat = hipFree(dcsrColInd)
    stat = hipFree(dcsrVal)
    stat = hipFree(dbuffer)

    stat = hipsparseDestroyCsrilu02Info(info)
    stat = hipsparseDestroyMatDescr(descr)
    stat = hipsparseDestroy(handle)

end program example_hipsparse_csrilu02
! [doc example end]
