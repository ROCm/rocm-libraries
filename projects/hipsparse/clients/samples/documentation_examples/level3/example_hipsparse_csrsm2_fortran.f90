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
program example_hipsparse_csrsm2
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

        function hipsparseSetMatFillMode(descr, fillMode) &
                bind(c, name = 'hipsparseSetMatFillMode')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseSetMatFillMode
            type(c_ptr), value :: descr
            integer(c_int), value :: fillMode
        end function hipsparseSetMatFillMode

        function hipsparseSetMatDiagType(descr, diagType) &
                bind(c, name = 'hipsparseSetMatDiagType')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseSetMatDiagType
            type(c_ptr), value :: descr
            integer(c_int), value :: diagType
        end function hipsparseSetMatDiagType

        function hipsparseCreateCsrsm2Info(info) &
                bind(c, name = 'hipsparseCreateCsrsm2Info')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseCreateCsrsm2Info
            type(c_ptr) :: info
        end function hipsparseCreateCsrsm2Info

        function hipsparseDestroyCsrsm2Info(info) &
                bind(c, name = 'hipsparseDestroyCsrsm2Info')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseDestroyCsrsm2Info
            type(c_ptr), value :: info
        end function hipsparseDestroyCsrsm2Info

        function hipsparseDcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, &
                                                alpha, descr, csrSortedValA, csrSortedRowPtrA, &
                                                csrSortedColIndA, B, ldb, info, policy, pBufferSize) &
                bind(c, name = 'hipsparseDcsrsm2_bufferSizeExt')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseDcsrsm2_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), value :: descr
            type(c_ptr), intent(in), value :: csrSortedValA
            type(c_ptr), intent(in), value :: csrSortedRowPtrA
            type(c_ptr), intent(in), value :: csrSortedColIndA
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: pBufferSize
        end function hipsparseDcsrsm2_bufferSizeExt

        function hipsparseDcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, &
                                           descr, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, &
                                           B, ldb, info, policy, pBuffer) &
                bind(c, name = 'hipsparseDcsrsm2_analysis')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseDcsrsm2_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), value :: descr
            type(c_ptr), intent(in), value :: csrSortedValA
            type(c_ptr), intent(in), value :: csrSortedRowPtrA
            type(c_ptr), intent(in), value :: csrSortedColIndA
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: pBuffer
        end function hipsparseDcsrsm2_analysis

        function hipsparseDcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descr, &
                                        csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, &
                                        info, policy, pBuffer) &
                bind(c, name = 'hipsparseDcsrsm2_solve')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseDcsrsm2_solve
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), value :: descr
            type(c_ptr), intent(in), value :: csrSortedValA
            type(c_ptr), intent(in), value :: csrSortedRowPtrA
            type(c_ptr), intent(in), value :: csrSortedColIndA
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: pBuffer
        end function hipsparseDcsrsm2_solve

        function hipsparseXcsrsm2_zeroPivot(handle, info, position) &
                bind(c, name = 'hipsparseXcsrsm2_zeroPivot')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseXcsrsm2_zeroPivot
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function hipsparseXcsrsm2_zeroPivot
    end interface

    integer, parameter :: HIPSPARSE_OPERATION_NON_TRANSPOSE = 0
    integer, parameter :: HIPSPARSE_FILL_MODE_LOWER = 0
    integer, parameter :: HIPSPARSE_DIAG_TYPE_NON_UNIT = 0
    integer, parameter :: HIPSPARSE_SOLVE_POLICY_NO_LEVEL = 0
    integer, parameter :: HIPSPARSE_STATUS_ZERO_PIVOT = 6

    ! Variables
    type(c_ptr) :: handle
    type(c_ptr) :: descr
    type(c_ptr) :: info
    integer :: i, stat
    integer, target :: pivot
    integer(c_size_t), target :: buffer_size
    
    ! Lower triangular matrix A (4x4) in CSR format
    ! A = ( 1.0  0.0  0.0  0.0 )
    !     ( 2.0  3.0  0.0  0.0 )
    !     ( 4.0  5.0  6.0  0.0 )
    !     ( 7.0  0.0  8.0  9.0 )
    integer, parameter :: m = 4
    integer, parameter :: n = 4
    integer, parameter :: nrhs = 4
    integer, parameter :: nnz = 9
    
    integer, dimension(m+1), target :: hcsrRowPtr = (/0, 1, 3, 6, 9/)
    integer, dimension(nnz), target :: hcsrColInd = (/0, 0, 1, 0, 1, 2, 0, 2, 3/)
    real(c_double), dimension(nnz), target :: hcsrVal = (/1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0, 6.0d0, 7.0d0, 8.0d0, 9.0d0/)
    
    ! Right-hand side matrix B (4x4) - column major
    integer, parameter :: ldb = n
    real(c_double), dimension(n*nrhs), target :: hB = (/ &
        1.0d0, 2.0d0, 3.0d0, 4.0d0, &
        5.0d0, 6.0d0, 7.0d0, 8.0d0, &
        9.0d0, 10.0d0, 11.0d0, 12.0d0, &
        13.0d0, 14.0d0, 15.0d0, 16.0d0 /)
    
    ! Scalar values
    real(c_double), target :: alpha = 1.0d0
    
    ! Device pointers
    type(c_ptr) :: dcsrRowPtr
    type(c_ptr) :: dcsrColInd
    type(c_ptr) :: dcsrVal
    type(c_ptr) :: dB
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

    ! Set matrix fill mode (lower triangular)
    stat = hipsparseSetMatFillMode(descr, HIPSPARSE_FILL_MODE_LOWER)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseSetMatFillMode failed'
        stop
    end if

    ! Set matrix diagonal type
    stat = hipsparseSetMatDiagType(descr, HIPSPARSE_DIAG_TYPE_NON_UNIT)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseSetMatDiagType failed'
        stop
    end if

    ! Create csrsm2 info
    stat = hipsparseCreateCsrsm2Info(info)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseCreateCsrsm2Info failed'
        stop
    end if

    ! Allocate device memory
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

    stat = hipMalloc(dcsrVal, int(nnz * 8, c_size_t))
    if (stat /= 0) then
        write(*,*) 'Error: hipMalloc dcsrVal failed'
        stop
    end if

    stat = hipMalloc(dB, int(n * nrhs * 8, c_size_t))
    if (stat /= 0) then
        write(*,*) 'Error: hipMalloc dB failed'
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

    stat = hipMemcpy(dcsrVal, c_loc(hcsrVal), int(nnz * 8, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) then
        write(*,*) 'Error: hipMemcpy dcsrVal failed'
        stop
    end if

    stat = hipMemcpy(dB, c_loc(hB), int(n * nrhs * 8, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) then
        write(*,*) 'Error: hipMemcpy dB failed'
        stop
    end if

    ! Obtain required buffer size
    stat = hipsparseDcsrsm2_bufferSizeExt(handle, &
                                          0, &
                                          HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                                          HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                                          m, &
                                          nrhs, &
                                          nnz, &
                                          c_loc(alpha), &
                                          descr, &
                                          dcsrVal, &
                                          dcsrRowPtr, &
                                          dcsrColInd, &
                                          dB, &
                                          ldb, &
                                          info, &
                                          HIPSPARSE_SOLVE_POLICY_NO_LEVEL, &
                                          c_loc(buffer_size))
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseDcsrsm2_bufferSizeExt failed'
        stop
    end if

    ! Allocate temporary buffer
    stat = hipMalloc(dbuffer, buffer_size)
    if (stat /= 0) then
        write(*,*) 'Error: hipMalloc dbuffer failed'
        stop
    end if

    ! Perform analysis step
    stat = hipsparseDcsrsm2_analysis(handle, &
                                     0, &
                                     HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                                     HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                                     m, &
                                     nrhs, &
                                     nnz, &
                                     c_loc(alpha), &
                                     descr, &
                                     dcsrVal, &
                                     dcsrRowPtr, &
                                     dcsrColInd, &
                                     dB, &
                                     ldb, &
                                     info, &
                                     HIPSPARSE_SOLVE_POLICY_NO_LEVEL, &
                                     dbuffer)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseDcsrsm2_analysis failed'
        stop
    end if

    ! Perform solve: solve Ax = B for x
    stat = hipsparseDcsrsm2_solve(handle, &
                                  0, &
                                  HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                                  HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                                  m, &
                                  nrhs, &
                                  nnz, &
                                  c_loc(alpha), &
                                  descr, &
                                  dcsrVal, &
                                  dcsrRowPtr, &
                                  dcsrColInd, &
                                  dB, &
                                  ldb, &
                                  info, &
                                  HIPSPARSE_SOLVE_POLICY_NO_LEVEL, &
                                  dbuffer)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseDcsrsm2_solve failed'
        stop
    end if

    ! Check for zero pivots
    stat = hipsparseXcsrsm2_zeroPivot(handle, info, c_loc(pivot))
    if (stat == HIPSPARSE_STATUS_ZERO_PIVOT) then
        write(*,*) 'Found zero pivot in matrix row ', pivot
    end if

    ! Copy result back to host
    stat = hipMemcpy(c_loc(hB), dB, int(m * nrhs * 8, c_size_t), hipMemcpyDeviceToHost)
    if (stat /= 0) then
        write(*,*) 'Error: hipMemcpy hB failed'
        stop
    end if

    ! Print result
    write(*,*) 'Solution:'
    do i = 1, m * nrhs
        write(*,*) hB(i)
    end do

    ! Clean up
    stat = hipFree(dcsrRowPtr)
    stat = hipFree(dcsrColInd)
    stat = hipFree(dcsrVal)
    stat = hipFree(dB)
    stat = hipFree(dbuffer)

    stat = hipsparseDestroyCsrsm2Info(info)
    stat = hipsparseDestroyMatDescr(descr)
    stat = hipsparseDestroy(handle)

end program example_hipsparse_csrsm2
! [doc example end]

