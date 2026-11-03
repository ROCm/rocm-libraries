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
program example_hipsparse_csrgemm2
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

        function hipsparseCreateCsrgemm2Info(info) &
                bind(c, name = 'hipsparseCreateCsrgemm2Info')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseCreateCsrgemm2Info
            type(c_ptr) :: info
        end function hipsparseCreateCsrgemm2Info

        function hipsparseDestroyCsrgemm2Info(info) &
                bind(c, name = 'hipsparseDestroyCsrgemm2Info')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseDestroyCsrgemm2Info
            type(c_ptr), value :: info
        end function hipsparseDestroyCsrgemm2Info

        function hipsparseScsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, nnzA, csrRowPtrA, &
                                                   csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, &
                                                   beta, descrD, nnzD, csrRowPtrD, csrColIndD, &
                                                   info, pBufferSizeInBytes) &
                bind(c, name = 'hipsparseScsrgemm2_bufferSizeExt')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseScsrgemm2_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), value :: info
            type(c_ptr), value :: pBufferSizeInBytes
        end function hipsparseScsrgemm2_bufferSizeExt

        function hipsparseXcsrgemm2Nnz(handle, m, n, k, descrA, nnzA, csrRowPtrA, csrColIndA, &
                                       descrB, nnzB, csrRowPtrB, csrColIndB, descrD, nnzD, &
                                       csrRowPtrD, csrColIndD, descrC, csrRowPtrC, nnzTotalDevHostPtr, &
                                       info, pBuffer) &
                bind(c, name = 'hipsparseXcsrgemm2Nnz')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseXcsrgemm2Nnz
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), value :: descrC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: nnzTotalDevHostPtr
            type(c_ptr), value :: info
            type(c_ptr), value :: pBuffer
        end function hipsparseXcsrgemm2Nnz

        function hipsparseScsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrValA, csrRowPtrA, &
                                    csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, &
                                    beta, descrD, nnzD, csrValD, csrRowPtrD, csrColIndD, &
                                    descrC, csrValC, csrRowPtrC, csrColIndC, info, pBuffer) &
                bind(c, name = 'hipsparseScsrgemm2')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseScsrgemm2
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrValD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), value :: info
            type(c_ptr), value :: pBuffer
        end function hipsparseScsrgemm2
    end interface

    ! Variables
    type(c_ptr) :: handle
    type(c_ptr) :: descrA, descrB, descrC, descrD
    type(c_ptr) :: info
    integer :: i, stat, start_idx, end_idx
    integer(c_size_t), target :: bufferSize
    
    ! C = alpha * A * B + beta * D
    ! A is m x k, B is k x n, D is m x n, C is m x n
    integer, parameter :: m = 4
    integer, parameter :: k = 3
    integer, parameter :: n = 2
    integer, parameter :: nnzA = 7
    integer, parameter :: nnzB = 3
    integer, parameter :: nnzD = 6
    integer(c_int), target :: nnzC
    
    ! Matrix A (4x3)
    integer, dimension(m+1), target :: hcsrRowPtrA = (/0, 1, 3, 6, 7/)
    integer, dimension(nnzA), target :: hcsrColIndA = (/0, 0, 1, 0, 1, 2, 2/)
    real(c_float), dimension(nnzA), target :: hcsrValA = (/1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0/)
    
    ! Matrix B (3x2)
    integer, dimension(k+1), target :: hcsrRowPtrB = (/0, 1, 2, 3/)
    integer, dimension(nnzB), target :: hcsrColIndB = (/1, 0, 1/)
    real(c_float), dimension(nnzB), target :: hcsrValB = (/1.0, 1.0, 1.0/)
    
    ! Matrix D (4x2)
    integer, dimension(m+1), target :: hcsrRowPtrD = (/0, 1, 3, 5, 6/)
    integer, dimension(nnzD), target :: hcsrColIndD = (/1, 0, 1, 0, 1, 1/)
    real(c_float), dimension(nnzD), target :: hcsrValD = (/1.0, 2.0, 3.0, 4.0, 5.0, 6.0/)
    
    ! Matrix C (will be allocated after nnzC is determined)
    integer, dimension(:), allocatable, target :: hcsrRowPtrC
    integer, dimension(:), allocatable, target :: hcsrColIndC
    real(c_float), dimension(:), allocatable, target :: hcsrValC
    
    ! Scalar values
    real(c_float), target :: alpha = 1.0
    real(c_float), target :: beta = 1.0
    
    ! Device pointers
    type(c_ptr) :: dcsrRowPtrA, dcsrColIndA, dcsrValA
    type(c_ptr) :: dcsrRowPtrB, dcsrColIndB, dcsrValB
    type(c_ptr) :: dcsrRowPtrD, dcsrColIndD, dcsrValD
    type(c_ptr) :: dcsrRowPtrC, dcsrColIndC, dcsrValC
    type(c_ptr) :: dbuffer

    ! Create hipSPARSE handle
    stat = hipsparseCreate(handle)
    if (stat /= 0) stop

    ! Create matrix descriptors
    stat = hipsparseCreateMatDescr(descrA)
    if (stat /= 0) stop
    stat = hipsparseCreateMatDescr(descrB)
    if (stat /= 0) stop
    stat = hipsparseCreateMatDescr(descrC)
    if (stat /= 0) stop
    stat = hipsparseCreateMatDescr(descrD)
    if (stat /= 0) stop

    ! Create csrgemm2 info
    stat = hipsparseCreateCsrgemm2Info(info)
    if (stat /= 0) stop

    ! Allocate device memory for A, B, and D
    stat = hipMalloc(dcsrRowPtrA, int((m + 1) * 4, c_size_t))
    if (stat /= 0) stop
    stat = hipMalloc(dcsrColIndA, int(nnzA * 4, c_size_t))
    if (stat /= 0) stop
    stat = hipMalloc(dcsrValA, int(nnzA * 4, c_size_t))
    if (stat /= 0) stop

    stat = hipMalloc(dcsrRowPtrB, int((k + 1) * 4, c_size_t))
    if (stat /= 0) stop
    stat = hipMalloc(dcsrColIndB, int(nnzB * 4, c_size_t))
    if (stat /= 0) stop
    stat = hipMalloc(dcsrValB, int(nnzB * 4, c_size_t))
    if (stat /= 0) stop

    stat = hipMalloc(dcsrRowPtrD, int((m + 1) * 4, c_size_t))
    if (stat /= 0) stop
    stat = hipMalloc(dcsrColIndD, int(nnzD * 4, c_size_t))
    if (stat /= 0) stop
    stat = hipMalloc(dcsrValD, int(nnzD * 4, c_size_t))
    if (stat /= 0) stop

    stat = hipMalloc(dcsrRowPtrC, int((m + 1) * 4, c_size_t))
    if (stat /= 0) stop

    ! Copy A, B, and D to device
    stat = hipMemcpy(dcsrRowPtrA, c_loc(hcsrRowPtrA), int((m + 1) * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop
    stat = hipMemcpy(dcsrColIndA, c_loc(hcsrColIndA), int(nnzA * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop
    stat = hipMemcpy(dcsrValA, c_loc(hcsrValA), int(nnzA * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop

    stat = hipMemcpy(dcsrRowPtrB, c_loc(hcsrRowPtrB), int((k + 1) * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop
    stat = hipMemcpy(dcsrColIndB, c_loc(hcsrColIndB), int(nnzB * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop
    stat = hipMemcpy(dcsrValB, c_loc(hcsrValB), int(nnzB * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop

    stat = hipMemcpy(dcsrRowPtrD, c_loc(hcsrRowPtrD), int((m + 1) * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop
    stat = hipMemcpy(dcsrColIndD, c_loc(hcsrColIndD), int(nnzD * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop
    stat = hipMemcpy(dcsrValD, c_loc(hcsrValD), int(nnzD * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop

    ! Get buffer size
    stat = hipsparseScsrgemm2_bufferSizeExt(handle, &
                                            m, &
                                            n, &
                                            k, &
                                            c_loc(alpha), &
                                            descrA, &
                                            nnzA, &
                                            dcsrRowPtrA, &
                                            dcsrColIndA, &
                                            descrB, &
                                            nnzB, &
                                            dcsrRowPtrB, &
                                            dcsrColIndB, &
                                            c_loc(beta), &
                                            descrD, &
                                            nnzD, &
                                            dcsrRowPtrD, &
                                            dcsrColIndD, &
                                            info, &
                                            c_loc(bufferSize))
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseScsrgemm2_bufferSizeExt failed'
        stop
    end if

    ! Allocate temporary buffer
    stat = hipMalloc(dbuffer, bufferSize)
    if (stat /= 0) stop

    ! Determine nnzC
    stat = hipsparseXcsrgemm2Nnz(handle, &
                                 m, &
                                 n, &
                                 k, &
                                 descrA, &
                                 nnzA, &
                                 dcsrRowPtrA, &
                                 dcsrColIndA, &
                                 descrB, &
                                 nnzB, &
                                 dcsrRowPtrB, &
                                 dcsrColIndB, &
                                 descrD, &
                                 nnzD, &
                                 dcsrRowPtrD, &
                                 dcsrColIndD, &
                                 descrC, &
                                 dcsrRowPtrC, &
                                 c_loc(nnzC), &
                                 info, &
                                 dbuffer)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseXcsrgemm2Nnz failed'
        stop
    end if

    ! Allocate device memory for C
    stat = hipMalloc(dcsrColIndC, int(nnzC * 4, c_size_t))
    if (stat /= 0) stop
    stat = hipMalloc(dcsrValC, int(nnzC * 4, c_size_t))
    if (stat /= 0) stop

    ! Compute C = alpha * A * B + beta * D
    stat = hipsparseScsrgemm2(handle, &
                              m, &
                              n, &
                              k, &
                              c_loc(alpha), &
                              descrA, &
                              nnzA, &
                              dcsrValA, &
                              dcsrRowPtrA, &
                              dcsrColIndA, &
                              descrB, &
                              nnzB, &
                              dcsrValB, &
                              dcsrRowPtrB, &
                              dcsrColIndB, &
                              c_loc(beta), &
                              descrD, &
                              nnzD, &
                              dcsrValD, &
                              dcsrRowPtrD, &
                              dcsrColIndD, &
                              descrC, &
                              dcsrValC, &
                              dcsrRowPtrC, &
                              dcsrColIndC, &
                              info, &
                              dbuffer)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseScsrgemm2 failed'
        stop
    end if

    ! Allocate host memory for C
    allocate(hcsrRowPtrC(m+1))
    allocate(hcsrColIndC(nnzC))
    allocate(hcsrValC(nnzC))

    ! Copy result back to host
    stat = hipMemcpy(c_loc(hcsrRowPtrC), dcsrRowPtrC, int((m + 1) * 4, c_size_t), hipMemcpyDeviceToHost)
    if (stat /= 0) stop
    stat = hipMemcpy(c_loc(hcsrColIndC), dcsrColIndC, int(nnzC * 4, c_size_t), hipMemcpyDeviceToHost)
    if (stat /= 0) stop
    stat = hipMemcpy(c_loc(hcsrValC), dcsrValC, int(nnzC * 4, c_size_t), hipMemcpyDeviceToHost)
    if (stat /= 0) stop

    ! Print result
    write(*,*) 'Matrix C (result of alpha*A*B + beta*D):'
    do i = 1, m
        start_idx = hcsrRowPtrC(i) + 1
        end_idx = hcsrRowPtrC(i + 1)
        write(*,fmt='(A,I0,A)',advance='no') 'Row ', i-1, ':'
        do stat = start_idx, end_idx
            write(*,fmt='(A,I0,A,F0.2,A)',advance='no') ' (', hcsrColIndC(stat), ',', hcsrValC(stat), ')'
        end do
        write(*,*)
    end do

    ! Clean up
    deallocate(hcsrRowPtrC)
    deallocate(hcsrColIndC)
    deallocate(hcsrValC)

    stat = hipFree(dcsrRowPtrA)
    stat = hipFree(dcsrColIndA)
    stat = hipFree(dcsrValA)
    stat = hipFree(dcsrRowPtrB)
    stat = hipFree(dcsrColIndB)
    stat = hipFree(dcsrValB)
    stat = hipFree(dcsrRowPtrD)
    stat = hipFree(dcsrColIndD)
    stat = hipFree(dcsrValD)
    stat = hipFree(dcsrRowPtrC)
    stat = hipFree(dcsrColIndC)
    stat = hipFree(dcsrValC)
    stat = hipFree(dbuffer)

    stat = hipsparseDestroyCsrgemm2Info(info)
    stat = hipsparseDestroyMatDescr(descrA)
    stat = hipsparseDestroyMatDescr(descrB)
    stat = hipsparseDestroyMatDescr(descrC)
    stat = hipsparseDestroyMatDescr(descrD)
    stat = hipsparseDestroy(handle)

end program example_hipsparse_csrgemm2
! [doc example end]
