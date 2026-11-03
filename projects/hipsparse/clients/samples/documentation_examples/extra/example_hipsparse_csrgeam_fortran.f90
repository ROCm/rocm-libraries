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
program example_hipsparse_csrgeam
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

        function hipsparseXcsrgeamNnz(handle, m, n, descrA, nnzA, csrRowPtrA, csrColIndA, descrB, &
                                      nnzB, csrRowPtrB, csrColIndB, descrC, csrRowPtrC, nnzTotalDevHostPtr) &
                bind(c, name = 'hipsparseXcsrgeamNnz')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseXcsrgeamNnz
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), value :: descrC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: nnzTotalDevHostPtr
        end function hipsparseXcsrgeamNnz

        function hipsparseScsrgeam(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, &
                                   csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, &
                                   csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC) &
                bind(c, name = 'hipsparseScsrgeam')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseScsrgeam
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrSortedValA
            type(c_ptr), intent(in), value :: csrSortedRowPtrA
            type(c_ptr), intent(in), value :: csrSortedColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrSortedValB
            type(c_ptr), intent(in), value :: csrSortedRowPtrB
            type(c_ptr), intent(in), value :: csrSortedColIndB
            type(c_ptr), value :: descrC
            type(c_ptr), value :: csrSortedValC
            type(c_ptr), value :: csrSortedRowPtrC
            type(c_ptr), value :: csrSortedColIndC
        end function hipsparseScsrgeam
    end interface

    ! Variables
    type(c_ptr) :: handle
    type(c_ptr) :: descrA, descrB, descrC
    integer :: i, stat
    
    ! Matrix addition: C = alpha * A + beta * B (no transpose!)
    integer, parameter :: m = 4
    integer, parameter :: n = 4
    integer, parameter :: nnzA = 9
    integer, parameter :: nnzB = 6
    integer(c_int), target :: nnzC
    
    ! Matrix A (4x4)
    integer, dimension(m+1), target :: hcsrRowPtrA = (/0, 2, 4, 8, 9/)
    integer, dimension(nnzA), target :: hcsrColIndA = (/0, 3, 0, 1, 0, 1, 2, 3, 2/)
    real(c_float), dimension(nnzA), target :: hcsrValA = (/1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0/)
    
    ! Matrix B (4x4)
    integer, dimension(m+1), target :: hcsrRowPtrB = (/0, 1, 3, 5, 6/)
    integer, dimension(nnzB), target :: hcsrColIndB = (/1, 0, 2, 1, 3, 2/)
    real(c_float), dimension(nnzB), target :: hcsrValB = (/1.0, 1.0, 1.0, 1.0, 1.0, 1.0/)
    
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
    type(c_ptr) :: dcsrRowPtrC, dcsrColIndC, dcsrValC

    ! Create hipSPARSE handle
    stat = hipsparseCreate(handle)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseCreate failed'
        stop
    end if

    ! Create matrix descriptors
    stat = hipsparseCreateMatDescr(descrA)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseCreateMatDescr descrA failed'
        stop
    end if

    stat = hipsparseCreateMatDescr(descrB)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseCreateMatDescr descrB failed'
        stop
    end if

    stat = hipsparseCreateMatDescr(descrC)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseCreateMatDescr descrC failed'
        stop
    end if

    ! Allocate device memory for A and B
    stat = hipMalloc(dcsrRowPtrA, int((m + 1) * 4, c_size_t))
    if (stat /= 0) stop
    stat = hipMalloc(dcsrColIndA, int(nnzA * 4, c_size_t))
    if (stat /= 0) stop
    stat = hipMalloc(dcsrValA, int(nnzA * 4, c_size_t))
    if (stat /= 0) stop

    stat = hipMalloc(dcsrRowPtrB, int((m + 1) * 4, c_size_t))
    if (stat /= 0) stop
    stat = hipMalloc(dcsrColIndB, int(nnzB * 4, c_size_t))
    if (stat /= 0) stop
    stat = hipMalloc(dcsrValB, int(nnzB * 4, c_size_t))
    if (stat /= 0) stop

    stat = hipMalloc(dcsrRowPtrC, int((m + 1) * 4, c_size_t))
    if (stat /= 0) stop

    ! Copy A and B to device
    stat = hipMemcpy(dcsrRowPtrA, c_loc(hcsrRowPtrA), int((m + 1) * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop
    stat = hipMemcpy(dcsrColIndA, c_loc(hcsrColIndA), int(nnzA * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop
    stat = hipMemcpy(dcsrValA, c_loc(hcsrValA), int(nnzA * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop

    stat = hipMemcpy(dcsrRowPtrB, c_loc(hcsrRowPtrB), int((m + 1) * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop
    stat = hipMemcpy(dcsrColIndB, c_loc(hcsrColIndB), int(nnzB * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop
    stat = hipMemcpy(dcsrValB, c_loc(hcsrValB), int(nnzB * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) stop

    ! Determine nnzC
    stat = hipsparseXcsrgeamNnz(handle, &
                                m, &
                                n, &
                                descrA, &
                                nnzA, &
                                dcsrRowPtrA, &
                                dcsrColIndA, &
                                descrB, &
                                nnzB, &
                                dcsrRowPtrB, &
                                dcsrColIndB, &
                                descrC, &
                                dcsrRowPtrC, &
                                c_loc(nnzC))
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseXcsrgeamNnz failed'
        stop
    end if

    ! Allocate device memory for C
    stat = hipMalloc(dcsrColIndC, int(nnzC * 4, c_size_t))
    if (stat /= 0) stop
    stat = hipMalloc(dcsrValC, int(nnzC * 4, c_size_t))
    if (stat /= 0) stop

    ! Perform matrix addition: C = alpha * A + beta * B
    stat = hipsparseScsrgeam(handle, &
                             m, &
                             n, &
                             c_loc(alpha), &
                             descrA, &
                             nnzA, &
                             dcsrValA, &
                             dcsrRowPtrA, &
                             dcsrColIndA, &
                             c_loc(beta), &
                             descrB, &
                             nnzB, &
                             dcsrValB, &
                             dcsrRowPtrB, &
                             dcsrColIndB, &
                             descrC, &
                             dcsrValC, &
                             dcsrRowPtrC, &
                             dcsrColIndC)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseScsrgeam failed'
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
    write(*,*) 'Matrix C (result of A + B):'
    write(*,*) 'nnzC =', nnzC
    write(*,*) 'csrRowPtrC:'
    do i = 1, m + 1
        write(*,*) hcsrRowPtrC(i)
    end do
    write(*,*) 'csrColIndC:'
    do i = 1, nnzC
        write(*,*) hcsrColIndC(i)
    end do
    write(*,*) 'csrValC:'
    do i = 1, nnzC
        write(*,*) hcsrValC(i)
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
    stat = hipFree(dcsrRowPtrC)
    stat = hipFree(dcsrColIndC)
    stat = hipFree(dcsrValC)

    stat = hipsparseDestroyMatDescr(descrA)
    stat = hipsparseDestroyMatDescr(descrB)
    stat = hipsparseDestroyMatDescr(descrC)
    stat = hipsparseDestroy(handle)

end program example_hipsparse_csrgeam
! [doc example end]
