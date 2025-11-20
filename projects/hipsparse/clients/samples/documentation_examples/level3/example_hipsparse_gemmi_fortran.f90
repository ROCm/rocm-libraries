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
program example_hipsparse_gemmi
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

        function hipsparseSgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, &
                                 cscRowIndB, beta, C, ldc) &
                bind(c, name = 'hipsparseSgemmi')
            use iso_c_binding
            implicit none
            integer(c_int) :: hipsparseSgemmi
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: cscValB
            type(c_ptr), intent(in), value :: cscColPtrB
            type(c_ptr), intent(in), value :: cscRowIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseSgemmi
    end interface

    ! Variables
    type(c_ptr) :: handle
    integer :: i, stat
    
    ! C = alpha * A * B + beta * C
    ! A is dense (m x k), B is sparse CSC (k x n), C is dense (m x n)
    integer, parameter :: m = 3
    integer, parameter :: n = 5
    integer, parameter :: k = 4
    integer, parameter :: lda = m
    integer, parameter :: ldc = m
    integer, parameter :: nnz_A = m * k
    integer, parameter :: nnz_B = 10
    integer, parameter :: nnz_C = m * n
    
    ! Sparse matrix B in CSC format
    integer, dimension(n+1), target :: hcscColPtr = (/0, 2, 5, 7, 8, 10/)
    integer, dimension(nnz_B), target :: hcscRowInd = (/0, 2, 0, 1, 3, 1, 3, 2, 0, 2/)
    real(c_float), dimension(nnz_B), target :: hcsc_val = (/1.0, 6.0, 2.0, 4.0, 9.0, 5.0, 2.0, 7.0, 3.0, 8.0/)
    
    ! Dense matrices A and C
    real(c_float), dimension(nnz_A), target :: hA
    real(c_float), dimension(nnz_C), target :: hC
    
    ! Scalar values
    real(c_float), target :: alpha = 0.5
    real(c_float), target :: beta = 0.25
    
    ! Device pointers
    type(c_ptr) :: dcscColPtr
    type(c_ptr) :: dcscRowInd
    type(c_ptr) :: dcsc_val
    type(c_ptr) :: dA
    type(c_ptr) :: dC
    
    ! Initialize dense matrices with 1.0
    do i = 1, nnz_A
        hA(i) = 1.0
    end do
    
    do i = 1, nnz_C
        hC(i) = 1.0
    end do

    ! Create hipSPARSE handle
    stat = hipsparseCreate(handle)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseCreate failed'
        stop
    end if

    ! Allocate device memory for CSC matrix B
    stat = hipMalloc(dcscColPtr, int((n + 1) * 4, c_size_t))
    if (stat /= 0) then
        write(*,*) 'Error: hipMalloc dcscColPtr failed'
        stop
    end if

    stat = hipMalloc(dcscRowInd, int(nnz_B * 4, c_size_t))
    if (stat /= 0) then
        write(*,*) 'Error: hipMalloc dcscRowInd failed'
        stop
    end if

    stat = hipMalloc(dcsc_val, int(nnz_B * 4, c_size_t))
    if (stat /= 0) then
        write(*,*) 'Error: hipMalloc dcsc_val failed'
        stop
    end if

    ! Allocate device memory for dense matrices A and C
    stat = hipMalloc(dA, int(nnz_A * 4, c_size_t))
    if (stat /= 0) then
        write(*,*) 'Error: hipMalloc dA failed'
        stop
    end if

    stat = hipMalloc(dC, int(nnz_C * 4, c_size_t))
    if (stat /= 0) then
        write(*,*) 'Error: hipMalloc dC failed'
        stop
    end if

    ! Copy data to device
    stat = hipMemcpy(dcscColPtr, c_loc(hcscColPtr), int((n + 1) * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) then
        write(*,*) 'Error: hipMemcpy dcscColPtr failed'
        stop
    end if

    stat = hipMemcpy(dcscRowInd, c_loc(hcscRowInd), int(nnz_B * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) then
        write(*,*) 'Error: hipMemcpy dcscRowInd failed'
        stop
    end if

    stat = hipMemcpy(dcsc_val, c_loc(hcsc_val), int(nnz_B * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) then
        write(*,*) 'Error: hipMemcpy dcsc_val failed'
        stop
    end if

    stat = hipMemcpy(dA, c_loc(hA), int(nnz_A * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) then
        write(*,*) 'Error: hipMemcpy dA failed'
        stop
    end if

    stat = hipMemcpy(dC, c_loc(hC), int(nnz_C * 4, c_size_t), hipMemcpyHostToDevice)
    if (stat /= 0) then
        write(*,*) 'Error: hipMemcpy dC failed'
        stop
    end if

    ! Perform operation: C = alpha * A * B + beta * C
    stat = hipsparseSgemmi(handle, &
                           m, &
                           n, &
                           k, &
                           nnz_B, &
                           c_loc(alpha), &
                           dA, &
                           lda, &
                           dcsc_val, &
                           dcscColPtr, &
                           dcscRowInd, &
                           c_loc(beta), &
                           dC, &
                           ldc)
    if (stat /= 0) then
        write(*,*) 'Error: hipsparseSgemmi failed'
        stop
    end if

    ! Copy result back to host
    stat = hipMemcpy(c_loc(hC), dC, int(nnz_C * 4, c_size_t), hipMemcpyDeviceToHost)
    if (stat /= 0) then
        write(*,*) 'Error: hipMemcpy hC failed'
        stop
    end if

    ! Print result
    write(*,*) 'hC:'
    do i = 1, nnz_C
        write(*,*) hC(i)
    end do

    ! Clean up
    stat = hipFree(dcscColPtr)
    stat = hipFree(dcscRowInd)
    stat = hipFree(dcsc_val)
    stat = hipFree(dA)
    stat = hipFree(dC)

    stat = hipsparseDestroy(handle)

end program example_hipsparse_gemmi
! [doc example end]
