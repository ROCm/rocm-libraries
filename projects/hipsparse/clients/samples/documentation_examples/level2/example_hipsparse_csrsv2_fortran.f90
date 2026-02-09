!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

subroutine HIP_CHECK(stat)
    use iso_c_binding

    implicit none

    integer(c_int) :: stat

    if(stat /= 0) then
        write(*,*) 'Error: hip error'
        stop
    end if

end subroutine HIP_CHECK

subroutine HIPSPARSE_CHECK(stat)
    use iso_c_binding

    implicit none

    integer(c_int) :: stat

    if(stat /= 0) then
        write(*,*) 'Error: hipsparse error'
        stop
    end if

end subroutine HIPSPARSE_CHECK

! [doc example start]
program example_fortran_csrsv2
    use iso_c_binding
    use hipsparse

    implicit none

    interface
        function hipMalloc(ptr, size) &
                bind(c, name = 'hipMalloc')
            use iso_c_binding
            implicit none
            integer :: hipMalloc
            type(c_ptr) :: ptr
            integer(c_size_t), value :: size
        end function hipMalloc

        function hipFree(ptr) &
                bind(c, name = 'hipFree')
            use iso_c_binding
            implicit none
            integer :: hipFree
            type(c_ptr), value :: ptr
        end function hipFree

        function hipMemcpy(dst, src, size, kind) &
                bind(c, name = 'hipMemcpy')
            use iso_c_binding
            implicit none
            integer :: hipMemcpy
            type(c_ptr), value :: dst
            type(c_ptr), intent(in), value :: src
            integer(c_size_t), value :: size
            integer(c_int), value :: kind
        end function hipMemcpy
    end interface

    integer, target :: h_csr_row_ptr(5), h_csr_col_ind(13)
    real(8), target :: h_csr_val(13), h_f(4), h_x(4)

    type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val
    type(c_ptr) :: d_f, d_x, d_buffer

    integer :: i
    integer(c_int) :: m, nnz, trans, policy
    integer(c_int), target :: buffer_size

    real(c_double), target :: alpha

    type(c_ptr) :: handle, descr, info

    integer :: version

!   Input data
    m = 4
    nnz = 13

!   Fill CSR structure
    h_csr_row_ptr = (/0, 2, 6, 10, 13/)
    h_csr_col_ind = (/0, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 3/)
    h_csr_val     = (/1.0d0, 2.0d0, 3.0d0, 2.0d0, 4.0d0, 1.0d0, 5.0d0, 6.0d0, 1.0d0, 3.0d0, 7.0d0, 8.0d0, 0.6d0/)

!   Transposition and policy
    trans = HIPSPARSE_OPERATION_NON_TRANSPOSE
    policy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL

!   Scalar alpha
    alpha = 1.0d0

!   f and x vectors
    h_f = (/32.0d0, 14.7d0, 33.6d0, 10.0d0/)
    h_x = 0.0d0

!   Allocate device memory
    call HIP_CHECK(hipMalloc(d_csr_row_ptr, int(m + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_col_ind, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_val, int(nnz, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(d_f, int(m, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(d_x, int(m, c_size_t) * 8))

!   Copy host data to device
    call HIP_CHECK(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr), int(m + 1, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind), int(nnz, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_csr_val, c_loc(h_csr_val), int(nnz, c_size_t) * 8, 1))
    call HIP_CHECK(hipMemcpy(d_f, c_loc(h_f), int(m, c_size_t) * 8, 1))

!   Create hipSPARSE handle
    call HIPSPARSE_CHECK(hipsparseCreate(handle))

!   Create matrix descriptor
    call HIPSPARSE_CHECK(hipsparseCreateMatDescr(descr))

!   Set matrix properties
    call HIPSPARSE_CHECK(hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO))
    call HIPSPARSE_CHECK(hipsparseSetMatFillMode(descr, HIPSPARSE_FILL_MODE_LOWER))
    call HIPSPARSE_CHECK(hipsparseSetMatDiagType(descr, HIPSPARSE_DIAG_TYPE_UNIT))

!   Create csrsv2 info
    call HIPSPARSE_CHECK(hipsparseCreateCsrsv2Info(info))

!   Get hipSPARSE version
    call HIPSPARSE_CHECK(hipsparseGetVersion(handle, version))
    write(*,fmt='(A,I0,A,I0,A,I0)') 'hipSPARSE version: ', version / 100000, '.', &
        mod(version / 100, 1000), '.', mod(version, 100)

!   Get buffer size
    call HIPSPARSE_CHECK(hipsparseDcsrsv2_bufferSize(handle, &
                                                     trans, &
                                                     m, &
                                                     nnz, &
                                                     descr, &
                                                     d_csr_val, &
                                                     d_csr_row_ptr, &
                                                     d_csr_col_ind, &
                                                     info, &
                                                     c_loc(buffer_size)))

!   Allocate buffer
    call HIP_CHECK(hipMalloc(d_buffer, int(buffer_size, c_size_t)))

!   Analysis step
    call HIPSPARSE_CHECK(hipsparseDcsrsv2_analysis(handle, &
                                                   trans, &
                                                   m, &
                                                   nnz, &
                                                   descr, &
                                                   d_csr_val, &
                                                   d_csr_row_ptr, &
                                                   d_csr_col_ind, &
                                                   info, &
                                                   policy, &
                                                   d_buffer))

!   Call dcsrsv2_solve to perform alpha * A * x = f
    call HIPSPARSE_CHECK(hipsparseDcsrsv2_solve(handle, &
                                                trans, &
                                                m, &
                                                nnz, &
                                                c_loc(alpha), &
                                                descr, &
                                                d_csr_val, &
                                                d_csr_row_ptr, &
                                                d_csr_col_ind, &
                                                info, &
                                                d_f, &
                                                d_x, &
                                                policy, &
                                                d_buffer))

!   Copy result back to host
    call HIP_CHECK(hipMemcpy(c_loc(h_x), d_x, int(m, c_size_t) * 8, 2))

!   Print result
    write(*,fmt='(A)',advance='no') 'hx:'
    do i = 1, m
        write(*,fmt='(A,F0.2)',advance='no') ' ', h_x(i)
    end do
    write(*,*)

!   Clear hipSPARSE
    call HIPSPARSE_CHECK(hipsparseDestroyCsrsv2Info(info))
    call HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr))
    call HIPSPARSE_CHECK(hipsparseDestroy(handle))

!   Clear device memory
    call HIP_CHECK(hipFree(d_csr_row_ptr))
    call HIP_CHECK(hipFree(d_csr_col_ind))
    call HIP_CHECK(hipFree(d_csr_val))
    call HIP_CHECK(hipFree(d_f))
    call HIP_CHECK(hipFree(d_x))
    call HIP_CHECK(hipFree(d_buffer))

end program example_fortran_csrsv2
! [doc example end]
