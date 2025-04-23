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


module rocblas
    use iso_c_binding

    !!!!!!!!!!!!!!!!!!!!!!!
    !    rocBLAS types    !
    !!!!!!!!!!!!!!!!!!!!!!!
    enum, bind(c)
        enumerator :: rocblas_operation_none = 111
        enumerator :: rocblas_operation_transpose = 112
        enumerator :: rocblas_operation_conjugate_transpose = 113
    end enum

    enum, bind(c)
        enumerator :: rocblas_fill_upper = 121
        enumerator :: rocblas_fill_lower = 122
        enumerator :: rocblas_fill_full  = 123
    end enum

    enum, bind(c)
        enumerator :: rocblas_diagonal_non_unit = 131
        enumerator :: rocblas_diagonal_unit     = 132
    end enum

    enum, bind(c)
        enumerator :: rocblas_side_left  = 141
        enumerator :: rocblas_side_right = 142
        enumerator :: rocblas_side_both  = 143
    end enum

    enum, bind(c)
        enumerator :: rocblas_status_success         = 0
        enumerator :: rocblas_status_invalid_handle  = 1
        enumerator :: rocblas_status_not_implemented = 2
        enumerator :: rocblas_status_invalid_pointer = 3
        enumerator :: rocblas_status_invalid_size    = 4
        enumerator :: rocblas_status_memory_error    = 5
        enumerator :: rocblas_status_internal_error  = 6
        enumerator :: rocblas_status_perf_degraded   = 7
        enumerator :: rocblas_status_size_query_mismatch = 8
        enumerator :: rocblas_status_size_increased      = 9
        enumerator :: rocblas_status_size_unchanged      = 10
        enumerator :: rocblas_status_invalid_value       = 11
        enumerator :: rocblas_status_continue            = 12
    end enum

    enum, bind(c)
        enumerator :: rocblas_pointer_mode_host = 0
        enumerator :: rocblas_pointer_mode_device = 1
    end enum

    

    !!!!!!!!!!!!!!!!!!!!!!!
    ! rocblas-auxiliary.h !
    !!!!!!!!!!!!!!!!!!!!!!!

    interface
        function rocblas_create_handle(handle) &
                result(c_int) &
                bind(c, name = 'rocblas_create_handle')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
        end function rocblas_create_handle
    end interface

    interface
        function rocblas_destroy_handle(handle) &
                result(c_int) &
                bind(c, name = 'rocblas_destroy_handle')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
        end function rocblas_destroy_handle
    end interface

    interface
        function rocblas_add_stream(handle, stream) &
                result(c_int) &
                bind(c, name = 'rocblas_add_stream')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: stream
        end function rocblas_add_stream
    end interface

    interface
        function rocblas_set_stream(handle, stream) &
                result(c_int) &
                bind(c, name = 'rocblas_set_stream')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: stream
        end function rocblas_set_stream
    end interface

    interface
        function rocblas_get_stream(handle, stream) &
                result(c_int) &
                bind(c, name = 'rocblas_get_stream')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: stream
        end function rocblas_get_stream
    end interface

    interface
        function rocblas_set_pointer_mode(handle, pointer_mode) &
                result(c_int) &
                bind(c, name = 'rocblas_set_pointer_mode')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: pointer_mode
        end function rocblas_set_pointer_mode
    end interface

    interface
        function rocblas_get_pointer_mode(handle, pointer_mode) &
                result(c_int) &
                bind(c, name = 'rocblas_get_pointer_mode')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int) :: pointer_mode
        end function rocblas_get_pointer_mode
    end interface

    ! TODO: is this good?
    interface
        function rocblas_pointer_to_mode(ptr) &
                result(c_int) &
                bind(c, name = 'rocblas_pointer_to_mode')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: ptr
        end function rocblas_pointer_to_mode
    end interface

    interface
        function rocblas_set_vector(n, elem_size, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_set_vector')
            use iso_c_binding
            implicit none
            integer(c_int), value :: n
            integer(c_int), value :: elem_size
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_set_vector
    end interface

    interface
        function rocblas_get_vector(n, elem_size, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_get_vector')
            use iso_c_binding
            implicit none
            integer(c_int), value :: n
            integer(c_int), value :: elem_size
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_get_vector
    end interface

    interface
        function rocblas_set_matrix(rows, cols, elem_size, a, lda, b, ldb) &
                result(c_int) &
                bind(c, name = 'rocblas_set_matrix')
            use iso_c_binding
            implicit none
            integer(c_int), value :: rows
            integer(c_int), value :: cols
            integer(c_int), value :: elem_size
            type(c_ptr), value :: a
            integer(c_int), value :: lda
            type(c_ptr), value :: b
            integer(c_int), value :: ldb
        end function rocblas_set_matrix
    end interface

    interface
        function rocblas_get_matrix(rows, cols, elem_size, a, lda, b, ldb) &
                result(c_int) &
                bind(c, name = 'rocblas_get_matrix')
            use iso_c_binding
            implicit none
            integer(c_int), value :: rows
            integer(c_int), value :: cols
            integer(c_int), value :: elem_size
            type(c_ptr), value :: a
            integer(c_int), value :: lda
            type(c_ptr), value :: b
            integer(c_int), value :: ldb
        end function rocblas_get_matrix
    end interface

    interface
        function rocblas_set_vector_async(n, elem_size, x, incx, y, incy, stream) &
                result(c_int) &
                bind(c, name = 'rocblas_set_vector_async')
            use iso_c_binding
            implicit none
            integer(c_int), value :: n
            integer(c_int), value :: elem_size
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: stream
        end function rocblas_set_vector_async
    end interface

    interface
        function rocblas_get_vector_async(n, elem_size, x, incx, y, incy, stream) &
                result(c_int) &
                bind(c, name = 'rocblas_get_vector_async')
            use iso_c_binding
            implicit none
            integer(c_int), value :: n
            integer(c_int), value :: elem_size
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: stream
        end function rocblas_get_vector_async
    end interface

    interface
        function rocblas_set_matrix_async(rows, cols, elem_size, a, lda, b, ldb, stream) &
                result(c_int) &
                bind(c, name = 'rocblas_set_matrix_async')
            use iso_c_binding
            implicit none
            integer(c_int), value :: rows
            integer(c_int), value :: cols
            integer(c_int), value :: elem_size
            type(c_ptr), value :: a
            integer(c_int), value :: lda
            type(c_ptr), value :: b
            integer(c_int), value :: ldb
            type(c_ptr), value :: stream
        end function rocblas_set_matrix_async
    end interface

    interface
        function rocblas_get_matrix_async(rows, cols, elem_size, a, lda, b, ldb, stream) &
                result(c_int) &
                bind(c, name = 'rocblas_get_matrix_async')
            use iso_c_binding
            implicit none
            integer(c_int), value :: rows
            integer(c_int), value :: cols
            integer(c_int), value :: elem_size
            type(c_ptr), value :: a
            integer(c_int), value :: lda
            type(c_ptr), value :: b
            integer(c_int), value :: ldb
            type(c_ptr), value :: stream
        end function rocblas_get_matrix_async
    end interface

    ! TODO: ?
    interface
        function rocblas_set_start_stop_events(handle, start_event, stop_event) &
                result(c_int) &
                bind(c, name = 'rocblas_set_start_stop_events')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: start_event
            type(c_ptr), value :: stop_event
        end function rocblas_set_start_stop_events
    end interface

    !!!!!!!!!!!!!!!!!!!!!!!
    ! rocblas-functions.h !
    !!!!!!!!!!!!!!!!!!!!!!!

    !--------!
    ! blas 1 !
    !--------!

    ! scal
    interface
        function rocblas_sscal(handle, n, alpha, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_sscal')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_sscal
    end interface

    interface
        function rocblas_dscal(handle, n, alpha, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_dscal')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_dscal
    end interface

    interface
        function rocblas_cscal(handle, n, alpha, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_cscal')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_cscal
    end interface

    interface
        function rocblas_zscal(handle, n, alpha, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_zscal')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_zscal
    end interface

    interface
        function rocblas_csscal(handle, n, alpha, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_csscal')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_csscal
    end interface

    interface
        function rocblas_zdscal(handle, n, alpha, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_zdscal')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_zdscal
    end interface

    ! scal_batched
    interface
        function rocblas_sscal_batched(handle, n, alpha, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sscal_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_sscal_batched
    end interface

    interface
        function rocblas_dscal_batched(handle, n, alpha, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dscal_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_dscal_batched
    end interface

    interface
        function rocblas_cscal_batched(handle, n, alpha, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cscal_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_cscal_batched
    end interface

    interface
        function rocblas_zscal_batched(handle, n, alpha, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zscal_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_zscal_batched
    end interface

    interface
        function rocblas_csscal_batched(handle, n, alpha, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csscal_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_csscal_batched
    end interface

    interface
        function rocblas_zdscal_batched(handle, n, alpha, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zdscal_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_zdscal_batched
    end interface

    ! scal_strided_batched
    interface
        function rocblas_sscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sscal_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_sscal_strided_batched
    end interface

    interface
        function rocblas_dscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dscal_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_dscal_strided_batched
    end interface

    interface
        function rocblas_cscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cscal_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_cscal_strided_batched
    end interface

    interface
        function rocblas_zscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zscal_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_zscal_strided_batched
    end interface

    interface
        function rocblas_csscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csscal_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_csscal_strided_batched
    end interface

    interface
        function rocblas_zdscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zdscal_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_zdscal_strided_batched
    end interface

    ! copy
    interface
        function rocblas_scopy(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_scopy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_scopy
    end interface

    interface
        function rocblas_dcopy(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_dcopy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_dcopy
    end interface

    interface
        function rocblas_ccopy(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_ccopy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_ccopy
    end interface

    interface
        function rocblas_zcopy(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_zcopy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_zcopy
    end interface

    ! copy_batched
    interface
        function rocblas_scopy_batched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_scopy_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_scopy_batched
    end interface

    interface
        function rocblas_dcopy_batched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dcopy_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_dcopy_batched
    end interface

    interface
        function rocblas_ccopy_batched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ccopy_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_ccopy_batched
    end interface

    interface
        function rocblas_zcopy_batched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zcopy_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_zcopy_batched
    end interface

    ! copy_strided_batched
    interface
        function rocblas_scopy_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_scopy_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_scopy_strided_batched
    end interface

    interface
        function rocblas_dcopy_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dcopy_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_dcopy_strided_batched
    end interface

    interface
        function rocblas_ccopy_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ccopy_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_ccopy_strided_batched
    end interface

    interface
        function rocblas_zcopy_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zcopy_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_zcopy_strided_batched
    end interface

    ! dot
    interface
        function rocblas_sdot(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'rocblas_sdot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function rocblas_sdot
    end interface

    interface
        function rocblas_ddot(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'rocblas_ddot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function rocblas_ddot
    end interface

    interface
        function rocblas_hdot(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'rocblas_hdot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function rocblas_hdot
    end interface

    interface
        function rocblas_bfdot(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'rocblas_bfdot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function rocblas_bfdot
    end interface

    interface
        function rocblas_cdotu(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'rocblas_cdotu')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function rocblas_cdotu
    end interface

    interface
        function rocblas_cdotc(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'rocblas_cdotc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function rocblas_cdotc
    end interface

    interface
        function rocblas_zdotu(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'rocblas_zdotu')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function rocblas_zdotu
    end interface

    interface
        function rocblas_zdotc(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'rocblas_zdotc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function rocblas_zdotc
    end interface

    ! dot_batched
    interface
        function rocblas_sdot_batched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_sdot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_sdot_batched
    end interface

    interface
        function rocblas_ddot_batched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_ddot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_ddot_batched
    end interface

    interface
        function rocblas_hdot_batched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_hdot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_hdot_batched
    end interface

    interface
        function rocblas_bfdot_batched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_bfdot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_bfdot_batched
    end interface

    interface
        function rocblas_cdotu_batched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_cdotu_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_cdotu_batched
    end interface

    interface
        function rocblas_cdotc_batched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_cdotc_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_cdotc_batched
    end interface

    interface
        function rocblas_zdotu_batched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_zdotu_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_zdotu_batched
    end interface

    interface
        function rocblas_zdotc_batched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_zdotc_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_zdotc_batched
    end interface

    ! dot_strided_batched
    interface
        function rocblas_sdot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_sdot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_sdot_strided_batched
    end interface

    interface
        function rocblas_ddot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_ddot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_ddot_strided_batched
    end interface

    interface
        function rocblas_hdot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_hdot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_hdot_strided_batched
    end interface

    interface
        function rocblas_bfdot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_bfdot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_bfdot_strided_batched
    end interface

    interface
        function rocblas_cdotu_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_cdotu_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_cdotu_strided_batched
    end interface

    interface
        function rocblas_cdotc_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_cdotc_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_cdotc_strided_batched
    end interface

    interface
        function rocblas_zdotu_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_zdotu_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_zdotu_strided_batched
    end interface

    interface
        function rocblas_zdotc_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_zdotc_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_zdotc_strided_batched
    end interface

    ! swap
    interface
        function rocblas_sswap(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_sswap')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_sswap
    end interface

    interface
        function rocblas_dswap(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_dswap')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_dswap
    end interface

    interface
        function rocblas_cswap(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_cswap')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_cswap
    end interface

    interface
        function rocblas_zswap(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_zswap')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_zswap
    end interface

    ! swap_batched
    interface
        function rocblas_sswap_batched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sswap_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_sswap_batched
    end interface

    interface
        function rocblas_dswap_batched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dswap_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_dswap_batched
    end interface

    interface
        function rocblas_cswap_batched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cswap_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_cswap_batched
    end interface

    interface
        function rocblas_zswap_batched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zswap_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_zswap_batched
    end interface

    ! swap_strided_batched
    interface
        function rocblas_sswap_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sswap_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_sswap_strided_batched
    end interface

    interface
        function rocblas_dswap_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dswap_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_dswap_strided_batched
    end interface

    interface
        function rocblas_cswap_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cswap_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_cswap_strided_batched
    end interface

    interface
        function rocblas_zswap_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zswap_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_zswap_strided_batched
    end interface

    ! axpy
    interface
        function rocblas_haxpy(handle, n, alpha, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_haxpy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_haxpy
    end interface

    interface
        function rocblas_saxpy(handle, n, alpha, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_saxpy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_saxpy
    end interface

    interface
        function rocblas_daxpy(handle, n, alpha, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_daxpy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_daxpy
    end interface

    interface
        function rocblas_caxpy(handle, n, alpha, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_caxpy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_caxpy
    end interface

    interface
        function rocblas_zaxpy(handle, n, alpha, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_zaxpy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_zaxpy
    end interface

    ! axpy_batched
    interface
        function rocblas_haxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_haxpy_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_haxpy_batched
    end interface

    interface
        function rocblas_saxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_saxpy_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_saxpy_batched
    end interface

    interface
        function rocblas_daxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_daxpy_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_daxpy_batched
    end interface

    interface
        function rocblas_caxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_caxpy_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_caxpy_batched
    end interface

    interface
        function rocblas_zaxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zaxpy_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_zaxpy_batched
    end interface

    ! axpy_strided_batched
    interface
        function rocblas_haxpy_strided_batched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_haxpy_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_haxpy_strided_batched
    end interface

    interface
        function rocblas_saxpy_strided_batched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_saxpy_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_saxpy_strided_batched
    end interface

    interface
        function rocblas_daxpy_strided_batched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_daxpy_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_daxpy_strided_batched
    end interface

    interface
        function rocblas_caxpy_strided_batched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_caxpy_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_caxpy_strided_batched
    end interface

    interface
        function rocblas_zaxpy_strided_batched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zaxpy_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_zaxpy_strided_batched
    end interface

    ! asum
    interface
        function rocblas_sasum(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_sasum')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_sasum
    end interface

    interface
        function rocblas_dasum(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dasum')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_dasum
    end interface

    interface
        function rocblas_scasum(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_scasum')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_scasum
    end interface

    interface
        function rocblas_dzasum(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dzasum')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_dzasum
    end interface

    ! asum_batched
    interface
        function rocblas_sasum_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_sasum_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_sasum_batched
    end interface

    interface
        function rocblas_dasum_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dasum_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_dasum_batched
    end interface

    interface
        function rocblas_scasum_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_scasum_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_scasum_batched
    end interface

    interface
        function rocblas_dzasum_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dzasum_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_dzasum_batched
    end interface

    ! asum_strided_batched
    interface
        function rocblas_sasum_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_sasum_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_sasum_strided_batched
    end interface

    interface
        function rocblas_dasum_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dasum_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_dasum_strided_batched
    end interface

    interface
        function rocblas_scasum_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_scasum_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_scasum_strided_batched
    end interface

    interface
        function rocblas_dzasum_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dzasum_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_dzasum_strided_batched
    end interface

    ! nrm2
    interface
        function rocblas_snrm2(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_snrm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_snrm2
    end interface

    interface
        function rocblas_dnrm2(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dnrm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_dnrm2
    end interface

    interface
        function rocblas_scnrm2(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_scnrm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_scnrm2
    end interface

    interface
        function rocblas_dznrm2(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dznrm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_dznrm2
    end interface

    ! nrm2_batched
    interface
        function rocblas_snrm2_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_snrm2_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_snrm2_batched
    end interface

    interface
        function rocblas_dnrm2_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dnrm2_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_dnrm2_batched
    end interface

    interface
        function rocblas_scnrm2_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_scnrm2_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_scnrm2_batched
    end interface

    interface
        function rocblas_dznrm2_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dznrm2_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_dznrm2_batched
    end interface

    ! nrm2_strided_batched
    interface
        function rocblas_snrm2_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_snrm2_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_snrm2_strided_batched
    end interface

    interface
        function rocblas_dnrm2_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dnrm2_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_dnrm2_strided_batched
    end interface

    interface
        function rocblas_scnrm2_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_scnrm2_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_scnrm2_strided_batched
    end interface

    interface
        function rocblas_dznrm2_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dznrm2_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_dznrm2_strided_batched
    end interface

    ! amax
    interface
        function rocblas_isamax(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_isamax')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_isamax
    end interface

    interface
        function rocblas_idamax(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_idamax')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_idamax
    end interface

    interface
        function rocblas_icamax(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_icamax')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_icamax
    end interface

    interface
        function rocblas_izamax(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_izamax')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_izamax
    end interface

    ! amax_batched
    interface
        function rocblas_isamax_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_isamax_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_isamax_batched
    end interface

    interface
        function rocblas_idamax_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_idamax_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_idamax_batched
    end interface

    interface
        function rocblas_icamax_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_icamax_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_icamax_batched
    end interface

    interface
        function rocblas_izamax_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_izamax_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_izamax_batched
    end interface

    ! amax_strided_batched
    interface
        function rocblas_isamax_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_isamax_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_isamax_strided_batched
    end interface

    interface
        function rocblas_idamax_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_idamax_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_idamax_strided_batched
    end interface

    interface
        function rocblas_icamax_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_icamax_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_icamax_strided_batched
    end interface

    interface
        function rocblas_izamax_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_izamax_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_izamax_strided_batched
    end interface

    ! amin
    interface
        function rocblas_isamin(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_isamin')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_isamin
    end interface

    interface
        function rocblas_idamin(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_idamin')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_idamin
    end interface

    interface
        function rocblas_icamin(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_icamin')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_icamin
    end interface

    interface
        function rocblas_izamin(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_izamin')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_izamin
    end interface

    ! amin_batched
    interface
        function rocblas_isamin_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_isamin_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_isamin_batched
    end interface

    interface
        function rocblas_idamin_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_idamin_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_idamin_batched
    end interface

    interface
        function rocblas_icamin_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_icamin_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_icamin_batched
    end interface

    interface
        function rocblas_izamin_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_izamin_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_izamin_batched
    end interface

    ! amin_strided_batched
    interface
        function rocblas_isamin_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_isamin_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_isamin_strided_batched
    end interface

    interface
        function rocblas_idamin_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_idamin_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_idamin_strided_batched
    end interface

    interface
        function rocblas_icamin_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_icamin_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_icamin_strided_batched
    end interface

    interface
        function rocblas_izamin_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_izamin_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_izamin_strided_batched
    end interface

    ! rot
    interface
        function rocblas_srot(handle, n, x, incx, y, incy, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_srot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_srot
    end interface

    interface
        function rocblas_drot(handle, n, x, incx, y, incy, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_drot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_drot
    end interface

    interface
        function rocblas_crot(handle, n, x, incx, y, incy, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_crot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_crot
    end interface

    interface
        function rocblas_csrot(handle, n, x, incx, y, incy, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_csrot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_csrot
    end interface

    interface
        function rocblas_zrot(handle, n, x, incx, y, incy, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_zrot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_zrot
    end interface

    interface
        function rocblas_zdrot(handle, n, x, incx, y, incy, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_zdrot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_zdrot
    end interface

    ! rot_batched
    interface
        function rocblas_srot_batched(handle, n, x, incx, y, incy, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_srot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_srot_batched
    end interface

    interface
        function rocblas_drot_batched(handle, n, x, incx, y, incy, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_drot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_drot_batched
    end interface

    interface
        function rocblas_crot_batched(handle, n, x, incx, y, incy, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_crot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_crot_batched
    end interface

    interface
        function rocblas_csrot_batched(handle, n, x, incx, y, incy, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csrot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_csrot_batched
    end interface

    interface
        function rocblas_zrot_batched(handle, n, x, incx, y, incy, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zrot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_zrot_batched
    end interface

    interface
        function rocblas_zdrot_batched(handle, n, x, incx, y, incy, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zdrot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_zdrot_batched
    end interface

    ! rot_strided_batched
    interface
        function rocblas_srot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_srot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_srot_strided_batched
    end interface

    interface
        function rocblas_drot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_drot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_drot_strided_batched
    end interface

    interface
        function rocblas_crot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_crot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_crot_strided_batched
    end interface

    interface
        function rocblas_csrot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csrot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_csrot_strided_batched
    end interface

    interface
        function rocblas_zrot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zrot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_zrot_strided_batched
    end interface

    interface
        function rocblas_zdrot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zdrot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_zdrot_strided_batched
    end interface

    ! rotg
    interface
        function rocblas_srotg(handle, a, b, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_srotg')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_srotg
    end interface

    interface
        function rocblas_drotg(handle, a, b, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_drotg')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_drotg
    end interface

    interface
        function rocblas_crotg(handle, a, b, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_crotg')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_crotg
    end interface

    interface
        function rocblas_zrotg(handle, a, b, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_zrotg')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_zrotg
    end interface

    ! rotg_batched
    interface
        function rocblas_srotg_batched(handle, a, b, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_srotg_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_srotg_batched
    end interface

    interface
        function rocblas_drotg_batched(handle, a, b, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_drotg_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_drotg_batched
    end interface

    interface
        function rocblas_crotg_batched(handle, a, b, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_crotg_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_crotg_batched
    end interface

    interface
        function rocblas_zrotg_batched(handle, a, b, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zrotg_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_zrotg_batched
    end interface

    ! rotg_strided_batched
    interface
        function rocblas_srotg_strided_batched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_srotg_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            integer(c_int), value :: stride_a
            type(c_ptr), value :: b
            integer(c_int), value :: stride_b
            type(c_ptr), value :: c
            integer(c_int), value :: stride_c
            type(c_ptr), value :: s
            integer(c_int), value :: stride_s
            integer(c_int), value :: batch_count
        end function rocblas_srotg_strided_batched
    end interface

    interface
        function rocblas_drotg_strided_batched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_drotg_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            integer(c_int), value :: stride_a
            type(c_ptr), value :: b
            integer(c_int), value :: stride_b
            type(c_ptr), value :: c
            integer(c_int), value :: stride_c
            type(c_ptr), value :: s
            integer(c_int), value :: stride_s
            integer(c_int), value :: batch_count
        end function rocblas_drotg_strided_batched
    end interface

    interface
        function rocblas_crotg_strided_batched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_crotg_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            integer(c_int), value :: stride_a
            type(c_ptr), value :: b
            integer(c_int), value :: stride_b
            type(c_ptr), value :: c
            integer(c_int), value :: stride_c
            type(c_ptr), value :: s
            integer(c_int), value :: stride_s
            integer(c_int), value :: batch_count
        end function rocblas_crotg_strided_batched
    end interface

    interface
        function rocblas_zrotg_strided_batched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zrotg_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            integer(c_int), value :: stride_a
            type(c_ptr), value :: b
            integer(c_int), value :: stride_b
            type(c_ptr), value :: c
            integer(c_int), value :: stride_c
            type(c_ptr), value :: s
            integer(c_int), value :: stride_s
            integer(c_int), value :: batch_count
        end function rocblas_zrotg_strided_batched
    end interface

    ! rotm
    interface
        function rocblas_srotm(handle, n, x, incx, y, incy, param) &
                result(c_int) &
                bind(c, name = 'rocblas_srotm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: param
        end function rocblas_srotm
    end interface

    interface
        function rocblas_drotm(handle, n, x, incx, y, incy, param) &
                result(c_int) &
                bind(c, name = 'rocblas_drotm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: param
        end function rocblas_drotm
    end interface

    ! rotm_batched
    interface
        function rocblas_srotm_batched(handle, n, x, incx, y, incy, param, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_srotm_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: param
            integer(c_int), value :: batch_count
        end function rocblas_srotm_batched
    end interface

    interface
        function rocblas_drotm_batched(handle, n, x, incx, y, incy, param, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_drotm_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: param
            integer(c_int), value :: batch_count
        end function rocblas_drotm_batched
    end interface

    ! rotm_strided_batched
    interface
        function rocblas_srotm_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, param, stride_param, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_srotm_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            type(c_ptr), value :: param
            integer(c_int), value :: stride_param
            integer(c_int), value :: batch_count
        end function rocblas_srotm_strided_batched
    end interface

    interface
        function rocblas_drotm_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, param, stride_param, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_drotm_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: stride_y
            type(c_ptr), value :: param
            integer(c_int), value :: stride_param
            integer(c_int), value :: batch_count
        end function rocblas_drotm_strided_batched
    end interface

    ! rotmg
    interface
        function rocblas_srotmg(handle, d1, d2, x1, y1, param) &
                result(c_int) &
                bind(c, name = 'rocblas_srotmg')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: d1
            type(c_ptr), value :: d2
            type(c_ptr), value :: x1
            type(c_ptr), value :: y1
            type(c_ptr), value :: param
        end function rocblas_srotmg
    end interface

    interface
        function rocblas_drotmg(handle, d1, d2, x1, y1, param) &
                result(c_int) &
                bind(c, name = 'rocblas_drotmg')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: d1
            type(c_ptr), value :: d2
            type(c_ptr), value :: x1
            type(c_ptr), value :: y1
            type(c_ptr), value :: param
        end function rocblas_drotmg
    end interface

    ! rotmg_batched
    interface
        function rocblas_srotmg_batched(handle, d1, d2, x1, y1, param, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_srotmg_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: d1
            type(c_ptr), value :: d2
            type(c_ptr), value :: x1
            type(c_ptr), value :: y1
            type(c_ptr), value :: param
            integer(c_int), value :: batch_count
        end function rocblas_srotmg_batched
    end interface

    interface
        function rocblas_drotmg_batched(handle, d1, d2, x1, y1, param, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_drotmg_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: d1
            type(c_ptr), value :: d2
            type(c_ptr), value :: x1
            type(c_ptr), value :: y1
            type(c_ptr), value :: param
            integer(c_int), value :: batch_count
        end function rocblas_drotmg_batched
    end interface

    ! rotmg_strided_batched
    interface
        function rocblas_srotmg_strided_batched(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, &
             y1, stride_y1, param, stride_param, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_srotmg_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: d1
            integer(c_int), value :: stride_d1
            type(c_ptr), value :: d2
            integer(c_int), value :: stride_d2
            type(c_ptr), value :: x1
            integer(c_int), value :: stride_x1
            type(c_ptr), value :: y1
            integer(c_int), value :: stride_y1
            type(c_ptr), value :: param
            integer(c_int), value :: stride_param
            integer(c_int), value :: batch_count
        end function rocblas_srotmg_strided_batched
    end interface

    interface
        function rocblas_drotmg_strided_batched(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, &
             y1, stride_y1, param, stride_param, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_drotmg_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: d1
            integer(c_int), value :: stride_d1
            type(c_ptr), value :: d2
            integer(c_int), value :: stride_d2
            type(c_ptr), value :: x1
            integer(c_int), value :: stride_x1
            type(c_ptr), value :: y1
            integer(c_int), value :: stride_y1
            type(c_ptr), value :: param
            integer(c_int), value :: stride_param
            integer(c_int), value :: batch_count
        end function rocblas_drotmg_strided_batched
    end interface

    contains
     subroutine rocblasCheck(rocblasError_t)
       implicit none
       integer(kind(rocblas_status_success)) :: rocblasError_t
       if (rocblasError_t /= rocblas_status_success) then
          write(*,*) "ROCBLAS ERROR: Error code = ", rocblasError_t
          call exit(rocblasError_t)
       end if
     end subroutine rocblasCheck

end module rocblas