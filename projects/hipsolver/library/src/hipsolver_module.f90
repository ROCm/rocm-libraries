!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!                                 DEPRECATED
!
! These Fortran bindings are deprecated and may be removed in a future
! release. The hipfort project provides hipsolver Fortran bindings that
! are recommended for use instead.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module hipsolver_enums
    use iso_c_binding

    !---------------------!
    !   hipSOLVER types   !
    !---------------------!

    enum, bind(c)
        enumerator :: HIPSOLVER_OP_N = 111
        enumerator :: HIPSOLVER_OP_T = 112
        enumerator :: HIPSOLVER_OP_C = 113
    end enum

    enum, bind(c)
        enumerator :: HIPSOLVER_FILL_MODE_UPPER = 121
        enumerator :: HIPSOLVER_FILL_MODE_LOWER = 122
    end enum

    enum, bind(c)
        enumerator :: HIPSOLVER_SIDE_LEFT  = 141
        enumerator :: HIPSOLVER_SIDE_RIGHT = 142
    end enum

    enum, bind(c)
        enumerator :: HIPSOLVER_EIG_MODE_NOVECTOR = 201
        enumerator :: HIPSOLVER_EIG_MODE_VECTOR   = 202
    end enum

    enum, bind(c)
        enumerator :: HIPSOLVER_EIG_TYPE_1 = 211
        enumerator :: HIPSOLVER_EIG_TYPE_2 = 212
        enumerator :: HIPSOLVER_EIG_TYPE_3 = 213
    end enum

    enum, bind(c)
        enumerator :: HIPSOLVER_EIG_RANGE_ALL = 221
        enumerator :: HIPSOLVER_EIG_RANGE_V   = 222
        enumerator :: HIPSOLVER_EIG_RANGE_I   = 223
    end enum

    enum, bind(c)
        enumerator :: HIPSOLVER_STATUS_SUCCESS           = 0
        enumerator :: HIPSOLVER_STATUS_NOT_INITIALIZED   = 1
        enumerator :: HIPSOLVER_STATUS_ALLOC_FAILED      = 2
        enumerator :: HIPSOLVER_STATUS_INVALID_VALUE     = 3
        enumerator :: HIPSOLVER_STATUS_MAPPING_ERROR     = 4
        enumerator :: HIPSOLVER_STATUS_EXECUTION_FAILED  = 5
        enumerator :: HIPSOLVER_STATUS_INTERNAL_ERROR    = 6
        enumerator :: HIPSOLVER_STATUS_NOT_SUPPORTED     = 7
        enumerator :: HIPSOLVER_STATUS_ARCH_MISMATCH     = 8
        enumerator :: HIPSOLVER_STATUS_HANDLE_IS_NULLPTR = 9
        enumerator :: HIPSOLVER_STATUS_INVALID_ENUM      = 10
        enumerator :: HIPSOLVER_STATUS_UNKNOWN           = 11
    end enum

end module hipsolver_enums

module hipsolver
    use iso_c_binding

    !---------!
    !   Aux   !
    !---------!
    
    interface
        function hipsolverCreate(handle) &
                bind(c, name = 'hipsolverCreate')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCreate
            type(c_ptr), value :: handle
        end function hipsolverCreate
    end interface

    interface
        function hipsolverDestroy(handle) &
                bind(c, name = 'hipsolverDestroy')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDestroy
            type(c_ptr), value :: handle
        end function hipsolverDestroy
    end interface

    interface
        function hipsolverSetStream(handle, streamId) &
                bind(c, name = 'hipsolverSetStream')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSetStream
            type(c_ptr), value :: handle
            type(c_ptr), value :: streamId
        end function hipsolverSetStream
    end interface

    interface
        function hipsolverGetStream(handle, streamId) &
                bind(c, name = 'hipsolverGetStream')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverGetStream
            type(c_ptr), value :: handle
            type(c_ptr), value :: streamId
        end function hipsolverGetStream
    end interface
    
    ! ******************** GESVDJ PARAMS ********************
    interface
        function hipsolverCreateGesvdjInfo(info) &
                bind(c, name = 'hipsolverCreateGesvdjInfo')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCreateGesvdjInfo
            type(c_ptr), value :: info
        end function hipsolverCreateGesvdjInfo
    end interface
    
    interface
        function hipsolverDestroyGesvdjInfo(info) &
                bind(c, name = 'hipsolverDestroyGesvdjInfo')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDestroyGesvdjInfo
            type(c_ptr), value :: info
        end function hipsolverDestroyGesvdjInfo
    end interface
    
    interface
        function hipsolverXgesvdjSetMaxSweeps(info, max_sweeps) &
                bind(c, name = 'hipsolverXgesvdjSetMaxSweeps')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverXgesvdjSetMaxSweeps
            type(c_ptr), value :: info
            integer(c_int), value :: max_sweeps
        end function hipsolverXgesvdjSetMaxSweeps
    end interface
    
    interface
        function hipsolverXgesvdjSetSortEig(info, sort_eig) &
                bind(c, name = 'hipsolverXgesvdjSetSortEig')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverXgesvdjSetSortEig
            type(c_ptr), value :: info
            integer(c_int), value :: sort_eig
        end function hipsolverXgesvdjSetSortEig
    end interface
    
    interface
        function hipsolverXgesvdjSetTolerance(info, tolerance) &
                bind(c, name = 'hipsolverXgesvdjSetTolerance')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverXgesvdjSetTolerance
            type(c_ptr), value :: info
            real(c_double), value :: tolerance
        end function hipsolverXgesvdjSetTolerance
    end interface
    
    interface
        function hipsolverXgesvdjGetResidual(handle, info, residual) &
                bind(c, name = 'hipsolverXgesvdjGetResidual')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverXgesvdjGetResidual
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: residual
        end function hipsolverXgesvdjGetResidual
    end interface
    
    interface
        function hipsolverXgesvdjGetSweeps(handle, info, executed_sweeps) &
                bind(c, name = 'hipsolverXgesvdjGetSweeps')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverXgesvdjGetSweeps
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: executed_sweeps
        end function hipsolverXgesvdjGetSweeps
    end interface
    
    ! ******************** SYEVJ PARAMS ********************
    interface
        function hipsolverCreateSyevjInfo(info) &
                bind(c, name = 'hipsolverCreateSyevjInfo')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCreateSyevjInfo
            type(c_ptr), value :: info
        end function hipsolverCreateSyevjInfo
    end interface
    
    interface
        function hipsolverDestroySyevjInfo(info) &
                bind(c, name = 'hipsolverDestroySyevjInfo')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDestroySyevjInfo
            type(c_ptr), value :: info
        end function hipsolverDestroySyevjInfo
    end interface
    
    interface
        function hipsolverXsyevjSetMaxSweeps(info, max_sweeps) &
                bind(c, name = 'hipsolverXsyevjSetMaxSweeps')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverXsyevjSetMaxSweeps
            type(c_ptr), value :: info
            integer(c_int), value :: max_sweeps
        end function hipsolverXsyevjSetMaxSweeps
    end interface
    
    interface
        function hipsolverXsyevjSetSortEig(info, sort_eig) &
                bind(c, name = 'hipsolverXsyevjSetSortEig')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverXsyevjSetSortEig
            type(c_ptr), value :: info
            integer(c_int), value :: sort_eig
        end function hipsolverXsyevjSetSortEig
    end interface
    
    interface
        function hipsolverXsyevjSetTolerance(info, tolerance) &
                bind(c, name = 'hipsolverXsyevjSetTolerance')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverXsyevjSetTolerance
            type(c_ptr), value :: info
            real(c_double), value :: tolerance
        end function hipsolverXsyevjSetTolerance
    end interface
    
    interface
        function hipsolverXsyevjGetResidual(handle, info, residual) &
                bind(c, name = 'hipsolverXsyevjSetTolerance')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverXsyevjGetResidual
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: residual
        end function hipsolverXsyevjGetResidual
    end interface
    
    interface
        function hipsolverXsyevjGetSweeps(handle, info, executed_sweeps) &
                bind(c, name = 'hipsolverXsyevjSetTolerance')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverXsyevjGetSweeps
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: executed_sweeps
        end function hipsolverXsyevjGetSweeps
    end interface

    !------------!
    !   LAPACK   !
    !------------!
    
    ! ******************** ORGBR/UNGBR ********************
    interface
        function hipsolverSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverSorgbr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgbr_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverSorgbr_bufferSize
    end interface
    
    interface
        function hipsolverDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverDorgbr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgbr_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverDorgbr_bufferSize
    end interface
    
    interface
        function hipsolverCungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverCungbr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungbr_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverCungbr_bufferSize
    end interface
    
    interface
        function hipsolverZungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverZungbr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungbr_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverZungbr_bufferSize
    end interface
    
    interface
        function hipsolverSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverSorgbr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgbr
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSorgbr
    end interface
    
    interface
        function hipsolverDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverDorgbr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgbr
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDorgbr
    end interface
    
    interface
        function hipsolverCungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverCungbr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungbr
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCungbr
    end interface
    
    interface
        function hipsolverZungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverZungbr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungbr
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZungbr
    end interface
    
    ! ******************** ORGQR/UNGQR ********************
    interface
        function hipsolverSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverSorgqr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgqr_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverSorgqr_bufferSize
    end interface
    
    interface
        function hipsolverDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverDorgqr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgqr_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverDorgqr_bufferSize
    end interface
    
    interface
        function hipsolverCungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverCungqr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungqr_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverCungqr_bufferSize
    end interface
    
    interface
        function hipsolverZungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverZungqr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungqr_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverZungqr_bufferSize
    end interface
    
    interface
        function hipsolverSorgqr(handle, m, n, k, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverSorgqr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgqr
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSorgqr
    end interface
    
    interface
        function hipsolverDorgqr(handle, m, n, k, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverDorgqr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgqr
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDorgqr
    end interface
    
    interface
        function hipsolverCungqr(handle, m, n, k, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverCungqr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungqr
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCungqr
    end interface
    
    interface
        function hipsolverZungqr(handle, m, n, k, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverZungqr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungqr
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZungqr
    end interface
    
    ! ******************** ORGTR/UNGTR ********************
    interface
        function hipsolverSorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverSorgtr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgtr_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverSorgtr_bufferSize
    end interface
    
    interface
        function hipsolverDorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverDorgtr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgtr_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverDorgtr_bufferSize
    end interface
    
    interface
        function hipsolverCungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverCungtr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungtr_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverCungtr_bufferSize
    end interface
    
    interface
        function hipsolverZungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverZungtr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungtr_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverZungtr_bufferSize
    end interface
    
    interface
        function hipsolverSorgtr(handle, uplo, n, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverSorgtr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgtr
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSorgtr
    end interface
    
    interface
        function hipsolverDorgtr(handle, uplo, n, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverDorgtr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgtr
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDorgtr
    end interface
    
    interface
        function hipsolverCungtr(handle, uplo, n, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverCungtr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungtr
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCungtr
    end interface
    
    interface
        function hipsolverZungtr(handle, uplo, n, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverZungtr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungtr
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZungtr
    end interface
    
    ! ******************** ORMQR/UNMQR ********************
    interface
        function hipsolverSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork) &
                bind(c, name = 'hipsolverSormqr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormqr_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: lwork
        end function hipsolverSormqr_bufferSize
    end interface
    
    interface
        function hipsolverDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork) &
                bind(c, name = 'hipsolverDormqr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormqr_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: lwork
        end function hipsolverDormqr_bufferSize
    end interface
    
    interface
        function hipsolverCunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork) &
                bind(c, name = 'hipsolverCunmqr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmqr_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: lwork
        end function hipsolverCunmqr_bufferSize
    end interface
    
    interface
        function hipsolverZunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork) &
                bind(c, name = 'hipsolverZunmqr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmqr_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: lwork
        end function hipsolverZunmqr_bufferSize
    end interface
    
    interface
        function hipsolverSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info) &
                bind(c, name = 'hipsolverSormqr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormqr
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSormqr
    end interface
    
    interface
        function hipsolverDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info) &
                bind(c, name = 'hipsolverDormqr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormqr
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDormqr
    end interface
    
    interface
        function hipsolverCunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info) &
                bind(c, name = 'hipsolverCunmqr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmqr
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCunmqr
    end interface
    
    interface
        function hipsolverZunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info) &
                bind(c, name = 'hipsolverZunmqr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmqr
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZunmqr
    end interface
    
    ! ******************** ORMTR/UNMTR ********************
    interface
        function hipsolverSormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork) &
                bind(c, name = 'hipsolverSormtr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormtr_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: lwork
        end function hipsolverSormtr_bufferSize
    end interface
    
    interface
        function hipsolverDormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork) &
                bind(c, name = 'hipsolverDormtr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormtr_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: lwork
        end function hipsolverDormtr_bufferSize
    end interface
    
    interface
        function hipsolverCunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork) &
                bind(c, name = 'hipsolverCunmtr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmtr_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: lwork
        end function hipsolverCunmtr_bufferSize
    end interface
    
    interface
        function hipsolverZunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork) &
                bind(c, name = 'hipsolverZunmtr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmtr_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: lwork
        end function hipsolverZunmtr_bufferSize
    end interface
    
    interface
        function hipsolverSormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info) &
                bind(c, name = 'hipsolverSormtr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormtr
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSormtr
    end interface
    
    interface
        function hipsolverDormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info) &
                bind(c, name = 'hipsolverDormtr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormtr
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDormtr
    end interface
    
    interface
        function hipsolverCunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info) &
                bind(c, name = 'hipsolverCunmtr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmtr
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCunmtr
    end interface
    
    interface
        function hipsolverZunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info) &
                bind(c, name = 'hipsolverZunmtr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmtr
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZunmtr
    end interface
    
    ! ******************** GEBRD ********************
    interface
        function hipsolverSgebrd_bufferSize(handle, m, n, lwork) &
                bind(c, name = 'hipsolverSgebrd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgebrd_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: lwork
        end function hipsolverSgebrd_bufferSize
    end interface
    
    interface
        function hipsolverDgebrd_bufferSize(handle, m, n, lwork) &
                bind(c, name = 'hipsolverDgebrd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgebrd_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: lwork
        end function hipsolverDgebrd_bufferSize
    end interface
    
    interface
        function hipsolverCgebrd_bufferSize(handle, m, n, lwork) &
                bind(c, name = 'hipsolverCgebrd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgebrd_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: lwork
        end function hipsolverCgebrd_bufferSize
    end interface
    
    interface
        function hipsolverZgebrd_bufferSize(handle, m, n, lwork) &
                bind(c, name = 'hipsolverZgebrd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgebrd_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: lwork
        end function hipsolverZgebrd_bufferSize
    end interface

    interface
        function hipsolverSgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info) &
                bind(c, name = 'hipsolverSgebrd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgebrd
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tauq
            type(c_ptr), value :: taup
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSgebrd
    end interface

    interface
        function hipsolverDgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info) &
                bind(c, name = 'hipsolverDgebrd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgebrd
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tauq
            type(c_ptr), value :: taup
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDgebrd
    end interface

    interface
        function hipsolverCgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info) &
                bind(c, name = 'hipsolverCgebrd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgebrd
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tauq
            type(c_ptr), value :: taup
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCgebrd
    end interface

    interface
        function hipsolverZgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info) &
                bind(c, name = 'hipsolverZgebrd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgebrd
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tauq
            type(c_ptr), value :: taup
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZgebrd
    end interface
    
    ! ******************** GELS ********************
    interface
        function hipsolverSSgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork) &
                bind(c, name = 'hipsolverSSgels_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSSgels_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: lwork
        end function hipsolverSSgels_bufferSize
    end interface
    
    interface
        function hipsolverDDgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork) &
                bind(c, name = 'hipsolverDDgels_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDDgels_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: lwork
        end function hipsolverDDgels_bufferSize
    end interface
    
    interface
        function hipsolverCCgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork) &
                bind(c, name = 'hipsolverCCgels_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCCgels_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: lwork
        end function hipsolverCCgels_bufferSize
    end interface
    
    interface
        function hipsolverZZgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork) &
                bind(c, name = 'hipsolverZZgels_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZZgels_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: lwork
        end function hipsolverZZgels_bufferSize
    end interface

    interface
        function hipsolverSSgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info) &
                bind(c, name = 'hipsolverSSgels')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSSgels
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: niters
            type(c_ptr), value :: info
        end function hipsolverSSgels
    end interface

    interface
        function hipsolverDDgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info) &
                bind(c, name = 'hipsolverDDgels')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDDgels
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: niters
            type(c_ptr), value :: info
        end function hipsolverDDgels
    end interface

    interface
        function hipsolverCCgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info) &
                bind(c, name = 'hipsolverCCgels')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCCgels
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: niters
            type(c_ptr), value :: info
        end function hipsolverCCgels
    end interface

    interface
        function hipsolverZZgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info) &
                bind(c, name = 'hipsolverZZgels')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZZgels
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: niters
            type(c_ptr), value :: info
        end function hipsolverZZgels
    end interface
    
    ! ******************** GEQRF ********************
    interface
        function hipsolverSgeqrf_bufferSize(handle, m, n, A, lda, lwork) &
                bind(c, name = 'hipsolverSgeqrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgeqrf_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverSgeqrf_bufferSize
    end interface

    interface
        function hipsolverDgeqrf_bufferSize(handle, m, n, A, lda, lwork) &
                bind(c, name = 'hipsolverDgeqrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgeqrf_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverDgeqrf_bufferSize
    end interface

    interface
        function hipsolverCgeqrf_bufferSize(handle, m, n, A, lda, lwork) &
                bind(c, name = 'hipsolverCgeqrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgeqrf_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverCgeqrf_bufferSize
    end interface

    interface
        function hipsolverZgeqrf_bufferSize(handle, m, n, A, lda, lwork) &
                bind(c, name = 'hipsolverZgeqrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgeqrf_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverZgeqrf_bufferSize
    end interface

    interface
        function hipsolverSgeqrf(handle, m, n, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverSgeqrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgeqrf
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSgeqrf
    end interface

    interface
        function hipsolverDgeqrf(handle, m, n, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverDgeqrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgeqrf
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDgeqrf
    end interface

    interface
        function hipsolverCgeqrf(handle, m, n, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverCgeqrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgeqrf
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCgeqrf
    end interface

    interface
        function hipsolverZgeqrf(handle, m, n, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverZgeqrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgeqrf
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZgeqrf
    end interface

    ! ******************** GESV ********************
    interface
        function hipsolverSSgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork) &
                bind(c, name = 'hipsolverSSgesv_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSSgesv_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: lwork
        end function hipsolverSSgesv_bufferSize
    end interface
    
    interface
        function hipsolverDDgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork) &
                bind(c, name = 'hipsolverDDgesv_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDDgesv_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: lwork
        end function hipsolverDDgesv_bufferSize
    end interface
    
    interface
        function hipsolverCCgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork) &
                bind(c, name = 'hipsolverCCgesv_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCCgesv_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: lwork
        end function hipsolverCCgesv_bufferSize
    end interface
    
    interface
        function hipsolverZZgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork) &
                bind(c, name = 'hipsolverZZgesv_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZZgesv_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: lwork
        end function hipsolverZZgesv_bufferSize
    end interface

    interface
        function hipsolverSSgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info) &
                bind(c, name = 'hipsolverSSgesv')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSSgesv
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: niters
            type(c_ptr), value :: info
        end function hipsolverSSgesv
    end interface

    interface
        function hipsolverDDgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info) &
                bind(c, name = 'hipsolverDDgesv')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDDgesv
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: niters
            type(c_ptr), value :: info
        end function hipsolverDDgesv
    end interface

    interface
        function hipsolverCCgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info) &
                bind(c, name = 'hipsolverCCgesv')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCCgesv
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: niters
            type(c_ptr), value :: info
        end function hipsolverCCgesv
    end interface

    interface
        function hipsolverZZgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info) &
                bind(c, name = 'hipsolverZZgesv')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZZgesv
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: niters
            type(c_ptr), value :: info
        end function hipsolverZZgesv
    end interface
    
    ! ******************** GESVD ********************
    interface
        function hipsolverSgesvd_bufferSize(handle, jobu, jobv, m, n, lwork) &
                bind(c, name = 'hipsolverSgesvd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgesvd_bufferSize
            type(c_ptr), value :: handle
            integer(c_signed_char), value :: jobu
            integer(c_signed_char), value :: jobv
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: lwork
        end function hipsolverSgesvd_bufferSize
    end interface
    
    interface
        function hipsolverDgesvd_bufferSize(handle, jobu, jobv, m, n, lwork) &
                bind(c, name = 'hipsolverDgesvd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgesvd_bufferSize
            type(c_ptr), value :: handle
            integer(c_signed_char), value :: jobu
            integer(c_signed_char), value :: jobv
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: lwork
        end function hipsolverDgesvd_bufferSize
    end interface
    
    interface
        function hipsolverCgesvd_bufferSize(handle, jobu, jobv, m, n, lwork) &
                bind(c, name = 'hipsolverCgesvd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgesvd_bufferSize
            type(c_ptr), value :: handle
            integer(c_signed_char), value :: jobu
            integer(c_signed_char), value :: jobv
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: lwork
        end function hipsolverCgesvd_bufferSize
    end interface
    
    interface
        function hipsolverZgesvd_bufferSize(handle, jobu, jobv, m, n, lwork) &
                bind(c, name = 'hipsolverZgesvd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgesvd_bufferSize
            type(c_ptr), value :: handle
            integer(c_signed_char), value :: jobu
            integer(c_signed_char), value :: jobv
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: lwork
        end function hipsolverZgesvd_bufferSize
    end interface

    interface
        function hipsolverSgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info) &
                bind(c, name = 'hipsolverSgesvd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgesvd
            type(c_ptr), value :: handle
            integer(c_signed_char), value :: jobu
            integer(c_signed_char), value :: jobv
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: rwork
            type(c_ptr), value :: info
        end function hipsolverSgesvd
    end interface

    interface
        function hipsolverDgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info) &
                bind(c, name = 'hipsolverDgesvd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgesvd
            type(c_ptr), value :: handle
            integer(c_signed_char), value :: jobu
            integer(c_signed_char), value :: jobv
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: rwork
            type(c_ptr), value :: info
        end function hipsolverDgesvd
    end interface

    interface
        function hipsolverCgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info) &
                bind(c, name = 'hipsolverCgesvd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgesvd
            type(c_ptr), value :: handle
            integer(c_signed_char), value :: jobu
            integer(c_signed_char), value :: jobv
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: rwork
            type(c_ptr), value :: info
        end function hipsolverCgesvd
    end interface

    interface
        function hipsolverZgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info) &
                bind(c, name = 'hipsolverZgesvd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgesvd
            type(c_ptr), value :: handle
            integer(c_signed_char), value :: jobu
            integer(c_signed_char), value :: jobv
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: rwork
            type(c_ptr), value :: info
        end function hipsolverZgesvd
    end interface
    
    ! ******************** GESVDJ ********************
    interface
        function hipsolverSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params) &
                bind(c, name = 'hipsolverSgesvdj_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgesvdj_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(c_int), value :: econ
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
        end function hipsolverSgesvdj_bufferSize
    end interface
    
    interface
        function hipsolverDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params) &
                bind(c, name = 'hipsolverDgesvdj_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgesvdj_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(c_int), value :: econ
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
        end function hipsolverDgesvdj_bufferSize
    end interface
    
    interface
        function hipsolverCgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params) &
                bind(c, name = 'hipsolverCgesvdj_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgesvdj_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(c_int), value :: econ
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
        end function hipsolverCgesvdj_bufferSize
    end interface
    
    interface
        function hipsolverZgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params) &
                bind(c, name = 'hipsolverZgesvdj_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgesvdj_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(c_int), value :: econ
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
        end function hipsolverZgesvdj_bufferSize
    end interface

    interface
        function hipsolverSgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params) &
                bind(c, name = 'hipsolverSgesvdj')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgesvdj
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(c_int), value :: econ
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
        end function hipsolverSgesvdj
    end interface

    interface
        function hipsolverDgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params) &
                bind(c, name = 'hipsolverDgesvdj')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgesvdj
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(c_int), value :: econ
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
        end function hipsolverDgesvdj
    end interface

    interface
        function hipsolverCgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params) &
                bind(c, name = 'hipsolverCgesvdj')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgesvdj
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(c_int), value :: econ
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
        end function hipsolverCgesvdj
    end interface

    interface
        function hipsolverZgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params) &
                bind(c, name = 'hipsolverZgesvdj')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgesvdj
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(c_int), value :: econ
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
        end function hipsolverZgesvdj
    end interface
    
    ! ******************** GESVDJ_BATCHED ********************
    interface
        function hipsolverSgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batch_count) &
                bind(c, name = 'hipsolverSgesvdjBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgesvdjBatched_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
            integer(c_int), value :: batch_count
        end function hipsolverSgesvdjBatched_bufferSize
    end interface
    
    interface
        function hipsolverDgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batch_count) &
                bind(c, name = 'hipsolverDgesvdjBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgesvdjBatched_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
            integer(c_int), value :: batch_count
        end function hipsolverDgesvdjBatched_bufferSize
    end interface
    
    interface
        function hipsolverCgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batch_count) &
                bind(c, name = 'hipsolverCgesvdjBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgesvdjBatched_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
            integer(c_int), value :: batch_count
        end function hipsolverCgesvdjBatched_bufferSize
    end interface
    
    interface
        function hipsolverZgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batch_count) &
                bind(c, name = 'hipsolverZgesvdjBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgesvdjBatched_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
            integer(c_int), value :: batch_count
        end function hipsolverZgesvdjBatched_bufferSize
    end interface

    interface
        function hipsolverSgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batch_count) &
                bind(c, name = 'hipsolverSgesvdjBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgesvdjBatched
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
            integer(c_int), value :: batch_count
        end function hipsolverSgesvdjBatched
    end interface

    interface
        function hipsolverDgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batch_count) &
                bind(c, name = 'hipsolverDgesvdjBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgesvdjBatched
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
            integer(c_int), value :: batch_count
        end function hipsolverDgesvdjBatched
    end interface

    interface
        function hipsolverCgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batch_count) &
                bind(c, name = 'hipsolverCgesvdjBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgesvdjBatched
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
            integer(c_int), value :: batch_count
        end function hipsolverCgesvdjBatched
    end interface

    interface
        function hipsolverZgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batch_count) &
                bind(c, name = 'hipsolverZgesvdjBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgesvdjBatched
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
            integer(c_int), value :: batch_count
        end function hipsolverZgesvdjBatched
    end interface

    ! ******************** GETRF ********************
    interface
        function hipsolverSgetrf_bufferSize(handle, m, n, A, lda, lwork) &
                bind(c, name = 'hipsolverSgetrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrf_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverSgetrf_bufferSize
    end interface
    
    interface
        function hipsolverDgetrf_bufferSize(handle, m, n, A, lda, lwork) &
                bind(c, name = 'hipsolverDgetrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrf_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverDgetrf_bufferSize
    end interface
    
    interface
        function hipsolverCgetrf_bufferSize(handle, m, n, A, lda, lwork) &
                bind(c, name = 'hipsolverCgetrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrf_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverCgetrf_bufferSize
    end interface
    
    interface
        function hipsolverZgetrf_bufferSize(handle, m, n, A, lda, lwork) &
                bind(c, name = 'hipsolverZgetrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrf_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverZgetrf_bufferSize
    end interface

    interface
        function hipsolverSgetrf(handle, m, n, A, lda, work, lwork, ipiv, info) &
                bind(c, name = 'hipsolverSgetrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrf
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: info
        end function hipsolverSgetrf
    end interface
    
    interface
        function hipsolverDgetrf(handle, m, n, A, lda, work, lwork, ipiv, info) &
                bind(c, name = 'hipsolverDgetrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrf
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: info
        end function hipsolverDgetrf
    end interface
    
    interface
        function hipsolverCgetrf(handle, m, n, A, lda, work, lwork, ipiv, info) &
                bind(c, name = 'hipsolverCgetrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrf
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: info
        end function hipsolverCgetrf
    end interface
    
    interface
        function hipsolverZgetrf(handle, m, n, A, lda, work, lwork, ipiv, info) &
                bind(c, name = 'hipsolverZgetrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrf
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: info
        end function hipsolverZgetrf
    end interface

    ! ******************** GETRS ********************
    interface
        function hipsolverSgetrs_bufferSize(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork) &
                bind(c, name = 'hipsolverSgetrs_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrs_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
        end function hipsolverSgetrs_bufferSize
    end interface
    
    interface
        function hipsolverDgetrs_bufferSize(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork) &
                bind(c, name = 'hipsolverDgetrs_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrs_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
        end function hipsolverDgetrs_bufferSize
    end interface
    
    interface
        function hipsolverCgetrs_bufferSize(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork) &
                bind(c, name = 'hipsolverCgetrs_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrs_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
        end function hipsolverCgetrs_bufferSize
    end interface
    
    interface
        function hipsolverZgetrs_bufferSize(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork) &
                bind(c, name = 'hipsolverZgetrs_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrs_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
        end function hipsolverZgetrs_bufferSize
    end interface

    interface
        function hipsolverSgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info) &
                bind(c, name = 'hipsolverSgetrs')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrs
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSgetrs
    end interface
    
    interface
        function hipsolverDgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info) &
                bind(c, name = 'hipsolverDgetrs')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrs
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDgetrs
    end interface
    
    interface
        function hipsolverCgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info) &
                bind(c, name = 'hipsolverCgetrs')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrs
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCgetrs
    end interface
    
    interface
        function hipsolverZgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info) &
                bind(c, name = 'hipsolverZgetrs')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrs
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZgetrs
    end interface

    ! ******************** POTRF ********************
    interface
        function hipsolverSpotrf_bufferSize(handle, uplo, n, A, lda, lwork) &
                bind(c, name = 'hipsolverSpotrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrf_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverSpotrf_bufferSize
    end interface
    
    interface
        function hipsolverDpotrf_bufferSize(handle, uplo, n, A, lda, lwork) &
                bind(c, name = 'hipsolverDpotrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrf_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverDpotrf_bufferSize
    end interface
    
    interface
        function hipsolverCpotrf_bufferSize(handle, uplo, n, A, lda, lwork) &
                bind(c, name = 'hipsolverCpotrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrf_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverCpotrf_bufferSize
    end interface
    
    interface
        function hipsolverZpotrf_bufferSize(handle, uplo, n, A, lda, lwork) &
                bind(c, name = 'hipsolverZpotrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrf_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverZpotrf_bufferSize
    end interface

    interface
        function hipsolverSpotrf(handle, uplo, n, A, lda, work, lwork, info) &
                bind(c, name = 'hipsolverSpotrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrf
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSpotrf
    end interface

    interface
        function hipsolverDpotrf(handle, uplo, n, A, lda, work, lwork, info) &
                bind(c, name = 'hipsolverDpotrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrf
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDpotrf
    end interface

    interface
        function hipsolverCpotrf(handle, uplo, n, A, lda, work, lwork, info) &
                bind(c, name = 'hipsolverCpotrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrf
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCpotrf
    end interface

    interface
        function hipsolverZpotrf(handle, uplo, n, A, lda, work, lwork, info) &
                bind(c, name = 'hipsolverZpotrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrf
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZpotrf
    end interface
    
    ! ******************** POTRF_BATCHED ********************
    interface
        function hipsolverSpotrfBatched_bufferSize(handle, uplo, n, A, lda, lwork, batch_count) &
                bind(c, name = 'hipsolverSpotrfBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrfBatched_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
            integer(c_int), value :: batch_count
        end function hipsolverSpotrfBatched_bufferSize
    end interface
    
    interface
        function hipsolverDpotrfBatched_bufferSize(handle, uplo, n, A, lda, lwork, batch_count) &
                bind(c, name = 'hipsolverDpotrfBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrfBatched_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
            integer(c_int), value :: batch_count
        end function hipsolverDpotrfBatched_bufferSize
    end interface
    
    interface
        function hipsolverCpotrfBatched_bufferSize(handle, uplo, n, A, lda, lwork, batch_count) &
                bind(c, name = 'hipsolverCpotrfBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrfBatched_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
            integer(c_int), value :: batch_count
        end function hipsolverCpotrfBatched_bufferSize
    end interface
    
    interface
        function hipsolverZpotrfBatched_bufferSize(handle, uplo, n, A, lda, lwork, batch_count) &
                bind(c, name = 'hipsolverZpotrfBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrfBatched_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
            integer(c_int), value :: batch_count
        end function hipsolverZpotrfBatched_bufferSize
    end interface

    interface
        function hipsolverSpotrfBatched(handle, uplo, n, A, lda, work, lwork, info, batch_count) &
                bind(c, name = 'hipsolverSpotrfBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrfBatched
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipsolverSpotrfBatched
    end interface
    
    interface
        function hipsolverDpotrfBatched(handle, uplo, n, A, lda, work, lwork, info, batch_count) &
                bind(c, name = 'hipsolverDpotrfBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrfBatched
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipsolverDpotrfBatched
    end interface
    
    interface
        function hipsolverCpotrfBatched(handle, uplo, n, A, lda, work, lwork, info, batch_count) &
                bind(c, name = 'hipsolverCpotrfBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrfBatched
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipsolverCpotrfBatched
    end interface
    
    interface
        function hipsolverZpotrfBatched(handle, uplo, n, A, lda, work, lwork, info, batch_count) &
                bind(c, name = 'hipsolverZpotrfBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrfBatched
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipsolverZpotrfBatched
    end interface

    ! ******************** POTRI ********************
    interface
        function hipsolverSpotri_bufferSize(handle, uplo, n, A, lda, lwork) &
                bind(c, name = 'hipsolverSpotri_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotri_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverSpotri_bufferSize
    end interface
    
    interface
        function hipsolverDpotri_bufferSize(handle, uplo, n, A, lda, lwork) &
                bind(c, name = 'hipsolverDpotri_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotri_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverDpotri_bufferSize
    end interface
    
    interface
        function hipsolverCpotri_bufferSize(handle, uplo, n, A, lda, lwork) &
                bind(c, name = 'hipsolverCpotri_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotri_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverCpotri_bufferSize
    end interface
    
    interface
        function hipsolverZpotri_bufferSize(handle, uplo, n, A, lda, lwork) &
                bind(c, name = 'hipsolverZpotri_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotri_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverZpotri_bufferSize
    end interface

    interface
        function hipsolverSpotri(handle, uplo, n, A, lda, work, lwork, info) &
                bind(c, name = 'hipsolverSpotri')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotri
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSpotri
    end interface

    interface
        function hipsolverDpotri(handle, uplo, n, A, lda, work, lwork, info) &
                bind(c, name = 'hipsolverDpotri')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotri
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDpotri
    end interface

    interface
        function hipsolverCpotri(handle, uplo, n, A, lda, work, lwork, info) &
                bind(c, name = 'hipsolverCpotri')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotri
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCpotri
    end interface

    interface
        function hipsolverZpotri(handle, uplo, n, A, lda, work, lwork, info) &
                bind(c, name = 'hipsolverZpotri')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotri
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZpotri
    end interface

    ! ******************** POTRS ********************
    interface
        function hipsolverSpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork) &
                bind(c, name = 'hipsolverSpotrs_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrs_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
        end function hipsolverSpotrs_bufferSize
    end interface
    
    interface
        function hipsolverDpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork) &
                bind(c, name = 'hipsolverDpotrs_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrs_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
        end function hipsolverDpotrs_bufferSize
    end interface
    
    interface
        function hipsolverCpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork) &
                bind(c, name = 'hipsolverCpotrs_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrs_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
        end function hipsolverCpotrs_bufferSize
    end interface
    
    interface
        function hipsolverZpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork) &
                bind(c, name = 'hipsolverZpotrs_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrs_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
        end function hipsolverZpotrs_bufferSize
    end interface

    interface
        function hipsolverSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info) &
                bind(c, name = 'hipsolverSpotrs')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrs
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSpotrs
    end interface

    interface
        function hipsolverDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info) &
                bind(c, name = 'hipsolverDpotrs')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrs
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDpotrs
    end interface

    interface
        function hipsolverCpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info) &
                bind(c, name = 'hipsolverCpotrs')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrs
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCpotrs
    end interface

    interface
        function hipsolverZpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info) &
                bind(c, name = 'hipsolverZpotrs')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrs
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZpotrs
    end interface

    ! ******************** POTRS_BATCHED ********************
    interface
        function hipsolverSpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork, batch_count) &
                bind(c, name = 'hipsolverSpotrsBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrsBatched_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
            integer(c_int), value :: batch_count
        end function hipsolverSpotrsBatched_bufferSize
    end interface
    
    interface
        function hipsolverDpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork, batch_count) &
                bind(c, name = 'hipsolverDpotrsBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrsBatched_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
            integer(c_int), value :: batch_count
        end function hipsolverDpotrsBatched_bufferSize
    end interface
    
    interface
        function hipsolverCpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork, batch_count) &
                bind(c, name = 'hipsolverCpotrsBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrsBatched_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
            integer(c_int), value :: batch_count
        end function hipsolverCpotrsBatched_bufferSize
    end interface
    
    interface
        function hipsolverZpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork, batch_count) &
                bind(c, name = 'hipsolverZpotrsBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrsBatched_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
            integer(c_int), value :: batch_count
        end function hipsolverZpotrsBatched_bufferSize
    end interface

    interface
        function hipsolverSpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, batch_count) &
                bind(c, name = 'hipsolverSpotrsBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrsBatched
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipsolverSpotrsBatched
    end interface

    interface
        function hipsolverDpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, batch_count) &
                bind(c, name = 'hipsolverDpotrsBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrsBatched
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipsolverDpotrsBatched
    end interface

    interface
        function hipsolverCpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, batch_count) &
                bind(c, name = 'hipsolverCpotrsBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrsBatched
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipsolverCpotrsBatched
    end interface

    interface
        function hipsolverZpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, batch_count) &
                bind(c, name = 'hipsolverZpotrsBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrsBatched
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipsolverZpotrsBatched
    end interface

    ! ******************** SYEVD/HEEVD ********************
    interface
        function hipsolverSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork) &
                bind(c, name = 'hipsolverSsyevd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsyevd_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
        end function hipsolverSsyevd_bufferSize
    end interface
    
    interface
        function hipsolverDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork) &
                bind(c, name = 'hipsolverDsyevd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsyevd_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
        end function hipsolverDsyevd_bufferSize
    end interface
    
    interface
        function hipsolverCheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork) &
                bind(c, name = 'hipsolverCheevd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCheevd_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
        end function hipsolverCheevd_bufferSize
    end interface
    
    interface
        function hipsolverZheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork) &
                bind(c, name = 'hipsolverZheevd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZheevd_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
        end function hipsolverZheevd_bufferSize
    end interface

    interface
        function hipsolverSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info) &
                bind(c, name = 'hipsolverSsyevd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsyevd
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSsyevd
    end interface

    interface
        function hipsolverDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info) &
                bind(c, name = 'hipsolverDsyevd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsyevd
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDsyevd
    end interface

    interface
        function hipsolverCheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info) &
                bind(c, name = 'hipsolverCheevd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCheevd
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCheevd
    end interface

    interface
        function hipsolverZheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info) &
                bind(c, name = 'hipsolverZheevd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZheevd
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZheevd
    end interface

    ! ******************** SYEVJ/HEEVJ ********************
    interface
        function hipsolverSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params) &
                bind(c, name = 'hipsolverSsyevj_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsyevj_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
        end function hipsolverSsyevj_bufferSize
    end interface
    
    interface
        function hipsolverDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params) &
                bind(c, name = 'hipsolverDsyevj_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsyevj_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
        end function hipsolverDsyevj_bufferSize
    end interface
    
    interface
        function hipsolverCheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params) &
                bind(c, name = 'hipsolverCheevj_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCheevj_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
        end function hipsolverCheevj_bufferSize
    end interface
    
    interface
        function hipsolverZheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params) &
                bind(c, name = 'hipsolverZheevj_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZheevj_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
        end function hipsolverZheevj_bufferSize
    end interface

    interface
        function hipsolverSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params) &
                bind(c, name = 'hipsolverSsyevj')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsyevj
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
        end function hipsolverSsyevj
    end interface

    interface
        function hipsolverDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params) &
                bind(c, name = 'hipsolverDsyevj')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsyevj
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
        end function hipsolverDsyevj
    end interface

    interface
        function hipsolverCheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params) &
                bind(c, name = 'hipsolverCheevj')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCheevj
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
        end function hipsolverCheevj
    end interface

    interface
        function hipsolverZheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params) &
                bind(c, name = 'hipsolverZheevj')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZheevj
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
        end function hipsolverZheevj
    end interface

    ! ******************** SYEVJ_BATCHED/HEEVJ_BATCHED ********************
    interface
        function hipsolverSsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, batch_count) &
                bind(c, name = 'hipsolverSsyevjBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsyevjBatched_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
            integer(c_int), value :: batch_count
        end function hipsolverSsyevjBatched_bufferSize
    end interface
    
    interface
        function hipsolverDsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, batch_count) &
                bind(c, name = 'hipsolverDsyevjBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsyevjBatched_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
            integer(c_int), value :: batch_count
        end function hipsolverDsyevjBatched_bufferSize
    end interface
    
    interface
        function hipsolverCheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, batch_count) &
                bind(c, name = 'hipsolverCheevjBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCheevjBatched_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
            integer(c_int), value :: batch_count
        end function hipsolverCheevjBatched_bufferSize
    end interface
    
    interface
        function hipsolverZheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, batch_count) &
                bind(c, name = 'hipsolverZheevjBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZheevjBatched_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
            integer(c_int), value :: batch_count
        end function hipsolverZheevjBatched_bufferSize
    end interface

    interface
        function hipsolverSsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batch_count) &
                bind(c, name = 'hipsolverSsyevjBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsyevjBatched
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
            integer(c_int), value :: batch_count
        end function hipsolverSsyevjBatched
    end interface

    interface
        function hipsolverDsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batch_count) &
                bind(c, name = 'hipsolverDsyevjBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsyevjBatched
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
            integer(c_int), value :: batch_count
        end function hipsolverDsyevjBatched
    end interface

    interface
        function hipsolverCheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batch_count) &
                bind(c, name = 'hipsolverCheevjBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCheevjBatched
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
            integer(c_int), value :: batch_count
        end function hipsolverCheevjBatched
    end interface

    interface
        function hipsolverZheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batch_count) &
                bind(c, name = 'hipsolverZheevjBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZheevjBatched
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
            integer(c_int), value :: batch_count
        end function hipsolverZheevjBatched
    end interface

    ! ******************** SYGVD/HEGVD ********************
    interface
        function hipsolverSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork) &
                bind(c, name = 'hipsolverSsygvd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsygvd_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
        end function hipsolverSsygvd_bufferSize
    end interface
    
    interface
        function hipsolverDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork) &
                bind(c, name = 'hipsolverDsygvd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsygvd_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
        end function hipsolverDsygvd_bufferSize
    end interface
    
    interface
        function hipsolverChegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork) &
                bind(c, name = 'hipsolverChegvd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChegvd_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
        end function hipsolverChegvd_bufferSize
    end interface
    
    interface
        function hipsolverZhegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork) &
                bind(c, name = 'hipsolverZhegvd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhegvd_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
        end function hipsolverZhegvd_bufferSize
    end interface

    interface
        function hipsolverSsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info) &
                bind(c, name = 'hipsolverSsygvd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsygvd
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSsygvd
    end interface

    interface
        function hipsolverDsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info) &
                bind(c, name = 'hipsolverDsygvd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsygvd
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDsygvd
    end interface

    interface
        function hipsolverChegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info) &
                bind(c, name = 'hipsolverChegvd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChegvd
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverChegvd
    end interface

    interface
        function hipsolverZhegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info) &
                bind(c, name = 'hipsolverZhegvd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhegvd
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZhegvd
    end interface

    ! ******************** SYGVJ/HEGVJ ********************
    interface
        function hipsolverSsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params) &
                bind(c, name = 'hipsolverSsygvj_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsygvj_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
        end function hipsolverSsygvj_bufferSize
    end interface
    
    interface
        function hipsolverDsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params) &
                bind(c, name = 'hipsolverDsygvj_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsygvj_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
        end function hipsolverDsygvj_bufferSize
    end interface
    
    interface
        function hipsolverChegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params) &
                bind(c, name = 'hipsolverChegvj_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChegvj_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
        end function hipsolverChegvj_bufferSize
    end interface
    
    interface
        function hipsolverZhegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params) &
                bind(c, name = 'hipsolverZhegvj_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhegvj_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: W
            type(c_ptr), value :: lwork
            type(c_ptr), value :: params
        end function hipsolverZhegvj_bufferSize
    end interface

    interface
        function hipsolverSsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params) &
                bind(c, name = 'hipsolverSsygvj')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsygvj
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
        end function hipsolverSsygvj
    end interface

    interface
        function hipsolverDsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params) &
                bind(c, name = 'hipsolverDsygvj')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsygvj
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
        end function hipsolverDsygvj
    end interface

    interface
        function hipsolverChegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params) &
                bind(c, name = 'hipsolverChegvj')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChegvj
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
        end function hipsolverChegvj
    end interface

    interface
        function hipsolverZhegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params) &
                bind(c, name = 'hipsolverZhegvj')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhegvj
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: W
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            type(c_ptr), value :: params
        end function hipsolverZhegvj
    end interface

    ! ******************** SYTRD/HETRD ********************
    interface
        function hipsolverSsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork) &
                bind(c, name = 'hipsolverSsytrd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrd_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverSsytrd_bufferSize
    end interface
    
    interface
        function hipsolverDsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork) &
                bind(c, name = 'hipsolverDsytrd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrd_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverDsytrd_bufferSize
    end interface
    
    interface
        function hipsolverChetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork) &
                bind(c, name = 'hipsolverChetrd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChetrd_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverChetrd_bufferSize
    end interface
    
    interface
        function hipsolverZhetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork) &
                bind(c, name = 'hipsolverZhetrd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhetrd_bufferSize
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverZhetrd_bufferSize
    end interface

    interface
        function hipsolverSsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, info) &
                bind(c, name = 'hipsolverSsytrd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrd
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSsytrd
    end interface

    interface
        function hipsolverDsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, info) &
                bind(c, name = 'hipsolverDsytrd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrd
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDsytrd
    end interface

    interface
        function hipsolverChetrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, info) &
                bind(c, name = 'hipsolverChetrd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChetrd
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverChetrd
    end interface

    interface
        function hipsolverZhetrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, info) &
                bind(c, name = 'hipsolverZhetrd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhetrd
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZhetrd
    end interface

    ! ******************** SYTRF ********************
    interface
        function hipsolverSsytrf_bufferSize(handle, n, A, lda, lwork) &
                bind(c, name = 'hipsolverSsytrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrf_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverSsytrf_bufferSize
    end interface
    
    interface
        function hipsolverDsytrf_bufferSize(handle, n, A, lda, lwork) &
                bind(c, name = 'hipsolverDsytrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrf_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverDsytrf_bufferSize
    end interface
    
    interface
        function hipsolverCsytrf_bufferSize(handle, n, A, lda, lwork) &
                bind(c, name = 'hipsolverCsytrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCsytrf_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverCsytrf_bufferSize
    end interface
    
    interface
        function hipsolverZsytrf_bufferSize(handle, n, A, lda, lwork) &
                bind(c, name = 'hipsolverZsytrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZsytrf_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverZsytrf_bufferSize
    end interface

    interface
        function hipsolverSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info) &
                bind(c, name = 'hipsolverSsytrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrf
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSsytrf
    end interface

    interface
        function hipsolverDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info) &
                bind(c, name = 'hipsolverDsytrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrf
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDsytrf
    end interface

    interface
        function hipsolverCsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info) &
                bind(c, name = 'hipsolverCsytrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCsytrf
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCsytrf
    end interface

    interface
        function hipsolverZsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info) &
                bind(c, name = 'hipsolverZsytrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZsytrf
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZsytrf
    end interface
    
end module hipsolver
