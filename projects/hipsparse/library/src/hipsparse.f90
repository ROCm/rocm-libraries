!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2020 Advanced Micro Devices, Inc. All rights Reserved.
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

module hipsparse
    use hipsparse_enums
    use iso_c_binding
    implicit none

! ===========================================================================
!   auxiliary SPARSE
! ===========================================================================

    interface

!       hipsparseHandle_t
        function hipsparseCreate(handle) &
                bind(c, name = 'hipsparseCreate')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCreate
            type(c_ptr) :: handle
        end function hipsparseCreate

        function hipsparseDestroy(handle) &
                bind(c, name = 'hipsparseDestroy')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDestroy
            type(c_ptr), value :: handle
        end function hipsparseDestroy

!       hipsparseVersion
        function hipsparseGetVersion(handle, version) &
                bind(c, name = 'hipsparseGetVersion')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseGetVersion
            type(c_ptr), value :: handle
            integer(c_int) :: version
        end function hipsparseGetVersion

!       hipsparseStream
        function hipsparseSetStream(handle, stream) &
                bind(c, name = 'hipsparseSetStream')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSetStream
            type(c_ptr), value :: handle
            type(c_ptr), value :: stream
        end function hipsparseSetStream

        function hipsparseGetStream(handle, stream) &
                bind(c, name = 'hipsparseGetStream')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseGetStream
            type(c_ptr), value :: handle
            type(c_ptr) :: stream
        end function hipsparseGetStream

!       hipsparsePointerMode_t
        function hipsparseSetPointerMode(handle, mode) &
                bind(c, name = 'hipsparseSetPointerMode')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSetPointerMode
            type(c_ptr), value :: handle
            integer(c_int), value :: mode
        end function hipsparseSetPointerMode

        function hipsparseGetPointerMode(handle, mode) &
                bind(c, name = 'hipsparseGetPointerMode')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseGetPointerMode
            type(c_ptr), value :: handle
            integer(c_int) :: mode
        end function hipsparseGetPointerMode

!       hipsparseMatDescr_t
        function hipsparseCreateMatDescr(descr) &
                bind(c, name = 'hipsparseCreateMatDescr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCreateMatDescr
            type(c_ptr) :: descr
        end function hipsparseCreateMatDescr

        function hipsparseDestroyMatDescr(descr) &
                bind(c, name = 'hipsparseDestroyMatDescr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDestroyMatDescr
            type(c_ptr), value :: descr
        end function hipsparseDestroyMatDescr

        function hipsparseCopyMatDescr(dest, src) &
                bind(c, name = 'hipsparseCopyMatDescr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCopyMatDescr
            type(c_ptr), value :: dest
            type(c_ptr), intent(in), value :: src
        end function hipsparseCopyMatDescr

!       hipsparseMatrixType_t
        function hipsparseSetMatType(descr, mtype) &
                bind(c, name = 'hipsparseSetMatType')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSetMatType
            type(c_ptr), value :: descr
            integer(c_int), value :: mtype
        end function hipsparseSetMatType

        function hipsparseGetMatType(descr) &
                bind(c, name = 'hipsparseGetMatType')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseGetMatType
            type(c_ptr), intent(in), value :: descr
        end function hipsparseGetMatType

!       hipsparseFillMode_t
        function hipsparseSetMatFillMode(descr, fill) &
                bind(c, name = 'hipsparseSetMatFillMode')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSetMatFillMode
            type(c_ptr), value :: descr
            integer(c_int), value :: fill
        end function hipsparseSetMatFillMode

        function hipsparseGetMatFillMode(descr) &
                bind(c, name = 'hipsparseGetMatFillMode')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseGetMatFillMode
            type(c_ptr), intent(in), value :: descr
        end function hipsparseGetMatFillMode

!       hipsparseDiagType_t
        function hipsparseSetMatDiagType(descr, diag) &
                bind(c, name = 'hipsparseSetMatDiagType')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSetMatDiagType
            type(c_ptr), value :: descr
            integer(c_int), value :: diag
        end function hipsparseSetMatDiagType

        function hipsparseGetMatDiagType(descr) &
                bind(c, name = 'hipsparseGetMatDiagType')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseGetMatDiagType
            type(c_ptr), intent(in), value :: descr
        end function hipsparseGetMatDiagType

!       hipsparseIndexBase_t
        function hipsparseSetMatIndexBase(descr, base) &
                bind(c, name = 'hipsparseSetMatIndexBase')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSetMatIndexBase
            type(c_ptr), value :: descr
            integer(c_int), value :: base
        end function hipsparseSetMatIndexBase

        function hipsparseGetMatIndexBase(descr) &
                bind(c, name = 'hipsparseGetMatIndexBase')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseGetMatIndexBase
            type(c_ptr), intent(in), value :: descr
        end function hipsparseGetMatIndexBase

!       hipsparseHybMat_t
        function hipsparseCreateHybMat(hyb) &
                bind(c, name = 'hipsparseCreateHybMat')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCreateHybMat
            type(c_ptr) :: hyb
        end function hipsparseCreateHybMat

        function hipsparseDestroyHybMat(hyb) &
                bind(c, name = 'hipsparseDestroyHybMat')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDestroyHybMat
            type(c_ptr), value :: hyb
        end function hipsparseDestroyHybMat

!       csrsv2Info_t
        function hipsparseCreateCsrsv2Info(info) &
                bind(c, name = 'hipsparseCreateCsrsv2Info')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCreateCsrsv2Info
            type(c_ptr) :: info
        end function hipsparseCreateCsrsv2Info

        function hipsparseDestroyCsrsv2Info(info) &
                bind(c, name = 'hipsparseDestroyCsrsv2Info')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDestroyCsrsv2Info
            type(c_ptr), value :: info
        end function hipsparseDestroyCsrsv2Info

!       csrsm2Info_t
        function hipsparseCreateCsrsm2Info(info) &
                bind(c, name = 'hipsparseCreateCsrsm2Info')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCreateCsrsm2Info
            type(c_ptr) :: info
        end function hipsparseCreateCsrsm2Info

        function hipsparseDestroyCsrsm2Info(info) &
                bind(c, name = 'hipsparseDestroyCsrsm2Info')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDestroyCsrsm2Info
            type(c_ptr), value :: info
        end function hipsparseDestroyCsrsm2Info

!       bsrilu02Info_t
        function hipsparseCreateBsrilu02Info(info) &
                bind(c, name = 'hipsparseCreateBsrilu02Info')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCreateBsrilu02Info
            type(c_ptr) :: info
        end function hipsparseCreateBsrilu02Info

        function hipsparseDestroyBsrilu02Info(info) &
                bind(c, name = 'hipsparseDestroyBsrilu02Info')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDestroyBsrilu02Info
            type(c_ptr), value :: info
        end function hipsparseDestroyBsrilu02Info

!       csrilu02Info_t
        function hipsparseCreateCsrilu02Info(info) &
                bind(c, name = 'hipsparseCreateCsrilu02Info')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCreateCsrilu02Info
            type(c_ptr) :: info
        end function hipsparseCreateCsrilu02Info

        function hipsparseDestroyCsrilu02Info(info) &
                bind(c, name = 'hipsparseDestroyCsrilu02Info')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDestroyCsrilu02Info
            type(c_ptr), value :: info
        end function hipsparseDestroyCsrilu02Info

!       bsric02Info_t
        function hipsparseCreateBsric02Info(info) &
                bind(c, name = 'hipsparseCreateBsric02Info')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCreateBsric02Info
            type(c_ptr) :: info
        end function hipsparseCreateBsric02Info

        function hipsparseDestroyBsric02Info(info) &
                bind(c, name = 'hipsparseDestroyBsric02Info')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDestroyBsric02Info
            type(c_ptr), value :: info
        end function hipsparseDestroyBsric02Info

!       csric02Info_t
        function hipsparseCreateCsric02Info(info) &
                bind(c, name = 'hipsparseCreateCsric02Info')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCreateCsric02Info
            type(c_ptr) :: info
        end function hipsparseCreateCsric02Info

        function hipsparseDestroyCsric02Info(info) &
                bind(c, name = 'hipsparseDestroyCsric02Info')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDestroyCsric02Info
            type(c_ptr), value :: info
        end function hipsparseDestroyCsric02Info

!       csrgemm2Info_t
        function hipsparseCreateCsrgemm2Info(info) &
                bind(c, name = 'hipsparseCreateCsrgemm2Info')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCreateCsrgemm2Info
            type(c_ptr) :: info
        end function hipsparseCreateCsrgemm2Info

        function hipsparseDestroyCsrgemm2Info(info) &
                bind(c, name = 'hipsparseDestroyCsrgemm2Info')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDestroyCsrgemm2Info
            type(c_ptr), value :: info
        end function hipsparseDestroyCsrgemm2Info

! ===========================================================================
!   level 1 SPARSE
! ===========================================================================

!       hipsparseXaxpyi
        function hipsparseSaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase) &
                bind(c, name = 'hipsparseSaxpyi')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSaxpyi
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
        end function hipsparseSaxpyi

        function hipsparseDaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase) &
                bind(c, name = 'hipsparseDaxpyi')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDaxpyi
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
        end function hipsparseDaxpyi

        function hipsparseCaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase) &
                bind(c, name = 'hipsparseCaxpyi')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCaxpyi
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
        end function hipsparseCaxpyi

        function hipsparseZaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase) &
                bind(c, name = 'hipsparseZaxpyi')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZaxpyi
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
        end function hipsparseZaxpyi

!       hipsparseXdoti
        function hipsparseSdoti(handle, nnz, xVal, xInd, y, result, idxBase) &
                bind(c, name = 'hipsparseSdoti')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSdoti
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idxBase
        end function hipsparseSdoti

        function hipsparseDdoti(handle, nnz, xVal, xInd, y, result, idxBase) &
                bind(c, name = 'hipsparseDdoti')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDdoti
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idxBase
        end function hipsparseDdoti

        function hipsparseCdoti(handle, nnz, xVal, xInd, y, result, idxBase) &
                bind(c, name = 'hipsparseCdoti')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCdoti
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idxBase
        end function hipsparseCdoti

        function hipsparseZdoti(handle, nnz, xVal, xInd, y, result, idxBase) &
                bind(c, name = 'hipsparseZdoti')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZdoti
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idxBase
        end function hipsparseZdoti

!       hipsparseXdotci
        function hipsparseCdotci(handle, nnz, xVal, xInd, y, result, idxBase) &
                bind(c, name = 'hipsparseCdotci')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCdotci
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idxBase
        end function hipsparseCdotci

        function hipsparseZdotci(handle, nnz, xVal, xInd, y, result, idxBase) &
                bind(c, name = 'hipsparseZdotci')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZdotci
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idxBase
        end function hipsparseZdotci

!       hipsparseXgthr
        function hipsparseSgthr(handle, nnz, y, xVal, xInd, idxBase) &
                bind(c, name = 'hipsparseSgthr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSgthr
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            integer(c_int), value :: idxBase
        end function hipsparseSgthr

        function hipsparseDgthr(handle, nnz, y, xVal, xInd, idxBase) &
                bind(c, name = 'hipsparseDgthr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDgthr
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            integer(c_int), value :: idxBase
        end function hipsparseDgthr

        function hipsparseCgthr(handle, nnz, y, xVal, xInd, idxBase) &
                bind(c, name = 'hipsparseCgthr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCgthr
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            integer(c_int), value :: idxBase
        end function hipsparseCgthr

        function hipsparseZgthr(handle, nnz, y, xVal, xInd, idxBase) &
                bind(c, name = 'hipsparseZgthr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZgthr
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            integer(c_int), value :: idxBase
        end function hipsparseZgthr

!       hipsparseXgthrz
        function hipsparseSgthrz(handle, nnz, y, xVal, xInd, idxBase) &
                bind(c, name = 'hipsparseSgthrz')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSgthrz
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), value :: y
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            integer(c_int), value :: idxBase
        end function hipsparseSgthrz

        function hipsparseDgthrz(handle, nnz, y, xVal, xInd, idxBase) &
                bind(c, name = 'hipsparseDgthrz')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDgthrz
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), value :: y
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            integer(c_int), value :: idxBase
        end function hipsparseDgthrz

        function hipsparseCgthrz(handle, nnz, y, xVal, xInd, idxBase) &
                bind(c, name = 'hipsparseCgthrz')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCgthrz
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), value :: y
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            integer(c_int), value :: idxBase
        end function hipsparseCgthrz

        function hipsparseZgthrz(handle, nnz, y, xVal, xInd, idxBase) &
                bind(c, name = 'hipsparseZgthrz')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZgthrz
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), value :: y
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            integer(c_int), value :: idxBase
        end function hipsparseZgthrz

!       hipsparseXroti
        function hipsparseSroti(handle, nnz, xVal, xInd, y, c, s, idxBase) &
                bind(c, name = 'hipsparseSroti')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSroti
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            type(c_ptr), intent(in), value :: c
            type(c_ptr), intent(in), value :: s
            integer(c_int), value :: idxBase
        end function hipsparseSroti

        function hipsparseDroti(handle, nnz, xVal, xInd, y, c, s, idxBase) &
                bind(c, name = 'hipsparseDroti')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDroti
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            type(c_ptr), intent(in), value :: c
            type(c_ptr), intent(in), value :: s
            integer(c_int), value :: idxBase
        end function hipsparseDroti

!       hipsparseXsctr
        function hipsparseSsctr(handle, nnz, xVal, xInd, y, idxBase) &
                bind(c, name = 'hipsparseSsctr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSsctr
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
        end function hipsparseSsctr

        function hipsparseDsctr(handle, nnz, xVal, xInd, y, idxBase) &
                bind(c, name = 'hipsparseDsctr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDsctr
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
        end function hipsparseDsctr

        function hipsparseCsctr(handle, nnz, xVal, xInd, y, idxBase) &
                bind(c, name = 'hipsparseCsctr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCsctr
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
        end function hipsparseCsctr

        function hipsparseZsctr(handle, nnz, xVal, xInd, y, idxBase) &
                bind(c, name = 'hipsparseZsctr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZsctr
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
        end function hipsparseZsctr

! ===========================================================================
!   level 2 SPARSE
! ===========================================================================

!       hipsparseXcsrmv
        function hipsparseScsrmv(handle, trans, m, n, nnz, alpha, descr, csrVal, &
                csrRowPtr, csrColInd, x, beta, y) &
                bind(c, name = 'hipsparseScsrmv')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrmv
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseScsrmv

        function hipsparseDcsrmv(handle, trans, m, n, nnz, alpha, descr, csrVal, &
                csrRowPtr, csrColInd, x, beta, y) &
                bind(c, name = 'hipsparseDcsrmv')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrmv
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseDcsrmv

        function hipsparseCcsrmv(handle, trans, m, n, nnz, alpha, descr, csrVal, &
                csrRowPtr, csrColInd, x, beta, y) &
                bind(c, name = 'hipsparseCcsrmv')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrmv
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseCcsrmv

        function hipsparseZcsrmv(handle, trans, m, n, nnz, alpha, descr, csrVal, &
                csrRowPtr, csrColInd, x, beta, y) &
                bind(c, name = 'hipsparseZcsrmv')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrmv
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseZcsrmv

!       hipsparseXcsrsv2_zeroPivot
        function hipsparseXcsrsv2_zeroPivot(handle, info, position) &
                bind(c, name = 'hipsparseXcsrsv2_zeroPivot')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXcsrsv2_zeroPivot
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function hipsparseXcsrsv2_zeroPivot

!       hipsparseXcsrsv2_bufferSize
        function hipsparseScsrsv2_bufferSize(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseScsrsv2_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrsv2_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseScsrsv2_bufferSize

        function hipsparseDcsrsv2_bufferSize(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseDcsrsv2_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrsv2_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseDcsrsv2_bufferSize

        function hipsparseCcsrsv2_bufferSize(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseCcsrsv2_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrsv2_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseCcsrsv2_bufferSize

        function hipsparseZcsrsv2_bufferSize(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseZcsrsv2_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrsv2_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseZcsrsv2_bufferSize

!       hipsparseXcsrsv2_bufferSizeExt
        function hipsparseScsrsv2_bufferSizeExt(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseScsrsv2_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrsv2_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseScsrsv2_bufferSizeExt

        function hipsparseDcsrsv2_bufferSizeExt(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseDcsrsv2_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrsv2_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseDcsrsv2_bufferSizeExt

        function hipsparseCcsrsv2_bufferSizeExt(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseCcsrsv2_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrsv2_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseCcsrsv2_bufferSizeExt

        function hipsparseZcsrsv2_bufferSizeExt(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseZcsrsv2_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrsv2_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseZcsrsv2_bufferSizeExt

!       hipsparseXcsrsv2_analysis
        function hipsparseScsrsv2_analysis(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseScsrsv2_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrsv2_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseScsrsv2_analysis

        function hipsparseDcsrsv2_analysis(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseDcsrsv2_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrsv2_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDcsrsv2_analysis

        function hipsparseCcsrsv2_analysis(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseCcsrsv2_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrsv2_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCcsrsv2_analysis

        function hipsparseZcsrsv2_analysis(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseZcsrsv2_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrsv2_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZcsrsv2_analysis

!       hipsparseXcsrsv2_solve
        function hipsparseScsrsv2_solve(handle, trans, m, nnz, alpha, descr, csrVal, &
                csrRowPtr, csrColInd, info, x, y, policy, buffer) &
                bind(c, name = 'hipsparseScsrsv2_solve')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrsv2_solve
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), value :: y
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseScsrsv2_solve

        function hipsparseDcsrsv2_solve(handle, trans, m, nnz, alpha, descr, csrVal, &
                csrRowPtr, csrColInd, info, x, y, policy, buffer) &
                bind(c, name = 'hipsparseDcsrsv2_solve')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrsv2_solve
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), value :: y
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDcsrsv2_solve

        function hipsparseCcsrsv2_solve(handle, trans, m, nnz, alpha, descr, csrVal, &
                csrRowPtr, csrColInd, info, x, y, policy, buffer) &
                bind(c, name = 'hipsparseCcsrsv2_solve')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrsv2_solve
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), value :: y
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCcsrsv2_solve

        function hipsparseZcsrsv2_solve(handle, trans, m, nnz, alpha, descr, csrVal, &
                csrRowPtr, csrColInd, info, x, y, policy, buffer) &
                bind(c, name = 'hipsparseZcsrsv2_solve')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrsv2_solve
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), value :: y
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZcsrsv2_solve

!       hipsparseXhybmv
        function hipsparseShybmv(handle, trans, alpha, descr, hyb, x, beta, y) &
                bind(c, name = 'hipsparseShybmv')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseShybmv
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseShybmv

        function hipsparseDhybmv(handle, trans, alpha, descr, hyb, x, beta, y) &
                bind(c, name = 'hipsparseDhybmv')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDhybmv
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseDhybmv

        function hipsparseChybmv(handle, trans, alpha, descr, hyb, x, beta, y) &
                bind(c, name = 'hipsparseChybmv')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseChybmv
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseChybmv

        function hipsparseZhybmv(handle, trans, alpha, descr, hyb, x, beta, y) &
                bind(c, name = 'hipsparseZhybmv')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZhybmv
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseZhybmv

!       hipsparseXbsrmv
        function hipsparseSbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y) &
                bind(c, name = 'hipsparseSbsrmv')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSbsrmv
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseSbsrmv

        function hipsparseDbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y) &
                bind(c, name = 'hipsparseDbsrmv')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDbsrmv
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseDbsrmv

        function hipsparseCbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y) &
                bind(c, name = 'hipsparseCbsrmv')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCbsrmv
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseCbsrmv

        function hipsparseZbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y) &
                bind(c, name = 'hipsparseZbsrmv')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZbsrmv
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseZbsrmv


!       hipsparseXbsrxmv
        function hipsparseSbsrxmv(handle, dir, trans, sizeOfMask, &
                mb, nb, nnzb, alpha, descr, bsrVal, &
                bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y) &
                bind(c, name = 'hipsparseSbsrxmv')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSbsrxmv
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: sizeOfMask
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrMaskPtr
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrEndPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseSbsrxmv

        function hipsparseDbsrxmv(handle, dir, trans, sizeOfMask, &
                mb, nb, nnzb, alpha, descr, bsrVal, &
                bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y) &
                bind(c, name = 'hipsparseDbsrxmv')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDbsrxmv
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: sizeOfMask
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrMaskPtr
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrEndPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseDbsrxmv

        function hipsparseCbsrxmv(handle, dir, trans, sizeOfMask, &
                mb, nb, nnzb, alpha, descr, bsrVal, &
                bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y) &
                bind(c, name = 'hipsparseCbsrxmv')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCbsrxmv
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: sizeOfMask
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrMaskPtr
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrEndPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseCbsrxmv

        function hipsparseZbsrxmv(handle, dir, trans, sizeOfMask, &
                mb, nb, nnzb, alpha, descr, bsrVal, &
                bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim, x, beta, y) &
                bind(c, name = 'hipsparseZbsrxmv')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZbsrxmv
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: sizeOfMask
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrMaskPtr
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrEndPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseZbsrxmv

        function hipsparseSgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSizeInBytes) &
                bind(c, name = 'hipsparseSgemvi_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSgemvi_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), value :: pBufferSizeInBytes
        end function hipsparseSgemvi_bufferSize

        function hipsparseDgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSizeInBytes) &
                bind(c, name = 'hipsparseDgemvi_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDgemvi_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), value :: pBufferSizeInBytes
        end function hipsparseDgemvi_bufferSize

        function hipsparseCgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSizeInBytes) &
                bind(c, name = 'hipsparseCgemvi_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCgemvi_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), value :: pBufferSizeInBytes
        end function hipsparseCgemvi_bufferSize

        function hipsparseZgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSizeInBytes) &
                bind(c, name = 'hipsparseZgemvi_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZgemvi_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), value :: pBufferSizeInBytes
        end function hipsparseZgemvi_bufferSize

        function hipsparseSgemvi(handle, transA, m, n, alpha, A, lda, nnz, x, xInd, &
                beta, y, idxBase, pBuffer) &
                bind(c, name = 'hipsparseSgemvi')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSgemvi
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
            type(c_ptr), value :: pBuffer
        end function hipsparseSgemvi

        function hipsparseDgemvi(handle, transA, m, n, alpha, A, lda, nnz, x, xInd, &
                beta, y, idxBase, pBuffer) &
                bind(c, name = 'hipsparseDgemvi')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDgemvi
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
            type(c_ptr), value :: pBuffer
        end function hipsparseDgemvi

        function hipsparseCgemvi(handle, transA, m, n, alpha, A, lda, nnz, x, xInd, &
                beta, y, idxBase, pBuffer) &
                bind(c, name = 'hipsparseCgemvi')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCgemvi
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
            type(c_ptr), value :: pBuffer
        end function hipsparseCgemvi

        function hipsparseZgemvi(handle, transA, m, n, alpha, A, lda, nnz, x, xInd, &
                beta, y, idxBase, pBuffer) &
                bind(c, name = 'hipsparseZgemvi')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZgemvi
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
            type(c_ptr), value :: pBuffer
        end function hipsparseZgemvi

! ===========================================================================
!   level 3 SPARSE
! ===========================================================================

!       hipsparseXbsrmm
        function hipsparseSbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, &
                bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc) &
                bind(c, name = 'hipsparseSbsrmm')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSbsrmm
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: mb
            integer(c_int), value :: n
            integer(c_int), value :: kb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: bsrValA
            type(c_ptr), intent(in), value :: bsrRowPtrA
            type(c_ptr), intent(in), value :: bsrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseSbsrmm

        function hipsparseDbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, &
                bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc) &
                bind(c, name = 'hipsparseDbsrmm')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDbsrmm
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: mb
            integer(c_int), value :: n
            integer(c_int), value :: kb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: bsrValA
            type(c_ptr), intent(in), value :: bsrRowPtrA
            type(c_ptr), intent(in), value :: bsrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseDbsrmm

        function hipsparseCbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, &
                bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc) &
                bind(c, name = 'hipsparseCbsrmm')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCbsrmm
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: mb
            integer(c_int), value :: n
            integer(c_int), value :: kb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: bsrValA
            type(c_ptr), intent(in), value :: bsrRowPtrA
            type(c_ptr), intent(in), value :: bsrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseCbsrmm

        function hipsparseZbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, &
                bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc) &
                bind(c, name = 'hipsparseZbsrmm')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZbsrmm
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: mb
            integer(c_int), value :: n
            integer(c_int), value :: kb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: bsrValA
            type(c_ptr), intent(in), value :: bsrRowPtrA
            type(c_ptr), intent(in), value :: bsrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseZbsrmm

!       hipsparseXcsrmm
        function hipsparseScsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, &
                csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc) &
                bind(c, name = 'hipsparseScsrmm')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrmm
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseScsrmm

        function hipsparseDcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, &
                csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc) &
                bind(c, name = 'hipsparseDcsrmm')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrmm
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseDcsrmm

        function hipsparseCcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, &
                csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc) &
                bind(c, name = 'hipsparseCcsrmm')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrmm
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseCcsrmm

        function hipsparseZcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, &
                csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc) &
                bind(c, name = 'hipsparseZcsrmm')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrmm
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseZcsrmm

!       hipsparseXcsrmm2
        function hipsparseScsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, &
                csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc) &
                bind(c, name = 'hipsparseScsrmm2')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrmm2
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseScsrmm2

        function hipsparseDcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, &
                csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc) &
                bind(c, name = 'hipsparseDcsrmm2')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrmm2
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseDcsrmm2

        function hipsparseCcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, &
                csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc) &
                bind(c, name = 'hipsparseCcsrmm2')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrmm2
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseCcsrmm2

        function hipsparseZcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, &
                csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc) &
                bind(c, name = 'hipsparseZcsrmm2')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrmm2
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseZcsrmm2

!       hipsparseXcsrsm2_zeroPivot
        function hipsparseXcsrsm2_zeroPivot(handle, info, position) &
                bind(c, name = 'hipsparseXcsrsm2_zeroPivot')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXcsrsm2_zeroPivot
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function hipsparseXcsrsm2_zeroPivot

!       hipsparseXcsrsm2_bufferSizeExt
        function hipsparseScsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, &
                nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, &
                policy, bufferSize) &
                bind(c, name = 'hipsparseScsrsm2_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrsm2_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: bufferSize
        end function hipsparseScsrsm2_bufferSizeExt

        function hipsparseDcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, &
                nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, &
                policy, bufferSize) &
                bind(c, name = 'hipsparseDcsrsm2_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrsm2_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: bufferSize
        end function hipsparseDcsrsm2_bufferSizeExt

        function hipsparseCcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, &
                nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, &
                policy, bufferSize) &
                bind(c, name = 'hipsparseCcsrsm2_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrsm2_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: bufferSize
        end function hipsparseCcsrsm2_bufferSizeExt

        function hipsparseZcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, &
                nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, &
                policy, bufferSize) &
                bind(c, name = 'hipsparseZcsrsm2_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrsm2_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: bufferSize
        end function hipsparseZcsrsm2_bufferSizeExt

!       hipsparseXcsrsm2_analysis
        function hipsparseScsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, &
                alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, &
                buffer) &
                bind(c, name = 'hipsparseScsrsm2_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrsm2_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseScsrsm2_analysis

        function hipsparseDcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, &
                alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, &
                buffer) &
                bind(c, name = 'hipsparseDcsrsm2_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrsm2_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDcsrsm2_analysis

        function hipsparseCcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, &
                alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, &
                buffer) &
                bind(c, name = 'hipsparseCcsrsm2_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrsm2_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCcsrsm2_analysis

        function hipsparseZcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, &
                alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, &
                buffer) &
                bind(c, name = 'hipsparseZcsrsm2_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrsm2_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZcsrsm2_analysis

!       hipsparseXcsrsm2_solve
        function hipsparseScsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, &
                alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, &
                buffer) &
                bind(c, name = 'hipsparseScsrsm2_solve')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrsm2_solve
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseScsrsm2_solve

        function hipsparseDcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, &
                alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, &
                buffer) &
                bind(c, name = 'hipsparseDcsrsm2_solve')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrsm2_solve
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDcsrsm2_solve

        function hipsparseCcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, &
                alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, &
                buffer) &
                bind(c, name = 'hipsparseCcsrsm2_solve')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrsm2_solve
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCcsrsm2_solve

        function hipsparseZcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, &
                alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, &
                buffer) &
                bind(c, name = 'hipsparseZcsrsm2_solve')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrsm2_solve
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZcsrsm2_solve

!       hipsparseXgemmi
        function hipsparseSgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, &
                cscColPtrB, cscRowIndB, beta, C, ldc) &
                bind(c, name = 'hipsparseSgemmi')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSgemmi
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

        function hipsparseDgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, &
                cscColPtrB, cscRowIndB, beta, C, ldc) &
                bind(c, name = 'hipsparseDgemmi')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDgemmi
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
        end function hipsparseDgemmi

        function hipsparseCgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, &
                cscColPtrB, cscRowIndB, beta, C, ldc) &
                bind(c, name = 'hipsparseCgemmi')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCgemmi
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
        end function hipsparseCgemmi

        function hipsparseZgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, &
                cscColPtrB, cscRowIndB, beta, C, ldc) &
                bind(c, name = 'hipsparseZgemmi')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZgemmi
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
        end function hipsparseZgemmi

! ===========================================================================
!   extra SPARSE
! ===========================================================================

!       hipsparseXcsrgeamNnz
        function hipsparseXcsrgeamNnz(handle, m, n, descrA, nnzA, csrRowPtrA, &
                csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrC, &
                csrRowPtrC, nnzTotalDevHostPtr) &
                bind(c, name = 'hipsparseXcsrgeamNnz')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXcsrgeamNnz
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: nnzTotalDevHostPtr
        end function hipsparseXcsrgeamNnz

!       hipsparseXcsrgeam
        function hipsparseScsrgeam(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC) &
                bind(c, name = 'hipsparseScsrgeam')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrgeam
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseScsrgeam

        function hipsparseDcsrgeam(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC) &
                bind(c, name = 'hipsparseDcsrgeam')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrgeam
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseDcsrgeam

        function hipsparseCcsrgeam(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC) &
                bind(c, name = 'hipsparseCcsrgeam')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrgeam
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseCcsrgeam

        function hipsparseZcsrgeam(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC) &
                bind(c, name = 'hipsparseZcsrgeam')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrgeam
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseZcsrgeam

!       hipsparseXcsrgeam2_bufferSizeExt
        function hipsparseScsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, bufferSize) &
                bind(c, name = 'hipsparseScsrgeam2_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrgeam2_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), intent(in), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), intent(in), value :: csrColIndC
            type(c_ptr), value :: bufferSize
        end function hipsparseScsrgeam2_bufferSizeExt

        function hipsparseDcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, bufferSize) &
                bind(c, name = 'hipsparseDcsrgeam2_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrgeam2_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), intent(in), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), intent(in), value :: csrColIndC
            type(c_ptr), value :: bufferSize
        end function hipsparseDcsrgeam2_bufferSizeExt

        function hipsparseCcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, bufferSize) &
                bind(c, name = 'hipsparseCcsrgeam2_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrgeam2_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), intent(in), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), intent(in), value :: csrColIndC
            type(c_ptr), value :: bufferSize
        end function hipsparseCcsrgeam2_bufferSizeExt

        function hipsparseZcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, bufferSize) &
                bind(c, name = 'hipsparseZcsrgeam2_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrgeam2_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), intent(in), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), intent(in), value :: csrColIndC
            type(c_ptr), value :: bufferSize
        end function hipsparseZcsrgeam2_bufferSizeExt

!       hipsparseXcsrgeam2Nnz
        function hipsparseXcsrgeam2Nnz(handle, m, n, descrA, nnzA, csrRowPtrA, &
                csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrC, &
                csrRowPtrC, nnzTotalDevHostPtr, workspace) &
                bind(c, name = 'hipsparseXcsrgeam2Nnz')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXcsrgeam2Nnz
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: nnzTotalDevHostPtr
            type(c_ptr), value :: workspace
        end function hipsparseXcsrgeam2Nnz

!       hipsparseXcsrgeam2
        function hipsparseScsrgeam2(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, buffer) &
                bind(c, name = 'hipsparseScsrgeam2')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrgeam2
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), value :: buffer
        end function hipsparseScsrgeam2

        function hipsparseDcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, buffer) &
                bind(c, name = 'hipsparseDcsrgeam2')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrgeam2
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), value :: buffer
        end function hipsparseDcsrgeam2

        function hipsparseCcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, buffer) &
                bind(c, name = 'hipsparseCcsrgeam2')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrgeam2
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), value :: buffer
        end function hipsparseCcsrgeam2

        function hipsparseZcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, buffer) &
                bind(c, name = 'hipsparseZcsrgeam2')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrgeam2
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), value :: buffer
        end function hipsparseZcsrgeam2

!       hipsparseXcsrgemmNnz
        function hipsparseXcsrgemmNnz(handle, transA, transB, m, n, k, descrA, nnzA, &
                csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrC, &
                csrRowPtrC, nnzTotalDevHostPtr) &
                bind(c, name = 'hipsparseXcsrgemmNnz')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXcsrgemmNnz
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: nnzTotalDevHostPtr
        end function hipsparseXcsrgemmNnz

!       hipsparseScsrgemm
        function hipsparseScsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, &
                csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC) &
                bind(c, name = 'hipsparseScsrgemm')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrgemm
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseScsrgemm

        function hipsparseDcsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, &
                csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC) &
                bind(c, name = 'hipsparseDcsrgemm')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrgemm
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseDcsrgemm

        function hipsparseCcsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, &
                csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC) &
                bind(c, name = 'hipsparseCcsrgemm')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrgemm
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseCcsrgemm

        function hipsparseZcsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, &
                csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC) &
                bind(c, name = 'hipsparseZcsrgemm')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrgemm
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseZcsrgemm

!       hipsparseXcsrgemm2_bufferSizeExt
        function hipsparseScsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, &
                nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, &
                beta, descrD, nnzD, csrRowPtrD, csrColIndD, info, bufferSize) &
                bind(c, name = 'hipsparseScsrgemm2_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrgemm2_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseScsrgemm2_bufferSizeExt

        function hipsparseDcsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, &
                nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, &
                beta, descrD, nnzD, csrRowPtrD, csrColIndD, info, bufferSize) &
                bind(c, name = 'hipsparseDcsrgemm2_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrgemm2_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseDcsrgemm2_bufferSizeExt

        function hipsparseCcsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, &
                nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, &
                beta, descrD, nnzD, csrRowPtrD, csrColIndD, info, bufferSize) &
                bind(c, name = 'hipsparseCcsrgemm2_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrgemm2_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseCcsrgemm2_bufferSizeExt

        function hipsparseZcsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, &
                nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, &
                beta, descrD, nnzD, csrRowPtrD, csrColIndD, info, bufferSize) &
                bind(c, name = 'hipsparseZcsrgemm2_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrgemm2_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseZcsrgemm2_bufferSizeExt

!       hipsparseXcsrgemm2Nnz
        function hipsparseXcsrgemm2Nnz(handle, m, n, k, descrA, nnzA, csrRowPtrA, &
                csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrD, nnzD, &
                csrRowPtrD, csrColIndD, descrC, csrRowPtrC, nnzTotalDevHostPtr, info, &
                buffer) &
                bind(c, name = 'hipsparseXcsrgemm2Nnz')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXcsrgemm2Nnz
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: nnzTotalDevHostPtr
            type(c_ptr), intent(in), value :: info
            type(c_ptr), value :: buffer
        end function hipsparseXcsrgemm2Nnz

!       hipsparseXcsrgemm2
        function hipsparseScsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, &
                beta, descrD, nnzD, csrValD, csrRowPtrD, csrColIndD, descrC, csrValC, &
                csrRowPtrC, csrColIndC, info, buffer) &
                bind(c, name = 'hipsparseScsrgemm2')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrgemm2
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrValD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), intent(in), value :: info
            type(c_ptr), value :: buffer
        end function hipsparseScsrgemm2

        function hipsparseDcsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, &
                beta, descrD, nnzD, csrValD, csrRowPtrD, csrColIndD, descrC, csrValC, &
                csrRowPtrC, csrColIndC, info, buffer) &
                bind(c, name = 'hipsparseDcsrgemm2')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrgemm2
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrValD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), intent(in), value :: info
            type(c_ptr), value :: buffer
        end function hipsparseDcsrgemm2

        function hipsparseCcsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, &
                beta, descrD, nnzD, csrValD, csrRowPtrD, csrColIndD, descrC, csrValC, &
                csrRowPtrC, csrColIndC, info, buffer) &
                bind(c, name = 'hipsparseCcsrgemm2')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrgemm2
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrValD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), intent(in), value :: info
            type(c_ptr), value :: buffer
        end function hipsparseCcsrgemm2

        function hipsparseZcsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, &
                beta, descrD, nnzD, csrValD, csrRowPtrD, csrColIndD, descrC, csrValC, &
                csrRowPtrC, csrColIndC, info, buffer) &
                bind(c, name = 'hipsparseZcsrgemm2')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrgemm2
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrValD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), intent(in), value :: info
            type(c_ptr), value :: buffer
        end function hipsparseZcsrgemm2

! ===========================================================================
!   preconditioner SPARSE
! ===========================================================================

!       hipsparseXbsrilu02_zeroPivot
        function hipsparseXbsrilu02_zeroPivot(handle, info, position) &
                bind(c, name = 'hipsparseXbsrilu02_zeroPivot')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXbsrilu02_zeroPivot
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function hipsparseXbsrilu02_zeroPivot

!       hipsparseXbsrilu02_bufferSize
        function hipsparseSbsrilu02_bufferSize(handle, dir, mb, nnzb, descr, bsrVal, &
                bsrRowPtr, bsrColInd, blockDim, info, bufferSize) &
                bind(c, name = 'hipsparseSbsrilu02_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSbsrilu02_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseSbsrilu02_bufferSize

        function hipsparseDbsrilu02_bufferSize(handle, dir, mb, nnzb, descr, bsrVal, &
                bsrRowPtr, bsrColInd, blockDim, info, bufferSize) &
                bind(c, name = 'hipsparseDbsrilu02_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDbsrilu02_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseDbsrilu02_bufferSize

        function hipsparseCbsrilu02_bufferSize(handle, dir, mb, nnzb, descr, bsrVal, &
                bsrRowPtr, bsrColInd, blockDim, info, bufferSize) &
                bind(c, name = 'hipsparseCbsrilu02_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCbsrilu02_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseCbsrilu02_bufferSize

        function hipsparseZbsrilu02_bufferSize(handle, dir, mb, nnzb, descr, bsrVal, &
                bsrRowPtr, bsrColInd, blockDim, info, bufferSize) &
                bind(c, name = 'hipsparseZbsrilu02_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZbsrilu02_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseZbsrilu02_bufferSize

!       hipsparseXbsrilu02_analysis
        function hipsparseSbsrilu02_analysis(handle, dir, mb, nnzb, descr, bsrVal, &
                bsrRowPtr, bsrColInd, blockDim, info, policy, buffer) &
                bind(c, name = 'hipsparseSbsrilu02_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSbsrilu02_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseSbsrilu02_analysis

        function hipsparseDbsrilu02_analysis(handle, dir, mb, nnzb, descr, bsrVal, &
                bsrRowPtr, bsrColInd, blockDim, info, policy, buffer) &
                bind(c, name = 'hipsparseDbsrilu02_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDbsrilu02_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDbsrilu02_analysis

        function hipsparseCbsrilu02_analysis(handle, dir, mb, nnzb, descr, bsrVal, &
                bsrRowPtr, bsrColInd, blockDim, info, policy, buffer) &
                bind(c, name = 'hipsparseCbsrilu02_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCbsrilu02_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCbsrilu02_analysis

        function hipsparseZbsrilu02_analysis(handle, dir, mb, nnzb, descr, bsrVal, &
                bsrRowPtr, bsrColInd, blockDim, info, policy, buffer) &
                bind(c, name = 'hipsparseZbsrilu02_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZbsrilu02_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZbsrilu02_analysis

!       hipsparseXbsrilu02
        function hipsparseSbsrilu02(handle, dir, mb, nnzb, descr, bsrVal, bsrRowPtr, &
                bsrColInd, blockDim, info, policy, buffer) &
                bind(c, name = 'hipsparseSbsrilu02')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSbsrilu02
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseSbsrilu02

        function hipsparseDbsrilu02(handle, dir, mb, nnzb, descr, bsrVal, bsrRowPtr, &
                bsrColInd, blockDim, info, policy, buffer) &
                bind(c, name = 'hipsparseDbsrilu02')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDbsrilu02
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDbsrilu02

        function hipsparseCbsrilu02(handle, dir, mb, nnzb, descr, bsrVal, bsrRowPtr, &
                bsrColInd, blockDim, info, policy, buffer) &
                bind(c, name = 'hipsparseCbsrilu02')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCbsrilu02
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCbsrilu02

        function hipsparseZbsrilu02(handle, dir, mb, nnzb, descr, bsrVal, bsrRowPtr, &
                bsrColInd, blockDim, info, policy, buffer) &
                bind(c, name = 'hipsparseZbsrilu02')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZbsrilu02
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZbsrilu02

!       hipsparseXcsrilu02_zeroPivot
        function hipsparseXcsrilu02_zeroPivot(handle, info, position) &
                bind(c, name = 'hipsparseXcsrilu02_zeroPivot')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXcsrilu02_zeroPivot
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function hipsparseXcsrilu02_zeroPivot

!       hipsparseXcsrilu02_bufferSize
        function hipsparseScsrilu02_bufferSize(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseScsrilu02_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrilu02_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseScsrilu02_bufferSize

        function hipsparseDcsrilu02_bufferSize(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseDcsrilu02_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrilu02_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseDcsrilu02_bufferSize

        function hipsparseCcsrilu02_bufferSize(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseCcsrilu02_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrilu02_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseCcsrilu02_bufferSize

        function hipsparseZcsrilu02_bufferSize(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseZcsrilu02_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrilu02_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseZcsrilu02_bufferSize

!       hipsparseXcsrilu02_bufferSizeExt
        function hipsparseScsrilu02_bufferSizeExt(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseScsrilu02_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrilu02_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseScsrilu02_bufferSizeExt

        function hipsparseDcsrilu02_bufferSizeExt(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseDcsrilu02_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrilu02_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseDcsrilu02_bufferSizeExt

        function hipsparseCcsrilu02_bufferSizeExt(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseCcsrilu02_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrilu02_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseCcsrilu02_bufferSizeExt

        function hipsparseZcsrilu02_bufferSizeExt(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseZcsrilu02_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrilu02_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseZcsrilu02_bufferSizeExt

!       hipsparseXcsrilu02_analysis
        function hipsparseScsrilu02_analysis(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseScsrilu02_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrilu02_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseScsrilu02_analysis

        function hipsparseDcsrilu02_analysis(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseDcsrilu02_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrilu02_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDcsrilu02_analysis

        function hipsparseCcsrilu02_analysis(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseCcsrilu02_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrilu02_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCcsrilu02_analysis

        function hipsparseZcsrilu02_analysis(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseZcsrilu02_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrilu02_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZcsrilu02_analysis

!       hipsparseXcsrilu02
        function hipsparseScsrilu02(handle, m, nnz, descr, csrVal, csrRowPtr, &
                csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseScsrilu02')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsrilu02
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseScsrilu02

        function hipsparseDcsrilu02(handle, m, nnz, descr, csrVal, csrRowPtr, &
                csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseDcsrilu02')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsrilu02
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDcsrilu02

        function hipsparseCcsrilu02(handle, m, nnz, descr, csrVal, csrRowPtr, &
                csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseCcsrilu02')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsrilu02
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCcsrilu02

        function hipsparseZcsrilu02(handle, m, nnz, descr, csrVal, csrRowPtr, &
                csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseZcsrilu02')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsrilu02
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZcsrilu02

!       hipsparseXbsric02_zeroPivot
        function hipsparseXbsric02_zeroPivot(handle, info, position) &
                bind(c, name = 'hipsparseXbsric02_zeroPivot')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXbsric02_zeroPivot
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function hipsparseXbsric02_zeroPivot

!       hipsparseXbsric02_bufferSize
        function hipsparseSbsric02_bufferSize(handle, dir, mb, nnzb, descr, bsrVal, &
                bsrRowPtr, bsrColInd, blockDim, info, bufferSize) &
                bind(c, name = 'hipsparseSbsric02_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSbsric02_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseSbsric02_bufferSize

        function hipsparseDbsric02_bufferSize(handle, dir, mb, nnzb, descr, bsrVal, &
                bsrRowPtr, bsrColInd, blockDim, info, bufferSize) &
                bind(c, name = 'hipsparseDbsric02_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDbsric02_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseDbsric02_bufferSize

        function hipsparseCbsric02_bufferSize(handle, dir, mb, nnzb, descr, bsrVal, &
                bsrRowPtr, bsrColInd, blockDim, info, bufferSize) &
                bind(c, name = 'hipsparseCbsric02_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCbsric02_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseCbsric02_bufferSize

        function hipsparseZbsric02_bufferSize(handle, dir, mb, nnzb, descr, bsrVal, &
                bsrRowPtr, bsrColInd, blockDim, info, bufferSize) &
                bind(c, name = 'hipsparseZbsric02_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZbsric02_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseZbsric02_bufferSize

!       hipsparseXbsric02_analysis
        function hipsparseSbsric02_analysis(handle, dir, mb, nnzb, descr, bsrVal, &
                bsrRowPtr, bsrColInd, blockDim, info, policy, buffer) &
                bind(c, name = 'hipsparseSbsric02_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSbsric02_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseSbsric02_analysis

        function hipsparseDbsric02_analysis(handle, dir, mb, nnzb, descr, bsrVal, &
                bsrRowPtr, bsrColInd, blockDim, info, policy, buffer) &
                bind(c, name = 'hipsparseDbsric02_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDbsric02_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDbsric02_analysis

        function hipsparseCbsric02_analysis(handle, dir, mb, nnzb, descr, bsrVal, &
                bsrRowPtr, bsrColInd, blockDim, info, policy, buffer) &
                bind(c, name = 'hipsparseCbsric02_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCbsric02_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCbsric02_analysis

        function hipsparseZbsric02_analysis(handle, dir, mb, nnzb, descr, bsrVal, &
                bsrRowPtr, bsrColInd, blockDim, info, policy, buffer) &
                bind(c, name = 'hipsparseZbsric02_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZbsric02_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZbsric02_analysis

!       hipsparseXbsric02
        function hipsparseSbsric02(handle, dir, mb, nnzb, descr, bsrVal, bsrRowPtr, &
                bsrColInd, blockDim, info, policy, buffer) &
                bind(c, name = 'hipsparseSbsric02')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSbsric02
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseSbsric02

        function hipsparseDbsric02(handle, dir, mb, nnzb, descr, bsrVal, bsrRowPtr, &
                bsrColInd, blockDim, info, policy, buffer) &
                bind(c, name = 'hipsparseDbsric02')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDbsric02
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDbsric02

        function hipsparseCbsric02(handle, dir, mb, nnzb, descr, bsrVal, bsrRowPtr, &
                bsrColInd, blockDim, info, policy, buffer) &
                bind(c, name = 'hipsparseCbsric02')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCbsric02
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCbsric02

        function hipsparseZbsric02(handle, dir, mb, nnzb, descr, bsrVal, bsrRowPtr, &
                bsrColInd, blockDim, info, policy, buffer) &
                bind(c, name = 'hipsparseZbsric02')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZbsric02
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZbsric02

!       hipsparseXcsric02_zeroPivot
        function hipsparseXcsric02_zeroPivot(handle, info, position) &
                bind(c, name = 'hipsparseXcsric02_zeroPivot')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXcsric02_zeroPivot
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function hipsparseXcsric02_zeroPivot

!       hipsparseXcsric02_bufferSize
        function hipsparseScsric02_bufferSize(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseScsric02_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsric02_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseScsric02_bufferSize

        function hipsparseDcsric02_bufferSize(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseDcsric02_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsric02_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseDcsric02_bufferSize

        function hipsparseCcsric02_bufferSize(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseCcsric02_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsric02_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseCcsric02_bufferSize

        function hipsparseZcsric02_bufferSize(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseZcsric02_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsric02_bufferSize
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseZcsric02_bufferSize

!       hipsparseXcsric02_bufferSizeExt
        function hipsparseScsric02_bufferSizeExt(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseScsric02_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsric02_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseScsric02_bufferSizeExt

        function hipsparseDcsric02_bufferSizeExt(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseDcsric02_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsric02_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseDcsric02_bufferSizeExt

        function hipsparseCcsric02_bufferSizeExt(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseCcsric02_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsric02_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseCcsric02_bufferSizeExt

        function hipsparseZcsric02_bufferSizeExt(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                bind(c, name = 'hipsparseZcsric02_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsric02_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseZcsric02_bufferSizeExt

!       hipsparseXcsric02_analysis
        function hipsparseScsric02_analysis(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseScsric02_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsric02_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseScsric02_analysis

        function hipsparseDcsric02_analysis(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseDcsric02_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsric02_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDcsric02_analysis

        function hipsparseCcsric02_analysis(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseCcsric02_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsric02_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCcsric02_analysis

        function hipsparseZcsric02_analysis(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseZcsric02_analysis')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsric02_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZcsric02_analysis

!       hipsparseXcsric02
        function hipsparseScsric02(handle, m, nnz, descr, csrVal, csrRowPtr, &
                csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseScsric02')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsric02
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseScsric02

        function hipsparseDcsric02(handle, m, nnz, descr, csrVal, csrRowPtr, &
                csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseDcsric02')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsric02
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDcsric02

        function hipsparseCcsric02(handle, m, nnz, descr, csrVal, csrRowPtr, &
                csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseCcsric02')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsric02
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCcsric02

        function hipsparseZcsric02(handle, m, nnz, descr, csrVal, csrRowPtr, &
                csrColInd, info, policy, buffer) &
                bind(c, name = 'hipsparseZcsric02')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsric02
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZcsric02

! ===========================================================================
!   conversion SPARSE
! ===========================================================================

!       hipsparseXnnz
        function hipsparseSnnz(handle, dir, m, n, descrA, A, lda, &
                nnzPerRowColumn, nnzTotalDevHostPtr) &
                bind(c, name = 'hipsparseSnnz')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSnnz
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: nnzPerRowColumn
            type(c_ptr), value :: nnzTotalDevHostPtr
        end function hipsparseSnnz

        function hipsparseDnnz(handle, dir, m, n, descrA, A, lda, &
                nnzPerRowColumn, nnzTotalDevHostPtr) &
                bind(c, name = 'hipsparseDnnz')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDnnz
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: nnzPerRowColumn
            type(c_ptr), value :: nnzTotalDevHostPtr
        end function hipsparseDnnz

        function hipsparseCnnz(handle, dir, m, n, descrA, A, lda, &
                nnzPerRowColumn, nnzTotalDevHostPtr) &
                bind(c, name = 'hipsparseCnnz')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCnnz
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: nnzPerRowColumn
            type(c_ptr), value :: nnzTotalDevHostPtr
        end function hipsparseCnnz

        function hipsparseZnnz(handle, dir, m, n, descrA, A, lda, &
                nnzPerRowColumn, nnzTotalDevHostPtr) &
                bind(c, name = 'hipsparseZnnz')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZnnz
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: nnzPerRowColumn
            type(c_ptr), value :: nnzTotalDevHostPtr
        end function hipsparseZnnz

!       hipsparseXdense2csr
        function hipsparseSdense2csr(handle, m, n, descrA, A, lda, nnzPerRows, &
                csrValA, csrRowPtrA, csrColIndA) &
                bind(c, name = 'hipsparseSdense2csr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSdense2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: nnzPerRows
            type(c_ptr), value :: csrValA
            type(c_ptr), value :: csrRowPtrA
            type(c_ptr), value :: csrColIndA
        end function hipsparseSdense2csr

        function hipsparseDdense2csr(handle, m, n, descrA, A, lda, nnzPerRows, &
                csrValA, csrRowPtrA, csrColIndA) &
                bind(c, name = 'hipsparseDdense2csr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDdense2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: nnzPerRows
            type(c_ptr), value :: csrValA
            type(c_ptr), value :: csrRowPtrA
            type(c_ptr), value :: csrColIndA
        end function hipsparseDdense2csr

        function hipsparseCdense2csr(handle, m, n, descrA, A, lda, nnzPerRows, &
                csrValA, csrRowPtrA, csrColIndA) &
                bind(c, name = 'hipsparseCdense2csr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCdense2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: nnzPerRows
            type(c_ptr), value :: csrValA
            type(c_ptr), value :: csrRowPtrA
            type(c_ptr), value :: csrColIndA
        end function hipsparseCdense2csr

        function hipsparseZdense2csr(handle, m, n, descrA, A, lda, nnzPerRows, &
                csrValA, csrRowPtrA, csrColIndA) &
                bind(c, name = 'hipsparseZdense2csr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZdense2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: nnzPerRows
            type(c_ptr), value :: csrValA
            type(c_ptr), value :: csrRowPtrA
            type(c_ptr), value :: csrColIndA
        end function hipsparseZdense2csr

!       hipsparseXdense2csc
        function hipsparseSdense2csc(handle, m, n, descrA, A, lda, nnzPerColumns, &
                cscValA, cscRowPtrA, cscColIndA) &
                bind(c, name = 'hipsparseSdense2csc')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSdense2csc
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: nnzPerColumns
            type(c_ptr), value :: cscValA
            type(c_ptr), value :: cscRowPtrA
            type(c_ptr), value :: cscColIndA
        end function hipsparseSdense2csc

        function hipsparseDdense2csc(handle, m, n, descrA, A, lda, nnzPerColumns, &
                cscValA, cscRowPtrA, cscColIndA) &
                bind(c, name = 'hipsparseDdense2csc')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDdense2csc
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: nnzPerColumns
            type(c_ptr), value :: cscValA
            type(c_ptr), value :: cscRowPtrA
            type(c_ptr), value :: cscColIndA
        end function hipsparseDdense2csc

        function hipsparseCdense2csc(handle, m, n, descrA, A, lda, nnzPerColumns, &
                cscValA, cscRowPtrA, cscColIndA) &
                bind(c, name = 'hipsparseCdense2csc')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCdense2csc
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: nnzPerColumns
            type(c_ptr), value :: cscValA
            type(c_ptr), value :: cscRowPtrA
            type(c_ptr), value :: cscColIndA
        end function hipsparseCdense2csc

        function hipsparseZdense2csc(handle, m, n, descrA, A, lda, nnzPerColumns, &
                cscValA, cscRowPtrA, cscColIndA) &
                bind(c, name = 'hipsparseZdense2csc')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZdense2csc
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: nnzPerColumns
            type(c_ptr), value :: cscValA
            type(c_ptr), value :: cscRowPtrA
            type(c_ptr), value :: cscColIndA
        end function hipsparseZdense2csc

!       hipsparseXcsr2dense
        function hipsparseScsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, A, lda) &
                bind(c, name = 'hipsparseScsr2dense')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsr2dense
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function hipsparseScsr2dense

        function hipsparseDcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, A, lda) &
                bind(c, name = 'hipsparseDcsr2dense')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsr2dense
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function hipsparseDcsr2dense

        function hipsparseCcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, A, lda) &
                bind(c, name = 'hipsparseCcsr2dense')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsr2dense
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function hipsparseCcsr2dense

        function hipsparseZcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, A, lda) &
                bind(c, name = 'hipsparseZcsr2dense')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsr2dense
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function hipsparseZcsr2dense

!       hipsparseXcsc2dense
        function hipsparseScsc2dense(handle, m, n, descrA, cscValA, cscRowPtrA, &
                cscColIndA, A, lda) &
                bind(c, name = 'hipsparseScsc2dense')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsc2dense
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: cscValA
            type(c_ptr), intent(in), value :: cscRowPtrA
            type(c_ptr), intent(in), value :: cscColIndA
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function hipsparseScsc2dense

        function hipsparseDcsc2dense(handle, m, n, descrA, cscValA, cscRowPtrA, &
                cscColIndA, A, lda) &
                bind(c, name = 'hipsparseDcsc2dense')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsc2dense
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: cscValA
            type(c_ptr), intent(in), value :: cscRowPtrA
            type(c_ptr), intent(in), value :: cscColIndA
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function hipsparseDcsc2dense

        function hipsparseCcsc2dense(handle, m, n, descrA, cscValA, cscRowPtrA, &
                cscColIndA, A, lda) &
                bind(c, name = 'hipsparseCcsc2dense')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsc2dense
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: cscValA
            type(c_ptr), intent(in), value :: cscRowPtrA
            type(c_ptr), intent(in), value :: cscColIndA
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function hipsparseCcsc2dense

        function hipsparseZcsc2dense(handle, m, n, descrA, cscValA, cscRowPtrA, &
                cscColIndA, A, lda) &
                bind(c, name = 'hipsparseZcsc2dense')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsc2dense
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: cscValA
            type(c_ptr), intent(in), value :: cscRowPtrA
            type(c_ptr), intent(in), value :: cscColIndA
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function hipsparseZcsc2dense

!       hipsparseXnnz_compress
        function hipsparseSnnz_compress(handle, m, descrA, csrValA, csrRowPtrA, &
                nnzPerRow, nnzC, tol) &
                bind(c, name = 'hipsparseSnnz_compress')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSnnz_compress
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), value :: nnzPerRow
            type(c_ptr), value :: nnzC
            real(c_float), value :: tol
        end function hipsparseSnnz_compress

        function hipsparseDnnz_compress(handle, m, descrA, csrValA, csrRowPtrA, &
                nnzPerRow, nnzC, tol) &
                bind(c, name = 'hipsparseDnnz_compress')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDnnz_compress
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), value :: nnzPerRow
            type(c_ptr), value :: nnzC
            real(c_double), value :: tol
        end function hipsparseDnnz_compress

        function hipsparseCnnz_compress(handle, m, descrA, csrValA, csrRowPtrA, &
                nnzPerRow, nnzC, tol) &
                bind(c, name = 'hipsparseCnnz_compress')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCnnz_compress
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), value :: nnzPerRow
            type(c_ptr), value :: nnzC
            complex(c_float_complex), value :: tol
        end function hipsparseCnnz_compress

        function hipsparseZnnz_compress(handle, m, descrA, csrValA, csrRowPtrA, &
                nnzPerRow, nnzC, tol) &
                bind(c, name = 'hipsparseZnnz_compress')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZnnz_compress
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), value :: nnzPerRow
            type(c_ptr), value :: nnzC
            complex(c_double_complex), value :: tol
        end function hipsparseZnnz_compress

!       hipsparseXcsr2coo
        function hipsparseXcsr2coo(handle, csrRowPtr, nnz, m, cooRowInd, idxBase) &
                bind(c, name = 'hipsparseXcsr2coo')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXcsr2coo
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: csrRowPtr
            integer(c_int), value :: nnz
            integer(c_int), value :: m
            type(c_ptr), value :: cooRowInd
            integer(c_int), value :: idxBase
        end function hipsparseXcsr2coo

!       hipsparseXcsr2csc
        function hipsparseScsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, &
                cscVal, cscRowInd, cscColPtr, copyValues, idxBase) &
                bind(c, name = 'hipsparseScsr2csc')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsr2csc
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: cscVal
            type(c_ptr), value :: cscRowInd
            type(c_ptr), value :: cscColPtr
            integer(c_int), value :: copyValues
            integer(c_int), value :: idxBase
        end function hipsparseScsr2csc

        function hipsparseDcsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, &
                cscVal, cscRowInd, cscColPtr, copyValues, idxBase) &
                bind(c, name = 'hipsparseDcsr2csc')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsr2csc
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: cscVal
            type(c_ptr), value :: cscRowInd
            type(c_ptr), value :: cscColPtr
            integer(c_int), value :: copyValues
            integer(c_int), value :: idxBase
        end function hipsparseDcsr2csc

        function hipsparseCcsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, &
                cscVal, cscRowInd, cscColPtr, copyValues, idxBase) &
                bind(c, name = 'hipsparseCcsr2csc')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsr2csc
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: cscVal
            type(c_ptr), value :: cscRowInd
            type(c_ptr), value :: cscColPtr
            integer(c_int), value :: copyValues
            integer(c_int), value :: idxBase
        end function hipsparseCcsr2csc

        function hipsparseZcsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, &
                cscVal, cscRowInd, cscColPtr, copyValues, idxBase) &
                bind(c, name = 'hipsparseZcsr2csc')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsr2csc
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: cscVal
            type(c_ptr), value :: cscRowInd
            type(c_ptr), value :: cscColPtr
            integer(c_int), value :: copyValues
            integer(c_int), value :: idxBase
        end function hipsparseZcsr2csc

!       hipsparseXcsr2hyb
        function hipsparseScsr2hyb(handle, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, hybA, userEllWidth, partitionType) &
                bind(c, name = 'hipsparseScsr2hyb')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsr2hyb
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: hybA
            integer(c_int), value :: userEllWidth
            integer(c_int), value :: partitionType
        end function hipsparseScsr2hyb

        function hipsparseDcsr2hyb(handle, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, hybA, userEllWidth, partitionType) &
                bind(c, name = 'hipsparseDcsr2hyb')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsr2hyb
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: hybA
            integer(c_int), value :: userEllWidth
            integer(c_int), value :: partitionType
        end function hipsparseDcsr2hyb

        function hipsparseCcsr2hyb(handle, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, hybA, userEllWidth, partitionType) &
                bind(c, name = 'hipsparseCcsr2hyb')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsr2hyb
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: hybA
            integer(c_int), value :: userEllWidth
            integer(c_int), value :: partitionType
        end function hipsparseCcsr2hyb

        function hipsparseZcsr2hyb(handle, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, hybA, userEllWidth, partitionType) &
                bind(c, name = 'hipsparseZcsr2hyb')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsr2hyb
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: hybA
            integer(c_int), value :: userEllWidth
            integer(c_int), value :: partitionType
        end function hipsparseZcsr2hyb

!     hipsparseXgebsr2gebsc_bufferSize
        function hipsparseSgebsr2gebsc_bufferSize( &
        handle, &
        mb,&
        nb, &
        nnzb, &
        bsr_val, &
        bsr_row_ptr, &
        bsr_col_ind, &
        row_block_dim, &
        col_block_dim, &
        p_buffer_size) &
            bind(c, name = 'hipsparseSgebsr2gebsc_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) &
            :: hipsparseSgebsr2gebsc_bufferSize

            type(c_ptr), value :: handle
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: p_buffer_size
        end function hipsparseSgebsr2gebsc_bufferSize

        function hipsparseDgebsr2gebsc_bufferSize( &
        handle, &
        mb,&
        nb, &
        nnzb, &
        bsr_val, &
        bsr_row_ptr, &
        bsr_col_ind, &
        row_block_dim, &
        col_block_dim, &
        p_buffer_size) &
            bind(c, name = 'hipsparseDgebsr2gebsc_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) &
            :: hipsparseDgebsr2gebsc_bufferSize

            type(c_ptr), value :: handle
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: p_buffer_size
        end function hipsparseDgebsr2gebsc_bufferSize

        function hipsparseCgebsr2gebsc_bufferSize( &
        handle, &
        mb,&
        nb, &
        nnzb, &
        bsr_val, &
        bsr_row_ptr, &
        bsr_col_ind, &
        row_block_dim, &
        col_block_dim, &
        p_buffer_size) &
            bind(c, name = 'hipsparseCgebsr2gebsc_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) &
            :: hipsparseCgebsr2gebsc_bufferSize

            type(c_ptr), value :: handle
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: p_buffer_size
        end function hipsparseCgebsr2gebsc_bufferSize

        function hipsparseZgebsr2gebsc_bufferSize( &
        handle, &
        mb,&
        nb, &
        nnzb, &
        bsr_val, &
        bsr_row_ptr, &
        bsr_col_ind, &
        row_block_dim, &
        col_block_dim, &
        p_buffer_size) &
            bind(c, name = 'hipsparseZgebsr2gebsc_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) &
            :: hipsparseZgebsr2gebsc_bufferSize

            type(c_ptr), value :: handle
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: p_buffer_size
        end function hipsparseZgebsr2gebsc_bufferSize





        function hipsparseSgebsr2gebsc( &
        handle, &
        mb,&
        nb, &
        nnzb, &
        bsr_val, &
        bsr_row_ptr, &
        bsr_col_ind, &
        row_block_dim, &
        col_block_dim, &
        bsc_val, &
        bsc_row_ind, &
        bsc_col_ptr, &
        copy_values, &
        idx_base, &
        p_buffer) &
            bind(c, name = 'hipsparseSgebsr2gebsc')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) &
            :: hipsparseSgebsr2gebsc

            type(c_ptr), value :: handle
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), intent(in), value :: bsc_val
            type(c_ptr), intent(in), value :: bsc_row_ind
            type(c_ptr), intent(in), value :: bsc_col_ptr
            integer(c_int), value :: copy_values
            integer(c_int), value :: idx_base
            type(c_ptr), intent(in), value :: p_buffer
        end function hipsparseSgebsr2gebsc

        function hipsparseDgebsr2gebsc( &
        handle, &
        mb,&
        nb, &
        nnzb, &
        bsr_val, &
        bsr_row_ptr, &
        bsr_col_ind, &
        row_block_dim, &
        col_block_dim, &
        bsc_val, &
        bsc_row_ind, &
        bsc_col_ptr, &
        copy_values, &
        idx_base, &
        p_buffer) &
            bind(c, name = 'hipsparseDgebsr2gebsc')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) &
            :: hipsparseDgebsr2gebsc

            type(c_ptr), value :: handle
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), intent(in), value :: bsc_val
            type(c_ptr), intent(in), value :: bsc_row_ind
            type(c_ptr), intent(in), value :: bsc_col_ptr
            integer(c_int), value :: copy_values
            integer(c_int), value :: idx_base
            type(c_ptr), intent(in), value :: p_buffer
        end function hipsparseDgebsr2gebsc


        function hipsparseCgebsr2gebsc( &
        handle, &
        mb,&
        nb, &
        nnzb, &
        bsr_val, &
        bsr_row_ptr, &
        bsr_col_ind, &
        row_block_dim, &
        col_block_dim, &
        bsc_val, &
        bsc_row_ind, &
        bsc_col_ptr, &
        copy_values, &
        idx_base, &
        p_buffer) &
            bind(c, name = 'hipsparseCgebsr2gebsc')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) &
            :: hipsparseCgebsr2gebsc

            type(c_ptr), value :: handle
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), intent(in), value :: bsc_val
            type(c_ptr), intent(in), value :: bsc_row_ind
            type(c_ptr), intent(in), value :: bsc_col_ptr
            integer(c_int), value :: copy_values
            integer(c_int), value :: idx_base
            type(c_ptr), intent(in), value :: p_buffer
        end function hipsparseCgebsr2gebsc

        function hipsparseZgebsr2gebsc( &
        handle, &
        mb,&
        nb, &
        nnzb, &
        bsr_val, &
        bsr_row_ptr, &
        bsr_col_ind, &
        row_block_dim, &
        col_block_dim, &
        bsc_val, &
        bsc_row_ind, &
        bsc_col_ptr, &
        copy_values, &
        idx_base, &
        p_buffer) &
            bind(c, name = 'hipsparseZgebsr2gebsc')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) &
            :: hipsparseZgebsr2gebsc

            type(c_ptr), value :: handle
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), intent(in), value :: bsc_val
            type(c_ptr), intent(in), value :: bsc_row_ind
            type(c_ptr), intent(in), value :: bsc_col_ptr
            integer(c_int), value :: copy_values
            integer(c_int), value :: idx_base
            type(c_ptr), intent(in), value :: p_buffer
        end function hipsparseZgebsr2gebsc








!     hipsparseXcsr2gebsr_bufferSize
        function hipsparseScsr2gebsr_bufferSize( &
                handle, &
                dir, &
                m, &
                n, &
                csr_descr, &
                csr_val, &
                csr_row_ptr, &
                csr_col_ind, &
                row_block_dim, &
                col_block_dim, &
                p_buffer_size) &
                bind(c, name = 'hipsparseScsr2gebsr_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) &
            :: hipsparseScsr2gebsr_bufferSize

            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: p_buffer_size
        end function hipsparseScsr2gebsr_bufferSize

        function hipsparseDcsr2gebsr_bufferSize( &
                handle, &
                dir, &
                m, &
                n, &
                csr_descr, &
                csr_val, &
                csr_row_ptr, &
                csr_col_ind, &
                row_block_dim, &
                col_block_dim, &
                p_buffer_size) &
                bind(c, name = 'hipsparseDcsr2gebsr_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) &
            :: hipsparseDcsr2gebsr_bufferSize

            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: p_buffer_size
        end function hipsparseDcsr2gebsr_bufferSize

        function hipsparseCcsr2gebsr_bufferSize( &
                handle, &
                dir, &
                m, &
                n, &
                csr_descr, &
                csr_val, &
                csr_row_ptr, &
                csr_col_ind, &
                row_block_dim, &
                col_block_dim, &
                p_buffer_size) &
                bind(c, name = 'hipsparseCcsr2gebsr_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) &
            :: hipsparseCcsr2gebsr_bufferSize

            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: p_buffer_size
        end function hipsparseCcsr2gebsr_bufferSize


        function hipsparseZcsr2gebsr_bufferSize( &
                handle, &
                dir, &
                m, &
                n, &
                csr_descr, &
                csr_val, &
                csr_row_ptr, &
                csr_col_ind, &
                row_block_dim, &
                col_block_dim, &
                p_buffer_size) &
                bind(c, name = 'hipsparseZcsr2gebsr_bufferSize')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) &
            :: hipsparseZcsr2gebsr_bufferSize

            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: p_buffer_size
        end function hipsparseZcsr2gebsr_bufferSize

!     hipsparseXcsr2gebsrNnz
        function hipsparseXcsr2gebsrNnz( &
                handle, &
                dir, &
                m, &
                n, &
                csr_descr, &
                csr_row_ptr, &
                csr_col_ind, &
                bsr_descr, &
                bsr_row_ptr, &
                row_block_dim, &
                col_block_dim, &
                bsr_nnz_devhost, &
                p_buffer) &
                bind(c, name = 'hipsparseXcsr2gebsrNnz')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) &
            :: hipsparseXcsr2gebsrNnz

            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), intent(in), value :: bsr_row_ptr
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: bsr_nnz_devhost
            type(c_ptr), intent(in), value :: p_buffer
        end function hipsparseXcsr2gebsrNnz

!     hipsparseScsr2gebsr
        function hipsparseScsr2gebsr( &
                handle, &
                dir, &
                m, &
                n, &
                csr_descr, &
                csr_val, &
                csr_row_ptr, &
                csr_col_ind, &
                bsr_descr, &
                bsr_val, &
                bsr_row_ptr, &
                bsr_col_ind, &
                row_block_dim, &
                col_block_dim, &
                p_buffer) &
                bind(c, name = 'hipsparseScsr2gebsr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) &
            :: hipsparseScsr2gebsr

            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), intent(in), value :: p_buffer
        end function hipsparseScsr2gebsr


!     hipsparseDcsr2gebsr
        function hipsparseDcsr2gebsr( &
                handle, &
                dir, &
                m, &
                n, &
                csr_descr, &
                csr_val, &
                csr_row_ptr, &
                csr_col_ind, &
                bsr_descr, &
                bsr_val, &
                bsr_row_ptr, &
                bsr_col_ind, &
                row_block_dim, &
                col_block_dim, &
                p_buffer) &
                bind(c, name = 'hipsparseDcsr2gebsr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) &
            :: hipsparseDcsr2gebsr

            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), intent(in), value :: p_buffer
        end function hipsparseDcsr2gebsr


        function hipsparseCcsr2gebsr( &
                handle, &
                dir, &
                m, &
                n, &
                csr_descr, &
                csr_val, &
                csr_row_ptr, &
                csr_col_ind, &
                bsr_descr, &
                bsr_val, &
                bsr_row_ptr, &
                bsr_col_ind, &
                row_block_dim, &
                col_block_dim, &
                p_buffer) &
                bind(c, name = 'hipsparseCcsr2gebsr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) &
            :: hipsparseCcsr2gebsr

            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), intent(in), value :: p_buffer
        end function hipsparseCcsr2gebsr

        function hipsparseZcsr2gebsr( &
                handle, &
                dir, &
                m, &
                n, &
                csr_descr, &
                csr_val, &
                csr_row_ptr, &
                csr_col_ind, &
                bsr_descr, &
                bsr_val, &
                bsr_row_ptr, &
                bsr_col_ind, &
                row_block_dim, &
                col_block_dim, &
                p_buffer) &
                bind(c, name = 'hipsparseZcsr2gebsr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) &
            :: hipsparseZcsr2gebsr

            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), intent(in), value :: p_buffer
        end function hipsparseZcsr2gebsr


!       hipsparseXcsr2bsrNnz
        function hipsparseXcsr2bsrNnz(handle, dirA, m, n, descrA, csrRowPtrA, &
                csrColIndA, blockDim, descrC, bsrRowPtrC, bsrNnzb) &
                bind(c, name = 'hipsparseXcsr2bsrNnz')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXcsr2bsrNnz
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: bsrRowPtrC
            type(c_ptr), value :: bsrNnzb
        end function hipsparseXcsr2bsrNnz

!       hipsparseXcsr2bsr
        function hipsparseScsr2bsr(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, blockDim, descrC, bsrValC, bsrRowPtrC, bsrColIndC) &
                bind(c, name = 'hipsparseScsr2bsr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsr2bsr
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: bsrValC
            type(c_ptr), value :: bsrRowPtrC
            type(c_ptr), value :: bsrColIndC
        end function hipsparseScsr2bsr

        function hipsparseDcsr2bsr(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, blockDim, descrC, bsrValC, bsrRowPtrC, bsrColIndC) &
                bind(c, name = 'hipsparseDcsr2bsr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsr2bsr
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: bsrValC
            type(c_ptr), value :: bsrRowPtrC
            type(c_ptr), value :: bsrColIndC
        end function hipsparseDcsr2bsr

        function hipsparseCcsr2bsr(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, blockDim, descrC, bsrValC, bsrRowPtrC, bsrColIndC) &
                bind(c, name = 'hipsparseCcsr2bsr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsr2bsr
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: bsrValC
            type(c_ptr), value :: bsrRowPtrC
            type(c_ptr), value :: bsrColIndC
        end function hipsparseCcsr2bsr

        function hipsparseZcsr2bsr(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, blockDim, descrC, bsrValC, bsrRowPtrC, bsrColIndC) &
                bind(c, name = 'hipsparseZcsr2bsr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsr2bsr
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: bsrValC
            type(c_ptr), value :: bsrRowPtrC
            type(c_ptr), value :: bsrColIndC
        end function hipsparseZcsr2bsr

!       hipsparseXbsr2csr
        function hipsparseSbsr2csr(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, &
                bsrColIndA, blockDim, descrC, csrValC, csrRowPtrC, csrColIndC) &
                bind(c, name = 'hipsparseSbsr2csr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSbsr2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: bsrValA
            type(c_ptr), intent(in), value :: bsrRowPtrA
            type(c_ptr), intent(in), value :: bsrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseSbsr2csr

        function hipsparseDbsr2csr(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, &
                bsrColIndA, blockDim, descrC, csrValC, csrRowPtrC, csrColIndC) &
                bind(c, name = 'hipsparseDbsr2csr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDbsr2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: bsrValA
            type(c_ptr), intent(in), value :: bsrRowPtrA
            type(c_ptr), intent(in), value :: bsrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseDbsr2csr

        function hipsparseCbsr2csr(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, &
                bsrColIndA, blockDim, descrC, csrValC, csrRowPtrC, csrColIndC) &
                bind(c, name = 'hipsparseCbsr2csr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCbsr2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: bsrValA
            type(c_ptr), intent(in), value :: bsrRowPtrA
            type(c_ptr), intent(in), value :: bsrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseCbsr2csr

        function hipsparseZbsr2csr(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, &
                bsrColIndA, blockDim, descrC, csrValC, csrRowPtrC, csrColIndC) &
                bind(c, name = 'hipsparseZbsr2csr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZbsr2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: bsrValA
            type(c_ptr), intent(in), value :: bsrRowPtrA
            type(c_ptr), intent(in), value :: bsrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseZbsr2csr



!       hipsparseXgebsr2csr
        function hipsparseSgebsr2csr(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, &
                bsrColIndA, rowBlockDim, colBlockDim,descrC, csrValC, csrRowPtrC, csrColIndC) &
                bind(c, name = 'hipsparseSgebsr2csr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseSgebsr2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: bsrValA
            type(c_ptr), intent(in), value :: bsrRowPtrA
            type(c_ptr), intent(in), value :: bsrColIndA
            integer(c_int), value :: rowBlockDim
            integer(c_int), value :: colBlockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseSgebsr2csr

        function hipsparseDgebsr2csr(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, &
                bsrColIndA, rowBlockDim, colBlockDim,descrC, csrValC, csrRowPtrC, csrColIndC) &
                bind(c, name = 'hipsparseDgebsr2csr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDgebsr2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: bsrValA
            type(c_ptr), intent(in), value :: bsrRowPtrA
            type(c_ptr), intent(in), value :: bsrColIndA
            integer(c_int), value :: rowBlockDim
            integer(c_int), value :: colBlockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseDgebsr2csr

        function hipsparseCgebsr2csr(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, &
                bsrColIndA, rowBlockDim, colBlockDim,descrC, csrValC, csrRowPtrC, csrColIndC) &
                bind(c, name = 'hipsparseCgebsr2csr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCgebsr2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: bsrValA
            type(c_ptr), intent(in), value :: bsrRowPtrA
            type(c_ptr), intent(in), value :: bsrColIndA
            integer(c_int), value :: rowBlockDim
            integer(c_int), value :: colBlockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseCgebsr2csr

        function hipsparseZgebsr2csr(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, &
                bsrColIndA, rowBlockDim, colBlockDim,descrC, csrValC, csrRowPtrC, csrColIndC) &
                bind(c, name = 'hipsparseZgebsr2csr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZgebsr2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: bsrValA
            type(c_ptr), intent(in), value :: bsrRowPtrA
            type(c_ptr), intent(in), value :: bsrColIndA
            integer(c_int), value :: rowBlockDim
            integer(c_int), value :: colBlockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseZgebsr2csr




!       hipsparseXcsr2csr_compress
        function hipsparseScsr2csr_compress(handle, m, n, descrA, csrValA, csrColIndA, &
                csrRowPtrA, nnzA, nnzPerRow, csrValC, csrColIndC, csrRowPtrC, tol) &
                bind(c, name = 'hipsparseScsr2csr_compress')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseScsr2csr_compress
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: csrRowPtrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: nnzPerRow
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), value :: csrRowPtrC
            real(c_float), value :: tol
        end function hipsparseScsr2csr_compress

        function hipsparseDcsr2csr_compress(handle, m, n, descrA, csrValA, csrColIndA, &
                csrRowPtrA, nnzA, nnzPerRow, csrValC, csrColIndC, csrRowPtrC, tol) &
                bind(c, name = 'hipsparseDcsr2csr_compress')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDcsr2csr_compress
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: csrRowPtrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: nnzPerRow
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), value :: csrRowPtrC
            real(c_double), value :: tol
        end function hipsparseDcsr2csr_compress

        function hipsparseCcsr2csr_compress(handle, m, n, descrA, csrValA, csrColIndA, &
                csrRowPtrA, nnzA, nnzPerRow, csrValC, csrColIndC, csrRowPtrC, tol) &
                bind(c, name = 'hipsparseCcsr2csr_compress')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCcsr2csr_compress
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: csrRowPtrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: nnzPerRow
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), value :: csrRowPtrC
            complex(c_float_complex), value :: tol
        end function hipsparseCcsr2csr_compress

        function hipsparseZcsr2csr_compress(handle, m, n, descrA, csrValA, csrColIndA, &
                csrRowPtrA, nnzA, nnzPerRow, csrValC, csrColIndC, csrRowPtrC, tol) &
                bind(c, name = 'hipsparseZcsr2csr_compress')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZcsr2csr_compress
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: csrRowPtrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: nnzPerRow
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), value :: csrRowPtrC
            complex(c_double_complex), value :: tol
        end function hipsparseZcsr2csr_compress

!       hipsparseXhyb2csr
        function hipsparseShyb2csr(handle, descrA, hybA, csrValA, csrRowPtrA, &
                csrColIndA) &
                bind(c, name = 'hipsparseShyb2csr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseShyb2csr
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: hybA
            type(c_ptr), value :: csrValA
            type(c_ptr), value :: csrRowPtrA
            type(c_ptr), value :: csrColIndA
        end function hipsparseShyb2csr

        function hipsparseDhyb2csr(handle, descrA, hybA, csrValA, csrRowPtrA, &
                csrColIndA) &
                bind(c, name = 'hipsparseDhyb2csr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseDhyb2csr
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: hybA
            type(c_ptr), value :: csrValA
            type(c_ptr), value :: csrRowPtrA
            type(c_ptr), value :: csrColIndA
        end function hipsparseDhyb2csr

        function hipsparseChyb2csr(handle, descrA, hybA, csrValA, csrRowPtrA, &
                csrColIndA) &
                bind(c, name = 'hipsparseChyb2csr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseChyb2csr
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: hybA
            type(c_ptr), value :: csrValA
            type(c_ptr), value :: csrRowPtrA
            type(c_ptr), value :: csrColIndA
        end function hipsparseChyb2csr

        function hipsparseZhyb2csr(handle, descrA, hybA, csrValA, csrRowPtrA, &
                csrColIndA) &
                bind(c, name = 'hipsparseZhyb2csr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseZhyb2csr
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: hybA
            type(c_ptr), value :: csrValA
            type(c_ptr), value :: csrRowPtrA
            type(c_ptr), value :: csrColIndA
        end function hipsparseZhyb2csr

!       hipsparseXcoo2csr
        function hipsparseXcoo2csr(handle, cooRowInd, nnz, m, csrRowPtr, idxBase) &
                bind(c, name = 'hipsparseXcoo2csr')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXcoo2csr
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: cooRowInd
            integer(c_int), value :: nnz
            integer(c_int), value :: m
            type(c_ptr), value :: csrRowPtr
            integer(c_int), value :: idxBase
        end function hipsparseXcoo2csr

!       hipsparseCreateIdentityPermutation
        function hipsparseCreateIdentityPermutation(handle, n, p) &
                bind(c, name = 'hipsparseCreateIdentityPermutation')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseCreateIdentityPermutation
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: p
        end function hipsparseCreateIdentityPermutation

!       hipsparseXcsrsort_bufferSizeExt
        function hipsparseXcsrsort_bufferSizeExt(handle, m, n, nnz, csrRowPtr, &
                csrColInd, bufferSize) &
                bind(c, name = 'hipsparseXcsrsort_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXcsrsort_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: bufferSize
        end function hipsparseXcsrsort_bufferSizeExt

!       hipsparseXcsrsort
        function hipsparseXcsrsort(handle, m, n, nnz, descr, csrRowPtr, csrColInd, &
                perm, buffer) &
                bind(c, name = 'hipsparseXcsrsort')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXcsrsort
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), value :: csrColInd
            type(c_ptr), value :: perm
            type(c_ptr), value :: buffer
        end function hipsparseXcsrsort

!       hipsparseXcscsort_bufferSizeExt
        function hipsparseXcscsort_bufferSizeExt(handle, m, n, nnz, cscRowPtr, &
                cscColInd, bufferSize) &
                bind(c, name = 'hipsparseXcscsort_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXcscsort_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: cscRowPtr
            type(c_ptr), intent(in), value :: cscColInd
            type(c_ptr), value :: bufferSize
        end function hipsparseXcscsort_bufferSizeExt

!       hipsparseXcscsort
        function hipsparseXcscsort(handle, m, n, nnz, descr, cscRowPtr, cscColInd, &
                perm, buffer) &
                bind(c, name = 'hipsparseXcscsort')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXcscsort
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: cscRowPtr
            type(c_ptr), value :: cscColInd
            type(c_ptr), value :: perm
            type(c_ptr), value :: buffer
        end function hipsparseXcscsort

!       hipsparseXcoosort_bufferSizeExt
        function hipsparseXcoosort_bufferSizeExt(handle, m, n, nnz, cooRowInd, &
                cooColInd, bufferSize) &
                bind(c, name = 'hipsparseXcoosort_bufferSizeExt')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXcoosort_bufferSizeExt
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: cooRowInd
            type(c_ptr), intent(in), value :: cooColInd
            type(c_ptr), value :: bufferSize
        end function hipsparseXcoosort_bufferSizeExt

!       hipsparseXcoosortByRow
        function hipsparseXcoosortByRow(handle, m, n, nnz, cooRowInd, cooColInd, &
                perm, buffer) &
                bind(c, name = 'hipsparseXcoosortByRow')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXcoosortByRow
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), value :: cooRowInd
            type(c_ptr), value :: cooColInd
            type(c_ptr), value :: perm
            type(c_ptr), value :: buffer
        end function hipsparseXcoosortByRow

!       hipsparseXcoosortByColumn
        function hipsparseXcoosortByColumn(handle, m, n, nnz, cooRowInd, cooColInd, &
                perm, buffer) &
                bind(c, name = 'hipsparseXcoosortByColumn')
            use hipsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(HIPSPARSE_STATUS_SUCCESS)) :: hipsparseXcoosortByColumn
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), value :: cooRowInd
            type(c_ptr), value :: cooColInd
            type(c_ptr), value :: perm
            type(c_ptr), value :: buffer
        end function hipsparseXcoosortByColumn

    end interface

end module hipsparse
