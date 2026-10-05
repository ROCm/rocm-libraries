!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
!
! SPDX-License-Identifier: MIT
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

module rocsolver
  use rocblas, only: rocblas_diagonal_non_unit, rocblas_fill_upper, rocblas_operation_none, &
    rocblas_side_left, rocblas_status_success
  use, intrinsic :: iso_c_binding
  implicit none

  ! rocblas_layer_mode_ex_
  enum, bind(c)
    enumerator :: rocblas_layer_mode_ex_log_kernel = 16
  end enum

  ! rocblas_direct_
  enum, bind(c)
    enumerator :: rocblas_forward_direction = 171
    enumerator :: rocblas_backward_direction = 172
  end enum

  ! rocblas_storev_
  enum, bind(c)
    enumerator :: rocblas_column_wise = 181
    enumerator :: rocblas_row_wise = 182
  end enum

  ! rocblas_svect_
  enum, bind(c)
    enumerator :: rocblas_svect_all = 191
    enumerator :: rocblas_svect_singular = 192
    enumerator :: rocblas_svect_overwrite = 193
    enumerator :: rocblas_svect_none = 194
  end enum

  ! rocblas_workmode_
  enum, bind(c)
    enumerator :: rocblas_outofplace = 201
    enumerator :: rocblas_inplace = 202
  end enum

  ! rocblas_evect_
  enum, bind(c)
    enumerator :: rocblas_evect_original = 211
    enumerator :: rocblas_evect_tridiagonal = 212
    enumerator :: rocblas_evect_none = 213
  end enum

  ! rocblas_eform_
  enum, bind(c)
    enumerator :: rocblas_eform_ax = 221
    enumerator :: rocblas_eform_abx = 222
    enumerator :: rocblas_eform_bax = 223
  end enum

  ! rocblas_erange_
  enum, bind(c)
    enumerator :: rocblas_erange_all = 231
    enumerator :: rocblas_erange_value = 232
    enumerator :: rocblas_erange_index = 233
  end enum

  ! rocblas_eorder_
  enum, bind(c)
    enumerator :: rocblas_eorder_blocks = 241
    enumerator :: rocblas_eorder_entire = 242
  end enum

  ! rocblas_esort_
  enum, bind(c)
    enumerator :: rocblas_esort_none = 251
    enumerator :: rocblas_esort_ascending = 252
  end enum

  ! rocblas_srange_
  enum, bind(c)
    enumerator :: rocblas_srange_all = 261
    enumerator :: rocblas_srange_value = 262
    enumerator :: rocblas_srange_index = 263
  end enum

  ! rocsolver_rfinfo_mode_
  enum, bind(c)
    enumerator :: rocsolver_rfinfo_mode_lu = 271
    enumerator :: rocsolver_rfinfo_mode_cholesky = 272
  end enum

  ! rocblas_pivot_
  enum, bind(c)
    enumerator :: rocblas_pivot_variable = 281
    enumerator :: rocblas_pivot_top = 282
    enumerator :: rocblas_pivot_bottom = 283
  end enum

  ! rocsolver_alg_mode_
  enum, bind(c)
    enumerator :: rocsolver_alg_mode_gpu = 291
    enumerator :: rocsolver_alg_mode_hybrid = 292
    enumerator :: rocsolver_alg_mode_mixed = 293
    enumerator :: rocsolver_alg_mode_1stage = 294
    enumerator :: rocsolver_alg_mode_2stage = 295
    enumerator :: rocsolver_alg_mode_auto = 296
  end enum

  ! rocsolver_norm_type_
  enum, bind(c)
    enumerator :: rocsolver_norm_type_one = 301
    enumerator :: rocsolver_norm_type_frobenius = 302
    enumerator :: rocsolver_norm_type_infinity = 303
    enumerator :: rocsolver_norm_type_max = 304
  end enum

  ! rocsolver_cholqr_shift_
  enum, bind(c)
    enumerator :: rocsolver_cholqr_shift_none = 311
    enumerator :: rocsolver_cholqr_shift_computed = 312
    enumerator :: rocsolver_cholqr_shift_provided = 313
  end enum

  ! rocsolver_function_
  enum, bind(c)
    enumerator :: rocsolver_function_bdsqr = 401
    enumerator :: rocsolver_function_gesvd = 402
    enumerator :: rocsolver_function_sterf = 403
    enumerator :: rocsolver_function_steqr = 404
    enumerator :: rocsolver_function_syev_heev = 405
    enumerator :: rocsolver_function_syev = 405
    enumerator :: rocsolver_function_heev = 405
    enumerator :: rocsolver_function_sytrd_hetrd = 406
    enumerator :: rocsolver_function_sytrd = 406
    enumerator :: rocsolver_function_hetrd = 406
  end enum


  interface

    !---------------------------------------------
    ! rocsolver_get_version_string
    !---------------------------------------------
    function rocsolver_get_version_string(buf, len) &
       result(get_version_string) &
       bind(C, name="rocsolver_get_version_string")
       import :: c_ptr, c_size_t, rocblas_status_success
       type(c_ptr), value :: buf
       integer(c_size_t), value :: len
       integer(kind(rocblas_status_success)) :: get_version_string
    end function rocsolver_get_version_string

    !---------------------------------------------
    ! rocsolver_get_version_string_size
    !---------------------------------------------
    function rocsolver_get_version_string_size(len) &
       result(get_version_string_size) &
       bind(C, name="rocsolver_get_version_string_size")
       import :: c_size_t, rocblas_status_success
       integer(c_size_t) :: len
       integer(kind(rocblas_status_success)) :: get_version_string_size
    end function rocsolver_get_version_string_size

    !---------------------------------------------
    ! rocsolver_log_begin
    !---------------------------------------------
    function rocsolver_log_begin() &
       result(log_begin) &
       bind(C, name="rocsolver_log_begin")
       import :: rocblas_status_success
       integer(kind(rocblas_status_success)) :: log_begin
    end function rocsolver_log_begin

    !---------------------------------------------
    ! rocsolver_log_end
    !---------------------------------------------
    function rocsolver_log_end() &
       result(log_end) &
       bind(C, name="rocsolver_log_end")
       import :: rocblas_status_success
       integer(kind(rocblas_status_success)) :: log_end
    end function rocsolver_log_end

    !---------------------------------------------
    ! rocsolver_log_set_layer_mode
    !---------------------------------------------
    function rocsolver_log_set_layer_mode(layer_mode) &
       result(log_set_layer_mode) &
       bind(C, name="rocsolver_log_set_layer_mode")
       import :: c_int, rocblas_status_success
       integer(c_int), value :: layer_mode
       integer(kind(rocblas_status_success)) :: log_set_layer_mode
    end function rocsolver_log_set_layer_mode

    !---------------------------------------------
    ! rocsolver_log_set_max_levels
    !---------------------------------------------
    function rocsolver_log_set_max_levels(max_levels) &
       result(log_set_max_levels) &
       bind(C, name="rocsolver_log_set_max_levels")
       import :: c_int, rocblas_status_success
       integer(c_int), value :: max_levels
       integer(kind(rocblas_status_success)) :: log_set_max_levels
    end function rocsolver_log_set_max_levels

    !---------------------------------------------
    ! rocsolver_log_restore_defaults
    !---------------------------------------------
    function rocsolver_log_restore_defaults() &
       result(log_restore_defaults) &
       bind(C, name="rocsolver_log_restore_defaults")
       import :: rocblas_status_success
       integer(kind(rocblas_status_success)) :: log_restore_defaults
    end function rocsolver_log_restore_defaults

    !---------------------------------------------
    ! rocsolver_log_write_profile
    !---------------------------------------------
    function rocsolver_log_write_profile() &
       result(log_write_profile) &
       bind(C, name="rocsolver_log_write_profile")
       import :: rocblas_status_success
       integer(kind(rocblas_status_success)) :: log_write_profile
    end function rocsolver_log_write_profile

    !---------------------------------------------
    ! rocsolver_log_flush_profile
    !---------------------------------------------
    function rocsolver_log_flush_profile() &
       result(log_flush_profile) &
       bind(C, name="rocsolver_log_flush_profile")
       import :: rocblas_status_success
       integer(kind(rocblas_status_success)) :: log_flush_profile
    end function rocsolver_log_flush_profile

    !---------------------------------------------
    ! rocsolver_set_alg_mode
    !---------------------------------------------
    function rocsolver_set_alg_mode(handle, func, mode) &
       result(set_alg_mode) &
       bind(C, name="rocsolver_set_alg_mode")
       import :: c_ptr, rocsolver_function_bdsqr, rocsolver_alg_mode_gpu, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_function_bdsqr)), value :: func
       integer(kind(rocsolver_alg_mode_gpu)), value :: mode
       integer(kind(rocblas_status_success)) :: set_alg_mode
    end function rocsolver_set_alg_mode

    !---------------------------------------------
    ! rocsolver_get_alg_mode
    !---------------------------------------------
    function rocsolver_get_alg_mode(handle, func, mode) &
       result(get_alg_mode) &
       bind(C, name="rocsolver_get_alg_mode")
       import :: c_ptr, rocsolver_function_bdsqr, rocsolver_alg_mode_gpu, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_function_bdsqr)), value :: func
       integer(kind(rocsolver_alg_mode_gpu)) :: mode
       integer(kind(rocblas_status_success)) :: get_alg_mode
    end function rocsolver_get_alg_mode

    !---------------------------------------------
    ! rocsolver_clacgv
    !---------------------------------------------
    function rocsolver_clacgv(handle, n, x, incx) &
       result(clacgv) &
       bind(C, name="rocsolver_clacgv")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: x
       integer(c_int), value :: incx
       integer(kind(rocblas_status_success)) :: clacgv
    end function rocsolver_clacgv

    !---------------------------------------------
    ! rocsolver_zlacgv
    !---------------------------------------------
    function rocsolver_zlacgv(handle, n, x, incx) &
       result(zlacgv) &
       bind(C, name="rocsolver_zlacgv")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: x
       integer(c_int), value :: incx
       integer(kind(rocblas_status_success)) :: zlacgv
    end function rocsolver_zlacgv

    !---------------------------------------------
    ! rocsolver_clacgv_64
    !---------------------------------------------
    function rocsolver_clacgv_64(handle, n, x, incx) &
       result(clacgv_64) &
       bind(C, name="rocsolver_clacgv_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: n
       type(c_ptr), value :: x
       integer(c_int64_t), value :: incx
       integer(kind(rocblas_status_success)) :: clacgv_64
    end function rocsolver_clacgv_64

    !---------------------------------------------
    ! rocsolver_zlacgv_64
    !---------------------------------------------
    function rocsolver_zlacgv_64(handle, n, x, incx) &
       result(zlacgv_64) &
       bind(C, name="rocsolver_zlacgv_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: n
       type(c_ptr), value :: x
       integer(c_int64_t), value :: incx
       integer(kind(rocblas_status_success)) :: zlacgv_64
    end function rocsolver_zlacgv_64

    !---------------------------------------------
    ! rocsolver_slange
    !---------------------------------------------
    function rocsolver_slange(handle, norm_type, m, n, A, lda, norm) &
       result(slange) &
       bind(C, name="rocsolver_slange")
       import :: c_ptr, rocsolver_norm_type_one, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_norm_type_one)), value :: norm_type
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: norm
       integer(kind(rocblas_status_success)) :: slange
    end function rocsolver_slange

    !---------------------------------------------
    ! rocsolver_dlange
    !---------------------------------------------
    function rocsolver_dlange(handle, norm_type, m, n, A, lda, norm) &
       result(dlange) &
       bind(C, name="rocsolver_dlange")
       import :: c_ptr, rocsolver_norm_type_one, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_norm_type_one)), value :: norm_type
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: norm
       integer(kind(rocblas_status_success)) :: dlange
    end function rocsolver_dlange

    !---------------------------------------------
    ! rocsolver_clange
    !---------------------------------------------
    function rocsolver_clange(handle, norm_type, m, n, A, lda, norm) &
       result(clange) &
       bind(C, name="rocsolver_clange")
       import :: c_ptr, rocsolver_norm_type_one, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_norm_type_one)), value :: norm_type
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: norm
       integer(kind(rocblas_status_success)) :: clange
    end function rocsolver_clange

    !---------------------------------------------
    ! rocsolver_zlange
    !---------------------------------------------
    function rocsolver_zlange(handle, norm_type, m, n, A, lda, norm) &
       result(zlange) &
       bind(C, name="rocsolver_zlange")
       import :: c_ptr, rocsolver_norm_type_one, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_norm_type_one)), value :: norm_type
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: norm
       integer(kind(rocblas_status_success)) :: zlange
    end function rocsolver_zlange

    !---------------------------------------------
    ! rocsolver_slange_64
    !---------------------------------------------
    function rocsolver_slange_64(handle, norm_type, m, n, A, lda, norm) &
       result(slange_64) &
       bind(C, name="rocsolver_slange_64")
       import :: c_ptr, rocsolver_norm_type_one, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_norm_type_one)), value :: norm_type
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: norm
       integer(kind(rocblas_status_success)) :: slange_64
    end function rocsolver_slange_64

    !---------------------------------------------
    ! rocsolver_dlange_64
    !---------------------------------------------
    function rocsolver_dlange_64(handle, norm_type, m, n, A, lda, norm) &
       result(dlange_64) &
       bind(C, name="rocsolver_dlange_64")
       import :: c_ptr, rocsolver_norm_type_one, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_norm_type_one)), value :: norm_type
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: norm
       integer(kind(rocblas_status_success)) :: dlange_64
    end function rocsolver_dlange_64

    !---------------------------------------------
    ! rocsolver_clange_64
    !---------------------------------------------
    function rocsolver_clange_64(handle, norm_type, m, n, A, lda, norm) &
       result(clange_64) &
       bind(C, name="rocsolver_clange_64")
       import :: c_ptr, rocsolver_norm_type_one, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_norm_type_one)), value :: norm_type
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: norm
       integer(kind(rocblas_status_success)) :: clange_64
    end function rocsolver_clange_64

    !---------------------------------------------
    ! rocsolver_zlange_64
    !---------------------------------------------
    function rocsolver_zlange_64(handle, norm_type, m, n, A, lda, norm) &
       result(zlange_64) &
       bind(C, name="rocsolver_zlange_64")
       import :: c_ptr, rocsolver_norm_type_one, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_norm_type_one)), value :: norm_type
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: norm
       integer(kind(rocblas_status_success)) :: zlange_64
    end function rocsolver_zlange_64

    !---------------------------------------------
    ! rocsolver_sgecon
    !---------------------------------------------
    function rocsolver_sgecon(handle, norm_type, n, A, lda, anorm, rcond) &
       result(sgecon) &
       bind(C, name="rocsolver_sgecon")
       import :: c_ptr, rocsolver_norm_type_one, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_norm_type_one)), value :: norm_type
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: anorm
       type(c_ptr), value :: rcond
       integer(kind(rocblas_status_success)) :: sgecon
    end function rocsolver_sgecon

    !---------------------------------------------
    ! rocsolver_dgecon
    !---------------------------------------------
    function rocsolver_dgecon(handle, norm_type, n, A, lda, anorm, rcond) &
       result(dgecon) &
       bind(C, name="rocsolver_dgecon")
       import :: c_ptr, rocsolver_norm_type_one, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_norm_type_one)), value :: norm_type
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: anorm
       type(c_ptr), value :: rcond
       integer(kind(rocblas_status_success)) :: dgecon
    end function rocsolver_dgecon

    !---------------------------------------------
    ! rocsolver_cgecon
    !---------------------------------------------
    function rocsolver_cgecon(handle, norm_type, n, A, lda, anorm, rcond) &
       result(cgecon) &
       bind(C, name="rocsolver_cgecon")
       import :: c_ptr, rocsolver_norm_type_one, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_norm_type_one)), value :: norm_type
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: anorm
       type(c_ptr), value :: rcond
       integer(kind(rocblas_status_success)) :: cgecon
    end function rocsolver_cgecon

    !---------------------------------------------
    ! rocsolver_zgecon
    !---------------------------------------------
    function rocsolver_zgecon(handle, norm_type, n, A, lda, anorm, rcond) &
       result(zgecon) &
       bind(C, name="rocsolver_zgecon")
       import :: c_ptr, rocsolver_norm_type_one, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_norm_type_one)), value :: norm_type
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: anorm
       type(c_ptr), value :: rcond
       integer(kind(rocblas_status_success)) :: zgecon
    end function rocsolver_zgecon

    !---------------------------------------------
    ! rocsolver_sgecon_64
    !---------------------------------------------
    function rocsolver_sgecon_64(handle, norm_type, n, A, lda, anorm, rcond) &
       result(sgecon_64) &
       bind(C, name="rocsolver_sgecon_64")
       import :: c_ptr, rocsolver_norm_type_one, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_norm_type_one)), value :: norm_type
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: anorm
       type(c_ptr), value :: rcond
       integer(kind(rocblas_status_success)) :: sgecon_64
    end function rocsolver_sgecon_64

    !---------------------------------------------
    ! rocsolver_dgecon_64
    !---------------------------------------------
    function rocsolver_dgecon_64(handle, norm_type, n, A, lda, anorm, rcond) &
       result(dgecon_64) &
       bind(C, name="rocsolver_dgecon_64")
       import :: c_ptr, rocsolver_norm_type_one, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_norm_type_one)), value :: norm_type
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: anorm
       type(c_ptr), value :: rcond
       integer(kind(rocblas_status_success)) :: dgecon_64
    end function rocsolver_dgecon_64

    !---------------------------------------------
    ! rocsolver_cgecon_64
    !---------------------------------------------
    function rocsolver_cgecon_64(handle, norm_type, n, A, lda, anorm, rcond) &
       result(cgecon_64) &
       bind(C, name="rocsolver_cgecon_64")
       import :: c_ptr, rocsolver_norm_type_one, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_norm_type_one)), value :: norm_type
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: anorm
       type(c_ptr), value :: rcond
       integer(kind(rocblas_status_success)) :: cgecon_64
    end function rocsolver_cgecon_64

    !---------------------------------------------
    ! rocsolver_zgecon_64
    !---------------------------------------------
    function rocsolver_zgecon_64(handle, norm_type, n, A, lda, anorm, rcond) &
       result(zgecon_64) &
       bind(C, name="rocsolver_zgecon_64")
       import :: c_ptr, rocsolver_norm_type_one, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_norm_type_one)), value :: norm_type
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: anorm
       type(c_ptr), value :: rcond
       integer(kind(rocblas_status_success)) :: zgecon_64
    end function rocsolver_zgecon_64

    !---------------------------------------------
    ! rocsolver_slaswp
    !---------------------------------------------
    function rocsolver_slaswp(handle, n, A, lda, k1, k2, ipiv, incx) &
       result(slaswp) &
       bind(C, name="rocsolver_slaswp")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int), value :: k1
       integer(c_int), value :: k2
       type(c_ptr), value :: ipiv
       integer(c_int), value :: incx
       integer(kind(rocblas_status_success)) :: slaswp
    end function rocsolver_slaswp

    !---------------------------------------------
    ! rocsolver_dlaswp
    !---------------------------------------------
    function rocsolver_dlaswp(handle, n, A, lda, k1, k2, ipiv, incx) &
       result(dlaswp) &
       bind(C, name="rocsolver_dlaswp")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int), value :: k1
       integer(c_int), value :: k2
       type(c_ptr), value :: ipiv
       integer(c_int), value :: incx
       integer(kind(rocblas_status_success)) :: dlaswp
    end function rocsolver_dlaswp

    !---------------------------------------------
    ! rocsolver_claswp
    !---------------------------------------------
    function rocsolver_claswp(handle, n, A, lda, k1, k2, ipiv, incx) &
       result(claswp) &
       bind(C, name="rocsolver_claswp")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int), value :: k1
       integer(c_int), value :: k2
       type(c_ptr), value :: ipiv
       integer(c_int), value :: incx
       integer(kind(rocblas_status_success)) :: claswp
    end function rocsolver_claswp

    !---------------------------------------------
    ! rocsolver_zlaswp
    !---------------------------------------------
    function rocsolver_zlaswp(handle, n, A, lda, k1, k2, ipiv, incx) &
       result(zlaswp) &
       bind(C, name="rocsolver_zlaswp")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int), value :: k1
       integer(c_int), value :: k2
       type(c_ptr), value :: ipiv
       integer(c_int), value :: incx
       integer(kind(rocblas_status_success)) :: zlaswp
    end function rocsolver_zlaswp

    !---------------------------------------------
    ! rocsolver_slarfg
    !---------------------------------------------
    function rocsolver_slarfg(handle, n, alpha, x, incx, tau) &
       result(slarfg) &
       bind(C, name="rocsolver_slarfg")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: alpha
       type(c_ptr), value :: x
       integer(c_int), value :: incx
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: slarfg
    end function rocsolver_slarfg

    !---------------------------------------------
    ! rocsolver_dlarfg
    !---------------------------------------------
    function rocsolver_dlarfg(handle, n, alpha, x, incx, tau) &
       result(dlarfg) &
       bind(C, name="rocsolver_dlarfg")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: alpha
       type(c_ptr), value :: x
       integer(c_int), value :: incx
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: dlarfg
    end function rocsolver_dlarfg

    !---------------------------------------------
    ! rocsolver_clarfg
    !---------------------------------------------
    function rocsolver_clarfg(handle, n, alpha, x, incx, tau) &
       result(clarfg) &
       bind(C, name="rocsolver_clarfg")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: alpha
       type(c_ptr), value :: x
       integer(c_int), value :: incx
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: clarfg
    end function rocsolver_clarfg

    !---------------------------------------------
    ! rocsolver_zlarfg
    !---------------------------------------------
    function rocsolver_zlarfg(handle, n, alpha, x, incx, tau) &
       result(zlarfg) &
       bind(C, name="rocsolver_zlarfg")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: alpha
       type(c_ptr), value :: x
       integer(c_int), value :: incx
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: zlarfg
    end function rocsolver_zlarfg

    !---------------------------------------------
    ! rocsolver_slarfg_64
    !---------------------------------------------
    function rocsolver_slarfg_64(handle, n, alpha, x, incx, tau) &
       result(slarfg_64) &
       bind(C, name="rocsolver_slarfg_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: n
       type(c_ptr), value :: alpha
       type(c_ptr), value :: x
       integer(c_int64_t), value :: incx
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: slarfg_64
    end function rocsolver_slarfg_64

    !---------------------------------------------
    ! rocsolver_dlarfg_64
    !---------------------------------------------
    function rocsolver_dlarfg_64(handle, n, alpha, x, incx, tau) &
       result(dlarfg_64) &
       bind(C, name="rocsolver_dlarfg_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: n
       type(c_ptr), value :: alpha
       type(c_ptr), value :: x
       integer(c_int64_t), value :: incx
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: dlarfg_64
    end function rocsolver_dlarfg_64

    !---------------------------------------------
    ! rocsolver_clarfg_64
    !---------------------------------------------
    function rocsolver_clarfg_64(handle, n, alpha, x, incx, tau) &
       result(clarfg_64) &
       bind(C, name="rocsolver_clarfg_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: n
       type(c_ptr), value :: alpha
       type(c_ptr), value :: x
       integer(c_int64_t), value :: incx
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: clarfg_64
    end function rocsolver_clarfg_64

    !---------------------------------------------
    ! rocsolver_zlarfg_64
    !---------------------------------------------
    function rocsolver_zlarfg_64(handle, n, alpha, x, incx, tau) &
       result(zlarfg_64) &
       bind(C, name="rocsolver_zlarfg_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: n
       type(c_ptr), value :: alpha
       type(c_ptr), value :: x
       integer(c_int64_t), value :: incx
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: zlarfg_64
    end function rocsolver_zlarfg_64

    !---------------------------------------------
    ! rocsolver_slarft
    !---------------------------------------------
    function rocsolver_slarft(handle, myDirect, storev, n, k, V, ldv, tau, T, ldt) &
       result(slarft) &
       bind(C, name="rocsolver_slarft")
       import :: c_ptr, rocblas_forward_direction, rocblas_column_wise, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_forward_direction)), value :: myDirect
       integer(kind(rocblas_column_wise)), value :: storev
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: tau
       type(c_ptr), value :: T
       integer(c_int), value :: ldt
       integer(kind(rocblas_status_success)) :: slarft
    end function rocsolver_slarft

    !---------------------------------------------
    ! rocsolver_dlarft
    !---------------------------------------------
    function rocsolver_dlarft(handle, myDirect, storev, n, k, V, ldv, tau, T, ldt) &
       result(dlarft) &
       bind(C, name="rocsolver_dlarft")
       import :: c_ptr, rocblas_forward_direction, rocblas_column_wise, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_forward_direction)), value :: myDirect
       integer(kind(rocblas_column_wise)), value :: storev
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: tau
       type(c_ptr), value :: T
       integer(c_int), value :: ldt
       integer(kind(rocblas_status_success)) :: dlarft
    end function rocsolver_dlarft

    !---------------------------------------------
    ! rocsolver_clarft
    !---------------------------------------------
    function rocsolver_clarft(handle, myDirect, storev, n, k, V, ldv, tau, T, ldt) &
       result(clarft) &
       bind(C, name="rocsolver_clarft")
       import :: c_ptr, rocblas_forward_direction, rocblas_column_wise, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_forward_direction)), value :: myDirect
       integer(kind(rocblas_column_wise)), value :: storev
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: tau
       type(c_ptr), value :: T
       integer(c_int), value :: ldt
       integer(kind(rocblas_status_success)) :: clarft
    end function rocsolver_clarft

    !---------------------------------------------
    ! rocsolver_zlarft
    !---------------------------------------------
    function rocsolver_zlarft(handle, myDirect, storev, n, k, V, ldv, tau, T, ldt) &
       result(zlarft) &
       bind(C, name="rocsolver_zlarft")
       import :: c_ptr, rocblas_forward_direction, rocblas_column_wise, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_forward_direction)), value :: myDirect
       integer(kind(rocblas_column_wise)), value :: storev
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: tau
       type(c_ptr), value :: T
       integer(c_int), value :: ldt
       integer(kind(rocblas_status_success)) :: zlarft
    end function rocsolver_zlarft

    !---------------------------------------------
    ! rocsolver_slarft_64
    !---------------------------------------------
    function rocsolver_slarft_64(handle, myDirect, storev, n, k, V, ldv, tau, T, ldt) &
       result(slarft_64) &
       bind(C, name="rocsolver_slarft_64")
       import :: c_ptr, rocblas_forward_direction, rocblas_column_wise, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_forward_direction)), value :: myDirect
       integer(kind(rocblas_column_wise)), value :: storev
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: k
       type(c_ptr), value :: V
       integer(c_int64_t), value :: ldv
       type(c_ptr), value :: tau
       type(c_ptr), value :: T
       integer(c_int64_t), value :: ldt
       integer(kind(rocblas_status_success)) :: slarft_64
    end function rocsolver_slarft_64

    !---------------------------------------------
    ! rocsolver_dlarft_64
    !---------------------------------------------
    function rocsolver_dlarft_64(handle, myDirect, storev, n, k, V, ldv, tau, T, ldt) &
       result(dlarft_64) &
       bind(C, name="rocsolver_dlarft_64")
       import :: c_ptr, rocblas_forward_direction, rocblas_column_wise, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_forward_direction)), value :: myDirect
       integer(kind(rocblas_column_wise)), value :: storev
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: k
       type(c_ptr), value :: V
       integer(c_int64_t), value :: ldv
       type(c_ptr), value :: tau
       type(c_ptr), value :: T
       integer(c_int64_t), value :: ldt
       integer(kind(rocblas_status_success)) :: dlarft_64
    end function rocsolver_dlarft_64

    !---------------------------------------------
    ! rocsolver_clarft_64
    !---------------------------------------------
    function rocsolver_clarft_64(handle, myDirect, storev, n, k, V, ldv, tau, T, ldt) &
       result(clarft_64) &
       bind(C, name="rocsolver_clarft_64")
       import :: c_ptr, rocblas_forward_direction, rocblas_column_wise, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_forward_direction)), value :: myDirect
       integer(kind(rocblas_column_wise)), value :: storev
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: k
       type(c_ptr), value :: V
       integer(c_int64_t), value :: ldv
       type(c_ptr), value :: tau
       type(c_ptr), value :: T
       integer(c_int64_t), value :: ldt
       integer(kind(rocblas_status_success)) :: clarft_64
    end function rocsolver_clarft_64

    !---------------------------------------------
    ! rocsolver_zlarft_64
    !---------------------------------------------
    function rocsolver_zlarft_64(handle, myDirect, storev, n, k, V, ldv, tau, T, ldt) &
       result(zlarft_64) &
       bind(C, name="rocsolver_zlarft_64")
       import :: c_ptr, rocblas_forward_direction, rocblas_column_wise, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_forward_direction)), value :: myDirect
       integer(kind(rocblas_column_wise)), value :: storev
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: k
       type(c_ptr), value :: V
       integer(c_int64_t), value :: ldv
       type(c_ptr), value :: tau
       type(c_ptr), value :: T
       integer(c_int64_t), value :: ldt
       integer(kind(rocblas_status_success)) :: zlarft_64
    end function rocsolver_zlarft_64

    !---------------------------------------------
    ! rocsolver_slarf
    !---------------------------------------------
    function rocsolver_slarf(handle, side, m, n, x, incx, alpha, A, lda) &
       result(slarf) &
       bind(C, name="rocsolver_slarf")
       import :: c_ptr, rocblas_side_left, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: x
       integer(c_int), value :: incx
       type(c_ptr), value :: alpha
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(kind(rocblas_status_success)) :: slarf
    end function rocsolver_slarf

    !---------------------------------------------
    ! rocsolver_dlarf
    !---------------------------------------------
    function rocsolver_dlarf(handle, side, m, n, x, incx, alpha, A, lda) &
       result(dlarf) &
       bind(C, name="rocsolver_dlarf")
       import :: c_ptr, rocblas_side_left, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: x
       integer(c_int), value :: incx
       type(c_ptr), value :: alpha
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(kind(rocblas_status_success)) :: dlarf
    end function rocsolver_dlarf

    !---------------------------------------------
    ! rocsolver_clarf
    !---------------------------------------------
    function rocsolver_clarf(handle, side, m, n, x, incx, alpha, A, lda) &
       result(clarf) &
       bind(C, name="rocsolver_clarf")
       import :: c_ptr, rocblas_side_left, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: x
       integer(c_int), value :: incx
       type(c_ptr), value :: alpha
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(kind(rocblas_status_success)) :: clarf
    end function rocsolver_clarf

    !---------------------------------------------
    ! rocsolver_zlarf
    !---------------------------------------------
    function rocsolver_zlarf(handle, side, m, n, x, incx, alpha, A, lda) &
       result(zlarf) &
       bind(C, name="rocsolver_zlarf")
       import :: c_ptr, rocblas_side_left, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: x
       integer(c_int), value :: incx
       type(c_ptr), value :: alpha
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(kind(rocblas_status_success)) :: zlarf
    end function rocsolver_zlarf

    !---------------------------------------------
    ! rocsolver_slarf_64
    !---------------------------------------------
    function rocsolver_slarf_64(handle, side, m, n, x, incx, alpha, A, lda) &
       result(slarf_64) &
       bind(C, name="rocsolver_slarf_64")
       import :: c_ptr, rocblas_side_left, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: x
       integer(c_int64_t), value :: incx
       type(c_ptr), value :: alpha
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(kind(rocblas_status_success)) :: slarf_64
    end function rocsolver_slarf_64

    !---------------------------------------------
    ! rocsolver_dlarf_64
    !---------------------------------------------
    function rocsolver_dlarf_64(handle, side, m, n, x, incx, alpha, A, lda) &
       result(dlarf_64) &
       bind(C, name="rocsolver_dlarf_64")
       import :: c_ptr, rocblas_side_left, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: x
       integer(c_int64_t), value :: incx
       type(c_ptr), value :: alpha
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(kind(rocblas_status_success)) :: dlarf_64
    end function rocsolver_dlarf_64

    !---------------------------------------------
    ! rocsolver_clarf_64
    !---------------------------------------------
    function rocsolver_clarf_64(handle, side, m, n, x, incx, alpha, A, lda) &
       result(clarf_64) &
       bind(C, name="rocsolver_clarf_64")
       import :: c_ptr, rocblas_side_left, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: x
       integer(c_int64_t), value :: incx
       type(c_ptr), value :: alpha
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(kind(rocblas_status_success)) :: clarf_64
    end function rocsolver_clarf_64

    !---------------------------------------------
    ! rocsolver_zlarf_64
    !---------------------------------------------
    function rocsolver_zlarf_64(handle, side, m, n, x, incx, alpha, A, lda) &
       result(zlarf_64) &
       bind(C, name="rocsolver_zlarf_64")
       import :: c_ptr, rocblas_side_left, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: x
       integer(c_int64_t), value :: incx
       type(c_ptr), value :: alpha
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(kind(rocblas_status_success)) :: zlarf_64
    end function rocsolver_zlarf_64

    !---------------------------------------------
    ! rocsolver_slarfb
    !---------------------------------------------
    function rocsolver_slarfb(handle, side, trans, myDirect, storev, m, n, k, V, ldv, T, ldt, A, &
                              lda) &
       result(slarfb) &
       bind(C, name="rocsolver_slarfb")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, rocblas_forward_direction, &
                 rocblas_column_wise, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(kind(rocblas_forward_direction)), value :: myDirect
       integer(kind(rocblas_column_wise)), value :: storev
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: T
       integer(c_int), value :: ldt
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(kind(rocblas_status_success)) :: slarfb
    end function rocsolver_slarfb

    !---------------------------------------------
    ! rocsolver_dlarfb
    !---------------------------------------------
    function rocsolver_dlarfb(handle, side, trans, myDirect, storev, m, n, k, V, ldv, T, ldt, A, &
                              lda) &
       result(dlarfb) &
       bind(C, name="rocsolver_dlarfb")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, rocblas_forward_direction, &
                 rocblas_column_wise, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(kind(rocblas_forward_direction)), value :: myDirect
       integer(kind(rocblas_column_wise)), value :: storev
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: T
       integer(c_int), value :: ldt
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(kind(rocblas_status_success)) :: dlarfb
    end function rocsolver_dlarfb

    !---------------------------------------------
    ! rocsolver_clarfb
    !---------------------------------------------
    function rocsolver_clarfb(handle, side, trans, myDirect, storev, m, n, k, V, ldv, T, ldt, A, &
                              lda) &
       result(clarfb) &
       bind(C, name="rocsolver_clarfb")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, rocblas_forward_direction, &
                 rocblas_column_wise, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(kind(rocblas_forward_direction)), value :: myDirect
       integer(kind(rocblas_column_wise)), value :: storev
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: T
       integer(c_int), value :: ldt
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(kind(rocblas_status_success)) :: clarfb
    end function rocsolver_clarfb

    !---------------------------------------------
    ! rocsolver_zlarfb
    !---------------------------------------------
    function rocsolver_zlarfb(handle, side, trans, myDirect, storev, m, n, k, V, ldv, T, ldt, A, &
                              lda) &
       result(zlarfb) &
       bind(C, name="rocsolver_zlarfb")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, rocblas_forward_direction, &
                 rocblas_column_wise, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(kind(rocblas_forward_direction)), value :: myDirect
       integer(kind(rocblas_column_wise)), value :: storev
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: T
       integer(c_int), value :: ldt
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(kind(rocblas_status_success)) :: zlarfb
    end function rocsolver_zlarfb

    !---------------------------------------------
    ! rocsolver_slasr
    !---------------------------------------------
    function rocsolver_slasr(handle, side, pivot, myDirect, m, n, C, S, A, lda) &
       result(slasr) &
       bind(C, name="rocsolver_slasr")
       import :: c_ptr, rocblas_side_left, rocblas_pivot_variable, rocblas_forward_direction, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_pivot_variable)), value :: pivot
       integer(kind(rocblas_forward_direction)), value :: myDirect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: C
       type(c_ptr), value :: S
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(kind(rocblas_status_success)) :: slasr
    end function rocsolver_slasr

    !---------------------------------------------
    ! rocsolver_dlasr
    !---------------------------------------------
    function rocsolver_dlasr(handle, side, pivot, myDirect, m, n, C, S, A, lda) &
       result(dlasr) &
       bind(C, name="rocsolver_dlasr")
       import :: c_ptr, rocblas_side_left, rocblas_pivot_variable, rocblas_forward_direction, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_pivot_variable)), value :: pivot
       integer(kind(rocblas_forward_direction)), value :: myDirect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: C
       type(c_ptr), value :: S
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(kind(rocblas_status_success)) :: dlasr
    end function rocsolver_dlasr

    !---------------------------------------------
    ! rocsolver_clasr
    !---------------------------------------------
    function rocsolver_clasr(handle, side, pivot, myDirect, m, n, C, S, A, lda) &
       result(clasr) &
       bind(C, name="rocsolver_clasr")
       import :: c_ptr, rocblas_side_left, rocblas_pivot_variable, rocblas_forward_direction, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_pivot_variable)), value :: pivot
       integer(kind(rocblas_forward_direction)), value :: myDirect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: C
       type(c_ptr), value :: S
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(kind(rocblas_status_success)) :: clasr
    end function rocsolver_clasr

    !---------------------------------------------
    ! rocsolver_zlasr
    !---------------------------------------------
    function rocsolver_zlasr(handle, side, pivot, myDirect, m, n, C, S, A, lda) &
       result(zlasr) &
       bind(C, name="rocsolver_zlasr")
       import :: c_ptr, rocblas_side_left, rocblas_pivot_variable, rocblas_forward_direction, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_pivot_variable)), value :: pivot
       integer(kind(rocblas_forward_direction)), value :: myDirect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: C
       type(c_ptr), value :: S
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(kind(rocblas_status_success)) :: zlasr
    end function rocsolver_zlasr

    !---------------------------------------------
    ! rocsolver_slabrd
    !---------------------------------------------
    function rocsolver_slabrd(handle, m, n, k, A, lda, D, E, tauq, taup, X, ldx, Y, ldy) &
       result(slabrd) &
       bind(C, name="rocsolver_slabrd")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: Y
       integer(c_int), value :: ldy
       integer(kind(rocblas_status_success)) :: slabrd
    end function rocsolver_slabrd

    !---------------------------------------------
    ! rocsolver_dlabrd
    !---------------------------------------------
    function rocsolver_dlabrd(handle, m, n, k, A, lda, D, E, tauq, taup, X, ldx, Y, ldy) &
       result(dlabrd) &
       bind(C, name="rocsolver_dlabrd")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: Y
       integer(c_int), value :: ldy
       integer(kind(rocblas_status_success)) :: dlabrd
    end function rocsolver_dlabrd

    !---------------------------------------------
    ! rocsolver_clabrd
    !---------------------------------------------
    function rocsolver_clabrd(handle, m, n, k, A, lda, D, E, tauq, taup, X, ldx, Y, ldy) &
       result(clabrd) &
       bind(C, name="rocsolver_clabrd")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: Y
       integer(c_int), value :: ldy
       integer(kind(rocblas_status_success)) :: clabrd
    end function rocsolver_clabrd

    !---------------------------------------------
    ! rocsolver_zlabrd
    !---------------------------------------------
    function rocsolver_zlabrd(handle, m, n, k, A, lda, D, E, tauq, taup, X, ldx, Y, ldy) &
       result(zlabrd) &
       bind(C, name="rocsolver_zlabrd")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: Y
       integer(c_int), value :: ldy
       integer(kind(rocblas_status_success)) :: zlabrd
    end function rocsolver_zlabrd

    !---------------------------------------------
    ! rocsolver_slatrd
    !---------------------------------------------
    function rocsolver_slatrd(handle, uplo, n, k, A, lda, E, tau, W, ldw) &
       result(slatrd) &
       bind(C, name="rocsolver_slatrd")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       type(c_ptr), value :: W
       integer(c_int), value :: ldw
       integer(kind(rocblas_status_success)) :: slatrd
    end function rocsolver_slatrd

    !---------------------------------------------
    ! rocsolver_dlatrd
    !---------------------------------------------
    function rocsolver_dlatrd(handle, uplo, n, k, A, lda, E, tau, W, ldw) &
       result(dlatrd) &
       bind(C, name="rocsolver_dlatrd")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       type(c_ptr), value :: W
       integer(c_int), value :: ldw
       integer(kind(rocblas_status_success)) :: dlatrd
    end function rocsolver_dlatrd

    !---------------------------------------------
    ! rocsolver_clatrd
    !---------------------------------------------
    function rocsolver_clatrd(handle, uplo, n, k, A, lda, E, tau, W, ldw) &
       result(clatrd) &
       bind(C, name="rocsolver_clatrd")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       type(c_ptr), value :: W
       integer(c_int), value :: ldw
       integer(kind(rocblas_status_success)) :: clatrd
    end function rocsolver_clatrd

    !---------------------------------------------
    ! rocsolver_zlatrd
    !---------------------------------------------
    function rocsolver_zlatrd(handle, uplo, n, k, A, lda, E, tau, W, ldw) &
       result(zlatrd) &
       bind(C, name="rocsolver_zlatrd")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       type(c_ptr), value :: W
       integer(c_int), value :: ldw
       integer(kind(rocblas_status_success)) :: zlatrd
    end function rocsolver_zlatrd

    !---------------------------------------------
    ! rocsolver_slasyf
    !---------------------------------------------
    function rocsolver_slasyf(handle, uplo, n, nb, kb, A, lda, ipiv, myInfo) &
       result(slasyf) &
       bind(C, name="rocsolver_slasyf")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nb
       type(c_ptr), value :: kb
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: slasyf
    end function rocsolver_slasyf

    !---------------------------------------------
    ! rocsolver_dlasyf
    !---------------------------------------------
    function rocsolver_dlasyf(handle, uplo, n, nb, kb, A, lda, ipiv, myInfo) &
       result(dlasyf) &
       bind(C, name="rocsolver_dlasyf")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nb
       type(c_ptr), value :: kb
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dlasyf
    end function rocsolver_dlasyf

    !---------------------------------------------
    ! rocsolver_clasyf
    !---------------------------------------------
    function rocsolver_clasyf(handle, uplo, n, nb, kb, A, lda, ipiv, myInfo) &
       result(clasyf) &
       bind(C, name="rocsolver_clasyf")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nb
       type(c_ptr), value :: kb
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: clasyf
    end function rocsolver_clasyf

    !---------------------------------------------
    ! rocsolver_zlasyf
    !---------------------------------------------
    function rocsolver_zlasyf(handle, uplo, n, nb, kb, A, lda, ipiv, myInfo) &
       result(zlasyf) &
       bind(C, name="rocsolver_zlasyf")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nb
       type(c_ptr), value :: kb
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zlasyf
    end function rocsolver_zlasyf

    !---------------------------------------------
    ! rocsolver_slauum
    !---------------------------------------------
    function rocsolver_slauum(handle, uplo, n, A, lda) &
       result(slauum) &
       bind(C, name="rocsolver_slauum")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(kind(rocblas_status_success)) :: slauum
    end function rocsolver_slauum

    !---------------------------------------------
    ! rocsolver_dlauum
    !---------------------------------------------
    function rocsolver_dlauum(handle, uplo, n, A, lda) &
       result(dlauum) &
       bind(C, name="rocsolver_dlauum")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(kind(rocblas_status_success)) :: dlauum
    end function rocsolver_dlauum

    !---------------------------------------------
    ! rocsolver_clauum
    !---------------------------------------------
    function rocsolver_clauum(handle, uplo, n, A, lda) &
       result(clauum) &
       bind(C, name="rocsolver_clauum")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(kind(rocblas_status_success)) :: clauum
    end function rocsolver_clauum

    !---------------------------------------------
    ! rocsolver_zlauum
    !---------------------------------------------
    function rocsolver_zlauum(handle, uplo, n, A, lda) &
       result(zlauum) &
       bind(C, name="rocsolver_zlauum")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(kind(rocblas_status_success)) :: zlauum
    end function rocsolver_zlauum

    !---------------------------------------------
    ! rocsolver_sorg2r
    !---------------------------------------------
    function rocsolver_sorg2r(handle, m, n, k, A, lda, ipiv) &
       result(sorg2r) &
       bind(C, name="rocsolver_sorg2r")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: sorg2r
    end function rocsolver_sorg2r

    !---------------------------------------------
    ! rocsolver_dorg2r
    !---------------------------------------------
    function rocsolver_dorg2r(handle, m, n, k, A, lda, ipiv) &
       result(dorg2r) &
       bind(C, name="rocsolver_dorg2r")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: dorg2r
    end function rocsolver_dorg2r

    !---------------------------------------------
    ! rocsolver_cung2r
    !---------------------------------------------
    function rocsolver_cung2r(handle, m, n, k, A, lda, ipiv) &
       result(cung2r) &
       bind(C, name="rocsolver_cung2r")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: cung2r
    end function rocsolver_cung2r

    !---------------------------------------------
    ! rocsolver_zung2r
    !---------------------------------------------
    function rocsolver_zung2r(handle, m, n, k, A, lda, ipiv) &
       result(zung2r) &
       bind(C, name="rocsolver_zung2r")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: zung2r
    end function rocsolver_zung2r

    !---------------------------------------------
    ! rocsolver_sorgqr
    !---------------------------------------------
    function rocsolver_sorgqr(handle, m, n, k, A, lda, ipiv) &
       result(sorgqr) &
       bind(C, name="rocsolver_sorgqr")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: sorgqr
    end function rocsolver_sorgqr

    !---------------------------------------------
    ! rocsolver_dorgqr
    !---------------------------------------------
    function rocsolver_dorgqr(handle, m, n, k, A, lda, ipiv) &
       result(dorgqr) &
       bind(C, name="rocsolver_dorgqr")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: dorgqr
    end function rocsolver_dorgqr

    !---------------------------------------------
    ! rocsolver_cungqr
    !---------------------------------------------
    function rocsolver_cungqr(handle, m, n, k, A, lda, ipiv) &
       result(cungqr) &
       bind(C, name="rocsolver_cungqr")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: cungqr
    end function rocsolver_cungqr

    !---------------------------------------------
    ! rocsolver_zungqr
    !---------------------------------------------
    function rocsolver_zungqr(handle, m, n, k, A, lda, ipiv) &
       result(zungqr) &
       bind(C, name="rocsolver_zungqr")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: zungqr
    end function rocsolver_zungqr

    !---------------------------------------------
    ! rocsolver_sorgl2
    !---------------------------------------------
    function rocsolver_sorgl2(handle, m, n, k, A, lda, ipiv) &
       result(sorgl2) &
       bind(C, name="rocsolver_sorgl2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: sorgl2
    end function rocsolver_sorgl2

    !---------------------------------------------
    ! rocsolver_dorgl2
    !---------------------------------------------
    function rocsolver_dorgl2(handle, m, n, k, A, lda, ipiv) &
       result(dorgl2) &
       bind(C, name="rocsolver_dorgl2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: dorgl2
    end function rocsolver_dorgl2

    !---------------------------------------------
    ! rocsolver_cungl2
    !---------------------------------------------
    function rocsolver_cungl2(handle, m, n, k, A, lda, ipiv) &
       result(cungl2) &
       bind(C, name="rocsolver_cungl2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: cungl2
    end function rocsolver_cungl2

    !---------------------------------------------
    ! rocsolver_zungl2
    !---------------------------------------------
    function rocsolver_zungl2(handle, m, n, k, A, lda, ipiv) &
       result(zungl2) &
       bind(C, name="rocsolver_zungl2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: zungl2
    end function rocsolver_zungl2

    !---------------------------------------------
    ! rocsolver_sorglq
    !---------------------------------------------
    function rocsolver_sorglq(handle, m, n, k, A, lda, ipiv) &
       result(sorglq) &
       bind(C, name="rocsolver_sorglq")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: sorglq
    end function rocsolver_sorglq

    !---------------------------------------------
    ! rocsolver_dorglq
    !---------------------------------------------
    function rocsolver_dorglq(handle, m, n, k, A, lda, ipiv) &
       result(dorglq) &
       bind(C, name="rocsolver_dorglq")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: dorglq
    end function rocsolver_dorglq

    !---------------------------------------------
    ! rocsolver_cunglq
    !---------------------------------------------
    function rocsolver_cunglq(handle, m, n, k, A, lda, ipiv) &
       result(cunglq) &
       bind(C, name="rocsolver_cunglq")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: cunglq
    end function rocsolver_cunglq

    !---------------------------------------------
    ! rocsolver_zunglq
    !---------------------------------------------
    function rocsolver_zunglq(handle, m, n, k, A, lda, ipiv) &
       result(zunglq) &
       bind(C, name="rocsolver_zunglq")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: zunglq
    end function rocsolver_zunglq

    !---------------------------------------------
    ! rocsolver_sorg2l
    !---------------------------------------------
    function rocsolver_sorg2l(handle, m, n, k, A, lda, ipiv) &
       result(sorg2l) &
       bind(C, name="rocsolver_sorg2l")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: sorg2l
    end function rocsolver_sorg2l

    !---------------------------------------------
    ! rocsolver_dorg2l
    !---------------------------------------------
    function rocsolver_dorg2l(handle, m, n, k, A, lda, ipiv) &
       result(dorg2l) &
       bind(C, name="rocsolver_dorg2l")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: dorg2l
    end function rocsolver_dorg2l

    !---------------------------------------------
    ! rocsolver_cung2l
    !---------------------------------------------
    function rocsolver_cung2l(handle, m, n, k, A, lda, ipiv) &
       result(cung2l) &
       bind(C, name="rocsolver_cung2l")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: cung2l
    end function rocsolver_cung2l

    !---------------------------------------------
    ! rocsolver_zung2l
    !---------------------------------------------
    function rocsolver_zung2l(handle, m, n, k, A, lda, ipiv) &
       result(zung2l) &
       bind(C, name="rocsolver_zung2l")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: zung2l
    end function rocsolver_zung2l

    !---------------------------------------------
    ! rocsolver_sorgql
    !---------------------------------------------
    function rocsolver_sorgql(handle, m, n, k, A, lda, ipiv) &
       result(sorgql) &
       bind(C, name="rocsolver_sorgql")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: sorgql
    end function rocsolver_sorgql

    !---------------------------------------------
    ! rocsolver_dorgql
    !---------------------------------------------
    function rocsolver_dorgql(handle, m, n, k, A, lda, ipiv) &
       result(dorgql) &
       bind(C, name="rocsolver_dorgql")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: dorgql
    end function rocsolver_dorgql

    !---------------------------------------------
    ! rocsolver_cungql
    !---------------------------------------------
    function rocsolver_cungql(handle, m, n, k, A, lda, ipiv) &
       result(cungql) &
       bind(C, name="rocsolver_cungql")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: cungql
    end function rocsolver_cungql

    !---------------------------------------------
    ! rocsolver_zungql
    !---------------------------------------------
    function rocsolver_zungql(handle, m, n, k, A, lda, ipiv) &
       result(zungql) &
       bind(C, name="rocsolver_zungql")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: zungql
    end function rocsolver_zungql

    !---------------------------------------------
    ! rocsolver_sorgbr
    !---------------------------------------------
    function rocsolver_sorgbr(handle, storev, m, n, k, A, lda, ipiv) &
       result(sorgbr) &
       bind(C, name="rocsolver_sorgbr")
       import :: c_ptr, rocblas_column_wise, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_column_wise)), value :: storev
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: sorgbr
    end function rocsolver_sorgbr

    !---------------------------------------------
    ! rocsolver_dorgbr
    !---------------------------------------------
    function rocsolver_dorgbr(handle, storev, m, n, k, A, lda, ipiv) &
       result(dorgbr) &
       bind(C, name="rocsolver_dorgbr")
       import :: c_ptr, rocblas_column_wise, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_column_wise)), value :: storev
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: dorgbr
    end function rocsolver_dorgbr

    !---------------------------------------------
    ! rocsolver_cungbr
    !---------------------------------------------
    function rocsolver_cungbr(handle, storev, m, n, k, A, lda, ipiv) &
       result(cungbr) &
       bind(C, name="rocsolver_cungbr")
       import :: c_ptr, rocblas_column_wise, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_column_wise)), value :: storev
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: cungbr
    end function rocsolver_cungbr

    !---------------------------------------------
    ! rocsolver_zungbr
    !---------------------------------------------
    function rocsolver_zungbr(handle, storev, m, n, k, A, lda, ipiv) &
       result(zungbr) &
       bind(C, name="rocsolver_zungbr")
       import :: c_ptr, rocblas_column_wise, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_column_wise)), value :: storev
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: zungbr
    end function rocsolver_zungbr

    !---------------------------------------------
    ! rocsolver_sorgtr
    !---------------------------------------------
    function rocsolver_sorgtr(handle, uplo, n, A, lda, ipiv) &
       result(sorgtr) &
       bind(C, name="rocsolver_sorgtr")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: sorgtr
    end function rocsolver_sorgtr

    !---------------------------------------------
    ! rocsolver_dorgtr
    !---------------------------------------------
    function rocsolver_dorgtr(handle, uplo, n, A, lda, ipiv) &
       result(dorgtr) &
       bind(C, name="rocsolver_dorgtr")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: dorgtr
    end function rocsolver_dorgtr

    !---------------------------------------------
    ! rocsolver_cungtr
    !---------------------------------------------
    function rocsolver_cungtr(handle, uplo, n, A, lda, ipiv) &
       result(cungtr) &
       bind(C, name="rocsolver_cungtr")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: cungtr
    end function rocsolver_cungtr

    !---------------------------------------------
    ! rocsolver_zungtr
    !---------------------------------------------
    function rocsolver_zungtr(handle, uplo, n, A, lda, ipiv) &
       result(zungtr) &
       bind(C, name="rocsolver_zungtr")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: zungtr
    end function rocsolver_zungtr

    !---------------------------------------------
    ! rocsolver_sorm2r
    !---------------------------------------------
    function rocsolver_sorm2r(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(sorm2r) &
       bind(C, name="rocsolver_sorm2r")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: sorm2r
    end function rocsolver_sorm2r

    !---------------------------------------------
    ! rocsolver_dorm2r
    !---------------------------------------------
    function rocsolver_dorm2r(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(dorm2r) &
       bind(C, name="rocsolver_dorm2r")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: dorm2r
    end function rocsolver_dorm2r

    !---------------------------------------------
    ! rocsolver_cunm2r
    !---------------------------------------------
    function rocsolver_cunm2r(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(cunm2r) &
       bind(C, name="rocsolver_cunm2r")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: cunm2r
    end function rocsolver_cunm2r

    !---------------------------------------------
    ! rocsolver_zunm2r
    !---------------------------------------------
    function rocsolver_zunm2r(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(zunm2r) &
       bind(C, name="rocsolver_zunm2r")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: zunm2r
    end function rocsolver_zunm2r

    !---------------------------------------------
    ! rocsolver_sormqr
    !---------------------------------------------
    function rocsolver_sormqr(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(sormqr) &
       bind(C, name="rocsolver_sormqr")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: sormqr
    end function rocsolver_sormqr

    !---------------------------------------------
    ! rocsolver_dormqr
    !---------------------------------------------
    function rocsolver_dormqr(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(dormqr) &
       bind(C, name="rocsolver_dormqr")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: dormqr
    end function rocsolver_dormqr

    !---------------------------------------------
    ! rocsolver_cunmqr
    !---------------------------------------------
    function rocsolver_cunmqr(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(cunmqr) &
       bind(C, name="rocsolver_cunmqr")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: cunmqr
    end function rocsolver_cunmqr

    !---------------------------------------------
    ! rocsolver_zunmqr
    !---------------------------------------------
    function rocsolver_zunmqr(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(zunmqr) &
       bind(C, name="rocsolver_zunmqr")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: zunmqr
    end function rocsolver_zunmqr

    !---------------------------------------------
    ! rocsolver_sorml2
    !---------------------------------------------
    function rocsolver_sorml2(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(sorml2) &
       bind(C, name="rocsolver_sorml2")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: sorml2
    end function rocsolver_sorml2

    !---------------------------------------------
    ! rocsolver_dorml2
    !---------------------------------------------
    function rocsolver_dorml2(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(dorml2) &
       bind(C, name="rocsolver_dorml2")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: dorml2
    end function rocsolver_dorml2

    !---------------------------------------------
    ! rocsolver_cunml2
    !---------------------------------------------
    function rocsolver_cunml2(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(cunml2) &
       bind(C, name="rocsolver_cunml2")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: cunml2
    end function rocsolver_cunml2

    !---------------------------------------------
    ! rocsolver_zunml2
    !---------------------------------------------
    function rocsolver_zunml2(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(zunml2) &
       bind(C, name="rocsolver_zunml2")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: zunml2
    end function rocsolver_zunml2

    !---------------------------------------------
    ! rocsolver_sormlq
    !---------------------------------------------
    function rocsolver_sormlq(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(sormlq) &
       bind(C, name="rocsolver_sormlq")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: sormlq
    end function rocsolver_sormlq

    !---------------------------------------------
    ! rocsolver_dormlq
    !---------------------------------------------
    function rocsolver_dormlq(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(dormlq) &
       bind(C, name="rocsolver_dormlq")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: dormlq
    end function rocsolver_dormlq

    !---------------------------------------------
    ! rocsolver_cunmlq
    !---------------------------------------------
    function rocsolver_cunmlq(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(cunmlq) &
       bind(C, name="rocsolver_cunmlq")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: cunmlq
    end function rocsolver_cunmlq

    !---------------------------------------------
    ! rocsolver_zunmlq
    !---------------------------------------------
    function rocsolver_zunmlq(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(zunmlq) &
       bind(C, name="rocsolver_zunmlq")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: zunmlq
    end function rocsolver_zunmlq

    !---------------------------------------------
    ! rocsolver_sorm2l
    !---------------------------------------------
    function rocsolver_sorm2l(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(sorm2l) &
       bind(C, name="rocsolver_sorm2l")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: sorm2l
    end function rocsolver_sorm2l

    !---------------------------------------------
    ! rocsolver_dorm2l
    !---------------------------------------------
    function rocsolver_dorm2l(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(dorm2l) &
       bind(C, name="rocsolver_dorm2l")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: dorm2l
    end function rocsolver_dorm2l

    !---------------------------------------------
    ! rocsolver_cunm2l
    !---------------------------------------------
    function rocsolver_cunm2l(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(cunm2l) &
       bind(C, name="rocsolver_cunm2l")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: cunm2l
    end function rocsolver_cunm2l

    !---------------------------------------------
    ! rocsolver_zunm2l
    !---------------------------------------------
    function rocsolver_zunm2l(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(zunm2l) &
       bind(C, name="rocsolver_zunm2l")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: zunm2l
    end function rocsolver_zunm2l

    !---------------------------------------------
    ! rocsolver_sormql
    !---------------------------------------------
    function rocsolver_sormql(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(sormql) &
       bind(C, name="rocsolver_sormql")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: sormql
    end function rocsolver_sormql

    !---------------------------------------------
    ! rocsolver_dormql
    !---------------------------------------------
    function rocsolver_dormql(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(dormql) &
       bind(C, name="rocsolver_dormql")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: dormql
    end function rocsolver_dormql

    !---------------------------------------------
    ! rocsolver_cunmql
    !---------------------------------------------
    function rocsolver_cunmql(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(cunmql) &
       bind(C, name="rocsolver_cunmql")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: cunmql
    end function rocsolver_cunmql

    !---------------------------------------------
    ! rocsolver_zunmql
    !---------------------------------------------
    function rocsolver_zunmql(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(zunmql) &
       bind(C, name="rocsolver_zunmql")
       import :: c_ptr, rocblas_side_left, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: zunmql
    end function rocsolver_zunmql

    !---------------------------------------------
    ! rocsolver_sormbr
    !---------------------------------------------
    function rocsolver_sormbr(handle, storev, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(sormbr) &
       bind(C, name="rocsolver_sormbr")
       import :: c_ptr, rocblas_column_wise, rocblas_side_left, rocblas_operation_none, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_column_wise)), value :: storev
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: sormbr
    end function rocsolver_sormbr

    !---------------------------------------------
    ! rocsolver_dormbr
    !---------------------------------------------
    function rocsolver_dormbr(handle, storev, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(dormbr) &
       bind(C, name="rocsolver_dormbr")
       import :: c_ptr, rocblas_column_wise, rocblas_side_left, rocblas_operation_none, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_column_wise)), value :: storev
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: dormbr
    end function rocsolver_dormbr

    !---------------------------------------------
    ! rocsolver_cunmbr
    !---------------------------------------------
    function rocsolver_cunmbr(handle, storev, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(cunmbr) &
       bind(C, name="rocsolver_cunmbr")
       import :: c_ptr, rocblas_column_wise, rocblas_side_left, rocblas_operation_none, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_column_wise)), value :: storev
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: cunmbr
    end function rocsolver_cunmbr

    !---------------------------------------------
    ! rocsolver_zunmbr
    !---------------------------------------------
    function rocsolver_zunmbr(handle, storev, side, trans, m, n, k, A, lda, ipiv, C, ldc) &
       result(zunmbr) &
       bind(C, name="rocsolver_zunmbr")
       import :: c_ptr, rocblas_column_wise, rocblas_side_left, rocblas_operation_none, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_column_wise)), value :: storev
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: zunmbr
    end function rocsolver_zunmbr

    !---------------------------------------------
    ! rocsolver_sormtr
    !---------------------------------------------
    function rocsolver_sormtr(handle, side, uplo, trans, m, n, A, lda, ipiv, C, ldc) &
       result(sormtr) &
       bind(C, name="rocsolver_sormtr")
       import :: c_ptr, rocblas_side_left, rocblas_fill_upper, rocblas_operation_none, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: sormtr
    end function rocsolver_sormtr

    !---------------------------------------------
    ! rocsolver_dormtr
    !---------------------------------------------
    function rocsolver_dormtr(handle, side, uplo, trans, m, n, A, lda, ipiv, C, ldc) &
       result(dormtr) &
       bind(C, name="rocsolver_dormtr")
       import :: c_ptr, rocblas_side_left, rocblas_fill_upper, rocblas_operation_none, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: dormtr
    end function rocsolver_dormtr

    !---------------------------------------------
    ! rocsolver_cunmtr
    !---------------------------------------------
    function rocsolver_cunmtr(handle, side, uplo, trans, m, n, A, lda, ipiv, C, ldc) &
       result(cunmtr) &
       bind(C, name="rocsolver_cunmtr")
       import :: c_ptr, rocblas_side_left, rocblas_fill_upper, rocblas_operation_none, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: cunmtr
    end function rocsolver_cunmtr

    !---------------------------------------------
    ! rocsolver_zunmtr
    !---------------------------------------------
    function rocsolver_zunmtr(handle, side, uplo, trans, m, n, A, lda, ipiv, C, ldc) &
       result(zunmtr) &
       bind(C, name="rocsolver_zunmtr")
       import :: c_ptr, rocblas_side_left, rocblas_fill_upper, rocblas_operation_none, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_side_left)), value :: side
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(kind(rocblas_status_success)) :: zunmtr
    end function rocsolver_zunmtr

    !---------------------------------------------
    ! rocsolver_sbdsqr
    !---------------------------------------------
    function rocsolver_sbdsqr(handle, uplo, n, nv, nu, nc, D, E, V, ldv, U, ldu, C, ldc, myInfo) &
       result(sbdsqr) &
       bind(C, name="rocsolver_sbdsqr")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nv
       integer(c_int), value :: nu
       integer(c_int), value :: nc
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sbdsqr
    end function rocsolver_sbdsqr

    !---------------------------------------------
    ! rocsolver_dbdsqr
    !---------------------------------------------
    function rocsolver_dbdsqr(handle, uplo, n, nv, nu, nc, D, E, V, ldv, U, ldu, C, ldc, myInfo) &
       result(dbdsqr) &
       bind(C, name="rocsolver_dbdsqr")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nv
       integer(c_int), value :: nu
       integer(c_int), value :: nc
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dbdsqr
    end function rocsolver_dbdsqr

    !---------------------------------------------
    ! rocsolver_cbdsqr
    !---------------------------------------------
    function rocsolver_cbdsqr(handle, uplo, n, nv, nu, nc, D, E, V, ldv, U, ldu, C, ldc, myInfo) &
       result(cbdsqr) &
       bind(C, name="rocsolver_cbdsqr")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nv
       integer(c_int), value :: nu
       integer(c_int), value :: nc
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cbdsqr
    end function rocsolver_cbdsqr

    !---------------------------------------------
    ! rocsolver_zbdsqr
    !---------------------------------------------
    function rocsolver_zbdsqr(handle, uplo, n, nv, nu, nc, D, E, V, ldv, U, ldu, C, ldc, myInfo) &
       result(zbdsqr) &
       bind(C, name="rocsolver_zbdsqr")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nv
       integer(c_int), value :: nu
       integer(c_int), value :: nc
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zbdsqr
    end function rocsolver_zbdsqr

    !---------------------------------------------
    ! rocsolver_ssterf
    !---------------------------------------------
    function rocsolver_ssterf(handle, n, D, E, myInfo) &
       result(ssterf) &
       bind(C, name="rocsolver_ssterf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ssterf
    end function rocsolver_ssterf

    !---------------------------------------------
    ! rocsolver_dsterf
    !---------------------------------------------
    function rocsolver_dsterf(handle, n, D, E, myInfo) &
       result(dsterf) &
       bind(C, name="rocsolver_dsterf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dsterf
    end function rocsolver_dsterf

    !---------------------------------------------
    ! rocsolver_ssteqr
    !---------------------------------------------
    function rocsolver_ssteqr(handle, evect, n, D, E, C, ldc, myInfo) &
       result(ssteqr) &
       bind(C, name="rocsolver_ssteqr")
       import :: c_ptr, rocblas_evect_original, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(c_int), value :: n
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ssteqr
    end function rocsolver_ssteqr

    !---------------------------------------------
    ! rocsolver_dsteqr
    !---------------------------------------------
    function rocsolver_dsteqr(handle, evect, n, D, E, C, ldc, myInfo) &
       result(dsteqr) &
       bind(C, name="rocsolver_dsteqr")
       import :: c_ptr, rocblas_evect_original, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(c_int), value :: n
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dsteqr
    end function rocsolver_dsteqr

    !---------------------------------------------
    ! rocsolver_csteqr
    !---------------------------------------------
    function rocsolver_csteqr(handle, evect, n, D, E, C, ldc, myInfo) &
       result(csteqr) &
       bind(C, name="rocsolver_csteqr")
       import :: c_ptr, rocblas_evect_original, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(c_int), value :: n
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: csteqr
    end function rocsolver_csteqr

    !---------------------------------------------
    ! rocsolver_zsteqr
    !---------------------------------------------
    function rocsolver_zsteqr(handle, evect, n, D, E, C, ldc, myInfo) &
       result(zsteqr) &
       bind(C, name="rocsolver_zsteqr")
       import :: c_ptr, rocblas_evect_original, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(c_int), value :: n
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zsteqr
    end function rocsolver_zsteqr

    !---------------------------------------------
    ! rocsolver_sstedc
    !---------------------------------------------
    function rocsolver_sstedc(handle, evect, n, D, E, C, ldc, myInfo) &
       result(sstedc) &
       bind(C, name="rocsolver_sstedc")
       import :: c_ptr, rocblas_evect_original, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(c_int), value :: n
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sstedc
    end function rocsolver_sstedc

    !---------------------------------------------
    ! rocsolver_dstedc
    !---------------------------------------------
    function rocsolver_dstedc(handle, evect, n, D, E, C, ldc, myInfo) &
       result(dstedc) &
       bind(C, name="rocsolver_dstedc")
       import :: c_ptr, rocblas_evect_original, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(c_int), value :: n
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dstedc
    end function rocsolver_dstedc

    !---------------------------------------------
    ! rocsolver_cstedc
    !---------------------------------------------
    function rocsolver_cstedc(handle, evect, n, D, E, C, ldc, myInfo) &
       result(cstedc) &
       bind(C, name="rocsolver_cstedc")
       import :: c_ptr, rocblas_evect_original, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(c_int), value :: n
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cstedc
    end function rocsolver_cstedc

    !---------------------------------------------
    ! rocsolver_zstedc
    !---------------------------------------------
    function rocsolver_zstedc(handle, evect, n, D, E, C, ldc, myInfo) &
       result(zstedc) &
       bind(C, name="rocsolver_zstedc")
       import :: c_ptr, rocblas_evect_original, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(c_int), value :: n
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zstedc
    end function rocsolver_zstedc

    !---------------------------------------------
    ! rocsolver_sstebz
    !---------------------------------------------
    function rocsolver_sstebz(handle, erange, eorder, n, vl, vu, il, iu, abstol, D, E, nev, &
                              nsplit, W, iblock, isplit, myInfo) &
       result(sstebz) &
       bind(C, name="rocsolver_sstebz")
       import :: c_ptr, rocblas_erange_all, rocblas_eorder_blocks, c_int, c_float, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_eorder_blocks)), value :: eorder
       integer(c_int), value :: n
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_float), value :: abstol
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: nev
       type(c_ptr), value :: nsplit
       type(c_ptr), value :: W
       type(c_ptr), value :: iblock
       type(c_ptr), value :: isplit
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sstebz
    end function rocsolver_sstebz

    !---------------------------------------------
    ! rocsolver_dstebz
    !---------------------------------------------
    function rocsolver_dstebz(handle, erange, eorder, n, vl, vu, il, iu, abstol, D, E, nev, &
                              nsplit, W, iblock, isplit, myInfo) &
       result(dstebz) &
       bind(C, name="rocsolver_dstebz")
       import :: c_ptr, rocblas_erange_all, rocblas_eorder_blocks, c_int, c_double, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_eorder_blocks)), value :: eorder
       integer(c_int), value :: n
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_double), value :: abstol
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: nev
       type(c_ptr), value :: nsplit
       type(c_ptr), value :: W
       type(c_ptr), value :: iblock
       type(c_ptr), value :: isplit
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dstebz
    end function rocsolver_dstebz

    !---------------------------------------------
    ! rocsolver_sstein
    !---------------------------------------------
    function rocsolver_sstein(handle, n, D, E, nev, W, iblock, isplit, Z, ldz, ifail, myInfo) &
       result(sstein) &
       bind(C, name="rocsolver_sstein")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: iblock
       type(c_ptr), value :: isplit
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sstein
    end function rocsolver_sstein

    !---------------------------------------------
    ! rocsolver_dstein
    !---------------------------------------------
    function rocsolver_dstein(handle, n, D, E, nev, W, iblock, isplit, Z, ldz, ifail, myInfo) &
       result(dstein) &
       bind(C, name="rocsolver_dstein")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: iblock
       type(c_ptr), value :: isplit
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dstein
    end function rocsolver_dstein

    !---------------------------------------------
    ! rocsolver_cstein
    !---------------------------------------------
    function rocsolver_cstein(handle, n, D, E, nev, W, iblock, isplit, Z, ldz, ifail, myInfo) &
       result(cstein) &
       bind(C, name="rocsolver_cstein")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: iblock
       type(c_ptr), value :: isplit
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cstein
    end function rocsolver_cstein

    !---------------------------------------------
    ! rocsolver_zstein
    !---------------------------------------------
    function rocsolver_zstein(handle, n, D, E, nev, W, iblock, isplit, Z, ldz, ifail, myInfo) &
       result(zstein) &
       bind(C, name="rocsolver_zstein")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: iblock
       type(c_ptr), value :: isplit
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zstein
    end function rocsolver_zstein

    !---------------------------------------------
    ! rocsolver_sbdsvdx
    !---------------------------------------------
    function rocsolver_sbdsvdx(handle, uplo, svect, srange, n, D, E, vl, vu, il, iu, nsv, S, Z, &
                               ldz, ifail, myInfo) &
       result(sbdsvdx) &
       bind(C, name="rocsolver_sbdsvdx")
       import :: c_ptr, rocblas_fill_upper, rocblas_svect_all, rocblas_srange_all, c_int, c_float, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(kind(rocblas_svect_all)), value :: svect
       integer(kind(rocblas_srange_all)), value :: srange
       integer(c_int), value :: n
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nsv
       type(c_ptr), value :: S
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sbdsvdx
    end function rocsolver_sbdsvdx

    !---------------------------------------------
    ! rocsolver_dbdsvdx
    !---------------------------------------------
    function rocsolver_dbdsvdx(handle, uplo, svect, srange, n, D, E, vl, vu, il, iu, nsv, S, Z, &
                               ldz, ifail, myInfo) &
       result(dbdsvdx) &
       bind(C, name="rocsolver_dbdsvdx")
       import :: c_ptr, rocblas_fill_upper, rocblas_svect_all, rocblas_srange_all, c_int, c_double, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(kind(rocblas_svect_all)), value :: svect
       integer(kind(rocblas_srange_all)), value :: srange
       integer(c_int), value :: n
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nsv
       type(c_ptr), value :: S
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dbdsvdx
    end function rocsolver_dbdsvdx

    !---------------------------------------------
    ! rocsolver_sgetf2_npvt
    !---------------------------------------------
    function rocsolver_sgetf2_npvt(handle, m, n, A, lda, myInfo) &
       result(sgetf2_npvt) &
       bind(C, name="rocsolver_sgetf2_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgetf2_npvt
    end function rocsolver_sgetf2_npvt

    !---------------------------------------------
    ! rocsolver_dgetf2_npvt
    !---------------------------------------------
    function rocsolver_dgetf2_npvt(handle, m, n, A, lda, myInfo) &
       result(dgetf2_npvt) &
       bind(C, name="rocsolver_dgetf2_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgetf2_npvt
    end function rocsolver_dgetf2_npvt

    !---------------------------------------------
    ! rocsolver_cgetf2_npvt
    !---------------------------------------------
    function rocsolver_cgetf2_npvt(handle, m, n, A, lda, myInfo) &
       result(cgetf2_npvt) &
       bind(C, name="rocsolver_cgetf2_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgetf2_npvt
    end function rocsolver_cgetf2_npvt

    !---------------------------------------------
    ! rocsolver_zgetf2_npvt
    !---------------------------------------------
    function rocsolver_zgetf2_npvt(handle, m, n, A, lda, myInfo) &
       result(zgetf2_npvt) &
       bind(C, name="rocsolver_zgetf2_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgetf2_npvt
    end function rocsolver_zgetf2_npvt

    !---------------------------------------------
    ! rocsolver_sgetf2_npvt_64
    !---------------------------------------------
    function rocsolver_sgetf2_npvt_64(handle, m, n, A, lda, myInfo) &
       result(sgetf2_npvt_64) &
       bind(C, name="rocsolver_sgetf2_npvt_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgetf2_npvt_64
    end function rocsolver_sgetf2_npvt_64

    !---------------------------------------------
    ! rocsolver_dgetf2_npvt_64
    !---------------------------------------------
    function rocsolver_dgetf2_npvt_64(handle, m, n, A, lda, myInfo) &
       result(dgetf2_npvt_64) &
       bind(C, name="rocsolver_dgetf2_npvt_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgetf2_npvt_64
    end function rocsolver_dgetf2_npvt_64

    !---------------------------------------------
    ! rocsolver_cgetf2_npvt_64
    !---------------------------------------------
    function rocsolver_cgetf2_npvt_64(handle, m, n, A, lda, myInfo) &
       result(cgetf2_npvt_64) &
       bind(C, name="rocsolver_cgetf2_npvt_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgetf2_npvt_64
    end function rocsolver_cgetf2_npvt_64

    !---------------------------------------------
    ! rocsolver_zgetf2_npvt_64
    !---------------------------------------------
    function rocsolver_zgetf2_npvt_64(handle, m, n, A, lda, myInfo) &
       result(zgetf2_npvt_64) &
       bind(C, name="rocsolver_zgetf2_npvt_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgetf2_npvt_64
    end function rocsolver_zgetf2_npvt_64

    !---------------------------------------------
    ! rocsolver_sgetf2_npvt_batched
    !---------------------------------------------
    function rocsolver_sgetf2_npvt_batched(handle, m, n, A, lda, myInfo, batch_count) &
       result(sgetf2_npvt_batched) &
       bind(C, name="rocsolver_sgetf2_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetf2_npvt_batched
    end function rocsolver_sgetf2_npvt_batched

    !---------------------------------------------
    ! rocsolver_dgetf2_npvt_batched
    !---------------------------------------------
    function rocsolver_dgetf2_npvt_batched(handle, m, n, A, lda, myInfo, batch_count) &
       result(dgetf2_npvt_batched) &
       bind(C, name="rocsolver_dgetf2_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetf2_npvt_batched
    end function rocsolver_dgetf2_npvt_batched

    !---------------------------------------------
    ! rocsolver_cgetf2_npvt_batched
    !---------------------------------------------
    function rocsolver_cgetf2_npvt_batched(handle, m, n, A, lda, myInfo, batch_count) &
       result(cgetf2_npvt_batched) &
       bind(C, name="rocsolver_cgetf2_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetf2_npvt_batched
    end function rocsolver_cgetf2_npvt_batched

    !---------------------------------------------
    ! rocsolver_zgetf2_npvt_batched
    !---------------------------------------------
    function rocsolver_zgetf2_npvt_batched(handle, m, n, A, lda, myInfo, batch_count) &
       result(zgetf2_npvt_batched) &
       bind(C, name="rocsolver_zgetf2_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetf2_npvt_batched
    end function rocsolver_zgetf2_npvt_batched

    !---------------------------------------------
    ! rocsolver_sgetf2_npvt_batched_64
    !---------------------------------------------
    function rocsolver_sgetf2_npvt_batched_64(handle, m, n, A, lda, myInfo, batch_count) &
       result(sgetf2_npvt_batched_64) &
       bind(C, name="rocsolver_sgetf2_npvt_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetf2_npvt_batched_64
    end function rocsolver_sgetf2_npvt_batched_64

    !---------------------------------------------
    ! rocsolver_dgetf2_npvt_batched_64
    !---------------------------------------------
    function rocsolver_dgetf2_npvt_batched_64(handle, m, n, A, lda, myInfo, batch_count) &
       result(dgetf2_npvt_batched_64) &
       bind(C, name="rocsolver_dgetf2_npvt_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetf2_npvt_batched_64
    end function rocsolver_dgetf2_npvt_batched_64

    !---------------------------------------------
    ! rocsolver_cgetf2_npvt_batched_64
    !---------------------------------------------
    function rocsolver_cgetf2_npvt_batched_64(handle, m, n, A, lda, myInfo, batch_count) &
       result(cgetf2_npvt_batched_64) &
       bind(C, name="rocsolver_cgetf2_npvt_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetf2_npvt_batched_64
    end function rocsolver_cgetf2_npvt_batched_64

    !---------------------------------------------
    ! rocsolver_zgetf2_npvt_batched_64
    !---------------------------------------------
    function rocsolver_zgetf2_npvt_batched_64(handle, m, n, A, lda, myInfo, batch_count) &
       result(zgetf2_npvt_batched_64) &
       bind(C, name="rocsolver_zgetf2_npvt_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetf2_npvt_batched_64
    end function rocsolver_zgetf2_npvt_batched_64

    !---------------------------------------------
    ! rocsolver_sgetf2_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_sgetf2_npvt_strided_batched(handle, m, n, A, lda, strideA, myInfo, &
                                                   batch_count) &
       result(sgetf2_npvt_strided_batched) &
       bind(C, name="rocsolver_sgetf2_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetf2_npvt_strided_batched
    end function rocsolver_sgetf2_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_dgetf2_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_dgetf2_npvt_strided_batched(handle, m, n, A, lda, strideA, myInfo, &
                                                   batch_count) &
       result(dgetf2_npvt_strided_batched) &
       bind(C, name="rocsolver_dgetf2_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetf2_npvt_strided_batched
    end function rocsolver_dgetf2_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_cgetf2_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_cgetf2_npvt_strided_batched(handle, m, n, A, lda, strideA, myInfo, &
                                                   batch_count) &
       result(cgetf2_npvt_strided_batched) &
       bind(C, name="rocsolver_cgetf2_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetf2_npvt_strided_batched
    end function rocsolver_cgetf2_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_zgetf2_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_zgetf2_npvt_strided_batched(handle, m, n, A, lda, strideA, myInfo, &
                                                   batch_count) &
       result(zgetf2_npvt_strided_batched) &
       bind(C, name="rocsolver_zgetf2_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetf2_npvt_strided_batched
    end function rocsolver_zgetf2_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_sgetf2_npvt_strided_batched_64
    !---------------------------------------------
    function rocsolver_sgetf2_npvt_strided_batched_64(handle, m, n, A, lda, strideA, myInfo, &
                                                      batch_count) &
       result(sgetf2_npvt_strided_batched_64) &
       bind(C, name="rocsolver_sgetf2_npvt_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetf2_npvt_strided_batched_64
    end function rocsolver_sgetf2_npvt_strided_batched_64

    !---------------------------------------------
    ! rocsolver_dgetf2_npvt_strided_batched_64
    !---------------------------------------------
    function rocsolver_dgetf2_npvt_strided_batched_64(handle, m, n, A, lda, strideA, myInfo, &
                                                      batch_count) &
       result(dgetf2_npvt_strided_batched_64) &
       bind(C, name="rocsolver_dgetf2_npvt_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetf2_npvt_strided_batched_64
    end function rocsolver_dgetf2_npvt_strided_batched_64

    !---------------------------------------------
    ! rocsolver_cgetf2_npvt_strided_batched_64
    !---------------------------------------------
    function rocsolver_cgetf2_npvt_strided_batched_64(handle, m, n, A, lda, strideA, myInfo, &
                                                      batch_count) &
       result(cgetf2_npvt_strided_batched_64) &
       bind(C, name="rocsolver_cgetf2_npvt_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetf2_npvt_strided_batched_64
    end function rocsolver_cgetf2_npvt_strided_batched_64

    !---------------------------------------------
    ! rocsolver_zgetf2_npvt_strided_batched_64
    !---------------------------------------------
    function rocsolver_zgetf2_npvt_strided_batched_64(handle, m, n, A, lda, strideA, myInfo, &
                                                      batch_count) &
       result(zgetf2_npvt_strided_batched_64) &
       bind(C, name="rocsolver_zgetf2_npvt_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetf2_npvt_strided_batched_64
    end function rocsolver_zgetf2_npvt_strided_batched_64

    !---------------------------------------------
    ! rocsolver_sgetrf_npvt
    !---------------------------------------------
    function rocsolver_sgetrf_npvt(handle, m, n, A, lda, myInfo) &
       result(sgetrf_npvt) &
       bind(C, name="rocsolver_sgetrf_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgetrf_npvt
    end function rocsolver_sgetrf_npvt

    !---------------------------------------------
    ! rocsolver_dgetrf_npvt
    !---------------------------------------------
    function rocsolver_dgetrf_npvt(handle, m, n, A, lda, myInfo) &
       result(dgetrf_npvt) &
       bind(C, name="rocsolver_dgetrf_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgetrf_npvt
    end function rocsolver_dgetrf_npvt

    !---------------------------------------------
    ! rocsolver_cgetrf_npvt
    !---------------------------------------------
    function rocsolver_cgetrf_npvt(handle, m, n, A, lda, myInfo) &
       result(cgetrf_npvt) &
       bind(C, name="rocsolver_cgetrf_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgetrf_npvt
    end function rocsolver_cgetrf_npvt

    !---------------------------------------------
    ! rocsolver_zgetrf_npvt
    !---------------------------------------------
    function rocsolver_zgetrf_npvt(handle, m, n, A, lda, myInfo) &
       result(zgetrf_npvt) &
       bind(C, name="rocsolver_zgetrf_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgetrf_npvt
    end function rocsolver_zgetrf_npvt

    !---------------------------------------------
    ! rocsolver_sgetrf_npvt_64
    !---------------------------------------------
    function rocsolver_sgetrf_npvt_64(handle, m, n, A, lda, myInfo) &
       result(sgetrf_npvt_64) &
       bind(C, name="rocsolver_sgetrf_npvt_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgetrf_npvt_64
    end function rocsolver_sgetrf_npvt_64

    !---------------------------------------------
    ! rocsolver_dgetrf_npvt_64
    !---------------------------------------------
    function rocsolver_dgetrf_npvt_64(handle, m, n, A, lda, myInfo) &
       result(dgetrf_npvt_64) &
       bind(C, name="rocsolver_dgetrf_npvt_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgetrf_npvt_64
    end function rocsolver_dgetrf_npvt_64

    !---------------------------------------------
    ! rocsolver_cgetrf_npvt_64
    !---------------------------------------------
    function rocsolver_cgetrf_npvt_64(handle, m, n, A, lda, myInfo) &
       result(cgetrf_npvt_64) &
       bind(C, name="rocsolver_cgetrf_npvt_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgetrf_npvt_64
    end function rocsolver_cgetrf_npvt_64

    !---------------------------------------------
    ! rocsolver_zgetrf_npvt_64
    !---------------------------------------------
    function rocsolver_zgetrf_npvt_64(handle, m, n, A, lda, myInfo) &
       result(zgetrf_npvt_64) &
       bind(C, name="rocsolver_zgetrf_npvt_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgetrf_npvt_64
    end function rocsolver_zgetrf_npvt_64

    !---------------------------------------------
    ! rocsolver_sgetrf_npvt_batched
    !---------------------------------------------
    function rocsolver_sgetrf_npvt_batched(handle, m, n, A, lda, myInfo, batch_count) &
       result(sgetrf_npvt_batched) &
       bind(C, name="rocsolver_sgetrf_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetrf_npvt_batched
    end function rocsolver_sgetrf_npvt_batched

    !---------------------------------------------
    ! rocsolver_dgetrf_npvt_batched
    !---------------------------------------------
    function rocsolver_dgetrf_npvt_batched(handle, m, n, A, lda, myInfo, batch_count) &
       result(dgetrf_npvt_batched) &
       bind(C, name="rocsolver_dgetrf_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetrf_npvt_batched
    end function rocsolver_dgetrf_npvt_batched

    !---------------------------------------------
    ! rocsolver_cgetrf_npvt_batched
    !---------------------------------------------
    function rocsolver_cgetrf_npvt_batched(handle, m, n, A, lda, myInfo, batch_count) &
       result(cgetrf_npvt_batched) &
       bind(C, name="rocsolver_cgetrf_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetrf_npvt_batched
    end function rocsolver_cgetrf_npvt_batched

    !---------------------------------------------
    ! rocsolver_zgetrf_npvt_batched
    !---------------------------------------------
    function rocsolver_zgetrf_npvt_batched(handle, m, n, A, lda, myInfo, batch_count) &
       result(zgetrf_npvt_batched) &
       bind(C, name="rocsolver_zgetrf_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetrf_npvt_batched
    end function rocsolver_zgetrf_npvt_batched

    !---------------------------------------------
    ! rocsolver_sgetrf_npvt_batched_64
    !---------------------------------------------
    function rocsolver_sgetrf_npvt_batched_64(handle, m, n, A, lda, myInfo, batch_count) &
       result(sgetrf_npvt_batched_64) &
       bind(C, name="rocsolver_sgetrf_npvt_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetrf_npvt_batched_64
    end function rocsolver_sgetrf_npvt_batched_64

    !---------------------------------------------
    ! rocsolver_dgetrf_npvt_batched_64
    !---------------------------------------------
    function rocsolver_dgetrf_npvt_batched_64(handle, m, n, A, lda, myInfo, batch_count) &
       result(dgetrf_npvt_batched_64) &
       bind(C, name="rocsolver_dgetrf_npvt_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetrf_npvt_batched_64
    end function rocsolver_dgetrf_npvt_batched_64

    !---------------------------------------------
    ! rocsolver_cgetrf_npvt_batched_64
    !---------------------------------------------
    function rocsolver_cgetrf_npvt_batched_64(handle, m, n, A, lda, myInfo, batch_count) &
       result(cgetrf_npvt_batched_64) &
       bind(C, name="rocsolver_cgetrf_npvt_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetrf_npvt_batched_64
    end function rocsolver_cgetrf_npvt_batched_64

    !---------------------------------------------
    ! rocsolver_zgetrf_npvt_batched_64
    !---------------------------------------------
    function rocsolver_zgetrf_npvt_batched_64(handle, m, n, A, lda, myInfo, batch_count) &
       result(zgetrf_npvt_batched_64) &
       bind(C, name="rocsolver_zgetrf_npvt_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetrf_npvt_batched_64
    end function rocsolver_zgetrf_npvt_batched_64

    !---------------------------------------------
    ! rocsolver_sgetrf_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_sgetrf_npvt_strided_batched(handle, m, n, A, lda, strideA, myInfo, &
                                                   batch_count) &
       result(sgetrf_npvt_strided_batched) &
       bind(C, name="rocsolver_sgetrf_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetrf_npvt_strided_batched
    end function rocsolver_sgetrf_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_dgetrf_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_dgetrf_npvt_strided_batched(handle, m, n, A, lda, strideA, myInfo, &
                                                   batch_count) &
       result(dgetrf_npvt_strided_batched) &
       bind(C, name="rocsolver_dgetrf_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetrf_npvt_strided_batched
    end function rocsolver_dgetrf_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_cgetrf_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_cgetrf_npvt_strided_batched(handle, m, n, A, lda, strideA, myInfo, &
                                                   batch_count) &
       result(cgetrf_npvt_strided_batched) &
       bind(C, name="rocsolver_cgetrf_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetrf_npvt_strided_batched
    end function rocsolver_cgetrf_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_zgetrf_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_zgetrf_npvt_strided_batched(handle, m, n, A, lda, strideA, myInfo, &
                                                   batch_count) &
       result(zgetrf_npvt_strided_batched) &
       bind(C, name="rocsolver_zgetrf_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetrf_npvt_strided_batched
    end function rocsolver_zgetrf_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_sgetrf_npvt_strided_batched_64
    !---------------------------------------------
    function rocsolver_sgetrf_npvt_strided_batched_64(handle, m, n, A, lda, strideA, myInfo, &
                                                      batch_count) &
       result(sgetrf_npvt_strided_batched_64) &
       bind(C, name="rocsolver_sgetrf_npvt_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetrf_npvt_strided_batched_64
    end function rocsolver_sgetrf_npvt_strided_batched_64

    !---------------------------------------------
    ! rocsolver_dgetrf_npvt_strided_batched_64
    !---------------------------------------------
    function rocsolver_dgetrf_npvt_strided_batched_64(handle, m, n, A, lda, strideA, myInfo, &
                                                      batch_count) &
       result(dgetrf_npvt_strided_batched_64) &
       bind(C, name="rocsolver_dgetrf_npvt_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetrf_npvt_strided_batched_64
    end function rocsolver_dgetrf_npvt_strided_batched_64

    !---------------------------------------------
    ! rocsolver_cgetrf_npvt_strided_batched_64
    !---------------------------------------------
    function rocsolver_cgetrf_npvt_strided_batched_64(handle, m, n, A, lda, strideA, myInfo, &
                                                      batch_count) &
       result(cgetrf_npvt_strided_batched_64) &
       bind(C, name="rocsolver_cgetrf_npvt_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetrf_npvt_strided_batched_64
    end function rocsolver_cgetrf_npvt_strided_batched_64

    !---------------------------------------------
    ! rocsolver_zgetrf_npvt_strided_batched_64
    !---------------------------------------------
    function rocsolver_zgetrf_npvt_strided_batched_64(handle, m, n, A, lda, strideA, myInfo, &
                                                      batch_count) &
       result(zgetrf_npvt_strided_batched_64) &
       bind(C, name="rocsolver_zgetrf_npvt_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetrf_npvt_strided_batched_64
    end function rocsolver_zgetrf_npvt_strided_batched_64

    !---------------------------------------------
    ! rocsolver_sgetf2
    !---------------------------------------------
    function rocsolver_sgetf2(handle, m, n, A, lda, ipiv, myInfo) &
       result(sgetf2) &
       bind(C, name="rocsolver_sgetf2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgetf2
    end function rocsolver_sgetf2

    !---------------------------------------------
    ! rocsolver_dgetf2
    !---------------------------------------------
    function rocsolver_dgetf2(handle, m, n, A, lda, ipiv, myInfo) &
       result(dgetf2) &
       bind(C, name="rocsolver_dgetf2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgetf2
    end function rocsolver_dgetf2

    !---------------------------------------------
    ! rocsolver_cgetf2
    !---------------------------------------------
    function rocsolver_cgetf2(handle, m, n, A, lda, ipiv, myInfo) &
       result(cgetf2) &
       bind(C, name="rocsolver_cgetf2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgetf2
    end function rocsolver_cgetf2

    !---------------------------------------------
    ! rocsolver_zgetf2
    !---------------------------------------------
    function rocsolver_zgetf2(handle, m, n, A, lda, ipiv, myInfo) &
       result(zgetf2) &
       bind(C, name="rocsolver_zgetf2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgetf2
    end function rocsolver_zgetf2

    !---------------------------------------------
    ! rocsolver_sgetf2_64
    !---------------------------------------------
    function rocsolver_sgetf2_64(handle, m, n, A, lda, ipiv, myInfo) &
       result(sgetf2_64) &
       bind(C, name="rocsolver_sgetf2_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgetf2_64
    end function rocsolver_sgetf2_64

    !---------------------------------------------
    ! rocsolver_dgetf2_64
    !---------------------------------------------
    function rocsolver_dgetf2_64(handle, m, n, A, lda, ipiv, myInfo) &
       result(dgetf2_64) &
       bind(C, name="rocsolver_dgetf2_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgetf2_64
    end function rocsolver_dgetf2_64

    !---------------------------------------------
    ! rocsolver_cgetf2_64
    !---------------------------------------------
    function rocsolver_cgetf2_64(handle, m, n, A, lda, ipiv, myInfo) &
       result(cgetf2_64) &
       bind(C, name="rocsolver_cgetf2_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgetf2_64
    end function rocsolver_cgetf2_64

    !---------------------------------------------
    ! rocsolver_zgetf2_64
    !---------------------------------------------
    function rocsolver_zgetf2_64(handle, m, n, A, lda, ipiv, myInfo) &
       result(zgetf2_64) &
       bind(C, name="rocsolver_zgetf2_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgetf2_64
    end function rocsolver_zgetf2_64

    !---------------------------------------------
    ! rocsolver_sgetf2_batched
    !---------------------------------------------
    function rocsolver_sgetf2_batched(handle, m, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(sgetf2_batched) &
       bind(C, name="rocsolver_sgetf2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetf2_batched
    end function rocsolver_sgetf2_batched

    !---------------------------------------------
    ! rocsolver_dgetf2_batched
    !---------------------------------------------
    function rocsolver_dgetf2_batched(handle, m, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(dgetf2_batched) &
       bind(C, name="rocsolver_dgetf2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetf2_batched
    end function rocsolver_dgetf2_batched

    !---------------------------------------------
    ! rocsolver_cgetf2_batched
    !---------------------------------------------
    function rocsolver_cgetf2_batched(handle, m, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(cgetf2_batched) &
       bind(C, name="rocsolver_cgetf2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetf2_batched
    end function rocsolver_cgetf2_batched

    !---------------------------------------------
    ! rocsolver_zgetf2_batched
    !---------------------------------------------
    function rocsolver_zgetf2_batched(handle, m, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(zgetf2_batched) &
       bind(C, name="rocsolver_zgetf2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetf2_batched
    end function rocsolver_zgetf2_batched

    !---------------------------------------------
    ! rocsolver_sgetf2_batched_64
    !---------------------------------------------
    function rocsolver_sgetf2_batched_64(handle, m, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(sgetf2_batched_64) &
       bind(C, name="rocsolver_sgetf2_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetf2_batched_64
    end function rocsolver_sgetf2_batched_64

    !---------------------------------------------
    ! rocsolver_dgetf2_batched_64
    !---------------------------------------------
    function rocsolver_dgetf2_batched_64(handle, m, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(dgetf2_batched_64) &
       bind(C, name="rocsolver_dgetf2_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetf2_batched_64
    end function rocsolver_dgetf2_batched_64

    !---------------------------------------------
    ! rocsolver_cgetf2_batched_64
    !---------------------------------------------
    function rocsolver_cgetf2_batched_64(handle, m, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(cgetf2_batched_64) &
       bind(C, name="rocsolver_cgetf2_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetf2_batched_64
    end function rocsolver_cgetf2_batched_64

    !---------------------------------------------
    ! rocsolver_zgetf2_batched_64
    !---------------------------------------------
    function rocsolver_zgetf2_batched_64(handle, m, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(zgetf2_batched_64) &
       bind(C, name="rocsolver_zgetf2_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetf2_batched_64
    end function rocsolver_zgetf2_batched_64

    !---------------------------------------------
    ! rocsolver_sgetf2_strided_batched
    !---------------------------------------------
    function rocsolver_sgetf2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              myInfo, batch_count) &
       result(sgetf2_strided_batched) &
       bind(C, name="rocsolver_sgetf2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetf2_strided_batched
    end function rocsolver_sgetf2_strided_batched

    !---------------------------------------------
    ! rocsolver_dgetf2_strided_batched
    !---------------------------------------------
    function rocsolver_dgetf2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              myInfo, batch_count) &
       result(dgetf2_strided_batched) &
       bind(C, name="rocsolver_dgetf2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetf2_strided_batched
    end function rocsolver_dgetf2_strided_batched

    !---------------------------------------------
    ! rocsolver_cgetf2_strided_batched
    !---------------------------------------------
    function rocsolver_cgetf2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              myInfo, batch_count) &
       result(cgetf2_strided_batched) &
       bind(C, name="rocsolver_cgetf2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetf2_strided_batched
    end function rocsolver_cgetf2_strided_batched

    !---------------------------------------------
    ! rocsolver_zgetf2_strided_batched
    !---------------------------------------------
    function rocsolver_zgetf2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              myInfo, batch_count) &
       result(zgetf2_strided_batched) &
       bind(C, name="rocsolver_zgetf2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetf2_strided_batched
    end function rocsolver_zgetf2_strided_batched

    !---------------------------------------------
    ! rocsolver_sgetf2_strided_batched_64
    !---------------------------------------------
    function rocsolver_sgetf2_strided_batched_64(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                                 myInfo, batch_count) &
       result(sgetf2_strided_batched_64) &
       bind(C, name="rocsolver_sgetf2_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetf2_strided_batched_64
    end function rocsolver_sgetf2_strided_batched_64

    !---------------------------------------------
    ! rocsolver_dgetf2_strided_batched_64
    !---------------------------------------------
    function rocsolver_dgetf2_strided_batched_64(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                                 myInfo, batch_count) &
       result(dgetf2_strided_batched_64) &
       bind(C, name="rocsolver_dgetf2_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetf2_strided_batched_64
    end function rocsolver_dgetf2_strided_batched_64

    !---------------------------------------------
    ! rocsolver_cgetf2_strided_batched_64
    !---------------------------------------------
    function rocsolver_cgetf2_strided_batched_64(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                                 myInfo, batch_count) &
       result(cgetf2_strided_batched_64) &
       bind(C, name="rocsolver_cgetf2_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetf2_strided_batched_64
    end function rocsolver_cgetf2_strided_batched_64

    !---------------------------------------------
    ! rocsolver_zgetf2_strided_batched_64
    !---------------------------------------------
    function rocsolver_zgetf2_strided_batched_64(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                                 myInfo, batch_count) &
       result(zgetf2_strided_batched_64) &
       bind(C, name="rocsolver_zgetf2_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetf2_strided_batched_64
    end function rocsolver_zgetf2_strided_batched_64

    !---------------------------------------------
    ! rocsolver_sgetrf
    !---------------------------------------------
    function rocsolver_sgetrf(handle, m, n, A, lda, ipiv, myInfo) &
       result(sgetrf) &
       bind(C, name="rocsolver_sgetrf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgetrf
    end function rocsolver_sgetrf

    !---------------------------------------------
    ! rocsolver_dgetrf
    !---------------------------------------------
    function rocsolver_dgetrf(handle, m, n, A, lda, ipiv, myInfo) &
       result(dgetrf) &
       bind(C, name="rocsolver_dgetrf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgetrf
    end function rocsolver_dgetrf

    !---------------------------------------------
    ! rocsolver_cgetrf
    !---------------------------------------------
    function rocsolver_cgetrf(handle, m, n, A, lda, ipiv, myInfo) &
       result(cgetrf) &
       bind(C, name="rocsolver_cgetrf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgetrf
    end function rocsolver_cgetrf

    !---------------------------------------------
    ! rocsolver_zgetrf
    !---------------------------------------------
    function rocsolver_zgetrf(handle, m, n, A, lda, ipiv, myInfo) &
       result(zgetrf) &
       bind(C, name="rocsolver_zgetrf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgetrf
    end function rocsolver_zgetrf

    !---------------------------------------------
    ! rocsolver_sgetrf_64
    !---------------------------------------------
    function rocsolver_sgetrf_64(handle, m, n, A, lda, ipiv, myInfo) &
       result(sgetrf_64) &
       bind(C, name="rocsolver_sgetrf_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgetrf_64
    end function rocsolver_sgetrf_64

    !---------------------------------------------
    ! rocsolver_dgetrf_64
    !---------------------------------------------
    function rocsolver_dgetrf_64(handle, m, n, A, lda, ipiv, myInfo) &
       result(dgetrf_64) &
       bind(C, name="rocsolver_dgetrf_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgetrf_64
    end function rocsolver_dgetrf_64

    !---------------------------------------------
    ! rocsolver_cgetrf_64
    !---------------------------------------------
    function rocsolver_cgetrf_64(handle, m, n, A, lda, ipiv, myInfo) &
       result(cgetrf_64) &
       bind(C, name="rocsolver_cgetrf_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgetrf_64
    end function rocsolver_cgetrf_64

    !---------------------------------------------
    ! rocsolver_zgetrf_64
    !---------------------------------------------
    function rocsolver_zgetrf_64(handle, m, n, A, lda, ipiv, myInfo) &
       result(zgetrf_64) &
       bind(C, name="rocsolver_zgetrf_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgetrf_64
    end function rocsolver_zgetrf_64

    !---------------------------------------------
    ! rocsolver_sgetrf_batched
    !---------------------------------------------
    function rocsolver_sgetrf_batched(handle, m, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(sgetrf_batched) &
       bind(C, name="rocsolver_sgetrf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetrf_batched
    end function rocsolver_sgetrf_batched

    !---------------------------------------------
    ! rocsolver_dgetrf_batched
    !---------------------------------------------
    function rocsolver_dgetrf_batched(handle, m, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(dgetrf_batched) &
       bind(C, name="rocsolver_dgetrf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetrf_batched
    end function rocsolver_dgetrf_batched

    !---------------------------------------------
    ! rocsolver_cgetrf_batched
    !---------------------------------------------
    function rocsolver_cgetrf_batched(handle, m, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(cgetrf_batched) &
       bind(C, name="rocsolver_cgetrf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetrf_batched
    end function rocsolver_cgetrf_batched

    !---------------------------------------------
    ! rocsolver_zgetrf_batched
    !---------------------------------------------
    function rocsolver_zgetrf_batched(handle, m, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(zgetrf_batched) &
       bind(C, name="rocsolver_zgetrf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetrf_batched
    end function rocsolver_zgetrf_batched

    !---------------------------------------------
    ! rocsolver_sgetrf_batched_64
    !---------------------------------------------
    function rocsolver_sgetrf_batched_64(handle, m, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(sgetrf_batched_64) &
       bind(C, name="rocsolver_sgetrf_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetrf_batched_64
    end function rocsolver_sgetrf_batched_64

    !---------------------------------------------
    ! rocsolver_dgetrf_batched_64
    !---------------------------------------------
    function rocsolver_dgetrf_batched_64(handle, m, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(dgetrf_batched_64) &
       bind(C, name="rocsolver_dgetrf_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetrf_batched_64
    end function rocsolver_dgetrf_batched_64

    !---------------------------------------------
    ! rocsolver_cgetrf_batched_64
    !---------------------------------------------
    function rocsolver_cgetrf_batched_64(handle, m, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(cgetrf_batched_64) &
       bind(C, name="rocsolver_cgetrf_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetrf_batched_64
    end function rocsolver_cgetrf_batched_64

    !---------------------------------------------
    ! rocsolver_zgetrf_batched_64
    !---------------------------------------------
    function rocsolver_zgetrf_batched_64(handle, m, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(zgetrf_batched_64) &
       bind(C, name="rocsolver_zgetrf_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetrf_batched_64
    end function rocsolver_zgetrf_batched_64

    !---------------------------------------------
    ! rocsolver_sgetrf_strided_batched
    !---------------------------------------------
    function rocsolver_sgetrf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              myInfo, batch_count) &
       result(sgetrf_strided_batched) &
       bind(C, name="rocsolver_sgetrf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetrf_strided_batched
    end function rocsolver_sgetrf_strided_batched

    !---------------------------------------------
    ! rocsolver_dgetrf_strided_batched
    !---------------------------------------------
    function rocsolver_dgetrf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              myInfo, batch_count) &
       result(dgetrf_strided_batched) &
       bind(C, name="rocsolver_dgetrf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetrf_strided_batched
    end function rocsolver_dgetrf_strided_batched

    !---------------------------------------------
    ! rocsolver_cgetrf_strided_batched
    !---------------------------------------------
    function rocsolver_cgetrf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              myInfo, batch_count) &
       result(cgetrf_strided_batched) &
       bind(C, name="rocsolver_cgetrf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetrf_strided_batched
    end function rocsolver_cgetrf_strided_batched

    !---------------------------------------------
    ! rocsolver_zgetrf_strided_batched
    !---------------------------------------------
    function rocsolver_zgetrf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              myInfo, batch_count) &
       result(zgetrf_strided_batched) &
       bind(C, name="rocsolver_zgetrf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetrf_strided_batched
    end function rocsolver_zgetrf_strided_batched

    !---------------------------------------------
    ! rocsolver_sgetrf_strided_batched_64
    !---------------------------------------------
    function rocsolver_sgetrf_strided_batched_64(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                                 myInfo, batch_count) &
       result(sgetrf_strided_batched_64) &
       bind(C, name="rocsolver_sgetrf_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetrf_strided_batched_64
    end function rocsolver_sgetrf_strided_batched_64

    !---------------------------------------------
    ! rocsolver_dgetrf_strided_batched_64
    !---------------------------------------------
    function rocsolver_dgetrf_strided_batched_64(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                                 myInfo, batch_count) &
       result(dgetrf_strided_batched_64) &
       bind(C, name="rocsolver_dgetrf_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetrf_strided_batched_64
    end function rocsolver_dgetrf_strided_batched_64

    !---------------------------------------------
    ! rocsolver_cgetrf_strided_batched_64
    !---------------------------------------------
    function rocsolver_cgetrf_strided_batched_64(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                                 myInfo, batch_count) &
       result(cgetrf_strided_batched_64) &
       bind(C, name="rocsolver_cgetrf_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetrf_strided_batched_64
    end function rocsolver_cgetrf_strided_batched_64

    !---------------------------------------------
    ! rocsolver_zgetrf_strided_batched_64
    !---------------------------------------------
    function rocsolver_zgetrf_strided_batched_64(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                                 myInfo, batch_count) &
       result(zgetrf_strided_batched_64) &
       bind(C, name="rocsolver_zgetrf_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetrf_strided_batched_64
    end function rocsolver_zgetrf_strided_batched_64

    !---------------------------------------------
    ! rocsolver_sgeqr2
    !---------------------------------------------
    function rocsolver_sgeqr2(handle, m, n, A, lda, ipiv) &
       result(sgeqr2) &
       bind(C, name="rocsolver_sgeqr2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: sgeqr2
    end function rocsolver_sgeqr2

    !---------------------------------------------
    ! rocsolver_dgeqr2
    !---------------------------------------------
    function rocsolver_dgeqr2(handle, m, n, A, lda, ipiv) &
       result(dgeqr2) &
       bind(C, name="rocsolver_dgeqr2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: dgeqr2
    end function rocsolver_dgeqr2

    !---------------------------------------------
    ! rocsolver_cgeqr2
    !---------------------------------------------
    function rocsolver_cgeqr2(handle, m, n, A, lda, ipiv) &
       result(cgeqr2) &
       bind(C, name="rocsolver_cgeqr2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: cgeqr2
    end function rocsolver_cgeqr2

    !---------------------------------------------
    ! rocsolver_zgeqr2
    !---------------------------------------------
    function rocsolver_zgeqr2(handle, m, n, A, lda, ipiv) &
       result(zgeqr2) &
       bind(C, name="rocsolver_zgeqr2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: zgeqr2
    end function rocsolver_zgeqr2

    !---------------------------------------------
    ! rocsolver_sgeqr2_64
    !---------------------------------------------
    function rocsolver_sgeqr2_64(handle, m, n, A, lda, ipiv) &
       result(sgeqr2_64) &
       bind(C, name="rocsolver_sgeqr2_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: sgeqr2_64
    end function rocsolver_sgeqr2_64

    !---------------------------------------------
    ! rocsolver_dgeqr2_64
    !---------------------------------------------
    function rocsolver_dgeqr2_64(handle, m, n, A, lda, ipiv) &
       result(dgeqr2_64) &
       bind(C, name="rocsolver_dgeqr2_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: dgeqr2_64
    end function rocsolver_dgeqr2_64

    !---------------------------------------------
    ! rocsolver_cgeqr2_64
    !---------------------------------------------
    function rocsolver_cgeqr2_64(handle, m, n, A, lda, ipiv) &
       result(cgeqr2_64) &
       bind(C, name="rocsolver_cgeqr2_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: cgeqr2_64
    end function rocsolver_cgeqr2_64

    !---------------------------------------------
    ! rocsolver_zgeqr2_64
    !---------------------------------------------
    function rocsolver_zgeqr2_64(handle, m, n, A, lda, ipiv) &
       result(zgeqr2_64) &
       bind(C, name="rocsolver_zgeqr2_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: zgeqr2_64
    end function rocsolver_zgeqr2_64

    !---------------------------------------------
    ! rocsolver_sgeqr2_batched
    !---------------------------------------------
    function rocsolver_sgeqr2_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(sgeqr2_batched) &
       bind(C, name="rocsolver_sgeqr2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgeqr2_batched
    end function rocsolver_sgeqr2_batched

    !---------------------------------------------
    ! rocsolver_dgeqr2_batched
    !---------------------------------------------
    function rocsolver_dgeqr2_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(dgeqr2_batched) &
       bind(C, name="rocsolver_dgeqr2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgeqr2_batched
    end function rocsolver_dgeqr2_batched

    !---------------------------------------------
    ! rocsolver_cgeqr2_batched
    !---------------------------------------------
    function rocsolver_cgeqr2_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(cgeqr2_batched) &
       bind(C, name="rocsolver_cgeqr2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgeqr2_batched
    end function rocsolver_cgeqr2_batched

    !---------------------------------------------
    ! rocsolver_zgeqr2_batched
    !---------------------------------------------
    function rocsolver_zgeqr2_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(zgeqr2_batched) &
       bind(C, name="rocsolver_zgeqr2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgeqr2_batched
    end function rocsolver_zgeqr2_batched

    !---------------------------------------------
    ! rocsolver_sgeqr2_batched_64
    !---------------------------------------------
    function rocsolver_sgeqr2_batched_64(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(sgeqr2_batched_64) &
       bind(C, name="rocsolver_sgeqr2_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgeqr2_batched_64
    end function rocsolver_sgeqr2_batched_64

    !---------------------------------------------
    ! rocsolver_dgeqr2_batched_64
    !---------------------------------------------
    function rocsolver_dgeqr2_batched_64(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(dgeqr2_batched_64) &
       bind(C, name="rocsolver_dgeqr2_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgeqr2_batched_64
    end function rocsolver_dgeqr2_batched_64

    !---------------------------------------------
    ! rocsolver_cgeqr2_batched_64
    !---------------------------------------------
    function rocsolver_cgeqr2_batched_64(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(cgeqr2_batched_64) &
       bind(C, name="rocsolver_cgeqr2_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgeqr2_batched_64
    end function rocsolver_cgeqr2_batched_64

    !---------------------------------------------
    ! rocsolver_zgeqr2_batched_64
    !---------------------------------------------
    function rocsolver_zgeqr2_batched_64(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(zgeqr2_batched_64) &
       bind(C, name="rocsolver_zgeqr2_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgeqr2_batched_64
    end function rocsolver_zgeqr2_batched_64

    !---------------------------------------------
    ! rocsolver_sgeqr2_strided_batched
    !---------------------------------------------
    function rocsolver_sgeqr2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(sgeqr2_strided_batched) &
       bind(C, name="rocsolver_sgeqr2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgeqr2_strided_batched
    end function rocsolver_sgeqr2_strided_batched

    !---------------------------------------------
    ! rocsolver_dgeqr2_strided_batched
    !---------------------------------------------
    function rocsolver_dgeqr2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(dgeqr2_strided_batched) &
       bind(C, name="rocsolver_dgeqr2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgeqr2_strided_batched
    end function rocsolver_dgeqr2_strided_batched

    !---------------------------------------------
    ! rocsolver_cgeqr2_strided_batched
    !---------------------------------------------
    function rocsolver_cgeqr2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(cgeqr2_strided_batched) &
       bind(C, name="rocsolver_cgeqr2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgeqr2_strided_batched
    end function rocsolver_cgeqr2_strided_batched

    !---------------------------------------------
    ! rocsolver_zgeqr2_strided_batched
    !---------------------------------------------
    function rocsolver_zgeqr2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(zgeqr2_strided_batched) &
       bind(C, name="rocsolver_zgeqr2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgeqr2_strided_batched
    end function rocsolver_zgeqr2_strided_batched

    !---------------------------------------------
    ! rocsolver_sgeqr2_strided_batched_64
    !---------------------------------------------
    function rocsolver_sgeqr2_strided_batched_64(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                                 batch_count) &
       result(sgeqr2_strided_batched_64) &
       bind(C, name="rocsolver_sgeqr2_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgeqr2_strided_batched_64
    end function rocsolver_sgeqr2_strided_batched_64

    !---------------------------------------------
    ! rocsolver_dgeqr2_strided_batched_64
    !---------------------------------------------
    function rocsolver_dgeqr2_strided_batched_64(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                                 batch_count) &
       result(dgeqr2_strided_batched_64) &
       bind(C, name="rocsolver_dgeqr2_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgeqr2_strided_batched_64
    end function rocsolver_dgeqr2_strided_batched_64

    !---------------------------------------------
    ! rocsolver_cgeqr2_strided_batched_64
    !---------------------------------------------
    function rocsolver_cgeqr2_strided_batched_64(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                                 batch_count) &
       result(cgeqr2_strided_batched_64) &
       bind(C, name="rocsolver_cgeqr2_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgeqr2_strided_batched_64
    end function rocsolver_cgeqr2_strided_batched_64

    !---------------------------------------------
    ! rocsolver_zgeqr2_strided_batched_64
    !---------------------------------------------
    function rocsolver_zgeqr2_strided_batched_64(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                                 batch_count) &
       result(zgeqr2_strided_batched_64) &
       bind(C, name="rocsolver_zgeqr2_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgeqr2_strided_batched_64
    end function rocsolver_zgeqr2_strided_batched_64

    !---------------------------------------------
    ! rocsolver_sgerq2
    !---------------------------------------------
    function rocsolver_sgerq2(handle, m, n, A, lda, ipiv) &
       result(sgerq2) &
       bind(C, name="rocsolver_sgerq2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: sgerq2
    end function rocsolver_sgerq2

    !---------------------------------------------
    ! rocsolver_dgerq2
    !---------------------------------------------
    function rocsolver_dgerq2(handle, m, n, A, lda, ipiv) &
       result(dgerq2) &
       bind(C, name="rocsolver_dgerq2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: dgerq2
    end function rocsolver_dgerq2

    !---------------------------------------------
    ! rocsolver_cgerq2
    !---------------------------------------------
    function rocsolver_cgerq2(handle, m, n, A, lda, ipiv) &
       result(cgerq2) &
       bind(C, name="rocsolver_cgerq2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: cgerq2
    end function rocsolver_cgerq2

    !---------------------------------------------
    ! rocsolver_zgerq2
    !---------------------------------------------
    function rocsolver_zgerq2(handle, m, n, A, lda, ipiv) &
       result(zgerq2) &
       bind(C, name="rocsolver_zgerq2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: zgerq2
    end function rocsolver_zgerq2

    !---------------------------------------------
    ! rocsolver_sgerq2_batched
    !---------------------------------------------
    function rocsolver_sgerq2_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(sgerq2_batched) &
       bind(C, name="rocsolver_sgerq2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgerq2_batched
    end function rocsolver_sgerq2_batched

    !---------------------------------------------
    ! rocsolver_dgerq2_batched
    !---------------------------------------------
    function rocsolver_dgerq2_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(dgerq2_batched) &
       bind(C, name="rocsolver_dgerq2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgerq2_batched
    end function rocsolver_dgerq2_batched

    !---------------------------------------------
    ! rocsolver_cgerq2_batched
    !---------------------------------------------
    function rocsolver_cgerq2_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(cgerq2_batched) &
       bind(C, name="rocsolver_cgerq2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgerq2_batched
    end function rocsolver_cgerq2_batched

    !---------------------------------------------
    ! rocsolver_zgerq2_batched
    !---------------------------------------------
    function rocsolver_zgerq2_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(zgerq2_batched) &
       bind(C, name="rocsolver_zgerq2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgerq2_batched
    end function rocsolver_zgerq2_batched

    !---------------------------------------------
    ! rocsolver_sgerq2_strided_batched
    !---------------------------------------------
    function rocsolver_sgerq2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(sgerq2_strided_batched) &
       bind(C, name="rocsolver_sgerq2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgerq2_strided_batched
    end function rocsolver_sgerq2_strided_batched

    !---------------------------------------------
    ! rocsolver_dgerq2_strided_batched
    !---------------------------------------------
    function rocsolver_dgerq2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(dgerq2_strided_batched) &
       bind(C, name="rocsolver_dgerq2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgerq2_strided_batched
    end function rocsolver_dgerq2_strided_batched

    !---------------------------------------------
    ! rocsolver_cgerq2_strided_batched
    !---------------------------------------------
    function rocsolver_cgerq2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(cgerq2_strided_batched) &
       bind(C, name="rocsolver_cgerq2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgerq2_strided_batched
    end function rocsolver_cgerq2_strided_batched

    !---------------------------------------------
    ! rocsolver_zgerq2_strided_batched
    !---------------------------------------------
    function rocsolver_zgerq2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(zgerq2_strided_batched) &
       bind(C, name="rocsolver_zgerq2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgerq2_strided_batched
    end function rocsolver_zgerq2_strided_batched

    !---------------------------------------------
    ! rocsolver_sgeql2
    !---------------------------------------------
    function rocsolver_sgeql2(handle, m, n, A, lda, ipiv) &
       result(sgeql2) &
       bind(C, name="rocsolver_sgeql2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: sgeql2
    end function rocsolver_sgeql2

    !---------------------------------------------
    ! rocsolver_dgeql2
    !---------------------------------------------
    function rocsolver_dgeql2(handle, m, n, A, lda, ipiv) &
       result(dgeql2) &
       bind(C, name="rocsolver_dgeql2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: dgeql2
    end function rocsolver_dgeql2

    !---------------------------------------------
    ! rocsolver_cgeql2
    !---------------------------------------------
    function rocsolver_cgeql2(handle, m, n, A, lda, ipiv) &
       result(cgeql2) &
       bind(C, name="rocsolver_cgeql2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: cgeql2
    end function rocsolver_cgeql2

    !---------------------------------------------
    ! rocsolver_zgeql2
    !---------------------------------------------
    function rocsolver_zgeql2(handle, m, n, A, lda, ipiv) &
       result(zgeql2) &
       bind(C, name="rocsolver_zgeql2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: zgeql2
    end function rocsolver_zgeql2

    !---------------------------------------------
    ! rocsolver_sgeql2_batched
    !---------------------------------------------
    function rocsolver_sgeql2_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(sgeql2_batched) &
       bind(C, name="rocsolver_sgeql2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgeql2_batched
    end function rocsolver_sgeql2_batched

    !---------------------------------------------
    ! rocsolver_dgeql2_batched
    !---------------------------------------------
    function rocsolver_dgeql2_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(dgeql2_batched) &
       bind(C, name="rocsolver_dgeql2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgeql2_batched
    end function rocsolver_dgeql2_batched

    !---------------------------------------------
    ! rocsolver_cgeql2_batched
    !---------------------------------------------
    function rocsolver_cgeql2_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(cgeql2_batched) &
       bind(C, name="rocsolver_cgeql2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgeql2_batched
    end function rocsolver_cgeql2_batched

    !---------------------------------------------
    ! rocsolver_zgeql2_batched
    !---------------------------------------------
    function rocsolver_zgeql2_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(zgeql2_batched) &
       bind(C, name="rocsolver_zgeql2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgeql2_batched
    end function rocsolver_zgeql2_batched

    !---------------------------------------------
    ! rocsolver_sgeql2_strided_batched
    !---------------------------------------------
    function rocsolver_sgeql2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(sgeql2_strided_batched) &
       bind(C, name="rocsolver_sgeql2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgeql2_strided_batched
    end function rocsolver_sgeql2_strided_batched

    !---------------------------------------------
    ! rocsolver_dgeql2_strided_batched
    !---------------------------------------------
    function rocsolver_dgeql2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(dgeql2_strided_batched) &
       bind(C, name="rocsolver_dgeql2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgeql2_strided_batched
    end function rocsolver_dgeql2_strided_batched

    !---------------------------------------------
    ! rocsolver_cgeql2_strided_batched
    !---------------------------------------------
    function rocsolver_cgeql2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(cgeql2_strided_batched) &
       bind(C, name="rocsolver_cgeql2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgeql2_strided_batched
    end function rocsolver_cgeql2_strided_batched

    !---------------------------------------------
    ! rocsolver_zgeql2_strided_batched
    !---------------------------------------------
    function rocsolver_zgeql2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(zgeql2_strided_batched) &
       bind(C, name="rocsolver_zgeql2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgeql2_strided_batched
    end function rocsolver_zgeql2_strided_batched

    !---------------------------------------------
    ! rocsolver_sgelq2
    !---------------------------------------------
    function rocsolver_sgelq2(handle, m, n, A, lda, ipiv) &
       result(sgelq2) &
       bind(C, name="rocsolver_sgelq2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: sgelq2
    end function rocsolver_sgelq2

    !---------------------------------------------
    ! rocsolver_dgelq2
    !---------------------------------------------
    function rocsolver_dgelq2(handle, m, n, A, lda, ipiv) &
       result(dgelq2) &
       bind(C, name="rocsolver_dgelq2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: dgelq2
    end function rocsolver_dgelq2

    !---------------------------------------------
    ! rocsolver_cgelq2
    !---------------------------------------------
    function rocsolver_cgelq2(handle, m, n, A, lda, ipiv) &
       result(cgelq2) &
       bind(C, name="rocsolver_cgelq2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: cgelq2
    end function rocsolver_cgelq2

    !---------------------------------------------
    ! rocsolver_zgelq2
    !---------------------------------------------
    function rocsolver_zgelq2(handle, m, n, A, lda, ipiv) &
       result(zgelq2) &
       bind(C, name="rocsolver_zgelq2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: zgelq2
    end function rocsolver_zgelq2

    !---------------------------------------------
    ! rocsolver_sgelq2_batched
    !---------------------------------------------
    function rocsolver_sgelq2_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(sgelq2_batched) &
       bind(C, name="rocsolver_sgelq2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgelq2_batched
    end function rocsolver_sgelq2_batched

    !---------------------------------------------
    ! rocsolver_dgelq2_batched
    !---------------------------------------------
    function rocsolver_dgelq2_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(dgelq2_batched) &
       bind(C, name="rocsolver_dgelq2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgelq2_batched
    end function rocsolver_dgelq2_batched

    !---------------------------------------------
    ! rocsolver_cgelq2_batched
    !---------------------------------------------
    function rocsolver_cgelq2_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(cgelq2_batched) &
       bind(C, name="rocsolver_cgelq2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgelq2_batched
    end function rocsolver_cgelq2_batched

    !---------------------------------------------
    ! rocsolver_zgelq2_batched
    !---------------------------------------------
    function rocsolver_zgelq2_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(zgelq2_batched) &
       bind(C, name="rocsolver_zgelq2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgelq2_batched
    end function rocsolver_zgelq2_batched

    !---------------------------------------------
    ! rocsolver_sgelq2_strided_batched
    !---------------------------------------------
    function rocsolver_sgelq2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(sgelq2_strided_batched) &
       bind(C, name="rocsolver_sgelq2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgelq2_strided_batched
    end function rocsolver_sgelq2_strided_batched

    !---------------------------------------------
    ! rocsolver_dgelq2_strided_batched
    !---------------------------------------------
    function rocsolver_dgelq2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(dgelq2_strided_batched) &
       bind(C, name="rocsolver_dgelq2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgelq2_strided_batched
    end function rocsolver_dgelq2_strided_batched

    !---------------------------------------------
    ! rocsolver_cgelq2_strided_batched
    !---------------------------------------------
    function rocsolver_cgelq2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(cgelq2_strided_batched) &
       bind(C, name="rocsolver_cgelq2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgelq2_strided_batched
    end function rocsolver_cgelq2_strided_batched

    !---------------------------------------------
    ! rocsolver_zgelq2_strided_batched
    !---------------------------------------------
    function rocsolver_zgelq2_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(zgelq2_strided_batched) &
       bind(C, name="rocsolver_zgelq2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgelq2_strided_batched
    end function rocsolver_zgelq2_strided_batched

    !---------------------------------------------
    ! rocsolver_sgeqrf
    !---------------------------------------------
    function rocsolver_sgeqrf(handle, m, n, A, lda, ipiv) &
       result(sgeqrf) &
       bind(C, name="rocsolver_sgeqrf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: sgeqrf
    end function rocsolver_sgeqrf

    !---------------------------------------------
    ! rocsolver_dgeqrf
    !---------------------------------------------
    function rocsolver_dgeqrf(handle, m, n, A, lda, ipiv) &
       result(dgeqrf) &
       bind(C, name="rocsolver_dgeqrf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: dgeqrf
    end function rocsolver_dgeqrf

    !---------------------------------------------
    ! rocsolver_cgeqrf
    !---------------------------------------------
    function rocsolver_cgeqrf(handle, m, n, A, lda, ipiv) &
       result(cgeqrf) &
       bind(C, name="rocsolver_cgeqrf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: cgeqrf
    end function rocsolver_cgeqrf

    !---------------------------------------------
    ! rocsolver_zgeqrf
    !---------------------------------------------
    function rocsolver_zgeqrf(handle, m, n, A, lda, ipiv) &
       result(zgeqrf) &
       bind(C, name="rocsolver_zgeqrf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: zgeqrf
    end function rocsolver_zgeqrf

    !---------------------------------------------
    ! rocsolver_sgeqrf_64
    !---------------------------------------------
    function rocsolver_sgeqrf_64(handle, m, n, A, lda, ipiv) &
       result(sgeqrf_64) &
       bind(C, name="rocsolver_sgeqrf_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: sgeqrf_64
    end function rocsolver_sgeqrf_64

    !---------------------------------------------
    ! rocsolver_dgeqrf_64
    !---------------------------------------------
    function rocsolver_dgeqrf_64(handle, m, n, A, lda, ipiv) &
       result(dgeqrf_64) &
       bind(C, name="rocsolver_dgeqrf_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: dgeqrf_64
    end function rocsolver_dgeqrf_64

    !---------------------------------------------
    ! rocsolver_cgeqrf_64
    !---------------------------------------------
    function rocsolver_cgeqrf_64(handle, m, n, A, lda, ipiv) &
       result(cgeqrf_64) &
       bind(C, name="rocsolver_cgeqrf_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: cgeqrf_64
    end function rocsolver_cgeqrf_64

    !---------------------------------------------
    ! rocsolver_zgeqrf_64
    !---------------------------------------------
    function rocsolver_zgeqrf_64(handle, m, n, A, lda, ipiv) &
       result(zgeqrf_64) &
       bind(C, name="rocsolver_zgeqrf_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: zgeqrf_64
    end function rocsolver_zgeqrf_64

    !---------------------------------------------
    ! rocsolver_sgeqrf_batched
    !---------------------------------------------
    function rocsolver_sgeqrf_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(sgeqrf_batched) &
       bind(C, name="rocsolver_sgeqrf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgeqrf_batched
    end function rocsolver_sgeqrf_batched

    !---------------------------------------------
    ! rocsolver_dgeqrf_batched
    !---------------------------------------------
    function rocsolver_dgeqrf_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(dgeqrf_batched) &
       bind(C, name="rocsolver_dgeqrf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgeqrf_batched
    end function rocsolver_dgeqrf_batched

    !---------------------------------------------
    ! rocsolver_cgeqrf_batched
    !---------------------------------------------
    function rocsolver_cgeqrf_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(cgeqrf_batched) &
       bind(C, name="rocsolver_cgeqrf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgeqrf_batched
    end function rocsolver_cgeqrf_batched

    !---------------------------------------------
    ! rocsolver_zgeqrf_batched
    !---------------------------------------------
    function rocsolver_zgeqrf_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(zgeqrf_batched) &
       bind(C, name="rocsolver_zgeqrf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgeqrf_batched
    end function rocsolver_zgeqrf_batched

    !---------------------------------------------
    ! rocsolver_sgeqrf_batched_64
    !---------------------------------------------
    function rocsolver_sgeqrf_batched_64(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(sgeqrf_batched_64) &
       bind(C, name="rocsolver_sgeqrf_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgeqrf_batched_64
    end function rocsolver_sgeqrf_batched_64

    !---------------------------------------------
    ! rocsolver_dgeqrf_batched_64
    !---------------------------------------------
    function rocsolver_dgeqrf_batched_64(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(dgeqrf_batched_64) &
       bind(C, name="rocsolver_dgeqrf_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgeqrf_batched_64
    end function rocsolver_dgeqrf_batched_64

    !---------------------------------------------
    ! rocsolver_cgeqrf_batched_64
    !---------------------------------------------
    function rocsolver_cgeqrf_batched_64(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(cgeqrf_batched_64) &
       bind(C, name="rocsolver_cgeqrf_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgeqrf_batched_64
    end function rocsolver_cgeqrf_batched_64

    !---------------------------------------------
    ! rocsolver_zgeqrf_batched_64
    !---------------------------------------------
    function rocsolver_zgeqrf_batched_64(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(zgeqrf_batched_64) &
       bind(C, name="rocsolver_zgeqrf_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgeqrf_batched_64
    end function rocsolver_zgeqrf_batched_64

    !---------------------------------------------
    ! rocsolver_sgeqrf_strided_batched
    !---------------------------------------------
    function rocsolver_sgeqrf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(sgeqrf_strided_batched) &
       bind(C, name="rocsolver_sgeqrf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgeqrf_strided_batched
    end function rocsolver_sgeqrf_strided_batched

    !---------------------------------------------
    ! rocsolver_dgeqrf_strided_batched
    !---------------------------------------------
    function rocsolver_dgeqrf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(dgeqrf_strided_batched) &
       bind(C, name="rocsolver_dgeqrf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgeqrf_strided_batched
    end function rocsolver_dgeqrf_strided_batched

    !---------------------------------------------
    ! rocsolver_cgeqrf_strided_batched
    !---------------------------------------------
    function rocsolver_cgeqrf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(cgeqrf_strided_batched) &
       bind(C, name="rocsolver_cgeqrf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgeqrf_strided_batched
    end function rocsolver_cgeqrf_strided_batched

    !---------------------------------------------
    ! rocsolver_zgeqrf_strided_batched
    !---------------------------------------------
    function rocsolver_zgeqrf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(zgeqrf_strided_batched) &
       bind(C, name="rocsolver_zgeqrf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgeqrf_strided_batched
    end function rocsolver_zgeqrf_strided_batched

    !---------------------------------------------
    ! rocsolver_sgeqrf_strided_batched_64
    !---------------------------------------------
    function rocsolver_sgeqrf_strided_batched_64(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                                 batch_count) &
       result(sgeqrf_strided_batched_64) &
       bind(C, name="rocsolver_sgeqrf_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgeqrf_strided_batched_64
    end function rocsolver_sgeqrf_strided_batched_64

    !---------------------------------------------
    ! rocsolver_dgeqrf_strided_batched_64
    !---------------------------------------------
    function rocsolver_dgeqrf_strided_batched_64(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                                 batch_count) &
       result(dgeqrf_strided_batched_64) &
       bind(C, name="rocsolver_dgeqrf_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgeqrf_strided_batched_64
    end function rocsolver_dgeqrf_strided_batched_64

    !---------------------------------------------
    ! rocsolver_cgeqrf_strided_batched_64
    !---------------------------------------------
    function rocsolver_cgeqrf_strided_batched_64(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                                 batch_count) &
       result(cgeqrf_strided_batched_64) &
       bind(C, name="rocsolver_cgeqrf_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgeqrf_strided_batched_64
    end function rocsolver_cgeqrf_strided_batched_64

    !---------------------------------------------
    ! rocsolver_zgeqrf_strided_batched_64
    !---------------------------------------------
    function rocsolver_zgeqrf_strided_batched_64(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                                 batch_count) &
       result(zgeqrf_strided_batched_64) &
       bind(C, name="rocsolver_zgeqrf_strided_batched_64")
       import :: c_ptr, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgeqrf_strided_batched_64
    end function rocsolver_zgeqrf_strided_batched_64

    !---------------------------------------------
    ! rocsolver_sgerqf
    !---------------------------------------------
    function rocsolver_sgerqf(handle, m, n, A, lda, ipiv) &
       result(sgerqf) &
       bind(C, name="rocsolver_sgerqf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: sgerqf
    end function rocsolver_sgerqf

    !---------------------------------------------
    ! rocsolver_dgerqf
    !---------------------------------------------
    function rocsolver_dgerqf(handle, m, n, A, lda, ipiv) &
       result(dgerqf) &
       bind(C, name="rocsolver_dgerqf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: dgerqf
    end function rocsolver_dgerqf

    !---------------------------------------------
    ! rocsolver_cgerqf
    !---------------------------------------------
    function rocsolver_cgerqf(handle, m, n, A, lda, ipiv) &
       result(cgerqf) &
       bind(C, name="rocsolver_cgerqf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: cgerqf
    end function rocsolver_cgerqf

    !---------------------------------------------
    ! rocsolver_zgerqf
    !---------------------------------------------
    function rocsolver_zgerqf(handle, m, n, A, lda, ipiv) &
       result(zgerqf) &
       bind(C, name="rocsolver_zgerqf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: zgerqf
    end function rocsolver_zgerqf

    !---------------------------------------------
    ! rocsolver_sgerqf_batched
    !---------------------------------------------
    function rocsolver_sgerqf_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(sgerqf_batched) &
       bind(C, name="rocsolver_sgerqf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgerqf_batched
    end function rocsolver_sgerqf_batched

    !---------------------------------------------
    ! rocsolver_dgerqf_batched
    !---------------------------------------------
    function rocsolver_dgerqf_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(dgerqf_batched) &
       bind(C, name="rocsolver_dgerqf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgerqf_batched
    end function rocsolver_dgerqf_batched

    !---------------------------------------------
    ! rocsolver_cgerqf_batched
    !---------------------------------------------
    function rocsolver_cgerqf_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(cgerqf_batched) &
       bind(C, name="rocsolver_cgerqf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgerqf_batched
    end function rocsolver_cgerqf_batched

    !---------------------------------------------
    ! rocsolver_zgerqf_batched
    !---------------------------------------------
    function rocsolver_zgerqf_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(zgerqf_batched) &
       bind(C, name="rocsolver_zgerqf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgerqf_batched
    end function rocsolver_zgerqf_batched

    !---------------------------------------------
    ! rocsolver_sgerqf_strided_batched
    !---------------------------------------------
    function rocsolver_sgerqf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(sgerqf_strided_batched) &
       bind(C, name="rocsolver_sgerqf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgerqf_strided_batched
    end function rocsolver_sgerqf_strided_batched

    !---------------------------------------------
    ! rocsolver_dgerqf_strided_batched
    !---------------------------------------------
    function rocsolver_dgerqf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(dgerqf_strided_batched) &
       bind(C, name="rocsolver_dgerqf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgerqf_strided_batched
    end function rocsolver_dgerqf_strided_batched

    !---------------------------------------------
    ! rocsolver_cgerqf_strided_batched
    !---------------------------------------------
    function rocsolver_cgerqf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(cgerqf_strided_batched) &
       bind(C, name="rocsolver_cgerqf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgerqf_strided_batched
    end function rocsolver_cgerqf_strided_batched

    !---------------------------------------------
    ! rocsolver_zgerqf_strided_batched
    !---------------------------------------------
    function rocsolver_zgerqf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(zgerqf_strided_batched) &
       bind(C, name="rocsolver_zgerqf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgerqf_strided_batched
    end function rocsolver_zgerqf_strided_batched

    !---------------------------------------------
    ! rocsolver_sgeqlf
    !---------------------------------------------
    function rocsolver_sgeqlf(handle, m, n, A, lda, ipiv) &
       result(sgeqlf) &
       bind(C, name="rocsolver_sgeqlf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: sgeqlf
    end function rocsolver_sgeqlf

    !---------------------------------------------
    ! rocsolver_dgeqlf
    !---------------------------------------------
    function rocsolver_dgeqlf(handle, m, n, A, lda, ipiv) &
       result(dgeqlf) &
       bind(C, name="rocsolver_dgeqlf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: dgeqlf
    end function rocsolver_dgeqlf

    !---------------------------------------------
    ! rocsolver_cgeqlf
    !---------------------------------------------
    function rocsolver_cgeqlf(handle, m, n, A, lda, ipiv) &
       result(cgeqlf) &
       bind(C, name="rocsolver_cgeqlf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: cgeqlf
    end function rocsolver_cgeqlf

    !---------------------------------------------
    ! rocsolver_zgeqlf
    !---------------------------------------------
    function rocsolver_zgeqlf(handle, m, n, A, lda, ipiv) &
       result(zgeqlf) &
       bind(C, name="rocsolver_zgeqlf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: zgeqlf
    end function rocsolver_zgeqlf

    !---------------------------------------------
    ! rocsolver_sgeqlf_batched
    !---------------------------------------------
    function rocsolver_sgeqlf_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(sgeqlf_batched) &
       bind(C, name="rocsolver_sgeqlf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgeqlf_batched
    end function rocsolver_sgeqlf_batched

    !---------------------------------------------
    ! rocsolver_dgeqlf_batched
    !---------------------------------------------
    function rocsolver_dgeqlf_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(dgeqlf_batched) &
       bind(C, name="rocsolver_dgeqlf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgeqlf_batched
    end function rocsolver_dgeqlf_batched

    !---------------------------------------------
    ! rocsolver_cgeqlf_batched
    !---------------------------------------------
    function rocsolver_cgeqlf_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(cgeqlf_batched) &
       bind(C, name="rocsolver_cgeqlf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgeqlf_batched
    end function rocsolver_cgeqlf_batched

    !---------------------------------------------
    ! rocsolver_zgeqlf_batched
    !---------------------------------------------
    function rocsolver_zgeqlf_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(zgeqlf_batched) &
       bind(C, name="rocsolver_zgeqlf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgeqlf_batched
    end function rocsolver_zgeqlf_batched

    !---------------------------------------------
    ! rocsolver_sgeqlf_strided_batched
    !---------------------------------------------
    function rocsolver_sgeqlf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(sgeqlf_strided_batched) &
       bind(C, name="rocsolver_sgeqlf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgeqlf_strided_batched
    end function rocsolver_sgeqlf_strided_batched

    !---------------------------------------------
    ! rocsolver_dgeqlf_strided_batched
    !---------------------------------------------
    function rocsolver_dgeqlf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(dgeqlf_strided_batched) &
       bind(C, name="rocsolver_dgeqlf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgeqlf_strided_batched
    end function rocsolver_dgeqlf_strided_batched

    !---------------------------------------------
    ! rocsolver_cgeqlf_strided_batched
    !---------------------------------------------
    function rocsolver_cgeqlf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(cgeqlf_strided_batched) &
       bind(C, name="rocsolver_cgeqlf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgeqlf_strided_batched
    end function rocsolver_cgeqlf_strided_batched

    !---------------------------------------------
    ! rocsolver_zgeqlf_strided_batched
    !---------------------------------------------
    function rocsolver_zgeqlf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(zgeqlf_strided_batched) &
       bind(C, name="rocsolver_zgeqlf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgeqlf_strided_batched
    end function rocsolver_zgeqlf_strided_batched

    !---------------------------------------------
    ! rocsolver_sgelqf
    !---------------------------------------------
    function rocsolver_sgelqf(handle, m, n, A, lda, ipiv) &
       result(sgelqf) &
       bind(C, name="rocsolver_sgelqf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: sgelqf
    end function rocsolver_sgelqf

    !---------------------------------------------
    ! rocsolver_dgelqf
    !---------------------------------------------
    function rocsolver_dgelqf(handle, m, n, A, lda, ipiv) &
       result(dgelqf) &
       bind(C, name="rocsolver_dgelqf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: dgelqf
    end function rocsolver_dgelqf

    !---------------------------------------------
    ! rocsolver_cgelqf
    !---------------------------------------------
    function rocsolver_cgelqf(handle, m, n, A, lda, ipiv) &
       result(cgelqf) &
       bind(C, name="rocsolver_cgelqf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: cgelqf
    end function rocsolver_cgelqf

    !---------------------------------------------
    ! rocsolver_zgelqf
    !---------------------------------------------
    function rocsolver_zgelqf(handle, m, n, A, lda, ipiv) &
       result(zgelqf) &
       bind(C, name="rocsolver_zgelqf")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(kind(rocblas_status_success)) :: zgelqf
    end function rocsolver_zgelqf

    !---------------------------------------------
    ! rocsolver_sgelqf_batched
    !---------------------------------------------
    function rocsolver_sgelqf_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(sgelqf_batched) &
       bind(C, name="rocsolver_sgelqf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgelqf_batched
    end function rocsolver_sgelqf_batched

    !---------------------------------------------
    ! rocsolver_dgelqf_batched
    !---------------------------------------------
    function rocsolver_dgelqf_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(dgelqf_batched) &
       bind(C, name="rocsolver_dgelqf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgelqf_batched
    end function rocsolver_dgelqf_batched

    !---------------------------------------------
    ! rocsolver_cgelqf_batched
    !---------------------------------------------
    function rocsolver_cgelqf_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(cgelqf_batched) &
       bind(C, name="rocsolver_cgelqf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgelqf_batched
    end function rocsolver_cgelqf_batched

    !---------------------------------------------
    ! rocsolver_zgelqf_batched
    !---------------------------------------------
    function rocsolver_zgelqf_batched(handle, m, n, A, lda, ipiv, strideP, batch_count) &
       result(zgelqf_batched) &
       bind(C, name="rocsolver_zgelqf_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgelqf_batched
    end function rocsolver_zgelqf_batched

    !---------------------------------------------
    ! rocsolver_sgelqf_strided_batched
    !---------------------------------------------
    function rocsolver_sgelqf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(sgelqf_strided_batched) &
       bind(C, name="rocsolver_sgelqf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgelqf_strided_batched
    end function rocsolver_sgelqf_strided_batched

    !---------------------------------------------
    ! rocsolver_dgelqf_strided_batched
    !---------------------------------------------
    function rocsolver_dgelqf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(dgelqf_strided_batched) &
       bind(C, name="rocsolver_dgelqf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgelqf_strided_batched
    end function rocsolver_dgelqf_strided_batched

    !---------------------------------------------
    ! rocsolver_cgelqf_strided_batched
    !---------------------------------------------
    function rocsolver_cgelqf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(cgelqf_strided_batched) &
       bind(C, name="rocsolver_cgelqf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgelqf_strided_batched
    end function rocsolver_cgelqf_strided_batched

    !---------------------------------------------
    ! rocsolver_zgelqf_strided_batched
    !---------------------------------------------
    function rocsolver_zgelqf_strided_batched(handle, m, n, A, lda, strideA, ipiv, strideP, &
                                              batch_count) &
       result(zgelqf_strided_batched) &
       bind(C, name="rocsolver_zgelqf_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgelqf_strided_batched
    end function rocsolver_zgelqf_strided_batched

    !---------------------------------------------
    ! rocsolver_sgebd2
    !---------------------------------------------
    function rocsolver_sgebd2(handle, m, n, A, lda, D, E, tauq, taup) &
       result(sgebd2) &
       bind(C, name="rocsolver_sgebd2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       integer(kind(rocblas_status_success)) :: sgebd2
    end function rocsolver_sgebd2

    !---------------------------------------------
    ! rocsolver_dgebd2
    !---------------------------------------------
    function rocsolver_dgebd2(handle, m, n, A, lda, D, E, tauq, taup) &
       result(dgebd2) &
       bind(C, name="rocsolver_dgebd2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       integer(kind(rocblas_status_success)) :: dgebd2
    end function rocsolver_dgebd2

    !---------------------------------------------
    ! rocsolver_cgebd2
    !---------------------------------------------
    function rocsolver_cgebd2(handle, m, n, A, lda, D, E, tauq, taup) &
       result(cgebd2) &
       bind(C, name="rocsolver_cgebd2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       integer(kind(rocblas_status_success)) :: cgebd2
    end function rocsolver_cgebd2

    !---------------------------------------------
    ! rocsolver_zgebd2
    !---------------------------------------------
    function rocsolver_zgebd2(handle, m, n, A, lda, D, E, tauq, taup) &
       result(zgebd2) &
       bind(C, name="rocsolver_zgebd2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       integer(kind(rocblas_status_success)) :: zgebd2
    end function rocsolver_zgebd2

    !---------------------------------------------
    ! rocsolver_sgebd2_batched
    !---------------------------------------------
    function rocsolver_sgebd2_batched(handle, m, n, A, lda, D, strideD, E, strideE, tauq, strideQ, &
                                      taup, strideP, batch_count) &
       result(sgebd2_batched) &
       bind(C, name="rocsolver_sgebd2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tauq
       integer(c_int64_t), value :: strideQ
       type(c_ptr), value :: taup
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgebd2_batched
    end function rocsolver_sgebd2_batched

    !---------------------------------------------
    ! rocsolver_dgebd2_batched
    !---------------------------------------------
    function rocsolver_dgebd2_batched(handle, m, n, A, lda, D, strideD, E, strideE, tauq, strideQ, &
                                      taup, strideP, batch_count) &
       result(dgebd2_batched) &
       bind(C, name="rocsolver_dgebd2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tauq
       integer(c_int64_t), value :: strideQ
       type(c_ptr), value :: taup
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgebd2_batched
    end function rocsolver_dgebd2_batched

    !---------------------------------------------
    ! rocsolver_cgebd2_batched
    !---------------------------------------------
    function rocsolver_cgebd2_batched(handle, m, n, A, lda, D, strideD, E, strideE, tauq, strideQ, &
                                      taup, strideP, batch_count) &
       result(cgebd2_batched) &
       bind(C, name="rocsolver_cgebd2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tauq
       integer(c_int64_t), value :: strideQ
       type(c_ptr), value :: taup
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgebd2_batched
    end function rocsolver_cgebd2_batched

    !---------------------------------------------
    ! rocsolver_zgebd2_batched
    !---------------------------------------------
    function rocsolver_zgebd2_batched(handle, m, n, A, lda, D, strideD, E, strideE, tauq, strideQ, &
                                      taup, strideP, batch_count) &
       result(zgebd2_batched) &
       bind(C, name="rocsolver_zgebd2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tauq
       integer(c_int64_t), value :: strideQ
       type(c_ptr), value :: taup
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgebd2_batched
    end function rocsolver_zgebd2_batched

    !---------------------------------------------
    ! rocsolver_sgebd2_strided_batched
    !---------------------------------------------
    function rocsolver_sgebd2_strided_batched(handle, m, n, A, lda, strideA, D, strideD, E, &
                                              strideE, tauq, strideQ, taup, strideP, batch_count) &
       result(sgebd2_strided_batched) &
       bind(C, name="rocsolver_sgebd2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tauq
       integer(c_int64_t), value :: strideQ
       type(c_ptr), value :: taup
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgebd2_strided_batched
    end function rocsolver_sgebd2_strided_batched

    !---------------------------------------------
    ! rocsolver_dgebd2_strided_batched
    !---------------------------------------------
    function rocsolver_dgebd2_strided_batched(handle, m, n, A, lda, strideA, D, strideD, E, &
                                              strideE, tauq, strideQ, taup, strideP, batch_count) &
       result(dgebd2_strided_batched) &
       bind(C, name="rocsolver_dgebd2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tauq
       integer(c_int64_t), value :: strideQ
       type(c_ptr), value :: taup
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgebd2_strided_batched
    end function rocsolver_dgebd2_strided_batched

    !---------------------------------------------
    ! rocsolver_cgebd2_strided_batched
    !---------------------------------------------
    function rocsolver_cgebd2_strided_batched(handle, m, n, A, lda, strideA, D, strideD, E, &
                                              strideE, tauq, strideQ, taup, strideP, batch_count) &
       result(cgebd2_strided_batched) &
       bind(C, name="rocsolver_cgebd2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tauq
       integer(c_int64_t), value :: strideQ
       type(c_ptr), value :: taup
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgebd2_strided_batched
    end function rocsolver_cgebd2_strided_batched

    !---------------------------------------------
    ! rocsolver_zgebd2_strided_batched
    !---------------------------------------------
    function rocsolver_zgebd2_strided_batched(handle, m, n, A, lda, strideA, D, strideD, E, &
                                              strideE, tauq, strideQ, taup, strideP, batch_count) &
       result(zgebd2_strided_batched) &
       bind(C, name="rocsolver_zgebd2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tauq
       integer(c_int64_t), value :: strideQ
       type(c_ptr), value :: taup
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgebd2_strided_batched
    end function rocsolver_zgebd2_strided_batched

    !---------------------------------------------
    ! rocsolver_sgebrd
    !---------------------------------------------
    function rocsolver_sgebrd(handle, m, n, A, lda, D, E, tauq, taup) &
       result(sgebrd) &
       bind(C, name="rocsolver_sgebrd")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       integer(kind(rocblas_status_success)) :: sgebrd
    end function rocsolver_sgebrd

    !---------------------------------------------
    ! rocsolver_dgebrd
    !---------------------------------------------
    function rocsolver_dgebrd(handle, m, n, A, lda, D, E, tauq, taup) &
       result(dgebrd) &
       bind(C, name="rocsolver_dgebrd")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       integer(kind(rocblas_status_success)) :: dgebrd
    end function rocsolver_dgebrd

    !---------------------------------------------
    ! rocsolver_cgebrd
    !---------------------------------------------
    function rocsolver_cgebrd(handle, m, n, A, lda, D, E, tauq, taup) &
       result(cgebrd) &
       bind(C, name="rocsolver_cgebrd")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       integer(kind(rocblas_status_success)) :: cgebrd
    end function rocsolver_cgebrd

    !---------------------------------------------
    ! rocsolver_zgebrd
    !---------------------------------------------
    function rocsolver_zgebrd(handle, m, n, A, lda, D, E, tauq, taup) &
       result(zgebrd) &
       bind(C, name="rocsolver_zgebrd")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       integer(kind(rocblas_status_success)) :: zgebrd
    end function rocsolver_zgebrd

    !---------------------------------------------
    ! rocsolver_sgebrd_batched
    !---------------------------------------------
    function rocsolver_sgebrd_batched(handle, m, n, A, lda, D, strideD, E, strideE, tauq, strideQ, &
                                      taup, strideP, batch_count) &
       result(sgebrd_batched) &
       bind(C, name="rocsolver_sgebrd_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tauq
       integer(c_int64_t), value :: strideQ
       type(c_ptr), value :: taup
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgebrd_batched
    end function rocsolver_sgebrd_batched

    !---------------------------------------------
    ! rocsolver_dgebrd_batched
    !---------------------------------------------
    function rocsolver_dgebrd_batched(handle, m, n, A, lda, D, strideD, E, strideE, tauq, strideQ, &
                                      taup, strideP, batch_count) &
       result(dgebrd_batched) &
       bind(C, name="rocsolver_dgebrd_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tauq
       integer(c_int64_t), value :: strideQ
       type(c_ptr), value :: taup
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgebrd_batched
    end function rocsolver_dgebrd_batched

    !---------------------------------------------
    ! rocsolver_cgebrd_batched
    !---------------------------------------------
    function rocsolver_cgebrd_batched(handle, m, n, A, lda, D, strideD, E, strideE, tauq, strideQ, &
                                      taup, strideP, batch_count) &
       result(cgebrd_batched) &
       bind(C, name="rocsolver_cgebrd_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tauq
       integer(c_int64_t), value :: strideQ
       type(c_ptr), value :: taup
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgebrd_batched
    end function rocsolver_cgebrd_batched

    !---------------------------------------------
    ! rocsolver_zgebrd_batched
    !---------------------------------------------
    function rocsolver_zgebrd_batched(handle, m, n, A, lda, D, strideD, E, strideE, tauq, strideQ, &
                                      taup, strideP, batch_count) &
       result(zgebrd_batched) &
       bind(C, name="rocsolver_zgebrd_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tauq
       integer(c_int64_t), value :: strideQ
       type(c_ptr), value :: taup
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgebrd_batched
    end function rocsolver_zgebrd_batched

    !---------------------------------------------
    ! rocsolver_sgebrd_strided_batched
    !---------------------------------------------
    function rocsolver_sgebrd_strided_batched(handle, m, n, A, lda, strideA, D, strideD, E, &
                                              strideE, tauq, strideQ, taup, strideP, batch_count) &
       result(sgebrd_strided_batched) &
       bind(C, name="rocsolver_sgebrd_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tauq
       integer(c_int64_t), value :: strideQ
       type(c_ptr), value :: taup
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgebrd_strided_batched
    end function rocsolver_sgebrd_strided_batched

    !---------------------------------------------
    ! rocsolver_dgebrd_strided_batched
    !---------------------------------------------
    function rocsolver_dgebrd_strided_batched(handle, m, n, A, lda, strideA, D, strideD, E, &
                                              strideE, tauq, strideQ, taup, strideP, batch_count) &
       result(dgebrd_strided_batched) &
       bind(C, name="rocsolver_dgebrd_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tauq
       integer(c_int64_t), value :: strideQ
       type(c_ptr), value :: taup
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgebrd_strided_batched
    end function rocsolver_dgebrd_strided_batched

    !---------------------------------------------
    ! rocsolver_cgebrd_strided_batched
    !---------------------------------------------
    function rocsolver_cgebrd_strided_batched(handle, m, n, A, lda, strideA, D, strideD, E, &
                                              strideE, tauq, strideQ, taup, strideP, batch_count) &
       result(cgebrd_strided_batched) &
       bind(C, name="rocsolver_cgebrd_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tauq
       integer(c_int64_t), value :: strideQ
       type(c_ptr), value :: taup
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgebrd_strided_batched
    end function rocsolver_cgebrd_strided_batched

    !---------------------------------------------
    ! rocsolver_zgebrd_strided_batched
    !---------------------------------------------
    function rocsolver_zgebrd_strided_batched(handle, m, n, A, lda, strideA, D, strideD, E, &
                                              strideE, tauq, strideQ, taup, strideP, batch_count) &
       result(zgebrd_strided_batched) &
       bind(C, name="rocsolver_zgebrd_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tauq
       integer(c_int64_t), value :: strideQ
       type(c_ptr), value :: taup
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgebrd_strided_batched
    end function rocsolver_zgebrd_strided_batched

    !---------------------------------------------
    ! rocsolver_sgetrs
    !---------------------------------------------
    function rocsolver_sgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb) &
       result(sgetrs) &
       bind(C, name="rocsolver_sgetrs")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: sgetrs
    end function rocsolver_sgetrs

    !---------------------------------------------
    ! rocsolver_dgetrs
    !---------------------------------------------
    function rocsolver_dgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb) &
       result(dgetrs) &
       bind(C, name="rocsolver_dgetrs")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: dgetrs
    end function rocsolver_dgetrs

    !---------------------------------------------
    ! rocsolver_cgetrs
    !---------------------------------------------
    function rocsolver_cgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb) &
       result(cgetrs) &
       bind(C, name="rocsolver_cgetrs")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: cgetrs
    end function rocsolver_cgetrs

    !---------------------------------------------
    ! rocsolver_zgetrs
    !---------------------------------------------
    function rocsolver_zgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb) &
       result(zgetrs) &
       bind(C, name="rocsolver_zgetrs")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: zgetrs
    end function rocsolver_zgetrs

    !---------------------------------------------
    ! rocsolver_sgetrs_64
    !---------------------------------------------
    function rocsolver_sgetrs_64(handle, trans, n, nrhs, A, lda, ipiv, B, ldb) &
       result(sgetrs_64) &
       bind(C, name="rocsolver_sgetrs_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(kind(rocblas_status_success)) :: sgetrs_64
    end function rocsolver_sgetrs_64

    !---------------------------------------------
    ! rocsolver_dgetrs_64
    !---------------------------------------------
    function rocsolver_dgetrs_64(handle, trans, n, nrhs, A, lda, ipiv, B, ldb) &
       result(dgetrs_64) &
       bind(C, name="rocsolver_dgetrs_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(kind(rocblas_status_success)) :: dgetrs_64
    end function rocsolver_dgetrs_64

    !---------------------------------------------
    ! rocsolver_cgetrs_64
    !---------------------------------------------
    function rocsolver_cgetrs_64(handle, trans, n, nrhs, A, lda, ipiv, B, ldb) &
       result(cgetrs_64) &
       bind(C, name="rocsolver_cgetrs_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(kind(rocblas_status_success)) :: cgetrs_64
    end function rocsolver_cgetrs_64

    !---------------------------------------------
    ! rocsolver_zgetrs_64
    !---------------------------------------------
    function rocsolver_zgetrs_64(handle, trans, n, nrhs, A, lda, ipiv, B, ldb) &
       result(zgetrs_64) &
       bind(C, name="rocsolver_zgetrs_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(kind(rocblas_status_success)) :: zgetrs_64
    end function rocsolver_zgetrs_64

    !---------------------------------------------
    ! rocsolver_sgetrs_batched
    !---------------------------------------------
    function rocsolver_sgetrs_batched(handle, trans, n, nrhs, A, lda, ipiv, strideP, B, ldb, &
                                      batch_count) &
       result(sgetrs_batched) &
       bind(C, name="rocsolver_sgetrs_batched")
       import :: c_ptr, rocblas_operation_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetrs_batched
    end function rocsolver_sgetrs_batched

    !---------------------------------------------
    ! rocsolver_dgetrs_batched
    !---------------------------------------------
    function rocsolver_dgetrs_batched(handle, trans, n, nrhs, A, lda, ipiv, strideP, B, ldb, &
                                      batch_count) &
       result(dgetrs_batched) &
       bind(C, name="rocsolver_dgetrs_batched")
       import :: c_ptr, rocblas_operation_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetrs_batched
    end function rocsolver_dgetrs_batched

    !---------------------------------------------
    ! rocsolver_cgetrs_batched
    !---------------------------------------------
    function rocsolver_cgetrs_batched(handle, trans, n, nrhs, A, lda, ipiv, strideP, B, ldb, &
                                      batch_count) &
       result(cgetrs_batched) &
       bind(C, name="rocsolver_cgetrs_batched")
       import :: c_ptr, rocblas_operation_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetrs_batched
    end function rocsolver_cgetrs_batched

    !---------------------------------------------
    ! rocsolver_zgetrs_batched
    !---------------------------------------------
    function rocsolver_zgetrs_batched(handle, trans, n, nrhs, A, lda, ipiv, strideP, B, ldb, &
                                      batch_count) &
       result(zgetrs_batched) &
       bind(C, name="rocsolver_zgetrs_batched")
       import :: c_ptr, rocblas_operation_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetrs_batched
    end function rocsolver_zgetrs_batched

    !---------------------------------------------
    ! rocsolver_sgetrs_batched_64
    !---------------------------------------------
    function rocsolver_sgetrs_batched_64(handle, trans, n, nrhs, A, lda, ipiv, strideP, B, ldb, &
                                         batch_count) &
       result(sgetrs_batched_64) &
       bind(C, name="rocsolver_sgetrs_batched_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetrs_batched_64
    end function rocsolver_sgetrs_batched_64

    !---------------------------------------------
    ! rocsolver_dgetrs_batched_64
    !---------------------------------------------
    function rocsolver_dgetrs_batched_64(handle, trans, n, nrhs, A, lda, ipiv, strideP, B, ldb, &
                                         batch_count) &
       result(dgetrs_batched_64) &
       bind(C, name="rocsolver_dgetrs_batched_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetrs_batched_64
    end function rocsolver_dgetrs_batched_64

    !---------------------------------------------
    ! rocsolver_cgetrs_batched_64
    !---------------------------------------------
    function rocsolver_cgetrs_batched_64(handle, trans, n, nrhs, A, lda, ipiv, strideP, B, ldb, &
                                         batch_count) &
       result(cgetrs_batched_64) &
       bind(C, name="rocsolver_cgetrs_batched_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetrs_batched_64
    end function rocsolver_cgetrs_batched_64

    !---------------------------------------------
    ! rocsolver_zgetrs_batched_64
    !---------------------------------------------
    function rocsolver_zgetrs_batched_64(handle, trans, n, nrhs, A, lda, ipiv, strideP, B, ldb, &
                                         batch_count) &
       result(zgetrs_batched_64) &
       bind(C, name="rocsolver_zgetrs_batched_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetrs_batched_64
    end function rocsolver_zgetrs_batched_64

    !---------------------------------------------
    ! rocsolver_sgetrs_strided_batched
    !---------------------------------------------
    function rocsolver_sgetrs_strided_batched(handle, trans, n, nrhs, A, lda, strideA, ipiv, &
                                              strideP, B, ldb, strideB, batch_count) &
       result(sgetrs_strided_batched) &
       bind(C, name="rocsolver_sgetrs_strided_batched")
       import :: c_ptr, rocblas_operation_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetrs_strided_batched
    end function rocsolver_sgetrs_strided_batched

    !---------------------------------------------
    ! rocsolver_dgetrs_strided_batched
    !---------------------------------------------
    function rocsolver_dgetrs_strided_batched(handle, trans, n, nrhs, A, lda, strideA, ipiv, &
                                              strideP, B, ldb, strideB, batch_count) &
       result(dgetrs_strided_batched) &
       bind(C, name="rocsolver_dgetrs_strided_batched")
       import :: c_ptr, rocblas_operation_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetrs_strided_batched
    end function rocsolver_dgetrs_strided_batched

    !---------------------------------------------
    ! rocsolver_cgetrs_strided_batched
    !---------------------------------------------
    function rocsolver_cgetrs_strided_batched(handle, trans, n, nrhs, A, lda, strideA, ipiv, &
                                              strideP, B, ldb, strideB, batch_count) &
       result(cgetrs_strided_batched) &
       bind(C, name="rocsolver_cgetrs_strided_batched")
       import :: c_ptr, rocblas_operation_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetrs_strided_batched
    end function rocsolver_cgetrs_strided_batched

    !---------------------------------------------
    ! rocsolver_zgetrs_strided_batched
    !---------------------------------------------
    function rocsolver_zgetrs_strided_batched(handle, trans, n, nrhs, A, lda, strideA, ipiv, &
                                              strideP, B, ldb, strideB, batch_count) &
       result(zgetrs_strided_batched) &
       bind(C, name="rocsolver_zgetrs_strided_batched")
       import :: c_ptr, rocblas_operation_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetrs_strided_batched
    end function rocsolver_zgetrs_strided_batched

    !---------------------------------------------
    ! rocsolver_sgetrs_strided_batched_64
    !---------------------------------------------
    function rocsolver_sgetrs_strided_batched_64(handle, trans, n, nrhs, A, lda, strideA, ipiv, &
                                                 strideP, B, ldb, strideB, batch_count) &
       result(sgetrs_strided_batched_64) &
       bind(C, name="rocsolver_sgetrs_strided_batched_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetrs_strided_batched_64
    end function rocsolver_sgetrs_strided_batched_64

    !---------------------------------------------
    ! rocsolver_dgetrs_strided_batched_64
    !---------------------------------------------
    function rocsolver_dgetrs_strided_batched_64(handle, trans, n, nrhs, A, lda, strideA, ipiv, &
                                                 strideP, B, ldb, strideB, batch_count) &
       result(dgetrs_strided_batched_64) &
       bind(C, name="rocsolver_dgetrs_strided_batched_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetrs_strided_batched_64
    end function rocsolver_dgetrs_strided_batched_64

    !---------------------------------------------
    ! rocsolver_cgetrs_strided_batched_64
    !---------------------------------------------
    function rocsolver_cgetrs_strided_batched_64(handle, trans, n, nrhs, A, lda, strideA, ipiv, &
                                                 strideP, B, ldb, strideB, batch_count) &
       result(cgetrs_strided_batched_64) &
       bind(C, name="rocsolver_cgetrs_strided_batched_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetrs_strided_batched_64
    end function rocsolver_cgetrs_strided_batched_64

    !---------------------------------------------
    ! rocsolver_zgetrs_strided_batched_64
    !---------------------------------------------
    function rocsolver_zgetrs_strided_batched_64(handle, trans, n, nrhs, A, lda, strideA, ipiv, &
                                                 strideP, B, ldb, strideB, batch_count) &
       result(zgetrs_strided_batched_64) &
       bind(C, name="rocsolver_zgetrs_strided_batched_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetrs_strided_batched_64
    end function rocsolver_zgetrs_strided_batched_64

    !---------------------------------------------
    ! rocsolver_ssytrs
    !---------------------------------------------
    function rocsolver_ssytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb) &
       result(ssytrs) &
       bind(C, name="rocsolver_ssytrs")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: ssytrs
    end function rocsolver_ssytrs

    !---------------------------------------------
    ! rocsolver_dsytrs
    !---------------------------------------------
    function rocsolver_dsytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb) &
       result(dsytrs) &
       bind(C, name="rocsolver_dsytrs")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: dsytrs
    end function rocsolver_dsytrs

    !---------------------------------------------
    ! rocsolver_csytrs
    !---------------------------------------------
    function rocsolver_csytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb) &
       result(csytrs) &
       bind(C, name="rocsolver_csytrs")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: csytrs
    end function rocsolver_csytrs

    !---------------------------------------------
    ! rocsolver_zsytrs
    !---------------------------------------------
    function rocsolver_zsytrs(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb) &
       result(zsytrs) &
       bind(C, name="rocsolver_zsytrs")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: zsytrs
    end function rocsolver_zsytrs

    !---------------------------------------------
    ! rocsolver_ssytrs_64
    !---------------------------------------------
    function rocsolver_ssytrs_64(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb) &
       result(ssytrs_64) &
       bind(C, name="rocsolver_ssytrs_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(kind(rocblas_status_success)) :: ssytrs_64
    end function rocsolver_ssytrs_64

    !---------------------------------------------
    ! rocsolver_dsytrs_64
    !---------------------------------------------
    function rocsolver_dsytrs_64(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb) &
       result(dsytrs_64) &
       bind(C, name="rocsolver_dsytrs_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(kind(rocblas_status_success)) :: dsytrs_64
    end function rocsolver_dsytrs_64

    !---------------------------------------------
    ! rocsolver_csytrs_64
    !---------------------------------------------
    function rocsolver_csytrs_64(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb) &
       result(csytrs_64) &
       bind(C, name="rocsolver_csytrs_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(kind(rocblas_status_success)) :: csytrs_64
    end function rocsolver_csytrs_64

    !---------------------------------------------
    ! rocsolver_zsytrs_64
    !---------------------------------------------
    function rocsolver_zsytrs_64(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb) &
       result(zsytrs_64) &
       bind(C, name="rocsolver_zsytrs_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(kind(rocblas_status_success)) :: zsytrs_64
    end function rocsolver_zsytrs_64

    !---------------------------------------------
    ! rocsolver_ssytrs_batched
    !---------------------------------------------
    function rocsolver_ssytrs_batched(handle, uplo, n, nrhs, A, lda, ipiv, strideP, B, ldb, &
                                      batch_count) &
       result(ssytrs_batched) &
       bind(C, name="rocsolver_ssytrs_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssytrs_batched
    end function rocsolver_ssytrs_batched

    !---------------------------------------------
    ! rocsolver_dsytrs_batched
    !---------------------------------------------
    function rocsolver_dsytrs_batched(handle, uplo, n, nrhs, A, lda, ipiv, strideP, B, ldb, &
                                      batch_count) &
       result(dsytrs_batched) &
       bind(C, name="rocsolver_dsytrs_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsytrs_batched
    end function rocsolver_dsytrs_batched

    !---------------------------------------------
    ! rocsolver_csytrs_batched
    !---------------------------------------------
    function rocsolver_csytrs_batched(handle, uplo, n, nrhs, A, lda, ipiv, strideP, B, ldb, &
                                      batch_count) &
       result(csytrs_batched) &
       bind(C, name="rocsolver_csytrs_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: csytrs_batched
    end function rocsolver_csytrs_batched

    !---------------------------------------------
    ! rocsolver_zsytrs_batched
    !---------------------------------------------
    function rocsolver_zsytrs_batched(handle, uplo, n, nrhs, A, lda, ipiv, strideP, B, ldb, &
                                      batch_count) &
       result(zsytrs_batched) &
       bind(C, name="rocsolver_zsytrs_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zsytrs_batched
    end function rocsolver_zsytrs_batched

    !---------------------------------------------
    ! rocsolver_ssytrs_batched_64
    !---------------------------------------------
    function rocsolver_ssytrs_batched_64(handle, uplo, n, nrhs, A, lda, ipiv, strideP, B, ldb, &
                                         batch_count) &
       result(ssytrs_batched_64) &
       bind(C, name="rocsolver_ssytrs_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssytrs_batched_64
    end function rocsolver_ssytrs_batched_64

    !---------------------------------------------
    ! rocsolver_dsytrs_batched_64
    !---------------------------------------------
    function rocsolver_dsytrs_batched_64(handle, uplo, n, nrhs, A, lda, ipiv, strideP, B, ldb, &
                                         batch_count) &
       result(dsytrs_batched_64) &
       bind(C, name="rocsolver_dsytrs_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsytrs_batched_64
    end function rocsolver_dsytrs_batched_64

    !---------------------------------------------
    ! rocsolver_csytrs_batched_64
    !---------------------------------------------
    function rocsolver_csytrs_batched_64(handle, uplo, n, nrhs, A, lda, ipiv, strideP, B, ldb, &
                                         batch_count) &
       result(csytrs_batched_64) &
       bind(C, name="rocsolver_csytrs_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: csytrs_batched_64
    end function rocsolver_csytrs_batched_64

    !---------------------------------------------
    ! rocsolver_zsytrs_batched_64
    !---------------------------------------------
    function rocsolver_zsytrs_batched_64(handle, uplo, n, nrhs, A, lda, ipiv, strideP, B, ldb, &
                                         batch_count) &
       result(zsytrs_batched_64) &
       bind(C, name="rocsolver_zsytrs_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zsytrs_batched_64
    end function rocsolver_zsytrs_batched_64

    !---------------------------------------------
    ! rocsolver_ssytrs_strided_batched
    !---------------------------------------------
    function rocsolver_ssytrs_strided_batched(handle, uplo, n, nrhs, A, lda, strideA, ipiv, &
                                              strideP, B, ldb, strideB, batch_count) &
       result(ssytrs_strided_batched) &
       bind(C, name="rocsolver_ssytrs_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssytrs_strided_batched
    end function rocsolver_ssytrs_strided_batched

    !---------------------------------------------
    ! rocsolver_dsytrs_strided_batched
    !---------------------------------------------
    function rocsolver_dsytrs_strided_batched(handle, uplo, n, nrhs, A, lda, strideA, ipiv, &
                                              strideP, B, ldb, strideB, batch_count) &
       result(dsytrs_strided_batched) &
       bind(C, name="rocsolver_dsytrs_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsytrs_strided_batched
    end function rocsolver_dsytrs_strided_batched

    !---------------------------------------------
    ! rocsolver_csytrs_strided_batched
    !---------------------------------------------
    function rocsolver_csytrs_strided_batched(handle, uplo, n, nrhs, A, lda, strideA, ipiv, &
                                              strideP, B, ldb, strideB, batch_count) &
       result(csytrs_strided_batched) &
       bind(C, name="rocsolver_csytrs_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: csytrs_strided_batched
    end function rocsolver_csytrs_strided_batched

    !---------------------------------------------
    ! rocsolver_zsytrs_strided_batched
    !---------------------------------------------
    function rocsolver_zsytrs_strided_batched(handle, uplo, n, nrhs, A, lda, strideA, ipiv, &
                                              strideP, B, ldb, strideB, batch_count) &
       result(zsytrs_strided_batched) &
       bind(C, name="rocsolver_zsytrs_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zsytrs_strided_batched
    end function rocsolver_zsytrs_strided_batched

    !---------------------------------------------
    ! rocsolver_ssytrs_strided_batched_64
    !---------------------------------------------
    function rocsolver_ssytrs_strided_batched_64(handle, uplo, n, nrhs, A, lda, strideA, ipiv, &
                                                 strideP, B, ldb, strideB, batch_count) &
       result(ssytrs_strided_batched_64) &
       bind(C, name="rocsolver_ssytrs_strided_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssytrs_strided_batched_64
    end function rocsolver_ssytrs_strided_batched_64

    !---------------------------------------------
    ! rocsolver_dsytrs_strided_batched_64
    !---------------------------------------------
    function rocsolver_dsytrs_strided_batched_64(handle, uplo, n, nrhs, A, lda, strideA, ipiv, &
                                                 strideP, B, ldb, strideB, batch_count) &
       result(dsytrs_strided_batched_64) &
       bind(C, name="rocsolver_dsytrs_strided_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsytrs_strided_batched_64
    end function rocsolver_dsytrs_strided_batched_64

    !---------------------------------------------
    ! rocsolver_csytrs_strided_batched_64
    !---------------------------------------------
    function rocsolver_csytrs_strided_batched_64(handle, uplo, n, nrhs, A, lda, strideA, ipiv, &
                                                 strideP, B, ldb, strideB, batch_count) &
       result(csytrs_strided_batched_64) &
       bind(C, name="rocsolver_csytrs_strided_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: csytrs_strided_batched_64
    end function rocsolver_csytrs_strided_batched_64

    !---------------------------------------------
    ! rocsolver_zsytrs_strided_batched_64
    !---------------------------------------------
    function rocsolver_zsytrs_strided_batched_64(handle, uplo, n, nrhs, A, lda, strideA, ipiv, &
                                                 strideP, B, ldb, strideB, batch_count) &
       result(zsytrs_strided_batched_64) &
       bind(C, name="rocsolver_zsytrs_strided_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zsytrs_strided_batched_64
    end function rocsolver_zsytrs_strided_batched_64

    !---------------------------------------------
    ! rocsolver_sgesv
    !---------------------------------------------
    function rocsolver_sgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, myInfo) &
       result(sgesv) &
       bind(C, name="rocsolver_sgesv")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgesv
    end function rocsolver_sgesv

    !---------------------------------------------
    ! rocsolver_dgesv
    !---------------------------------------------
    function rocsolver_dgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, myInfo) &
       result(dgesv) &
       bind(C, name="rocsolver_dgesv")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgesv
    end function rocsolver_dgesv

    !---------------------------------------------
    ! rocsolver_cgesv
    !---------------------------------------------
    function rocsolver_cgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, myInfo) &
       result(cgesv) &
       bind(C, name="rocsolver_cgesv")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgesv
    end function rocsolver_cgesv

    !---------------------------------------------
    ! rocsolver_zgesv
    !---------------------------------------------
    function rocsolver_zgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, myInfo) &
       result(zgesv) &
       bind(C, name="rocsolver_zgesv")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgesv
    end function rocsolver_zgesv

    !---------------------------------------------
    ! rocsolver_sgesv_batched
    !---------------------------------------------
    function rocsolver_sgesv_batched(handle, n, nrhs, A, lda, ipiv, strideP, B, ldb, myInfo, &
                                     batch_count) &
       result(sgesv_batched) &
       bind(C, name="rocsolver_sgesv_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgesv_batched
    end function rocsolver_sgesv_batched

    !---------------------------------------------
    ! rocsolver_dgesv_batched
    !---------------------------------------------
    function rocsolver_dgesv_batched(handle, n, nrhs, A, lda, ipiv, strideP, B, ldb, myInfo, &
                                     batch_count) &
       result(dgesv_batched) &
       bind(C, name="rocsolver_dgesv_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgesv_batched
    end function rocsolver_dgesv_batched

    !---------------------------------------------
    ! rocsolver_cgesv_batched
    !---------------------------------------------
    function rocsolver_cgesv_batched(handle, n, nrhs, A, lda, ipiv, strideP, B, ldb, myInfo, &
                                     batch_count) &
       result(cgesv_batched) &
       bind(C, name="rocsolver_cgesv_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgesv_batched
    end function rocsolver_cgesv_batched

    !---------------------------------------------
    ! rocsolver_zgesv_batched
    !---------------------------------------------
    function rocsolver_zgesv_batched(handle, n, nrhs, A, lda, ipiv, strideP, B, ldb, myInfo, &
                                     batch_count) &
       result(zgesv_batched) &
       bind(C, name="rocsolver_zgesv_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgesv_batched
    end function rocsolver_zgesv_batched

    !---------------------------------------------
    ! rocsolver_sgesv_strided_batched
    !---------------------------------------------
    function rocsolver_sgesv_strided_batched(handle, n, nrhs, A, lda, strideA, ipiv, strideP, B, &
                                             ldb, strideB, myInfo, batch_count) &
       result(sgesv_strided_batched) &
       bind(C, name="rocsolver_sgesv_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgesv_strided_batched
    end function rocsolver_sgesv_strided_batched

    !---------------------------------------------
    ! rocsolver_dgesv_strided_batched
    !---------------------------------------------
    function rocsolver_dgesv_strided_batched(handle, n, nrhs, A, lda, strideA, ipiv, strideP, B, &
                                             ldb, strideB, myInfo, batch_count) &
       result(dgesv_strided_batched) &
       bind(C, name="rocsolver_dgesv_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgesv_strided_batched
    end function rocsolver_dgesv_strided_batched

    !---------------------------------------------
    ! rocsolver_cgesv_strided_batched
    !---------------------------------------------
    function rocsolver_cgesv_strided_batched(handle, n, nrhs, A, lda, strideA, ipiv, strideP, B, &
                                             ldb, strideB, myInfo, batch_count) &
       result(cgesv_strided_batched) &
       bind(C, name="rocsolver_cgesv_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgesv_strided_batched
    end function rocsolver_cgesv_strided_batched

    !---------------------------------------------
    ! rocsolver_zgesv_strided_batched
    !---------------------------------------------
    function rocsolver_zgesv_strided_batched(handle, n, nrhs, A, lda, strideA, ipiv, strideP, B, &
                                             ldb, strideB, myInfo, batch_count) &
       result(zgesv_strided_batched) &
       bind(C, name="rocsolver_zgesv_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgesv_strided_batched
    end function rocsolver_zgesv_strided_batched

    !---------------------------------------------
    ! rocsolver_sgetrs_npvt
    !---------------------------------------------
    function rocsolver_sgetrs_npvt(handle, trans, n, nrhs, A, lda, B, ldb) &
       result(sgetrs_npvt) &
       bind(C, name="rocsolver_sgetrs_npvt")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: sgetrs_npvt
    end function rocsolver_sgetrs_npvt

    !---------------------------------------------
    ! rocsolver_dgetrs_npvt
    !---------------------------------------------
    function rocsolver_dgetrs_npvt(handle, trans, n, nrhs, A, lda, B, ldb) &
       result(dgetrs_npvt) &
       bind(C, name="rocsolver_dgetrs_npvt")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: dgetrs_npvt
    end function rocsolver_dgetrs_npvt

    !---------------------------------------------
    ! rocsolver_cgetrs_npvt
    !---------------------------------------------
    function rocsolver_cgetrs_npvt(handle, trans, n, nrhs, A, lda, B, ldb) &
       result(cgetrs_npvt) &
       bind(C, name="rocsolver_cgetrs_npvt")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: cgetrs_npvt
    end function rocsolver_cgetrs_npvt

    !---------------------------------------------
    ! rocsolver_zgetrs_npvt
    !---------------------------------------------
    function rocsolver_zgetrs_npvt(handle, trans, n, nrhs, A, lda, B, ldb) &
       result(zgetrs_npvt) &
       bind(C, name="rocsolver_zgetrs_npvt")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: zgetrs_npvt
    end function rocsolver_zgetrs_npvt

    !---------------------------------------------
    ! rocsolver_sgetrs_npvt_64
    !---------------------------------------------
    function rocsolver_sgetrs_npvt_64(handle, trans, n, nrhs, A, lda, B, ldb) &
       result(sgetrs_npvt_64) &
       bind(C, name="rocsolver_sgetrs_npvt_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(kind(rocblas_status_success)) :: sgetrs_npvt_64
    end function rocsolver_sgetrs_npvt_64

    !---------------------------------------------
    ! rocsolver_dgetrs_npvt_64
    !---------------------------------------------
    function rocsolver_dgetrs_npvt_64(handle, trans, n, nrhs, A, lda, B, ldb) &
       result(dgetrs_npvt_64) &
       bind(C, name="rocsolver_dgetrs_npvt_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(kind(rocblas_status_success)) :: dgetrs_npvt_64
    end function rocsolver_dgetrs_npvt_64

    !---------------------------------------------
    ! rocsolver_cgetrs_npvt_64
    !---------------------------------------------
    function rocsolver_cgetrs_npvt_64(handle, trans, n, nrhs, A, lda, B, ldb) &
       result(cgetrs_npvt_64) &
       bind(C, name="rocsolver_cgetrs_npvt_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(kind(rocblas_status_success)) :: cgetrs_npvt_64
    end function rocsolver_cgetrs_npvt_64

    !---------------------------------------------
    ! rocsolver_zgetrs_npvt_64
    !---------------------------------------------
    function rocsolver_zgetrs_npvt_64(handle, trans, n, nrhs, A, lda, B, ldb) &
       result(zgetrs_npvt_64) &
       bind(C, name="rocsolver_zgetrs_npvt_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(kind(rocblas_status_success)) :: zgetrs_npvt_64
    end function rocsolver_zgetrs_npvt_64

    !---------------------------------------------
    ! rocsolver_sgetrs_npvt_batched
    !---------------------------------------------
    function rocsolver_sgetrs_npvt_batched(handle, trans, n, nrhs, A, lda, B, ldb, batch_count) &
       result(sgetrs_npvt_batched) &
       bind(C, name="rocsolver_sgetrs_npvt_batched")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetrs_npvt_batched
    end function rocsolver_sgetrs_npvt_batched

    !---------------------------------------------
    ! rocsolver_dgetrs_npvt_batched
    !---------------------------------------------
    function rocsolver_dgetrs_npvt_batched(handle, trans, n, nrhs, A, lda, B, ldb, batch_count) &
       result(dgetrs_npvt_batched) &
       bind(C, name="rocsolver_dgetrs_npvt_batched")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetrs_npvt_batched
    end function rocsolver_dgetrs_npvt_batched

    !---------------------------------------------
    ! rocsolver_cgetrs_npvt_batched
    !---------------------------------------------
    function rocsolver_cgetrs_npvt_batched(handle, trans, n, nrhs, A, lda, B, ldb, batch_count) &
       result(cgetrs_npvt_batched) &
       bind(C, name="rocsolver_cgetrs_npvt_batched")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetrs_npvt_batched
    end function rocsolver_cgetrs_npvt_batched

    !---------------------------------------------
    ! rocsolver_zgetrs_npvt_batched
    !---------------------------------------------
    function rocsolver_zgetrs_npvt_batched(handle, trans, n, nrhs, A, lda, B, ldb, batch_count) &
       result(zgetrs_npvt_batched) &
       bind(C, name="rocsolver_zgetrs_npvt_batched")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetrs_npvt_batched
    end function rocsolver_zgetrs_npvt_batched

    !---------------------------------------------
    ! rocsolver_sgetrs_npvt_batched_64
    !---------------------------------------------
    function rocsolver_sgetrs_npvt_batched_64(handle, trans, n, nrhs, A, lda, B, ldb, batch_count) &
       result(sgetrs_npvt_batched_64) &
       bind(C, name="rocsolver_sgetrs_npvt_batched_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetrs_npvt_batched_64
    end function rocsolver_sgetrs_npvt_batched_64

    !---------------------------------------------
    ! rocsolver_dgetrs_npvt_batched_64
    !---------------------------------------------
    function rocsolver_dgetrs_npvt_batched_64(handle, trans, n, nrhs, A, lda, B, ldb, batch_count) &
       result(dgetrs_npvt_batched_64) &
       bind(C, name="rocsolver_dgetrs_npvt_batched_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetrs_npvt_batched_64
    end function rocsolver_dgetrs_npvt_batched_64

    !---------------------------------------------
    ! rocsolver_cgetrs_npvt_batched_64
    !---------------------------------------------
    function rocsolver_cgetrs_npvt_batched_64(handle, trans, n, nrhs, A, lda, B, ldb, batch_count) &
       result(cgetrs_npvt_batched_64) &
       bind(C, name="rocsolver_cgetrs_npvt_batched_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetrs_npvt_batched_64
    end function rocsolver_cgetrs_npvt_batched_64

    !---------------------------------------------
    ! rocsolver_zgetrs_npvt_batched_64
    !---------------------------------------------
    function rocsolver_zgetrs_npvt_batched_64(handle, trans, n, nrhs, A, lda, B, ldb, batch_count) &
       result(zgetrs_npvt_batched_64) &
       bind(C, name="rocsolver_zgetrs_npvt_batched_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetrs_npvt_batched_64
    end function rocsolver_zgetrs_npvt_batched_64

    !---------------------------------------------
    ! rocsolver_sgetrs_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_sgetrs_npvt_strided_batched(handle, trans, n, nrhs, A, lda, strideA, B, &
                                                   ldb, strideB, batch_count) &
       result(sgetrs_npvt_strided_batched) &
       bind(C, name="rocsolver_sgetrs_npvt_strided_batched")
       import :: c_ptr, rocblas_operation_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetrs_npvt_strided_batched
    end function rocsolver_sgetrs_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_dgetrs_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_dgetrs_npvt_strided_batched(handle, trans, n, nrhs, A, lda, strideA, B, &
                                                   ldb, strideB, batch_count) &
       result(dgetrs_npvt_strided_batched) &
       bind(C, name="rocsolver_dgetrs_npvt_strided_batched")
       import :: c_ptr, rocblas_operation_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetrs_npvt_strided_batched
    end function rocsolver_dgetrs_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_cgetrs_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_cgetrs_npvt_strided_batched(handle, trans, n, nrhs, A, lda, strideA, B, &
                                                   ldb, strideB, batch_count) &
       result(cgetrs_npvt_strided_batched) &
       bind(C, name="rocsolver_cgetrs_npvt_strided_batched")
       import :: c_ptr, rocblas_operation_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetrs_npvt_strided_batched
    end function rocsolver_cgetrs_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_zgetrs_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_zgetrs_npvt_strided_batched(handle, trans, n, nrhs, A, lda, strideA, B, &
                                                   ldb, strideB, batch_count) &
       result(zgetrs_npvt_strided_batched) &
       bind(C, name="rocsolver_zgetrs_npvt_strided_batched")
       import :: c_ptr, rocblas_operation_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetrs_npvt_strided_batched
    end function rocsolver_zgetrs_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_sgetrs_npvt_strided_batched_64
    !---------------------------------------------
    function rocsolver_sgetrs_npvt_strided_batched_64(handle, trans, n, nrhs, A, lda, strideA, B, &
                                                      ldb, strideB, batch_count) &
       result(sgetrs_npvt_strided_batched_64) &
       bind(C, name="rocsolver_sgetrs_npvt_strided_batched_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetrs_npvt_strided_batched_64
    end function rocsolver_sgetrs_npvt_strided_batched_64

    !---------------------------------------------
    ! rocsolver_dgetrs_npvt_strided_batched_64
    !---------------------------------------------
    function rocsolver_dgetrs_npvt_strided_batched_64(handle, trans, n, nrhs, A, lda, strideA, B, &
                                                      ldb, strideB, batch_count) &
       result(dgetrs_npvt_strided_batched_64) &
       bind(C, name="rocsolver_dgetrs_npvt_strided_batched_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetrs_npvt_strided_batched_64
    end function rocsolver_dgetrs_npvt_strided_batched_64

    !---------------------------------------------
    ! rocsolver_cgetrs_npvt_strided_batched_64
    !---------------------------------------------
    function rocsolver_cgetrs_npvt_strided_batched_64(handle, trans, n, nrhs, A, lda, strideA, B, &
                                                      ldb, strideB, batch_count) &
       result(cgetrs_npvt_strided_batched_64) &
       bind(C, name="rocsolver_cgetrs_npvt_strided_batched_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetrs_npvt_strided_batched_64
    end function rocsolver_cgetrs_npvt_strided_batched_64

    !---------------------------------------------
    ! rocsolver_zgetrs_npvt_strided_batched_64
    !---------------------------------------------
    function rocsolver_zgetrs_npvt_strided_batched_64(handle, trans, n, nrhs, A, lda, strideA, B, &
                                                      ldb, strideB, batch_count) &
       result(zgetrs_npvt_strided_batched_64) &
       bind(C, name="rocsolver_zgetrs_npvt_strided_batched_64")
       import :: c_ptr, rocblas_operation_none, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetrs_npvt_strided_batched_64
    end function rocsolver_zgetrs_npvt_strided_batched_64

    !---------------------------------------------
    ! rocsolver_sgetri
    !---------------------------------------------
    function rocsolver_sgetri(handle, n, A, lda, ipiv, myInfo) &
       result(sgetri) &
       bind(C, name="rocsolver_sgetri")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgetri
    end function rocsolver_sgetri

    !---------------------------------------------
    ! rocsolver_dgetri
    !---------------------------------------------
    function rocsolver_dgetri(handle, n, A, lda, ipiv, myInfo) &
       result(dgetri) &
       bind(C, name="rocsolver_dgetri")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgetri
    end function rocsolver_dgetri

    !---------------------------------------------
    ! rocsolver_cgetri
    !---------------------------------------------
    function rocsolver_cgetri(handle, n, A, lda, ipiv, myInfo) &
       result(cgetri) &
       bind(C, name="rocsolver_cgetri")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgetri
    end function rocsolver_cgetri

    !---------------------------------------------
    ! rocsolver_zgetri
    !---------------------------------------------
    function rocsolver_zgetri(handle, n, A, lda, ipiv, myInfo) &
       result(zgetri) &
       bind(C, name="rocsolver_zgetri")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgetri
    end function rocsolver_zgetri

    !---------------------------------------------
    ! rocsolver_sgetri_batched
    !---------------------------------------------
    function rocsolver_sgetri_batched(handle, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(sgetri_batched) &
       bind(C, name="rocsolver_sgetri_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetri_batched
    end function rocsolver_sgetri_batched

    !---------------------------------------------
    ! rocsolver_dgetri_batched
    !---------------------------------------------
    function rocsolver_dgetri_batched(handle, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(dgetri_batched) &
       bind(C, name="rocsolver_dgetri_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetri_batched
    end function rocsolver_dgetri_batched

    !---------------------------------------------
    ! rocsolver_cgetri_batched
    !---------------------------------------------
    function rocsolver_cgetri_batched(handle, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(cgetri_batched) &
       bind(C, name="rocsolver_cgetri_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetri_batched
    end function rocsolver_cgetri_batched

    !---------------------------------------------
    ! rocsolver_zgetri_batched
    !---------------------------------------------
    function rocsolver_zgetri_batched(handle, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(zgetri_batched) &
       bind(C, name="rocsolver_zgetri_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetri_batched
    end function rocsolver_zgetri_batched

    !---------------------------------------------
    ! rocsolver_sgetri_strided_batched
    !---------------------------------------------
    function rocsolver_sgetri_strided_batched(handle, n, A, lda, strideA, ipiv, strideP, myInfo, &
                                              batch_count) &
       result(sgetri_strided_batched) &
       bind(C, name="rocsolver_sgetri_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetri_strided_batched
    end function rocsolver_sgetri_strided_batched

    !---------------------------------------------
    ! rocsolver_dgetri_strided_batched
    !---------------------------------------------
    function rocsolver_dgetri_strided_batched(handle, n, A, lda, strideA, ipiv, strideP, myInfo, &
                                              batch_count) &
       result(dgetri_strided_batched) &
       bind(C, name="rocsolver_dgetri_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetri_strided_batched
    end function rocsolver_dgetri_strided_batched

    !---------------------------------------------
    ! rocsolver_cgetri_strided_batched
    !---------------------------------------------
    function rocsolver_cgetri_strided_batched(handle, n, A, lda, strideA, ipiv, strideP, myInfo, &
                                              batch_count) &
       result(cgetri_strided_batched) &
       bind(C, name="rocsolver_cgetri_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetri_strided_batched
    end function rocsolver_cgetri_strided_batched

    !---------------------------------------------
    ! rocsolver_zgetri_strided_batched
    !---------------------------------------------
    function rocsolver_zgetri_strided_batched(handle, n, A, lda, strideA, ipiv, strideP, myInfo, &
                                              batch_count) &
       result(zgetri_strided_batched) &
       bind(C, name="rocsolver_zgetri_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetri_strided_batched
    end function rocsolver_zgetri_strided_batched

    !---------------------------------------------
    ! rocsolver_sgetri_npvt
    !---------------------------------------------
    function rocsolver_sgetri_npvt(handle, n, A, lda, myInfo) &
       result(sgetri_npvt) &
       bind(C, name="rocsolver_sgetri_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgetri_npvt
    end function rocsolver_sgetri_npvt

    !---------------------------------------------
    ! rocsolver_dgetri_npvt
    !---------------------------------------------
    function rocsolver_dgetri_npvt(handle, n, A, lda, myInfo) &
       result(dgetri_npvt) &
       bind(C, name="rocsolver_dgetri_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgetri_npvt
    end function rocsolver_dgetri_npvt

    !---------------------------------------------
    ! rocsolver_cgetri_npvt
    !---------------------------------------------
    function rocsolver_cgetri_npvt(handle, n, A, lda, myInfo) &
       result(cgetri_npvt) &
       bind(C, name="rocsolver_cgetri_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgetri_npvt
    end function rocsolver_cgetri_npvt

    !---------------------------------------------
    ! rocsolver_zgetri_npvt
    !---------------------------------------------
    function rocsolver_zgetri_npvt(handle, n, A, lda, myInfo) &
       result(zgetri_npvt) &
       bind(C, name="rocsolver_zgetri_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgetri_npvt
    end function rocsolver_zgetri_npvt

    !---------------------------------------------
    ! rocsolver_sgetri_npvt_batched
    !---------------------------------------------
    function rocsolver_sgetri_npvt_batched(handle, n, A, lda, myInfo, batch_count) &
       result(sgetri_npvt_batched) &
       bind(C, name="rocsolver_sgetri_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetri_npvt_batched
    end function rocsolver_sgetri_npvt_batched

    !---------------------------------------------
    ! rocsolver_dgetri_npvt_batched
    !---------------------------------------------
    function rocsolver_dgetri_npvt_batched(handle, n, A, lda, myInfo, batch_count) &
       result(dgetri_npvt_batched) &
       bind(C, name="rocsolver_dgetri_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetri_npvt_batched
    end function rocsolver_dgetri_npvt_batched

    !---------------------------------------------
    ! rocsolver_cgetri_npvt_batched
    !---------------------------------------------
    function rocsolver_cgetri_npvt_batched(handle, n, A, lda, myInfo, batch_count) &
       result(cgetri_npvt_batched) &
       bind(C, name="rocsolver_cgetri_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetri_npvt_batched
    end function rocsolver_cgetri_npvt_batched

    !---------------------------------------------
    ! rocsolver_zgetri_npvt_batched
    !---------------------------------------------
    function rocsolver_zgetri_npvt_batched(handle, n, A, lda, myInfo, batch_count) &
       result(zgetri_npvt_batched) &
       bind(C, name="rocsolver_zgetri_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetri_npvt_batched
    end function rocsolver_zgetri_npvt_batched

    !---------------------------------------------
    ! rocsolver_sgetri_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_sgetri_npvt_strided_batched(handle, n, A, lda, strideA, myInfo, &
                                                   batch_count) &
       result(sgetri_npvt_strided_batched) &
       bind(C, name="rocsolver_sgetri_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetri_npvt_strided_batched
    end function rocsolver_sgetri_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_dgetri_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_dgetri_npvt_strided_batched(handle, n, A, lda, strideA, myInfo, &
                                                   batch_count) &
       result(dgetri_npvt_strided_batched) &
       bind(C, name="rocsolver_dgetri_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetri_npvt_strided_batched
    end function rocsolver_dgetri_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_cgetri_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_cgetri_npvt_strided_batched(handle, n, A, lda, strideA, myInfo, &
                                                   batch_count) &
       result(cgetri_npvt_strided_batched) &
       bind(C, name="rocsolver_cgetri_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetri_npvt_strided_batched
    end function rocsolver_cgetri_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_zgetri_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_zgetri_npvt_strided_batched(handle, n, A, lda, strideA, myInfo, &
                                                   batch_count) &
       result(zgetri_npvt_strided_batched) &
       bind(C, name="rocsolver_zgetri_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetri_npvt_strided_batched
    end function rocsolver_zgetri_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_sgels
    !---------------------------------------------
    function rocsolver_sgels(handle, trans, m, n, nrhs, A, lda, B, ldb, myInfo) &
       result(sgels) &
       bind(C, name="rocsolver_sgels")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgels
    end function rocsolver_sgels

    !---------------------------------------------
    ! rocsolver_dgels
    !---------------------------------------------
    function rocsolver_dgels(handle, trans, m, n, nrhs, A, lda, B, ldb, myInfo) &
       result(dgels) &
       bind(C, name="rocsolver_dgels")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgels
    end function rocsolver_dgels

    !---------------------------------------------
    ! rocsolver_cgels
    !---------------------------------------------
    function rocsolver_cgels(handle, trans, m, n, nrhs, A, lda, B, ldb, myInfo) &
       result(cgels) &
       bind(C, name="rocsolver_cgels")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgels
    end function rocsolver_cgels

    !---------------------------------------------
    ! rocsolver_zgels
    !---------------------------------------------
    function rocsolver_zgels(handle, trans, m, n, nrhs, A, lda, B, ldb, myInfo) &
       result(zgels) &
       bind(C, name="rocsolver_zgels")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgels
    end function rocsolver_zgels

    !---------------------------------------------
    ! rocsolver_sgels_batched
    !---------------------------------------------
    function rocsolver_sgels_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, myInfo, &
                                     batch_count) &
       result(sgels_batched) &
       bind(C, name="rocsolver_sgels_batched")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgels_batched
    end function rocsolver_sgels_batched

    !---------------------------------------------
    ! rocsolver_dgels_batched
    !---------------------------------------------
    function rocsolver_dgels_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, myInfo, &
                                     batch_count) &
       result(dgels_batched) &
       bind(C, name="rocsolver_dgels_batched")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgels_batched
    end function rocsolver_dgels_batched

    !---------------------------------------------
    ! rocsolver_cgels_batched
    !---------------------------------------------
    function rocsolver_cgels_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, myInfo, &
                                     batch_count) &
       result(cgels_batched) &
       bind(C, name="rocsolver_cgels_batched")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgels_batched
    end function rocsolver_cgels_batched

    !---------------------------------------------
    ! rocsolver_zgels_batched
    !---------------------------------------------
    function rocsolver_zgels_batched(handle, trans, m, n, nrhs, A, lda, B, ldb, myInfo, &
                                     batch_count) &
       result(zgels_batched) &
       bind(C, name="rocsolver_zgels_batched")
       import :: c_ptr, rocblas_operation_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgels_batched
    end function rocsolver_zgels_batched

    !---------------------------------------------
    ! rocsolver_sgels_strided_batched
    !---------------------------------------------
    function rocsolver_sgels_strided_batched(handle, trans, m, n, nrhs, A, lda, strideA, B, ldb, &
                                             strideB, myInfo, batch_count) &
       result(sgels_strided_batched) &
       bind(C, name="rocsolver_sgels_strided_batched")
       import :: c_ptr, rocblas_operation_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgels_strided_batched
    end function rocsolver_sgels_strided_batched

    !---------------------------------------------
    ! rocsolver_dgels_strided_batched
    !---------------------------------------------
    function rocsolver_dgels_strided_batched(handle, trans, m, n, nrhs, A, lda, strideA, B, ldb, &
                                             strideB, myInfo, batch_count) &
       result(dgels_strided_batched) &
       bind(C, name="rocsolver_dgels_strided_batched")
       import :: c_ptr, rocblas_operation_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgels_strided_batched
    end function rocsolver_dgels_strided_batched

    !---------------------------------------------
    ! rocsolver_cgels_strided_batched
    !---------------------------------------------
    function rocsolver_cgels_strided_batched(handle, trans, m, n, nrhs, A, lda, strideA, B, ldb, &
                                             strideB, myInfo, batch_count) &
       result(cgels_strided_batched) &
       bind(C, name="rocsolver_cgels_strided_batched")
       import :: c_ptr, rocblas_operation_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgels_strided_batched
    end function rocsolver_cgels_strided_batched

    !---------------------------------------------
    ! rocsolver_zgels_strided_batched
    !---------------------------------------------
    function rocsolver_zgels_strided_batched(handle, trans, m, n, nrhs, A, lda, strideA, B, ldb, &
                                             strideB, myInfo, batch_count) &
       result(zgels_strided_batched) &
       bind(C, name="rocsolver_zgels_strided_batched")
       import :: c_ptr, rocblas_operation_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_operation_none)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgels_strided_batched
    end function rocsolver_zgels_strided_batched

    !---------------------------------------------
    ! rocsolver_spotf2
    !---------------------------------------------
    function rocsolver_spotf2(handle, uplo, n, A, lda, myInfo) &
       result(spotf2) &
       bind(C, name="rocsolver_spotf2")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: spotf2
    end function rocsolver_spotf2

    !---------------------------------------------
    ! rocsolver_dpotf2
    !---------------------------------------------
    function rocsolver_dpotf2(handle, uplo, n, A, lda, myInfo) &
       result(dpotf2) &
       bind(C, name="rocsolver_dpotf2")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dpotf2
    end function rocsolver_dpotf2

    !---------------------------------------------
    ! rocsolver_cpotf2
    !---------------------------------------------
    function rocsolver_cpotf2(handle, uplo, n, A, lda, myInfo) &
       result(cpotf2) &
       bind(C, name="rocsolver_cpotf2")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cpotf2
    end function rocsolver_cpotf2

    !---------------------------------------------
    ! rocsolver_zpotf2
    !---------------------------------------------
    function rocsolver_zpotf2(handle, uplo, n, A, lda, myInfo) &
       result(zpotf2) &
       bind(C, name="rocsolver_zpotf2")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zpotf2
    end function rocsolver_zpotf2

    !---------------------------------------------
    ! rocsolver_spotf2_64
    !---------------------------------------------
    function rocsolver_spotf2_64(handle, uplo, n, A, lda, myInfo) &
       result(spotf2_64) &
       bind(C, name="rocsolver_spotf2_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: spotf2_64
    end function rocsolver_spotf2_64

    !---------------------------------------------
    ! rocsolver_dpotf2_64
    !---------------------------------------------
    function rocsolver_dpotf2_64(handle, uplo, n, A, lda, myInfo) &
       result(dpotf2_64) &
       bind(C, name="rocsolver_dpotf2_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dpotf2_64
    end function rocsolver_dpotf2_64

    !---------------------------------------------
    ! rocsolver_cpotf2_64
    !---------------------------------------------
    function rocsolver_cpotf2_64(handle, uplo, n, A, lda, myInfo) &
       result(cpotf2_64) &
       bind(C, name="rocsolver_cpotf2_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cpotf2_64
    end function rocsolver_cpotf2_64

    !---------------------------------------------
    ! rocsolver_zpotf2_64
    !---------------------------------------------
    function rocsolver_zpotf2_64(handle, uplo, n, A, lda, myInfo) &
       result(zpotf2_64) &
       bind(C, name="rocsolver_zpotf2_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zpotf2_64
    end function rocsolver_zpotf2_64

    !---------------------------------------------
    ! rocsolver_spotf2_batched
    !---------------------------------------------
    function rocsolver_spotf2_batched(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(spotf2_batched) &
       bind(C, name="rocsolver_spotf2_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: spotf2_batched
    end function rocsolver_spotf2_batched

    !---------------------------------------------
    ! rocsolver_dpotf2_batched
    !---------------------------------------------
    function rocsolver_dpotf2_batched(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(dpotf2_batched) &
       bind(C, name="rocsolver_dpotf2_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dpotf2_batched
    end function rocsolver_dpotf2_batched

    !---------------------------------------------
    ! rocsolver_cpotf2_batched
    !---------------------------------------------
    function rocsolver_cpotf2_batched(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(cpotf2_batched) &
       bind(C, name="rocsolver_cpotf2_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cpotf2_batched
    end function rocsolver_cpotf2_batched

    !---------------------------------------------
    ! rocsolver_zpotf2_batched
    !---------------------------------------------
    function rocsolver_zpotf2_batched(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(zpotf2_batched) &
       bind(C, name="rocsolver_zpotf2_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zpotf2_batched
    end function rocsolver_zpotf2_batched

    !---------------------------------------------
    ! rocsolver_spotf2_batched_64
    !---------------------------------------------
    function rocsolver_spotf2_batched_64(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(spotf2_batched_64) &
       bind(C, name="rocsolver_spotf2_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: spotf2_batched_64
    end function rocsolver_spotf2_batched_64

    !---------------------------------------------
    ! rocsolver_dpotf2_batched_64
    !---------------------------------------------
    function rocsolver_dpotf2_batched_64(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(dpotf2_batched_64) &
       bind(C, name="rocsolver_dpotf2_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dpotf2_batched_64
    end function rocsolver_dpotf2_batched_64

    !---------------------------------------------
    ! rocsolver_cpotf2_batched_64
    !---------------------------------------------
    function rocsolver_cpotf2_batched_64(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(cpotf2_batched_64) &
       bind(C, name="rocsolver_cpotf2_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cpotf2_batched_64
    end function rocsolver_cpotf2_batched_64

    !---------------------------------------------
    ! rocsolver_zpotf2_batched_64
    !---------------------------------------------
    function rocsolver_zpotf2_batched_64(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(zpotf2_batched_64) &
       bind(C, name="rocsolver_zpotf2_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zpotf2_batched_64
    end function rocsolver_zpotf2_batched_64

    !---------------------------------------------
    ! rocsolver_spotf2_strided_batched
    !---------------------------------------------
    function rocsolver_spotf2_strided_batched(handle, uplo, n, A, lda, strideA, myInfo, &
                                              batch_count) &
       result(spotf2_strided_batched) &
       bind(C, name="rocsolver_spotf2_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: spotf2_strided_batched
    end function rocsolver_spotf2_strided_batched

    !---------------------------------------------
    ! rocsolver_dpotf2_strided_batched
    !---------------------------------------------
    function rocsolver_dpotf2_strided_batched(handle, uplo, n, A, lda, strideA, myInfo, &
                                              batch_count) &
       result(dpotf2_strided_batched) &
       bind(C, name="rocsolver_dpotf2_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dpotf2_strided_batched
    end function rocsolver_dpotf2_strided_batched

    !---------------------------------------------
    ! rocsolver_cpotf2_strided_batched
    !---------------------------------------------
    function rocsolver_cpotf2_strided_batched(handle, uplo, n, A, lda, strideA, myInfo, &
                                              batch_count) &
       result(cpotf2_strided_batched) &
       bind(C, name="rocsolver_cpotf2_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cpotf2_strided_batched
    end function rocsolver_cpotf2_strided_batched

    !---------------------------------------------
    ! rocsolver_zpotf2_strided_batched
    !---------------------------------------------
    function rocsolver_zpotf2_strided_batched(handle, uplo, n, A, lda, strideA, myInfo, &
                                              batch_count) &
       result(zpotf2_strided_batched) &
       bind(C, name="rocsolver_zpotf2_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zpotf2_strided_batched
    end function rocsolver_zpotf2_strided_batched

    !---------------------------------------------
    ! rocsolver_spotf2_strided_batched_64
    !---------------------------------------------
    function rocsolver_spotf2_strided_batched_64(handle, uplo, n, A, lda, strideA, myInfo, &
                                                 batch_count) &
       result(spotf2_strided_batched_64) &
       bind(C, name="rocsolver_spotf2_strided_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: spotf2_strided_batched_64
    end function rocsolver_spotf2_strided_batched_64

    !---------------------------------------------
    ! rocsolver_dpotf2_strided_batched_64
    !---------------------------------------------
    function rocsolver_dpotf2_strided_batched_64(handle, uplo, n, A, lda, strideA, myInfo, &
                                                 batch_count) &
       result(dpotf2_strided_batched_64) &
       bind(C, name="rocsolver_dpotf2_strided_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dpotf2_strided_batched_64
    end function rocsolver_dpotf2_strided_batched_64

    !---------------------------------------------
    ! rocsolver_cpotf2_strided_batched_64
    !---------------------------------------------
    function rocsolver_cpotf2_strided_batched_64(handle, uplo, n, A, lda, strideA, myInfo, &
                                                 batch_count) &
       result(cpotf2_strided_batched_64) &
       bind(C, name="rocsolver_cpotf2_strided_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cpotf2_strided_batched_64
    end function rocsolver_cpotf2_strided_batched_64

    !---------------------------------------------
    ! rocsolver_zpotf2_strided_batched_64
    !---------------------------------------------
    function rocsolver_zpotf2_strided_batched_64(handle, uplo, n, A, lda, strideA, myInfo, &
                                                 batch_count) &
       result(zpotf2_strided_batched_64) &
       bind(C, name="rocsolver_zpotf2_strided_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zpotf2_strided_batched_64
    end function rocsolver_zpotf2_strided_batched_64

    !---------------------------------------------
    ! rocsolver_spotrf
    !---------------------------------------------
    function rocsolver_spotrf(handle, uplo, n, A, lda, myInfo) &
       result(spotrf) &
       bind(C, name="rocsolver_spotrf")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: spotrf
    end function rocsolver_spotrf

    !---------------------------------------------
    ! rocsolver_dpotrf
    !---------------------------------------------
    function rocsolver_dpotrf(handle, uplo, n, A, lda, myInfo) &
       result(dpotrf) &
       bind(C, name="rocsolver_dpotrf")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dpotrf
    end function rocsolver_dpotrf

    !---------------------------------------------
    ! rocsolver_cpotrf
    !---------------------------------------------
    function rocsolver_cpotrf(handle, uplo, n, A, lda, myInfo) &
       result(cpotrf) &
       bind(C, name="rocsolver_cpotrf")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cpotrf
    end function rocsolver_cpotrf

    !---------------------------------------------
    ! rocsolver_zpotrf
    !---------------------------------------------
    function rocsolver_zpotrf(handle, uplo, n, A, lda, myInfo) &
       result(zpotrf) &
       bind(C, name="rocsolver_zpotrf")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zpotrf
    end function rocsolver_zpotrf

    !---------------------------------------------
    ! rocsolver_spotrf_64
    !---------------------------------------------
    function rocsolver_spotrf_64(handle, uplo, n, A, lda, myInfo) &
       result(spotrf_64) &
       bind(C, name="rocsolver_spotrf_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: spotrf_64
    end function rocsolver_spotrf_64

    !---------------------------------------------
    ! rocsolver_dpotrf_64
    !---------------------------------------------
    function rocsolver_dpotrf_64(handle, uplo, n, A, lda, myInfo) &
       result(dpotrf_64) &
       bind(C, name="rocsolver_dpotrf_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dpotrf_64
    end function rocsolver_dpotrf_64

    !---------------------------------------------
    ! rocsolver_cpotrf_64
    !---------------------------------------------
    function rocsolver_cpotrf_64(handle, uplo, n, A, lda, myInfo) &
       result(cpotrf_64) &
       bind(C, name="rocsolver_cpotrf_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cpotrf_64
    end function rocsolver_cpotrf_64

    !---------------------------------------------
    ! rocsolver_zpotrf_64
    !---------------------------------------------
    function rocsolver_zpotrf_64(handle, uplo, n, A, lda, myInfo) &
       result(zpotrf_64) &
       bind(C, name="rocsolver_zpotrf_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zpotrf_64
    end function rocsolver_zpotrf_64

    !---------------------------------------------
    ! rocsolver_spotrf_batched
    !---------------------------------------------
    function rocsolver_spotrf_batched(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(spotrf_batched) &
       bind(C, name="rocsolver_spotrf_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: spotrf_batched
    end function rocsolver_spotrf_batched

    !---------------------------------------------
    ! rocsolver_dpotrf_batched
    !---------------------------------------------
    function rocsolver_dpotrf_batched(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(dpotrf_batched) &
       bind(C, name="rocsolver_dpotrf_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dpotrf_batched
    end function rocsolver_dpotrf_batched

    !---------------------------------------------
    ! rocsolver_cpotrf_batched
    !---------------------------------------------
    function rocsolver_cpotrf_batched(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(cpotrf_batched) &
       bind(C, name="rocsolver_cpotrf_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cpotrf_batched
    end function rocsolver_cpotrf_batched

    !---------------------------------------------
    ! rocsolver_zpotrf_batched
    !---------------------------------------------
    function rocsolver_zpotrf_batched(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(zpotrf_batched) &
       bind(C, name="rocsolver_zpotrf_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zpotrf_batched
    end function rocsolver_zpotrf_batched

    !---------------------------------------------
    ! rocsolver_spotrf_batched_64
    !---------------------------------------------
    function rocsolver_spotrf_batched_64(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(spotrf_batched_64) &
       bind(C, name="rocsolver_spotrf_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: spotrf_batched_64
    end function rocsolver_spotrf_batched_64

    !---------------------------------------------
    ! rocsolver_dpotrf_batched_64
    !---------------------------------------------
    function rocsolver_dpotrf_batched_64(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(dpotrf_batched_64) &
       bind(C, name="rocsolver_dpotrf_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dpotrf_batched_64
    end function rocsolver_dpotrf_batched_64

    !---------------------------------------------
    ! rocsolver_cpotrf_batched_64
    !---------------------------------------------
    function rocsolver_cpotrf_batched_64(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(cpotrf_batched_64) &
       bind(C, name="rocsolver_cpotrf_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cpotrf_batched_64
    end function rocsolver_cpotrf_batched_64

    !---------------------------------------------
    ! rocsolver_zpotrf_batched_64
    !---------------------------------------------
    function rocsolver_zpotrf_batched_64(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(zpotrf_batched_64) &
       bind(C, name="rocsolver_zpotrf_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zpotrf_batched_64
    end function rocsolver_zpotrf_batched_64

    !---------------------------------------------
    ! rocsolver_spotrf_strided_batched
    !---------------------------------------------
    function rocsolver_spotrf_strided_batched(handle, uplo, n, A, lda, strideA, myInfo, &
                                              batch_count) &
       result(spotrf_strided_batched) &
       bind(C, name="rocsolver_spotrf_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: spotrf_strided_batched
    end function rocsolver_spotrf_strided_batched

    !---------------------------------------------
    ! rocsolver_dpotrf_strided_batched
    !---------------------------------------------
    function rocsolver_dpotrf_strided_batched(handle, uplo, n, A, lda, strideA, myInfo, &
                                              batch_count) &
       result(dpotrf_strided_batched) &
       bind(C, name="rocsolver_dpotrf_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dpotrf_strided_batched
    end function rocsolver_dpotrf_strided_batched

    !---------------------------------------------
    ! rocsolver_cpotrf_strided_batched
    !---------------------------------------------
    function rocsolver_cpotrf_strided_batched(handle, uplo, n, A, lda, strideA, myInfo, &
                                              batch_count) &
       result(cpotrf_strided_batched) &
       bind(C, name="rocsolver_cpotrf_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cpotrf_strided_batched
    end function rocsolver_cpotrf_strided_batched

    !---------------------------------------------
    ! rocsolver_zpotrf_strided_batched
    !---------------------------------------------
    function rocsolver_zpotrf_strided_batched(handle, uplo, n, A, lda, strideA, myInfo, &
                                              batch_count) &
       result(zpotrf_strided_batched) &
       bind(C, name="rocsolver_zpotrf_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zpotrf_strided_batched
    end function rocsolver_zpotrf_strided_batched

    !---------------------------------------------
    ! rocsolver_spotrf_strided_batched_64
    !---------------------------------------------
    function rocsolver_spotrf_strided_batched_64(handle, uplo, n, A, lda, strideA, myInfo, &
                                                 batch_count) &
       result(spotrf_strided_batched_64) &
       bind(C, name="rocsolver_spotrf_strided_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: spotrf_strided_batched_64
    end function rocsolver_spotrf_strided_batched_64

    !---------------------------------------------
    ! rocsolver_dpotrf_strided_batched_64
    !---------------------------------------------
    function rocsolver_dpotrf_strided_batched_64(handle, uplo, n, A, lda, strideA, myInfo, &
                                                 batch_count) &
       result(dpotrf_strided_batched_64) &
       bind(C, name="rocsolver_dpotrf_strided_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dpotrf_strided_batched_64
    end function rocsolver_dpotrf_strided_batched_64

    !---------------------------------------------
    ! rocsolver_cpotrf_strided_batched_64
    !---------------------------------------------
    function rocsolver_cpotrf_strided_batched_64(handle, uplo, n, A, lda, strideA, myInfo, &
                                                 batch_count) &
       result(cpotrf_strided_batched_64) &
       bind(C, name="rocsolver_cpotrf_strided_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cpotrf_strided_batched_64
    end function rocsolver_cpotrf_strided_batched_64

    !---------------------------------------------
    ! rocsolver_zpotrf_strided_batched_64
    !---------------------------------------------
    function rocsolver_zpotrf_strided_batched_64(handle, uplo, n, A, lda, strideA, myInfo, &
                                                 batch_count) &
       result(zpotrf_strided_batched_64) &
       bind(C, name="rocsolver_zpotrf_strided_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zpotrf_strided_batched_64
    end function rocsolver_zpotrf_strided_batched_64

    !---------------------------------------------
    ! rocsolver_spotrs
    !---------------------------------------------
    function rocsolver_spotrs(handle, uplo, n, nrhs, A, lda, B, ldb) &
       result(spotrs) &
       bind(C, name="rocsolver_spotrs")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: spotrs
    end function rocsolver_spotrs

    !---------------------------------------------
    ! rocsolver_dpotrs
    !---------------------------------------------
    function rocsolver_dpotrs(handle, uplo, n, nrhs, A, lda, B, ldb) &
       result(dpotrs) &
       bind(C, name="rocsolver_dpotrs")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: dpotrs
    end function rocsolver_dpotrs

    !---------------------------------------------
    ! rocsolver_cpotrs
    !---------------------------------------------
    function rocsolver_cpotrs(handle, uplo, n, nrhs, A, lda, B, ldb) &
       result(cpotrs) &
       bind(C, name="rocsolver_cpotrs")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: cpotrs
    end function rocsolver_cpotrs

    !---------------------------------------------
    ! rocsolver_zpotrs
    !---------------------------------------------
    function rocsolver_zpotrs(handle, uplo, n, nrhs, A, lda, B, ldb) &
       result(zpotrs) &
       bind(C, name="rocsolver_zpotrs")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: zpotrs
    end function rocsolver_zpotrs

    !---------------------------------------------
    ! rocsolver_spotrs_64
    !---------------------------------------------
    function rocsolver_spotrs_64(handle, uplo, n, nrhs, A, lda, B, ldb) &
       result(spotrs_64) &
       bind(C, name="rocsolver_spotrs_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(kind(rocblas_status_success)) :: spotrs_64
    end function rocsolver_spotrs_64

    !---------------------------------------------
    ! rocsolver_dpotrs_64
    !---------------------------------------------
    function rocsolver_dpotrs_64(handle, uplo, n, nrhs, A, lda, B, ldb) &
       result(dpotrs_64) &
       bind(C, name="rocsolver_dpotrs_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(kind(rocblas_status_success)) :: dpotrs_64
    end function rocsolver_dpotrs_64

    !---------------------------------------------
    ! rocsolver_cpotrs_64
    !---------------------------------------------
    function rocsolver_cpotrs_64(handle, uplo, n, nrhs, A, lda, B, ldb) &
       result(cpotrs_64) &
       bind(C, name="rocsolver_cpotrs_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(kind(rocblas_status_success)) :: cpotrs_64
    end function rocsolver_cpotrs_64

    !---------------------------------------------
    ! rocsolver_zpotrs_64
    !---------------------------------------------
    function rocsolver_zpotrs_64(handle, uplo, n, nrhs, A, lda, B, ldb) &
       result(zpotrs_64) &
       bind(C, name="rocsolver_zpotrs_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(kind(rocblas_status_success)) :: zpotrs_64
    end function rocsolver_zpotrs_64

    !---------------------------------------------
    ! rocsolver_spotrs_batched
    !---------------------------------------------
    function rocsolver_spotrs_batched(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count) &
       result(spotrs_batched) &
       bind(C, name="rocsolver_spotrs_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: spotrs_batched
    end function rocsolver_spotrs_batched

    !---------------------------------------------
    ! rocsolver_dpotrs_batched
    !---------------------------------------------
    function rocsolver_dpotrs_batched(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count) &
       result(dpotrs_batched) &
       bind(C, name="rocsolver_dpotrs_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dpotrs_batched
    end function rocsolver_dpotrs_batched

    !---------------------------------------------
    ! rocsolver_cpotrs_batched
    !---------------------------------------------
    function rocsolver_cpotrs_batched(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count) &
       result(cpotrs_batched) &
       bind(C, name="rocsolver_cpotrs_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cpotrs_batched
    end function rocsolver_cpotrs_batched

    !---------------------------------------------
    ! rocsolver_zpotrs_batched
    !---------------------------------------------
    function rocsolver_zpotrs_batched(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count) &
       result(zpotrs_batched) &
       bind(C, name="rocsolver_zpotrs_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zpotrs_batched
    end function rocsolver_zpotrs_batched

    !---------------------------------------------
    ! rocsolver_spotrs_batched_64
    !---------------------------------------------
    function rocsolver_spotrs_batched_64(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count) &
       result(spotrs_batched_64) &
       bind(C, name="rocsolver_spotrs_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: spotrs_batched_64
    end function rocsolver_spotrs_batched_64

    !---------------------------------------------
    ! rocsolver_dpotrs_batched_64
    !---------------------------------------------
    function rocsolver_dpotrs_batched_64(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count) &
       result(dpotrs_batched_64) &
       bind(C, name="rocsolver_dpotrs_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dpotrs_batched_64
    end function rocsolver_dpotrs_batched_64

    !---------------------------------------------
    ! rocsolver_cpotrs_batched_64
    !---------------------------------------------
    function rocsolver_cpotrs_batched_64(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count) &
       result(cpotrs_batched_64) &
       bind(C, name="rocsolver_cpotrs_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cpotrs_batched_64
    end function rocsolver_cpotrs_batched_64

    !---------------------------------------------
    ! rocsolver_zpotrs_batched_64
    !---------------------------------------------
    function rocsolver_zpotrs_batched_64(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count) &
       result(zpotrs_batched_64) &
       bind(C, name="rocsolver_zpotrs_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zpotrs_batched_64
    end function rocsolver_zpotrs_batched_64

    !---------------------------------------------
    ! rocsolver_spotrs_strided_batched
    !---------------------------------------------
    function rocsolver_spotrs_strided_batched(handle, uplo, n, nrhs, A, lda, strideA, B, ldb, &
                                              strideB, batch_count) &
       result(spotrs_strided_batched) &
       bind(C, name="rocsolver_spotrs_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: spotrs_strided_batched
    end function rocsolver_spotrs_strided_batched

    !---------------------------------------------
    ! rocsolver_dpotrs_strided_batched
    !---------------------------------------------
    function rocsolver_dpotrs_strided_batched(handle, uplo, n, nrhs, A, lda, strideA, B, ldb, &
                                              strideB, batch_count) &
       result(dpotrs_strided_batched) &
       bind(C, name="rocsolver_dpotrs_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dpotrs_strided_batched
    end function rocsolver_dpotrs_strided_batched

    !---------------------------------------------
    ! rocsolver_cpotrs_strided_batched
    !---------------------------------------------
    function rocsolver_cpotrs_strided_batched(handle, uplo, n, nrhs, A, lda, strideA, B, ldb, &
                                              strideB, batch_count) &
       result(cpotrs_strided_batched) &
       bind(C, name="rocsolver_cpotrs_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cpotrs_strided_batched
    end function rocsolver_cpotrs_strided_batched

    !---------------------------------------------
    ! rocsolver_zpotrs_strided_batched
    !---------------------------------------------
    function rocsolver_zpotrs_strided_batched(handle, uplo, n, nrhs, A, lda, strideA, B, ldb, &
                                              strideB, batch_count) &
       result(zpotrs_strided_batched) &
       bind(C, name="rocsolver_zpotrs_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zpotrs_strided_batched
    end function rocsolver_zpotrs_strided_batched

    !---------------------------------------------
    ! rocsolver_spotrs_strided_batched_64
    !---------------------------------------------
    function rocsolver_spotrs_strided_batched_64(handle, uplo, n, nrhs, A, lda, strideA, B, ldb, &
                                                 strideB, batch_count) &
       result(spotrs_strided_batched_64) &
       bind(C, name="rocsolver_spotrs_strided_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: spotrs_strided_batched_64
    end function rocsolver_spotrs_strided_batched_64

    !---------------------------------------------
    ! rocsolver_dpotrs_strided_batched_64
    !---------------------------------------------
    function rocsolver_dpotrs_strided_batched_64(handle, uplo, n, nrhs, A, lda, strideA, B, ldb, &
                                                 strideB, batch_count) &
       result(dpotrs_strided_batched_64) &
       bind(C, name="rocsolver_dpotrs_strided_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dpotrs_strided_batched_64
    end function rocsolver_dpotrs_strided_batched_64

    !---------------------------------------------
    ! rocsolver_cpotrs_strided_batched_64
    !---------------------------------------------
    function rocsolver_cpotrs_strided_batched_64(handle, uplo, n, nrhs, A, lda, strideA, B, ldb, &
                                                 strideB, batch_count) &
       result(cpotrs_strided_batched_64) &
       bind(C, name="rocsolver_cpotrs_strided_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cpotrs_strided_batched_64
    end function rocsolver_cpotrs_strided_batched_64

    !---------------------------------------------
    ! rocsolver_zpotrs_strided_batched_64
    !---------------------------------------------
    function rocsolver_zpotrs_strided_batched_64(handle, uplo, n, nrhs, A, lda, strideA, B, ldb, &
                                                 strideB, batch_count) &
       result(zpotrs_strided_batched_64) &
       bind(C, name="rocsolver_zpotrs_strided_batched_64")
       import :: c_ptr, rocblas_fill_upper, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zpotrs_strided_batched_64
    end function rocsolver_zpotrs_strided_batched_64

    !---------------------------------------------
    ! rocsolver_sposv
    !---------------------------------------------
    function rocsolver_sposv(handle, uplo, n, nrhs, A, lda, B, ldb, myInfo) &
       result(sposv) &
       bind(C, name="rocsolver_sposv")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sposv
    end function rocsolver_sposv

    !---------------------------------------------
    ! rocsolver_dposv
    !---------------------------------------------
    function rocsolver_dposv(handle, uplo, n, nrhs, A, lda, B, ldb, myInfo) &
       result(dposv) &
       bind(C, name="rocsolver_dposv")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dposv
    end function rocsolver_dposv

    !---------------------------------------------
    ! rocsolver_cposv
    !---------------------------------------------
    function rocsolver_cposv(handle, uplo, n, nrhs, A, lda, B, ldb, myInfo) &
       result(cposv) &
       bind(C, name="rocsolver_cposv")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cposv
    end function rocsolver_cposv

    !---------------------------------------------
    ! rocsolver_zposv
    !---------------------------------------------
    function rocsolver_zposv(handle, uplo, n, nrhs, A, lda, B, ldb, myInfo) &
       result(zposv) &
       bind(C, name="rocsolver_zposv")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zposv
    end function rocsolver_zposv

    !---------------------------------------------
    ! rocsolver_sposv_batched
    !---------------------------------------------
    function rocsolver_sposv_batched(handle, uplo, n, nrhs, A, lda, B, ldb, myInfo, batch_count) &
       result(sposv_batched) &
       bind(C, name="rocsolver_sposv_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sposv_batched
    end function rocsolver_sposv_batched

    !---------------------------------------------
    ! rocsolver_dposv_batched
    !---------------------------------------------
    function rocsolver_dposv_batched(handle, uplo, n, nrhs, A, lda, B, ldb, myInfo, batch_count) &
       result(dposv_batched) &
       bind(C, name="rocsolver_dposv_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dposv_batched
    end function rocsolver_dposv_batched

    !---------------------------------------------
    ! rocsolver_cposv_batched
    !---------------------------------------------
    function rocsolver_cposv_batched(handle, uplo, n, nrhs, A, lda, B, ldb, myInfo, batch_count) &
       result(cposv_batched) &
       bind(C, name="rocsolver_cposv_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cposv_batched
    end function rocsolver_cposv_batched

    !---------------------------------------------
    ! rocsolver_zposv_batched
    !---------------------------------------------
    function rocsolver_zposv_batched(handle, uplo, n, nrhs, A, lda, B, ldb, myInfo, batch_count) &
       result(zposv_batched) &
       bind(C, name="rocsolver_zposv_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zposv_batched
    end function rocsolver_zposv_batched

    !---------------------------------------------
    ! rocsolver_sposv_strided_batched
    !---------------------------------------------
    function rocsolver_sposv_strided_batched(handle, uplo, n, nrhs, A, lda, strideA, B, ldb, &
                                             strideB, myInfo, batch_count) &
       result(sposv_strided_batched) &
       bind(C, name="rocsolver_sposv_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sposv_strided_batched
    end function rocsolver_sposv_strided_batched

    !---------------------------------------------
    ! rocsolver_dposv_strided_batched
    !---------------------------------------------
    function rocsolver_dposv_strided_batched(handle, uplo, n, nrhs, A, lda, strideA, B, ldb, &
                                             strideB, myInfo, batch_count) &
       result(dposv_strided_batched) &
       bind(C, name="rocsolver_dposv_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dposv_strided_batched
    end function rocsolver_dposv_strided_batched

    !---------------------------------------------
    ! rocsolver_cposv_strided_batched
    !---------------------------------------------
    function rocsolver_cposv_strided_batched(handle, uplo, n, nrhs, A, lda, strideA, B, ldb, &
                                             strideB, myInfo, batch_count) &
       result(cposv_strided_batched) &
       bind(C, name="rocsolver_cposv_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cposv_strided_batched
    end function rocsolver_cposv_strided_batched

    !---------------------------------------------
    ! rocsolver_zposv_strided_batched
    !---------------------------------------------
    function rocsolver_zposv_strided_batched(handle, uplo, n, nrhs, A, lda, strideA, B, ldb, &
                                             strideB, myInfo, batch_count) &
       result(zposv_strided_batched) &
       bind(C, name="rocsolver_zposv_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zposv_strided_batched
    end function rocsolver_zposv_strided_batched

    !---------------------------------------------
    ! rocsolver_spotri
    !---------------------------------------------
    function rocsolver_spotri(handle, uplo, n, A, lda, myInfo) &
       result(spotri) &
       bind(C, name="rocsolver_spotri")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: spotri
    end function rocsolver_spotri

    !---------------------------------------------
    ! rocsolver_dpotri
    !---------------------------------------------
    function rocsolver_dpotri(handle, uplo, n, A, lda, myInfo) &
       result(dpotri) &
       bind(C, name="rocsolver_dpotri")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dpotri
    end function rocsolver_dpotri

    !---------------------------------------------
    ! rocsolver_cpotri
    !---------------------------------------------
    function rocsolver_cpotri(handle, uplo, n, A, lda, myInfo) &
       result(cpotri) &
       bind(C, name="rocsolver_cpotri")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cpotri
    end function rocsolver_cpotri

    !---------------------------------------------
    ! rocsolver_zpotri
    !---------------------------------------------
    function rocsolver_zpotri(handle, uplo, n, A, lda, myInfo) &
       result(zpotri) &
       bind(C, name="rocsolver_zpotri")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zpotri
    end function rocsolver_zpotri

    !---------------------------------------------
    ! rocsolver_spotri_batched
    !---------------------------------------------
    function rocsolver_spotri_batched(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(spotri_batched) &
       bind(C, name="rocsolver_spotri_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: spotri_batched
    end function rocsolver_spotri_batched

    !---------------------------------------------
    ! rocsolver_dpotri_batched
    !---------------------------------------------
    function rocsolver_dpotri_batched(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(dpotri_batched) &
       bind(C, name="rocsolver_dpotri_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dpotri_batched
    end function rocsolver_dpotri_batched

    !---------------------------------------------
    ! rocsolver_cpotri_batched
    !---------------------------------------------
    function rocsolver_cpotri_batched(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(cpotri_batched) &
       bind(C, name="rocsolver_cpotri_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cpotri_batched
    end function rocsolver_cpotri_batched

    !---------------------------------------------
    ! rocsolver_zpotri_batched
    !---------------------------------------------
    function rocsolver_zpotri_batched(handle, uplo, n, A, lda, myInfo, batch_count) &
       result(zpotri_batched) &
       bind(C, name="rocsolver_zpotri_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zpotri_batched
    end function rocsolver_zpotri_batched

    !---------------------------------------------
    ! rocsolver_spotri_strided_batched
    !---------------------------------------------
    function rocsolver_spotri_strided_batched(handle, uplo, n, A, lda, strideA, myInfo, &
                                              batch_count) &
       result(spotri_strided_batched) &
       bind(C, name="rocsolver_spotri_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: spotri_strided_batched
    end function rocsolver_spotri_strided_batched

    !---------------------------------------------
    ! rocsolver_dpotri_strided_batched
    !---------------------------------------------
    function rocsolver_dpotri_strided_batched(handle, uplo, n, A, lda, strideA, myInfo, &
                                              batch_count) &
       result(dpotri_strided_batched) &
       bind(C, name="rocsolver_dpotri_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dpotri_strided_batched
    end function rocsolver_dpotri_strided_batched

    !---------------------------------------------
    ! rocsolver_cpotri_strided_batched
    !---------------------------------------------
    function rocsolver_cpotri_strided_batched(handle, uplo, n, A, lda, strideA, myInfo, &
                                              batch_count) &
       result(cpotri_strided_batched) &
       bind(C, name="rocsolver_cpotri_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cpotri_strided_batched
    end function rocsolver_cpotri_strided_batched

    !---------------------------------------------
    ! rocsolver_zpotri_strided_batched
    !---------------------------------------------
    function rocsolver_zpotri_strided_batched(handle, uplo, n, A, lda, strideA, myInfo, &
                                              batch_count) &
       result(zpotri_strided_batched) &
       bind(C, name="rocsolver_zpotri_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zpotri_strided_batched
    end function rocsolver_zpotri_strided_batched

    !---------------------------------------------
    ! rocsolver_sgesvd
    !---------------------------------------------
    function rocsolver_sgesvd(handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv, E, &
                              fast_alg, myInfo) &
       result(sgesvd) &
       bind(C, name="rocsolver_sgesvd")
       import :: c_ptr, rocblas_svect_all, c_int, rocblas_outofplace, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: E
       integer(kind(rocblas_outofplace)), value :: fast_alg
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgesvd
    end function rocsolver_sgesvd

    !---------------------------------------------
    ! rocsolver_dgesvd
    !---------------------------------------------
    function rocsolver_dgesvd(handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv, E, &
                              fast_alg, myInfo) &
       result(dgesvd) &
       bind(C, name="rocsolver_dgesvd")
       import :: c_ptr, rocblas_svect_all, c_int, rocblas_outofplace, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: E
       integer(kind(rocblas_outofplace)), value :: fast_alg
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgesvd
    end function rocsolver_dgesvd

    !---------------------------------------------
    ! rocsolver_cgesvd
    !---------------------------------------------
    function rocsolver_cgesvd(handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv, E, &
                              fast_alg, myInfo) &
       result(cgesvd) &
       bind(C, name="rocsolver_cgesvd")
       import :: c_ptr, rocblas_svect_all, c_int, rocblas_outofplace, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: E
       integer(kind(rocblas_outofplace)), value :: fast_alg
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgesvd
    end function rocsolver_cgesvd

    !---------------------------------------------
    ! rocsolver_zgesvd
    !---------------------------------------------
    function rocsolver_zgesvd(handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv, E, &
                              fast_alg, myInfo) &
       result(zgesvd) &
       bind(C, name="rocsolver_zgesvd")
       import :: c_ptr, rocblas_svect_all, c_int, rocblas_outofplace, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: E
       integer(kind(rocblas_outofplace)), value :: fast_alg
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgesvd
    end function rocsolver_zgesvd

    !---------------------------------------------
    ! rocsolver_sgesvd_batched
    !---------------------------------------------
    function rocsolver_sgesvd_batched(handle, left_svect, right_svect, m, n, A, lda, S, strideS, &
                                      U, ldu, strideU, V, ldv, strideV, E, strideE, fast_alg, &
                                      myInfo, batch_count) &
       result(sgesvd_batched) &
       bind(C, name="rocsolver_sgesvd_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, rocblas_outofplace, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       integer(kind(rocblas_outofplace)), value :: fast_alg
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgesvd_batched
    end function rocsolver_sgesvd_batched

    !---------------------------------------------
    ! rocsolver_dgesvd_batched
    !---------------------------------------------
    function rocsolver_dgesvd_batched(handle, left_svect, right_svect, m, n, A, lda, S, strideS, &
                                      U, ldu, strideU, V, ldv, strideV, E, strideE, fast_alg, &
                                      myInfo, batch_count) &
       result(dgesvd_batched) &
       bind(C, name="rocsolver_dgesvd_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, rocblas_outofplace, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       integer(kind(rocblas_outofplace)), value :: fast_alg
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgesvd_batched
    end function rocsolver_dgesvd_batched

    !---------------------------------------------
    ! rocsolver_cgesvd_batched
    !---------------------------------------------
    function rocsolver_cgesvd_batched(handle, left_svect, right_svect, m, n, A, lda, S, strideS, &
                                      U, ldu, strideU, V, ldv, strideV, E, strideE, fast_alg, &
                                      myInfo, batch_count) &
       result(cgesvd_batched) &
       bind(C, name="rocsolver_cgesvd_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, rocblas_outofplace, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       integer(kind(rocblas_outofplace)), value :: fast_alg
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgesvd_batched
    end function rocsolver_cgesvd_batched

    !---------------------------------------------
    ! rocsolver_zgesvd_batched
    !---------------------------------------------
    function rocsolver_zgesvd_batched(handle, left_svect, right_svect, m, n, A, lda, S, strideS, &
                                      U, ldu, strideU, V, ldv, strideV, E, strideE, fast_alg, &
                                      myInfo, batch_count) &
       result(zgesvd_batched) &
       bind(C, name="rocsolver_zgesvd_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, rocblas_outofplace, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       integer(kind(rocblas_outofplace)), value :: fast_alg
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgesvd_batched
    end function rocsolver_zgesvd_batched

    !---------------------------------------------
    ! rocsolver_sgesvd_strided_batched
    !---------------------------------------------
    function rocsolver_sgesvd_strided_batched(handle, left_svect, right_svect, m, n, A, lda, &
                                              strideA, S, strideS, U, ldu, strideU, V, ldv, &
                                              strideV, E, strideE, fast_alg, myInfo, batch_count) &
       result(sgesvd_strided_batched) &
       bind(C, name="rocsolver_sgesvd_strided_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, rocblas_outofplace, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       integer(kind(rocblas_outofplace)), value :: fast_alg
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgesvd_strided_batched
    end function rocsolver_sgesvd_strided_batched

    !---------------------------------------------
    ! rocsolver_dgesvd_strided_batched
    !---------------------------------------------
    function rocsolver_dgesvd_strided_batched(handle, left_svect, right_svect, m, n, A, lda, &
                                              strideA, S, strideS, U, ldu, strideU, V, ldv, &
                                              strideV, E, strideE, fast_alg, myInfo, batch_count) &
       result(dgesvd_strided_batched) &
       bind(C, name="rocsolver_dgesvd_strided_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, rocblas_outofplace, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       integer(kind(rocblas_outofplace)), value :: fast_alg
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgesvd_strided_batched
    end function rocsolver_dgesvd_strided_batched

    !---------------------------------------------
    ! rocsolver_cgesvd_strided_batched
    !---------------------------------------------
    function rocsolver_cgesvd_strided_batched(handle, left_svect, right_svect, m, n, A, lda, &
                                              strideA, S, strideS, U, ldu, strideU, V, ldv, &
                                              strideV, E, strideE, fast_alg, myInfo, batch_count) &
       result(cgesvd_strided_batched) &
       bind(C, name="rocsolver_cgesvd_strided_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, rocblas_outofplace, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       integer(kind(rocblas_outofplace)), value :: fast_alg
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgesvd_strided_batched
    end function rocsolver_cgesvd_strided_batched

    !---------------------------------------------
    ! rocsolver_zgesvd_strided_batched
    !---------------------------------------------
    function rocsolver_zgesvd_strided_batched(handle, left_svect, right_svect, m, n, A, lda, &
                                              strideA, S, strideS, U, ldu, strideU, V, ldv, &
                                              strideV, E, strideE, fast_alg, myInfo, batch_count) &
       result(zgesvd_strided_batched) &
       bind(C, name="rocsolver_zgesvd_strided_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, rocblas_outofplace, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       integer(kind(rocblas_outofplace)), value :: fast_alg
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgesvd_strided_batched
    end function rocsolver_zgesvd_strided_batched

    !---------------------------------------------
    ! rocsolver_sgesdd
    !---------------------------------------------
    function rocsolver_sgesdd(handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv, &
                              myInfo) &
       result(sgesdd) &
       bind(C, name="rocsolver_sgesdd")
       import :: c_ptr, rocblas_svect_all, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgesdd
    end function rocsolver_sgesdd

    !---------------------------------------------
    ! rocsolver_dgesdd
    !---------------------------------------------
    function rocsolver_dgesdd(handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv, &
                              myInfo) &
       result(dgesdd) &
       bind(C, name="rocsolver_dgesdd")
       import :: c_ptr, rocblas_svect_all, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgesdd
    end function rocsolver_dgesdd

    !---------------------------------------------
    ! rocsolver_cgesdd
    !---------------------------------------------
    function rocsolver_cgesdd(handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv, &
                              myInfo) &
       result(cgesdd) &
       bind(C, name="rocsolver_cgesdd")
       import :: c_ptr, rocblas_svect_all, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgesdd
    end function rocsolver_cgesdd

    !---------------------------------------------
    ! rocsolver_zgesdd
    !---------------------------------------------
    function rocsolver_zgesdd(handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv, &
                              myInfo) &
       result(zgesdd) &
       bind(C, name="rocsolver_zgesdd")
       import :: c_ptr, rocblas_svect_all, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgesdd
    end function rocsolver_zgesdd

    !---------------------------------------------
    ! rocsolver_sgesdd_batched
    !---------------------------------------------
    function rocsolver_sgesdd_batched(handle, left_svect, right_svect, m, n, A, lda, S, strideS, &
                                      U, ldu, strideU, V, ldv, strideV, myInfo, batch_count) &
       result(sgesdd_batched) &
       bind(C, name="rocsolver_sgesdd_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgesdd_batched
    end function rocsolver_sgesdd_batched

    !---------------------------------------------
    ! rocsolver_dgesdd_batched
    !---------------------------------------------
    function rocsolver_dgesdd_batched(handle, left_svect, right_svect, m, n, A, lda, S, strideS, &
                                      U, ldu, strideU, V, ldv, strideV, myInfo, batch_count) &
       result(dgesdd_batched) &
       bind(C, name="rocsolver_dgesdd_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgesdd_batched
    end function rocsolver_dgesdd_batched

    !---------------------------------------------
    ! rocsolver_cgesdd_batched
    !---------------------------------------------
    function rocsolver_cgesdd_batched(handle, left_svect, right_svect, m, n, A, lda, S, strideS, &
                                      U, ldu, strideU, V, ldv, strideV, myInfo, batch_count) &
       result(cgesdd_batched) &
       bind(C, name="rocsolver_cgesdd_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgesdd_batched
    end function rocsolver_cgesdd_batched

    !---------------------------------------------
    ! rocsolver_zgesdd_batched
    !---------------------------------------------
    function rocsolver_zgesdd_batched(handle, left_svect, right_svect, m, n, A, lda, S, strideS, &
                                      U, ldu, strideU, V, ldv, strideV, myInfo, batch_count) &
       result(zgesdd_batched) &
       bind(C, name="rocsolver_zgesdd_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgesdd_batched
    end function rocsolver_zgesdd_batched

    !---------------------------------------------
    ! rocsolver_sgesdd_strided_batched
    !---------------------------------------------
    function rocsolver_sgesdd_strided_batched(handle, left_svect, right_svect, m, n, A, lda, &
                                              strideA, S, strideS, U, ldu, strideU, V, ldv, &
                                              strideV, myInfo, batch_count) &
       result(sgesdd_strided_batched) &
       bind(C, name="rocsolver_sgesdd_strided_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgesdd_strided_batched
    end function rocsolver_sgesdd_strided_batched

    !---------------------------------------------
    ! rocsolver_dgesdd_strided_batched
    !---------------------------------------------
    function rocsolver_dgesdd_strided_batched(handle, left_svect, right_svect, m, n, A, lda, &
                                              strideA, S, strideS, U, ldu, strideU, V, ldv, &
                                              strideV, myInfo, batch_count) &
       result(dgesdd_strided_batched) &
       bind(C, name="rocsolver_dgesdd_strided_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgesdd_strided_batched
    end function rocsolver_dgesdd_strided_batched

    !---------------------------------------------
    ! rocsolver_cgesdd_strided_batched
    !---------------------------------------------
    function rocsolver_cgesdd_strided_batched(handle, left_svect, right_svect, m, n, A, lda, &
                                              strideA, S, strideS, U, ldu, strideU, V, ldv, &
                                              strideV, myInfo, batch_count) &
       result(cgesdd_strided_batched) &
       bind(C, name="rocsolver_cgesdd_strided_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgesdd_strided_batched
    end function rocsolver_cgesdd_strided_batched

    !---------------------------------------------
    ! rocsolver_zgesdd_strided_batched
    !---------------------------------------------
    function rocsolver_zgesdd_strided_batched(handle, left_svect, right_svect, m, n, A, lda, &
                                              strideA, S, strideS, U, ldu, strideU, V, ldv, &
                                              strideV, myInfo, batch_count) &
       result(zgesdd_strided_batched) &
       bind(C, name="rocsolver_zgesdd_strided_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgesdd_strided_batched
    end function rocsolver_zgesdd_strided_batched

    !---------------------------------------------
    ! rocsolver_sgesvdj
    !---------------------------------------------
    function rocsolver_sgesvdj(handle, left_svect, right_svect, m, n, A, lda, abstol, residual, &
                               max_sweeps, n_sweeps, S, U, ldu, V, ldv, myInfo) &
       result(sgesvdj) &
       bind(C, name="rocsolver_sgesvdj")
       import :: c_ptr, rocblas_svect_all, c_int, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgesvdj
    end function rocsolver_sgesvdj

    !---------------------------------------------
    ! rocsolver_dgesvdj
    !---------------------------------------------
    function rocsolver_dgesvdj(handle, left_svect, right_svect, m, n, A, lda, abstol, residual, &
                               max_sweeps, n_sweeps, S, U, ldu, V, ldv, myInfo) &
       result(dgesvdj) &
       bind(C, name="rocsolver_dgesvdj")
       import :: c_ptr, rocblas_svect_all, c_int, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgesvdj
    end function rocsolver_dgesvdj

    !---------------------------------------------
    ! rocsolver_cgesvdj
    !---------------------------------------------
    function rocsolver_cgesvdj(handle, left_svect, right_svect, m, n, A, lda, abstol, residual, &
                               max_sweeps, n_sweeps, S, U, ldu, V, ldv, myInfo) &
       result(cgesvdj) &
       bind(C, name="rocsolver_cgesvdj")
       import :: c_ptr, rocblas_svect_all, c_int, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgesvdj
    end function rocsolver_cgesvdj

    !---------------------------------------------
    ! rocsolver_zgesvdj
    !---------------------------------------------
    function rocsolver_zgesvdj(handle, left_svect, right_svect, m, n, A, lda, abstol, residual, &
                               max_sweeps, n_sweeps, S, U, ldu, V, ldv, myInfo) &
       result(zgesvdj) &
       bind(C, name="rocsolver_zgesvdj")
       import :: c_ptr, rocblas_svect_all, c_int, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgesvdj
    end function rocsolver_zgesvdj

    !---------------------------------------------
    ! rocsolver_sgesvdj_batched
    !---------------------------------------------
    function rocsolver_sgesvdj_batched(handle, left_svect, right_svect, m, n, A, lda, abstol, &
                                       residual, max_sweeps, n_sweeps, S, strideS, U, ldu, &
                                       strideU, V, ldv, strideV, myInfo, batch_count) &
       result(sgesvdj_batched) &
       bind(C, name="rocsolver_sgesvdj_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_float, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgesvdj_batched
    end function rocsolver_sgesvdj_batched

    !---------------------------------------------
    ! rocsolver_dgesvdj_batched
    !---------------------------------------------
    function rocsolver_dgesvdj_batched(handle, left_svect, right_svect, m, n, A, lda, abstol, &
                                       residual, max_sweeps, n_sweeps, S, strideS, U, ldu, &
                                       strideU, V, ldv, strideV, myInfo, batch_count) &
       result(dgesvdj_batched) &
       bind(C, name="rocsolver_dgesvdj_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_double, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgesvdj_batched
    end function rocsolver_dgesvdj_batched

    !---------------------------------------------
    ! rocsolver_cgesvdj_batched
    !---------------------------------------------
    function rocsolver_cgesvdj_batched(handle, left_svect, right_svect, m, n, A, lda, abstol, &
                                       residual, max_sweeps, n_sweeps, S, strideS, U, ldu, &
                                       strideU, V, ldv, strideV, myInfo, batch_count) &
       result(cgesvdj_batched) &
       bind(C, name="rocsolver_cgesvdj_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_float, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgesvdj_batched
    end function rocsolver_cgesvdj_batched

    !---------------------------------------------
    ! rocsolver_zgesvdj_batched
    !---------------------------------------------
    function rocsolver_zgesvdj_batched(handle, left_svect, right_svect, m, n, A, lda, abstol, &
                                       residual, max_sweeps, n_sweeps, S, strideS, U, ldu, &
                                       strideU, V, ldv, strideV, myInfo, batch_count) &
       result(zgesvdj_batched) &
       bind(C, name="rocsolver_zgesvdj_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_double, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgesvdj_batched
    end function rocsolver_zgesvdj_batched

    !---------------------------------------------
    ! rocsolver_sgesvdj_strided_batched
    !---------------------------------------------
    function rocsolver_sgesvdj_strided_batched(handle, left_svect, right_svect, m, n, A, lda, &
                                               strideA, abstol, residual, max_sweeps, n_sweeps, S, &
                                               strideS, U, ldu, strideU, V, ldv, strideV, myInfo, &
                                               batch_count) &
       result(sgesvdj_strided_batched) &
       bind(C, name="rocsolver_sgesvdj_strided_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_float), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgesvdj_strided_batched
    end function rocsolver_sgesvdj_strided_batched

    !---------------------------------------------
    ! rocsolver_dgesvdj_strided_batched
    !---------------------------------------------
    function rocsolver_dgesvdj_strided_batched(handle, left_svect, right_svect, m, n, A, lda, &
                                               strideA, abstol, residual, max_sweeps, n_sweeps, S, &
                                               strideS, U, ldu, strideU, V, ldv, strideV, myInfo, &
                                               batch_count) &
       result(dgesvdj_strided_batched) &
       bind(C, name="rocsolver_dgesvdj_strided_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_double), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgesvdj_strided_batched
    end function rocsolver_dgesvdj_strided_batched

    !---------------------------------------------
    ! rocsolver_cgesvdj_strided_batched
    !---------------------------------------------
    function rocsolver_cgesvdj_strided_batched(handle, left_svect, right_svect, m, n, A, lda, &
                                               strideA, abstol, residual, max_sweeps, n_sweeps, S, &
                                               strideS, U, ldu, strideU, V, ldv, strideV, myInfo, &
                                               batch_count) &
       result(cgesvdj_strided_batched) &
       bind(C, name="rocsolver_cgesvdj_strided_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_float), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgesvdj_strided_batched
    end function rocsolver_cgesvdj_strided_batched

    !---------------------------------------------
    ! rocsolver_zgesvdj_strided_batched
    !---------------------------------------------
    function rocsolver_zgesvdj_strided_batched(handle, left_svect, right_svect, m, n, A, lda, &
                                               strideA, abstol, residual, max_sweeps, n_sweeps, S, &
                                               strideS, U, ldu, strideU, V, ldv, strideV, myInfo, &
                                               batch_count) &
       result(zgesvdj_strided_batched) &
       bind(C, name="rocsolver_zgesvdj_strided_batched")
       import :: c_ptr, rocblas_svect_all, c_int, c_int64_t, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_double), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgesvdj_strided_batched
    end function rocsolver_zgesvdj_strided_batched

    !---------------------------------------------
    ! rocsolver_sgesvdx
    !---------------------------------------------
    function rocsolver_sgesvdx(handle, left_svect, right_svect, srange, m, n, A, lda, vl, vu, il, &
                               iu, nsv, S, U, ldu, V, ldv, ifail, myInfo) &
       result(sgesvdx) &
       bind(C, name="rocsolver_sgesvdx")
       import :: c_ptr, rocblas_svect_all, rocblas_srange_all, c_int, c_float, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(kind(rocblas_srange_all)), value :: srange
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nsv
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: ifail
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgesvdx
    end function rocsolver_sgesvdx

    !---------------------------------------------
    ! rocsolver_dgesvdx
    !---------------------------------------------
    function rocsolver_dgesvdx(handle, left_svect, right_svect, srange, m, n, A, lda, vl, vu, il, &
                               iu, nsv, S, U, ldu, V, ldv, ifail, myInfo) &
       result(dgesvdx) &
       bind(C, name="rocsolver_dgesvdx")
       import :: c_ptr, rocblas_svect_all, rocblas_srange_all, c_int, c_double, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(kind(rocblas_srange_all)), value :: srange
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nsv
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: ifail
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgesvdx
    end function rocsolver_dgesvdx

    !---------------------------------------------
    ! rocsolver_cgesvdx
    !---------------------------------------------
    function rocsolver_cgesvdx(handle, left_svect, right_svect, srange, m, n, A, lda, vl, vu, il, &
                               iu, nsv, S, U, ldu, V, ldv, ifail, myInfo) &
       result(cgesvdx) &
       bind(C, name="rocsolver_cgesvdx")
       import :: c_ptr, rocblas_svect_all, rocblas_srange_all, c_int, c_float, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(kind(rocblas_srange_all)), value :: srange
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nsv
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: ifail
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgesvdx
    end function rocsolver_cgesvdx

    !---------------------------------------------
    ! rocsolver_zgesvdx
    !---------------------------------------------
    function rocsolver_zgesvdx(handle, left_svect, right_svect, srange, m, n, A, lda, vl, vu, il, &
                               iu, nsv, S, U, ldu, V, ldv, ifail, myInfo) &
       result(zgesvdx) &
       bind(C, name="rocsolver_zgesvdx")
       import :: c_ptr, rocblas_svect_all, rocblas_srange_all, c_int, c_double, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(kind(rocblas_srange_all)), value :: srange
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nsv
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: ifail
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgesvdx
    end function rocsolver_zgesvdx

    !---------------------------------------------
    ! rocsolver_sgesvdx_batched
    !---------------------------------------------
    function rocsolver_sgesvdx_batched(handle, left_svect, right_svect, srange, m, n, A, lda, vl, &
                                       vu, il, iu, nsv, S, strideS, U, ldu, strideU, V, ldv, &
                                       strideV, ifail, strideF, myInfo, batch_count) &
       result(sgesvdx_batched) &
       bind(C, name="rocsolver_sgesvdx_batched")
       import :: c_ptr, rocblas_svect_all, rocblas_srange_all, c_int, c_float, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(kind(rocblas_srange_all)), value :: srange
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nsv
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgesvdx_batched
    end function rocsolver_sgesvdx_batched

    !---------------------------------------------
    ! rocsolver_dgesvdx_batched
    !---------------------------------------------
    function rocsolver_dgesvdx_batched(handle, left_svect, right_svect, srange, m, n, A, lda, vl, &
                                       vu, il, iu, nsv, S, strideS, U, ldu, strideU, V, ldv, &
                                       strideV, ifail, strideF, myInfo, batch_count) &
       result(dgesvdx_batched) &
       bind(C, name="rocsolver_dgesvdx_batched")
       import :: c_ptr, rocblas_svect_all, rocblas_srange_all, c_int, c_double, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(kind(rocblas_srange_all)), value :: srange
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nsv
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgesvdx_batched
    end function rocsolver_dgesvdx_batched

    !---------------------------------------------
    ! rocsolver_cgesvdx_batched
    !---------------------------------------------
    function rocsolver_cgesvdx_batched(handle, left_svect, right_svect, srange, m, n, A, lda, vl, &
                                       vu, il, iu, nsv, S, strideS, U, ldu, strideU, V, ldv, &
                                       strideV, ifail, strideF, myInfo, batch_count) &
       result(cgesvdx_batched) &
       bind(C, name="rocsolver_cgesvdx_batched")
       import :: c_ptr, rocblas_svect_all, rocblas_srange_all, c_int, c_float, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(kind(rocblas_srange_all)), value :: srange
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nsv
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgesvdx_batched
    end function rocsolver_cgesvdx_batched

    !---------------------------------------------
    ! rocsolver_zgesvdx_batched
    !---------------------------------------------
    function rocsolver_zgesvdx_batched(handle, left_svect, right_svect, srange, m, n, A, lda, vl, &
                                       vu, il, iu, nsv, S, strideS, U, ldu, strideU, V, ldv, &
                                       strideV, ifail, strideF, myInfo, batch_count) &
       result(zgesvdx_batched) &
       bind(C, name="rocsolver_zgesvdx_batched")
       import :: c_ptr, rocblas_svect_all, rocblas_srange_all, c_int, c_double, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(kind(rocblas_srange_all)), value :: srange
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nsv
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgesvdx_batched
    end function rocsolver_zgesvdx_batched

    !---------------------------------------------
    ! rocsolver_sgesvdx_strided_batched
    !---------------------------------------------
    function rocsolver_sgesvdx_strided_batched(handle, left_svect, right_svect, srange, m, n, A, &
                                               lda, strideA, vl, vu, il, iu, nsv, S, strideS, U, &
                                               ldu, strideU, V, ldv, strideV, ifail, strideF, &
                                               myInfo, batch_count) &
       result(sgesvdx_strided_batched) &
       bind(C, name="rocsolver_sgesvdx_strided_batched")
       import :: c_ptr, rocblas_svect_all, rocblas_srange_all, c_int, c_int64_t, c_float, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(kind(rocblas_srange_all)), value :: srange
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nsv
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgesvdx_strided_batched
    end function rocsolver_sgesvdx_strided_batched

    !---------------------------------------------
    ! rocsolver_dgesvdx_strided_batched
    !---------------------------------------------
    function rocsolver_dgesvdx_strided_batched(handle, left_svect, right_svect, srange, m, n, A, &
                                               lda, strideA, vl, vu, il, iu, nsv, S, strideS, U, &
                                               ldu, strideU, V, ldv, strideV, ifail, strideF, &
                                               myInfo, batch_count) &
       result(dgesvdx_strided_batched) &
       bind(C, name="rocsolver_dgesvdx_strided_batched")
       import :: c_ptr, rocblas_svect_all, rocblas_srange_all, c_int, c_int64_t, c_double, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(kind(rocblas_srange_all)), value :: srange
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nsv
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgesvdx_strided_batched
    end function rocsolver_dgesvdx_strided_batched

    !---------------------------------------------
    ! rocsolver_cgesvdx_strided_batched
    !---------------------------------------------
    function rocsolver_cgesvdx_strided_batched(handle, left_svect, right_svect, srange, m, n, A, &
                                               lda, strideA, vl, vu, il, iu, nsv, S, strideS, U, &
                                               ldu, strideU, V, ldv, strideV, ifail, strideF, &
                                               myInfo, batch_count) &
       result(cgesvdx_strided_batched) &
       bind(C, name="rocsolver_cgesvdx_strided_batched")
       import :: c_ptr, rocblas_svect_all, rocblas_srange_all, c_int, c_int64_t, c_float, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(kind(rocblas_srange_all)), value :: srange
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nsv
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgesvdx_strided_batched
    end function rocsolver_cgesvdx_strided_batched

    !---------------------------------------------
    ! rocsolver_zgesvdx_strided_batched
    !---------------------------------------------
    function rocsolver_zgesvdx_strided_batched(handle, left_svect, right_svect, srange, m, n, A, &
                                               lda, strideA, vl, vu, il, iu, nsv, S, strideS, U, &
                                               ldu, strideU, V, ldv, strideV, ifail, strideF, &
                                               myInfo, batch_count) &
       result(zgesvdx_strided_batched) &
       bind(C, name="rocsolver_zgesvdx_strided_batched")
       import :: c_ptr, rocblas_svect_all, rocblas_srange_all, c_int, c_int64_t, c_double, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_svect_all)), value :: left_svect
       integer(kind(rocblas_svect_all)), value :: right_svect
       integer(kind(rocblas_srange_all)), value :: srange
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nsv
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgesvdx_strided_batched
    end function rocsolver_zgesvdx_strided_batched

    !---------------------------------------------
    ! rocsolver_ssytd2
    !---------------------------------------------
    function rocsolver_ssytd2(handle, uplo, n, A, lda, D, E, tau) &
       result(ssytd2) &
       bind(C, name="rocsolver_ssytd2")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: ssytd2
    end function rocsolver_ssytd2

    !---------------------------------------------
    ! rocsolver_dsytd2
    !---------------------------------------------
    function rocsolver_dsytd2(handle, uplo, n, A, lda, D, E, tau) &
       result(dsytd2) &
       bind(C, name="rocsolver_dsytd2")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: dsytd2
    end function rocsolver_dsytd2

    !---------------------------------------------
    ! rocsolver_chetd2
    !---------------------------------------------
    function rocsolver_chetd2(handle, uplo, n, A, lda, D, E, tau) &
       result(chetd2) &
       bind(C, name="rocsolver_chetd2")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: chetd2
    end function rocsolver_chetd2

    !---------------------------------------------
    ! rocsolver_zhetd2
    !---------------------------------------------
    function rocsolver_zhetd2(handle, uplo, n, A, lda, D, E, tau) &
       result(zhetd2) &
       bind(C, name="rocsolver_zhetd2")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: zhetd2
    end function rocsolver_zhetd2

    !---------------------------------------------
    ! rocsolver_ssytd2_batched
    !---------------------------------------------
    function rocsolver_ssytd2_batched(handle, uplo, n, A, lda, D, strideD, E, strideE, tau, &
                                      strideP, batch_count) &
       result(ssytd2_batched) &
       bind(C, name="rocsolver_ssytd2_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssytd2_batched
    end function rocsolver_ssytd2_batched

    !---------------------------------------------
    ! rocsolver_dsytd2_batched
    !---------------------------------------------
    function rocsolver_dsytd2_batched(handle, uplo, n, A, lda, D, strideD, E, strideE, tau, &
                                      strideP, batch_count) &
       result(dsytd2_batched) &
       bind(C, name="rocsolver_dsytd2_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsytd2_batched
    end function rocsolver_dsytd2_batched

    !---------------------------------------------
    ! rocsolver_chetd2_batched
    !---------------------------------------------
    function rocsolver_chetd2_batched(handle, uplo, n, A, lda, D, strideD, E, strideE, tau, &
                                      strideP, batch_count) &
       result(chetd2_batched) &
       bind(C, name="rocsolver_chetd2_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chetd2_batched
    end function rocsolver_chetd2_batched

    !---------------------------------------------
    ! rocsolver_zhetd2_batched
    !---------------------------------------------
    function rocsolver_zhetd2_batched(handle, uplo, n, A, lda, D, strideD, E, strideE, tau, &
                                      strideP, batch_count) &
       result(zhetd2_batched) &
       bind(C, name="rocsolver_zhetd2_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhetd2_batched
    end function rocsolver_zhetd2_batched

    !---------------------------------------------
    ! rocsolver_ssytd2_strided_batched
    !---------------------------------------------
    function rocsolver_ssytd2_strided_batched(handle, uplo, n, A, lda, strideA, D, strideD, E, &
                                              strideE, tau, strideP, batch_count) &
       result(ssytd2_strided_batched) &
       bind(C, name="rocsolver_ssytd2_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssytd2_strided_batched
    end function rocsolver_ssytd2_strided_batched

    !---------------------------------------------
    ! rocsolver_dsytd2_strided_batched
    !---------------------------------------------
    function rocsolver_dsytd2_strided_batched(handle, uplo, n, A, lda, strideA, D, strideD, E, &
                                              strideE, tau, strideP, batch_count) &
       result(dsytd2_strided_batched) &
       bind(C, name="rocsolver_dsytd2_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsytd2_strided_batched
    end function rocsolver_dsytd2_strided_batched

    !---------------------------------------------
    ! rocsolver_chetd2_strided_batched
    !---------------------------------------------
    function rocsolver_chetd2_strided_batched(handle, uplo, n, A, lda, strideA, D, strideD, E, &
                                              strideE, tau, strideP, batch_count) &
       result(chetd2_strided_batched) &
       bind(C, name="rocsolver_chetd2_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chetd2_strided_batched
    end function rocsolver_chetd2_strided_batched

    !---------------------------------------------
    ! rocsolver_zhetd2_strided_batched
    !---------------------------------------------
    function rocsolver_zhetd2_strided_batched(handle, uplo, n, A, lda, strideA, D, strideD, E, &
                                              strideE, tau, strideP, batch_count) &
       result(zhetd2_strided_batched) &
       bind(C, name="rocsolver_zhetd2_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhetd2_strided_batched
    end function rocsolver_zhetd2_strided_batched

    !---------------------------------------------
    ! rocsolver_ssytrd
    !---------------------------------------------
    function rocsolver_ssytrd(handle, uplo, n, A, lda, D, E, tau) &
       result(ssytrd) &
       bind(C, name="rocsolver_ssytrd")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: ssytrd
    end function rocsolver_ssytrd

    !---------------------------------------------
    ! rocsolver_dsytrd
    !---------------------------------------------
    function rocsolver_dsytrd(handle, uplo, n, A, lda, D, E, tau) &
       result(dsytrd) &
       bind(C, name="rocsolver_dsytrd")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: dsytrd
    end function rocsolver_dsytrd

    !---------------------------------------------
    ! rocsolver_chetrd
    !---------------------------------------------
    function rocsolver_chetrd(handle, uplo, n, A, lda, D, E, tau) &
       result(chetrd) &
       bind(C, name="rocsolver_chetrd")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: chetrd
    end function rocsolver_chetrd

    !---------------------------------------------
    ! rocsolver_zhetrd
    !---------------------------------------------
    function rocsolver_zhetrd(handle, uplo, n, A, lda, D, E, tau) &
       result(zhetrd) &
       bind(C, name="rocsolver_zhetrd")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: zhetrd
    end function rocsolver_zhetrd

    !---------------------------------------------
    ! rocsolver_ssytrd_batched
    !---------------------------------------------
    function rocsolver_ssytrd_batched(handle, uplo, n, A, lda, D, strideD, E, strideE, tau, &
                                      strideP, batch_count) &
       result(ssytrd_batched) &
       bind(C, name="rocsolver_ssytrd_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssytrd_batched
    end function rocsolver_ssytrd_batched

    !---------------------------------------------
    ! rocsolver_dsytrd_batched
    !---------------------------------------------
    function rocsolver_dsytrd_batched(handle, uplo, n, A, lda, D, strideD, E, strideE, tau, &
                                      strideP, batch_count) &
       result(dsytrd_batched) &
       bind(C, name="rocsolver_dsytrd_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsytrd_batched
    end function rocsolver_dsytrd_batched

    !---------------------------------------------
    ! rocsolver_chetrd_batched
    !---------------------------------------------
    function rocsolver_chetrd_batched(handle, uplo, n, A, lda, D, strideD, E, strideE, tau, &
                                      strideP, batch_count) &
       result(chetrd_batched) &
       bind(C, name="rocsolver_chetrd_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chetrd_batched
    end function rocsolver_chetrd_batched

    !---------------------------------------------
    ! rocsolver_zhetrd_batched
    !---------------------------------------------
    function rocsolver_zhetrd_batched(handle, uplo, n, A, lda, D, strideD, E, strideE, tau, &
                                      strideP, batch_count) &
       result(zhetrd_batched) &
       bind(C, name="rocsolver_zhetrd_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhetrd_batched
    end function rocsolver_zhetrd_batched

    !---------------------------------------------
    ! rocsolver_ssytrd_strided_batched
    !---------------------------------------------
    function rocsolver_ssytrd_strided_batched(handle, uplo, n, A, lda, strideA, D, strideD, E, &
                                              strideE, tau, strideP, batch_count) &
       result(ssytrd_strided_batched) &
       bind(C, name="rocsolver_ssytrd_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssytrd_strided_batched
    end function rocsolver_ssytrd_strided_batched

    !---------------------------------------------
    ! rocsolver_dsytrd_strided_batched
    !---------------------------------------------
    function rocsolver_dsytrd_strided_batched(handle, uplo, n, A, lda, strideA, D, strideD, E, &
                                              strideE, tau, strideP, batch_count) &
       result(dsytrd_strided_batched) &
       bind(C, name="rocsolver_dsytrd_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsytrd_strided_batched
    end function rocsolver_dsytrd_strided_batched

    !---------------------------------------------
    ! rocsolver_chetrd_strided_batched
    !---------------------------------------------
    function rocsolver_chetrd_strided_batched(handle, uplo, n, A, lda, strideA, D, strideD, E, &
                                              strideE, tau, strideP, batch_count) &
       result(chetrd_strided_batched) &
       bind(C, name="rocsolver_chetrd_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chetrd_strided_batched
    end function rocsolver_chetrd_strided_batched

    !---------------------------------------------
    ! rocsolver_zhetrd_strided_batched
    !---------------------------------------------
    function rocsolver_zhetrd_strided_batched(handle, uplo, n, A, lda, strideA, D, strideD, E, &
                                              strideE, tau, strideP, batch_count) &
       result(zhetrd_strided_batched) &
       bind(C, name="rocsolver_zhetrd_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhetrd_strided_batched
    end function rocsolver_zhetrd_strided_batched

    !---------------------------------------------
    ! rocsolver_ssygs2
    !---------------------------------------------
    function rocsolver_ssygs2(handle, itype, uplo, n, A, lda, B, ldb) &
       result(ssygs2) &
       bind(C, name="rocsolver_ssygs2")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: ssygs2
    end function rocsolver_ssygs2

    !---------------------------------------------
    ! rocsolver_dsygs2
    !---------------------------------------------
    function rocsolver_dsygs2(handle, itype, uplo, n, A, lda, B, ldb) &
       result(dsygs2) &
       bind(C, name="rocsolver_dsygs2")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: dsygs2
    end function rocsolver_dsygs2

    !---------------------------------------------
    ! rocsolver_chegs2
    !---------------------------------------------
    function rocsolver_chegs2(handle, itype, uplo, n, A, lda, B, ldb) &
       result(chegs2) &
       bind(C, name="rocsolver_chegs2")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: chegs2
    end function rocsolver_chegs2

    !---------------------------------------------
    ! rocsolver_zhegs2
    !---------------------------------------------
    function rocsolver_zhegs2(handle, itype, uplo, n, A, lda, B, ldb) &
       result(zhegs2) &
       bind(C, name="rocsolver_zhegs2")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: zhegs2
    end function rocsolver_zhegs2

    !---------------------------------------------
    ! rocsolver_ssygs2_batched
    !---------------------------------------------
    function rocsolver_ssygs2_batched(handle, itype, uplo, n, A, lda, B, ldb, batch_count) &
       result(ssygs2_batched) &
       bind(C, name="rocsolver_ssygs2_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssygs2_batched
    end function rocsolver_ssygs2_batched

    !---------------------------------------------
    ! rocsolver_dsygs2_batched
    !---------------------------------------------
    function rocsolver_dsygs2_batched(handle, itype, uplo, n, A, lda, B, ldb, batch_count) &
       result(dsygs2_batched) &
       bind(C, name="rocsolver_dsygs2_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsygs2_batched
    end function rocsolver_dsygs2_batched

    !---------------------------------------------
    ! rocsolver_chegs2_batched
    !---------------------------------------------
    function rocsolver_chegs2_batched(handle, itype, uplo, n, A, lda, B, ldb, batch_count) &
       result(chegs2_batched) &
       bind(C, name="rocsolver_chegs2_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chegs2_batched
    end function rocsolver_chegs2_batched

    !---------------------------------------------
    ! rocsolver_zhegs2_batched
    !---------------------------------------------
    function rocsolver_zhegs2_batched(handle, itype, uplo, n, A, lda, B, ldb, batch_count) &
       result(zhegs2_batched) &
       bind(C, name="rocsolver_zhegs2_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhegs2_batched
    end function rocsolver_zhegs2_batched

    !---------------------------------------------
    ! rocsolver_ssygs2_strided_batched
    !---------------------------------------------
    function rocsolver_ssygs2_strided_batched(handle, itype, uplo, n, A, lda, strideA, B, ldb, &
                                              strideB, batch_count) &
       result(ssygs2_strided_batched) &
       bind(C, name="rocsolver_ssygs2_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssygs2_strided_batched
    end function rocsolver_ssygs2_strided_batched

    !---------------------------------------------
    ! rocsolver_dsygs2_strided_batched
    !---------------------------------------------
    function rocsolver_dsygs2_strided_batched(handle, itype, uplo, n, A, lda, strideA, B, ldb, &
                                              strideB, batch_count) &
       result(dsygs2_strided_batched) &
       bind(C, name="rocsolver_dsygs2_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsygs2_strided_batched
    end function rocsolver_dsygs2_strided_batched

    !---------------------------------------------
    ! rocsolver_chegs2_strided_batched
    !---------------------------------------------
    function rocsolver_chegs2_strided_batched(handle, itype, uplo, n, A, lda, strideA, B, ldb, &
                                              strideB, batch_count) &
       result(chegs2_strided_batched) &
       bind(C, name="rocsolver_chegs2_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chegs2_strided_batched
    end function rocsolver_chegs2_strided_batched

    !---------------------------------------------
    ! rocsolver_zhegs2_strided_batched
    !---------------------------------------------
    function rocsolver_zhegs2_strided_batched(handle, itype, uplo, n, A, lda, strideA, B, ldb, &
                                              strideB, batch_count) &
       result(zhegs2_strided_batched) &
       bind(C, name="rocsolver_zhegs2_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhegs2_strided_batched
    end function rocsolver_zhegs2_strided_batched

    !---------------------------------------------
    ! rocsolver_ssygst
    !---------------------------------------------
    function rocsolver_ssygst(handle, itype, uplo, n, A, lda, B, ldb) &
       result(ssygst) &
       bind(C, name="rocsolver_ssygst")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: ssygst
    end function rocsolver_ssygst

    !---------------------------------------------
    ! rocsolver_dsygst
    !---------------------------------------------
    function rocsolver_dsygst(handle, itype, uplo, n, A, lda, B, ldb) &
       result(dsygst) &
       bind(C, name="rocsolver_dsygst")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: dsygst
    end function rocsolver_dsygst

    !---------------------------------------------
    ! rocsolver_chegst
    !---------------------------------------------
    function rocsolver_chegst(handle, itype, uplo, n, A, lda, B, ldb) &
       result(chegst) &
       bind(C, name="rocsolver_chegst")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: chegst
    end function rocsolver_chegst

    !---------------------------------------------
    ! rocsolver_zhegst
    !---------------------------------------------
    function rocsolver_zhegst(handle, itype, uplo, n, A, lda, B, ldb) &
       result(zhegst) &
       bind(C, name="rocsolver_zhegst")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(kind(rocblas_status_success)) :: zhegst
    end function rocsolver_zhegst

    !---------------------------------------------
    ! rocsolver_ssygst_batched
    !---------------------------------------------
    function rocsolver_ssygst_batched(handle, itype, uplo, n, A, lda, B, ldb, batch_count) &
       result(ssygst_batched) &
       bind(C, name="rocsolver_ssygst_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssygst_batched
    end function rocsolver_ssygst_batched

    !---------------------------------------------
    ! rocsolver_dsygst_batched
    !---------------------------------------------
    function rocsolver_dsygst_batched(handle, itype, uplo, n, A, lda, B, ldb, batch_count) &
       result(dsygst_batched) &
       bind(C, name="rocsolver_dsygst_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsygst_batched
    end function rocsolver_dsygst_batched

    !---------------------------------------------
    ! rocsolver_chegst_batched
    !---------------------------------------------
    function rocsolver_chegst_batched(handle, itype, uplo, n, A, lda, B, ldb, batch_count) &
       result(chegst_batched) &
       bind(C, name="rocsolver_chegst_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chegst_batched
    end function rocsolver_chegst_batched

    !---------------------------------------------
    ! rocsolver_zhegst_batched
    !---------------------------------------------
    function rocsolver_zhegst_batched(handle, itype, uplo, n, A, lda, B, ldb, batch_count) &
       result(zhegst_batched) &
       bind(C, name="rocsolver_zhegst_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhegst_batched
    end function rocsolver_zhegst_batched

    !---------------------------------------------
    ! rocsolver_ssygst_strided_batched
    !---------------------------------------------
    function rocsolver_ssygst_strided_batched(handle, itype, uplo, n, A, lda, strideA, B, ldb, &
                                              strideB, batch_count) &
       result(ssygst_strided_batched) &
       bind(C, name="rocsolver_ssygst_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssygst_strided_batched
    end function rocsolver_ssygst_strided_batched

    !---------------------------------------------
    ! rocsolver_dsygst_strided_batched
    !---------------------------------------------
    function rocsolver_dsygst_strided_batched(handle, itype, uplo, n, A, lda, strideA, B, ldb, &
                                              strideB, batch_count) &
       result(dsygst_strided_batched) &
       bind(C, name="rocsolver_dsygst_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsygst_strided_batched
    end function rocsolver_dsygst_strided_batched

    !---------------------------------------------
    ! rocsolver_chegst_strided_batched
    !---------------------------------------------
    function rocsolver_chegst_strided_batched(handle, itype, uplo, n, A, lda, strideA, B, ldb, &
                                              strideB, batch_count) &
       result(chegst_strided_batched) &
       bind(C, name="rocsolver_chegst_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chegst_strided_batched
    end function rocsolver_chegst_strided_batched

    !---------------------------------------------
    ! rocsolver_zhegst_strided_batched
    !---------------------------------------------
    function rocsolver_zhegst_strided_batched(handle, itype, uplo, n, A, lda, strideA, B, ldb, &
                                              strideB, batch_count) &
       result(zhegst_strided_batched) &
       bind(C, name="rocsolver_zhegst_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhegst_strided_batched
    end function rocsolver_zhegst_strided_batched

    !---------------------------------------------
    ! rocsolver_sgehd2
    !---------------------------------------------
    function rocsolver_sgehd2(handle, n, ilo, ihi, A, lda, tau) &
       result(sgehd2) &
       bind(C, name="rocsolver_sgehd2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: sgehd2
    end function rocsolver_sgehd2

    !---------------------------------------------
    ! rocsolver_dgehd2
    !---------------------------------------------
    function rocsolver_dgehd2(handle, n, ilo, ihi, A, lda, tau) &
       result(dgehd2) &
       bind(C, name="rocsolver_dgehd2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: dgehd2
    end function rocsolver_dgehd2

    !---------------------------------------------
    ! rocsolver_cgehd2
    !---------------------------------------------
    function rocsolver_cgehd2(handle, n, ilo, ihi, A, lda, tau) &
       result(cgehd2) &
       bind(C, name="rocsolver_cgehd2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: cgehd2
    end function rocsolver_cgehd2

    !---------------------------------------------
    ! rocsolver_zgehd2
    !---------------------------------------------
    function rocsolver_zgehd2(handle, n, ilo, ihi, A, lda, tau) &
       result(zgehd2) &
       bind(C, name="rocsolver_zgehd2")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: zgehd2
    end function rocsolver_zgehd2

    !---------------------------------------------
    ! rocsolver_sgehd2_batched
    !---------------------------------------------
    function rocsolver_sgehd2_batched(handle, n, ilo, ihi, A, lda, tau, strideP, batch_count) &
       result(sgehd2_batched) &
       bind(C, name="rocsolver_sgehd2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgehd2_batched
    end function rocsolver_sgehd2_batched

    !---------------------------------------------
    ! rocsolver_dgehd2_batched
    !---------------------------------------------
    function rocsolver_dgehd2_batched(handle, n, ilo, ihi, A, lda, tau, strideP, batch_count) &
       result(dgehd2_batched) &
       bind(C, name="rocsolver_dgehd2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgehd2_batched
    end function rocsolver_dgehd2_batched

    !---------------------------------------------
    ! rocsolver_cgehd2_batched
    !---------------------------------------------
    function rocsolver_cgehd2_batched(handle, n, ilo, ihi, A, lda, tau, strideP, batch_count) &
       result(cgehd2_batched) &
       bind(C, name="rocsolver_cgehd2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgehd2_batched
    end function rocsolver_cgehd2_batched

    !---------------------------------------------
    ! rocsolver_zgehd2_batched
    !---------------------------------------------
    function rocsolver_zgehd2_batched(handle, n, ilo, ihi, A, lda, tau, strideP, batch_count) &
       result(zgehd2_batched) &
       bind(C, name="rocsolver_zgehd2_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgehd2_batched
    end function rocsolver_zgehd2_batched

    !---------------------------------------------
    ! rocsolver_sgehd2_strided_batched
    !---------------------------------------------
    function rocsolver_sgehd2_strided_batched(handle, n, ilo, ihi, A, lda, strideA, tau, strideP, &
                                              batch_count) &
       result(sgehd2_strided_batched) &
       bind(C, name="rocsolver_sgehd2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgehd2_strided_batched
    end function rocsolver_sgehd2_strided_batched

    !---------------------------------------------
    ! rocsolver_dgehd2_strided_batched
    !---------------------------------------------
    function rocsolver_dgehd2_strided_batched(handle, n, ilo, ihi, A, lda, strideA, tau, strideP, &
                                              batch_count) &
       result(dgehd2_strided_batched) &
       bind(C, name="rocsolver_dgehd2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgehd2_strided_batched
    end function rocsolver_dgehd2_strided_batched

    !---------------------------------------------
    ! rocsolver_cgehd2_strided_batched
    !---------------------------------------------
    function rocsolver_cgehd2_strided_batched(handle, n, ilo, ihi, A, lda, strideA, tau, strideP, &
                                              batch_count) &
       result(cgehd2_strided_batched) &
       bind(C, name="rocsolver_cgehd2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgehd2_strided_batched
    end function rocsolver_cgehd2_strided_batched

    !---------------------------------------------
    ! rocsolver_zgehd2_strided_batched
    !---------------------------------------------
    function rocsolver_zgehd2_strided_batched(handle, n, ilo, ihi, A, lda, strideA, tau, strideP, &
                                              batch_count) &
       result(zgehd2_strided_batched) &
       bind(C, name="rocsolver_zgehd2_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgehd2_strided_batched
    end function rocsolver_zgehd2_strided_batched

    !---------------------------------------------
    ! rocsolver_sgehrd
    !---------------------------------------------
    function rocsolver_sgehrd(handle, n, ilo, ihi, A, lda, tau) &
       result(sgehrd) &
       bind(C, name="rocsolver_sgehrd")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: sgehrd
    end function rocsolver_sgehrd

    !---------------------------------------------
    ! rocsolver_dgehrd
    !---------------------------------------------
    function rocsolver_dgehrd(handle, n, ilo, ihi, A, lda, tau) &
       result(dgehrd) &
       bind(C, name="rocsolver_dgehrd")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: dgehrd
    end function rocsolver_dgehrd

    !---------------------------------------------
    ! rocsolver_cgehrd
    !---------------------------------------------
    function rocsolver_cgehrd(handle, n, ilo, ihi, A, lda, tau) &
       result(cgehrd) &
       bind(C, name="rocsolver_cgehrd")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: cgehrd
    end function rocsolver_cgehrd

    !---------------------------------------------
    ! rocsolver_zgehrd
    !---------------------------------------------
    function rocsolver_zgehrd(handle, n, ilo, ihi, A, lda, tau) &
       result(zgehrd) &
       bind(C, name="rocsolver_zgehrd")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(kind(rocblas_status_success)) :: zgehrd
    end function rocsolver_zgehrd

    !---------------------------------------------
    ! rocsolver_sgehrd_batched
    !---------------------------------------------
    function rocsolver_sgehrd_batched(handle, n, ilo, ihi, A, lda, tau, strideP, batch_count) &
       result(sgehrd_batched) &
       bind(C, name="rocsolver_sgehrd_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgehrd_batched
    end function rocsolver_sgehrd_batched

    !---------------------------------------------
    ! rocsolver_dgehrd_batched
    !---------------------------------------------
    function rocsolver_dgehrd_batched(handle, n, ilo, ihi, A, lda, tau, strideP, batch_count) &
       result(dgehrd_batched) &
       bind(C, name="rocsolver_dgehrd_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgehrd_batched
    end function rocsolver_dgehrd_batched

    !---------------------------------------------
    ! rocsolver_cgehrd_batched
    !---------------------------------------------
    function rocsolver_cgehrd_batched(handle, n, ilo, ihi, A, lda, tau, strideP, batch_count) &
       result(cgehrd_batched) &
       bind(C, name="rocsolver_cgehrd_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgehrd_batched
    end function rocsolver_cgehrd_batched

    !---------------------------------------------
    ! rocsolver_zgehrd_batched
    !---------------------------------------------
    function rocsolver_zgehrd_batched(handle, n, ilo, ihi, A, lda, tau, strideP, batch_count) &
       result(zgehrd_batched) &
       bind(C, name="rocsolver_zgehrd_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgehrd_batched
    end function rocsolver_zgehrd_batched

    !---------------------------------------------
    ! rocsolver_sgehrd_strided_batched
    !---------------------------------------------
    function rocsolver_sgehrd_strided_batched(handle, n, ilo, ihi, A, lda, strideA, tau, strideP, &
                                              batch_count) &
       result(sgehrd_strided_batched) &
       bind(C, name="rocsolver_sgehrd_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgehrd_strided_batched
    end function rocsolver_sgehrd_strided_batched

    !---------------------------------------------
    ! rocsolver_dgehrd_strided_batched
    !---------------------------------------------
    function rocsolver_dgehrd_strided_batched(handle, n, ilo, ihi, A, lda, strideA, tau, strideP, &
                                              batch_count) &
       result(dgehrd_strided_batched) &
       bind(C, name="rocsolver_dgehrd_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgehrd_strided_batched
    end function rocsolver_dgehrd_strided_batched

    !---------------------------------------------
    ! rocsolver_cgehrd_strided_batched
    !---------------------------------------------
    function rocsolver_cgehrd_strided_batched(handle, n, ilo, ihi, A, lda, strideA, tau, strideP, &
                                              batch_count) &
       result(cgehrd_strided_batched) &
       bind(C, name="rocsolver_cgehrd_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgehrd_strided_batched
    end function rocsolver_cgehrd_strided_batched

    !---------------------------------------------
    ! rocsolver_zgehrd_strided_batched
    !---------------------------------------------
    function rocsolver_zgehrd_strided_batched(handle, n, ilo, ihi, A, lda, strideA, tau, strideP, &
                                              batch_count) &
       result(zgehrd_strided_batched) &
       bind(C, name="rocsolver_zgehrd_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: ilo
       integer(c_int), value :: ihi
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: tau
       integer(c_int64_t), value :: strideP
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgehrd_strided_batched
    end function rocsolver_zgehrd_strided_batched

    !---------------------------------------------
    ! rocsolver_ssyev
    !---------------------------------------------
    function rocsolver_ssyev(handle, evect, uplo, n, A, lda, D, E, myInfo) &
       result(ssyev) &
       bind(C, name="rocsolver_ssyev")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ssyev
    end function rocsolver_ssyev

    !---------------------------------------------
    ! rocsolver_dsyev
    !---------------------------------------------
    function rocsolver_dsyev(handle, evect, uplo, n, A, lda, D, E, myInfo) &
       result(dsyev) &
       bind(C, name="rocsolver_dsyev")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dsyev
    end function rocsolver_dsyev

    !---------------------------------------------
    ! rocsolver_ssyev_64
    !---------------------------------------------
    function rocsolver_ssyev_64(handle, evect, uplo, n, A, lda, D, E, myInfo) &
       result(ssyev_64) &
       bind(C, name="rocsolver_ssyev_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ssyev_64
    end function rocsolver_ssyev_64

    !---------------------------------------------
    ! rocsolver_dsyev_64
    !---------------------------------------------
    function rocsolver_dsyev_64(handle, evect, uplo, n, A, lda, D, E, myInfo) &
       result(dsyev_64) &
       bind(C, name="rocsolver_dsyev_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dsyev_64
    end function rocsolver_dsyev_64

    !---------------------------------------------
    ! rocsolver_cheev
    !---------------------------------------------
    function rocsolver_cheev(handle, evect, uplo, n, A, lda, D, E, myInfo) &
       result(cheev) &
       bind(C, name="rocsolver_cheev")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cheev
    end function rocsolver_cheev

    !---------------------------------------------
    ! rocsolver_zheev
    !---------------------------------------------
    function rocsolver_zheev(handle, evect, uplo, n, A, lda, D, E, myInfo) &
       result(zheev) &
       bind(C, name="rocsolver_zheev")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zheev
    end function rocsolver_zheev

    !---------------------------------------------
    ! rocsolver_cheev_64
    !---------------------------------------------
    function rocsolver_cheev_64(handle, evect, uplo, n, A, lda, D, E, myInfo) &
       result(cheev_64) &
       bind(C, name="rocsolver_cheev_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cheev_64
    end function rocsolver_cheev_64

    !---------------------------------------------
    ! rocsolver_zheev_64
    !---------------------------------------------
    function rocsolver_zheev_64(handle, evect, uplo, n, A, lda, D, E, myInfo) &
       result(zheev_64) &
       bind(C, name="rocsolver_zheev_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zheev_64
    end function rocsolver_zheev_64

    !---------------------------------------------
    ! rocsolver_ssyev_batched
    !---------------------------------------------
    function rocsolver_ssyev_batched(handle, evect, uplo, n, A, lda, D, strideD, E, strideE, &
                                     myInfo, batch_count) &
       result(ssyev_batched) &
       bind(C, name="rocsolver_ssyev_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssyev_batched
    end function rocsolver_ssyev_batched

    !---------------------------------------------
    ! rocsolver_dsyev_batched
    !---------------------------------------------
    function rocsolver_dsyev_batched(handle, evect, uplo, n, A, lda, D, strideD, E, strideE, &
                                     myInfo, batch_count) &
       result(dsyev_batched) &
       bind(C, name="rocsolver_dsyev_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsyev_batched
    end function rocsolver_dsyev_batched

    !---------------------------------------------
    ! rocsolver_ssyev_batched_64
    !---------------------------------------------
    function rocsolver_ssyev_batched_64(handle, evect, uplo, n, A, lda, D, strideD, E, strideE, &
                                        myInfo, batch_count) &
       result(ssyev_batched_64) &
       bind(C, name="rocsolver_ssyev_batched_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssyev_batched_64
    end function rocsolver_ssyev_batched_64

    !---------------------------------------------
    ! rocsolver_dsyev_batched_64
    !---------------------------------------------
    function rocsolver_dsyev_batched_64(handle, evect, uplo, n, A, lda, D, strideD, E, strideE, &
                                        myInfo, batch_count) &
       result(dsyev_batched_64) &
       bind(C, name="rocsolver_dsyev_batched_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsyev_batched_64
    end function rocsolver_dsyev_batched_64

    !---------------------------------------------
    ! rocsolver_cheev_batched
    !---------------------------------------------
    function rocsolver_cheev_batched(handle, evect, uplo, n, A, lda, D, strideD, E, strideE, &
                                     myInfo, batch_count) &
       result(cheev_batched) &
       bind(C, name="rocsolver_cheev_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cheev_batched
    end function rocsolver_cheev_batched

    !---------------------------------------------
    ! rocsolver_zheev_batched
    !---------------------------------------------
    function rocsolver_zheev_batched(handle, evect, uplo, n, A, lda, D, strideD, E, strideE, &
                                     myInfo, batch_count) &
       result(zheev_batched) &
       bind(C, name="rocsolver_zheev_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zheev_batched
    end function rocsolver_zheev_batched

    !---------------------------------------------
    ! rocsolver_cheev_batched_64
    !---------------------------------------------
    function rocsolver_cheev_batched_64(handle, evect, uplo, n, A, lda, D, strideD, E, strideE, &
                                        myInfo, batch_count) &
       result(cheev_batched_64) &
       bind(C, name="rocsolver_cheev_batched_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cheev_batched_64
    end function rocsolver_cheev_batched_64

    !---------------------------------------------
    ! rocsolver_zheev_batched_64
    !---------------------------------------------
    function rocsolver_zheev_batched_64(handle, evect, uplo, n, A, lda, D, strideD, E, strideE, &
                                        myInfo, batch_count) &
       result(zheev_batched_64) &
       bind(C, name="rocsolver_zheev_batched_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zheev_batched_64
    end function rocsolver_zheev_batched_64

    !---------------------------------------------
    ! rocsolver_ssyev_strided_batched
    !---------------------------------------------
    function rocsolver_ssyev_strided_batched(handle, evect, uplo, n, A, lda, strideA, D, strideD, &
                                             E, strideE, myInfo, batch_count) &
       result(ssyev_strided_batched) &
       bind(C, name="rocsolver_ssyev_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssyev_strided_batched
    end function rocsolver_ssyev_strided_batched

    !---------------------------------------------
    ! rocsolver_dsyev_strided_batched
    !---------------------------------------------
    function rocsolver_dsyev_strided_batched(handle, evect, uplo, n, A, lda, strideA, D, strideD, &
                                             E, strideE, myInfo, batch_count) &
       result(dsyev_strided_batched) &
       bind(C, name="rocsolver_dsyev_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsyev_strided_batched
    end function rocsolver_dsyev_strided_batched

    !---------------------------------------------
    ! rocsolver_ssyev_strided_batched_64
    !---------------------------------------------
    function rocsolver_ssyev_strided_batched_64(handle, evect, uplo, n, A, lda, strideA, D, &
                                                strideD, E, strideE, myInfo, batch_count) &
       result(ssyev_strided_batched_64) &
       bind(C, name="rocsolver_ssyev_strided_batched_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssyev_strided_batched_64
    end function rocsolver_ssyev_strided_batched_64

    !---------------------------------------------
    ! rocsolver_dsyev_strided_batched_64
    !---------------------------------------------
    function rocsolver_dsyev_strided_batched_64(handle, evect, uplo, n, A, lda, strideA, D, &
                                                strideD, E, strideE, myInfo, batch_count) &
       result(dsyev_strided_batched_64) &
       bind(C, name="rocsolver_dsyev_strided_batched_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsyev_strided_batched_64
    end function rocsolver_dsyev_strided_batched_64

    !---------------------------------------------
    ! rocsolver_cheev_strided_batched
    !---------------------------------------------
    function rocsolver_cheev_strided_batched(handle, evect, uplo, n, A, lda, strideA, D, strideD, &
                                             E, strideE, myInfo, batch_count) &
       result(cheev_strided_batched) &
       bind(C, name="rocsolver_cheev_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cheev_strided_batched
    end function rocsolver_cheev_strided_batched

    !---------------------------------------------
    ! rocsolver_zheev_strided_batched
    !---------------------------------------------
    function rocsolver_zheev_strided_batched(handle, evect, uplo, n, A, lda, strideA, D, strideD, &
                                             E, strideE, myInfo, batch_count) &
       result(zheev_strided_batched) &
       bind(C, name="rocsolver_zheev_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zheev_strided_batched
    end function rocsolver_zheev_strided_batched

    !---------------------------------------------
    ! rocsolver_cheev_strided_batched_64
    !---------------------------------------------
    function rocsolver_cheev_strided_batched_64(handle, evect, uplo, n, A, lda, strideA, D, &
                                                strideD, E, strideE, myInfo, batch_count) &
       result(cheev_strided_batched_64) &
       bind(C, name="rocsolver_cheev_strided_batched_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cheev_strided_batched_64
    end function rocsolver_cheev_strided_batched_64

    !---------------------------------------------
    ! rocsolver_zheev_strided_batched_64
    !---------------------------------------------
    function rocsolver_zheev_strided_batched_64(handle, evect, uplo, n, A, lda, strideA, D, &
                                                strideD, E, strideE, myInfo, batch_count) &
       result(zheev_strided_batched_64) &
       bind(C, name="rocsolver_zheev_strided_batched_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zheev_strided_batched_64
    end function rocsolver_zheev_strided_batched_64

    !---------------------------------------------
    ! rocsolver_ssyevd
    !---------------------------------------------
    function rocsolver_ssyevd(handle, evect, uplo, n, A, lda, D, E, myInfo) &
       result(ssyevd) &
       bind(C, name="rocsolver_ssyevd")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ssyevd
    end function rocsolver_ssyevd

    !---------------------------------------------
    ! rocsolver_dsyevd
    !---------------------------------------------
    function rocsolver_dsyevd(handle, evect, uplo, n, A, lda, D, E, myInfo) &
       result(dsyevd) &
       bind(C, name="rocsolver_dsyevd")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dsyevd
    end function rocsolver_dsyevd

    !---------------------------------------------
    ! rocsolver_ssyevd_64
    !---------------------------------------------
    function rocsolver_ssyevd_64(handle, evect, uplo, n, A, lda, D, E, myInfo) &
       result(ssyevd_64) &
       bind(C, name="rocsolver_ssyevd_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ssyevd_64
    end function rocsolver_ssyevd_64

    !---------------------------------------------
    ! rocsolver_dsyevd_64
    !---------------------------------------------
    function rocsolver_dsyevd_64(handle, evect, uplo, n, A, lda, D, E, myInfo) &
       result(dsyevd_64) &
       bind(C, name="rocsolver_dsyevd_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dsyevd_64
    end function rocsolver_dsyevd_64

    !---------------------------------------------
    ! rocsolver_cheevd
    !---------------------------------------------
    function rocsolver_cheevd(handle, evect, uplo, n, A, lda, D, E, myInfo) &
       result(cheevd) &
       bind(C, name="rocsolver_cheevd")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cheevd
    end function rocsolver_cheevd

    !---------------------------------------------
    ! rocsolver_zheevd
    !---------------------------------------------
    function rocsolver_zheevd(handle, evect, uplo, n, A, lda, D, E, myInfo) &
       result(zheevd) &
       bind(C, name="rocsolver_zheevd")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zheevd
    end function rocsolver_zheevd

    !---------------------------------------------
    ! rocsolver_cheevd_64
    !---------------------------------------------
    function rocsolver_cheevd_64(handle, evect, uplo, n, A, lda, D, E, myInfo) &
       result(cheevd_64) &
       bind(C, name="rocsolver_cheevd_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cheevd_64
    end function rocsolver_cheevd_64

    !---------------------------------------------
    ! rocsolver_zheevd_64
    !---------------------------------------------
    function rocsolver_zheevd_64(handle, evect, uplo, n, A, lda, D, E, myInfo) &
       result(zheevd_64) &
       bind(C, name="rocsolver_zheevd_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zheevd_64
    end function rocsolver_zheevd_64

    !---------------------------------------------
    ! rocsolver_ssyevd_batched
    !---------------------------------------------
    function rocsolver_ssyevd_batched(handle, evect, uplo, n, A, lda, D, strideD, E, strideE, &
                                      myInfo, batch_count) &
       result(ssyevd_batched) &
       bind(C, name="rocsolver_ssyevd_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssyevd_batched
    end function rocsolver_ssyevd_batched

    !---------------------------------------------
    ! rocsolver_dsyevd_batched
    !---------------------------------------------
    function rocsolver_dsyevd_batched(handle, evect, uplo, n, A, lda, D, strideD, E, strideE, &
                                      myInfo, batch_count) &
       result(dsyevd_batched) &
       bind(C, name="rocsolver_dsyevd_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsyevd_batched
    end function rocsolver_dsyevd_batched

    !---------------------------------------------
    ! rocsolver_ssyevd_batched_64
    !---------------------------------------------
    function rocsolver_ssyevd_batched_64(handle, evect, uplo, n, A, lda, D, strideD, E, strideE, &
                                         myInfo, batch_count) &
       result(ssyevd_batched_64) &
       bind(C, name="rocsolver_ssyevd_batched_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssyevd_batched_64
    end function rocsolver_ssyevd_batched_64

    !---------------------------------------------
    ! rocsolver_dsyevd_batched_64
    !---------------------------------------------
    function rocsolver_dsyevd_batched_64(handle, evect, uplo, n, A, lda, D, strideD, E, strideE, &
                                         myInfo, batch_count) &
       result(dsyevd_batched_64) &
       bind(C, name="rocsolver_dsyevd_batched_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsyevd_batched_64
    end function rocsolver_dsyevd_batched_64

    !---------------------------------------------
    ! rocsolver_cheevd_batched
    !---------------------------------------------
    function rocsolver_cheevd_batched(handle, evect, uplo, n, A, lda, D, strideD, E, strideE, &
                                      myInfo, batch_count) &
       result(cheevd_batched) &
       bind(C, name="rocsolver_cheevd_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cheevd_batched
    end function rocsolver_cheevd_batched

    !---------------------------------------------
    ! rocsolver_zheevd_batched
    !---------------------------------------------
    function rocsolver_zheevd_batched(handle, evect, uplo, n, A, lda, D, strideD, E, strideE, &
                                      myInfo, batch_count) &
       result(zheevd_batched) &
       bind(C, name="rocsolver_zheevd_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zheevd_batched
    end function rocsolver_zheevd_batched

    !---------------------------------------------
    ! rocsolver_cheevd_batched_64
    !---------------------------------------------
    function rocsolver_cheevd_batched_64(handle, evect, uplo, n, A, lda, D, strideD, E, strideE, &
                                         myInfo, batch_count) &
       result(cheevd_batched_64) &
       bind(C, name="rocsolver_cheevd_batched_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cheevd_batched_64
    end function rocsolver_cheevd_batched_64

    !---------------------------------------------
    ! rocsolver_zheevd_batched_64
    !---------------------------------------------
    function rocsolver_zheevd_batched_64(handle, evect, uplo, n, A, lda, D, strideD, E, strideE, &
                                         myInfo, batch_count) &
       result(zheevd_batched_64) &
       bind(C, name="rocsolver_zheevd_batched_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zheevd_batched_64
    end function rocsolver_zheevd_batched_64

    !---------------------------------------------
    ! rocsolver_ssyevd_strided_batched
    !---------------------------------------------
    function rocsolver_ssyevd_strided_batched(handle, evect, uplo, n, A, lda, strideA, D, strideD, &
                                              E, strideE, myInfo, batch_count) &
       result(ssyevd_strided_batched) &
       bind(C, name="rocsolver_ssyevd_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssyevd_strided_batched
    end function rocsolver_ssyevd_strided_batched

    !---------------------------------------------
    ! rocsolver_dsyevd_strided_batched
    !---------------------------------------------
    function rocsolver_dsyevd_strided_batched(handle, evect, uplo, n, A, lda, strideA, D, strideD, &
                                              E, strideE, myInfo, batch_count) &
       result(dsyevd_strided_batched) &
       bind(C, name="rocsolver_dsyevd_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsyevd_strided_batched
    end function rocsolver_dsyevd_strided_batched

    !---------------------------------------------
    ! rocsolver_ssyevd_strided_batched_64
    !---------------------------------------------
    function rocsolver_ssyevd_strided_batched_64(handle, evect, uplo, n, A, lda, strideA, D, &
                                                 strideD, E, strideE, myInfo, batch_count) &
       result(ssyevd_strided_batched_64) &
       bind(C, name="rocsolver_ssyevd_strided_batched_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssyevd_strided_batched_64
    end function rocsolver_ssyevd_strided_batched_64

    !---------------------------------------------
    ! rocsolver_dsyevd_strided_batched_64
    !---------------------------------------------
    function rocsolver_dsyevd_strided_batched_64(handle, evect, uplo, n, A, lda, strideA, D, &
                                                 strideD, E, strideE, myInfo, batch_count) &
       result(dsyevd_strided_batched_64) &
       bind(C, name="rocsolver_dsyevd_strided_batched_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsyevd_strided_batched_64
    end function rocsolver_dsyevd_strided_batched_64

    !---------------------------------------------
    ! rocsolver_cheevd_strided_batched
    !---------------------------------------------
    function rocsolver_cheevd_strided_batched(handle, evect, uplo, n, A, lda, strideA, D, strideD, &
                                              E, strideE, myInfo, batch_count) &
       result(cheevd_strided_batched) &
       bind(C, name="rocsolver_cheevd_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cheevd_strided_batched
    end function rocsolver_cheevd_strided_batched

    !---------------------------------------------
    ! rocsolver_zheevd_strided_batched
    !---------------------------------------------
    function rocsolver_zheevd_strided_batched(handle, evect, uplo, n, A, lda, strideA, D, strideD, &
                                              E, strideE, myInfo, batch_count) &
       result(zheevd_strided_batched) &
       bind(C, name="rocsolver_zheevd_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zheevd_strided_batched
    end function rocsolver_zheevd_strided_batched

    !---------------------------------------------
    ! rocsolver_cheevd_strided_batched_64
    !---------------------------------------------
    function rocsolver_cheevd_strided_batched_64(handle, evect, uplo, n, A, lda, strideA, D, &
                                                 strideD, E, strideE, myInfo, batch_count) &
       result(cheevd_strided_batched_64) &
       bind(C, name="rocsolver_cheevd_strided_batched_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: cheevd_strided_batched_64
    end function rocsolver_cheevd_strided_batched_64

    !---------------------------------------------
    ! rocsolver_zheevd_strided_batched_64
    !---------------------------------------------
    function rocsolver_zheevd_strided_batched_64(handle, evect, uplo, n, A, lda, strideA, D, &
                                                 strideD, E, strideE, myInfo, batch_count) &
       result(zheevd_strided_batched_64) &
       bind(C, name="rocsolver_zheevd_strided_batched_64")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zheevd_strided_batched_64
    end function rocsolver_zheevd_strided_batched_64

    !---------------------------------------------
    ! rocsolver_ssyevdj
    !---------------------------------------------
    function rocsolver_ssyevdj(handle, evect, uplo, n, A, lda, D, myInfo) &
       result(ssyevdj) &
       bind(C, name="rocsolver_ssyevdj")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ssyevdj
    end function rocsolver_ssyevdj

    !---------------------------------------------
    ! rocsolver_dsyevdj
    !---------------------------------------------
    function rocsolver_dsyevdj(handle, evect, uplo, n, A, lda, D, myInfo) &
       result(dsyevdj) &
       bind(C, name="rocsolver_dsyevdj")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dsyevdj
    end function rocsolver_dsyevdj

    !---------------------------------------------
    ! rocsolver_cheevdj
    !---------------------------------------------
    function rocsolver_cheevdj(handle, evect, uplo, n, A, lda, D, myInfo) &
       result(cheevdj) &
       bind(C, name="rocsolver_cheevdj")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cheevdj
    end function rocsolver_cheevdj

    !---------------------------------------------
    ! rocsolver_zheevdj
    !---------------------------------------------
    function rocsolver_zheevdj(handle, evect, uplo, n, A, lda, D, myInfo) &
       result(zheevdj) &
       bind(C, name="rocsolver_zheevdj")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zheevdj
    end function rocsolver_zheevdj

    !---------------------------------------------
    ! rocsolver_ssyevdj_batched
    !---------------------------------------------
    function rocsolver_ssyevdj_batched(handle, evect, uplo, n, A, lda, D, strideD, myInfo, &
                                       batch_count) &
       result(ssyevdj_batched) &
       bind(C, name="rocsolver_ssyevdj_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssyevdj_batched
    end function rocsolver_ssyevdj_batched

    !---------------------------------------------
    ! rocsolver_dsyevdj_batched
    !---------------------------------------------
    function rocsolver_dsyevdj_batched(handle, evect, uplo, n, A, lda, D, strideD, myInfo, &
                                       batch_count) &
       result(dsyevdj_batched) &
       bind(C, name="rocsolver_dsyevdj_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsyevdj_batched
    end function rocsolver_dsyevdj_batched

    !---------------------------------------------
    ! rocsolver_cheevdj_batched
    !---------------------------------------------
    function rocsolver_cheevdj_batched(handle, evect, uplo, n, A, lda, D, strideD, myInfo, &
                                       batch_count) &
       result(cheevdj_batched) &
       bind(C, name="rocsolver_cheevdj_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cheevdj_batched
    end function rocsolver_cheevdj_batched

    !---------------------------------------------
    ! rocsolver_zheevdj_batched
    !---------------------------------------------
    function rocsolver_zheevdj_batched(handle, evect, uplo, n, A, lda, D, strideD, myInfo, &
                                       batch_count) &
       result(zheevdj_batched) &
       bind(C, name="rocsolver_zheevdj_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zheevdj_batched
    end function rocsolver_zheevdj_batched

    !---------------------------------------------
    ! rocsolver_ssyevdj_strided_batched
    !---------------------------------------------
    function rocsolver_ssyevdj_strided_batched(handle, evect, uplo, n, A, lda, strideA, D, &
                                               strideD, myInfo, batch_count) &
       result(ssyevdj_strided_batched) &
       bind(C, name="rocsolver_ssyevdj_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssyevdj_strided_batched
    end function rocsolver_ssyevdj_strided_batched

    !---------------------------------------------
    ! rocsolver_dsyevdj_strided_batched
    !---------------------------------------------
    function rocsolver_dsyevdj_strided_batched(handle, evect, uplo, n, A, lda, strideA, D, &
                                               strideD, myInfo, batch_count) &
       result(dsyevdj_strided_batched) &
       bind(C, name="rocsolver_dsyevdj_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsyevdj_strided_batched
    end function rocsolver_dsyevdj_strided_batched

    !---------------------------------------------
    ! rocsolver_cheevdj_strided_batched
    !---------------------------------------------
    function rocsolver_cheevdj_strided_batched(handle, evect, uplo, n, A, lda, strideA, D, &
                                               strideD, myInfo, batch_count) &
       result(cheevdj_strided_batched) &
       bind(C, name="rocsolver_cheevdj_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cheevdj_strided_batched
    end function rocsolver_cheevdj_strided_batched

    !---------------------------------------------
    ! rocsolver_zheevdj_strided_batched
    !---------------------------------------------
    function rocsolver_zheevdj_strided_batched(handle, evect, uplo, n, A, lda, strideA, D, &
                                               strideD, myInfo, batch_count) &
       result(zheevdj_strided_batched) &
       bind(C, name="rocsolver_zheevdj_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_fill_upper, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zheevdj_strided_batched
    end function rocsolver_zheevdj_strided_batched

    !---------------------------------------------
    ! rocsolver_ssygvdj
    !---------------------------------------------
    function rocsolver_ssygvdj(handle, itype, evect, uplo, n, A, lda, B, ldb, D, myInfo) &
       result(ssygvdj) &
       bind(C, name="rocsolver_ssygvdj")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ssygvdj
    end function rocsolver_ssygvdj

    !---------------------------------------------
    ! rocsolver_dsygvdj
    !---------------------------------------------
    function rocsolver_dsygvdj(handle, itype, evect, uplo, n, A, lda, B, ldb, D, myInfo) &
       result(dsygvdj) &
       bind(C, name="rocsolver_dsygvdj")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dsygvdj
    end function rocsolver_dsygvdj

    !---------------------------------------------
    ! rocsolver_chegvdj
    !---------------------------------------------
    function rocsolver_chegvdj(handle, itype, evect, uplo, n, A, lda, B, ldb, D, myInfo) &
       result(chegvdj) &
       bind(C, name="rocsolver_chegvdj")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: chegvdj
    end function rocsolver_chegvdj

    !---------------------------------------------
    ! rocsolver_zhegvdj
    !---------------------------------------------
    function rocsolver_zhegvdj(handle, itype, evect, uplo, n, A, lda, B, ldb, D, myInfo) &
       result(zhegvdj) &
       bind(C, name="rocsolver_zhegvdj")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zhegvdj
    end function rocsolver_zhegvdj

    !---------------------------------------------
    ! rocsolver_ssygvdj_batched
    !---------------------------------------------
    function rocsolver_ssygvdj_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, strideD, &
                                       myInfo, batch_count) &
       result(ssygvdj_batched) &
       bind(C, name="rocsolver_ssygvdj_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssygvdj_batched
    end function rocsolver_ssygvdj_batched

    !---------------------------------------------
    ! rocsolver_dsygvdj_batched
    !---------------------------------------------
    function rocsolver_dsygvdj_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, strideD, &
                                       myInfo, batch_count) &
       result(dsygvdj_batched) &
       bind(C, name="rocsolver_dsygvdj_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsygvdj_batched
    end function rocsolver_dsygvdj_batched

    !---------------------------------------------
    ! rocsolver_chegvdj_batched
    !---------------------------------------------
    function rocsolver_chegvdj_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, strideD, &
                                       myInfo, batch_count) &
       result(chegvdj_batched) &
       bind(C, name="rocsolver_chegvdj_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chegvdj_batched
    end function rocsolver_chegvdj_batched

    !---------------------------------------------
    ! rocsolver_zhegvdj_batched
    !---------------------------------------------
    function rocsolver_zhegvdj_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, strideD, &
                                       myInfo, batch_count) &
       result(zhegvdj_batched) &
       bind(C, name="rocsolver_zhegvdj_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhegvdj_batched
    end function rocsolver_zhegvdj_batched

    !---------------------------------------------
    ! rocsolver_ssygvdj_strided_batched
    !---------------------------------------------
    function rocsolver_ssygvdj_strided_batched(handle, itype, evect, uplo, n, A, lda, strideA, B, &
                                               ldb, strideB, D, strideD, myInfo, batch_count) &
       result(ssygvdj_strided_batched) &
       bind(C, name="rocsolver_ssygvdj_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssygvdj_strided_batched
    end function rocsolver_ssygvdj_strided_batched

    !---------------------------------------------
    ! rocsolver_dsygvdj_strided_batched
    !---------------------------------------------
    function rocsolver_dsygvdj_strided_batched(handle, itype, evect, uplo, n, A, lda, strideA, B, &
                                               ldb, strideB, D, strideD, myInfo, batch_count) &
       result(dsygvdj_strided_batched) &
       bind(C, name="rocsolver_dsygvdj_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsygvdj_strided_batched
    end function rocsolver_dsygvdj_strided_batched

    !---------------------------------------------
    ! rocsolver_chegvdj_strided_batched
    !---------------------------------------------
    function rocsolver_chegvdj_strided_batched(handle, itype, evect, uplo, n, A, lda, strideA, B, &
                                               ldb, strideB, D, strideD, myInfo, batch_count) &
       result(chegvdj_strided_batched) &
       bind(C, name="rocsolver_chegvdj_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chegvdj_strided_batched
    end function rocsolver_chegvdj_strided_batched

    !---------------------------------------------
    ! rocsolver_zhegvdj_strided_batched
    !---------------------------------------------
    function rocsolver_zhegvdj_strided_batched(handle, itype, evect, uplo, n, A, lda, strideA, B, &
                                               ldb, strideB, D, strideD, myInfo, batch_count) &
       result(zhegvdj_strided_batched) &
       bind(C, name="rocsolver_zhegvdj_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhegvdj_strided_batched
    end function rocsolver_zhegvdj_strided_batched

    !---------------------------------------------
    ! rocsolver_ssyevj
    !---------------------------------------------
    function rocsolver_ssyevj(handle, esort, evect, uplo, n, A, lda, abstol, residual, max_sweeps, &
                              n_sweeps, W, myInfo) &
       result(ssyevj) &
       bind(C, name="rocsolver_ssyevj")
       import :: c_ptr, rocblas_esort_none, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_esort_none)), value :: esort
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ssyevj
    end function rocsolver_ssyevj

    !---------------------------------------------
    ! rocsolver_dsyevj
    !---------------------------------------------
    function rocsolver_dsyevj(handle, esort, evect, uplo, n, A, lda, abstol, residual, max_sweeps, &
                              n_sweeps, W, myInfo) &
       result(dsyevj) &
       bind(C, name="rocsolver_dsyevj")
       import :: c_ptr, rocblas_esort_none, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_esort_none)), value :: esort
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dsyevj
    end function rocsolver_dsyevj

    !---------------------------------------------
    ! rocsolver_cheevj
    !---------------------------------------------
    function rocsolver_cheevj(handle, esort, evect, uplo, n, A, lda, abstol, residual, max_sweeps, &
                              n_sweeps, W, myInfo) &
       result(cheevj) &
       bind(C, name="rocsolver_cheevj")
       import :: c_ptr, rocblas_esort_none, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_esort_none)), value :: esort
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cheevj
    end function rocsolver_cheevj

    !---------------------------------------------
    ! rocsolver_zheevj
    !---------------------------------------------
    function rocsolver_zheevj(handle, esort, evect, uplo, n, A, lda, abstol, residual, max_sweeps, &
                              n_sweeps, W, myInfo) &
       result(zheevj) &
       bind(C, name="rocsolver_zheevj")
       import :: c_ptr, rocblas_esort_none, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_esort_none)), value :: esort
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zheevj
    end function rocsolver_zheevj

    !---------------------------------------------
    ! rocsolver_ssyevj_batched
    !---------------------------------------------
    function rocsolver_ssyevj_batched(handle, esort, evect, uplo, n, A, lda, abstol, residual, &
                                      max_sweeps, n_sweeps, W, strideW, myInfo, batch_count) &
       result(ssyevj_batched) &
       bind(C, name="rocsolver_ssyevj_batched")
       import :: c_ptr, rocblas_esort_none, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_float, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_esort_none)), value :: esort
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssyevj_batched
    end function rocsolver_ssyevj_batched

    !---------------------------------------------
    ! rocsolver_dsyevj_batched
    !---------------------------------------------
    function rocsolver_dsyevj_batched(handle, esort, evect, uplo, n, A, lda, abstol, residual, &
                                      max_sweeps, n_sweeps, W, strideW, myInfo, batch_count) &
       result(dsyevj_batched) &
       bind(C, name="rocsolver_dsyevj_batched")
       import :: c_ptr, rocblas_esort_none, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_double, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_esort_none)), value :: esort
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsyevj_batched
    end function rocsolver_dsyevj_batched

    !---------------------------------------------
    ! rocsolver_cheevj_batched
    !---------------------------------------------
    function rocsolver_cheevj_batched(handle, esort, evect, uplo, n, A, lda, abstol, residual, &
                                      max_sweeps, n_sweeps, W, strideW, myInfo, batch_count) &
       result(cheevj_batched) &
       bind(C, name="rocsolver_cheevj_batched")
       import :: c_ptr, rocblas_esort_none, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_float, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_esort_none)), value :: esort
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cheevj_batched
    end function rocsolver_cheevj_batched

    !---------------------------------------------
    ! rocsolver_zheevj_batched
    !---------------------------------------------
    function rocsolver_zheevj_batched(handle, esort, evect, uplo, n, A, lda, abstol, residual, &
                                      max_sweeps, n_sweeps, W, strideW, myInfo, batch_count) &
       result(zheevj_batched) &
       bind(C, name="rocsolver_zheevj_batched")
       import :: c_ptr, rocblas_esort_none, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_double, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_esort_none)), value :: esort
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zheevj_batched
    end function rocsolver_zheevj_batched

    !---------------------------------------------
    ! rocsolver_ssyevj_strided_batched
    !---------------------------------------------
    function rocsolver_ssyevj_strided_batched(handle, esort, evect, uplo, n, A, lda, strideA, &
                                              abstol, residual, max_sweeps, n_sweeps, W, strideW, &
                                              myInfo, batch_count) &
       result(ssyevj_strided_batched) &
       bind(C, name="rocsolver_ssyevj_strided_batched")
       import :: c_ptr, rocblas_esort_none, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_esort_none)), value :: esort
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_float), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssyevj_strided_batched
    end function rocsolver_ssyevj_strided_batched

    !---------------------------------------------
    ! rocsolver_dsyevj_strided_batched
    !---------------------------------------------
    function rocsolver_dsyevj_strided_batched(handle, esort, evect, uplo, n, A, lda, strideA, &
                                              abstol, residual, max_sweeps, n_sweeps, W, strideW, &
                                              myInfo, batch_count) &
       result(dsyevj_strided_batched) &
       bind(C, name="rocsolver_dsyevj_strided_batched")
       import :: c_ptr, rocblas_esort_none, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_esort_none)), value :: esort
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_double), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsyevj_strided_batched
    end function rocsolver_dsyevj_strided_batched

    !---------------------------------------------
    ! rocsolver_cheevj_strided_batched
    !---------------------------------------------
    function rocsolver_cheevj_strided_batched(handle, esort, evect, uplo, n, A, lda, strideA, &
                                              abstol, residual, max_sweeps, n_sweeps, W, strideW, &
                                              myInfo, batch_count) &
       result(cheevj_strided_batched) &
       bind(C, name="rocsolver_cheevj_strided_batched")
       import :: c_ptr, rocblas_esort_none, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_esort_none)), value :: esort
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_float), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cheevj_strided_batched
    end function rocsolver_cheevj_strided_batched

    !---------------------------------------------
    ! rocsolver_zheevj_strided_batched
    !---------------------------------------------
    function rocsolver_zheevj_strided_batched(handle, esort, evect, uplo, n, A, lda, strideA, &
                                              abstol, residual, max_sweeps, n_sweeps, W, strideW, &
                                              myInfo, batch_count) &
       result(zheevj_strided_batched) &
       bind(C, name="rocsolver_zheevj_strided_batched")
       import :: c_ptr, rocblas_esort_none, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_esort_none)), value :: esort
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_double), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zheevj_strided_batched
    end function rocsolver_zheevj_strided_batched

    !---------------------------------------------
    ! rocsolver_ssyevx
    !---------------------------------------------
    function rocsolver_ssyevx(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol, nev, &
                              W, Z, ldz, ifail, myInfo) &
       result(ssyevx) &
       bind(C, name="rocsolver_ssyevx")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_float), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ssyevx
    end function rocsolver_ssyevx

    !---------------------------------------------
    ! rocsolver_dsyevx
    !---------------------------------------------
    function rocsolver_dsyevx(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol, nev, &
                              W, Z, ldz, ifail, myInfo) &
       result(dsyevx) &
       bind(C, name="rocsolver_dsyevx")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_double), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dsyevx
    end function rocsolver_dsyevx

    !---------------------------------------------
    ! rocsolver_cheevx
    !---------------------------------------------
    function rocsolver_cheevx(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol, nev, &
                              W, Z, ldz, ifail, myInfo) &
       result(cheevx) &
       bind(C, name="rocsolver_cheevx")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_float), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cheevx
    end function rocsolver_cheevx

    !---------------------------------------------
    ! rocsolver_zheevx
    !---------------------------------------------
    function rocsolver_zheevx(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol, nev, &
                              W, Z, ldz, ifail, myInfo) &
       result(zheevx) &
       bind(C, name="rocsolver_zheevx")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_double), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zheevx
    end function rocsolver_zheevx

    !---------------------------------------------
    ! rocsolver_ssyevx_batched
    !---------------------------------------------
    function rocsolver_ssyevx_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, &
                                      abstol, nev, W, strideW, Z, ldz, ifail, strideF, myInfo, &
                                      batch_count) &
       result(ssyevx_batched) &
       bind(C, name="rocsolver_ssyevx_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_float, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_float), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssyevx_batched
    end function rocsolver_ssyevx_batched

    !---------------------------------------------
    ! rocsolver_dsyevx_batched
    !---------------------------------------------
    function rocsolver_dsyevx_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, &
                                      abstol, nev, W, strideW, Z, ldz, ifail, strideF, myInfo, &
                                      batch_count) &
       result(dsyevx_batched) &
       bind(C, name="rocsolver_dsyevx_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_double, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_double), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsyevx_batched
    end function rocsolver_dsyevx_batched

    !---------------------------------------------
    ! rocsolver_cheevx_batched
    !---------------------------------------------
    function rocsolver_cheevx_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, &
                                      abstol, nev, W, strideW, Z, ldz, ifail, strideF, myInfo, &
                                      batch_count) &
       result(cheevx_batched) &
       bind(C, name="rocsolver_cheevx_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_float, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_float), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cheevx_batched
    end function rocsolver_cheevx_batched

    !---------------------------------------------
    ! rocsolver_zheevx_batched
    !---------------------------------------------
    function rocsolver_zheevx_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, &
                                      abstol, nev, W, strideW, Z, ldz, ifail, strideF, myInfo, &
                                      batch_count) &
       result(zheevx_batched) &
       bind(C, name="rocsolver_zheevx_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_double, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_double), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zheevx_batched
    end function rocsolver_zheevx_batched

    !---------------------------------------------
    ! rocsolver_ssyevx_strided_batched
    !---------------------------------------------
    function rocsolver_ssyevx_strided_batched(handle, evect, erange, uplo, n, A, lda, strideA, vl, &
                                              vu, il, iu, abstol, nev, W, strideW, Z, ldz, &
                                              strideZ, ifail, strideF, myInfo, batch_count) &
       result(ssyevx_strided_batched) &
       bind(C, name="rocsolver_ssyevx_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_int64_t, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_float), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       integer(c_int64_t), value :: strideZ
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssyevx_strided_batched
    end function rocsolver_ssyevx_strided_batched

    !---------------------------------------------
    ! rocsolver_dsyevx_strided_batched
    !---------------------------------------------
    function rocsolver_dsyevx_strided_batched(handle, evect, erange, uplo, n, A, lda, strideA, vl, &
                                              vu, il, iu, abstol, nev, W, strideW, Z, ldz, &
                                              strideZ, ifail, strideF, myInfo, batch_count) &
       result(dsyevx_strided_batched) &
       bind(C, name="rocsolver_dsyevx_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_int64_t, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_double), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       integer(c_int64_t), value :: strideZ
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsyevx_strided_batched
    end function rocsolver_dsyevx_strided_batched

    !---------------------------------------------
    ! rocsolver_cheevx_strided_batched
    !---------------------------------------------
    function rocsolver_cheevx_strided_batched(handle, evect, erange, uplo, n, A, lda, strideA, vl, &
                                              vu, il, iu, abstol, nev, W, strideW, Z, ldz, &
                                              strideZ, ifail, strideF, myInfo, batch_count) &
       result(cheevx_strided_batched) &
       bind(C, name="rocsolver_cheevx_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_int64_t, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_float), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       integer(c_int64_t), value :: strideZ
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cheevx_strided_batched
    end function rocsolver_cheevx_strided_batched

    !---------------------------------------------
    ! rocsolver_zheevx_strided_batched
    !---------------------------------------------
    function rocsolver_zheevx_strided_batched(handle, evect, erange, uplo, n, A, lda, strideA, vl, &
                                              vu, il, iu, abstol, nev, W, strideW, Z, ldz, &
                                              strideZ, ifail, strideF, myInfo, batch_count) &
       result(zheevx_strided_batched) &
       bind(C, name="rocsolver_zheevx_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_int64_t, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_double), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       integer(c_int64_t), value :: strideZ
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zheevx_strided_batched
    end function rocsolver_zheevx_strided_batched

    !---------------------------------------------
    ! rocsolver_ssygv
    !---------------------------------------------
    function rocsolver_ssygv(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, myInfo) &
       result(ssygv) &
       bind(C, name="rocsolver_ssygv")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ssygv
    end function rocsolver_ssygv

    !---------------------------------------------
    ! rocsolver_dsygv
    !---------------------------------------------
    function rocsolver_dsygv(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, myInfo) &
       result(dsygv) &
       bind(C, name="rocsolver_dsygv")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dsygv
    end function rocsolver_dsygv

    !---------------------------------------------
    ! rocsolver_chegv
    !---------------------------------------------
    function rocsolver_chegv(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, myInfo) &
       result(chegv) &
       bind(C, name="rocsolver_chegv")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: chegv
    end function rocsolver_chegv

    !---------------------------------------------
    ! rocsolver_zhegv
    !---------------------------------------------
    function rocsolver_zhegv(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, myInfo) &
       result(zhegv) &
       bind(C, name="rocsolver_zhegv")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zhegv
    end function rocsolver_zhegv

    !---------------------------------------------
    ! rocsolver_ssygv_batched
    !---------------------------------------------
    function rocsolver_ssygv_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, strideD, E, &
                                     strideE, myInfo, batch_count) &
       result(ssygv_batched) &
       bind(C, name="rocsolver_ssygv_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssygv_batched
    end function rocsolver_ssygv_batched

    !---------------------------------------------
    ! rocsolver_dsygv_batched
    !---------------------------------------------
    function rocsolver_dsygv_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, strideD, E, &
                                     strideE, myInfo, batch_count) &
       result(dsygv_batched) &
       bind(C, name="rocsolver_dsygv_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsygv_batched
    end function rocsolver_dsygv_batched

    !---------------------------------------------
    ! rocsolver_chegv_batched
    !---------------------------------------------
    function rocsolver_chegv_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, strideD, E, &
                                     strideE, myInfo, batch_count) &
       result(chegv_batched) &
       bind(C, name="rocsolver_chegv_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chegv_batched
    end function rocsolver_chegv_batched

    !---------------------------------------------
    ! rocsolver_zhegv_batched
    !---------------------------------------------
    function rocsolver_zhegv_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, strideD, E, &
                                     strideE, myInfo, batch_count) &
       result(zhegv_batched) &
       bind(C, name="rocsolver_zhegv_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhegv_batched
    end function rocsolver_zhegv_batched

    !---------------------------------------------
    ! rocsolver_ssygv_strided_batched
    !---------------------------------------------
    function rocsolver_ssygv_strided_batched(handle, itype, evect, uplo, n, A, lda, strideA, B, &
                                             ldb, strideB, D, strideD, E, strideE, myInfo, &
                                             batch_count) &
       result(ssygv_strided_batched) &
       bind(C, name="rocsolver_ssygv_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssygv_strided_batched
    end function rocsolver_ssygv_strided_batched

    !---------------------------------------------
    ! rocsolver_dsygv_strided_batched
    !---------------------------------------------
    function rocsolver_dsygv_strided_batched(handle, itype, evect, uplo, n, A, lda, strideA, B, &
                                             ldb, strideB, D, strideD, E, strideE, myInfo, &
                                             batch_count) &
       result(dsygv_strided_batched) &
       bind(C, name="rocsolver_dsygv_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsygv_strided_batched
    end function rocsolver_dsygv_strided_batched

    !---------------------------------------------
    ! rocsolver_chegv_strided_batched
    !---------------------------------------------
    function rocsolver_chegv_strided_batched(handle, itype, evect, uplo, n, A, lda, strideA, B, &
                                             ldb, strideB, D, strideD, E, strideE, myInfo, &
                                             batch_count) &
       result(chegv_strided_batched) &
       bind(C, name="rocsolver_chegv_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chegv_strided_batched
    end function rocsolver_chegv_strided_batched

    !---------------------------------------------
    ! rocsolver_zhegv_strided_batched
    !---------------------------------------------
    function rocsolver_zhegv_strided_batched(handle, itype, evect, uplo, n, A, lda, strideA, B, &
                                             ldb, strideB, D, strideD, E, strideE, myInfo, &
                                             batch_count) &
       result(zhegv_strided_batched) &
       bind(C, name="rocsolver_zhegv_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhegv_strided_batched
    end function rocsolver_zhegv_strided_batched

    !---------------------------------------------
    ! rocsolver_ssygvd
    !---------------------------------------------
    function rocsolver_ssygvd(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, myInfo) &
       result(ssygvd) &
       bind(C, name="rocsolver_ssygvd")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ssygvd
    end function rocsolver_ssygvd

    !---------------------------------------------
    ! rocsolver_dsygvd
    !---------------------------------------------
    function rocsolver_dsygvd(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, myInfo) &
       result(dsygvd) &
       bind(C, name="rocsolver_dsygvd")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dsygvd
    end function rocsolver_dsygvd

    !---------------------------------------------
    ! rocsolver_chegvd
    !---------------------------------------------
    function rocsolver_chegvd(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, myInfo) &
       result(chegvd) &
       bind(C, name="rocsolver_chegvd")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: chegvd
    end function rocsolver_chegvd

    !---------------------------------------------
    ! rocsolver_zhegvd
    !---------------------------------------------
    function rocsolver_zhegvd(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, myInfo) &
       result(zhegvd) &
       bind(C, name="rocsolver_zhegvd")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zhegvd
    end function rocsolver_zhegvd

    !---------------------------------------------
    ! rocsolver_ssygvd_batched
    !---------------------------------------------
    function rocsolver_ssygvd_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, strideD, &
                                      E, strideE, myInfo, batch_count) &
       result(ssygvd_batched) &
       bind(C, name="rocsolver_ssygvd_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssygvd_batched
    end function rocsolver_ssygvd_batched

    !---------------------------------------------
    ! rocsolver_dsygvd_batched
    !---------------------------------------------
    function rocsolver_dsygvd_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, strideD, &
                                      E, strideE, myInfo, batch_count) &
       result(dsygvd_batched) &
       bind(C, name="rocsolver_dsygvd_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsygvd_batched
    end function rocsolver_dsygvd_batched

    !---------------------------------------------
    ! rocsolver_chegvd_batched
    !---------------------------------------------
    function rocsolver_chegvd_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, strideD, &
                                      E, strideE, myInfo, batch_count) &
       result(chegvd_batched) &
       bind(C, name="rocsolver_chegvd_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chegvd_batched
    end function rocsolver_chegvd_batched

    !---------------------------------------------
    ! rocsolver_zhegvd_batched
    !---------------------------------------------
    function rocsolver_zhegvd_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, D, strideD, &
                                      E, strideE, myInfo, batch_count) &
       result(zhegvd_batched) &
       bind(C, name="rocsolver_zhegvd_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhegvd_batched
    end function rocsolver_zhegvd_batched

    !---------------------------------------------
    ! rocsolver_ssygvd_strided_batched
    !---------------------------------------------
    function rocsolver_ssygvd_strided_batched(handle, itype, evect, uplo, n, A, lda, strideA, B, &
                                              ldb, strideB, D, strideD, E, strideE, myInfo, &
                                              batch_count) &
       result(ssygvd_strided_batched) &
       bind(C, name="rocsolver_ssygvd_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssygvd_strided_batched
    end function rocsolver_ssygvd_strided_batched

    !---------------------------------------------
    ! rocsolver_dsygvd_strided_batched
    !---------------------------------------------
    function rocsolver_dsygvd_strided_batched(handle, itype, evect, uplo, n, A, lda, strideA, B, &
                                              ldb, strideB, D, strideD, E, strideE, myInfo, &
                                              batch_count) &
       result(dsygvd_strided_batched) &
       bind(C, name="rocsolver_dsygvd_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsygvd_strided_batched
    end function rocsolver_dsygvd_strided_batched

    !---------------------------------------------
    ! rocsolver_chegvd_strided_batched
    !---------------------------------------------
    function rocsolver_chegvd_strided_batched(handle, itype, evect, uplo, n, A, lda, strideA, B, &
                                              ldb, strideB, D, strideD, E, strideE, myInfo, &
                                              batch_count) &
       result(chegvd_strided_batched) &
       bind(C, name="rocsolver_chegvd_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chegvd_strided_batched
    end function rocsolver_chegvd_strided_batched

    !---------------------------------------------
    ! rocsolver_zhegvd_strided_batched
    !---------------------------------------------
    function rocsolver_zhegvd_strided_batched(handle, itype, evect, uplo, n, A, lda, strideA, B, &
                                              ldb, strideB, D, strideD, E, strideE, myInfo, &
                                              batch_count) &
       result(zhegvd_strided_batched) &
       bind(C, name="rocsolver_zhegvd_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: D
       integer(c_int64_t), value :: strideD
       type(c_ptr), value :: E
       integer(c_int64_t), value :: strideE
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhegvd_strided_batched
    end function rocsolver_zhegvd_strided_batched

    !---------------------------------------------
    ! rocsolver_ssygvj
    !---------------------------------------------
    function rocsolver_ssygvj(handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, residual, &
                              max_sweeps, n_sweeps, W, myInfo) &
       result(ssygvj) &
       bind(C, name="rocsolver_ssygvj")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, c_float, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ssygvj
    end function rocsolver_ssygvj

    !---------------------------------------------
    ! rocsolver_dsygvj
    !---------------------------------------------
    function rocsolver_dsygvj(handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, residual, &
                              max_sweeps, n_sweeps, W, myInfo) &
       result(dsygvj) &
       bind(C, name="rocsolver_dsygvj")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dsygvj
    end function rocsolver_dsygvj

    !---------------------------------------------
    ! rocsolver_chegvj
    !---------------------------------------------
    function rocsolver_chegvj(handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, residual, &
                              max_sweeps, n_sweeps, W, myInfo) &
       result(chegvj) &
       bind(C, name="rocsolver_chegvj")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, c_float, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: chegvj
    end function rocsolver_chegvj

    !---------------------------------------------
    ! rocsolver_zhegvj
    !---------------------------------------------
    function rocsolver_zhegvj(handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, residual, &
                              max_sweeps, n_sweeps, W, myInfo) &
       result(zhegvj) &
       bind(C, name="rocsolver_zhegvj")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zhegvj
    end function rocsolver_zhegvj

    !---------------------------------------------
    ! rocsolver_ssygvj_batched
    !---------------------------------------------
    function rocsolver_ssygvj_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, &
                                      residual, max_sweeps, n_sweeps, W, strideW, myInfo, &
                                      batch_count) &
       result(ssygvj_batched) &
       bind(C, name="rocsolver_ssygvj_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, c_float, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssygvj_batched
    end function rocsolver_ssygvj_batched

    !---------------------------------------------
    ! rocsolver_dsygvj_batched
    !---------------------------------------------
    function rocsolver_dsygvj_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, &
                                      residual, max_sweeps, n_sweeps, W, strideW, myInfo, &
                                      batch_count) &
       result(dsygvj_batched) &
       bind(C, name="rocsolver_dsygvj_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_double, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsygvj_batched
    end function rocsolver_dsygvj_batched

    !---------------------------------------------
    ! rocsolver_chegvj_batched
    !---------------------------------------------
    function rocsolver_chegvj_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, &
                                      residual, max_sweeps, n_sweeps, W, strideW, myInfo, &
                                      batch_count) &
       result(chegvj_batched) &
       bind(C, name="rocsolver_chegvj_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, c_float, &
                 c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chegvj_batched
    end function rocsolver_chegvj_batched

    !---------------------------------------------
    ! rocsolver_zhegvj_batched
    !---------------------------------------------
    function rocsolver_zhegvj_batched(handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, &
                                      residual, max_sweeps, n_sweeps, W, strideW, myInfo, &
                                      batch_count) &
       result(zhegvj_batched) &
       bind(C, name="rocsolver_zhegvj_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_double, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhegvj_batched
    end function rocsolver_zhegvj_batched

    !---------------------------------------------
    ! rocsolver_ssygvj_strided_batched
    !---------------------------------------------
    function rocsolver_ssygvj_strided_batched(handle, itype, evect, uplo, n, A, lda, strideA, B, &
                                              ldb, strideB, abstol, residual, max_sweeps, &
                                              n_sweeps, W, strideW, myInfo, batch_count) &
       result(ssygvj_strided_batched) &
       bind(C, name="rocsolver_ssygvj_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       real(c_float), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssygvj_strided_batched
    end function rocsolver_ssygvj_strided_batched

    !---------------------------------------------
    ! rocsolver_dsygvj_strided_batched
    !---------------------------------------------
    function rocsolver_dsygvj_strided_batched(handle, itype, evect, uplo, n, A, lda, strideA, B, &
                                              ldb, strideB, abstol, residual, max_sweeps, &
                                              n_sweeps, W, strideW, myInfo, batch_count) &
       result(dsygvj_strided_batched) &
       bind(C, name="rocsolver_dsygvj_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       real(c_double), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsygvj_strided_batched
    end function rocsolver_dsygvj_strided_batched

    !---------------------------------------------
    ! rocsolver_chegvj_strided_batched
    !---------------------------------------------
    function rocsolver_chegvj_strided_batched(handle, itype, evect, uplo, n, A, lda, strideA, B, &
                                              ldb, strideB, abstol, residual, max_sweeps, &
                                              n_sweeps, W, strideW, myInfo, batch_count) &
       result(chegvj_strided_batched) &
       bind(C, name="rocsolver_chegvj_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       real(c_float), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chegvj_strided_batched
    end function rocsolver_chegvj_strided_batched

    !---------------------------------------------
    ! rocsolver_zhegvj_strided_batched
    !---------------------------------------------
    function rocsolver_zhegvj_strided_batched(handle, itype, evect, uplo, n, A, lda, strideA, B, &
                                              ldb, strideB, abstol, residual, max_sweeps, &
                                              n_sweeps, W, strideW, myInfo, batch_count) &
       result(zhegvj_strided_batched) &
       bind(C, name="rocsolver_zhegvj_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_fill_upper, c_int, &
                 c_int64_t, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       real(c_double), value :: abstol
       type(c_ptr), value :: residual
       integer(c_int), value :: max_sweeps
       type(c_ptr), value :: n_sweeps
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhegvj_strided_batched
    end function rocsolver_zhegvj_strided_batched

    !---------------------------------------------
    ! rocsolver_ssygvx
    !---------------------------------------------
    function rocsolver_ssygvx(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, &
                              iu, abstol, nev, W, Z, ldz, ifail, myInfo) &
       result(ssygvx) &
       bind(C, name="rocsolver_ssygvx")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_float), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ssygvx
    end function rocsolver_ssygvx

    !---------------------------------------------
    ! rocsolver_dsygvx
    !---------------------------------------------
    function rocsolver_dsygvx(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, &
                              iu, abstol, nev, W, Z, ldz, ifail, myInfo) &
       result(dsygvx) &
       bind(C, name="rocsolver_dsygvx")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_double), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dsygvx
    end function rocsolver_dsygvx

    !---------------------------------------------
    ! rocsolver_chegvx
    !---------------------------------------------
    function rocsolver_chegvx(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, &
                              iu, abstol, nev, W, Z, ldz, ifail, myInfo) &
       result(chegvx) &
       bind(C, name="rocsolver_chegvx")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_float), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: chegvx
    end function rocsolver_chegvx

    !---------------------------------------------
    ! rocsolver_zhegvx
    !---------------------------------------------
    function rocsolver_zhegvx(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, &
                              iu, abstol, nev, W, Z, ldz, ifail, myInfo) &
       result(zhegvx) &
       bind(C, name="rocsolver_zhegvx")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_double), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zhegvx
    end function rocsolver_zhegvx

    !---------------------------------------------
    ! rocsolver_ssygvx_batched
    !---------------------------------------------
    function rocsolver_ssygvx_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, &
                                      vu, il, iu, abstol, nev, W, strideW, Z, ldz, ifail, strideF, &
                                      myInfo, batch_count) &
       result(ssygvx_batched) &
       bind(C, name="rocsolver_ssygvx_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_float, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_float), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssygvx_batched
    end function rocsolver_ssygvx_batched

    !---------------------------------------------
    ! rocsolver_dsygvx_batched
    !---------------------------------------------
    function rocsolver_dsygvx_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, &
                                      vu, il, iu, abstol, nev, W, strideW, Z, ldz, ifail, strideF, &
                                      myInfo, batch_count) &
       result(dsygvx_batched) &
       bind(C, name="rocsolver_dsygvx_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_double, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_double), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsygvx_batched
    end function rocsolver_dsygvx_batched

    !---------------------------------------------
    ! rocsolver_chegvx_batched
    !---------------------------------------------
    function rocsolver_chegvx_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, &
                                      vu, il, iu, abstol, nev, W, strideW, Z, ldz, ifail, strideF, &
                                      myInfo, batch_count) &
       result(chegvx_batched) &
       bind(C, name="rocsolver_chegvx_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_float, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_float), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chegvx_batched
    end function rocsolver_chegvx_batched

    !---------------------------------------------
    ! rocsolver_zhegvx_batched
    !---------------------------------------------
    function rocsolver_zhegvx_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, &
                                      vu, il, iu, abstol, nev, W, strideW, Z, ldz, ifail, strideF, &
                                      myInfo, batch_count) &
       result(zhegvx_batched) &
       bind(C, name="rocsolver_zhegvx_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_double, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_double), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhegvx_batched
    end function rocsolver_zhegvx_batched

    !---------------------------------------------
    ! rocsolver_ssygvx_strided_batched
    !---------------------------------------------
    function rocsolver_ssygvx_strided_batched(handle, itype, evect, erange, uplo, n, A, lda, &
                                              strideA, B, ldb, strideB, vl, vu, il, iu, abstol, &
                                              nev, W, strideW, Z, ldz, strideZ, ifail, strideF, &
                                              myInfo, batch_count) &
       result(ssygvx_strided_batched) &
       bind(C, name="rocsolver_ssygvx_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_int64_t, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_float), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       integer(c_int64_t), value :: strideZ
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssygvx_strided_batched
    end function rocsolver_ssygvx_strided_batched

    !---------------------------------------------
    ! rocsolver_dsygvx_strided_batched
    !---------------------------------------------
    function rocsolver_dsygvx_strided_batched(handle, itype, evect, erange, uplo, n, A, lda, &
                                              strideA, B, ldb, strideB, vl, vu, il, iu, abstol, &
                                              nev, W, strideW, Z, ldz, strideZ, ifail, strideF, &
                                              myInfo, batch_count) &
       result(dsygvx_strided_batched) &
       bind(C, name="rocsolver_dsygvx_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_int64_t, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_double), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       integer(c_int64_t), value :: strideZ
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsygvx_strided_batched
    end function rocsolver_dsygvx_strided_batched

    !---------------------------------------------
    ! rocsolver_chegvx_strided_batched
    !---------------------------------------------
    function rocsolver_chegvx_strided_batched(handle, itype, evect, erange, uplo, n, A, lda, &
                                              strideA, B, ldb, strideB, vl, vu, il, iu, abstol, &
                                              nev, W, strideW, Z, ldz, strideZ, ifail, strideF, &
                                              myInfo, batch_count) &
       result(chegvx_strided_batched) &
       bind(C, name="rocsolver_chegvx_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_int64_t, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_float), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       integer(c_int64_t), value :: strideZ
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chegvx_strided_batched
    end function rocsolver_chegvx_strided_batched

    !---------------------------------------------
    ! rocsolver_zhegvx_strided_batched
    !---------------------------------------------
    function rocsolver_zhegvx_strided_batched(handle, itype, evect, erange, uplo, n, A, lda, &
                                              strideA, B, ldb, strideB, vl, vu, il, iu, abstol, &
                                              nev, W, strideW, Z, ldz, strideZ, ifail, strideF, &
                                              myInfo, batch_count) &
       result(zhegvx_strided_batched) &
       bind(C, name="rocsolver_zhegvx_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_int64_t, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       real(c_double), value :: abstol
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       integer(c_int64_t), value :: strideZ
       type(c_ptr), value :: ifail
       integer(c_int64_t), value :: strideF
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhegvx_strided_batched
    end function rocsolver_zhegvx_strided_batched

    !---------------------------------------------
    ! rocsolver_sgetri_outofplace
    !---------------------------------------------
    function rocsolver_sgetri_outofplace(handle, n, A, lda, ipiv, C, ldc, myInfo) &
       result(sgetri_outofplace) &
       bind(C, name="rocsolver_sgetri_outofplace")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgetri_outofplace
    end function rocsolver_sgetri_outofplace

    !---------------------------------------------
    ! rocsolver_dgetri_outofplace
    !---------------------------------------------
    function rocsolver_dgetri_outofplace(handle, n, A, lda, ipiv, C, ldc, myInfo) &
       result(dgetri_outofplace) &
       bind(C, name="rocsolver_dgetri_outofplace")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgetri_outofplace
    end function rocsolver_dgetri_outofplace

    !---------------------------------------------
    ! rocsolver_cgetri_outofplace
    !---------------------------------------------
    function rocsolver_cgetri_outofplace(handle, n, A, lda, ipiv, C, ldc, myInfo) &
       result(cgetri_outofplace) &
       bind(C, name="rocsolver_cgetri_outofplace")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgetri_outofplace
    end function rocsolver_cgetri_outofplace

    !---------------------------------------------
    ! rocsolver_zgetri_outofplace
    !---------------------------------------------
    function rocsolver_zgetri_outofplace(handle, n, A, lda, ipiv, C, ldc, myInfo) &
       result(zgetri_outofplace) &
       bind(C, name="rocsolver_zgetri_outofplace")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgetri_outofplace
    end function rocsolver_zgetri_outofplace

    !---------------------------------------------
    ! rocsolver_sgetri_outofplace_batched
    !---------------------------------------------
    function rocsolver_sgetri_outofplace_batched(handle, n, A, lda, ipiv, strideP, C, ldc, myInfo, &
                                                 batch_count) &
       result(sgetri_outofplace_batched) &
       bind(C, name="rocsolver_sgetri_outofplace_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetri_outofplace_batched
    end function rocsolver_sgetri_outofplace_batched

    !---------------------------------------------
    ! rocsolver_dgetri_outofplace_batched
    !---------------------------------------------
    function rocsolver_dgetri_outofplace_batched(handle, n, A, lda, ipiv, strideP, C, ldc, myInfo, &
                                                 batch_count) &
       result(dgetri_outofplace_batched) &
       bind(C, name="rocsolver_dgetri_outofplace_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetri_outofplace_batched
    end function rocsolver_dgetri_outofplace_batched

    !---------------------------------------------
    ! rocsolver_cgetri_outofplace_batched
    !---------------------------------------------
    function rocsolver_cgetri_outofplace_batched(handle, n, A, lda, ipiv, strideP, C, ldc, myInfo, &
                                                 batch_count) &
       result(cgetri_outofplace_batched) &
       bind(C, name="rocsolver_cgetri_outofplace_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetri_outofplace_batched
    end function rocsolver_cgetri_outofplace_batched

    !---------------------------------------------
    ! rocsolver_zgetri_outofplace_batched
    !---------------------------------------------
    function rocsolver_zgetri_outofplace_batched(handle, n, A, lda, ipiv, strideP, C, ldc, myInfo, &
                                                 batch_count) &
       result(zgetri_outofplace_batched) &
       bind(C, name="rocsolver_zgetri_outofplace_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetri_outofplace_batched
    end function rocsolver_zgetri_outofplace_batched

    !---------------------------------------------
    ! rocsolver_sgetri_outofplace_strided_batched
    !---------------------------------------------
    function rocsolver_sgetri_outofplace_strided_batched(handle, n, A, lda, strideA, ipiv, &
                                                         strideP, C, ldc, strideC, myInfo, &
                                                         batch_count) &
       result(sgetri_outofplace_strided_batched) &
       bind(C, name="rocsolver_sgetri_outofplace_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetri_outofplace_strided_batched
    end function rocsolver_sgetri_outofplace_strided_batched

    !---------------------------------------------
    ! rocsolver_dgetri_outofplace_strided_batched
    !---------------------------------------------
    function rocsolver_dgetri_outofplace_strided_batched(handle, n, A, lda, strideA, ipiv, &
                                                         strideP, C, ldc, strideC, myInfo, &
                                                         batch_count) &
       result(dgetri_outofplace_strided_batched) &
       bind(C, name="rocsolver_dgetri_outofplace_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetri_outofplace_strided_batched
    end function rocsolver_dgetri_outofplace_strided_batched

    !---------------------------------------------
    ! rocsolver_cgetri_outofplace_strided_batched
    !---------------------------------------------
    function rocsolver_cgetri_outofplace_strided_batched(handle, n, A, lda, strideA, ipiv, &
                                                         strideP, C, ldc, strideC, myInfo, &
                                                         batch_count) &
       result(cgetri_outofplace_strided_batched) &
       bind(C, name="rocsolver_cgetri_outofplace_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetri_outofplace_strided_batched
    end function rocsolver_cgetri_outofplace_strided_batched

    !---------------------------------------------
    ! rocsolver_zgetri_outofplace_strided_batched
    !---------------------------------------------
    function rocsolver_zgetri_outofplace_strided_batched(handle, n, A, lda, strideA, ipiv, &
                                                         strideP, C, ldc, strideC, myInfo, &
                                                         batch_count) &
       result(zgetri_outofplace_strided_batched) &
       bind(C, name="rocsolver_zgetri_outofplace_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetri_outofplace_strided_batched
    end function rocsolver_zgetri_outofplace_strided_batched

    !---------------------------------------------
    ! rocsolver_sgetri_npvt_outofplace
    !---------------------------------------------
    function rocsolver_sgetri_npvt_outofplace(handle, n, A, lda, C, ldc, myInfo) &
       result(sgetri_npvt_outofplace) &
       bind(C, name="rocsolver_sgetri_npvt_outofplace")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgetri_npvt_outofplace
    end function rocsolver_sgetri_npvt_outofplace

    !---------------------------------------------
    ! rocsolver_dgetri_npvt_outofplace
    !---------------------------------------------
    function rocsolver_dgetri_npvt_outofplace(handle, n, A, lda, C, ldc, myInfo) &
       result(dgetri_npvt_outofplace) &
       bind(C, name="rocsolver_dgetri_npvt_outofplace")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgetri_npvt_outofplace
    end function rocsolver_dgetri_npvt_outofplace

    !---------------------------------------------
    ! rocsolver_cgetri_npvt_outofplace
    !---------------------------------------------
    function rocsolver_cgetri_npvt_outofplace(handle, n, A, lda, C, ldc, myInfo) &
       result(cgetri_npvt_outofplace) &
       bind(C, name="rocsolver_cgetri_npvt_outofplace")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgetri_npvt_outofplace
    end function rocsolver_cgetri_npvt_outofplace

    !---------------------------------------------
    ! rocsolver_zgetri_npvt_outofplace
    !---------------------------------------------
    function rocsolver_zgetri_npvt_outofplace(handle, n, A, lda, C, ldc, myInfo) &
       result(zgetri_npvt_outofplace) &
       bind(C, name="rocsolver_zgetri_npvt_outofplace")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgetri_npvt_outofplace
    end function rocsolver_zgetri_npvt_outofplace

    !---------------------------------------------
    ! rocsolver_sgetri_npvt_outofplace_batched
    !---------------------------------------------
    function rocsolver_sgetri_npvt_outofplace_batched(handle, n, A, lda, C, ldc, myInfo, &
                                                      batch_count) &
       result(sgetri_npvt_outofplace_batched) &
       bind(C, name="rocsolver_sgetri_npvt_outofplace_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetri_npvt_outofplace_batched
    end function rocsolver_sgetri_npvt_outofplace_batched

    !---------------------------------------------
    ! rocsolver_dgetri_npvt_outofplace_batched
    !---------------------------------------------
    function rocsolver_dgetri_npvt_outofplace_batched(handle, n, A, lda, C, ldc, myInfo, &
                                                      batch_count) &
       result(dgetri_npvt_outofplace_batched) &
       bind(C, name="rocsolver_dgetri_npvt_outofplace_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetri_npvt_outofplace_batched
    end function rocsolver_dgetri_npvt_outofplace_batched

    !---------------------------------------------
    ! rocsolver_cgetri_npvt_outofplace_batched
    !---------------------------------------------
    function rocsolver_cgetri_npvt_outofplace_batched(handle, n, A, lda, C, ldc, myInfo, &
                                                      batch_count) &
       result(cgetri_npvt_outofplace_batched) &
       bind(C, name="rocsolver_cgetri_npvt_outofplace_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetri_npvt_outofplace_batched
    end function rocsolver_cgetri_npvt_outofplace_batched

    !---------------------------------------------
    ! rocsolver_zgetri_npvt_outofplace_batched
    !---------------------------------------------
    function rocsolver_zgetri_npvt_outofplace_batched(handle, n, A, lda, C, ldc, myInfo, &
                                                      batch_count) &
       result(zgetri_npvt_outofplace_batched) &
       bind(C, name="rocsolver_zgetri_npvt_outofplace_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetri_npvt_outofplace_batched
    end function rocsolver_zgetri_npvt_outofplace_batched

    !---------------------------------------------
    ! rocsolver_sgetri_npvt_outofplace_strided_batched
    !---------------------------------------------
    function rocsolver_sgetri_npvt_outofplace_strided_batched(handle, n, A, lda, strideA, C, ldc, &
                                                              strideC, myInfo, batch_count) &
       result(sgetri_npvt_outofplace_strided_batched) &
       bind(C, name="rocsolver_sgetri_npvt_outofplace_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgetri_npvt_outofplace_strided_batched
    end function rocsolver_sgetri_npvt_outofplace_strided_batched

    !---------------------------------------------
    ! rocsolver_dgetri_npvt_outofplace_strided_batched
    !---------------------------------------------
    function rocsolver_dgetri_npvt_outofplace_strided_batched(handle, n, A, lda, strideA, C, ldc, &
                                                              strideC, myInfo, batch_count) &
       result(dgetri_npvt_outofplace_strided_batched) &
       bind(C, name="rocsolver_dgetri_npvt_outofplace_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgetri_npvt_outofplace_strided_batched
    end function rocsolver_dgetri_npvt_outofplace_strided_batched

    !---------------------------------------------
    ! rocsolver_cgetri_npvt_outofplace_strided_batched
    !---------------------------------------------
    function rocsolver_cgetri_npvt_outofplace_strided_batched(handle, n, A, lda, strideA, C, ldc, &
                                                              strideC, myInfo, batch_count) &
       result(cgetri_npvt_outofplace_strided_batched) &
       bind(C, name="rocsolver_cgetri_npvt_outofplace_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgetri_npvt_outofplace_strided_batched
    end function rocsolver_cgetri_npvt_outofplace_strided_batched

    !---------------------------------------------
    ! rocsolver_zgetri_npvt_outofplace_strided_batched
    !---------------------------------------------
    function rocsolver_zgetri_npvt_outofplace_strided_batched(handle, n, A, lda, strideA, C, ldc, &
                                                              strideC, myInfo, batch_count) &
       result(zgetri_npvt_outofplace_strided_batched) &
       bind(C, name="rocsolver_zgetri_npvt_outofplace_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgetri_npvt_outofplace_strided_batched
    end function rocsolver_zgetri_npvt_outofplace_strided_batched

    !---------------------------------------------
    ! rocsolver_strtri
    !---------------------------------------------
    function rocsolver_strtri(handle, uplo, diag, n, A, lda, myInfo) &
       result(strtri) &
       bind(C, name="rocsolver_strtri")
       import :: c_ptr, rocblas_fill_upper, rocblas_diagonal_non_unit, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(kind(rocblas_diagonal_non_unit)), value :: diag
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: strtri
    end function rocsolver_strtri

    !---------------------------------------------
    ! rocsolver_dtrtri
    !---------------------------------------------
    function rocsolver_dtrtri(handle, uplo, diag, n, A, lda, myInfo) &
       result(dtrtri) &
       bind(C, name="rocsolver_dtrtri")
       import :: c_ptr, rocblas_fill_upper, rocblas_diagonal_non_unit, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(kind(rocblas_diagonal_non_unit)), value :: diag
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dtrtri
    end function rocsolver_dtrtri

    !---------------------------------------------
    ! rocsolver_ctrtri
    !---------------------------------------------
    function rocsolver_ctrtri(handle, uplo, diag, n, A, lda, myInfo) &
       result(ctrtri) &
       bind(C, name="rocsolver_ctrtri")
       import :: c_ptr, rocblas_fill_upper, rocblas_diagonal_non_unit, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(kind(rocblas_diagonal_non_unit)), value :: diag
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ctrtri
    end function rocsolver_ctrtri

    !---------------------------------------------
    ! rocsolver_ztrtri
    !---------------------------------------------
    function rocsolver_ztrtri(handle, uplo, diag, n, A, lda, myInfo) &
       result(ztrtri) &
       bind(C, name="rocsolver_ztrtri")
       import :: c_ptr, rocblas_fill_upper, rocblas_diagonal_non_unit, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(kind(rocblas_diagonal_non_unit)), value :: diag
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ztrtri
    end function rocsolver_ztrtri

    !---------------------------------------------
    ! rocsolver_strtri_batched
    !---------------------------------------------
    function rocsolver_strtri_batched(handle, uplo, diag, n, A, lda, myInfo, batch_count) &
       result(strtri_batched) &
       bind(C, name="rocsolver_strtri_batched")
       import :: c_ptr, rocblas_fill_upper, rocblas_diagonal_non_unit, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(kind(rocblas_diagonal_non_unit)), value :: diag
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: strtri_batched
    end function rocsolver_strtri_batched

    !---------------------------------------------
    ! rocsolver_dtrtri_batched
    !---------------------------------------------
    function rocsolver_dtrtri_batched(handle, uplo, diag, n, A, lda, myInfo, batch_count) &
       result(dtrtri_batched) &
       bind(C, name="rocsolver_dtrtri_batched")
       import :: c_ptr, rocblas_fill_upper, rocblas_diagonal_non_unit, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(kind(rocblas_diagonal_non_unit)), value :: diag
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dtrtri_batched
    end function rocsolver_dtrtri_batched

    !---------------------------------------------
    ! rocsolver_ctrtri_batched
    !---------------------------------------------
    function rocsolver_ctrtri_batched(handle, uplo, diag, n, A, lda, myInfo, batch_count) &
       result(ctrtri_batched) &
       bind(C, name="rocsolver_ctrtri_batched")
       import :: c_ptr, rocblas_fill_upper, rocblas_diagonal_non_unit, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(kind(rocblas_diagonal_non_unit)), value :: diag
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ctrtri_batched
    end function rocsolver_ctrtri_batched

    !---------------------------------------------
    ! rocsolver_ztrtri_batched
    !---------------------------------------------
    function rocsolver_ztrtri_batched(handle, uplo, diag, n, A, lda, myInfo, batch_count) &
       result(ztrtri_batched) &
       bind(C, name="rocsolver_ztrtri_batched")
       import :: c_ptr, rocblas_fill_upper, rocblas_diagonal_non_unit, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(kind(rocblas_diagonal_non_unit)), value :: diag
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ztrtri_batched
    end function rocsolver_ztrtri_batched

    !---------------------------------------------
    ! rocsolver_strtri_strided_batched
    !---------------------------------------------
    function rocsolver_strtri_strided_batched(handle, uplo, diag, n, A, lda, strideA, myInfo, &
                                              batch_count) &
       result(strtri_strided_batched) &
       bind(C, name="rocsolver_strtri_strided_batched")
       import :: c_ptr, rocblas_fill_upper, rocblas_diagonal_non_unit, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(kind(rocblas_diagonal_non_unit)), value :: diag
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: strtri_strided_batched
    end function rocsolver_strtri_strided_batched

    !---------------------------------------------
    ! rocsolver_dtrtri_strided_batched
    !---------------------------------------------
    function rocsolver_dtrtri_strided_batched(handle, uplo, diag, n, A, lda, strideA, myInfo, &
                                              batch_count) &
       result(dtrtri_strided_batched) &
       bind(C, name="rocsolver_dtrtri_strided_batched")
       import :: c_ptr, rocblas_fill_upper, rocblas_diagonal_non_unit, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(kind(rocblas_diagonal_non_unit)), value :: diag
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dtrtri_strided_batched
    end function rocsolver_dtrtri_strided_batched

    !---------------------------------------------
    ! rocsolver_ctrtri_strided_batched
    !---------------------------------------------
    function rocsolver_ctrtri_strided_batched(handle, uplo, diag, n, A, lda, strideA, myInfo, &
                                              batch_count) &
       result(ctrtri_strided_batched) &
       bind(C, name="rocsolver_ctrtri_strided_batched")
       import :: c_ptr, rocblas_fill_upper, rocblas_diagonal_non_unit, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(kind(rocblas_diagonal_non_unit)), value :: diag
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ctrtri_strided_batched
    end function rocsolver_ctrtri_strided_batched

    !---------------------------------------------
    ! rocsolver_ztrtri_strided_batched
    !---------------------------------------------
    function rocsolver_ztrtri_strided_batched(handle, uplo, diag, n, A, lda, strideA, myInfo, &
                                              batch_count) &
       result(ztrtri_strided_batched) &
       bind(C, name="rocsolver_ztrtri_strided_batched")
       import :: c_ptr, rocblas_fill_upper, rocblas_diagonal_non_unit, c_int, c_int64_t, &
                 rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(kind(rocblas_diagonal_non_unit)), value :: diag
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ztrtri_strided_batched
    end function rocsolver_ztrtri_strided_batched

    !---------------------------------------------
    ! rocsolver_ssytf2
    !---------------------------------------------
    function rocsolver_ssytf2(handle, uplo, n, A, lda, ipiv, myInfo) &
       result(ssytf2) &
       bind(C, name="rocsolver_ssytf2")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ssytf2
    end function rocsolver_ssytf2

    !---------------------------------------------
    ! rocsolver_dsytf2
    !---------------------------------------------
    function rocsolver_dsytf2(handle, uplo, n, A, lda, ipiv, myInfo) &
       result(dsytf2) &
       bind(C, name="rocsolver_dsytf2")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dsytf2
    end function rocsolver_dsytf2

    !---------------------------------------------
    ! rocsolver_csytf2
    !---------------------------------------------
    function rocsolver_csytf2(handle, uplo, n, A, lda, ipiv, myInfo) &
       result(csytf2) &
       bind(C, name="rocsolver_csytf2")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: csytf2
    end function rocsolver_csytf2

    !---------------------------------------------
    ! rocsolver_zsytf2
    !---------------------------------------------
    function rocsolver_zsytf2(handle, uplo, n, A, lda, ipiv, myInfo) &
       result(zsytf2) &
       bind(C, name="rocsolver_zsytf2")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zsytf2
    end function rocsolver_zsytf2

    !---------------------------------------------
    ! rocsolver_ssytf2_batched
    !---------------------------------------------
    function rocsolver_ssytf2_batched(handle, uplo, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(ssytf2_batched) &
       bind(C, name="rocsolver_ssytf2_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssytf2_batched
    end function rocsolver_ssytf2_batched

    !---------------------------------------------
    ! rocsolver_dsytf2_batched
    !---------------------------------------------
    function rocsolver_dsytf2_batched(handle, uplo, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(dsytf2_batched) &
       bind(C, name="rocsolver_dsytf2_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsytf2_batched
    end function rocsolver_dsytf2_batched

    !---------------------------------------------
    ! rocsolver_csytf2_batched
    !---------------------------------------------
    function rocsolver_csytf2_batched(handle, uplo, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(csytf2_batched) &
       bind(C, name="rocsolver_csytf2_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: csytf2_batched
    end function rocsolver_csytf2_batched

    !---------------------------------------------
    ! rocsolver_zsytf2_batched
    !---------------------------------------------
    function rocsolver_zsytf2_batched(handle, uplo, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(zsytf2_batched) &
       bind(C, name="rocsolver_zsytf2_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zsytf2_batched
    end function rocsolver_zsytf2_batched

    !---------------------------------------------
    ! rocsolver_ssytf2_strided_batched
    !---------------------------------------------
    function rocsolver_ssytf2_strided_batched(handle, uplo, n, A, lda, strideA, ipiv, strideP, &
                                              myInfo, batch_count) &
       result(ssytf2_strided_batched) &
       bind(C, name="rocsolver_ssytf2_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssytf2_strided_batched
    end function rocsolver_ssytf2_strided_batched

    !---------------------------------------------
    ! rocsolver_dsytf2_strided_batched
    !---------------------------------------------
    function rocsolver_dsytf2_strided_batched(handle, uplo, n, A, lda, strideA, ipiv, strideP, &
                                              myInfo, batch_count) &
       result(dsytf2_strided_batched) &
       bind(C, name="rocsolver_dsytf2_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsytf2_strided_batched
    end function rocsolver_dsytf2_strided_batched

    !---------------------------------------------
    ! rocsolver_csytf2_strided_batched
    !---------------------------------------------
    function rocsolver_csytf2_strided_batched(handle, uplo, n, A, lda, strideA, ipiv, strideP, &
                                              myInfo, batch_count) &
       result(csytf2_strided_batched) &
       bind(C, name="rocsolver_csytf2_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: csytf2_strided_batched
    end function rocsolver_csytf2_strided_batched

    !---------------------------------------------
    ! rocsolver_zsytf2_strided_batched
    !---------------------------------------------
    function rocsolver_zsytf2_strided_batched(handle, uplo, n, A, lda, strideA, ipiv, strideP, &
                                              myInfo, batch_count) &
       result(zsytf2_strided_batched) &
       bind(C, name="rocsolver_zsytf2_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zsytf2_strided_batched
    end function rocsolver_zsytf2_strided_batched

    !---------------------------------------------
    ! rocsolver_ssytrf
    !---------------------------------------------
    function rocsolver_ssytrf(handle, uplo, n, A, lda, ipiv, myInfo) &
       result(ssytrf) &
       bind(C, name="rocsolver_ssytrf")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ssytrf
    end function rocsolver_ssytrf

    !---------------------------------------------
    ! rocsolver_dsytrf
    !---------------------------------------------
    function rocsolver_dsytrf(handle, uplo, n, A, lda, ipiv, myInfo) &
       result(dsytrf) &
       bind(C, name="rocsolver_dsytrf")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dsytrf
    end function rocsolver_dsytrf

    !---------------------------------------------
    ! rocsolver_csytrf
    !---------------------------------------------
    function rocsolver_csytrf(handle, uplo, n, A, lda, ipiv, myInfo) &
       result(csytrf) &
       bind(C, name="rocsolver_csytrf")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: csytrf
    end function rocsolver_csytrf

    !---------------------------------------------
    ! rocsolver_zsytrf
    !---------------------------------------------
    function rocsolver_zsytrf(handle, uplo, n, A, lda, ipiv, myInfo) &
       result(zsytrf) &
       bind(C, name="rocsolver_zsytrf")
       import :: c_ptr, rocblas_fill_upper, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zsytrf
    end function rocsolver_zsytrf

    !---------------------------------------------
    ! rocsolver_ssytrf_batched
    !---------------------------------------------
    function rocsolver_ssytrf_batched(handle, uplo, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(ssytrf_batched) &
       bind(C, name="rocsolver_ssytrf_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssytrf_batched
    end function rocsolver_ssytrf_batched

    !---------------------------------------------
    ! rocsolver_dsytrf_batched
    !---------------------------------------------
    function rocsolver_dsytrf_batched(handle, uplo, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(dsytrf_batched) &
       bind(C, name="rocsolver_dsytrf_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsytrf_batched
    end function rocsolver_dsytrf_batched

    !---------------------------------------------
    ! rocsolver_csytrf_batched
    !---------------------------------------------
    function rocsolver_csytrf_batched(handle, uplo, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(csytrf_batched) &
       bind(C, name="rocsolver_csytrf_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: csytrf_batched
    end function rocsolver_csytrf_batched

    !---------------------------------------------
    ! rocsolver_zsytrf_batched
    !---------------------------------------------
    function rocsolver_zsytrf_batched(handle, uplo, n, A, lda, ipiv, strideP, myInfo, batch_count) &
       result(zsytrf_batched) &
       bind(C, name="rocsolver_zsytrf_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zsytrf_batched
    end function rocsolver_zsytrf_batched

    !---------------------------------------------
    ! rocsolver_ssytrf_strided_batched
    !---------------------------------------------
    function rocsolver_ssytrf_strided_batched(handle, uplo, n, A, lda, strideA, ipiv, strideP, &
                                              myInfo, batch_count) &
       result(ssytrf_strided_batched) &
       bind(C, name="rocsolver_ssytrf_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssytrf_strided_batched
    end function rocsolver_ssytrf_strided_batched

    !---------------------------------------------
    ! rocsolver_dsytrf_strided_batched
    !---------------------------------------------
    function rocsolver_dsytrf_strided_batched(handle, uplo, n, A, lda, strideA, ipiv, strideP, &
                                              myInfo, batch_count) &
       result(dsytrf_strided_batched) &
       bind(C, name="rocsolver_dsytrf_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsytrf_strided_batched
    end function rocsolver_dsytrf_strided_batched

    !---------------------------------------------
    ! rocsolver_csytrf_strided_batched
    !---------------------------------------------
    function rocsolver_csytrf_strided_batched(handle, uplo, n, A, lda, strideA, ipiv, strideP, &
                                              myInfo, batch_count) &
       result(csytrf_strided_batched) &
       bind(C, name="rocsolver_csytrf_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: csytrf_strided_batched
    end function rocsolver_csytrf_strided_batched

    !---------------------------------------------
    ! rocsolver_zsytrf_strided_batched
    !---------------------------------------------
    function rocsolver_zsytrf_strided_batched(handle, uplo, n, A, lda, strideA, ipiv, strideP, &
                                              myInfo, batch_count) &
       result(zsytrf_strided_batched) &
       bind(C, name="rocsolver_zsytrf_strided_batched")
       import :: c_ptr, rocblas_fill_upper, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: ipiv
       integer(c_int64_t), value :: strideP
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zsytrf_strided_batched
    end function rocsolver_zsytrf_strided_batched

    !---------------------------------------------
    ! rocsolver_sgeblttrf_npvt
    !---------------------------------------------
    function rocsolver_sgeblttrf_npvt(handle, nb, nblocks, A, lda, B, ldb, C, ldc, myInfo) &
       result(sgeblttrf_npvt) &
       bind(C, name="rocsolver_sgeblttrf_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: sgeblttrf_npvt
    end function rocsolver_sgeblttrf_npvt

    !---------------------------------------------
    ! rocsolver_dgeblttrf_npvt
    !---------------------------------------------
    function rocsolver_dgeblttrf_npvt(handle, nb, nblocks, A, lda, B, ldb, C, ldc, myInfo) &
       result(dgeblttrf_npvt) &
       bind(C, name="rocsolver_dgeblttrf_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dgeblttrf_npvt
    end function rocsolver_dgeblttrf_npvt

    !---------------------------------------------
    ! rocsolver_cgeblttrf_npvt
    !---------------------------------------------
    function rocsolver_cgeblttrf_npvt(handle, nb, nblocks, A, lda, B, ldb, C, ldc, myInfo) &
       result(cgeblttrf_npvt) &
       bind(C, name="rocsolver_cgeblttrf_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cgeblttrf_npvt
    end function rocsolver_cgeblttrf_npvt

    !---------------------------------------------
    ! rocsolver_zgeblttrf_npvt
    !---------------------------------------------
    function rocsolver_zgeblttrf_npvt(handle, nb, nblocks, A, lda, B, ldb, C, ldc, myInfo) &
       result(zgeblttrf_npvt) &
       bind(C, name="rocsolver_zgeblttrf_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zgeblttrf_npvt
    end function rocsolver_zgeblttrf_npvt

    !---------------------------------------------
    ! rocsolver_sgeblttrf_npvt_batched
    !---------------------------------------------
    function rocsolver_sgeblttrf_npvt_batched(handle, nb, nblocks, A, lda, B, ldb, C, ldc, myInfo, &
                                              batch_count) &
       result(sgeblttrf_npvt_batched) &
       bind(C, name="rocsolver_sgeblttrf_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgeblttrf_npvt_batched
    end function rocsolver_sgeblttrf_npvt_batched

    !---------------------------------------------
    ! rocsolver_dgeblttrf_npvt_batched
    !---------------------------------------------
    function rocsolver_dgeblttrf_npvt_batched(handle, nb, nblocks, A, lda, B, ldb, C, ldc, myInfo, &
                                              batch_count) &
       result(dgeblttrf_npvt_batched) &
       bind(C, name="rocsolver_dgeblttrf_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgeblttrf_npvt_batched
    end function rocsolver_dgeblttrf_npvt_batched

    !---------------------------------------------
    ! rocsolver_cgeblttrf_npvt_batched
    !---------------------------------------------
    function rocsolver_cgeblttrf_npvt_batched(handle, nb, nblocks, A, lda, B, ldb, C, ldc, myInfo, &
                                              batch_count) &
       result(cgeblttrf_npvt_batched) &
       bind(C, name="rocsolver_cgeblttrf_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgeblttrf_npvt_batched
    end function rocsolver_cgeblttrf_npvt_batched

    !---------------------------------------------
    ! rocsolver_zgeblttrf_npvt_batched
    !---------------------------------------------
    function rocsolver_zgeblttrf_npvt_batched(handle, nb, nblocks, A, lda, B, ldb, C, ldc, myInfo, &
                                              batch_count) &
       result(zgeblttrf_npvt_batched) &
       bind(C, name="rocsolver_zgeblttrf_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgeblttrf_npvt_batched
    end function rocsolver_zgeblttrf_npvt_batched

    !---------------------------------------------
    ! rocsolver_sgeblttrf_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_sgeblttrf_npvt_strided_batched(handle, nb, nblocks, A, lda, strideA, B, &
                                                      ldb, strideB, C, ldc, strideC, myInfo, &
                                                      batch_count) &
       result(sgeblttrf_npvt_strided_batched) &
       bind(C, name="rocsolver_sgeblttrf_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgeblttrf_npvt_strided_batched
    end function rocsolver_sgeblttrf_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_dgeblttrf_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_dgeblttrf_npvt_strided_batched(handle, nb, nblocks, A, lda, strideA, B, &
                                                      ldb, strideB, C, ldc, strideC, myInfo, &
                                                      batch_count) &
       result(dgeblttrf_npvt_strided_batched) &
       bind(C, name="rocsolver_dgeblttrf_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgeblttrf_npvt_strided_batched
    end function rocsolver_dgeblttrf_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_cgeblttrf_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_cgeblttrf_npvt_strided_batched(handle, nb, nblocks, A, lda, strideA, B, &
                                                      ldb, strideB, C, ldc, strideC, myInfo, &
                                                      batch_count) &
       result(cgeblttrf_npvt_strided_batched) &
       bind(C, name="rocsolver_cgeblttrf_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgeblttrf_npvt_strided_batched
    end function rocsolver_cgeblttrf_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_zgeblttrf_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_zgeblttrf_npvt_strided_batched(handle, nb, nblocks, A, lda, strideA, B, &
                                                      ldb, strideB, C, ldc, strideC, myInfo, &
                                                      batch_count) &
       result(zgeblttrf_npvt_strided_batched) &
       bind(C, name="rocsolver_zgeblttrf_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgeblttrf_npvt_strided_batched
    end function rocsolver_zgeblttrf_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_sgeblttrf_npvt_interleaved_batched
    !---------------------------------------------
    function rocsolver_sgeblttrf_npvt_interleaved_batched(handle, nb, nblocks, A, inca, lda, &
                                                          strideA, B, incb, ldb, strideB, C, incc, &
                                                          ldc, strideC, myInfo, batch_count) &
       result(sgeblttrf_npvt_interleaved_batched) &
       bind(C, name="rocsolver_sgeblttrf_npvt_interleaved_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       type(c_ptr), value :: A
       integer(c_int), value :: inca
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: incb
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: C
       integer(c_int), value :: incc
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgeblttrf_npvt_interleaved_batched
    end function rocsolver_sgeblttrf_npvt_interleaved_batched

    !---------------------------------------------
    ! rocsolver_dgeblttrf_npvt_interleaved_batched
    !---------------------------------------------
    function rocsolver_dgeblttrf_npvt_interleaved_batched(handle, nb, nblocks, A, inca, lda, &
                                                          strideA, B, incb, ldb, strideB, C, incc, &
                                                          ldc, strideC, myInfo, batch_count) &
       result(dgeblttrf_npvt_interleaved_batched) &
       bind(C, name="rocsolver_dgeblttrf_npvt_interleaved_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       type(c_ptr), value :: A
       integer(c_int), value :: inca
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: incb
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: C
       integer(c_int), value :: incc
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgeblttrf_npvt_interleaved_batched
    end function rocsolver_dgeblttrf_npvt_interleaved_batched

    !---------------------------------------------
    ! rocsolver_cgeblttrf_npvt_interleaved_batched
    !---------------------------------------------
    function rocsolver_cgeblttrf_npvt_interleaved_batched(handle, nb, nblocks, A, inca, lda, &
                                                          strideA, B, incb, ldb, strideB, C, incc, &
                                                          ldc, strideC, myInfo, batch_count) &
       result(cgeblttrf_npvt_interleaved_batched) &
       bind(C, name="rocsolver_cgeblttrf_npvt_interleaved_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       type(c_ptr), value :: A
       integer(c_int), value :: inca
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: incb
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: C
       integer(c_int), value :: incc
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgeblttrf_npvt_interleaved_batched
    end function rocsolver_cgeblttrf_npvt_interleaved_batched

    !---------------------------------------------
    ! rocsolver_zgeblttrf_npvt_interleaved_batched
    !---------------------------------------------
    function rocsolver_zgeblttrf_npvt_interleaved_batched(handle, nb, nblocks, A, inca, lda, &
                                                          strideA, B, incb, ldb, strideB, C, incc, &
                                                          ldc, strideC, myInfo, batch_count) &
       result(zgeblttrf_npvt_interleaved_batched) &
       bind(C, name="rocsolver_zgeblttrf_npvt_interleaved_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       type(c_ptr), value :: A
       integer(c_int), value :: inca
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: incb
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: C
       integer(c_int), value :: incc
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgeblttrf_npvt_interleaved_batched
    end function rocsolver_zgeblttrf_npvt_interleaved_batched

    !---------------------------------------------
    ! rocsolver_sgeblttrs_npvt
    !---------------------------------------------
    function rocsolver_sgeblttrs_npvt(handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, X, ldx) &
       result(sgeblttrs_npvt) &
       bind(C, name="rocsolver_sgeblttrs_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(kind(rocblas_status_success)) :: sgeblttrs_npvt
    end function rocsolver_sgeblttrs_npvt

    !---------------------------------------------
    ! rocsolver_dgeblttrs_npvt
    !---------------------------------------------
    function rocsolver_dgeblttrs_npvt(handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, X, ldx) &
       result(dgeblttrs_npvt) &
       bind(C, name="rocsolver_dgeblttrs_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(kind(rocblas_status_success)) :: dgeblttrs_npvt
    end function rocsolver_dgeblttrs_npvt

    !---------------------------------------------
    ! rocsolver_cgeblttrs_npvt
    !---------------------------------------------
    function rocsolver_cgeblttrs_npvt(handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, X, ldx) &
       result(cgeblttrs_npvt) &
       bind(C, name="rocsolver_cgeblttrs_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(kind(rocblas_status_success)) :: cgeblttrs_npvt
    end function rocsolver_cgeblttrs_npvt

    !---------------------------------------------
    ! rocsolver_zgeblttrs_npvt
    !---------------------------------------------
    function rocsolver_zgeblttrs_npvt(handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, X, ldx) &
       result(zgeblttrs_npvt) &
       bind(C, name="rocsolver_zgeblttrs_npvt")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(kind(rocblas_status_success)) :: zgeblttrs_npvt
    end function rocsolver_zgeblttrs_npvt

    !---------------------------------------------
    ! rocsolver_sgeblttrs_npvt_batched
    !---------------------------------------------
    function rocsolver_sgeblttrs_npvt_batched(handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, &
                                              X, ldx, batch_count) &
       result(sgeblttrs_npvt_batched) &
       bind(C, name="rocsolver_sgeblttrs_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgeblttrs_npvt_batched
    end function rocsolver_sgeblttrs_npvt_batched

    !---------------------------------------------
    ! rocsolver_dgeblttrs_npvt_batched
    !---------------------------------------------
    function rocsolver_dgeblttrs_npvt_batched(handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, &
                                              X, ldx, batch_count) &
       result(dgeblttrs_npvt_batched) &
       bind(C, name="rocsolver_dgeblttrs_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgeblttrs_npvt_batched
    end function rocsolver_dgeblttrs_npvt_batched

    !---------------------------------------------
    ! rocsolver_cgeblttrs_npvt_batched
    !---------------------------------------------
    function rocsolver_cgeblttrs_npvt_batched(handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, &
                                              X, ldx, batch_count) &
       result(cgeblttrs_npvt_batched) &
       bind(C, name="rocsolver_cgeblttrs_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgeblttrs_npvt_batched
    end function rocsolver_cgeblttrs_npvt_batched

    !---------------------------------------------
    ! rocsolver_zgeblttrs_npvt_batched
    !---------------------------------------------
    function rocsolver_zgeblttrs_npvt_batched(handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, &
                                              X, ldx, batch_count) &
       result(zgeblttrs_npvt_batched) &
       bind(C, name="rocsolver_zgeblttrs_npvt_batched")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgeblttrs_npvt_batched
    end function rocsolver_zgeblttrs_npvt_batched

    !---------------------------------------------
    ! rocsolver_sgeblttrs_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_sgeblttrs_npvt_strided_batched(handle, nb, nblocks, nrhs, A, lda, strideA, &
                                                      B, ldb, strideB, C, ldc, strideC, X, ldx, &
                                                      strideX, batch_count) &
       result(sgeblttrs_npvt_strided_batched) &
       bind(C, name="rocsolver_sgeblttrs_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(c_int64_t), value :: strideX
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgeblttrs_npvt_strided_batched
    end function rocsolver_sgeblttrs_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_dgeblttrs_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_dgeblttrs_npvt_strided_batched(handle, nb, nblocks, nrhs, A, lda, strideA, &
                                                      B, ldb, strideB, C, ldc, strideC, X, ldx, &
                                                      strideX, batch_count) &
       result(dgeblttrs_npvt_strided_batched) &
       bind(C, name="rocsolver_dgeblttrs_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(c_int64_t), value :: strideX
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgeblttrs_npvt_strided_batched
    end function rocsolver_dgeblttrs_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_cgeblttrs_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_cgeblttrs_npvt_strided_batched(handle, nb, nblocks, nrhs, A, lda, strideA, &
                                                      B, ldb, strideB, C, ldc, strideC, X, ldx, &
                                                      strideX, batch_count) &
       result(cgeblttrs_npvt_strided_batched) &
       bind(C, name="rocsolver_cgeblttrs_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(c_int64_t), value :: strideX
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgeblttrs_npvt_strided_batched
    end function rocsolver_cgeblttrs_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_zgeblttrs_npvt_strided_batched
    !---------------------------------------------
    function rocsolver_zgeblttrs_npvt_strided_batched(handle, nb, nblocks, nrhs, A, lda, strideA, &
                                                      B, ldb, strideB, C, ldc, strideC, X, ldx, &
                                                      strideX, batch_count) &
       result(zgeblttrs_npvt_strided_batched) &
       bind(C, name="rocsolver_zgeblttrs_npvt_strided_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(c_int64_t), value :: strideX
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgeblttrs_npvt_strided_batched
    end function rocsolver_zgeblttrs_npvt_strided_batched

    !---------------------------------------------
    ! rocsolver_sgeblttrs_npvt_interleaved_batched
    !---------------------------------------------
    function rocsolver_sgeblttrs_npvt_interleaved_batched(handle, nb, nblocks, nrhs, A, inca, lda, &
                                                          strideA, B, incb, ldb, strideB, C, incc, &
                                                          ldc, strideC, X, incx, ldx, strideX, &
                                                          batch_count) &
       result(sgeblttrs_npvt_interleaved_batched) &
       bind(C, name="rocsolver_sgeblttrs_npvt_interleaved_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: inca
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: incb
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: C
       integer(c_int), value :: incc
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: X
       integer(c_int), value :: incx
       integer(c_int), value :: ldx
       integer(c_int64_t), value :: strideX
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: sgeblttrs_npvt_interleaved_batched
    end function rocsolver_sgeblttrs_npvt_interleaved_batched

    !---------------------------------------------
    ! rocsolver_dgeblttrs_npvt_interleaved_batched
    !---------------------------------------------
    function rocsolver_dgeblttrs_npvt_interleaved_batched(handle, nb, nblocks, nrhs, A, inca, lda, &
                                                          strideA, B, incb, ldb, strideB, C, incc, &
                                                          ldc, strideC, X, incx, ldx, strideX, &
                                                          batch_count) &
       result(dgeblttrs_npvt_interleaved_batched) &
       bind(C, name="rocsolver_dgeblttrs_npvt_interleaved_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: inca
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: incb
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: C
       integer(c_int), value :: incc
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: X
       integer(c_int), value :: incx
       integer(c_int), value :: ldx
       integer(c_int64_t), value :: strideX
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dgeblttrs_npvt_interleaved_batched
    end function rocsolver_dgeblttrs_npvt_interleaved_batched

    !---------------------------------------------
    ! rocsolver_cgeblttrs_npvt_interleaved_batched
    !---------------------------------------------
    function rocsolver_cgeblttrs_npvt_interleaved_batched(handle, nb, nblocks, nrhs, A, inca, lda, &
                                                          strideA, B, incb, ldb, strideB, C, incc, &
                                                          ldc, strideC, X, incx, ldx, strideX, &
                                                          batch_count) &
       result(cgeblttrs_npvt_interleaved_batched) &
       bind(C, name="rocsolver_cgeblttrs_npvt_interleaved_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: inca
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: incb
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: C
       integer(c_int), value :: incc
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: X
       integer(c_int), value :: incx
       integer(c_int), value :: ldx
       integer(c_int64_t), value :: strideX
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cgeblttrs_npvt_interleaved_batched
    end function rocsolver_cgeblttrs_npvt_interleaved_batched

    !---------------------------------------------
    ! rocsolver_zgeblttrs_npvt_interleaved_batched
    !---------------------------------------------
    function rocsolver_zgeblttrs_npvt_interleaved_batched(handle, nb, nblocks, nrhs, A, inca, lda, &
                                                          strideA, B, incb, ldb, strideB, C, incc, &
                                                          ldc, strideC, X, incx, ldx, strideX, &
                                                          batch_count) &
       result(zgeblttrs_npvt_interleaved_batched) &
       bind(C, name="rocsolver_zgeblttrs_npvt_interleaved_batched")
       import :: c_ptr, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: nb
       integer(c_int), value :: nblocks
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: inca
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: incb
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       type(c_ptr), value :: C
       integer(c_int), value :: incc
       integer(c_int), value :: ldc
       integer(c_int64_t), value :: strideC
       type(c_ptr), value :: X
       integer(c_int), value :: incx
       integer(c_int), value :: ldx
       integer(c_int64_t), value :: strideX
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zgeblttrs_npvt_interleaved_batched
    end function rocsolver_zgeblttrs_npvt_interleaved_batched

    !---------------------------------------------
    ! rocsolver_create_rfinfo
    !---------------------------------------------
    function rocsolver_create_rfinfo(rfinfo, handle) &
       result(create_rfinfo) &
       bind(C, name="rocsolver_create_rfinfo")
       import :: c_ptr, rocblas_status_success
       type(c_ptr) :: rfinfo
       type(c_ptr), value :: handle
       integer(kind(rocblas_status_success)) :: create_rfinfo
    end function rocsolver_create_rfinfo

    !---------------------------------------------
    ! rocsolver_destroy_rfinfo
    !---------------------------------------------
    function rocsolver_destroy_rfinfo(rfinfo) &
       result(destroy_rfinfo) &
       bind(C, name="rocsolver_destroy_rfinfo")
       import :: c_ptr, rocblas_status_success
       type(c_ptr), value :: rfinfo
       integer(kind(rocblas_status_success)) :: destroy_rfinfo
    end function rocsolver_destroy_rfinfo

    !---------------------------------------------
    ! rocsolver_set_rfinfo_mode
    !---------------------------------------------
    function rocsolver_set_rfinfo_mode(rfinfo, mode) &
       result(set_rfinfo_mode) &
       bind(C, name="rocsolver_set_rfinfo_mode")
       import :: c_ptr, rocsolver_rfinfo_mode_lu, rocblas_status_success
       type(c_ptr), value :: rfinfo
       integer(kind(rocsolver_rfinfo_mode_lu)), value :: mode
       integer(kind(rocblas_status_success)) :: set_rfinfo_mode
    end function rocsolver_set_rfinfo_mode

    !---------------------------------------------
    ! rocsolver_get_rfinfo_mode
    !---------------------------------------------
    function rocsolver_get_rfinfo_mode(rfinfo, mode) &
       result(get_rfinfo_mode) &
       bind(C, name="rocsolver_get_rfinfo_mode")
       import :: c_ptr, rocsolver_rfinfo_mode_lu, rocblas_status_success
       type(c_ptr), value :: rfinfo
       integer(kind(rocsolver_rfinfo_mode_lu)) :: mode
       integer(kind(rocblas_status_success)) :: get_rfinfo_mode
    end function rocsolver_get_rfinfo_mode

    !---------------------------------------------
    ! rocsolver_scsrrf_sumlu
    !---------------------------------------------
    function rocsolver_scsrrf_sumlu(handle, n, nnzL, ptrL, indL, valL, nnzU, ptrU, indU, valU, &
                                    ptrT, indT, valT) &
       result(scsrrf_sumlu) &
       bind(C, name="rocsolver_scsrrf_sumlu")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nnzL
       type(c_ptr), value :: ptrL
       type(c_ptr), value :: indL
       type(c_ptr), value :: valL
       integer(c_int), value :: nnzU
       type(c_ptr), value :: ptrU
       type(c_ptr), value :: indU
       type(c_ptr), value :: valU
       type(c_ptr), value :: ptrT
       type(c_ptr), value :: indT
       type(c_ptr), value :: valT
       integer(kind(rocblas_status_success)) :: scsrrf_sumlu
    end function rocsolver_scsrrf_sumlu

    !---------------------------------------------
    ! rocsolver_dcsrrf_sumlu
    !---------------------------------------------
    function rocsolver_dcsrrf_sumlu(handle, n, nnzL, ptrL, indL, valL, nnzU, ptrU, indU, valU, &
                                    ptrT, indT, valT) &
       result(dcsrrf_sumlu) &
       bind(C, name="rocsolver_dcsrrf_sumlu")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nnzL
       type(c_ptr), value :: ptrL
       type(c_ptr), value :: indL
       type(c_ptr), value :: valL
       integer(c_int), value :: nnzU
       type(c_ptr), value :: ptrU
       type(c_ptr), value :: indU
       type(c_ptr), value :: valU
       type(c_ptr), value :: ptrT
       type(c_ptr), value :: indT
       type(c_ptr), value :: valT
       integer(kind(rocblas_status_success)) :: dcsrrf_sumlu
    end function rocsolver_dcsrrf_sumlu

    !---------------------------------------------
    ! rocsolver_scsrrf_splitlu
    !---------------------------------------------
    function rocsolver_scsrrf_splitlu(handle, n, nnzT, ptrT, indT, valT, ptrL, indL, valL, ptrU, &
                                      indU, valU) &
       result(scsrrf_splitlu) &
       bind(C, name="rocsolver_scsrrf_splitlu")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nnzT
       type(c_ptr), value :: ptrT
       type(c_ptr), value :: indT
       type(c_ptr), value :: valT
       type(c_ptr), value :: ptrL
       type(c_ptr), value :: indL
       type(c_ptr), value :: valL
       type(c_ptr), value :: ptrU
       type(c_ptr), value :: indU
       type(c_ptr), value :: valU
       integer(kind(rocblas_status_success)) :: scsrrf_splitlu
    end function rocsolver_scsrrf_splitlu

    !---------------------------------------------
    ! rocsolver_dcsrrf_splitlu
    !---------------------------------------------
    function rocsolver_dcsrrf_splitlu(handle, n, nnzT, ptrT, indT, valT, ptrL, indL, valL, ptrU, &
                                      indU, valU) &
       result(dcsrrf_splitlu) &
       bind(C, name="rocsolver_dcsrrf_splitlu")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nnzT
       type(c_ptr), value :: ptrT
       type(c_ptr), value :: indT
       type(c_ptr), value :: valT
       type(c_ptr), value :: ptrL
       type(c_ptr), value :: indL
       type(c_ptr), value :: valL
       type(c_ptr), value :: ptrU
       type(c_ptr), value :: indU
       type(c_ptr), value :: valU
       integer(kind(rocblas_status_success)) :: dcsrrf_splitlu
    end function rocsolver_dcsrrf_splitlu

    !---------------------------------------------
    ! rocsolver_scsrrf_analysis
    !---------------------------------------------
    function rocsolver_scsrrf_analysis(handle, n, nrhs, nnzM, ptrM, indM, valM, nnzT, ptrT, indT, &
                                       valT, pivP, pivQ, B, ldb, rfinfo) &
       result(scsrrf_analysis) &
       bind(C, name="rocsolver_scsrrf_analysis")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       integer(c_int), value :: nnzM
       type(c_ptr), value :: ptrM
       type(c_ptr), value :: indM
       type(c_ptr), value :: valM
       integer(c_int), value :: nnzT
       type(c_ptr), value :: ptrT
       type(c_ptr), value :: indT
       type(c_ptr), value :: valT
       type(c_ptr), value :: pivP
       type(c_ptr), value :: pivQ
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: rfinfo
       integer(kind(rocblas_status_success)) :: scsrrf_analysis
    end function rocsolver_scsrrf_analysis

    !---------------------------------------------
    ! rocsolver_dcsrrf_analysis
    !---------------------------------------------
    function rocsolver_dcsrrf_analysis(handle, n, nrhs, nnzM, ptrM, indM, valM, nnzT, ptrT, indT, &
                                       valT, pivP, pivQ, B, ldb, rfinfo) &
       result(dcsrrf_analysis) &
       bind(C, name="rocsolver_dcsrrf_analysis")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       integer(c_int), value :: nnzM
       type(c_ptr), value :: ptrM
       type(c_ptr), value :: indM
       type(c_ptr), value :: valM
       integer(c_int), value :: nnzT
       type(c_ptr), value :: ptrT
       type(c_ptr), value :: indT
       type(c_ptr), value :: valT
       type(c_ptr), value :: pivP
       type(c_ptr), value :: pivQ
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: rfinfo
       integer(kind(rocblas_status_success)) :: dcsrrf_analysis
    end function rocsolver_dcsrrf_analysis

    !---------------------------------------------
    ! rocsolver_scsrrf_refactlu
    !---------------------------------------------
    function rocsolver_scsrrf_refactlu(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT, indT, valT, &
                                       pivP, pivQ, rfinfo) &
       result(scsrrf_refactlu) &
       bind(C, name="rocsolver_scsrrf_refactlu")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nnzA
       type(c_ptr), value :: ptrA
       type(c_ptr), value :: indA
       type(c_ptr), value :: valA
       integer(c_int), value :: nnzT
       type(c_ptr), value :: ptrT
       type(c_ptr), value :: indT
       type(c_ptr), value :: valT
       type(c_ptr), value :: pivP
       type(c_ptr), value :: pivQ
       type(c_ptr), value :: rfinfo
       integer(kind(rocblas_status_success)) :: scsrrf_refactlu
    end function rocsolver_scsrrf_refactlu

    !---------------------------------------------
    ! rocsolver_dcsrrf_refactlu
    !---------------------------------------------
    function rocsolver_dcsrrf_refactlu(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT, indT, valT, &
                                       pivP, pivQ, rfinfo) &
       result(dcsrrf_refactlu) &
       bind(C, name="rocsolver_dcsrrf_refactlu")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nnzA
       type(c_ptr), value :: ptrA
       type(c_ptr), value :: indA
       type(c_ptr), value :: valA
       integer(c_int), value :: nnzT
       type(c_ptr), value :: ptrT
       type(c_ptr), value :: indT
       type(c_ptr), value :: valT
       type(c_ptr), value :: pivP
       type(c_ptr), value :: pivQ
       type(c_ptr), value :: rfinfo
       integer(kind(rocblas_status_success)) :: dcsrrf_refactlu
    end function rocsolver_dcsrrf_refactlu

    !---------------------------------------------
    ! rocsolver_scsrrf_refactchol
    !---------------------------------------------
    function rocsolver_scsrrf_refactchol(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT, indT, &
                                         valT, pivQ, rfinfo) &
       result(scsrrf_refactchol) &
       bind(C, name="rocsolver_scsrrf_refactchol")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nnzA
       type(c_ptr), value :: ptrA
       type(c_ptr), value :: indA
       type(c_ptr), value :: valA
       integer(c_int), value :: nnzT
       type(c_ptr), value :: ptrT
       type(c_ptr), value :: indT
       type(c_ptr), value :: valT
       type(c_ptr), value :: pivQ
       type(c_ptr), value :: rfinfo
       integer(kind(rocblas_status_success)) :: scsrrf_refactchol
    end function rocsolver_scsrrf_refactchol

    !---------------------------------------------
    ! rocsolver_dcsrrf_refactchol
    !---------------------------------------------
    function rocsolver_dcsrrf_refactchol(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT, indT, &
                                         valT, pivQ, rfinfo) &
       result(dcsrrf_refactchol) &
       bind(C, name="rocsolver_dcsrrf_refactchol")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nnzA
       type(c_ptr), value :: ptrA
       type(c_ptr), value :: indA
       type(c_ptr), value :: valA
       integer(c_int), value :: nnzT
       type(c_ptr), value :: ptrT
       type(c_ptr), value :: indT
       type(c_ptr), value :: valT
       type(c_ptr), value :: pivQ
       type(c_ptr), value :: rfinfo
       integer(kind(rocblas_status_success)) :: dcsrrf_refactchol
    end function rocsolver_dcsrrf_refactchol

    !---------------------------------------------
    ! rocsolver_scsrrf_solve
    !---------------------------------------------
    function rocsolver_scsrrf_solve(handle, n, nrhs, nnzT, ptrT, indT, valT, pivP, pivQ, B, ldb, &
                                    rfinfo) &
       result(scsrrf_solve) &
       bind(C, name="rocsolver_scsrrf_solve")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       integer(c_int), value :: nnzT
       type(c_ptr), value :: ptrT
       type(c_ptr), value :: indT
       type(c_ptr), value :: valT
       type(c_ptr), value :: pivP
       type(c_ptr), value :: pivQ
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: rfinfo
       integer(kind(rocblas_status_success)) :: scsrrf_solve
    end function rocsolver_scsrrf_solve

    !---------------------------------------------
    ! rocsolver_dcsrrf_solve
    !---------------------------------------------
    function rocsolver_dcsrrf_solve(handle, n, nrhs, nnzT, ptrT, indT, valT, pivP, pivQ, B, ldb, &
                                    rfinfo) &
       result(dcsrrf_solve) &
       bind(C, name="rocsolver_dcsrrf_solve")
       import :: c_ptr, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       integer(c_int), value :: nnzT
       type(c_ptr), value :: ptrT
       type(c_ptr), value :: indT
       type(c_ptr), value :: valT
       type(c_ptr), value :: pivP
       type(c_ptr), value :: pivQ
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: rfinfo
       integer(kind(rocblas_status_success)) :: dcsrrf_solve
    end function rocsolver_dcsrrf_solve

    !---------------------------------------------
    ! rocsolver_ssyevdx
    !---------------------------------------------
    function rocsolver_ssyevdx(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, nev, W, Z, &
                               ldz, myInfo) &
       result(ssyevdx) &
       bind(C, name="rocsolver_ssyevdx")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ssyevdx
    end function rocsolver_ssyevdx

    !---------------------------------------------
    ! rocsolver_dsyevdx
    !---------------------------------------------
    function rocsolver_dsyevdx(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, nev, W, Z, &
                               ldz, myInfo) &
       result(dsyevdx) &
       bind(C, name="rocsolver_dsyevdx")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dsyevdx
    end function rocsolver_dsyevdx

    !---------------------------------------------
    ! rocsolver_cheevdx
    !---------------------------------------------
    function rocsolver_cheevdx(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, nev, W, Z, &
                               ldz, myInfo) &
       result(cheevdx) &
       bind(C, name="rocsolver_cheevdx")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: cheevdx
    end function rocsolver_cheevdx

    !---------------------------------------------
    ! rocsolver_zheevdx
    !---------------------------------------------
    function rocsolver_zheevdx(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, nev, W, Z, &
                               ldz, myInfo) &
       result(zheevdx) &
       bind(C, name="rocsolver_zheevdx")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zheevdx
    end function rocsolver_zheevdx

    !---------------------------------------------
    ! rocsolver_ssyevdx_batched
    !---------------------------------------------
    function rocsolver_ssyevdx_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, &
                                       nev, W, strideW, Z, ldz, myInfo, batch_count) &
       result(ssyevdx_batched) &
       bind(C, name="rocsolver_ssyevdx_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_float, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssyevdx_batched
    end function rocsolver_ssyevdx_batched

    !---------------------------------------------
    ! rocsolver_dsyevdx_batched
    !---------------------------------------------
    function rocsolver_dsyevdx_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, &
                                       nev, W, strideW, Z, ldz, myInfo, batch_count) &
       result(dsyevdx_batched) &
       bind(C, name="rocsolver_dsyevdx_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_double, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsyevdx_batched
    end function rocsolver_dsyevdx_batched

    !---------------------------------------------
    ! rocsolver_cheevdx_batched
    !---------------------------------------------
    function rocsolver_cheevdx_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, &
                                       nev, W, strideW, Z, ldz, myInfo, batch_count) &
       result(cheevdx_batched) &
       bind(C, name="rocsolver_cheevdx_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_float, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cheevdx_batched
    end function rocsolver_cheevdx_batched

    !---------------------------------------------
    ! rocsolver_zheevdx_batched
    !---------------------------------------------
    function rocsolver_zheevdx_batched(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, &
                                       nev, W, strideW, Z, ldz, myInfo, batch_count) &
       result(zheevdx_batched) &
       bind(C, name="rocsolver_zheevdx_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_double, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zheevdx_batched
    end function rocsolver_zheevdx_batched

    !---------------------------------------------
    ! rocsolver_ssyevdx_strided_batched
    !---------------------------------------------
    function rocsolver_ssyevdx_strided_batched(handle, evect, erange, uplo, n, A, lda, strideA, &
                                               vl, vu, il, iu, nev, W, strideW, Z, ldz, strideZ, &
                                               myInfo, batch_count) &
       result(ssyevdx_strided_batched) &
       bind(C, name="rocsolver_ssyevdx_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_int64_t, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       integer(c_int64_t), value :: strideZ
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssyevdx_strided_batched
    end function rocsolver_ssyevdx_strided_batched

    !---------------------------------------------
    ! rocsolver_dsyevdx_strided_batched
    !---------------------------------------------
    function rocsolver_dsyevdx_strided_batched(handle, evect, erange, uplo, n, A, lda, strideA, &
                                               vl, vu, il, iu, nev, W, strideW, Z, ldz, strideZ, &
                                               myInfo, batch_count) &
       result(dsyevdx_strided_batched) &
       bind(C, name="rocsolver_dsyevdx_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_int64_t, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       integer(c_int64_t), value :: strideZ
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsyevdx_strided_batched
    end function rocsolver_dsyevdx_strided_batched

    !---------------------------------------------
    ! rocsolver_cheevdx_strided_batched
    !---------------------------------------------
    function rocsolver_cheevdx_strided_batched(handle, evect, erange, uplo, n, A, lda, strideA, &
                                               vl, vu, il, iu, nev, W, strideW, Z, ldz, strideZ, &
                                               myInfo, batch_count) &
       result(cheevdx_strided_batched) &
       bind(C, name="rocsolver_cheevdx_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_int64_t, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       integer(c_int64_t), value :: strideZ
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: cheevdx_strided_batched
    end function rocsolver_cheevdx_strided_batched

    !---------------------------------------------
    ! rocsolver_zheevdx_strided_batched
    !---------------------------------------------
    function rocsolver_zheevdx_strided_batched(handle, evect, erange, uplo, n, A, lda, strideA, &
                                               vl, vu, il, iu, nev, W, strideW, Z, ldz, strideZ, &
                                               myInfo, batch_count) &
       result(zheevdx_strided_batched) &
       bind(C, name="rocsolver_zheevdx_strided_batched")
       import :: c_ptr, rocblas_evect_original, rocblas_erange_all, rocblas_fill_upper, c_int, &
                 c_int64_t, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       integer(c_int64_t), value :: strideZ
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zheevdx_strided_batched
    end function rocsolver_zheevdx_strided_batched

    !---------------------------------------------
    ! rocsolver_ssygvdx
    !---------------------------------------------
    function rocsolver_ssygvdx(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, &
                               iu, nev, W, Z, ldz, myInfo) &
       result(ssygvdx) &
       bind(C, name="rocsolver_ssygvdx")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: ssygvdx
    end function rocsolver_ssygvdx

    !---------------------------------------------
    ! rocsolver_dsygvdx
    !---------------------------------------------
    function rocsolver_dsygvdx(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, &
                               iu, nev, W, Z, ldz, myInfo) &
       result(dsygvdx) &
       bind(C, name="rocsolver_dsygvdx")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: dsygvdx
    end function rocsolver_dsygvdx

    !---------------------------------------------
    ! rocsolver_chegvdx
    !---------------------------------------------
    function rocsolver_chegvdx(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, &
                               iu, nev, W, Z, ldz, myInfo) &
       result(chegvdx) &
       bind(C, name="rocsolver_chegvdx")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: chegvdx
    end function rocsolver_chegvdx

    !---------------------------------------------
    ! rocsolver_zhegvdx
    !---------------------------------------------
    function rocsolver_zhegvdx(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, &
                               iu, nev, W, Z, ldz, myInfo) &
       result(zhegvdx) &
       bind(C, name="rocsolver_zhegvdx")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: myInfo
       integer(kind(rocblas_status_success)) :: zhegvdx
    end function rocsolver_zhegvdx

    !---------------------------------------------
    ! rocsolver_ssygvdx_batched
    !---------------------------------------------
    function rocsolver_ssygvdx_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, &
                                       vu, il, iu, nev, W, strideW, Z, ldz, myInfo, batch_count) &
       result(ssygvdx_batched) &
       bind(C, name="rocsolver_ssygvdx_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_float, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssygvdx_batched
    end function rocsolver_ssygvdx_batched

    !---------------------------------------------
    ! rocsolver_dsygvdx_batched
    !---------------------------------------------
    function rocsolver_dsygvdx_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, &
                                       vu, il, iu, nev, W, strideW, Z, ldz, myInfo, batch_count) &
       result(dsygvdx_batched) &
       bind(C, name="rocsolver_dsygvdx_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_double, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsygvdx_batched
    end function rocsolver_dsygvdx_batched

    !---------------------------------------------
    ! rocsolver_chegvdx_batched
    !---------------------------------------------
    function rocsolver_chegvdx_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, &
                                       vu, il, iu, nev, W, strideW, Z, ldz, myInfo, batch_count) &
       result(chegvdx_batched) &
       bind(C, name="rocsolver_chegvdx_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_float, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chegvdx_batched
    end function rocsolver_chegvdx_batched

    !---------------------------------------------
    ! rocsolver_zhegvdx_batched
    !---------------------------------------------
    function rocsolver_zhegvdx_batched(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, &
                                       vu, il, iu, nev, W, strideW, Z, ldz, myInfo, batch_count) &
       result(zhegvdx_batched) &
       bind(C, name="rocsolver_zhegvdx_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_double, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhegvdx_batched
    end function rocsolver_zhegvdx_batched

    !---------------------------------------------
    ! rocsolver_ssygvdx_strided_batched
    !---------------------------------------------
    function rocsolver_ssygvdx_strided_batched(handle, itype, evect, erange, uplo, n, A, lda, &
                                               strideA, B, ldb, strideB, vl, vu, il, iu, nev, W, &
                                               strideW, Z, ldz, strideZ, myInfo, batch_count) &
       result(ssygvdx_strided_batched) &
       bind(C, name="rocsolver_ssygvdx_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_int64_t, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       integer(c_int64_t), value :: strideZ
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ssygvdx_strided_batched
    end function rocsolver_ssygvdx_strided_batched

    !---------------------------------------------
    ! rocsolver_dsygvdx_strided_batched
    !---------------------------------------------
    function rocsolver_dsygvdx_strided_batched(handle, itype, evect, erange, uplo, n, A, lda, &
                                               strideA, B, ldb, strideB, vl, vu, il, iu, nev, W, &
                                               strideW, Z, ldz, strideZ, myInfo, batch_count) &
       result(dsygvdx_strided_batched) &
       bind(C, name="rocsolver_dsygvdx_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_int64_t, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       integer(c_int64_t), value :: strideZ
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dsygvdx_strided_batched
    end function rocsolver_dsygvdx_strided_batched

    !---------------------------------------------
    ! rocsolver_chegvdx_strided_batched
    !---------------------------------------------
    function rocsolver_chegvdx_strided_batched(handle, itype, evect, erange, uplo, n, A, lda, &
                                               strideA, B, ldb, strideB, vl, vu, il, iu, nev, W, &
                                               strideW, Z, ldz, strideZ, myInfo, batch_count) &
       result(chegvdx_strided_batched) &
       bind(C, name="rocsolver_chegvdx_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_int64_t, c_float, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       integer(c_int64_t), value :: strideZ
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: chegvdx_strided_batched
    end function rocsolver_chegvdx_strided_batched

    !---------------------------------------------
    ! rocsolver_zhegvdx_strided_batched
    !---------------------------------------------
    function rocsolver_zhegvdx_strided_batched(handle, itype, evect, erange, uplo, n, A, lda, &
                                               strideA, B, ldb, strideB, vl, vu, il, iu, nev, W, &
                                               strideW, Z, ldz, strideZ, myInfo, batch_count) &
       result(zhegvdx_strided_batched) &
       bind(C, name="rocsolver_zhegvdx_strided_batched")
       import :: c_ptr, rocblas_eform_ax, rocblas_evect_original, rocblas_erange_all, &
                 rocblas_fill_upper, c_int, c_int64_t, c_double, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocblas_eform_ax)), value :: itype
       integer(kind(rocblas_evect_original)), value :: evect
       integer(kind(rocblas_erange_all)), value :: erange
       integer(kind(rocblas_fill_upper)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int64_t), value :: strideB
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       type(c_ptr), value :: nev
       type(c_ptr), value :: W
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: Z
       integer(c_int), value :: ldz
       integer(c_int64_t), value :: strideZ
       type(c_ptr), value :: myInfo
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zhegvdx_strided_batched
    end function rocsolver_zhegvdx_strided_batched

    !---------------------------------------------
    ! rocsolver_scholqr
    !---------------------------------------------
    function rocsolver_scholqr(handle, cholshift, cholnum, m, n, A, lda, W, ldw, sigma, nr) &
       result(scholqr) &
       bind(C, name="rocsolver_scholqr")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int), value :: ldw
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(kind(rocblas_status_success)) :: scholqr
    end function rocsolver_scholqr

    !---------------------------------------------
    ! rocsolver_dcholqr
    !---------------------------------------------
    function rocsolver_dcholqr(handle, cholshift, cholnum, m, n, A, lda, W, ldw, sigma, nr) &
       result(dcholqr) &
       bind(C, name="rocsolver_dcholqr")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int), value :: ldw
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(kind(rocblas_status_success)) :: dcholqr
    end function rocsolver_dcholqr

    !---------------------------------------------
    ! rocsolver_ccholqr
    !---------------------------------------------
    function rocsolver_ccholqr(handle, cholshift, cholnum, m, n, A, lda, W, ldw, sigma, nr) &
       result(ccholqr) &
       bind(C, name="rocsolver_ccholqr")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int), value :: ldw
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(kind(rocblas_status_success)) :: ccholqr
    end function rocsolver_ccholqr

    !---------------------------------------------
    ! rocsolver_zcholqr
    !---------------------------------------------
    function rocsolver_zcholqr(handle, cholshift, cholnum, m, n, A, lda, W, ldw, sigma, nr) &
       result(zcholqr) &
       bind(C, name="rocsolver_zcholqr")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int), value :: ldw
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(kind(rocblas_status_success)) :: zcholqr
    end function rocsolver_zcholqr

    !---------------------------------------------
    ! rocsolver_scholqr_64
    !---------------------------------------------
    function rocsolver_scholqr_64(handle, cholshift, cholnum, m, n, A, lda, W, ldw, sigma, nr) &
       result(scholqr_64) &
       bind(C, name="rocsolver_scholqr_64")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: W
       integer(c_int64_t), value :: ldw
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(kind(rocblas_status_success)) :: scholqr_64
    end function rocsolver_scholqr_64

    !---------------------------------------------
    ! rocsolver_dcholqr_64
    !---------------------------------------------
    function rocsolver_dcholqr_64(handle, cholshift, cholnum, m, n, A, lda, W, ldw, sigma, nr) &
       result(dcholqr_64) &
       bind(C, name="rocsolver_dcholqr_64")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: W
       integer(c_int64_t), value :: ldw
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(kind(rocblas_status_success)) :: dcholqr_64
    end function rocsolver_dcholqr_64

    !---------------------------------------------
    ! rocsolver_ccholqr_64
    !---------------------------------------------
    function rocsolver_ccholqr_64(handle, cholshift, cholnum, m, n, A, lda, W, ldw, sigma, nr) &
       result(ccholqr_64) &
       bind(C, name="rocsolver_ccholqr_64")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: W
       integer(c_int64_t), value :: ldw
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(kind(rocblas_status_success)) :: ccholqr_64
    end function rocsolver_ccholqr_64

    !---------------------------------------------
    ! rocsolver_zcholqr_64
    !---------------------------------------------
    function rocsolver_zcholqr_64(handle, cholshift, cholnum, m, n, A, lda, W, ldw, sigma, nr) &
       result(zcholqr_64) &
       bind(C, name="rocsolver_zcholqr_64")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: W
       integer(c_int64_t), value :: ldw
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(kind(rocblas_status_success)) :: zcholqr_64
    end function rocsolver_zcholqr_64

    !---------------------------------------------
    ! rocsolver_scholqr_batched
    !---------------------------------------------
    function rocsolver_scholqr_batched(handle, cholshift, cholnum, m, n, A, lda, W, ldw, strideW, &
                                       sigma, nr, batch_count) &
       result(scholqr_batched) &
       bind(C, name="rocsolver_scholqr_batched")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int), value :: ldw
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: scholqr_batched
    end function rocsolver_scholqr_batched

    !---------------------------------------------
    ! rocsolver_dcholqr_batched
    !---------------------------------------------
    function rocsolver_dcholqr_batched(handle, cholshift, cholnum, m, n, A, lda, W, ldw, strideW, &
                                       sigma, nr, batch_count) &
       result(dcholqr_batched) &
       bind(C, name="rocsolver_dcholqr_batched")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int), value :: ldw
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dcholqr_batched
    end function rocsolver_dcholqr_batched

    !---------------------------------------------
    ! rocsolver_ccholqr_batched
    !---------------------------------------------
    function rocsolver_ccholqr_batched(handle, cholshift, cholnum, m, n, A, lda, W, ldw, strideW, &
                                       sigma, nr, batch_count) &
       result(ccholqr_batched) &
       bind(C, name="rocsolver_ccholqr_batched")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int), value :: ldw
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ccholqr_batched
    end function rocsolver_ccholqr_batched

    !---------------------------------------------
    ! rocsolver_zcholqr_batched
    !---------------------------------------------
    function rocsolver_zcholqr_batched(handle, cholshift, cholnum, m, n, A, lda, W, ldw, strideW, &
                                       sigma, nr, batch_count) &
       result(zcholqr_batched) &
       bind(C, name="rocsolver_zcholqr_batched")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int), value :: ldw
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zcholqr_batched
    end function rocsolver_zcholqr_batched

    !---------------------------------------------
    ! rocsolver_scholqr_batched_64
    !---------------------------------------------
    function rocsolver_scholqr_batched_64(handle, cholshift, cholnum, m, n, A, lda, W, ldw, &
                                          strideW, sigma, nr, batch_count) &
       result(scholqr_batched_64) &
       bind(C, name="rocsolver_scholqr_batched_64")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: W
       integer(c_int64_t), value :: ldw
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: scholqr_batched_64
    end function rocsolver_scholqr_batched_64

    !---------------------------------------------
    ! rocsolver_dcholqr_batched_64
    !---------------------------------------------
    function rocsolver_dcholqr_batched_64(handle, cholshift, cholnum, m, n, A, lda, W, ldw, &
                                          strideW, sigma, nr, batch_count) &
       result(dcholqr_batched_64) &
       bind(C, name="rocsolver_dcholqr_batched_64")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: W
       integer(c_int64_t), value :: ldw
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dcholqr_batched_64
    end function rocsolver_dcholqr_batched_64

    !---------------------------------------------
    ! rocsolver_ccholqr_batched_64
    !---------------------------------------------
    function rocsolver_ccholqr_batched_64(handle, cholshift, cholnum, m, n, A, lda, W, ldw, &
                                          strideW, sigma, nr, batch_count) &
       result(ccholqr_batched_64) &
       bind(C, name="rocsolver_ccholqr_batched_64")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: W
       integer(c_int64_t), value :: ldw
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: ccholqr_batched_64
    end function rocsolver_ccholqr_batched_64

    !---------------------------------------------
    ! rocsolver_zcholqr_batched_64
    !---------------------------------------------
    function rocsolver_zcholqr_batched_64(handle, cholshift, cholnum, m, n, A, lda, W, ldw, &
                                          strideW, sigma, nr, batch_count) &
       result(zcholqr_batched_64) &
       bind(C, name="rocsolver_zcholqr_batched_64")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: W
       integer(c_int64_t), value :: ldw
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zcholqr_batched_64
    end function rocsolver_zcholqr_batched_64

    !---------------------------------------------
    ! rocsolver_scholqr_strided_batched
    !---------------------------------------------
    function rocsolver_scholqr_strided_batched(handle, cholshift, cholnum, m, n, A, lda, strideA, &
                                               W, ldw, strideW, sigma, nr, batch_count) &
       result(scholqr_strided_batched) &
       bind(C, name="rocsolver_scholqr_strided_batched")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: W
       integer(c_int), value :: ldw
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: scholqr_strided_batched
    end function rocsolver_scholqr_strided_batched

    !---------------------------------------------
    ! rocsolver_dcholqr_strided_batched
    !---------------------------------------------
    function rocsolver_dcholqr_strided_batched(handle, cholshift, cholnum, m, n, A, lda, strideA, &
                                               W, ldw, strideW, sigma, nr, batch_count) &
       result(dcholqr_strided_batched) &
       bind(C, name="rocsolver_dcholqr_strided_batched")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: W
       integer(c_int), value :: ldw
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: dcholqr_strided_batched
    end function rocsolver_dcholqr_strided_batched

    !---------------------------------------------
    ! rocsolver_ccholqr_strided_batched
    !---------------------------------------------
    function rocsolver_ccholqr_strided_batched(handle, cholshift, cholnum, m, n, A, lda, strideA, &
                                               W, ldw, strideW, sigma, nr, batch_count) &
       result(ccholqr_strided_batched) &
       bind(C, name="rocsolver_ccholqr_strided_batched")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: W
       integer(c_int), value :: ldw
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: ccholqr_strided_batched
    end function rocsolver_ccholqr_strided_batched

    !---------------------------------------------
    ! rocsolver_zcholqr_strided_batched
    !---------------------------------------------
    function rocsolver_zcholqr_strided_batched(handle, cholshift, cholnum, m, n, A, lda, strideA, &
                                               W, ldw, strideW, sigma, nr, batch_count) &
       result(zcholqr_strided_batched) &
       bind(C, name="rocsolver_zcholqr_strided_batched")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: W
       integer(c_int), value :: ldw
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(c_int), value :: batch_count
       integer(kind(rocblas_status_success)) :: zcholqr_strided_batched
    end function rocsolver_zcholqr_strided_batched

    !---------------------------------------------
    ! rocsolver_scholqr_strided_batched_64
    !---------------------------------------------
    function rocsolver_scholqr_strided_batched_64(handle, cholshift, cholnum, m, n, A, lda, &
                                                  strideA, W, ldw, strideW, sigma, nr, &
                                                  batch_count) &
       result(scholqr_strided_batched_64) &
       bind(C, name="rocsolver_scholqr_strided_batched_64")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: W
       integer(c_int64_t), value :: ldw
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: scholqr_strided_batched_64
    end function rocsolver_scholqr_strided_batched_64

    !---------------------------------------------
    ! rocsolver_dcholqr_strided_batched_64
    !---------------------------------------------
    function rocsolver_dcholqr_strided_batched_64(handle, cholshift, cholnum, m, n, A, lda, &
                                                  strideA, W, ldw, strideW, sigma, nr, &
                                                  batch_count) &
       result(dcholqr_strided_batched_64) &
       bind(C, name="rocsolver_dcholqr_strided_batched_64")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: W
       integer(c_int64_t), value :: ldw
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: dcholqr_strided_batched_64
    end function rocsolver_dcholqr_strided_batched_64

    !---------------------------------------------
    ! rocsolver_ccholqr_strided_batched_64
    !---------------------------------------------
    function rocsolver_ccholqr_strided_batched_64(handle, cholshift, cholnum, m, n, A, lda, &
                                                  strideA, W, ldw, strideW, sigma, nr, &
                                                  batch_count) &
       result(ccholqr_strided_batched_64) &
       bind(C, name="rocsolver_ccholqr_strided_batched_64")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: W
       integer(c_int64_t), value :: ldw
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: ccholqr_strided_batched_64
    end function rocsolver_ccholqr_strided_batched_64

    !---------------------------------------------
    ! rocsolver_zcholqr_strided_batched_64
    !---------------------------------------------
    function rocsolver_zcholqr_strided_batched_64(handle, cholshift, cholnum, m, n, A, lda, &
                                                  strideA, W, ldw, strideW, sigma, nr, &
                                                  batch_count) &
       result(zcholqr_strided_batched_64) &
       bind(C, name="rocsolver_zcholqr_strided_batched_64")
       import :: c_ptr, rocsolver_cholqr_shift_none, c_int, c_int64_t, rocblas_status_success
       type(c_ptr), value :: handle
       integer(kind(rocsolver_cholqr_shift_none)), value :: cholshift
       integer(c_int), value :: cholnum
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: W
       integer(c_int64_t), value :: ldw
       integer(c_int64_t), value :: strideW
       type(c_ptr), value :: sigma
       type(c_ptr), value :: nr
       integer(c_int64_t), value :: batch_count
       integer(kind(rocblas_status_success)) :: zcholqr_strided_batched_64
    end function rocsolver_zcholqr_strided_batched_64

  end interface


  contains

    subroutine rocsolverCheck(status)
      implicit none
      integer(kind(ROCBLAS_STATUS_SUCCESS)) :: status
      if (status /= ROCBLAS_STATUS_SUCCESS) then
        write (*, *) "ROCSOLVER ERROR: code = ", status
        stop 1
      end if
    end subroutine rocsolverCheck
end module rocsolver
