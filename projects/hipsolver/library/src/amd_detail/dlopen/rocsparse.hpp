/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#include "lib_macros.hpp"

#ifdef HAVE_ROCSPARSE
#include <rocsparse/rocsparse.h>
#else

#if defined(rocsparse_ILP64)
typedef int64_t rocsparse_int;
#else
typedef int32_t rocsparse_int;
#endif

// type definitions
typedef enum rocsparse_status_
{
    rocsparse_status_success                 = 0,
    rocsparse_status_invalid_handle          = 1,
    rocsparse_status_not_implemented         = 2,
    rocsparse_status_invalid_pointer         = 3,
    rocsparse_status_invalid_size            = 4,
    rocsparse_status_memory_error            = 5,
    rocsparse_status_internal_error          = 6,
    rocsparse_status_invalid_value           = 7,
    rocsparse_status_arch_mismatch           = 8,
    rocsparse_status_zero_pivot              = 9,
    rocsparse_status_not_initialized         = 10,
    rocsparse_status_type_mismatch           = 11,
    rocsparse_status_requires_sorted_storage = 12,
    rocsparse_status_thrown_exception        = 13,
    rocsparse_status_continue                = 14
} rocsparse_status;

typedef enum rocsparse_index_base_
{
    rocsparse_index_base_zero = 0, /**< zero based indexing. */
    rocsparse_index_base_one  = 1 /**< one based indexing. */
} rocsparse_index_base;

typedef enum rocsparse_matrix_type_
{
    rocsparse_matrix_type_general    = 0, /**< general matrix type. */
    rocsparse_matrix_type_symmetric  = 1, /**< symmetric matrix type. */
    rocsparse_matrix_type_hermitian  = 2, /**< hermitian matrix type. */
    rocsparse_matrix_type_triangular = 3 /**< triangular matrix type. */
} rocsparse_matrix_type;

typedef struct _rocsparse_handle*    rocsparse_handle;
typedef struct _rocsparse_mat_descr* rocsparse_mat_descr;

typedef struct
{
    float x, y;
} rocsparse_float_complex;

typedef struct
{
    double x, y;
} rocsparse_double_complex;

HIPSOLVER_BEGIN_NAMESPACE

// function declarations
typedef rocsparse_status (*fp_rocsparse_create_handle)(rocsparse_handle* handle);
extern fp_rocsparse_create_handle g_rocsparse_create_handle;
#define rocsparse_create_handle ::hipsolver::g_rocsparse_create_handle

typedef rocsparse_status (*fp_rocsparse_destroy_handle)(rocsparse_handle handle);
extern fp_rocsparse_destroy_handle g_rocsparse_destroy_handle;
#define rocsparse_destroy_handle ::hipsolver::g_rocsparse_destroy_handle

typedef rocsparse_status (*fp_rocsparse_create_mat_descr)(rocsparse_mat_descr* descr);
extern fp_rocsparse_create_mat_descr g_rocsparse_create_mat_descr;
#define rocsparse_create_mat_descr ::hipsolver::g_rocsparse_create_mat_descr

typedef rocsparse_status (*fp_rocsparse_destroy_mat_descr)(rocsparse_mat_descr descr);
extern fp_rocsparse_destroy_mat_descr g_rocsparse_destroy_mat_descr;
#define rocsparse_destroy_mat_descr ::hipsolver::g_rocsparse_destroy_mat_descr

typedef rocsparse_matrix_type (*fp_rocsparse_get_mat_type)(rocsparse_mat_descr descr);
extern fp_rocsparse_get_mat_type g_rocsparse_get_mat_type;
#define rocsparse_get_mat_type ::hipsolver::g_rocsparse_get_mat_type

typedef rocsparse_index_base (*fp_rocsparse_get_mat_index_base)(rocsparse_mat_descr descr);
extern fp_rocsparse_get_mat_index_base g_rocsparse_get_mat_index_base;
#define rocsparse_get_mat_index_base ::hipsolver::g_rocsparse_get_mat_index_base

typedef rocsparse_status (*fp_rocsparse_scsr2dense)(rocsparse_handle          handle,
                                                    rocsparse_int             m,
                                                    rocsparse_int             n,
                                                    const rocsparse_mat_descr descr,
                                                    const float*              csr_val,
                                                    const rocsparse_int*      csr_row_ptr,
                                                    const rocsparse_int*      csr_col_ind,
                                                    float*                    A,
                                                    rocsparse_int             ld);
extern fp_rocsparse_scsr2dense g_rocsparse_scsr2dense;
#define rocsparse_scsr2dense ::hipsolver::g_rocsparse_scsr2dense

typedef rocsparse_status (*fp_rocsparse_dcsr2dense)(rocsparse_handle          handle,
                                                    rocsparse_int             m,
                                                    rocsparse_int             n,
                                                    const rocsparse_mat_descr descr,
                                                    const double*             csr_val,
                                                    const rocsparse_int*      csr_row_ptr,
                                                    const rocsparse_int*      csr_col_ind,
                                                    double*                   A,
                                                    rocsparse_int             ld);
extern fp_rocsparse_dcsr2dense g_rocsparse_dcsr2dense;
#define rocsparse_dcsr2dense ::hipsolver::g_rocsparse_dcsr2dense

typedef rocsparse_status (*fp_rocsparse_ccsr2dense)(rocsparse_handle               handle,
                                                    rocsparse_int                  m,
                                                    rocsparse_int                  n,
                                                    const rocsparse_mat_descr      descr,
                                                    const rocsparse_float_complex* csr_val,
                                                    const rocsparse_int*           csr_row_ptr,
                                                    const rocsparse_int*           csr_col_ind,
                                                    rocsparse_float_complex*       A,
                                                    rocsparse_int                  ld);
extern fp_rocsparse_ccsr2dense g_rocsparse_ccsr2dense;
#define rocsparse_ccsr2dense ::hipsolver::g_rocsparse_ccsr2dense

typedef rocsparse_status (*fp_rocsparse_zcsr2dense)(rocsparse_handle                handle,
                                                    rocsparse_int                   m,
                                                    rocsparse_int                   n,
                                                    const rocsparse_mat_descr       descr,
                                                    const rocsparse_double_complex* csr_val,
                                                    const rocsparse_int*            csr_row_ptr,
                                                    const rocsparse_int*            csr_col_ind,
                                                    rocsparse_double_complex*       A,
                                                    rocsparse_int                   ld);
extern fp_rocsparse_zcsr2dense g_rocsparse_zcsr2dense;
#define rocsparse_zcsr2dense ::hipsolver::g_rocsparse_zcsr2dense

HIPSOLVER_END_NAMESPACE

#endif // HAVE_ROCSPARSE

HIPSOLVER_BEGIN_NAMESPACE

// load methods
bool try_load_rocsparse();

HIPSOLVER_END_NAMESPACE
