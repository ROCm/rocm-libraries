/* **************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

//
//
// IMPORTANT:
//
// THESE ALIASES ARE NOT MAINTAINED ANYMORE. THEY EXIST ONLY FOR BACKWARDS
// COMPATIBILITY. USE ROCBLAS TYPES AND FUNCTIONS DIRECTLY.
//
//

#ifndef ROCSOLVER_ALIASES_H
#define ROCSOLVER_ALIASES_H

#include "rocsolver-export.h"
#include "rocsolver-extra-types.h"
#include <rocblas/rocblas.h>

#ifndef ROCSOLVER_DEPRECATED_X
#if defined(__GNUC__)
#define ROCSOLVER_DEPRECATED_X(x) __attribute__((deprecated(x))) // GCC or Clang
#elif defined(_MSC_VER)
#define ROCSOLVER_DEPRECATED_X(x) __declspec(deprecated(x)) // MSVC
#else
#define ROCSOLVER_DEPRECATED_X(x) // other compilers
#endif
#endif

// rocblas original types

/*! \deprecated Use \c rocblas_int.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_int") typedef rocblas_int rocsolver_int;

/*! \deprecated Use \c rocblas_stride.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_stride") typedef rocblas_stride rocsolver_stride;

/*! \deprecated Use \c rocblas_float_complex.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_float_complex")
typedef rocblas_float_complex rocsolver_float_complex;

/*! \deprecated Use \c rocblas_double_complex.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_double_complex")
typedef rocblas_double_complex rocsolver_double_complex;

/*! \deprecated Use \c rocblas_half.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_half") typedef rocblas_half rocsolver_half;

/*! \deprecated Use \c rocblas_handle.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_handle") typedef rocblas_handle rocsolver_handle;

/*! \deprecated Use \c rocblas_operation.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_operation") typedef rocblas_operation rocsolver_operation;

/*! \deprecated Use \c rocblas_fill.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_fill") typedef rocblas_fill rocsolver_fill;

/*! \deprecated Use \c rocblas_diagonal.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_diagonal") typedef rocblas_diagonal rocsolver_diagonal;

/*! \deprecated Use \c rocblas_side.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_side") typedef rocblas_side rocsolver_side;

/*! \deprecated Use \c rocblas_status.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_status") typedef rocblas_status rocsolver_status;

/*! \deprecated Use \c rocblas_layer_mode.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_layer_mode") typedef rocblas_layer_mode rocsolver_layer_mode;

// extras types only used in rocsolver

/*! \deprecated Use \c rocblas_direct
 */
ROCSOLVER_DEPRECATED_X("use rocblas_direct") typedef rocblas_direct rocsolver_direction;

/*! \deprecated Use \c rocblas_storev.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_storev") typedef rocblas_storev rocsolver_storev;

// auxiliaries
#ifdef __cplusplus
extern "C" {
#endif

// deprecated functions use deprecated types, so ignore the warnings
#if defined(__GNUC__) // GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER) // MSVC
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

/*! \deprecated Use \c rocblas_create_handle.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_create_handle")
ROCSOLVER_EXPORT rocsolver_status rocsolver_create_handle(rocsolver_handle* handle);

/*! \deprecated Use \c rocblas_destroy_handle.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_destroy_handle")
ROCSOLVER_EXPORT rocsolver_status rocsolver_destroy_handle(rocsolver_handle handle);

// rocblas_add_stream was removed in ROCm 3.6; use rocblas_set_stream
//
// ROCSOLVER_EXPORT rocsolver_status
// rocsolver_add_stream(rocsolver_handle handle, hipStream_t stream) {
//  return rocblas_add_stream(handle, stream);
//}

/*! \deprecated Use \c rocblas_set_stream.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_set_stream")
ROCSOLVER_EXPORT rocsolver_status rocsolver_set_stream(rocsolver_handle handle, hipStream_t stream);

/*! \deprecated Use \c rocblas_get_stream.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_get_stream")
ROCSOLVER_EXPORT rocsolver_status rocsolver_get_stream(rocsolver_handle handle, hipStream_t* stream);

/*! \deprecated Use \c rocblas_set_vector.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_set_vector")
ROCSOLVER_EXPORT rocsolver_status rocsolver_set_vector(rocsolver_int n,
                                                       rocsolver_int elem_size,
                                                       const void* x,
                                                       rocsolver_int incx,
                                                       void* y,
                                                       rocsolver_int incy);

/*! \deprecated Use \c rocblas_get_vector.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_get_vector")
ROCSOLVER_EXPORT rocsolver_status rocsolver_get_vector(rocsolver_int n,
                                                       rocsolver_int elem_size,
                                                       const void* x,
                                                       rocsolver_int incx,
                                                       void* y,
                                                       rocsolver_int incy);

/*! \deprecated Use \c rocblas_set_matrix.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_set_matrix")
ROCSOLVER_EXPORT rocsolver_status rocsolver_set_matrix(rocsolver_int rows,
                                                       rocsolver_int cols,
                                                       rocsolver_int elem_size,
                                                       const void* a,
                                                       rocsolver_int lda,
                                                       void* b,
                                                       rocsolver_int ldb);

/*! \deprecated Use \c rocblas_get_matrix.
 */
ROCSOLVER_DEPRECATED_X("use rocblas_get_matrix")
ROCSOLVER_EXPORT rocsolver_status rocsolver_get_matrix(rocsolver_int rows,
                                                       rocsolver_int cols,
                                                       rocsolver_int elem_size,
                                                       const void* a,
                                                       rocsolver_int lda,
                                                       void* b,
                                                       rocsolver_int ldb);

// re-enable deprecation warnings
#if defined(__GNUC__) // GCC or Clang
#pragma GCC diagnostic pop
#elif defined(_MSC_VER) // MSVC
#pragma warning(pop)
#endif

#ifdef __cplusplus
}
#endif

#undef ROCSOLVER_DEPRECATED_X

#endif /* ROCSOLVER_ALIASES_H */
