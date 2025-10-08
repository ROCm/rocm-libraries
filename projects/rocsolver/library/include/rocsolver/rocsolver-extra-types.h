/* **************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCSOLVER_EXTRA_TYPES_H
#define ROCSOLVER_EXTRA_TYPES_H

#include <stdint.h>

/*! \brief Used to specify the logging layer mode using a bitwise combination
 *of rocblas_layer_mode values.
 ********************************************************************************/
typedef uint32_t rocblas_layer_mode_flags;

/*! \brief Used to expand the logging layer modes offered for rocSOLVER logging.
 ********************************************************************************/
typedef enum rocblas_layer_mode_ex_
{
    rocblas_layer_mode_ex_log_kernel = 0x10, /**< Enable logging for kernel calls. */
} rocblas_layer_mode_ex;

/*! \brief Used to specify the order in which multiple Householder matrices or plane rotations are
 *applied together (see the documentation of the specific routines for more details).
 ********************************************************************************/
typedef enum rocblas_direct_
{
    rocblas_forward_direction = 171, /**< Forward ordering. */
    rocblas_backward_direction = 172, /**< Backward ordering. */
} rocblas_direct;

/*! \brief Used to specify how householder vectors are stored in a matrix of
 *vectors
 ********************************************************************************/
typedef enum rocblas_storev_
{
    rocblas_column_wise = 181, /**< Householder vectors are stored in the columns of a matrix. */
    rocblas_row_wise = 182, /**< Householder vectors are stored in the rows of a matrix. */
} rocblas_storev;

/*! \brief Used to specify how the singular vectors are to be computed and
 *stored
 ********************************************************************************/
typedef enum rocblas_svect_
{
    rocblas_svect_all = 191, /**< The entire associated orthogonal/unitary matrix is computed. */
    rocblas_svect_singular = 192, /**< Only the singular vectors are computed and
                                       stored in output array. */
    rocblas_svect_overwrite = 193, /**< Only the singular vectors are computed and
                                        overwrite the input matrix. */
    rocblas_svect_none = 194, /**< No singular vectors are computed. */
} rocblas_svect;

/*! \brief Used to enable the use of fast algorithms (with out-of-place
 *computations) in some of the routines
 ********************************************************************************/
typedef enum rocblas_workmode_
{
    rocblas_outofplace = 201, /**< Out-of-place computations are allowed; this
                                   requires extra device memory for workspace. */
    rocblas_inplace = 202, /**< If not enough memory is available, this forces in-place computations.  */
} rocblas_workmode;

/*! \brief Used to specify how the eigenvectors are to be computed
 ********************************************************************************/
typedef enum rocblas_evect_
{
    rocblas_evect_original = 211, /**< Compute eigenvectors for the original symmetric/Hermitian
                                       matrix. */
    rocblas_evect_tridiagonal = 212, /**< Compute eigenvectors for the symmetric tridiagonal
                                           matrix. */
    rocblas_evect_none = 213, /**< No eigenvectors are computed. */
} rocblas_evect;

/*! \brief Used to specify the form of the generalized eigenproblem
 ********************************************************************************/
typedef enum rocblas_eform_
{
    rocblas_eform_ax = 221, /**< The problem is \f$Ax = \lambda Bx\f$. */
    rocblas_eform_abx = 222, /**< The problem is \f$ABx = \lambda x\f$. */
    rocblas_eform_bax = 223, /**< The problem is \f$BAx = \lambda x\f$. */
} rocblas_eform;

/*! \brief Used to specify the type of range in which eigenvalues will be found
 *in partial eigenvalue decompositions
 ********************************************************************************/
typedef enum rocblas_erange_
{
    rocblas_erange_all = 231, /**< All eigenvalues will be found. */
    rocblas_erange_value = 232, /**< All eigenvalues in the half-open interval
                                     \f$(vl, vu]\f$ will be found. */
    rocblas_erange_index = 233, /**< The \f$il\f$-th through \f$iu\f$-th eigenvalues will be found.*/
} rocblas_erange;

/*! \brief Used to specify whether the eigenvalues are grouped and ordered by blocks
 ********************************************************************************/
typedef enum rocblas_eorder_
{
    rocblas_eorder_blocks = 241, /**< The computed eigenvalues will be grouped by split-off
                                      blocks and arranged in increasing order within each block. */
    rocblas_eorder_entire = 242, /**< All computed eigenvalues of the entire matrix will be
                                      ordered from smallest to largest. */
} rocblas_eorder;

/*! \brief Used in the Jacobi methods to specify whether the eigenvalues are sorted
 *in increasing order
 ********************************************************************************/
typedef enum rocblas_esort_
{
    rocblas_esort_none = 251, /**< The computed eigenvalues will not be sorted. */
    rocblas_esort_ascending = 252, /**< The computed eigenvalues will be sorted in ascending order. */
} rocblas_esort;

/*! \brief Used to specify the type of range in which singular values will be found
 *in partial singular value decompositions
 ********************************************************************************/
typedef enum rocblas_srange_
{
    rocblas_srange_all = 261, /**< All singular values will be found. */
    rocblas_srange_value = 262, /**< All singular values in the half-open interval
                                     \f$(vl, vu]\f$ will be found. */
    rocblas_srange_index = 263, /**< The \f$il\f$-th through \f$iu\f$-th singular values will be found.*/
} rocblas_srange;

/*! \brief Forward-declaration of opaque struct containing data used for the re-factorization interfaces.
 ********************************************************************************/
struct rocsolver_rfinfo_;

/*! \brief A handle to a structure containing matrix descriptors and metadata required to interact
 *with rocSPARSE when using the rocSOLVER re-factorization functionality. It needs to be initialized
 *with \ref rocsolver_create_rfinfo and destroyed with \ref rocsolver_destroy_rfinfo.
 ********************************************************************************/
typedef struct rocsolver_rfinfo_* rocsolver_rfinfo;

/*! \brief Used to specify the mode of the rfinfo struct required by the re-factorization functionality.
 ********************************************************************************/
typedef enum rocsolver_rfinfo_mode_
{
    rocsolver_rfinfo_mode_lu
    = 271, /**< To work with LU factorization (for general sparse matrices). This is the default mode. */
    rocsolver_rfinfo_mode_cholesky
    = 272, /**< To work with Cholesky factorization (for symmetric positive definite sparse matrices). */
} rocsolver_rfinfo_mode;

/*! \brief Used to specify the planes on which a sequence of Givens rotations is applied.
 ********************************************************************************/
typedef enum rocblas_pivot_
{
    rocblas_pivot_variable = 281, /**< The i-th rotation is applied on plane (i,i+1). */
    rocblas_pivot_top = 282, /**< The i-th rotation is applied on plane (1,i+1). */
    rocblas_pivot_bottom = 283, /**< The i-th rotation is applied on plane (i,m) or (i,n). */
} rocblas_pivot;

/*! \brief Used by specific functions to specify the algorithm mode.
 ********************************************************************************/
typedef enum rocsolver_alg_mode_
{
    rocsolver_alg_mode_gpu
    = 291, /**< Computations are all performed on the GPU. This is the default mode. */
    rocsolver_alg_mode_hybrid = 292, /**< Computations are performed on the CPU and GPU. */
    rocsolver_alg_mode_mixed = 293, /**< Nested functions use a mixture of hybrid and GPU-only modes. */
} rocsolver_alg_mode;

/*! \brief Used to specify a function with multiple supported algorithm modes.
 ********************************************************************************/
typedef enum rocsolver_function_
{
    rocsolver_function_bdsqr = 401,
    rocsolver_function_gesvd = 402, /**< Affected by bdsqr. */
    rocsolver_function_sterf = 403,
    rocsolver_function_steqr = 404,
    rocsolver_function_syev_heev = 405, /**< Affected by sterf and steqr. */
} rocsolver_function;

#endif /* ROCSOLVER_EXTRA_TYPES_H */
