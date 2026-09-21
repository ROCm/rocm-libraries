/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#ifndef HIPSPARSE_SPGEAM_H
#define HIPSPARSE_SPGEAM_H

// The generic SpGEAM API is gated behind the HIPSPARSE_WITH_SPGEAM build-time feature flag.
// Use the include-root-relative path so it resolves both in the hipSPARSE build and for
// downstream consumers that only add the include root (e.g. -I <prefix>/include) to their path.
#include "hipsparse/hipsparse-config.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Create a sparse matrix sparse matrix addition descriptor.
*
*  \details
*  \p hipsparseSpGEAM_createDescr creates a sparse matrix sparse matrix addition descriptor. It should be
*  destroyed at the end using \ref hipsparseSpGEAM_destroyDescr().
*/
// clang-format 19 (math-ci) rewrites "#if(" to "#if (", which clang-format
// 18 (the repo pre-commit hook) reverts; disable so both formatters agree.
// clang-format off
#if(defined(HIPSPARSE_WITH_SPGEAM) && (!defined(CUDART_VERSION) || CUDART_VERSION >= 13030))
// clang-format on
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEAM_createDescr(hipsparseSpGEAMDescr_t* descr);
#endif

/*! \ingroup generic_module
*  \brief Destroy a sparse matrix sparse matrix addition descriptor.
*
*  \details
*  \p hipsparseSpGEAM_destroyDescr destroys a sparse matrix sparse matrix addition descriptor and releases all
*  resources used by the descriptor.
*/
// clang-format off
#if(defined(HIPSPARSE_WITH_SPGEAM) && (!defined(CUDART_VERSION) || CUDART_VERSION >= 13030))
// clang-format on
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEAM_destroyDescr(hipsparseSpGEAMDescr_t descr);
#endif

/*! \ingroup generic_module
*  \brief Buffer size step of the sparse matrix sparse matrix addition:
*  \f[
*    C := \alpha \cdot op(A) + \beta \cdot op(B),
*  \f]
*  where \f$A\f$, \f$B\f$, and \f$C\f$ are sparse matrices.
*
*  \details
*  \p hipsparseSpGEAM_bufferSize computes the size in bytes of the user-allocated temporary storage buffer
*  that is used by \ref hipsparseSpGEAM_nnz and \ref hipsparseSpGEAM. The same buffer can be passed to both
*  routines.
*
*  @param[in]
*  handle           handle to the hipSPARSE library context queue.
*  @param[in]
*  opA              sparse matrix \f$A\f$ operation type.
*  @param[in]
*  opB              sparse matrix \f$B\f$ operation type.
*  @param[in]
*  alpha            scalar \f$\alpha\f$.
*  @param[in]
*  matA             sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  beta             scalar \f$\beta\f$.
*  @param[in]
*  matB             sparse matrix \f$B\f$ descriptor.
*  @param[in]
*  matC             sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  computeType      floating point precision for the SpGEAM computation.
*  @param[in]
*  alg              SpGEAM algorithm for the SpGEAM computation.
*  @param[in]
*  spgeamDescr      SpGEAM descriptor.
*  @param[out]
*  bufferSize       number of bytes of the temporary storage buffer.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_NOT_INITIALIZED \p handle is not initialized.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p beta, \p matA, \p matB, \p matC,
*          \p spgeamDescr, or \p bufferSize is nullptr, or \p opA or \p opB is invalid.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED \p opA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
*          \p opB != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE.
*/
// clang-format off
#if(defined(HIPSPARSE_WITH_SPGEAM) && (!defined(CUDART_VERSION) || CUDART_VERSION >= 13030))
// clang-format on
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEAM_bufferSize(hipsparseHandle_t          handle,
                                             hipsparseOperation_t       opA,
                                             hipsparseOperation_t       opB,
                                             const void*                alpha,
                                             hipsparseConstSpMatDescr_t matA,
                                             const void*                beta,
                                             hipsparseConstSpMatDescr_t matB,
                                             hipsparseSpMatDescr_t      matC,
                                             hipDataType                computeType,
                                             hipsparseSpGEAMAlg_t       alg,
                                             hipsparseSpGEAMDescr_t     spgeamDescr,
                                             size_t*                    bufferSize);
#endif

/*! \ingroup generic_module
*  \brief Computes the number of non-zero elements of \f$C\f$ for the sparse matrix-sparse matrix addition:
*  \f[
*    C := \alpha \cdot op(A) + \beta \cdot op(B),
*  \f]
*  where \f$A\f$, \f$B\f$, and \f$C\f$ are sparse matrices.
*
*  \details
*  \p hipsparseSpGEAM_nnz computes the number of non-zero elements and the row offsets array of the output
*  sparse matrix \f$C\f$. The \f$C\f$ matrix must be created with its row offsets array already allocated.
*  After the call, the number of non-zero elements of \f$C\f$ can be retrieved using \ref hipsparseSpMatGetSize.
*  The user can then allocate the column indices and values arrays of \f$C\f$, set them using
*  \ref hipsparseCsrSetPointers, and finally call \ref hipsparseSpGEAM. If the sparsity pattern of \f$C\f$ is
*  already known, this step can be skipped.
*
*  @param[in]
*  handle           handle to the hipSPARSE library context queue.
*  @param[in]
*  opA              sparse matrix \f$A\f$ operation type.
*  @param[in]
*  opB              sparse matrix \f$B\f$ operation type.
*  @param[in]
*  alpha            scalar \f$\alpha\f$.
*  @param[in]
*  matA             sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  beta             scalar \f$\beta\f$.
*  @param[in]
*  matB             sparse matrix \f$B\f$ descriptor.
*  @param[out]
*  matC             sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  computeType      floating point precision for the SpGEAM computation.
*  @param[in]
*  alg              SpGEAM algorithm for the SpGEAM computation.
*  @param[in]
*  spgeamDescr      SpGEAM descriptor.
*  @param[in]
*  externalBuffer   temporary storage buffer allocated by the user.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p beta, \p matA, \p matB, \p matC, or
*          \p spgeamDescr is nullptr.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED \p opA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
*          \p opB != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE.
*/
// clang-format off
#if(defined(HIPSPARSE_WITH_SPGEAM) && (!defined(CUDART_VERSION) || CUDART_VERSION >= 13030))
// clang-format on
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEAM_nnz(hipsparseHandle_t          handle,
                                      hipsparseOperation_t       opA,
                                      hipsparseOperation_t       opB,
                                      const void*                alpha,
                                      hipsparseConstSpMatDescr_t matA,
                                      const void*                beta,
                                      hipsparseConstSpMatDescr_t matB,
                                      hipsparseSpMatDescr_t      matC,
                                      hipDataType                computeType,
                                      hipsparseSpGEAMAlg_t       alg,
                                      hipsparseSpGEAMDescr_t     spgeamDescr,
                                      void*                      externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Compute step of the sparse matrix sparse matrix addition:
*  \f[
*    C := \alpha \cdot op(A) + \beta \cdot op(B),
*  \f]
*  where \f$A\f$, \f$B\f$, and \f$C\f$ are sparse matrices.
*
*  \details
*  \p hipsparseSpGEAM computes the column indices and values arrays of the output sparse matrix \f$C\f$. It
*  must be called after \ref hipsparseSpGEAM_nnz (unless the sparsity pattern of \f$C\f$ is already known) and
*  after the column indices and values arrays of \f$C\f$ have been allocated and set using
*  \ref hipsparseCsrSetPointers.
*
*  \p hipsparseSpGEAM supports the following uniform precision data type for the sparse matrices \f$A\f$,
*  \f$B\f$, \f$C\f$ and the compute type for \f$\alpha\f$ and \f$\beta\f$.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spgeam_uniform">Uniform Precisions</caption>
*  <tr><th>A / B / C / compute_type
*  <tr><td>HIP_R_32F
*  <tr><td>HIP_R_64F
*  <tr><td>HIP_C_32F
*  <tr><td>HIP_C_64F
*  </table>
*
*  \p hipsparseSpGEAM supports \ref HIPSPARSE_INDEX_32I and \ref HIPSPARSE_INDEX_64I index precisions for
*  storing the row offsets and column indices arrays of the sparse matrices.
*
*  \note Currently, only \ref HIPSPARSE_FORMAT_CSR format is supported.
*  \note Currently, only \p opA and \p opB equal to \ref HIPSPARSE_OPERATION_NON_TRANSPOSE are supported.
*  \note The column indices of \f$A\f$ and \f$B\f$ must be sorted. The column indices of \f$C\f$ are returned sorted.
*
*  @param[in]
*  handle           handle to the hipSPARSE library context queue.
*  @param[in]
*  opA              sparse matrix \f$A\f$ operation type.
*  @param[in]
*  opB              sparse matrix \f$B\f$ operation type.
*  @param[in]
*  alpha            scalar \f$\alpha\f$.
*  @param[in]
*  matA             sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  beta             scalar \f$\beta\f$.
*  @param[in]
*  matB             sparse matrix \f$B\f$ descriptor.
*  @param[out]
*  matC             sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  computeType      floating point precision for the SpGEAM computation.
*  @param[in]
*  alg              SpGEAM algorithm for the SpGEAM computation.
*  @param[in]
*  spgeamDescr      SpGEAM descriptor.
*  @param[in]
*  externalBuffer   temporary storage buffer allocated by the user.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p beta, \p matA, \p matB, \p matC, or
*          \p spgeamDescr is nullptr.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED \p opA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
*          \p opB != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE.
*/
// clang-format off
#if(defined(HIPSPARSE_WITH_SPGEAM) && (!defined(CUDART_VERSION) || CUDART_VERSION >= 13030))
// clang-format on
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEAM(hipsparseHandle_t          handle,
                                  hipsparseOperation_t       opA,
                                  hipsparseOperation_t       opB,
                                  const void*                alpha,
                                  hipsparseConstSpMatDescr_t matA,
                                  const void*                beta,
                                  hipsparseConstSpMatDescr_t matB,
                                  hipsparseSpMatDescr_t      matC,
                                  hipDataType                computeType,
                                  hipsparseSpGEAMAlg_t       alg,
                                  hipsparseSpGEAMDescr_t     spgeamDescr,
                                  void*                      externalBuffer);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_SPGEAM_H */
