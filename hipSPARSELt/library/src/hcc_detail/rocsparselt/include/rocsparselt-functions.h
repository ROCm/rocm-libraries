/* ************************************************************************
* Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
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

/*! \file
*  \brief rocsparselt-functions.h provides Sparse Linear Algebra Subprograms
*  of Level 1, 2 and 3, using HIP optimized for AMD GPU hardware.
*/

#pragma once
#ifndef _ROCSPARSELT_FUNCTIONS_H_
#define _ROCSPARSELT_FUNCTIONS_H_

#include "rocsparselt-types.h"
#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Initialize rocSPARSELt on the current HIP device, to avoid costly startup time at the first call on that device.
    \details

    Calling `rocsparselt_initialize()`` allows upfront intialization including device specific kernel setup,
    otherwise this function is automatically called on the first function call that requires these initializations (mainly SPMM).

 ******************************************************************************/
void rocsparselt_initialize(void);

/*
* ===========================================================================
*    SPARSE Matrix Multiplication
* ===========================================================================
*/

/*! \ingroup spmm_module
 *  \brief Determines the required workspace size.
 *  \details
 *  \p rocsparselt_matmul_get_workspace determines the required workspace size
 *  associated to the selected algorithm.
 *
 *  @param[out]
 *  workspaceSize    Workspace size in bytes
 *
 *  @param[in]
 *  handle           rocsparselt library handle
 *  plan             Matrix multiplication plan.
 *
 *  \retval rocsparselt_status_success the operation completed successfully.
 *  \retval rocsparselt_status_invalid_handle \p handle or \p plan is invalid.
 *  \retval rocsparselt_status_invalid_pointer \p workspaceSize pointer is invalid.
 */
rocsparselt_status rocsparselt_matmul_get_workspace(const rocsparselt_handle*      handle,
                                                    const rocsparselt_matmul_plan* plan,
                                                    size_t*                        workspaceSize);

/*! \ingroup spmm_module
 *  \brief Sparse matrix dense matrix multiplication
 *
 *  \details
 *  \p rocsparselt_matmul computes the matrix multiplication of matrices A and B to
 *  produce the output matrix D, according to the following operation:
 *  \f[
 *    D := Activation(\alpha \cdot op(A) \cdot op(B) + \beta \cdot C)
 *  \f]
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only supports the case where D has the same shape of C.
 *
 *  @param[out]
 *  d_D         Pointer to the dense matrix D
 *
 *  @param[in]
 *  handle      rocsparselt library handle
 *  plan        Matrix multiplication plan
 *  alpha       scalar \f$\alpha\f$. (float)
 *  d_A         Pointer to the structured matrix A
 *  d_B         Pointer to the dense matrix B
 *  beta        scalar \f$\beta\f$. (float)
 *  d_C         Pointer to the dense matrix C
 *  workspace   Pointor to the worksapce
 *  streams     Pointer to HIP stream array for the computation
 *  numStreams  Number of HIP streams in \p streams

 *  \retval     rocsparselt_status_success the operation completed successfully.
 *  \retval     rocsparselt_status_invalid_handle \p handle or \p plan is invalid.
 *  \retval     rocsparselt_status_invalid_pointer \p alpha, \p A, \p B, \p beta, \p C or \p D
 *              pointer is invalid.
 *  \retval     rocsparselt_status_invalid_value workspace is invalide or streams and numStreams are invalid
 *  \retval     rocsparselt_status_not_implemented the problme is not supported
 */
rocsparselt_status rocsparselt_matmul(const rocsparselt_handle*      handle,
                                      const rocsparselt_matmul_plan* plan,
                                      const void*                    alpha,
                                      const void*                    d_A,
                                      const void*                    d_B,
                                      const void*                    beta,
                                      const void*                    d_C,
                                      void*                          d_D,
                                      void*                          workspace,
                                      hipStream_t*                   streams,
                                      int32_t                        numStreams);

/*! \ingroup spmm_module
 *  \brief Sparse matrix dense matrix multiplication
 *
 *  \details
 *  \p rocsparselt_matmul_search evaluates all available algorithms for the matrix multiplication
 *  and automatically updates the plan by selecting the fastest one.
 *  The functionality is intended to be used for auto-tuning purposes when the same operation
 *  is repeated multiple times over different inputs.
 *
 *  \note
 *  This function's behavior is the same of rocsparselt_matmul
 *
 *  \note
 *  d_C and d_D must be two different memory buffers, otherwise the output will be incorrect.
 *
 *  \note
 *  This function is NOT asynchronous with respect to streams[0] (blocking call)
 *
 *  \note
 *  The number of iterations for the evaluation can be set by using
 *  rocsparselt_matmul_alg_set_attribute() with rocsparselt_matmul_search_iterations.
 *
 *  \note
o*	The selected algorithm id can be retrieved by using
 *
 *  @param[out]
 *  d_D         Pointer to the dense matrix D
 *
 *  @param[in]
 *  handle      rocsparselt library handle
 *  plan        Matrix multiplication plan
 *  alpha       scalar \f$\alpha\f$. (float)
 *  d_A         Pointer to the structured matrix A
 *  d_B         Pointer to the dense matrix B
 *  beta        scalar \f$\beta\f$. (float)
 *  d_C         Pointer to the dense matrix C
 *  workspace   Pointor to the worksapce
 *  streams     Pointer to HIP stream array for the computation
 *  numStreams  Number of HIP streams in \p streams

 *  \retval     rocsparselt_status_success the operation completed successfully.
 *  \retval     rocsparselt_status_invalid_handle \p handle or \p plan is invalid.
 *  \retval     rocsparselt_status_invalid_pointer \p alpha, \p A, \p B, \p beta, \p C or \p D
 *              pointer is invalid.
 *  \retval     rocsparselt_status_invalid_value workspace is invalide or streams and numStreams are invalid
 *  \retval     rocsparselt_status_not_implemented the problme is not supported
 */
rocsparselt_status rocsparselt_matmul_search(const rocsparselt_handle* handle,
                                             rocsparselt_matmul_plan*  plan,
                                             const void*               alpha,
                                             const void*               d_A,
                                             const void*               d_B,
                                             const void*               beta,
                                             const void*               d_C,
                                             void*                     d_D,
                                             void*                     workspace,
                                             hipStream_t*              streams,
                                             int32_t                   numStreams);

/*! \ingroup spmm_module
 *  \brief Purnes a dense matrix.
 *
 *  \details
 *  \p rocsparselt_smfmac_prune prunes a dense matrix d_in according to the specified
 *  algorithm pruneAlg, \ref rocsparselt_prune_smfmac_tile or \ref rocsparselt_prune_smfmac_strip.
 *
 *  \note
 o	The function requires no extra storage.
 *
 *  \note
 *  This function supports asynchronous execution with respect to stream.
 *
 *  @param[out]
 *  d_out       pointer to the pruned matrix.
 *
 *  @param[in]
 *  handle      rocsparselt library handle
 *  matmulDescr matrix multiplication descriptor.
 *  d_in        pointer to the dense matrix.
 *  pruneAlg    pruning algorithm.
 *  stream      HIP stream for the computation.
 *
 *  \retval     rocsparselt_status_success the operation completed successfully.
 *  \retval     rocsparselt_status_invalid_handle \p handle or \p matmulDescr is invalid.
 *  \retval     rocsparselt_status_invalid_pointer \p d_in, \p d_out pointer is invalid.
 *  \retval     rocsparselt_status_not_implemented the problem or \p pruneAlg is not support
 */
rocsparselt_status rocsparselt_smfmac_prune(const rocsparselt_handle*       handle,
                                            const rocsparselt_matmul_descr* matmulDescr,
                                            const void*                     d_in,
                                            void*                           d_out,
                                            rocsparselt_prune_alg           pruneAlg,
                                            hipStream_t                     stream);

/*! \ingroup spmm_module
 *  \brief Purnes a dense matrix.
 *
 *  \details
 *  \p rocsparselt_smfmac_prune2 prunes a dense matrix d_in according to the specified
 *  algorithm pruneAlg, \ref rocsparselt_prune_smfmac_tile or \ref rocsparselt_prune_smfmac_strip.
 *
 *  \note
 o	The function requires no extra storage.
 *
 *  \note
 *  This function supports asynchronous execution with respect to stream.
 *
 *  @param[out]
 *  d_out       pointer to the pruned matrix.
 *
 *  @param[in]
 *  handle         rocsparselt library handle
 *  sparseMatDescr structured(sparse) matrix descriptor.
 *  isSparseA      specify if the structured (sparse) matrix is in the first position (matA or matB) (Currently, only support matA)
 *  op             operation that will be applied to the structured (sparse) matrix in the multiplication
 *  d_in           pointer to the dense matrix.
 *  pruneAlg       pruning algorithm.
 *  stream         HIP stream for the computation.
 *
 *  \retval     rocsparselt_status_success the operation completed successfully.
 *  \retval     rocsparselt_status_invalid_handle \p handle or \p sparseMatDescr is invalid.
 *  \retval     rocsparselt_status_invalid_pointer \p d_in, \p d_out pointer is invalid.
 *  \retval     rocsparselt_status_invalid_value \p op is invalid.
 *  \retval     rocsparselt_status_not_implemented the problem or \p pruneAlg is not support
 */
rocsparselt_status rocsparselt_smfmac_prune2(const rocsparselt_handle*    handle,
                                             const rocsparselt_mat_descr* sparseMatDescr,
                                             int                          isSparseA,
                                             rocsparselt_operation        op,
                                             const void*                  d_in,
                                             void*                        d_out,
                                             rocsparselt_prune_alg        pruneAlg,
                                             hipStream_t                  stream);

/*! \ingroup spmm_module
 *  \brief checks the correctness of the pruning structure for a given matrix.
 *
 *  \details
 *  \p rocsparselt_smfmac_prune_check checks the correctness of the pruning structure for a given matrix.
 *  Contents in the given matrix must be sparsity 2:4.
 *
 *
 *  @param[out]
 *  d_valid     validation results (0 correct, 1 wrong).
 *
 *  @param[in]
 *  handle      rocsparselt library handle
 *  matmulDescr matrix multiplication descriptor.
 *  d_in        pointer to the matrix to check.
 *  stream      HIP stream for the computation.
 *
 *  \retval     rocsparselt_status_success the operation completed successfully.
 *  \retval     rocsparselt_status_invalid_handle \p handle or \p matmulDescr is invalid.
 *  \retval     rocsparselt_status_invalid_pointer \p d_in, \p d_valid pointer is invalid.
 *  \retval     rocsparselt_status_not_implemented the problem is not support
 */
rocsparselt_status rocsparselt_smfmac_prune_check(const rocsparselt_handle*       handle,
                                                  const rocsparselt_matmul_descr* matmulDescr,
                                                  const void*                     d_in,
                                                  int*                            d_valid,
                                                  hipStream_t                     stream);
/*! \ingroup spmm_module
 *  \brief checks the correctness of the pruning structure for a given matrix.
 *
 *  \details
 *  \p rocsparselt_smfmac_prune_check2 checks the correctness of the pruning structure for a given matrix.
 *  Contents in the given matrix must be sparsity 2:4.
 *
 *
 *  @param[out]
 *  d_valid     validation results (0 correct, 1 wrong).
 *
 *  @param[in]
 *  handle         rocsparselt library handle
 *  sparseMatDescr structured(sparse) matrix descriptor.
 *  isSparseA      specify if the structured (sparse) matrix is in the first position (matA or matB) (Currently, only support matA)
 *  op             operation that will be applied to the structured (sparse) matrix in the multiplication
 *  d_in           pointer to the matrix to check.
 *  stream         HIP stream for the computation.
 *
 *  \retval     rocsparselt_status_success the operation completed successfully.
 *  \retval     rocsparselt_status_invalid_handle \p handle or \p sparseMatDescr is invalid.
 *  \retval     rocsparselt_status_invalid_pointer \p d_in, \p d_valid pointer is invalid.
 *  \retval     rocsparselt_status_invalid_value \p op is invalid.
 *  \retval     rocsparselt_status_not_implemented the problem is not support
 */
rocsparselt_status rocsparselt_smfmac_prune_check2(const rocsparselt_handle*    handle,
                                                   const rocsparselt_mat_descr* sparseMatDescr,
                                                   int                          isSparseA,
                                                   rocsparselt_operation        op,
                                                   const void*                  d_in,
                                                   int*                         d_valid,
                                                   hipStream_t                  stream);

/*! \ingroup spmm_module
 *  \brief provide the size of the compressed matrix.
 *
 *  \details
 *  \p rocsparselt_smfmac_compressed_size provides the size of the compressed matrix
 *  to be allocated before calling rocsparselt_smfmac_compress()
 *
 *
 *  @param[out]
 *  compressedSize      size in bytes of the compressed matrix.
 *  @param[out]
 *  compressBufferSize  size in bytes for the buffer needed for the matrix compression
 *
 *  @param[in]
 *  handle             rocsparselt library handle
 *  plan               matrix multiplication plan descriptor.
 *
 *  \retval     rocsparselt_status_success the operation completed successfully.
 *  \retval     rocsparselt_status_invalid_handle \p handle or \p plan is invalid.
 *  \retval     rocsparselt_status_invalid_pointer \p compressedSize pointer is invalid.
 *  \retval     rocsparselt_status_not_implemented the problem is not support
 */
rocsparselt_status rocsparselt_smfmac_compressed_size(const rocsparselt_handle*      handle,
                                                      const rocsparselt_matmul_plan* plan,
                                                      size_t*                        compressedSize,
                                                      size_t* compressBufferSize);

/*! \ingroup spmm_module
 *  \brief provide the size of the compressed matrix.
 *
 *  \details
 *  \p rocsparselt_smfmac_compressed_size provides the size of the compressed matrix
 *  to be allocated before calling rocsparselt_smfmac_compress()
 *
 *
 *  @param[out]
 *  compressedSize      size in bytes of the compressed matrix.
 *  @param[out]
 *  compressBufferSize  size in bytes for the buffer needed for the matrix compression
 *
 *  @param[in]
 *  handle             rocsparselt library handle
 *  sparseMatDescr structured(sparse) matrix descriptor.
 *
 *  \retval     rocsparselt_status_success the operation completed successfully.
 *  \retval     rocsparselt_status_invalid_handle \p handle or \p sparseMatDescr is invalid.
 *  \retval     rocsparselt_status_invalid_pointer \p compressedSize pointer is invalid.
 *  \retval     rocsparselt_status_not_implemented the problem is not support
 */
rocsparselt_status rocsparselt_smfmac_compressed_size2(const rocsparselt_handle*    handle,
                                                       const rocsparselt_mat_descr* sparseMatDescr,
                                                       size_t*                      compressedSize,
                                                       size_t* compresseBufferSize);

/*! \ingroup spmm_module
 *  \brief compresses a dense matrix to structured matrix.
 *
 *  \details
 *  \p rocsparselt_smfmac_compress compresses a dense matrix d_dense.
 *  The compressed matrix is intended to be used as the first/second operand A/B
 *  in the rocsparselt_matmul() function.
 *
 *  @param[out]
 *  d_compressed       compressed matrix and metadata
 *  @param[out]
 *  d_compressBuffer   temporary buffer for the compression
 *
 *  @param[in]
 *  handle         handle to the rocsparselt library context queue.
 *  plan           matrix multiplication plan descriptor.
 *  d_dense        pointer to the dense matrix.
 *  stream         HIP stream for the computation.
 *
 *  \retval     rocsparselt_status_success the operation completed successfully.
 *  \retval     rocsparselt_status_invalid_handle \p handle or \p plan is invalid.
 *  \retval     rocsparselt_status_invalid_pointer \p d_dense or \p d_compressed pointer is invalid.
 *  \retval     rocsparselt_status_not_implemented the problem is not support
 */
rocsparselt_status rocsparselt_smfmac_compress(const rocsparselt_handle*      handle,
                                               const rocsparselt_matmul_plan* plan,
                                               const void*                    d_dense,
                                               void*                          d_compressed,
                                               void*                          d_compressBuffer,
                                               hipStream_t                    stream);

/*! \ingroup spmm_module
 *  \brief compresses a dense matrix to structured matrix.
 *
 *  \details
 *  \p rocsparselt_smfmac_compress2 compresses a dense matrix d_dense.
 *  The compressed matrix is intended to be used as the first/second operand A/B
 *  in the rocsparselt_matmul() function.
 *
 *  @param[out]
 *  d_compressed       compressed matrix and metadata
 *  @param[out]
 *  d_compressBuffer   temporary buffer for the compression
 *
 *  @param[in]
 *  handle         handle to the rocsparselt library context queue.
 *  sparseMatDescr structured(sparse) matrix descriptor.
 *  isSparseA      specify if the structured (sparse) matrix is in the first position (matA or matB) (Currently, only support matA)
 *  op             operation that will be applied to the structured (sparse) matrix in the multiplication
 *  d_dense        pointer to the dense matrix.
 *  stream         HIP stream for the computation.
 *
 *  \retval     rocsparselt_status_success the operation completed successfully.
 *  \retval     rocsparselt_status_invalid_handle \p handle or \p sparseMatDescr is invalid.
 *  \retval     rocsparselt_status_invalid_pointer \p d_dense or \p d_compressed pointer is invalid.
 *  \retval     rocsparselt_status_invalid_value \p op is invalid.
 *  \retval     rocsparselt_status_not_implemented the problem is not support
 */
rocsparselt_status rocsparselt_smfmac_compress2(const rocsparselt_handle*    handle,
                                                const rocsparselt_mat_descr* sparseMatDescr,
                                                int                          isSparseA,
                                                rocsparselt_operation        op,
                                                const void*                  d_dense,
                                                void*                        d_compressed,
                                                void*                        d_compressBuffer,
                                                hipStream_t                  stream);

#ifdef __cplusplus
}
#endif

#endif /* _ROCSPARSELT_FUNCTIONS_H_ */
