/*! \file */
/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCSPARSE_SPGEAM_H
#define ROCSPARSE_SPGEAM_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \details
*  \p rocsparse_spgeam_buffer_size returns the size of the required buffer to execute the given stage of the SpGEAM operation.
*  This routine is used in conjunction with \ref rocsparse_spgeam(). See \ref rocsparse_spgeam for full description and example.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  descr        SpGEAM descriptor
*  @param[in]
*  mat_A        sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  mat_B        sparse matrix \f$B\f$ descriptor.
*  @param[in]
*  mat_C        sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  stage        SpGEAM stage for the SpGEAM computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p mat_A, \p mat_B, \p descr or \p buffer_size pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spgeam_buffer_size(rocsparse_handle            handle,
                                              rocsparse_spgeam_descr      descr,
                                              rocsparse_const_spmat_descr mat_A,
                                              rocsparse_const_spmat_descr mat_B,
                                              rocsparse_const_spmat_descr mat_C,
                                              rocsparse_spgeam_stage      stage,
                                              size_t*                     buffer_size);

/*! \ingroup generic_module
*  \brief Sparse matrix sparse matrix addition
*
*  \details
*  \p rocsparse_spgeam multiplies the scalar \f$\alpha\f$ with the sparse \f$m \times n\f$ CSR matrix \f$op(A)\f$
*  and adds it to \f$\beta\f$ multiplied by the sparse \f$m \times n\f$ matrix \f$op(B)\f$. The final result is
*  stored in the sparse \f$m \times n\f$ matrix \f$C\f$,
*  such that
*  \f[
*    C := \alpha op(A) + \beta op(B),
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans_A == rocsparse_operation_none}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == rocsparse_operation_none}
*    \end{array}
*    \right.
*  \f]
*
*  \p rocsparse_spgeam requires multiple steps to complete. First, the user must create a \ref rocsparse_spgeam_descr by
*  calling \ref rocsparse_create_spgeam_descr. The user sets the SpGEAM algorithm (currently only
*  \ref rocsparse_spgeam_alg_default supported) as well as the compute type and the transpose operation type for the sparse
*  matrices \f$op(A)\f$ and \f$op(B)\f$ using \ref rocsparse_spgeam_set_input. Next, the user must calculate the total nonzeros
*  that will exist in the sparse matrix \f$C\f$. To do so, the user calls \ref rocsparse_spgeam_buffer_size with the stage set
*  to \ref rocsparse_spgeam_stage_analysis. This will fill the \p buffer_size parameter allowing the user to then allocate this buffer.
*  Once the buffer has been allocated, the user calls \p rocsparse_spgeam with the same stage \ref rocsparse_spgeam_stage_analysis.
*  The total nonzeros and the row offset array for \f$C\f$ has now been calculated and is stored internally in the
*  \ref rocsparse_spgeam_descr. The user now needs to retrieve the nonzero count using \ref rocsparse_spgeam_get_output and then
*  allocate the \f$C\f$ matrix. To complete the computation, the user repeats the process (this time passing the stage
*  \ref rocsparse_spgeam_stage_compute) by calling \ref rocsparse_spgeam_buffer_size to determine the required buffer size, then
*  allocating the buffer, and finally calling \p rocsparse_spgeam. The user allocated buffers can be freed after each call to
*  \p rocsparse_spgeam. Once the computation is complete and the SpGEAM descriptor is no longer needed, the user must call
*  \ref rocsparse_destroy_spgeam_descr. See full code example below.
*
*  \p rocsparse_spgeam supports multiple combinations of index types, data types, and compute types. The tables below indicate
*  the currently supported different index and data types that can be used for the sparse matrices \f$op(A)\f$, \f$op(B)\f$, and
*  \f$C\f$, and the compute type for \f$\alpha\f$ and \f$\beta\f$. The advantage of using different index and data types is to save on
*  memory bandwidth and storage when a user application allows while performing the actual computation in a higher precision.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spgeam_uniform">Uniform Precisions</caption>
*  <tr><th>A / B / C / compute_type
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \par Uniform Index Types:
*  <table>
*  <caption id="spgeam_csr_uniform_indextype">CSR Uniform Index Types</caption>
*  <tr><th>CSR Row offset                 <th>CSR Column indices
*  <tr><td>rocsparse_datatype_f32_r</td>  <td>rocsparse_datatype_f32_r</td>
*  <tr><td>rocsparse_datatype_f64_r</td>  <td>rocsparse_datatype_f64_r</td>
*  </table>
*
*  \par Mixed Index Types:
*  <table>
*  <caption id="spgeam_csr_mixed_indextype">CSR Mixed Index Types</caption>
*  <tr><th>CSR Row offset                 <th>CSR Column indices
*  <tr><td>rocsparse_datatype_f64_r</td>  <td>rocsparse_datatype_f32_r</td>
*  </table>
*
*  In general, when adding two sparse matrices together, it is entirely possible that the resulting matrix will require a
*  a larger index representation to store correctly. For example, when adding \f$A + B\f$ using
*  \ref rocsparse_indextype_i32 index types for the row pointer and column indices arrays, it may be the case that the row pointer
*  of the resulting \f$C\f$ matrix would require index type \ref rocsparse_indextype_i64. This is currently not supported. In this
*  scenario, the user would need to store the \f$A\f$, \f$B\f$, and \f$C\f$ matrices using the higher index precision.
*
*  Additionally, all three matrices \f$A\f$, \f$B\f$, and \f$C\f$ must use the same index types. For example, if \f$A\f$ uses the
*  index type \ref rocsparse_datatype_f32_r for the row offset array and the index type \ref rocsparse_datatype_f32_r for the column
*  indices array, then both \f$B\f$ and \f$C\f$ must also use these same index types for their respective row offset and column index
*  arrays. In the scenario where \f$C\f$ requires a larger index type for the row offset array, the user would need to store all three
*  matrices using the larger index type \ref rocsparse_datatype_f64_r for the row offsets array.
*
*  \note If \f$\alpha == 0\f$, then \f$C = \beta \cdot op(B)\f$ will be computed.
*  \note If \f$\beta == 0\f$, then \f$C = \alpha \cdot op(A) \f$ will be
*  computed.
*  \note Currently only CSR format is supported.
*  \note \f$\alpha == beta == 0\f$ is invalid.
*  \note Currently, only \p trans_A == \ref rocsparse_operation_none is supported.
*  \note Currently, only \p trans_B == \ref rocsparse_operation_none is supported.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  descr        SpGEAM descriptor
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  mat_A        sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[in]
*  mat_B        sparse matrix \f$B\f$ descriptor.
*  @param[out]
*  mat_C        sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  stage        SpGEAM stage for the SpGEAM computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer. \p buffer_size is
*               determined by calling \ref rocsparse_spgeam_buffer_size.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p alpha and \p beta are invalid,
*          \p mat_A, \p mat_B, \p mat_C, \p descr or \p buffer_size pointer is invalid.
*
*  \par Example
*  \code{.c}
*   // A - m x n
*   // B - m x n
*   // C - m x n
*   int m = 4;
*   int n = 6;
*
*   // 1 2 0 0 3 7
*   // 0 0 1 4 6 8
*   // 0 2 0 4 0 0
*   // 9 8 0 0 2 0
*   std::vector<int> hcsr_row_ptr_A = {0, 4, 8, 10, 13}; // host A m x n matrix
*   std::vector<int> hcsr_col_ind_A = {0, 1, 4, 5, 2, 3, 4, 5, 1, 3, 0, 1, 4}; // host A m x n matrix
*   std::vector<float> hcsr_val_A = {1, 2, 3, 7, 1, 4, 6, 8, 2, 4, 9, 8, 2};   // host A m x n matrix
*
*   // 0 2 1 0 0 5
*   // 0 1 1 3 0 2
*   // 0 0 0 0 0 0
*   // 1 2 3 4 5 6
*   std::vector<int> hcsr_row_ptr_B = {0, 3, 7, 7, 13}; // host B m x n matrix
*   std::vector<int> hcsr_col_ind_B = {1, 2, 5, 1, 2, 3, 5, 0, 1, 2, 3, 4, 5}; // host B m x n matrix
*   std::vector<float> hcsr_val_B = {2, 1, 5, 1, 1, 3, 2, 1, 2, 3, 4, 5, 6};   // host B m x n matrix
*
*   int nnz_A = hcsr_val_A.size();
*   int nnz_B = hcsr_val_B.size();
*
*   float alpha            = 1.0f;
*   float beta             = 1.0f;
*
*   int* dcsr_row_ptr_A = nullptr;
*   int* dcsr_col_ind_A = nullptr;
*   float* dcsr_val_A = nullptr;
*
*   int* dcsr_row_ptr_B = nullptr;
*   int* dcsr_col_ind_B = nullptr;
*   float* dcsr_val_B = nullptr;
*
*   hipMalloc((void**)&dcsr_row_ptr_A, (m + 1) * sizeof(int));
*   hipMalloc((void**)&dcsr_col_ind_A, nnz_A * sizeof(int));
*   hipMalloc((void**)&dcsr_val_A, nnz_A * sizeof(float));
*
*   hipMalloc((void**)&dcsr_row_ptr_B, (m + 1) * sizeof(int));
*   hipMalloc((void**)&dcsr_col_ind_B, nnz_B * sizeof(int));
*   hipMalloc((void**)&dcsr_val_B, nnz_B * sizeof(float));
*
*   hipMemcpy(dcsr_row_ptr_A, hcsr_row_ptr_A.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice);
*   hipMemcpy(dcsr_col_ind_A, hcsr_col_ind_A.data(), nnz_A * sizeof(int), hipMemcpyHostToDevice);
*   hipMemcpy(dcsr_val_A, hcsr_val_A.data(), nnz_A * sizeof(float), hipMemcpyHostToDevice);
*
*   hipMemcpy(dcsr_row_ptr_B, hcsr_row_ptr_B.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice);
*   hipMemcpy(dcsr_col_ind_B, hcsr_col_ind_B.data(), nnz_B * sizeof(int), hipMemcpyHostToDevice);
*   hipMemcpy(dcsr_val_B, hcsr_val_B.data(), nnz_B * sizeof(float), hipMemcpyHostToDevice);
*
*   rocsparse_handle     handle;
*   rocsparse_spmat_descr matA, matB, matC;
*
*   rocsparse_index_base index_base = rocsparse_index_base_zero;
*   rocsparse_indextype itype = rocsparse_indextype_i32;
*   rocsparse_indextype jtype = rocsparse_indextype_i32;
*   rocsparse_datatype  ttype = rocsparse_datatype_f32_r;
*
*   rocsparse_create_handle(&handle);
*
*   hipStream_t stream;
*   rocsparse_get_stream(handle, &stream);
*
*   // Create sparse matrix A in CSR format
*   rocsparse_create_csr_descr(&matA, m, n, nnz_A,
*                       dcsr_row_ptr_A, dcsr_col_ind_A, dcsr_val_A,
*                       itype, jtype,
*                       index_base, ttype);
*
*   // Create sparse matrix B in CSR format
*   rocsparse_create_csr_descr(&matB, m, n, nnz_B,
*                       dcsr_row_ptr_B, dcsr_col_ind_B, dcsr_val_B,
*                       itype, jtype,
*                       index_base, ttype);
*
*   // Create SpGEAM descriptor.
*   rocsparse_spgeam_descr descr;
*   rocsparse_create_spgeam_descr(&descr);
*
*   // Set the algorithm on the descriptor
*   const rocsparse_spgeam_alg alg = rocsparse_spgeam_alg_default;
*   rocsparse_spgeam_set_input(handle, descr, rocsparse_spgeam_input_alg, &alg, sizeof(alg));
*
*   // Set the transpose operation for sparses matrix A and B on the descriptor
*   const rocsparse_operation trans_A = rocsparse_operation_none;
*   const rocsparse_operation trans_B = rocsparse_operation_none;
*   rocsparse_spgeam_set_input(handle, descr, rocsparse_spgeam_input_operation_A, &trans_A, sizeof(trans_A));
*   rocsparse_spgeam_set_input(handle, descr, rocsparse_spgeam_input_operation_B, &trans_B, sizeof(trans_B));
*
*   // Set the compute type on the descriptor
*   const rocsparse_datatype datatype = rocsparse_datatype_f32_r;
*   rocsparse_spgeam_set_input(handle, descr, rocsparse_spgeam_input_compute_datatype, &datatype, sizeof(datatype));
*
*   // Calculate NNZ phase
*   size_t buffer_size_in_bytes;
*   void * buffer;
*   rocsparse_spgeam_buffer_size(handle, descr, matA, matB, nullptr, rocsparse_spgeam_stage_analysis, &buffer_size_in_bytes);
*
*   hipMalloc(&buffer, buffer_size_in_bytes);
*   rocsparse_spgeam(handle, descr, &alpha, matA, &beta, matB, nullptr, rocsparse_spgeam_stage_analysis, buffer_size_in_bytes, buffer);
*   hipFree(buffer);
*
*   // Ensure analysis stage is complete before grabbing C non-zero count
*   hipStreamSynchronize(stream);
*
*   int64_t nnz_C;
*   rocsparse_spgeam_get_output(handle, descr, rocsparse_spgeam_output_nnz, &nnz_C, sizeof(int64_t));
*
*   // Compute column indices and values of C
*   int* dcsr_row_ptr_C = nullptr;
*   int* dcsr_col_ind_C = nullptr;
*   float* dcsr_val_C = nullptr;
*   hipMalloc((void**)&dcsr_row_ptr_C, (m + 1) * sizeof(int));
*   hipMalloc((void**)&dcsr_col_ind_C, sizeof(int32_t) * nnz_C);
*   hipMalloc((void**)&dcsr_val_C, sizeof(float) * nnz_C);
*
*   // Create sparse matrix C in CSR format
*   rocsparse_create_csr_descr(&matC, m, n, nnz_C,
*                       dcsr_row_ptr_C, dcsr_col_ind_C, dcsr_val_C,
*                       itype, jtype,
*                       index_base, ttype);
*
*   // Compute phase
*   rocsparse_spgeam_buffer_size(handle, descr, matA, matB, matC, rocsparse_spgeam_stage_compute, &buffer_size_in_bytes);
*
*   hipMalloc(&buffer, buffer_size_in_bytes);
*   rocsparse_spgeam(handle, descr, &alpha, matA, &beta, matB, matC, rocsparse_spgeam_stage_compute, buffer_size_in_bytes, buffer);
*   hipFree(buffer);
*
*   // Copy C matrix result back to host
*   std::vector<int> hcsr_row_ptr_C(m + 1);
*   std::vector<int> hcsr_col_ind_C(nnz_C);
*   std::vector<float>  hcsr_val_C(nnz_C);
*
*   hipMemcpy(hcsr_row_ptr_C.data(), dcsr_row_ptr_C, sizeof(int) * (m + 1), hipMemcpyDeviceToHost);
*   hipMemcpy(hcsr_col_ind_C.data(), dcsr_col_ind_C, sizeof(int) * nnz_C, hipMemcpyDeviceToHost);
*   hipMemcpy(hcsr_val_C.data(), dcsr_val_C, sizeof(float) * nnz_C, hipMemcpyDeviceToHost);
*
*   // Destroy matrix descriptors
*   rocsparse_destroy_spmat_descr(matA);
*   rocsparse_destroy_spmat_descr(matB);
*   rocsparse_destroy_spmat_descr(matC);
*   rocsparse_destroy_handle(handle);
*
*   // Free device arrays
*   hipFree(dcsr_row_ptr_A);
*   hipFree(dcsr_col_ind_A);
*   hipFree(dcsr_val_A);
*
*   hipFree(dcsr_row_ptr_B);
*   hipFree(dcsr_col_ind_B);
*   hipFree(dcsr_val_B);
*
*   hipFree(dcsr_row_ptr_C);
*   hipFree(dcsr_col_ind_C);
*   hipFree(dcsr_val_C);
*  \endcode
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spgeam(rocsparse_handle            handle,
                                  rocsparse_spgeam_descr      descr,
                                  const void*                 alpha,
                                  rocsparse_const_spmat_descr mat_A,
                                  const void*                 beta,
                                  rocsparse_const_spmat_descr mat_B,
                                  rocsparse_spmat_descr       mat_C,
                                  rocsparse_spgeam_stage      stage,
                                  size_t                      buffer_size,
                                  void*                       temp_buffer);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPGEAM_H */
