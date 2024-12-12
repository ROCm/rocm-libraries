/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_SPVV_H
#define ROCSPARSE_SPVV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Sparse vector inner dot product
*
*  \details
*  \p rocsparse_spvv computes the inner dot product of the sparse vector \f$x\f$ with the
*  dense vector \f$y\f$, such that
*  \f[
*    \text{result} := op(x) \cdot y,
*  \f]
*  with
*  \f[
*    op(x) = \left\{
*    \begin{array}{ll}
*        x,   & \text{if trans == rocsparse_operation_none} \\
*        \bar{x}, & \text{if trans == rocsparse_operation_conjugate_transpose} \\
*    \end{array}
*    \right.
*  \f]
*
*  \code{.c}
*      result = 0;
*      for(i = 0; i < nnz; ++i)
*      {
*          result += x_val[i] * y[x_ind[i]];
*      }
*  \endcode
*
*  \p rocsparse_spvv supports the following uniform and mixed precision data types for the sparse and dense vectors x and 
*  y and compute types for the scalar \f$result\f$.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spvv_uniform">Uniform Precisions</caption>
*  <tr><th>X / Y / compute_type
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \par Mixed precisions:
*  <table>
*  <caption id="spvv_mixed">Mixed Precisions</caption>
*  <tr><th>X / Y                   <th>compute_type / result
*  <tr><td>rocsparse_datatype_i8_r <td>rocsparse_datatype_i32_r
*  <tr><td>rocsparse_datatype_i8_r <td>rocsparse_datatype_f32_r
*  </table>
*
*  \note
*  This function writes the required allocation size (in bytes) to \p buffer_size and
*  returns without performing the SpVV operation, when a nullptr is passed for
*  \p temp_buffer.
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  trans        sparse vector operation type.
*  @param[in]
*  x            sparse vector descriptor.
*  @param[in]
*  y            dense vector descriptor.
*  @param[out]
*  result       pointer to the result, can be host or device memory
*  @param[in]
*  compute_type floating point precision for the SpVV computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer. buffer_size is set when
*               \p temp_buffer is nullptr.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user. When a nullptr is passed,
*               the required allocation size (in bytes) is written to \p buffer_size and
*               function returns without performing the SpVV operation.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p x, \p y, \p result or \p buffer_size
*               pointer is invalid.
*  \retval      rocsparse_status_not_implemented \p compute_type is currently not
*               supported.
*
*  \par Example
*  \code{.c}
*   // Number of non-zeros of the sparse vector
*   int nnz = 3;
*
*   // Size of sparse and dense vector
*   int size = 9;
*
*   // Sparse index vector
*   std::vector<int> hx_ind = {0, 3, 5};
*
*   // Sparse value vector
*   std::vector<float> hx_val = {1.0f, 2.0f, 3.0f};
*
*   // Dense vector
*   std::vector<float> hy = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
*
*   // Offload data to device
*   int* dx_ind;
*   float* dx_val;
*   float* dy;
*   hipMalloc((void**)&dx_ind, sizeof(int) * nnz);
*   hipMalloc((void**)&dx_val, sizeof(float) * nnz);
*   hipMalloc((void**)&dy, sizeof(float) * size);
*
*   hipMemcpy(dx_ind, hx_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
*   hipMemcpy(dx_val, hx_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice);
*   hipMemcpy(dy, hy.data(), sizeof(float) * size, hipMemcpyHostToDevice);
*
*   rocsparse_handle     handle;
*   rocsparse_spvec_descr vecX;
*   rocsparse_dnvec_descr vecY;
*
*   rocsparse_indextype idx_type = rocsparse_indextype_i32;
*   rocsparse_datatype  data_type = rocsparse_datatype_f32_r;
*   rocsparse_datatype  compute_type = rocsparse_datatype_f32_r;
*   rocsparse_operation trans = rocsparse_operation_none;
*   rocsparse_index_base idx_base = rocsparse_index_base_zero;
*
*   rocsparse_create_handle(&handle);
*
*   // Create sparse vector X
*   rocsparse_create_spvec_descr(&vecX,
*                                size,
*                                nnz,
*                                dx_ind,
*                                dx_val,
*                                idx_type,
*                                idx_base,
*                                data_type);
*
*   // Create dense vector Y
*   rocsparse_create_dnvec_descr(&vecY,
*                                size,
*                                dy,
*                                data_type);
*
*   // Obtain buffer size
*   float hresult = 0.0f;
*   size_t buffer_size;
*   rocsparse_spvv(handle,
*                  trans,
*                  vecX,
*                  vecY,
*                  &hresult,
*                  compute_type,
*                  &buffer_size,
*                  nullptr);
*
*   void* temp_buffer;
*   hipMalloc(&temp_buffer, buffer_size);
*
*   // SpVV
*   rocsparse_spvv(handle,
*                  trans,
*                  vecX,
*                  vecY,
*                  &hresult,
*                  compute_type,
*                  &buffer_size,
*                  temp_buffer);
*
*   hipDeviceSynchronize();
*
*   // Clear rocSPARSE
*   rocsparse_destroy_spvec_descr(vecX);
*   rocsparse_destroy_dnvec_descr(vecY);
*   rocsparse_destroy_handle(handle);
*
*   // Clear device memory
*   hipFree(dx_ind);
*   hipFree(dx_val);
*   hipFree(dy);
*   hipFree(temp_buffer);
*  \endcode
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spvv(rocsparse_handle            handle,
                                rocsparse_operation         trans,
                                rocsparse_const_spvec_descr x,
                                rocsparse_const_dnvec_descr y,
                                void*                       result,
                                rocsparse_datatype          compute_type,
                                size_t*                     buffer_size,
                                void*                       temp_buffer);
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPVV_H */
