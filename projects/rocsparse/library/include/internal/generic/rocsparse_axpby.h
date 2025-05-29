/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_AXPBY_H
#define ROCSPARSE_AXPBY_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Scale a sparse vector and add it to a scaled dense vector.
*
*  \details
*  \p rocsparse_axpby multiplies the sparse vector \f$x\f$ with scalar \f$\alpha\f$ and
*  adds the result to the dense vector \f$y\f$ that is multiplied with scalar
*  \f$\beta\f$, such that
*
*  \f[
*      y := \alpha \cdot x + \beta \cdot y
*  \f]
*
*  \code{.c}
*      for(i = 0; i < size; ++i)
*      {
*          y[i] = beta * y[i]
*      }
*      for(i = 0; i < nnz; ++i)
*      {
*          y[x_ind[i]] += alpha * x_val[i]
*      }
*  \endcode
*
*  \p rocsparse_axpby supports the following uniform precision data types for the sparse and dense vectors x and
*  y and compute types for the scalars \f$\alpha\f$ and \f$\beta\f$.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="axpby_uniform">Uniform Precisions</caption>
*  <tr><th>X / Y / compute_type
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \par Mixed precisions:
*  <table>
*  <caption id="axpby_mixed">Mixed Precisions</caption>
*  <tr><th>X / Y                   <th>compute_type
*  <tr><td>rocsparse_datatype_f16_r <td>rocsparse_datatype_f32_r
*  </table>
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  x           sparse matrix descriptor.
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  y           dense matrix descriptor.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p alpha, \p x, \p beta or \p y pointer is
*          invalid.
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
*   // Scalar alpha
*   float alpha = 3.7f;
*
*   // Scalar beta
*   float beta = 1.2f;
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
*   // Call axpby to perform y = beta * y + alpha * x
*   rocsparse_axpby(handle,
*                   &alpha,
*                   vecX,
*                   &beta,
*                   vecY);
*
*   rocsparse_dnvec_get_values(vecY, (void**)&dy);
*
*   // Copy result back to host
*   hipMemcpy(hy.data(), dy, sizeof(float) * size, hipMemcpyDeviceToHost);
*
*   std::cout << "y" << std::endl;
*   for(size_t i = 0; i < hy.size(); ++i)
*   {
*       std::cout << hy[i] << " ";
*   }
*   std::cout << std::endl;
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
*  \endcode
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_axpby(rocsparse_handle            handle,
                                 const void*                 alpha,
                                 rocsparse_const_spvec_descr x,
                                 const void*                 beta,
                                 rocsparse_dnvec_descr       y);
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_AXPBY_H */
