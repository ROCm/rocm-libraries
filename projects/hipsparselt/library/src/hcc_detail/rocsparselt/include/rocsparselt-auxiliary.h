/* ************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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
 *  \brief rocsparselt-auxiliary.h provides auxilary functions in rocsparselt
 */

#pragma once
#ifndef _ROCSPARSELT_AUXILIARY_H_
#define _ROCSPARSELT_AUXILIARY_H_

#include <stdint.h>
#include <hip/hip_runtime_api.h>
#include "rocsparselt-types.h"

std::string rocsparselt_internal_get_arch_name();
std::string rocsparselt_internal_get_arch_name(const hipDeviceProp_t& deviceProperties);

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 *  \brief Create a rocsparselt handle
 *
 *  \details
 *  \p rocsparselt_init creates the rocSPARSELt library context. It must be
 *  initialized before any other rocSPARSELt API function is invoked and must be passed to
 *  all subsequent library function calls. The handle should be destroyed at the end
 *  using rocsparselt_destroy_handle().
 *
 *  @param[out]
 *  handle  rocsparselt library handle
 *
 *  \retval rocsparselt_status_success the initialization succeeded.
 *  \retval rocsparselt_status_invalid_pointer \p handle pointer is invalid.
 */
rocsparselt_status rocsparselt_init(rocsparselt_handle* handle);

/*! \ingroup aux_module
 *  \brief Destroy a rocsparselt handle
 *
 *  \details
 *  \p rocsparselt_destroy destroys the rocSPARSELt library context and releases all
 *  resources used by the rocSPARSELt library.
 *
 *  @param[in]
 *  handle  rocsparselt library handle
 *
 *  \retval rocsparselt_status_success the operation completed successfully.
 *  \retval rocsparselt_status_invalid_handle \p handle is invalid.
 */
rocsparselt_status rocsparselt_destroy(const rocsparselt_handle* handle);

/*! \ingroup aux_module
 *  \brief Create a descriptor for dense matrix
 *  \details
 *  \p rocsparselt_dense_descr_init creates a matrix descriptor It initializes
 *  \ref rocsparselt_matrix_type to \ref rocsparselt_matrix_type_dense and
 *  It should be destroyed at the end using rocsparselt_mat_descr_destroy().
 *
 *  @param[out]
 *  matDescr   the pointer to the dense matrix descriptor
 *
 *  \retval rocsparselt_status_success the operation completed successfully.
 *  \retval rocsparselt_status_invalid_handle \p handle is invalid.
 *  \retval rocsparselt_status_invalid_pointer \p descr pointer is invalid.
 *  \retval rocsparselt_status_invalid_size \p row, \p cols or \p ld is invalid.
 *  \retval rocsparselt_status_not_implemented \ref hipDataType or \ref rocsparselt_order is invalid.
 */
rocsparselt_status rocsparselt_dense_descr_init(const rocsparselt_handle* handle,
                                                rocsparselt_mat_descr*    matDescr,
                                                int64_t                   rows,
                                                int64_t                   cols,
                                                int64_t                   ld,
                                                uint32_t                  alignment,
                                                hipDataType               valueType,
                                                rocsparselt_order         order);

/*! \ingroup aux_module
 *  \brief Create a descriptor for structured matrix
 *  \details
 *  \p rocsparselt_dense_descr_init creates a matrix descriptor It initializes
 *  \ref rocsparselt_matrix_type to \ref rocsparselt_matrix_type_structured and
 *  It should be destroyed at the end using rocsparselt_mat_descr_destroy().
 *
 *  @param[out]
 *  matDescr   the pointer to the dense matrix descriptor
 *
 *  \retval rocsparselt_status_success the operation completed successfully.
 *  \retval rocsparselt_status_invalid_handle \p handle is invalid.
 *  \retval rocsparselt_status_invalid_pointer \p descr pointer is invalid.
 *  \retval rocsparselt_status_invalid_size \p row, \p cols or \p ld is invalid.
 *  \retval rocsparselt_status_not_implemented \ref hipDataType or \ref rocsparselt_order is invalid.
 */
rocsparselt_status rocsparselt_structured_descr_init(const rocsparselt_handle* handle,
                                                     rocsparselt_mat_descr*    matDescr,
                                                     int64_t                   rows,
                                                     int64_t                   cols,
                                                     int64_t                   ld,
                                                     uint32_t                  alignment,
                                                     hipDataType               valueType,
                                                     rocsparselt_order         order,
                                                     rocsparselt_sparsity      sparsity);

/*! \ingroup aux_module
 *  \brief Destroy a matrix descriptor
 *
 *  \details
 *  \p rocsparselt_mat_descr_destroy destroys a matrix descriptor and releases all
 *  resources used by the descriptor
 *
 *  @param[in]
 *  descr   the matrix descriptor
 *
 *  \retval rocsparselt_status_success the operation completed successfully.
 *  \retval rocsparselt_status_invalid_handle \p descr is invalid.
 */
rocsparselt_status rocsparselt_mat_descr_destroy(const rocsparselt_mat_descr* descr);

/*! \ingroup aux_module
 *  \brief Specify the matrix attribute of a matrix descriptor
 *
 *  \details
 *  \p rocsparselt_mat_descr_set_attribute sets the value of the specified attribute belonging
 *  to matrix descr such as number of batches and their stride.
 *
 *  @param[inout]
 *  matDescr        the matrix descriptor
 *  @param[in]
 *  handle          the rocsparselt handle
 *  matAttribute    \ref rocsparselt_mat_num_batches, \ref rocsparselt_mat_batch_stride.
 *  data            pointer to the value to which the specified attribute will be set.
 *  dataSize        size in bytes of the attribute value used for verification.
 *
 *  \retval rocsparselt_status_success the operation completed successfully.
 *  \retval rocsparselt_status_invalid_handle \p handle or \p descr is invalid.
 *  \retval rocsparselt_status_invalid_pointer \p data pointer is invalid.
 *  \retval rocsparselt_status_invalid_value \p data content is invalid, see \ref rocsparselt_mat_descr_attribute.
 *  \retval rocsparselt_status_invalid_size \p dataSize is invalid
 */
rocsparselt_status rocsparselt_mat_descr_set_attribute(const rocsparselt_handle*       handle,
                                                       rocsparselt_mat_descr*          matDescr,
                                                       rocsparselt_mat_descr_attribute matAttribute,
                                                       const void*                     data,
                                                       size_t                          dataSize);

/*! \ingroup aux_module
 *  \brief Get the matrix type of a matrix descriptor
 *
 *  \details
 *  \p rocsparselt_mat_descr_get_attribute returns the matrix attribute of a matrix descriptor
 *
 *  @param[inout]
 *  data            the memory address containing the attribute value retrieved by this function
 *
 *  @param[in]
 *  handle          the rocsparselt handle
 *  matAttribute    \ref rocsparselt_mat_num_batches, \ref rocsparselt_mat_batch_stride.
 *  matDescr        the matrix descriptor
 *  dataSize        size in bytes of the attribute value used for verification.
 *
 *  \retval rocsparselt_status_success the operation completed successfully.
 *  \retval rocsparselt_status_invalid_handle \p handle or \p descr is invalid.
 *  \retval rocsparselt_status_invalid_pointer \p data pointer is invalid.
 *  \retval rocsparselt_status_invalid_value \p data content is invalid, see \ref rocsparselt_mat_descr_attribute.
 *  \retval rocsparselt_status_invalid_size \p dataSize is invalid
 */
rocsparselt_status rocsparselt_mat_descr_get_attribute(const rocsparselt_handle*       handle,
                                                       const rocsparselt_mat_descr*    matDescr,
                                                       rocsparselt_mat_descr_attribute matAttribute,
                                                       void*                           data,
                                                       size_t                          dataSize);

/*! \ingroup aux_module
 *  \brief  Initializes the matrix multiplication descriptor.
 *
 *  \details
 *  \p rocsparselt_matmul_descr_init creates a matrix multiplication descriptor.
 *  It should be destroyed at the end using rocsparselt_matmul_descr_destroy().
 *
 *  @param[inout]
 *  matmulDescr     the matrix multiplication descriptor
 *  @param[in]
 *  handle   the rocsparselt handle
 *  opA      rocsparse operation for Matrix A.
 *  opB      rocsparse operation for Matrix B.
 *  matA     the matrix descriptor
 *  matB     the matrix descriptor
 *  matC     the matrix descriptor
 *  matD     the matrix descriptor
 *  computeType        size in bytes of the attribute value used for verification.
 *
 *  \retval rocsparselt_status_success the operation completed successfully.
 *  \retval rocsparselt_status_invalid_pointer \p matmulDescr pointer is invalid.
 *  \retval rocsparselt_status_invalid_handle \p handle or \p matA or \p matB or \p matC or \p matD is invalid.
 *  \retval rocsparselt_status_invalid_value \p computeType is invalid.
 *  \retval rocsparselt_status_invalid_size sizes of matrix A,b,C,D is invalid.
 */
rocsparselt_status rocsparselt_matmul_descr_init(const rocsparselt_handle*    handle,
                                                 rocsparselt_matmul_descr*    matmulDescr,
                                                 rocsparselt_operation        opA,
                                                 rocsparselt_operation        opB,
                                                 const rocsparselt_mat_descr* matA,
                                                 const rocsparselt_mat_descr* matB,
                                                 const rocsparselt_mat_descr* matC,
                                                 const rocsparselt_mat_descr* matD,
                                                 rocsparselt_compute_type     computeType);

/*! \ingroup aux_module
 *  \brief Specify the matrix attribute of a matrix descriptor
 *
 *  \details
 *  \p rocsparselt_matmul_descr_set_attribute sets the value of the specified attribute belonging
 *  to matrix descr such as number of batches and their stride.
 *
 *  @param[inout]
 *  matDescr        the matrix descriptor
 *  @param[in]
 *  handle          the rocsparselt handle
 *  matAttribute    \ref rocsparselt_matmul_activation_relu, \ref rocsparselt_matmul_activation_relu_upperbound,
 *                  \ref rocsparselt_matmul_activation_relu_threashold, \ref rocsparselt_matmul_activation_gelu,
 *                  \ref rocsparselt_matmul_bias_pointer, \ref rocsparselt_matmul_bias_stride
 *  data            pointer to the value to which the specified attribute will be set.
 *  dataSize        size in bytes of the attribute value used for verification.
 *
 *  \retval rocsparselt_status_success the operation completed successfully.
 *  \retval rocsparselt_status_invalid_handle \p handle or \p matDescr is invalid.
 *  \retval rocsparselt_status_invalid_pointer \p data pointer is invalid.
 *  \retval rocsparselt_status_invalid_value \p data content is invalid, see \ref rocsparselt_matmul_descr_attribute.
 *  \retval rocsparselt_status_not_implemented \p rocsparselt_mat_descr_attribute is not supported.
 *  \retval rocsparselt_status_invalid_size \p dataSize is invalid
 */
rocsparselt_status
    rocsparselt_matmul_descr_set_attribute(const rocsparselt_handle*          handle,
                                           rocsparselt_matmul_descr*          matDescr,
                                           rocsparselt_matmul_descr_attribute matAttribute,
                                           const void*                        data,
                                           size_t                             dataSize);

/*! \ingroup aux_module
 *  \brief Get the matrix type of a matrix descriptor
 *
 *  \details
 *  \p rocsparselt_matmul_descr_get_attribute returns the matrix attribute of a matrix descriptor
 *
 *  @param[inout]
 *  data            the memory address containing the attribute value retrieved by this function
 *
 *  @param[in]
 *  handle          the rocsparselt handle
 *  matAttribute    \ref rocsparselt_matmul_activation_relu, \ref rocsparselt_matmul_activation_relu_upperbound,
 *                  \ref rocsparselt_matmul_activation_relu_threashold, \ref rocsparselt_matmul_activation_gelu,
 *                  \ref rocsparselt_matmul_bias_pointer, \ref rocsparselt_matmul_bias_stride
 *  matDescr        the matrix descriptor
 *  dataSize        size in bytes of the attribute value used for verification.
 *
 *  \retval rocsparselt_status_success the operation completed successfully.
 *  \retval rocsparselt_status_invalid_handle \p handle or \p matDescr is invalid.
 *  \retval rocsparselt_status_invalid_pointer \p data pointer is invalid.
 *  \retval rocsparselt_status_invalid_value \p data content is invalid, see \ref rocsparselt_matmul_descr_attribute.
 *  \retval rocsparselt_status_not_implemented \p rocsparselt_mat_descr_attribute is not supported.
 *  \retval rocsparselt_status_invalid_size \p dataSize is invalid
 */
rocsparselt_status
    rocsparselt_matmul_descr_get_attribute(const rocsparselt_handle*          handle,
                                           const rocsparselt_matmul_descr*    matDescr,
                                           rocsparselt_matmul_descr_attribute matAttribute,
                                           void*                              data,
                                           size_t                             dataSize);

/*! \ingroup aux_module
 *  \brief Initializes the algorithm selection descriptor
 *  \details
 *  \p rocsparselt_matmul_alg_selection_init creates a algorithm selection descriptor.
 *  It should be destroyed at the end using rocsparselt_matmul_alg_selection_destroy().
 *
 *  @param[out]
 *  algSelection the pointer to the algorithm selection descriptor
 *
 *  \retval rocsparselt_status_success the operation completed successfully.
 *  \retval rocsparselt_status_invalid_handle \p handle or \p matmulDescr is invalid.
 *  \retval rocsparselt_status_invalid_pointer \p algSelection pointer is invalid.
 *  \retval rocsparselt_status_not_implemented no solution be found for this specific problem (defined by matmulDescr).
 */
rocsparselt_status
    rocsparselt_matmul_alg_selection_init(const rocsparselt_handle*         handle,
                                          rocsparselt_matmul_alg_selection* algSelection,
                                          const rocsparselt_matmul_descr*   matmulDescr,
                                          rocsparselt_matmul_alg            alg);

/*! \ingroup aux_module
 *  \brief Specify the algorithm attribute of a algorithm selection descriptor
 *
 *  \details
 *  \p rocsparselt_matmul_alg_set_attribute sets the value of the specified attribute
 *  belonging to algorithm selection descriptor.
 *
 *  @param[inout]
 *  algSelection    the algorithm selection descriptor
 *  @param[in]
 *  handle          the rocsparselt handle
 *  attribute
 *  data            pointer to the value to which the specified attribute will be set.
 *  dataSize        size in bytes of the attribute value used for verification.
 *
 *  \retval rocsparselt_status_success the operation completed successfully.
 *  \retval rocsparselt_status_invalid_handle \p handle or \p algSelection is invalid.
 *  \retval rocsparselt_status_invalid_pointer \p data pointer is invalid.
 *  \retval rocsparselt_status_invalid_value \p data content is invalid, see \ref rocsparselt_matmul_alg_attribute.
 *  \retval rocsparselt_status_invalid_size \p dataSize is invalid
 */
rocsparselt_status
    rocsparselt_matmul_alg_set_attribute(const rocsparselt_handle*         handle,
                                         rocsparselt_matmul_alg_selection* algSelection,
                                         rocsparselt_matmul_alg_attribute  attribute,
                                         const void*                       data,
                                         size_t                            dataSize);

/*! \ingroup aux_module
 *  \brief Get the specific algorithm attribute from algorithm selection descriptor
 *
 *  \details
 *  \p rocsparselt_matmul_alg_get_attribute returns the value of the queried attribute belonging
 *  to algorithm selection descriptor.
 *
 *  @param[inout]
 *  data            the memory address containing the attribute value retrieved by this function
 *
 *  @param[in]
 *  handle          the rocsparselt handle
 *  algSelection    the algorithm selection descriptor
 *  dataSize        size in bytes of the attribute value used for verification.
 *
 *  \retval rocsparselt_status_success the operation completed successfully.
 *  \retval rocsparselt_status_invalid_handle \p handle or \p algSelection is invalid.
 *  \retval rocsparselt_status_invalid_pointer \p data pointer is invalid.
 *  \retval rocsparselt_status_invalid_value \p data content is invalid, see \ref rocsparselt_matmul_alg_attribute.
 *  \retval rocsparselt_status_invalid_size \p dataSize is invalid
 */
rocsparselt_status
    rocsparselt_matmul_alg_get_attribute(const rocsparselt_handle*               handle,
                                         const rocsparselt_matmul_alg_selection* algSelection,
                                         rocsparselt_matmul_alg_attribute        attribute,
                                         void*                                   data,
                                         size_t                                  dataSize);

/*! \ingroup aux_module
 *  \brief Initializes the matrix multiplication plan descriptor
 *  \details
 *  \p rocsparselt_matmul_plan_init creates a matrix multiplication plan descriptor.
 *  It should be destroyed at the end using rocsparselt_matmul_matmul_plan_destroy().
 *
 *  @param[out]
 *  plan the pointer to the matrix multiplication plan descriptor
 *
 *  \retval rocsparselt_status_success the operation completed successfully.
 *  \retval rocsparselt_status_invalid_pointer \p plan pointer is invalid.
 *  \retval rocsparselt_status_invalid_handle \p handle or \p matmulDescr or \p algSelection is invalid.
 *  \retval rocsparselt_status_invalid_size values of \p rocsparselt_mat_num_batches from matrix A to D are inconisistent
 */
rocsparselt_status
    rocsparselt_matmul_plan_init(const rocsparselt_handle*               handle,
                                 rocsparselt_matmul_plan*                plan,
                                 const rocsparselt_matmul_descr*         matmulDescr,
                                 const rocsparselt_matmul_alg_selection* algSelection);

/*! \ingroup aux_module
 *  \brief Destroy a matrix multiplication plan descriptor
 *  \details
 *  \p rocsparselt_matmul_plan_destroy releases the resources used by an instance
 *  of the matrix multiplication plan. This function is the last call with a specific plan
 *  instance.
 *
 *  @param[in]
 *  plan the matrix multiplication plan descriptor
 *
 *  \retval rocsparselt_status_success the operation completed successfully.
 *  \retval rocsparselt_status_invalid_handle \p plan is invalid.
 */
rocsparselt_status rocsparselt_matmul_plan_destroy(const rocsparselt_matmul_plan* plan);

#ifdef __cplusplus
}
#endif

#endif /* _ROCSPARSELT_AUXILIARY_H_ */
