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

/*! \file
 *  \brief rocsparse_idvec_descr.h provides auxilary functions in rocsparse
 */

#ifndef ROCSPARSE_IDVEC_DESCR_H
#define ROCSPARSE_IDVEC_DESCR_H

#include "rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
   *  \brief Create an indexing vector descriptor
   *  \details
   *  \p rocsparse_idvec_create creates an indexing vector descriptor. It should be
   *  destroyed at the end using rocsparse_idvec_destroy().
   *
   *  This offers the flexibility to define data such that the ith element \f$e\f$
   *  \f[
   *    e := A[ i * inc ], where A is the appropriate pointer type.
   *  \f]
   *
   *  @param[in]
   *  handle       handle to the rocsparse library context queue.
   *  @param[out]
   *  descr   the pointer to the indexing vector descriptor.
   *  @param[in]
   *  indextype selected from  \ref rocsparse_indextype.
   *  @param[in]
   *  base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
   *  @param[in]
   *  size   size of the indexing vector, must be positive.
   *  @param[in]
   *  inc   increment, arbitrary value.
   *  @param[in]
   *  const_data  non-mutable non-zero values in the indexing vector (must be array of length \p size ).
   *  @param[in]
   *  data   mutable non-zero values in the indexing vector (must be array of length \p size ).
   *  @param[out]
   *  p_error      error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
   *  \retval rocsparse_status_invalid_pointer if \p descr, \p const_data is invalid, \p const_data are invalid if these are null pointers wheras \p size is positive; or if \p data is a non-null pointer and is different from \p const_data.
   *  \retval rocsparse_status_invalid_size if \p size is negative.
   *  \retval rocsparse_status_invalid_value if \p indextype is invalid.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_idvec_create(rocsparse_handle       handle,
                                        rocsparse_idvec_descr* descr,
                                        rocsparse_indextype    indextype,
                                        rocsparse_index_base   base,
                                        int64_t                size,
                                        int64_t                inc,
                                        const void*            const_data,
                                        void*                  data,
                                        rocsparse_error*       p_error);

/*! \ingroup aux_module
   *  \brief Create a batched indexing vector descriptor
   *  \details
   *  \p rocsparse_idvec_create_batched creates a batched indexing vector descriptor. It should be
   *  destroyed at the end using rocsparse_idvec_destroy().
   *
   *  In the case of a strided batched data type, \ref rocsparse_batchtype_strided, this offers the flexibility to define batched data such that the \f$i^{th}\f$ element \f$e\f$ of the \f$j^{th}\f$ vector is:
   *  \f[
   *    e := A[ j * batch_dist + i * inc], where j is the batch index, and A is the appropriate pointer type.
   *  \f]
   *
   *  In the case of an array of pointers data type, \ref rocsparse_batchtype_pointerarray, this offers the flexibility to define batched data such that the \f$i^{th}\f$ element \f$e\f$ of the \f$j^{th}\f$ vector is:
   *  \f[
   *    e := A[j * batch_dist][i * inc], where j is the batch index, and A is the appropriate pointer type.
   *  \f]
   *
   *  Note: The values of the batch distance \p batch_dist and the increment \p inc are voluntarily left as arbitrary to maximize the flexbility.
   *
   *  @param[in]
   *  handle       handle to the rocsparse library context queue.
   *  @param[out]
   *  descr   the pointer to the indexing vector descriptor.
   *  @param[in]
   *  indextype   indextype from \ref rocsparse_indextype.
   *  @param[in]
   *  base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
   *  @param[in]
   *  size   size of the indexing vector, must be positive.
   *  @param[in]
   *  inc   increment, arbitrary value.
   *  @param[in]
   *  batch_type  type of the batched data.
   *  @param[in]
   *  batch_storage  storage type of the batched data.
   *  @param[in]
   *  batch_count  size of the batched data, must be positive.
   *  @param[in]
   *  batch_dist   batch distance, arbitrary value.
   *  @param[in]
   *  const_data  non-mutable non-zero values in the indexing vector (must be array of length \p size ).
   *  @param[in]
   *  data   mutable non-zero values in the indexing vector (must be array of length \p size ).
   *  @param[out]
   *  p_error      error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
   *  \retval rocsparse_status_invalid_pointer if \p descr, \p const_data is invalid, \p const_data are invalid if these are null pointers wheras \p size is positive; or if \p data is a non-null pointer and is different from \p const_data.
   *  \retval rocsparse_status_invalid_size if \p size is negative.
   *  \retval rocsparse_status_invalid_value if \p batch_type or \p indextype is invalid.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_idvec_create_batched(rocsparse_handle       handle,
                                                rocsparse_idvec_descr* descr,
                                                rocsparse_indextype    indextype,
                                                rocsparse_index_base   base,
                                                int64_t                size,
                                                int64_t                inc,
                                                rocsparse_batchtype    batch_type,
                                                rocsparse_batchstorage batch_storage,
                                                int64_t                batch_count,
                                                int64_t                batch_dist,
                                                const void*            const_data,
                                                void*                  data,
                                                rocsparse_error*       p_error);

/*! \ingroup aux_module
   *  \brief Destroy an indexing vector descriptor
   *
   *  \details
   *  \p rocsparse_idvec_destroy destroys an indexing vector descriptor and releases all
   *  resources used by the descriptor.
   *
   *  @param[in]
   *  handle  handle to the rocsparse library context queue.
   *  @param[in]
   *  descr   the matrix descriptor.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
   *  \retval rocsparse_status_invalid_pointer \p descr is invalid.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_idvec_destroy(rocsparse_handle      handle,
                                         rocsparse_idvec_descr descr,
                                         rocsparse_error*      p_error);

/*! \ingroup aux_module
   *  \brief Set a property of the indexing vector descriptor
   *
   *  \details
   *  \p rocsparse_idvec_set_prop sets a property of the indexing vector descriptor.
   *
   *  @param[in]
   *  handle  handle to the rocsparse library context queue.
   *  @param[in]
   *  descr   the matrix descriptor.
   *  @param[in]
   *  prop   select from \ref rocsparse_idvec_prop.
   *  @param[out]
   *  p_value   pointer to the value.
   *  @param[in]
   *  value_size_in_bytes size in bytes of the memory \p p_value points to, this must match the required size given from the underlying type given by the documentation of \ref rocsparse_idvec_prop.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
   *  \retval rocsparse_status_invalid_pointer if \p descr or \p p_value is invalid.
   *  \retval rocsparse_status_invalid_value if \p prop is invalid or if \p value_size_in_bytes does not match the required size.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_idvec_get_prop(rocsparse_handle            handle,
                                          rocsparse_const_idvec_descr descr,
                                          rocsparse_idvec_prop        prop,
                                          void*                       p_value,
                                          size_t                      value_size_in_bytes,
                                          rocsparse_error*            p_error);

/*! \ingroup aux_module
   *  \brief Set a property of the indexing vector descriptor
   *
   *  \details
   *  \p rocsparse_idvec_set_prop sets a property of the indexing vector descriptor.
   *
   *  @param[in]
   *  handle  handle to the rocsparse library context queue.
   *  @param[in]
   *  descr   the matrix descriptor.
   *  @param[in]
   *  prop   select from \ref rocsparse_idvec_prop.
   *  @param[in]
   *  p_value   pointer to the value.
   *  @param[in]
   *  value_size_in_bytes size in bytes of the memory \p p_value points to, this must match the required size given from the underlying type given by the documentation of \ref rocsparse_idvec_prop.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
   *  \retval rocsparse_status_invalid_pointer if \p descr or \p p_value is invalid.
   *  \retval rocsparse_status_invalid_value if \p prop is invalid or if \p value_size_in_bytes does not match the required size.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_idvec_set_prop(rocsparse_handle      handle,
                                          rocsparse_idvec_descr descr,
                                          rocsparse_idvec_prop  prop,
                                          const void*           p_value,
                                          size_t                value_size_in_bytes,
                                          rocsparse_error*      p_error);

/*! \ingroup aux_module
   *  \brief Get mutable data to the indexing vector descriptor
   *
   *  \details
   *  \p rocsparse_idvec_get_data gets the pointer of mutable data of the indexing vector descriptor.
   *
   *  \note The pointer to mutable data is null if the vector has been defined with non-mutanle data only.
   *
   *  @param[in]
   *  handle  handle to the rocsparse library context queue.
   *  @param[in]
   *  descr   the matrix descriptor.
   *  @param[out]
   *  p_data   pointer to mutable data.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
   *  \retval rocsparse_status_invalid_pointer if \p descr or \p p_data is invalid.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_idvec_get_data(rocsparse_handle      handle,
                                          rocsparse_idvec_descr descr,
                                          void**                p_data,
                                          rocsparse_error*      p_error);

/*! \ingroup aux_module
   *  \brief Get non-mutable data to the indexing vector descriptor
   *
   *  \details
   *  \p rocsparse_idvec_get_const_data gets the pointer of non-mutable data of the indexing vector descriptor.
   *
   *  \note The pointer to non-mutable data is always available.
   *
   *  @param[in]
   *  handle  handle to the rocsparse library context queue.
   *  @param[in]
   *  descr   the matrix descriptor.
   *  @param[out]
   *  p_const_data   pointer to non-mutable data.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
   *  \retval rocsparse_status_invalid_pointer if \p descr or \p p_const_data is invalid.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_idvec_get_const_data(rocsparse_handle            handle,
                                                rocsparse_const_idvec_descr descr,
                                                const void**                p_const_data,
                                                rocsparse_error*            p_error);

/*! \ingroup aux_module
   *  \brief Set mutable data to the indexing vector descriptor
   *
   *  \details
   *  \p rocsparse_idvec_set_data sets the data of the indexing vector descriptor.
   *
   *  \note This sets the mutable and non-mutable pointers to the data.
   *
   *  @param[in]
   *  handle  handle to the rocsparse library context queue.
   *  @param[in]
   *  descr   the matrix descriptor.
   *  @param[in]
   *  data   pointer to mutable data.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
   *  \retval rocsparse_status_invalid_pointer if \p descr or \p data is invalid.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_idvec_set_data(rocsparse_handle      handle,
                                          rocsparse_idvec_descr descr,
                                          void*                 data,
                                          rocsparse_error*      p_error);

/*! \ingroup aux_module
   *  \brief Set non-mutable data to the indexing vector descriptor
   *
   *  \details
   *  \p rocsparse_idvec_set_const_data sets the pointer of non-mutable data of the indexing vector descriptor.
   *
   *  \note This only sets the non-mutable pointer to the data, the mutable pointer is set to null.
   *
   *  @param[in]
   *  handle  handle to the rocsparse library context queue.
   *  @param[in]
   *  descr   the matrix descriptor.
   *  @param[in]
   *  const_data   pointer to non-mutable data.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
   *  \retval rocsparse_status_invalid_pointer if \p descr or \p const_data is invalid.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_idvec_set_const_data(rocsparse_handle      handle,
                                                rocsparse_idvec_descr descr,
                                                const void*           const_data,
                                                rocsparse_error*      p_error);

#ifdef __cplusplus
}
#endif

#endif
