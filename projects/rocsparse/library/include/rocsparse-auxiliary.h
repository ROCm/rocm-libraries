/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
 *  \brief rocsparse-auxiliary.h provides auxiliary functions in rocsparse
 */

#ifndef ROCSPARSE_AUXILIARY_H
#define ROCSPARSE_AUXILIARY_H

#include "internal/auxiliary/rocsparse_dnmat_descr.h"
#include "internal/auxiliary/rocsparse_dnvec_descr.h"
#include "internal/auxiliary/rocsparse_idvec_descr.h"
#include "internal/auxiliary/rocsparse_mat_info.h"
#include "internal/auxiliary/rocsparse_spattern_descr.h"
#include "internal/auxiliary/rocsparse_spgeam_descr.h"
#include "internal/auxiliary/rocsparse_spic0_descr.h"
#include "internal/auxiliary/rocsparse_spilu0_descr.h"
#include "internal/auxiliary/rocsparse_spmat_descr.h"
#include "internal/auxiliary/rocsparse_spmv_descr.h"
#include "internal/auxiliary/rocsparse_sptrsm_descr.h"
#include "internal/auxiliary/rocsparse_sptrsv_descr.h"

#include "internal/auxiliary/rocsparse_dnmat_descr_backward.h"
#include "internal/auxiliary/rocsparse_dnvec_descr_backward.h"
#include "internal/auxiliary/rocsparse_extract_descr_backward.h"
#include "internal/auxiliary/rocsparse_mat_info_backward.h"
#include "internal/auxiliary/rocsparse_sparse_to_sparse_descr_backward.h"
#include "internal/auxiliary/rocsparse_spgeam_descr_backward.h"
#include "internal/auxiliary/rocsparse_spmat_descr_backward.h"
#include "internal/auxiliary/rocsparse_spmv_descr_backward.h"
#include "internal/auxiliary/rocsparse_sptrsm_descr_backward.h"
#include "internal/auxiliary/rocsparse_sptrsv_descr_backward.h"

#include "rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"
#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 *  \brief Create a rocsparse handle
 *
 *  \details
 *  \p rocsparse_create_handle creates the rocSPARSE library context. It must be
 *  initialized before any other rocSPARSE API function is invoked and must be passed to
 *  all subsequent library function calls. The handle should be destroyed at the end
 *  using rocsparse_destroy_handle().
 *
 *  @param[out]
 *  handle  the pointer to the handle to the rocSPARSE library context.
 *
 *  \retval rocsparse_status_success the initialization succeeded.
 *  \retval rocsparse_status_invalid_handle \p handle pointer is invalid.
 *  \retval rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_handle(rocsparse_handle* handle);

/*! \ingroup aux_module
 *  \brief Destroy a rocsparse handle
 *
 *  \details
 *  \p rocsparse_destroy_handle destroys the rocSPARSE library context and releases all
 *  resources used by the rocSPARSE library.
 *
 *  @param[in]
 *  handle  the handle to the rocSPARSE library context.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle \p handle is invalid.
 *  \retval rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_handle(rocsparse_handle handle);

/*! \ingroup aux_module
 *  \brief Destroy a rocsparse error descriptor.
 *
 *  \details
 *  \p rocsparse_destroy_error destroys the rocSPARSE error descriptor.
 *
 *  @param[in]
 *  error  the pointer to the rocSPARSE error descriptor, it can be a null pointer.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_error(rocsparse_error error);

/*! \ingroup aux_module
 *  \brief Eerror message from a rocsparse error descriptor.
 *
 *  \details
 *  \p rocsparse_error_message returns a C-style string that provides detail for the error.
 *
 *  @param[in]
 *  error  the error to the rocSPARSE error descriptor.
 *
 *  @return an error message from a rocsparse error descriptor.
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
const char* rocsparse_error_get_message(rocsparse_error error);

/*! \ingroup aux_module
 *  \brief Return the string representation of a rocSPARSE status code enum name
 *
 *  \details
 *  \p rocsparse_get_status_name takes a rocSPARSE status as input and returns the string representation of this status.
 *  If the status is not recognized, the function returns "Unrecognized status code"
 *
 *  @param[in]
 *  status  a rocSPARSE status
 *
 *  \retval pointer to null terminated string
 */
ROCSPARSE_EXPORT
const char* rocsparse_get_status_name(rocsparse_status status);

/*! \ingroup aux_module
 *  \brief Return the rocSPARSE status code description as a string
 *
 *  \details
 *  \p rocsparse_get_status_description takes a rocSPARSE status as input and returns the status description as a string.
 *  If the status is not recognized, the function returns "Unrecognized status code"
 *
 *  @param[in]
 *  status  a rocSPARSE status
 *
 *  \retval pointer to null terminated string
 */
ROCSPARSE_EXPORT
const char* rocsparse_get_status_description(rocsparse_status status);

/*! \ingroup aux_module
 *  \brief Specify user defined HIP stream
 *
 *  \details
 *  \p rocsparse_set_stream specifies the stream to be used by the rocSPARSE library
 *  context and all subsequent function calls.
 *
 *  @param[inout]
 *  handle  the handle to the rocSPARSE library context.
 *  @param[in]
 *  stream  the stream to be used by the rocSPARSE library context.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle \p handle is invalid.
 *
 *  \par Example
 *  This example illustrates, how a user defined stream can be used in rocSPARSE.
 *  \code{.c}
 *      // Create rocSPARSE handle
 *      rocsparse_handle handle;
 *      rocsparse_create_handle(&handle);
 *
 *      // Create stream
 *      hipStream_t stream;
 *      hipStreamCreate(&stream);
 *
 *      // Set stream to rocSPARSE handle
 *      rocsparse_set_stream(handle, stream);
 *
 *      // Do some work
 *      // ...
 *
 *      // Clean up
 *      rocsparse_destroy_handle(handle);
 *      hipStreamDestroy(stream);
 *  \endcode
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_stream(rocsparse_handle handle, hipStream_t stream);

/*! \ingroup aux_module
 *  \brief Get current stream from library context
 *
 *  \details
 *  \p rocsparse_get_stream gets the rocSPARSE library context stream which is currently
 *  used for all subsequent function calls.
 *
 *  @param[in]
 *  handle the handle to the rocSPARSE library context.
 *  @param[out]
 *  stream the stream currently used by the rocSPARSE library context.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle \p handle is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_get_stream(rocsparse_handle handle, hipStream_t* stream);

/*! \ingroup aux_module
 *  \brief Specify pointer mode
 *
 *  \details
 *  \p rocsparse_set_pointer_mode specifies the pointer mode to be used by the rocSPARSE
 *  library context and all subsequent function calls. For example, many rocSPARSE routines take
 *  \f$\alpha\f$ and \f$\beta\f$ pointers as parameters. These can be either host memory pointers
 *  or device memory pointers depending on what the pointer mode is set to. By default, all values are passed
 *  using host pointer mode. Valid pointer modes are \ref rocsparse_pointer_mode_host
 *  or \ref rocsparse_pointer_mode_device.
 *
 *  @param[in]
 *  handle          the handle to the rocSPARSE library context.
 *  @param[in]
 *  pointer_mode    the pointer mode to be used by the rocSPARSE library context.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle \p handle is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_pointer_mode(rocsparse_handle       handle,
                                            rocsparse_pointer_mode pointer_mode);

/*! \ingroup aux_module
 *  \brief Get current pointer mode from library context
 *
 *  \details
 *  \p rocsparse_get_pointer_mode gets the rocSPARSE library context pointer mode which
 *  is currently used for all subsequent function calls.
 *
 *  @param[in]
 *  handle          the handle to the rocSPARSE library context.
 *  @param[out]
 *  pointer_mode    the pointer mode that is currently used by the rocSPARSE library
 *                  context.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle \p handle is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_get_pointer_mode(rocsparse_handle        handle,
                                            rocsparse_pointer_mode* pointer_mode);

/*! \ingroup aux_module
 *  \brief Get rocSPARSE version
 *
 *  \details
 *  \p rocsparse_get_version gets the rocSPARSE library version number.
 *  - patch = version % 100
 *  - minor = version / 100 % 1000
 *  - major = version / 100000
 *
 *  @param[in]
 *  handle  the handle to the rocSPARSE library context.
 *  @param[out]
 *  version the version number of the rocSPARSE library.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle \p handle is invalid.
 *  \par Example
 *  \code{.c}
 *   rocsparse_handle handle;
 *   rocsparse_create_handle(&handle);
 *   rocsparse_get_version(handle, &rocsparse_ver);
 *   rocsparse_destroy_handle(handle);
 *  \endcode
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_get_version(rocsparse_handle handle, int* version);

/*! \ingroup aux_module
 *  \brief Get rocSPARSE git revision
 *
 *  \details
 *  \p rocsparse_get_git_rev gets the rocSPARSE library git commit revision (SHA-1).
 *
 *  @param[in]
 *  handle  the handle to the rocSPARSE library context.
 *  @param[out]
 *  rev     the git commit revision (SHA-1).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle \p handle is invalid.
 *  \par Example
 *  \code{.c}
 *   rocsparse_handle handle;
 *   rocsparse_create_handle(&handle);
 *   rocsparse_get_git_rev(handle, rocsparse_rev);
 *   rocsparse_destroy_handle(handle);
 *  \endcode
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_get_git_rev(rocsparse_handle handle, char* rev);

/*! \ingroup aux_module
 *  \brief Create a matrix descriptor
 *  \details
 *  \p rocsparse_create_mat_descr creates a matrix descriptor. It initializes
 *  \ref rocsparse_matrix_type to \ref rocsparse_matrix_type_general, \ref rocsparse_fill_mode
 *  to \ref rocsparse_fill_mode_lower, \ref rocsparse_diag_type to \ref rocsparse_diag_type_non_unit,
 *  \ref rocsparse_index_base to \ref rocsparse_index_base_zero, and \ref rocsparse_storage_mode
 *  to \ref rocsparse_storage_mode_sorted.  It should be destroyed at the end using
 *  \ref rocsparse_destroy_mat_descr().
 *
 *  The matrix type, fill mode, diag type, index base, and storage mode can be set using the
 *  \ref rocsparse_set_mat_type, \ref rocsparse_set_mat_fill_mode, \ref rocsparse_set_mat_diag_type,
 *  \ref rocsparse_set_mat_index_base, and \ref rocsparse_set_mat_storage_mode APIs respectively.
 *
 *  @param[out]
 *  descr   the pointer to the matrix descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_mat_descr(rocsparse_mat_descr* descr);

/*! \ingroup aux_module
 *  \brief Copy a matrix descriptor
 *  \details
 *  \p rocsparse_copy_mat_descr copies a matrix descriptor. Both, source and destination
 *  matrix descriptors must be initialized prior to calling \p rocsparse_copy_mat_descr.
 *
 *  @param[out]
 *  dest    the pointer to the destination matrix descriptor.
 *  @param[in]
 *  src     the pointer to the source matrix descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p src or \p dest pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_copy_mat_descr(rocsparse_mat_descr dest, const rocsparse_mat_descr src);

/*! \ingroup aux_module
 *  \brief Destroy a matrix descriptor
 *
 *  \details
 *  \p rocsparse_destroy_mat_descr destroys a matrix descriptor and releases all
 *  resources used by the descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_mat_descr(rocsparse_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Specify the index base of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_set_mat_index_base sets the index base of a matrix descriptor. Valid
 *  options are \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  @param[inout]
 *  descr   the matrix descriptor.
 *  @param[in]
 *  base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr pointer is invalid.
 *  \retval rocsparse_status_invalid_value \p base is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_mat_index_base(rocsparse_mat_descr descr, rocsparse_index_base base);

/*! \ingroup aux_module
 *  \brief Get the index base of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_get_mat_index_base returns the index base of a matrix descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 */
ROCSPARSE_EXPORT
rocsparse_index_base rocsparse_get_mat_index_base(const rocsparse_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Specify the matrix type of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_set_mat_type sets the matrix type of a matrix descriptor. Valid
 *  matrix types are \ref rocsparse_matrix_type_general,
 *  \ref rocsparse_matrix_type_symmetric, \ref rocsparse_matrix_type_hermitian or
 *  \ref rocsparse_matrix_type_triangular.
 *
 *  @param[inout]
 *  descr   the matrix descriptor.
 *  @param[in]
 *  type    \ref rocsparse_matrix_type_general, \ref rocsparse_matrix_type_symmetric,
 *          \ref rocsparse_matrix_type_hermitian or
 *          \ref rocsparse_matrix_type_triangular.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr pointer is invalid.
 *  \retval rocsparse_status_invalid_value \p type is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_mat_type(rocsparse_mat_descr descr, rocsparse_matrix_type type);

/*! \ingroup aux_module
 *  \brief Get the matrix type of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_get_mat_type returns the matrix type of a matrix descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns    \ref rocsparse_matrix_type_general, \ref rocsparse_matrix_type_symmetric,
 *              \ref rocsparse_matrix_type_hermitian or
 *              \ref rocsparse_matrix_type_triangular.
 */
ROCSPARSE_EXPORT
rocsparse_matrix_type rocsparse_get_mat_type(const rocsparse_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Specify the matrix fill mode of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_set_mat_fill_mode sets the matrix fill mode of a matrix descriptor.
 *  Valid fill modes are \ref rocsparse_fill_mode_lower or
 *  \ref rocsparse_fill_mode_upper.
 *
 *  @param[inout]
 *  descr       the matrix descriptor.
 *  @param[in]
 *  fill_mode   \ref rocsparse_fill_mode_lower or \ref rocsparse_fill_mode_upper.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr pointer is invalid.
 *  \retval rocsparse_status_invalid_value \p fill_mode is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_mat_fill_mode(rocsparse_mat_descr descr,
                                             rocsparse_fill_mode fill_mode);

/*! \ingroup aux_module
 *  \brief Get the matrix fill mode of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_get_mat_fill_mode returns the matrix fill mode of a matrix descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns    \ref rocsparse_fill_mode_lower or \ref rocsparse_fill_mode_upper.
 */
ROCSPARSE_EXPORT
rocsparse_fill_mode rocsparse_get_mat_fill_mode(const rocsparse_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Specify the matrix diagonal type of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_set_mat_diag_type sets the matrix diagonal type of a matrix
 *  descriptor. Valid diagonal types are \ref rocsparse_diag_type_unit or
 *  \ref rocsparse_diag_type_non_unit.
 *
 *  @param[inout]
 *  descr       the matrix descriptor.
 *  @param[in]
 *  diag_type   \ref rocsparse_diag_type_unit or \ref rocsparse_diag_type_non_unit.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr pointer is invalid.
 *  \retval rocsparse_status_invalid_value \p diag_type is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_mat_diag_type(rocsparse_mat_descr descr,
                                             rocsparse_diag_type diag_type);

/*! \ingroup aux_module
 *  \brief Get the matrix diagonal type of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_get_mat_diag_type returns the matrix diagonal type of a matrix
 *  descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns \ref rocsparse_diag_type_unit or \ref rocsparse_diag_type_non_unit.
 */
ROCSPARSE_EXPORT
rocsparse_diag_type rocsparse_get_mat_diag_type(const rocsparse_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Specify the matrix storage mode of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_set_mat_storage_mode sets the matrix storage mode of a matrix descriptor.
 *  Valid fill modes are \ref rocsparse_storage_mode_sorted or
 *  \ref rocsparse_storage_mode_unsorted.
 *
 *  @param[inout]
 *  descr           the matrix descriptor.
 *  @param[in]
 *  storage_mode    \ref rocsparse_storage_mode_sorted or
 *                  \ref rocsparse_storage_mode_unsorted.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr pointer is invalid.
 *  \retval rocsparse_status_invalid_value \p storage_mode is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_mat_storage_mode(rocsparse_mat_descr    descr,
                                                rocsparse_storage_mode storage_mode);

/*! \ingroup aux_module
 *  \brief Get the matrix storage mode of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_get_mat_storage_mode returns the matrix storage mode of a matrix descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns    \ref rocsparse_storage_mode_sorted or \ref rocsparse_storage_mode_unsorted.
 */
ROCSPARSE_EXPORT
rocsparse_storage_mode rocsparse_get_mat_storage_mode(const rocsparse_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Create a \p HYB matrix structure
 *
 *  \details
 *  \p rocsparse_create_hyb_mat creates a structure that holds the matrix in \p HYB
 *  storage format. It should be destroyed at the end using rocsparse_destroy_hyb_mat().
 *
 *  @param[inout]
 *  hyb the pointer to the hybrid matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p hyb pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_hyb_mat(rocsparse_hyb_mat* hyb);

/*! \ingroup aux_module
 *  \brief Copy a \p HYB matrix structure
 *
 *  \details
 *  \p rocsparse_copy_hyb_mat copies a matrix info structure. Both, source and destination
 *  matrix info structure must be initialized prior to calling \p rocsparse_copy_hyb_mat.
 *
 *  @param[out]
 *  dest    the pointer to the destination matrix info structure.
 *  @param[in]
 *  src     the pointer to the source matrix info structure.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p hyb pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_copy_hyb_mat(rocsparse_hyb_mat dest, const rocsparse_hyb_mat src);

/*! \ingroup aux_module
 *  \brief Destroy a \p HYB matrix structure
 *
 *  \details
 *  \p rocsparse_destroy_hyb_mat destroys a \p HYB structure.
 *
 *  @param[in]
 *  hyb the hybrid matrix structure.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p hyb pointer is invalid.
 *  \retval rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_hyb_mat(rocsparse_hyb_mat hyb);

/*! \ingroup aux_module
 *  \brief Create a color info structure
 *
 *  \details
 *  \p rocsparse_create_color_info creates a structure that holds the color info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using rocsparse_destroy_color_info().
 *
 *  @param[inout]
 *  info    the pointer to the info structure.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p info pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_color_info(rocsparse_color_info* info);

/*! \ingroup aux_module
 *  \brief Copy a color info structure
 *  \details
 *  \p rocsparse_copy_color_info copies a color info structure. Both, source and destination
 *  color info structure must be initialized prior to calling \p rocsparse_copy_color_info.
 *
 *  @param[out]
 *  dest    the pointer to the destination color info structure.
 *  @param[in]
 *  src     the pointer to the source color info structure.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p src or \p dest pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_copy_color_info(rocsparse_color_info       dest,
                                           const rocsparse_color_info src);

/*! \ingroup aux_module
 *  \brief Destroy a color info structure
 *
 *  \details
 *  \p rocsparse_destroy_color_info destroys a color info structure.
 *
 *  @param[in]
 *  info    the info structure.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p info pointer is invalid.
 *  \retval rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_color_info(rocsparse_color_info info);

// Generic API

/*! \ingroup aux_module
 *  \brief Create a sparse vector descriptor
 *  \details
 *  \p rocsparse_create_spvec_descr creates a sparse vector descriptor. It should be
 *  destroyed at the end using rocsparse_destroy_mat_descr().
 *
 *  @param[out]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[in]
 *  size   size of the sparse vector.
 *  @param[in]
 *  nnz   number of non-zeros in sparse vector.
 *  @param[in]
 *  indices   indices of the sparse vector where non-zeros occur (must be array of length \p nnz ).
 *  @param[in]
 *  values   non-zero values in the sparse vector (must be array of length \p nnz ).
 *  @param[in]
 *  idx_type   \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base   \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p indices or \p values is invalid.
 *  \retval rocsparse_status_invalid_size if \p size or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_spvec_descr(rocsparse_spvec_descr* descr,
                                              int64_t                size,
                                              int64_t                nnz,
                                              void*                  indices,
                                              void*                  values,
                                              rocsparse_indextype    idx_type,
                                              rocsparse_index_base   idx_base,
                                              rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_const_spvec_descr(rocsparse_const_spvec_descr* descr,
                                                    int64_t                      size,
                                                    int64_t                      nnz,
                                                    const void*                  indices,
                                                    const void*                  values,
                                                    rocsparse_indextype          idx_type,
                                                    rocsparse_index_base         idx_base,
                                                    rocsparse_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Destroy a sparse vector descriptor
 *
 *  \details
 *  \p rocsparse_destroy_spvec_descr destroys a sparse vector descriptor and releases all
 *  resources used by the descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_spvec_descr(rocsparse_const_spvec_descr descr);

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse vector descriptor
 *  \details
 *  \p rocsparse_spvec_get gets the fields of the sparse vector descriptor
 *
 *  @param[in]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[out]
 *  size   size of the sparse vector.
 *  @param[out]
 *  nnz   number of non-zeros in sparse vector.
 *  @param[out]
 *  indices   indices of the sparse vector where non-zeros occur (must be array of length \p nnz ).
 *  @param[out]
 *  values   non-zero values in the sparse vector (must be array of length \p nnz ).
 *  @param[out]
 *  idx_type   \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base   \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p indices or \p values is invalid.
 *  \retval rocsparse_status_invalid_size if \p size or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spvec_get(const rocsparse_spvec_descr descr,
                                     int64_t*                    size,
                                     int64_t*                    nnz,
                                     void**                      indices,
                                     void**                      values,
                                     rocsparse_indextype*        idx_type,
                                     rocsparse_index_base*       idx_base,
                                     rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_spvec_get(rocsparse_const_spvec_descr descr,
                                           int64_t*                    size,
                                           int64_t*                    nnz,
                                           const void**                indices,
                                           const void**                values,
                                           rocsparse_indextype*        idx_type,
                                           rocsparse_index_base*       idx_base,
                                           rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the index base stored in the sparse vector descriptor
 *
 *  @param[in]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[out]
 *  idx_base   \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_base is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spvec_get_index_base(rocsparse_const_spvec_descr descr,
                                                rocsparse_index_base*       idx_base);

/*! \ingroup aux_module
 *  \brief Get the values array stored in the sparse vector descriptor
 *
 *  @param[in]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[out]
 *  values   non-zero values in the sparse vector (must be array of length \p nnz ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spvec_get_values(const rocsparse_spvec_descr descr, void** values);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_spvec_get_values(rocsparse_const_spvec_descr descr,
                                                  const void**                values);
/**@}*/

/*! \ingroup aux_module
 *  \brief Set the values array in the sparse vector descriptor
 *
 *  @param[inout]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[in]
 *  values   non-zero values in the sparse vector (must be array of length \p nnz ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spvec_set_values(rocsparse_spvec_descr descr, void* values);

//
// If ROCSPARSE_WITH_MEMSTAT is defined
// then a set of extra routines is offered
// to manage memory with a recording of some traces.
//
#ifdef ROCSPARSE_WITH_MEMSTAT
/*! \ingroup aux_module
   *  \brief Set the memory report filename.
   *
   *  \details
   *  \p rocsparse_memstat_report set the filename to use for the memory report.
   *  This routine is optional, but it must be called before any hip memory operation.
   *  Note that the default memory report filename is 'rocsparse_memstat.json'.
   *  Also note that if any operation occurs before calling this routine, the default filename rocsparse_memstat.json
   *  will be used but renamed after this call.
   *  The content of the memory report summarizes memory operations from the use of the routines
   *  \ref rocsparse_hip_malloc,
   *  \ref rocsparse_hip_free,
   *  \ref rocsparse_hip_host_malloc,
   *  \ref rocsparse_hip_host_free,
   *  \ref rocsparse_hip_host_managed,
   *  \ref rocsparse_hip_free_managed.
   *
   *  @param[in]
   *  filename  the memory report filename.
   *
   *  \retval rocsparse_status_success the operation succeeded.
   *  \retval rocsparse_status_invalid_pointer \p handle filename is an invalid pointer.
   *  \retval rocsparse_status_internal_error an internal error occurred.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_memstat_report(const char* filename);

/*! \ingroup aux_module
   *  \brief Wrap hipFree.
   *
   *  @param[in]
   *  mem  memory pointer
   *  @param[in]
   *  tag  tag to attach to the operation.
   *
   *  \retval error from the related hip operation.
   */
ROCSPARSE_EXPORT
hipError_t rocsparse_hip_free(void* mem, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipMalloc.
   *
   *  @param[in]
   *  mem  pointer of memory pointer
   *  @param[in]
   *  nbytes  number of bytes
   *  @param[in]
   *  tag  tag to attach to the operation
   *
   *  \retval error from the related hip operation
   */
ROCSPARSE_EXPORT
hipError_t rocsparse_hip_malloc(void** mem, size_t nbytes, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipFreeAsync.
   *
   *  @param[in]
   *  mem  memory pointer
   *  @param[in]
   *  stream  the stream to be used by the asynchronous operation
   *  @param[in]
   *  tag  tag to attach to the operation.
   *
   *  \retval error from the related hip operation.
   */
ROCSPARSE_EXPORT
hipError_t rocsparse_hip_free_async(void* mem, hipStream_t stream, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipMallocAsync.
   *
   *  @param[in]
   *  mem  pointer of memory pointer
   *  @param[in]
   *  nbytes  number of bytes
   *  @param[in]
   *  stream  the stream to be used by the asynchronous operation
   *  @param[in]
   *  tag  tag to attach to the operation
   *
   *  \retval error from the related hip operation
   */
ROCSPARSE_EXPORT
hipError_t
    rocsparse_hip_malloc_async(void** mem, size_t nbytes, hipStream_t stream, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipHostFree.
   *
   *  @param[in]
   *  mem  memory pointer
   *  @param[in]
   *  tag  tag to attach to the operation.
   *
   *  \retval error from the related hip operation.
   */
ROCSPARSE_EXPORT
hipError_t rocsparse_hip_host_free(void* mem, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipHostMalloc.
   *
   *  @param[in]
   *  mem  pointer of memory pointer
   *  @param[in]
   *  nbytes  number of bytes
   *  @param[in]
   *  tag  tag to attach to the operation
   *
   *  \retval error from the related hip operation
   */
ROCSPARSE_EXPORT
hipError_t rocsparse_hip_host_malloc(void** mem, size_t nbytes, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipFreeManaged.
   *
   *  @param[in]
   *  mem  memory pointer
   *  @param[in]
   *  tag  tag to attach to the operation.
   *
   *  \retval error from the related hip operation.
   */
ROCSPARSE_EXPORT
hipError_t rocsparse_hip_free_managed(void* mem, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipMallocManaged.
   *
   *  @param[in]
   *  mem  pointer of memory pointer
   *  @param[in]
   *  nbytes  number of bytes
   *  @param[in]
   *  tag  tag to attach to the operation
   *
   *  \retval error from the related hip operation
   */
ROCSPARSE_EXPORT
hipError_t rocsparse_hip_malloc_managed(void** mem, size_t nbytes, const char* tag);

#endif

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_AUXILIARY_H */
