/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc.
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

/*!\file
 * \brief rocsparseio.h includes other *.h and exposes a common interface
 */
#ifndef _ROCSPARSEIO_H_
#define _ROCSPARSEIO_H_

#define ROCSPARSEIO_VERSION_MAJOR 1

#include <stddef.h>
#include <stdint.h>

/*! \ingroup types_module
 *  \brief Specifies whether int32 or int64 is used.
 */
#ifdef __cplusplus
#include <cstdint>
#else
typedef long int      int32_t;
typedef long long int int64_t;
#endif

#include <stdarg.h>

typedef char rocsparseio_string[512];

/*! \ingroup types_module
 *  \brief Info structure to hold all matrix meta data.
 *
 *  \details
 *  The rocSPARSE matrix info is a structure holding all matrix information that
 * is gathered during analysis routines. It must be initialized using
 *  rocsparse_create_mat_info() and the returned info structure must be passed
 * to all subsequent library calls that require additional matrix information.
 * It should be destroyed at the end using rocsparse_destroy_mat_info().
 */

/*! \ingroup types_module
 *  \brief Type used for enums in rocsparseio.
 *
 *  \details
 *  The type \ref rocsparseio_enum_t must be the type used by enums.
 */
typedef int32_t rocsparseio_enum_t;

/*! \ingroup types_module
 *  \brief Enumeration type for status.
 */
typedef rocsparseio_enum_t rocsparseio_status_t;
typedef rocsparseio_enum_t rocsparseio_rwmode_t;
typedef rocsparseio_enum_t rocsparseio_order_t;
typedef rocsparseio_enum_t rocsparseio_direction_t;
typedef rocsparseio_enum_t rocsparseio_type_t;
typedef rocsparseio_enum_t rocsparseio_format_t;
typedef rocsparseio_enum_t rocsparseio_index_base_t;

/*! \ingroup types_module
 *  \brief Enumerates status.
 */
#define ROCSPARSEIO_STATUS_SUCCESS 0
#define ROCSPARSEIO_STATUS_INVALID_HANDLE 1
#define ROCSPARSEIO_STATUS_INVALID_POINTER 2
#define ROCSPARSEIO_STATUS_INVALID_VALUE 3
#define ROCSPARSEIO_STATUS_INVALID_ENUM 4
#define ROCSPARSEIO_STATUS_INVALID_FILE 5
#define ROCSPARSEIO_STATUS_INVALID_FILE_OPERATION 6
#define ROCSPARSEIO_STATUS_INVALID_FORMAT 7
#define ROCSPARSEIO_STATUS_INVALID_MODE 8
#define ROCSPARSEIO_STATUS_INVALID_SIZE 9
#define ROCSPARSEIO_STATUS_INVALID_MEMORY 10

#define ROCSPARSEIO_RWMODE_READ 0
#define ROCSPARSEIO_RWMODE_WRITE 1

#define ROCSPARSEIO_ORDER_ROW 0
#define ROCSPARSEIO_ORDER_COLUMN 1

#define ROCSPARSEIO_INDEX_BASE_ZERO 0
#define ROCSPARSEIO_INDEX_BASE_ONE 1

#define ROCSPARSEIO_DIRECTION_ROW 0
#define ROCSPARSEIO_DIRECTION_COLUMN 1

#define ROCSPARSEIO_TYPE_INT32 0
#define ROCSPARSEIO_TYPE_INT64 1
#define ROCSPARSEIO_TYPE_FLOAT32 2
#define ROCSPARSEIO_TYPE_FLOAT64 3
#define ROCSPARSEIO_TYPE_COMPLEX32 4
#define ROCSPARSEIO_TYPE_COMPLEX64 5
#define ROCSPARSEIO_TYPE_INT8 6

#define ROCSPARSEIO_FORMAT_DENSE_VECTOR 0
#define ROCSPARSEIO_FORMAT_DENSE_MATRIX 1
#define ROCSPARSEIO_FORMAT_SPARSE_CSX 2
#define ROCSPARSEIO_FORMAT_SPARSE_GEBSX 3
#define ROCSPARSEIO_FORMAT_SPARSE_COO 4
#define ROCSPARSEIO_FORMAT_SPARSE_DIA 5
#define ROCSPARSEIO_FORMAT_SPARSE_ELL 6
#define ROCSPARSEIO_FORMAT_SPARSE_HYB 7
#define ROCSPARSEIO_FORMAT_SPARSE_MCSX 8

typedef enum rocsparseio_index_base_
{
    rocsparseio_index_base_zero = 0,
    rocsparseio_index_base_one  = 1
} rocsparseio_index_base;

typedef enum rocsparseio_rwmode_
{
    rocsparseio_rwmode_read  = ROCSPARSEIO_RWMODE_READ,
    rocsparseio_rwmode_write = ROCSPARSEIO_RWMODE_WRITE
} rocsparseio_rwmode;

typedef enum rocsparseio_type_
{
    rocsparseio_type_int32     = ROCSPARSEIO_TYPE_INT32,
    rocsparseio_type_int64     = ROCSPARSEIO_TYPE_INT64,
    rocsparseio_type_float32   = ROCSPARSEIO_TYPE_FLOAT32,
    rocsparseio_type_float64   = ROCSPARSEIO_TYPE_FLOAT64,
    rocsparseio_type_complex32 = ROCSPARSEIO_TYPE_COMPLEX32,
    rocsparseio_type_complex64 = ROCSPARSEIO_TYPE_COMPLEX64,
    rocsparseio_type_int8      = ROCSPARSEIO_TYPE_INT8
} rocsparseio_type;

typedef enum rocsparseio_format_
{
    rocsparseio_format_dense_vector = ROCSPARSEIO_FORMAT_DENSE_VECTOR,
    rocsparseio_format_dense_matrix = ROCSPARSEIO_FORMAT_DENSE_MATRIX,
    rocsparseio_format_sparse_csx   = ROCSPARSEIO_FORMAT_SPARSE_CSX,
    rocsparseio_format_sparse_gebsx = ROCSPARSEIO_FORMAT_SPARSE_GEBSX,
    rocsparseio_format_sparse_coo   = ROCSPARSEIO_FORMAT_SPARSE_COO,
    rocsparseio_format_sparse_dia   = ROCSPARSEIO_FORMAT_SPARSE_DIA,
    rocsparseio_format_sparse_ell   = ROCSPARSEIO_FORMAT_SPARSE_ELL,
    rocsparseio_format_sparse_hyb   = ROCSPARSEIO_FORMAT_SPARSE_HYB,
    rocsparseio_format_sparse_mcsx  = ROCSPARSEIO_FORMAT_SPARSE_MCSX
} rocsparseio_format;

typedef enum rocsparseio_direction_
{
    rocsparseio_direction_row    = ROCSPARSEIO_DIRECTION_ROW,
    rocsparseio_direction_column = ROCSPARSEIO_DIRECTION_COLUMN
} rocsparseio_direction;

typedef enum rocsparseio_order_
{
    rocsparseio_order_row    = 0,
    rocsparseio_order_column = 1
} rocsparseio_order;

/*! \ingroup types_module
 *  \brief Enumerates status.
 */
typedef enum rocsparseio_status_
{
    rocsparseio_status_success                = ROCSPARSEIO_STATUS_SUCCESS,
    rocsparseio_status_invalid_handle         = ROCSPARSEIO_STATUS_INVALID_HANDLE,
    rocsparseio_status_invalid_pointer        = ROCSPARSEIO_STATUS_INVALID_POINTER,
    rocsparseio_status_invalid_value          = ROCSPARSEIO_STATUS_INVALID_VALUE,
    rocsparseio_status_invalid_enum           = ROCSPARSEIO_STATUS_INVALID_ENUM,
    rocsparseio_status_invalid_file           = ROCSPARSEIO_STATUS_INVALID_FILE,
    rocsparseio_status_invalid_file_operation = ROCSPARSEIO_STATUS_INVALID_FILE_OPERATION,
    rocsparseio_status_invalid_format         = ROCSPARSEIO_STATUS_INVALID_FORMAT,
    rocsparseio_status_invalid_mode           = ROCSPARSEIO_STATUS_INVALID_MODE,
    rocsparseio_status_invalid_size           = ROCSPARSEIO_STATUS_INVALID_SIZE,
    rocsparseio_status_invalid_memory         = ROCSPARSEIO_STATUS_INVALID_MEMORY
} rocsparseio_status;

//
// Opaque structure.
//
typedef struct _rocsparseio_handle* rocsparseio_handle;

#ifdef __cplusplus
extern "C" {
#endif

//! @brief Open a rocsparseio file
//! @param[out] p_handle pointer to the rocsparseio handle to be created.
//! @param[in] mode indicates how to handle the file.
//! @param[in] filename flexible C-printf style file name.
//! @retval rocsparseio_status_success the operation completed successfully.
//! @retval rocsparseio_status_invalid_handle \p p_handle is a null pointer.
//! @retval rocsparseio_status_invalid_value \p mode is invalid
//! @retval rocsparseio_status_invalid_pointer \p filename is a null pointer.

rocsparseio_status rocsparseio_open(rocsparseio_handle* p_handle,
                                    rocsparseio_rwmode  mode,
                                    const char*         filename,
                                    ...);

//! @brief Close the rocsparseio file.
//! @param[in] handle pointer to the rocsparseio handle.
//! @retval rocsparseio_status_success the operation completed successfully.
//! @retval rocsparseio_status_invalid_handle \p handle is a null pointer.

rocsparseio_status rocsparseio_close(rocsparseio_handle handle);

//! @brief Get the description of a \ref rocsparseio_status.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[in] status status to get the description from.
//! @param[out] description.
//! @retval rocsparseio_status_success the operation completed successfully.
//! @retval rocsparseio_status_invalid_handle \p handle is a null pointer.
//! @retval rocsparseio_status_invalid_pointer \p description is a null pointer.
rocsparseio_status rocsparseio_status_description(rocsparseio_handle handle,
                                                  rocsparseio_status status,
                                                  rocsparseio_string description);

//! @brief Read the name of the object.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name of object.
//! @retval rocsparseio_status_success the operation completed successfully.
//! @retval rocsparseio_status_invalid_handle \p handle is a null pointer.
//! @retval rocsparseio_status_invalid_pointer \p format is a null pointer
rocsparseio_status rocsparseio_read_name(rocsparseio_handle handle, rocsparseio_string name);

//! @brief Read what kind of object is recorded.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] format of object.
//! @retval rocsparseio_status_success the operation completed successfully.
//! @retval rocsparseio_status_invalid_handle \p handle is a null pointer.
//! @retval rocsparseio_status_invalid_pointer \p format is a null pointer

rocsparseio_status rocsparseio_read_format(rocsparseio_handle handle, rocsparseio_format* format);

//! @brief Size in byte of a given \ref rocsparseio_type.
//! @param[in] type
//! @param[out] size
//! @retval rocsparseio_status_success the operation completed successfully.
//! @retval rocsparseio_status_invalid_value \p type is invalid
//! @retval rocsparseio_status_invalid_pointer \p size is a null pointer

rocsparseio_status rocsparseio_type_get_size(rocsparseio_type type, uint64_t* size);

//! @brief Write vector.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[in] data_type rocsparseio data type.
//! @param[in] data_nmemb length of the vector with respect to the data type.
//! @param[in] data values
//! @param[in] data_ld leading dimension of data.
//! @param[in] name flexible C-printf style name.
rocsparseio_status rocsparseio_write_dense_vector(rocsparseio_handle handle,
                                                  rocsparseio_type   data_type,
                                                  uint64_t           data_nmemb,
                                                  const void*        data,
                                                  uint64_t           data_ld,
                                                  const char*        name,
                                                  ...);

//! @brief Read vector.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[out] data_type rocsparseio data type.
//! @param[out] data_nmemb length of the vector with respect to the data type.
//! @param[out] data array of values internally created with malloc, the user is
//! responsible to free.
rocsparseio_status rocsparseio_read_dense_vector(rocsparseio_handle handle,
                                                 rocsparseio_type*  data_type,
                                                 uint64_t*          data_nmemb,
                                                 void**             data);

//! @brief Read metadata vector.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[out] data_type rocsparseio data type.
//! @param[out] data_nmemb length of the vector with respect to the data type.
//! @note
//! - This method is useful if the user wants to have control over the memory
//! allocation.
//! - It is expected \ref rocsparseiox_read_dense_vector to be the next
//! rocsparseio routine being called.

rocsparseio_status rocsparseiox_read_metadata_dense_vector(rocsparseio_handle handle,
                                                           rocsparseio_type*  data_type,
                                                           uint64_t*          data_nmemb);

//! @brief Read vector.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] data array of values to be filled.
//! @param[in] data_ld leading dimension of data.

rocsparseio_status
    rocsparseiox_read_dense_vector(rocsparseio_handle handle, void* data, uint64_t data_ld);

//! @brief Write dense
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[in] order indicates if memory layout is row- or column-oriented.
//! @param[in] m number of rows
//! @param[in] n number of columns
//! @param[in] data_type rocsparseio data type.
//! @param[in] data
//! @param[in] data_ld leading dimension of data.
//! @param[in] name flexible C-printf style name.

rocsparseio_status rocsparseio_write_dense_matrix(rocsparseio_handle handle,
                                                  rocsparseio_order  order,
                                                  uint64_t           m,
                                                  uint64_t           n,
                                                  rocsparseio_type   data_type,
                                                  const void*        data,
                                                  uint64_t           data_ld,
                                                  const char*        name,
                                                  ...);

//! @brief Read dense
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[out] order indicates if memory layout is row- or column-oriented.
//! @param[out] m number of rows
//! @param[out] n number of columns
//! @param[out] data_type rocsparseio data type.
//! @param[out] data array of values filled but created with malloc, the user is
//! responsible to free.
rocsparseio_status rocsparseio_read_dense_matrix(rocsparseio_handle handle,
                                                 rocsparseio_order* order,
                                                 uint64_t*          m,
                                                 uint64_t*          n,
                                                 rocsparseio_type*  data_type,
                                                 void**             data);

//! @brief Read metadata dense
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[out] order indicates if memory layout is row- or column-oriented.
//! @param[out] m number of rows
//! @param[out] n number of columns
//! @param[out] data_type rocsparseio data type.

rocsparseio_status rocsparseiox_read_metadata_dense_matrix(rocsparseio_handle handle,
                                                           rocsparseio_order* order,
                                                           uint64_t*          m,
                                                           uint64_t*          n,
                                                           rocsparseio_type*  data_type);

//! @brief Read metadata dense
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[in] data array of values to be filled.
//! @param[in] data_ld leading dimension..

rocsparseio_status
    rocsparseiox_read_dense_matrix(rocsparseio_handle handle, void* data, uint64_t data_ld);

//! @brief Write a sparse csr/csc matrix.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[in] dir indicates if the matrix is using a Compressed Sparse Row or
//! Column storage.
//! @param[in] m number of rows.
//! @param[in] n number of columns.
//! @param[in] nnz number of non-zeros.
//! @param[in] ptr_type type of elements of the array \p ptr.
//! @param[in] ptr array of offsets.
//! @param[in] ind_type type of elements of the array \p ind.
//! @param[in] ind array of column/row indices.
//! @param[in] val_type type of elements of the array \p val.
//! @param[in] val array of values.
//! @param[in] base index base used in arrays \p ptr and \p ind.
//! @param[in] name flexible C-printf style name.
//! @retval rocsparseio_status

rocsparseio_status rocsparseio_write_sparse_csx(rocsparseio_handle     handle,
                                                rocsparseio_direction  dir,
                                                uint64_t               m,
                                                uint64_t               n,
                                                uint64_t               nnz,
                                                rocsparseio_type       ptr_type,
                                                const void*            ptr,
                                                rocsparseio_type       ind_type,
                                                const void*            ind,
                                                rocsparseio_type       val_type,
                                                const void*            val,
                                                rocsparseio_index_base base,
                                                const char*            name,
                                                ...);

//! @brief Read a sparse csr/csc matrix.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[out] dir indicates if the matrix is using with a Compressed Sparse
//! Row or Column storage.
//! @param[out] m number of rows.
//! @param[out] n number of columns.
//! @param[out] nnz number of non-zeros.
//! @param[out] ptr_type type of elements of the array \p ptr.
//! @param[out] ptr array of offsets.
//! @param[out] ind_type type of elements of the array \p ind.
//! @param[out] ind array of column/row indices.
//! @param[out] val_type type of elements of the array \p val.
//! @param[out] val array of values.
//! @param[out] base index base used in arrays \p ptr and \p ind.
//! @retval rocsparseio_status
//! @note allocation is performed with standard malloc function, the user is
//! responsible to the standard free function to free the memory.

rocsparseio_status rocsparseio_read_sparse_csx(rocsparseio_handle      handle,
                                               rocsparseio_direction*  dir,
                                               uint64_t*               m,
                                               uint64_t*               n,
                                               uint64_t*               nnz,
                                               rocsparseio_type*       ptr_type,
                                               void**                  ptr,
                                               rocsparseio_type*       ind_type,
                                               void**                  ind,
                                               rocsparseio_type*       val_type,
                                               void**                  val,
                                               rocsparseio_index_base* base);

//! @brief Read metadata information of the sparse csr/csc matrix.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[out] dir indicates if the matrix is using with a Compressed Sparse
//! Row or Column storage.
//! @param[out] m number of rows.
//! @param[out] n number of columns.
//! @param[out] nnz number of non-zeros.
//! @param[out] ptr_type type of elements of the array \p ptr.
//! @param[out] ind_type type of elements of the array \p ind.
//! @param[out] val_type type of elements of the array \p val.
//! @param[out] base index base used in arrays \p ptr and \p ind.
//! @retval rocsparseio_status
rocsparseio_status rocsparseiox_read_metadata_sparse_csx(rocsparseio_handle      handle,
                                                         rocsparseio_direction*  dir,
                                                         uint64_t*               m,
                                                         uint64_t*               n,
                                                         uint64_t*               nnz,
                                                         rocsparseio_type*       ptr_type,
                                                         rocsparseio_type*       ind_type,
                                                         rocsparseio_type*       data_type,
                                                         rocsparseio_index_base* base);

//! @brief Read sparse csr/csc matrix data arrays.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[in] ptr array of offsets to fill.
//! @param[in] ind array of column/row indices to fill.
//! @param[in] val array of values to fill.
//! @retval rocsparseio_status
rocsparseio_status
    rocsparseiox_read_sparse_csx(rocsparseio_handle handle, void* ptr, void* ind, void* data);

//! @brief Write a sparse modified csr/csc matrix.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[in] dir indicates if the matrix is using a Modified Compressed Sparse Row or
//! Column storage.
//! @param[in] m number of rows.
//! @param[in] n number of columns.
//! @param[in] nnz number of non-zeros.
//! @param[in] ptr_type type of elements of the array \p ptr.
//! @param[in] ptr array of offsets.
//! @param[in] ind_type type of elements of the array \p ind.
//! @param[in] ind array of column/row indices.
//! @param[in] val_type type of elements of the array \p val.
//! @param[in] val array of values.
//! @param[in] base index base used in arrays \p ptr and \p ind.
//! @param[in] name flexible C-printf style name.
//! @retval rocsparseio_status

rocsparseio_status rocsparseio_write_sparse_mcsx(rocsparseio_handle     handle,
                                                 rocsparseio_direction  dir,
                                                 uint64_t               m,
                                                 uint64_t               n,
                                                 uint64_t               nnz,
                                                 rocsparseio_type       ptr_type,
                                                 const void*            ptr,
                                                 rocsparseio_type       ind_type,
                                                 const void*            ind,
                                                 rocsparseio_type       val_type,
                                                 const void*            val,
                                                 rocsparseio_index_base base,
                                                 const char*            name,
                                                 ...);

//! @brief Read a sparse modified csr/csc matrix.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[out] dir indicates if the matrix is using with a Modified Compressed Sparse
//! Row or Column storage.
//! @param[out] m number of rows.
//! @param[out] n number of columns.
//! @param[out] nnz number of non-zeros.
//! @param[out] ptr_type type of elements of the array \p ptr.
//! @param[out] ptr array of offsets.
//! @param[out] ind_type type of elements of the array \p ind.
//! @param[out] ind array of column/row indices.
//! @param[out] val_type type of elements of the array \p val.
//! @param[out] val array of values.
//! @param[out] base index base used in arrays \p ptr and \p ind.
//! @retval rocsparseio_status
//! @note allocation is performed with standard malloc function, the user is
//! responsible to the standard free function to free the memory.

rocsparseio_status rocsparseio_read_sparse_mcsx(rocsparseio_handle      handle,
                                                rocsparseio_direction*  dir,
                                                uint64_t*               m,
                                                uint64_t*               n,
                                                uint64_t*               nnz,
                                                rocsparseio_type*       ptr_type,
                                                void**                  ptr,
                                                rocsparseio_type*       ind_type,
                                                void**                  ind,
                                                rocsparseio_type*       val_type,
                                                void**                  val,
                                                rocsparseio_index_base* base);

//! @brief Read metadata information of the sparse modified csr/csc matrix.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[out] dir indicates if the matrix is using with a Modifed Compressed Sparse
//! Row or Column storage.
//! @param[out] m number of rows.
//! @param[out] n number of columns.
//! @param[out] nnz number of non-zeros.
//! @param[out] ptr_type type of elements of the array \p ptr.
//! @param[out] ind_type type of elements of the array \p ind.
//! @param[out] val_type type of elements of the array \p val.
//! @param[out] base index base used in arrays \p ptr and \p ind.
//! @retval rocsparseio_status
rocsparseio_status rocsparseiox_read_metadata_sparse_mcsx(rocsparseio_handle      handle,
                                                          rocsparseio_direction*  dir,
                                                          uint64_t*               m,
                                                          uint64_t*               n,
                                                          uint64_t*               nnz,
                                                          rocsparseio_type*       ptr_type,
                                                          rocsparseio_type*       ind_type,
                                                          rocsparseio_type*       data_type,
                                                          rocsparseio_index_base* base);

//! @brief Read sparse modified csr/csc matrix data arrays.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[in] ptr array of offsets to fill.
//! @param[in] ind array of column/row indices to fill.
//! @param[in] val array of values to fill.
//! @retval rocsparseio_status
rocsparseio_status
    rocsparseiox_read_sparse_mcsx(rocsparseio_handle handle, void* ptr, void* ind, void* data);

//! @brief Write a sparse gebsr/gebsc matrix.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[in] dir indicates if the matrix is using a GEneral Block Sparse Row
//! or Column storage.
//! @param[in] dirb indicates storage ordering of dense block matrices.
//! @param[in] mb number of rows.
//! @param[in] nb number of columns.
//! @param[in] nnzb number of non-zeros.
//! @param[in] row_block_dim number of rows in a single dense block.
//! @param[in] col_block_dim number of columns in a single dense block.
//! @param[in] ptr_type type of elements of the array \p ptr.
//! @param[in] ptr array of offsets.
//! @param[in] ind_type type of elements of the array \p ind.
//! @param[in] ind array of column/row indices.
//! @param[in] val_type type of elements of the array \p val.
//! @param[in] val array of values.
//! @param[in] base index base used in arrays \p ptr and \p ind.
//! @param[in] name flexible C-printf style name.
//! @retval rocsparseio_status
rocsparseio_status rocsparseio_write_sparse_gebsx(rocsparseio_handle     handle,
                                                  rocsparseio_direction  dir,
                                                  rocsparseio_direction  dirb,
                                                  uint64_t               mb,
                                                  uint64_t               nb,
                                                  uint64_t               nnzb,
                                                  uint64_t               row_block_dim,
                                                  uint64_t               col_block_dim,
                                                  rocsparseio_type       ptr_type,
                                                  const void*            ptr,
                                                  rocsparseio_type       ind_type,
                                                  const void*            ind,
                                                  rocsparseio_type       val_type,
                                                  const void*            val,
                                                  rocsparseio_index_base base,
                                                  const char*            name,
                                                  ...);

//! @brief Read a sparse gebsr/gebsc matrix.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[out] dir indicates if the matrix is using a GEneral Block Sparse Row
//! or Column storage.
//! @param[out] dirb indicates storage ordering of dense block matrices.
//! @param[out] mb number of rows.
//! @param[out] nb number of columns.
//! @param[out] nnzb number of non-zeros.
//! @param[out] row_block_dim number of rows in a single dense block.
//! @param[out] col_block_dim number of columns in a single dense block.
//! @param[out] ptr_type type of elements of the array \p ptr.
//! @param[out] ptr array of offsets.
//! @param[out] ind_type type of elements of the array \p ind.
//! @param[out] ind array of column/row indices.
//! @param[out] val_type type of elements of the array \p val.
//! @param[out] val array of values.
//! @param[out] base index base used in arrays \p ptr and \p ind.
//! @retval rocsparseio_status
//! @note allocation is performed with standard malloc function, the user is
//! responsible to the standard free function to free the memory.
rocsparseio_status rocsparseio_read_sparse_gebsx(rocsparseio_handle      handle,
                                                 rocsparseio_direction*  dir,
                                                 rocsparseio_direction*  dirb,
                                                 uint64_t*               mb,
                                                 uint64_t*               nb,
                                                 uint64_t*               nnzb,
                                                 uint64_t*               row_block_dim,
                                                 uint64_t*               col_block_dim,
                                                 rocsparseio_type*       ptr_type,
                                                 void**                  ptr,
                                                 rocsparseio_type*       ind_type,
                                                 void**                  ind,
                                                 rocsparseio_type*       val_type,
                                                 void**                  val,
                                                 rocsparseio_index_base* base);

//! @brief Read metadata information of a sparse gebsr/gebsc matrix.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[out] dir indicates if the matrix is using a GEneral Block Sparse Row
//! or Column storage.
//! @param[out] dirb indicates storage ordering of dense block matrices.
//! @param[out] mb number of rows.
//! @param[out] nb number of columns.
//! @param[out] nnzb number of non-zeros.
//! @param[out] row_block_dim number of rows in a single dense block.
//! @param[out] col_block_dim number of columns in a single dense block.
//! @param[out] ptr_type type of elements of the array \p ptr.
//! @param[out] ind_type type of elements of the array \p ind.
//! @param[out] val_type type of elements of the array \p val.
//! @param[out] base index base used in arrays \p ptr and \p ind.
//! @retval rocsparseio_status
rocsparseio_status rocsparseiox_read_metadata_sparse_gebsx(rocsparseio_handle      handle,
                                                           rocsparseio_direction*  dir,
                                                           rocsparseio_direction*  dirb,
                                                           uint64_t*               mb,
                                                           uint64_t*               nb,
                                                           uint64_t*               nnzb,
                                                           uint64_t*               row_block_dim,
                                                           uint64_t*               col_block_dim,
                                                           rocsparseio_type*       ptr_type,
                                                           rocsparseio_type*       ind_type,
                                                           rocsparseio_type*       val_type,
                                                           rocsparseio_index_base* base);

//! @brief Read a sparse gebsr/gebsc matrix.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] ptr array of offsets.
//! @param[out] ind array of column/row indices.
//! @param[out] val array of values.
//! @retval rocsparseio_status
rocsparseio_status
    rocsparseiox_read_sparse_gebsx(rocsparseio_handle handle, void* ptr, void* ind, void* val);

//! @brief Write a sparse matrix with coordinates format.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[in] m number of rows.
//! @param[in] n number of columns.
//! @param[in] nnz number of non-zeros.
//! @param[in] row_ind_type type of elements of the array \p row_ind.
//! @param[in] row_ind array of row indices.
//! @param[in] col_ind_type type of elements of the array \p col_ind.
//! @param[in] col_ind array of column indices.
//! @param[in] val_type type of elements of the array \p val.
//! @param[in] val array of values.
//! @param[in] base index base used in arrays \p row_ind and \p col_ind.
//! @param[in] name flexible C-printf style name.
//! @retval rocsparseio_status

rocsparseio_status rocsparseio_write_sparse_coo(rocsparseio_handle     handle,
                                                uint64_t               m,
                                                uint64_t               n,
                                                uint64_t               nnz,
                                                rocsparseio_type       row_ind_type,
                                                const void*            row_ind,
                                                rocsparseio_type       col_ind_type,
                                                const void*            col_ind,
                                                rocsparseio_type       val_type,
                                                const void*            val,
                                                rocsparseio_index_base base,
                                                const char*            name,
                                                ...);

//! @brief Read a sparse matrix with coordinates format.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[out] m number of rows.
//! @param[out] n number of columns.
//! @param[out] nnz number of non-zeros.
//! @param[out] row_ind_type type of elements of the array \p row_ind.
//! @param[out] row_ind array of offsets.
//! @param[out] col_ind_type type of elements of the array \p col_ind.
//! @param[out] col_ind array of column/row indices.
//! @param[out] val_type type of elements of the array \p val.
//! @param[out] val array of values.
//! @param[out] base index base used in arrays \p row_ind and \p col_ind.
//! @retval rocsparseio_status
//! @note allocation is performed with standard malloc function, the user is
//! responsible to the standard free function to free the memory.
rocsparseio_status rocsparseio_read_sparse_coo(rocsparseio_handle      handle,
                                               uint64_t*               m,
                                               uint64_t*               n,
                                               uint64_t*               nnz,
                                               rocsparseio_type*       row_ind_type,
                                               void**                  row_ind,
                                               rocsparseio_type*       col_ind_type,
                                               void**                  col_ind,
                                               rocsparseio_type*       val_type,
                                               void**                  val,
                                               rocsparseio_index_base* base);

//! @brief Read metadata information of a sparse matrix with coordinates format.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[out] m number of rows.
//! @param[out] n number of columns.
//! @param[out] nnz number of non-zeros.
//! @param[out] row_ind_type type of elements of the array of row indices.
//! @param[out] col_ind_type type of elements of the array of column indices.
//! @param[out] val_type type of elements of the array.
//! @param[out] base index base used in arrays row and column indices.
//! @retval rocsparseio_status
rocsparseio_status rocsparseiox_read_metadata_sparse_coo(rocsparseio_handle      handle,
                                                         uint64_t*               m,
                                                         uint64_t*               n,
                                                         uint64_t*               nnz,
                                                         rocsparseio_type*       row_ind_type,
                                                         rocsparseio_type*       col_ind_type,
                                                         rocsparseio_type*       val_type,
                                                         rocsparseio_index_base* base);

//! @brief Read a sparse matrix with coordinates format.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] row_ind array of row indices.
//! @param[out] col_ind array of column indices.
//! @param[out] val     array of values.
//! @retval rocsparseio_status
rocsparseio_status rocsparseiox_read_sparse_coo(rocsparseio_handle handle,
                                                void*              row_ind,
                                                void*              col_ind,
                                                void*              val);

//! @brief Write a sparse ell matrix.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[in] m number of rows.
//! @param[in] n number of columns.
//! @param[in] width width of the matrix.
//! @param[in] ind_type type of elements of the array \p ind.
//! @param[in] ind array of column/row indices.
//! @param[in] val_type type of elements of the array \p val.
//! @param[in] val array of values.
//! @param[in] base index base used in arrays and \p ind.
//! @param[in] name flexible C-printf style name.
//! @retval rocsparseio_status
rocsparseio_status rocsparseio_write_sparse_ell(rocsparseio_handle     handle,
                                                uint64_t               m,
                                                uint64_t               n,
                                                uint64_t               width,
                                                rocsparseio_type       ind_type,
                                                const void*            ind,
                                                rocsparseio_type       val_type,
                                                const void*            val,
                                                rocsparseio_index_base base,
                                                const char*            name,
                                                ...);

//! @brief Read a sparse ell matrix.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[out] m number of rows.
//! @param[out] n number of columns.
//! @param[out] width width of the matrix.
//! @param[out] ind_type type of elements of the array \p ind.
//! @param[out] ind array of column/row indices.
//! @param[out] val_type type of elements of the array \p val.
//! @param[out] val array of values.
//! @param[out] base index base used in arrays \p ptr and \p ind.
//! @retval rocsparseio_status
//! @note allocation is performed with standard malloc function, the user is
//! responsible to the standard free function to free the memory.
rocsparseio_status rocsparseio_read_sparse_ell(rocsparseio_handle      handle,
                                               uint64_t*               m,
                                               uint64_t*               n,
                                               uint64_t*               width,
                                               rocsparseio_type*       ind_type,
                                               void**                  ind,
                                               rocsparseio_type*       val_type,
                                               void**                  val,
                                               rocsparseio_index_base* base);

//! @brief Read metadata information of a sparse ell matrix.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[out] m number of rows.
//! @param[out] n number of columns.
//! @param[out] width witdh of the matrix.
//! @param[out] ind_type type of elements of the array \p ind.
//! @param[out] val_type type of elements of the array \p val.
//! @param[out] base index base used in arrays \p ptr and \p ind.
//! @retval rocsparseio_status
rocsparseio_status rocsparseiox_read_metadata_sparse_ell(rocsparseio_handle      handle,
                                                         uint64_t*               m,
                                                         uint64_t*               n,
                                                         uint64_t*               width,
                                                         rocsparseio_type*       ind_type,
                                                         rocsparseio_type*       val_type,
                                                         rocsparseio_index_base* base);

//! @brief Read a sparse ell matrix.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] ind array of column indices.
//! @param[out] val array of values.
//! @retval rocsparseio_status
rocsparseio_status rocsparseiox_read_sparse_ell(rocsparseio_handle handle, void* ind, void* val);

//! @brief Write a sparse dia matrix.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[in] m number of rows.
//! @param[in] n number of columns.
//! @param[in] ndiag number of diagonals.
//! @param[in] ind_type type of elements of the array \p ind.
//! @param[in] ind array of diagonal indices.
//! @param[in] val_type type of elements of the array \p val.
//! @param[in] val array of values.
//! @param[in] base index base used in arrays and \p ind.
//! @param[in] name flexible C-printf style name.
//! @retval rocsparseio_status
rocsparseio_status rocsparseio_write_sparse_dia(rocsparseio_handle     handle,
                                                uint64_t               m,
                                                uint64_t               n,
                                                uint64_t               ndiag,
                                                rocsparseio_type       ind_type,
                                                const void*            ind,
                                                rocsparseio_type       val_type,
                                                const void*            val,
                                                rocsparseio_index_base base,
                                                const char*            name,
                                                ...);

//! @brief Read a sparse dia matrix.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[out] m number of rows.
//! @param[out] n number of columns.
//! @param[out] ndiag number of diagonals.
//! @param[out] ind_type type of elements of the array \p ind.
//! @param[out] ind array of diagonal indices.
//! @param[out] val_type type of elements of the array \p val.
//! @param[out] val array of values.
//! @param[out] base index base used in arrays \p ptr and \p ind.
//! @retval rocsparseio_status
//! @note allocation is performed with standard malloc function, the user is
//! responsible to the standard free function to free the memory.
rocsparseio_status rocsparseio_read_sparse_dia(rocsparseio_handle      handle,
                                               uint64_t*               m,
                                               uint64_t*               n,
                                               uint64_t*               ndiag,
                                               rocsparseio_type*       ind_type,
                                               void**                  ind,
                                               rocsparseio_type*       val_type,
                                               void**                  val,
                                               rocsparseio_index_base* base);

//! @brief Read metadata information of a sparse dia matrix.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[out] m number of rows.
//! @param[out] n number of columns.
//! @param[in] ndiag number of diagonals.
//! @param[out] ind_type type of elements of the array \p ind.
//! @param[out] val_type type of elements of the array \p val.
//! @param[out] base index base used in arrays \p ptr and \p ind.
//! @retval rocsparseio_status
rocsparseio_status rocsparseiox_read_metadata_sparse_dia(rocsparseio_handle      handle,
                                                         uint64_t*               m,
                                                         uint64_t*               n,
                                                         uint64_t*               ndiag,
                                                         rocsparseio_type*       ind_type,
                                                         rocsparseio_type*       val_type,
                                                         rocsparseio_index_base* base);

//! @brief Read a sparse ell matrix.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] ind array of diagonal indices.
//! @param[out] val array of values.
//! @retval rocsparseio_status
rocsparseio_status rocsparseiox_read_sparse_dia(rocsparseio_handle handle, void* ind, void* val);

//! @brief Write a sparse matrix with hyb format.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[in] m number of rows.
//! @param[in] n number of columns.
//! @param[in] coo_nnz number of non-zeros.
//! @param[in] coo_row_ind_type type of elements of the array \p row_ind.
//! @param[in] coo_row_ind array of row indices.
//! @param[in] coo_col_ind_type type of elements of the array \p col_ind.
//! @param[in] coo_col_ind array of column indices.
//! @param[in] coo_val_type type of elements of the array \p val.
//! @param[in] coo_val array of values.
//! @param[in] coo_base index base used in arrays \p row_ind and \p col_ind.
//! @param[in] ell_width width of the matrix.
//! @param[in] ell_ind_type type of elements of the array \p ind.
//! @param[in] ell_ind array of column/row indices.
//! @param[in] ell_val_type type of elements of the array \p val.
//! @param[in] ell_val array of values.
//! @param[in] ell_base index base used in arrays and \p ind.
//! @param[in] name flexible C-printf style name.
//! @retval rocsparseio_status
rocsparseio_status rocsparseio_write_sparse_hyb(rocsparseio_handle     handle,
                                                uint64_t               m,
                                                uint64_t               n,
                                                uint64_t               coo_nnz,
                                                rocsparseio_type       coo_row_ind_type,
                                                const void*            coo_row_ind,
                                                rocsparseio_type       coo_col_ind_type,
                                                const void*            coo_col_ind,
                                                rocsparseio_type       coo_val_type,
                                                const void*            coo_val,
                                                rocsparseio_index_base coo_base,

                                                uint64_t               ell_width,
                                                rocsparseio_type       ell_col_ind_type,
                                                const void*            ell_col_ind,
                                                rocsparseio_type       ell_val_type,
                                                const void*            ell_val,
                                                rocsparseio_index_base ell_base,
                                                const char*            name,
                                                ...);

//! @brief Read a sparse matrix with coordinates format.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[out] m number of rows.
//! @param[out] n number of columns.
//! @param[out] coo_nnz number of non-zeros.
//! @param[out] coo_row_ind_type type of elements of the array \p row_ind.
//! @param[out] coo_row_ind array of offsets.
//! @param[out] coo_col_ind_type type of elements of the array \p col_ind.
//! @param[out] coo_col_ind array of column/row indices.
//! @param[out] coo_val_type type of elements of the array \p val.
//! @param[out] coo_val array of values.
//! @param[out] coo_base index base used in arrays \p row_ind and \p col_ind.
//! @param[out] ell_width width of the matrix.
//! @param[out] ell_ind_type type of elements of the array \p ind.
//! @param[out] ell_ind array of column/row indices.
//! @param[out] ell_val_type type of elements of the array \p val.
//! @param[out] ell_val array of values.
//! @param[out] ell_base index base used in arrays \p ptr and \p ind.
//! @retval rocsparseio_status
//! @note allocation is performed with standard malloc function, the user is
//! responsible to the standard free function to free the memory.
rocsparseio_status rocsparseio_read_sparse_hyb(rocsparseio_handle      handle,
                                               uint64_t*               m,
                                               uint64_t*               n,
                                               uint64_t*               coo_nnz,
                                               rocsparseio_type*       coo_row_ind_type,
                                               void**                  coo_row_ind,
                                               rocsparseio_type*       coo_col_ind_type,
                                               void**                  coo_col_ind,
                                               rocsparseio_type*       coo_val_type,
                                               void**                  coo_val,
                                               rocsparseio_index_base* coo_base,
                                               uint64_t*               ell_width,
                                               rocsparseio_type*       ell_col_ind_type,
                                               void**                  ell_col_ind,
                                               rocsparseio_type*       ell_val_type,
                                               void**                  ell_val,
                                               rocsparseio_index_base* ell_base);

//! @brief Read metadata information of a sparse matrix with coordinates format.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] name name of the matrix.
//! @param[out] m number of rows.
//! @param[out] n number of columns.
//! @param[out] coo_nnz number of non-zeros.
//! @param[out] coo_row_ind_type type of elements of the array of row indices.
//! @param[out] coo_col_ind_type type of elements of the array of column indices.
//! @param[out] coo_val_type type of elements of the array.
//! @param[out] coo_base index base used in arrays row and column indices.
//! @param[out] ell_width witdh of the matrix.
//! @param[out] ell_ind_type type of elements of the array \p ind.
//! @param[out] ell_val_type type of elements of the array \p val.
//! @param[out] ell_base index base used in arrays \p ptr and \p ind.
//! @retval rocsparseio_status
rocsparseio_status rocsparseiox_read_metadata_sparse_hyb(rocsparseio_handle      handle,
                                                         uint64_t*               m,
                                                         uint64_t*               n,
                                                         uint64_t*               coo_nnz,
                                                         rocsparseio_type*       coo_row_ind_type,
                                                         rocsparseio_type*       coo_col_ind_type,
                                                         rocsparseio_type*       coo_val_type,
                                                         rocsparseio_index_base* coo_base,
                                                         uint64_t*               ell_width,
                                                         rocsparseio_type*       ell_col_ind_type,
                                                         rocsparseio_type*       ell_val_type,
                                                         rocsparseio_index_base* ell_base);

//! @brief Read a sparse matrix with coordinates format.
//! @param[in] handle pointer to the rocsparseio handle.
//! @param[out] coo_row_ind array of row indices.
//! @param[out] coo_col_ind array of column indices.
//! @param[out] coo_val     array of values.
//! @param[out] ell_col_ind array of column indices.
//! @param[out] ell_val     array of values.
//! @retval rocsparseio_status
rocsparseio_status rocsparseiox_read_sparse_hyb(rocsparseio_handle handle,
                                                void*              coo_row_ind,
                                                void*              coo_col_ind,
                                                void*              coo_val,
                                                void*              ell_col_ind,
                                                void*              ell_val);

#ifdef __cplusplus
}
#endif

#endif /* _ROCSPARSEIO_H_ */
