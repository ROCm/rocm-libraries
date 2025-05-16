/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCBLAS_AUXILIARY_H
#define ROCBLAS_AUXILIARY_H

#include "rocblas-export.h"
#include "rocblas-macros.h"
#include "rocblas-types.h"

/*!\file
 * \brief rocblas-auxiliary.h provides auxilary functions in rocblas
 */

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Create handle
 */
ROCBLAS_EXPORT rocblas_status rocblas_create_handle(rocblas_handle* handle);

/*! \brief Destroy handle
 */
ROCBLAS_EXPORT rocblas_status rocblas_destroy_handle(rocblas_handle handle);

/*! \brief Set stream for handle
 */
ROCBLAS_EXPORT rocblas_status rocblas_set_stream(rocblas_handle handle, hipStream_t stream);

/*! \brief Get stream [0] from handle
 */
ROCBLAS_EXPORT rocblas_status rocblas_get_stream(rocblas_handle handle, hipStream_t* stream);

/*! \brief Set rocblas_pointer_mode
 */
ROCBLAS_EXPORT rocblas_status rocblas_set_pointer_mode(rocblas_handle       handle,
                                                       rocblas_pointer_mode pointer_mode);
/*! \brief Get rocblas_pointer_mode
 */
ROCBLAS_EXPORT rocblas_status rocblas_get_pointer_mode(rocblas_handle        handle,
                                                       rocblas_pointer_mode* pointer_mode);

/*! \brief Set rocblas_atomics_mode
 *  \details
 *  Some rocBLAS functions may have implementations which use atomic operations to increase performance.
 *  By using atomic operations, results are not guaranteed to be identical between multiple runs.
 *  Results will be accurate with or without atomic operations. Atomic operations in rocBLAS are turned
 *  off by default. They can be turned on or off for a handle by calling rocblas_set_atomics_mode.
 */
ROCBLAS_EXPORT rocblas_status rocblas_set_atomics_mode(rocblas_handle       handle,
                                                       rocblas_atomics_mode atomics_mode);

/*! \brief Get rocblas_atomics_mode
 */
ROCBLAS_EXPORT rocblas_status rocblas_get_atomics_mode(rocblas_handle        handle,
                                                       rocblas_atomics_mode* atomics_mode);

/*! \brief Set rocblas_math_mode
 */
ROCBLAS_EXPORT rocblas_status rocblas_set_math_mode(rocblas_handle    handle,
                                                    rocblas_math_mode math_mode);

/*! \brief Get rocblas_math_mode
 */
ROCBLAS_EXPORT rocblas_status rocblas_get_math_mode(rocblas_handle     handle,
                                                    rocblas_math_mode* math_mode);

/*! \brief  Indicates whether the pointer is on the host or device.
 */
ROCBLAS_EXPORT rocblas_pointer_mode rocblas_pointer_to_mode(void* ptr);

/*! \brief Copy vector from host to device
    @param[in]
    n           [rocblas_int]
                number of elements in the vector
    @param[in]
    elem_size   [rocblas_int]
                number of bytes per element in the matrix
    @param[in]
    x           pointer to vector on the host
    @param[in]
    incx        [rocblas_int]
                specifies the increment for the elements of the vector
    @param[out]
    y           pointer to vector on the device
    @param[in]
    incy        [rocblas_int]
                specifies the increment for the elements of the vector
 */
ROCBLAS_EXPORT rocblas_status rocblas_set_vector(rocblas_int n,
                                                 rocblas_int elem_size,
                                                 const void* x,
                                                 rocblas_int incx,
                                                 void*       y,
                                                 rocblas_int incy);

ROCBLAS_EXPORT rocblas_status rocblas_set_vector_64(
    int64_t n, int64_t elem_size, const void* x, int64_t incx, void* y, int64_t incy);

/*! \brief Copy vector from device to host
    @param[in]
    n           [rocblas_int]
                number of elements in the vector
    @param[in]
    elem_size   [rocblas_int]
                number of bytes per element in the matrix
    @param[in]
    x           pointer to vector on the device
    @param[in]
    incx        [rocblas_int]
                specifies the increment for the elements of the vector
    @param[out]
    y           pointer to vector on the host
    @param[in]
    incy        [rocblas_int]
                specifies the increment for the elements of the vector
 */
ROCBLAS_EXPORT rocblas_status rocblas_get_vector(rocblas_int n,
                                                 rocblas_int elem_size,
                                                 const void* x,
                                                 rocblas_int incx,
                                                 void*       y,
                                                 rocblas_int incy);

ROCBLAS_EXPORT rocblas_status rocblas_get_vector_64(
    int64_t n, int64_t elem_size, const void* x, int64_t incx, void* y, int64_t incy);

/*! \brief Copy matrix from host to device
    @param[in]
    rows        [rocblas_int]
                number of rows in matrices
    @param[in]
    cols        [rocblas_int]
                number of columns in matrices
    @param[in]
    elem_size   [rocblas_int]
                number of bytes per element in the matrix
    @param[in]
    a           pointer to matrix on the host
    @param[in]
    lda         [rocblas_int]
                specifies the leading dimension of A, lda >= rows
    @param[out]
    b           pointer to matrix on the GPU
    @param[in]
    ldb         [rocblas_int]
                specifies the leading dimension of B, ldb >= rows
 */
ROCBLAS_EXPORT rocblas_status rocblas_set_matrix(rocblas_int rows,
                                                 rocblas_int cols,
                                                 rocblas_int elem_size,
                                                 const void* a,
                                                 rocblas_int lda,
                                                 void*       b,
                                                 rocblas_int ldb);

ROCBLAS_EXPORT rocblas_status rocblas_set_matrix_64(int64_t     rows,
                                                    int64_t     cols,
                                                    int64_t     elem_size,
                                                    const void* a,
                                                    int64_t     lda,
                                                    void*       b,
                                                    int64_t     ldb);

/*! \brief Copy matrix from device to host
    @param[in]
    rows        [rocblas_int]
                number of rows in matrices
    @param[in]
    cols        [rocblas_int]
                number of columns in matrices
    @param[in]
    elem_size   [rocblas_int]
                number of bytes per element in the matrix
    @param[in]
    a           pointer to matrix on the GPU
    @param[in]
    lda         [rocblas_int]
                specifies the leading dimension of A, lda >= rows
    @param[out]
    b           pointer to matrix on the host
    @param[in]
    ldb         [rocblas_int]
                specifies the leading dimension of B, ldb >= rows
 */
ROCBLAS_EXPORT rocblas_status rocblas_get_matrix(rocblas_int rows,
                                                 rocblas_int cols,
                                                 rocblas_int elem_size,
                                                 const void* a,
                                                 rocblas_int lda,
                                                 void*       b,
                                                 rocblas_int ldb);

ROCBLAS_EXPORT rocblas_status rocblas_get_matrix_64(int64_t     rows,
                                                    int64_t     cols,
                                                    int64_t     elem_size,
                                                    const void* a,
                                                    int64_t     lda,
                                                    void*       b,
                                                    int64_t     ldb);

/*! \brief Asynchronously copy vector from host to device
     \details
    rocblas_set_vector_async copies a vector from pinned host memory to device memory asynchronously.
    Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
    @param[in]
    n           [rocblas_int]
                number of elements in the vector
    @param[in]
    elem_size   [rocblas_int]
                number of bytes per element in the matrix
    @param[in]
    x           pointer to vector on the host
    @param[in]
    incx        [rocblas_int]
                specifies the increment for the elements of the vector
    @param[out]
    y           pointer to vector on the device
    @param[in]
    incy        [rocblas_int]
                specifies the increment for the elements of the vector
    @param[in]
    stream      specifies the stream into which this transfer request is queued
     ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_set_vector_async(rocblas_int n,
                                                       rocblas_int elem_size,
                                                       const void* x,
                                                       rocblas_int incx,
                                                       void*       y,
                                                       rocblas_int incy,
                                                       hipStream_t stream);

ROCBLAS_EXPORT rocblas_status rocblas_set_vector_async_64(int64_t     n,
                                                          int64_t     elem_size,
                                                          const void* x,
                                                          int64_t     incx,
                                                          void*       y,
                                                          int64_t     incy,
                                                          hipStream_t stream);

/*! \brief Asynchronously copy vector from device to host
     \details
    rocblas_get_vector_async copies a vector from pinned host memory to device memory asynchronously.
    Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
    @param[in]
    n           [rocblas_int]
                number of elements in the vector
    @param[in]
    elem_size   [rocblas_int]
                number of bytes per element in the matrix
    @param[in]
    x           pointer to vector on the device
    @param[in]
    incx        [rocblas_int]
                specifies the increment for the elements of the vector
    @param[out]
    y           pointer to vector on the host
    @param[in]
    incy        [rocblas_int]
                specifies the increment for the elements of the vector
    @param[in]
    stream      specifies the stream into which this transfer request is queued
     ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_get_vector_async(rocblas_int n,
                                                       rocblas_int elem_size,
                                                       const void* x,
                                                       rocblas_int incx,
                                                       void*       y,
                                                       rocblas_int incy,
                                                       hipStream_t stream);

ROCBLAS_EXPORT rocblas_status rocblas_get_vector_async_64(int64_t     n,
                                                          int64_t     elem_size,
                                                          const void* x,
                                                          int64_t     incx,
                                                          void*       y,
                                                          int64_t     incy,
                                                          hipStream_t stream);

/*! \brief Asynchronously copy matrix from host to device
     \details
    rocblas_set_matrix_async copies a matrix from pinned host memory to device memory asynchronously.
    Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
    @param[in]
    rows        [rocblas_int]
                number of rows in matrices
    @param[in]
    cols        [rocblas_int]
                number of columns in matrices
    @param[in]
    elem_size   [rocblas_int]
                number of bytes per element in the matrix
    @param[in]
    a           pointer to matrix on the host
    @param[in]
    lda         [rocblas_int]
                specifies the leading dimension of A, lda >= rows
    @param[out]
    b           pointer to matrix on the GPU
    @param[in]
    ldb         [rocblas_int]
                specifies the leading dimension of B, ldb >= rows
    @param[in]
    stream      specifies the stream into which this transfer request is queued
     ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_set_matrix_async(rocblas_int rows,
                                                       rocblas_int cols,
                                                       rocblas_int elem_size,
                                                       const void* a,
                                                       rocblas_int lda,
                                                       void*       b,
                                                       rocblas_int ldb,
                                                       hipStream_t stream);

ROCBLAS_EXPORT rocblas_status rocblas_set_matrix_async_64(int64_t     rows,
                                                          int64_t     cols,
                                                          int64_t     elem_size,
                                                          const void* a,
                                                          int64_t     lda,
                                                          void*       b,
                                                          int64_t     ldb,
                                                          hipStream_t stream);

/*! \brief asynchronously copy matrix from device to host
     \details
    rocblas_get_matrix_async copies a matrix from device memory to pinned host memory asynchronously.
    Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
    @param[in]
    rows        [rocblas_int]
                number of rows in matrices
    @param[in]
    cols        [rocblas_int]
                number of columns in matrices
    @param[in]
    elem_size   [rocblas_int]
                number of bytes per element in the matrix
    @param[in]
    a           pointer to matrix on the GPU
    @param[in]
    lda         [rocblas_int]
                specifies the leading dimension of A, lda >= rows
    @param[out]
    b           pointer to matrix on the host
    @param[in]
    ldb         [rocblas_int]
                specifies the leading dimension of B, ldb >= rows
    @param[in]
    stream      specifies the stream into which this transfer request is queued
     ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_get_matrix_async(rocblas_int rows,
                                                       rocblas_int cols,
                                                       rocblas_int elem_size,
                                                       const void* a,
                                                       rocblas_int lda,
                                                       void*       b,
                                                       rocblas_int ldb,
                                                       hipStream_t stream);

ROCBLAS_EXPORT rocblas_status rocblas_get_matrix_async_64(int64_t     rows,
                                                          int64_t     cols,
                                                          int64_t     elem_size,
                                                          const void* a,
                                                          int64_t     lda,
                                                          void*       b,
                                                          int64_t     ldb,
                                                          hipStream_t stream);

/*******************************************************************************
 * Function to set start/stop event handlers (for internal use only)
 ******************************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_set_start_stop_events(rocblas_handle handle,
                                                            hipEvent_t     startEvent,
                                                            hipEvent_t     stopEvent);
#define ROCBLAS_INVOKE_START_STOP_EVENTS(handle, startEvent, stopEvent, call) \
    do                                                                        \
    {                                                                         \
        rocblas_handle tmp_h = (handle);                                      \
        rocblas_set_start_stop_events(tmp_h, (startEvent), (stopEvent));      \
        call;                                                                 \
        rocblas_set_start_stop_events(tmp_h, (hipEvent_t)0, (hipEvent_t)0);   \
    } while(0)

// For testing solution selection fitness -- for internal testing only
ROCBLAS_EXPORT rocblas_status rocblas_set_solution_fitness_query(rocblas_handle handle,
                                                                 double*        fitness);

/*! \brief specifies the performance metric that solution selection uses
     \details
    Determines which performance metric will be used by Tensile when selecting the optimal solution
    for gemm problems. If a valid solution benchmarked for this performance metric does not exist
    for a problem, Tensile will default to a solution benchmarked for overall performance instead.
    @param[in]
    handle      [rocblas_handle]
                the handle of device
    @param[in]
    metric      [rocblas_performance_metric]
                the performance metric to be used
     ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_set_performance_metric(rocblas_handle             handle,
                                                             rocblas_performance_metric metric);
/*! \brief returns the performance metric being used for solution selection
     \details
    Returns the performance metric used by Tensile to select the optimal solution for gemm problems.
    @param[in]
    handle      [rocblas_handle]
                the handle of device
    @param[out]
    metric      [rocblas_performance_metric*]
                pointer to where the metric will be stored
     ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_get_performance_metric(rocblas_handle              handle,
                                                             rocblas_performance_metric* metric);

#ifdef __cplusplus
}
#endif

#endif /* ROCBLAS_AUXILIARY_H */
