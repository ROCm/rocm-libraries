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
 *
 * ************************************************************************ */
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas-auxiliary.h"
#include <cctype>
#include <cstdlib>
#include <memory>
#include <string>

/* ============================================================================================ */

/*******************************************************************************
 * ! \brief  indicates whether the pointer is on the host or device.
 * currently HIP API can only recoginize the input ptr on deive or not
 *  can not recoginize it is on host or not
 ******************************************************************************/
extern "C" rocblas_pointer_mode rocblas_pointer_to_mode(void* ptr)
{
    hipPointerAttribute_t attribute;
    hipPointerGetAttributes(&attribute, ptr);
    return ptr == attribute.devicePointer ? rocblas_pointer_mode_device : rocblas_pointer_mode_host;
}

/*******************************************************************************
 * ! \brief get pointer mode, can be host or device
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_pointer_mode(rocblas_handle        handle,
                                                   rocblas_pointer_mode* mode)
try
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;
    *mode = handle->pointer_mode;
    rocblas_internal_logger logger;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        logger.log_trace(handle, "rocblas_get_pointer_mode", *mode);
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * ! \brief set pointer mode to host or device
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_pointer_mode(rocblas_handle handle, rocblas_pointer_mode mode)
try
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;
    rocblas_internal_logger logger;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
    {
        logger.log_trace(handle, "rocblas_set_pointer_mode", mode);
    }
    handle->pointer_mode = mode;
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * ! \brief get atomics mode
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_atomics_mode(rocblas_handle        handle,
                                                   rocblas_atomics_mode* mode)
try
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;
    *mode = handle->atomics_mode;
    rocblas_internal_logger logger;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        logger.log_trace(handle, "rocblas_get_atomics_mode", *mode);
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * ! \brief set atomics mode
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_atomics_mode(rocblas_handle handle, rocblas_atomics_mode mode)
try
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;
    rocblas_internal_logger logger;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        logger.log_trace(handle, "rocblas_set_atomics_mode", mode);
    handle->atomics_mode = mode;
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * ! \brief get math mode
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_math_mode(rocblas_handle handle, rocblas_math_mode* mode)
try
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;
    rocblas_internal_logger logger;
    *mode = handle->math_mode;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        logger.log_trace(handle, "rocblas_get_math_mode", *mode);
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * ! \brief set math mode
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_math_mode(rocblas_handle handle, rocblas_math_mode mode)
try
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;

    rocblas_internal_logger logger;

    bool supported = true;
    switch(mode)
    {
    case rocblas_default_math:
        supported = true;
        break;
    case rocblas_xf32_xdl_math_op:
        supported = rocblas_internal_tensile_supports_xdl_math_op(mode);
        break;
    default:
        supported = false;
        break;
    }

    if(!supported)
    {
        if(handle->layer_mode & rocblas_layer_mode_log_trace)
            logger.log_trace(handle, "rocblas_set_math_mode", mode, "is not supported");
    }
    else
    {
        if(handle->layer_mode & rocblas_layer_mode_log_trace)
            logger.log_trace(handle, "rocblas_set_math_mode", mode);

        handle->math_mode = mode;
    }
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 * ! \brief create rocblas handle called before any rocblas library routines
 ******************************************************************************/
extern "C" rocblas_status rocblas_create_handle(rocblas_handle* handle)
try
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;

    // allocate on heap
    *handle = new _rocblas_handle;
    rocblas_internal_logger logger;
    if((*handle)->layer_mode & rocblas_layer_mode_log_trace)
        logger.log_trace(*handle, "rocblas_create_handle");

    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 *! \brief release rocblas handle, will implicitly synchronize host and device
 ******************************************************************************/
extern "C" rocblas_status rocblas_destroy_handle(rocblas_handle handle)
try
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;
    rocblas_internal_logger logger;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        logger.log_trace(handle, "rocblas_destroy_handle");
    // call destructor
    delete handle;

    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 *! \brief   set rocblas stream used for all subsequent library function calls.
 *   If not set, all hip kernels will take the default NULL stream.
 *   stream must be created before this call
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_stream(rocblas_handle handle, hipStream_t stream)
try
{
    // If handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;
    rocblas_internal_logger logger;
    // Log rocblas_set_stream
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        logger.log_trace(handle, "rocblas_set_stream", stream);

    // If the stream is unchanged, return immediately
    if(stream == handle->stream)
        return rocblas_status_success;

    //Verify if the new stream is in capture mode
    hipStreamCaptureStatus stream_status = hipStreamCaptureStatusNone;
    if(stream != 0)
    {
        bool status = hipStreamIsCapturing(stream, &stream_status) == hipSuccess;

        if(!status)
            return rocblas_status_invalid_value;
    }

    // Stream capture does not allow use of hipStreamQuery
    // If the current stream or new stream is in capture mode, skip use of hipStreamQuery()
    if((handle->stream == 0 || !handle->is_stream_in_capture_mode())
       && stream_status == hipStreamCaptureStatusNone)
    {
        // The new stream must be valid
        if(stream != 0 && hipStreamQuery(stream) == hipErrorInvalidHandle)
            return rocblas_status_invalid_value;
    }

    // Set the new stream
    handle->stream = stream;
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 *! \brief   get rocblas stream used for all subsequent library function calls.
 *   If not set, all hip kernels will take the default NULL stream.
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_stream(rocblas_handle handle, hipStream_t* stream_id)
try
{
    // if handle not valid
    if(!handle)
        return rocblas_status_invalid_handle;
    if(!stream_id)
        return rocblas_status_invalid_pointer;
    rocblas_internal_logger logger;
    if(handle->layer_mode & rocblas_layer_mode_log_trace)
        logger.log_trace(handle, "rocblas_get_stream", *stream_id);
    *stream_id = handle->get_stream();
    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

/*******************************************************************************
 *! \brief   copies void* vector x with stride incx on host to void* vector
     y with stride incy on device. Vectors have n elements of size elem_size.
  TODO: Need to replace device memory allocation with new system
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_vector_64(
    int64_t n, int64_t elem_size, const void* x_h, int64_t incx, void* y_d, int64_t incy)
try
{
    if(n == 0) // quick return
        return rocblas_status_success;
    if(n < 0 || incx <= 0 || incy <= 0 || elem_size <= 0)
        return rocblas_status_invalid_size;
    if(!x_h || !y_d)
        return rocblas_status_invalid_pointer;

    size_t elem_size_u64 = size_t(elem_size);

    if(incx == 1 && incy == 1) // contiguous host vector -> contiguous device vector
    {
        RETURN_IF_HIP_ERROR(hipMemcpy(y_d, x_h, elem_size_u64 * n, hipMemcpyHostToDevice));
    }
    else // either non-contiguous host vector or non-contiguous device vector
    {
        // pretend data is 2D to compensate for non unit increments
        RETURN_IF_HIP_ERROR(hipMemcpy2D(y_d,
                                        elem_size_u64 * incy,
                                        x_h,
                                        elem_size_u64 * incx,
                                        elem_size,
                                        n,
                                        hipMemcpyHostToDevice));
    }
    return rocblas_status_success;
}
catch(...) // catch all exceptions
{
    return exception_to_rocblas_status();
}

extern "C" rocblas_status rocblas_set_vector(rocblas_int n,
                                             rocblas_int elem_size,
                                             const void* x_h,
                                             rocblas_int incx,
                                             void*       y_d,
                                             rocblas_int incy)
{
    return rocblas_set_vector_64(n, elem_size, x_h, incx, y_d, incy);
}

/*******************************************************************************
 *! \brief   copies void* vector x with stride incx on device to void* vector
     y with stride incy on host. Vectors have n elements of size elem_size.
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_vector_64(
    int64_t n, int64_t elem_size, const void* x_d, int64_t incx, void* y_h, int64_t incy)
try
{
    if(n == 0) // quick return
        return rocblas_status_success;
    if(n < 0 || incx <= 0 || incy <= 0 || elem_size <= 0)
        return rocblas_status_invalid_size;
    if(!x_d || !y_h)
        return rocblas_status_invalid_pointer;

    size_t elem_size_u64 = size_t(elem_size);

    if(incx == 1 && incy == 1) // congiguous device vector -> congiguous host vector
    {
        RETURN_IF_HIP_ERROR(hipMemcpy(y_h, x_d, elem_size_u64 * n, hipMemcpyDeviceToHost));
    }
    else
    {
        // pretend data is 2D to compensate for non unit increments
        RETURN_IF_HIP_ERROR(hipMemcpy2D(y_h,
                                        elem_size_u64 * incy,
                                        x_d,
                                        elem_size_u64 * incx,
                                        elem_size,
                                        n,
                                        hipMemcpyDeviceToHost));
    }
    return rocblas_status_success;
}
catch(...) // catch all exceptions
{
    return exception_to_rocblas_status();
}

extern "C" rocblas_status rocblas_get_vector(rocblas_int n,
                                             rocblas_int elem_size,
                                             const void* x_d,
                                             rocblas_int incx,
                                             void*       y_h,
                                             rocblas_int incy)
{
    return rocblas_get_vector_64(n, elem_size, x_d, incx, y_h, incy);
}

/*******************************************************************************
 *! \brief   copies void* vector x with stride incx on host to void* vector
     y with stride incy on device. Vectors have n elements of size elem_size.
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_vector_async_64(int64_t     n,
                                                      int64_t     elem_size,
                                                      const void* x_h,
                                                      int64_t     incx,
                                                      void*       y_d,
                                                      int64_t     incy,
                                                      hipStream_t stream)
try
{
    if(n == 0) // quick return
        return rocblas_status_success;
    if(n < 0 || incx <= 0 || incy <= 0 || elem_size <= 0)
        return rocblas_status_invalid_size;
    if(!x_h || !y_d)
        return rocblas_status_invalid_pointer;

    size_t elem_size_u64 = size_t(elem_size);

    if(incx == 1 && incy == 1) // contiguous host vector -> contiguous device vector
    {
        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(y_d, x_h, elem_size_u64 * n, hipMemcpyHostToDevice, stream));
    }
    else // either non-contiguous host vector or non-contiguous device vector
    {
        // pretend data is 2D to compensate for non unit increments
        RETURN_IF_HIP_ERROR(hipMemcpy2DAsync(y_d,
                                             elem_size_u64 * incy,
                                             x_h,
                                             elem_size_u64 * incx,
                                             elem_size,
                                             n,
                                             hipMemcpyHostToDevice,
                                             stream));
    }
    return rocblas_status_success;
}
catch(...) // catch all exceptions
{
    return exception_to_rocblas_status();
}

extern "C" rocblas_status rocblas_set_vector_async(rocblas_int n,
                                                   rocblas_int elem_size,
                                                   const void* x_h,
                                                   rocblas_int incx,
                                                   void*       y_d,
                                                   rocblas_int incy,
                                                   hipStream_t stream)
{
    return rocblas_set_vector_async_64(n, elem_size, x_h, incx, y_d, incy, stream);
}

/*******************************************************************************
 *! \brief   copies void* vector x with stride incx on device to void* vector
     y with stride incy on host. Vectors have n elements of size elem_size.
 ******************************************************************************/
extern "C" rocblas_status rocblas_get_vector_async_64(int64_t     n,
                                                      int64_t     elem_size,
                                                      const void* x_d,
                                                      int64_t     incx,
                                                      void*       y_h,
                                                      int64_t     incy,
                                                      hipStream_t stream)
try
{
    if(n == 0) // quick return
        return rocblas_status_success;
    if(n < 0 || incx <= 0 || incy <= 0 || elem_size <= 0)
        return rocblas_status_invalid_size;
    if(!x_d || !y_h)
        return rocblas_status_invalid_pointer;

    size_t elem_size_u64 = size_t(elem_size);

    if(incx == 1 && incy == 1) // congiguous device vector -> congiguous host vector
    {
        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(y_h, x_d, elem_size_u64 * n, hipMemcpyDeviceToHost, stream));
    }
    else // either device or host vector is non-contiguous
    {
        // pretend data is 2D to compensate for non unit increments
        RETURN_IF_HIP_ERROR(hipMemcpy2DAsync(y_h,
                                             elem_size_u64 * incy,
                                             x_d,
                                             elem_size_u64 * incx,
                                             elem_size,
                                             n,
                                             hipMemcpyDeviceToHost,
                                             stream));
    }
    return rocblas_status_success;
}
catch(...) // catch all exceptions
{
    return exception_to_rocblas_status();
}

extern "C" rocblas_status rocblas_get_vector_async(rocblas_int n,
                                                   rocblas_int elem_size,
                                                   const void* x_d,
                                                   rocblas_int incx,
                                                   void*       y_h,
                                                   rocblas_int incy,
                                                   hipStream_t stream)
{
    return rocblas_get_vector_async_64(n, elem_size, x_d, incx, y_h, incy, stream);
}

/*******************************************************************************
 *! \brief  Matrix copy on device. Matrices are void pointers with element
     size elem_size
 ******************************************************************************/

constexpr size_t      MAT_BUFF_MAX_BYTES = 1048576;
constexpr rocblas_int MATRIX_DIM_X       = 128;
constexpr rocblas_int MATRIX_DIM_Y       = 8;

template <rocblas_int DIM_X, rocblas_int DIM_Y>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_copy_void_ptr_matrix_kernel(rocblas_int rows,
                                    rocblas_int cols,
                                    size_t      elem_size_u64,
                                    const void* a,
                                    rocblas_int lda,
                                    void*       b,
                                    rocblas_int ldb)
{
    rocblas_int tx = blockIdx.x * blockDim.x + threadIdx.x;
    rocblas_int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if(tx < rows && ty < cols)
        memcpy((char*)b + (tx + size_t(ldb) * ty) * elem_size_u64,
               (const char*)a + (tx + size_t(lda) * ty) * elem_size_u64,
               elem_size_u64);
}

/*******************************************************************************
 *! \brief   copies void* matrix a_h with leading dimentsion lda on host to
     void* matrix b_d with leading dimension ldb on device. Matrices have
     size rows * cols with element size elem_size.
 ******************************************************************************/

extern "C" rocblas_status rocblas_set_matrix_64(int64_t     rows,
                                                int64_t     cols,
                                                int64_t     elem_size,
                                                const void* a_h,
                                                int64_t     lda,
                                                void*       b_d,
                                                int64_t     ldb)
try
{
    if(rows == 0 || cols == 0) // quick return
        return rocblas_status_success;
    if(rows < 0 || cols < 0 || lda <= 0 || ldb <= 0 || rows > lda || rows > ldb || elem_size <= 0)
        return rocblas_status_invalid_size;
    if(!a_h || !b_d)
        return rocblas_status_invalid_pointer;

    size_t elem_size_u64 = size_t(elem_size);

    // contiguous host matrix -> contiguous device matrix
    if(lda == rows && ldb == rows)
    {
        size_t bytes_to_copy = elem_size_u64 * rows * cols;
        RETURN_IF_HIP_ERROR(hipMemcpy(b_d, a_h, bytes_to_copy, hipMemcpyHostToDevice));
    }
    else
    {
        // width is column vector in matrix
        RETURN_IF_HIP_ERROR(hipMemcpy2D(b_d,
                                        elem_size_u64 * ldb,
                                        a_h,
                                        elem_size_u64 * lda,
                                        elem_size_u64 * rows,
                                        cols,
                                        hipMemcpyHostToDevice));
    }
    return rocblas_status_success;
}
catch(...) // catch all exceptions
{
    return exception_to_rocblas_status();
}

extern "C" rocblas_status rocblas_set_matrix(rocblas_int rows,
                                             rocblas_int cols,
                                             rocblas_int elem_size,
                                             const void* a_h,
                                             rocblas_int lda,
                                             void*       b_d,
                                             rocblas_int ldb)
{
    return rocblas_set_matrix_64(rows, cols, elem_size, a_h, lda, b_d, ldb);
}

/*******************************************************************************
 *! \brief   copies void* matrix a_h with leading dimentsion lda on host to
     void* matrix b_d with leading dimension ldb on device. Matrices have
     size rows * cols with element size elem_size.
 ******************************************************************************/

extern "C" rocblas_status rocblas_get_matrix_64(int64_t     rows,
                                                int64_t     cols,
                                                int64_t     elem_size,
                                                const void* a_d,
                                                int64_t     lda,
                                                void*       b_h,
                                                int64_t     ldb)
try
{
    if(rows == 0 || cols == 0) // quick return
        return rocblas_status_success;
    if(rows < 0 || cols < 0 || lda <= 0 || ldb <= 0 || rows > lda || rows > ldb || elem_size <= 0)
        return rocblas_status_invalid_size;
    if(!a_d || !b_h)
        return rocblas_status_invalid_pointer;

    size_t elem_size_u64 = size_t(elem_size);

    // congiguous device matrix -> congiguous host matrix
    if(lda == rows && ldb == rows)
    {
        size_t bytes_to_copy = elem_size_u64 * rows * cols;
        RETURN_IF_HIP_ERROR(hipMemcpy(b_h, a_d, bytes_to_copy, hipMemcpyDeviceToHost));
    }
    else
    {
        // width is column vector in matrix
        RETURN_IF_HIP_ERROR(hipMemcpy2D(b_h,
                                        elem_size_u64 * ldb,
                                        a_d,
                                        elem_size_u64 * lda,
                                        elem_size_u64 * rows,
                                        cols,
                                        hipMemcpyDeviceToHost));
    }
    return rocblas_status_success;
}
catch(...) // catch all exceptions
{
    return exception_to_rocblas_status();
}

extern "C" rocblas_status rocblas_get_matrix(rocblas_int rows,
                                             rocblas_int cols,
                                             rocblas_int elem_size,
                                             const void* a_d,
                                             rocblas_int lda,
                                             void*       b_h,
                                             rocblas_int ldb)
{
    return rocblas_get_matrix_64(rows, cols, elem_size, a_d, lda, b_h, ldb);
}

/*******************************************************************************
 *! \brief   copies void* matrix a_h with leading dimentsion lda on host to
     void* matrix b_d with leading dimension ldb on device. Matrices have
     size rows * cols with element size elem_size.
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_matrix_async_64(int64_t     rows,
                                                      int64_t     cols,
                                                      int64_t     elem_size,
                                                      const void* a_h,
                                                      int64_t     lda,
                                                      void*       b_d,
                                                      int64_t     ldb,
                                                      hipStream_t stream)
try
{
    if(rows == 0 || cols == 0) // quick return
        return rocblas_status_success;
    if(rows < 0 || cols < 0 || lda <= 0 || ldb <= 0 || rows > lda || rows > ldb || elem_size <= 0)
        return rocblas_status_invalid_size;
    if(!a_h || !b_d)
        return rocblas_status_invalid_pointer;

    size_t elem_size_u64(elem_size);

    // contiguous host matrix -> contiguous device matrix
    if(lda == rows && ldb == rows)
    {
        size_t bytes_to_copy = elem_size_u64 * rows * cols;
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(b_d, a_h, bytes_to_copy, hipMemcpyHostToDevice, stream));
    }
    else
    {
        // width is column vector in matrix
        RETURN_IF_HIP_ERROR(hipMemcpy2DAsync(b_d,
                                             elem_size_u64 * ldb,
                                             a_h,
                                             elem_size_u64 * lda,
                                             elem_size_u64 * rows,
                                             cols,
                                             hipMemcpyHostToDevice,
                                             stream));
    }
    return rocblas_status_success;
}
catch(...) // catch all exceptions
{
    return exception_to_rocblas_status();
}

extern "C" rocblas_status rocblas_set_matrix_async(rocblas_int rows,
                                                   rocblas_int cols,
                                                   rocblas_int elem_size,
                                                   const void* a_h,
                                                   rocblas_int lda,
                                                   void*       b_d,
                                                   rocblas_int ldb,
                                                   hipStream_t stream)
{
    return rocblas_set_matrix_async_64(rows, cols, elem_size, a_h, lda, b_d, ldb, stream);
}

/*******************************************************************************
 *! \brief   copies void* matrix a_h with leading dimentsion lda on host to
     void* matrix b_d with leading dimension ldb on device. Matrices have
     size rows * cols with element size elem_size.
 ******************************************************************************/

extern "C" rocblas_status rocblas_get_matrix_async_64(int64_t     rows,
                                                      int64_t     cols,
                                                      int64_t     elem_size,
                                                      const void* a_d,
                                                      int64_t     lda,
                                                      void*       b_h,
                                                      int64_t     ldb,
                                                      hipStream_t stream)
try
{
    if(rows == 0 || cols == 0) // quick return
        return rocblas_status_success;
    if(rows < 0 || cols < 0 || lda <= 0 || ldb <= 0 || rows > lda || rows > ldb || elem_size <= 0)
        return rocblas_status_invalid_size;
    if(!a_d || !b_h)
        return rocblas_status_invalid_pointer;

    size_t elem_size_u64(elem_size);

    // contiguous host matrix -> contiguous device matrix
    if(lda == rows && ldb == rows)
    {
        size_t bytes_to_copy = elem_size_u64 * rows * cols;
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(b_h, a_d, bytes_to_copy, hipMemcpyDeviceToHost, stream));
    }
    else
    {
        // width is column vector in matrix
        RETURN_IF_HIP_ERROR(hipMemcpy2DAsync(b_h,
                                             elem_size_u64 * ldb,
                                             a_d,
                                             elem_size_u64 * lda,
                                             elem_size_u64 * rows,
                                             cols,
                                             hipMemcpyDeviceToHost,
                                             stream));
    }
    return rocblas_status_success;
}
catch(...) // catch all exceptions
{
    return exception_to_rocblas_status();
}

extern "C" rocblas_status rocblas_get_matrix_async(rocblas_int rows,
                                                   rocblas_int cols,
                                                   rocblas_int elem_size,
                                                   const void* a_d,
                                                   rocblas_int lda,
                                                   void*       b_h,
                                                   rocblas_int ldb,
                                                   hipStream_t stream)
{
    return rocblas_get_matrix_async_64(rows, cols, elem_size, a_d, lda, b_h, ldb, stream);
}

// Convert rocblas_status to string
extern "C" const char* rocblas_status_to_string(rocblas_status status)
{
#define CASE(x) \
    case x:     \
        return #x
    switch(status)
    {
        CASE(rocblas_status_success);
        CASE(rocblas_status_invalid_handle);
        CASE(rocblas_status_not_implemented);
        CASE(rocblas_status_invalid_pointer);
        CASE(rocblas_status_invalid_size);
        CASE(rocblas_status_memory_error);
        CASE(rocblas_status_internal_error);
        CASE(rocblas_status_perf_degraded);
        CASE(rocblas_status_size_query_mismatch);
        CASE(rocblas_status_size_increased);
        CASE(rocblas_status_size_unchanged);
        CASE(rocblas_status_invalid_value);
        CASE(rocblas_status_continue);
        CASE(rocblas_status_check_numerics_fail);
        CASE(rocblas_status_excluded_from_build);
        CASE(rocblas_status_arch_mismatch);
    }
#undef CASE
    // We don't use default: so that the compiler warns us if any valid enums are missing
    // from our switch. If the value is not a valid rocblas_status, we return this string.
    return "<undefined rocblas_status value>";
}

/*******************************************************************************
 * Function to set start/stop event handlers (for internal use only)
 ******************************************************************************/
extern "C" rocblas_status rocblas_set_start_stop_events(rocblas_handle handle,
                                                        hipEvent_t     startEvent,
                                                        hipEvent_t     stopEvent)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    handle->startEvent = startEvent;
    handle->stopEvent  = stopEvent;
    return rocblas_status_success;
}

/*******************************************************************************
 * GPU architecture-related functions
 ******************************************************************************/

// Emulate C++17 std::void_t
template <typename...>
using void_t = void;

// If gcnArchName not present, return empty string
template <typename PROP, typename = void>
struct ArchName
{
    std::string operator()(const PROP& prop) const
    {
        return "";
    }
};

// If gcnArchName exists as a member, use it instead
template <typename PROP>
struct ArchName<PROP, void_t<decltype(PROP::gcnArchName)>>
{
    std::string operator()(const PROP& prop) const
    {
        // strip out xnack/ecc from name
        std::string gcnArchName(prop.gcnArchName);
        std::string gcnArch = gcnArchName.substr(0, gcnArchName.find(":"));
        return gcnArch;
    }
};

// If gcnArchName not present, no xnack mode
template <typename PROP, typename = void>
struct XnackMode
{
    std::string operator()(const PROP& prop) const
    {
        return "";
    }
};

// If gcnArchName exists as a member, use it
template <typename PROP>
struct XnackMode<PROP, void_t<decltype(PROP::gcnArchName)>>
{
    std::string operator()(const PROP& prop) const
    {
        // strip out xnack/ecc from name
        std::string gcnArchName(prop.gcnArchName);
        auto        loc = gcnArchName.find("xnack");
        std::string xnackMode;
        if(loc != std::string::npos)
        {
            xnackMode = gcnArchName.substr(loc, 6);
            // guard against missing +/- at end of xnack mode
            if(xnackMode.size() < 6)
                xnackMode = "";
        }
        return xnackMode;
    }
};

int rocblas_internal_get_arch(rocblas_handle handle)
{
    return handle->getArch();
}

bool rocblas_internal_tensile_supports_ldc_ne_ldd(rocblas_handle handle)
{
    return handle->getArch() >= 906;
}

bool rocblas_internal_tensile_supports_xdl_math_op(rocblas_math_mode mode)
{
    int deviceId;
    hipGetDevice(&deviceId);
    hipDeviceProp_t deviceProperties;
    hipGetDeviceProperties(&deviceProperties, deviceId);
    std::string deviceString(deviceProperties.gcnArchName);
    return (deviceString.find("gfx942") != std::string::npos);
}

// exported. Get architecture name
std::string rocblas_internal_get_arch_name()
{
    int deviceId;
    hipGetDevice(&deviceId);
    hipDeviceProp_t deviceProperties;
    hipGetDeviceProperties(&deviceProperties, deviceId);
    return ArchName<hipDeviceProp_t>{}(deviceProperties);
}

// exported. Get xnack mode
std::string rocblas_internal_get_xnack_mode()
{
    int deviceId;
    hipGetDevice(&deviceId);
    hipDeviceProp_t deviceProperties;
    hipGetDeviceProperties(&deviceProperties, deviceId);
    return XnackMode<hipDeviceProp_t>{}(deviceProperties);
}

/*******************************************************************************
 * exported. Whether to skip buffer alloc/init/copy when tracing kernel names in Tensile *
 *******************************************************************************/
bool rocblas_internal_tensile_debug_skip_launch()
{
    static const bool skip_launch = [] {
        const char* db2 = std::getenv("TENSILE_DB2");
        return db2 && (strtol(db2, nullptr, 0) & 1) != 0;
    }();
    return skip_launch;
}
