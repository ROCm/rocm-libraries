/* **************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "common_host_helpers.hpp"
#include "lib_host_helpers.hpp"
#include "rocsolver/rocsolver.h"

#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif

ROCSOLVER_BEGIN_NAMESPACE

/* ROCSOLVER_HYBRID_STORAGE provides a wrapper for data arrays that is intended to simplify host memory
   allocation as well as data transfers to and from the device. Hybrid algorithms can use rocsolver_hybrid_storage
   to read device data to the host so that the data may be operated upon, and then write the resultant data
   back to the device. Two modes are supported:

   * Standard mode: Used to read device data onto the host, which may then be operated upon by host code.
     A typical workflow will call init_async to allocate a host buffer and read the data into the buffer,
     execute host code on the buffered data through the pointer provided by the [] operator, and then call
     write_to_device_async to write the resultant data back to the device.

   * Pointers only mode: Used to read device pointers from a batched array onto the host, so that device
     kernels may be called on each individual batch instance. A typical workflow will call init_pointers_only
     to allocate a host buffer for the device pointers, and then execute device kernels on the pointers
     provided by the [] operator. */
template <typename T,
          typename I,
          typename U,
          std::enable_if_t<std::is_trivial<T>::value && std::is_standard_layout<T>::value
                               && !std::is_pointer<T>::value,
                           int> = 0>
struct rocsolver_hybrid_storage
{
    I dim, batch_count;
    rocblas_stride shift;
    rocblas_stride stride;

    U src_array;
    T** batch_array;
    T* val_array;

    /* Constructor */
    rocsolver_hybrid_storage()
        : src_array(nullptr)
        , batch_array(nullptr)
        , val_array(nullptr)
    {
    }

    /* Disallow copying. */
    rocsolver_hybrid_storage(const rocsolver_hybrid_storage&) = delete;

    /* Disallow assigning. */
    rocsolver_hybrid_storage& operator=(const rocsolver_hybrid_storage&) = delete;

    /* Destructor */
    ~rocsolver_hybrid_storage()
    {
        if(val_array)
        {
            memset(val_array, 0, sizeof(T) * this->dim * this->batch_count);
#ifdef _WIN32
            _aligned_free(val_array);
#else
            free(val_array);
#endif
        }
        if(batch_array && (val_array || this->dim < 0))
        {
            memset(batch_array, 0, sizeof(T*) * this->batch_count);
#ifdef _WIN32
            _aligned_free(batch_array);
#else
            free(batch_array);
#endif
        }
    }

    /* Used to read device pointers from a batched array for use on the host; no other data is read from the
       device. */
    rocblas_status init_pointers_only(U array,
                                      rocblas_stride shift,
                                      rocblas_stride stride,
                                      I batch_count,
                                      hipStream_t stream)
    {
        if(val_array)
        {
            memset(val_array, 0, sizeof(T) * this->dim * this->batch_count);
#ifdef _WIN32
            _aligned_free(val_array);
#else
            free(val_array);
#endif
            val_array = nullptr;
        }
        if(batch_array && (val_array || this->dim < 0))
        {
            memset(batch_array, 0, sizeof(T*) * this->batch_count);
#ifdef _WIN32
            _aligned_free(batch_array);
#else
            free(batch_array);
#endif
            batch_array = nullptr;
        }

        this->dim = -1;
        this->src_array = array;
        this->shift = shift;
        this->stride = stride;
        this->batch_count = batch_count;

        bool constexpr is_strided = (std::is_same<U, T*>::value || std::is_same<U, T* const>::value);
        bool is_device = is_device_pointer((void*)array);

        if(is_device)
        {
            // pointers only; don't allocate val_array

            if(!is_strided)
            {
                // data is batched; read device pointers into batch_array
                size_t batch_bytes = sizeof(T*) * batch_count;
#ifdef _WIN32
                batch_array = (T**)_aligned_malloc(batch_bytes, sizeof(void*));
                if(!batch_array)
                    return rocblas_status_memory_error;
#else
                if(posix_memalign((void**)&batch_array, sizeof(void*), batch_bytes) != 0)
                    return rocblas_status_memory_error;
#endif
                memset(batch_array, 0, batch_bytes);
                HIP_CHECK(
                    hipMemcpyAsync(batch_array, array, batch_bytes, hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipStreamSynchronize(stream));
            }
        }
        else
        {
            // data on host; use src_array directly

            if(!is_strided)
                batch_array = (T**)src_array;
        }

        return rocblas_status_success;
    }
    /* Used to read device data into a host buffer, which is managed by this class. Data for all batch instances
       will be read into the buffer. */
    rocblas_status init_async(I dim,
                              U array,
                              rocblas_stride shift,
                              rocblas_stride stride,
                              I batch_count,
                              hipStream_t stream)
    {
        if(val_array)
        {
            memset(val_array, 0, sizeof(T) * this->dim * this->batch_count);
#ifdef _WIN32
            _aligned_free(val_array);
#else
            free(val_array);
#endif
            val_array = nullptr;
        }
        if(batch_array && (val_array || this->dim < 0))
        {
            memset(batch_array, 0, sizeof(T*) * this->batch_count);
#ifdef _WIN32
            _aligned_free(batch_array);
#else
            free(batch_array);
#endif
            batch_array = nullptr;
        }

        if(dim < 0)
            return rocblas_status_internal_error;
        if(dim == 0)
        {
            this->dim = 0;
            this->src_array = 0;
            this->shift = 0;
            this->stride = 0;
            this->batch_count = 0;
            return rocblas_status_success;
        }

        this->dim = dim;
        this->src_array = array;
        this->shift = shift;
        this->stride = stride;
        this->batch_count = batch_count;

        bool constexpr is_strided = (std::is_same<U, T*>::value || std::is_same<U, T* const>::value);
        bool is_device = is_device_pointer((void*)array);

        if(is_device)
        {
            // allocate space on host for data from device
            size_t dim_bytes = sizeof(T) * dim;
            size_t val_bytes = sizeof(T) * dim * batch_count;
#ifdef _WIN32
            val_array = (T*)_aligned_malloc(val_bytes, sizeof(void*));
            if(!val_array)
                return rocblas_status_memory_error;
#else
            if(posix_memalign((void**)&val_array, sizeof(void*), val_bytes) != 0)
                return rocblas_status_memory_error;
#endif
            memset(val_array, 0, val_bytes);

            if(is_strided)
            {
                // data is strided; batch_array not needed

                // read data to val_array
                if(batch_count == 1 || stride == dim)
                {
                    HIP_CHECK(hipMemcpyAsync(val_array, src_array + shift, val_bytes,
                                             hipMemcpyDeviceToHost, stream));
                }
                else
                {
                    for(I bid = 0; bid < batch_count; bid++)
                    {
                        HIP_CHECK(hipMemcpyAsync(val_array + bid * dim,
                                                 src_array + shift + bid * stride, dim_bytes,
                                                 hipMemcpyDeviceToHost, stream));
                    }
                }
            }
            else
            {
                // data is batched; read device pointers into batch_array
                size_t batch_bytes = sizeof(T*) * batch_count;
#ifdef _WIN32
                batch_array = (T**)_aligned_malloc(batch_bytes, sizeof(void*));
                if(!batch_array)
                    return rocblas_status_memory_error;
#else
                if(posix_memalign((void**)&batch_array, sizeof(void*), batch_bytes) != 0)
                    return rocblas_status_memory_error;
#endif
                memset(batch_array, 0, batch_bytes);
                HIP_CHECK(
                    hipMemcpyAsync(batch_array, array, batch_bytes, hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipStreamSynchronize(stream));

                // read data to val_array
                for(I bid = 0; bid < batch_count; bid++)
                {
                    HIP_CHECK(hipMemcpyAsync(val_array + bid * dim, batch_array[bid] + shift,
                                             dim_bytes, hipMemcpyDeviceToHost, stream));
                }
            }
        }
        else
        {
            // data on host; use src_array directly

            if(!is_strided)
                batch_array = (T**)src_array;
        }

        return rocblas_status_success;
    }

    /* Copies data from host buffer back to the device. Returns an error if initialized for pointers
       only. */
    rocblas_status write_to_device_async(hipStream_t stream)
    {
        if(!src_array)
            return rocblas_status_internal_error;
        if(dim < 0)
            return rocblas_status_internal_error;
        if(dim == 0)
            return rocblas_status_success;

        if(val_array)
        {
            size_t dim_bytes = sizeof(T) * dim;
            size_t val_bytes = sizeof(T) * dim * batch_count;

            if(!batch_array)
            {
                if(batch_count == 1 || stride == dim)
                {
                    HIP_CHECK(hipMemcpyAsync(src_array, val_array, val_bytes, hipMemcpyHostToDevice,
                                             stream));
                }
                else
                {
                    for(I bid = 0; bid < batch_count; bid++)
                    {
                        HIP_CHECK(hipMemcpyAsync(src_array + bid * stride, val_array + bid * dim,
                                                 dim_bytes, hipMemcpyHostToDevice, stream));
                    }
                }
            }
            else
            {
                for(I bid = 0; bid < batch_count; bid++)
                {
                    HIP_CHECK(hipMemcpyAsync(batch_array[bid], val_array + bid * dim, dim_bytes,
                                             hipMemcpyHostToDevice, stream));
                }
            }
        }

        return rocblas_status_success;
    }

    /* Gets a pointer to the data for batch index bid. If initialized for pointers only, the
       returned pointer may be on the device. Otherwise, the pointer is on the host. */
    T* operator[](I bid)
    {
        if(!src_array)
            return nullptr;

        if(val_array)
            return val_array + bid * dim;
        else
        {
            if(batch_array)
                return batch_array[bid] + shift;
            else
                return (T*)(src_array + shift + bid * stride);
        }
    }
};

ROCSOLVER_END_NAMESPACE
