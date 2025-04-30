/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once
#ifndef ROCSPARSE_ALLOCATOR_HPP
#define ROCSPARSE_ALLOCATOR_HPP

#include "rocsparse_random.hpp"
#include <hip/hip_runtime_api.h>
#ifdef GOOGLE_TEST
#include "rocsparse_test.hpp"
#endif
#include <cinttypes>
#include <iostream>
#include <locale.h>
struct memory_mode
{
    typedef enum _value
    {
        device = 0,
        host,
        managed
    } value_t;

    static constexpr hipMemcpyKind get_hipMemcpyKind(memory_mode::value_t TARGET,
                                                     memory_mode::value_t SOURCE)
    {
        switch(TARGET)
        {
        case memory_mode::host:
        {
            switch(SOURCE)
            {
            case memory_mode::host:
            {
                return hipMemcpyHostToHost;
            }
            case memory_mode::device:
            {
                return hipMemcpyDeviceToHost;
            }
            case memory_mode::managed:
            {
                return hipMemcpyHostToHost;
            }
            }
        }
        case memory_mode::device:
        {
            switch(SOURCE)
            {
            case memory_mode::host:
            {
                return hipMemcpyHostToDevice;
            }
            case memory_mode::device:
            {
                return hipMemcpyDeviceToDevice;
            }
            case memory_mode::managed:
            {
                return hipMemcpyDeviceToDevice;
            }
            }
        }
        case memory_mode::managed:
        {
            switch(SOURCE)
            {
            case memory_mode::host:
            {
                return hipMemcpyHostToHost;
            }
            case memory_mode::managed:
            {
                return hipMemcpyHostToHost;
            }
            case memory_mode::device:
            {
                return hipMemcpyDeviceToDevice;
            }
            }
        }
        }
        throw(rocsparse_status_invalid_value);
    }
};

#ifndef ROCSPARSE_WITH_MEMSTAT

#define rocsparse_hipMalloc(p_, nbytes_) hipMalloc(p_, nbytes_)

#define rocsparse_hipFree(p_) hipFree(p_)

#define rocsparse_hipHostMalloc(p_, nbytes_) hipHostMalloc(p_, nbytes_)
#define rocsparse_hipHostFree(p_) hipHostFree(p_)

#define rocsparse_hipMallocManaged(p_, nbytes_) hipMallocManaged(p_, nbytes_)
#define rocsparse_hipFreeManaged(p_) hipFree(p_)

#else

template <typename T>
hipError_t rocsparse_hip_free_template(T* mem, const char* tag)
{
    return rocsparse_hip_free(static_cast<void*>(mem), tag);
}

template <typename T>
hipError_t rocsparse_hip_malloc_template(T** mem, size_t nbytes, const char* tag)
{
    void* m;
    auto  error = rocsparse_hip_malloc(&m, nbytes, tag);
    if(error != hipSuccess)
    {
        return error;
    }
    mem[0] = static_cast<T*>(m);
    return hipSuccess;
}

template <typename T>
hipError_t rocsparse_hip_host_free_template(T* mem, const char* tag)
{
    return rocsparse_hip_host_free(static_cast<void*>(mem), tag);
}

template <typename T>
hipError_t rocsparse_hip_host_malloc_template(T** mem, size_t nbytes, const char* tag)
{
    void* m;
    auto  error = rocsparse_hip_host_malloc(&m, nbytes, tag);
    if(error != hipSuccess)
    {
        return error;
    }
    mem[0] = static_cast<T*>(m);
    return hipSuccess;
}
template <typename T>
hipError_t rocsparse_hip_free_managed_template(T* mem, const char* tag)
{
    return rocsparse_hip_free_managed(static_cast<void*>(mem), tag);
}

template <typename T>
hipError_t rocsparse_hip_malloc_managed_template(T** mem, size_t nbytes, const char* tag)
{
    void* m;
    auto  error = rocsparse_hip_malloc_managed(&m, nbytes, tag);
    if(error != hipSuccess)
    {
        return error;
    }
    mem[0] = static_cast<T*>(m);
    return hipSuccess;
}

#define ROCSPARSE_CLIENTS_MEMORY_SOURCE_LINE(line_) #line_
#define ROCSPARSE_CLIENTS_MEMORY_SOURCE_TAG(line_) \
    __FILE__ " " ROCSPARSE_CLIENTS_MEMORY_SOURCE_LINE(line_)

#define rocsparse_hipMalloc(p_, nbytes_) \
    rocsparse_hip_malloc_template(p_, nbytes_, ROCSPARSE_CLIENTS_MEMORY_SOURCE_TAG(__LINE__))

#define rocsparse_hipFree(p_) \
    rocsparse_hip_free_template(p_, ROCSPARSE_CLIENTS_MEMORY_SOURCE_TAG(__LINE__))

#define rocsparse_hipHostMalloc(p_, nbytes_) \
    rocsparse_hip_host_malloc_template(p_, nbytes_, ROCSPARSE_CLIENTS_MEMORY_SOURCE_TAG(__LINE__))

#define rocsparse_hipHostFree(p_) \
    rocsparse_hip_host_free_template(p_, ROCSPARSE_CLIENTS_MEMORY_SOURCE_TAG(__LINE__))

#define rocsparse_hipMallocManaged(p_, nbytes_) \
    rocsparse_hip_malloc_managed_template(      \
        p_, nbytes_, ROCSPARSE_CLIENTS_MEMORY_SOURCE_TAG(__LINE__))

#define rocsparse_hipFreeManaged(p_) \
    rocsparse_hip_free_managed_template(p_, ROCSPARSE_CLIENTS_MEMORY_SOURCE_TAG(__LINE__))

#endif

template <memory_mode::value_t MODE, typename T, size_t PAD = 4096, typename U = T>
struct rocsparse_allocator
{

private:
    static size_t compute_nbytes(size_t s)
    {
#ifdef GOOGLE_TEST
        return sizeof(T) * (s + PAD * 3);
#else
        return sizeof(T) * s;
#endif
    }

#ifdef GOOGLE_TEST
    static void init_guards(U* A, size_t N)
    {
        for(size_t i = 0; i < N; ++i)
            A[i] = U(rocsparse_nan_rng());
    }

    static T* off_guards(T* d)
    {
        if(PAD > 0)
        {
            d = (T*)(((U*)d) - 2 * PAD);
        }
        return d;
    }

    static T* install_guards(T* d, size_t size)
    {
        if(d != nullptr)
        {
            if(PAD > 0)
            {
                U guard[PAD];
                init_guards(guard, PAD);

                // Copy guard to device memory before allocated memory
                EXPECT_EQ(hipMemcpy(d,
                                    guard,
                                    sizeof(guard),
                                    memory_mode::get_hipMemcpyKind(MODE, memory_mode::host)),
                          0);
                EXPECT_EQ(hipMemcpy(d + PAD,
                                    guard,
                                    sizeof(guard),
                                    memory_mode::get_hipMemcpyKind(MODE, memory_mode::host)),
                          0);

                // Point to allocated block
                d += 2 * PAD;

                // Copy guard to device memory after allocated memory
                EXPECT_EQ(hipMemcpy(d + size,
                                    guard,
                                    sizeof(guard),
                                    memory_mode::get_hipMemcpyKind(MODE, memory_mode::host)),
                          0);
            }
        }
        return d;
    }
#endif

public:
    static T* malloc(size_t size)
    {
        if(size == 0)
        {
            return (T*)nullptr;
        }
        size_t nbytes = compute_nbytes(size);
        T*     d;
        switch(MODE)
        {
        case memory_mode::host:
        {
            if(rocsparse_hipHostMalloc(&d, nbytes) != hipSuccess)
            {
                fprintf(stderr, "Error allocating %'zu bytes (%zu GB)\n", nbytes, nbytes >> 30);
                d = nullptr;
                throw std::bad_alloc();
            }
            break;
        }
        case memory_mode::device:
        {
            if(rocsparse_hipMalloc(&d, nbytes) != hipSuccess)
            {
                fprintf(stderr, "Error allocating %'zu bytes (%zu GB)\n", nbytes, nbytes >> 30);
                d = nullptr;
                throw std::bad_alloc();
            }
            break;
        }
        case memory_mode::managed:
        {

            if(rocsparse_hipMallocManaged(&d, nbytes) != hipSuccess)
            {
                fprintf(stderr, "Error allocating %'zu bytes (%zu GB)\n", nbytes, nbytes >> 30);
                d = nullptr;
                throw std::bad_alloc();
            }

            break;
        }
        }
#ifdef GOOGLE_TEST
        return install_guards(d, size);
#else
        return d;
#endif
    }

#ifdef GOOGLE_TEST
    static void check_guards(T* d, size_t size)
    {
        if(d != nullptr)
        {
            if(PAD > 0)
            {
                U host[PAD], guard[PAD];
                // Copy device memory after allocated memory to host
                EXPECT_EQ(hipMemcpy(guard,
                                    ((U*)d) - 2 * PAD,
                                    sizeof(guard),
                                    memory_mode::get_hipMemcpyKind(memory_mode::host, MODE)),
                          0);

                // Copy device memory after allocated memory to host
                EXPECT_EQ(hipMemcpy(host,
                                    d + size,
                                    sizeof(guard),
                                    memory_mode::get_hipMemcpyKind(memory_mode::host, MODE)),
                          0);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);

                // Point to guard before allocated memory
                d -= PAD;

                // Copy device memory after allocated memory to host
                EXPECT_EQ(hipMemcpy(host,
                                    d,
                                    sizeof(guard),
                                    memory_mode::get_hipMemcpyKind(memory_mode::host, MODE)),
                          0);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);
            }
        }
    }
#endif

    static void free(T* d)
    {
        if(d != nullptr)
        {
#ifdef GOOGLE_TEST
            d = off_guards(d);
#endif
            switch(MODE)
            {
            case memory_mode::host:
            {
                auto status = rocsparse_hipHostFree(d);
                if(status != hipSuccess)
                {
                    throw status;
                }

                break;
            }
            case memory_mode::device:
            {
                // Free device memory
                auto status = rocsparse_hipFree(d);
                if(status != hipSuccess)
                {
                    throw status;
                }
                break;
            }
            case memory_mode::managed:
            {
                // Free device memory
                auto status = rocsparse_hipFreeManaged(d);
                if(status != hipSuccess)
                {
                    throw status;
                }
                break;
            }
            }
        }
    }
};

template <typename T>
using rocsparse_host_allocator = rocsparse_allocator<memory_mode::host, T>;

template <typename T>
using rocsparse_device_allocator = rocsparse_allocator<memory_mode::device, T>;

template <typename T>
using rocsparse_managed_allocator = rocsparse_allocator<memory_mode::managed, T>;

#endif // ROCSPARSE_ALLOCATOR_HPP
