/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#pragma once

#include "hipsolver.h"
#include "rocblas_init.hpp"
//#include "rocblas_test.hpp"
#include <cinttypes>
#include <cstdio>

using rocblas_int    = int;
using rocblas_stride = ptrdiff_t;

/* ============================================================================================
 */
/*! \brief  base-class to allocate/deallocate device memory */
template <typename T, size_t PAD, typename U>
class d_vector
{
private:
    size_t size, bytes;

public:
    inline size_t nmemb() const noexcept
    {
        return size;
    }

#ifdef GOOGLE_TEST
    U guard[PAD];
    d_vector(size_t s)
        : size(s)
        , bytes((s + PAD * 2) * sizeof(T))
    {
        // Initialize guard with random data
        if(PAD > 0)
        {
            rocblas_init_nan(guard, PAD);
        }
    }
#else
    d_vector(size_t s)
        : size(s)
        , bytes(s ? s * sizeof(T) : sizeof(T))
    {
    }
#endif

#ifdef GOOGLE_TEST
    // Helper function to setup guards - can use ASSERT_EQ to abort on failure
    void setup_guards(T* d)
    {
        if(PAD > 0)
        {
            // Copy guard to device memory before allocated memory
            ASSERT_EQ(hipMemcpy(d, guard, sizeof(guard), hipMemcpyHostToDevice), hipSuccess);

            // Copy guard to device memory after allocated memory
            ASSERT_EQ(hipMemcpy(d + PAD + size, guard, sizeof(guard), hipMemcpyHostToDevice),
                      hipSuccess);
        }
    }

    // Helper function to check guards - can use ASSERT_EQ to abort on failure
    void check_guards(T* d)
    {
        if(PAD > 0)
        {
            U host[PAD];

            // Copy device memory after allocated memory to host
            ASSERT_EQ(hipMemcpy(host, d + size, sizeof(guard), hipMemcpyDeviceToHost), hipSuccess);

            // Make sure no corruption has occurred
            ASSERT_EQ(memcmp(host, guard, sizeof(guard)), 0);

            // Copy device memory before allocated memory to host
            ASSERT_EQ(hipMemcpy(host, d - PAD, sizeof(guard), hipMemcpyDeviceToHost), hipSuccess);

            // Make sure no corruption has occurred
            ASSERT_EQ(memcmp(host, guard, sizeof(guard)), 0);
        }
    }
#endif

    T* device_vector_setup()
    {
        T* d;
        if((hipMalloc)(&d, bytes) != hipSuccess)
        {
            static char* lc = setlocale(LC_NUMERIC, "");
            fprintf(stderr, "Error allocating %'zu bytes (%zu GB)\n", bytes, bytes >> 30);
            d = nullptr;
        }
#ifdef GOOGLE_TEST
        else
        {
            setup_guards(d);
            if(PAD > 0)
                d += PAD; // Point to allocated block
        }
#endif
        return d;
    }

    void device_vector_check(T* d)
    {
#ifdef GOOGLE_TEST
        check_guards(d);
#endif
    }

    void device_vector_teardown(T* d)
    {
        if(d != nullptr)
        {
#ifdef GOOGLE_TEST
            check_guards(d);
            if(PAD > 0)
                d -= PAD; // Point to guard before allocated memory
#endif
            // Free device memory
            CHECK_HIP_ERROR((hipFree)(d));
        }
    }
};
