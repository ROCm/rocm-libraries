/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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
            if(PAD > 0)
            {
                // Copy guard to device memory before allocated memory
                hipMemcpy(d, guard, sizeof(guard), hipMemcpyHostToDevice);

                // Point to allocated block
                d += PAD;

                // Copy guard to device memory after allocated memory
                hipMemcpy(d + size, guard, sizeof(guard), hipMemcpyHostToDevice);
            }
        }
#endif
        return d;
    }

    void device_vector_check(T* d)
    {
#ifdef GOOGLE_TEST
        if(PAD > 0)
        {
            U host[PAD];

            // Copy device memory after allocated memory to host
            hipMemcpy(host, d + this->size, sizeof(guard), hipMemcpyDeviceToHost);

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);

            // Point to guard before allocated memory
            d -= PAD;

            // Copy device memory after allocated memory to host
            hipMemcpy(host, d, sizeof(guard), hipMemcpyDeviceToHost);

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);
        }
#endif
    }

    void device_vector_teardown(T* d)
    {
        if(d != nullptr)
        {
#ifdef GOOGLE_TEST
            if(PAD > 0)
            {
                U host[PAD];

                // Copy device memory after allocated memory to host
                hipMemcpy(host, d + this->size, sizeof(guard), hipMemcpyDeviceToHost);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);

                // Point to guard before allocated memory
                d -= PAD;

                // Copy device memory after allocated memory to host
                hipMemcpy(host, d, sizeof(guard), hipMemcpyDeviceToHost);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);
            }
#endif
            // Free device memory
            CHECK_HIP_ERROR((hipFree)(d));
        }
    }
};
