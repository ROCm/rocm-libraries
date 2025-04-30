/* **************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cinttypes>
#include <cstdio>

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <rocblas/rocblas.h>

#include "common/misc/data_initializer.hpp"
#include "common/misc/rocblas_test.hpp"
#include "common_host_helpers.hpp"

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

    d_vector(size_t s)
        : size(s)
        , bytes(s ? s * sizeof(T) : sizeof(T))
    {
    }

    T* device_vector_setup()
    {
        T* d = nullptr;
        if((hipMalloc)(&d, bytes) != hipSuccess)
        {
            fmt::print(stderr, "Error allocating {} bytes ({} GB)\n", bytes, bytes >> 30);
            d = nullptr;
        }
        if(d != nullptr)
        {
            auto status = (hipMemset)(d, 0, bytes);
            if(status != hipSuccess)
            {
                fmt::print(stderr, "error: {} ({}) at {}:{}\n", hipGetErrorString(status),
                           static_cast<std::int32_t>(status), __FILE__, __LINE__);
                rocblas_abort();
            }
        }
        return d;
    }

    void device_vector_check(T* d) {}

    void device_vector_teardown(T* d)
    {
        if(d != nullptr)
        {
            // Free device memory
            CHECK_HIP_ERROR((hipFree)(d));
        }
    }
};
