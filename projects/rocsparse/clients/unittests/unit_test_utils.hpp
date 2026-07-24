/*! \file */
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

//
// Shared helpers for the host-path unit tests (rocsparse-unit-test-device).
//
// These tests drive the public C API so they exercise the HOST dispatch and
// validation code that rocSPARSE code coverage counts (coverage is host-only).
// They require a GPU (create a handle, allocate device memory, may launch
// kernels through the public API).
//
#pragma once

#include "rocsparse.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace rocsparse_ut
{
#define UT_CHECK_HIP(cmd)                                             \
    do                                                                \
    {                                                                 \
        hipError_t status_ = (cmd);                                   \
        ASSERT_EQ(status_, hipSuccess) << hipGetErrorString(status_); \
    } while(0)

#define UT_EXPECT_ROC(cmd, expected) EXPECT_EQ((cmd), (expected))

    // ----- type <-> enum mappings -------------------------------------------
    template <typename T>
    rocsparse_datatype dt_of();
    template <>
    inline rocsparse_datatype dt_of<float>()
    {
        return rocsparse_datatype_f32_r;
    }
    template <>
    inline rocsparse_datatype dt_of<double>()
    {
        return rocsparse_datatype_f64_r;
    }
    template <>
    inline rocsparse_datatype dt_of<rocsparse_float_complex>()
    {
        return rocsparse_datatype_f32_c;
    }
    template <>
    inline rocsparse_datatype dt_of<rocsparse_double_complex>()
    {
        return rocsparse_datatype_f64_c;
    }

    template <typename I>
    rocsparse_indextype it_of();
    template <>
    inline rocsparse_indextype it_of<int32_t>()
    {
        return rocsparse_indextype_i32;
    }
    template <>
    inline rocsparse_indextype it_of<int64_t>()
    {
        return rocsparse_indextype_i64;
    }

    template <typename T>
    T scalar(float v)
    {
        return static_cast<T>(v);
    }
    template <>
    inline rocsparse_float_complex scalar<rocsparse_float_complex>(float v)
    {
        return rocsparse_float_complex(v, 0.0f);
    }
    template <>
    inline rocsparse_double_complex scalar<rocsparse_double_complex>(float v)
    {
        return rocsparse_double_complex(v, 0.0);
    }

    // ----- tiny device buffer RAII ------------------------------------------
    template <typename T>
    struct device_vector
    {
        T*     ptr = nullptr;
        size_t n   = 0;

        device_vector() = default;
        explicit device_vector(const std::vector<T>& host)
        {
            n = host.size();
            if(hipMalloc(&ptr, n * sizeof(T)) != hipSuccess)
            {
                ptr = nullptr;
                return;
            }
            (void)hipMemcpy(ptr, host.data(), n * sizeof(T), hipMemcpyHostToDevice);
        }
        explicit device_vector(size_t count)
            : n(count)
        {
            if(hipMalloc(&ptr, n * sizeof(T)) != hipSuccess)
                ptr = nullptr;
        }
        device_vector(const device_vector&) = delete;
        device_vector& operator=(const device_vector&) = delete;
        ~device_vector()
        {
            if(ptr)
                (void)hipFree(ptr);
        }
        operator T*() const
        {
            return ptr;
        }
    };

    // ----- fixture owning a handle ------------------------------------------
    class HandleTest : public ::testing::Test
    {
    protected:
        rocsparse_handle handle = nullptr;
        void             SetUp() override
        {
            ASSERT_EQ(rocsparse_create_handle(&handle), rocsparse_status_success);
        }
        void TearDown() override
        {
            if(handle)
                EXPECT_EQ(rocsparse_destroy_handle(handle), rocsparse_status_success);
        }
    };
} // namespace rocsparse_ut
