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
// Device (GPU) unit tests for level-1 compute kernels ("Phase 3" micro-tests).
//
// Each test launches a single level-1 device function/kernel in isolation with
// tiny, hand-built inputs and analytically known outputs. Unlike the public-API
// integration suite, a failure here points directly at the kernel under test.
//
// This translation unit is compiled into the separate rocsparse-unit-test-device
// binary (links hip::device) and must run on a GPU (via the serializer, e.g.
// HIP_VISIBLE_DEVICES=0 gpu-run ./rocsparse-unit-test-device).
//
#include "rocsparse_common.hpp" // ROCSPARSE_DEVICE_ILF / ROCSPARSE_KERNEL / rocsparse::fma

#include "axpyi_device.h"
#include "gthr_device.h"
#include "gthrz_device.h"
#include "roti_device.h"
#include "sctr_device.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace
{
    constexpr uint32_t BS = 256; // block size (all test cases use nnz <= BS, one block)

#define CHECK_HIP(cmd)                                                \
    do                                                                \
    {                                                                 \
        hipError_t status_ = (cmd);                                   \
        ASSERT_EQ(status_, hipSuccess) << hipGetErrorString(status_); \
    } while(0)

    template <typename T>
    T* device_from(const std::vector<T>& h)
    {
        T* d = nullptr;
        if(hipMalloc(&d, h.size() * sizeof(T)) != hipSuccess)
        {
            return nullptr;
        }
        if(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice) != hipSuccess)
        {
            (void)hipFree(d);
            return nullptr;
        }
        return d;
    }

    template <typename T>
    std::vector<T> host_from(const T* d, size_t n)
    {
        std::vector<T> h(n);
        (void)hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost);
        return h;
    }

    void device_free(void* d)
    {
        (void)hipFree(d);
    }

    // Wrapper kernels for the ROCSPARSE_DEVICE_ILF functions (which are meant to
    // be called from within a kernel). This exercises the exact device code.
    template <typename T, typename I>
    __global__ void
        k_axpyi(I nnz, T alpha, const T* x_val, const I* x_ind, T* y, rocsparse_index_base base)
    {
        rocsparse::axpyi_device<BS, T, I, T, T>(nnz, alpha, x_val, x_ind, y, base);
    }

    template <typename T, typename I>
    __global__ void k_gthr(I nnz, const T* y, T* x_val, const I* x_ind, rocsparse_index_base base)
    {
        rocsparse::gthr_device<BS, I, T>(nnz, y, x_val, x_ind, base);
    }

    template <typename T, typename I>
    __global__ void
        k_roti(I nnz, T* x_val, const I* x_ind, T* y, T c, T s, rocsparse_index_base base)
    {
        rocsparse::roti_device<BS, I, T>(nnz, x_val, x_ind, y, c, s, base);
    }
} // namespace

// ---------------------------------------------------------------------------
// axpyi : y[x_ind[i]] += alpha * x_val[i]   (sparse x, dense y)
// ---------------------------------------------------------------------------
TEST(device_level1_axpyi, basic)
{
    using T = float;
    using I = int32_t;

    const std::vector<T> x_val{10, 20, 30};
    const std::vector<I> x_ind{0, 2, 4};
    std::vector<T>       y{1, 1, 1, 1, 1};
    const T              alpha = 2;

    T* d_xval = device_from(x_val);
    I* d_xind = device_from(x_ind);
    T* d_y    = device_from(y);
    ASSERT_NE(d_xval, nullptr);
    ASSERT_NE(d_xind, nullptr);
    ASSERT_NE(d_y, nullptr);

    hipLaunchKernelGGL((k_axpyi<T, I>),
                       dim3(1),
                       dim3(BS),
                       0,
                       0,
                       (I)x_ind.size(),
                       alpha,
                       d_xval,
                       d_xind,
                       d_y,
                       rocsparse_index_base_zero);
    CHECK_HIP(hipDeviceSynchronize());

    const std::vector<T> got = host_from(d_y, y.size());
    const std::vector<T> exp{21, 1, 41, 1, 61};
    EXPECT_EQ(got, exp);

    device_free(d_xval);
    device_free(d_xind);
    device_free(d_y);
}

// ---------------------------------------------------------------------------
// gthr : x_val[i] = y[x_ind[i]]   (gather)
// ---------------------------------------------------------------------------
TEST(device_level1_gthr, basic)
{
    using T = float;
    using I = int32_t;

    const std::vector<T> y{5, 6, 7, 8, 9};
    const std::vector<I> x_ind{4, 0, 2};
    std::vector<T>       x_val(x_ind.size(), -1);

    T* d_y    = device_from(y);
    I* d_xind = device_from(x_ind);
    T* d_xval = device_from(x_val);
    ASSERT_NE(d_y, nullptr);
    ASSERT_NE(d_xind, nullptr);
    ASSERT_NE(d_xval, nullptr);

    hipLaunchKernelGGL((k_gthr<T, I>),
                       dim3(1),
                       dim3(BS),
                       0,
                       0,
                       (I)x_ind.size(),
                       d_y,
                       d_xval,
                       d_xind,
                       rocsparse_index_base_zero);
    CHECK_HIP(hipDeviceSynchronize());

    const std::vector<T> got = host_from(d_xval, x_val.size());
    const std::vector<T> exp{9, 5, 7};
    EXPECT_EQ(got, exp);

    device_free(d_y);
    device_free(d_xind);
    device_free(d_xval);
}

// gthr with one-based indexing exercises the idx_base subtraction.
TEST(device_level1_gthr, index_base_one)
{
    using T = float;
    using I = int32_t;

    const std::vector<T> y{5, 6, 7, 8, 9};
    const std::vector<I> x_ind{5, 1, 3}; // one-based -> elements 9, 5, 7
    std::vector<T>       x_val(x_ind.size(), -1);

    T* d_y    = device_from(y);
    I* d_xind = device_from(x_ind);
    T* d_xval = device_from(x_val);
    ASSERT_NE(d_y, nullptr);
    ASSERT_NE(d_xind, nullptr);
    ASSERT_NE(d_xval, nullptr);

    hipLaunchKernelGGL((k_gthr<T, I>),
                       dim3(1),
                       dim3(BS),
                       0,
                       0,
                       (I)x_ind.size(),
                       d_y,
                       d_xval,
                       d_xind,
                       rocsparse_index_base_one);
    CHECK_HIP(hipDeviceSynchronize());

    const std::vector<T> got = host_from(d_xval, x_val.size());
    const std::vector<T> exp{9, 5, 7};
    EXPECT_EQ(got, exp);

    device_free(d_y);
    device_free(d_xind);
    device_free(d_xval);
}

// ---------------------------------------------------------------------------
// gthrz : x_val[i] = y[x_ind[i]]; y[x_ind[i]] = 0   (gather-and-zero)
// ---------------------------------------------------------------------------
TEST(device_level1_gthrz, basic)
{
    using T = float;

    std::vector<T>                   y{5, 6, 7, 8, 9};
    const std::vector<rocsparse_int> x_ind{1, 3};
    std::vector<T>                   x_val(x_ind.size(), -1);

    T*             d_y    = device_from(y);
    rocsparse_int* d_xind = device_from(x_ind);
    T*             d_xval = device_from(x_val);
    ASSERT_NE(d_y, nullptr);
    ASSERT_NE(d_xind, nullptr);
    ASSERT_NE(d_xval, nullptr);

    hipLaunchKernelGGL((rocsparse::gthrz_kernel<BS, T>),
                       dim3(1),
                       dim3(BS),
                       0,
                       0,
                       (rocsparse_int)x_ind.size(),
                       d_y,
                       d_xval,
                       d_xind,
                       rocsparse_index_base_zero);
    CHECK_HIP(hipDeviceSynchronize());

    const std::vector<T> got_xval = host_from(d_xval, x_val.size());
    const std::vector<T> got_y    = host_from(d_y, y.size());
    EXPECT_EQ(got_xval, (std::vector<T>{6, 8}));
    EXPECT_EQ(got_y, (std::vector<T>{5, 0, 7, 0, 9}));

    device_free(d_y);
    device_free(d_xind);
    device_free(d_xval);
}

// ---------------------------------------------------------------------------
// sctr : y[x_ind[i]] = x_val[i]   (scatter)
// ---------------------------------------------------------------------------
TEST(device_level1_sctr, basic)
{
    using T = float;
    using I = int32_t;

    const std::vector<T> x_val{11, 22};
    const std::vector<I> x_ind{0, 3};
    std::vector<T>       y(5, 0);

    T* d_xval = device_from(x_val);
    I* d_xind = device_from(x_ind);
    T* d_y    = device_from(y);
    ASSERT_NE(d_xval, nullptr);
    ASSERT_NE(d_xind, nullptr);
    ASSERT_NE(d_y, nullptr);

    hipLaunchKernelGGL((rocsparse::sctr_kernel<BS, I, T>),
                       dim3(1),
                       dim3(BS),
                       0,
                       0,
                       (I)x_ind.size(),
                       d_xval,
                       d_xind,
                       d_y,
                       rocsparse_index_base_zero);
    CHECK_HIP(hipDeviceSynchronize());

    const std::vector<T> got = host_from(d_y, y.size());
    const std::vector<T> exp{11, 0, 0, 22, 0};
    EXPECT_EQ(got, exp);

    device_free(d_xval);
    device_free(d_xind);
    device_free(d_y);
}

// ---------------------------------------------------------------------------
// roti : Givens rotation of sparse x against dense y
//   x' = c*x + s*y ; y' = c*y - s*x
// ---------------------------------------------------------------------------
TEST(device_level1_roti, rotation_90deg)
{
    using T = float;
    using I = int32_t;

    // c=0, s=1 -> x' = y ; y' = -x
    std::vector<T>       x_val{1, 2};
    const std::vector<I> x_ind{0, 2};
    std::vector<T>       y{10, 0, 20, 0, 0};

    T* d_xval = device_from(x_val);
    I* d_xind = device_from(x_ind);
    T* d_y    = device_from(y);
    ASSERT_NE(d_xval, nullptr);
    ASSERT_NE(d_xind, nullptr);
    ASSERT_NE(d_y, nullptr);

    hipLaunchKernelGGL((k_roti<T, I>),
                       dim3(1),
                       dim3(BS),
                       0,
                       0,
                       (I)x_ind.size(),
                       d_xval,
                       d_xind,
                       d_y,
                       (T)0,
                       (T)1,
                       rocsparse_index_base_zero);
    CHECK_HIP(hipDeviceSynchronize());

    const std::vector<T> got_xval = host_from(d_xval, x_val.size());
    const std::vector<T> got_y    = host_from(d_y, y.size());
    EXPECT_EQ(got_xval, (std::vector<T>{10, 20})); // x' = y
    EXPECT_EQ(got_y, (std::vector<T>{-1, 0, -2, 0, 0})); // y' = -x

    device_free(d_xval);
    device_free(d_xind);
    device_free(d_y);
}

// Identity rotation (c=1, s=0) must leave both vectors unchanged.
TEST(device_level1_roti, identity)
{
    using T = float;
    using I = int32_t;

    std::vector<T>       x_val{3, 4};
    const std::vector<I> x_ind{1, 4};
    std::vector<T>       y{0, 7, 0, 0, 8};

    T* d_xval = device_from(x_val);
    I* d_xind = device_from(x_ind);
    T* d_y    = device_from(y);
    ASSERT_NE(d_xval, nullptr);
    ASSERT_NE(d_xind, nullptr);
    ASSERT_NE(d_y, nullptr);

    hipLaunchKernelGGL((k_roti<T, I>),
                       dim3(1),
                       dim3(BS),
                       0,
                       0,
                       (I)x_ind.size(),
                       d_xval,
                       d_xind,
                       d_y,
                       (T)1,
                       (T)0,
                       rocsparse_index_base_zero);
    CHECK_HIP(hipDeviceSynchronize());

    EXPECT_EQ(host_from(d_xval, x_val.size()), (std::vector<T>{3, 4}));
    EXPECT_EQ(host_from(d_y, y.size()), (std::vector<T>{0, 7, 0, 0, 8}));

    device_free(d_xval);
    device_free(d_xind);
    device_free(d_y);
}
