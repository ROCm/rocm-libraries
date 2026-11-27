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
#include "unit_test_utils.hpp"

#include "axpyi_device.h"
#include "dotci_device.h"
#include "doti_device.h"
#include "gthr_device.h"
#include "gthrz_device.h"
#include "roti_device.h"
#include "sctr_device.h"

#include <chrono>
#include <cstdlib>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <limits>
#include <thread>
#include <vector>

namespace
{
    constexpr uint32_t BS = 256; // block size (all test cases use nnz <= BS, one block)

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
        rocsparse::gthr_device<BS, true, I, T>(nnz, y, x_val, x_ind, base);
    }

    template <typename T, typename I, bool GRID_STRIDE = false>
    __global__ void
        k_roti(I nnz, T* x_val, const I* x_ind, T* y, T c, T s, rocsparse_index_base base)
    {
        rocsparse::roti_device<BS, GRID_STRIDE, I, T>(nnz, x_val, x_ind, y, c, s, base);
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

    rocsparse_ut::device_vector<T> d_xval(x_val);
    rocsparse_ut::device_vector<I> d_xind(x_ind);
    rocsparse_ut::device_vector<T> d_y(y);

    (void)hipGetLastError();
    hipLaunchKernelGGL((k_axpyi<T, I>),
                       dim3(1),
                       dim3(BS),
                       0,
                       0,
                       (I)x_ind.size(),
                       alpha,
                       d_xval.ptr,
                       d_xind.ptr,
                       d_y.ptr,
                       rocsparse_index_base_zero);
    UT_CHECK_HIP(hipGetLastError());
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<T> got = rocsparse_ut::to_host(d_y);
    const std::vector<T> exp{21, 1, 41, 1, 61};
    EXPECT_EQ(got, exp);
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

    rocsparse_ut::device_vector<T> d_y(y);
    rocsparse_ut::device_vector<I> d_xind(x_ind);
    rocsparse_ut::device_vector<T> d_xval(x_val);

    (void)hipGetLastError();
    hipLaunchKernelGGL((k_gthr<T, I>),
                       dim3(1),
                       dim3(BS),
                       0,
                       0,
                       (I)x_ind.size(),
                       d_y.ptr,
                       d_xval.ptr,
                       d_xind.ptr,
                       rocsparse_index_base_zero);
    UT_CHECK_HIP(hipGetLastError());
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<T> got = rocsparse_ut::to_host(d_xval);
    const std::vector<T> exp{9, 5, 7};
    EXPECT_EQ(got, exp);
}

// gthr with one-based indexing exercises the idx_base subtraction.
TEST(device_level1_gthr, index_base_one)
{
    using T = float;
    using I = int32_t;

    const std::vector<T> y{5, 6, 7, 8, 9};
    const std::vector<I> x_ind{5, 1, 3}; // one-based -> elements 9, 5, 7
    std::vector<T>       x_val(x_ind.size(), -1);

    rocsparse_ut::device_vector<T> d_y(y);
    rocsparse_ut::device_vector<I> d_xind(x_ind);
    rocsparse_ut::device_vector<T> d_xval(x_val);

    (void)hipGetLastError();
    hipLaunchKernelGGL((k_gthr<T, I>),
                       dim3(1),
                       dim3(BS),
                       0,
                       0,
                       (I)x_ind.size(),
                       d_y.ptr,
                       d_xval.ptr,
                       d_xind.ptr,
                       rocsparse_index_base_one);
    UT_CHECK_HIP(hipGetLastError());
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<T> got = rocsparse_ut::to_host(d_xval);
    const std::vector<T> exp{9, 5, 7};
    EXPECT_EQ(got, exp);
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

    rocsparse_ut::device_vector<T>             d_y(y);
    rocsparse_ut::device_vector<rocsparse_int> d_xind(x_ind);
    rocsparse_ut::device_vector<T>             d_xval(x_val);

    (void)hipGetLastError();
    hipLaunchKernelGGL((rocsparse::gthrz_kernel<BS, T>),
                       dim3(1),
                       dim3(BS),
                       0,
                       0,
                       (rocsparse_int)x_ind.size(),
                       d_y.ptr,
                       d_xval.ptr,
                       d_xind.ptr,
                       rocsparse_index_base_zero);
    UT_CHECK_HIP(hipGetLastError());
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<T> got_xval = rocsparse_ut::to_host(d_xval);
    const std::vector<T> got_y    = rocsparse_ut::to_host(d_y);
    EXPECT_EQ(got_xval, (std::vector<T>{6, 8}));
    EXPECT_EQ(got_y, (std::vector<T>{5, 0, 7, 0, 9}));
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

    rocsparse_ut::device_vector<T> d_xval(x_val);
    rocsparse_ut::device_vector<I> d_xind(x_ind);
    rocsparse_ut::device_vector<T> d_y(y);

    (void)hipGetLastError();
    hipLaunchKernelGGL((rocsparse::sctr_kernel<BS, I, T>),
                       dim3(1),
                       dim3(BS),
                       0,
                       0,
                       (I)x_ind.size(),
                       d_xval.ptr,
                       d_xind.ptr,
                       d_y.ptr,
                       rocsparse_index_base_zero);
    UT_CHECK_HIP(hipGetLastError());
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<T> got = rocsparse_ut::to_host(d_y);
    const std::vector<T> exp{11, 0, 0, 22, 0};
    EXPECT_EQ(got, exp);
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

    rocsparse_ut::device_vector<T> d_xval(x_val);
    rocsparse_ut::device_vector<I> d_xind(x_ind);
    rocsparse_ut::device_vector<T> d_y(y);

    (void)hipGetLastError();
    hipLaunchKernelGGL((k_roti<T, I>),
                       dim3(1),
                       dim3(BS),
                       0,
                       0,
                       (I)x_ind.size(),
                       d_xval.ptr,
                       d_xind.ptr,
                       d_y.ptr,
                       (T)0,
                       (T)1,
                       rocsparse_index_base_zero);
    UT_CHECK_HIP(hipGetLastError());
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<T> got_xval = rocsparse_ut::to_host(d_xval);
    const std::vector<T> got_y    = rocsparse_ut::to_host(d_y);
    EXPECT_EQ(got_xval, (std::vector<T>{10, 20})); // x' = y
    EXPECT_EQ(got_y, (std::vector<T>{-1, 0, -2, 0, 0})); // y' = -x
}

// The grid-stride variant must rotate every element even when launched with
// fewer blocks than ceil(nnz / BS).
TEST(device_level1_roti, grid_stride)
{
    using T = float;
    using I = int64_t;

    const I        nnz = 2 * BS + 3;
    std::vector<T> x_val(nnz);
    std::vector<I> x_ind(nnz);
    std::vector<T> y(nnz, 0);
    for(I i = 0; i < nnz; ++i)
    {
        x_val[i] = static_cast<T>(i + 1);
        x_ind[i] = i;
    }

    rocsparse_ut::device_vector<T> d_xval(x_val);
    rocsparse_ut::device_vector<I> d_xind(x_ind);
    rocsparse_ut::device_vector<T> d_y(y);

    (void)hipGetLastError();
    // c=2, s=1, y=0 -> x' = 2*x ; y' = -x
    hipLaunchKernelGGL((k_roti<T, I, true>),
                       dim3(1),
                       dim3(BS),
                       0,
                       0,
                       nnz,
                       d_xval.ptr,
                       d_xind.ptr,
                       d_y.ptr,
                       (T)2,
                       (T)1,
                       rocsparse_index_base_zero);
    UT_CHECK_HIP(hipGetLastError());
    UT_CHECK_HIP(hipDeviceSynchronize());

    std::vector<T> want_xval(nnz);
    std::vector<T> want_y(nnz);
    for(I i = 0; i < nnz; ++i)
    {
        want_xval[i] = 2 * x_val[i];
        want_y[i]    = -x_val[i];
    }
    EXPECT_EQ(rocsparse_ut::to_host(d_xval), want_xval);
    EXPECT_EQ(rocsparse_ut::to_host(d_y), want_y);
}

// Identity rotation (c=1, s=0) must leave both vectors unchanged.
TEST(device_level1_roti, identity)
{
    using T = float;
    using I = int32_t;

    std::vector<T>       x_val{3, 4};
    const std::vector<I> x_ind{1, 4};
    std::vector<T>       y{0, 7, 0, 0, 8};

    rocsparse_ut::device_vector<T> d_xval(x_val);
    rocsparse_ut::device_vector<I> d_xind(x_ind);
    rocsparse_ut::device_vector<T> d_y(y);

    (void)hipGetLastError();
    hipLaunchKernelGGL((k_roti<T, I>),
                       dim3(1),
                       dim3(BS),
                       0,
                       0,
                       (I)x_ind.size(),
                       d_xval.ptr,
                       d_xind.ptr,
                       d_y.ptr,
                       (T)1,
                       (T)0,
                       rocsparse_index_base_zero);
    UT_CHECK_HIP(hipGetLastError());
    UT_CHECK_HIP(hipDeviceSynchronize());

    EXPECT_EQ(rocsparse_ut::to_host(d_xval), (std::vector<T>{3, 4}));
    EXPECT_EQ(rocsparse_ut::to_host(d_y), (std::vector<T>{0, 7, 0, 0, 8}));
}

// ---------------------------------------------------------------------------
// doti / dotci : sum of y[x_ind[i]] * x_val[i]  (conj(x_val[i]) for dotci)
//
// Regression tests for AISPARSE-1104. Before the fix, both kernels kept their
// grid-stride index in the 32-bit index type. On the 256 x 256 grid that
// rocsparse_doti and rocsparse_dotci launch, the index wrapped negative once
// nnz exceeded 2147352576 (doti) or 2147418112 (dotci), so the kernels read
// out of bounds or never terminated.
//
// These launch both kernel parts on that grid with 32-bit indices and
// nnz = INT32_MAX. The public API cannot reach the dotci wrap cheaply: spvv
// sends real types to doti even for a conjugate transpose, and complex data of
// this size needs about 26 GB. The int8 data here needs about 10 GiB. It is
// zero except for probes past each old wrap point and at the last element, so
// an exact result shows that the kernel terminated and read each probe once.
// ---------------------------------------------------------------------------
namespace
{
    constexpr uint32_t DOT_DIM    = 256;
    constexpr uint32_t DOTI_LOOPS = 2;

    // A kernel that never terminates cannot be cancelled, and freeing its
    // buffers would block, so a timeout aborts the process.
    void synchronize_or_abort(std::chrono::seconds limit)
    {
        hipEvent_t done;
        UT_CHECK_HIP(hipEventCreateWithFlags(&done, hipEventDisableTiming));
        UT_CHECK_HIP(hipEventRecord(done, 0));

        const auto deadline = std::chrono::steady_clock::now() + limit;
        hipError_t status;
        while((status = hipEventQuery(done)) == hipErrorNotReady)
        {
            if(std::chrono::steady_clock::now() > deadline)
            {
                ADD_FAILURE() << "kernel did not finish within " << limit.count() << " s";
                std::abort();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        UT_CHECK_HIP(status);
        UT_CHECK_HIP(hipEventDestroy(done));
    }

    template <bool CONJ>
    void dot_int32_max_nnz()
    {
        using I = int32_t;
        using X = int8_t;
        using T = int32_t;

        const I      nnz  = std::numeric_limits<I>::max();
        const size_t need = static_cast<size_t>(nnz) * (sizeof(I) + sizeof(X));

        size_t free_bytes  = 0;
        size_t total_bytes = 0;
        if(hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess
           || free_bytes < need + (size_t(256) << 20))
        {
            GTEST_SKIP() << "Insufficient device memory: need about " << (need >> 20) << " MiB";
        }

        rocsparse_ut::device_vector<I> d_xind(static_cast<size_t>(nnz));
        rocsparse_ut::device_vector<X> d_xval(static_cast<size_t>(nnz));
        UT_CHECK_HIP(hipMemset(d_xind.ptr, 0, sizeof(I) * nnz));
        UT_CHECK_HIP(hipMemset(d_xval.ptr, 0, sizeof(X) * nnz));

        // The first index past the old doti wrap, the first past the old dotci
        // wrap, and the last element. Each probe reads its own entry of y, and
        // a missed or repeated probe changes the sum.
        const I              probe[3]     = {2147352576, 2147418112, nnz - 1};
        const X              probe_val[3] = {2, 3, 4};
        const std::vector<X> y{0, 5, 7, 11};

        T expected = 0;
        for(I p = 0; p < 3; ++p)
        {
            const I ind = p + 1;
            UT_CHECK_HIP(hipMemcpy(d_xind.ptr + probe[p], &ind, sizeof(I), hipMemcpyHostToDevice));
            UT_CHECK_HIP(
                hipMemcpy(d_xval.ptr + probe[p], &probe_val[p], sizeof(X), hipMemcpyHostToDevice));
            expected += static_cast<T>(probe_val[p]) * y[ind];
        }

        rocsparse_ut::device_vector<X> d_y(y);
        rocsparse_ut::device_vector<T> d_workspace(DOT_DIM);
        rocsparse_ut::device_vector<T> d_result(1);

        (void)hipGetLastError();
        if(CONJ)
        {
            hipLaunchKernelGGL((rocsparse::dotci_kernel_part1<DOT_DIM, I, X, X, T>),
                               dim3(DOT_DIM),
                               dim3(DOT_DIM),
                               0,
                               0,
                               nnz,
                               d_xval.ptr,
                               d_xind.ptr,
                               d_y.ptr,
                               d_workspace.ptr,
                               rocsparse_index_base_zero);
            hipLaunchKernelGGL((rocsparse::dotci_kernel_part2<DOT_DIM, T>),
                               dim3(1),
                               dim3(DOT_DIM),
                               0,
                               0,
                               d_workspace.ptr,
                               d_result.ptr);
        }
        else
        {
            hipLaunchKernelGGL((rocsparse::doti_kernel_part1<DOT_DIM, DOTI_LOOPS, I, X, X, T>),
                               dim3(DOT_DIM),
                               dim3(DOT_DIM),
                               0,
                               0,
                               nnz,
                               d_xval.ptr,
                               d_xind.ptr,
                               d_y.ptr,
                               d_workspace.ptr,
                               rocsparse_index_base_zero);
            hipLaunchKernelGGL((rocsparse::doti_kernel_part2<DOT_DIM, T>),
                               dim3(1),
                               dim3(DOT_DIM),
                               0,
                               0,
                               d_workspace.ptr,
                               d_result.ptr);
        }
        UT_CHECK_HIP(hipGetLastError());
        synchronize_or_abort(std::chrono::seconds(60));

        EXPECT_EQ(rocsparse_ut::to_host(d_result), std::vector<T>{expected});
    }
} // namespace

TEST(device_level1_doti, int32_max_nnz)
{
    dot_int32_max_nnz<false>();
}

TEST(device_level1_dotci, int32_max_nnz)
{
    dot_int32_max_nnz<true>();
}
