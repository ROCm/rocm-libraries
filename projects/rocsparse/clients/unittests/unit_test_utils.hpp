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

    // ----- device -> host readback ------------------------------------------
    // Copies `count` elements from a device pointer into a host vector so tests
    // can assert on the *numerical output* of a routine (not just its status).
    template <typename T>
    std::vector<T> to_host(const T* device_ptr, size_t count)
    {
        std::vector<T> host(count);
        if(count > 0 && device_ptr != nullptr)
        {
            (void)hipMemcpy(host.data(), device_ptr, count * sizeof(T), hipMemcpyDeviceToHost);
        }
        return host;
    }

    template <typename T>
    std::vector<T> to_host(const device_vector<T>& d)
    {
        return to_host<T>(d.ptr, d.n);
    }

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

    // ========================================================================
    // Device test-kernel harness (shared by the internal-routine device tests)
    // ========================================================================
    //
    // rocSPARSE's device building blocks (block/warp collectives, shuffles,
    // segmented reductions, ...) are ROCSPARSE_DEVICE_ILF functions declared in
    // library/src/include/rocsparse_common.hpp and the level-* device headers.
    // They are meant to be *called from inside a kernel*, so a unit test cannot
    // call them directly from the host. The proven pattern (see
    // unit_test_device_level1.cpp) is:
    //
    //   1. Write a thin __global__ wrapper in your test TU that calls the
    //      internal device function and stores its result into a device buffer.
    //   2. Launch that wrapper on ONE block (or ONE warp) with tiny inputs.
    //   3. Read the result back with rocsparse_ut::to_host and assert on it.
    //
    // The helpers below remove the launch/synchronize boilerplate. They are
    // header-only and only compiled when the including TU is built as HIP device
    // code (the rocsparse-unit-test-device target); on the host-only
    // rocsparse-unit-test target only the runtime warp-size accessor is defined.
    //
    // ------------------------------------------------------------------------
    // Warp-size accessors
    // ------------------------------------------------------------------------
    // rocSPARSE's warp collectives are templated on the wavefront size (32 or
    // 64). Tests must NOT hard-code or skip a wavefront: they instantiate the
    // building block for BOTH 32 and 64 (so the wave64 instantiation is compiled
    // and validated by CI's wave64 parts, gfx94x/gfx950) and, at run time,
    // launch the instantiation matching the active device (see
    // launch_warp_by_size / device_warp_size below).
    //
    // `device_warp_size()` (below) is the runtime wavefront size of the active
    // device and is the value tests dispatch on.

    // Runtime wavefront size of the active HIP device (32 on gfx1201, 64 on
    // wave64 parts). Returns 0 if the device cannot be queried.
    inline int device_warp_size()
    {
        int             dev   = 0;
        hipDeviceProp_t props = {};
        if(hipGetDevice(&dev) != hipSuccess)
            return 0;
        if(hipGetDeviceProperties(&props, dev) != hipSuccess)
            return 0;
        return props.warpSize;
    }

#ifdef __HIPCC__
    // ------------------------------------------------------------------------
    // Single-block / single-warp launch helpers
    // ------------------------------------------------------------------------
    // Launch `kernel` (a __global__ wrapper, or any callable usable with
    // hipLaunchKernelGGL) on exactly ONE block of `block_dim` threads, forward
    // `args...`, then hipDeviceSynchronize(). Returns the first non-success
    // hipError_t (launch error or sync error), else hipSuccess. Pair with
    // rocsparse_ut::to_host to read results back.
    //
    // Usage (inside a device test TU):
    //
    //   __global__ void k_reduce(const float* in, float* out)
    //   { *out = rocsparse::blockreduce_sum<256>(threadIdx.x, in[threadIdx.x]); }
    //
    //   rocsparse_ut::device_vector<float> d_in(host_in), d_out(size_t{1});
    //   ASSERT_EQ(rocsparse_ut::launch_single_block(k_reduce, 256, d_in.ptr, d_out.ptr),
    //             hipSuccess);
    //   EXPECT_FLOAT_EQ(rocsparse_ut::to_host(d_out)[0], expected);
    template <typename Kernel, typename... Args>
    hipError_t launch_single_block(Kernel kernel, unsigned int block_dim, Args... args)
    {
        hipLaunchKernelGGL(kernel, dim3(1), dim3(block_dim), 0, 0, args...);
        const hipError_t launch = hipGetLastError();
        if(launch != hipSuccess)
            return launch;
        return hipDeviceSynchronize();
    }

    // Launch `kernel` on exactly ONE warp (block of device_warp_size() threads).
    // Convenient for warp-collective wrappers (wfreduce_*, shfl_*, ...).
    template <typename Kernel, typename... Args>
    hipError_t launch_single_warp(Kernel kernel, Args... args)
    {
        const int ws = device_warp_size();
        return launch_single_block(kernel, static_cast<unsigned int>(ws > 0 ? ws : 32), args...);
    }

    // ------------------------------------------------------------------------
    // Runtime wavefront-size dispatch for warp collectives
    // ------------------------------------------------------------------------
    // `k32` and `k64` are the WFSIZE==32 and WFSIZE==64 instantiations of a
    // warp-collective __global__ wrapper. BOTH are referenced here, so both are
    // compiled: CI's wave64 parts (gfx94x/gfx950) compile AND run the 64-lane
    // path, this wave32 card compiles both and runs the 32-lane path. At run
    // time we launch exactly one wavefront of the *device's* width and dispatch
    // to the matching instantiation -- no wavefront is skipped. Returns
    // hipErrorInvalidValue if the device reports an unsupported wavefront size.
    template <typename Kernel32, typename Kernel64, typename... Args>
    hipError_t launch_warp_by_size(Kernel32 k32, Kernel64 k64, Args... args)
    {
        const int ws = device_warp_size();
        if(ws == 64)
            return launch_single_block(k64, 64u, args...);
        if(ws == 32)
            return launch_single_block(k32, 32u, args...);
        return hipErrorInvalidValue;
    }
#endif // __HIPCC__
} // namespace rocsparse_ut
