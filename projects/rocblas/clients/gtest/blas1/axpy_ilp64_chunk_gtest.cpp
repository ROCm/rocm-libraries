/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#include "client_utility.hpp"
#include "rocblas.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_test.hpp"
#include "type_dispatch.hpp"
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
    // Regression for the axpy_64 chunked batch launch. The 64-bit-increment path
    // in rocblas_internal_axpy_launcher_64 chunks the batch on the host and sizes
    // grid.z from the per-chunk count, but passed the *global* batch_count as the
    // kernel argument. rocblas_axpy_kernel grid-strides its batch loop by
    // c_YZ_grid_launch_limit (65535), so a bound larger than the chunk makes each
    // chunk continue into batch items the next chunk also covers, and past the end
    // of the batch entirely on the last chunk.
    //
    // A reference comparison is a poor detector here: with n == 1 only element
    // zero of each item is touched, so a narrowing LP64 reference still produces
    // the right answer and the out-of-bounds write is invisible to it. This test
    // therefore computes the expected result analytically and checks a guard.
    //
    // batch_count notes:
    //   65535 -- control. Two host chunks (65520 + 15), but 0 + 65535 is not
    //            < 65535, so neither chunk's sweep repeats. Passes pre-fix. Its
    //            job is to pin the threshold at 65536 rather than at the host
    //            chunk size of 65520, which is a tempting misreading.
    //   65536 -- smallest detector. Chunk 0 additionally processes local item
    //            65535 (global 65535), which chunk 1 also processes as its local
    //            item 15, so global 65535 receives alpha*x twice. Chunk 1's
    //            blockIdx.z == 0 also reaches local 65535, i.e. global 131055,
    //            which is past the end of a 65536-item batch.
    //   131070 -- three host chunks (65520 + 65520 + 30), so two of them are
    //            saturated: tens of thousands of doubled items and a deeper
    //            overrun than 65536 produces.
    //
    // Every batch_count is run in both pointer modes, because the launcher has a
    // separate kernel launch for each and the defect was in both.
    //
    // n is 1 deliberately. The defect is in the batch dimension, so the element
    // dimension only needs to be non-empty, and n == 1 keeps the operands at a
    // few megabytes instead of the gigabytes a large 64-bit increment would
    // otherwise imply: the span of one item is (n-1)*|incx| + 1.

    // Must exceed c_i32_max so the launcher takes its 64-bit-increment branch.
    // With n == 1 the value is never used to stride, so it costs no memory.
    constexpr int64_t c_incx_64bit = 2147483649;

    // Elements of headroom past the batch, sized to contain the deepest overrun
    // so it is caught as a guard trip rather than a fault.
    //
    // The furthest item a chunk reaches is (chunk base) + (grid.z - 1) + 65535.
    // At batch_count 65536 that is 65520 + 0 + 65535 = 131055, i.e. 65519 items
    // past the end; at 131070 it is 131040 + 29 + 65535 = 196604, i.e. 65534 past
    // the end. One sweep stride therefore bounds the overrun for any batch_count,
    // because a second stride would exceed the global bound and stop the loop.
    constexpr int64_t c_tail_elements = 65536;

    constexpr unsigned char c_guard_byte = 0xA5;

    template <typename T>
    struct padded_vector
    {
        T*     device  = nullptr;
        size_t n_live  = 0;
        size_t n_total = 0;
        // Retained so the caller can tell a VRAM shortfall, which is a skip, from
        // any other allocation failure, which is a test failure.
        hipError_t alloc_status = hipSuccess;

        padded_vector(size_t live, size_t tail)
            : n_live(live)
            , n_total(live + tail)
        {
            alloc_status = (hipMalloc)(&device, n_total * sizeof(T));
            if(alloc_status != hipSuccess)
                device = nullptr;
        }

        ~padded_vector()
        {
            if(device)
                (void)(hipFree)(device);
        }

        padded_vector(const padded_vector&)            = delete;
        padded_vector& operator=(const padded_vector&) = delete;

        size_t tail_bytes() const
        {
            return (n_total - n_live) * sizeof(T);
        }

        T* tail()
        {
            return device + n_live;
        }
    };

    // Reports how many guard bytes moved and where, in the manner of
    // device_vector's guard check, because "it was overwritten" is much less
    // useful than how far past the end and by how much.
    template <typename T>
    void expect_tail_clean(padded_vector<T>& v, const char* which)
    {
        std::vector<unsigned char> host(v.tail_bytes());
        ASSERT_EQ(hipMemcpy(host.data(), v.tail(), v.tail_bytes(), hipMemcpyDeviceToHost),
                  hipSuccess);

        size_t differing = 0;
        size_t first     = host.size();
        for(size_t i = 0; i < host.size(); ++i)
        {
            if(host[i] != c_guard_byte)
            {
                if(first == host.size())
                    first = i;
                ++differing;
            }
        }

        EXPECT_EQ(differing, size_t(0))
            << differing << " guard byte(s) past the end of " << which << " differ; the first is "
            << first << " byte(s) past the " << v.n_live * sizeof(T)
            << " byte batch, i.e. batch item " << (first / sizeof(T));
    }

    template <typename T>
    void testing_axpy_ilp64_chunk(const Arguments& arg)
    {
        const int64_t batch_count = arg.batch_count;
        ASSERT_GT(batch_count, 0);

        const size_t live = size_t(batch_count);
        const T      alpha_value(1);

        padded_vector<T> x(live, size_t(c_tail_elements));
        padded_vector<T> y(live, size_t(c_tail_elements));
        CHECK_DEVICE_ALLOCATION(x.alloc_status);
        CHECK_DEVICE_ALLOCATION(y.alloc_status);

        // x is 1 across the batch. Its tail is 1 as well, rather than a guard
        // pattern: the overrunning launch reads x before it writes y, so the tail
        // has to be a defined value for the resulting write to be deterministic,
        // and 1 keeps the value written equal to the value a correct launch would
        // write. That is deliberate -- it means the guard trip on y proves an
        // out-of-bounds *write* occurred, and is not an artifact of x's tail
        // holding something exotic.
        const std::vector<T> hx(x.n_total, T(1));
        ASSERT_EQ(hipMemcpy(x.device, hx.data(), x.n_total * sizeof(T), hipMemcpyHostToDevice),
                  hipSuccess);

        // y is 0 across the batch; only its tail carries the guard pattern.
        ASSERT_EQ(hipMemset(y.device, 0, live * sizeof(T)), hipSuccess);
        ASSERT_EQ(hipMemset(y.tail(), c_guard_byte, y.tail_bytes()), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        rocblas_local_handle handle{arg};

        // The launcher has a separate kernel launch per pointer mode and the
        // defect was in both, so the mode is data and both are exercised. Without
        // this the handle keeps the rocBLAS default of host mode and a fix that
        // corrected only the host launch would pass every row.
        //
        // rocblas_local_handle does not apply arg.pointer_mode_device itself; the
        // suites that care set the mode explicitly, as here.
        // Always one element rather than zero when unused: hipMalloc(0) is a
        // needless special case for the sake of a handful of bytes.
        padded_vector<T> d_alpha(1, 0);
        const T*         alpha_ptr = &alpha_value;
        if(arg.pointer_mode_device)
        {
            CHECK_DEVICE_ALLOCATION(d_alpha.alloc_status);
            ASSERT_EQ(hipMemcpy(d_alpha.device, &alpha_value, sizeof(T), hipMemcpyHostToDevice),
                      hipSuccess);
            ASSERT_EQ(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device),
                      rocblas_status_success);
            alpha_ptr = d_alpha.device;
        }
        else
        {
            ASSERT_EQ(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host),
                      rocblas_status_success);
        }

        // stride 1 with n == 1 packs the batch contiguously, so batch item i is
        // element i and an overrun lands in the tail rather than inside the batch.
        const rocblas_status status = rocblas_axpy_strided_batched_64<T>(handle,
                                                                        1,
                                                                        alpha_ptr,
                                                                        x.device,
                                                                        c_incx_64bit,
                                                                        1,
                                                                        y.device,
                                                                        c_incx_64bit,
                                                                        1,
                                                                        batch_count);
        EXPECT_EQ(status, rocblas_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        std::vector<T> hy(live);
        ASSERT_EQ(hipMemcpy(hy.data(), y.device, live * sizeof(T), hipMemcpyDeviceToHost),
                  hipSuccess);

        // Every item must have had alpha*x applied exactly once: 0 + 1*1 == 1.
        // A doubled item reads 2. Report the first few rather than one line per
        // item, since a saturated second sweep can affect tens of thousands.
        size_t  wrong     = 0;
        int64_t first_bad = -1;
        for(int64_t i = 0; i < batch_count; ++i)
        {
            if(hy[i] != T(1))
            {
                if(first_bad < 0)
                    first_bad = i;
                ++wrong;
            }
        }
        EXPECT_EQ(wrong, size_t(0)) << wrong << " of " << batch_count
                                    << " batch items did not receive alpha*x exactly once; the "
                                       "first is item "
                                    << first_bad << ", which holds "
                                    << (first_bad >= 0 ? double(hy[first_bad]) : 0.0)
                                    << " where 1 is expected (2 means it was applied twice)";

        expect_tail_clean(y, "y");
    }

    bool is_axpy_ilp64_chunk_function(const std::string& fn)
    {
        return fn == "axpy_strided_batched_ilp64_chunk";
    }

    // By default this test does not apply to any type.
    template <typename, typename = void>
    struct axpy_ilp64_chunk_fun : rocblas_test_invalid
    {
    };

    // Single and double precision, and the choice is deliberate. The defect is
    // host-side arithmetic in the launcher -- the kernel argument disagreeing with
    // grid.z -- which is computed identically for every type. One type therefore
    // exercises it fully; the second is cheap here (n == 1 keeps operands at a few
    // megabytes) and guards against the launcher later being specialised per type
    // without this coverage following.
    //
    //   rocblas_half -- excluded on purpose, not for cost. A half axpy routes
    //                   through a separate specialisation in the same launcher
    //                   (the mod-8 vectorised path), whose own batch indexing is
    //                   inconsistent with the grid it is launched on. Including
    //                   half here would conflate two independent defects in one
    //                   regression, so that a failure would no longer identify
    //                   which one regressed. That path warrants its own ticket and
    //                   its own test.
    //   complex      -- exercises the same host arithmetic a third time and adds
    //                   no coverage of it.
    template <typename T>
    struct axpy_ilp64_chunk_fun<
        T,
        std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            testing_axpy_ilp64_chunk<T>(arg);
        }
    };

    struct axpy_ilp64_chunk_gtest
        : RocBLAS_Test<axpy_ilp64_chunk_gtest, axpy_ilp64_chunk_fun>
    {
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<type_filter_functor>(arg);
        }

        static bool function_filter(const Arguments& arg)
        {
            return is_axpy_ilp64_chunk_function(arg.function);
        }

        static std::string name_suffix(const Arguments& arg)
        {
            // The precision is part of the name because the rows are otherwise
            // identical across types; without it the two instances differ only by
            // the counter the harness appends to break the tie, and a failure does
            // not say which precision failed.
            return RocBLAS_TestName<axpy_ilp64_chunk_gtest>(arg.name)
                   << '_' << rocblas_datatype2string(arg.a_type) << "_bc_"
                   << arg.batch_count;
        }
    };

    TEST_P(axpy_ilp64_chunk_gtest, pre_checkin)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<axpy_ilp64_chunk_fun>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(axpy_ilp64_chunk_gtest)

} // namespace
