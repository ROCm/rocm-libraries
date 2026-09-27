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
    // Regression for the dot_64 single-block negative-increment shift.
    //
    // rocblas_internal_dot_launcher already walks a negative increment from the
    // end of the data (rocblas_dot_kernels.hpp:409-410). The src64 single-block
    // path computed the same shift itself and passed the result into those
    // parameters, so it was applied twice; the second line also tested and scaled
    // by incx_64 where it should have used incy_64.
    //
    // Reaching the defective path requires both:
    //   * |incx| or |incy| above c_ILP64_i32_max, to take the 64-bit-increment
    //     branch rather than the truncated 32-bit call, and
    //   * n at or below rocblas_dot_one_block_threshold<T>() (31000 for float),
    //     to take the single-block path rather than the multi-block path, whose
    //     own shift is correct.
    //
    // Only incx_64 < 0 triggers anything. With incx_64 > 0 both lines collapse to
    // the plain offsets and the launcher does the right thing regardless of incy,
    // which is why the existing { incx: 1, incy: -1 } coverage passes today and
    // must continue to pass: fixing only the incx_64/incy_64 typo while leaving
    // the double shift would break precisely that case.
    //
    // Memory. One vector must span (n-1)*|inc| + 1 elements, and |inc| has to
    // exceed 2^31-1, so roughly 2^31 elements are unavoidable -- about 8.6 GB at
    // single precision. Two choices keep it to that minimum rather than more:
    //
    //   * n is 2, the smallest value that shifts at all, since the shift scales
    //     with (n-1). A larger n multiplies the allocation directly, which is why
    //     a case at n in the thousands cannot be written at this increment.
    //   * the large increment is placed on incy while incx is -1. Putting a large
    //     *negative* increment on incx instead would double a 2^31 shift to 2^32
    //     and read about 8.6 GB past the end, which faults rather than landing in
    //     a tail element. Here the doubled shift overruns by exactly one element.
    //
    // Lower precision is not an option even though rocblas_hdot_64 exists: the
    // products below would have to stay inside the half range, and a wrong read
    // can produce a value that saturates to infinity, so the test would stop
    // discriminating for a reason unrelated to the defect.
    //
    // dot only reads its operands, so a byte guard cannot detect this. The tail
    // elements instead hold known values chosen to make the incorrect result a
    // specific, recognisable number.

    // Smallest magnitude above c_ILP64_i32_max, which minimises the allocation.
    constexpr int64_t c_incy_64bit = 2147483648;

    // Owns a device allocation so that an ASSERT_* between the allocation and the
    // end of the test cannot leak it. ASSERT_* returns from the enclosing function
    // immediately, so a hand-written hipFree further down is not reached, and
    // these allocations are large enough that leaking one would push every later
    // test in the binary into a VRAM skip.
    template <typename T>
    struct device_buffer
    {
        T* device = nullptr;

        explicit device_buffer(size_t count)
        {
            if((hipMalloc)(&device, count * sizeof(T)) != hipSuccess)
                device = nullptr;
        }

        ~device_buffer()
        {
            if(device)
                (void)(hipFree)(device);
        }

        device_buffer(const device_buffer&)            = delete;
        device_buffer& operator=(const device_buffer&) = delete;
    };

    template <typename T>
    void testing_dot_ilp64_chunk(const Arguments& arg)
    {
        constexpr int64_t n = 2;

        // Signs come from the data; magnitudes are fixed. incx stays at 1 so x
        // costs two elements, and incy carries the 64-bit magnitude that selects
        // the branch.
        //
        // Three combinations are needed, and the second is the important one:
        //
        //   incx < 0, incy > 0  detector. shiftx is doubled and shifty is shifted
        //                       when it should not be, reading y one element past
        //                       the end.
        //   incx > 0, incy < 0  control, and the reason it exists. Both caller
        //                       lines collapse to the plain offsets, so this case
        //                       is correct before the fix and must stay correct
        //                       after it. Fixing only the incx_64/incy_64 typo
        //                       while leaving the double shift would give shifty
        //                       twice the intended offset and break exactly this
        //                       case, so without this row that mistake would look
        //                       like a clean pass.
        //   incx < 0, incy < 0  both lines wrong at once.
        const int64_t incx = arg.incx < 0 ? -1 : 1;
        const int64_t incy = arg.incy < 0 ? -c_incy_64bit : c_incy_64bit;

        // y must be addressable at (n-1)*|incy| == 2147483648, so it needs
        // 2147483649 live elements, plus one tail element for the overrun.
        const size_t y_live  = size_t(c_incy_64bit) + 1;
        const size_t y_total = y_live + 1;
        const size_t x_live  = 2;
        const size_t x_total = x_live + 1;

        size_t free_bytes = 0, total_bytes = 0;
        if(hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess
           || free_bytes < y_total * sizeof(T) + (64u << 20))
            GTEST_SKIP() << LIMITED_VRAM_STRING;

        device_buffer<T> x_buf(x_total);
        device_buffer<T> y_buf(y_total);
        if(!x_buf.device || !y_buf.device)
            GTEST_SKIP() << LIMITED_VRAM_STRING;
        T* const dx = x_buf.device;
        T* const dy = y_buf.device;

        // Values are small integers, so every product and the sum are exact in
        // single precision and the comparison below is exact rather than
        // tolerance-based. They are also distinct and mutually coprime, so any
        // pair of wrong operands produces a value that cannot coincide with the
        // right one.
        //
        // x: [3, 5] live, 100 in the tail.
        // y: y[0] = 7, y[1] = 13, y[2147483648] = 11, y[2147483649] = 17.
        //
        // The tail values are what a pre-fix read past the end returns, so the
        // incorrect result is a specific recognisable number rather than whatever
        // happened to follow the allocation. Worked through:
        //
        //   incx < 0, incy > 0  correct 5*7 + 3*11 = 68; pre-fix the caller passed
        //                       shiftx = shifty = 1, the launcher advanced shiftx
        //                       to 2, giving x[2]*y[1] + x[1]*y[2147483649]
        //                       = 100*13 + 5*17 = 1385.
        //   incx > 0, incy < 0  correct 3*11 + 5*7 = 68, unchanged by the fix.
        //   incx < 0, incy < 0  correct 5*11 + 3*7 = 76; pre-fix
        //                       100*17 + 5*13 = 1765.
        const std::vector<T> hx{T(3), T(5), T(100)};
        ASSERT_EQ(hipMemcpy(dx, hx.data(), x_total * sizeof(T), hipMemcpyHostToDevice),
                  hipSuccess);
        ASSERT_EQ(hipMemset(dy, 0, y_total * sizeof(T)), hipSuccess);

        const T y_head[2] = {T(7), T(13)};
        ASSERT_EQ(hipMemcpy(dy, y_head, sizeof(y_head), hipMemcpyHostToDevice), hipSuccess);
        const T y_far[2] = {T(11), T(17)};
        ASSERT_EQ(hipMemcpy(dy + c_incy_64bit, y_far, sizeof(y_far), hipMemcpyHostToDevice),
                  hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        rocblas_local_handle handle{arg};

        T result(0);
        const rocblas_status status = rocblas_dot_64<T>(handle, n, dx, incx, dy, incy, &result);

        EXPECT_EQ(status, rocblas_status_success);
        const hipError_t sync = hipDeviceSynchronize();

        // A fault here is itself a failure worth naming: the pre-fix read is out
        // of bounds, and whether it faults or returns a wrong value depends on
        // what follows the allocation.
        ASSERT_EQ(sync, hipSuccess) << "device fault after dot, which indicates an out-of-bounds "
                                       "access: "
                                    << hipGetErrorString(sync);

        // Derived from BLAS traversal order rather than hard-coded per row, so the
        // expectation cannot drift out of step with the data. A negative increment
        // walks its operand from the far end, which reverses that operand's
        // logical order; mixed signs therefore reverse the pairing, and that is
        // correct behaviour rather than a symptom.
        const double x0 = incx < 0 ? 5.0 : 3.0; // x[1] : x[0]
        const double x1 = incx < 0 ? 3.0 : 5.0; // x[0] : x[1]
        const double y0 = incy < 0 ? 11.0 : 7.0; // y[2147483648] : y[0]
        const double y1 = incy < 0 ? 7.0 : 11.0; // y[0] : y[2147483648]
        const double expected = x0 * y0 + x1 * y1;

        EXPECT_EQ(double(result), expected)
            << "dot returned " << double(result) << " where " << expected
            << " is correct for incx " << incx << " and incy " << incy
            << "; 1385 or 1765 means the negative-increment shift was applied twice and y was "
               "read one element past the end";
    }

    bool is_dot_ilp64_chunk_function(const std::string& fn)
    {
        return fn == "dot_ilp64_single_block_chunk";
    }

    template <typename, typename = void>
    struct dot_ilp64_chunk_fun : rocblas_test_invalid
    {
    };

    // Single precision only, and the restriction is deliberate rather than
    // incidental. The defect is host-side index arithmetic in the launcher,
    // computed identically for every type, so one type exercises it fully and
    // each additional type buys coverage of nothing while costing another 2^31
    // element allocation.
    //
    //   double        -- the same shape is 17.2 GB rather than 8.6 GB, because the
    //                    element count is fixed by the increment and cannot be
    //                    reduced. It would skip for want of memory on much of the
    //                    fleet, which turns a regression test into a silent no-op.
    //   rocblas_half  -- available as rocblas_hdot_64, and rejected on numerical
    //                    grounds. The incorrect result here is a product of
    //                    sentinel values; in half the finite range ends at 65504,
    //                    so a wrong read can saturate to infinity, and every
    //                    expected value would have to be exactly representable.
    //                    The test would then be constrained, and could stop
    //                    discriminating, for reasons that have nothing to do with
    //                    the defect under test.
    //   complex       -- same host arithmetic again, at two or four times the
    //                    footprint.
    template <typename T>
    struct dot_ilp64_chunk_fun<T, std::enable_if_t<std::is_same_v<T, float>>> : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            testing_dot_ilp64_chunk<T>(arg);
        }
    };

    struct dot_ilp64_chunk_gtest : RocBLAS_Test<dot_ilp64_chunk_gtest, dot_ilp64_chunk_fun>
    {
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<type_filter_functor>(arg);
        }

        static bool function_filter(const Arguments& arg)
        {
            return is_dot_ilp64_chunk_function(arg.function);
        }

        static std::string name_suffix(const Arguments& arg)
        {
            return RocBLAS_TestName<dot_ilp64_chunk_gtest>(arg.name);
        }
    };

    TEST_P(dot_ilp64_chunk_gtest, nightly)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<dot_ilp64_chunk_fun>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(dot_ilp64_chunk_gtest)

} // namespace
