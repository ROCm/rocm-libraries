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
#include <algorithm>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
    // Covers rocblas_internal_rot_launcher_64 on the chunked X path: n above
    // c_ILP64_i32_max with a negative increment, where each chunk's base offset
    // must account for the 32-bit launcher traversing a negative increment from
    // the end of the chunk (see rocblas_rot_kernels_64.cpp for the offset math).
    // Below that n the launcher takes a single 32-bit call and never chunks, so
    // n is at least 2^31 here, roughly 8.6 GB per operand at single precision.
    //
    // The four rows exercise both offset lines independently: increments positive
    // (control), incx negative, incy negative, and both. Detection is twofold. The
    // value check compares every element against BLAS traversal order; two sites
    // per chunk carry distinct values so the expected result depends on the actual
    // x/y pairing, while the full scan confirms the chunk ranges partition the
    // vector. The guard check catches an out-of-bounds write as a guard trip
    // rather than a device fault, which would abort the whole binary.
    //
    // n is exactly 2^31 = 8 * c_i64_grid_X_chunk: the smallest n that chunks, and
    // an even multiple. c_below is the 7-chunk span a lost negative-increment
    // offset reaches below the pointer; c_above is one chunk, bounding an
    // off-by-one that writes (n_64 - 1 - n_base) instead of (n_64 - n - n_base).
    //
    // Valid only against the production constants. A -DROCBLAS_DEV_TEST_ILP64 build
    // sets c_i64_grid_X_chunk to 512 and c_ILP64_i32_max to 0 while c_n stays 2^31,
    // so the guard sizes no longer match the chunk layout; do not run these rows in
    // that mode. The category does not fence them off, since the harness drops a
    // category only when the filter is empty.
    constexpr int64_t c_grid_x_chunk = int64_t(1) << 28; // c_i64_grid_X_chunk
    constexpr int64_t c_n            = int64_t(1) << 31; // 8 * c_grid_x_chunk
    constexpr int64_t c_below        = 7 * c_grid_x_chunk; // 7-chunk guard span below
    constexpr int64_t c_above        = c_grid_x_chunk; // one-chunk guard span above

    constexpr unsigned char c_guard_byte = 0xA5;

    // Elements copied per host staging buffer. The device allocations here reach
    // 16 GiB each, so nothing is ever copied in one piece: a host buffer the size
    // of the device one would fail the test on a machine that has the VRAM but not
    // the RAM, which is a false failure on correct code.
    constexpr size_t c_slice = size_t(1) << 24; // 16 Mi elements, 64 MB at f32

    // Owns a device allocation so that an ASSERT_* between the allocation and the
    // end of the test cannot leak it. ASSERT_* returns from the enclosing function
    // immediately, so a hand-written hipFree further down is not reached, and
    // these allocations are large enough that leaking one would push every later
    // test in the binary into a VRAM skip.
    //
    // The hipMalloc status is kept rather than reduced to a null pointer, so the
    // caller can hand it to CHECK_DEVICE_ALLOCATION and get a skip only on
    // hipErrorOutOfMemory and a failure on anything else. Reading every failure as
    // "not enough VRAM" would drop all coverage of this regression whenever the
    // allocation failed for an unrelated reason, which is the one outcome a
    // regression test must not have.
    template <typename T>
    struct device_buffer
    {
        T*         device = nullptr;
        hipError_t status = hipSuccess;

        explicit device_buffer(size_t count)
        {
            status = (hipMalloc)(&device, count * sizeof(T));
            if(status != hipSuccess)
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

    // The sample sites, and the values placed there.
    //
    // Two sites per chunk: the first element of the chunk, and the element that
    // pairs with it when an increment is reversed. The set is therefore closed
    // under p -> n-1-p, which is what a negative increment does, so a sampled
    // element is always paired with another sampled element rather than with the
    // surrounding fill. Without that closure most samples would be checked against
    // a fill value and a wrong pairing between two chunks could go unnoticed.
    //
    // k in [0,8)  -> k*C,           the head of chunk k
    // k in [8,16) -> n-1-(k-8)*C,   its mirror, the tail of chunk 7-(k-8)
    //
    // The two halves cannot collide: the first are multiples of C and the second
    // are all congruent to -1 mod C.
    //
    // The values are distinct from each other and from both fills, so an element
    // that was never touched, an element rotated twice, and an element rotated
    // against the wrong partner all read differently. They are small integers, so
    // 0.5*v and 2*v and their sum are exact in binary floating point and the
    // comparison is exact rather than tolerance-based.
    constexpr int c_samples = 16;

    int64_t sample_index(int k)
    {
        return k < 8 ? k * c_grid_x_chunk : c_n - 1 - (k - 8) * c_grid_x_chunk;
    }

    // Equivalent to searching sample_index, in constant time. The sites are the
    // multiples of the chunk size and their predecessors, so an element is a
    // sample exactly when its offset within its chunk is the first or the last.
    // The chunk size is a power of two, so the modulo is a mask.
    //
    // This matters because the value check below runs over every one of the 2^31
    // elements, and a sixteen-entry search per element would dominate the
    // test's runtime by orders of magnitude.
    bool is_sample(int64_t p)
    {
        const int64_t within = p & (c_grid_x_chunk - 1);
        return within == 0 || within == c_grid_x_chunk - 1;
    }

    double x_original(int64_t idx)
    {
        if(!is_sample(idx))
            return 1.0; // x fill
        for(int k = 0; k < c_samples; ++k)
            if(sample_index(k) == idx)
                return 10.0 + k; // 10 .. 25
        return 1.0;
    }

    double y_original(int64_t idx)
    {
        if(!is_sample(idx))
            return 2.0; // y fill
        for(int k = 0; k < c_samples; ++k)
            if(sample_index(k) == idx)
                return 100.0 + k; // 100 .. 115
        return 2.0;
    }

    template <typename T>
    void testing_rot_ilp64_chunk(const Arguments& arg)
    {
        // Only the sign is taken from the data; the magnitude must stay 1 so that
        // the operands remain n elements rather than n * |inc|, and must stay at
        // or below c_ILP64_i32_max so the launcher takes the chunked branch.
        const int64_t incx = arg.incx < 0 ? -1 : 1;
        const int64_t incy = arg.incy < 0 ? -1 : 1;

        // Guarded room is needed exactly where an increment is negative. The
        // all-positive row is a control and needs no guard at all, which also makes
        // it the cheapest row and the last to be skipped for want of memory.
        const size_t below_x = incx < 0 ? size_t(c_below) : 0;
        const size_t below_y = incy < 0 ? size_t(c_below) : 0;
        const size_t above_x = incx < 0 ? size_t(c_above) : 0;
        const size_t above_y = incy < 0 ? size_t(c_above) : 0;
        const size_t x_total = below_x + size_t(c_n) + above_x;
        const size_t y_total = below_y + size_t(c_n) + above_y;
        const size_t needed  = (x_total + y_total) * sizeof(T) + (256u << 20);

        // A failed query is a failure, not a skip: it says nothing about the free
        // capacity, and silently skipping on it would hide a broken run. Only the
        // capacity the query reports decides whether the device can supply this.
        size_t free_bytes = 0, total_bytes = 0;
        CHECK_HIP_ERROR(hipMemGetInfo(&free_bytes, &total_bytes));
        if(free_bytes < needed)
            GTEST_SKIP() << LIMITED_VRAM_STRING;

        // The query above is advisory -- it can pass and the allocation still fail,
        // for instance against another process -- so the status of each allocation
        // is checked as well, and only hipErrorOutOfMemory skips.
        device_buffer<T> x_buf(x_total);
        CHECK_DEVICE_ALLOCATION(x_buf.status);
        device_buffer<T> y_buf(y_total);
        CHECK_DEVICE_ALLOCATION(y_buf.status);

        // The pointers rocBLAS is given sit above their lower guard regions.
        T* const dx = x_buf.device + below_x;
        T* const dy = y_buf.device + below_y;

        if(below_x)
            ASSERT_EQ(hipMemset(x_buf.device, c_guard_byte, below_x * sizeof(T)), hipSuccess);
        if(below_y)
            ASSERT_EQ(hipMemset(y_buf.device, c_guard_byte, below_y * sizeof(T)), hipSuccess);
        if(above_x)
            ASSERT_EQ(hipMemset(dx + c_n, c_guard_byte, above_x * sizeof(T)), hipSuccess);
        if(above_y)
            ASSERT_EQ(hipMemset(dy + c_n, c_guard_byte, above_y * sizeof(T)), hipSuccess);

        // (c, s) is deliberately not normalised. With c*c + s*s == 1 a rotation
        // applied twice is still a rotation and preserves magnitude, which is
        // harder to distinguish; with c = 0.5 and s = 2 a second application moves
        // the result well away from the first. Both are exact in binary floating
        // point.
        const T c_val(0.5f), s_val(2.0f);

        {
            // Fill in slices; see c_slice.
            const std::vector<T> hx(c_slice, T(1));
            const std::vector<T> hy(c_slice, T(2));
            for(size_t off = 0; off < size_t(c_n); off += c_slice)
            {
                const size_t count = std::min(c_slice, size_t(c_n) - off);
                ASSERT_EQ(
                    hipMemcpy(dx + off, hx.data(), count * sizeof(T), hipMemcpyHostToDevice),
                    hipSuccess);
                ASSERT_EQ(
                    hipMemcpy(dy + off, hy.data(), count * sizeof(T), hipMemcpyHostToDevice),
                    hipSuccess);
            }
        }

        // Distinct values at the sample sites, written after the fill so they are
        // not overwritten by it.
        for(int k = 0; k < c_samples; ++k)
        {
            const int64_t idx = sample_index(k);
            const T       xv(float(x_original(idx)));
            const T       yv(float(y_original(idx)));
            ASSERT_EQ(hipMemcpy(dx + idx, &xv, sizeof(T), hipMemcpyHostToDevice), hipSuccess);
            ASSERT_EQ(hipMemcpy(dy + idx, &yv, sizeof(T), hipMemcpyHostToDevice), hipSuccess);
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        rocblas_local_handle handle{arg};

        const rocblas_status status
            = rocblas_rot_64<T, T, T>(handle, c_n, dx, incx, dy, incy, &c_val, &s_val);
        ASSERT_EQ(status, rocblas_status_success);
        const hipError_t sync = hipDeviceSynchronize();
        ASSERT_EQ(sync, hipSuccess)
            << "device fault after rot, which indicates an out-of-bounds access beyond the "
               "guarded region: "
            << hipGetErrorString(sync);

        // Scan the guards in slices rather than copying them out whole; each is up
        // to 7 GiB.
        auto expect_guard_clean
            = [](const T* base, size_t elems, const char* which, bool below) {
                  if(!elems)
                      return;
                  std::vector<unsigned char> host(c_slice * sizeof(T));
                  size_t                     differing = 0;
                  size_t                     first     = elems;
                  for(size_t off = 0; off < elems; off += c_slice)
                  {
                      const size_t count = std::min(c_slice, elems - off);
                      ASSERT_EQ(hipMemcpy(host.data(),
                                          base + off,
                                          count * sizeof(T),
                                          hipMemcpyDeviceToHost),
                                hipSuccess);
                      for(size_t i = 0; i < count * sizeof(T); ++i)
                      {
                          if(host[i] != c_guard_byte)
                          {
                              if(first == elems)
                                  first = off + i / sizeof(T);
                              ++differing;
                          }
                      }
                  }
                  // Reported relative to the pointer rocBLAS was given, so a
                  // below-guard trip reads as a negative element index rather
                  // than as an offset into the guard, which would look like
                  // element 0 of the vector.
                  EXPECT_EQ(differing, size_t(0))
                      << differing << " guard byte(s) " << (below ? "below " : "above ") << which
                      << " were written; the furthest is element "
                      << (below ? -int64_t(elems - first) : c_n + int64_t(first))
                      << ". Below the vector means a chunk offset lost its "
                         "negative-increment compensation; above it means the offset "
                         "overran the top by up to one chunk.";
              };

        expect_guard_clean(x_buf.device, below_x, "x", true);
        expect_guard_clean(y_buf.device, below_y, "y", true);
        expect_guard_clean(dx + c_n, above_x, "x", false);
        expect_guard_clean(dy + c_n, above_y, "y", false);

        // Expected values follow from BLAS traversal order rather than being
        // hard-coded, so they cannot drift out of step with the increment signs.
        //
        // A negative increment walks its operand from the far end, so physical
        // element p of x carries logical element i = n-1-p, and the y it is paired
        // with is whichever physical element carries the same logical index.
        auto expected = [&](int64_t p, bool want_x) {
            if(want_x)
            {
                // x at physical p holds logical i; its partner is y at physical q.
                const int64_t i = incx < 0 ? c_n - 1 - p : p;
                const int64_t q = incy < 0 ? c_n - 1 - i : i;
                return 0.5 * x_original(p) + 2.0 * y_original(q);
            }
            // y at physical p holds logical j; its partner is x at physical r.
            const int64_t j = incy < 0 ? c_n - 1 - p : p;
            const int64_t r = incx < 0 ? c_n - 1 - j : j;
            return 0.5 * y_original(p) - 2.0 * x_original(r);
        };

        // Every element is checked, not a sample of them.
        //
        // The sample sites exist to make the *pairing* observable, and sixteen of
        // them are enough for that, but they are not enough to show that the chunk
        // ranges partition the vector: getting all sixteen right is consistent with
        // skipping or double-covering elements between them. Away from the sample
        // sites both operands are constant, so the expected value is uniform --
        // 0.5*1 + 2*2 = 4.5 for x and 0.5*2 - 2*1 = -1 for y, both exact -- which
        // makes the full check no harder to write than the sampled one. This holds
        // for every increment-sign combination because the sample set is closed
        // under p -> n-1-p, so a non-sample element is always paired with another
        // non-sample element.
        //
        // The cost is a device-to-host pass over each operand, reusing the same
        // slice-sized staging buffer as the guard scan. That is time on a nightly
        // row, not memory, and coverage is worth more than the time.
        auto expect_vector_correct = [&](const T* base, bool want_x, const char* which) {
            // Away from the sample sites the expectation is the same for every
            // element, so it is computed once and the inner loop is a single
            // comparison. Only the sixteen sample sites take the derived path.
            const double   uniform = want_x ? 0.5 * 1.0 + 2.0 * 2.0 : 0.5 * 2.0 - 2.0 * 1.0;
            std::vector<T> host(c_slice);
            size_t         wrong     = 0;
            int64_t        first_bad = -1;
            double         first_got = 0.0, first_want = 0.0;
            for(int64_t off = 0; off < c_n; off += int64_t(c_slice))
            {
                const size_t count = std::min(c_slice, size_t(c_n - off));
                ASSERT_EQ(
                    hipMemcpy(host.data(), base + off, count * sizeof(T), hipMemcpyDeviceToHost),
                    hipSuccess);
                for(size_t e = 0; e < count; ++e)
                {
                    const int64_t p    = off + int64_t(e);
                    const double  want = is_sample(p) ? expected(p, want_x) : uniform;
                    if(double(host[e]) != want)
                    {
                        if(first_bad < 0)
                        {
                            first_bad  = p;
                            first_got  = double(host[e]);
                            first_want = want;
                        }
                        ++wrong;
                    }
                }
            }
            EXPECT_EQ(wrong, size_t(0))
                << wrong << " of " << c_n << " elements of " << which
                << " are wrong; the first is " << which << "[" << first_bad << "], which is "
                << first_got << " where " << first_want
                << " is expected. A value equal to the original means the chunk covering "
                   "that element addressed a different range; any other value means it was "
                   "paired with the wrong element of the other operand.";
        };

        expect_vector_correct(dx, true, "x");
        expect_vector_correct(dy, false, "y");
    }

    bool is_rot_ilp64_chunk_function(const std::string& fn)
    {
        return fn == "rot_ilp64_negative_inc_chunk";
    }

    template <typename, typename = void>
    struct rot_ilp64_chunk_fun : rocblas_test_invalid
    {
    };

    // Single precision only. Unlike the sibling regressions this is not a free
    // choice -- it is what the API and the footprint allow.
    //
    //   rocblas_half -- does not exist for rot. There is no rocblas_hrot_64 in the
    //                   public API, so half is not an option at any size.
    //   double       -- the element count is fixed at 2^31 by the branch
    //                   condition, so double precision doubles this test to 64 GiB
    //                   for the both-negative row, which does not fit a single
    //                   MI250 GCD alongside anything else. It would skip
    //                   everywhere and protect nothing.
    //   complex      -- csrot and zdrot reach the same launcher and the same index
    //                   arithmetic, at two and four times the footprint
    //                   respectively, so they cost more and cover nothing further.
    //
    // The arithmetic under test is host-side and identical for every type, so a
    // single type is sufficient coverage of it.
    template <typename T>
    struct rot_ilp64_chunk_fun<T, std::enable_if_t<std::is_same_v<T, float>>> : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            testing_rot_ilp64_chunk<T>(arg);
        }
    };

    struct rot_ilp64_chunk_gtest : RocBLAS_Test<rot_ilp64_chunk_gtest, rot_ilp64_chunk_fun>
    {
        static bool type_filter(const Arguments& arg)
        {
            return rocblas_simple_dispatch<type_filter_functor>(arg);
        }

        static bool function_filter(const Arguments& arg)
        {
            return is_rot_ilp64_chunk_function(arg.function);
        }

        static std::string name_suffix(const Arguments& arg)
        {
            return RocBLAS_TestName<rot_ilp64_chunk_gtest>(arg.name);
        }
    };

    // PrintToStringParamName already prefixes each test name with the YAML
    // category, so this name carries the BLAS level rather than repeating one, as
    // the other nine blas1 suites do. Naming it after a category would make that
    // category's filter match these rows whatever the YAML says.
    TEST_P(rot_ilp64_chunk_gtest, blas1)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<rot_ilp64_chunk_fun>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(rot_ilp64_chunk_gtest)

} // namespace
