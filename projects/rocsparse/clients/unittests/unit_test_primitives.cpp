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
// Device (GPU) unit tests for the internal rocsparse::primitives building
// blocks (radix sort, scan, reduce, run-length-encode).
//
// These primitives are template functions declared in
// library/src/include/rocsparse_primitives.hpp and *defined* + explicitly
// instantiated in library/src/primitives/*.cpp. librocsparse is built with
// hidden symbol visibility, so those instantiations are NOT exported from the
// shared object and cannot be reached by linking roc::rocsparse. To exercise
// them in isolation we compile the primitive translation units directly into
// this test binary (the same "compile the library .cpp into the test target"
// mechanism already used by rocsparse-unit-test for host-pure utilities). See
// ROCSPARSE_UNIT_TEST_DEVICE_SOURCES in clients/unittests/CMakeLists.txt.
//
// Each test drives a single primitive with a tiny (<= 8 element) input and a
// hand-computed expected output, copying the result back to the host for an
// exact comparison. The primitives take a rocsparse_handle plus device buffers
// and follow the _buffer_size -> allocate -> execute convention.
//
// This TU is compiled into rocsparse-unit-test-device (links hip::device) and
// must run on a GPU (via the serializer, e.g.
// HIP_VISIBLE_DEVICES=0 gpu-run ./rocsparse-unit-test-device).
//
#include "rocsparse_primitives.hpp"

#include "unit_test_utils.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace
{
    using rocsparse_ut::device_vector;

    namespace prim = rocsparse::primitives;

    // Fixture owning a rocsparse_handle; the "Primitive" suite prefix lets CI
    // select these with --gtest_filter='Primitive*'.
    class PrimitiveTest : public rocsparse_ut::HandleTest
    {
    };

    // Assert that a primitive returned success. Wrapping the call in an extra
    // pair of parentheses protects commas in template argument lists (e.g.
    // foo<I, J>(...)) from being parsed as gtest-macro argument separators.
    inline void expect_success(rocsparse_status status)
    {
        ASSERT_EQ(status, rocsparse_status_success);
    }

    // Copy n elements from a device pointer back to a host vector.
    template <typename T>
    std::vector<T> to_host(const T* d, size_t n)
    {
        std::vector<T> h(n);
        (void)hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost);
        return h;
    }
} // namespace

// ---------------------------------------------------------------------------
// inclusive_scan : out[i] = sum(in[0..i])
// ---------------------------------------------------------------------------
TEST_F(PrimitiveTest, inclusive_scan_i32)
{
    using I = int32_t;
    using J = int32_t;

    const std::vector<I> in{1, 2, 3, 4};
    const size_t         length = in.size();

    device_vector<I> d_in(in);
    device_vector<J> d_out(length);
    ASSERT_NE(d_in.ptr, nullptr);
    ASSERT_NE(d_out.ptr, nullptr);

    size_t buffer_size = 0;
    expect_success(prim::inclusive_scan_buffer_size<I, J>(handle, length, &buffer_size));

    device_vector<char> d_buffer(buffer_size ? buffer_size : 1);
    ASSERT_NE(d_buffer.ptr, nullptr);

    expect_success(prim::inclusive_scan<I, J>(handle, d_in, d_out, length, buffer_size, d_buffer));
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<J> got = to_host(d_out.ptr, length);
    EXPECT_EQ(got, (std::vector<J>{1, 3, 6, 10}));
}

// ---------------------------------------------------------------------------
// exclusive_scan : out[i] = initial + sum(in[0..i-1])
// ---------------------------------------------------------------------------
TEST_F(PrimitiveTest, exclusive_scan_i32)
{
    using I = int32_t;
    using J = int32_t;

    const std::vector<I> in{1, 2, 3, 4};
    const size_t         length        = in.size();
    const J              initial_value = 0;

    device_vector<I> d_in(in);
    device_vector<J> d_out(length);
    ASSERT_NE(d_in.ptr, nullptr);
    ASSERT_NE(d_out.ptr, nullptr);

    size_t buffer_size = 0;
    expect_success(
        prim::exclusive_scan_buffer_size<I, J>(handle, initial_value, length, &buffer_size));

    device_vector<char> d_buffer(buffer_size ? buffer_size : 1);
    ASSERT_NE(d_buffer.ptr, nullptr);

    expect_success(prim::exclusive_scan<I, J>(
        handle, d_in, d_out, initial_value, length, buffer_size, d_buffer));
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<J> got = to_host(d_out.ptr, length);
    EXPECT_EQ(got, (std::vector<J>{0, 1, 3, 6}));
}

// exclusive_scan honours a non-zero initial value (offset arithmetic).
TEST_F(PrimitiveTest, exclusive_scan_i32_nonzero_initial)
{
    using I = int32_t;
    using J = int32_t;

    const std::vector<I> in{5, 10, 15};
    const size_t         length        = in.size();
    const J              initial_value = 100;

    device_vector<I> d_in(in);
    device_vector<J> d_out(length);
    ASSERT_NE(d_in.ptr, nullptr);
    ASSERT_NE(d_out.ptr, nullptr);

    size_t buffer_size = 0;
    expect_success(
        prim::exclusive_scan_buffer_size<I, J>(handle, initial_value, length, &buffer_size));

    device_vector<char> d_buffer(buffer_size ? buffer_size : 1);
    ASSERT_NE(d_buffer.ptr, nullptr);

    expect_success(prim::exclusive_scan<I, J>(
        handle, d_in, d_out, initial_value, length, buffer_size, d_buffer));
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<J> got = to_host(d_out.ptr, length);
    EXPECT_EQ(got, (std::vector<J>{100, 105, 115}));
}

// ---------------------------------------------------------------------------
// find_max : reduce with maximum
// ---------------------------------------------------------------------------
TEST_F(PrimitiveTest, find_max_i32)
{
    using I = int32_t;
    using J = int32_t;

    const std::vector<I> in{3, 1, 4, 1, 5, 9, 2, 6};
    const size_t         length = in.size();

    device_vector<I> d_in(in);
    device_vector<J> d_max(1);
    ASSERT_NE(d_in.ptr, nullptr);
    ASSERT_NE(d_max.ptr, nullptr);

    size_t buffer_size = 0;
    expect_success(prim::find_max_buffer_size<I, J>(handle, length, &buffer_size));

    device_vector<char> d_buffer(buffer_size ? buffer_size : 1);
    ASSERT_NE(d_buffer.ptr, nullptr);

    expect_success(prim::find_max<I, J>(handle, d_in, d_max, length, buffer_size, d_buffer));
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<J> got = to_host(d_max.ptr, 1);
    EXPECT_EQ(got, (std::vector<J>{9}));
}

// ---------------------------------------------------------------------------
// find_sum : reduce with plus
// ---------------------------------------------------------------------------
TEST_F(PrimitiveTest, find_sum_i32)
{
    using I = int32_t;
    using J = int32_t;

    const std::vector<I> in{1, 2, 3, 4, 5, 6};
    const size_t         length = in.size();

    device_vector<I> d_in(in);
    device_vector<J> d_sum(1);
    ASSERT_NE(d_in.ptr, nullptr);
    ASSERT_NE(d_sum.ptr, nullptr);

    size_t buffer_size = 0;
    expect_success(prim::find_sum_buffer_size<I, J>(handle, length, &buffer_size));

    device_vector<char> d_buffer(buffer_size ? buffer_size : 1);
    ASSERT_NE(d_buffer.ptr, nullptr);

    expect_success(prim::find_sum<I, J>(handle, d_in, d_sum, length, buffer_size, d_buffer));
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<J> got = to_host(d_sum.ptr, 1);
    EXPECT_EQ(got, (std::vector<J>{21}));
}

// ---------------------------------------------------------------------------
// run_length_encode : compress consecutive runs into (unique, counts) + #runs
// ---------------------------------------------------------------------------
TEST_F(PrimitiveTest, run_length_encode_i32)
{
    using J = int32_t;

    const std::vector<J> in{1, 1, 2, 3, 3, 3};
    const size_t         length = in.size();

    device_vector<J> d_in(in);
    device_vector<J> d_unique(length);
    device_vector<J> d_counts(length);
    device_vector<J> d_runs(1);
    ASSERT_NE(d_in.ptr, nullptr);
    ASSERT_NE(d_unique.ptr, nullptr);
    ASSERT_NE(d_counts.ptr, nullptr);
    ASSERT_NE(d_runs.ptr, nullptr);

    size_t buffer_size = 0;
    expect_success(prim::run_length_encode_buffer_size<J>(handle, length, &buffer_size));

    device_vector<char> d_buffer(buffer_size ? buffer_size : 1);
    ASSERT_NE(d_buffer.ptr, nullptr);

    expect_success(prim::run_length_encode<J>(
        handle, d_in, d_unique, d_counts, d_runs, length, buffer_size, d_buffer));
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<J> runs = to_host(d_runs.ptr, 1);
    ASSERT_EQ(runs, (std::vector<J>{3}));

    const size_t         num_runs = static_cast<size_t>(runs[0]);
    const std::vector<J> unique   = to_host(d_unique.ptr, num_runs);
    const std::vector<J> counts   = to_host(d_counts.ptr, num_runs);
    EXPECT_EQ(unique, (std::vector<J>{1, 2, 3}));
    EXPECT_EQ(counts, (std::vector<J>{2, 1, 3}));
}

// ---------------------------------------------------------------------------
// radix_sort_keys : ascending sort of keys via the double_buffer API
// ---------------------------------------------------------------------------
TEST_F(PrimitiveTest, radix_sort_keys_f32)
{
    using K = float;

    const std::vector<K> keys_in{3.0f, 1.0f, 4.0f, 1.5f, 2.0f};
    const size_t         length   = keys_in.size();
    const uint32_t       startbit = 0;
    const uint32_t       endbit   = sizeof(K) * 8;

    device_vector<K> d_keys(keys_in);
    device_vector<K> d_keys_alt(length);
    ASSERT_NE(d_keys.ptr, nullptr);
    ASSERT_NE(d_keys_alt.ptr, nullptr);

    size_t buffer_size = 0;
    expect_success(
        prim::radix_sort_keys_buffer_size<K>(handle, length, startbit, endbit, &buffer_size));

    device_vector<char> d_buffer(buffer_size ? buffer_size : 1);
    ASSERT_NE(d_buffer.ptr, nullptr);

    prim::double_buffer<K> keys(d_keys.ptr, d_keys_alt.ptr);
    expect_success(
        prim::radix_sort_keys<K>(handle, keys, length, startbit, endbit, buffer_size, d_buffer));
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<K> got = to_host(keys.current(), length);
    EXPECT_EQ(got, (std::vector<K>{1.0f, 1.5f, 2.0f, 3.0f, 4.0f}));
}

// ---------------------------------------------------------------------------
// radix_sort_pairs : sort keys and carry the associated values along
// ---------------------------------------------------------------------------
TEST_F(PrimitiveTest, radix_sort_pairs_i32_i32)
{
    using K = int32_t;
    using V = int32_t;

    const std::vector<K> keys_in{3, 1, 4, 2};
    const std::vector<V> vals_in{30, 10, 40, 20};
    const size_t         length   = keys_in.size();
    const uint32_t       startbit = 0;
    const uint32_t       endbit   = sizeof(K) * 8;

    device_vector<K> d_keys(keys_in);
    device_vector<K> d_keys_alt(length);
    device_vector<V> d_vals(vals_in);
    device_vector<V> d_vals_alt(length);
    ASSERT_NE(d_keys.ptr, nullptr);
    ASSERT_NE(d_keys_alt.ptr, nullptr);
    ASSERT_NE(d_vals.ptr, nullptr);
    ASSERT_NE(d_vals_alt.ptr, nullptr);

    size_t buffer_size = 0;
    expect_success(prim::radix_sort_pairs_buffer_size<K, V>(
        handle, length, startbit, endbit, &buffer_size, true));

    device_vector<char> d_buffer(buffer_size ? buffer_size : 1);
    ASSERT_NE(d_buffer.ptr, nullptr);

    prim::double_buffer<K> keys(d_keys.ptr, d_keys_alt.ptr);
    prim::double_buffer<V> vals(d_vals.ptr, d_vals_alt.ptr);
    expect_success(prim::radix_sort_pairs<K, V>(
        handle, keys, vals, length, startbit, endbit, buffer_size, d_buffer));
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<K> got_keys = to_host(keys.current(), length);
    const std::vector<V> got_vals = to_host(vals.current(), length);
    EXPECT_EQ(got_keys, (std::vector<K>{1, 2, 3, 4}));
    EXPECT_EQ(got_vals, (std::vector<V>{10, 20, 30, 40}));
}

// radix_sort_pairs also exposes a non-double-buffer (input/output) overload.
TEST_F(PrimitiveTest, radix_sort_pairs_i32_i32_io)
{
    using K = int32_t;
    using V = int32_t;

    const std::vector<K> keys_in{5, 2, 8, 1};
    const std::vector<V> vals_in{50, 20, 80, 10};
    const size_t         length   = keys_in.size();
    const uint32_t       startbit = 0;
    const uint32_t       endbit   = sizeof(K) * 8;

    device_vector<K> d_keys_in(keys_in);
    device_vector<K> d_keys_out(length);
    device_vector<V> d_vals_in(vals_in);
    device_vector<V> d_vals_out(length);
    ASSERT_NE(d_keys_in.ptr, nullptr);
    ASSERT_NE(d_keys_out.ptr, nullptr);
    ASSERT_NE(d_vals_in.ptr, nullptr);
    ASSERT_NE(d_vals_out.ptr, nullptr);

    // The input/output overload requires an explicit (non double-buffer) size.
    size_t buffer_size = 0;
    expect_success(prim::radix_sort_pairs_buffer_size<K, V>(
        handle, length, startbit, endbit, &buffer_size, false));

    device_vector<char> d_buffer(buffer_size ? buffer_size : 1);
    ASSERT_NE(d_buffer.ptr, nullptr);

    expect_success(prim::radix_sort_pairs<K, V>(handle,
                                                d_keys_in,
                                                d_keys_out,
                                                d_vals_in,
                                                d_vals_out,
                                                length,
                                                startbit,
                                                endbit,
                                                buffer_size,
                                                d_buffer));
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<K> got_keys = to_host(d_keys_out.ptr, length);
    const std::vector<V> got_vals = to_host(d_vals_out.ptr, length);
    EXPECT_EQ(got_keys, (std::vector<K>{1, 2, 5, 8}));
    EXPECT_EQ(got_vals, (std::vector<V>{10, 20, 50, 80}));
}

// ---------------------------------------------------------------------------
// segmented_radix_sort_keys : ascending sort of keys WITHIN each segment. The
// segment boundaries are given by device begin/end offset arrays. Elements are
// only reordered inside their own [begin,end) range.
//
// The two .cpp translation units for the segmented primitives are compiled into
// this binary via unit_test_primitives_support.cpp (they are not in the CMake
// primitive source list). Instantiations available: keys<int32_t,int32_t> and
// pairs<int32_t,int32_t,int32_t>.
// ---------------------------------------------------------------------------
TEST_F(PrimitiveTest, segmented_radix_sort_keys_i32_full_range)
{
    using K = int32_t;
    using I = int32_t;

    // Two segments: [0,3) = {4,2,6}, [3,6) = {9,1,5}.
    const std::vector<K> keys_in{4, 2, 6, 9, 1, 5};
    const std::vector<I> begin{0, 3};
    const std::vector<I> end{3, 6};
    const size_t         length   = keys_in.size();
    const size_t         segments = begin.size();
    const uint32_t       startbit = 0;
    const uint32_t       endbit   = sizeof(K) * 8;

    device_vector<K> d_keys(keys_in);
    device_vector<K> d_keys_alt(length);
    device_vector<I> d_begin(begin);
    device_vector<I> d_end(end);
    ASSERT_NE(d_keys.ptr, nullptr);
    ASSERT_NE(d_keys_alt.ptr, nullptr);
    ASSERT_NE(d_begin.ptr, nullptr);
    ASSERT_NE(d_end.ptr, nullptr);

    size_t buffer_size = 0;
    expect_success(prim::segmented_radix_sort_keys_buffer_size<K, I>(
        handle, length, segments, startbit, endbit, &buffer_size));

    device_vector<char> d_buffer(buffer_size ? buffer_size : 1);
    ASSERT_NE(d_buffer.ptr, nullptr);

    prim::double_buffer<K> keys(d_keys.ptr, d_keys_alt.ptr);
    expect_success(prim::segmented_radix_sort_keys<K, I>(handle,
                                                         keys,
                                                         length,
                                                         segments,
                                                         d_begin.ptr,
                                                         d_end.ptr,
                                                         startbit,
                                                         endbit,
                                                         buffer_size,
                                                         d_buffer));
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<K> got = to_host(keys.current(), length);
    EXPECT_EQ(got, (std::vector<K>{2, 4, 6, 1, 5, 9}));
}

// Restricted bit range: all keys are < 16 so a 4-bit window [0,4) still sorts
// them fully, exercising the non-default begin_bit/end_bit branch.
TEST_F(PrimitiveTest, segmented_radix_sort_keys_i32_narrow_range)
{
    using K = int32_t;
    using I = int32_t;

    const std::vector<K> keys_in{7, 3, 5, 2, 8, 1};
    const std::vector<I> begin{0, 3};
    const std::vector<I> end{3, 6};
    const size_t         length   = keys_in.size();
    const size_t         segments = begin.size();
    const uint32_t       startbit = 0;
    const uint32_t       endbit   = 4;

    device_vector<K> d_keys(keys_in);
    device_vector<K> d_keys_alt(length);
    device_vector<I> d_begin(begin);
    device_vector<I> d_end(end);
    ASSERT_NE(d_keys.ptr, nullptr);
    ASSERT_NE(d_keys_alt.ptr, nullptr);
    ASSERT_NE(d_begin.ptr, nullptr);
    ASSERT_NE(d_end.ptr, nullptr);

    size_t buffer_size = 0;
    expect_success(prim::segmented_radix_sort_keys_buffer_size<K, I>(
        handle, length, segments, startbit, endbit, &buffer_size));

    device_vector<char> d_buffer(buffer_size ? buffer_size : 1);
    ASSERT_NE(d_buffer.ptr, nullptr);

    prim::double_buffer<K> keys(d_keys.ptr, d_keys_alt.ptr);
    expect_success(prim::segmented_radix_sort_keys<K, I>(handle,
                                                         keys,
                                                         length,
                                                         segments,
                                                         d_begin.ptr,
                                                         d_end.ptr,
                                                         startbit,
                                                         endbit,
                                                         buffer_size,
                                                         d_buffer));
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<K> got = to_host(keys.current(), length);
    EXPECT_EQ(got, (std::vector<K>{3, 5, 7, 1, 2, 8}));
}

// ---------------------------------------------------------------------------
// segmented_radix_sort_pairs : segment-local ascending sort of keys, carrying
// the paired values along.
// ---------------------------------------------------------------------------
TEST_F(PrimitiveTest, segmented_radix_sort_pairs_i32_i32)
{
    using K = int32_t;
    using V = int32_t;
    using I = int32_t;

    // Two segments: [0,3) keys {4,2,6}, [3,6) keys {9,1,5}.
    const std::vector<K> keys_in{4, 2, 6, 9, 1, 5};
    const std::vector<V> vals_in{40, 20, 60, 90, 10, 50};
    const std::vector<I> begin{0, 3};
    const std::vector<I> end{3, 6};
    const size_t         length   = keys_in.size();
    const size_t         segments = begin.size();
    const uint32_t       startbit = 0;
    const uint32_t       endbit   = sizeof(K) * 8;

    device_vector<K> d_keys(keys_in);
    device_vector<K> d_keys_alt(length);
    device_vector<V> d_vals(vals_in);
    device_vector<V> d_vals_alt(length);
    device_vector<I> d_begin(begin);
    device_vector<I> d_end(end);
    ASSERT_NE(d_keys.ptr, nullptr);
    ASSERT_NE(d_keys_alt.ptr, nullptr);
    ASSERT_NE(d_vals.ptr, nullptr);
    ASSERT_NE(d_vals_alt.ptr, nullptr);
    ASSERT_NE(d_begin.ptr, nullptr);
    ASSERT_NE(d_end.ptr, nullptr);

    size_t buffer_size = 0;
    expect_success(prim::segmented_radix_sort_pairs_buffer_size<K, V, I>(
        handle, length, segments, startbit, endbit, &buffer_size));

    device_vector<char> d_buffer(buffer_size ? buffer_size : 1);
    ASSERT_NE(d_buffer.ptr, nullptr);

    prim::double_buffer<K> keys(d_keys.ptr, d_keys_alt.ptr);
    prim::double_buffer<V> vals(d_vals.ptr, d_vals_alt.ptr);
    expect_success(prim::segmented_radix_sort_pairs<K, V, I>(handle,
                                                             keys,
                                                             vals,
                                                             length,
                                                             segments,
                                                             d_begin.ptr,
                                                             d_end.ptr,
                                                             startbit,
                                                             endbit,
                                                             buffer_size,
                                                             d_buffer));
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<K> got_keys = to_host(keys.current(), length);
    const std::vector<V> got_vals = to_host(vals.current(), length);
    EXPECT_EQ(got_keys, (std::vector<K>{2, 4, 6, 1, 5, 9}));
    EXPECT_EQ(got_vals, (std::vector<V>{20, 40, 60, 10, 50, 90}));
}
