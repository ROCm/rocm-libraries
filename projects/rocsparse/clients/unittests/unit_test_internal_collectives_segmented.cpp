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
// Device (GPU) unit tests for rocSPARSE internal SEGMENTED reductions
// (rocsparse::wfsegmented_reduce and rocsparse::segmented_blockreduce). Split
// out of unit_test_internal_collectives.cpp by collective family. Both are
// exercised on exactly one wavefront/block via the launch helpers in
// unit_test_utils.hpp.
//
#include "unit_test_utils.hpp"

#include "unit_test_internal_collectives_common.hpp"

#include "rocsparse_common.hpp" // wfsegmented_reduce

// segmented_blockreduce lives in the level2 device header. The device
// unit-test target only puts library/src/{include,level1,level3} on the
// include path, so we reach the level2 header with a source-relative include
// (this TU lives in clients/unittests/). This keeps the addition local to this
// file and avoids a shared CMakeLists.txt include-dir change.
#include "../../library/src/level2/coomv_device.h" // rocsparse::segmented_blockreduce

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

using rocsparse_ut::device_vector;
using rocsparse_ut::launch_single_block;
using rocsparse_ut::launch_warp_by_size;
using rocsparse_ut::to_host;

using namespace rocsparse_ut_collectives;

namespace
{
    // ---- segmented wavefront reduce ----------------------------------------
    template <uint32_t WFSIZE, typename R, typename T>
    __global__ void k_wfsegmented_reduce(const R* row, const T* val, T* out)
    {
        const int lane = threadIdx.x;
        out[lane]      = rocsparse::wfsegmented_reduce<WFSIZE>(row[lane], val[lane]);
    }

    // ---- segmented block reduce (coomv_device.h) ---------------------------
    template <uint32_t BS, typename I, typename T>
    __global__ void k_segmented_blockreduce(const I* rin, const T* vin, T* vout)
    {
        __shared__ I sr[BS];
        __shared__ T sv[BS];
        const int    tid = threadIdx.x;
        sr[tid]          = rin[tid];
        sv[tid]          = vin[tid];
        __syncthreads();
        rocsparse::segmented_blockreduce<BS>(sr, sv);
        vout[tid] = sv[tid];
    }
    // wfsegmented_reduce<WFSIZE>(row, val): inclusive scan of val restricted to
    // runs of equal row key. Host reference mirrors the exact shfl_up scan, so
    // every lane's result is validated.
    template <typename R, typename T, typename RowGen, typename ValGen>
    void run_wfsegmented_reduce(RowGen rowgen, ValGen valgen)
    {
        const uint32_t wf = require_wavefront_size();
        std::vector<R> row(wf);
        std::vector<T> val(wf);
        for(uint32_t l = 0; l < wf; ++l)
        {
            row[l] = rowgen(l);
            val[l] = valgen(l);
        }
        std::vector<T> v = val;
        for(uint32_t j = 1; j < wf; j <<= 1)
        {
            std::vector<T> nv = v;
            for(uint32_t l = j; l < wf; ++l)
            {
                if(row[l] == row[l - j])
                    nv[l] = v[l] + v[l - j];
            }
            v = nv;
        }
        device_vector<R> d_row(row);
        device_vector<T> d_val(val), d_out(size_t{wf});
        ASSERT_NE(d_row.ptr, nullptr);
        ASSERT_NE(d_val.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_warp_by_size(k_wfsegmented_reduce<32, R, T>,
                                      k_wfsegmented_reduce<64, R, T>,
                                      d_row.ptr,
                                      d_val.ptr,
                                      d_out.ptr),
                  hipSuccess);
        auto h = to_host(d_out);
        for(uint32_t l = 0; l < wf; ++l)
            expect_close(h[l], v[l]);
    }

    // Contiguous runs of equal row keys of varying lengths; valid for wf 32 or 64
    // (the final run simply extends on wider wavefronts).
    template <typename R>
    R seg_row_key(uint32_t l)
    {
        if(l < 5)
            return static_cast<R>(0);
        if(l < 6)
            return static_cast<R>(1);
        if(l < 14)
            return static_cast<R>(2);
        if(l < 20)
            return static_cast<R>(3);
        return static_cast<R>(4);
    }
    // segmented_blockreduce<BS>(row, val): block-wide Hillis-Steele inclusive
    // scan restricted to equal-row-key runs. Host reference mirrors the exact
    // scan; every one of the BS outputs is checked.
    template <uint32_t BS, typename I, typename T>
    void run_segmented_blockreduce(const std::vector<I>& row, const std::vector<T>& val)
    {
        ASSERT_EQ(row.size(), size_t{BS});
        ASSERT_EQ(val.size(), size_t{BS});
        std::vector<T> v = val;
        for(uint32_t j = 1; j < BS; j <<= 1)
        {
            std::vector<T> add(BS, T(0));
            for(uint32_t t = j; t < BS; ++t)
            {
                if(row[t] == row[t - j])
                    add[t] = v[t - j];
            }
            for(uint32_t t = 0; t < BS; ++t)
                v[t] = v[t] + add[t];
        }
        device_vector<I> d_row(row);
        device_vector<T> d_val(val), d_out(size_t{BS});
        ASSERT_NE(d_row.ptr, nullptr);
        ASSERT_NE(d_val.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_block(
                      k_segmented_blockreduce<BS, I, T>, BS, d_row.ptr, d_val.ptr, d_out.ptr),
                  hipSuccess);
        auto h = to_host(d_out);
        for(uint32_t t = 0; t < BS; ++t)
            expect_close(h[t], v[t]);
    }

    template <typename I>
    std::vector<I> seg_rows64()
    {
        std::vector<I> r(64);
        for(uint32_t t = 0; t < 64; ++t)
            r[t] = static_cast<I>(t / 5); // runs of length 5 (last run shorter)
        return r;
    }
} // namespace

TEST(internal_collectives_wfsegmented_reduce, row_i32_val_i32)
{
    run_wfsegmented_reduce<int32_t, int32_t>(
        seg_row_key<int32_t>, [](uint32_t l) { return static_cast<int32_t>((l % 4) + 1); });
}
TEST(internal_collectives_wfsegmented_reduce, row_i32_val_f32)
{
    run_wfsegmented_reduce<int32_t, float>(
        seg_row_key<int32_t>, [](uint32_t l) { return static_cast<float>((l % 4) + 1); });
}
TEST(internal_collectives_wfsegmented_reduce, row_i64_val_f64)
{
    run_wfsegmented_reduce<int64_t, double>(
        seg_row_key<int64_t>, [](uint32_t l) { return static_cast<double>((l % 4) + 1); });
}
TEST(internal_collectives_segmented_blockreduce, row_i32_val_f32)
{
    std::vector<float> val(64);
    for(uint32_t t = 0; t < 64; ++t)
        val[t] = static_cast<float>((t % 3) + 1);
    run_segmented_blockreduce<64, int32_t, float>(seg_rows64<int32_t>(), val);
}
TEST(internal_collectives_segmented_blockreduce, row_i64_val_f64)
{
    std::vector<double> val(64);
    for(uint32_t t = 0; t < 64; ++t)
        val[t] = static_cast<double>((t % 3) + 1);
    run_segmented_blockreduce<64, int64_t, double>(seg_rows64<int64_t>(), val);
}
// Edge case: a single segment (all row keys equal) makes segmented_blockreduce
// degenerate to a plain block-wide inclusive prefix sum.
TEST(internal_collectives_segmented_blockreduce, single_segment_all_same)
{
    std::vector<int32_t> row(64, 0);
    std::vector<float>   val(64);
    for(uint32_t t = 0; t < 64; ++t)
        val[t] = static_cast<float>((t % 5) + 1);
    run_segmented_blockreduce<64, int32_t, float>(row, val);
}
// Edge case: all-distinct row keys make every element its own segment, so each
// output must equal its input (no accumulation crosses a segment boundary).
TEST(internal_collectives_segmented_blockreduce, all_distinct_segments)
{
    std::vector<int32_t> row(64);
    for(uint32_t t = 0; t < 64; ++t)
        row[t] = static_cast<int32_t>(t);
    std::vector<double> val(64);
    for(uint32_t t = 0; t < 64; ++t)
        val[t] = static_cast<double>((t % 7) + 1);
    run_segmented_blockreduce<64, int32_t, double>(row, val);
}
// wfsegmented_reduce with all-distinct row keys: every lane is its own segment,
// so the scan result equals the input value on every lane.
TEST(internal_collectives_wfsegmented_reduce, all_distinct_segments)
{
    run_wfsegmented_reduce<int32_t, float>(
        [](uint32_t l) { return static_cast<int32_t>(l); },
        [](uint32_t l) { return static_cast<float>((l % 4) + 1); });
}
