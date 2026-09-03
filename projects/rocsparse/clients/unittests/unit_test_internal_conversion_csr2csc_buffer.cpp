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
// csr2csc must not write outside of the temporary buffer whose size it
// reported through its buffer-size query.
//
// The numeric (rocsparse_action_numeric) code path sorts (column index, offset)
// pairs, i.e. rocPRIM keys of type J and values of type I, while the symbolic
// path sorts <J, J>. rocPRIM's temporary storage requirement depends on BOTH
// type parameters (its onesweep configuration is auto-tuned per key/value size,
// and the number of lookback states follows from the resulting items-per-block),
// so querying the size for the wrong pair silently under-allocates and the sort
// scribbles past the end of the user buffer.
//
// The tests below run the conversion with a poisoned guard band placed directly
// behind the buffer that rocSPARSE asked for, and fail if a single guard byte
// changed.
//
// Note on hardware coverage: whether <J,J> and <J,I> actually resolve to
// different rocPRIM configurations depends on the architecture (they differ on
// gfx942/gfx950/gfx1030/gfx12xx, they happen to coincide on gfx908/gfx90a/
// gfx1100). On the architectures where they coincide these tests pass even with
// an under-sized buffer, so they must run on a wide set of GPUs to be effective.
//
#include "unit_test_utils.hpp"

#include "rocsparse.h"

#include <algorithm>
#include <numeric>

using namespace rocsparse_ut;

namespace
{
    constexpr rocsparse_index_base BASE = rocsparse_index_base_zero;

    // rocPRIM sorts up to 1M items with a merge sort whose temporary storage is
    // independent of the value type; the onesweep path taken above that limit is
    // the one whose storage requirement differs. Hence > 1M non-zeros.
    constexpr int64_t M            = 4200;
    constexpr int64_t N            = 4096;
    constexpr int64_t NNZ_PER_ROW  = 256;
    constexpr int64_t COLUMN_STEP  = N / NNZ_PER_ROW;
    constexpr int64_t NNZ          = M * NNZ_PER_ROW;
    constexpr size_t  GUARD_BYTES  = size_t{8} << 20;
    constexpr uint8_t GUARD_FILLER = 0xAB;

    static_assert(NNZ > (1 << 20), "matrix too small to reach the rocPRIM onesweep path");

    // Column of the k-th non-zero of row i. Strictly increasing in k, spread over
    // all N columns so that the sort covers the full radix range.
    int64_t column_of(int64_t i, int64_t k)
    {
        return k * COLUMN_STEP + (i % COLUMN_STEP);
    }

    template <typename I, typename J, typename T>
    void csr2csc_does_not_overflow_its_buffer()
    {
        std::vector<I> hcsr_row_ptr(M + 1);
        std::vector<J> hcsr_col_ind(NNZ);
        std::vector<T> hcsr_val(NNZ);
        for(int64_t i = 0; i < M; ++i)
        {
            hcsr_row_ptr[i] = static_cast<I>(i * NNZ_PER_ROW);
            for(int64_t k = 0; k < NNZ_PER_ROW; ++k)
            {
                const int64_t at = i * NNZ_PER_ROW + k;
                hcsr_col_ind[at] = static_cast<J>(column_of(i, k));
                hcsr_val[at]     = static_cast<T>(at % 1024);
            }
        }
        hcsr_row_ptr[M] = static_cast<I>(NNZ);

        device_vector<I> dcsr_row_ptr(hcsr_row_ptr);
        device_vector<J> dcsr_col_ind(hcsr_col_ind);
        device_vector<T> dcsr_val(hcsr_val);
        const size_t     nnz_count = NNZ;
        device_vector<I> dcsc_col_ptr(size_t(N + 1));
        device_vector<J> dcsc_row_ind(nnz_count);
        device_vector<T> dcsc_val(nnz_count);
        ASSERT_NE(dcsr_row_ptr.ptr, nullptr);
        ASSERT_NE(dcsr_col_ind.ptr, nullptr);
        ASSERT_NE(dcsr_val.ptr, nullptr);
        ASSERT_NE(dcsc_col_ptr.ptr, nullptr);
        ASSERT_NE(dcsc_row_ind.ptr, nullptr);
        ASSERT_NE(dcsc_val.ptr, nullptr);

        rocsparse_handle handle = nullptr;
        ASSERT_EQ(rocsparse_create_handle(&handle), rocsparse_status_success);

        rocsparse_const_spmat_descr source = nullptr;
        ASSERT_EQ(rocsparse_create_const_csr_descr(&source,
                                                   M,
                                                   N,
                                                   NNZ,
                                                   dcsr_row_ptr.ptr,
                                                   dcsr_col_ind.ptr,
                                                   dcsr_val.ptr,
                                                   it_of<I>(),
                                                   it_of<J>(),
                                                   BASE,
                                                   dt_of<T>()),
                  rocsparse_status_success);

        rocsparse_spmat_descr target = nullptr;
        ASSERT_EQ(rocsparse_create_csc_descr(&target,
                                             M,
                                             N,
                                             NNZ,
                                             dcsc_col_ptr.ptr,
                                             dcsc_row_ind.ptr,
                                             dcsc_val.ptr,
                                             it_of<I>(),
                                             it_of<J>(),
                                             BASE,
                                             dt_of<T>()),
                  rocsparse_status_success);

        rocsparse_sparse_to_sparse_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(
                      &descr, source, target, rocsparse_sparse_to_sparse_alg_default),
                  rocsparse_status_success);

        size_t analysis_size = 0;
        ASSERT_EQ(rocsparse_sparse_to_sparse_buffer_size(handle,
                                                         descr,
                                                         source,
                                                         target,
                                                         rocsparse_sparse_to_sparse_stage_analysis,
                                                         &analysis_size),
                  rocsparse_status_success);
        {
            device_vector<uint8_t> analysis_buffer(analysis_size);
            ASSERT_EQ(rocsparse_sparse_to_sparse(handle,
                                                 descr,
                                                 source,
                                                 target,
                                                 rocsparse_sparse_to_sparse_stage_analysis,
                                                 analysis_size,
                                                 analysis_buffer.ptr),
                      rocsparse_status_success);
        }

        size_t compute_size = 0;
        ASSERT_EQ(rocsparse_sparse_to_sparse_buffer_size(handle,
                                                         descr,
                                                         source,
                                                         target,
                                                         rocsparse_sparse_to_sparse_stage_compute,
                                                         &compute_size),
                  rocsparse_status_success);
        ASSERT_GT(compute_size, 0u);

        // Buffer of exactly the requested size, followed by a poisoned guard band.
        device_vector<uint8_t> compute_buffer(compute_size + GUARD_BYTES);
        ASSERT_NE(compute_buffer.ptr, nullptr);
        UT_CHECK_HIP(hipMemset(compute_buffer.ptr + compute_size, GUARD_FILLER, GUARD_BYTES));

        ASSERT_EQ(rocsparse_sparse_to_sparse(handle,
                                             descr,
                                             source,
                                             target,
                                             rocsparse_sparse_to_sparse_stage_compute,
                                             compute_size,
                                             compute_buffer.ptr),
                  rocsparse_status_success);
        UT_CHECK_HIP(hipDeviceSynchronize());

        const std::vector<uint8_t> guard
            = to_host<uint8_t>(compute_buffer.ptr + compute_size, GUARD_BYTES);
        const size_t clobbered = std::count_if(
            guard.begin(), guard.end(), [](uint8_t b) { return b != GUARD_FILLER; });
        EXPECT_EQ(clobbered, 0u) << "csr2csc wrote " << clobbered << " byte(s) past the end of the "
                                 << compute_size << "-byte buffer it asked for";

        // The conversion itself must still be correct: column i holds one entry per
        // row whose non-zero pattern covers it.
        std::vector<I> expected_col_ptr(N + 1, static_cast<I>(0));
        for(int64_t i = 0; i < M; ++i)
        {
            for(int64_t k = 0; k < NNZ_PER_ROW; ++k)
            {
                ++expected_col_ptr[column_of(i, k) + 1];
            }
        }
        std::partial_sum(
            expected_col_ptr.begin(), expected_col_ptr.end(), expected_col_ptr.begin());
        EXPECT_EQ(to_host<I>(dcsc_col_ptr.ptr, N + 1), expected_col_ptr);

        EXPECT_EQ(rocsparse_destroy_sparse_to_sparse_descr(descr), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(target), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(source), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_handle(handle), rocsparse_status_success);
    }
}

TEST(conversion_csr2csc_buffer, i32_ptr_i32_ind)
{
    csr2csc_does_not_overflow_its_buffer<int32_t, int32_t, float>();
}

TEST(conversion_csr2csc_buffer, i64_ptr_i32_ind)
{
    csr2csc_does_not_overflow_its_buffer<int64_t, int32_t, float>();
}

TEST(conversion_csr2csc_buffer, i32_ptr_i64_ind)
{
    csr2csc_does_not_overflow_its_buffer<int32_t, int64_t, float>();
}

TEST(conversion_csr2csc_buffer, i64_ptr_i64_ind)
{
    csr2csc_does_not_overflow_its_buffer<int64_t, int64_t, float>();
}
