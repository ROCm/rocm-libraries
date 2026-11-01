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
// Grid-stride tests for the block-row grids of gebsrmv, gebsr2csr and bsrgeam.
//
// These launches clamp grid.x with rocsparse::get_grid_size_x, to
// min(maxGridSize[0], (2^32 - 1) / blockDim.x) blocks, and grid-stride over the
// block rows.
//
// The forced-limit cases shrink handle->properties.maxGridSize[0] with
// ScopedMaxGridSizeX to 3 blocks and to 1 block and run MB = 200 block rows, so
// most block rows are reached only by the grid-stride loops. They pass on a
// launcher that ignores maxGridSize[0], so they check the loops, not the clamp.
//
// The real_threshold cases keep the device limit and use mb = (2^32 - 1) /
// blockDim.x + 2 block rows, so get_grid_size_x clamps on its own. Without the
// clamp the launch is rejected, so the routine returns an error or leaves its
// output unwritten; with the clamp but no grid-stride loop the last two block
// rows are left unwritten. Only block rows 0, limit and limit + 1 hold a block,
// so the footprint is a few hundred MB; a case skips when less than twice that
// is free.
//
// rocsparse_gebsrmv sends square blocks to bsrmv and rocsparse_gebsr2csr sends
// them to bsr2csr, so every gebsrmv and gebsr2csr shape here is non-square.
//
// TARGET: rocsparse-unit-test-device.
//
#include "unit_test_utils.hpp"

// ScopedMaxGridSizeX: shrinks handle->properties.maxGridSize[0], the limit
// get_grid_size_x clamps grid.x against.
#include "unit_test_grid_clamp.hpp"

#include "rocsparse.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    constexpr rocsparse_index_base BASE = rocsparse_index_base_zero;

    // Number of block rows in the forced-limit cases.
    constexpr rocsparse_int MB = 200;

    // The shrunk grid.x limit. At MB = 200 every grid under test is wider: the
    // 1xn-4xn kernels pack 32 block rows per block and still need 7 blocks.
    constexpr int CLAMPED_GRID_X = 3;

    // y entries keep this value where the routine must not write, and the beta
    // term is computed from it. Non-zero so the beta != 0 branch is exercised.
    constexpr float Y_INIT = 7.0f;

    constexpr float ALPHA = 2.0f;
    constexpr float BETA  = 3.0f;

    struct MatDescr
    {
        rocsparse_mat_descr d = nullptr;
        MatDescr()
        {
            (void)rocsparse_create_mat_descr(&d);
        }
        ~MatDescr()
        {
            if(d)
                (void)rocsparse_destroy_mat_descr(d);
        }
        MatDescr(const MatDescr&) = delete;

        MatDescr& operator=(const MatDescr&) = delete;
    };

    // A block-diagonal GEBSR matrix: block row i holds a single
    // row_block_dim x col_block_dim block in block column i. All values are
    // small integers so the device and the host reference agree exactly in
    // float, and they vary with (block, row, column) so a kernel that mixes up
    // or drops block rows cannot accidentally produce the reference result.
    struct gebsr_matrix
    {
        rocsparse_int              mb   = MB;
        rocsparse_int              nb   = MB;
        rocsparse_int              nnzb = MB;
        rocsparse_int              row_block_dim;
        rocsparse_int              col_block_dim;
        rocsparse_direction        dir;
        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<float>         val;

        // Value of entry (r, c) of the block in block row i.
        static float entry(rocsparse_int i, rocsparse_int r, rocsparse_int c)
        {
            return static_cast<float>(1 + ((i + 2 * r + 3 * c) % 7));
        }

        // Offset of entry (r, c) inside block j, honouring the storage
        // direction of the blocks.
        size_t at(rocsparse_int j, rocsparse_int r, rocsparse_int c) const
        {
            const size_t base = static_cast<size_t>(j) * row_block_dim * col_block_dim;
            return base
                   + (dir == rocsparse_direction_row ? static_cast<size_t>(r) * col_block_dim + c
                                                     : static_cast<size_t>(c) * row_block_dim + r);
        }
    };

    gebsr_matrix make_gebsr(rocsparse_int rbd, rocsparse_int cbd, rocsparse_direction dir)
    {
        gebsr_matrix m;
        m.row_block_dim = rbd;
        m.col_block_dim = cbd;
        m.dir           = dir;

        m.row_ptr.resize(MB + 1);
        m.col_ind.resize(MB);
        for(rocsparse_int i = 0; i < MB; ++i)
        {
            m.row_ptr[i] = i;
            m.col_ind[i] = i;
        }
        m.row_ptr[MB] = MB;

        m.val.resize(static_cast<size_t>(MB) * rbd * cbd);
        for(rocsparse_int i = 0; i < MB; ++i)
        {
            for(rocsparse_int r = 0; r < rbd; ++r)
            {
                for(rocsparse_int c = 0; c < cbd; ++c)
                {
                    m.val[m.at(i, r, c)] = gebsr_matrix::entry(i, r, c);
                }
            }
        }
        return m;
    }

    // Report the first index at which `got` and `want` differ, else -1. One
    // precise failure beats thousands of EXPECT_* in a loop.
    template <typename T>
    int64_t first_mismatch(const std::vector<T>& got, const std::vector<T>& want)
    {
        for(size_t i = 0; i < got.size(); ++i)
        {
            if(got[i] != want[i])
            {
                return static_cast<int64_t>(i);
            }
        }
        return got.size() == want.size() ? -1 : 0;
    }

    // ---------------------------------------------------------------------
    // Real-threshold helpers.
    // ---------------------------------------------------------------------

    // Largest grid.x a launch with `threads` threads per block can use.
    rocsparse_int dispatch_limit(int64_t threads)
    {
        return static_cast<rocsparse_int>(((int64_t(1) << 32) - 1) / threads);
    }

    // The block rows that hold a block in the real-threshold cases. The last two
    // lie past the clamped grid.
    struct three_rows
    {
        rocsparse_int row[3];

        explicit three_rows(rocsparse_int limit)
            : row{0, limit, limit + 1}
        {
        }

        // Index of the block in block row i, or -1 if the row is empty.
        int block_of(rocsparse_int i) const
        {
            for(int b = 0; b < 3; ++b)
            {
                if(row[b] == i)
                {
                    return b;
                }
            }
            return -1;
        }
    };

    // Block-row pointer of an mb-row matrix with one block in each of `rows`.
    std::vector<rocsparse_int> sparse_row_ptr(rocsparse_int mb, const three_rows& rows)
    {
        std::vector<rocsparse_int> ptr(static_cast<size_t>(mb) + 1);
        rocsparse_int              acc = 0;
        for(rocsparse_int i = 0; i < mb; ++i)
        {
            ptr[i] = acc;
            acc += (rows.block_of(i) >= 0) ? 1 : 0;
        }
        ptr[mb] = acc;
        return ptr;
    }

    bool device_has(size_t need)
    {
        size_t free_bytes  = 0;
        size_t total_bytes = 0;
        return hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess && free_bytes >= 2 * need;
    }
}

// ---------------------------------------------------------------------------
// gebsrmv: 16 grids across the eight rocsparse_gebsrmv_template_row_block_dim_*
// files, behind seven device functions (general, 1xn, 2xn, 3xn, 4xn, mxn,
// mxn_16).
// ---------------------------------------------------------------------------
class GebsrmvGridClamp : public HandleTest
{
protected:
    void check(rocsparse_int rbd, rocsparse_int cbd, rocsparse_direction dir, int grid_x)
    {
        const gebsr_matrix m = make_gebsr(rbd, cbd, dir);

        std::vector<float> x(static_cast<size_t>(m.nb) * cbd);
        for(size_t j = 0; j < x.size(); ++j)
        {
            x[j] = static_cast<float>(1 + (j % 5));
        }

        device_vector<rocsparse_int> d_row_ptr{m.row_ptr};
        device_vector<rocsparse_int> d_col_ind{m.col_ind};
        device_vector<float>         d_val{m.val};
        device_vector<float>         d_x{x};
        device_vector<float> d_y{std::vector<float>(static_cast<size_t>(m.mb) * rbd, Y_INIT)};
        ASSERT_TRUE(d_row_ptr.ptr && d_col_ind.ptr && d_val.ptr && d_x.ptr && d_y.ptr);

        MatDescr    descr;
        const float alpha = ALPHA;
        const float beta  = BETA;

        {
            ScopedMaxGridSizeX clamp(handle, grid_x);

            ASSERT_EQ(rocsparse_sgebsrmv(handle,
                                         dir,
                                         rocsparse_operation_none,
                                         m.mb,
                                         m.nb,
                                         m.nnzb,
                                         &alpha,
                                         descr.d,
                                         d_val,
                                         d_row_ptr,
                                         d_col_ind,
                                         rbd,
                                         cbd,
                                         d_x,
                                         &beta,
                                         d_y),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        }

        // Host reference: y = alpha * A * x + beta * y over the block diagonal.
        std::vector<float> want(static_cast<size_t>(m.mb) * rbd);
        for(rocsparse_int i = 0; i < m.mb; ++i)
        {
            for(rocsparse_int r = 0; r < rbd; ++r)
            {
                float acc = 0.0f;
                for(rocsparse_int c = 0; c < cbd; ++c)
                {
                    acc += m.val[m.at(i, r, c)] * x[static_cast<size_t>(i) * cbd + c];
                }
                want[static_cast<size_t>(i) * rbd + r] = ALPHA * acc + BETA * Y_INIT;
            }
        }

        const std::vector<float> got = to_host(d_y.ptr, d_y.n);
        const int64_t            bad = first_mismatch(got, want);
        EXPECT_EQ(bad, -1) << "row_block_dim=" << rbd << " col_block_dim=" << cbd
                           << " dir=" << static_cast<int>(dir) << ": first wrong y entry " << bad
                           << " (block row " << (bad / rbd) << " of " << m.mb << ") got "
                           << got[bad] << " want " << want[bad] << ". grid.x was clamped to "
                           << grid_x;
    }

    // One shape per grid. With one block per block row, blocks_per_row is 1, so
    // the 1xn-4xn files pick WFSIZE 4.
    //   (1, 2)   gebsrmvn_1xn_kernel<128, 2, 4>      row_block_dim_1.cpp
    //   (1, 20)  gebsrmvn_general_kernel<32, 32>     row_block_dim_1.cpp
    //   (2, 3)   gebsrmvn_2xn_kernel<128, 3, 4>      row_block_dim_2.cpp
    //   (2, 20)  gebsrmvn_general_kernel<32, 32>     row_block_dim_2.cpp
    //   (3, 2)   gebsrmvn_3xn_kernel<128, 2, 4>      row_block_dim_3.cpp
    //   (3, 20)  gebsrmvn_general_kernel<96, 32>     row_block_dim_3.cpp
    //   (4, 3)   gebsrmvn_4xn_kernel<128, 3, 4>      row_block_dim_4.cpp
    //   (4, 20)  gebsrmvn_general_kernel<128, 32>    row_block_dim_4.cpp
    //   (5, 4)   gebsrmvn_mxn_kernel<60, 5, 4>       row_block_dim_5_8.cpp
    //   (5, 9)   gebsrmvn_mxn_16_kernel<90, 5, 9>    row_block_dim_5_8.cpp
    //   (5, 12)  gebsrmvn_general_kernel<128, 16>    row_block_dim_5_8.cpp
    //   (10, 4)  gebsrmvn_mxn_16_kernel<40, 10, 4>   row_block_dim_9_12.cpp
    //   (10, 12) gebsrmvn_general_kernel<256, 16>    row_block_dim_9_12.cpp
    //   (14, 4)  gebsrmvn_mxn_16_kernel<56, 14, 4>   row_block_dim_13_16.cpp
    //   (14, 20) gebsrmvn_general_kernel<512, 32>    row_block_dim_13_16.cpp
    //   (20, 5)  gebsrmvn_general_kernel<256, 8>     row_block_dim_17_inf.cpp
    static std::vector<std::pair<rocsparse_int, rocsparse_int>> shapes()
    {
        return {{1, 2},
                {1, 20},
                {2, 3},
                {2, 20},
                {3, 2},
                {3, 20},
                {4, 3},
                {4, 20},
                {5, 4},
                {5, 9},
                {5, 12},
                {10, 4},
                {10, 12},
                {14, 4},
                {14, 20},
                {20, 5}};
    }

    // y = alpha * A * x + beta * y with mb = dispatch_limit(threads) + 2 and
    // blocks only in block rows 0, limit and limit + 1.
    void check_real_threshold(rocsparse_int rbd, rocsparse_int cbd, rocsparse_int threads)
    {
        const rocsparse_int     limit = dispatch_limit(threads);
        const rocsparse_int     mb    = limit + 2;
        const rocsparse_int     nb    = 1;
        constexpr rocsparse_int nnzb  = 3;
        const three_rows        rows(limit);
        const size_t            y_n = static_cast<size_t>(mb) * rbd;

        const size_t need
            = (static_cast<size_t>(mb) + 1) * sizeof(rocsparse_int) + y_n * sizeof(float);
        if(!device_has(need))
        {
            GTEST_SKIP() << "Insufficient device memory: need about " << need << " bytes";
        }

        const std::vector<rocsparse_int> row_ptr = sparse_row_ptr(mb, rows);
        const std::vector<rocsparse_int> col_ind(nnzb, 0);
        std::vector<float>               val(static_cast<size_t>(nnzb) * rbd * cbd);
        for(size_t k = 0; k < val.size(); ++k)
        {
            val[k] = static_cast<float>(1 + (k % 7));
        }
        std::vector<float> x(cbd);
        for(size_t j = 0; j < x.size(); ++j)
        {
            x[j] = static_cast<float>(1 + (j % 5));
        }

        device_vector<rocsparse_int> d_row_ptr{row_ptr};
        device_vector<rocsparse_int> d_col_ind{col_ind};
        device_vector<float>         d_val{val};
        device_vector<float>         d_x{x};
        device_vector<float>         d_y{std::vector<float>(y_n, Y_INIT)};
        ASSERT_TRUE(d_row_ptr.ptr && d_col_ind.ptr && d_val.ptr && d_x.ptr && d_y.ptr);

        MatDescr    descr;
        const float alpha = ALPHA;
        const float beta  = BETA;

        ASSERT_EQ(rocsparse_sgebsrmv(handle,
                                     rocsparse_direction_row,
                                     rocsparse_operation_none,
                                     mb,
                                     nb,
                                     nnzb,
                                     &alpha,
                                     descr.d,
                                     d_val,
                                     d_row_ptr,
                                     d_col_ind,
                                     rbd,
                                     cbd,
                                     d_x,
                                     &beta,
                                     d_y),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const std::vector<float> got = to_host(d_y);
        for(rocsparse_int i = 0; i < mb; ++i)
        {
            const int b = rows.block_of(i);
            for(rocsparse_int r = 0; r < rbd; ++r)
            {
                float acc = 0.0f;
                for(rocsparse_int c = 0; b >= 0 && c < cbd; ++c)
                {
                    acc += val[(static_cast<size_t>(b) * rbd + r) * cbd + c] * x[c];
                }
                const float  want = ALPHA * acc + BETA * Y_INIT;
                const size_t k    = static_cast<size_t>(i) * rbd + r;
                if(got[k] != want)
                {
                    ADD_FAILURE() << "y[" << k << "] (block row " << i << " of " << mb << ") is "
                                  << got[k] << ", want " << want << "; grid.x limit " << limit;
                    return;
                }
            }
        }
    }
};

// grid.x clamped well below the block-row count.
TEST_F(GebsrmvGridClamp, grid_smaller_than_block_rows)
{
    for(const auto& s : shapes())
    {
        check(s.first, s.second, rocsparse_direction_row, CLAMPED_GRID_X);
        check(s.first, s.second, rocsparse_direction_column, CLAMPED_GRID_X);
    }
}

// grid.x == 1, so a single block sweeps every block row. For the mxn and mxn_16
// kernels that also runs consecutive rows through the same shared sdata
// reduction, which may expose a missing barrier between iterations.
TEST_F(GebsrmvGridClamp, single_block_sweeps_every_block_row)
{
    for(const auto& s : shapes())
    {
        check(s.first, s.second, rocsparse_direction_row, 1);
        check(s.first, s.second, rocsparse_direction_column, 1);
    }
}

// (17, 18) reaches gebsrmvn_general_kernel<1024, 32> in row_block_dim_17_inf.cpp.
TEST_F(GebsrmvGridClamp, real_threshold_17x18)
{
    check_real_threshold(17, 18, 1024);
}

// ---------------------------------------------------------------------------
// gebsr2csr: 2 grids, one per __global__ kernel
// (gebsr2csr_block_per_row_1_32_kernel and _33_128_kernel).
// ---------------------------------------------------------------------------
class Gebsr2csrGridClamp : public HandleTest
{
protected:
    void check(rocsparse_int rbd, rocsparse_int cbd, rocsparse_direction dir, int grid_x)
    {
        const gebsr_matrix m = make_gebsr(rbd, cbd, dir);

        const rocsparse_int csr_m   = m.mb * rbd;
        const size_t        csr_nnz = static_cast<size_t>(m.nnzb) * rbd * cbd;

        device_vector<rocsparse_int> d_row_ptr{m.row_ptr};
        device_vector<rocsparse_int> d_col_ind{m.col_ind};
        device_vector<float>         d_val{m.val};
        device_vector<rocsparse_int> d_csr_row_ptr{std::vector<rocsparse_int>(csr_m + 1, -1)};
        device_vector<rocsparse_int> d_csr_col_ind{std::vector<rocsparse_int>(csr_nnz, -1)};
        device_vector<float>         d_csr_val{std::vector<float>(csr_nnz, -1.0f)};
        ASSERT_TRUE(d_row_ptr.ptr && d_col_ind.ptr && d_val.ptr && d_csr_row_ptr.ptr
                    && d_csr_col_ind.ptr && d_csr_val.ptr);

        MatDescr bsr_descr;
        MatDescr csr_descr;

        {
            ScopedMaxGridSizeX clamp(handle, grid_x);

            ASSERT_EQ(rocsparse_sgebsr2csr(handle,
                                           dir,
                                           m.mb,
                                           m.nb,
                                           bsr_descr.d,
                                           d_val,
                                           d_row_ptr,
                                           d_col_ind,
                                           rbd,
                                           cbd,
                                           csr_descr.d,
                                           d_csr_val,
                                           d_csr_row_ptr,
                                           d_csr_col_ind),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        }

        // Host reference. The block diagonal gives every CSR row exactly cbd
        // entries, in block column i == block row i.
        std::vector<rocsparse_int> want_row_ptr(csr_m + 1);
        std::vector<rocsparse_int> want_col_ind(csr_nnz);
        std::vector<float>         want_val(csr_nnz);
        for(rocsparse_int i = 0; i < m.mb; ++i)
        {
            for(rocsparse_int r = 0; r < rbd; ++r)
            {
                const rocsparse_int row   = i * rbd + r;
                const size_t        start = static_cast<size_t>(row) * cbd;
                want_row_ptr[row]         = static_cast<rocsparse_int>(start) + BASE;
                for(rocsparse_int c = 0; c < cbd; ++c)
                {
                    want_col_ind[start + c] = i * cbd + c + BASE;
                    want_val[start + c]     = m.val[m.at(i, r, c)];
                }
            }
        }
        want_row_ptr[csr_m] = static_cast<rocsparse_int>(csr_nnz) + BASE;

        const std::vector<rocsparse_int> got_row_ptr = to_host(d_csr_row_ptr.ptr, d_csr_row_ptr.n);
        const std::vector<rocsparse_int> got_col_ind = to_host(d_csr_col_ind.ptr, d_csr_col_ind.n);
        const std::vector<float>         got_val     = to_host(d_csr_val.ptr, d_csr_val.n);

        const std::string where = " row_block_dim=" + std::to_string(rbd) + " col_block_dim="
                                  + std::to_string(cbd) + " grid.x=" + std::to_string(grid_x);
        EXPECT_EQ(first_mismatch(got_row_ptr, want_row_ptr), -1) << "csr_row_ptr:" << where;
        EXPECT_EQ(first_mismatch(got_col_ind, want_col_ind), -1) << "csr_col_ind:" << where;
        EXPECT_EQ(first_mismatch(got_val, want_val), -1) << "csr_val:" << where;
    }

    // gebsr2csr with mb = dispatch_limit(threads) + 2 and blocks only in block
    // rows 0, limit and limit + 1. The kernels write csr_row_ptr for every block
    // row, so the empty rows past the clamped grid are checked too.
    void check_real_threshold(rocsparse_int rbd, rocsparse_int cbd, rocsparse_int threads)
    {
        const rocsparse_int     limit = dispatch_limit(threads);
        const rocsparse_int     mb    = limit + 2;
        const rocsparse_int     nb    = 1;
        constexpr rocsparse_int nnzb  = 3;
        const three_rows        rows(limit);
        const rocsparse_int     m   = mb * rbd;
        const rocsparse_int     nnz = nnzb * rbd * cbd;

        const size_t need
            = (static_cast<size_t>(mb) + static_cast<size_t>(m) + 2) * sizeof(rocsparse_int);
        if(!device_has(need))
        {
            GTEST_SKIP() << "Insufficient device memory: need about " << need << " bytes";
        }

        const std::vector<rocsparse_int> row_ptr = sparse_row_ptr(mb, rows);
        const std::vector<rocsparse_int> col_ind(nnzb, 0);
        std::vector<float>               val(static_cast<size_t>(nnz));
        for(size_t k = 0; k < val.size(); ++k)
        {
            val[k] = static_cast<float>(k + 1);
        }

        device_vector<rocsparse_int> d_row_ptr{row_ptr};
        device_vector<rocsparse_int> d_col_ind{col_ind};
        device_vector<float>         d_val{val};
        device_vector<rocsparse_int> d_csr_row_ptr(static_cast<size_t>(m) + 1);
        device_vector<rocsparse_int> d_csr_col_ind(static_cast<size_t>(nnz));
        device_vector<float>         d_csr_val(static_cast<size_t>(nnz));
        ASSERT_TRUE(d_row_ptr.ptr && d_col_ind.ptr && d_val.ptr && d_csr_row_ptr.ptr
                    && d_csr_col_ind.ptr && d_csr_val.ptr);
        ASSERT_EQ(hipMemset(d_csr_row_ptr.ptr, 0xFF, d_csr_row_ptr.n * sizeof(rocsparse_int)),
                  hipSuccess);
        ASSERT_EQ(hipMemset(d_csr_col_ind.ptr, 0xFF, d_csr_col_ind.n * sizeof(rocsparse_int)),
                  hipSuccess);
        ASSERT_EQ(hipMemset(d_csr_val.ptr, 0xFF, d_csr_val.n * sizeof(float)), hipSuccess);

        MatDescr bsr_descr;
        MatDescr csr_descr;

        ASSERT_EQ(rocsparse_sgebsr2csr(handle,
                                       rocsparse_direction_row,
                                       mb,
                                       nb,
                                       bsr_descr.d,
                                       d_val,
                                       d_row_ptr,
                                       d_col_ind,
                                       rbd,
                                       cbd,
                                       csr_descr.d,
                                       d_csr_val,
                                       d_csr_row_ptr,
                                       d_csr_col_ind),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const std::vector<rocsparse_int> got_ptr = to_host(d_csr_row_ptr);
        ASSERT_EQ(got_ptr[0], BASE) << "csr_row_ptr[0]";
        rocsparse_int running = BASE;
        for(rocsparse_int i = 0; i < mb; ++i)
        {
            const rocsparse_int per_row = (rows.block_of(i) >= 0) ? cbd : 0;
            for(rocsparse_int r = 0; r < rbd; ++r)
            {
                running += per_row;
                const size_t k = static_cast<size_t>(i) * rbd + r + 1;
                if(got_ptr[k] != running)
                {
                    ADD_FAILURE() << "csr_row_ptr[" << k << "] (block row " << i << " of " << mb
                                  << ") is " << got_ptr[k] << ", want " << running
                                  << "; grid.x limit " << limit;
                    return;
                }
            }
        }

        // Each block row holds one block in block column 0, so in row direction
        // the CSR entries of block b are bsr_val of block b in the same order.
        const std::vector<rocsparse_int> got_ind = to_host(d_csr_col_ind);
        const std::vector<float>         got_val = to_host(d_csr_val);
        for(rocsparse_int k = 0; k < nnz; ++k)
        {
            if(got_ind[k] != (k % cbd) + BASE || got_val[k] != val[k])
            {
                ADD_FAILURE() << "csr entry " << k << " ind " << got_ind[k] << " val "
                              << got_val[k];
                return;
            }
        }
    }
};

// (2, 4) and (4, 2) reach gebsr2csr_block_per_row_1_32_kernel<256, 2, 4> and
// <256, 4, 2>; (2, 64) reaches gebsr2csr_block_per_row_33_128_kernel.
TEST_F(Gebsr2csrGridClamp, grid_smaller_than_block_rows)
{
    for(int grid_x : {CLAMPED_GRID_X, 1})
    {
        for(const auto dir : {rocsparse_direction_row, rocsparse_direction_column})
        {
            check(2, 4, dir, grid_x);
            check(4, 2, dir, grid_x);
            check(2, 64, dir, grid_x);
        }
    }
}

// gebsr2csr_block_per_row_1_32_kernel<1024, 4, 16>.
TEST_F(Gebsr2csrGridClamp, real_threshold_4x16)
{
    check_real_threshold(4, 16, 1024);
}

// gebsr2csr_block_per_row_33_128_kernel<1024, 4, 64, 4, 32>.
TEST_F(Gebsr2csrGridClamp, real_threshold_4x64)
{
    check_real_threshold(4, 64, 1024);
}

// ---------------------------------------------------------------------------
// bsrgeam: 4 grids. block_dim 4 and 5-8 launch bsrgeam_wf_per_row_multipass_kernel
// <64, 4> and <64, 8>; 9-16 and 17-32 launch
// bsrgeam_block_per_row_multipass_kernel2 <256, 16> and <256, 32>.
// ---------------------------------------------------------------------------
class BsrgeamGridClamp : public HandleTest
{
protected:
    void check(rocsparse_int block_dim, rocsparse_direction dir, int grid_x)
    {
        // A on the block diagonal, B on the block diagonal too, so C has the
        // same structure and every block of C mixes a block of A and one of B.
        const gebsr_matrix a = make_gebsr(block_dim, block_dim, dir);
        gebsr_matrix       b = make_gebsr(block_dim, block_dim, dir);
        for(size_t i = 0; i < b.val.size(); ++i)
        {
            b.val[i] += 1.0f;
        }

        device_vector<rocsparse_int> d_row_ptr_a{a.row_ptr};
        device_vector<rocsparse_int> d_col_ind_a{a.col_ind};
        device_vector<float>         d_val_a{a.val};
        device_vector<rocsparse_int> d_row_ptr_b{b.row_ptr};
        device_vector<rocsparse_int> d_col_ind_b{b.col_ind};
        device_vector<float>         d_val_b{b.val};
        device_vector<rocsparse_int> d_row_ptr_c{std::vector<rocsparse_int>(MB + 1, -1)};
        ASSERT_TRUE(d_row_ptr_a.ptr && d_col_ind_a.ptr && d_val_a.ptr && d_row_ptr_b.ptr
                    && d_col_ind_b.ptr && d_val_b.ptr && d_row_ptr_c.ptr);

        MatDescr descr_a, descr_b, descr_c;

        // The structure pass is a different set of kernels; run it at the real
        // grid limit.
        rocsparse_int nnzb_c = 0;
        ASSERT_EQ(rocsparse_bsrgeam_nnzb(handle,
                                         dir,
                                         a.mb,
                                         a.nb,
                                         block_dim,
                                         descr_a.d,
                                         a.nnzb,
                                         d_row_ptr_a,
                                         d_col_ind_a,
                                         descr_b.d,
                                         b.nnzb,
                                         d_row_ptr_b,
                                         d_col_ind_b,
                                         descr_c.d,
                                         d_row_ptr_c,
                                         &nnzb_c),
                  rocsparse_status_success);
        ASSERT_EQ(nnzb_c, MB);

        device_vector<rocsparse_int> d_col_ind_c{std::vector<rocsparse_int>(nnzb_c, -1)};
        device_vector<float>         d_val_c{
            std::vector<float>(static_cast<size_t>(nnzb_c) * block_dim * block_dim, Y_INIT)};
        ASSERT_TRUE(d_col_ind_c.ptr && d_val_c.ptr);

        const float alpha = ALPHA;
        const float beta  = BETA;

        {
            ScopedMaxGridSizeX clamp(handle, grid_x);

            ASSERT_EQ(rocsparse_sbsrgeam(handle,
                                         dir,
                                         a.mb,
                                         a.nb,
                                         block_dim,
                                         &alpha,
                                         descr_a.d,
                                         a.nnzb,
                                         d_val_a,
                                         d_row_ptr_a,
                                         d_col_ind_a,
                                         &beta,
                                         descr_b.d,
                                         b.nnzb,
                                         d_val_b,
                                         d_row_ptr_b,
                                         d_col_ind_b,
                                         descr_c.d,
                                         d_val_c,
                                         d_row_ptr_c,
                                         d_col_ind_c),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        }

        std::vector<rocsparse_int> want_col(MB);
        std::vector<float>         want(static_cast<size_t>(nnzb_c) * block_dim * block_dim);
        for(rocsparse_int i = 0; i < MB; ++i)
        {
            want_col[i] = i + BASE;
            for(rocsparse_int r = 0; r < block_dim; ++r)
            {
                for(rocsparse_int c = 0; c < block_dim; ++c)
                {
                    want[a.at(i, r, c)]
                        = ALPHA * a.val[a.at(i, r, c)] + BETA * b.val[b.at(i, r, c)];
                }
            }
        }

        const std::vector<rocsparse_int> got_col = to_host(d_col_ind_c.ptr, d_col_ind_c.n);
        const int64_t                    bad_col = first_mismatch(got_col, want_col);
        EXPECT_EQ(bad_col, -1) << "block_dim=" << block_dim << " dir=" << static_cast<int>(dir)
                               << ": first wrong bsr_col_ind_C entry " << bad_col << " of " << MB
                               << ". grid.x was clamped to " << grid_x;

        const std::vector<float> got = to_host(d_val_c.ptr, d_val_c.n);
        const int64_t            bad = first_mismatch(got, want);
        EXPECT_EQ(bad, -1) << "block_dim=" << block_dim << " dir=" << static_cast<int>(dir)
                           << ": first wrong bsr_val_C entry " << bad << " (block row "
                           << (bad / (block_dim * block_dim)) << " of " << MB << ") got "
                           << got[bad] << " want " << want[bad] << ". grid.x was clamped to "
                           << grid_x;
    }

    // C = alpha * A + beta * B with mb = dispatch_limit(threads) + 2 and A, B
    // and C all holding one block in each of block rows 0, limit and limit + 1.
    // A, B and C share one row pointer; the routine only reads it.
    void check_real_threshold(rocsparse_int block_dim, rocsparse_int threads)
    {
        const rocsparse_int     limit = dispatch_limit(threads);
        const rocsparse_int     mb    = limit + 2;
        const rocsparse_int     nb    = 1;
        constexpr rocsparse_int nnzb  = 3;
        const three_rows        rows(limit);
        const size_t            bs = static_cast<size_t>(block_dim) * block_dim;

        const size_t need = (static_cast<size_t>(mb) + 1) * sizeof(rocsparse_int);
        if(!device_has(need))
        {
            GTEST_SKIP() << "Insufficient device memory: need about " << need << " bytes";
        }

        const std::vector<rocsparse_int> row_ptr = sparse_row_ptr(mb, rows);
        const std::vector<rocsparse_int> col_ind(nnzb, 0);
        std::vector<float>               val_a(nnzb * bs);
        std::vector<float>               val_b(nnzb * bs);
        for(size_t k = 0; k < val_a.size(); ++k)
        {
            val_a[k] = static_cast<float>(1 + (k % 7));
            val_b[k] = static_cast<float>(1 + (k % 5));
        }

        device_vector<rocsparse_int> d_row_ptr{row_ptr};
        device_vector<rocsparse_int> d_col_ind{col_ind};
        device_vector<float>         d_val_a{val_a};
        device_vector<float>         d_val_b{val_b};
        device_vector<rocsparse_int> d_col_ind_c{std::vector<rocsparse_int>(nnzb, -1)};
        device_vector<float>         d_val_c{std::vector<float>(nnzb * bs, Y_INIT)};
        ASSERT_TRUE(d_row_ptr.ptr && d_col_ind.ptr && d_val_a.ptr && d_val_b.ptr && d_col_ind_c.ptr
                    && d_val_c.ptr);

        MatDescr    descr_a, descr_b, descr_c;
        const float alpha = ALPHA;
        const float beta  = BETA;

        ASSERT_EQ(rocsparse_sbsrgeam(handle,
                                     rocsparse_direction_row,
                                     mb,
                                     nb,
                                     block_dim,
                                     &alpha,
                                     descr_a.d,
                                     nnzb,
                                     d_val_a,
                                     d_row_ptr,
                                     d_col_ind,
                                     &beta,
                                     descr_b.d,
                                     nnzb,
                                     d_val_b,
                                     d_row_ptr,
                                     d_col_ind,
                                     descr_c.d,
                                     d_val_c,
                                     d_row_ptr,
                                     d_col_ind_c),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const std::vector<rocsparse_int> got_col = to_host(d_col_ind_c);
        const std::vector<float>         got_val = to_host(d_val_c);
        for(int b = 0; b < nnzb; ++b)
        {
            EXPECT_EQ(got_col[b], BASE) << "bsr_col_ind_C of block row " << rows.row[b] << " of "
                                        << mb << "; grid.x limit " << limit;
            for(size_t k = b * bs; k < (b + 1) * bs; ++k)
            {
                const float want = ALPHA * val_a[k] + BETA * val_b[k];
                if(got_val[k] != want)
                {
                    ADD_FAILURE() << "bsr_val_C[" << k << "] (block row " << rows.row[b] << " of "
                                  << mb << ") is " << got_val[k] << ", want " << want
                                  << "; grid.x limit " << limit;
                    return;
                }
            }
        }
    }
};

// block_dim 4 and 6 reach the wf_per_row kernels, 16 and 32 the block_per_row
// ones. grid.x == 1 also runs consecutive rows through the same shared arrays,
// which may expose a missing barrier between iterations.
TEST_F(BsrgeamGridClamp, grid_smaller_than_block_rows)
{
    for(int grid_x : {CLAMPED_GRID_X, 1})
    {
        for(const auto dir : {rocsparse_direction_row, rocsparse_direction_column})
        {
            check(4, dir, grid_x);
            check(6, dir, grid_x);
            check(16, dir, grid_x);
            check(32, dir, grid_x);
        }
    }
}

// bsrgeam_wf_per_row_multipass_kernel<64, 8>: one block row per 64-thread block.
TEST_F(BsrgeamGridClamp, real_threshold_block_dim_8)
{
    check_real_threshold(8, 64);
}

// bsrgeam_block_per_row_multipass_kernel2<256, 16>.
TEST_F(BsrgeamGridClamp, real_threshold_block_dim_16)
{
    check_real_threshold(16, 256);
}

// bsrgeam_block_per_row_multipass_kernel2<256, 32>.
TEST_F(BsrgeamGridClamp, real_threshold_block_dim_32)
{
    check_real_threshold(32, 256);
}
