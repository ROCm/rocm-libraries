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
// Regression tests for the one-block-per-block-row compute grids in gebsrmv,
// gebsr2csr and bsrgeam (AISPARSE-705).
//
// FOCUS: twenty launches across those three routines sized grid.x at one block
// (or one wavefront) per block row straight from a rocsparse_int block-row
// count, with no clamp against handle->properties.maxGridSize[0] and no
// grid-stride loop in the kernel behind it. With BUILD_ROCSPARSE_ILP64=ON
// rocsparse_int is int64_t, so dim3 narrows the count to unsigned int and the
// kernels silently leave most of the matrix unprocessed. The fix clamps every
// grid and grid-strides the block rows.
//
// HOW THIS IS TESTED CHEAPLY: reproducing the narrowing for real needs more
// than 2^32 block rows, which is far more memory than any GPU here has -- and
// it needs an ILP64 build, which nothing in CI produces. Instead these tests
// shrink handle->properties.maxGridSize[0] -- the very limit the new code
// clamps against -- to a handful of blocks and drive a SMALL matrix through the
// public entry points. That reproduces the defect's actual mechanism, a launch
// grid smaller than the work, on a few kilobytes: with MB = 40 block rows and
// the limit set to 3, block rows 3..39 are reached ONLY by the new grid-stride
// loops. This is the AISPARSE-702 idiom.
//
// WHAT THESE CASES DO AND DO NOT PROVE. They are load-bearing for the
// grid-stride loops, which are the half of the fix that can silently corrupt a
// result at any matrix size. They cannot fail on the clamp alone: removing only
// the clamp makes the grid larger (40 blocks instead of 3), and a larger grid
// is still covered correctly by the stride loop. The clamp guards the dim3
// narrowing, whose only observable effect needs an ILP64 build and more than
// 2^32 block rows, so no affordable test can reach it. Both halves of that
// statement were checked by breaking the source and rebuilding with
// CCACHE_DISABLE=1.
//
// There is deliberately NO device-memory guard and no size-based skip: the
// footprint is a few hundred kilobytes, so every case below runs on every GPU,
// including the 15 GB gfx1201.
//
#include "unit_test_utils.hpp"

#include "rocsparse.h"

// Internal handle definition. These tests need the complete _rocsparse_handle
// type to shrink handle->properties.maxGridSize[0]; that field is what the
// hardened launch paths clamp their grids against, and shrinking it is the only
// way to put the grid below the work without an enormous matrix.
#include "rocsparse_handle.hpp"

#include <vector>

using namespace rocsparse_ut;

namespace
{
    constexpr rocsparse_index_base BASE = rocsparse_index_base_zero;

    // Number of block rows. Small enough to stay in a few hundred kilobytes,
    // far more than the clamped grid below so the grid-stride loops do most of
    // the work.
    constexpr rocsparse_int MB = 40;

    // The shrunk grid.x limit. 3 blocks for 40 block rows means ~92% of the
    // block rows are reached only by a grid-stride loop.
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
        MatDescr(const MatDescr&)            = delete;
        MatDescr& operator=(const MatDescr&) = delete;
    };

    // Temporarily shrink the grid.x limit the hardened launch paths clamp
    // against, and restore it on scope exit so a failed assertion cannot leak
    // the override into another assertion in the same test.
    //
    // Duplicated from AISPARSE-699/700's clients/unittests/unit_test_grid_clamp.hpp
    // rather than included from it: that header is on a sibling branch that has
    // not merged, and a test that does not compile standalone is worse than
    // fifteen duplicated lines. Collapse to an include once it lands.
    struct ScopedMaxGridSizeX
    {
        rocsparse_handle handle;
        int              saved;

        ScopedMaxGridSizeX(rocsparse_handle h, int limit)
            : handle(h)
            , saved(h->properties.maxGridSize[0])
        {
            handle->properties.maxGridSize[0] = limit;
        }
        ~ScopedMaxGridSizeX()
        {
            handle->properties.maxGridSize[0] = saved;
        }
        ScopedMaxGridSizeX(const ScopedMaxGridSizeX&)            = delete;
        ScopedMaxGridSizeX& operator=(const ScopedMaxGridSizeX&) = delete;
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
}

// ---------------------------------------------------------------------------
// gebsrmv: 16 grids across the eight rocsparse_gebsrmv_template_row_block_dim_*
// files, behind seven device functions (general, 1xn, 2xn, 3xn, 4xn, mxn,
// mxn_16). The (row_block_dim, col_block_dim) pairs below are chosen to reach
// every one of the seven.
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
                           << grid_x
                           << ", so block rows past it are covered only by the kernel's "
                              "grid-stride loop.";
    }

    // One pair per device function reached by rocsparse_gebsrmv's dispatch:
    //   (1, 2)   -> gebsrmvn_1xn_device      (row_block_dim_1.cpp)
    //   (2, 2)   -> gebsrmvn_2xn_device      (row_block_dim_2.cpp)
    //   (3, 3)   -> gebsrmvn_3xn_device      (row_block_dim_3.cpp)
    //   (4, 4)   -> gebsrmvn_4xn_device      (row_block_dim_4.cpp)
    //   (5, 5)   -> gebsrmvn_mxn_device      (row_block_dim_5_8.cpp)
    //   (5, 9)   -> gebsrmvn_mxn_16_device   (row_block_dim_5_8.cpp)
    //   (10, 10) -> gebsrmvn_mxn_16_device   (row_block_dim_9_12.cpp)
    //   (14, 14) -> gebsrmvn_mxn_16_device   (row_block_dim_13_16.cpp)
    //   (5, 12)  -> gebsrmvn_general_device  (row_block_dim_5_8.cpp)
    //   (20, 20) -> gebsrmvn_general_device  (row_block_dim_17_inf.cpp)
    static std::vector<std::pair<rocsparse_int, rocsparse_int>> shapes()
    {
        return {
            {1, 2}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {5, 9}, {10, 10}, {14, 14}, {5, 12}, {20, 20}};
    }
};

// grid.x clamped well below the block-row count: block rows 3..39 are reached
// only by the grid-stride loop.
TEST_F(GebsrmvGridClamp, grid_smaller_than_block_rows)
{
    for(const auto& s : shapes())
    {
        check(s.first, s.second, rocsparse_direction_row, CLAMPED_GRID_X);
        check(s.first, s.second, rocsparse_direction_column, CLAMPED_GRID_X);
    }
}

// Hardest case for the loop: grid.x == 1, so a single block sweeps all 40 block
// rows sequentially. For the mxn and mxn_16 kernels that also puts 40
// consecutive iterations through the shared sdata reduction, so a missing
// __syncthreads() between iterations -- one iteration's trailing reads of sdata
// racing the next one's first store -- shows up here as a wrong y entry.
TEST_F(GebsrmvGridClamp, single_block_sweeps_every_block_row)
{
    for(const auto& s : shapes())
    {
        check(s.first, s.second, rocsparse_direction_row, 1);
        check(s.first, s.second, rocsparse_direction_column, 1);
    }
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
};

// (2, 2) reaches gebsr2csr_block_per_row_1_32_kernel, (2, 64) reaches
// gebsr2csr_block_per_row_33_128_kernel; both grids are sized from mb.
TEST_F(Gebsr2csrGridClamp, grid_smaller_than_block_rows)
{
    for(int grid_x : {CLAMPED_GRID_X, 1})
    {
        check(2, 2, rocsparse_direction_row, grid_x);
        check(2, 2, rocsparse_direction_column, grid_x);
        check(2, 64, rocsparse_direction_row, grid_x);
        check(2, 64, rocsparse_direction_column, grid_x);
    }
}

// ---------------------------------------------------------------------------
// bsrgeam: 2 grids, both behind bsrgeam_block_per_row_multipass_device2, which
// rocsparse_bsrgeam dispatches to for 8 < block_dim <= 32.
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

        // The structure pass is a different set of kernels, outside this
        // change; run it at the real grid limit.
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

        std::vector<float> want(static_cast<size_t>(nnzb_c) * block_dim * block_dim);
        for(rocsparse_int i = 0; i < MB; ++i)
        {
            for(rocsparse_int r = 0; r < block_dim; ++r)
            {
                for(rocsparse_int c = 0; c < block_dim; ++c)
                {
                    want[a.at(i, r, c)]
                        = ALPHA * a.val[a.at(i, r, c)] + BETA * b.val[b.at(i, r, c)];
                }
            }
        }

        const std::vector<float> got = to_host(d_val_c.ptr, d_val_c.n);
        const int64_t            bad = first_mismatch(got, want);
        EXPECT_EQ(bad, -1) << "block_dim=" << block_dim << " dir=" << static_cast<int>(dir)
                           << ": first wrong bsr_val_C entry " << bad << " (block row "
                           << (bad / (block_dim * block_dim)) << " of " << MB << ") got "
                           << got[bad] << " want " << want[bad] << ". grid.x was clamped to "
                           << grid_x
                           << ", so block rows past it are covered only by the kernel's "
                              "grid-stride loop.";
    }
};

// block_dim 16 reaches the <= 16 launch, block_dim 32 the <= 32 one; those are
// the two grids AISPARSE-705 covers in this file. grid.x == 1 additionally runs
// 40 consecutive multipass sweeps through the same shared table/data arrays,
// which is what the __syncthreads() added after each iteration guards.
TEST_F(BsrgeamGridClamp, grid_smaller_than_block_rows)
{
    for(int grid_x : {CLAMPED_GRID_X, 1})
    {
        check(16, rocsparse_direction_row, grid_x);
        check(16, rocsparse_direction_column, grid_x);
        check(32, rocsparse_direction_row, grid_x);
        check(32, rocsparse_direction_column, grid_x);
    }
}
