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
// Regression tests for the rocsparse_Xbsrxmv one-block-per-block-row compute
// grids (AISPARSE-702).
//
// FOCUS: the bsrxmv specialised kernels launch one block per block row (or per
// small group of block rows) and used to pass the block-row count straight into
// dim3. That count is the template parameter J, instantiated as int64_t, so
// dim3 narrowed it to unsigned int: past 2^32-1 block rows the grid wrapped and
// the kernels -- which read hipBlockIdx_x with no grid-stride loop -- silently
// left most of the matrix unprocessed and returned a wrong result with no
// error. The fix clamps every grid against handle->properties.maxGridSize[0]
// and adds a grid-stride loop over block rows to each kernel.
//
// HOW THIS IS TESTED CHEAPLY: reproducing the overflow for real needs ~2^31
// block rows (about 17 GB of row pointers). Instead these tests shrink
// handle->properties.maxGridSize[0] -- the very limit the new code clamps
// against -- to a handful of blocks and run a SMALL matrix through the public
// rocsparse_sbsrxmv entry point. That reproduces the defect's actual mechanism
// (a launch grid smaller than the work) on a few kilobytes: with mb = 40 block
// rows and the limit set to 3, block rows 3..39 are reached ONLY by the new
// grid-stride loop. Drop that loop and the tail of y keeps its input value,
// which these tests detect.
//
// WHAT THESE CASES DO AND DO NOT PROVE. They are load-bearing for the
// grid-stride loop, which is the half of the fix that can silently corrupt a
// result at any matrix size: disable the stride and all four cases fail at
// block row 3, the first row past the clamped grid. They cannot fail on the
// clamp alone. Removing the clamp only makes the grid larger (40 blocks
// instead of 3), and a larger grid is still covered correctly by the stride
// loop, so all four cases still pass. The clamp guards the dim3 narrowing,
// whose only observable effect needs more than 2^32 block rows -- the 17 GB
// this file exists to avoid allocating -- so no affordable test can reach it.
// Both halves of that statement were checked by breaking the source and
// rebuilding with CCACHE_DISABLE=1.
//
// There is deliberately NO device-memory guard and no size-based skip: the
// footprint is well under a megabyte, so every case below runs on every GPU,
// including the 15 GB gfx1201.
//
// ARCHITECTURE COVERAGE: rocsparse::bsrxmv_template_dispatch routes *every*
// block_dim through bsrxmvn_general when handle->wavefront_size == 32, so on a
// wave32 part these cases exercise the five grids in
// rocsparse_bsrxmv_spzl_general.cpp. On wave64 parts the same cases dispatch to
// the block-dim specialisations, so block_dim 5/8/16/17/32 additionally cover
// rocsparse_bsrxmv_spzl_5x5.cpp, _8x8.cpp, _16x16.cpp and _17_32.cpp. One test
// case set, both architectures.
//
#include "unit_test_utils.hpp"

#include "rocsparse.h"

// Internal handle definition. These tests need the complete _rocsparse_handle
// type to shrink handle->properties.maxGridSize[0]; that field is what the
// bsrxmv launch paths clamp their grids against, and shrinking it is the only
// way to put the grid below the work without allocating a 17 GB matrix.
#include "rocsparse_handle.hpp"

#include <vector>

using namespace rocsparse_ut;

namespace
{
    constexpr rocsparse_index_base BASE = rocsparse_index_base_zero;

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
    };

    // Temporarily shrink the grid.x limit the bsrxmv launch paths clamp against,
    // and restore it on scope exit so a failed assertion cannot leak the
    // override into another assertion in the same test.
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
    };

    // Number of block rows. Small enough to stay in a few kilobytes, far more
    // than the clamped grid below so the grid-stride loop does most of the work.
    constexpr rocsparse_int MB = 40;

    // The shrunk grid.x limit. 3 blocks for 40 (or 20 masked) block rows means
    // ~92% of the block rows are reached only by the grid-stride loop.
    constexpr int CLAMPED_GRID_X = 3;

    // y entries of block rows the routine must NOT touch keep this value, and
    // the beta term of the rows it does touch is computed from it. Non-zero so
    // the beta != 0 branch of every kernel is the one exercised.
    constexpr float Y_INIT = 7.0f;

    constexpr float ALPHA = 2.0f;
    constexpr float BETA  = 3.0f;

    // One block per block row, on the diagonal: block row i holds a single
    // block_dim x block_dim block in block column i.
    struct bsr_matrix
    {
        rocsparse_int              mb;
        rocsparse_int              nb;
        rocsparse_int              nnzb;
        rocsparse_int              block_dim;
        std::vector<rocsparse_int> row_ptr; // start offset of block row i
        std::vector<rocsparse_int> end_ptr; // end offset of block row i
        std::vector<rocsparse_int> col_ind;
        std::vector<float>         val;
        std::vector<float>         x;
    };

    // Build the diagonal test matrix. Entries vary with (block, row, column) so
    // a kernel that mixes up block rows, or drops some, cannot accidentally
    // produce the reference result. All values are small integers, so both the
    // device and the host reference evaluate them exactly in float.
    bsr_matrix make_matrix(rocsparse_int block_dim, rocsparse_direction dir)
    {
        bsr_matrix m;
        m.mb        = MB;
        m.nb        = MB;
        m.nnzb      = MB;
        m.block_dim = block_dim;

        m.row_ptr.resize(MB);
        m.end_ptr.resize(MB);
        m.col_ind.resize(MB);
        for(rocsparse_int i = 0; i < MB; ++i)
        {
            m.row_ptr[i] = i;
            m.end_ptr[i] = i + 1;
            m.col_ind[i] = i;
        }

        m.val.resize(static_cast<size_t>(MB) * block_dim * block_dim);
        for(rocsparse_int i = 0; i < MB; ++i)
        {
            for(rocsparse_int r = 0; r < block_dim; ++r)
            {
                for(rocsparse_int c = 0; c < block_dim; ++c)
                {
                    const float  v    = static_cast<float>(1 + ((i + 2 * r + 3 * c) % 7));
                    const size_t base = static_cast<size_t>(i) * block_dim * block_dim;
                    // rocSPARSE stores each block either row-major
                    // (rocsparse_direction_row) or column-major.
                    m.val[base
                          + (dir == rocsparse_direction_row
                                 ? static_cast<size_t>(r) * block_dim + c
                                 : static_cast<size_t>(c) * block_dim + r)]
                        = v;
                }
            }
        }

        m.x.resize(static_cast<size_t>(MB) * block_dim);
        for(size_t j = 0; j < m.x.size(); ++j)
        {
            m.x[j] = static_cast<float>(1 + (j % 5));
        }
        return m;
    }

    // y = alpha * A * x + beta * y, restricted to the block rows `mask`
    // selects (all of them when `mask` is empty). Block rows outside the mask
    // are left at Y_INIT, exactly as the routine must leave them.
    std::vector<float> reference(const bsr_matrix&                 m,
                                 rocsparse_direction               dir,
                                 const std::vector<rocsparse_int>& mask)
    {
        std::vector<float> y(static_cast<size_t>(m.mb) * m.block_dim, Y_INIT);

        std::vector<rocsparse_int> rows;
        if(mask.empty())
        {
            for(rocsparse_int i = 0; i < m.mb; ++i)
            {
                rows.push_back(i);
            }
        }
        else
        {
            rows = mask;
        }

        for(rocsparse_int i : rows)
        {
            for(rocsparse_int r = 0; r < m.block_dim; ++r)
            {
                float acc = 0.0f;
                for(rocsparse_int j = m.row_ptr[i]; j < m.end_ptr[i]; ++j)
                {
                    const rocsparse_int col  = m.col_ind[j];
                    const size_t        base = static_cast<size_t>(j) * m.block_dim * m.block_dim;
                    for(rocsparse_int c = 0; c < m.block_dim; ++c)
                    {
                        const float a = m.val[base
                                              + (dir == rocsparse_direction_row
                                                     ? static_cast<size_t>(r) * m.block_dim + c
                                                     : static_cast<size_t>(c) * m.block_dim + r)];
                        acc += a * m.x[static_cast<size_t>(col) * m.block_dim + c];
                    }
                }
                const size_t idx = static_cast<size_t>(i) * m.block_dim + r;
                y[idx]           = ALPHA * acc + BETA * Y_INIT;
            }
        }
        return y;
    }

    // Report the first index at which `got` and `want` differ, else -1. A single
    // precise failure beats thousands of EXPECT_FLOAT_EQ macros in a loop.
    int64_t first_mismatch(const std::vector<float>& got, const std::vector<float>& want)
    {
        for(size_t i = 0; i < got.size(); ++i)
        {
            if(got[i] != want[i])
            {
                return static_cast<int64_t>(i);
            }
        }
        return -1;
    }
}

class BsrxmvGridClamp : public HandleTest
{
protected:
    // Run rocsparse_sbsrxmv on the small diagonal matrix with grid.x clamped to
    // CLAMPED_GRID_X, and compare the whole of y against the host reference.
    // `mask` empty means the unmasked path (grid sized from mb); otherwise the
    // masked path (grid sized from size_of_mask).
    void check(rocsparse_int                     block_dim,
               rocsparse_direction               dir,
               const std::vector<rocsparse_int>& mask)
    {
        check_with_grid_x(block_dim, dir, mask, CLAMPED_GRID_X);
    }

    // As `check`, but with the grid.x limit given explicitly.
    void check_with_grid_x(rocsparse_int                     block_dim,
                           rocsparse_direction               dir,
                           const std::vector<rocsparse_int>& mask,
                           int                               grid_x)
    {
        const bsr_matrix m = make_matrix(block_dim, dir);

        device_vector<rocsparse_int> d_row_ptr{m.row_ptr};
        device_vector<rocsparse_int> d_end_ptr{m.end_ptr};
        device_vector<rocsparse_int> d_col_ind{m.col_ind};
        device_vector<float>         d_val{m.val};
        device_vector<float>         d_x{m.x};
        device_vector<float> d_y{std::vector<float>(static_cast<size_t>(m.mb) * block_dim, Y_INIT)};
        ASSERT_TRUE(d_row_ptr.ptr && d_end_ptr.ptr && d_col_ind.ptr && d_val.ptr && d_x.ptr
                    && d_y.ptr);

        // The unmasked path is selected by a NULL bsr_mask_ptr with
        // size_of_mask = 0 (ROCSPARSE_CHECKARG_ARRAY only rejects a null
        // pointer when the size is non-zero). The kernels branch on the
        // pointer, not the size, so a non-null pointer with size 0 would take
        // the masked path over an empty mask instead.
        device_vector<rocsparse_int> d_mask{mask.empty() ? std::vector<rocsparse_int>{0} : mask};
        ASSERT_TRUE(d_mask.ptr);
        const rocsparse_int* mask_arg = mask.empty() ? nullptr : d_mask.ptr;

        MatDescr    descr;
        const float alpha = ALPHA;
        const float beta  = BETA;

        {
            ScopedMaxGridSizeX clamp(handle, grid_x);

            ASSERT_EQ(rocsparse_sbsrxmv(handle,
                                        dir,
                                        rocsparse_operation_none,
                                        static_cast<rocsparse_int>(mask.size()),
                                        m.mb,
                                        m.nb,
                                        m.nnzb,
                                        &alpha,
                                        descr.d,
                                        d_val,
                                        mask_arg,
                                        d_row_ptr,
                                        d_end_ptr,
                                        d_col_ind,
                                        block_dim,
                                        d_x,
                                        &beta,
                                        d_y),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        }

        const std::vector<float> got  = to_host(d_y.ptr, d_y.n);
        const std::vector<float> want = reference(m, dir, mask);
        ASSERT_EQ(got.size(), want.size());

        const int64_t bad = first_mismatch(got, want);
        EXPECT_EQ(bad, -1) << "block_dim=" << block_dim << " dir=" << static_cast<int>(dir)
                           << (mask.empty() ? " unmasked" : " masked") << ": first wrong y entry "
                           << bad << " (block row " << (bad / block_dim) << " of " << m.mb
                           << ") got " << got[bad] << " want " << want[bad]
                           << ". grid.x was clamped to " << grid_x
                           << ", so block rows past it are covered only by the kernel's "
                              "grid-stride loop.";
    }

    // Block dims whose wave64 dispatch target is one of the files AISPARSE-702
    // fixes (5x5, 8x8, 16x16, 17_32). Safe to assert on every architecture.
    static std::vector<rocsparse_int> block_dims_in_scope()
    {
        return {5, 8, 16, 17, 32};
    }
};

// Unmasked: the grid is sized from mb, and every block row must be computed.
TEST_F(BsrxmvGridClamp, unmasked_grid_smaller_than_block_rows)
{
    for(rocsparse_int block_dim : block_dims_in_scope())
    {
        check(block_dim, rocsparse_direction_row, {});
        check(block_dim, rocsparse_direction_column, {});
    }
}

// Masked: the grid is sized from size_of_mask instead of mb. Both are template
// J, so both need the clamp and the grid-stride loop; this case covers the
// bsr_mask_ptr != nullptr side of that branch, including that block rows the
// mask does not select stay untouched.
TEST_F(BsrxmvGridClamp, masked_grid_smaller_than_mask_size)
{
    // Every other block row, so 20 mask entries against a grid of 3.
    std::vector<rocsparse_int> mask;
    for(rocsparse_int i = 0; i < MB; i += 2)
    {
        mask.push_back(i);
    }

    for(rocsparse_int block_dim : block_dims_in_scope())
    {
        check(block_dim, rocsparse_direction_row, mask);
        check(block_dim, rocsparse_direction_column, mask);
    }
}

// Hardest case for the grid-stride loop: grid.x == 1, so one single block must
// sweep all 40 block rows sequentially instead of ~13. On wave64 parts, where
// these block dims reach the block-dim specialisations, that also puts 40
// consecutive iterations through each kernel's shared sdata reduction, so a
// missing __syncthreads() between iterations (one iteration's trailing reads of
// sdata racing the next one's first store) shows up here as a wrong y entry.
TEST_F(BsrxmvGridClamp, single_block_sweeps_every_block_row)
{
    std::vector<rocsparse_int> mask;
    for(rocsparse_int i = 0; i < MB; i += 2)
    {
        mask.push_back(i);
    }

    for(rocsparse_int block_dim : block_dims_in_scope())
    {
        check_with_grid_x(block_dim, rocsparse_direction_row, {}, 1);
        check_with_grid_x(block_dim, rocsparse_direction_column, {}, 1);
        check_with_grid_x(block_dim, rocsparse_direction_row, mask, 1);
        check_with_grid_x(block_dim, rocsparse_direction_column, mask, 1);
    }
}

// The five grids in rocsparse_bsrxmv_spzl_general.cpp are bucketed by
// block_dim: <= 2, <= 4, <= 8, <= 16 and > 16, and the first two buckets are
// gated on wavefront_size == 32. block_dim 2/3/4 therefore reach the two small
// general buckets on a wave32 part, which is the only place they are reachable.
//
// These block dims are NOT asserted on wave64, where they dispatch to
// rocsparse_bsrxmv_spzl_2x2/3x3/4x4.cpp instead. Those three files have the
// same unclamped-grid defect and are outside the five files AISPARSE-702
// covers, so asserting here would fail for a defect this change does not fix.
TEST_F(BsrxmvGridClamp, wave32_small_block_dims_reach_the_general_buckets)
{
    if(device_warp_size() != 32)
    {
        GTEST_SUCCEED() << "block_dim 2/3/4 dispatch to the (still unfixed) 2x2/3x3/4x4 "
                           "specialisations on wave64; covered on wave32 parts.";
        return;
    }

    std::vector<rocsparse_int> mask;
    for(rocsparse_int i = 0; i < MB; i += 2)
    {
        mask.push_back(i);
    }

    for(rocsparse_int block_dim : {2, 3, 4})
    {
        check(block_dim, rocsparse_direction_row, {});
        check(block_dim, rocsparse_direction_column, {});
        check(block_dim, rocsparse_direction_row, mask);
    }
}
