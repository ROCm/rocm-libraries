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
// Tests for the rocsparse_Xbsrxmv one-block-per-block-row compute grids.
//
// The bsrxmv kernels launch one block per block row (or per small group of
// block rows). The launchers size grid.x with rocsparse::get_grid_size_x, which
// clamps it to min(maxGridSize[0], (2^32 - 1) / blockDim.x), and the kernels
// grid-stride over the block rows the clamped grid does not reach.
//
// FORCED-LIMIT CASES: these shrink handle->properties.maxGridSize[0] to a
// handful of blocks and run a small matrix (MB block rows) through the public
// entry points. Block rows past the first clamped sweep are reached only by the
// grid-stride loop; drop the loop and the tail of y keeps its input value. A
// launcher that ignores maxGridSize[0] launches the full grid and still passes,
// so these cases cover the stride loops, not the clamp.
//
// REAL-LIMIT CASE: real_threshold_block_dim_32 leaves maxGridSize[0] alone and
// runs 1024-thread blocks with mb just past (2^32 - 1) / 1024, so
// get_grid_size_x clamps and the stride loop covers the last block rows.
// Without the clamp the launch is invalid and y is not written. It needs about
// 0.6 GB of device memory and skips when that is not available.
//
// ARCHITECTURE COVERAGE: rocsparse::bsrxmv_template_dispatch routes *every*
// block_dim through bsrxmvn_general when handle->wavefront_size == 32, so on a
// wave32 part these cases exercise the five grids in
// rocsparse_bsrxmv_spzl_general.cpp. On wave64 parts the same cases dispatch to
// the block-dim specialisations, so block_dim 5/8/16/17/32 additionally cover
// rocsparse_bsrxmv_spzl_5x5.cpp, _8x8.cpp, _16x16.cpp and _17_32.cpp. In those
// two files float takes the half-warp kernels and only the double cases reach
// the templated block-per-row kernels. One test case set, both architectures.
//
#include "unit_test_utils.hpp"

#include "unit_test_grid_clamp.hpp"

#include "rocsparse.h"

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

    // Number of block rows, about a megabyte of values at most. Large enough that
    // every case clamps at CLAMPED_GRID_X, including the masked ones (MB / 2 mask
    // entries) on the wave64 float half-warp kernels, which cover 8 (5x5) and
    // 4 (8x8) block rows per block.
    constexpr rocsparse_int MB = 100;

    // The shrunk grid.x limit. Block rows past the first clamped sweep
    // (CLAMPED_GRID_X blocks times the block rows per block) are reached only by
    // the grid-stride loop.
    constexpr int CLAMPED_GRID_X = 3;

    // y entries of block rows the routine must NOT touch keep this value, and
    // the beta term of the rows it does touch is computed from it. Non-zero so
    // the beta != 0 branch of every kernel is the one exercised.
    constexpr float Y_INIT = 7.0f;

    constexpr float ALPHA = 2.0f;
    constexpr float BETA  = 3.0f;

    // One block per block row, on the diagonal: block row i holds a single
    // block_dim x block_dim block in block column i.
    template <typename T>
    struct bsr_matrix
    {
        rocsparse_int              mb;
        rocsparse_int              nb;
        rocsparse_int              nnzb;
        rocsparse_int              block_dim;
        std::vector<rocsparse_int> row_ptr; // start offset of block row i
        std::vector<rocsparse_int> end_ptr; // end offset of block row i
        std::vector<rocsparse_int> col_ind;
        std::vector<T>             val;
        std::vector<T>             x;
    };

    // Build the diagonal test matrix. Entries vary with (block, row, column) so
    // a kernel that mixes up block rows, or drops some, cannot accidentally
    // produce the reference result. All values are small integers, so both the
    // device and the host reference evaluate them exactly in float and double.
    template <typename T>
    bsr_matrix<T> make_matrix(rocsparse_int block_dim, rocsparse_direction dir)
    {
        bsr_matrix<T> m;
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
                    const T      v    = static_cast<T>(1 + ((i + 2 * r + 3 * c) % 7));
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
            m.x[j] = static_cast<T>(1 + (j % 5));
        }
        return m;
    }

    // y = alpha * A * x + beta * y, restricted to the block rows `mask`
    // selects (all of them when `mask` is empty). Block rows outside the mask
    // are left at Y_INIT, exactly as the routine must leave them.
    template <typename T>
    std::vector<T> reference(const bsr_matrix<T>&              m,
                             rocsparse_direction               dir,
                             const std::vector<rocsparse_int>& mask)
    {
        std::vector<T> y(static_cast<size_t>(m.mb) * m.block_dim, static_cast<T>(Y_INIT));

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
                T acc = static_cast<T>(0);
                for(rocsparse_int j = m.row_ptr[i]; j < m.end_ptr[i]; ++j)
                {
                    const rocsparse_int col  = m.col_ind[j];
                    const size_t        base = static_cast<size_t>(j) * m.block_dim * m.block_dim;
                    for(rocsparse_int c = 0; c < m.block_dim; ++c)
                    {
                        const T a = m.val[base
                                          + (dir == rocsparse_direction_row
                                                 ? static_cast<size_t>(r) * m.block_dim + c
                                                 : static_cast<size_t>(c) * m.block_dim + r)];
                        acc += a * m.x[static_cast<size_t>(col) * m.block_dim + c];
                    }
                }
                const size_t idx = static_cast<size_t>(i) * m.block_dim + r;
                y[idx]
                    = static_cast<T>(ALPHA) * acc + static_cast<T>(BETA) * static_cast<T>(Y_INIT);
            }
        }
        return y;
    }

    // Report the first index at which `got` and `want` differ, else -1. A single
    // precise failure beats thousands of EXPECT_FLOAT_EQ macros in a loop.
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
        return -1;
    }

    rocsparse_status xbsrxmv(rocsparse_handle          handle,
                             rocsparse_direction       dir,
                             rocsparse_operation       trans,
                             rocsparse_int             size_of_mask,
                             rocsparse_int             mb,
                             rocsparse_int             nb,
                             rocsparse_int             nnzb,
                             const float*              alpha,
                             const rocsparse_mat_descr descr,
                             const float*              bsr_val,
                             const rocsparse_int*      bsr_mask_ptr,
                             const rocsparse_int*      bsr_row_ptr,
                             const rocsparse_int*      bsr_end_ptr,
                             const rocsparse_int*      bsr_col_ind,
                             rocsparse_int             block_dim,
                             const float*              x,
                             const float*              beta,
                             float*                    y)
    {
        return rocsparse_sbsrxmv(handle,
                                 dir,
                                 trans,
                                 size_of_mask,
                                 mb,
                                 nb,
                                 nnzb,
                                 alpha,
                                 descr,
                                 bsr_val,
                                 bsr_mask_ptr,
                                 bsr_row_ptr,
                                 bsr_end_ptr,
                                 bsr_col_ind,
                                 block_dim,
                                 x,
                                 beta,
                                 y);
    }

    rocsparse_status xbsrxmv(rocsparse_handle          handle,
                             rocsparse_direction       dir,
                             rocsparse_operation       trans,
                             rocsparse_int             size_of_mask,
                             rocsparse_int             mb,
                             rocsparse_int             nb,
                             rocsparse_int             nnzb,
                             const double*             alpha,
                             const rocsparse_mat_descr descr,
                             const double*             bsr_val,
                             const rocsparse_int*      bsr_mask_ptr,
                             const rocsparse_int*      bsr_row_ptr,
                             const rocsparse_int*      bsr_end_ptr,
                             const rocsparse_int*      bsr_col_ind,
                             rocsparse_int             block_dim,
                             const double*             x,
                             const double*             beta,
                             double*                   y)
    {
        return rocsparse_dbsrxmv(handle,
                                 dir,
                                 trans,
                                 size_of_mask,
                                 mb,
                                 nb,
                                 nnzb,
                                 alpha,
                                 descr,
                                 bsr_val,
                                 bsr_mask_ptr,
                                 bsr_row_ptr,
                                 bsr_end_ptr,
                                 bsr_col_ind,
                                 block_dim,
                                 x,
                                 beta,
                                 y);
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

    // As `check`, but with the grid.x limit and the value type given explicitly.
    template <typename T = float>
    void check_with_grid_x(rocsparse_int                     block_dim,
                           rocsparse_direction               dir,
                           const std::vector<rocsparse_int>& mask,
                           int                               grid_x)
    {
        const bsr_matrix<T> m = make_matrix<T>(block_dim, dir);

        device_vector<rocsparse_int> d_row_ptr{m.row_ptr};
        device_vector<rocsparse_int> d_end_ptr{m.end_ptr};
        device_vector<rocsparse_int> d_col_ind{m.col_ind};
        device_vector<T>             d_val{m.val};
        device_vector<T>             d_x{m.x};
        device_vector<T>             d_y{
            std::vector<T>(static_cast<size_t>(m.mb) * block_dim, static_cast<T>(Y_INIT))};
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

        MatDescr descr;
        const T  alpha = static_cast<T>(ALPHA);
        const T  beta  = static_cast<T>(BETA);

        {
            ScopedMaxGridSizeX clamp(handle, grid_x);

            ASSERT_EQ(xbsrxmv(handle,
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

        const std::vector<T> got  = to_host(d_y.ptr, d_y.n);
        const std::vector<T> want = reference(m, dir, mask);
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

    // As check_with_grid_x, through rocsparse_dbsrmv: no mask, and the kernels
    // read the row end from bsr_row_ptr[row + 1] because bsr_end_ptr is null.
    void check_bsrmv_with_grid_x(rocsparse_int block_dim, rocsparse_direction dir, int grid_x)
    {
        const bsr_matrix<double> m = make_matrix<double>(block_dim, dir);

        std::vector<rocsparse_int> row_ptr(m.row_ptr);
        row_ptr.push_back(m.end_ptr.back());

        device_vector<rocsparse_int> d_row_ptr{row_ptr};
        device_vector<rocsparse_int> d_col_ind{m.col_ind};
        device_vector<double>        d_val{m.val};
        device_vector<double>        d_x{m.x};
        device_vector<double>        d_y{std::vector<double>(static_cast<size_t>(m.mb) * block_dim,
                                                      static_cast<double>(Y_INIT))};
        ASSERT_TRUE(d_row_ptr.ptr && d_col_ind.ptr && d_val.ptr && d_x.ptr && d_y.ptr);

        MatDescr     descr;
        const double alpha = ALPHA;
        const double beta  = BETA;

        {
            ScopedMaxGridSizeX clamp(handle, grid_x);

            ASSERT_EQ(rocsparse_dbsrmv(handle,
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
                                       block_dim,
                                       nullptr,
                                       d_x,
                                       &beta,
                                       d_y),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        }

        const std::vector<double> got  = to_host(d_y.ptr, d_y.n);
        const std::vector<double> want = reference(m, dir, {});
        ASSERT_EQ(got.size(), want.size());

        const int64_t bad = first_mismatch(got, want);
        EXPECT_EQ(bad, -1) << "bsrmv block_dim=" << block_dim << " dir=" << static_cast<int>(dir)
                           << ": first wrong y entry " << bad << " (block row " << (bad / block_dim)
                           << " of " << m.mb << ") got " << got[bad] << " want " << want[bad]
                           << ". grid.x was clamped to " << grid_x;
    }

    // A non-null bsr_mask_ptr with size_of_mask == 0 selects no block rows: the
    // call must succeed and leave y untouched.
    template <typename T>
    void check_empty_mask(rocsparse_int block_dim)
    {
        const bsr_matrix<T> m = make_matrix<T>(block_dim, rocsparse_direction_row);

        device_vector<rocsparse_int> d_row_ptr{m.row_ptr};
        device_vector<rocsparse_int> d_end_ptr{m.end_ptr};
        device_vector<rocsparse_int> d_col_ind{m.col_ind};
        device_vector<T>             d_val{m.val};
        device_vector<T>             d_x{m.x};
        device_vector<T>             d_y{
            std::vector<T>(static_cast<size_t>(m.mb) * block_dim, static_cast<T>(Y_INIT))};
        device_vector<rocsparse_int> d_mask{std::vector<rocsparse_int>{0}};
        ASSERT_TRUE(d_row_ptr.ptr && d_end_ptr.ptr && d_col_ind.ptr && d_val.ptr && d_x.ptr
                    && d_y.ptr && d_mask.ptr);

        MatDescr descr;
        const T  alpha = static_cast<T>(ALPHA);
        const T  beta  = static_cast<T>(BETA);

        EXPECT_EQ(xbsrxmv(handle,
                          rocsparse_direction_row,
                          rocsparse_operation_none,
                          0,
                          m.mb,
                          m.nb,
                          m.nnzb,
                          &alpha,
                          descr.d,
                          d_val,
                          d_mask,
                          d_row_ptr,
                          d_end_ptr,
                          d_col_ind,
                          block_dim,
                          d_x,
                          &beta,
                          d_y),
                  rocsparse_status_success)
            << "block_dim=" << block_dim;
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const std::vector<T> got = to_host(d_y.ptr, d_y.n);
        const std::vector<T> want(got.size(), static_cast<T>(Y_INIT));
        const int64_t        bad = first_mismatch(got, want);
        EXPECT_EQ(bad, -1) << "block_dim=" << block_dim << ": empty mask wrote y entry " << bad;
    }

    // Block dims whose wave64 dispatch target clamps its grid (5x5, 8x8, 16x16,
    // 17_32). Safe to assert on every architecture.
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
    // Every other block row, so MB / 2 mask entries against a grid of 3.
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
// sweep all MB block rows sequentially. On wave64 parts, where these block dims
// reach the block-dim specialisations, that also puts MB consecutive iterations
// through each kernel's shared sdata reduction, so a missing __syncthreads()
// between iterations (one iteration's trailing reads of sdata racing the next
// one's first store) may show up here as a wrong y entry. The race is timing
// dependent, and the 50-thread 5x5 kernel is a single wave on wave64, so it
// cannot race there at all.
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
// These block dims are skipped on wave64, where they dispatch to
// rocsparse_bsrxmv_spzl_2x2/3x3/4x4.cpp instead. Those launchers do not read
// maxGridSize[0], so the cases would pass there without testing anything.
TEST_F(BsrxmvGridClamp, wave32_small_block_dims_reach_the_general_buckets)
{
    if(device_warp_size() != 32)
    {
        GTEST_SKIP() << "block_dim 2/3/4 dispatch to the 2x2/3x3/4x4 kernels on wave64, which "
                        "do not clamp their grids (AISPARSE-937)";
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

// rocsparse_sbsrxmv on wave64 sends block_dim 5 and 8 to the half-warp
// sbsrxmvn_5x5_kernel / sbsrxmvn_8x8_kernel. The templated bsrxmvn_5x5_kernel
// and bsrxmvn_8x8_kernel, whose grid-stride loop fences the shared sdata
// reduction with __syncthreads(), only run for the other value types, so
// repeat the clamped cases in double. On wave32 every block_dim goes through
// bsrxmvn_general, so this also covers the general kernel in double.
TEST_F(BsrxmvGridClamp, double_block_dims_5_and_8_reach_the_templated_kernels)
{
    std::vector<rocsparse_int> mask;
    for(rocsparse_int i = 0; i < MB; i += 2)
    {
        mask.push_back(i);
    }

    for(rocsparse_int block_dim : {5, 8})
    {
        for(int grid_x : {1, CLAMPED_GRID_X})
        {
            check_with_grid_x<double>(block_dim, rocsparse_direction_row, {}, grid_x);
            check_with_grid_x<double>(block_dim, rocsparse_direction_column, {}, grid_x);
            check_with_grid_x<double>(block_dim, rocsparse_direction_row, mask, grid_x);
            check_with_grid_x<double>(block_dim, rocsparse_direction_column, mask, grid_x);
        }
    }
}

// block_dim 32 runs 1024-thread blocks on every architecture
// (bsrxmvn_general_kernel<1024, 32> on wave32, bsrxmvn_17_32_kernel<32> on
// wave64), so get_grid_size_x clamps grid.x to (2^32 - 1) / 1024 blocks. mb is
// 65 block rows past that. Every block row is empty except block rows 0,
// grid_x and mb - 1, which hold one block each, so the alpha term is checked
// on block rows inside and past the clamped grid; every other entry must be
// beta * Y_INIT.
TEST_F(BsrxmvGridClamp, real_threshold_block_dim_32)
{
    constexpr rocsparse_int block_dim = 32;
    constexpr rocsparse_int bs        = block_dim * block_dim;
    const rocsparse_int     grid_x    = static_cast<rocsparse_int>(((uint64_t(1) << 32) - 1) / bs);
    const rocsparse_int     mb        = grid_x + 65;
    const rocsparse_int     nb        = 1;
    constexpr rocsparse_int nnzb      = 3;
    const rocsparse_int     nz_rows[nnzb] = {0, grid_x, mb - 1};

    const size_t y_size = static_cast<size_t>(mb) * block_dim;
    const size_t need
        = 2 * static_cast<size_t>(mb) * sizeof(rocsparse_int) + y_size * sizeof(float);
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    if(hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess || free_bytes < 2 * need)
    {
        GTEST_SKIP() << "Insufficient device memory: need about " << need << " bytes";
    }

    std::vector<rocsparse_int> h_row_ptr(static_cast<size_t>(mb));
    std::vector<rocsparse_int> h_end_ptr(static_cast<size_t>(mb));
    rocsparse_int              acc = 0;
    rocsparse_int              nz  = 0;
    for(rocsparse_int row = 0; row < mb; ++row)
    {
        h_row_ptr[static_cast<size_t>(row)] = acc;
        if(nz < nnzb && row == nz_rows[nz])
        {
            ++acc;
            ++nz;
        }
        h_end_ptr[static_cast<size_t>(row)] = acc;
    }
    ASSERT_EQ(acc, nnzb);

    std::vector<rocsparse_int> h_col_ind(nnzb, 0);
    std::vector<float>         h_val(static_cast<size_t>(nnzb) * bs);
    for(rocsparse_int b = 0; b < nnzb; ++b)
    {
        for(rocsparse_int k = 0; k < bs; ++k)
        {
            h_val[static_cast<size_t>(b) * bs + k] = static_cast<float>(1 + (b + k) % 7);
        }
    }
    std::vector<float> h_x(block_dim);
    for(rocsparse_int c = 0; c < block_dim; ++c)
    {
        h_x[c] = static_cast<float>(1 + c % 5);
    }

    device_vector<rocsparse_int> d_row_ptr{h_row_ptr};
    device_vector<rocsparse_int> d_end_ptr{h_end_ptr};
    device_vector<rocsparse_int> d_col_ind{h_col_ind};
    device_vector<float>         d_val{h_val};
    device_vector<float>         d_x{h_x};
    device_vector<float>         d_y{std::vector<float>(y_size, Y_INIT)};
    if(d_row_ptr.ptr == nullptr || d_end_ptr.ptr == nullptr || d_col_ind.ptr == nullptr
       || d_val.ptr == nullptr || d_x.ptr == nullptr || d_y.ptr == nullptr)
    {
        GTEST_SKIP() << "Insufficient device memory: allocation of about " << need
                     << " bytes failed";
    }

    MatDescr    descr;
    const float alpha = ALPHA;
    const float beta  = BETA;
    ASSERT_EQ(xbsrxmv(handle,
                      rocsparse_direction_row,
                      rocsparse_operation_none,
                      0,
                      mb,
                      nb,
                      nnzb,
                      &alpha,
                      descr.d,
                      d_val,
                      nullptr,
                      d_row_ptr,
                      d_end_ptr,
                      d_col_ind,
                      block_dim,
                      d_x,
                      &beta,
                      d_y),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    const std::vector<float> got = to_host(d_y.ptr, d_y.n);
    ASSERT_EQ(got.size(), y_size);

    rocsparse_int next_nz = 0;
    for(rocsparse_int row = 0; row < mb; ++row)
    {
        const bool nonzero = next_nz < nnzb && row == nz_rows[next_nz];
        for(rocsparse_int r = 0; r < block_dim; ++r)
        {
            float want = BETA * Y_INIT;
            if(nonzero)
            {
                float sum = 0.0f;
                for(rocsparse_int c = 0; c < block_dim; ++c)
                {
                    sum += h_val[static_cast<size_t>(next_nz) * bs + r * block_dim + c] * h_x[c];
                }
                want = ALPHA * sum + BETA * Y_INIT;
            }
            const size_t idx = static_cast<size_t>(row) * block_dim + r;
            if(got[idx] != want)
            {
                ADD_FAILURE() << "y[" << idx << "] (block row " << row << " of " << mb
                              << ", grid.x clamps at " << grid_x << ") is " << got[idx] << ", want "
                              << want;
                return;
            }
        }
        if(nonzero)
        {
            ++next_nz;
        }
    }
}

// rocsparse_Xbsrmv reaches the same kernels with a null bsr_end_ptr. Double
// reaches the templated 5x5 and 8x8 kernels on wave64.
TEST_F(BsrxmvGridClamp, bsrmv_null_end_ptr)
{
    for(rocsparse_int block_dim : block_dims_in_scope())
    {
        for(int grid_x : {1, CLAMPED_GRID_X})
        {
            check_bsrmv_with_grid_x(block_dim, rocsparse_direction_row, grid_x);
            check_bsrmv_with_grid_x(block_dim, rocsparse_direction_column, grid_x);
        }
    }
}

// Every launcher that sizes its grid as one block per (masked) block row,
// including the general one (block_dim 7 and 40), with an empty mask. Double
// reaches the templated 5x5 and 8x8 kernels on wave64.
TEST_F(BsrxmvGridClamp, empty_mask_leaves_y_untouched)
{
    for(rocsparse_int block_dim : {5, 7, 8, 16, 17, 32, 40})
    {
        check_empty_mask<double>(block_dim);
    }
}
