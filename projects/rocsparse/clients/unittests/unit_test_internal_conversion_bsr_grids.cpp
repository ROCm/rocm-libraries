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
// Forced-clamp tests for the csr2bsr_nnz, csr2bsr and bsr2csr block-row grids
// (AISPARSE-703).
//
// FOCUS. Every block_dim > 1 launch of the three conversions sized grid.x from
// the block-row count mb (or mb / rows-per-block for the wavefront kernels).
// The fix clamps each with rocsparse::get_grid_size_x and grid-strides the
// kernels over mb; bsr2csr additionally keeps a straight-line GRID_STRIDE =
// false instantiation that it selects whenever the grid is not clamped. The
// csr2bsr_nnz block-per-row kernel keeps its row count in shared memory, and a
// looping block must reset it from thread 0 only, or another wave can zero it
// before thread 0 stores the previous row's count.
//
// WHY NOT TEST THE REAL THRESHOLD. The clamp binds at (2^32 - 1) / blockDim.x
// blocks, about 16.7M block rows at 256 threads and 4.2M at 1024. These tests
// shrink handle->properties.maxGridSize[0] with ScopedMaxGridSizeX to 1, 3 and 7
// blocks instead, so a few hundred block rows take the clamped, looping path.
// This is the AISPARSE-702 idiom.
//
// WHAT MAKES EACH CASE LOAD-BEARING. block_dim <= 16 uses mb = 150: the
// wavefront kernels cover 16 (block_dim <= 4) or 256 / wavefront size block
// rows per block, so their unclamped grid is at least 10 blocks. block_dim > 16
// uses mb = 20 with one block per block row. Every grid under test is therefore
// wider than the largest limit, 7. Rows have varied lengths with a dense row
// every 29 rows, so block rows have many block columns and the multipass
// kernels take several chunks. m and n are one short of a multiple of
// block_dim, so the last block row and column are partial.
//
// block_dim = 1 is a control: those launches are not clamped by the fix.
//
// Each case compares nnzb, the row pointer, the column indices and the values
// exactly, against a host conversion and against the unclamped run.
//
// TARGET: rocsparse-unit-test-device. The tests drive the public
// rocsparse_sparse_to_sparse entry point, which reaches csr2bsr_nnz in its
// analysis stage and csr2bsr or bsr2csr in its compute stage, for 32- and 64-bit
// indices.
//
#include "unit_test_utils.hpp"

// ScopedMaxGridSizeX: shrinks handle->properties.maxGridSize[0], the limit
// get_grid_size_x clamps grid.x against.
#include "unit_test_grid_clamp.hpp"

#include "rocsparse.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    // grid.x limits the tests shrink maxGridSize[0] to. 0 means the device limit.
    constexpr int clamped_limits[] = {1, 3, 7};

    // Host sparse matrix. For BSR, m, n, ptr and ind are at block level and val
    // holds block_dim * block_dim entries per block in the block direction.
    struct HostSparse
    {
        int64_t              m         = 0;
        int64_t              n         = 0;
        int64_t              block_dim = 1;
        std::vector<int64_t> ptr;
        std::vector<int64_t> ind;
        std::vector<double>  val;

        int64_t nnz() const
        {
            return ptr.empty() ? 0 : ptr.back();
        }
    };

    int64_t block_entry(rocsparse_direction dir, int64_t block_dim, int64_t r, int64_t c)
    {
        return (dir == rocsparse_direction_row) ? r * block_dim + c : c * block_dim + r;
    }

    // mb block rows of block_dim, less one row, so the last block row is partial;
    // likewise for the columns.
    HostSparse make_csr(int64_t block_dim)
    {
        const int64_t blocks = (block_dim <= 16) ? 150 : 20;
        const int64_t shrink = (block_dim > 1) ? 1 : 0;

        std::mt19937 rng(703);

        HostSparse a;
        a.m = blocks * block_dim - shrink;
        a.n = blocks * block_dim - shrink;
        a.ptr.push_back(0);

        std::uniform_int_distribution<int64_t> short_len(0, 5);
        std::uniform_int_distribution<int64_t> col(0, a.n - 1);
        std::uniform_int_distribution<int>     value(1, 9);
        std::vector<char>                      used(a.n, 0);

        for(int64_t i = 0; i < a.m; ++i)
        {
            const int64_t len = (i % 29 == 0) ? a.n / 3 : short_len(rng);

            std::vector<int64_t> cols;
            while(static_cast<int64_t>(cols.size()) < len)
            {
                const int64_t c = col(rng);
                if(!used[c])
                {
                    used[c] = 1;
                    cols.push_back(c);
                }
            }
            std::sort(cols.begin(), cols.end());
            for(const int64_t c : cols)
            {
                used[c] = 0;
                a.ind.push_back(c);
                a.val.push_back(static_cast<double>((i + c) % 2 == 0 ? value(rng) : -value(rng)));
            }
            a.ptr.push_back(static_cast<int64_t>(a.ind.size()));
        }
        return a;
    }

    HostSparse host_csr2bsr(const HostSparse& a, int64_t block_dim, rocsparse_direction dir)
    {
        const int64_t bs = block_dim * block_dim;

        HostSparse b;
        b.m         = (a.m + block_dim - 1) / block_dim;
        b.n         = (a.n + block_dim - 1) / block_dim;
        b.block_dim = block_dim;
        b.ptr.push_back(0);

        std::vector<int64_t> slot(b.n, -1);
        for(int64_t bi = 0; bi < b.m; ++bi)
        {
            const int64_t row_begin = bi * block_dim;
            const int64_t row_end   = std::min(a.m, row_begin + block_dim);

            std::vector<int64_t> bcols;
            for(int64_t i = row_begin; i < row_end; ++i)
            {
                for(int64_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                {
                    bcols.push_back(a.ind[p] / block_dim);
                }
            }
            std::sort(bcols.begin(), bcols.end());
            bcols.erase(std::unique(bcols.begin(), bcols.end()), bcols.end());

            const int64_t first = static_cast<int64_t>(b.ind.size());
            for(size_t k = 0; k < bcols.size(); ++k)
            {
                slot[bcols[k]] = first + static_cast<int64_t>(k);
                b.ind.push_back(bcols[k]);
            }
            b.val.resize(b.ind.size() * bs, 0.0);

            for(int64_t i = row_begin; i < row_end; ++i)
            {
                for(int64_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                {
                    const int64_t bj = a.ind[p] / block_dim;
                    b.val[slot[bj] * bs
                          + block_entry(dir, block_dim, i - row_begin, a.ind[p] - bj * block_dim)]
                        = a.val[p];
                }
            }
            for(const int64_t bj : bcols)
            {
                slot[bj] = -1;
            }
            b.ptr.push_back(static_cast<int64_t>(b.ind.size()));
        }
        return b;
    }

    // Every block expands to block_dim full rows of block_dim entries, zeros
    // included.
    HostSparse host_bsr2csr(const HostSparse& b, rocsparse_direction dir)
    {
        const int64_t bd = b.block_dim;
        const int64_t bs = bd * bd;

        HostSparse a;
        a.m = b.m * bd;
        a.n = b.n * bd;
        a.ptr.push_back(0);
        for(int64_t bi = 0; bi < b.m; ++bi)
        {
            for(int64_t r = 0; r < bd; ++r)
            {
                for(int64_t p = b.ptr[bi]; p < b.ptr[bi + 1]; ++p)
                {
                    for(int64_t c = 0; c < bd; ++c)
                    {
                        a.ind.push_back(b.ind[p] * bd + c);
                        a.val.push_back(b.val[p * bs + block_entry(dir, bd, r, c)]);
                    }
                }
                a.ptr.push_back(static_cast<int64_t>(a.ind.size()));
            }
        }
        return a;
    }

    // Empty if equal, otherwise where the first difference is.
    std::string first_difference(const HostSparse& got, const HostSparse& want)
    {
        std::ostringstream msg;
        if(got.nnz() != want.nnz())
        {
            msg << "nnz is " << got.nnz() << ", want " << want.nnz() << "; ";
        }
        if(got.ptr != want.ptr)
        {
            size_t i = 0;
            while(i < got.ptr.size() && i < want.ptr.size() && got.ptr[i] == want.ptr[i])
            {
                ++i;
            }
            msg << "row pointer differs at " << i;
        }
        else if(got.ind != want.ind)
        {
            size_t i = 0;
            while(got.ind[i] == want.ind[i])
            {
                ++i;
            }
            msg << "column index " << i << " is " << got.ind[i] << ", want " << want.ind[i];
        }
        else if(got.val != want.val)
        {
            size_t i = 0;
            while(got.val[i] == want.val[i])
            {
                ++i;
            }
            msg << "value " << i << " is " << got.val[i] << ", want " << want.val[i];
        }
        return msg.str();
    }

    // Owns the conversion descriptors so each one created is destroyed on every
    // return path.
    struct ConversionDescrs
    {
        rocsparse_spmat_descr            source = nullptr;
        rocsparse_spmat_descr            target = nullptr;
        rocsparse_sparse_to_sparse_descr descr  = nullptr;

        ConversionDescrs() = default;

        ConversionDescrs(const ConversionDescrs&) = delete;

        ConversionDescrs& operator=(const ConversionDescrs&) = delete;

        ~ConversionDescrs()
        {
            if(descr != nullptr)
            {
                (void)rocsparse_destroy_sparse_to_sparse_descr(descr);
            }
            for(rocsparse_spmat_descr m : {source, target})
            {
                if(m != nullptr)
                {
                    (void)rocsparse_destroy_spmat_descr(m);
                }
            }
        }
    };

    template <typename I, typename J, typename T>
    rocsparse_status create_descr(rocsparse_spmat_descr* descr,
                                  const HostSparse&      h,
                                  int64_t                nnz,
                                  void*                  ptr,
                                  void*                  ind,
                                  void*                  val,
                                  bool                   bsr,
                                  rocsparse_direction    dir)
    {
        if(bsr)
        {
            return rocsparse_create_bsr_descr(descr,
                                              h.m,
                                              h.n,
                                              nnz,
                                              dir,
                                              h.block_dim,
                                              ptr,
                                              ind,
                                              val,
                                              it_of<I>(),
                                              it_of<J>(),
                                              rocsparse_index_base_zero,
                                              dt_of<T>());
        }
        return rocsparse_create_csr_descr(descr,
                                          h.m,
                                          h.n,
                                          nnz,
                                          ptr,
                                          ind,
                                          val,
                                          it_of<I>(),
                                          it_of<J>(),
                                          rocsparse_index_base_zero,
                                          dt_of<T>());
    }

    // Convert `source` (CSR when to_bsr, otherwise BSR) to the other format
    // through rocsparse_sparse_to_sparse with grid.x limited to `limit` blocks (0
    // keeps the device limit) for both stages. `shape` gives the target's m, n and
    // block_dim. Records a gtest failure and returns an empty matrix on error.
    template <typename I, typename J, typename T>
    HostSparse device_convert(rocsparse_handle    handle,
                              const HostSparse&   source,
                              HostSparse          shape,
                              bool                to_bsr,
                              rocsparse_direction dir,
                              int                 limit)
    {
        device_vector<I> s_ptr(std::vector<I>(source.ptr.begin(), source.ptr.end()));
        device_vector<J> s_ind(std::vector<J>(source.ind.begin(), source.ind.end()));
        device_vector<T> s_val(std::vector<T>(source.val.begin(), source.val.end()));
        device_vector<I> t_ptr(static_cast<size_t>(shape.m + 1));
        if(s_ptr.ptr == nullptr || s_ind.ptr == nullptr || s_val.ptr == nullptr
           || t_ptr.ptr == nullptr)
        {
            ADD_FAILURE() << "device allocation failed";
            return {};
        }

        ConversionDescrs descrs;
        if(create_descr<I, J, T>(
               &descrs.source, source, source.nnz(), s_ptr.ptr, s_ind.ptr, s_val.ptr, !to_bsr, dir)
               != rocsparse_status_success
           || create_descr<I, J, T>(
                  &descrs.target, shape, 0, t_ptr.ptr, nullptr, nullptr, to_bsr, dir)
                  != rocsparse_status_success
           || rocsparse_create_sparse_to_sparse_descr(&descrs.descr,
                                                      descrs.source,
                                                      descrs.target,
                                                      rocsparse_sparse_to_sparse_alg_default)
                  != rocsparse_status_success)
        {
            ADD_FAILURE() << "descriptor creation failed";
            return {};
        }

        std::unique_ptr<ScopedMaxGridSizeX> clamp;
        if(limit > 0)
        {
            clamp.reset(new ScopedMaxGridSizeX(handle, limit));
        }

        auto run_stage = [&](rocsparse_sparse_to_sparse_stage stage) {
            size_t           buffer_size = 0;
            rocsparse_status status      = rocsparse_sparse_to_sparse_buffer_size(
                handle, descrs.descr, descrs.source, descrs.target, stage, &buffer_size);
            void* buffer = nullptr;
            if(status == rocsparse_status_success
               && hipMalloc(&buffer, std::max<size_t>(buffer_size, 1)) != hipSuccess)
            {
                return rocsparse_status_memory_error;
            }
            if(status == rocsparse_status_success)
            {
                status = rocsparse_sparse_to_sparse(
                    handle, descrs.descr, descrs.source, descrs.target, stage, buffer_size, buffer);
            }
            if(status == rocsparse_status_success && hipDeviceSynchronize() != hipSuccess)
            {
                status = rocsparse_status_internal_error;
            }
            if(buffer != nullptr)
            {
                (void)hipFree(buffer);
            }
            return status;
        };

        std::unique_ptr<device_vector<J>> t_ind;
        std::unique_ptr<device_vector<T>> t_val;
        const int64_t                     bs = shape.block_dim * shape.block_dim;

        int64_t          rows   = 0;
        int64_t          cols   = 0;
        int64_t          nnz    = 0;
        rocsparse_status status = run_stage(rocsparse_sparse_to_sparse_stage_analysis);
        if(status == rocsparse_status_success)
        {
            status = rocsparse_spmat_get_size(descrs.target, &rows, &cols, &nnz);
        }
        if(status == rocsparse_status_success)
        {
            t_ind.reset(new device_vector<J>(static_cast<size_t>(std::max<int64_t>(nnz, 1))));
            t_val.reset(new device_vector<T>(static_cast<size_t>(std::max<int64_t>(nnz * bs, 1))));
            if(t_ind->ptr == nullptr || t_val->ptr == nullptr)
            {
                status = rocsparse_status_memory_error;
            }
            else if(to_bsr)
            {
                status
                    = rocsparse_bsr_set_pointers(descrs.target, t_ptr.ptr, t_ind->ptr, t_val->ptr);
            }
            else
            {
                status
                    = rocsparse_csr_set_pointers(descrs.target, t_ptr.ptr, t_ind->ptr, t_val->ptr);
            }
        }
        if(status == rocsparse_status_success)
        {
            status = run_stage(rocsparse_sparse_to_sparse_stage_compute);
        }
        if(status != rocsparse_status_success)
        {
            ADD_FAILURE() << "rocsparse_sparse_to_sparse failed with status " << status
                          << " at limit " << limit;
            return {};
        }

        const std::vector<I> h_ptr = to_host(t_ptr);
        const std::vector<J> h_ind = to_host<J>(t_ind->ptr, static_cast<size_t>(nnz));
        const std::vector<T> h_val = to_host<T>(t_val->ptr, static_cast<size_t>(nnz * bs));

        shape.ptr.assign(h_ptr.begin(), h_ptr.end());
        shape.ind.assign(h_ind.begin(), h_ind.end());
        shape.val.assign(h_val.begin(), h_val.end());
        if(shape.nnz() != nnz)
        {
            ADD_FAILURE() << "analysis reported nnz " << nnz << " but the row pointer ends at "
                          << shape.nnz();
        }
        return shape;
    }

    // Unclamped control against the host conversion, then every clamped limit
    // against both the host conversion and the unclamped result.
    template <typename I, typename J, typename T>
    void check_all_limits(rocsparse_handle    handle,
                          const HostSparse&   source,
                          const HostSparse&   want,
                          bool                to_bsr,
                          rocsparse_direction dir)
    {
        HostSparse shape;
        shape.m         = want.m;
        shape.n         = want.n;
        shape.block_dim = want.block_dim;

        const HostSparse unclamped = device_convert<I, J, T>(handle, source, shape, to_bsr, dir, 0);
        const std::string control  = first_difference(unclamped, want);
        ASSERT_TRUE(control.empty()) << "unclamped control: " << control;

        for(const int limit : clamped_limits)
        {
            const HostSparse got
                = device_convert<I, J, T>(handle, source, shape, to_bsr, dir, limit);
            const std::string vs_host = first_difference(got, want);
            EXPECT_TRUE(vs_host.empty())
                << "grid.x clamped to " << limit << " blocks, against the host: " << vs_host;
            const std::string vs_unclamped = first_difference(got, unclamped);
            EXPECT_TRUE(vs_unclamped.empty())
                << "grid.x clamped to " << limit << " blocks, against unclamped: " << vs_unclamped;
        }
    }

    template <typename I, typename J, typename T>
    void check_csr2bsr(rocsparse_handle handle, int64_t block_dim, rocsparse_direction dir)
    {
        const HostSparse csr  = make_csr(block_dim);
        const HostSparse want = host_csr2bsr(csr, block_dim, dir);
        check_all_limits<I, J, T>(handle, csr, want, true, dir);
    }

    template <typename I, typename J, typename T>
    void check_bsr2csr(rocsparse_handle handle, int64_t block_dim, rocsparse_direction dir)
    {
        const HostSparse bsr  = host_csr2bsr(make_csr(block_dim), block_dim, dir);
        const HostSparse want = host_bsr2csr(bsr, dir);
        check_all_limits<I, J, T>(handle, bsr, want, false, dir);
    }

    // Both directions, with 32-bit indices and float, 64-bit indices and double,
    // and 64-bit row pointers with 32-bit column indices and double.
    template <void (*Check32)(rocsparse_handle, int64_t, rocsparse_direction),
              void (*Check64)(rocsparse_handle, int64_t, rocsparse_direction),
              void (*CheckMixed)(rocsparse_handle, int64_t, rocsparse_direction)>
    void check_all_variants(rocsparse_handle handle, int64_t block_dim)
    {
        for(const rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
        {
            SCOPED_TRACE(dir == rocsparse_direction_row ? "row direction" : "column direction");
            {
                SCOPED_TRACE("float, int32 indices");
                Check32(handle, block_dim, dir);
            }
            {
                SCOPED_TRACE("double, int64 indices");
                Check64(handle, block_dim, dir);
            }
            {
                SCOPED_TRACE("double, int64 row pointer, int32 column indices");
                CheckMixed(handle, block_dim, dir);
            }
        }
    }

    void check_csr2bsr_all(rocsparse_handle handle, int64_t block_dim)
    {
        check_all_variants<check_csr2bsr<int32_t, int32_t, float>,
                           check_csr2bsr<int64_t, int64_t, double>,
                           check_csr2bsr<int64_t, int32_t, double>>(handle, block_dim);
    }

    void check_bsr2csr_all(rocsparse_handle handle, int64_t block_dim)
    {
        check_all_variants<check_bsr2csr<int32_t, int32_t, float>,
                           check_bsr2csr<int64_t, int64_t, double>,
                           check_bsr2csr<int64_t, int32_t, double>>(handle, block_dim);
    }

    using BsrConversionGrids = HandleTest;
}

// ---------------------------------------------------------------------------
// csr2bsr: the analysis stage runs csr2bsr_nnz, the compute stage csr2bsr.
// ---------------------------------------------------------------------------

// Control: block_dim 1 copies the CSR arrays, and its launches are not clamped.
TEST_F(BsrConversionGrids, csr2bsr_block_dim_1)
{
    check_csr2bsr_all(handle, 1);
}

// Wavefront-per-row multipass, 16 lanes, 16 block rows per block.
TEST_F(BsrConversionGrids, csr2bsr_block_dim_2)
{
    check_csr2bsr_all(handle, 2);
}

TEST_F(BsrConversionGrids, csr2bsr_block_dim_4)
{
    check_csr2bsr_all(handle, 4);
}

// Wavefront-per-row multipass, one wavefront per block row.
TEST_F(BsrConversionGrids, csr2bsr_block_dim_8)
{
    check_csr2bsr_all(handle, 8);
}

TEST_F(BsrConversionGrids, csr2bsr_block_dim_16)
{
    check_csr2bsr_all(handle, 16);
}

// Block-per-row multipass. csr2bsr_nnz keeps the block row's count in shared
// memory, which a looping block must reset from thread 0 only.
TEST_F(BsrConversionGrids, csr2bsr_block_dim_32)
{
    check_csr2bsr_all(handle, 32);
}

TEST_F(BsrConversionGrids, csr2bsr_block_dim_64)
{
    check_csr2bsr_all(handle, 64);
}

// csr2bsr_65_inf_kernel and csr2bsr_nnz_65_inf_kernel, whose global scratch is
// partitioned by physical block rather than by block row.
TEST_F(BsrConversionGrids, csr2bsr_block_dim_80)
{
    check_csr2bsr_all(handle, 80);
}

// ---------------------------------------------------------------------------
// bsr2csr. Each block_dim from 2 to 7 has its own instantiation of the 2-7
// kernel; 8, 16 and 32 take the 8-32 kernel and 64 and 128 the 33-256 kernel.
// ---------------------------------------------------------------------------

// Control: block_dim 1 copies the BSR arrays, and its launch is not clamped.
TEST_F(BsrConversionGrids, bsr2csr_block_dim_1)
{
    check_bsr2csr_all(handle, 1);
}

TEST_F(BsrConversionGrids, bsr2csr_block_dim_2_to_7)
{
    for(int64_t block_dim = 2; block_dim <= 7; ++block_dim)
    {
        SCOPED_TRACE("block_dim " + std::to_string(block_dim));
        check_bsr2csr_all(handle, block_dim);
    }
}

TEST_F(BsrConversionGrids, bsr2csr_block_dim_8)
{
    check_bsr2csr_all(handle, 8);
}

TEST_F(BsrConversionGrids, bsr2csr_block_dim_16)
{
    check_bsr2csr_all(handle, 16);
}

TEST_F(BsrConversionGrids, bsr2csr_block_dim_32)
{
    check_bsr2csr_all(handle, 32);
}

TEST_F(BsrConversionGrids, bsr2csr_block_dim_64)
{
    check_bsr2csr_all(handle, 64);
}

TEST_F(BsrConversionGrids, bsr2csr_block_dim_128)
{
    check_bsr2csr_all(handle, 128);
}
