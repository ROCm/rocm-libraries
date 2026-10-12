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
// Forced-clamp tests for the csr2bsr_nnz, csr2bsr and bsr2csr block-row grids.
//
// Every block_dim > 1 launch clamps grid.x with rocsparse::get_grid_size_x and
// grid-strides over the block rows. bsr2csr has one instantiation per kernel,
// the same shape as csr2bsr. The csr2bsr_nnz block-per-row kernel keeps its row
// count in shared memory, and a looping block must reset it from thread 0 only,
// or another wave can zero it before thread 0 stores the previous row's count.
//
// The cases below shrink handle->properties.maxGridSize[0] with
// ScopedMaxGridSizeX to 1, 3 and 7 blocks, so a few hundred block rows take the
// clamped path. The bsr2csr_real_threshold tests do not: rocsparse_sbsr2csr with
// block_dim 8 and 33 (1024 threads, mb = 2^22 + 1) and block_dim 2 (256
// threads, mb = 2^24 + 1) cross (2^32 - 1) / blockDim.x and have to clamp inside
// get_grid_size_x. Device targets are filled with 0xFF before launch, in both
// directions, so a kernel that skips block rows cannot pass on leftover bytes.
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
#include <limits>
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
        else if(got.ind.size() != want.ind.size())
        {
            msg << "column index count is " << got.ind.size() << ", want " << want.ind.size();
        }
        else if(got.ind != want.ind)
        {
            size_t i = 0;
            while(i < got.ind.size() && i < want.ind.size() && got.ind[i] == want.ind[i])
            {
                ++i;
            }
            msg << "column index differs at " << i;
            if(i < got.ind.size() && i < want.ind.size())
            {
                msg << ": " << got.ind[i] << ", want " << want.ind[i];
            }
        }
        else if(got.val.size() != want.val.size())
        {
            msg << "value count is " << got.val.size() << ", want " << want.val.size();
        }
        else if(got.val != want.val)
        {
            size_t i = 0;
            while(i < got.val.size() && i < want.val.size() && got.val[i] == want.val[i])
            {
                ++i;
            }
            msg << "value differs at " << i;
            if(i < got.val.size() && i < want.val.size())
            {
                msg << ": " << got.val[i] << ", want " << want.val[i];
            }
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
        // Poison the targets. The allocator hands the same blocks, still holding
        // a correct result, to the next run, so a kernel that skips block rows
        // would otherwise pass on leftover bytes.
        if(hipMemset(t_ptr.ptr, 0xFF, t_ptr.n * sizeof(I)) != hipSuccess)
        {
            ADD_FAILURE() << "hipMemset of the target row pointer failed";
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
            else if(hipMemset(t_ind->ptr, 0xFF, t_ind->n * sizeof(J)) != hipSuccess
                    || hipMemset(t_val->ptr, 0xFF, t_val->n * sizeof(T)) != hipSuccess)
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

    // bsr2csr launched with blockDim.x threads crosses (2^32 - 1) / blockDim.x
    // block rows at mb = (2^32 - 1) / blockDim.x + 2. Nonzero blocks sit on block
    // rows 0, grid_x and grid_x + 1. The last two are past the clamped grid.x, so
    // they are right only when the kernel grid-strides.
    void check_bsr2csr_real_threshold(rocsparse_handle handle,
                                      rocsparse_int    block_dim,
                                      rocsparse_int    threads)
    {
        const rocsparse_int grid_x
            = static_cast<rocsparse_int>(((uint64_t(1) << 32) - 1) / threads);
        const rocsparse_int     mb                  = grid_x + 2;
        const rocsparse_int     nb                  = 1;
        const rocsparse_int     bs                  = block_dim * block_dim;
        constexpr rocsparse_int nnz_blocks          = 3;
        const rocsparse_int     nz_rows[nnz_blocks] = {0, grid_x, grid_x + 1};

        // Device row pointers: BSR (mb + 1) and CSR (mb * block_dim + 1).
        const size_t need = (static_cast<size_t>(mb) * (1 + block_dim) + 2) * sizeof(rocsparse_int);
        size_t       free_bytes  = 0;
        size_t       total_bytes = 0;
        if(hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess || free_bytes < 2 * need)
        {
            GTEST_SKIP() << "Insufficient device memory: need about " << need << " bytes";
        }

        std::vector<rocsparse_int> h_bsr_ptr(static_cast<size_t>(mb) + 1, 0);
        rocsparse_int              acc = 0;
        size_t                     nz  = 0;
        for(rocsparse_int row = 0; row < mb; ++row)
        {
            h_bsr_ptr[static_cast<size_t>(row)] = acc;
            if(nz < nnz_blocks && row == nz_rows[nz])
            {
                ++acc;
                ++nz;
            }
        }
        h_bsr_ptr[static_cast<size_t>(mb)] = acc;
        ASSERT_EQ(acc, nnz_blocks);

        std::vector<rocsparse_int> h_bsr_ind(nnz_blocks, 0);
        std::vector<float>         h_bsr_val(static_cast<size_t>(nnz_blocks) * bs);
        for(rocsparse_int b = 0; b < nnz_blocks; ++b)
        {
            for(rocsparse_int k = 0; k < bs; ++k)
            {
                h_bsr_val[static_cast<size_t>(b) * bs + k] = static_cast<float>(1000 * b + k + 1);
            }
        }

        const rocsparse_int        m   = mb * block_dim;
        const rocsparse_int        nnz = nnz_blocks * bs;
        std::vector<rocsparse_int> want_ptr(static_cast<size_t>(m) + 1, 0);
        rocsparse_int              running = 0;
        for(rocsparse_int row = 0; row < mb; ++row)
        {
            const rocsparse_int blocks
                = h_bsr_ptr[static_cast<size_t>(row) + 1] - h_bsr_ptr[static_cast<size_t>(row)];
            for(rocsparse_int r = 0; r < block_dim; ++r)
            {
                running += blocks * block_dim;
                want_ptr[static_cast<size_t>(row) * block_dim + static_cast<size_t>(r) + 1]
                    = running;
            }
        }
        ASSERT_EQ(running, nnz);

        device_vector<rocsparse_int> bsr_ptr(h_bsr_ptr);
        device_vector<rocsparse_int> bsr_ind(h_bsr_ind);
        device_vector<float>         bsr_val(h_bsr_val);
        device_vector<rocsparse_int> csr_ptr(static_cast<size_t>(m) + 1);
        device_vector<rocsparse_int> csr_ind(static_cast<size_t>(nnz));
        device_vector<float>         csr_val(static_cast<size_t>(nnz));
        if(bsr_ptr.ptr == nullptr || bsr_ind.ptr == nullptr || bsr_val.ptr == nullptr
           || csr_ptr.ptr == nullptr || csr_ind.ptr == nullptr || csr_val.ptr == nullptr)
        {
            ADD_FAILURE() << "device allocation failed";
            return;
        }
        ASSERT_EQ(hipMemset(csr_ptr.ptr, 0xFF, csr_ptr.n * sizeof(rocsparse_int)), hipSuccess);
        ASSERT_EQ(hipMemset(csr_ind.ptr, 0xFF, csr_ind.n * sizeof(rocsparse_int)), hipSuccess);
        ASSERT_EQ(hipMemset(csr_val.ptr, 0xFF, csr_val.n * sizeof(float)), hipSuccess);

        struct MatDescrs
        {
            rocsparse_mat_descr bsr = nullptr;
            rocsparse_mat_descr csr = nullptr;

            ~MatDescrs()
            {
                if(bsr != nullptr)
                {
                    (void)rocsparse_destroy_mat_descr(bsr);
                }
                if(csr != nullptr)
                {
                    (void)rocsparse_destroy_mat_descr(csr);
                }
            }
        } descrs;

        ASSERT_EQ(rocsparse_create_mat_descr(&descrs.bsr), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_descr(&descrs.csr), rocsparse_status_success);

        ASSERT_EQ(rocsparse_sbsr2csr(handle,
                                     rocsparse_direction_row,
                                     mb,
                                     nb,
                                     descrs.bsr,
                                     bsr_val,
                                     bsr_ptr,
                                     bsr_ind,
                                     block_dim,
                                     descrs.csr,
                                     csr_val,
                                     csr_ptr,
                                     csr_ind),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const std::vector<rocsparse_int> got_ptr = to_host(csr_ptr);
        const std::vector<rocsparse_int> got_ind = to_host(csr_ind);
        const std::vector<float>         got_val = to_host(csr_val);

        ASSERT_EQ(got_ptr.size(), want_ptr.size());
        for(size_t i = 0; i < want_ptr.size(); ++i)
        {
            if(got_ptr[i] != want_ptr[i])
            {
                ADD_FAILURE() << "csr_row_ptr[" << i << "] is " << got_ptr[i] << ", want "
                              << want_ptr[i];
                return;
            }
        }
        ASSERT_EQ(got_ind.size(), static_cast<size_t>(nnz));
        ASSERT_EQ(got_val.size(), static_cast<size_t>(nnz));
        for(rocsparse_int k = 0; k < nnz; ++k)
        {
            if(got_ind[static_cast<size_t>(k)] != (k % block_dim)
               || got_val[static_cast<size_t>(k)] != h_bsr_val[static_cast<size_t>(k)])
            {
                ADD_FAILURE() << "csr entry " << k << " ind " << got_ind[static_cast<size_t>(k)]
                              << " val " << got_val[static_cast<size_t>(k)];
                return;
            }
        }
    }

    // csr2bsr_65_inf_kernel and csr2bsr_nnz_65_inf_kernel give each launched block
    // a scratch slice of 2 * rows_per_segment * 32 row bounds, starting at the
    // physical block index times the slice size. In 32-bit unsigned arithmetic
    // that product wraps at 2^32 elements. For block_dim 65 (rows_per_segment 3,
    // 192 elements per slice) block 22369622 is the first to wrap, and a wrapped
    // block lands across the slices of blocks phys - 22369622 and
    // phys - 22369621. A block uses its slice only while it runs, so two blocks
    // conflict only while both are resident.
    //
    // mb is the smallest that wraps plus `wrapped` block rows, and m = 65 * mb
    // still fits int32. The scratch of that launch is 17 GB for csr2bsr_nnz and
    // 34 GB for csr2bsr, so the test skips unless the device has twice the total
    // free. Block rows 0 to `wrapped` are the ones the wrapped blocks land on.
    // Each has two long rows and keeps its block running while the grid
    // dispatches the 22 million empty blocks after it, so a wrapped block starts
    // while the block it lands on is still working and overwrites its row bounds.
    // Every block row has its own range of columns, so a block that reads another
    // one's bounds returns visibly wrong blocks. The wrapped block rows hold short
    // rows, so their own output is checked as well. Every other block row is
    // empty, which keeps the CSR row pointer the only large input.
    //
    // The long rows have to outlast the dispatch of the empty blocks, which is
    // slow when memory is first touched in the kernel (about 20 s on an MI300A).
    // If they end first the old 32-bit offsets pass; the 64-bit ones always do.
    //
    // The expected block row pointer follows from the planted pattern, and the
    // block columns and values of csr2bsr come from host_csr2bsr on the planted
    // block rows alone.
    void check_65_inf_segment_base(rocsparse_handle handle, bool with_values)
    {
        constexpr int64_t block_dim        = 65;
        constexpr int64_t rows_per_segment = 3;
        constexpr int64_t block_size       = 32;
        constexpr int64_t wrapped          = 2;
        constexpr int64_t short_cols       = 200;
        const int64_t     slice            = 2 * rows_per_segment * block_size;
        const int64_t     first_wrapped    = ((int64_t(1) << 32) + slice - 1) / slice;
        const int64_t     mb               = first_wrapped + wrapped;
        const int64_t     m                = mb * block_dim;
        ASSERT_LE(m, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));

        // csr2bsr steps through every distinct csr column of a block row, and
        // csr2bsr_nnz through every distinct block column. The first test takes
        // consecutive columns, which keeps the BSR values small, the second one
        // column per block column.
        const int64_t stride    = with_values ? 1 : block_dim;
        const int64_t long_cols = with_values ? int64_t(1) << 18 : int64_t(1) << 20;
        const int64_t width     = (long_cols * stride + block_dim - 1) / block_dim * block_dim;

        std::vector<int64_t> planted;
        for(int64_t b = 0; b <= wrapped; ++b)
        {
            planted.push_back(b);
        }
        for(int64_t b = first_wrapped; b < mb; ++b)
        {
            planted.push_back(b);
        }
        const int64_t count = static_cast<int64_t>(planted.size());
        const int64_t n     = count * width;

        // Each planted block row has two nonempty rows, the second one in the
        // third segment row of the kernel.
        auto first_row = [&](int64_t b) { return (b * 7 + 3) % 64; };
        auto cols_of   = [&](int64_t b) { return (b <= wrapped) ? long_cols : short_cols; };
        auto slot_of
            = [&](int64_t b) { return (b <= wrapped) ? b : wrapped + 1 + b - first_wrapped; };
        auto col_at = [&](int64_t b, int64_t r, int64_t k) {
            return slot_of(b) * width + k * stride + (stride == 1 ? 0 : (b + r) % stride);
        };
        auto value_at = [&](int64_t b, int64_t r, int64_t k) {
            return static_cast<float>((b * 31 + r * 7 + k) % 251 + 1);
        };

        auto blocks_of = [&](int64_t b) {
            return (stride == 1) ? (cols_of(b) + block_dim - 1) / block_dim : cols_of(b);
        };
        int64_t nnzb    = 0;
        int64_t csr_nnz = 0;
        for(const int64_t b : planted)
        {
            nnzb += blocks_of(b);
            csr_nnz += 2 * cols_of(b);
        }

        const size_t scratch_per_block
            = static_cast<size_t>(slice) * sizeof(int32_t) * (with_values ? 2 : 1);
        const size_t need
            = sizeof(int32_t) * (static_cast<size_t>(m) + static_cast<size_t>(mb) + 2)
              + scratch_per_block * static_cast<size_t>(mb)
              + static_cast<size_t>(csr_nnz) * (sizeof(int32_t) + sizeof(float))
              + static_cast<size_t>(nnzb) * sizeof(int32_t)
              + (with_values ? static_cast<size_t>(nnzb) * block_dim * block_dim * sizeof(float)
                             : 0);
        size_t free_bytes  = 0;
        size_t total_bytes = 0;
        if(hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess || free_bytes < 2 * need)
        {
            GTEST_SKIP() << "Insufficient device memory: need about " << need << " bytes";
        }

        std::vector<int32_t> want_ptr(static_cast<size_t>(mb) + 1, 0);
        for(const int64_t b : planted)
        {
            want_ptr[static_cast<size_t>(b) + 1] = static_cast<int32_t>(blocks_of(b));
        }
        for(int64_t b = 0; b < mb; ++b)
        {
            want_ptr[static_cast<size_t>(b) + 1] += want_ptr[static_cast<size_t>(b)];
        }
        ASSERT_EQ(static_cast<int64_t>(want_ptr[static_cast<size_t>(mb)]), nnzb);

        // The CSR row pointer is constant between the planted rows. It is
        // written in runs on the device rather than from a 6 GB host array.
        device_vector<int32_t> csr_ptr(static_cast<size_t>(m) + 1);
        if(csr_ptr.ptr == nullptr)
        {
            ADD_FAILURE() << "device allocation failed";
            return;
        }
        std::vector<int32_t> h_ind;
        std::vector<float>   h_val;
        {
            int64_t next = 0;
            int32_t acc  = 0;
            auto    fill = [&](int64_t last) {
                ASSERT_EQ(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(csr_ptr.ptr + next),
                                       acc,
                                       static_cast<size_t>(last - next + 1)),
                          hipSuccess);
            };
            for(int64_t s = 0; s < count; ++s)
            {
                const int64_t b = planted[s];
                for(int64_t r = 0; r < block_dim; ++r)
                {
                    if(r != first_row(b) && r != block_dim - 1)
                    {
                        continue;
                    }
                    fill(b * block_dim + r);
                    next = b * block_dim + r + 1;
                    acc += static_cast<int32_t>(cols_of(b));
                    for(int64_t k = 0; k < cols_of(b); ++k)
                    {
                        h_ind.push_back(static_cast<int32_t>(col_at(b, r, k)));
                        h_val.push_back(value_at(b, r, k));
                    }
                }
            }
            fill(m);
        }
        ASSERT_EQ(static_cast<int64_t>(h_ind.size()), csr_nnz);

        device_vector<int32_t> csr_ind(h_ind);
        device_vector<float>   csr_val(h_val);
        device_vector<int32_t> bsr_ptr(static_cast<size_t>(mb) + 1);
        if(csr_ind.ptr == nullptr || csr_val.ptr == nullptr || bsr_ptr.ptr == nullptr)
        {
            ADD_FAILURE() << "device allocation failed";
            return;
        }

        struct MatDescrs
        {
            rocsparse_mat_descr bsr = nullptr;
            rocsparse_mat_descr csr = nullptr;

            ~MatDescrs()
            {
                if(bsr != nullptr)
                {
                    (void)rocsparse_destroy_mat_descr(bsr);
                }
                if(csr != nullptr)
                {
                    (void)rocsparse_destroy_mat_descr(csr);
                }
            }
        } descrs;
        ASSERT_EQ(rocsparse_create_mat_descr(&descrs.bsr), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_descr(&descrs.csr), rocsparse_status_success);

        const auto first_diff = [](const auto& got, const auto& expected) -> int64_t {
            for(size_t i = 0; i < got.size() && i < expected.size(); ++i)
            {
                if(got[i] != expected[i])
                {
                    return static_cast<int64_t>(i);
                }
            }
            return -1;
        };

        if(!with_values)
        {
            // Poison the row pointer so a skipped block row cannot pass.
            ASSERT_EQ(hipMemset(bsr_ptr.ptr, 0xFF, bsr_ptr.n * sizeof(int32_t)), hipSuccess);

            rocsparse_int bsr_nnz = -1;
            ASSERT_EQ(rocsparse_csr2bsr_nnz(handle,
                                            rocsparse_direction_row,
                                            static_cast<rocsparse_int>(m),
                                            static_cast<rocsparse_int>(n),
                                            descrs.csr,
                                            csr_ptr,
                                            csr_ind,
                                            static_cast<rocsparse_int>(block_dim),
                                            descrs.bsr,
                                            bsr_ptr,
                                            &bsr_nnz),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

            const std::vector<int32_t> got_ptr = to_host(bsr_ptr);
            const int64_t              diff    = first_diff(got_ptr, want_ptr);
            EXPECT_EQ(diff, -1) << "bsr_row_ptr differs from block row " << diff << ": "
                                << got_ptr[static_cast<size_t>(std::max<int64_t>(diff, 0))]
                                << ", want "
                                << want_ptr[static_cast<size_t>(std::max<int64_t>(diff, 0))];
            EXPECT_EQ(static_cast<int64_t>(bsr_nnz), nnzb);
            return;
        }

        // The reference holds the planted block rows alone, which are the only
        // ones with blocks, so its arrays line up with the device arrays.
        HostSparse sub;
        sub.m = count * block_dim;
        sub.n = n;
        sub.ptr.push_back(0);
        for(int64_t s = 0; s < count; ++s)
        {
            const int64_t b = planted[s];
            for(int64_t r = 0; r < block_dim; ++r)
            {
                if(r == first_row(b) || r == block_dim - 1)
                {
                    for(int64_t k = 0; k < cols_of(b); ++k)
                    {
                        sub.ind.push_back(col_at(b, r, k));
                        sub.val.push_back(static_cast<double>(value_at(b, r, k)));
                    }
                }
                sub.ptr.push_back(static_cast<int64_t>(sub.ind.size()));
            }
        }
        const HostSparse want = host_csr2bsr(sub, block_dim, rocsparse_direction_row);
        ASSERT_EQ(want.nnz(), nnzb);

        ASSERT_EQ(hipMemcpy(bsr_ptr.ptr,
                            want_ptr.data(),
                            want_ptr.size() * sizeof(int32_t),
                            hipMemcpyHostToDevice),
                  hipSuccess);
        device_vector<int32_t> bsr_ind(static_cast<size_t>(nnzb));
        device_vector<float>   bsr_val(static_cast<size_t>(nnzb) * block_dim * block_dim);
        if(bsr_ind.ptr == nullptr || bsr_val.ptr == nullptr)
        {
            ADD_FAILURE() << "device allocation failed";
            return;
        }
        ASSERT_EQ(hipMemset(bsr_ind.ptr, 0xFF, bsr_ind.n * sizeof(int32_t)), hipSuccess);
        ASSERT_EQ(hipMemset(bsr_val.ptr, 0xFF, bsr_val.n * sizeof(float)), hipSuccess);

        ASSERT_EQ(rocsparse_scsr2bsr(handle,
                                     rocsparse_direction_row,
                                     static_cast<rocsparse_int>(m),
                                     static_cast<rocsparse_int>(n),
                                     descrs.csr,
                                     csr_val,
                                     csr_ptr,
                                     csr_ind,
                                     static_cast<rocsparse_int>(block_dim),
                                     descrs.bsr,
                                     bsr_val,
                                     bsr_ptr,
                                     bsr_ind),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const std::vector<int32_t> got_ind  = to_host(bsr_ind);
        const int64_t              ind_diff = first_diff(got_ind, want.ind);
        EXPECT_EQ(ind_diff, -1) << "bsr_col_ind differs at block " << ind_diff;

        const std::vector<float> got_val = to_host(bsr_val);
        ASSERT_EQ(got_val.size(), want.val.size());
        for(size_t i = 0; i < got_val.size(); ++i)
        {
            if(got_val[i] != static_cast<float>(want.val[i]))
            {
                ADD_FAILURE() << "bsr_val differs at block " << i / (block_dim * block_dim)
                              << ", entry " << i % (block_dim * block_dim) << ": " << got_val[i]
                              << ", want " << want.val[i];
                return;
            }
        }
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
// kernel; 8, 16 and 32 take the 8-32 kernel and 64, 128 and 200 the 33-256
// kernel.
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

// The 129-256 bucket, launch_bsr2csr_block_per_row_33_256_kernel(1024, 256, 32).
TEST_F(BsrConversionGrids, bsr2csr_block_dim_200)
{
    check_bsr2csr_all(handle, 200);
}

// Legacy int32 rocsparse_sbsr2csr, 1024 threads, mb past (2^32 - 1) / 1024.
// Goes through get_grid_size_x with the device's real grid limit.
TEST_F(BsrConversionGrids, bsr2csr_real_threshold_block_dim_8)
{
    check_bsr2csr_real_threshold(handle, 8, 1024);
}

// The 2-7 kernels launch 256 threads, so their limit is (2^32 - 1) / 256.
TEST_F(BsrConversionGrids, bsr2csr_real_threshold_block_dim_2)
{
    check_bsr2csr_real_threshold(handle, 2, 256);
}

// The 33-256 kernel, 1024 threads. block_dim 33 is the smallest in the bucket,
// which keeps the CSR row pointer to about 550 MB.
TEST_F(BsrConversionGrids, bsr2csr_real_threshold_block_dim_33)
{
    check_bsr2csr_real_threshold(handle, 33, 1024);
}

// csr2bsr_nnz_65_inf_kernel with block_dim 65 and 22369624 block rows: the
// scratch slice start of the blocks from 22369622 on exceeds 2^32 elements.
// Needs about 23 GB of free device memory; skipped otherwise.
TEST_F(BsrConversionGrids, csr2bsr_nnz_segment_base_past_2_32)
{
    check_65_inf_segment_base(handle, false);
}

// The same for csr2bsr_65_inf_kernel, which needs about 41 GB.
TEST_F(BsrConversionGrids, csr2bsr_segment_base_past_2_32)
{
    check_65_inf_segment_base(handle, true);
}
