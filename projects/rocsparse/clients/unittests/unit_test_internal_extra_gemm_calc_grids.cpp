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
// Forced-clamp tests for the csrgemm and bsrgemm calc grids (AISPARSE-677).
//
// FOCUS. csrgemm_intermediate_products, the csrgemm nnz / fill / symbolic /
// numeric bucket kernels (wf-per-row groups 0 and 1, block-per-row groups 2 to
// 9, the group 10 multipass fallback) and every bsrgemm fill bucket clamp grid.x
// with rocsparse::get_grid_size_x and grid-stride over the rows of their bucket.
// rocsparse::dispatch_grid_stride_x selects the GRID_STRIDE instantiation only
// when the clamp binds, so a normally sized problem never runs the looping
// variant at all.
//
// WHY NOT TEST THE REAL THRESHOLD. The clamp binds at (2^32 - 1) / 256 =
// 16,777,215 blocks, i.e. about 16.7M rows in one block-per-row bucket. These
// tests shrink handle->properties.maxGridSize[0] with ScopedMaxGridSizeX to 1,
// 3, 7 and 64 blocks instead, so a problem of a few thousand rows takes the
// clamped, looping path. This is the AISPARSE-702 idiom.
//
// WHAT MAKES EACH CASE LOAD-BEARING. The operands are built from row classes,
// each aimed at one bucket (see csr_classes and bsr_classes). Every non-empty
// bucket holds at least 8 rows, so limits 1, 3 and 7 clamp every bucketed
// launch. The wf-per-row buckets, csrgemm_intermediate_products and the
// block_dim > 8 bsrgemm launch are more than 64 blocks wide, so limit 64 clamps
// those too. Each test checks the bucket populations on the host reference
// before it trusts a pass.
//
// EXACT ARITHMETIC. All values, alpha and beta are small integers, so every
// entry of C is exactly representable in float and does not depend on the
// order in which the hash tables accumulate. That allows an exact comparison
// against the host reference and against the unclamped run.
//
// TARGET: rocsparse-unit-test-device. The tests drive the public rocsparse_spgemm
// entry point and need the complete handle type to shrink the grid limit.
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
    constexpr int clamped_limits[] = {1, 3, 7, 64};

    constexpr double alpha_value = 2.0;
    constexpr double beta_value  = 3.0;

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

    // Rows of A that each pick a_per_row distinct rows of B from a private pool of
    // B rows with b_row_len random columns. Row nnz of C is then about the union
    // of a_per_row random column sets of length b_row_len.
    struct RowClass
    {
        int64_t rows;
        int64_t a_per_row;
        int64_t b_row_len;
    };

    // n = 50000 columns. The comment on each class is the csrgemm fill bucket
    // (row nnz of C, D included) it lands in; the nnz stage buckets by
    // intermediate products and sees classes 0 to 9 in its lower buckets and
    // class 10 (5 * 15000 = 75000 products) in its group 10.
    constexpr int64_t  csr_n         = 50000;
    constexpr RowClass csr_classes[] = {
        {2400, 2, 4}, //    <= 16, wf-per-row group 0, 75 blocks of 32 rows
        {1200, 3, 7}, // 17 - 32,  wf-per-row group 1, 75 blocks of 16 rows
        {72, 4, 25}, //  33 - 256,   block-per-row group 2
        {72, 4, 90}, //  257 - 512,  group 3
        {72, 4, 190}, // 513 - 1024, group 4
        {72, 4, 380}, // 1025 - 2048, group 5
        {72, 4, 770}, // 2049 - 4096, group 6
        {12, 4, 1600}, // 4097 - 8192, group 7 (group 10 if shared memory is short)
        {12, 4, 3300}, // 8193 - 16384, group 8 (likewise)
        {12, 4, 7000}, // 16385 - 32768, group 9 (likewise)
        {8, 5, 15000}, // > 32768, group 10 multipass
    };

    // Rows of C with at most 16 entries, so csrgemm skips the grouping and runs
    // one wf-per-row launch over all rows: 20000 rows are 625 blocks.
    constexpr int64_t  short_n         = 20000;
    constexpr RowClass short_classes[] = {{20000, 4, 4}};

    // nb = 2000 block columns. The comment is the bsrgemm bucket (nnzb per row
    // of C, D included). block_dim > 8 skips the grouping and runs all 2620 block
    // rows through one block-per-row launch.
    constexpr int64_t  bsr_nb        = 2000;
    constexpr RowClass bsr_classes[] = {
        {1100, 2, 3}, //   <= 8, group 0
        {1100, 3, 4}, //  9 - 16, group 1
        {100, 3, 8}, //  17 - 32, group 2
        {100, 4, 12}, // 33 - 64, group 3
        {100, 4, 25}, // 65 - 128, group 4
        {100, 4, 50}, // 129 - 256, group 5
        {12, 4, 100}, // 257 - 512, group 6
        {8, 4, 200}, //  > 512, group 7
    };

    // Entries of D per row, at block level for BSR.
    constexpr int64_t d_per_row = 2;

    // Distinct sorted random columns in [0, n).
    std::vector<int64_t> random_columns(std::mt19937& rng, int64_t n, int64_t count)
    {
        std::vector<int64_t>                   cols;
        std::vector<char>                      used(n, 0);
        std::uniform_int_distribution<int64_t> col(0, n - 1);
        while(static_cast<int64_t>(cols.size()) < count)
        {
            const int64_t c = col(rng);
            if(!used[c])
            {
                used[c] = 1;
                cols.push_back(c);
            }
        }
        std::sort(cols.begin(), cols.end());
        return cols;
    }

    // Small nonzero integers so every product and sum is exact.
    double random_value(std::mt19937& rng)
    {
        static const double values[] = {-3.0, -2.0, -1.0, 1.0, 2.0, 3.0};
        return values[std::uniform_int_distribution<int>(0, 5)(rng)];
    }

    void append_row(HostSparse& a, const std::vector<int64_t>& cols, std::mt19937& rng)
    {
        const int64_t block_size = a.block_dim * a.block_dim;
        for(const int64_t c : cols)
        {
            a.ind.push_back(c);
            for(int64_t e = 0; e < block_size; ++e)
            {
                a.val.push_back(random_value(rng));
            }
        }
        a.ptr.push_back(static_cast<int64_t>(a.ind.size()));
    }

    struct Operands
    {
        HostSparse a;
        HostSparse b;
        HostSparse d;
    };

    // Build A (m x k), B (k x n) and D (m x n) from the row classes. The rows of
    // A are shuffled so every bucket's rows are scattered and the permutation
    // that groups them is far from the identity.
    template <size_t N>
    Operands make_operands(const RowClass (&classes)[N], int64_t n, int64_t block_dim, bool with_d)
    {
        std::mt19937 rng(677);

        Operands op;
        op.b.n         = n;
        op.b.block_dim = block_dim;
        op.b.ptr.push_back(0);

        // One pool of B rows per class.
        std::vector<int64_t> pool_begin(N);
        std::vector<int64_t> pool_size(N);
        for(size_t c = 0; c < N; ++c)
        {
            pool_begin[c] = op.b.m;
            pool_size[c]  = std::max<int64_t>(8, 4 * classes[c].a_per_row);
            for(int64_t r = 0; r < pool_size[c]; ++r)
            {
                append_row(op.b, random_columns(rng, n, classes[c].b_row_len), rng);
                ++op.b.m;
            }
        }

        std::vector<size_t> row_class;
        for(size_t c = 0; c < N; ++c)
        {
            row_class.insert(row_class.end(), classes[c].rows, c);
        }
        std::shuffle(row_class.begin(), row_class.end(), rng);

        op.a.m         = static_cast<int64_t>(row_class.size());
        op.a.n         = op.b.m;
        op.a.block_dim = block_dim;
        op.a.ptr.push_back(0);

        op.d.m         = op.a.m;
        op.d.n         = n;
        op.d.block_dim = block_dim;
        op.d.ptr.push_back(0);

        for(const size_t c : row_class)
        {
            std::vector<int64_t> picks = random_columns(rng, pool_size[c], classes[c].a_per_row);
            for(int64_t& p : picks)
            {
                p += pool_begin[c];
            }
            append_row(op.a, picks, rng);
            append_row(
                op.d, with_d ? random_columns(rng, n, d_per_row) : std::vector<int64_t>(), rng);
        }
        return op;
    }

    int64_t block_entry(rocsparse_direction dir, int64_t block_dim, int64_t r, int64_t c)
    {
        return (dir == rocsparse_direction_row) ? r * block_dim + c : c * block_dim + r;
    }

    // C = alpha * A * B + beta * D on the host, with the columns of each row
    // sorted. The pattern is the structural union, as rocsparse computes it.
    HostSparse host_gemm(const Operands& op, bool with_d, rocsparse_direction dir)
    {
        const HostSparse& a  = op.a;
        const HostSparse& b  = op.b;
        const HostSparse& d  = op.d;
        const int64_t     bd = a.block_dim;
        const int64_t     bs = bd * bd;

        HostSparse c;
        c.m         = a.m;
        c.n         = b.n;
        c.block_dim = bd;
        c.ptr.push_back(0);

        std::vector<int64_t> slot(b.n, -1);
        std::vector<int64_t> cols;
        std::vector<double>  acc;

        for(int64_t i = 0; i < a.m; ++i)
        {
            cols.clear();
            acc.clear();

            auto slot_of = [&](int64_t col) {
                if(slot[col] < 0)
                {
                    slot[col] = static_cast<int64_t>(cols.size());
                    cols.push_back(col);
                    acc.resize(acc.size() + bs, 0.0);
                }
                return slot[col];
            };

            for(int64_t pa = a.ptr[i]; pa < a.ptr[i + 1]; ++pa)
            {
                const int64_t k     = a.ind[pa];
                const double* a_blk = &a.val[pa * bs];
                for(int64_t pb = b.ptr[k]; pb < b.ptr[k + 1]; ++pb)
                {
                    const double* b_blk = &b.val[pb * bs];
                    double*       c_blk = &acc[slot_of(b.ind[pb]) * bs];
                    for(int64_t r = 0; r < bd; ++r)
                    {
                        for(int64_t cc = 0; cc < bd; ++cc)
                        {
                            double sum = 0.0;
                            for(int64_t t = 0; t < bd; ++t)
                            {
                                sum += a_blk[block_entry(dir, bd, r, t)]
                                       * b_blk[block_entry(dir, bd, t, cc)];
                            }
                            c_blk[block_entry(dir, bd, r, cc)] += alpha_value * sum;
                        }
                    }
                }
            }

            if(with_d)
            {
                for(int64_t pd = d.ptr[i]; pd < d.ptr[i + 1]; ++pd)
                {
                    double* c_blk = &acc[slot_of(d.ind[pd]) * bs];
                    for(int64_t e = 0; e < bs; ++e)
                    {
                        c_blk[e] += beta_value * d.val[pd * bs + e];
                    }
                }
            }

            std::vector<int64_t> sorted = cols;
            std::sort(sorted.begin(), sorted.end());
            for(const int64_t col : sorted)
            {
                c.ind.push_back(col);
                c.val.insert(c.val.end(), &acc[slot[col] * bs], &acc[slot[col] * bs] + bs);
            }
            for(const int64_t col : cols)
            {
                slot[col] = -1;
            }
            c.ptr.push_back(static_cast<int64_t>(c.ind.size()));
        }
        return c;
    }

    // Rows of m whose entry count lies in [lo, hi].
    int64_t rows_with_nnz(const HostSparse& m, int64_t lo, int64_t hi)
    {
        int64_t count = 0;
        for(int64_t i = 0; i < m.m; ++i)
        {
            const int64_t nnz = m.ptr[i + 1] - m.ptr[i];
            count += (nnz >= lo && nnz <= hi) ? 1 : 0;
        }
        return count;
    }

    // Empty if equal, otherwise where the first difference is.
    std::string first_difference(const HostSparse& got, const HostSparse& want)
    {
        std::ostringstream msg;
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

    template <typename I, typename J, typename T>
    struct DeviceSparse
    {
        std::unique_ptr<device_vector<I>> ptr;
        std::unique_ptr<device_vector<J>> ind;
        std::unique_ptr<device_vector<T>> val;

        explicit DeviceSparse(const HostSparse& h)
        {
            ptr.reset(new device_vector<I>(std::vector<I>(h.ptr.begin(), h.ptr.end())));
            ind.reset(new device_vector<J>(std::vector<J>(h.ind.begin(), h.ind.end())));
            val.reset(new device_vector<T>(std::vector<T>(h.val.begin(), h.val.end())));
        }
    };

    // Owns the spgemm descriptors so each one created is destroyed on every
    // return path.
    struct SpgemmDescrs
    {
        rocsparse_spmat_descr a = nullptr;
        rocsparse_spmat_descr b = nullptr;
        rocsparse_spmat_descr c = nullptr;
        rocsparse_spmat_descr d = nullptr;

        SpgemmDescrs() = default;

        SpgemmDescrs(const SpgemmDescrs&) = delete;

        SpgemmDescrs& operator=(const SpgemmDescrs&) = delete;

        ~SpgemmDescrs()
        {
            for(rocsparse_spmat_descr m : {a, b, c, d})
            {
                if(m != nullptr)
                {
                    (void)rocsparse_destroy_spmat_descr(m);
                }
            }
        }
    };

    enum class Flow
    {
        // buffer_size, nnz, compute: csrgemm_nnz_calc and csrgemm_calc (bsrgemm_calc).
        nnz_compute,
        // buffer_size, nnz, symbolic, numeric: csrgemm_symbolic_calc and
        // csrgemm_numeric_calc. CSR only.
        symbolic_numeric
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

    // Run C = alpha * A * B (+ beta * D) through rocsparse_spgemm with grid.x
    // limited to `limit` blocks (0 keeps the device limit) for every stage, and
    // return C. Records a gtest failure and returns an empty matrix on error.
    template <typename I, typename J, typename T>
    HostSparse device_gemm(rocsparse_handle    handle,
                           const Operands&     op,
                           bool                with_d,
                           bool                bsr,
                           rocsparse_direction dir,
                           Flow                flow,
                           int                 limit)
    {
        const T alpha = static_cast<T>(alpha_value);
        const T beta  = static_cast<T>(beta_value);

        DeviceSparse<I, J, T> da(op.a);
        DeviceSparse<I, J, T> db(op.b);
        DeviceSparse<I, J, T> dd(op.d);
        device_vector<I>      c_ptr(static_cast<size_t>(op.a.m + 1));
        if(da.ptr->ptr == nullptr || da.ind->ptr == nullptr || da.val->ptr == nullptr
           || db.ptr->ptr == nullptr || db.ind->ptr == nullptr || db.val->ptr == nullptr
           || dd.ptr->ptr == nullptr || c_ptr.ptr == nullptr)
        {
            ADD_FAILURE() << "device allocation failed";
            return {};
        }

        HostSparse c_shape;
        c_shape.m         = op.a.m;
        c_shape.n         = op.b.n;
        c_shape.block_dim = op.a.block_dim;

        SpgemmDescrs descrs;
        if(create_descr<I, J, T>(
               &descrs.a, op.a, op.a.nnz(), da.ptr->ptr, da.ind->ptr, da.val->ptr, bsr, dir)
               != rocsparse_status_success
           || create_descr<I, J, T>(
                  &descrs.b, op.b, op.b.nnz(), db.ptr->ptr, db.ind->ptr, db.val->ptr, bsr, dir)
                  != rocsparse_status_success
           || create_descr<I, J, T>(
                  &descrs.d, op.d, op.d.nnz(), dd.ptr->ptr, dd.ind->ptr, dd.val->ptr, bsr, dir)
                  != rocsparse_status_success
           || create_descr<I, J, T>(&descrs.c, c_shape, 0, c_ptr.ptr, nullptr, nullptr, bsr, dir)
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

        std::unique_ptr<device_vector<J>> c_ind;
        std::unique_ptr<device_vector<T>> c_val;
        size_t                            buffer_size = 0;
        void*                             buffer      = nullptr;

        auto spgemm = [&](rocsparse_spgemm_stage stage) {
            return rocsparse_spgemm(handle,
                                    rocsparse_operation_none,
                                    rocsparse_operation_none,
                                    &alpha,
                                    descrs.a,
                                    descrs.b,
                                    with_d ? &beta : nullptr,
                                    descrs.d,
                                    descrs.c,
                                    dt_of<T>(),
                                    rocsparse_spgemm_alg_default,
                                    stage,
                                    &buffer_size,
                                    buffer);
        };

        rocsparse_status status = spgemm(rocsparse_spgemm_stage_buffer_size);
        if(status == rocsparse_status_success
           && hipMalloc(&buffer, std::max<size_t>(buffer_size, 1)) != hipSuccess)
        {
            buffer = nullptr;
            status = rocsparse_status_memory_error;
        }
        if(status == rocsparse_status_success)
        {
            status = spgemm(rocsparse_spgemm_stage_nnz);
        }

        int64_t rows = 0;
        int64_t cols = 0;
        int64_t nnz  = 0;
        if(status == rocsparse_status_success)
        {
            status = rocsparse_spmat_get_size(descrs.c, &rows, &cols, &nnz);
        }
        if(status == rocsparse_status_success)
        {
            const int64_t bs = op.a.block_dim * op.a.block_dim;
            c_ind.reset(new device_vector<J>(static_cast<size_t>(std::max<int64_t>(nnz, 1))));
            c_val.reset(new device_vector<T>(static_cast<size_t>(std::max<int64_t>(nnz * bs, 1))));
            if(c_ind->ptr == nullptr || c_val->ptr == nullptr)
            {
                status = rocsparse_status_memory_error;
            }
            else if(bsr)
            {
                status = rocsparse_bsr_set_pointers(descrs.c, c_ptr.ptr, c_ind->ptr, c_val->ptr);
            }
            else
            {
                status = rocsparse_csr_set_pointers(descrs.c, c_ptr.ptr, c_ind->ptr, c_val->ptr);
            }
        }
        if(status == rocsparse_status_success)
        {
            if(flow == Flow::nnz_compute)
            {
                status = spgemm(rocsparse_spgemm_stage_compute);
            }
            else
            {
                status = spgemm(rocsparse_spgemm_stage_symbolic);
                if(status == rocsparse_status_success)
                {
                    status = spgemm(rocsparse_spgemm_stage_numeric);
                }
            }
        }
        if(status == rocsparse_status_success && hipDeviceSynchronize() != hipSuccess)
        {
            status = rocsparse_status_internal_error;
        }
        if(buffer != nullptr)
        {
            (void)hipFree(buffer);
        }
        if(status != rocsparse_status_success)
        {
            ADD_FAILURE() << "rocsparse_spgemm failed with status " << status << " at limit "
                          << limit;
            return {};
        }

        const std::vector<I> h_ptr = to_host(c_ptr);
        const std::vector<J> h_ind = to_host<J>(c_ind->ptr, static_cast<size_t>(nnz));
        const std::vector<T> h_val
            = to_host<T>(c_val->ptr, static_cast<size_t>(nnz * op.a.block_dim * op.a.block_dim));

        c_shape.ptr.assign(h_ptr.begin(), h_ptr.end());
        c_shape.ind.assign(h_ind.begin(), h_ind.end());
        c_shape.val.assign(h_val.begin(), h_val.end());
        return c_shape;
    }

    // Unclamped control against the host reference, then every clamped limit
    // against both the host reference and the unclamped result.
    template <typename I, typename J, typename T>
    void check_all_limits(rocsparse_handle    handle,
                          const Operands&     op,
                          const HostSparse&   want,
                          bool                with_d,
                          bool                bsr,
                          rocsparse_direction dir,
                          Flow                flow)
    {
        const HostSparse  unclamped = device_gemm<I, J, T>(handle, op, with_d, bsr, dir, flow, 0);
        const std::string control   = first_difference(unclamped, want);
        ASSERT_TRUE(control.empty()) << "unclamped control: " << control;

        for(const int limit : clamped_limits)
        {
            const HostSparse  got = device_gemm<I, J, T>(handle, op, with_d, bsr, dir, flow, limit);
            const std::string vs_host = first_difference(got, want);
            EXPECT_TRUE(vs_host.empty())
                << "grid.x clamped to " << limit << " blocks, against the host: " << vs_host;
            const std::string vs_unclamped = first_difference(got, unclamped);
            EXPECT_TRUE(vs_unclamped.empty())
                << "grid.x clamped to " << limit << " blocks, against unclamped: " << vs_unclamped;
        }
    }

    // The csrgemm fill buckets of csr_classes must each hold enough rows that
    // the clamp binds; see the file header.
    void expect_csr_buckets(const HostSparse& c)
    {
        EXPECT_GT(rows_with_nnz(c, 0, 16), 64 * 32) << "wf-per-row group 0";
        EXPECT_GT(rows_with_nnz(c, 17, 32), 64 * 16) << "wf-per-row group 1";
        const int64_t bounds[] = {32, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
        for(size_t g = 0; g + 1 < sizeof(bounds) / sizeof(bounds[0]); ++g)
        {
            EXPECT_GE(rows_with_nnz(c, bounds[g] + 1, bounds[g + 1]), 8)
                << "block-per-row bucket " << bounds[g] + 1 << " - " << bounds[g + 1];
        }
        EXPECT_GE(rows_with_nnz(c, 32769, c.n), 8) << "group 10";
    }

    void expect_bsr_buckets(const HostSparse& c)
    {
        const int64_t bounds[] = {0, 8, 16, 32, 64, 128, 256, 512};
        for(size_t g = 0; g + 1 < sizeof(bounds) / sizeof(bounds[0]); ++g)
        {
            EXPECT_GE(rows_with_nnz(c, bounds[g] + 1, bounds[g + 1]), 8)
                << "bucket " << bounds[g] + 1 << " - " << bounds[g + 1];
        }
        EXPECT_GE(rows_with_nnz(c, 513, c.n), 8) << "group 7";
    }

    template <typename I, typename J, typename T>
    void check_csrgemm_buckets(rocsparse_handle handle, bool with_d, Flow flow)
    {
        const Operands   op   = make_operands(csr_classes, csr_n, 1, with_d);
        const HostSparse want = host_gemm(op, with_d, rocsparse_direction_row);
        expect_csr_buckets(want);
        check_all_limits<I, J, T>(handle, op, want, with_d, false, rocsparse_direction_row, flow);
    }

    template <typename I, typename J, typename T>
    void check_bsrgemm(rocsparse_handle handle, int64_t block_dim, rocsparse_direction dir)
    {
        const Operands   op   = make_operands(bsr_classes, bsr_nb, block_dim, true);
        const HostSparse want = host_gemm(op, true, dir);
        expect_bsr_buckets(want);
        check_all_limits<I, J, T>(handle, op, want, true, true, dir, Flow::nnz_compute);
    }

    using GemmCalcGrids = HandleTest;
}

// ---------------------------------------------------------------------------
// csrgemm, C = alpha * A * B + beta * D, nnz + compute stages.
// ---------------------------------------------------------------------------

TEST_F(GemmCalcGrids, csrgemm_multadd_compute_f32_i32)
{
    check_csrgemm_buckets<int32_t, int32_t, float>(handle, true, Flow::nnz_compute);
}

TEST_F(GemmCalcGrids, csrgemm_multadd_compute_f64_i32)
{
    check_csrgemm_buckets<int32_t, int32_t, double>(handle, true, Flow::nnz_compute);
}

TEST_F(GemmCalcGrids, csrgemm_multadd_compute_f32_i64)
{
    check_csrgemm_buckets<int64_t, int64_t, float>(handle, true, Flow::nnz_compute);
}

TEST_F(GemmCalcGrids, csrgemm_multadd_compute_f64_i64)
{
    check_csrgemm_buckets<int64_t, int64_t, double>(handle, true, Flow::nnz_compute);
}

// 64-bit row pointers with 32-bit column indices.
TEST_F(GemmCalcGrids, csrgemm_multadd_compute_f64_i64_i32)
{
    check_csrgemm_buckets<int64_t, int32_t, double>(handle, true, Flow::nnz_compute);
}

// ---------------------------------------------------------------------------
// csrgemm, symbolic + numeric stages (csrgemm_symbolic_calc, csrgemm_numeric_calc).
// ---------------------------------------------------------------------------

TEST_F(GemmCalcGrids, csrgemm_multadd_symbolic_numeric_f32_i32)
{
    check_csrgemm_buckets<int32_t, int32_t, float>(handle, true, Flow::symbolic_numeric);
}

TEST_F(GemmCalcGrids, csrgemm_multadd_symbolic_numeric_f64_i64)
{
    check_csrgemm_buckets<int64_t, int64_t, double>(handle, true, Flow::symbolic_numeric);
}

// ---------------------------------------------------------------------------
// csrgemm, C = alpha * A * B (mul only, beta == nullptr).
// ---------------------------------------------------------------------------

TEST_F(GemmCalcGrids, csrgemm_mult_compute_f64_i32)
{
    check_csrgemm_buckets<int32_t, int32_t, double>(handle, false, Flow::nnz_compute);
}

TEST_F(GemmCalcGrids, csrgemm_mult_compute_f32_i64)
{
    check_csrgemm_buckets<int64_t, int64_t, float>(handle, false, Flow::nnz_compute);
}

TEST_F(GemmCalcGrids, csrgemm_mult_symbolic_numeric_f64_i32)
{
    check_csrgemm_buckets<int32_t, int32_t, double>(handle, false, Flow::symbolic_numeric);
}

// Every row of C has at most 16 entries, so csrgemm skips the grouping and runs
// a single wf-per-row launch of 625 blocks over all rows, with no permutation.
TEST_F(GemmCalcGrids, csrgemm_short_rows_single_group)
{
    const Operands   op   = make_operands(short_classes, short_n, 1, false);
    const HostSparse want = host_gemm(op, false, rocsparse_direction_row);
    ASSERT_EQ(rows_with_nnz(want, 0, 16), want.m);
    check_all_limits<int32_t, int32_t, float>(
        handle, op, want, false, false, rocsparse_direction_row, Flow::nnz_compute);
    check_all_limits<int64_t, int64_t, double>(
        handle, op, want, false, false, rocsparse_direction_row, Flow::symbolic_numeric);
}

// ---------------------------------------------------------------------------
// bsrgemm, C = alpha * A * B + beta * D. block_dim 2 takes the 2x2 fill kernels,
// 4 the 3-4 path, 8 the 5-8 path, and 16 the ungrouped block-per-row multipass.
// ---------------------------------------------------------------------------

TEST_F(GemmCalcGrids, bsrgemm_bd2_row_f32_i32)
{
    check_bsrgemm<int32_t, int32_t, float>(handle, 2, rocsparse_direction_row);
}

TEST_F(GemmCalcGrids, bsrgemm_bd2_column_f32_i32)
{
    check_bsrgemm<int32_t, int32_t, float>(handle, 2, rocsparse_direction_column);
}

TEST_F(GemmCalcGrids, bsrgemm_bd2_row_f64_i64)
{
    check_bsrgemm<int64_t, int64_t, double>(handle, 2, rocsparse_direction_row);
}

TEST_F(GemmCalcGrids, bsrgemm_bd2_column_f64_i64)
{
    check_bsrgemm<int64_t, int64_t, double>(handle, 2, rocsparse_direction_column);
}

TEST_F(GemmCalcGrids, bsrgemm_bd4_row_f64_i32)
{
    check_bsrgemm<int32_t, int32_t, double>(handle, 4, rocsparse_direction_row);
}

TEST_F(GemmCalcGrids, bsrgemm_bd4_column_f32_i64)
{
    check_bsrgemm<int64_t, int64_t, float>(handle, 4, rocsparse_direction_column);
}

TEST_F(GemmCalcGrids, bsrgemm_bd8_row_f32_i64)
{
    check_bsrgemm<int64_t, int64_t, float>(handle, 8, rocsparse_direction_row);
}

TEST_F(GemmCalcGrids, bsrgemm_bd8_column_f64_i32)
{
    check_bsrgemm<int32_t, int32_t, double>(handle, 8, rocsparse_direction_column);
}

TEST_F(GemmCalcGrids, bsrgemm_bd16_row_f64_i64)
{
    check_bsrgemm<int64_t, int64_t, double>(handle, 16, rocsparse_direction_row);
}

TEST_F(GemmCalcGrids, bsrgemm_bd16_column_f32_i32)
{
    check_bsrgemm<int32_t, int32_t, float>(handle, 16, rocsparse_direction_column);
}
