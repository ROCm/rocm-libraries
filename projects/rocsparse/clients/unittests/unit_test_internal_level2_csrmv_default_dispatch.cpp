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
// Default CSR SpMV selects nnz-split only for a poor adaptive partition, and
// only when the caller asked for the default algorithm. These tests run the
// public SpMV routines, read which analysis buffers and algorithm were stored,
// and check y against a host reference.
//
#include "unit_test_utils.hpp"

#include "../../library/src/level2/rocsparse_csrmv.hpp"
#include "rocsparse_mat_info.hpp"
#include "rocsparse_spmat_descr.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    // Stored pattern: the rows of a CSR matrix, or the columns of a CSC matrix.
    struct Pattern
    {
        int64_t              m = 0;
        int64_t              n = 0;
        std::vector<int64_t> ptr{0};
        std::vector<int64_t> ind;

        int64_t nnz() const
        {
            return static_cast<int64_t>(ind.size());
        }

        void add(const std::vector<int64_t>& cols)
        {
            ind.insert(ind.end(), cols.begin(), cols.end());
            ptr.push_back(nnz());
            ++m;
        }

        void add(int64_t len)
        {
            std::vector<int64_t> cols(static_cast<size_t>(len));
            for(int64_t k = 0; k < len; ++k)
            {
                cols[static_cast<size_t>(k)] = k;
            }
            add(cols);
        }
    };

    // One nonzero at each end and a long empty gap between them. Adaptive keeps
    // that gap in one row block.
    Pattern scattered_gap()
    {
        Pattern a;
        a.n = 1;
        a.add(1);
        for(int i = 0; i < 20000; ++i)
        {
            a.add(0);
        }
        a.add(1);
        return a;
    }

    // The same gap in a square lower-triangular matrix, so it can be symmetric.
    Pattern scattered_gap_lower()
    {
        Pattern a;
        a.n = 20002;
        a.add(std::vector<int64_t>{0});
        for(int i = 0; i < 20000; ++i)
        {
            a.add(0);
        }
        a.add(std::vector<int64_t>{0, 20001});
        return a;
    }

    // Every row is short and populated, so one adaptive block holds the matrix.
    Pattern populated()
    {
        Pattern a;
        a.n = 8;
        for(int i = 0; i < 64; ++i)
        {
            a.add(8);
        }
        return a;
    }

    // A just-long row, then two short rows, repeated. Adaptive cuts a block on
    // every long-to-short transition.
    Pattern alternating()
    {
        Pattern a;
        a.n = 160;
        for(int g = 0; g < 8; ++g)
        {
            a.add(160);
            a.add(3);
            a.add(3);
        }
        return a;
    }

    // Rows longer than a block alternate with empty rows, so adaptive puts each
    // empty row in a block of its own.
    Pattern long_and_empty_rows()
    {
        Pattern a;
        a.n = 2000;
        for(int g = 0; g < 8; ++g)
        {
            a.add(2000);
            a.add(0);
        }
        return a;
    }

    // An empty prefix too short for adaptive to skip, then short rows. The
    // first block holds the prefix and the first 1024 short rows.
    Pattern short_empty_prefix()
    {
        Pattern a;
        a.n = 1;
        for(int i = 0; i < 8000; ++i)
        {
            a.add(0);
        }
        for(int i = 0; i < 2048; ++i)
        {
            a.add(1);
        }
        return a;
    }

    // Long rows, then an empty suffix adaptive already skips.
    Pattern long_then_empty()
    {
        Pattern a;
        a.n = 200;
        for(int i = 0; i < 4; ++i)
        {
            a.add(200);
        }
        for(int i = 0; i < 20000; ++i)
        {
            a.add(0);
        }
        return a;
    }

    using cd = std::complex<double>;

    // Small multiples of a quarter, exact in every tested precision.
    cd value_of(int64_t k)
    {
        return {1.0 + 0.25 * static_cast<double>(k % 4), 0.5 * static_cast<double>(k % 3 - 1)};
    }

    cd x_of(int64_t j)
    {
        return {1.0 + static_cast<double>(j % 3), 0.0};
    }

    cd y0_of(int64_t i)
    {
        return {0.5 * static_cast<double>(1 + i % 2), 0.0};
    }

    template <typename T>
    T to_type(const cd& v)
    {
        if constexpr(std::is_same<T, rocsparse_float_complex>{}
                     || std::is_same<T, rocsparse_double_complex>{})
        {
            return T(v.real(), v.imag());
        }
        else
        {
            return static_cast<T>(static_cast<float>(v.real()));
        }
    }

    template <typename T>
    cd to_cd(const T& v)
    {
        if constexpr(std::is_same<T, rocsparse_float_complex>{}
                     || std::is_same<T, rocsparse_double_complex>{})
        {
            return {static_cast<double>(std::real(v)), static_cast<double>(std::imag(v))};
        }
        else
        {
            return {static_cast<double>(static_cast<float>(v)), 0.0};
        }
    }

    struct Options
    {
        rocsparse_spmv_alg   alg         = rocsparse_spmv_alg_default;
        rocsparse_spmv_alg   compute_alg = rocsparse_spmv_alg_default;
        rocsparse_operation  op          = rocsparse_operation_none;
        rocsparse_operation  compute_op  = rocsparse_operation_none;
        bool                 as_csc      = false;
        bool                 symmetric   = false;
        rocsparse_index_base base        = rocsparse_index_base_zero;
        double               beta        = 0.0;
        bool                 v2          = false;
        // Point the descriptor at copies of the arrays between the two stages.
        bool repoint = false;
    };

    Options with_alg(rocsparse_spmv_alg alg)
    {
        Options o;
        o.alg         = alg;
        o.compute_alg = alg;
        return o;
    }

    Options with_op(rocsparse_operation op)
    {
        Options o;
        o.op         = op;
        o.compute_op = op;
        return o;
    }

    struct Result
    {
        bool                 has_info        = false;
        bool                 adaptive_blocks = false;
        bool                 nnzsplit_tuned  = false;
        bool                 copy_ok         = false;
        rocsparse::csrmv_alg selected        = rocsparse::csrmv_alg_default;
        rocsparse_status     compute_status  = rocsparse_status_internal_error;
    };

    // y = op(A) x + beta y0, with A the matrix the routine sees: m x n stored
    // rows for CSR, and the transpose of the stored pattern for CSC. Symmetric
    // matrices are only mirrored for a non-transposed CSR product.
    std::vector<cd> reference(const Pattern& a, const std::vector<cd>& val, const Options& o)
    {
        if(o.symmetric && (o.as_csc || o.compute_op != rocsparse_operation_none))
        {
            ADD_FAILURE() << "the reference does not cover this symmetric case";
        }
        const int64_t rows = o.as_csc ? a.n : a.m;
        const int64_t cols = o.as_csc ? a.m : a.n;
        const int64_t ylen = (o.compute_op == rocsparse_operation_none) ? rows : cols;

        std::vector<cd> y(static_cast<size_t>(ylen));
        for(int64_t i = 0; i < ylen; ++i)
        {
            y[static_cast<size_t>(i)] = o.beta * y0_of(i);
        }

        for(int64_t s = 0; s < a.m; ++s)
        {
            for(int64_t k = a.ptr[static_cast<size_t>(s)]; k < a.ptr[static_cast<size_t>(s) + 1];
                ++k)
            {
                const int64_t t = a.ind[static_cast<size_t>(k)];
                const int64_t r = o.as_csc ? t : s;
                const int64_t c = o.as_csc ? s : t;
                const cd      v = val[static_cast<size_t>(k)];
                switch(o.compute_op)
                {
                case rocsparse_operation_none:
                {
                    y[static_cast<size_t>(r)] += v * x_of(c);
                    if(o.symmetric && r != c)
                    {
                        y[static_cast<size_t>(c)] += v * x_of(r);
                    }
                    break;
                }
                case rocsparse_operation_transpose:
                {
                    y[static_cast<size_t>(c)] += v * x_of(r);
                    break;
                }
                case rocsparse_operation_conjugate_transpose:
                {
                    y[static_cast<size_t>(c)] += std::conj(v) * x_of(r);
                    break;
                }
                }
            }
        }
        return y;
    }

    bool same_device_bytes(const void* a, const void* b, size_t bytes)
    {
        std::vector<char> ha(bytes);
        std::vector<char> hb(bytes);
        return hipMemcpy(ha.data(), a, bytes, hipMemcpyDeviceToHost) == hipSuccess
               && hipMemcpy(hb.data(), b, bytes, hipMemcpyDeviceToHost) == hipSuccess
               && std::memcmp(ha.data(), hb.data(), bytes) == 0;
    }

    void inspect(rocsparse_csrmv_info info, Result& out)
    {
        if(info == nullptr)
        {
            return;
        }
        out.has_info        = true;
        out.adaptive_blocks = info->adaptive.row_blocks != nullptr;
        out.selected        = info->alg;
        out.nnzsplit_tuned
            = info->nnzsplit.get_starting_ids() != nullptr && info->nnzsplit.block_size != 0;
        if(!out.nnzsplit_tuned)
        {
            return;
        }

        rocsparse_csrmv_info copied = new _rocsparse_csrmv_info();
        out.copy_ok = rocsparse::copy_csrmv_info(copied, info) == rocsparse_status_success
                      && copied->nnzsplit.get_starting_ids() != nullptr
                      && copied->nnzsplit.get_starting_ids() != info->nnzsplit.get_starting_ids()
                      && copied->nnzsplit.get_size() == info->nnzsplit.get_size()
                      && copied->nnzsplit.get_use_starting_block_ids()
                             == info->nnzsplit.get_use_starting_block_ids()
                      && copied->nnzsplit.get_indextype() == info->nnzsplit.get_indextype()
                      && copied->nnzsplit.block_size == info->nnzsplit.block_size
                      && copied->nnzsplit.nnz_per_thread == info->nnzsplit.nnz_per_thread
                      && copied->nnzsplit.get_size_in_bytes() == info->nnzsplit.get_size_in_bytes()
                      && copied->alg == info->alg && copied->adaptive.row_blocks == nullptr
                      && same_device_bytes(copied->nnzsplit.get_starting_ids(),
                                           info->nnzsplit.get_starting_ids(),
                                           info->nnzsplit.get_size_in_bytes());
        delete copied;
    }

    // A is the matrix and x value type, Y the output type and T the compute type.
    template <typename I, typename J, typename A, typename Y, typename T = Y>
    Result run(rocsparse_handle handle, const Pattern& a, const Options& o)
    {
        Result out;

        auto require = [](rocsparse_status status) {
            EXPECT_EQ(status, rocsparse_status_success);
            return status == rocsparse_status_success;
        };

        const int64_t  b = (o.base == rocsparse_index_base_one) ? 1 : 0;
        std::vector<I> h_ptr(a.ptr.size());
        std::vector<J> h_ind(a.ind.size());
        std::vector<A> h_val(a.ind.size());
        for(size_t i = 0; i < a.ptr.size(); ++i)
        {
            h_ptr[i] = static_cast<I>(a.ptr[i] + b);
        }
        for(size_t k = 0; k < a.ind.size(); ++k)
        {
            h_ind[k] = static_cast<J>(a.ind[k] + b);
            h_val[k] = to_type<A>(value_of(static_cast<int64_t>(k)));
        }

        const int64_t rows = o.as_csc ? a.n : a.m;
        const int64_t cols = o.as_csc ? a.m : a.n;
        auto          xlen = [&](rocsparse_operation op) {
            return (op == rocsparse_operation_none) ? cols : rows;
        };
        auto ylen = [&](rocsparse_operation op) {
            return (op == rocsparse_operation_none) ? rows : cols;
        };
        auto vec = [](int64_t len, cd (*f)(int64_t)) {
            std::vector<Y> v(static_cast<size_t>(len));
            for(int64_t i = 0; i < len; ++i)
            {
                v[static_cast<size_t>(i)] = to_type<Y>(f(i));
            }
            return v;
        };
        auto xvec = [&](int64_t len) {
            std::vector<A> v(static_cast<size_t>(len));
            for(int64_t i = 0; i < len; ++i)
            {
                v[static_cast<size_t>(i)] = to_type<A>(x_of(i));
            }
            return v;
        };

        device_vector<I> d_ptr{h_ptr};
        device_vector<J> d_ind{h_ind};
        device_vector<A> d_val{h_val};
        device_vector<I> d_ptr2{h_ptr};
        device_vector<J> d_ind2{h_ind};
        device_vector<A> d_val2{h_val};
        device_vector<A> d_x{xvec(xlen(o.op))};
        device_vector<Y> d_y{vec(ylen(o.op), y0_of)};
        device_vector<A> d_cx{xvec(xlen(o.compute_op))};
        device_vector<Y> d_cy{vec(ylen(o.compute_op), y0_of)};
        if(!d_ptr.ptr || !d_ind.ptr || !d_val.ptr || !d_ptr2.ptr || !d_ind2.ptr || !d_val2.ptr
           || !d_x.ptr || !d_y.ptr || !d_cx.ptr || !d_cy.ptr)
        {
            ADD_FAILURE() << "device allocation failed";
            return out;
        }

        rocsparse_spmat_descr  mat    = nullptr;
        const rocsparse_status create = o.as_csc ? rocsparse_create_csc_descr(&mat,
                                                                              rows,
                                                                              cols,
                                                                              a.nnz(),
                                                                              d_ptr.ptr,
                                                                              d_ind.ptr,
                                                                              d_val.ptr,
                                                                              it_of<I>(),
                                                                              it_of<J>(),
                                                                              o.base,
                                                                              dt_of<A>())
                                                 : rocsparse_create_csr_descr(&mat,
                                                                              rows,
                                                                              cols,
                                                                              a.nnz(),
                                                                              d_ptr.ptr,
                                                                              d_ind.ptr,
                                                                              d_val.ptr,
                                                                              it_of<I>(),
                                                                              it_of<J>(),
                                                                              o.base,
                                                                              dt_of<A>());
        if(!require(create))
        {
            return out;
        }
        if(o.symmetric)
        {
            const rocsparse_matrix_type type = rocsparse_matrix_type_symmetric;
            const rocsparse_fill_mode   fill = rocsparse_fill_mode_lower;
            EXPECT_EQ(rocsparse_spmat_set_attribute(
                          mat, rocsparse_spmat_matrix_type, &type, sizeof(type)),
                      rocsparse_status_success);
            EXPECT_EQ(
                rocsparse_spmat_set_attribute(mat, rocsparse_spmat_fill_mode, &fill, sizeof(fill)),
                rocsparse_status_success);
        }

        rocsparse_dnvec_descr dx  = nullptr;
        rocsparse_dnvec_descr dy  = nullptr;
        rocsparse_dnvec_descr dcx = nullptr;
        rocsparse_dnvec_descr dcy = nullptr;
        EXPECT_EQ(rocsparse_create_dnvec_descr(&dx, xlen(o.op), d_x.ptr, dt_of<A>()),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_create_dnvec_descr(&dy, ylen(o.op), d_y.ptr, dt_of<Y>()),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_create_dnvec_descr(&dcx, xlen(o.compute_op), d_cx.ptr, dt_of<A>()),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_create_dnvec_descr(&dcy, ylen(o.compute_op), d_cy.ptr, dt_of<Y>()),
                  rocsparse_status_success);

        const T alpha = to_type<T>(cd(1.0, 0.0));
        const T beta  = to_type<T>(cd(o.beta, 0.0));

        auto repoint = [&]() {
            if(o.repoint)
            {
                EXPECT_EQ(o.as_csc
                              ? rocsparse_csc_set_pointers(mat, d_ptr2.ptr, d_ind2.ptr, d_val2.ptr)
                              : rocsparse_csr_set_pointers(mat, d_ptr2.ptr, d_ind2.ptr, d_val2.ptr),
                          rocsparse_status_success);
            }
        };

        if(o.v2)
        {
            rocsparse_spmv_descr descr   = nullptr;
            rocsparse_error      error[] = {nullptr};
            auto                 checked = [&](rocsparse_status status) {
                if(error[0] != nullptr)
                {
                    EXPECT_EQ(rocsparse_destroy_error(error[0]), rocsparse_status_success);
                    error[0] = nullptr;
                }
                return status;
            };
            EXPECT_EQ(rocsparse_create_spmv_descr(&descr), rocsparse_status_success);
            const rocsparse_datatype ytype = dt_of<T>();
            EXPECT_EQ(checked(rocsparse_spmv_set_input(
                          handle, descr, rocsparse_spmv_input_alg, &o.alg, sizeof(o.alg), error)),
                      rocsparse_status_success);
            EXPECT_EQ(
                checked(rocsparse_spmv_set_input(
                    handle, descr, rocsparse_spmv_input_operation, &o.op, sizeof(o.op), error)),
                rocsparse_status_success);
            EXPECT_EQ(checked(rocsparse_spmv_set_input(handle,
                                                       descr,
                                                       rocsparse_spmv_input_scalar_datatype,
                                                       &ytype,
                                                       sizeof(ytype),
                                                       error)),
                      rocsparse_status_success);
            EXPECT_EQ(checked(rocsparse_spmv_set_input(handle,
                                                       descr,
                                                       rocsparse_spmv_input_compute_datatype,
                                                       &ytype,
                                                       sizeof(ytype),
                                                       error)),
                      rocsparse_status_success);

            auto stage = [&](rocsparse_v2_spmv_stage s) {
                size_t size = 0;
                EXPECT_EQ(checked(rocsparse_v2_spmv_buffer_size(
                              handle, descr, mat, dcx, dcy, s, &size, error)),
                          rocsparse_status_success);
                void* buffer = nullptr;
                if(size > 0)
                {
                    EXPECT_EQ(hipMalloc(&buffer, size), hipSuccess);
                }
                const rocsparse_status status = checked(rocsparse_v2_spmv(
                    handle, descr, &alpha, mat, dcx, &beta, dcy, s, size, buffer, error));
                if(buffer != nullptr)
                {
                    EXPECT_EQ(hipFree(buffer), hipSuccess);
                }
                return status;
            };
            EXPECT_EQ(stage(rocsparse_v2_spmv_stage_analysis), rocsparse_status_success);
            repoint();
            out.compute_status = stage(rocsparse_v2_spmv_stage_compute);
            EXPECT_EQ(rocsparse_destroy_spmv_descr(descr), rocsparse_status_success);
        }
        else
        {
            auto spmv = [&](rocsparse_operation   op,
                            rocsparse_spmv_alg    alg,
                            rocsparse_dnvec_descr vx,
                            rocsparse_dnvec_descr vy,
                            rocsparse_spmv_stage  s,
                            size_t*               size,
                            void*                 buffer) {
                return rocsparse_spmv(
                    handle, op, &alpha, mat, vx, &beta, vy, dt_of<T>(), alg, s, size, buffer);
            };

            size_t size = 0;
            EXPECT_EQ(spmv(o.op, o.alg, dx, dy, rocsparse_spmv_stage_buffer_size, &size, nullptr),
                      rocsparse_status_success);
            size_t compute_size = 0;
            EXPECT_EQ(spmv(o.compute_op,
                           o.compute_alg,
                           dcx,
                           dcy,
                           rocsparse_spmv_stage_buffer_size,
                           &compute_size,
                           nullptr),
                      rocsparse_status_success);
            size         = std::max(size, compute_size);
            void* buffer = nullptr;
            EXPECT_EQ(hipMalloc(&buffer, std::max<size_t>(size, 1)), hipSuccess);

            if(require(spmv(o.op, o.alg, dx, dy, rocsparse_spmv_stage_preprocess, &size, buffer)))
            {
                inspect(mat->info->get_csrmv_info(), out);
                repoint();
                out.compute_status = spmv(o.compute_op,
                                          o.compute_alg,
                                          dcx,
                                          dcy,
                                          rocsparse_spmv_stage_compute,
                                          &size,
                                          buffer);
            }
            EXPECT_EQ(hipFree(buffer), hipSuccess);
        }

        if(out.compute_status == rocsparse_status_success)
        {
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
            const std::vector<Y> y = to_host(d_cy.ptr, d_cy.n);
            std::vector<cd>      val(h_val.size());
            for(size_t k = 0; k < h_val.size(); ++k)
            {
                val[k] = to_cd(h_val[k]);
            }
            const std::vector<cd> ref = reference(a, val, o);
            EXPECT_EQ(y.size(), ref.size());
            size_t mismatches = 0;
            for(size_t i = 0; i < std::min(y.size(), ref.size()); ++i)
            {
                const cd     got = to_cd(y[i]);
                const double tol = 1e-4 * std::max(1.0, std::abs(ref[i]));
                if(std::abs(got.real() - ref[i].real()) > tol
                   || std::abs(got.imag() - ref[i].imag()) > tol)
                {
                    if(mismatches++ < 5)
                    {
                        ADD_FAILURE() << "y[" << i << "] = " << got << ", expected " << ref[i];
                    }
                }
            }
            EXPECT_EQ(mismatches, 0u);
        }

        EXPECT_EQ(rocsparse_destroy_dnvec_descr(dx), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_dnvec_descr(dy), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_dnvec_descr(dcx), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_dnvec_descr(dcy), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
        return out;
    }

    Result run_f32(rocsparse_handle handle, const Pattern& a, const Options& o)
    {
        return run<int32_t, int32_t, float, float>(handle, a, o);
    }

    void expect_nnzsplit(const Result& got)
    {
        EXPECT_EQ(got.compute_status, rocsparse_status_success);
        EXPECT_TRUE(got.has_info);
        EXPECT_EQ(got.selected, rocsparse::csrmv_alg_nnzsplit);
        EXPECT_TRUE(got.nnzsplit_tuned);
        EXPECT_TRUE(got.copy_ok);
        EXPECT_FALSE(got.adaptive_blocks);
    }

    void expect_adaptive(const Result& got, rocsparse::csrmv_alg selected)
    {
        EXPECT_EQ(got.compute_status, rocsparse_status_success);
        EXPECT_TRUE(got.has_info);
        EXPECT_EQ(got.selected, selected);
        EXPECT_TRUE(got.adaptive_blocks);
        EXPECT_FALSE(got.nnzsplit_tuned);
    }
}

class CsrmvDefaultDispatch : public HandleTest
{
};

TEST_F(CsrmvDefaultDispatch, default_selects_nnzsplit_for_internal_empty_gap)
{
    expect_nnzsplit(run_f32(handle, scattered_gap(), Options{}));
}

TEST_F(CsrmvDefaultDispatch, explicit_adaptive_is_not_replaced)
{
    expect_adaptive(run_f32(handle, scattered_gap(), with_alg(rocsparse_spmv_alg_csr_adaptive)),
                    rocsparse::csrmv_alg_default);
}

TEST_F(CsrmvDefaultDispatch, default_keeps_adaptive_for_populated_rows)
{
    expect_adaptive(run_f32(handle, populated(), Options{}), rocsparse::csrmv_alg_adaptive);
}

TEST_F(CsrmvDefaultDispatch, default_selects_nnzsplit_for_alternating_row_lengths)
{
    expect_nnzsplit(run_f32(handle, alternating(), Options{}));
}

TEST_F(CsrmvDefaultDispatch, default_keeps_adaptive_when_empty_suffix_is_skipped)
{
    expect_adaptive(run_f32(handle, long_then_empty(), Options{}), rocsparse::csrmv_alg_adaptive);
}

TEST_F(CsrmvDefaultDispatch, default_selects_nnzsplit_for_empty_rows_between_long_rows)
{
    expect_nnzsplit(run_f32(handle, long_and_empty_rows(), Options{}));
}

TEST_F(CsrmvDefaultDispatch, default_keeps_adaptive_for_short_empty_prefix)
{
    expect_adaptive(run_f32(handle, short_empty_prefix(), Options{}),
                    rocsparse::csrmv_alg_adaptive);
}

TEST_F(CsrmvDefaultDispatch, switched_default_with_int64_indices)
{
    expect_nnzsplit(run<int64_t, int64_t, float, float>(handle, scattered_gap(), Options{}));
    expect_nnzsplit(run<int64_t, int32_t, float, float>(handle, alternating(), Options{}));
}

TEST_F(CsrmvDefaultDispatch, switched_default_with_base_one)
{
    Options o;
    o.base = rocsparse_index_base_one;
    expect_nnzsplit(run_f32(handle, scattered_gap(), o));
    expect_nnzsplit(run_f32(handle, alternating(), o));
}

TEST_F(CsrmvDefaultDispatch, switched_default_with_beta)
{
    Options o;
    o.beta = 2.0;
    expect_nnzsplit(run_f32(handle, scattered_gap(), o));
    expect_nnzsplit(run_f32(handle, alternating(), o));
}

TEST_F(CsrmvDefaultDispatch, switched_default_with_complex_values)
{
    Options o;
    o.beta = 0.5;
    expect_nnzsplit(run<int32_t, int32_t, rocsparse_float_complex, rocsparse_float_complex>(
        handle, alternating(), o));
    expect_nnzsplit(run<int64_t, int64_t, rocsparse_double_complex, rocsparse_double_complex>(
        handle, scattered_gap(), o));
}

TEST_F(CsrmvDefaultDispatch, switched_default_with_symmetric_matrix)
{
    Options o;
    o.symmetric = true;
    o.beta      = 1.0;
    expect_nnzsplit(run_f32(handle, scattered_gap_lower(), o));
    expect_nnzsplit(run<int32_t, int32_t, rocsparse_float_complex, rocsparse_float_complex>(
        handle, scattered_gap_lower(), o));
}

// The v2 analysis lives in the SpMV descriptor, which the test cannot read, so
// this checks y on matrices that switch with rocsparse_spmv.
TEST_F(CsrmvDefaultDispatch, switched_default_with_v2_spmv)
{
    Options o;
    o.v2   = true;
    o.beta = 1.0;
    EXPECT_EQ(run_f32(handle, scattered_gap(), o).compute_status, rocsparse_status_success);
    EXPECT_EQ(run_f32(handle, alternating(), o).compute_status, rocsparse_status_success);
    o.op         = rocsparse_operation_conjugate_transpose;
    o.compute_op = o.op;
    o.as_csc     = true;
    EXPECT_EQ((run<int64_t, int32_t, rocsparse_float_complex, rocsparse_float_complex>(
                   handle, alternating(), o)
                   .compute_status),
              rocsparse_status_success);
}

// Half-precision matrices keep adaptive whatever the output type.
TEST_F(CsrmvDefaultDispatch, half_precision_default_keeps_adaptive)
{
    expect_adaptive(run<int32_t, int32_t, _Float16, float>(handle, scattered_gap(), Options{}),
                    rocsparse::csrmv_alg_adaptive);
    expect_adaptive(
        run<int32_t, int32_t, _Float16, _Float16, float>(handle, scattered_gap(), Options{}),
        rocsparse::csrmv_alg_adaptive);
    expect_adaptive(run<int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float>(
                        handle, scattered_gap(), Options{}),
                    rocsparse::csrmv_alg_adaptive);
}

TEST_F(CsrmvDefaultDispatch, transposed_csc_default_uses_the_csr_rule)
{
    Options o = with_op(rocsparse_operation_transpose);
    o.as_csc  = true;
    o.beta    = 1.0;
    expect_nnzsplit(run_f32(handle, scattered_gap(), o));
    expect_nnzsplit(run_f32(handle, alternating(), o));
}

TEST_F(CsrmvDefaultDispatch, conjugate_transposed_csc_default_uses_the_csr_rule)
{
    Options o = with_op(rocsparse_operation_conjugate_transpose);
    o.as_csc  = true;
    expect_nnzsplit(run<int32_t, int32_t, rocsparse_float_complex, rocsparse_float_complex>(
        handle, alternating(), o));
    expect_nnzsplit(run<int64_t, int64_t, rocsparse_double_complex, rocsparse_double_complex>(
        handle, scattered_gap(), o));
}

// The selection is stored for the operation that was analysed. A transposed
// compute on the same descriptor uses rowsplit, which does not check the
// analysed pointers, so it succeeds after a repoint where a reused nnz-split
// analysis would return invalid_pointer.
TEST_F(CsrmvDefaultDispatch, preprocess_none_then_compute_transpose)
{
    Options o;
    o.repoint    = true;
    o.compute_op = rocsparse_operation_transpose;
    expect_nnzsplit(run_f32(handle, scattered_gap(), o));
    expect_nnzsplit(run_f32(handle, alternating(), o));
    o.compute_op = rocsparse_operation_conjugate_transpose;
    expect_nnzsplit(run<int32_t, int32_t, rocsparse_float_complex, rocsparse_float_complex>(
        handle, alternating(), o));
}

// nnz-split compute checks the analysed pointers, as adaptive compute does.
TEST_F(CsrmvDefaultDispatch, switched_default_rejects_changed_pointers)
{
    Options o;
    o.repoint = true;
    EXPECT_EQ(run_f32(handle, scattered_gap(), o).compute_status, rocsparse_status_invalid_pointer);
    EXPECT_EQ(run_f32(handle, populated(), o).compute_status, rocsparse_status_invalid_pointer);
    o.alg         = rocsparse_spmv_alg_csr_nnzsplit;
    o.compute_alg = o.alg;
    EXPECT_EQ(run_f32(handle, populated(), o).compute_status, rocsparse_status_invalid_pointer);
}

// An nnz-split analysis has no row blocks for an adaptive compute to launch.
TEST_F(CsrmvDefaultDispatch, adaptive_compute_rejects_switched_analysis)
{
    Options o;
    o.compute_alg    = rocsparse_spmv_alg_csr_adaptive;
    const Result got = run_f32(handle, scattered_gap(), o);
    EXPECT_EQ(got.selected, rocsparse::csrmv_alg_nnzsplit);
    EXPECT_EQ(got.compute_status, rocsparse_status_invalid_value);
}
