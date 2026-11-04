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
// Forced-clamp tests for the csrmv adaptive row-block launches.
//
// The adaptive kernels run one workgroup per row block. When the row-block
// count is above get_grid_size_x, csrmv splits the row blocks into consecutive
// launches of at most that many workgroups. The workgroups of a long row wait
// on a flag the row's first workgroup flips, so the split must stay correct
// when a long row starts in one launch and continues in the next.
//
// At real sizes the limit needs millions of row blocks. These tests shrink
// handle->properties.maxGridSize[0] instead, so a small matrix is split into
// launches of a few row blocks, and compare y against the unclamped result and
// a host reference. The row-block layout is read back from a csrmv analysis so
// that the limits include ones that put the first workgroup of a long row at
// the end of a launch, with the rest of the row in the next launch.
//
// EXACT ARITHMETIC. All values are small integers and alpha, beta are 2 and
// 1/2, so every product and partial sum is exact and the result does not
// depend on the order of the atomic adds; y is compared exactly.
//
#include "unit_test_utils.hpp"

#include "rocsparse.h"
#include "rocsparse_csrmv_info.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_mat_info.hpp"

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    // A CSR matrix on the host. Every entry is (1 + (row + col) % 3, (row + col) % 2),
    // the imaginary part being dropped for real types.
    struct HostCsr
    {
        int64_t              m = 0;
        int64_t              n = 0;
        std::vector<int64_t> row_ptr{0};
        std::vector<int64_t> col_ind;

        // Appends a row holding columns [first_col, first_col + len).
        void add_row(int64_t first_col, int64_t len)
        {
            for(int64_t k = 0; k < len; ++k)
            {
                col_ind.push_back(first_col + k);
            }
            row_ptr.push_back(static_cast<int64_t>(col_ind.size()));
            ++m;
        }

        int64_t nnz() const
        {
            return static_cast<int64_t>(col_ind.size());
        }
    };

    std::complex<double> entry(int64_t row, int64_t col)
    {
        return {static_cast<double>(1 + (row + col) % 3), static_cast<double>((row + col) % 2)};
    }

    std::complex<double> x_entry(int64_t i)
    {
        return {static_cast<double>(i % 7 + 1), static_cast<double>(i % 3) - 1.0};
    }

    std::complex<double> y0_entry(int64_t i)
    {
        return {static_cast<double>(i % 5) - 2.0, static_cast<double>(i % 4)};
    }

    // Lower bidiagonal: short rows only.
    HostCsr short_rows()
    {
        HostCsr A;
        A.n = 16384;
        for(int64_t i = 0; i < A.n; ++i)
        {
            A.add_row((i > 0) ? i - 1 : 0, (i > 0) ? 2 : 1);
        }
        return A;
    }

    // Lower triangular: short rows with long rows (more than one workgroup each)
    // between them, two of them adjacent and one the last row.
    HostCsr mixed_rows()
    {
        const struct
        {
            int64_t row;
            int64_t len;
        } long_rows[] = {{8000, 7000}, {8001, 4000}, {14000, 13000}, {19999, 9000}};

        HostCsr A;
        A.n = 20000;
        for(int64_t i = 0; i < A.n; ++i)
        {
            int64_t len = (i > 0) ? 2 : 1;
            for(const auto& r : long_rows)
            {
                if(r.row == i)
                {
                    len = r.len;
                }
            }
            A.add_row(i - len + 1, len);
        }
        return A;
    }

    // Rectangular with dense rows, so the general path picks its smaller
    // workgroup; every fifth row needs more than one workgroup.
    HostCsr dense_rows()
    {
        HostCsr A;
        A.n = 8192;
        for(int64_t i = 0; i < 512; ++i)
        {
            const int64_t len = (i % 5 == 0) ? 7000 : 40;
            A.add_row((i * 37) % (A.n - len + 1), len);
        }
        return A;
    }

    // y = alpha * A * x + beta * y0, where a symmetric A also applies the
    // transpose of every stored off-diagonal entry.
    std::vector<std::complex<double>> host_reference(const HostCsr&       A,
                                                     bool                 symmetric,
                                                     std::complex<double> alpha,
                                                     std::complex<double> beta,
                                                     bool                 complex_values)
    {
        auto val = [&](int64_t row, int64_t col) {
            const std::complex<double> v = entry(row, col);
            return complex_values ? v : std::complex<double>(v.real(), 0.0);
        };
        auto xv = [&](int64_t i) {
            const std::complex<double> v = x_entry(i);
            return complex_values ? v : std::complex<double>(v.real(), 0.0);
        };
        auto y0v = [&](int64_t i) {
            const std::complex<double> v = y0_entry(i);
            return complex_values ? v : std::complex<double>(v.real(), 0.0);
        };

        std::vector<std::complex<double>> ax(A.m, 0.0);
        for(int64_t i = 0; i < A.m; ++i)
        {
            for(int64_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
            {
                const int64_t j = A.col_ind[k];
                ax[i] += val(i, j) * xv(j);
                if(symmetric && j != i)
                {
                    ax[j] += val(i, j) * xv(i);
                }
            }
        }

        std::vector<std::complex<double>> y(A.m);
        for(int64_t i = 0; i < A.m; ++i)
        {
            y[i] = alpha * ax[i] + beta * y0v(i);
        }
        return y;
    }

    template <typename T>
    T make_value(std::complex<double> v)
    {
        return static_cast<T>(v.real());
    }
    template <>
    rocsparse_float_complex make_value<rocsparse_float_complex>(std::complex<double> v)
    {
        return rocsparse_float_complex(static_cast<float>(v.real()), static_cast<float>(v.imag()));
    }
    template <>
    rocsparse_double_complex make_value<rocsparse_double_complex>(std::complex<double> v)
    {
        return rocsparse_double_complex(v.real(), v.imag());
    }

    std::complex<double> to_complex(float v)
    {
        return {v, 0.0};
    }
    std::complex<double> to_complex(double v)
    {
        return {v, 0.0};
    }
    std::complex<double> to_complex(rocsparse_float_complex v)
    {
        return {std::real(v), std::imag(v)};
    }
    std::complex<double> to_complex(rocsparse_double_complex v)
    {
        return {std::real(v), std::imag(v)};
    }

    // Restores maxGridSize[0] even when an assertion returns early.
    struct GridLimitGuard
    {
        rocsparse_handle handle;
        int              saved;

        explicit GridLimitGuard(rocsparse_handle h)
            : handle(h)
            , saved(h->properties.maxGridSize[0])
        {
        }
        ~GridLimitGuard()
        {
            handle->properties.maxGridSize[0] = saved;
        }
        void set(int64_t limit)
        {
            handle->properties.maxGridSize[0] = static_cast<int>(limit);
        }
        void restore()
        {
            handle->properties.maxGridSize[0] = saved;
        }
    };

    // A long row runs on `count` consecutive row blocks starting at `first`.
    struct LongRow
    {
        int64_t first;
        int64_t count;
    };

    struct Layout
    {
        int64_t              nblocks = 0;
        std::vector<LongRow> long_rows;
    };

    // The row-block layout depends only on the row pointer, so it is read back
    // from a 32-bit csrmv analysis of the same pattern. A row block that starts
    // and stops at the same row is a workgroup of a long row other than its last.
    void learn_layout(rocsparse_handle handle, const HostCsr& A, Layout& layout)
    {
        std::vector<rocsparse_int> row_ptr(A.row_ptr.begin(), A.row_ptr.end());
        std::vector<rocsparse_int> col_ind(A.col_ind.begin(), A.col_ind.end());
        std::vector<float>         val(A.nnz(), 1.0f);

        device_vector<rocsparse_int> d_row_ptr{row_ptr};
        device_vector<rocsparse_int> d_col_ind{col_ind};
        device_vector<float>         d_val{val};
        ASSERT_TRUE(d_row_ptr.ptr && d_col_ind.ptr && d_val.ptr);

        rocsparse_mat_descr descr = nullptr;
        rocsparse_mat_info  info  = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);
        ASSERT_EQ(rocsparse_scsrmv_analysis(handle,
                                            rocsparse_operation_none,
                                            static_cast<rocsparse_int>(A.m),
                                            static_cast<rocsparse_int>(A.n),
                                            static_cast<rocsparse_int>(A.nnz()),
                                            descr,
                                            d_val,
                                            d_row_ptr,
                                            d_col_ind,
                                            info),
                  rocsparse_status_success);
        ASSERT_NE(info->get_csrmv_info(), nullptr);

        const size_t               size       = info->get_csrmv_info()->adaptive.size;
        std::vector<rocsparse_int> row_blocks = to_host(
            static_cast<const rocsparse_int*>(info->get_csrmv_info()->adaptive.row_blocks), size);

        layout.nblocks = static_cast<int64_t>(size) - 1;
        for(int64_t g = 0; g < layout.nblocks; ++g)
        {
            if(row_blocks[g] == row_blocks[g + 1])
            {
                if(layout.long_rows.empty()
                   || layout.long_rows.back().first + layout.long_rows.back().count - 1 != g)
                {
                    layout.long_rows.push_back({g, 1});
                }
                ++layout.long_rows.back().count;
            }
        }

        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    // Small limits, plus every limit that ends a launch exactly on the first
    // workgroup of a long row, and one that leaves only the last row block for
    // a second launch.
    std::vector<int64_t> limits_for(const Layout& layout)
    {
        std::set<int64_t> limits = {1, 2, 3, 5, 7, layout.nblocks - 1};
        for(const LongRow& r : layout.long_rows)
        {
            for(int64_t l = 1; l <= r.first + 1; ++l)
            {
                if((r.first + 1) % l == 0)
                {
                    limits.insert(l);
                }
            }
            limits.insert(r.first);
        }

        std::vector<int64_t> out;
        for(int64_t l : limits)
        {
            if(l >= 1 && l < layout.nblocks)
            {
                out.push_back(l);
            }
        }
        return out;
    }

    // True if some limit puts the first workgroup of a long row at the end of a
    // launch other than the first, with the rest of the row in the next launch.
    bool straddles_later_launch(const Layout& layout, const std::vector<int64_t>& limits)
    {
        for(int64_t l : limits)
        {
            for(const LongRow& r : layout.long_rows)
            {
                if(r.first % l == l - 1 && r.first / l >= 1)
                {
                    return true;
                }
            }
        }
        return false;
    }

    template <typename T_, typename I_, typename J_>
    struct Config
    {
        using T = T_;
        using I = I_;
        using J = J_;
    };

    template <typename C>
    class CsrmvAdaptiveLimit : public HandleTest
    {
    protected:
        using T = typename C::T;
        using I = typename C::I;
        using J = typename C::J;

        void run(const HostCsr& A, rocsparse_matrix_type type, bool long_rows)
        {
            const bool symmetric      = (type == rocsparse_matrix_type_symmetric);
            const bool complex_values = (dt_of<T>() == rocsparse_datatype_f32_c
                                         || dt_of<T>() == rocsparse_datatype_f64_c);

            Layout layout;
            ASSERT_NO_FATAL_FAILURE(learn_layout(handle, A, layout));
            ASSERT_GE(layout.nblocks, 8);
            ASSERT_EQ(layout.long_rows.empty(), !long_rows);
            const std::vector<int64_t> limits = limits_for(layout);
            if(long_rows)
            {
                ASSERT_TRUE(straddles_later_launch(layout, limits));
            }

            std::vector<I> row_ptr(A.row_ptr.begin(), A.row_ptr.end());
            std::vector<J> col_ind(A.col_ind.begin(), A.col_ind.end());
            std::vector<T> val(A.nnz());
            for(int64_t i = 0; i < A.m; ++i)
            {
                for(int64_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
                {
                    val[k] = make_value<T>(entry(i, A.col_ind[k]));
                }
            }
            std::vector<T> hx(A.n), hy0(A.m);
            for(int64_t i = 0; i < A.n; ++i)
            {
                hx[i] = make_value<T>(x_entry(i));
            }
            for(int64_t i = 0; i < A.m; ++i)
            {
                hy0[i] = make_value<T>(y0_entry(i));
            }

            device_vector<I> d_row_ptr{row_ptr};
            device_vector<J> d_col_ind{col_ind};
            device_vector<T> d_val{val};
            device_vector<T> d_x{hx};
            device_vector<T> d_y{hy0};
            ASSERT_TRUE(d_row_ptr.ptr && d_col_ind.ptr && d_val.ptr && d_x.ptr && d_y.ptr);

            rocsparse_spmat_descr mat  = nullptr;
            rocsparse_dnvec_descr vecx = nullptr;
            rocsparse_dnvec_descr vecy = nullptr;
            rocsparse_spmv_descr  spmv = nullptr;
            ASSERT_EQ(rocsparse_create_csr_descr(&mat,
                                                 A.m,
                                                 A.n,
                                                 A.nnz(),
                                                 d_row_ptr,
                                                 d_col_ind,
                                                 d_val,
                                                 it_of<I>(),
                                                 it_of<J>(),
                                                 rocsparse_index_base_zero,
                                                 dt_of<T>()),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_create_dnvec_descr(&vecx, A.n, d_x, dt_of<T>()),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_create_dnvec_descr(&vecy, A.m, d_y, dt_of<T>()),
                      rocsparse_status_success);
            if(symmetric)
            {
                const rocsparse_fill_mode fill = rocsparse_fill_mode_lower;
                ASSERT_EQ(rocsparse_spmat_set_attribute(
                              mat, rocsparse_spmat_matrix_type, &type, sizeof(type)),
                          rocsparse_status_success);
                ASSERT_EQ(rocsparse_spmat_set_attribute(
                              mat, rocsparse_spmat_fill_mode, &fill, sizeof(fill)),
                          rocsparse_status_success);
            }

            ASSERT_EQ(rocsparse_create_spmv_descr(&spmv), rocsparse_status_success);
            const rocsparse_spmv_alg  alg       = rocsparse_spmv_alg_csr_adaptive;
            const rocsparse_operation operation = rocsparse_operation_none;
            const rocsparse_datatype  datatype  = dt_of<T>();
            ASSERT_EQ(rocsparse_spmv_set_input(
                          handle, spmv, rocsparse_spmv_input_alg, &alg, sizeof(alg), nullptr),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_spmv_set_input(handle,
                                               spmv,
                                               rocsparse_spmv_input_operation,
                                               &operation,
                                               sizeof(operation),
                                               nullptr),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_spmv_set_input(handle,
                                               spmv,
                                               rocsparse_spmv_input_scalar_datatype,
                                               &datatype,
                                               sizeof(datatype),
                                               nullptr),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_spmv_set_input(handle,
                                               spmv,
                                               rocsparse_spmv_input_compute_datatype,
                                               &datatype,
                                               sizeof(datatype),
                                               nullptr),
                      rocsparse_status_success);

            const T alpha = make_value<T>(2.0);
            const T beta  = make_value<T>(0.5);

            auto spmv_stage = [&](rocsparse_v2_spmv_stage stage) {
                size_t buffer_size = 0;
                EXPECT_EQ(rocsparse_v2_spmv_buffer_size(
                              handle, spmv, mat, vecx, vecy, stage, &buffer_size, nullptr),
                          rocsparse_status_success);
                device_vector<char> buffer(buffer_size ? buffer_size : size_t(1));
                return rocsparse_v2_spmv(handle,
                                         spmv,
                                         &alpha,
                                         mat,
                                         vecx,
                                         &beta,
                                         vecy,
                                         stage,
                                         buffer_size,
                                         buffer_size ? buffer.ptr : nullptr,
                                         nullptr);
            };
            auto reset_y = [&]() {
                return hipMemcpy(d_y.ptr, hy0.data(), sizeof(T) * A.m, hipMemcpyHostToDevice);
            };

            GridLimitGuard guard(handle);

            // Analysis launches nothing and accepts any row-block count.
            guard.set(1);
            ASSERT_EQ(spmv_stage(rocsparse_v2_spmv_stage_analysis), rocsparse_status_success);
            guard.restore();

            // Unclamped: one launch, checked against the host.
            ASSERT_EQ(spmv_stage(rocsparse_v2_spmv_stage_compute), rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            const std::vector<T> unclamped = to_host(d_y);

            const std::vector<std::complex<double>> ref
                = host_reference(A, symmetric, 2.0, 0.5, complex_values);
            for(int64_t i = 0; i < A.m; ++i)
            {
                ASSERT_EQ(to_complex(unclamped[i]), ref[i]) << "row " << i;
            }

            // Clamped: the row blocks run over several launches.
            for(int64_t limit : limits)
            {
                ASSERT_EQ(reset_y(), hipSuccess);
                guard.set(limit);
                ASSERT_EQ(spmv_stage(rocsparse_v2_spmv_stage_compute), rocsparse_status_success);
                guard.restore();
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

                const std::vector<T> clamped = to_host(d_y);
                for(int64_t i = 0; i < A.m; ++i)
                {
                    ASSERT_EQ(to_complex(clamped[i]), to_complex(unclamped[i]))
                        << "limit " << limit << " of " << layout.nblocks << " row blocks, row "
                        << i;
                }
            }

            EXPECT_EQ(rocsparse_destroy_spmv_descr(spmv), rocsparse_status_success);
            EXPECT_EQ(rocsparse_destroy_dnvec_descr(vecy), rocsparse_status_success);
            EXPECT_EQ(rocsparse_destroy_dnvec_descr(vecx), rocsparse_status_success);
            EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
        }
    };

    using Configs = ::testing::Types<Config<float, int32_t, int32_t>,
                                     Config<float, int64_t, int32_t>,
                                     Config<float, int64_t, int64_t>,
                                     Config<double, int32_t, int32_t>,
                                     Config<double, int64_t, int32_t>,
                                     Config<double, int64_t, int64_t>,
                                     Config<rocsparse_float_complex, int32_t, int32_t>,
                                     Config<rocsparse_float_complex, int64_t, int32_t>,
                                     Config<rocsparse_float_complex, int64_t, int64_t>,
                                     Config<rocsparse_double_complex, int32_t, int32_t>,
                                     Config<rocsparse_double_complex, int64_t, int32_t>,
                                     Config<rocsparse_double_complex, int64_t, int64_t>>;
}

TYPED_TEST_SUITE(CsrmvAdaptiveLimit, Configs);

TYPED_TEST(CsrmvAdaptiveLimit, short_rows_general)
{
    this->run(short_rows(), rocsparse_matrix_type_general, false);
}

TYPED_TEST(CsrmvAdaptiveLimit, short_rows_symmetric)
{
    this->run(short_rows(), rocsparse_matrix_type_symmetric, false);
}

TYPED_TEST(CsrmvAdaptiveLimit, mixed_rows_general)
{
    this->run(mixed_rows(), rocsparse_matrix_type_general, true);
}

TYPED_TEST(CsrmvAdaptiveLimit, mixed_rows_symmetric)
{
    this->run(mixed_rows(), rocsparse_matrix_type_symmetric, true);
}

TYPED_TEST(CsrmvAdaptiveLimit, dense_rows_general)
{
    this->run(dense_rows(), rocsparse_matrix_type_general, true);
}
