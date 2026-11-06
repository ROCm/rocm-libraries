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
// Device unit tests of the COO, COO AoS, ELL, sliced ELL and HYB matrix vector
// products (legacy routines and the generic rocsparse_spmv): every operation,
// scalars, pointer modes, index bases, very long rows and empty matrices.
// Every result is compared with a host product of the CSR matrix.
//
#include "unit_test_device_level2_common.hpp"

#include <string>

using namespace level2;

namespace
{
    enum class fmt
    {
        coo,
        coo_aos,
        ell,
        sell
    };

    inline const char* fmt_name(fmt f)
    {
        switch(f)
        {
        case fmt::coo:
            return "coo";
        case fmt::coo_aos:
            return "coo_aos";
        case fmt::ell:
            return "ell";
        case fmt::sell:
            return "sell";
        }
        return "?";
    }

    struct mv_case
    {
        fmt                 format = fmt::coo;
        rocsparse_operation trans  = rocsparse_operation_none;
        float               alpha  = 1.5f;
        float               beta   = -0.5f;
        bool                device = false;
        int                 base   = 0;
        rocsparse_spmv_alg  alg    = rocsparse_spmv_alg_default;
        int32_t             slice  = 32;
        // Run the stage that analyses the matrix as well.
        bool preprocess = true;
        // 64 bit indices.
        bool i64 = false;
    };

    // y = alpha * op(A) * x + beta * y through rocsparse_spmv, checked against the host.
    template <typename T>
    void run_generic(rocsparse_handle handle, const host_csr<T>& A, const mv_case& c)
    {
        SCOPED_TRACE(::testing::Message()
                     << fmt_name(c.format) << " trans=" << c.trans << " alpha=" << c.alpha
                     << " beta=" << c.beta << " device=" << c.device << " base=" << c.base
                     << " i64=" << c.i64 << " alg=" << c.alg << " slice=" << c.slice << " m=" << A.m
                     << " n=" << A.n);

        const int32_t m  = A.m;
        const int32_t n  = A.n;
        const size_t  xs = trans_size(c.trans, m, n, true);
        const size_t  ys = trans_size(c.trans, m, n, false);

        const T alpha = case_scalar<T>(c.alpha, 0.0f);
        const T beta  = case_scalar<T>(c.beta, 1.0f);

        const std::vector<T> x    = make_vector<T>(xs, 2);
        const std::vector<T> y0   = make_vector<T>(ys, 9);
        const std::vector<T> want = csr_gemv(A, c.trans, alpha, x, beta, y0);

        device_vector<T> dx(x), dy(y0), dalpha(std::vector<T>(1, alpha)),
            dbeta(std::vector<T>(1, beta));
        ASSERT_TRUE(dx.ptr && dy.ptr && dalpha.ptr && dbeta.ptr);

        const host_coo<T>  coo  = to_coo(A);
        const host_ell<T>  ell  = to_ell(A);
        const host_sell<T> sell = to_sell(A, c.slice);

        std::vector<int32_t> aos(2 * coo.row.size());
        for(size_t k = 0; k < coo.row.size(); ++k)
        {
            aos[2 * k]     = coo.row[k];
            aos[2 * k + 1] = coo.col[k];
        }

        // Index arrays in both 32 and 64 bit; the descriptor picks one.
        auto widen = [](const std::vector<int32_t>& v) {
            return std::vector<int64_t>(v.begin(), v.end());
        };
        device_vector<int32_t> drow(coo.row), dcol(coo.col), daos(aos), dell_col(ell.col),
            dsell_col(sell.col), dsell_off(sell.offsets);
        device_vector<int64_t> drow64(widen(coo.row)), dcol64(widen(coo.col)), daos64(widen(aos)),
            dell_col64(widen(ell.col)), dsell_col64(widen(sell.col)),
            dsell_off64(widen(sell.offsets));
        device_vector<T> dcoo_val(coo.val), dell_val(ell.val), dsell_val(sell.val);
        // Empty arrays still need a valid address for the descriptors.
        device_vector<int32_t> dummy_i(std::vector<int32_t>(2, 0));
        device_vector<int64_t> dummy_i64(std::vector<int64_t>(2, 0));
        device_vector<T>       dummy_t(std::vector<T>(2, T{}));

        auto pick = [&](bool empty, const auto& a32, const auto& a64) -> void* {
            if(empty)
                return c.i64 ? static_cast<void*>(dummy_i64.ptr) : static_cast<void*>(dummy_i.ptr);
            return c.i64 ? static_cast<void*>(a64.ptr) : static_cast<void*>(a32.ptr);
        };

        spmat_guard mat;
        const auto  it = c.i64 ? rocsparse_indextype_i64 : rocsparse_indextype_i32;
        const auto  dt = rocsparse_ut::dt_of<T>();
        const auto  ib = to_base(c.base);
        switch(c.format)
        {
        case fmt::coo:
            ASSERT_EQ(rocsparse_create_coo_descr(&mat.d,
                                                 m,
                                                 n,
                                                 coo.val.size(),
                                                 pick(coo.val.empty(), drow, drow64),
                                                 pick(coo.val.empty(), dcol, dcol64),
                                                 coo.val.empty() ? dummy_t.ptr : dcoo_val.ptr,
                                                 it,
                                                 ib,
                                                 dt),
                      rocsparse_status_success);
            break;
        case fmt::coo_aos:
            ASSERT_EQ(rocsparse_create_coo_aos_descr(&mat.d,
                                                     m,
                                                     n,
                                                     coo.val.size(),
                                                     pick(coo.val.empty(), daos, daos64),
                                                     coo.val.empty() ? dummy_t.ptr : dcoo_val.ptr,
                                                     it,
                                                     ib,
                                                     dt),
                      rocsparse_status_success);
            break;
        case fmt::ell:
            ASSERT_EQ(rocsparse_create_ell_descr(&mat.d,
                                                 m,
                                                 n,
                                                 pick(ell.col.empty(), dell_col, dell_col64),
                                                 ell.col.empty() ? dummy_t.ptr : dell_val.ptr,
                                                 ell.width,
                                                 it,
                                                 ib,
                                                 dt),
                      rocsparse_status_success);
            break;
        case fmt::sell:
            ASSERT_EQ(rocsparse_create_sell_descr(&mat.d,
                                                  m,
                                                  n,
                                                  coo.val.size(),
                                                  c.slice,
                                                  sell.colval,
                                                  pick(false, dsell_off, dsell_off64),
                                                  pick(sell.col.empty(), dsell_col, dsell_col64),
                                                  sell.col.empty() ? dummy_t.ptr : dsell_val.ptr,
                                                  it,
                                                  it,
                                                  ib,
                                                  dt),
                      rocsparse_status_success);
            break;
        }
        dnvec_guard vx, vy;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, xs, dx, dt), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, ys, dy, dt), rocsparse_status_success);

        scoped_pointer_mode mode(handle, c.device);
        const void*         palpha
            = c.device ? static_cast<const void*>(dalpha.ptr) : static_cast<const void*>(&alpha);
        const void* pbeta
            = c.device ? static_cast<const void*>(dbeta.ptr) : static_cast<const void*>(&beta);

        if(c.format == fmt::sell)
        {
            // Sliced ELL is only available through the v2 interface.
            spmv_descr_guard descr;
            ASSERT_TRUE(descr.d);
            ASSERT_EQ(set_basic_inputs(handle, descr.d, c.alg, c.trans, dt, dt),
                      rocsparse_status_success);
            run_v2_spmv(handle, descr.d, mat.d, vx.d, vy.d, palpha, pbeta);
            if(::testing::Test::HasFatalFailure())
                return;
            EXPECT_TRUE(expect_near(want, to_host(dy), "spmv y"));
            return;
        }

        auto call = [&](rocsparse_spmv_stage stage, size_t* bytes, void* buffer) {
            return rocsparse_spmv(
                handle, c.trans, palpha, mat.d, vx.d, pbeta, vy.d, dt, c.alg, stage, bytes, buffer);
        };
        size_t bytes = 0;
        ASSERT_EQ(call(rocsparse_spmv_stage_buffer_size, &bytes, nullptr),
                  rocsparse_status_success);
        device_vector<char> dbuf(std::max<size_t>(bytes, 1));
        ASSERT_TRUE(dbuf.ptr);
        if(c.preprocess)
            ASSERT_EQ(call(rocsparse_spmv_stage_preprocess, &bytes, dbuf),
                      rocsparse_status_success);
        ASSERT_EQ(call(rocsparse_spmv_stage_compute, &bytes, dbuf), rocsparse_status_success);
        EXPECT_TRUE(expect_near(want, to_host(dy), "spmv y"));
    }

    class L2DeviceFormats : public rocsparse_ut::HandleTest
    {
    protected:
        // Every operation with a cycle of scalars, pointer modes and bases; the
        // float sweep (`dense`) additionally crosses the scalars with the operation.
        template <typename T>
        void sweep(
            fmt f, const host_csr<T>& base0, bool dense, rocsparse_spmv_alg alg, int32_t slice)
        {
            static const float alphas[] = {1.5f, 0.0f, 1.0f, -1.0f};
            static const float betas[]  = {-0.5f, 0.0f, 1.0f, 2.0f};
            int                n        = 0;
            for(const auto trans : all_ops)
                for(int s = 0; s < (dense ? 16 : 4); ++s, ++n)
                {
                    mv_case c;
                    c.format      = f;
                    c.trans       = trans;
                    c.alpha       = alphas[s % 4];
                    c.beta        = betas[(dense ? s / 4 : s) % 4];
                    c.device      = (n % 3) == 1;
                    c.base        = (n / 2) % 2;
                    c.alg         = alg;
                    c.slice       = slice;
                    c.preprocess  = (n % 4) != 0;
                    c.i64         = (n % 4) == 2;
                    host_csr<T> A = base0;
                    rebase(A, c.base);
                    run_generic<T>(handle, A, c);
                    if(::testing::Test::HasFailure())
                        return;
                }
        }
    };

    TEST_F(L2DeviceFormats, coo_float)
    {
        sweep<float>(
            fmt::coo, make_matrix<float>(300, 250, 0, 0, 11), true, rocsparse_spmv_alg_coo, 32);
        sweep<float>(fmt::coo,
                     make_matrix<float>(300, 250, 0, 0, 11),
                     false,
                     rocsparse_spmv_alg_coo_atomic,
                     32);
    }

    TEST_F(L2DeviceFormats, coo_long_row_float)
    {
        // A row with more than 10 * 256 entries selects the looping atomic kernels.
        const auto A = make_matrix<float>(40, 3000, 0, 2800, 0);
        sweep<float>(fmt::coo, A, false, rocsparse_spmv_alg_coo_atomic, 32);
        sweep<float>(fmt::coo, A, false, rocsparse_spmv_alg_coo, 32);
    }

    TEST_F(L2DeviceFormats, coo_double_and_complex)
    {
        sweep<double>(fmt::coo,
                      make_matrix<double>(200, 300, 0, 0, 5),
                      false,
                      rocsparse_spmv_alg_default,
                      32);
        sweep<rocsparse_float_complex>(fmt::coo,
                                       make_matrix<rocsparse_float_complex>(150, 90, 0, 0, 0),
                                       false,
                                       rocsparse_spmv_alg_coo_atomic,
                                       32);
        sweep<rocsparse_double_complex>(fmt::coo,
                                        make_matrix<rocsparse_double_complex>(150, 90, 0, 0, 7),
                                        false,
                                        rocsparse_spmv_alg_coo,
                                        32);
    }

    TEST_F(L2DeviceFormats, coo_aos_float)
    {
        sweep<float>(fmt::coo_aos,
                     make_matrix<float>(300, 250, 0, 0, 11),
                     true,
                     rocsparse_spmv_alg_default,
                     32);
        sweep<float>(fmt::coo_aos,
                     make_matrix<float>(40, 3000, 0, 2800, 0),
                     false,
                     rocsparse_spmv_alg_default,
                     32);
    }

    TEST_F(L2DeviceFormats, coo_aos_double_and_complex)
    {
        sweep<double>(fmt::coo_aos,
                      make_matrix<double>(200, 300, 0, 0, 5),
                      false,
                      rocsparse_spmv_alg_default,
                      32);
        sweep<rocsparse_float_complex>(fmt::coo_aos,
                                       make_matrix<rocsparse_float_complex>(150, 90, 0, 0, 0),
                                       false,
                                       rocsparse_spmv_alg_default,
                                       32);
        sweep<rocsparse_double_complex>(fmt::coo_aos,
                                        make_matrix<rocsparse_double_complex>(150, 90, 0, 0, 7),
                                        false,
                                        rocsparse_spmv_alg_default,
                                        32);
    }

    TEST_F(L2DeviceFormats, ell_float)
    {
        sweep<float>(
            fmt::ell, make_matrix<float>(300, 250, 0, 0, 11), true, rocsparse_spmv_alg_ell, 32);
        sweep<float>(fmt::ell,
                     make_matrix<float>(40, 3000, 0, 2800, 0),
                     false,
                     rocsparse_spmv_alg_default,
                     32);
    }

    TEST_F(L2DeviceFormats, sell_float_slice_sizes)
    {
        // Slices up to 128 rows use the shared memory kernel, larger ones the large
        // slice kernel.
        const auto A = make_matrix<float>(600, 250, 0, 0, 11);
        for(const int32_t slice : {1, 4, 16, 32, 64, 128, 129, 256, 500})
        {
            sweep<float>(fmt::sell, A, false, rocsparse_spmv_alg_sell, slice);
            if(HasFailure())
                return;
        }
    }

    TEST_F(L2DeviceFormats, sell_dense_sweep_float)
    {
        sweep<float>(fmt::sell,
                     make_matrix<float>(300, 250, 0, 0, 11),
                     true,
                     rocsparse_spmv_alg_default,
                     32);
    }

    TEST_F(L2DeviceFormats, empty_matrices_scale_y)
    {
        // nnz == 0: the matrix is not read and y is scaled by beta.
        for(const auto f : {fmt::coo, fmt::coo_aos, fmt::sell})
            for(const auto trans : {rocsparse_operation_none, rocsparse_operation_transpose})
                for(const bool device : {false, true})
                {
                    mv_case c;
                    c.format = f;
                    c.trans  = trans;
                    c.device = device;
                    c.slice  = 8;
                    run_generic<float>(handle, make_matrix<float>(40, 30, 0, 0, 1), c);
                    if(HasFailure())
                        return;
                }
    }
}
