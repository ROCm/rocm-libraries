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
// Device unit tests of the block level2 products: gebsrmv (row_block_dim !=
// col_block_dim), bsrxmv and bsrmv.
//
// Every call is compared with a host reference of y = alpha * A * x + beta * y.
//
// rocsparse_bsrmv / rocsparse_bsrxmv route every block dimension to the
// general kernel on a GPU whose wavefront is 32. The kernels specialised for
// 2x2, 3x3, 4x4, 5x5, 8x8, 16x16 and 17..32, and the 64-lane kernels selected by
// blocks_per_row >= 64, are only reached on a GPU whose wavefront is 64, so those
// tests skip elsewhere.
//
#include "unit_test_device_level2_common.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>

using namespace level2;

namespace
{
    inline std::string
        describe(const char* what, int32_t rbd, int32_t cbd, int32_t bpr, int dir, int base)
    {
        std::ostringstream s;
        s << what << " rbd=" << rbd << " cbd=" << cbd << " bpr=" << bpr << " dir=" << dir
          << " base=" << base;
        return s.str();
    }

    //
    // gebsrmv: y = alpha * A * x + beta * y for a (rbd x cbd)-blocked matrix.
    //
    template <typename T>
    void check_gebsrmv(rocsparse_handle    handle,
                       rocsparse_direction dir,
                       int32_t             mb,
                       int32_t             nb,
                       int32_t             bpr,
                       int32_t             rbd,
                       int32_t             cbd,
                       int                 base,
                       T                   alpha,
                       T                   beta,
                       bool                device_mode,
                       bool                poison = false)
    {
        SCOPED_TRACE(describe("gebsrmv", rbd, cbd, bpr, static_cast<int>(dir), base));

        host_bsr<T>          A    = make_bsr<T>(mb, nb, bpr, rbd, cbd, dir, base);
        std::vector<T>       x    = make_vector<T>(static_cast<size_t>(nb) * cbd, 1);
        const std::vector<T> y0   = make_vector<T>(static_cast<size_t>(mb) * rbd, 5);
        std::vector<T>       want = bsr_gemv(A, all_rows(mb), {}, alpha, x, beta, y0);
        if(poison)
        {
            // Only meaningful for alpha == 0 and beta == 1: y must come back untouched
            // without the matrix and x being read, so a computation of
            // beta * y + 0 * A * x would turn y into NaN.
            ASSERT_TRUE(alpha == T{} && beta == rocsparse_ut::scalar<T>(1.0f));
            const T nan = rocsparse_ut::scalar<T>(std::numeric_limits<float>::quiet_NaN());
            std::fill(A.val.begin(), A.val.end(), nan);
            std::fill(x.begin(), x.end(), nan);
            want = y0;
        }

        mat_descr descr;
        ASSERT_NE(descr.d, nullptr);
        ASSERT_EQ(rocsparse_set_mat_index_base(descr, to_base(base)), rocsparse_status_success);

        device_vector<rocsparse_int> dptr(convert_vector<rocsparse_int>(A.ptr)),
            dind(convert_vector<rocsparse_int>(A.ind));
        device_vector<T> dval(A.val), dx(x), dy(y0);
        device_vector<T> dalpha(std::vector<T>(1, alpha)), dbeta(std::vector<T>(1, beta));
        ASSERT_TRUE(dptr.ptr && dind.ptr && dval.ptr && dx.ptr && dy.ptr && dalpha.ptr
                    && dbeta.ptr);

        scoped_pointer_mode mode(handle, device_mode);
        const T*            palpha = device_mode ? dalpha.ptr : &alpha;
        const T*            pbeta  = device_mode ? dbeta.ptr : &beta;
        ASSERT_EQ(gebsrmv<T>(handle,
                             dir,
                             rocsparse_operation_none,
                             mb,
                             nb,
                             A.nnzb(),
                             palpha,
                             descr,
                             dval,
                             dptr,
                             dind,
                             rbd,
                             cbd,
                             dx,
                             pbeta,
                             dy),
                  rocsparse_status_success);
        EXPECT_TRUE(expect_near(want, to_host(dy), "gebsrmv y"));
    }

    class L2DeviceGebsrmv : public rocsparse_ut::HandleTest
    {
    protected:
        template <typename T>
        void sweep(const std::vector<int32_t>& rbds,
                   const std::vector<int32_t>& cbds,
                   const std::vector<int32_t>& bprs,
                   bool                        both_dirs)
        {
            for(int32_t rbd : rbds)
            {
                for(int32_t cbd : cbds)
                {
                    if(rbd == cbd)
                        continue;
                    for(int32_t bpr : bprs)
                    {
                        for(int d = 0; d < (both_dirs ? 2 : 1); ++d)
                        {
                            const rocsparse_direction dir
                                = (d == 0) ? rocsparse_direction_row : rocsparse_direction_column;
                            // Alternate the index base and the scalars so that every
                            // dispatch leaf sees both of them.
                            const int  base   = ((rbd + cbd + d) & 1);
                            const bool device = (((rbd * 3 + cbd + bpr) % 5) == 0);
                            check_gebsrmv<T>(handle,
                                             dir,
                                             5,
                                             bpr + 5,
                                             bpr,
                                             rbd,
                                             cbd,
                                             base,
                                             case_scalar<T>(2.0f, 0.0f),
                                             case_scalar<T>(-0.5f, 1.0f),
                                             device);
                            if(::testing::Test::HasFailure())
                                return;
                        }
                    }
                }
            }
        }
    };

    TEST_F(L2DeviceGebsrmv, row_block_dim_1_to_4)
    {
        // Blocks per row 2, 10, 20 and 40 select the 4, 8, 16 and 32 lane kernels.
        const std::vector<int32_t> cbds
            = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20};
        sweep<float>({1, 2, 3, 4}, cbds, {2, 10, 20, 40}, true);
    }

    TEST_F(L2DeviceGebsrmv, row_block_dim_5_to_8)
    {
        const std::vector<int32_t> cbds
            = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20};
        sweep<float>({5, 6, 7, 8}, cbds, {3}, true);
    }

    TEST_F(L2DeviceGebsrmv, row_block_dim_17_and_up)
    {
        // The general kernel picks its logical wavefront from col_block_dim:
        // <= 2, <= 4, <= 8, <= 16 and larger.
        sweep<float>({17, 20, 33}, {1, 2, 3, 4, 7, 8, 9, 16, 18}, {3}, true);
    }

    TEST_F(L2DeviceGebsrmv, wave64_long_rows)
    {
        // 70 blocks per row select the 64-lane kernels of the row block dimensions 2, 3
        // and 4, which only run on a GPU whose wavefront is 64.
        if(rocsparse_ut::device_warp_size() != 64)
            GTEST_SKIP() << "needs a GPU with a wavefront of 64";
        const std::vector<int32_t> cbds
            = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20};
        sweep<float>({2, 3, 4}, cbds, {70}, true);
    }

    TEST_F(L2DeviceGebsrmv, double_and_complex)
    {
        sweep<double>({1, 2, 3}, {2, 4, 5}, {2, 12}, true);
        sweep<double>({5, 7}, {3, 9}, {3}, true);
        sweep<rocsparse_float_complex>({2, 4, 6}, {3, 7}, {5}, true);
        sweep<rocsparse_double_complex>({1, 3, 8}, {5, 2}, {5}, true);
    }

    TEST_F(L2DeviceGebsrmv, alpha_zero_beta_one_leaves_y_untouched)
    {
        // alpha == 0 and beta == 1 leaves y untouched. The matrix values and x are NaN,
        // so a product that is computed (0 * NaN) shows. Device scalars bypass the
        // host side early return, so the kernels' own handling is exercised too.
        for(const int32_t rbd : {1, 3, 6, 18})
        {
            const int32_t cbd = (rbd == 3) ? 2 : 5;
            for(const bool device : {false, true})
                for(const int base : {0, 1})
                {
                    check_gebsrmv<float>(handle,
                                         rocsparse_direction_row,
                                         4,
                                         9,
                                         3,
                                         rbd,
                                         cbd,
                                         base,
                                         0.0f,
                                         1.0f,
                                         device,
                                         true);
                    check_gebsrmv<float>(handle,
                                         rocsparse_direction_column,
                                         4,
                                         9,
                                         3,
                                         rbd,
                                         cbd,
                                         base,
                                         0.0f,
                                         1.0f,
                                         device,
                                         true);
                }
        }
    }

    TEST_F(L2DeviceGebsrmv, empty_matrix_scales_y_by_beta)
    {
        // nb == 0: y (mb * row_block_dim entries) = beta * y. mb == 0: y is empty and
        // never accessed.
        const int32_t rbd = 3, cbd = 2;
        for(const bool device : {false, true})
            for(const int32_t mb : {4, 0})
            {
                SCOPED_TRACE("device=" + std::to_string(device) + " mb=" + std::to_string(mb));
                const std::vector<double> y0    = make_vector<double>(12, 3);
                const double              alpha = 2.0, beta = -0.75;
                mat_descr                 descr;
                ASSERT_NE(descr.d, nullptr);
                device_vector<rocsparse_int> ptr(std::vector<rocsparse_int>(5, 0));
                device_vector<double>        x(std::vector<double>(1, 0.0));
                device_vector<double>        y(y0);
                device_vector<double>        dalpha(std::vector<double>(1, alpha)),
                    dbeta(std::vector<double>(1, beta));
                ASSERT_TRUE(ptr.ptr && x.ptr && y.ptr && dalpha.ptr && dbeta.ptr);
                scoped_pointer_mode mode(handle, device);
                ASSERT_EQ(gebsrmv<double>(handle,
                                          rocsparse_direction_column,
                                          rocsparse_operation_none,
                                          mb,
                                          mb == 0 ? 5 : 0,
                                          0,
                                          device ? dalpha.ptr : &alpha,
                                          descr,
                                          nullptr,
                                          ptr.ptr,
                                          nullptr,
                                          rbd,
                                          cbd,
                                          x.ptr,
                                          device ? dbeta.ptr : &beta,
                                          y.ptr),
                          rocsparse_status_success);
                std::vector<double> want(y0);
                for(int32_t i = 0; i < mb * rbd; ++i)
                    want[i] *= beta;
                EXPECT_TRUE(expect_near(want, to_host(y), "gebsrmv beta scaling"));
            }

        mat_descr descr;
        ASSERT_NE(descr.d, nullptr);
        const double alpha = 1.0;
        EXPECT_EQ(gebsrmv<double>(handle,
                                  rocsparse_direction_row,
                                  rocsparse_operation_none,
                                  4,
                                  0,
                                  0,
                                  &alpha,
                                  descr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  rbd,
                                  cbd,
                                  nullptr,
                                  nullptr,
                                  nullptr),
                  rocsparse_status_invalid_pointer);
    }

    //
    // bsrxmv / bsrmv with a square (bd x bd) block.
    //
    struct xmv_case
    {
        int32_t             bd;
        int32_t             bpr;
        rocsparse_direction dir;
        bool                masked;
        bool                device;
        int                 base;
    };

    template <typename T>
    void check_bsrxmv(rocsparse_handle handle, const xmv_case& c, T alpha, T beta)
    {
        SCOPED_TRACE(describe("bsrxmv", c.bd, c.bd, c.bpr, static_cast<int>(c.dir), c.base)
                     + (c.masked ? " masked" : " all rows"));

        const int32_t        mb = 7;
        const int32_t        nb = c.bpr + 4;
        const host_bsr<T>    A  = make_bsr<T>(mb, nb, c.bpr, c.bd, c.bd, c.dir, c.base);
        const std::vector<T> x  = make_vector<T>(static_cast<size_t>(nb) * c.bd, 2);
        const std::vector<T> y0 = make_vector<T>(static_cast<size_t>(mb) * c.bd, 9);

        // Row ends. The masked variant shortens the first block of the even rows
        // away, so the end pointer array (and not ptr[i + 1]) decides the range.
        std::vector<int32_t> end(mb);
        for(int32_t i = 0; i < mb; ++i)
        {
            end[i] = A.ptr[i + 1];
            if(c.masked && (i % 2 == 0) && A.ptr[i + 1] - A.ptr[i] > 1)
                end[i] -= 1;
        }

        // Rows processed: every row in order, or every other row in reverse order.
        std::vector<int32_t> rows;
        if(c.masked)
        {
            for(int32_t i = mb - 1; i >= 0; i -= 2)
                rows.push_back(i);
        }
        else
        {
            rows = all_rows(mb);
        }
        std::vector<int32_t> mask(rows);
        for(auto& r : mask)
            r += c.base;
        const std::vector<T> want = bsr_gemv(A, rows, end, alpha, x, beta, y0);

        mat_descr descr;
        ASSERT_NE(descr.d, nullptr);
        ASSERT_EQ(rocsparse_set_mat_index_base(descr, to_base(c.base)), rocsparse_status_success);

        device_vector<rocsparse_int> dptr(convert_vector<rocsparse_int>(A.ptr)),
            dind(convert_vector<rocsparse_int>(A.ind)), dend(convert_vector<rocsparse_int>(end));
        device_vector<rocsparse_int> dmask(convert_vector<rocsparse_int>(mask));
        device_vector<T>             dval(A.val), dx(x), dy(y0);
        device_vector<T> dalpha(std::vector<T>(1, alpha)), dbeta(std::vector<T>(1, beta));
        ASSERT_TRUE(dptr.ptr && dind.ptr && dend.ptr && dmask.ptr && dval.ptr && dx.ptr && dy.ptr
                    && dalpha.ptr && dbeta.ptr);

        scoped_pointer_mode mode(handle, c.device);
        const T*            palpha = c.device ? dalpha.ptr : &alpha;
        const T*            pbeta  = c.device ? dbeta.ptr : &beta;
        ASSERT_EQ(bsrxmv<T>(handle,
                            c.dir,
                            rocsparse_operation_none,
                            static_cast<rocsparse_int>(mask.size()),
                            mb,
                            nb,
                            A.nnzb(),
                            palpha,
                            descr,
                            dval,
                            dmask.ptr,
                            dptr,
                            dend,
                            dind,
                            c.bd,
                            dx,
                            pbeta,
                            dy),
                  rocsparse_status_success);
        EXPECT_TRUE(expect_near(want, to_host(dy), "bsrxmv y"));
    }

    template <typename T>
    void
        check_bsrmv(rocsparse_handle handle, const xmv_case& c, T alpha, T beta, bool with_analysis)
    {
        SCOPED_TRACE(describe("bsrmv", c.bd, c.bd, c.bpr, static_cast<int>(c.dir), c.base));

        const int32_t        mb   = 6;
        const int32_t        nb   = c.bpr + 3;
        const host_bsr<T>    A    = make_bsr<T>(mb, nb, c.bpr, c.bd, c.bd, c.dir, c.base);
        const std::vector<T> x    = make_vector<T>(static_cast<size_t>(nb) * c.bd, 4);
        const std::vector<T> y0   = make_vector<T>(static_cast<size_t>(mb) * c.bd, 6);
        const std::vector<T> want = bsr_gemv(A, all_rows(mb), {}, alpha, x, beta, y0);

        mat_descr descr;
        ASSERT_NE(descr.d, nullptr);
        ASSERT_EQ(rocsparse_set_mat_index_base(descr, to_base(c.base)), rocsparse_status_success);
        device_vector<rocsparse_int> dptr(convert_vector<rocsparse_int>(A.ptr)),
            dind(convert_vector<rocsparse_int>(A.ind));
        device_vector<T> dval(A.val), dx(x), dy(y0);
        device_vector<T> dalpha(std::vector<T>(1, alpha)), dbeta(std::vector<T>(1, beta));
        ASSERT_TRUE(dptr.ptr && dind.ptr && dval.ptr && dx.ptr && dy.ptr && dalpha.ptr
                    && dbeta.ptr);

        scoped_pointer_mode mode(handle, c.device);
        const T*            palpha = c.device ? dalpha.ptr : &alpha;
        const T*            pbeta  = c.device ? dbeta.ptr : &beta;

        // The analysis only records something for block_dim == 1.
        mat_info info;
        ASSERT_NE(info.d, nullptr);
        if(with_analysis)
            ASSERT_EQ(bsrmv_analysis<T>(handle,
                                        c.dir,
                                        rocsparse_operation_none,
                                        mb,
                                        nb,
                                        A.nnzb(),
                                        descr,
                                        dval,
                                        dptr,
                                        dind,
                                        c.bd,
                                        info),
                      rocsparse_status_success);
        ASSERT_EQ(bsrmv<T>(handle,
                           c.dir,
                           rocsparse_operation_none,
                           mb,
                           nb,
                           A.nnzb(),
                           palpha,
                           descr,
                           dval,
                           dptr,
                           dind,
                           c.bd,
                           with_analysis ? info.d : nullptr,
                           dx,
                           pbeta,
                           dy),
                  rocsparse_status_success);
        EXPECT_TRUE(expect_near(want, to_host(dy), "bsrmv y"));
        if(with_analysis)
            EXPECT_EQ(rocsparse_bsrmv_clear(handle, info), rocsparse_status_success);
    }

    class L2DeviceBsrxmv : public rocsparse_ut::HandleTest
    {
    protected:
        template <typename T>
        void sweep_xmv(const std::vector<int32_t>& bds, const std::vector<int32_t>& bprs)
        {
            for(int32_t bd : bds)
            {
                for(int32_t bpr : bprs)
                {
                    for(int d = 0; d < 2; ++d)
                    {
                        for(int masked = 0; masked < 2; ++masked)
                        {
                            xmv_case c;
                            c.bd  = bd;
                            c.bpr = bpr;
                            c.dir = d == 0 ? rocsparse_direction_row : rocsparse_direction_column;
                            c.masked = masked != 0;
                            c.device = ((bd + bpr + d + masked) % 3) == 0;
                            c.base   = (bd + masked) & 1;
                            check_bsrxmv<T>(
                                handle, c, case_scalar<T>(1.5f, 0.0f), case_scalar<T>(0.25f, 1.0f));
                            if(::testing::Test::HasFailure())
                                return;
                        }
                    }
                }
            }
        }

        template <typename T>
        void sweep_mv(const std::vector<int32_t>& bds,
                      const std::vector<int32_t>& bprs,
                      bool                        with_analysis = false)
        {
            for(int32_t bd : bds)
            {
                for(int32_t bpr : bprs)
                {
                    for(int d = 0; d < 2; ++d)
                    {
                        xmv_case c;
                        c.bd     = bd;
                        c.bpr    = bpr;
                        c.dir    = d == 0 ? rocsparse_direction_row : rocsparse_direction_column;
                        c.masked = false;
                        c.device = ((bd + bpr + d) % 3) == 0;
                        c.base   = (bd + d) & 1;
                        check_bsrmv<T>(handle,
                                       c,
                                       case_scalar<T>(2.0f, 0.0f),
                                       case_scalar<T>(-1.0f, 1.0f),
                                       with_analysis);
                        if(::testing::Test::HasFailure())
                            return;
                    }
                }
            }
        }
    };

    TEST_F(L2DeviceBsrxmv, general_kernels_native_wavefront)
    {
        // The handle keeps the wavefront size of the device. On a wave32 GPU every block
        // dimension is handled by the general kernel (1 and 2 use the tight 2x2 launch,
        // 3..4 the 4x4 launch, 5..8 the 8x8, 9..16 the 16x16 and everything above the
        // 32x32 launch); on a wave64 GPU the specialised kernels are reached instead.
        sweep_xmv<float>({2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17, 24, 32, 33}, {3});
        sweep_xmv<double>({2, 4, 8, 16, 20}, {9});
        sweep_xmv<rocsparse_float_complex>({3, 5, 10}, {4});
        sweep_xmv<rocsparse_double_complex>({2, 7, 17}, {4});
    }

    TEST_F(L2DeviceBsrxmv, specialised_kernels_wave64)
    {
        if(rocsparse_ut::device_warp_size() != 64)
            GTEST_SKIP() << "needs a GPU with a wavefront of 64";
        // Dimensions 2, 3 and 4: blocks per row 2, 10, 20 and 40 select the 4, 8,
        // 16 and 32 lane kernels.
        sweep_xmv<float>({2, 3, 4}, {2, 10, 20, 40});
        sweep_xmv<double>({2, 3, 4}, {3, 17});
        sweep_xmv<rocsparse_float_complex>({2, 3, 4}, {5});
        sweep_xmv<rocsparse_double_complex>({2, 3, 4}, {12});
        sweep_xmv<float>({5, 8}, {3, 9});
        sweep_xmv<float>({16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}, {3});
        sweep_xmv<double>({5, 8, 16, 19, 32}, {7});
        sweep_xmv<rocsparse_float_complex>({5, 8, 16, 17, 25}, {2});
        sweep_xmv<rocsparse_double_complex>({5, 8, 16, 18, 30}, {2});
    }

    TEST_F(L2DeviceBsrxmv, wave64_long_rows)
    {
        // 70 blocks per row select the 64-lane 2x2, 3x3 and 4x4 kernels, which only run
        // on a GPU whose wavefront is 64.
        if(rocsparse_ut::device_warp_size() != 64)
            GTEST_SKIP() << "needs a GPU with a wavefront of 64";
        sweep_xmv<float>({2, 3, 4}, {70});
        sweep_xmv<double>({2, 3, 4}, {70});
    }

    TEST_F(L2DeviceBsrxmv, argument_paths)
    {
        mat_descr descr;
        ASSERT_NE(descr.d, nullptr);
        const float one = 1.0f;

        // ind doubles as the mask of the single block row.
        device_vector<rocsparse_int> ptr(std::vector<rocsparse_int>{0, 1}),
            ind(std::vector<rocsparse_int>{0});
        const float          nan = std::numeric_limits<float>::quiet_NaN();
        device_vector<float> val(std::vector<float>(4, nan));
        device_vector<float> x(std::vector<float>(2, nan)), y(std::vector<float>(2, 1.0f));
        ASSERT_TRUE(ptr.ptr && ind.ptr && val.ptr && x.ptr && y.ptr);

        // block_dim == 1 has no bsrxmv kernel.
        EXPECT_EQ(bsrxmv<float>(handle,
                                rocsparse_direction_row,
                                rocsparse_operation_none,
                                1,
                                1,
                                1,
                                1,
                                &one,
                                descr,
                                val.ptr,
                                ind.ptr,
                                ptr.ptr,
                                ptr.ptr + 1,
                                ind.ptr,
                                1,
                                x.ptr,
                                &one,
                                y.ptr),
                  rocsparse_status_not_implemented);
        // Transposition is not supported.
        EXPECT_EQ(bsrxmv<float>(handle,
                                rocsparse_direction_row,
                                rocsparse_operation_transpose,
                                1,
                                1,
                                1,
                                1,
                                &one,
                                descr,
                                val.ptr,
                                ind.ptr,
                                ptr.ptr,
                                ptr.ptr + 1,
                                ind.ptr,
                                2,
                                x.ptr,
                                &one,
                                y.ptr),
                  rocsparse_status_not_implemented);
        // alpha == 0 with beta == 1 returns before touching y (host pointer mode); the
        // matrix values and x are NaN, so computing beta * y + 0 * A * x shows.
        const float zero = 0.0f;
        EXPECT_EQ(bsrxmv<float>(handle,
                                rocsparse_direction_row,
                                rocsparse_operation_none,
                                1,
                                1,
                                1,
                                1,
                                &zero,
                                descr,
                                val.ptr,
                                ind.ptr,
                                ptr.ptr,
                                ptr.ptr + 1,
                                ind.ptr,
                                2,
                                x.ptr,
                                &one,
                                y.ptr),
                  rocsparse_status_success);
        EXPECT_TRUE(expect_near(std::vector<float>(2, 1.0f), to_host(y), "bsrxmv untouched y"));
    }

    TEST_F(L2DeviceBsrxmv, empty_matrix_scales_y_by_beta)
    {
        // nb == 0: the matrix is not read, y (selected rows) is scaled by beta. The
        // scaling kernel reads beta from device memory.
        const int32_t            mb   = 5;
        const int32_t            bd   = 3;
        const std::vector<float> y0   = make_vector<float>(static_cast<size_t>(mb) * bd, 3);
        const float              beta = -2.0f;

        mat_descr descr;
        ASSERT_NE(descr.d, nullptr);
        device_vector<rocsparse_int> ptr(std::vector<rocsparse_int>(mb + 1, 0));
        device_vector<rocsparse_int> every(convert_vector<rocsparse_int>(all_rows(mb)));
        device_vector<rocsparse_int> mask(std::vector<rocsparse_int>{4, 1});
        device_vector<float>         x(std::vector<float>(1, 0.0f));
        device_vector<float>         dalpha(std::vector<float>(1, 1.5f));
        device_vector<float>         dbeta(std::vector<float>(1, beta));
        ASSERT_TRUE(ptr.ptr && every.ptr && mask.ptr && x.ptr && dalpha.ptr && dbeta.ptr);

        scoped_pointer_mode mode(handle, rocsparse_pointer_mode_device);

        // All rows.
        {
            device_vector<float> y(y0);
            ASSERT_TRUE(y.ptr);
            ASSERT_EQ(bsrxmv<float>(handle,
                                    rocsparse_direction_row,
                                    rocsparse_operation_none,
                                    mb,
                                    mb,
                                    0,
                                    0,
                                    dalpha.ptr,
                                    descr,
                                    nullptr,
                                    every.ptr,
                                    ptr.ptr,
                                    ptr.ptr + 1,
                                    nullptr,
                                    bd,
                                    x.ptr,
                                    dbeta.ptr,
                                    y.ptr),
                      rocsparse_status_success);
            std::vector<float> want(y0);
            for(auto& v : want)
                v *= beta;
            EXPECT_TRUE(expect_near(want, to_host(y), "bsrxmv beta scaling, all rows"));
        }

        // Rows 4 and 1 only.
        {
            device_vector<float> y(y0);
            ASSERT_TRUE(y.ptr);
            ASSERT_EQ(bsrxmv<float>(handle,
                                    rocsparse_direction_row,
                                    rocsparse_operation_none,
                                    2,
                                    mb,
                                    0,
                                    0,
                                    dalpha.ptr,
                                    descr,
                                    nullptr,
                                    mask.ptr,
                                    ptr.ptr,
                                    ptr.ptr + 1,
                                    nullptr,
                                    bd,
                                    x.ptr,
                                    dbeta.ptr,
                                    y.ptr),
                      rocsparse_status_success);
            std::vector<float> want(y0);
            for(int32_t r : {4, 1})
                for(int32_t k = 0; k < bd; ++k)
                    want[static_cast<size_t>(r) * bd + k] *= beta;
            EXPECT_TRUE(expect_near(want, to_host(y), "bsrxmv beta scaling, masked rows"));
        }
    }

    class L2DeviceBsrmv : public L2DeviceBsrxmv
    {
    };

    TEST_F(L2DeviceBsrmv, specialised_kernels_wave64)
    {
        if(rocsparse_ut::device_warp_size() != 64)
            GTEST_SKIP() << "needs a GPU with a wavefront of 64";
        sweep_mv<float>({2, 3, 4}, {2, 10, 20});
        sweep_mv<float>({5, 8, 16, 17, 20, 25, 32}, {3});
        sweep_mv<double>({2, 3, 4, 5, 8, 16, 18}, {6});
        sweep_mv<rocsparse_float_complex>({3, 4, 5, 8}, {3});
        sweep_mv<rocsparse_double_complex>({2, 5, 8, 17}, {3});
    }

    TEST_F(L2DeviceBsrmv, with_analysis)
    {
        sweep_mv<float>({1, 2}, {3, 40}, true);
        sweep_mv<rocsparse_double_complex>({1, 3}, {4}, true);
    }

    TEST_F(L2DeviceBsrmv, empty_matrix_scales_y_by_beta)
    {
        const std::vector<float> y0   = make_vector<float>(12, 1);
        const float              beta = 0.5f;
        const float              one  = 1.0f;
        mat_descr                descr;
        ASSERT_NE(descr.d, nullptr);
        device_vector<rocsparse_int> ptr(std::vector<rocsparse_int>(5, 0));
        device_vector<float>         x(std::vector<float>(1, 0.0f));
        device_vector<float>         y(y0);
        ASSERT_TRUE(ptr.ptr && x.ptr && y.ptr);

        // nnzb == 0: y = beta * y.
        ASSERT_EQ(bsrmv<float>(handle,
                               rocsparse_direction_row,
                               rocsparse_operation_none,
                               4,
                               2,
                               0,
                               &one,
                               descr,
                               nullptr,
                               ptr.ptr,
                               nullptr,
                               3,
                               nullptr,
                               x.ptr,
                               &beta,
                               y.ptr),
                  rocsparse_status_success);
        std::vector<float> want(y0);
        for(auto& v : want)
            v *= beta;
        EXPECT_TRUE(expect_near(want, to_host(y), "bsrmv beta scaling"));
    }
}
