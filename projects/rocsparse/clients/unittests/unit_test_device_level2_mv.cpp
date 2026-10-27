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
// Device unit tests of the sparse matrix-vector products with a CSR matrix:
//   * the legacy rocsparse_Xcsrmv / rocsparse_Xcsrmv_analysis routines,
//   * the v2 spmv descriptor API (rocsparse_v2_spmv) with its algorithms,
//     mixed scalar / compute precisions and extra terms (gamma * z),
//   * the algorithm / format validation of rocsparse_spmv and rocsparse_v2_spmv.
// Results are compared with a host product on the same data.
//
#include "unit_test_device_level2_common.hpp"

#include <deque>
#include <memory>
#include <set>
#include <string>
#include <tuple>

using namespace level2;

namespace
{
    // ----- helpers -----------------------------------------------------------------
    template <typename CT, typename AT>
    host_csr<CT> cast_csr(const host_csr<AT>& a)
    {
        host_csr<CT> r;
        r.m    = a.m;
        r.n    = a.n;
        r.base = a.base;
        r.ptr  = a.ptr;
        r.ind  = a.ind;
        r.val.assign(a.val.begin(), a.val.end());
        return r;
    }

    // Rectangular CSR with rows of very different length: every seventh row is
    // empty, a few rows are (almost) dense when `big` is set, the others have one
    // to nine entries. With `empty` the matrix has no entry at all.
    template <typename T>
    host_csr<T> make_irregular_csr(int32_t m, int32_t n, int base, bool big, bool empty)
    {
        return make_csr_by_length<T>(m, n, base, [=](int32_t i) {
            if(empty || i % 7 == 3)
                return 0;
            return (big && i % 13 == 5) ? 200 : 1 + (i * 5) % 9;
        });
    }

    // Full symmetric matrix from the stored lower triangle.
    template <typename T>
    host_csr<T> symmetrize_lower(const host_csr<T>& low)
    {
        const int32_t                                   n = low.m;
        std::vector<std::vector<std::pair<int32_t, T>>> rows(n);
        for(int32_t i = 0; i < n; ++i)
            for(int32_t k = low.ptr[i] - low.base; k < low.ptr[i + 1] - low.base; ++k)
            {
                const int32_t j = low.ind[k] - low.base;
                rows[i].push_back({j, low.val[k]});
                if(i != j)
                    rows[j].push_back({i, low.val[k]});
            }
        host_csr<T> s;
        s.m    = n;
        s.n    = n;
        s.base = low.base;
        s.ptr.push_back(low.base);
        for(int32_t i = 0; i < n; ++i)
        {
            std::sort(rows[i].begin(), rows[i].end(), [](const auto& l, const auto& r) {
                return l.first < r.first;
            });
            for(const auto& e : rows[i])
            {
                s.ind.push_back(e.first + low.base);
                s.val.push_back(e.second);
            }
            s.ptr.push_back(static_cast<int32_t>(s.ind.size()) + low.base);
        }
        return s;
    }

    // ----- legacy csrmv ---------------------------------------------------------------
    struct csrmv_case
    {
        rocsparse_operation   trans     = rocsparse_operation_none;
        rocsparse_matrix_type mtype     = rocsparse_matrix_type_general;
        int                   info_mode = 0; // 0: no info, 1: fresh info, 2: analysed info
        bool                  device    = false;
        int                   base      = 0;
        int32_t               m         = 37;
        int32_t               n         = 29;
        float                 alpha     = 1.5f;
        float                 beta      = -0.5f;
        bool                  empty     = false;
        bool                  big       = false;
    };

    template <typename T>
    void run_csrmv(rocsparse_handle handle, const csrmv_case& c)
    {
        SCOPED_TRACE(::testing::Message()
                     << "csrmv trans=" << c.trans << " type=" << c.mtype << " info=" << c.info_mode
                     << " device=" << c.device << " base=" << c.base << " m=" << c.m << " n=" << c.n
                     << " alpha=" << c.alpha << " beta=" << c.beta << " empty=" << c.empty);

        const bool  sym = (c.mtype == rocsparse_matrix_type_symmetric);
        host_csr<T> store;
        host_csr<T> ref;
        if(sym)
        {
            store = make_triangular_pattern<T>(c.n, 3, true, false, true, c.base);
            ref   = symmetrize_lower(store);
        }
        else
        {
            store = make_irregular_csr<T>(c.m, c.n, c.base, c.big, c.empty);
            ref   = store;
        }
        const int32_t m  = store.m;
        const int32_t n  = store.n;
        const int32_t xs = (c.trans == rocsparse_operation_none) ? n : m;
        const int32_t ys = (c.trans == rocsparse_operation_none) ? m : n;

        const std::vector<T> x     = make_vector<T>(xs, 2);
        const std::vector<T> y0    = make_vector<T>(ys, 9);
        const T              alpha = case_scalar<T>(c.alpha, 0.0f);
        const T              beta  = case_scalar<T>(c.beta, 1.0f);
        const std::vector<T> want  = csr_gemv(ref, c.trans, alpha, x, beta, y0);

        mat_descr descr;
        mat_info  info;
        ASSERT_TRUE(descr.d && info.d);
        ASSERT_EQ(rocsparse_set_mat_type(descr, c.mtype), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_index_base(descr, to_base(c.base)), rocsparse_status_success);

        device_csr<T>    dA(store);
        device_vector<T> dx(x), dy(y0), dalpha(std::vector<T>(1, alpha)),
            dbeta(std::vector<T>(1, beta));
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr && dalpha.ptr && dbeta.ptr);

        if(c.info_mode == 2)
        {
            ASSERT_EQ(csrmv_analysis<T>(
                          handle, c.trans, m, n, store.nnz(), descr, dA.val, dA.ptr, dA.ind, info),
                      rocsparse_status_success);
        }

        rocsparse_status st;
        {
            scoped_pointer_mode mode(handle, c.device);
            st = csrmv<T>(handle,
                          c.trans,
                          m,
                          n,
                          store.nnz(),
                          c.device ? dalpha.ptr : &alpha,
                          descr,
                          dA.val,
                          dA.ptr,
                          dA.ind,
                          c.info_mode == 0 ? nullptr : info.d,
                          dx,
                          c.device ? dbeta.ptr : &beta,
                          dy);
        }
        // rocsparse_csrmv.h documents not_implemented for a matrix type other than
        // general; a symmetric product that succeeds is checked.
        if(sym)
            ASSERT_TRUE(st == rocsparse_status_success || st == rocsparse_status_not_implemented)
                << st;
        else
            ASSERT_EQ(st, rocsparse_status_success);
        if(st == rocsparse_status_success)
            EXPECT_TRUE(expect_near(want, to_host(dy), "csrmv y"));
        EXPECT_EQ(rocsparse_csrmv_clear(handle, info), rocsparse_status_success);
    }

    class L2DeviceCsrmv : public rocsparse_ut::HandleTest
    {
    protected:
        // The full cross product of operation, analysis mode, matrix type, index base
        // and pointer mode; each one is an independent loop. A symmetric matrix is
        // only checked with real data.
        template <typename T>
        void sweep()
        {
            std::set<std::tuple<int, int, int, int, int>> seen;
            for(const auto trans : {rocsparse_operation_none,
                                    rocsparse_operation_transpose,
                                    rocsparse_operation_conjugate_transpose})
                for(const int info_mode : {0, 1, 2})
                    for(const auto mtype :
                        {rocsparse_matrix_type_general, rocsparse_matrix_type_symmetric})
                    {
                        if(mtype == rocsparse_matrix_type_symmetric
                           && std::is_same<typename real_of<T>::type, T>::value == false)
                            continue; // symmetric products are only checked for real data
                        for(const int base : {0, 1})
                            for(const bool device : {false, true})
                            {
                                seen.insert(std::make_tuple(static_cast<int>(trans),
                                                            info_mode,
                                                            static_cast<int>(mtype),
                                                            base,
                                                            device ? 1 : 0));
                                csrmv_case c;
                                c.trans     = trans;
                                c.info_mode = info_mode;
                                c.mtype     = mtype;
                                c.device    = device;
                                c.base      = base;
                                run_csrmv<T>(handle, c);
                                if(::testing::Test::HasFailure())
                                    return;
                            }
                    }
            const bool real = std::is_same<typename real_of<T>::type, T>::value;
            EXPECT_EQ(seen.size(), size_t{3 * 3 * (real ? 2 : 1) * 2 * 2});
        }
    };

    TEST_F(L2DeviceCsrmv, float_all_variants)
    {
        sweep<float>();
    }

    TEST_F(L2DeviceCsrmv, double_and_complex)
    {
        sweep<double>();
        sweep<rocsparse_float_complex>();
        sweep<rocsparse_double_complex>();
    }

    TEST_F(L2DeviceCsrmv, scalars_that_skip_or_scale)
    {
        // alpha == 0 skips the product, beta == 0 overwrites y and beta == 1 keeps it.
        for(const bool device : {false, true})
            for(const float alpha : {0.0f, 1.0f, -2.0f})
                for(const float beta : {0.0f, 1.0f, 2.0f})
                {
                    csrmv_case c;
                    c.info_mode = 2;
                    c.device    = device;
                    c.alpha     = alpha;
                    c.beta      = beta;
                    run_csrmv<float>(handle, c);
                    if(::testing::Test::HasFailure())
                        return;
                }
    }

    TEST_F(L2DeviceCsrmv, long_rows)
    {
        for(const int info_mode : {0, 2})
            for(const auto trans : {rocsparse_operation_none, rocsparse_operation_transpose})
            {
                csrmv_case c;
                c.m         = 70;
                c.n         = 400;
                c.big       = true;
                c.info_mode = info_mode;
                c.trans     = trans;
                run_csrmv<float>(handle, c);
                if(::testing::Test::HasFailure())
                    return;
            }
    }

    TEST_F(L2DeviceCsrmv, empty_matrix_scales_y)
    {
        for(const bool device : {false, true})
            for(const auto trans : {rocsparse_operation_none, rocsparse_operation_transpose})
                for(const int info_mode : {0, 1, 2})
                {
                    csrmv_case c;
                    c.empty     = true;
                    c.device    = device;
                    c.trans     = trans;
                    c.info_mode = info_mode;
                    run_csrmv<float>(handle, c);
                    if(::testing::Test::HasFailure())
                        return;
                }
    }

    TEST_F(L2DeviceCsrmv, zero_sized_dimensions)
    {
        mat_descr descr;
        mat_info  info;
        ASSERT_TRUE(descr.d && info.d);
        device_vector<float> one(std::vector<float>(4, 1.0f)), y(std::vector<float>(4, 3.0f));
        device_vector<rocsparse_int> ptr(std::vector<rocsparse_int>(8, 0)),
            ind(std::vector<rocsparse_int>(4, 0));
        ASSERT_TRUE(one.ptr && y.ptr && ptr.ptr && ind.ptr);
        const float alpha = 1.0f;
        const float beta  = 2.0f;

        // m == 0: nothing to update.
        EXPECT_EQ(
            csrmv_analysis<float>(
                handle, rocsparse_operation_none, 0, 4, 0, descr, one.ptr, ptr.ptr, ind.ptr, info),
            rocsparse_status_success);
        EXPECT_EQ(csrmv<float>(handle,
                               rocsparse_operation_none,
                               0,
                               4,
                               0,
                               &alpha,
                               descr,
                               one.ptr,
                               ptr.ptr,
                               ind.ptr,
                               info,
                               one,
                               &beta,
                               y),
                  rocsparse_status_success);
        EXPECT_TRUE(expect_near(std::vector<float>(4, 3.0f), to_host(y), "csrmv y (m=0)"));

        // n == 0 with a non transposed product: y = beta * y.
        EXPECT_EQ(
            csrmv_analysis<float>(
                handle, rocsparse_operation_none, 4, 0, 0, descr, one.ptr, ptr.ptr, ind.ptr, info),
            rocsparse_status_success);
        EXPECT_EQ(csrmv<float>(handle,
                               rocsparse_operation_none,
                               4,
                               0,
                               0,
                               &alpha,
                               descr,
                               one.ptr,
                               ptr.ptr,
                               ind.ptr,
                               info,
                               one,
                               &beta,
                               y),
                  rocsparse_status_success);
        EXPECT_TRUE(expect_near(std::vector<float>(4, 6.0f), to_host(y), "csrmv y (n=0)"));
    }

    TEST_F(L2DeviceCsrmv, argument_checks)
    {
        const host_csr<float> A = make_irregular_csr<float>(12, 9, 0, false, false);
        device_csr<float>     dA(A);
        device_vector<float>  dx(make_vector<float>(9, 1)), dy(make_vector<float>(12, 2));
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr);
        mat_descr descr;
        mat_info  info;
        ASSERT_TRUE(descr.d && info.d);
        mat_descr unsorted;
        ASSERT_EQ(rocsparse_set_mat_storage_mode(unsorted, rocsparse_storage_mode_unsorted),
                  rocsparse_status_success);
        const float alpha           = 1.0f;
        const float beta            = 0.0f;
        const auto  op              = rocsparse_operation_none;
        const auto  invalid_pointer = rocsparse_status_invalid_pointer;

        {
            enum
            {
                HANDLE,
                TRANS,
                M,
                N,
                NNZ,
                ALPHA,
                DESCR,
                VAL,
                PTR,
                IND,
                INFO,
                X,
                BETA,
                Y
            };
            const arg_checker bad([](auto... a) { return csrmv<float>(a...); },
                                  handle,
                                  op,
                                  12,
                                  9,
                                  A.nnz(),
                                  &alpha,
                                  descr.d,
                                  dA.val.ptr,
                                  dA.ptr.ptr,
                                  dA.ind.ptr,
                                  info.d,
                                  dx.ptr,
                                  &beta,
                                  dy.ptr);
            bad.check<M>(12, rocsparse_status_success);
            bad.check<HANDLE>(nullptr, rocsparse_status_invalid_handle);
            bad.check_invalid<TRANS>(static_cast<rocsparse_operation>(77));
            bad.check_invalid<M>(-1);
            bad.check_invalid<N>(-1);
            bad.check_invalid<NNZ>(-1);
            bad.check<ALPHA>(nullptr, invalid_pointer);
            bad.check<DESCR>(nullptr, invalid_pointer);
            bad.check<VAL>(nullptr, invalid_pointer);
            bad.check<PTR>(nullptr, invalid_pointer);
            bad.check<IND>(nullptr, invalid_pointer);
            bad.check<X>(nullptr, invalid_pointer);
            bad.check<BETA>(nullptr, invalid_pointer);
            bad.check<Y>(nullptr, invalid_pointer);
            bad.check_fails<DESCR>(unsorted.d);
        }
        {
            enum
            {
                HANDLE,
                TRANS,
                M,
                N,
                NNZ,
                DESCR,
                VAL,
                PTR,
                IND,
                INFO
            };
            const arg_checker bad([](auto... a) { return csrmv_analysis<float>(a...); },
                                  handle,
                                  op,
                                  12,
                                  9,
                                  A.nnz(),
                                  descr.d,
                                  dA.val.ptr,
                                  dA.ptr.ptr,
                                  dA.ind.ptr,
                                  info.d);
            bad.check<HANDLE>(nullptr, rocsparse_status_invalid_handle);
            bad.check<DESCR>(nullptr, invalid_pointer);
            bad.check<VAL>(nullptr, invalid_pointer);
            bad.check<PTR>(nullptr, invalid_pointer);
            bad.check<IND>(nullptr, invalid_pointer);
            bad.check_invalid<M>(-1);
            bad.check_fails<DESCR>(unsorted.d);
        }
        EXPECT_EQ(rocsparse_csrmv_clear(nullptr, info), rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_csrmv_clear(handle, nullptr), rocsparse_status_invalid_pointer);
    }

    // ----- v2 spmv descriptor API ----------------------------------------------------------
    struct v2_case
    {
        rocsparse_spmv_alg  alg          = rocsparse_spmv_alg_default;
        rocsparse_operation trans        = rocsparse_operation_none;
        int                 extras       = 0;
        bool                device       = false;
        int                 base         = 0;
        bool                starting_ids = false;
        // 0: general data, 1: matrix without entries, 2: alpha = 0 and beta = 1
        int     special    = 0;
        bool    zero_gamma = false;
        bool    big        = false;
        int32_t m          = 41;
        int32_t n          = 33;
    };

    // A: matrix precision, CT: precision of x, y and of the computation, ST: precision of alpha,
    // beta and gamma.
    template <typename AT, typename CT, typename ST>
    void run_v2(rocsparse_handle handle, const v2_case& c)
    {
        SCOPED_TRACE(::testing::Message()
                     << "v2_spmv alg=" << c.alg << " trans=" << c.trans << " extras=" << c.extras
                     << " device=" << c.device << " base=" << c.base << " special=" << c.special
                     << " zero_gamma=" << c.zero_gamma << " ids=" << c.starting_ids);

        const host_csr<AT> A  = make_irregular_csr<AT>(c.m, c.n, c.base, c.big, c.special == 1);
        const host_csr<CT> Ac = cast_csr<CT>(A);
        const int32_t      xs = (c.trans == rocsparse_operation_none) ? c.n : c.m;
        const int32_t      ys = (c.trans == rocsparse_operation_none) ? c.m : c.n;

        const std::vector<CT> x  = make_vector<CT>(xs, 2);
        const std::vector<CT> y0 = make_vector<CT>(ys, 9);
        const ST alpha = (c.special == 2) ? scalar_of<ST>(0.0f, 0.0f) : scalar_of<ST>(1.5f, -0.5f);
        const ST beta  = (c.special == 2) ? scalar_of<ST>(1.0f, 0.0f) : scalar_of<ST>(-0.5f, 0.25f);

        std::vector<ST>              gamma(c.extras > 0 ? c.extras : 1);
        std::vector<std::vector<CT>> z(c.extras);
        for(int e = 0; e < c.extras; ++e)
        {
            gamma[e] = c.zero_gamma ? scalar_of<ST>(0.0f, 0.0f)
                                    : scalar_of<ST>(0.5f + 0.25f * e, 0.25f * (e + 1));
            z[e]     = make_vector<CT>(ys, 20 + e);
        }

        std::vector<CT> want
            = csr_gemv<CT>(Ac, c.trans, static_cast<CT>(alpha), x, static_cast<CT>(beta), y0);
        for(int e = 0; e < c.extras; ++e)
            for(int32_t j = 0; j < ys; ++j)
                want[j] = want[j] + static_cast<CT>(gamma[e]) * z[e][j];

        device_csr<AT, int32_t, int32_t> dA(A);
        device_vector<CT>                dx(x), dy(y0);
        device_vector<ST> dalpha(std::vector<ST>(1, alpha)), dbeta(std::vector<ST>(1, beta));
        device_vector<ST> dgamma(gamma);
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr && dalpha.ptr && dbeta.ptr && dgamma.ptr);
        std::deque<device_vector<CT>> dz;
        for(int e = 0; e < c.extras; ++e)
        {
            dz.emplace_back(z[e]);
            ASSERT_TRUE(dz.back().ptr);
        }

        spmat_guard mat;
        ASSERT_EQ(rocsparse_create_csr_descr(&mat.d,
                                             c.m,
                                             c.n,
                                             A.nnz(),
                                             dA.ptr,
                                             dA.ind,
                                             dA.val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             to_base(c.base),
                                             rocsparse_ut::dt_of<AT>()),
                  rocsparse_status_success);
        dnvec_guard vx, vy;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, xs, dx, rocsparse_ut::dt_of<CT>()),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, ys, dy, rocsparse_ut::dt_of<CT>()),
                  rocsparse_status_success);

        scoped_pointer_mode mode(handle, c.device);

        spmv_descr_guard descr;
        ASSERT_TRUE(descr.d);
        ASSERT_EQ(set_basic_inputs(handle,
                                   descr.d,
                                   c.alg,
                                   c.trans,
                                   rocsparse_ut::dt_of<ST>(),
                                   rocsparse_ut::dt_of<CT>()),
                  rocsparse_status_success);
        if(c.starting_ids)
            ASSERT_EQ(
                set_input(handle, descr.d, rocsparse_spmv_input_nnz_use_starting_block_ids, true),
                rocsparse_status_success);

        dnvec_guard                              vgamma;
        std::deque<dnvec_guard>                  vz;
        std::vector<rocsparse_const_dnvec_descr> zs;
        if(c.extras > 0)
        {
            ASSERT_EQ(rocsparse_create_dnvec_descr(&vgamma.d,
                                                   c.extras,
                                                   c.device ? static_cast<void*>(dgamma.ptr)
                                                            : static_cast<void*>(gamma.data()),
                                                   rocsparse_ut::dt_of<ST>()),
                      rocsparse_status_success);
            for(int e = 0; e < c.extras; ++e)
            {
                vz.emplace_back();
                ASSERT_EQ(rocsparse_create_dnvec_descr(
                              &vz.back().d, ys, dz[e].ptr, rocsparse_ut::dt_of<CT>()),
                          rocsparse_status_success);
                zs.push_back(vz.back().d);
            }
            ASSERT_EQ(
                rocsparse_spmv_set_extra(handle, descr.d, c.extras, vgamma.d, zs.data(), nullptr),
                rocsparse_status_success);
        }

        const void* palpha
            = c.device ? static_cast<const void*>(dalpha.ptr) : static_cast<const void*>(&alpha);
        const void* pbeta
            = c.device ? static_cast<const void*>(dbeta.ptr) : static_cast<const void*>(&beta);

        run_v2_spmv(handle, descr.d, mat.d, vx.d, vy.d, palpha, pbeta);
        if(::testing::Test::HasFatalFailure())
            return;
        EXPECT_TRUE(expect_near(want, to_host(dy), "v2_spmv y"));
    }

    class L2DeviceV2Spmv : public rocsparse_ut::HandleTest
    {
    protected:
        // Index base, pointer mode and (for nnzsplit) the starting block ids are
        // independent loops, crossed with the algorithm, operation and extras.
        template <typename AT, typename CT, typename ST>
        void sweep(const std::vector<rocsparse_spmv_alg>& algs, int max_extras, bool both_trans)
        {
            for(const auto alg : algs)
                for(const auto trans : {rocsparse_operation_none, rocsparse_operation_transpose})
                {
                    const bool trans_ok = (alg == rocsparse_spmv_alg_default
                                           || alg == rocsparse_spmv_alg_csr_rowsplit
                                           || alg == rocsparse_spmv_alg_csr_nnzsplit);
                    if(trans != rocsparse_operation_none && (!trans_ok || !both_trans))
                        continue;
                    const bool nnzsplit = (alg == rocsparse_spmv_alg_csr_nnzsplit);
                    for(int extras = 0; extras <= max_extras;
                        extras += (max_extras > 0 ? max_extras : 1))
                        for(const int base : {0, 1})
                            for(const bool device : {false, true})
                                for(const bool starting_ids : {false, true})
                                {
                                    if(starting_ids && !nnzsplit)
                                        continue;
                                    v2_case c;
                                    c.alg          = alg;
                                    c.trans        = trans;
                                    c.extras       = extras;
                                    c.device       = device;
                                    c.base         = base;
                                    c.starting_ids = starting_ids;
                                    run_v2<AT, CT, ST>(handle, c);
                                    if(::testing::Test::HasFailure())
                                        return;
                                }
                }
        }
    };

    const std::vector<rocsparse_spmv_alg> all_csr_algs = {rocsparse_spmv_alg_default,
                                                          rocsparse_spmv_alg_csr_rowsplit,
                                                          rocsparse_spmv_alg_csr_adaptive,
                                                          rocsparse_spmv_alg_csr_lrb,
                                                          rocsparse_spmv_alg_csr_nnzsplit};

    TEST_F(L2DeviceV2Spmv, float_algorithms_with_extras)
    {
        sweep<float, float, float>(all_csr_algs, 2, true);
    }

    TEST_F(L2DeviceV2Spmv, complex_algorithms_with_extras)
    {
        sweep<rocsparse_float_complex, rocsparse_float_complex, rocsparse_float_complex>(
            {rocsparse_spmv_alg_default,
             rocsparse_spmv_alg_csr_adaptive,
             rocsparse_spmv_alg_csr_lrb},
            2,
            false);
        sweep<rocsparse_double_complex, rocsparse_double_complex, rocsparse_double_complex>(
            {rocsparse_spmv_alg_csr_rowsplit, rocsparse_spmv_alg_csr_nnzsplit}, 1, true);
    }

    TEST_F(L2DeviceV2Spmv, scalar_type_differs_from_compute_type)
    {
        // Single precision scalars and matrix, double precision vectors.
        sweep<float, double, float>(
            {rocsparse_spmv_alg_default, rocsparse_spmv_alg_csr_adaptive}, 0, true);
        sweep<float, double, double>(
            {rocsparse_spmv_alg_csr_rowsplit, rocsparse_spmv_alg_csr_nnzsplit}, 0, true);
        sweep<rocsparse_float_complex, rocsparse_double_complex, rocsparse_double_complex>(
            {rocsparse_spmv_alg_csr_rowsplit, rocsparse_spmv_alg_csr_adaptive}, 0, false);
    }

    TEST_F(L2DeviceV2Spmv, matrix_without_entries_and_extras)
    {
        // The matrix is never read: y = beta * y + sum_i gamma_i z_i.
        for(const bool device : {false, true})
            for(const int extras : {0, 2})
                for(const auto trans : {rocsparse_operation_none, rocsparse_operation_transpose})
                {
                    v2_case c;
                    c.special = 1;
                    c.extras  = extras;
                    c.device  = device;
                    c.trans   = trans;
                    c.alg     = rocsparse_spmv_alg_csr_rowsplit;
                    run_v2<double, double, double>(handle, c);
                    if(::testing::Test::HasFailure())
                        return;
                }
    }

    TEST_F(L2DeviceV2Spmv, alpha_zero_beta_one)
    {
        // With alpha = 0 and beta = 1 (host pointers) only the extra terms act. When all
        // gamma are zero too, y must not change at all.
        for(const bool zero_gamma : {false, true})
            for(const int extras : {0, 2})
                for(const bool device : {false, true})
                {
                    v2_case c;
                    c.special    = 2;
                    c.extras     = extras;
                    c.zero_gamma = zero_gamma;
                    c.device     = device;
                    run_v2<double, double, double>(handle, c);
                    if(::testing::Test::HasFailure())
                        return;
                }
    }

    TEST_F(L2DeviceV2Spmv, stage_order_and_input_errors)
    {
        const host_csr<float>               A = make_irregular_csr<float>(12, 9, 0, false, false);
        device_csr<float, int32_t, int32_t> dA(A);
        device_vector<float> dx(make_vector<float>(9, 1)), dy(make_vector<float>(12, 2));
        ASSERT_TRUE(dA.ok() && dx.ptr && dy.ptr);
        spmat_guard mat;
        ASSERT_EQ(rocsparse_create_csr_descr(&mat.d,
                                             12,
                                             9,
                                             A.nnz(),
                                             dA.ptr,
                                             dA.ind,
                                             dA.val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        dnvec_guard vx, vy;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, 9, dx, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, 12, dy, rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        spmv_descr_guard descr;
        ASSERT_TRUE(descr.d);
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        size_t      bytes = 0;

        auto spmv = [&](rocsparse_v2_spmv_stage stage) {
            return rocsparse_v2_spmv(
                handle, descr.d, &alpha, mat.d, vx.d, &beta, vy.d, stage, 0, nullptr, nullptr);
        };

        // Inputs that were never set.
        EXPECT_NE(rocsparse_v2_spmv_buffer_size(handle,
                                                descr.d,
                                                mat.d,
                                                vx.d,
                                                vy.d,
                                                rocsparse_v2_spmv_stage_analysis,
                                                &bytes,
                                                nullptr),
                  rocsparse_status_success);
        EXPECT_NE(spmv(rocsparse_v2_spmv_stage_analysis), rocsparse_status_success);

        // set_input argument validation.
        const auto alg = rocsparse_spmv_alg_default;
        EXPECT_EQ(rocsparse_spmv_set_input(
                      nullptr, descr.d, rocsparse_spmv_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_spmv_set_input(
                      handle, nullptr, rocsparse_spmv_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            rocsparse_spmv_set_input(
                handle, descr.d, static_cast<rocsparse_spmv_input>(77), &alg, sizeof(alg), nullptr),
            rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_spmv_set_input(
                      handle, descr.d, rocsparse_spmv_input_alg, nullptr, sizeof(alg), nullptr),
                  rocsparse_status_invalid_pointer);
        for(const auto input : {rocsparse_spmv_input_alg,
                                rocsparse_spmv_input_operation,
                                rocsparse_spmv_input_scalar_datatype,
                                rocsparse_spmv_input_compute_datatype,
                                rocsparse_spmv_input_nnz_use_starting_block_ids,
                                rocsparse_spmv_input_enable_extra})
        {
            // a bool is one byte, so use another wrong size for that input
            const size_t wrong_size
                = (input == rocsparse_spmv_input_nnz_use_starting_block_ids) ? 4 : 1;
            EXPECT_NE(rocsparse_spmv_set_input(handle, descr.d, input, &alg, wrong_size, nullptr),
                      rocsparse_status_success);
        }
        // enable_extra needs extras to be set first.
        const int32_t enable = 1;
        EXPECT_EQ(set_input(handle, descr.d, rocsparse_spmv_input_enable_extra, enable),
                  rocsparse_status_invalid_value);

        ASSERT_EQ(set_basic_inputs(handle,
                                   descr.d,
                                   rocsparse_spmv_alg_csr_rowsplit,
                                   rocsparse_operation_none,
                                   rocsparse_datatype_f32_r,
                                   rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        // Compute before analysis.
        EXPECT_NE(spmv(rocsparse_v2_spmv_stage_compute), rocsparse_status_success);
        EXPECT_EQ(spmv(rocsparse_v2_spmv_stage_analysis), rocsparse_status_success);
        // Analysis twice.
        EXPECT_NE(spmv(rocsparse_v2_spmv_stage_analysis), rocsparse_status_success);
        EXPECT_EQ(spmv(rocsparse_v2_spmv_stage_compute), rocsparse_status_success);
        // Analysis after compute.
        EXPECT_NE(spmv(rocsparse_v2_spmv_stage_analysis), rocsparse_status_success);
        // The algorithm and the operation cannot change once the analysis ran.
        EXPECT_NE(set_input(handle, descr.d, rocsparse_spmv_input_alg, alg),
                  rocsparse_status_success);
        const auto trans = rocsparse_operation_transpose;
        EXPECT_NE(set_input(handle, descr.d, rocsparse_spmv_input_operation, trans),
                  rocsparse_status_success);

        // Invalid stage and pointers of the compute entry points.
        {
            enum
            {
                ALPHA  = 2,
                BETA   = 5,
                STAGE  = 7,
                BYTES  = 8,
                BUFFER = 9
            };
            const arg_checker bad([](auto... a) { return rocsparse_v2_spmv(a...); },
                                  handle,
                                  descr.d,
                                  &alpha,
                                  mat.d,
                                  vx.d,
                                  &beta,
                                  vy.d,
                                  rocsparse_v2_spmv_stage_compute,
                                  size_t(0),
                                  static_cast<void*>(nullptr),
                                  static_cast<rocsparse_error*>(nullptr));
            bad.check<STAGE>(static_cast<rocsparse_v2_spmv_stage>(77),
                             rocsparse_status_invalid_value);
            bad.check<ALPHA>(nullptr, rocsparse_status_invalid_pointer);
            bad.check<BETA>(nullptr, rocsparse_status_invalid_pointer);
            // a buffer without a size, and a size without a buffer
            bad.check_fails<BUFFER>(static_cast<void*>(dy.ptr));
            bad.check_fails<BYTES>(size_t(8));
        }
        EXPECT_EQ(rocsparse_v2_spmv_buffer_size(handle,
                                                descr.d,
                                                mat.d,
                                                vx.d,
                                                vy.d,
                                                static_cast<rocsparse_v2_spmv_stage>(77),
                                                &bytes,
                                                nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_v2_spmv_buffer_size(handle,
                                                descr.d,
                                                mat.d,
                                                vx.d,
                                                vy.d,
                                                rocsparse_v2_spmv_stage_compute,
                                                nullptr,
                                                nullptr),
                  rocsparse_status_invalid_pointer);

        // The result of the successful compute is the product A x.
        const std::vector<float> want = csr_gemv<float>(A,
                                                        rocsparse_operation_none,
                                                        1.0f,
                                                        make_vector<float>(9, 1),
                                                        0.0f,
                                                        make_vector<float>(12, 2));
        EXPECT_TRUE(expect_near(want, to_host(dy), "v2_spmv y"));
    }

    TEST_F(L2DeviceV2Spmv, set_extra_errors)
    {
        const int32_t         ys = 12;
        device_vector<double> gamma_dev(std::vector<double>(2, 1.0)),
            z0(make_vector<double>(ys, 1)), z1(make_vector<double>(ys, 2)),
            z_short(make_vector<double>(ys - 1, 3));
        device_vector<float> z_f32(make_vector<float>(ys, 4)),
            gamma_f32(std::vector<float>(2, 1.0f));
        ASSERT_TRUE(gamma_dev.ptr && z0.ptr && z1.ptr && z_short.ptr && z_f32.ptr && gamma_f32.ptr);

        const auto  f64 = rocsparse_datatype_f64_r;
        dnvec_guard g1, g2, a, b, shorter, single, gf32;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&g1.d, 1, gamma_dev, f64), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&g2.d, 2, gamma_dev, f64), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&a.d, ys, z0, f64), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&b.d, ys, z1, f64), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&shorter.d, ys - 1, z_short, f64),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&single.d, ys, z_f32, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&gf32.d, 1, gamma_f32, rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        spmv_descr_guard descr;
        ASSERT_TRUE(descr.d);
        rocsparse_const_dnvec_descr zs[2] = {a.d, b.d};

        // Datatypes not set yet.
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 2, g2.d, zs, nullptr),
                  rocsparse_status_invalid_value);
        ASSERT_EQ(
            set_basic_inputs(
                handle, descr.d, rocsparse_spmv_alg_default, rocsparse_operation_none, f64, f64),
            rocsparse_status_success);

        EXPECT_EQ(rocsparse_spmv_set_extra(nullptr, descr.d, 2, g2.d, zs, nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, nullptr, 2, g2.d, zs, nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 0, g2.d, zs, nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 2, nullptr, zs, nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 2, g2.d, nullptr, nullptr),
                  rocsparse_status_invalid_pointer);
        // gamma with the wrong datatype or size.
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 1, gf32.d, zs, nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 2, g1.d, zs, nullptr),
                  rocsparse_status_invalid_size);
        // z vectors with a null entry, the wrong datatype or different sizes.
        rocsparse_const_dnvec_descr z_null[2] = {a.d, nullptr};
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 2, g2.d, z_null, nullptr),
                  rocsparse_status_invalid_pointer);
        rocsparse_const_dnvec_descr z_type[1] = {single.d};
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 1, g1.d, z_type, nullptr),
                  rocsparse_status_invalid_value);
        rocsparse_const_dnvec_descr z_size[2] = {a.d, shorter.d};
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 2, g2.d, z_size, nullptr),
                  rocsparse_status_invalid_size);

        // Valid extras, then toggled with enable_extra and removed again.
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 2, g2.d, zs, nullptr),
                  rocsparse_status_success);
        for(const int32_t enable : {0, 1})
            EXPECT_EQ(set_input(handle, descr.d, rocsparse_spmv_input_enable_extra, enable),
                      rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmv_clear_extra(handle, descr.d, nullptr), rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmv_clear_extra(nullptr, descr.d, nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_spmv_clear_extra(handle, nullptr, nullptr),
                  rocsparse_status_invalid_pointer);

        // Different scalar and compute datatypes do not support extras.
        spmv_descr_guard mixed;
        ASSERT_TRUE(mixed.d);
        ASSERT_EQ(set_basic_inputs(handle,
                                   mixed.d,
                                   rocsparse_spmv_alg_default,
                                   rocsparse_operation_none,
                                   rocsparse_datatype_f32_r,
                                   f64),
                  rocsparse_status_success);
        rocsparse_const_dnvec_descr z_one[1] = {a.d};
        EXPECT_NE(rocsparse_spmv_set_extra(handle, mixed.d, 1, gf32.d, z_one, nullptr),
                  rocsparse_status_success);
    }

    // ----- algorithm / format validation -----------------------------------------------------
    enum class mv_format
    {
        csr,
        csc,
        coo,
        coo_aos,
        ell,
        bsr,
        bell,
        sell
    };

    bool alg_valid_for(mv_format f, rocsparse_spmv_alg alg)
    {
        if(alg == rocsparse_spmv_alg_default)
            return f != mv_format::bell;
        switch(f)
        {
        case mv_format::csr:
        case mv_format::csc:
            return alg == rocsparse_spmv_alg_csr_rowsplit || alg == rocsparse_spmv_alg_csr_adaptive
                   || alg == rocsparse_spmv_alg_csr_lrb || alg == rocsparse_spmv_alg_csr_nnzsplit;
        case mv_format::coo:
        case mv_format::coo_aos:
            return alg == rocsparse_spmv_alg_coo || alg == rocsparse_spmv_alg_coo_atomic;
        case mv_format::ell:
            return alg == rocsparse_spmv_alg_ell;
        case mv_format::bsr:
            return alg == rocsparse_spmv_alg_bsr;
        case mv_format::sell:
            return alg == rocsparse_spmv_alg_sell;
        case mv_format::bell:
            return false;
        }
        return false;
    }

    class L2DeviceSpmvAlgs : public rocsparse_ut::HandleTest
    {
    protected:
        static constexpr int32_t dim = 4;

        // Builds a small descriptor of each format. The arrays are valid device
        // memory but only descriptors are inspected by the stages used below.
        bool make_descriptor(mv_format f, spmat_guard& mat)
        {
            idx = std::make_unique<device_vector<int32_t>>(std::vector<int32_t>(64, 0));
            val = std::make_unique<device_vector<float>>(std::vector<float>(64, 1.0f));
            if(!idx->ptr || !val->ptr)
                return false;
            const auto it = rocsparse_indextype_i32;
            const auto b  = rocsparse_index_base_zero;
            const auto dt = rocsparse_datatype_f32_r;
            switch(f)
            {
            case mv_format::csr:
                return rocsparse_create_csr_descr(
                           &mat.d, dim, dim, dim, idx->ptr, idx->ptr, val->ptr, it, it, b, dt)
                       == rocsparse_status_success;
            case mv_format::csc:
                return rocsparse_create_csc_descr(
                           &mat.d, dim, dim, dim, idx->ptr, idx->ptr, val->ptr, it, it, b, dt)
                       == rocsparse_status_success;
            case mv_format::coo:
                return rocsparse_create_coo_descr(
                           &mat.d, dim, dim, dim, idx->ptr, idx->ptr, val->ptr, it, b, dt)
                       == rocsparse_status_success;
            case mv_format::coo_aos:
                return rocsparse_create_coo_aos_descr(
                           &mat.d, dim, dim, dim, idx->ptr, val->ptr, it, b, dt)
                       == rocsparse_status_success;
            case mv_format::ell:
                return rocsparse_create_ell_descr(
                           &mat.d, dim, dim, idx->ptr, val->ptr, 1, it, b, dt)
                       == rocsparse_status_success;
            case mv_format::bsr:
                return rocsparse_create_bsr_descr(&mat.d,
                                                  2,
                                                  2,
                                                  2,
                                                  rocsparse_direction_row,
                                                  2,
                                                  idx->ptr,
                                                  idx->ptr,
                                                  val->ptr,
                                                  it,
                                                  it,
                                                  b,
                                                  dt)
                       == rocsparse_status_success;
            case mv_format::bell:
                return rocsparse_create_bell_descr(&mat.d,
                                                   dim,
                                                   dim,
                                                   rocsparse_direction_row,
                                                   2,
                                                   1,
                                                   idx->ptr,
                                                   val->ptr,
                                                   it,
                                                   b,
                                                   dt)
                       == rocsparse_status_success;
            case mv_format::sell:
                return rocsparse_create_sell_descr(
                           &mat.d, dim, dim, dim, 2, 4, idx->ptr, idx->ptr, val->ptr, it, it, b, dt)
                       == rocsparse_status_success;
            }
            return false;
        }

        std::unique_ptr<device_vector<int32_t>> idx;
        std::unique_ptr<device_vector<float>>   val;
    };

    const std::vector<rocsparse_spmv_alg> every_alg = {rocsparse_spmv_alg_default,
                                                       rocsparse_spmv_alg_csr_rowsplit,
                                                       rocsparse_spmv_alg_csr_adaptive,
                                                       rocsparse_spmv_alg_csr_lrb,
                                                       rocsparse_spmv_alg_csr_nnzsplit,
                                                       rocsparse_spmv_alg_coo,
                                                       rocsparse_spmv_alg_coo_atomic,
                                                       rocsparse_spmv_alg_ell,
                                                       rocsparse_spmv_alg_sell,
                                                       rocsparse_spmv_alg_bsr};

    TEST_F(L2DeviceSpmvAlgs, spmv_buffer_size_checks_algorithm_per_format)
    {
        device_vector<float> dx(std::vector<float>(dim, 1.0f)), dy(std::vector<float>(dim, 1.0f));
        ASSERT_TRUE(dx.ptr && dy.ptr);
        dnvec_guard vx, vy;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, dim, dx, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, dim, dy, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        const float alpha = 1.0f;
        const float beta  = 0.0f;

        for(const auto f : {mv_format::csr,
                            mv_format::csc,
                            mv_format::coo,
                            mv_format::coo_aos,
                            mv_format::ell,
                            mv_format::bsr,
                            mv_format::bell,
                            mv_format::sell})
        {
            spmat_guard mat;
            ASSERT_TRUE(make_descriptor(f, mat)) << "format " << static_cast<int>(f);
            for(const auto alg : every_alg)
            {
                SCOPED_TRACE(::testing::Message()
                             << "format " << static_cast<int>(f) << " alg " << alg);
                size_t                 bytes = 99;
                const rocsparse_status st    = rocsparse_spmv(handle,
                                                           rocsparse_operation_none,
                                                           &alpha,
                                                           mat.d,
                                                           vx.d,
                                                           &beta,
                                                           vy.d,
                                                           rocsparse_datatype_f32_r,
                                                           alg,
                                                           rocsparse_spmv_stage_buffer_size,
                                                           &bytes,
                                                           nullptr);
                if(!alg_valid_for(f, alg))
                {
                    EXPECT_EQ(st, rocsparse_status_invalid_value);
                }
                else if(f == mv_format::sell)
                {
                    // Sliced ELL is only available through rocsparse_v2_spmv.
                    EXPECT_EQ(st, rocsparse_status_not_implemented);
                }
                else
                {
                    EXPECT_EQ(st, rocsparse_status_success);
                }
            }
        }
    }

    TEST_F(L2DeviceSpmvAlgs, v2_spmv_analysis_rejects_algorithm_of_other_format)
    {
        device_vector<float> dx(std::vector<float>(dim, 1.0f)), dy(std::vector<float>(dim, 1.0f));
        ASSERT_TRUE(dx.ptr && dy.ptr);
        dnvec_guard vx, vy;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, dim, dx, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, dim, dy, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        const float alpha = 1.0f;
        const float beta  = 0.0f;

        for(const auto f : {mv_format::csr,
                            mv_format::csc,
                            mv_format::coo,
                            mv_format::coo_aos,
                            mv_format::ell,
                            mv_format::bsr,
                            mv_format::bell,
                            mv_format::sell})
        {
            spmat_guard mat;
            ASSERT_TRUE(make_descriptor(f, mat)) << "format " << static_cast<int>(f);
            for(const auto alg : every_alg)
            {
                if(alg_valid_for(f, alg))
                    continue; // the valid combinations are exercised with real data elsewhere
                SCOPED_TRACE(::testing::Message()
                             << "format " << static_cast<int>(f) << " alg " << alg);
                spmv_descr_guard descr;
                ASSERT_TRUE(descr.d);
                ASSERT_EQ(set_basic_inputs(handle,
                                           descr.d,
                                           alg,
                                           rocsparse_operation_none,
                                           rocsparse_datatype_f32_r,
                                           rocsparse_datatype_f32_r),
                          rocsparse_status_success);
                EXPECT_EQ(rocsparse_v2_spmv(handle,
                                            descr.d,
                                            &alpha,
                                            mat.d,
                                            vx.d,
                                            &beta,
                                            vy.d,
                                            rocsparse_v2_spmv_stage_analysis,
                                            0,
                                            nullptr,
                                            nullptr),
                          rocsparse_status_invalid_value);
            }
        }
    }
} // namespace
