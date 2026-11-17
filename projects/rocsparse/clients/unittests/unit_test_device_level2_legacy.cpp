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
// Device unit tests of the legacy COO, ELL and HYB matrix vector products and of
// the dense matrix sparse vector product (gemvi). Every result is compared with
// a host reference computed from the CSR form of the matrix.
//
#include "unit_test_device_level2_common.hpp"

#include <set>

using namespace level2;

namespace
{
    // A device vector of an empty host vector has no valid address: keep one element.
    template <typename V>
    std::vector<V> nonempty(std::vector<V> v)
    {
        if(v.empty())
            v.push_back(V{});
        return v;
    }

    // Rows of 3, 4 or 5 entries (the longest row is below twice the average, as the
    // HYB conversion requires for the maximum partition), over distinct columns
    // when n is a prime above 40.
    template <typename T>
    host_csr<T> make_narrow(int32_t m, int32_t n, int base)
    {
        host_csr<T> a;
        a.m    = m;
        a.n    = n;
        a.base = 0;
        a.ptr.push_back(0);
        int64_t counter = 0;
        for(int32_t i = 0; i < m; ++i)
        {
            const int32_t     len = 3 + (i % 3);
            std::set<int32_t> cols;
            for(int32_t k = 0; k < len; ++k)
                cols.insert((i + 7 * k) % n);
            for(int32_t c : cols)
            {
                a.ind.push_back(c);
                a.val.push_back(value_of<T>(counter++, true));
            }
            a.ptr.push_back(static_cast<int32_t>(a.ind.size()));
        }
        rebase(a, base);
        return a;
    }

    struct scalars
    {
        float alpha;
        float beta;
    };

    // Scalars that exercise the general path, alpha == 0, beta == 0 and the
    // alpha == 0 && beta == 1 early return.
    const scalars all_scalars[]
        = {{1.5f, -0.5f}, {0.0f, 2.0f}, {-1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};

    enum class kind
    {
        coo,
        ell,
        hyb
    };

    struct legacy_case
    {
        kind                    k      = kind::coo;
        rocsparse_operation     trans  = rocsparse_operation_none;
        scalars                 s      = {1.5f, -0.5f};
        bool                    device = false;
        rocsparse_hyb_partition part   = rocsparse_hyb_partition_auto;
        int32_t                 width  = 0;
    };

    template <typename T>
    void run_legacy(rocsparse_handle handle, const host_csr<T>& A, const legacy_case& c)
    {
        SCOPED_TRACE(::testing::Message()
                     << "kind=" << static_cast<int>(c.k) << " trans=" << c.trans
                     << " alpha=" << c.s.alpha << " beta=" << c.s.beta << " device=" << c.device
                     << " base=" << A.base << " part=" << c.part << " width=" << c.width
                     << " m=" << A.m << " n=" << A.n << " nnz=" << A.nnz());

        const int32_t m  = A.m;
        const int32_t n  = A.n;
        const size_t  xs = trans_size(c.trans, m, n, true);
        const size_t  ys = trans_size(c.trans, m, n, false);

        const T alpha = case_scalar<T>(c.s.alpha, 0.0f);
        const T beta  = case_scalar<T>(c.s.beta, 1.0f);

        const std::vector<T> x    = make_vector<T>(xs, 2);
        const std::vector<T> y0   = make_vector<T>(ys, 9);
        const std::vector<T> want = csr_gemv(A, c.trans, alpha, x, beta, y0);

        device_vector<T> dx(nonempty(x)), dy(nonempty(y0)), dalpha(std::vector<T>(1, alpha)),
            dbeta(std::vector<T>(1, beta));
        ASSERT_TRUE(dx.ptr && dy.ptr && dalpha.ptr && dbeta.ptr);

        mat_descr descr;
        ASSERT_TRUE(descr.d);
        ASSERT_EQ(rocsparse_set_mat_index_base(descr.d, to_base(A.base)), rocsparse_status_success);

        scoped_pointer_mode mode(handle, c.device);
        const T*            palpha = c.device ? dalpha.ptr : &alpha;
        const T*            pbeta  = c.device ? dbeta.ptr : &beta;

        switch(c.k)
        {
        case kind::coo:
        {
            const host_coo<T>            coo = to_coo(A);
            device_vector<rocsparse_int> drow(convert_vector<rocsparse_int>(nonempty(coo.row))),
                dcol(convert_vector<rocsparse_int>(nonempty(coo.col)));
            device_vector<T> dval(nonempty(coo.val));
            ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr);
            ASSERT_EQ(coomv<T>(handle,
                               c.trans,
                               m,
                               n,
                               static_cast<rocsparse_int>(coo.val.size()),
                               palpha,
                               descr.d,
                               dval.ptr,
                               drow.ptr,
                               dcol.ptr,
                               dx.ptr,
                               pbeta,
                               dy.ptr),
                      rocsparse_status_success);
            break;
        }
        case kind::ell:
        {
            const host_ell<T>            ell = to_ell(A);
            device_vector<rocsparse_int> dcol(convert_vector<rocsparse_int>(nonempty(ell.col)));
            device_vector<T>             dval(nonempty(ell.val));
            ASSERT_TRUE(dcol.ptr && dval.ptr);
            ASSERT_EQ(ellmv<T>(handle,
                               c.trans,
                               m,
                               n,
                               palpha,
                               descr.d,
                               dval.ptr,
                               dcol.ptr,
                               ell.width,
                               dx.ptr,
                               pbeta,
                               dy.ptr),
                      rocsparse_status_success);
            break;
        }
        case kind::hyb:
        {
            device_csr<T> dA(A);
            ASSERT_TRUE(dA.ok());
            hyb_guard hyb;
            ASSERT_TRUE(hyb.d);
            // rocsparse_csr2hyb.h does not say whether a user ELL width of 0 is valid; the
            // library accepts it.
            ASSERT_EQ(
                csr2hyb<T>(handle, m, n, descr.d, dA.val, dA.ptr, dA.ind, hyb.d, c.width, c.part),
                rocsparse_status_success);
            ASSERT_EQ(hybmv<T>(handle, c.trans, palpha, descr.d, hyb.d, dx.ptr, pbeta, dy.ptr),
                      rocsparse_status_success);
            break;
        }
        }
        std::vector<T> got = to_host(dy);
        got.resize(want.size());
        EXPECT_TRUE(expect_near(want, got, "y"));
    }

    class L2DeviceLegacy : public rocsparse_ut::HandleTest
    {
    protected:
        template <typename T>
        void sweep(kind                    k,
                   const host_csr<T>&      base0,
                   rocsparse_hyb_partition part  = rocsparse_hyb_partition_auto,
                   int32_t                 width = 0)
        {
            int n = 0;
            for(const auto trans : all_ops)
                for(const auto s : all_scalars)
                {
                    legacy_case c;
                    c.k      = k;
                    c.trans  = trans;
                    c.s      = s;
                    c.device = ((n++) % 3) == 1;
                    c.part   = part;
                    c.width  = width;
                    run_legacy<T>(handle, base0, c);
                    if(HasFailure())
                        return;
                }
        }
    };

    TEST_F(L2DeviceLegacy, coomv_long_row_and_empty)
    {
        sweep<float>(kind::coo, make_matrix<float>(40, 3000, 0, 2800, 0));
        // No entries: only the scaling of y remains.
        sweep<float>(kind::coo, make_matrix<float>(40, 30, 0, 0, 1));
        // No rows or no columns.
        sweep<float>(kind::coo, make_matrix<float>(0, 30, 0, 0, 0));
        sweep<float>(kind::coo, make_matrix<float>(20, 0, 0, 0, 0));
    }

    TEST_F(L2DeviceLegacy, ellmv_empty)
    {
        // Width 0: nothing to multiply, y = beta * y.
        sweep<float>(kind::ell, make_matrix<float>(40, 30, 0, 0, 1));
        sweep<float>(kind::ell, make_matrix<float>(0, 30, 0, 0, 0));
        sweep<float>(kind::ell, make_matrix<float>(20, 0, 0, 0, 0));
    }

    TEST_F(L2DeviceLegacy, coomv_ellmv_other_precisions)
    {
        for(const auto k : {kind::coo, kind::ell})
        {
            sweep<double>(k, make_matrix<double>(110, 70, 0, 0, 5));
            sweep<rocsparse_float_complex>(k,
                                           make_matrix<rocsparse_float_complex>(90, 100, 1, 0, 0));
            sweep<rocsparse_double_complex>(k,
                                            make_matrix<rocsparse_double_complex>(90, 60, 0, 0, 6));
        }
    }

    TEST_F(L2DeviceLegacy, hybmv_partitions_float)
    {
        const auto A = make_matrix<float>(120, 90, 0, 0, 7);
        sweep<float>(kind::hyb, A, rocsparse_hyb_partition_auto, 0);
        // The ELL part holds the first entries of each row, the COO part the rest.
        sweep<float>(kind::hyb, A, rocsparse_hyb_partition_user, 3);
        sweep<float>(kind::hyb, A, rocsparse_hyb_partition_user, 6);
        sweep<float>(
            kind::hyb, make_matrix<float>(100, 80, 1, 0, 0), rocsparse_hyb_partition_user, 2);
        // ELL only.
        const auto N = make_narrow<float>(100, 89, 1);
        sweep<float>(kind::hyb, N, rocsparse_hyb_partition_max, 0);
        sweep<float>(kind::hyb, N, rocsparse_hyb_partition_user, 5);
        // COO only: a user ELL width of 0.
        sweep<float>(kind::hyb, N, rocsparse_hyb_partition_user, 0);
    }

    TEST_F(L2DeviceLegacy, hybmv_double_and_complex)
    {
        sweep<double>(
            kind::hyb, make_matrix<double>(110, 70, 0, 0, 5), rocsparse_hyb_partition_user, 4);
        sweep<rocsparse_float_complex>(kind::hyb,
                                       make_matrix<rocsparse_float_complex>(90, 100, 1, 0, 0),
                                       rocsparse_hyb_partition_user,
                                       2);
        sweep<rocsparse_double_complex>(kind::hyb,
                                        make_matrix<rocsparse_double_complex>(90, 60, 0, 0, 6),
                                        rocsparse_hyb_partition_auto,
                                        0);
    }

    TEST_F(L2DeviceLegacy, hybmv_empty)
    {
        sweep<float>(
            kind::hyb, make_matrix<float>(40, 30, 0, 0, 1), rocsparse_hyb_partition_auto, 0);
        // Neither an ELL nor a COO part: no entries and a user ELL width of 0.
        sweep<float>(
            kind::hyb, make_matrix<float>(40, 30, 0, 0, 1), rocsparse_hyb_partition_user, 0);
    }

    // ----- gemvi --------------------------------------------------------------------------

    template <typename T>
    void run_gemvi(rocsparse_handle handle,
                   int32_t          m,
                   int32_t          n,
                   int32_t          lda_extra,
                   int32_t          nnz,
                   int              base,
                   scalars          s,
                   bool             device)
    {
        SCOPED_TRACE(::testing::Message()
                     << "gemvi m=" << m << " n=" << n << " nnz=" << nnz << " base=" << base
                     << " alpha=" << s.alpha << " beta=" << s.beta << " device=" << device);

        const int32_t  lda = m + lda_extra;
        std::vector<T> A(static_cast<size_t>(lda) * n);
        for(size_t k = 0; k < A.size(); ++k)
            A[k] = value_of<T>(static_cast<int64_t>(k));

        // nnz distinct columns spread over [0, n), in decreasing order to cover an
        // unsorted sparse vector.
        std::vector<int32_t> ind(nnz);
        std::vector<T>       xval(nnz);
        for(int32_t k = 0; k < nnz; ++k)
        {
            ind[k]  = static_cast<int32_t>((static_cast<int64_t>(nnz - 1 - k) * n) / nnz);
            xval[k] = value_of<T>(k + 3, true);
        }
        std::vector<int32_t> dev_ind = ind;
        for(auto& i : dev_ind)
            i += base;

        const T alpha = case_scalar<T>(s.alpha, 0.0f);
        const T beta  = case_scalar<T>(s.beta, 1.0f);

        const std::vector<T> y0 = make_vector<T>(m, 9);
        std::vector<T>       want(m);
        for(int32_t i = 0; i < m; ++i)
        {
            T acc{};
            for(int32_t k = 0; k < nnz; ++k)
                acc = acc + A[static_cast<size_t>(ind[k]) * lda + i] * xval[k];
            want[i] = alpha * acc + beta * y0[i];
        }

        if(xval.empty())
        {
            xval.push_back(T{});
            dev_ind.push_back(base);
        }
        device_vector<T> dA(A), dx(xval), dy(y0), dalpha(std::vector<T>(1, alpha)),
            dbeta(std::vector<T>(1, beta));
        device_vector<rocsparse_int> dind(convert_vector<rocsparse_int>(dev_ind));
        ASSERT_TRUE(dA.ptr && dx.ptr && dy.ptr && dalpha.ptr && dbeta.ptr && dind.ptr);

        scoped_pointer_mode mode(handle, device);

        size_t bytes = 0;
        ASSERT_EQ(gemvi_buffer_size<T>(handle, rocsparse_operation_none, m, n, nnz, &bytes),
                  rocsparse_status_success);
        device_vector<char> dbuf(std::max<size_t>(bytes, 1));
        ASSERT_TRUE(dbuf.ptr);
        ASSERT_EQ(gemvi<T>(handle,
                           rocsparse_operation_none,
                           m,
                           n,
                           device ? dalpha.ptr : &alpha,
                           dA.ptr,
                           lda,
                           nnz,
                           dx.ptr,
                           dind.ptr,
                           device ? dbeta.ptr : &beta,
                           dy.ptr,
                           to_base(base),
                           dbuf.ptr),
                  rocsparse_status_success);
        EXPECT_TRUE(expect_near(want, to_host(dy), "gemvi y"));
    }

    TEST_F(L2DeviceLegacy, gemvi_float_shapes)
    {
        // Short sparse vectors (one wavefront), medium ones, and long ones that
        // split the sparse vector over several workers.
        struct shape
        {
            int32_t m, n, extra, nnz;
        };
        const shape shapes[] = {{33, 40, 0, 5},
                                {100, 300, 7, 100},
                                {70, 700, 0, 600},
                                {128, 3000, 3, 3000},
                                {200, 2200, 0, 1100},
                                {40, 5000, 0, 2500}};
        int         n        = 0;
        for(const auto& sh : shapes)
            for(const auto s : all_scalars)
            {
                run_gemvi<float>(handle, sh.m, sh.n, sh.extra, sh.nnz, n % 2, s, (n % 3) == 1);
                ++n;
                if(HasFailure())
                    return;
            }
    }

    TEST_F(L2DeviceLegacy, gemvi_many_rows)
    {
        // More thread blocks over the rows than the device can hold at once.
        run_gemvi<float>(handle, 9000, 1100, 0, 1100, 0, {1.5f, -0.5f}, false);
        run_gemvi<float>(handle, 9000, 1100, 5, 1100, 1, {1.0f, 0.0f}, true);
    }

    TEST_F(L2DeviceLegacy, gemvi_other_precisions)
    {
        run_gemvi<double>(handle, 100, 300, 7, 100, 0, {1.5f, -0.5f}, false);
        run_gemvi<rocsparse_float_complex>(handle, 70, 700, 0, 600, 1, {1.5f, -0.5f}, true);
        run_gemvi<rocsparse_double_complex>(handle, 33, 40, 3, 5, 0, {1.0f, 0.0f}, false);
    }

    TEST_F(L2DeviceLegacy, gemvi_zero_rows)
    {
        // Nothing to compute without rows.
        size_t bytes = 1;
        EXPECT_EQ(gemvi_buffer_size<float>(handle, rocsparse_operation_none, 0, 5, 0, &bytes),
                  rocsparse_status_success);
        EXPECT_EQ(bytes, 0u);
        const float alpha = 1.0f;
        const float beta  = 2.0f;
        EXPECT_EQ(gemvi<float>(handle,
                               rocsparse_operation_none,
                               0,
                               5,
                               &alpha,
                               nullptr,
                               1,
                               0,
                               nullptr,
                               nullptr,
                               &beta,
                               nullptr,
                               rocsparse_index_base_zero,
                               nullptr),
                  rocsparse_status_success);
    }

    TEST_F(L2DeviceLegacy, gemvi_empty_vector_scales_y)
    {
        // No entries in the sparse vector (or no rows or columns): y = beta * y.
        for(const bool device : {false, true})
        {
            run_gemvi<float>(handle, 50, 60, 0, 0, 0, {1.5f, -0.5f}, device);
            if(HasFailure())
                return;
        }
    }
}
