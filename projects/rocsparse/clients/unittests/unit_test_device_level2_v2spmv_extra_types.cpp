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
// Device unit tests of the extra terms of rocsparse_v2_spmv with a CSR matrix,
//   y = alpha * op(A) * x + beta * y + sum_i gamma_i * z_i,
// for every A / X / Y / compute type combination of the v2 SpMV type table. gamma
// has the scalar type and z the compute type, which differs from the type of y for
// the half precision outputs. Every CSR algorithm, both index widths, both index
// bases and both pointer modes are crossed with 0, 1 and 3 extra terms. The data
// are small integers, so the host reference in the compute type is exact.
//
#include "unit_test_device_level2_common.hpp"

#include <cstring>
#include <deque>
#include <limits>
#include <string>
#include <type_traits>

using namespace level2;
using rocsparse_ut::dt_of;

namespace
{
    template <typename T>
    constexpr bool is_integer()
    {
        return std::is_same<T, int8_t>::value || std::is_same<T, int32_t>::value;
    }

    template <typename T>
    T make(int re, int im)
    {
        if constexpr(is_complex_v<T>)
            return T(re, im);
        else if constexpr(std::is_same<T, rocsparse_bfloat16>::value)
            return rocsparse_bfloat16(static_cast<float>(re));
        else
            return static_cast<T>(re);
    }

    template <typename T>
    double real_part(const T& v)
    {
        if constexpr(is_complex_v<T>)
            return static_cast<double>(std::real(v));
        else if constexpr(std::is_same<T, rocsparse_bfloat16>::value)
            return static_cast<double>(static_cast<float>(v));
        else
            return static_cast<double>(v);
    }

    template <typename T>
    double imag_part(const T& v)
    {
        if constexpr(is_complex_v<T>)
            return static_cast<double>(std::imag(v));
        else
            return 0.0;
    }

    template <typename To, typename From>
    To convert(const From& v)
    {
        using R = typename real_of<To>::type;
        if constexpr(is_complex_v<To>)
            return To(static_cast<R>(real_part(v)), static_cast<R>(imag_part(v)));
        else if constexpr(std::is_same<To, rocsparse_bfloat16>::value)
            return rocsparse_bfloat16(static_cast<float>(real_part(v)));
        else
            return static_cast<To>(real_part(v));
    }

    // Small integers in [-2, 2] (and [-1, 1] for the imaginary part).
    template <typename T>
    T value_at(int64_t n)
    {
        const int re = static_cast<int>((n * 37 + 11) % 5) - 2;
        const int im = static_cast<int>((n * 23 + 5) % 3) - 1;
        return make<T>(re, im);
    }

    template <typename A_, typename X_, typename Y_, typename T_>
    struct combo
    {
        using A = A_;
        using X = X_;
        using Y = Y_;
        using T = T_;
    };

    // The A / X / Y / compute type rows of the v2 SpMV type table. The scalar (alpha,
    // beta, gamma) type is the compute type, the only combination rocsparse_spmv_set_extra
    // accepts.
    using all_combos = ::testing::Types<
        combo<float, float, float, float>,
        combo<double, double, double, double>,
        combo<rocsparse_float_complex,
              rocsparse_float_complex,
              rocsparse_float_complex,
              rocsparse_float_complex>,
        combo<rocsparse_double_complex,
              rocsparse_double_complex,
              rocsparse_double_complex,
              rocsparse_double_complex>,
        combo<int8_t, int8_t, int32_t, int32_t>,
        combo<int8_t, int8_t, float, float>,
        combo<_Float16, _Float16, float, float>,
        combo<_Float16, _Float16, _Float16, float>,
        combo<rocsparse_bfloat16, rocsparse_bfloat16, float, float>,
        combo<rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16, float>,
        combo<float, double, double, double>,
        combo<rocsparse_float_complex,
              rocsparse_double_complex,
              rocsparse_double_complex,
              rocsparse_double_complex>,
        combo<float, rocsparse_float_complex, rocsparse_float_complex, rocsparse_float_complex>,
        combo<double,
              rocsparse_double_complex,
              rocsparse_double_complex,
              rocsparse_double_complex>>;

    template <typename T>
    std::string type_name()
    {
        switch(dt_of<T>())
        {
        case rocsparse_datatype_i8_r:
            return "i8";
        case rocsparse_datatype_i32_r:
            return "i32";
        case rocsparse_datatype_f16_r:
            return "f16";
        case rocsparse_datatype_bf16_r:
            return "bf16";
        case rocsparse_datatype_f32_r:
            return "f32";
        case rocsparse_datatype_f64_r:
            return "f64";
        case rocsparse_datatype_f32_c:
            return "c32";
        case rocsparse_datatype_f64_c:
            return "c64";
        default:
            return "unknown";
        }
    }

    struct combo_names
    {
        template <typename C>
        static std::string GetName(int)
        {
            return type_name<typename C::A>() + "_" + type_name<typename C::X>() + "_"
                   + type_name<typename C::Y>() + "_" + type_name<typename C::T>();
        }
    };

    // Integers up to exact_limit are exact in Y. Above it a result rounded once to Y is
    // within one ulp, at most |want| * ulp_ratio.
    template <typename Y>
    double exact_limit()
    {
        if constexpr(std::is_same<Y, _Float16>::value)
            return 2048.0;
        else if constexpr(std::is_same<Y, rocsparse_bfloat16>::value)
            return 256.0;
        else
            return std::numeric_limits<double>::infinity();
    }

    template <typename Y>
    double ulp_ratio()
    {
        if constexpr(std::is_same<Y, _Float16>::value)
            return 1.0 / 1024.0;
        else if constexpr(std::is_same<Y, rocsparse_bfloat16>::value)
            return 1.0 / 128.0;
        else
            return 0.0;
    }

    template <typename Y>
    Y poison()
    {
        if constexpr(is_integer<Y>())
            return static_cast<Y>(12345);
        else
            return convert<Y>(std::numeric_limits<float>::quiet_NaN());
    }

    // Host CSR with 0-based structure; the base is applied when copying to the device.
    // A symmetric matrix stores its lower triangle.
    template <typename A>
    struct host_matrix
    {
        int32_t              m         = 0;
        int32_t              n         = 0;
        bool                 symmetric = false;
        std::vector<int64_t> ptr;
        std::vector<int64_t> col;
        std::vector<A>       val;
    };

    enum shape_kind
    {
        // Rows of 0 to 9 entries, every seventh row empty.
        shape_irregular,
        // A few rows of several thousand entries between short and empty rows, for the
        // long-row paths of the adaptive and LRB algorithms.
        shape_long_rows,
        // Rows of 0 to 2500 entries: the LRB short, medium (warp and block per row) and
        // long bins, and adaptive blocks of one row of 1025 to 3072 entries.
        shape_medium_rows,
        // 700 rows of 1 to 3 entries: adaptive blocks of more than 128 rows.
        shape_many_short_rows,
        // More than 8192 empty rows before and after the rows with entries, which the
        // adaptive algorithm leaves out of its row blocks.
        shape_trimmed,
        // The lower triangle of a square matrix of rows of 0 to 9 entries.
        shape_symmetric,
        // No entry at all; y = beta * y + sum_i gamma_i * z_i.
        shape_empty,
        // m = 0 and n = 0.
        shape_no_rows,
        shape_no_cols,
        // The 1 x 1 matrix of issue #15870.
        shape_one
    };

    template <typename A>
    host_matrix<A> make_matrix(shape_kind shape)
    {
        host_matrix<A> a;
        if(shape == shape_one)
        {
            a.m   = 1;
            a.n   = 1;
            a.ptr = {0, 1};
            a.col = {0};
            a.val = {make<A>(2, 0)};
            return a;
        }

        static const int32_t medium_lengths[] = {0, 1, 3, 17, 40, 100, 300, 700, 1500, 2500};
        static const int32_t trimmed_lead     = 9000;
        static const int32_t trimmed_band     = 300;

        switch(shape)
        {
        case shape_long_rows:
            a.m = 64;
            a.n = 4500;
            break;
        case shape_medium_rows:
            a.m = 300;
            a.n = 3000;
            break;
        case shape_many_short_rows:
            a.m = 700;
            a.n = 64;
            break;
        case shape_trimmed:
            a.m = trimmed_lead + trimmed_band + trimmed_lead + 500;
            a.n = 64;
            break;
        case shape_no_rows:
            a.m = 0;
            a.n = 33;
            break;
        case shape_no_cols:
            a.m = 41;
            a.n = 0;
            break;
        default:
            a.m = 41;
            a.n = (shape == shape_symmetric) ? 41 : 33;
            break;
        }
        a.symmetric = (shape == shape_symmetric);

        a.ptr.push_back(0);
        for(int32_t i = 0; i < a.m; ++i)
        {
            int32_t len = 0;
            switch(shape)
            {
            case shape_irregular:
            case shape_symmetric:
                len = (i % 7 == 3) ? 0 : 1 + (i * 5) % 9;
                break;
            case shape_long_rows:
                len = (i % 7 == 3) ? 0 : (i % 16 == 5) ? 4096 + i : 1 + i % 5;
                break;
            case shape_medium_rows:
                len = medium_lengths[i % 10];
                break;
            case shape_many_short_rows:
                len = 1 + i % 3;
                break;
            case shape_trimmed:
                len = (i >= trimmed_lead && i < trimmed_lead + trimmed_band) ? 1 + i % 3 : 0;
                break;
            default:
                break;
            }
            const int32_t        start = (a.n == 0) ? 0 : (i * 11) % a.n;
            std::vector<int64_t> cols;
            for(int32_t k = 0; k < len; ++k)
            {
                const int64_t j = (start + k) % a.n;
                if(!a.symmetric || j <= i)
                    cols.push_back(j);
            }
            std::sort(cols.begin(), cols.end());
            for(const int64_t j : cols)
            {
                a.col.push_back(j);
                a.val.push_back(value_at<A>(i * 131 + j));
            }
            a.ptr.push_back(static_cast<int64_t>(a.col.size()));
        }
        return a;
    }

    // y = alpha * op(A) * x + beta * y + sum_i gamma_i * z_i in the compute type T; beta == 0
    // does not read y.
    template <typename C>
    std::vector<typename C::T> reference(const host_matrix<typename C::A>&              a,
                                         rocsparse_operation                            trans,
                                         typename C::T                                  alpha,
                                         const std::vector<typename C::X>&              x,
                                         typename C::T                                  beta,
                                         const std::vector<typename C::Y>&              y0,
                                         const std::vector<typename C::T>&              gamma,
                                         const std::vector<std::vector<typename C::T>>& z)
    {
        using T = typename C::T;
        std::vector<T> ax(y0.size(), make<T>(0, 0));
        for(int32_t i = 0; i < a.m; ++i)
            for(int64_t k = a.ptr[i]; k < a.ptr[i + 1]; ++k)
            {
                const int64_t j   = a.col[k];
                T             aij = convert<T>(a.val[k]);
                if(trans == rocsparse_operation_conjugate_transpose)
                    aij = conj_of(aij);
                if(trans == rocsparse_operation_none)
                {
                    ax[i] = ax[i] + aij * convert<T>(x[j]);
                    if(a.symmetric && j != i)
                        ax[j] = ax[j] + aij * convert<T>(x[i]);
                }
                else
                    ax[j] = ax[j] + aij * convert<T>(x[i]);
            }
        std::vector<T> want(y0.size());
        for(size_t r = 0; r < y0.size(); ++r)
        {
            T v = alpha * ax[r];
            if(beta != make<T>(0, 0))
                v = v + beta * convert<T>(y0[r]);
            for(size_t e = 0; e < z.size(); ++e)
                v = v + gamma[e] * z[e][r];
            want[r] = v;
        }
        return want;
    }

    template <typename Y, typename T>
    bool expect_result(const std::vector<T>& want, const std::vector<Y>& got)
    {
        if(want.size() != got.size())
        {
            ADD_FAILURE() << "y size " << got.size() << " != " << want.size();
            return false;
        }
        size_t bad   = 0;
        size_t first = 0;
        for(size_t i = 0; i < want.size(); ++i)
        {
            const double wr  = real_part(want[i]);
            const double wi  = imag_part(want[i]);
            const double mag = std::max(std::abs(wr), std::abs(wi));
            const double tol = (mag <= exact_limit<Y>()) ? 0.0 : mag * ulp_ratio<Y>();
            if(!(std::abs(wr - real_part(got[i])) <= tol
                 && std::abs(wi - imag_part(got[i])) <= tol))
            {
                if(bad == 0)
                    first = i;
                ++bad;
            }
        }
        if(bad > 0)
        {
            ADD_FAILURE() << bad << " mismatches in y, first at " << first << " want=("
                          << real_part(want[first]) << "," << imag_part(want[first]) << ") got=("
                          << real_part(got[first]) << "," << imag_part(got[first]) << ")";
            return false;
        }
        return true;
    }

    enum index_kind
    {
        index_i32_i32,
        index_i64_i32,
        index_i64_i64
    };

    enum special_kind
    {
        special_none,
        // beta = 0 with y filled with NaN (or a sentinel for an integer y).
        special_beta_zero,
        // alpha = 0 and beta = 1.
        special_alpha_zero_beta_one
    };

    const special_kind all_specials[]
        = {special_none, special_beta_zero, special_alpha_zero_beta_one};

    enum gamma_kind
    {
        // 1, -1 + i, 3, ...
        gamma_nonzero,
        // {0, 2, 0}
        gamma_zero_two_zero,
        // {0}
        gamma_all_zero
    };

    enum toggle_kind
    {
        // The extras stay enabled.
        toggle_none,
        // Disabled after rocsparse_spmv_set_extra; the result must not contain them.
        toggle_disabled,
        // Disabled, then enabled again.
        toggle_reenabled
    };

    struct extra_case
    {
        rocsparse_spmv_alg  alg          = rocsparse_spmv_alg_default;
        rocsparse_operation trans        = rocsparse_operation_none;
        shape_kind          shape        = shape_irregular;
        index_kind          index        = index_i32_i32;
        special_kind        special      = special_none;
        gamma_kind          gammas       = gamma_nonzero;
        int                 extras       = 0;
        toggle_kind         extra_toggle = toggle_none;
        int                 base         = 0;
        bool                device       = false;
    };

    std::ostream& operator<<(std::ostream& os, const extra_case& c)
    {
        return os << "alg=" << c.alg << " trans=" << c.trans << " shape=" << c.shape
                  << " index=" << c.index << " special=" << c.special << " gammas=" << c.gammas
                  << " extras=" << c.extras << " toggle=" << c.extra_toggle << " base=" << c.base
                  << " device=" << c.device;
    }

    // Elements on each side of y that rocsparse_v2_spmv must not write.
    constexpr int32_t y_guard = 8;

    template <typename Y>
    Y guard_value()
    {
        return make<Y>(77, 0);
    }

    // x, y, alpha, beta, gamma and z of a case.
    template <typename C>
    struct case_data
    {
        using X = typename C::X;
        using Y = typename C::Y;
        using T = typename C::T;

        std::vector<X>              x;
        std::vector<Y>              y0;
        T                           alpha;
        T                           beta;
        std::vector<T>              gamma;
        std::vector<std::vector<T>> z;

        case_data(const extra_case& c, int32_t xs, int32_t ys)
            : x(xs)
            , y0(ys)
            , alpha(make<T>(2, 1))
            , beta(make<T>(-1, 1))
            , gamma(std::max(c.extras, 1))
            , z(c.extras, std::vector<T>(ys))
        {
            // Most of x is zero for the long rows, so that the partial row sums the
            // algorithms add to a half precision y stay exactly representable. op(A) = A^T
            // spreads a row over many entries of y, at most 5 terms each, so x stays dense.
            const bool    tr       = (c.trans != rocsparse_operation_none);
            const int32_t x_period = tr                               ? 1
                                     : (c.shape == shape_long_rows)   ? 64
                                     : (c.shape == shape_medium_rows) ? 16
                                                                      : 1;
            for(int32_t i = 0; i < xs; ++i)
                x[i] = (i % x_period != 0) ? make<X>(0, 0) : value_at<X>(3 * i + 1);
            for(int32_t i = 0; i < ys; ++i)
                y0[i] = (c.special == special_beta_zero) ? poison<Y>() : value_at<Y>(5 * i + 2);
            if(c.shape == shape_one)
            {
                x[0]  = make<X>(3, 0);
                y0[0] = make<Y>(5, 0);
                beta  = make<T>(1, 0);
            }
            if(c.special == special_beta_zero)
                beta = make<T>(0, 0);
            else if(c.special == special_alpha_zero_beta_one)
            {
                alpha = make<T>(0, 0);
                beta  = make<T>(1, 0);
            }

            for(int e = 0; e < c.extras; ++e)
            {
                switch(c.gammas)
                {
                case gamma_nonzero:
                    gamma[e] = make<T>(e == 1 ? -1 : 1 + e, e % 2);
                    break;
                case gamma_zero_two_zero:
                    gamma[e] = make<T>(e == 1 ? 2 : 0, 0);
                    break;
                case gamma_all_zero:
                    gamma[e] = make<T>(0, 0);
                    break;
                }
                for(int32_t i = 0; i < ys; ++i)
                    z[e][i] = value_at<T>(7 * i + 13 * e + 3);
            }
            if(c.shape == shape_one && c.extras > 0)
            {
                gamma[0] = make<T>(7, 0);
                z[0][0]  = make<T>(11, 0);
            }
        }
    };

    // CSR, x and guarded y on the device, with their descriptors.
    template <typename C, typename I, typename J>
    struct device_problem
    {
        using A = typename C::A;
        using X = typename C::X;
        using Y = typename C::Y;

        device_vector<I> dptr;
        device_vector<J> dcol;
        device_vector<A> dval;
        device_vector<X> dx;
        device_vector<Y> dy;
        int32_t          ys;
        spmat_guard      mat;
        dnvec_guard      vx, vy;

        static std::vector<I> row_ptr(const host_matrix<A>& a, int base)
        {
            std::vector<I> ptr(a.ptr.size());
            for(size_t i = 0; i < ptr.size(); ++i)
                ptr[i] = static_cast<I>(a.ptr[i] + base);
            return ptr;
        }
        static std::vector<J> col_ind(const host_matrix<A>& a, int base)
        {
            std::vector<J> col(std::max<size_t>(a.col.size(), 1), static_cast<J>(base));
            for(size_t k = 0; k < a.col.size(); ++k)
                col[k] = static_cast<J>(a.col[k] + base);
            return col;
        }
        static std::vector<A> values(const host_matrix<A>& a)
        {
            return a.val.empty() ? std::vector<A>(1, make<A>(0, 0)) : a.val;
        }
        static std::vector<X> padded(const std::vector<X>& x)
        {
            return x.empty() ? std::vector<X>(1, make<X>(0, 0)) : x;
        }
        static std::vector<Y> guarded(const std::vector<Y>& y0)
        {
            std::vector<Y> y(y0.size() + 2 * y_guard, guard_value<Y>());
            std::copy(y0.begin(), y0.end(), y.begin() + y_guard);
            return y;
        }

        device_problem(const host_matrix<A>& a,
                       int                   base,
                       const std::vector<X>& x,
                       const std::vector<Y>& y0)
            : dptr(row_ptr(a, base))
            , dcol(col_ind(a, base))
            , dval(values(a))
            , dx(padded(x))
            , dy(guarded(y0))
            , ys(static_cast<int32_t>(y0.size()))
        {
            EXPECT_EQ(rocsparse_create_csr_descr(&mat.d,
                                                 a.m,
                                                 a.n,
                                                 static_cast<int64_t>(a.col.size()),
                                                 dptr.ptr,
                                                 dcol.ptr,
                                                 dval.ptr,
                                                 rocsparse_ut::it_of<I>(),
                                                 rocsparse_ut::it_of<J>(),
                                                 to_base(base),
                                                 dt_of<A>()),
                      rocsparse_status_success);
            if(a.symmetric)
            {
                const rocsparse_matrix_type type = rocsparse_matrix_type_symmetric;
                EXPECT_EQ(rocsparse_spmat_set_attribute(
                              mat.d, rocsparse_spmat_matrix_type, &type, sizeof(type)),
                          rocsparse_status_success);
            }
            EXPECT_EQ(rocsparse_create_dnvec_descr(
                          &vx.d, static_cast<int64_t>(x.size()), dx.ptr, dt_of<X>()),
                      rocsparse_status_success);
            EXPECT_EQ(rocsparse_create_dnvec_descr(&vy.d, ys, dy.ptr + y_guard, dt_of<Y>()),
                      rocsparse_status_success);
        }

        void reset_y(const std::vector<Y>& y0)
        {
            const std::vector<Y> y = guarded(y0);
            ASSERT_EQ(hipMemcpy(dy.ptr, y.data(), y.size() * sizeof(Y), hipMemcpyHostToDevice),
                      hipSuccess);
        }

        // Compares y to `want` and checks that the guard elements are unchanged.
        void expect_y(const std::vector<typename C::T>& want)
        {
            const std::vector<Y> all = to_host(dy);
            const Y              g   = guard_value<Y>();
            for(int32_t i = 0; i < y_guard; ++i)
            {
                EXPECT_EQ(real_part(all[i]), real_part(g)) << "guard element " << i - y_guard;
                EXPECT_EQ(real_part(all[y_guard + ys + i]), real_part(g))
                    << "guard element " << ys + i;
            }
            expect_result<Y>(want, std::vector<Y>(all.begin() + y_guard, all.end() - y_guard));
        }
    };

    // gamma and z descriptors for rocsparse_spmv_set_extra.
    template <typename T>
    struct device_extras
    {
        device_vector<T>                         dgamma;
        std::deque<device_vector<T>>             dz;
        dnvec_guard                              vgamma;
        std::deque<dnvec_guard>                  vz;
        std::vector<rocsparse_const_dnvec_descr> zs;

        device_extras(const std::vector<T>&              gamma,
                      const std::vector<std::vector<T>>& z,
                      int32_t                            ys,
                      bool                               device)
            : dgamma(gamma)
        {
            const int extras = static_cast<int>(z.size());
            for(int e = 0; e < extras; ++e)
                dz.emplace_back(ys == 0 ? std::vector<T>(1, make<T>(0, 0)) : z[e]);
            if(extras == 0)
                return;
            EXPECT_EQ(rocsparse_create_dnvec_descr(
                          &vgamma.d,
                          extras,
                          device ? static_cast<void*>(dgamma.ptr)
                                 : static_cast<void*>(const_cast<T*>(gamma.data())),
                          dt_of<T>()),
                      rocsparse_status_success);
            for(int e = 0; e < extras; ++e)
            {
                vz.emplace_back();
                EXPECT_EQ(rocsparse_create_dnvec_descr(&vz.back().d, ys, dz[e].ptr, dt_of<T>()),
                          rocsparse_status_success);
                zs.push_back(vz.back().d);
            }
        }

        rocsparse_status set(rocsparse_handle handle, rocsparse_spmv_descr descr) const
        {
            return rocsparse_spmv_set_extra(handle,
                                            descr,
                                            static_cast<int64_t>(zs.size()),
                                            vgamma.d,
                                            const_cast<rocsparse_const_dnvec_descr*>(zs.data()),
                                            nullptr);
        }
    };

    template <typename C, typename I, typename J>
    void run_extra_case(rocsparse_handle handle, const extra_case& c)
    {
        using A = typename C::A;
        using T = typename C::T;
        SCOPED_TRACE(::testing::Message() << c);

        const host_matrix<A> a  = make_matrix<A>(c.shape);
        const bool           tr = (c.trans != rocsparse_operation_none);
        const int32_t        xs = tr ? a.m : a.n;
        const int32_t        ys = tr ? a.n : a.m;

        const case_data<C> d(c, xs, ys);

        const std::vector<std::vector<T>> no_z;
        const std::vector<T>              want = reference<C>(a,
                                                 c.trans,
                                                 d.alpha,
                                                 d.x,
                                                 d.beta,
                                                 d.y0,
                                                 d.gamma,
                                                 c.extra_toggle == toggle_disabled ? no_z : d.z);

        device_problem<C, I, J> p(a, c.base, d.x, d.y0);
        device_vector<T>        dalpha(std::vector<T>(1, d.alpha));
        device_vector<T>        dbeta(std::vector<T>(1, d.beta));
        device_extras<T>        ex(d.gamma, d.z, ys, c.device);
        if(::testing::Test::HasFailure())
            return;

        scoped_pointer_mode mode(handle, c.device);

        spmv_descr_guard descr;
        ASSERT_TRUE(descr.d);
        ASSERT_EQ(set_basic_inputs(handle, descr.d, c.alg, c.trans, dt_of<T>(), dt_of<T>()),
                  rocsparse_status_success);
        if(c.extras > 0)
        {
            ASSERT_EQ(ex.set(handle, descr.d), rocsparse_status_success);
            if(c.extra_toggle != toggle_none)
                ASSERT_EQ(set_input(handle, descr.d, rocsparse_spmv_input_enable_extra, int32_t(0)),
                          rocsparse_status_success);
            if(c.extra_toggle == toggle_reenabled)
                ASSERT_EQ(set_input(handle, descr.d, rocsparse_spmv_input_enable_extra, int32_t(1)),
                          rocsparse_status_success);
        }

        const void* palpha
            = c.device ? static_cast<const void*>(dalpha.ptr) : static_cast<const void*>(&d.alpha);
        const void* pbeta
            = c.device ? static_cast<const void*>(dbeta.ptr) : static_cast<const void*>(&d.beta);

        // LRB does not support symmetric matrices, with or without extras, unless alpha = 0
        // and beta = 1 in host mode without a nonzero gamma, which returns before the
        // algorithm is chosen.
        const bool quick_return
            = !c.device && c.special == special_alpha_zero_beta_one
              && (c.extras == 0 || c.extra_toggle == toggle_disabled || c.gammas == gamma_all_zero);
        const rocsparse_status expected
            = (a.symmetric && c.alg == rocsparse_spmv_alg_csr_lrb && !quick_return)
                  ? rocsparse_status_not_implemented
                  : rocsparse_status_success;
        rocsparse_status st = run_v2_spmv_stage(handle,
                                                descr.d,
                                                p.mat.d,
                                                p.vx.d,
                                                p.vy.d,
                                                palpha,
                                                pbeta,
                                                rocsparse_v2_spmv_stage_analysis);
        if(st == rocsparse_status_success)
            st = run_v2_spmv_stage(handle,
                                   descr.d,
                                   p.mat.d,
                                   p.vx.d,
                                   p.vy.d,
                                   palpha,
                                   pbeta,
                                   rocsparse_v2_spmv_stage_compute);
        ASSERT_EQ(st, expected);
        if(expected == rocsparse_status_success)
            p.expect_y(want);
    }

    // -0, +inf, -inf and a signaling NaN by bit pattern. Scaling one of them by 1 can change
    // its bits: the NaN is quieted, and -0 can come back as +0 after narrowing.
    template <typename R>
    R special_real(int k)
    {
        if constexpr(is_integer<R>())
            return static_cast<R>(k - 7);
        else
        {
            using B = std::conditional_t<sizeof(R) == 2,
                                         uint16_t,
                                         std::conditional_t<sizeof(R) == 4, uint32_t, uint64_t>>;
            static_assert(sizeof(B) == sizeof(R), "unexpected floating-point size");
            B bits[4];
            if constexpr(std::is_same<R, double>::value)
            {
                const B b[4] = {0x8000000000000000ull,
                                0x7ff0000000000000ull,
                                0xfff0000000000000ull,
                                0x7ff4000000000000ull};
                std::copy(b, b + 4, bits);
            }
            else if constexpr(std::is_same<R, float>::value)
            {
                const B b[4] = {0x80000000u, 0x7f800000u, 0xff800000u, 0x7fa00000u};
                std::copy(b, b + 4, bits);
            }
            else if constexpr(std::is_same<R, _Float16>::value)
            {
                const B b[4] = {0x8000, 0x7c00, 0xfc00, 0x7d00};
                std::copy(b, b + 4, bits);
            }
            else
            {
                const B b[4] = {0x8000, 0x7f80, 0xff80, 0x7fa0};
                std::copy(b, b + 4, bits);
            }
            R r;
            std::memcpy(&r, &bits[k % 4], sizeof(R));
            return r;
        }
    }

    // For a complex y, (1, inf) * 1 also has a NaN real part.
    template <typename Y>
    Y special_y(int64_t i)
    {
        if constexpr(is_complex_v<Y>)
        {
            using R = typename real_of<Y>::type;
            switch(i % 4)
            {
            case 0:
                return Y(R(1), special_real<R>(1));
            case 1:
                return Y(special_real<R>(0), R(-1));
            case 2:
                return Y(special_real<R>(3), R(2));
            default:
                return Y(special_real<R>(2), special_real<R>(0));
            }
        }
        else
            return special_real<Y>(static_cast<int>(i % 4));
    }

    // alpha = 0 and beta = 1 without extras leave every bit of y as it was, also in rows
    // that several workgroups share.
    template <typename C, typename I, typename J>
    void run_untouched_case(rocsparse_handle handle, const extra_case& c)
    {
        using A = typename C::A;
        using X = typename C::X;
        using Y = typename C::Y;
        using T = typename C::T;
        SCOPED_TRACE(::testing::Message() << c);

        const host_matrix<A> a = make_matrix<A>(c.shape);
        std::vector<X>       x(a.n);
        for(int32_t i = 0; i < a.n; ++i)
            x[i] = value_at<X>(i);
        // The long rows are 16 apart; i / 16 gives each of them a different value.
        std::vector<Y> y0(a.m);
        for(int32_t i = 0; i < a.m; ++i)
            y0[i] = special_y<Y>(i + i / 16);

        device_problem<C, I, J> p(a, c.base, x, y0);
        const T                 alpha = make<T>(0, 0);
        const T                 beta  = make<T>(1, 0);
        device_vector<T>        dalpha(std::vector<T>(1, alpha));
        device_vector<T>        dbeta(std::vector<T>(1, beta));
        if(::testing::Test::HasFailure())
            return;

        scoped_pointer_mode mode(handle, c.device);

        spmv_descr_guard descr;
        ASSERT_TRUE(descr.d);
        ASSERT_EQ(set_basic_inputs(handle, descr.d, c.alg, c.trans, dt_of<T>(), dt_of<T>()),
                  rocsparse_status_success);
        const void* palpha
            = c.device ? static_cast<const void*>(dalpha.ptr) : static_cast<const void*>(&alpha);
        const void* pbeta
            = c.device ? static_cast<const void*>(dbeta.ptr) : static_cast<const void*>(&beta);
        run_v2_spmv(handle, descr.d, p.mat.d, p.vx.d, p.vy.d, palpha, pbeta);
        if(::testing::Test::HasFailure())
            return;

        const std::vector<Y> want = device_problem<C, I, J>::guarded(y0);
        const std::vector<Y> got  = to_host(p.dy);
        ASSERT_EQ(got.size(), want.size());
        size_t bad   = 0;
        size_t first = 0;
        for(size_t i = 0; i < want.size(); ++i)
        {
            if(std::memcmp(&got[i], &want[i], sizeof(Y)) != 0)
            {
                if(bad == 0)
                    first = i;
                ++bad;
            }
        }
        EXPECT_EQ(bad, 0u) << "y elements changed, first at y["
                           << static_cast<int64_t>(first) - y_guard << "]";
    }

    // rocsparse_spmv_set_extra copies gamma, read with the pointer mode of that call. A host
    // mode compute with alpha = 0 and beta = 1 uses the copy too, after the caller zeros its
    // gamma buffer, whether the copy was read from host or from device memory.
    template <typename C, typename I, typename J>
    void run_captured_gamma_case(rocsparse_handle handle, const extra_case& c, bool set_on_device)
    {
        using A = typename C::A;
        using T = typename C::T;
        SCOPED_TRACE(::testing::Message() << c << " set_on_device=" << set_on_device);

        const host_matrix<A> a = make_matrix<A>(c.shape);
        extra_case           e = c;
        e.extras               = 1;
        e.special              = special_alpha_zero_beta_one;
        const case_data<C>   d(e, a.n, a.m);
        const std::vector<T> want
            = reference<C>(a, rocsparse_operation_none, d.alpha, d.x, d.beta, d.y0, d.gamma, d.z);

        device_problem<C, I, J> p(a, c.base, d.x, d.y0);
        std::vector<T>          host_gamma = d.gamma;
        device_extras<T>        ex(host_gamma, d.z, a.m, set_on_device);
        if(::testing::Test::HasFailure())
            return;

        spmv_descr_guard descr;
        ASSERT_TRUE(descr.d);
        ASSERT_EQ(set_basic_inputs(
                      handle, descr.d, c.alg, rocsparse_operation_none, dt_of<T>(), dt_of<T>()),
                  rocsparse_status_success);
        {
            scoped_pointer_mode mode(handle, set_on_device);
            ASSERT_EQ(ex.set(handle, descr.d), rocsparse_status_success);
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        std::fill(host_gamma.begin(), host_gamma.end(), make<T>(0, 0));
        ASSERT_EQ(hipMemset(ex.dgamma.ptr, 0, sizeof(T) * host_gamma.size()), hipSuccess);

        scoped_pointer_mode mode(handle, false);
        run_v2_spmv(handle, descr.d, p.mat.d, p.vx.d, p.vy.d, &d.alpha, &d.beta);
        if(::testing::Test::HasFailure())
            return;
        p.expect_y(want);
    }

    template <typename C>
    void run_extra_case(rocsparse_handle handle, const extra_case& c)
    {
        switch(c.index)
        {
        case index_i32_i32:
            return run_extra_case<C, int32_t, int32_t>(handle, c);
        case index_i64_i32:
            return run_extra_case<C, int64_t, int32_t>(handle, c);
        case index_i64_i64:
            return run_extra_case<C, int64_t, int64_t>(handle, c);
        }
    }

    // One analysis, then several computes on the same descriptor: with one extra, with the
    // extras disabled, with them enabled again, and after replacing them with three others.
    template <typename C, typename I, typename J>
    void run_reuse_case(rocsparse_handle handle, const extra_case& c)
    {
        using A = typename C::A;
        using T = typename C::T;
        SCOPED_TRACE(::testing::Message() << c);

        const host_matrix<A> a  = make_matrix<A>(c.shape);
        const bool           tr = (c.trans != rocsparse_operation_none);
        const int32_t        xs = tr ? a.m : a.n;
        const int32_t        ys = tr ? a.n : a.m;

        extra_case one   = c;
        one.extras       = 1;
        extra_case three = c;
        three.extras     = 3;
        const case_data<C> d1(one, xs, ys);
        const case_data<C> d3(three, xs, ys);

        const std::vector<std::vector<T>> no_z;
        const std::vector<T>              want_one
            = reference<C>(a, c.trans, d1.alpha, d1.x, d1.beta, d1.y0, d1.gamma, d1.z);
        const std::vector<T> want_none
            = reference<C>(a, c.trans, d1.alpha, d1.x, d1.beta, d1.y0, d1.gamma, no_z);
        const std::vector<T> want_three
            = reference<C>(a, c.trans, d3.alpha, d3.x, d3.beta, d3.y0, d3.gamma, d3.z);

        device_problem<C, I, J> p(a, c.base, d1.x, d1.y0);
        device_vector<T>        dalpha(std::vector<T>(1, d1.alpha));
        device_vector<T>        dbeta(std::vector<T>(1, d1.beta));
        device_extras<T>        ex1(d1.gamma, d1.z, ys, c.device);
        device_extras<T>        ex3(d3.gamma, d3.z, ys, c.device);
        if(::testing::Test::HasFailure())
            return;

        scoped_pointer_mode mode(handle, c.device);

        spmv_descr_guard descr;
        ASSERT_TRUE(descr.d);
        ASSERT_EQ(set_basic_inputs(handle, descr.d, c.alg, c.trans, dt_of<T>(), dt_of<T>()),
                  rocsparse_status_success);
        ASSERT_EQ(ex1.set(handle, descr.d), rocsparse_status_success);

        const void* palpha
            = c.device ? static_cast<const void*>(dalpha.ptr) : static_cast<const void*>(&d1.alpha);
        const void* pbeta
            = c.device ? static_cast<const void*>(dbeta.ptr) : static_cast<const void*>(&d1.beta);
        auto compute = [&](const std::vector<T>& want, const char* what) {
            SCOPED_TRACE(what);
            p.reset_y(d1.y0);
            ASSERT_EQ(run_v2_spmv_stage(handle,
                                        descr.d,
                                        p.mat.d,
                                        p.vx.d,
                                        p.vy.d,
                                        palpha,
                                        pbeta,
                                        rocsparse_v2_spmv_stage_compute),
                      rocsparse_status_success);
            p.expect_y(want);
        };

        ASSERT_EQ(run_v2_spmv_stage(handle,
                                    descr.d,
                                    p.mat.d,
                                    p.vx.d,
                                    p.vy.d,
                                    palpha,
                                    pbeta,
                                    rocsparse_v2_spmv_stage_analysis),
                  rocsparse_status_success);
        compute(want_one, "one extra");
        ASSERT_EQ(set_input(handle, descr.d, rocsparse_spmv_input_enable_extra, int32_t(0)),
                  rocsparse_status_success);
        compute(want_none, "extras disabled");
        ASSERT_EQ(set_input(handle, descr.d, rocsparse_spmv_input_enable_extra, int32_t(1)),
                  rocsparse_status_success);
        compute(want_one, "extras enabled again");
        ASSERT_EQ(rocsparse_spmv_clear_extra(handle, descr.d, nullptr), rocsparse_status_success);
        ASSERT_EQ(ex3.set(handle, descr.d), rocsparse_status_success);
        compute(want_three, "three other extras");
    }

    template <typename C>
    void run_reuse_case(rocsparse_handle handle, const extra_case& c)
    {
        switch(c.index)
        {
        case index_i32_i32:
            return run_reuse_case<C, int32_t, int32_t>(handle, c);
        case index_i64_i32:
            return run_reuse_case<C, int64_t, int32_t>(handle, c);
        case index_i64_i64:
            return run_reuse_case<C, int64_t, int64_t>(handle, c);
        }
    }

    const rocsparse_spmv_alg all_csr_algs[] = {rocsparse_spmv_alg_default,
                                               rocsparse_spmv_alg_csr_rowsplit,
                                               rocsparse_spmv_alg_csr_adaptive,
                                               rocsparse_spmv_alg_csr_lrb,
                                               rocsparse_spmv_alg_csr_nnzsplit};

    struct index_and_base
    {
        index_kind index;
        int        base;
    };

    const std::vector<index_and_base> all_indices_and_bases = {{index_i32_i32, 0},
                                                               {index_i32_i32, 1},
                                                               {index_i64_i32, 0},
                                                               {index_i64_i32, 1},
                                                               {index_i64_i64, 0},
                                                               {index_i64_i64, 1}};

    // csrmv returns before choosing an algorithm for a matrix without entries, so its index
    // types and base only reach the analysis.
    const std::vector<index_and_base> empty_matrix_indices_and_bases
        = {{index_i32_i32, 0}, {index_i64_i64, 1}};

    bool supports_transpose(rocsparse_spmv_alg alg)
    {
        return alg == rocsparse_spmv_alg_default || alg == rocsparse_spmv_alg_csr_rowsplit
               || alg == rocsparse_spmv_alg_csr_nnzsplit;
    }

    template <typename C>
    class L2DeviceV2SpmvExtraTypes : public rocsparse_ut::HandleTest
    {
    protected:
        // The operations of op(A): conjugate transpose only for a complex compute type.
        static std::vector<rocsparse_operation> transposes()
        {
            if(is_complex_v<typename C::T>)
                return {rocsparse_operation_transpose, rocsparse_operation_conjugate_transpose};
            return {rocsparse_operation_transpose};
        }

        // Runs `c` for every CSR algorithm, the given index widths and bases, and both
        // pointer modes.
        template <typename F>
        void sweep_with(extra_case c, F run, const std::vector<index_and_base>& indices)
        {
            for(const auto alg : all_csr_algs)
            {
                if(c.trans != rocsparse_operation_none && !supports_transpose(alg))
                    continue;
                for(const auto& ib : indices)
                    for(const bool device : {false, true})
                    {
                        c.alg    = alg;
                        c.index  = ib.index;
                        c.base   = ib.base;
                        c.device = device;
                        run(handle, c);
                        if(::testing::Test::HasFailure())
                            return;
                    }
            }
        }

        void sweep(const extra_case&                  c,
                   const std::vector<index_and_base>& indices = all_indices_and_bases)
        {
            sweep_with(
                c,
                [](rocsparse_handle h, const extra_case& e) { run_extra_case<C>(h, e); },
                indices);
        }

        void sweep_reuse(const extra_case& c)
        {
            sweep_with(
                c,
                [](rocsparse_handle h, const extra_case& e) { run_reuse_case<C>(h, e); },
                all_indices_and_bases);
        }

        // Sweeps `shape` with 0, 1 and 3 extras and every special case of alpha and beta.
        void sweep_specials(shape_kind                         shape,
                            rocsparse_operation                trans,
                            const std::vector<index_and_base>& indices = all_indices_and_bases)
        {
            for(const int extras : {0, 1, 3})
                for(const auto special : all_specials)
                {
                    extra_case c;
                    c.shape   = shape;
                    c.trans   = trans;
                    c.extras  = extras;
                    c.special = special;
                    sweep(c, indices);
                    if(::testing::Test::HasFailure())
                        return;
                }
        }
    };

    TYPED_TEST_SUITE(L2DeviceV2SpmvExtraTypes, all_combos, combo_names);

    TYPED_TEST(L2DeviceV2SpmvExtraTypes, issue_15870_one_by_one)
    {
        // y = 2 * 2 * 3 + 1 * 5 + 7 * 11 = 94 (17 without the extra term) for a real type.
        for(const int extras : {0, 1})
        {
            extra_case c;
            c.shape  = shape_one;
            c.extras = extras;
            this->sweep(c);
            if(::testing::Test::HasFailure())
                return;
        }
    }

    TYPED_TEST(L2DeviceV2SpmvExtraTypes, irregular)
    {
        this->sweep_specials(shape_irregular, rocsparse_operation_none);
    }

    TYPED_TEST(L2DeviceV2SpmvExtraTypes, long_rows)
    {
        this->sweep_specials(shape_long_rows, rocsparse_operation_none);
    }

    TYPED_TEST(L2DeviceV2SpmvExtraTypes, medium_rows)
    {
        this->sweep_specials(shape_medium_rows, rocsparse_operation_none);
    }

    TYPED_TEST(L2DeviceV2SpmvExtraTypes, many_short_rows)
    {
        this->sweep_specials(shape_many_short_rows, rocsparse_operation_none);
    }

    TYPED_TEST(L2DeviceV2SpmvExtraTypes, trimmed_empty_rows)
    {
        this->sweep_specials(shape_trimmed, rocsparse_operation_none);
    }

    TYPED_TEST(L2DeviceV2SpmvExtraTypes, symmetric)
    {
        this->sweep_specials(shape_symmetric, rocsparse_operation_none);
    }

    TYPED_TEST(L2DeviceV2SpmvExtraTypes, transpose)
    {
        for(const auto trans : this->transposes())
            for(const auto shape : {shape_irregular, shape_long_rows})
            {
                this->sweep_specials(shape, trans);
                if(::testing::Test::HasFailure())
                    return;
            }
    }

    TYPED_TEST(L2DeviceV2SpmvExtraTypes, matrix_without_entries)
    {
        std::vector<rocsparse_operation> ops = this->transposes();
        ops.insert(ops.begin(), rocsparse_operation_none);
        for(const auto trans : ops)
            for(const auto shape : {shape_empty, shape_no_rows, shape_no_cols})
            {
                this->sweep_specials(shape, trans, empty_matrix_indices_and_bases);
                if(::testing::Test::HasFailure())
                    return;
            }
    }

    TYPED_TEST(L2DeviceV2SpmvExtraTypes, zero_gamma)
    {
        // gamma = {0, 2, 0}, and gamma = {0}, which in host mode lets alpha = 0, beta = 1
        // return without touching y.
        for(const auto special : all_specials)
            for(const auto gammas : {gamma_zero_two_zero, gamma_all_zero})
            {
                extra_case c;
                c.shape   = shape_irregular;
                c.gammas  = gammas;
                c.extras  = (gammas == gamma_all_zero) ? 1 : 3;
                c.special = special;
                this->sweep(c);
                if(::testing::Test::HasFailure())
                    return;
            }
    }

    TYPED_TEST(L2DeviceV2SpmvExtraTypes, alpha_zero_beta_one_keeps_y_bits)
    {
        // The long rows are split over several workgroups by the adaptive algorithm and
        // fall in the LRB long-row bins; the medium rows reach LRB bins 11 and 12.
        for(const auto shape : {shape_long_rows, shape_medium_rows})
            for(const auto alg : all_csr_algs)
                for(const bool device : {false, true})
                {
                    extra_case c;
                    c.shape  = shape;
                    c.alg    = alg;
                    c.device = device;
                    run_untouched_case<TypeParam, int32_t, int32_t>(this->handle, c);
                    if(::testing::Test::HasFailure())
                        return;
                }
    }

    TYPED_TEST(L2DeviceV2SpmvExtraTypes, gamma_copied_from_host_buffer)
    {
        for(const auto alg : all_csr_algs)
        {
            extra_case c;
            c.shape = shape_irregular;
            c.alg   = alg;
            run_captured_gamma_case<TypeParam, int32_t, int32_t>(this->handle, c, false);
            if(::testing::Test::HasFailure())
                return;
        }
    }

    TYPED_TEST(L2DeviceV2SpmvExtraTypes, gamma_set_in_device_mode_computed_in_host_mode)
    {
        for(const auto alg : all_csr_algs)
        {
            extra_case c;
            c.shape = shape_irregular;
            c.alg   = alg;
            run_captured_gamma_case<TypeParam, int32_t, int32_t>(this->handle, c, true);
            if(::testing::Test::HasFailure())
                return;
        }
    }

    TYPED_TEST(L2DeviceV2SpmvExtraTypes, enable_extra_toggle)
    {
        for(const auto toggle : {toggle_disabled, toggle_reenabled})
            for(const int extras : {1, 3})
            {
                extra_case c;
                c.shape        = shape_irregular;
                c.extras       = extras;
                c.extra_toggle = toggle;
                this->sweep(c);
                if(::testing::Test::HasFailure())
                    return;
            }
    }

    TYPED_TEST(L2DeviceV2SpmvExtraTypes, descriptor_reuse)
    {
        for(const auto trans : {rocsparse_operation_none, rocsparse_operation_transpose})
            for(const auto shape : {shape_irregular, shape_long_rows})
            {
                extra_case c;
                c.shape = shape;
                c.trans = trans;
                this->sweep_reuse(c);
                if(::testing::Test::HasFailure())
                    return;
            }
    }
}
