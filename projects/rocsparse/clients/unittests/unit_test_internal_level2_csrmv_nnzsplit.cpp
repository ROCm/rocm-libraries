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
// CSR nnz-split SpMV [AISPARSE-1099].
//
// Every case runs the public SpMV (rocsparse_spmv, and rocsparse_v2_spmv with
// and without starting block ids) with rocsparse_spmv_alg_csr_nnzsplit and
// compares y with a host reference. All data are small integers, so the
// reference is exact in every tested precision. When beta is zero y is filled
// with a poison value beforehand, so a read of y shows up in the result.
// Floating-point types get a NaN; integral types get the bit pattern 0x5a5a5a5a.
//
// The nnz-per-thread granularity is chosen from the matrix (see
// nnzsplit_nnz_per_thread): 1 for small or sparse matrices, 8 when the longest
// row is much longer than the mean, and 4 or 8 for large matrices by density.
// The shapes below are sized to land in each of them, and the granularity that
// was selected is read back from the v1 analysis and checked.
//
#include "unit_test_device_level2_common.hpp"

#include "csrmv_device_nnzsplit.h"
#include "rocsparse_mat_info.hpp"
#include "rocsparse_spmat_descr.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    using cd = std::complex<double>;

    // ---- data -------------------------------------------------------------

    // CSR pattern with zero-based indices. Values, x and y0 are generated
    // unless a case sets them.
    struct Csr
    {
        int64_t              m = 0;
        int64_t              n = 0;
        std::vector<int64_t> ptr{0};
        std::vector<int64_t> ind;
        std::vector<cd>      val;
        std::vector<cd>      x;
        std::vector<cd>      y0;

        int64_t nnz() const
        {
            return static_cast<int64_t>(ind.size());
        }

        void add_row(int64_t first_col, int64_t len)
        {
            for(int64_t k = 0; k < len; ++k)
            {
                ind.push_back(first_col + k);
            }
            ptr.push_back(nnz());
            ++m;
        }
    };

    // Row r has lens[r] consecutive columns.
    Csr make(int64_t n, const std::vector<int64_t>& lens)
    {
        Csr a;
        a.n = n;
        for(size_t r = 0; r < lens.size(); ++r)
        {
            const int64_t len = lens[r];
            if(len > n)
            {
                ADD_FAILURE() << "row " << r << " is longer than n";
                return a;
            }
            a.add_row(len > 0 ? (static_cast<int64_t>(r) * 5) % (n - len + 1) : 0, len);
        }
        return a;
    }

    // Square lower triangle: row r has lens[r] <= r + 1 columns ending at the
    // diagonal.
    Csr make_lower(const std::vector<int64_t>& lens)
    {
        Csr a;
        a.n = static_cast<int64_t>(lens.size());
        for(size_t r = 0; r < lens.size(); ++r)
        {
            const int64_t len = std::min<int64_t>(lens[r], static_cast<int64_t>(r) + 1);
            a.add_row(len > 0 ? static_cast<int64_t>(r) - len + 1 : 0, len);
        }
        return a;
    }

    // Square upper triangle: row r has lens[r] <= n - r columns starting at the
    // diagonal.
    Csr make_upper(const std::vector<int64_t>& lens)
    {
        Csr a;
        a.n = static_cast<int64_t>(lens.size());
        for(size_t r = 0; r < lens.size(); ++r)
        {
            const int64_t len = std::min<int64_t>(lens[r], a.n - static_cast<int64_t>(r));
            a.add_row(static_cast<int64_t>(r), len);
        }
        return a;
    }

    cd value_of(int64_t k)
    {
        static const int re[7] = {1, 2, -1, 1, -2, 1, -1};
        static const int im[5] = {1, 0, -1, 0, 1};
        return {static_cast<double>(re[k % 7]), static_cast<double>(im[k % 5])};
    }

    cd x_of(int64_t j)
    {
        static const int re[4] = {1, 2, -1, 3};
        return {static_cast<double>(re[j % 4]), static_cast<double>(j % 3 - 1)};
    }

    cd y0_of(int64_t i)
    {
        return {static_cast<double>(1 + i % 3), static_cast<double>(i % 2)};
    }

    template <typename T>
    T to_type(const cd& v)
    {
        if constexpr(is_complex<T>{})
        {
            using R = decltype(std::real(std::declval<T>()));
            return T(static_cast<R>(v.real()), static_cast<R>(v.imag()));
        }
        else if constexpr(std::is_integral<T>{})
        {
            return static_cast<T>(v.real());
        }
        else
        {
            return static_cast<T>(static_cast<float>(v.real()));
        }
    }

    template <typename T>
    cd to_cd(const T& v)
    {
        if constexpr(is_complex<T>{})
        {
            return {static_cast<double>(std::real(v)), static_cast<double>(std::imag(v))};
        }
        else
        {
            return {static_cast<double>(static_cast<float>(v)), 0.0};
        }
    }

    template <typename T>
    T poison()
    {
        if constexpr(std::is_integral<T>{})
        {
            return static_cast<T>(0x5a5a5a5a);
        }
        else
        {
            return to_type<T>(cd(std::numeric_limits<double>::quiet_NaN(), 0.0));
        }
    }

    // ---- reporting --------------------------------------------------------

    int g_reports = 0;

    void report(const std::string& message)
    {
        constexpr int budget = 400;
        if(g_reports < budget)
        {
            ADD_FAILURE() << message;
        }
        else if(g_reports == budget)
        {
            ADD_FAILURE() << "further failures are suppressed";
        }
        ++g_reports;
    }

    // ---- one run ----------------------------------------------------------

    struct Opt
    {
        rocsparse_operation  op        = rocsparse_operation_none;
        rocsparse_spmv_alg   alg       = rocsparse_spmv_alg_csr_nnzsplit;
        rocsparse_index_base base      = rocsparse_index_base_zero;
        bool                 v2        = false;
        bool                 use_ids   = false;
        bool                 device    = false;
        bool                 symmetric = false;
        rocsparse_fill_mode  fill      = rocsparse_fill_mode_lower;
        double               alpha     = 2.0;
        double               beta      = 3.0;
        int                  repeat    = 1;
        // Leave the previous result in y between computes. Needs beta = 0.
        bool reset_y = true;
        // Store the rows of the pattern as the columns of a CSC matrix.
        bool csc = false;
    };

    struct Out
    {
        bool            ran              = false;
        uint32_t        npt              = 0;
        uint32_t        block            = 0;
        bool            use_ids_recorded = false;
        std::vector<cd> y;
    };

    // y = beta y0 + alpha op(A) x, with A mirrored when it is symmetric.
    std::vector<cd> reference(const Csr&             a,
                              const std::vector<cd>& val,
                              const std::vector<cd>& x,
                              const std::vector<cd>& y0,
                              const Opt&             o,
                              cd                     alpha,
                              cd                     beta)
    {
        const int64_t   ylen = (o.op == rocsparse_operation_none) ? a.m : a.n;
        std::vector<cd> y(static_cast<size_t>(ylen), cd(0.0, 0.0));
        // std::complex multiplies 0 * NaN into the imaginary part. A real
        // product stays real, including when the real part is NaN.
        const auto mul = [](cd lhs, cd rhs) {
            if(lhs.imag() == 0.0 && rhs.imag() == 0.0)
            {
                return cd(lhs.real() * rhs.real(), 0.0);
            }
            return lhs * rhs;
        };
        if(beta != cd(0.0, 0.0))
        {
            for(int64_t i = 0; i < ylen; ++i)
            {
                y[static_cast<size_t>(i)] = beta * y0[static_cast<size_t>(i)];
            }
        }
        for(int64_t r = 0; r < a.m; ++r)
        {
            for(int64_t k = a.ptr[static_cast<size_t>(r)]; k < a.ptr[static_cast<size_t>(r) + 1];
                ++k)
            {
                const int64_t c = a.ind[static_cast<size_t>(k)];
                const cd      v = val[static_cast<size_t>(k)];
                switch(o.op)
                {
                case rocsparse_operation_none:
                {
                    y[static_cast<size_t>(r)] += mul(mul(alpha, v), x[static_cast<size_t>(c)]);
                    if(o.symmetric && r != c)
                    {
                        y[static_cast<size_t>(c)] += mul(mul(alpha, v), x[static_cast<size_t>(r)]);
                    }
                    break;
                }
                case rocsparse_operation_transpose:
                {
                    y[static_cast<size_t>(c)] += mul(mul(alpha, v), x[static_cast<size_t>(r)]);
                    if(o.symmetric && r != c)
                    {
                        y[static_cast<size_t>(r)] += mul(mul(alpha, v), x[static_cast<size_t>(c)]);
                    }
                    break;
                }
                case rocsparse_operation_conjugate_transpose:
                {
                    const cd w = std::conj(v);
                    y[static_cast<size_t>(c)] += mul(mul(alpha, w), x[static_cast<size_t>(r)]);
                    if(o.symmetric && r != c)
                    {
                        y[static_cast<size_t>(r)] += mul(mul(alpha, w), x[static_cast<size_t>(c)]);
                    }
                    break;
                }
                }
            }
        }
        return y;
    }

    template <typename Y>
    bool compare(const std::vector<Y>& y, const std::vector<cd>& ref, const std::string& what)
    {
        bool ok = (y.size() == ref.size());
        if(!ok)
        {
            report(what + ": y has the wrong length");
            return false;
        }
        size_t bad = 0;
        for(size_t i = 0; i < y.size(); ++i)
        {
            const cd   got = to_cd(y[i]);
            const auto same
                = [](double a, double b) { return a == b || (std::isnan(a) && std::isnan(b)); };
            // A NaN in x can belong in y. NaN compares unequal to itself.
            if(!(same(got.real(), ref[i].real()) && same(got.imag(), ref[i].imag())))
            {
                if(bad++ < 3)
                {
                    report(what + ": y[" + std::to_string(i) + "] = " + std::to_string(got.real())
                           + "+" + std::to_string(got.imag()) + "i, expected "
                           + std::to_string(ref[i].real()) + "+" + std::to_string(ref[i].imag())
                           + "i");
                }
            }
        }
        return bad == 0;
    }

    // A is the matrix value type, X and Y the vector types and T the scalar
    // and compute type.
    template <typename I, typename J, typename A, typename X, typename Y, typename T>
    Out run(rocsparse_handle handle, const Csr& a, const Opt& o, const std::string& what)
    {
        Out out;

        if(!o.reset_y && o.beta != 0.0)
        {
            report(what + ": reset_y = false needs beta = 0");
            return out;
        }

        // The CSC matrix is the transpose of the pattern, so its product is
        // the opposite one of the pattern read as CSR. A symmetric matrix is
        // its own transpose, so its conjugate transpose stays one.
        Opt ro = o;
        if(o.csc)
        {
            if(o.op == rocsparse_operation_conjugate_transpose && !o.symmetric)
            {
                report(what + ": the reference does not cover this CSC case");
                return out;
            }
            if(o.op == rocsparse_operation_none)
            {
                ro.op = rocsparse_operation_transpose;
            }
            else if(o.op == rocsparse_operation_transpose)
            {
                ro.op = rocsparse_operation_none;
            }
        }

        const int64_t b    = (o.base == rocsparse_index_base_one) ? 1 : 0;
        const bool    none = (ro.op == rocsparse_operation_none);
        const int64_t xlen = none ? a.n : a.m;
        const int64_t ylen = none ? a.m : a.n;

        std::vector<I> h_ptr(a.ptr.size());
        std::vector<J> h_ind(a.ind.size());
        std::vector<A> h_val(a.ind.size());
        std::vector<X> h_x(static_cast<size_t>(xlen));
        std::vector<Y> h_y0(static_cast<size_t>(ylen));
        for(size_t i = 0; i < a.ptr.size(); ++i)
        {
            h_ptr[i] = static_cast<I>(a.ptr[i] + b);
        }
        for(size_t k = 0; k < a.ind.size(); ++k)
        {
            h_ind[k] = static_cast<J>(a.ind[k] + b);
            h_val[k] = to_type<A>(a.val.empty() ? value_of(static_cast<int64_t>(k)) : a.val[k]);
        }
        for(int64_t j = 0; j < xlen; ++j)
        {
            h_x[static_cast<size_t>(j)]
                = to_type<X>(a.x.empty() ? x_of(j) : a.x[static_cast<size_t>(j)]);
        }
        for(int64_t i = 0; i < ylen; ++i)
        {
            h_y0[static_cast<size_t>(i)]
                = to_type<Y>(a.y0.empty() ? y0_of(i) : a.y0[static_cast<size_t>(i)]);
        }

        const T alpha = to_type<T>(cd(o.alpha, 0.0));
        const T beta  = to_type<T>(cd(o.beta, 0.0));

        // The reference sees the values after conversion to the stored types.
        std::vector<cd> c_val(h_val.size());
        std::vector<cd> c_x(h_x.size());
        std::vector<cd> c_y0(h_y0.size());
        for(size_t k = 0; k < h_val.size(); ++k)
        {
            c_val[k] = to_cd(h_val[k]);
        }
        for(size_t j = 0; j < h_x.size(); ++j)
        {
            c_x[j] = to_cd(h_x[j]);
        }
        for(size_t i = 0; i < h_y0.size(); ++i)
        {
            c_y0[i] = to_cd(h_y0[i]);
        }
        const std::vector<cd> ref = reference(a, c_val, c_x, c_y0, ro, to_cd(alpha), to_cd(beta));

        // y is not read when beta is zero.
        const std::vector<Y> y_init
            = (o.beta == 0.0) ? std::vector<Y>(static_cast<size_t>(ylen), poison<Y>()) : h_y0;

        device_vector<I> d_ptr{h_ptr};
        device_vector<J> d_ind{h_ind};
        device_vector<A> d_val{h_val};
        device_vector<X> d_x{h_x};
        device_vector<Y> d_y{y_init};
        device_vector<T> d_alpha{std::vector<T>{alpha}};
        device_vector<T> d_beta{std::vector<T>{beta}};

        level2::scoped_pointer_mode pointer_mode(handle, o.device);
        const void* p_alpha = o.device ? static_cast<const void*>(d_alpha.ptr) : &alpha;
        const void* p_beta  = o.device ? static_cast<const void*>(d_beta.ptr) : &beta;

        level2::spmat_guard mat;
        level2::dnvec_guard vx;
        level2::dnvec_guard vy;
        rocsparse_status    created = rocsparse_status_success;
        if(o.csc)
        {
            created = rocsparse_create_csc_descr(&mat.d,
                                                 a.n,
                                                 a.m,
                                                 a.nnz(),
                                                 d_ptr.ptr,
                                                 d_ind.ptr,
                                                 d_val.ptr,
                                                 it_of<I>(),
                                                 it_of<J>(),
                                                 o.base,
                                                 dt_of<A>());
        }
        else
        {
            created = rocsparse_create_csr_descr(&mat.d,
                                                 a.m,
                                                 a.n,
                                                 a.nnz(),
                                                 d_ptr.ptr,
                                                 d_ind.ptr,
                                                 d_val.ptr,
                                                 it_of<I>(),
                                                 it_of<J>(),
                                                 o.base,
                                                 dt_of<A>());
        }
        if(created != rocsparse_status_success)
        {
            report(what + ": cannot create the matrix descriptor");
            return out;
        }
        if(o.symmetric)
        {
            const rocsparse_matrix_type type = rocsparse_matrix_type_symmetric;
            const rocsparse_fill_mode   fill = o.fill;
            if(rocsparse_spmat_set_attribute(mat, rocsparse_spmat_matrix_type, &type, sizeof(type))
                   != rocsparse_status_success
               || rocsparse_spmat_set_attribute(mat, rocsparse_spmat_fill_mode, &fill, sizeof(fill))
                      != rocsparse_status_success)
            {
                report(what + ": cannot set the symmetric attributes");
                return out;
            }
        }
        if(rocsparse_create_dnvec_descr(&vx.d, xlen, d_x.ptr, dt_of<X>())
               != rocsparse_status_success
           || rocsparse_create_dnvec_descr(&vy.d, ylen, d_y.ptr, dt_of<Y>())
                  != rocsparse_status_success)
        {
            report(what + ": cannot create the vector descriptors");
            return out;
        }

        auto reset = [&]() {
            return hipMemcpy(
                       d_y.ptr, y_init.data(), y_init.size() * sizeof(Y), hipMemcpyHostToDevice)
                   == hipSuccess;
        };

        auto check = [&](int rep) {
            if(hipDeviceSynchronize() != hipSuccess)
            {
                report(what + ": hipDeviceSynchronize failed");
                return;
            }
            const std::vector<Y> y = to_host(d_y.ptr, d_y.n);
            out.y.resize(y.size());
            for(size_t i = 0; i < y.size(); ++i)
            {
                out.y[i] = to_cd(y[i]);
            }
            compare(y,
                    ref,
                    what + " (" + (out.npt > 0 ? std::to_string(out.npt) + " nnz per thread, " : "")
                        + "compute " + std::to_string(rep) + ")");
        };

        auto status_name
            = [](rocsparse_status s) { return std::string(rocsparse_get_status_name(s)); };

        if(!o.v2)
        {
            auto spmv = [&](rocsparse_spmv_stage stage, size_t* size, void* buffer) {
                return rocsparse_spmv(handle,
                                      o.op,
                                      p_alpha,
                                      mat,
                                      vx,
                                      p_beta,
                                      vy,
                                      dt_of<T>(),
                                      o.alg,
                                      stage,
                                      size,
                                      buffer);
            };

            size_t           size = 0;
            rocsparse_status s    = spmv(rocsparse_spmv_stage_buffer_size, &size, nullptr);
            if(s != rocsparse_status_success)
            {
                report(what + ": buffer size returned " + status_name(s));
                return out;
            }
            void* buffer = nullptr;
            if(hipMalloc(&buffer, std::max<size_t>(size, 1)) != hipSuccess)
            {
                report(what + ": hipMalloc failed");
                return out;
            }
            s = spmv(rocsparse_spmv_stage_preprocess, &size, buffer);
            if(s != rocsparse_status_success)
            {
                report(what + ": preprocess returned " + status_name(s));
                (void)hipFree(buffer);
                return out;
            }
            rocsparse_csrmv_info info = mat.d->info->get_csrmv_info();
            if(info != nullptr && info->nnzsplit.get_starting_ids() != nullptr)
            {
                out.npt              = info->nnzsplit.nnz_per_thread;
                out.block            = info->nnzsplit.block_size;
                out.use_ids_recorded = info->nnzsplit.get_use_starting_block_ids();
            }
            for(int rep = 0; rep < o.repeat; ++rep)
            {
                if((rep == 0 || o.reset_y) && !reset())
                {
                    report(what + ": hipMemcpy failed");
                    break;
                }
                s = spmv(rocsparse_spmv_stage_compute, &size, buffer);
                if(s != rocsparse_status_success)
                {
                    report(what + ": compute returned " + status_name(s));
                    break;
                }
                check(rep);
            }
            (void)hipFree(buffer);
            out.ran = true;
            return out;
        }

        rocsparse_spmv_descr descr   = nullptr;
        rocsparse_error      error[] = {nullptr};
        auto                 checked = [&](rocsparse_status s) {
            if(error[0] != nullptr)
            {
                rocsparse_destroy_error(error[0]);
                error[0] = nullptr;
            }
            return s;
        };
        if(rocsparse_create_spmv_descr(&descr) != rocsparse_status_success)
        {
            report(what + ": cannot create the spmv descriptor");
            return out;
        }
        const rocsparse_datatype ctype = dt_of<T>();
        bool                     ok    = true;
        ok                             = ok
             && checked(rocsparse_spmv_set_input(
                    handle, descr, rocsparse_spmv_input_alg, &o.alg, sizeof(o.alg), error))
                    == rocsparse_status_success;
        ok = ok
             && checked(rocsparse_spmv_set_input(
                    handle, descr, rocsparse_spmv_input_operation, &o.op, sizeof(o.op), error))
                    == rocsparse_status_success;
        ok = ok
             && checked(rocsparse_spmv_set_input(handle,
                                                 descr,
                                                 rocsparse_spmv_input_scalar_datatype,
                                                 &ctype,
                                                 sizeof(ctype),
                                                 error))
                    == rocsparse_status_success;
        ok = ok
             && checked(rocsparse_spmv_set_input(handle,
                                                 descr,
                                                 rocsparse_spmv_input_compute_datatype,
                                                 &ctype,
                                                 sizeof(ctype),
                                                 error))
                    == rocsparse_status_success;
        ok = ok
             && checked(rocsparse_spmv_set_input(handle,
                                                 descr,
                                                 rocsparse_spmv_input_nnz_use_starting_block_ids,
                                                 &o.use_ids,
                                                 sizeof(o.use_ids),
                                                 error))
                    == rocsparse_status_success;
        if(!ok)
        {
            report(what + ": cannot configure the spmv descriptor");
            rocsparse_destroy_spmv_descr(descr);
            return out;
        }

        auto stage = [&](rocsparse_v2_spmv_stage s) {
            size_t size = 0;
            auto   st   = checked(
                rocsparse_v2_spmv_buffer_size(handle, descr, mat, vx, vy, s, &size, error));
            if(st != rocsparse_status_success)
            {
                return st;
            }
            void* buffer = nullptr;
            if(size > 0 && hipMalloc(&buffer, size) != hipSuccess)
            {
                return rocsparse_status_memory_error;
            }
            st = checked(rocsparse_v2_spmv(
                handle, descr, p_alpha, mat, vx, p_beta, vy, s, size, buffer, error));
            if(buffer != nullptr)
            {
                (void)hipFree(buffer);
            }
            return st;
        };

        rocsparse_status s = stage(rocsparse_v2_spmv_stage_analysis);
        if(s != rocsparse_status_success)
        {
            report(what + ": analysis returned " + status_name(s));
            rocsparse_destroy_spmv_descr(descr);
            return out;
        }
        for(int rep = 0; rep < o.repeat; ++rep)
        {
            if((rep == 0 || o.reset_y) && !reset())
            {
                report(what + ": hipMemcpy failed");
                break;
            }
            s = stage(rocsparse_v2_spmv_stage_compute);
            if(s != rocsparse_status_success)
            {
                report(what + ": compute returned " + status_name(s));
                break;
            }
            check(rep);
        }
        rocsparse_destroy_spmv_descr(descr);
        out.ran = true;
        return out;
    }

    Out run_f32(rocsparse_handle handle, const Csr& a, const Opt& o, const std::string& what)
    {
        return run<int32_t, int32_t, float, float, float, float>(handle, a, o, what);
    }

    Out run_f64(rocsparse_handle handle, const Csr& a, const Opt& o, const std::string& what)
    {
        return run<int32_t, int32_t, double, double, double, double>(handle, a, o, what);
    }

    // ---- shapes -----------------------------------------------------------

    using Named = std::pair<std::string, Csr>;

    // Row 0 is not empty, and its nonzeros share the last wavefront with the
    // last nonzero. y = 2 A x + 3 y0 = [65, 62, 9], not [127, 62, 9].
    Csr make_issue()
    {
        Csr a;
        a.m   = 3;
        a.n   = 2;
        a.ptr = {0, 2, 3, 3};
        a.ind = {0, 1, 1};
        a.val = {cd(2, 0), cd(3, 0), cd(4, 0)};
        a.x   = {cd(5, 0), cd(7, 0)};
        a.y0  = {cd(1, 0), cd(2, 0), cd(3, 0)};
        return a;
    }

    // 20002 x 2: row 0 = {0, 1}, 20000 empty rows and a last row of {0}. An
    // adaptive block would span the empty rows, so the default algorithm
    // selects nnz-split, and row 0 reaches the wavefront that holds the last
    // nonzero.
    Csr make_gap_first_row_reaches_tail()
    {
        Csr a;
        a.n = 2;
        a.add_row(0, 2);
        for(int i = 0; i < 20000; ++i)
        {
            a.add_row(0, 0);
        }
        a.add_row(0, 1);
        return a;
    }

    // Row 0 uses column 1, and row 1 uses column 0, so x[0] is read and there
    // is no (0, 0) entry. The transposed product is [3, NaN, 9].
    Csr make_nan_in_x0()
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        Csr          a;
        a.m   = 3;
        a.n   = 3;
        a.ptr = {0, 1, 3, 4};
        a.ind = {1, 0, 2, 2};
        a.val = {cd(2, 0), cd(3, 0), cd(1, 0), cd(4, 0)};
        a.x   = {cd(nan, 0), cd(1, 0), cd(2, 0)};
        return a;
    }

    // [[1, 2], [2, 3]] stored as its upper triangle, which is also the lower
    // triangle of a CSC matrix with the same arrays. Row 0 holds two of the
    // three nonzeros. y = 2 A x + 3 y0 = [13, 22], not [23, 22], under every
    // operation.
    Csr make_symmetric_upper_2x2()
    {
        Csr a = make_upper({2, 1});
        a.val = {cd(1, 0), cd(2, 0), cd(3, 0)};
        a.x   = {cd(1, 0), cd(2, 0)};
        a.y0  = {cd(1, 0), cd(2, 0)};
        return a;
    }

    // Shapes that stay at one nnz per thread. B is the number of nonzeros in a
    // block at that granularity and W the wavefront size.
    std::vector<Named> small_shapes(int64_t B, int64_t W)
    {
        std::vector<Named> s;
        s.emplace_back("first_row_reaches_tail", make_issue());

        for(int64_t len : {int64_t(1), int64_t(2), W - 1, W, W + 1, B - 1, B, B + 1, 3 * B + 5})
        {
            s.emplace_back("single_row_" + std::to_string(len), make(len, {len}));
        }
        s.emplace_back("single_column_dense", make(1, std::vector<int64_t>(300, 1)));
        {
            std::vector<int64_t> lens;
            for(int i = 0; i < 300; ++i)
            {
                lens.push_back((i % 3 == 0) ? 0 : 1);
            }
            s.emplace_back("single_column_gaps", make(1, lens));
        }
        s.emplace_back("first_row_empty", make(8, {0, 3, 5, 2}));
        s.emplace_back("first_rows_empty", make(8, {0, 0, 0, 4, 4}));
        s.emplace_back("last_row_empty", make(8, {3, 4, 0}));
        s.emplace_back("last_rows_empty", make(8, {3, 0, 0, 0}));
        s.emplace_back("first_and_last_rows_empty", make(8, {0, 0, 3, 4, 0, 0}));
        s.emplace_back("empty_rows_between", make(8, {3, 0, 0, 0, 4, 0, 5}));
        for(int64_t len : {int64_t(5), B + 5})
        {
            for(int pos : {0, 3, 6})
            {
                std::vector<int64_t> lens(7, 0);
                lens[static_cast<size_t>(pos)] = len;
                s.emplace_back("only_row_" + std::to_string(pos) + "_len_" + std::to_string(len),
                               make(len, lens));
            }
        }
        s.emplace_back("all_rows_empty", make(4, {0, 0, 0}));
        // Each block adds its part of a row to y atomically, in any order, so
        // every partial sum of a row must be an integer a bf16 y holds exactly
        // (at most 256). A first row of 2 * B + 7 stays below that on wave64;
        // 3 * B + 7 does not.
        s.emplace_back("first_row_long", make(2 * B + 7, {2 * B + 7, 4, 5, 0, 6}));
        s.emplace_back("first_row_one_block", make(B, {B, 4, 5, 0, 6}));
        s.emplace_back("first_row_one_block_plus_one", make(B + 1, {B + 1, 4, 5, 0, 6}));
        s.emplace_back("last_row_long", make(3 * B + 7, {4, 5, 0, 3 * B + 7}));
        s.emplace_back("middle_row_long", make(2 * B + 9, {4, 2 * B + 9, 3}));
        s.emplace_back("two_long_rows", make(2 * B + 9, {2 * B + 9, 2 * B + 9}));
        s.emplace_back("rows_end_on_block_boundary", make(B, {B, 7, B, B, 0, B}));
        s.emplace_back("rows_cross_block_boundary_by_one", make(B + 1, {B + 1, B + 1, B + 1, 3}));
        s.emplace_back("rows_stop_short_of_block_boundary", make(B, {B - 1, B - 1, B - 1, 3}));
        s.emplace_back("many_rows_of_one", make(1, std::vector<int64_t>(3 * B + 17, 1)));
        s.emplace_back("dense_short_rows", make(3, std::vector<int64_t>(50, 3)));
        s.emplace_back("wide_rows", make(500, {100, 100, 100}));
        {
            std::vector<int64_t> lens;
            uint32_t             state = 12345u;
            for(int i = 0; i < 40; ++i)
            {
                state = state * 1664525u + 1013904223u;
                lens.push_back((state >> 16) % 31);
            }
            s.emplace_back("irregular", make(40, lens));
        }
        return s;
    }

    // Rows of one nonzero and a few long ones, so that the longest row is far
    // above the mean and the skew guard picks 8 nonzeros per thread. The head
    // rows come first; the tail rows follow the singles.
    Csr skewed(int64_t                     n,
               const std::vector<int64_t>& head,
               int64_t                     singles,
               const std::vector<int64_t>& tail)
    {
        std::vector<int64_t> lens = head;
        lens.insert(lens.end(), static_cast<size_t>(singles), 1);
        lens.insert(lens.end(), tail.begin(), tail.end());
        return make(n, lens);
    }

    // The same, for a small matrix. B is the number of nonzeros in a block at
    // 8 nnz per thread and W the wavefront size.
    std::vector<Named> skewed_shapes(int64_t B, int64_t W)
    {
        const int64_t      heavy = 600;
        const int64_t      n     = 4 * B;
        std::vector<Named> s;
        s.emplace_back("skew_heavy_first", skewed(n, {heavy}, 3000, {}));
        s.emplace_back("skew_heavy_last", skewed(n, {}, 3000, {heavy}));
        s.emplace_back("skew_heavy_middle", skewed(n, {}, 1500, {heavy}));
        s.emplace_back("skew_heavy_both_ends", skewed(n, {heavy}, 3000, {heavy}));
        s.emplace_back("skew_first_row_empty", skewed(n, {0, heavy}, 3000, {}));
        s.emplace_back("skew_first_rows_empty", skewed(n, {0, 0, 0, heavy}, 3000, {}));
        s.emplace_back("skew_last_row_empty", skewed(n, {heavy}, 3000, {0}));
        s.emplace_back("skew_last_rows_empty", skewed(n, {heavy}, 3000, {0, 0, 0}));
        s.emplace_back("skew_first_row_spans_blocks", skewed(n, {3 * B + 17}, 3000, {}));
        s.emplace_back("skew_last_row_spans_blocks", skewed(n, {}, 3000, {3 * B + 17}));
        s.emplace_back("skew_first_row_one_block", skewed(n, {B}, 3000, {}));
        s.emplace_back("skew_first_row_one_block_plus_one", skewed(n, {B + 1}, 3000, {}));
        s.emplace_back("skew_first_row_one_block_minus_one", skewed(n, {B - 1}, 3000, {}));
        s.emplace_back("skew_heavy_rows_adjacent", skewed(n, {heavy, heavy, 0, heavy}, 3000, {}));
        {
            std::vector<int64_t> lens{heavy};
            lens.insert(lens.end(), 3000, 0);
            s.emplace_back("skew_only_first_row_filled", make(n, lens));
        }
        {
            std::vector<int64_t> head{heavy};
            head.insert(head.end(), 700, 0);
            s.emplace_back("skew_many_empty_rows", skewed(n, head, 700, {}));
        }

        // The whole matrix fits in one block, and the first row is not empty.
        s.emplace_back("skew_one_block_heavy_first", skewed(n, {300}, 99, {}));
        s.emplace_back("skew_one_block_heavy_second", skewed(n, {1, 300}, 98, {}));
        {
            std::vector<int64_t> lens(50, 1);
            lens.push_back(300);
            lens.insert(lens.end(), 50, 1);
            s.emplace_back("skew_one_block_heavy_middle", make(n, lens));
        }
        s.emplace_back("skew_one_block_heavy_last", skewed(n, {1}, 98, {300}));
        {
            std::vector<int64_t> lens(200, 0);
            lens[0] = 1;
            lens[1] = 70;
            s.emplace_back("skew_one_wavefront_heavy_second", make(n, lens));
            lens[0] = 70;
            lens[1] = 0;
            s.emplace_back("skew_one_wavefront_heavy_first", make(n, lens));
            lens[0]   = 3;
            lens[1]   = 0;
            lens[199] = 70;
            s.emplace_back("skew_one_wavefront_heavy_last", make(n, lens));
        }

        // The long first row ends in the same wavefront slice as the last
        // nonzero, behind which the slice is padded. The empty rows keep the
        // mean low enough for the skew guard. W / 4 is 8 on wave32, so each
        // tail length is kept once.
        std::vector<int64_t> tails;
        for(int64_t tail : {int64_t(1), int64_t(2), int64_t(3), int64_t(8), W / 4})
        {
            if(std::find(tails.begin(), tails.end(), tail) == tails.end())
            {
                tails.push_back(tail);
            }
        }
        for(int64_t tail : tails)
        {
            for(int64_t head : {9 * W + W / 2, B + W / 2, 2 * B + W / 2 + 3})
            {
                std::vector<int64_t> lens{head};
                lens.insert(lens.end(), 100, 0);
                lens.insert(lens.end(), static_cast<size_t>(tail), 1);
                s.emplace_back("skew_head_reaches_tail_" + std::to_string(head) + "_"
                                   + std::to_string(tail),
                               make(n, lens));
            }
        }
        return s;
    }

    // Large matrices whose density selects the granularity. nnz_needed is the
    // size above which the low-volume rule stops selecting one nnz per thread.
    std::vector<int64_t> uniform_lens(int64_t nnz_needed, int64_t lo, int64_t span)
    {
        std::vector<int64_t> lens;
        int64_t              total = 0;
        for(int64_t r = 0; total < nnz_needed + 2000; ++r)
        {
            const int64_t len = lo + (r * 37) % span;
            lens.push_back(len);
            total += len;
        }
        return lens;
    }

    // ---- type sets --------------------------------------------------------

    // A is the matrix type, X and Y the vector types and T the compute type.
#define NNZSPLIT_FOR_EACH_TYPE_SET(F)                                                       \
    F(float, float, float, float)                                                           \
    F(double, double, double, double)                                                       \
    F(rocsparse_float_complex,                                                              \
      rocsparse_float_complex,                                                              \
      rocsparse_float_complex,                                                              \
      rocsparse_float_complex)                                                              \
    F(rocsparse_double_complex,                                                             \
      rocsparse_double_complex,                                                             \
      rocsparse_double_complex,                                                             \
      rocsparse_double_complex)                                                             \
    F(_Float16, _Float16, _Float16, float)                                                  \
    F(_Float16, _Float16, float, float)                                                     \
    F(rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16, float)                    \
    F(rocsparse_bfloat16, rocsparse_bfloat16, float, float)                                 \
    F(int8_t, int8_t, int32_t, int32_t)                                                     \
    F(int8_t, int8_t, float, float)                                                         \
    F(float, rocsparse_float_complex, rocsparse_float_complex, rocsparse_float_complex)     \
    F(float, double, double, double)                                                        \
    F(double, rocsparse_double_complex, rocsparse_double_complex, rocsparse_double_complex) \
    F(rocsparse_float_complex,                                                              \
      rocsparse_double_complex,                                                             \
      rocsparse_double_complex,                                                             \
      rocsparse_double_complex)

    class CsrmvNnzsplit : public HandleTest
    {
    protected:
        hipDeviceProp_t prop{};
        uint32_t        block = 0;

        void SetUp() override
        {
            HandleTest::SetUp();
            g_reports  = 0;
            int device = 0;
            ASSERT_EQ(hipGetDevice(&device), hipSuccess);
            ASSERT_EQ(hipGetDeviceProperties(&prop, device), hipSuccess);
            // The block holds four wavefronts.
            block = 4u * static_cast<uint32_t>(prop.warpSize);
        }

        int64_t wavefront() const
        {
            return prop.warpSize;
        }

        // Nonzeros that stop the low-volume rule from selecting 1 nnz per
        // thread.
        int64_t volume_threshold() const
        {
            uint32_t per_cu = 1u;
            if(prop.maxThreadsPerMultiProcessor > 0)
            {
                per_cu = std::max<uint32_t>(
                    1u, static_cast<uint32_t>(prop.maxThreadsPerMultiProcessor) / block);
            }
            const int64_t saturation = static_cast<int64_t>(prop.multiProcessorCount) * per_cu;
            return saturation * static_cast<int64_t>(block) * 4;
        }

        // Checks the layout recorded by a v1 analysis. v1 never uses starting
        // block ids.
        void expect_npt(const Out& out, uint32_t npt, const std::string& what)
        {
            if(!out.ran)
            {
                report(what + ": the case did not run");
                return;
            }
            if(out.npt != npt)
            {
                report(what + ": the analysis selected " + std::to_string(out.npt)
                       + " nnz per thread, the case is meant for " + std::to_string(npt));
            }
            if(out.block != block)
            {
                report(what + ": the analysis recorded a block of " + std::to_string(out.block)
                       + " threads, expected " + std::to_string(block));
            }
            if(out.use_ids_recorded)
            {
                report(what + ": the v1 analysis recorded starting block ids");
            }
        }

        // Runs v1, v2 and v2 with starting block ids on host scalars.
        template <typename F>
        void each_mode(F&& f)
        {
            for(int mode = 0; mode < 3; ++mode)
            {
                Opt o;
                o.v2      = (mode > 0);
                o.use_ids = (mode == 2);
                f(o, std::string(mode == 0 ? "v1" : (mode == 1 ? "v2" : "v2+ids")));
            }
        }
    };

    template <uint32_t WFSIZE, uint32_t NPT>
    __global__ void
        k_coalesced_rows(int32_t* out, int32_t nnz, const int32_t* row_ptr, int32_t nrows)
    {
        constexpr uint32_t BLOCK = WFSIZE * 4;
        int32_t            rows[NPT];
        rocsparse::get_coalesced_row_indices<BLOCK, WFSIZE, NPT>(
            rows, int32_t(0), nrows, int32_t(0), nnz, row_ptr, rocsparse_index_base_zero);

        const uint32_t lid  = threadIdx.x & (WFSIZE - 1);
        const uint32_t wid  = threadIdx.x / WFSIZE;
        const int32_t  base = static_cast<int32_t>(wid * (WFSIZE * NPT));
        for(uint32_t i = 0; i < NPT; ++i)
        {
            out[base + static_cast<int32_t>(lid + i * WFSIZE)] = rows[i];
        }
    }

    // One block of four wavefronts. Both wavefront widths are instantiated; the
    // launch uses the device's width.
    template <uint32_t NPT>
    void launch_coalesced_rows(int32_t* out, int32_t nnz, const int32_t* row_ptr, int32_t nrows)
    {
        const int ws = rocsparse_ut::device_warp_size();
        if(ws == 64)
        {
            ASSERT_EQ(rocsparse_ut::launch_single_block(
                          k_coalesced_rows<64, NPT>, 64u * 4, out, nnz, row_ptr, nrows),
                      hipSuccess);
        }
        else
        {
            ASSERT_EQ(rocsparse_ut::launch_single_block(
                          k_coalesced_rows<32, NPT>, 32u * 4, out, nnz, row_ptr, nrows),
                      hipSuccess);
        }
    }

    template <uint32_t NPT>
    void check_coalesced_rows(int32_t nnz, const std::vector<int32_t>& row_ptr)
    {
        const int32_t ws = rocsparse_ut::device_warp_size();
        ASSERT_TRUE(ws == 32 || ws == 64);
        const int32_t                        slots = 4 * ws * static_cast<int32_t>(NPT);
        rocsparse_ut::device_vector<int32_t> d_ptr{row_ptr};
        rocsparse_ut::device_vector<int32_t> d_out(static_cast<size_t>(slots));
        launch_coalesced_rows<NPT>(
            d_out.ptr, nnz, d_ptr.ptr, static_cast<int32_t>(row_ptr.size() - 1));
        const std::vector<int32_t> got = rocsparse_ut::to_host(d_out);
        ASSERT_EQ(got.size(), static_cast<size_t>(slots));
        size_t bad = 0;
        for(int32_t pos = 0; pos < slots; ++pos)
        {
            int32_t expect = -1;
            if(pos < nnz)
            {
                const auto it = std::upper_bound(row_ptr.begin(), row_ptr.end(), pos);
                expect        = static_cast<int32_t>(it - row_ptr.begin()) - 1;
            }
            if(got[static_cast<size_t>(pos)] != expect && bad++ < 4)
            {
                ADD_FAILURE() << "npt " << NPT << " nnz " << nnz << " position " << pos
                              << " is row " << got[static_cast<size_t>(pos)] << ", expected "
                              << expect;
            }
        }
        EXPECT_EQ(bad, 0u);
    }
}

// Regression test for AISPARSE-1099 (GitHub #15869).
TEST_F(CsrmvNnzsplit, first_row_reaching_last_wavefront)
{
    const Csr a = make_issue();
    ASSERT_EQ(a.nnz(), 3);

    Opt o;
    o.v2          = false;
    const Out out = run_f32(handle, a, o, "issue v1");
    ASSERT_TRUE(out.ran);
    ASSERT_EQ(out.y.size(), 3u);
    EXPECT_EQ(out.y[0], cd(65, 0));
    EXPECT_EQ(out.y[1], cd(62, 0));
    EXPECT_EQ(out.y[2], cd(9, 0));
    expect_npt(out, 1, "issue v1");
}

TEST_F(CsrmvNnzsplit, first_row_reaching_last_wavefront_every_configuration)
{
    const Csr a = make_issue();
    for(bool device : {false, true})
    {
        for(rocsparse_index_base base : {rocsparse_index_base_zero, rocsparse_index_base_one})
        {
            each_mode([&](Opt o, const std::string& mode) {
                o.device               = device;
                o.base                 = base;
                const std::string what = "issue " + mode + (device ? " device" : " host")
                                         + (base == rocsparse_index_base_one ? " base1" : " base0");
                run<int32_t, int32_t, float, float, float, float>(handle, a, o, what + " i32/i32");
                run<int64_t, int32_t, float, float, float, float>(handle, a, o, what + " i64/i32");
                run<int64_t, int64_t, float, float, float, float>(handle, a, o, what + " i64/i64");
                run<int32_t, int32_t, double, double, double, double>(handle, a, o, what + " f64");
            });
        }
    }
}

// Same data, default algorithm and nnz-split. run() checks both against the
// host reference. The small shapes are good adaptive fits, so the default
// keeps adaptive there; the gap matrix must send it through nnz-split.
TEST_F(CsrmvNnzsplit, matches_default_algorithm)
{
    std::vector<Named> shapes = small_shapes(block, wavefront());
    shapes.emplace_back("gap_first_row_reaches_tail", make_gap_first_row_reaches_tail());
    for(const Named& shape : shapes)
    {
        Opt ns;
        Opt def;
        def.alg        = rocsparse_spmv_alg_default;
        const Out got  = run_f32(handle, shape.second, ns, shape.first + " nnzsplit");
        const Out want = run_f32(handle, shape.second, def, shape.first + " default");
        ASSERT_TRUE(got.ran && want.ran) << shape.first;
    }

    const Csr a = make_gap_first_row_reaches_tail();
    for(rocsparse_index_base base : {rocsparse_index_base_zero, rocsparse_index_base_one})
    {
        each_mode([&](Opt o, const std::string& mode) {
            o.alg                  = rocsparse_spmv_alg_default;
            o.base                 = base;
            const std::string what = "gap_first_row_reaches_tail default " + mode
                                     + (base == rocsparse_index_base_one ? " base1" : " base0");
            const Out out = run_f32(handle, a, o, what + " f32");
            if(!o.v2)
            {
                EXPECT_GT(out.npt, 0u) << what << ": the default algorithm kept adaptive";
            }
            run_f64(handle, a, o, what + " f64");
        });
    }
}

TEST_F(CsrmvNnzsplit, small_shapes)
{
    for(const Named& shape : small_shapes(block, wavefront()))
    {
        each_mode([&](Opt o, const std::string& mode) {
            const std::string what = shape.first + " " + mode;
            const Out         out  = run_f32(handle, shape.second, o, what);
            if(!o.v2 && shape.second.nnz() > 0)
            {
                expect_npt(out, 1, what);
            }
        });
    }
}

TEST_F(CsrmvNnzsplit, small_shapes_scalars)
{
    struct Scalars
    {
        double alpha;
        double beta;
    };
    const Scalars scalars[] = {{2, 3}, {1, 0}, {1, 1}, {0, 3}, {0, 0}, {-1, 2}, {2, 0}, {0, 1}};

    // The long-row and empty-edge shapes are the ones that differ.
    const std::string  single_row = "single_row_" + std::to_string(wavefront() + 1);
    std::vector<Named> shapes;
    for(const Named& shape : small_shapes(block, wavefront()))
    {
        if(shape.first == "first_row_reaches_tail" || shape.first == "first_row_long"
           || shape.first == "first_row_empty" || shape.first == "last_row_empty"
           || shape.first == single_row || shape.first == "irregular"
           || shape.first == "all_rows_empty" || shape.first == "single_column_dense")
        {
            shapes.push_back(shape);
        }
    }
    ASSERT_EQ(shapes.size(), 8u);
    for(const Named& shape : shapes)
    {
        for(const Scalars& sc : scalars)
        {
            for(bool device : {false, true})
            {
                each_mode([&](Opt o, const std::string& mode) {
                    o.alpha  = sc.alpha;
                    o.beta   = sc.beta;
                    o.device = device;
                    run_f32(handle,
                            shape.second,
                            o,
                            shape.first + " " + mode + " alpha=" + std::to_string(sc.alpha)
                                + " beta=" + std::to_string(sc.beta)
                                + (device ? " device" : " host"));
                });
            }
        }
    }
}

TEST_F(CsrmvNnzsplit, small_shapes_index_types_and_bases)
{
    for(const Named& shape : small_shapes(block, wavefront()))
    {
        for(rocsparse_index_base base : {rocsparse_index_base_zero, rocsparse_index_base_one})
        {
            each_mode([&](Opt o, const std::string& mode) {
                o.base                 = base;
                const std::string what = shape.first + " " + mode
                                         + (base == rocsparse_index_base_one ? " base1" : " base0");
                run<int64_t, int32_t, float, float, float, float>(
                    handle, shape.second, o, what + " i64/i32");
                run<int64_t, int64_t, float, float, float, float>(
                    handle, shape.second, o, what + " i64/i64");
                run<int32_t, int32_t, float, float, float, float>(
                    handle, shape.second, o, what + " i32/i32");
            });
        }
    }
}

// The first block holds the head of the matrix; move the row boundary through
// the first two blocks, one nonzero at a time. These splits do not reproduce
// the first-row bug: row 0 is empty, ends before the last nonzero, or has
// enough nonzeros after it to block the scan. first_row_reaches_last_window
// covers that layout.
TEST_F(CsrmvNnzsplit, boundary_sweep_one_nnz_per_thread)
{
    const int64_t B = block;
    for(int64_t s = 0; s <= 2 * B + 3; ++s)
    {
        const int64_t      n = 4 * B + 40;
        std::vector<Named> shapes;
        shapes.emplace_back("split_" + std::to_string(s), make(n, {s, B + 5, 3, 0, 2 * B}));
        shapes.emplace_back("split_first_row_empty_" + std::to_string(s),
                            make(n, {0, s, B + 5, 3, 0, 2 * B}));
        shapes.emplace_back("split_tail_" + std::to_string(s), make(n, {B + 5, s, 3}));
        for(const Named& shape : shapes)
        {
            Opt       o;
            const Out out = run_f32(handle, shape.second, o, shape.first + " v1");
            expect_npt(out, 1, shape.first);
        }
        // Starting block ids use a different launch layout for long rows.
        if(s % 7 == 0 || s % B <= 2 || s % B >= B - 2)
        {
            Opt o;
            o.v2      = true;
            o.use_ids = true;
            run_f32(handle,
                    make(n, {s, B + 5, 3, 0, 2 * B}),
                    o,
                    "split_" + std::to_string(s) + " v2+ids");
            run_f32(
                handle, make(n, {0, s, 4 * B}), o, "split_empty_" + std::to_string(s) + " v2+ids");
        }
    }
}

// Row 0 shares the last wavefront with the last nonzero. head is the length
// of row 0 and tail is the number of nonzeros after it, still inside that
// wavefront.
TEST_F(CsrmvNnzsplit, first_row_reaches_last_window)
{
    const int64_t W = wavefront();
    for(int64_t head = 1; head < W; ++head)
    {
        for(int64_t tail = 1; tail <= W - head; ++tail)
        {
            const std::string what
                = "head_" + std::to_string(head) + "_tail_" + std::to_string(tail);
            Opt o;
            expect_npt(run_f32(handle, make(W, {head, tail}), o, what), 1, what);
        }
    }
}

// A wavefront covers wavefront() nonzeros at a time; sweep the row boundary
// across the wavefront and block edges of the 8 nnz per thread layout. As in
// the one-per-thread sweep, row 0 does not share the last wavefront with the
// last nonzero, so this does not reproduce the first-row bug.
TEST_F(CsrmvNnzsplit, boundary_sweep_eight_nnz_per_thread)
{
    const int64_t B = 8 * static_cast<int64_t>(block);
    const int64_t W = 8 * wavefront();

    std::vector<int64_t> splits;
    for(int64_t s = 0; s <= 40; ++s)
    {
        splits.push_back(s);
    }
    for(int64_t edge = W; edge <= 2 * B + W; edge += W)
    {
        for(int64_t d = -3; d <= 3; ++d)
        {
            splits.push_back(edge + d);
        }
    }

    for(int64_t s : splits)
    {
        const int64_t n = 4 * B;
        for(int variant = 0; variant < 3; ++variant)
        {
            std::vector<int64_t> head;
            if(variant == 0)
            {
                head = {s, 600};
            }
            else if(variant == 1)
            {
                head = {0, s, 600};
            }
            else
            {
                head = {600, s};
            }
            const Csr         a = skewed(n, head, 3000, {});
            const std::string what
                = "split_" + std::to_string(s) + "_variant_" + std::to_string(variant);
            Opt o;
            expect_npt(run_f32(handle, a, o, what + " v1"), 8, what);
            if(s % 5 == 0)
            {
                o.v2 = true;
                run_f32(handle, a, o, what + " v2");
                o.use_ids = true;
                run_f32(handle, a, o, what + " v2+ids");
            }
        }
    }
}

TEST_F(CsrmvNnzsplit, skewed_shapes_eight_nnz_per_thread)
{
    for(const Named& shape : skewed_shapes(8 * block, wavefront()))
    {
        each_mode([&](Opt o, const std::string& mode) {
            const std::string what = shape.first + " " + mode;
            const Out         out  = run_f32(handle, shape.second, o, what);
            if(!o.v2)
            {
                expect_npt(out, 8, what);
            }
            o.alpha = 1.0;
            o.beta  = 0.0;
            run_f32(handle, shape.second, o, what + " beta=0");
            o.beta   = 1.0;
            o.device = true;
            run_f32(handle, shape.second, o, what + " beta=1 device");
        });
    }
}

TEST_F(CsrmvNnzsplit, skewed_shapes_index_types_and_bases)
{
    for(const Named& shape : skewed_shapes(8 * block, wavefront()))
    {
        for(rocsparse_index_base base : {rocsparse_index_base_zero, rocsparse_index_base_one})
        {
            each_mode([&](Opt o, const std::string& mode) {
                o.base                 = base;
                const std::string what = shape.first + " " + mode
                                         + (base == rocsparse_index_base_one ? " base1" : " base0");
                run<int64_t, int32_t, double, double, double, double>(
                    handle, shape.second, o, what + " i64/i32");
                run<int64_t, int64_t, double, double, double, double>(
                    handle, shape.second, o, what + " i64/i64");
            });
        }
    }
}

// Large matrices, so that the volume rule leaves 1 nnz per thread and the
// density picks 4 or 8. The first row is short and never reaches the last
// block here, so these cases do not reproduce the issue; they guard the
// granularity selection and the multi-block launches.
TEST_F(CsrmvNnzsplit, density_selected_granularities)
{
    struct Density
    {
        const char* name;
        uint32_t    npt;
        int64_t     lo;
        int64_t     span;
    };
    const Density densities[] = {{"npt4", 4, 33, 30}, {"npt8", 8, 100, 40}};

    const int64_t needed = volume_threshold();
    for(const Density& dn : densities)
    {
        const int64_t        B    = static_cast<int64_t>(block) * dn.npt;
        const int64_t        n    = 4 * B;
        std::vector<int64_t> base = uniform_lens(needed, dn.lo, dn.span);
        const std::string    tag  = dn.name;
        // The row spans three blocks but stays below the skew guard (64 times
        // the mean row length, at least 64 * dn.lo). 3 * B + 5 crosses it at
        // 4 nnz per thread on wave64.
        const int64_t long_row = 2 * B + 5;

        std::vector<Named> shapes;
        shapes.emplace_back(tag + "_uniform", make(n, base));
        {
            auto lens = base;
            lens.insert(lens.begin(), 0);
            shapes.emplace_back(tag + "_first_row_empty", make(n, lens));
        }
        {
            auto lens = base;
            lens.push_back(0);
            shapes.emplace_back(tag + "_last_row_empty", make(n, lens));
        }
        {
            auto lens = base;
            lens.insert(lens.begin(), long_row);
            shapes.emplace_back(tag + "_first_row_spans_blocks", make(n, lens));
        }
        {
            auto lens = base;
            lens.push_back(long_row);
            shapes.emplace_back(tag + "_last_row_spans_blocks", make(n, lens));
        }
        {
            auto lens = base;
            lens.insert(lens.begin() + static_cast<std::ptrdiff_t>(lens.size() / 2), long_row);
            shapes.emplace_back(tag + "_middle_row_spans_blocks", make(n, lens));
        }
        {
            auto lens = base;
            lens.insert(lens.begin(), 4, 0);
            shapes.emplace_back(tag + "_first_rows_empty", make(n, lens));
        }
        for(int64_t s : {int64_t(1), int64_t(2), B - 2, B - 1, B, B + 1, B + 2, 2 * B - 1, 2 * B})
        {
            auto lens = base;
            lens.insert(lens.begin(), s);
            shapes.emplace_back(tag + "_head_row_" + std::to_string(s), make(n, lens));
        }

        for(const Named& shape : shapes)
        {
            Opt o;
            expect_npt(run_f32(handle, shape.second, o, shape.first + " v1"), dn.npt, shape.first);
            o.v2 = true;
            run_f32(handle, shape.second, o, shape.first + " v2");
            o.use_ids = true;
            run_f32(handle, shape.second, o, shape.first + " v2+ids");
        }

        // Below the volume threshold the same density stays at 1 nnz per
        // thread, which checks volume_threshold() against the library.
        {
            std::vector<int64_t> lens;
            int64_t              total = 0;
            for(size_t r = 0; r < base.size(); ++r)
            {
                if(total + base[r] >= needed - 2000)
                {
                    break;
                }
                lens.push_back(base[r]);
                total += base[r];
            }
            Opt o;
            expect_npt(run_f32(handle, make(n, lens), o, tag + "_below_volume_threshold v1"),
                       1,
                       tag + "_below_volume_threshold");
        }

        Opt o;
        o.alpha = 1.0;
        o.beta  = 0.0;
        run_f64(handle, make(n, base), o, tag + " f64 beta=0");
        o.device = true;
        o.alpha  = 2.0;
        o.beta   = 1.0;
        run_f64(handle, make(n, base), o, tag + " f64 device");
    }
}

#define NNZSPLIT_TYPE_TEST_BODY(A, X, Y, T)                                                      \
    for(const Named& shape : shapes)                                                             \
    {                                                                                            \
        each_mode([&](Opt o, const std::string& mode) {                                          \
            const std::string what                                                               \
                = std::string(#A "," #X "," #Y "," #T " ") + shape.first + " " + mode;           \
            const Out out = run<int32_t, int32_t, A, X, Y, T>(handle, shape.second, o, what);    \
            if(!o.v2)                                                                            \
            {                                                                                    \
                expect_npt(out, expected_npt, what);                                             \
            }                                                                                    \
            o.device = true;                                                                     \
            o.beta   = 0.0;                                                                      \
            run<int64_t, int64_t, A, X, Y, T>(handle, shape.second, o, what + " device beta=0"); \
        });                                                                                      \
    }

TEST_F(CsrmvNnzsplit, value_types_one_nnz_per_thread)
{
    std::vector<Named> shapes;
    for(const Named& shape : small_shapes(block, wavefront()))
    {
        if(shape.first == "first_row_reaches_tail" || shape.first == "first_row_long"
           || shape.first == "irregular" || shape.first == "first_row_empty"
           || shape.first == "rows_cross_block_boundary_by_one")
        {
            shapes.push_back(shape);
        }
    }
    ASSERT_EQ(shapes.size(), 5u);
    const uint32_t expected_npt = 1;
    NNZSPLIT_FOR_EACH_TYPE_SET(NNZSPLIT_TYPE_TEST_BODY)
}

TEST_F(CsrmvNnzsplit, value_types_eight_nnz_per_thread)
{
    std::vector<Named> shapes;
    const int64_t      n = 4 * block;
    shapes.emplace_back("skew_first", skewed(n, {200}, 700, {}));
    shapes.emplace_back("skew_last", skewed(n, {}, 700, {200}));
    shapes.emplace_back("skew_first_row_empty", skewed(n, {0, 200}, 700, {}));
    // Smallest shape whose long first row shares the last wavefront with the
    // last nonzero, so the 8-per-thread fix is exercised in every type.
    {
        const int64_t        head = 9 * wavefront() + wavefront() / 2;
        std::vector<int64_t> lens{head};
        lens.insert(lens.end(), 100, 0);
        lens.push_back(1);
        shapes.emplace_back("skew_head_reaches_tail", make(n, lens));
    }
    const uint32_t expected_npt = 8;
    NNZSPLIT_FOR_EACH_TYPE_SET(NNZSPLIT_TYPE_TEST_BODY)
}

// The same analysis serves every compute call; a later compute must not see
// state left by an earlier one.
TEST_F(CsrmvNnzsplit, repeated_compute_with_one_analysis)
{
    std::vector<Named> shapes;
    for(const Named& shape : small_shapes(block, wavefront()))
    {
        shapes.push_back(shape);
    }
    for(const Named& shape : skewed_shapes(8 * block, wavefront()))
    {
        shapes.push_back(shape);
    }
    for(const Named& shape : shapes)
    {
        each_mode([&](Opt o, const std::string& mode) {
            o.repeat = 3;
            run_f32(handle, shape.second, o, shape.first + " " + mode + " repeat");

            // Without a reset the previous y is overwritten, not accumulated.
            o.beta    = 0.0;
            o.alpha   = 1.0;
            o.reset_y = false;
            run_f32(handle, shape.second, o, shape.first + " " + mode + " repeat beta=0");
        });
    }
}

TEST_F(CsrmvNnzsplit, transposed_operations)
{
    std::vector<Named> shapes;
    shapes.emplace_back("rect_tall", make(4, std::vector<int64_t>(37, 2)));
    shapes.emplace_back("rect_wide", make(300, {100, 100, 100}));
    shapes.emplace_back("first_row_empty", make(8, {0, 3, 5, 2}));
    shapes.emplace_back("last_row_empty", make(8, {3, 4, 0}));
    shapes.emplace_back("first_row_spans_three_blocks",
                        make(3 * block + 7, {3 * block + 7, 4, 5, 0, 6}));
    shapes.emplace_back("last_row_long", make(3 * block + 7, {4, 5, 0, 3 * block + 7}));
    shapes.emplace_back("single_row", make(40, {40}));
    shapes.emplace_back("single_column", make(1, std::vector<int64_t>(300, 1)));
    shapes.emplace_back("first_row_reaches_tail", make_issue());
    for(const Named& shape : skewed_shapes(8 * block, wavefront()))
    {
        shapes.push_back(shape);
    }

    for(rocsparse_operation op :
        {rocsparse_operation_transpose, rocsparse_operation_conjugate_transpose})
    {
        for(const Named& shape : shapes)
        {
            each_mode([&](Opt o, const std::string& mode) {
                o.op                   = op;
                const std::string what = shape.first + " " + mode
                                         + (op == rocsparse_operation_transpose ? " T" : " H");
                run_f32(handle, shape.second, o, what + " f32");
                run<int64_t,
                    int32_t,
                    rocsparse_float_complex,
                    rocsparse_float_complex,
                    rocsparse_float_complex,
                    rocsparse_float_complex>(handle, shape.second, o, what + " c32");
                o.device = true;
                o.alpha  = 1.0;
                o.beta   = 0.0;
                run<int32_t,
                    int32_t,
                    rocsparse_double_complex,
                    rocsparse_double_complex,
                    rocsparse_double_complex,
                    rocsparse_double_complex>(handle, shape.second, o, what + " c64 device beta=0");
                o.base = rocsparse_index_base_one;
                run_f64(handle, shape.second, o, what + " f64 base1");
            });
        }
    }
}

TEST_F(CsrmvNnzsplit, symmetric_matrices)
{
    std::vector<Named> shapes;
    shapes.emplace_back("diagonal", make_lower(std::vector<int64_t>(40, 1)));
    shapes.emplace_back("lower_irregular", make_lower({1, 2, 0, 3, 4, 0, 2, 5, 1, 7}));
    shapes.emplace_back("first_row_empty", make_lower({0, 2, 3, 3}));
    shapes.emplace_back("last_row_long", [&] {
        std::vector<int64_t> lens(3 * block + 7, 1);
        lens.back() = 3 * block + 7;
        return make_lower(lens);
    }());
    shapes.emplace_back("skew_last_row_long", [&] {
        std::vector<int64_t> lens(3000, 1);
        lens.back() = 1000;
        return make_lower(lens);
    }());
    shapes.emplace_back("skew_first_rows_empty", [&] {
        std::vector<int64_t> lens(3000, 1);
        lens[0]     = 0;
        lens[1]     = 0;
        lens.back() = 800;
        return make_lower(lens);
    }());

    for(const Named& shape : shapes)
    {
        each_mode([&](Opt o, const std::string& mode) {
            o.symmetric = true;
            run_f32(handle, shape.second, o, shape.first + " " + mode + " symmetric f32");
            o.beta   = 0.0;
            o.alpha  = 1.0;
            o.device = true;
            run<int64_t,
                int64_t,
                rocsparse_double_complex,
                rocsparse_double_complex,
                rocsparse_double_complex,
                rocsparse_double_complex>(
                handle, shape.second, o, shape.first + " " + mode + " symmetric c64 device beta=0");
        });
    }

    // Neither row 0 nor column 0 holds a nonzero, so a NaN in x[0] must not
    // reach y. In the transposed pass a position past the last nonzero has
    // row 0 and column 0, so the diagonal skip drops it even without the x
    // guard. unused_x_entry_does_not_reach_y covers that guard.
    std::vector<Named> unused;
    unused.emplace_back("row_and_column_0_unused", make_lower({0, 1, 2, 2}));
    unused.emplace_back("skew_row_and_column_0_unused", [&] {
        std::vector<int64_t> lens(3000, 1);
        lens[0]     = 0;
        lens.back() = 800;
        return make_lower(lens);
    }());
    for(Named& shape : unused)
    {
        Csr& a = shape.second;
        a.x.resize(static_cast<size_t>(a.n));
        for(int64_t j = 0; j < a.n; ++j)
        {
            a.x[static_cast<size_t>(j)] = x_of(j);
        }
        a.x[0] = cd(std::numeric_limits<double>::quiet_NaN(), 0.0);
        each_mode([&](Opt o, const std::string& mode) {
            o.symmetric            = true;
            const std::string what = shape.first + " " + mode;
            const Out         out  = run_f32(handle, a, o, what + " symmetric f32");
            if(!o.v2 && shape.first == "skew_row_and_column_0_unused")
            {
                expect_npt(out, 8, what);
            }
            o.alpha = 1.0;
            o.beta  = 0.0;
            run_f64(handle, a, o, what + " symmetric f64 beta=0");
        });
    }
}

// A symmetric matrix runs the forward kernel under every operation, so an
// upper triangle, whose row 0 can hold more than one nonzero, reaches the
// first-row bug as a transposed CSR product and as a non-transposed CSC
// product too.
TEST_F(CsrmvNnzsplit, symmetric_upper_first_row)
{
    const Csr a = make_symmetric_upper_2x2();
    for(bool csc : {false, true})
    {
        for(rocsparse_operation op : {rocsparse_operation_none,
                                      rocsparse_operation_transpose,
                                      rocsparse_operation_conjugate_transpose})
        {
            each_mode([&](Opt o, const std::string& mode) {
                o.symmetric = true;
                o.csc       = csc;
                o.fill      = csc ? rocsparse_fill_mode_lower : rocsparse_fill_mode_upper;
                o.op        = op;
                const std::string what = std::string(csc ? "csc lower " : "csr upper ") + mode
                                         + " op=" + std::to_string(op);
                const Out out = run_f32(handle, a, o, what + " f32");
                ASSERT_TRUE(out.ran) << what;
                ASSERT_EQ(out.y.size(), 2u) << what;
                EXPECT_EQ(out.y[0], cd(13, 0)) << what;
                EXPECT_EQ(out.y[1], cd(22, 0)) << what;
                if(!o.v2)
                {
                    expect_npt(out, 1, what);
                }
                run_f64(handle, a, o, what + " f64");
                run<int32_t,
                    int32_t,
                    rocsparse_float_complex,
                    rocsparse_float_complex,
                    rocsparse_float_complex,
                    rocsparse_float_complex>(handle, a, o, what + " c32");
            });
        }
    }
}

// Upper triangles at 1 and 8 nonzeros per thread, as CSR upper and as CSC
// lower with the same arrays, under every operation. At 8 nonzeros per
// thread the long row 0 ends in the wavefront that holds the last nonzero,
// and the empty rows keep the mean low enough for the skew guard.
TEST_F(CsrmvNnzsplit, symmetric_upper_shapes)
{
    struct Shape
    {
        std::string name;
        Csr         a;
        uint32_t    npt;
    };
    const int64_t      W = wavefront();
    std::vector<Shape> shapes;
    shapes.push_back({"upper_first_row_full", make_upper({5, 1, 0, 2, 1}), 1});
    {
        std::vector<int64_t> lens(static_cast<size_t>(W), 0);
        lens[0] = W / 2;
        std::fill(lens.end() - 3, lens.end(), 1);
        shapes.push_back({"upper_first_row_reaches_tail", make_upper(lens), 1});
    }
    {
        std::vector<int64_t> lens;
        uint32_t             state = 12345u;
        for(int i = 0; i < 40; ++i)
        {
            state = state * 1664525u + 1013904223u;
            lens.push_back((state >> 16) % 31);
        }
        lens[0] = 40;
        shapes.push_back({"upper_irregular", make_upper(lens), 1});
    }
    for(int64_t head : {9 * W + W / 2, 8 * static_cast<int64_t>(block) + W / 2})
    {
        for(int64_t tail : {int64_t(1), int64_t(3)})
        {
            std::vector<int64_t> lens(static_cast<size_t>(head + 100), 0);
            lens[0] = head;
            std::fill(lens.end() - tail, lens.end(), 1);
            shapes.push_back({"upper_skew_head_reaches_tail_" + std::to_string(head) + "_"
                                  + std::to_string(tail),
                              make_upper(lens),
                              8});
        }
    }

    for(const Shape& shape : shapes)
    {
        ASSERT_GT(shape.a.ptr[1], 1) << shape.name;
        for(bool csc : {false, true})
        {
            for(rocsparse_operation op : {rocsparse_operation_none,
                                          rocsparse_operation_transpose,
                                          rocsparse_operation_conjugate_transpose})
            {
                each_mode([&](Opt o, const std::string& mode) {
                    o.symmetric = true;
                    o.csc       = csc;
                    o.fill      = csc ? rocsparse_fill_mode_lower : rocsparse_fill_mode_upper;
                    o.op        = op;
                    const std::string what = shape.name + (csc ? " csc lower " : " csr upper ")
                                             + mode + " op=" + std::to_string(op);
                    const Out out = run_f32(handle, shape.a, o, what + " f32");
                    if(!o.v2)
                    {
                        expect_npt(out, shape.npt, what);
                    }
                    o.alpha  = 1.0;
                    o.beta   = 0.0;
                    o.device = true;
                    run<int64_t,
                        int32_t,
                        rocsparse_double_complex,
                        rocsparse_double_complex,
                        rocsparse_double_complex,
                        rocsparse_double_complex>(handle, shape.a, o, what + " c64 device beta=0");
                });
            }
        }
    }
}

// Every matrix of up to four rows with at most three entries per row. Most of
// them fit in a single wavefront with unused lanes behind the last nonzero.
TEST_F(CsrmvNnzsplit, every_tiny_matrix)
{
    size_t count = 0;
    for(int64_t m = 1; m <= 4; ++m)
    {
        int64_t combos = 1;
        for(int64_t r = 0; r < m; ++r)
        {
            combos *= 4;
        }
        for(int64_t code = 0; code < combos; ++code)
        {
            std::vector<int64_t> lens;
            int64_t              rest = code;
            for(int64_t r = 0; r < m; ++r)
            {
                lens.push_back(rest % 4);
                rest /= 4;
            }
            const Csr         a = make(3, lens);
            const std::string what
                = "lens " + std::to_string(code) + " of " + std::to_string(m) + " rows";
            Opt       o;
            const Out out = run_f32(handle, a, o, what + " v1");
            if(a.nnz() > 0)
            {
                expect_npt(out, 1, what);
            }
            if(code % 5 == 0)
            {
                o.v2      = true;
                o.use_ids = true;
                o.beta    = 0.0;
                run_f32(handle, a, o, what + " v2+ids beta=0");
            }
            ++count;
        }
    }
    EXPECT_EQ(count, 340u);
}

// A position past the last nonzero must not read or accumulate anything. An x
// entry that no nonzero uses may hold NaN without reaching y.
TEST_F(CsrmvNnzsplit, unused_x_entry_does_not_reach_y)
{
    const int64_t B = block;

    std::vector<Named> shapes;
    shapes.emplace_back("first_row_empty", make(8, {0, 3, 5, 2}));
    shapes.emplace_back("first_row_empty_long", make(3 * B + 7, {0, 3 * B + 7, 5}));
    shapes.emplace_back("first_row_empty_skewed", skewed(4 * B, {0, 300}, 99, {}));
    shapes.emplace_back("first_row_empty_skewed_large", skewed(8 * B, {0, 600}, 3000, {}));

    const double nan = std::numeric_limits<double>::quiet_NaN();
    for(const Named& shape : shapes)
    {
        for(rocsparse_operation op : {rocsparse_operation_none,
                                      rocsparse_operation_transpose,
                                      rocsparse_operation_conjugate_transpose})
        {
            Csr a = shape.second;
            if(op == rocsparse_operation_none)
            {
                // Move every column right so that column 0 is unused.
                for(int64_t& c : a.ind)
                {
                    ++c;
                }
                ++a.n;
            }
            const int64_t xlen = (op == rocsparse_operation_none) ? a.n : a.m;
            a.x.resize(static_cast<size_t>(xlen));
            for(int64_t j = 0; j < xlen; ++j)
            {
                a.x[static_cast<size_t>(j)] = x_of(j);
            }
            a.x[0] = cd(nan, 0.0);

            each_mode([&](Opt o, const std::string& mode) {
                o.op                   = op;
                const std::string what = shape.first + " " + mode + " op=" + std::to_string(op);
                run_f32(handle, a, o, what + " f32");
                o.alpha = 1.0;
                o.beta  = 0.0;
                run_f64(handle, a, o, what + " f64 beta=0");
            });
        }
    }
}

// Empty dimensions and empty matrices take an early exit, which must still
// scale y by beta (or leave it alone when y is empty).
TEST_F(CsrmvNnzsplit, degenerate_sizes)
{
    std::vector<Named> shapes;
    shapes.emplace_back("m0", make(5, {}));
    shapes.emplace_back("n0", make(0, {0, 0, 0}));
    shapes.emplace_back("m0_n0", make(0, {}));
    shapes.emplace_back("all_rows_empty", make(4, {0, 0, 0}));
    ASSERT_EQ(shapes[0].second.m, 0);
    ASSERT_EQ(shapes[1].second.n, 0);

    for(const Named& shape : shapes)
    {
        for(rocsparse_operation op : {rocsparse_operation_none,
                                      rocsparse_operation_transpose,
                                      rocsparse_operation_conjugate_transpose})
        {
            each_mode([&](Opt o, const std::string& mode) {
                o.op                   = op;
                const std::string what = shape.first + " " + mode + " op=" + std::to_string(op);
                run_f32(handle, shape.second, o, what + " f32");
                o.beta   = 0.0;
                o.device = true;
                run<int64_t,
                    int64_t,
                    rocsparse_double_complex,
                    rocsparse_double_complex,
                    rocsparse_double_complex,
                    rocsparse_double_complex>(handle, shape.second, o, what + " c64 device beta=0");
            });
        }
    }
}

// A position past the last nonzero must not turn y[0] into NaN.
TEST_F(CsrmvNnzsplit, transposed_nan_in_x0_stays_off_y0)
{
    const Csr a = make_nan_in_x0();
    for(rocsparse_operation op :
        {rocsparse_operation_transpose, rocsparse_operation_conjugate_transpose})
    {
        each_mode([&](Opt o, const std::string& mode) {
            o.op    = op;
            o.alpha = 1.0;
            o.beta  = 0.0;
            run_f32(handle, a, o, mode + (op == rocsparse_operation_transpose ? " T" : " H"));
        });
    }
}

// The same arrays as a CSC matrix. A non-transposed CSC product runs the
// transposed CSR kernel on them, so A x is [3, NaN, 9] again.
TEST_F(CsrmvNnzsplit, csc_nan_in_x0_stays_off_y0)
{
    const Csr a = make_nan_in_x0();
    for(bool v2 : {false, true})
    {
        Opt o;
        o.csc                  = true;
        o.v2                   = v2;
        o.alpha                = 1.0;
        o.beta                 = 0.0;
        const std::string what = std::string("csc ") + (v2 ? "v2" : "v1");
        const Out         out  = run_f32(handle, a, o, what);
        ASSERT_TRUE(out.ran) << what;
        ASSERT_EQ(out.y.size(), 3u) << what;
        EXPECT_EQ(out.y[0], cd(3, 0)) << what;
        EXPECT_TRUE(std::isnan(out.y[1].real())) << what;
        EXPECT_EQ(out.y[2], cd(9, 0)) << what;
    }
}

// get_coalesced_row_indices at 4 nonzeros per thread is not reachable from the
// public API on a multi-CU device. Call it directly, on a block that ends
// part-way through a wavefront.
TEST_F(CsrmvNnzsplit, coalesced_row_indices_mark_positions_past_nnz)
{
    const auto run_npt = [](auto npt) {
        constexpr uint32_t NPT = decltype(npt)::value;
        const int32_t      ws  = rocsparse_ut::device_warp_size();
        ASSERT_TRUE(ws == 32 || ws == 64);
        const int32_t mid
            = ws * static_cast<int32_t>(NPT) + (ws * static_cast<int32_t>(NPT)) / 2 + 3;
        for(int32_t nnz : {int32_t(3), mid})
        {
            std::vector<int32_t> row_ptr{0};
            const int32_t        head = std::max<int32_t>(nnz - 5, 1);
            const int32_t        rest = nnz - head;
            row_ptr.push_back(head);
            for(int32_t k = 0; k < rest; ++k)
            {
                row_ptr.push_back(row_ptr.back() + 1);
            }
            if(rest == 0)
            {
                row_ptr.back() = nnz;
            }
            check_coalesced_rows<NPT>(nnz, row_ptr);
        }
    };
    run_npt(std::integral_constant<uint32_t, 1>{});
    run_npt(std::integral_constant<uint32_t, 4>{});
    run_npt(std::integral_constant<uint32_t, 8>{});
}
