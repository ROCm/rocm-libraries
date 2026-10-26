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
// Shared helpers for the extra-routine device unit tests: small host reference
// types (sparse / dense matrices computed in double-complex precision),
// deterministic matrix generators, RAII wrappers for the generic-API
// descriptors, and output comparison helpers.
//
// Every test built on these helpers uploads a small matrix, calls the public C
// API, and compares the COMPLETE output with a host reference.
//
#pragma once

#include "unit_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace extra_ut
{
    using cplx = std::complex<double>;

    // ----- value conversions ------------------------------------------------
    template <typename T>
    inline cplx to_c(T v)
    {
        return cplx(static_cast<double>(v), 0.0);
    }
    template <>
    inline cplx to_c<rocsparse_float_complex>(rocsparse_float_complex v)
    {
        return cplx(std::real(v), std::imag(v));
    }
    template <>
    inline cplx to_c<rocsparse_double_complex>(rocsparse_double_complex v)
    {
        return cplx(std::real(v), std::imag(v));
    }

    template <typename T>
    inline T from_c(cplx c)
    {
        return static_cast<T>(c.real());
    }
    template <>
    inline rocsparse_float_complex from_c<rocsparse_float_complex>(cplx c)
    {
        return rocsparse_float_complex(static_cast<float>(c.real()), static_cast<float>(c.imag()));
    }
    template <>
    inline rocsparse_double_complex from_c<rocsparse_double_complex>(cplx c)
    {
        return rocsparse_double_complex(c.real(), c.imag());
    }

    template <typename T>
    struct is_complex_t
    {
        static constexpr bool value = false;
    };
    template <>
    struct is_complex_t<rocsparse_float_complex>
    {
        static constexpr bool value = true;
    };
    template <>
    struct is_complex_t<rocsparse_double_complex>
    {
        static constexpr bool value = true;
    };

    // The test data are small integers (and exactly representable fractions),
    // so sums and products are computed exactly in float and double: results
    // are compared exactly (tolerance 0). The few tests whose result is not
    // exactly representable (e.g. deep triangular solves) pass an explicit,
    // measured and justified tolerance.

    // Round a value to what T can represent, so the host reference uses exactly
    // the numbers the device sees.
    template <typename T>
    inline cplx round_to(cplx c)
    {
        return to_c<T>(from_c<T>(c));
    }

    // ----- deterministic generator -----------------------------------------
    struct Rng
    {
        uint64_t s;
        explicit Rng(uint64_t seed = 1)
            : s(seed * 6364136223846793005ULL + 1442695040888963407ULL)
        {
        }
        uint32_t next()
        {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            return static_cast<uint32_t>(s >> 33);
        }
        int64_t range(int64_t lo, int64_t hi) // inclusive
        {
            return lo + static_cast<int64_t>(next() % static_cast<uint64_t>(hi - lo + 1));
        }
    };

    // Non-zero small value (so that sums/products are exactly representable).
    template <typename T>
    inline cplx rand_val(Rng& rng)
    {
        const int64_t re_mag  = rng.range(1, 4);
        const int64_t re_sign = rng.range(0, 1);
        double        im      = 0.0;
        if(is_complex_t<T>::value)
        {
            const int64_t im_mag  = rng.range(1, 3);
            const int64_t im_sign = rng.range(0, 1);
            im                    = static_cast<double>(im_mag) * (im_sign ? 1.0 : -1.0);
        }
        return cplx(static_cast<double>(re_mag) * (re_sign ? 1.0 : -1.0), im);
    }

    // Diagonal entry of a triangular test matrix: +-(2.5 to 4.5), plus a small imaginary
    // part for complex types.
    template <typename T>
    inline cplx rand_diag(Rng& rng)
    {
        const int64_t sign = rng.range(0, 1);
        const int64_t mag  = rng.range(0, 4);
        const double  im   = is_complex_t<T>::value ? 0.25 * rng.range(-2, 2) : 0.0;
        return cplx((sign ? 1.0 : -1.0) * (2.5 + 0.5 * mag), im);
    }

    // ----- host sparse matrix (CSR, zero based, double complex values) -----
    struct Csr
    {
        int64_t              m = 0, n = 0;
        std::vector<int64_t> ptr; // m + 1
        std::vector<int64_t> col;
        std::vector<cplx>    val;
        int64_t              nnz() const
        {
            return static_cast<int64_t>(col.size());
        }
    };

    // Matrix with the given number of non-zeros in each row (sorted, distinct
    // columns, random positions).
    template <typename T>
    inline Csr csr_with_row_lengths(int64_t m, int64_t n, const std::vector<int64_t>& len, Rng& rng)
    {
        Csr A;
        A.m = m;
        A.n = n;
        A.ptr.assign(static_cast<size_t>(m) + 1, 0);
        for(int64_t i = 0; i < m; ++i)
        {
            const int64_t     l = std::min<int64_t>(len[static_cast<size_t>(i)], n);
            std::set<int64_t> cols;
            if(l == n)
            {
                for(int64_t j = 0; j < n; ++j)
                    cols.insert(j);
            }
            else
            {
                while(static_cast<int64_t>(cols.size()) < l)
                    cols.insert(rng.range(0, n - 1));
            }
            for(int64_t j : cols)
            {
                A.col.push_back(j);
                A.val.push_back(round_to<T>(rand_val<T>(rng)));
            }
            A.ptr[static_cast<size_t>(i) + 1] = static_cast<int64_t>(A.col.size());
        }
        return A;
    }

    template <typename T>
    inline Csr random_csr(int64_t m, int64_t n, int64_t max_row_nnz, Rng& rng)
    {
        std::vector<int64_t> len(static_cast<size_t>(m));
        for(auto& l : len)
            l = rng.range(0, max_row_nnz);
        return csr_with_row_lengths<T>(m, n, len, rng);
    }

    inline Csr transpose(const Csr& A)
    {
        Csr T;
        T.m = A.n;
        T.n = A.m;
        T.ptr.assign(static_cast<size_t>(A.n) + 1, 0);
        for(int64_t c : A.col)
            ++T.ptr[static_cast<size_t>(c) + 1];
        for(int64_t j = 0; j < A.n; ++j)
            T.ptr[static_cast<size_t>(j) + 1] += T.ptr[static_cast<size_t>(j)];
        T.col.resize(A.col.size());
        T.val.resize(A.col.size());
        std::vector<int64_t> pos(T.ptr.begin(), T.ptr.end() - 1);
        for(int64_t i = 0; i < A.m; ++i)
            for(int64_t p = A.ptr[static_cast<size_t>(i)]; p < A.ptr[static_cast<size_t>(i) + 1];
                ++p)
            {
                const int64_t d = pos[static_cast<size_t>(A.col[static_cast<size_t>(p)])]++;
                T.col[static_cast<size_t>(d)] = i;
                T.val[static_cast<size_t>(d)] = A.val[static_cast<size_t>(p)];
            }
        return T;
    }

    // Row-major m x n dense expansion.
    inline std::vector<cplx> to_dense(const Csr& A)
    {
        std::vector<cplx> D(static_cast<size_t>(A.m * A.n), cplx(0.0, 0.0));
        for(int64_t i = 0; i < A.m; ++i)
            for(int64_t p = A.ptr[static_cast<size_t>(i)]; p < A.ptr[static_cast<size_t>(i) + 1];
                ++p)
                D[static_cast<size_t>(i * A.n + A.col[static_cast<size_t>(p)])]
                    += A.val[static_cast<size_t>(p)];
        return D;
    }

    // ----- host dense reference matrices -----------------------------------
    struct Dense
    {
        int64_t           rows = 0, cols = 0;
        std::vector<cplx> v; // row-major logical
        Dense() = default;
        Dense(int64_t r, int64_t c)
            : rows(r)
            , cols(c)
            , v(static_cast<size_t>(r * c), cplx(0.0, 0.0))
        {
        }
        cplx& at(int64_t i, int64_t j)
        {
            return v[static_cast<size_t>(i * cols + j)];
        }
        const cplx& at(int64_t i, int64_t j) const
        {
            return v[static_cast<size_t>(i * cols + j)];
        }
    };

    inline Dense dense_of(const Csr& A)
    {
        Dense D;
        D.rows = A.m;
        D.cols = A.n;
        D.v    = to_dense(A);
        return D;
    }

    inline Dense apply_op(const Dense& A, rocsparse_operation op)
    {
        if(op == rocsparse_operation_none)
            return A;
        Dense R(A.cols, A.rows);
        for(int64_t i = 0; i < A.rows; ++i)
            for(int64_t j = 0; j < A.cols; ++j)
                R.at(j, i) = (op == rocsparse_operation_conjugate_transpose) ? std::conj(A.at(i, j))
                                                                             : A.at(i, j);
        return R;
    }

    inline Dense matmul(const Dense& A, const Dense& B)
    {
        Dense C(A.rows, B.cols);
        for(int64_t i = 0; i < A.rows; ++i)
            for(int64_t k = 0; k < A.cols; ++k)
            {
                const cplx a = A.at(i, k);
                if(a == cplx(0.0, 0.0))
                    continue;
                for(int64_t j = 0; j < B.cols; ++j)
                    C.at(i, j) += a * B.at(k, j);
            }
        return C;
    }

    inline Dense axpby(cplx alpha, const Dense& X, cplx beta, const Dense& Y)
    {
        Dense R(X.rows, X.cols);
        for(size_t i = 0; i < R.v.size(); ++i)
            R.v[i] = alpha * X.v[i] + beta * Y.v[i];
        return R;
    }

    // Random dense matrix with entries rounded to T.
    template <typename T>
    inline Dense random_dense(int64_t r, int64_t c, Rng& rng)
    {
        Dense D(r, c);
        for(auto& x : D.v)
            x = round_to<T>(rand_val<T>(rng));
        return D;
    }

    // Storage of a logical dense matrix with order / leading dimension. The
    // padding between rows/columns is filled with `pad` so tests can check that
    // the library does not write outside the logical matrix.
    template <typename T>
    inline std::vector<T>
        store_dense(const Dense& D, rocsparse_order order, int64_t ld, cplx pad = cplx(-77.0, 0.0))
    {
        const int64_t  lines = (order == rocsparse_order_column) ? D.cols : D.rows;
        std::vector<T> out(static_cast<size_t>(std::max<int64_t>(ld * lines, 1)), from_c<T>(pad));
        for(int64_t i = 0; i < D.rows; ++i)
            for(int64_t j = 0; j < D.cols; ++j)
            {
                const int64_t idx = (order == rocsparse_order_column) ? j * ld + i : i * ld + j;
                out[static_cast<size_t>(idx)] = from_c<T>(D.at(i, j));
            }
        return out;
    }

    inline int64_t min_ld(const Dense& D, rocsparse_order order)
    {
        return std::max<int64_t>(1, (order == rocsparse_order_column) ? D.rows : D.cols);
    }

    // Host reference: solve M X = R for a dense triangular M.
    inline Dense solve_triangular(const Dense& M, bool lower, const Dense& R)
    {
        Dense         X(R.rows, R.cols);
        const int64_t m = M.rows;
        for(int64_t col = 0; col < R.cols; ++col)
            for(int64_t s = 0; s < m; ++s)
            {
                const int64_t i   = lower ? s : (m - 1 - s);
                cplx          sum = R.at(i, col);
                for(int64_t j = 0; j < m; ++j)
                    if(j != i && (lower ? j < i : j > i))
                        sum -= M.at(i, j) * X.at(j, col);
                X.at(i, col) = sum / M.at(i, i);
            }
        return X;
    }

    // ----- comparison ------------------------------------------------------
    // |got - want| <= tol * (1 + |want|); tol == 0 is an exact comparison.
    template <typename T>
    inline bool close(T got, cplx want, double tol = 0.0)
    {
        const cplx   g = to_c<T>(got);
        const double e = std::abs(g - want);
        return std::isfinite(e) && e <= tol * (1.0 + std::abs(want));
    }

    // Compare the whole logical matrix (and the untouched padding) of a dense
    // matrix read back from the device.
    template <typename T>
    inline void expect_dense_equal(const std::vector<T>& got,
                                   const Dense&          want,
                                   rocsparse_order       order,
                                   int64_t               ld,
                                   const char*           what,
                                   cplx                  pad = cplx(-77.0, 0.0),
                                   double                tol = 0.0)
    {
        const int64_t lines = (order == rocsparse_order_column) ? want.cols : want.rows;
        ASSERT_GE(static_cast<int64_t>(got.size()), std::max<int64_t>(ld * lines, 1)) << what;
        int bad = 0;
        for(int64_t i = 0; i < want.rows && bad < 5; ++i)
            for(int64_t j = 0; j < want.cols && bad < 5; ++j)
            {
                const int64_t idx = (order == rocsparse_order_column) ? j * ld + i : i * ld + j;
                if(!close<T>(got[static_cast<size_t>(idx)], want.at(i, j), tol))
                {
                    ++bad;
                    ADD_FAILURE() << what << "(" << i << "," << j
                                  << ") = " << to_c<T>(got[static_cast<size_t>(idx)])
                                  << " expected " << want.at(i, j);
                }
            }
        // padding rows must keep their sentinel
        const int64_t inner = (order == rocsparse_order_column) ? want.rows : want.cols;
        for(int64_t l = 0; l < lines && bad < 5; ++l)
            for(int64_t p = inner; p < ld && bad < 5; ++p)
            {
                const size_t idx = static_cast<size_t>(l * ld + p);
                if(!close<T>(got[idx], pad))
                {
                    ++bad;
                    ADD_FAILURE() << what << " padding at line " << l << " pos " << p
                                  << " was modified";
                }
            }
    }

    template <typename T>
    inline void expect_vec_equal(const std::vector<T>&    got,
                                 const std::vector<cplx>& want,
                                 const char*              what,
                                 double                   tol = 0.0)
    {
        ASSERT_GE(got.size(), want.size()) << what;
        int bad = 0;
        for(size_t i = 0; i < want.size() && bad < 5; ++i)
            if(!close<T>(got[i], want[i], tol))
            {
                ++bad;
                ADD_FAILURE() << what << "[" << i << "] = " << to_c<T>(got[i]) << " expected "
                              << want[i];
            }
    }

    template <typename I>
    inline void expect_idx_equal(const std::vector<I>&       got,
                                 const std::vector<int64_t>& want,
                                 int64_t                     base,
                                 const char*                 what)
    {
        ASSERT_GE(got.size(), want.size()) << what;
        int bad = 0;
        for(size_t i = 0; i < want.size() && bad < 5; ++i)
            if(static_cast<int64_t>(got[i]) - base != want[i])
            {
                ++bad;
                ADD_FAILURE() << what << "[" << i << "] = " << static_cast<int64_t>(got[i]) - base
                              << " expected " << want[i];
            }
    }

    // The geam headers do not promise sorted rows in C: sort every row of a
    // read-back C by column before comparing (each column carries bb values).
    template <typename I, typename J, typename T>
    inline void sort_rows(const std::vector<I>& ptr,
                          int64_t               base,
                          std::vector<J>&       col,
                          std::vector<T>&       val,
                          int64_t               bb = 1)
    {
        for(size_t i = 0; i + 1 < ptr.size(); ++i)
        {
            const int64_t b = static_cast<int64_t>(ptr[i]) - base;
            const int64_t e = static_cast<int64_t>(ptr[i + 1]) - base;
            if(b < 0 || e - b < 2 || e > static_cast<int64_t>(col.size())
               || e * bb > static_cast<int64_t>(val.size()))
                continue;
            std::vector<std::pair<J, std::vector<T>>> row;
            for(int64_t p = b; p < e; ++p)
                row.emplace_back(col[static_cast<size_t>(p)],
                                 std::vector<T>(val.begin() + p * bb, val.begin() + (p + 1) * bb));
            std::stable_sort(row.begin(), row.end(), [](const auto& x, const auto& y) {
                return x.first < y.first;
            });
            for(int64_t p = b; p < e; ++p)
            {
                const auto& r               = row[static_cast<size_t>(p - b)];
                col[static_cast<size_t>(p)] = r.first;
                std::copy(r.second.begin(), r.second.end(), val.begin() + p * bb);
            }
        }
    }

    // ----- movable device buffer -----------------------------------------
    template <typename T>
    struct DevBuf
    {
        T*     ptr = nullptr;
        size_t n   = 0;
        DevBuf()   = default;
        explicit DevBuf(const std::vector<T>& h)
            : n(std::max<size_t>(h.size(), 1))
        {
            std::vector<T> tmp(n);
            std::copy(h.begin(), h.end(), tmp.begin());
            if(hipMalloc(&ptr, n * sizeof(T)) != hipSuccess)
            {
                ADD_FAILURE() << "hipMalloc of " << n * sizeof(T) << " bytes failed";
                ptr = nullptr;
                return;
            }
            EXPECT_EQ(hipMemcpy(ptr, tmp.data(), n * sizeof(T), hipMemcpyHostToDevice), hipSuccess);
        }
        explicit DevBuf(size_t count)
            : n(std::max<size_t>(count, 1))
        {
            if(hipMalloc(&ptr, n * sizeof(T)) != hipSuccess)
            {
                ADD_FAILURE() << "hipMalloc of " << n * sizeof(T) << " bytes failed";
                ptr = nullptr;
            }
            else
                EXPECT_EQ(hipMemset(ptr, 0, n * sizeof(T)), hipSuccess);
        }
        DevBuf(const DevBuf&) = delete;
        DevBuf& operator=(const DevBuf&) = delete;
        DevBuf(DevBuf&& o) noexcept
            : ptr(o.ptr)
            , n(o.n)
        {
            o.ptr = nullptr;
            o.n   = 0;
        }
        DevBuf& operator=(DevBuf&& o) noexcept
        {
            if(this != &o)
            {
                if(ptr)
                    (void)hipFree(ptr);
                ptr   = o.ptr;
                n     = o.n;
                o.ptr = nullptr;
                o.n   = 0;
            }
            return *this;
        }
        ~DevBuf()
        {
            if(ptr)
                (void)hipFree(ptr);
        }
        std::vector<T> host() const
        {
            std::vector<T> h(n);
            if(ptr)
                (void)hipMemcpy(h.data(), ptr, n * sizeof(T), hipMemcpyDeviceToHost);
            return h;
        }
    };

    // ----- uploads ---------------------------------------------------------
    // Device buffers always hold at least one element so the pointers stay valid.
    template <typename T>
    inline DevBuf<T> up_val(const std::vector<cplx>& v)
    {
        std::vector<T> h(v.size());
        for(size_t i = 0; i < v.size(); ++i)
            h[i] = from_c<T>(v[i]);
        return DevBuf<T>(h);
    }

    template <typename I>
    inline DevBuf<I> up_idx(const std::vector<int64_t>& v, int64_t base = 0)
    {
        std::vector<I> h(v.size());
        for(size_t i = 0; i < v.size(); ++i)
            h[i] = static_cast<I>(v[i] + base);
        return DevBuf<I>(h);
    }

    template <typename T>
    inline DevBuf<T> up_raw(const std::vector<T>& h)
    {
        return DevBuf<T>(h);
    }

    // ----- RAII descriptors -------------------------------------------------
    struct SpmatH
    {
        rocsparse_spmat_descr p = nullptr;
        SpmatH()                = default;
        SpmatH(const SpmatH&)   = delete;
        SpmatH& operator=(const SpmatH&) = delete;
        ~SpmatH()
        {
            if(p)
                (void)rocsparse_destroy_spmat_descr(p);
        }
    };

    struct DnmatH
    {
        rocsparse_dnmat_descr p = nullptr;
        DnmatH()                = default;
        DnmatH(const DnmatH&)   = delete;
        DnmatH& operator=(const DnmatH&) = delete;
        ~DnmatH()
        {
            if(p)
                (void)rocsparse_destroy_dnmat_descr(p);
        }
    };

    struct DnvecH
    {
        rocsparse_dnvec_descr p = nullptr;
        DnvecH()                = default;
        DnvecH(const DnvecH&)   = delete;
        DnvecH& operator=(const DnvecH&) = delete;
        ~DnvecH()
        {
            if(p)
                (void)rocsparse_destroy_dnvec_descr(p);
        }
    };

    struct MatDescrH
    {
        rocsparse_mat_descr d = nullptr;
        explicit MatDescrH(rocsparse_index_base base = rocsparse_index_base_zero)
        {
            (void)rocsparse_create_mat_descr(&d);
            (void)rocsparse_set_mat_index_base(d, base);
        }
        MatDescrH(const MatDescrH&) = delete;
        MatDescrH& operator=(const MatDescrH&) = delete;
        ~MatDescrH()
        {
            if(d)
                (void)rocsparse_destroy_mat_descr(d);
        }
    };

    struct MatInfoH
    {
        rocsparse_mat_info i = nullptr;
        MatInfoH()
        {
            (void)rocsparse_create_mat_info(&i);
        }
        MatInfoH(const MatInfoH&) = delete;
        MatInfoH& operator=(const MatInfoH&) = delete;
        ~MatInfoH()
        {
            if(i)
                (void)rocsparse_destroy_mat_info(i);
        }
    };

    // RAII owner of a bare SpGEAM descriptor.
    struct SpgeamDescrH
    {
        rocsparse_spgeam_descr d = nullptr;
        SpgeamDescrH()
        {
            (void)rocsparse_create_spgeam_descr(&d);
        }
        SpgeamDescrH(const SpgeamDescrH&) = delete;
        SpgeamDescrH& operator=(const SpgeamDescrH&) = delete;
        ~SpgeamDescrH()
        {
            if(d)
                (void)rocsparse_destroy_spgeam_descr(d);
        }
    };

    // Sets every input of an SpGEAM descriptor (default algorithm, operations
    // none unless given). alpha / beta are the scalar pointers of the handle's
    // pointer mode.
    inline void spgeam_set_inputs(rocsparse_handle       handle,
                                  rocsparse_spgeam_descr d,
                                  rocsparse_datatype     scalar_type,
                                  rocsparse_datatype     compute_type,
                                  const void*            alpha,
                                  const void*            beta,
                                  rocsparse_operation    op_A = rocsparse_operation_none,
                                  rocsparse_operation    op_B = rocsparse_operation_none)
    {
        const rocsparse_spgeam_alg alg = rocsparse_spgeam_alg_default;
        auto put = [&](rocsparse_spgeam_input which, const void* p, size_t n) {
            ASSERT_EQ(rocsparse_spgeam_set_input(handle, d, which, p, n, nullptr),
                      rocsparse_status_success);
        };
        put(rocsparse_spgeam_input_alg, &alg, sizeof(alg));
        put(rocsparse_spgeam_input_scalar_datatype, &scalar_type, sizeof(scalar_type));
        put(rocsparse_spgeam_input_compute_datatype, &compute_type, sizeof(compute_type));
        put(rocsparse_spgeam_input_operation_A, &op_A, sizeof(op_A));
        put(rocsparse_spgeam_input_operation_B, &op_B, sizeof(op_B));
        put(rocsparse_spgeam_input_scalar_alpha, alpha, sizeof(void*));
        put(rocsparse_spgeam_input_scalar_beta, beta, sizeof(void*));
    }

    // Expected status of a call: an exact status, or any_error where the
    // public header documents a failure but not which status is returned.
    struct AnyError
    {
    };
    constexpr AnyError any_error{};
    struct Expected
    {
        bool             any;
        rocsparse_status status;
        Expected(rocsparse_status s)
            : any(false)
            , status(s)
        {
        }
        Expected(AnyError)
            : any(true)
            , status(rocsparse_status_success)
        {
        }
    };

    inline ::testing::AssertionResult status_is(rocsparse_status got, Expected want)
    {
        if(want.any ? got != rocsparse_status_success : got == want.status)
            return ::testing::AssertionSuccess();
        return ::testing::AssertionFailure()
               << "status " << got << ", expected "
               << (want.any ? std::string("an error") : std::to_string(want.status));
    }

    // A call of a C entry point with a stored list of valid arguments.
    // set<I>(v) returns a copy with argument number I replaced, so that every
    // invalid-argument check is one line:
    //   const auto call = make_call(rocsparse_foo, handle, n, x, y);
    //   EXPECT_EQ(call.set<0>(nullptr)(), rocsparse_status_invalid_handle);
    //   EXPECT_EQ(call.set<1>(-1).set<2>(nullptr)(), rocsparse_status_invalid_size);
    template <typename F, typename... A>
    class ArgCall
    {
        F                f_;
        std::tuple<A...> a_;

    public:
        ArgCall(F f, A... a)
            : f_(f)
            , a_(a...)
        {
        }
        template <size_t I, typename V>
        ArgCall set(V v) const
        {
            ArgCall c         = *this;
            std::get<I>(c.a_) = v;
            return c;
        }
        rocsparse_status operator()() const
        {
            return std::apply(f_, a_);
        }
    };

    template <typename F, typename... A>
    ArgCall<F, A...> make_call(F f, A... a)
    {
        return ArgCall<F, A...>(f, a...);
    }

    struct PointerModeGuard
    {
        rocsparse_handle       handle = nullptr;
        rocsparse_pointer_mode prev   = rocsparse_pointer_mode_host;
        PointerModeGuard(rocsparse_handle h, rocsparse_pointer_mode mode)
            : handle(h)
        {
            (void)rocsparse_get_pointer_mode(handle, &prev);
            (void)rocsparse_set_pointer_mode(handle, mode);
        }
        PointerModeGuard(const PointerModeGuard&) = delete;
        PointerModeGuard& operator=(const PointerModeGuard&) = delete;
        ~PointerModeGuard()
        {
            (void)rocsparse_set_pointer_mode(handle, prev);
        }
    };

    // Scalars that are either host values or device pointers, depending on the
    // handle pointer mode that the caller selected.
    template <typename T>
    struct Scalar
    {
        T         h;
        DevBuf<T> d;
        bool      device;
        Scalar(cplx v, bool dev)
            : h(from_c<T>(v))
            , d(std::vector<T>{h})
            , device(dev)
        {
        }
        const void* get() const
        {
            return device ? static_cast<const void*>(d.ptr) : static_cast<const void*>(&h);
        }
        const T* typed() const
        {
            return device ? d.ptr : &h;
        }
    };

    inline rocsparse_pointer_mode mode_of(bool device)
    {
        return device ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host;
    }

    // Create a CSR descriptor from a host matrix on given index types / base.
    template <typename T, typename I, typename J>
    inline void make_csr_descr(SpmatH&              out,
                               const Csr&           A,
                               DevBuf<I>&           dptr,
                               DevBuf<J>&           dcol,
                               DevBuf<T>&           dval,
                               rocsparse_index_base base = rocsparse_index_base_zero)
    {
        dptr = up_idx<I>(A.ptr, static_cast<int64_t>(base));
        dcol = up_idx<J>(A.col, static_cast<int64_t>(base));
        dval = up_val<T>(A.val);
        ASSERT_TRUE(dptr.ptr && dcol.ptr && dval.ptr);
        ASSERT_EQ(rocsparse_create_csr_descr(&out.p,
                                             A.m,
                                             A.n,
                                             A.nnz(),
                                             dptr.ptr,
                                             dcol.ptr,
                                             dval.ptr,
                                             rocsparse_ut::it_of<I>(),
                                             rocsparse_ut::it_of<J>(),
                                             base,
                                             rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);
    }

    // ----- host block sparse matrix (BSR, zero based) ------------------------
    // Blocks are stored row-major (logical layout); bsr_pack() / bsr_unpack()
    // convert to and from the device layout of a given block direction.
    struct Bsr
    {
        int64_t              mb = 0, nb = 0, bd = 1;
        std::vector<int64_t> ptr; // mb + 1
        std::vector<int64_t> col;
        std::vector<cplx>    val; // nnzb * bd * bd
        int64_t              nnzb() const
        {
            return static_cast<int64_t>(col.size());
        }
    };

    template <typename T>
    inline Bsr random_bsr(int64_t mb, int64_t nb, int64_t bd, int64_t max_row_nnzb, Rng& rng)
    {
        const Csr pattern = random_csr<T>(mb, nb, max_row_nnzb, rng);
        Bsr       B;
        B.mb  = mb;
        B.nb  = nb;
        B.bd  = bd;
        B.ptr = pattern.ptr;
        B.col = pattern.col;
        B.val.resize(static_cast<size_t>(pattern.nnz() * bd * bd));
        for(auto& v : B.val)
            v = round_to<T>(rand_val<T>(rng));
        return B;
    }

    inline Bsr bsr_from_csr(const Csr& A)
    {
        Bsr B;
        B.mb  = A.m;
        B.nb  = A.n;
        B.bd  = 1;
        B.ptr = A.ptr;
        B.col = A.col;
        B.val = A.val;
        return B;
    }

    template <typename T>
    inline std::vector<T> bsr_pack(const Bsr& B, rocsparse_direction dir)
    {
        const int64_t  bd = B.bd;
        std::vector<T> out(B.val.size());
        for(int64_t p = 0; p < B.nnzb(); ++p)
            for(int64_t r = 0; r < bd; ++r)
                for(int64_t c = 0; c < bd; ++c)
                {
                    const size_t dst = static_cast<size_t>(
                        p * bd * bd + (dir == rocsparse_direction_row ? r * bd + c : c * bd + r));
                    out[dst] = from_c<T>(B.val[static_cast<size_t>(p * bd * bd + r * bd + c)]);
                }
        return out;
    }

    template <typename T>
    inline std::vector<cplx>
        bsr_unpack(const std::vector<T>& dev, int64_t nnzb, int64_t bd, rocsparse_direction dir)
    {
        std::vector<cplx> out(static_cast<size_t>(nnzb * bd * bd));
        for(int64_t p = 0; p < nnzb; ++p)
            for(int64_t r = 0; r < bd; ++r)
                for(int64_t c = 0; c < bd; ++c)
                    out[static_cast<size_t>(p * bd * bd + r * bd + c)]
                        = to_c<T>(dev[static_cast<size_t>(
                            p * bd * bd
                            + (dir == rocsparse_direction_row ? r * bd + c : c * bd + r))]);
        return out;
    }

    // C = alpha * A + beta * B (block pattern = union of both patterns).
    inline Bsr bsr_geam(cplx alpha, const Bsr& A, cplx beta, const Bsr& B)
    {
        const int64_t bd = A.bd;
        Bsr           C;
        C.mb = A.mb;
        C.nb = A.nb;
        C.bd = bd;
        C.ptr.assign(static_cast<size_t>(A.mb) + 1, 0);
        for(int64_t i = 0; i < A.mb; ++i)
        {
            std::vector<std::pair<int64_t, std::vector<cplx>>> row;
            auto                                               add_row = [&](const Bsr& M, cplx s) {
                for(int64_t p = M.ptr[static_cast<size_t>(i)];
                    p < M.ptr[static_cast<size_t>(i) + 1];
                    ++p)
                {
                    const int64_t j = M.col[static_cast<size_t>(p)];
                    auto          it = std::find_if(
                        row.begin(), row.end(), [&](const auto& e) { return e.first == j; });
                    if(it == row.end())
                    {
                        row.emplace_back(j, std::vector<cplx>(static_cast<size_t>(bd * bd)));
                        it = row.end() - 1;
                    }
                    for(int64_t e = 0; e < bd * bd; ++e)
                        it->second[static_cast<size_t>(e)]
                            += s * M.val[static_cast<size_t>(p * bd * bd + e)];
                }
            };
            add_row(A, alpha);
            add_row(B, beta);
            std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });
            for(const auto& e : row)
            {
                C.col.push_back(e.first);
                C.val.insert(C.val.end(), e.second.begin(), e.second.end());
            }
            C.ptr[static_cast<size_t>(i) + 1] = static_cast<int64_t>(C.col.size());
        }
        return C;
    }

    // C = alpha * A * B + beta * D, where `mul` / `add` select the terms.
    inline Bsr bsr_gemm(bool       mul,
                        cplx       alpha,
                        const Bsr& A,
                        const Bsr& B,
                        bool       add,
                        cplx       beta,
                        const Bsr& D,
                        int64_t    mb,
                        int64_t    nb)
    {
        const int64_t bd = D.bd;
        Bsr           C;
        C.mb = mb;
        C.nb = nb;
        C.bd = bd;
        C.ptr.assign(static_cast<size_t>(mb) + 1, 0);
        for(int64_t i = 0; i < mb; ++i)
        {
            std::map<int64_t, std::vector<cplx>> row;
            auto                                 slot = [&](int64_t j) -> std::vector<cplx>& {
                auto& blk = row[j];
                if(blk.empty())
                    blk.resize(static_cast<size_t>(bd * bd));
                return blk;
            };
            if(mul)
            {
                for(int64_t pa = A.ptr[static_cast<size_t>(i)];
                    pa < A.ptr[static_cast<size_t>(i) + 1];
                    ++pa)
                {
                    const int64_t kk = A.col[static_cast<size_t>(pa)];
                    for(int64_t pb = B.ptr[static_cast<size_t>(kk)];
                        pb < B.ptr[static_cast<size_t>(kk) + 1];
                        ++pb)
                    {
                        std::vector<cplx>& blk = slot(B.col[static_cast<size_t>(pb)]);
                        for(int64_t r = 0; r < bd; ++r)
                            for(int64_t c = 0; c < bd; ++c)
                                for(int64_t t = 0; t < bd; ++t)
                                    blk[static_cast<size_t>(r * bd + c)]
                                        += alpha
                                           * A.val[static_cast<size_t>(pa * bd * bd + r * bd + t)]
                                           * B.val[static_cast<size_t>(pb * bd * bd + t * bd + c)];
                    }
                }
            }
            if(add)
            {
                for(int64_t p = D.ptr[static_cast<size_t>(i)];
                    p < D.ptr[static_cast<size_t>(i) + 1];
                    ++p)
                {
                    std::vector<cplx>& blk = slot(D.col[static_cast<size_t>(p)]);
                    for(int64_t e = 0; e < bd * bd; ++e)
                        blk[static_cast<size_t>(e)]
                            += beta * D.val[static_cast<size_t>(p * bd * bd + e)];
                }
            }
            for(const auto& e : row)
            {
                C.col.push_back(e.first);
                C.val.insert(C.val.end(), e.second.begin(), e.second.end());
            }
            C.ptr[static_cast<size_t>(i) + 1] = static_cast<int64_t>(C.col.size());
        }
        return C;
    }

    // Which of the terms of C = alpha * A * B + beta * D take part.
    enum class Mode
    {
        multadd,
        mult,
        add
    };

    inline Bsr empty_bsr(int64_t mb, int64_t nb, int64_t bd)
    {
        Bsr E;
        E.mb  = mb;
        E.nb  = nb;
        E.bd  = bd;
        E.ptr = std::vector<int64_t>(static_cast<size_t>(mb) + 1, 0);
        return E;
    }

    // Compares a computed BSR matrix with the reference. With `sorted` the
    // blocks of every row have to come out sorted by column (guaranteed by the
    // documentation of csrgemm: "the output will be sorted"); the
    // documentation of bsrgemm and of the generic spgemm gives no such
    // guarantee, so there the blocks of a row are matched by column index and
    // the order the library emits them in is not checked.
    template <typename T, typename I, typename J>
    void expect_bsr_equal(const std::vector<I>& ptr,
                          const std::vector<J>& col,
                          const std::vector<T>& val_dev,
                          const Bsr&            want,
                          rocsparse_direction   dir,
                          int64_t               base,
                          bool                  sorted = false)
    {
        bool ptr_ok = ptr.size() >= want.ptr.size();
        for(size_t i = 0; ptr_ok && i < want.ptr.size(); ++i)
            ptr_ok = (static_cast<int64_t>(ptr[i]) - base == want.ptr[i]);
        if(!ptr_ok)
        {
            expect_idx_equal<I>(ptr, want.ptr, base, "row_ptr_C");
            return;
        }
        const int64_t bb      = want.bd * want.bd;
        const auto    got_val = bsr_unpack<T>(val_dev, want.nnzb(), want.bd, dir);
        for(int64_t i = 0; i < want.mb; ++i)
        {
            const int64_t        b = want.ptr[static_cast<size_t>(i)];
            const int64_t        e = want.ptr[static_cast<size_t>(i) + 1];
            std::vector<int64_t> order(static_cast<size_t>(e - b));
            for(int64_t p = b; p < e; ++p)
                order[static_cast<size_t>(p - b)] = p;
            if(!sorted)
                std::sort(order.begin(), order.end(), [&](int64_t x, int64_t y) {
                    return col[static_cast<size_t>(x)] < col[static_cast<size_t>(y)];
                });
            for(int64_t q = 0; q < e - b; ++q)
            {
                const int64_t p = order[static_cast<size_t>(q)];
                const int64_t w = b + q; // the reference is sorted by column
                if(static_cast<int64_t>(col[static_cast<size_t>(p)]) - base
                   != want.col[static_cast<size_t>(w)])
                {
                    ADD_FAILURE() << "row " << i << " block " << q << " column "
                                  << (col[static_cast<size_t>(p)] - base) << " expected "
                                  << want.col[static_cast<size_t>(w)];
                    return;
                }
                for(int64_t t = 0; t < bb; ++t)
                {
                    if(!close<T>(from_c<T>(got_val[static_cast<size_t>(p * bb + t)]),
                                 want.val[static_cast<size_t>(w * bb + t)]))
                    {
                        ADD_FAILURE()
                            << "row " << i << " block col " << want.col[static_cast<size_t>(w)]
                            << " entry " << t << " = " << got_val[static_cast<size_t>(p * bb + t)]
                            << " expected " << want.val[static_cast<size_t>(w * bb + t)];
                        return;
                    }
                }
            }
        }
    }

    // Storage formats of a generic-API sparse operand. bsr is only an
    // enumerator: make_sparse() does not build it (needs the block structure).
    enum class SFmt
    {
        csr,
        csc,
        coo,
        bsr
    };

    // Device storage + descriptor of the sparse operand.
    template <typename T, typename I, typename J>
    struct DevSparse
    {
        DevBuf<I> ptr; // CSR row ptr / CSC col ptr / COO row ind
        DevBuf<J> ind; // CSR col ind / CSC row ind
        DevBuf<I> coo_col;
        DevBuf<T> val;
        SpmatH    h;
    };

    template <typename T, typename I, typename J>
    void make_sparse(DevSparse<T, I, J>&  d,
                     SFmt                 fmt,
                     const Csr&           A,
                     rocsparse_index_base base = rocsparse_index_base_zero)
    {
        const rocsparse_datatype dt = rocsparse_ut::dt_of<T>();
        const int64_t            b  = static_cast<int64_t>(base);
        switch(fmt)
        {
        case SFmt::csr:
        {
            make_csr_descr<T, I, J>(d.h, A, d.ptr, d.ind, d.val, base);
            return;
        }
        case SFmt::csc:
        {
            const Csr At = transpose(A);
            d.ptr        = up_idx<I>(At.ptr, b);
            d.ind        = up_idx<J>(At.col, b);
            d.val        = up_val<T>(At.val);
            ASSERT_TRUE(d.ptr.ptr && d.ind.ptr && d.val.ptr);
            ASSERT_EQ(rocsparse_create_csc_descr(&d.h.p,
                                                 A.m,
                                                 A.n,
                                                 A.nnz(),
                                                 d.ptr.ptr,
                                                 d.ind.ptr,
                                                 d.val.ptr,
                                                 rocsparse_ut::it_of<I>(),
                                                 rocsparse_ut::it_of<J>(),
                                                 base,
                                                 dt),
                      rocsparse_status_success);
            return;
        }
        case SFmt::bsr:
        {
            ADD_FAILURE() << "make_sparse cannot build BSR (needs the block structure)";
            return;
        }
        case SFmt::coo:
        {
            std::vector<int64_t> rows;
            for(int64_t i = 0; i < A.m; ++i)
                for(int64_t p = A.ptr[static_cast<size_t>(i)];
                    p < A.ptr[static_cast<size_t>(i) + 1];
                    ++p)
                    rows.push_back(i);
            d.ptr     = up_idx<I>(rows, b);
            d.coo_col = up_idx<I>(A.col, b);
            d.val     = up_val<T>(A.val);
            ASSERT_TRUE(d.ptr.ptr && d.coo_col.ptr && d.val.ptr);
            ASSERT_EQ(rocsparse_create_coo_descr(&d.h.p,
                                                 A.m,
                                                 A.n,
                                                 A.nnz(),
                                                 d.ptr.ptr,
                                                 d.coo_col.ptr,
                                                 d.val.ptr,
                                                 rocsparse_ut::it_of<I>(),
                                                 base,
                                                 dt),
                      rocsparse_status_success);
            return;
        }
        }
    }

    // Typed access to the legacy s/d/c/z entry points.
    template <typename T>
    struct Api;
#define EXTRA_API_FN(P, NAME)             \
    template <typename... A>              \
    static rocsparse_status NAME(A... a)  \
    {                                     \
        return rocsparse_##P##NAME(a...); \
    }
#define EXTRA_API(TYPE, P)                   \
    template <>                              \
    struct Api<TYPE>                         \
    {                                        \
        EXTRA_API_FN(P, csrgemm_buffer_size) \
        EXTRA_API_FN(P, csrgemm)             \
        EXTRA_API_FN(P, bsrgemm_buffer_size) \
        EXTRA_API_FN(P, bsrgemm)             \
        EXTRA_API_FN(P, bsrgeam)             \
        EXTRA_API_FN(P, csrgeam)             \
        EXTRA_API_FN(P, bsrsm_buffer_size)   \
        EXTRA_API_FN(P, bsrsm_analysis)      \
        EXTRA_API_FN(P, bsrsm_solve)         \
        EXTRA_API_FN(P, gebsrmm)             \
        EXTRA_API_FN(P, gemmi)               \
    }
    EXTRA_API(float, s);
    EXTRA_API(double, d);
    EXTRA_API(rocsparse_float_complex, c);
    EXTRA_API(rocsparse_double_complex, z);
#undef EXTRA_API
#undef EXTRA_API_FN
}
