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
// Helpers shared by the level2 device unit tests: deterministic data, host sparse
// containers and formats, RAII descriptors, precision dispatch of the C API and
// host reference products.
//
#pragma once

#include "unit_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace level2
{
    using rocsparse_ut::device_vector;
    using rocsparse_ut::to_host;

    // ----- scalar traits ----------------------------------------------------
    template <typename T>
    struct real_of
    {
        using type = T;
    };
    template <>
    struct real_of<rocsparse_float_complex>
    {
        using type = float;
    };
    template <>
    struct real_of<rocsparse_double_complex>
    {
        using type = double;
    };

    template <typename T>
    constexpr bool is_single()
    {
        return std::is_same<typename real_of<T>::type, float>::value;
    }

    // Comparison tolerance used for products and solves of the small, exactly
    // representable data generated below.
    template <typename T>
    constexpr double tolerance()
    {
        return is_single<T>() ? 1.0e-4 : 1.0e-11;
    }

    template <typename T>
    inline double magnitude(const T& v)
    {
        return static_cast<double>(std::abs(v));
    }

    // Deterministic values: multiples of 1/16 in [-1.875, 1.875]. The real part has
    // period 61 and the imaginary part period 53 (both prime), so the sequence
    // does not repeat for any block size or row length used by the tests, and
    // an index that is off by a small multiple of a block dimension reads a
    // different value. All sums and products of these values (a few thousand
    // terms at most) are exact in single precision, which keeps the results of
    // atomic reductions independent of the summation order.
    // The value is zero for one `n` in 61 (a deliberate sprinkling of zeros);
    // with `nonzero` set the zero is replaced, so the value can be used for a
    // diagonal entry.
    template <typename T>
    inline T value_of(int64_t n, bool nonzero = false)
    {
        int64_t k = (n * 37 + 11) % 61 - 30;
        if(nonzero && k == 0)
            k = 7;
        return rocsparse_ut::scalar<T>(static_cast<float>(k) / 16.0f);
    }
    template <>
    inline rocsparse_float_complex value_of<rocsparse_float_complex>(int64_t n, bool nonzero)
    {
        const float im = static_cast<float>((n * 23 + 5) % 53 - 26) / 16.0f;
        return rocsparse_float_complex(value_of<float>(n, nonzero), im);
    }
    template <>
    inline rocsparse_double_complex value_of<rocsparse_double_complex>(int64_t n, bool nonzero)
    {
        const double im = static_cast<double>((n * 23 + 5) % 53 - 26) / 16.0;
        return rocsparse_double_complex(value_of<double>(n, nonzero), im);
    }

    // ----- comparison -------------------------------------------------------
    // Compares two vectors element by element. Reports the first mismatch (and
    // the count of mismatches) once, so a wrong kernel does not flood the log.
    template <typename T>
    inline bool expect_near(const std::vector<T>& want, const std::vector<T>& got, const char* what)
    {
        if(want.size() != got.size())
        {
            ADD_FAILURE() << what << ": size " << got.size() << " != " << want.size();
            return false;
        }
        const double tol   = tolerance<T>();
        size_t       bad   = 0;
        size_t       first = 0;
        for(size_t i = 0; i < want.size(); ++i)
        {
            const double err = magnitude(want[i] - got[i]);
            if(!(err <= tol * (1.0 + magnitude(want[i]))))
            {
                if(bad == 0)
                    first = i;
                ++bad;
            }
        }
        if(bad > 0)
        {
            ADD_FAILURE() << what << ": " << bad << " mismatches, first at " << first
                          << " want=" << want[first] << " got=" << got[first];
            return false;
        }
        return true;
    }

    // ----- RAII descriptors -------------------------------------------------
    // Owns a descriptor created by `create(&d)`; a failed creation leaves d null.
    template <typename D, auto create, auto destroy>
    struct created
    {
        D d = nullptr;
        created()
        {
            if(create(&d) != rocsparse_status_success)
                d = nullptr;
        }
        ~created()
        {
            if(d)
                (void)destroy(d);
        }
        created(const created&) = delete;
        created& operator=(const created&) = delete;
                 operator D() const
        {
            return d;
        }
    };

    // Owns a descriptor that the test creates itself (`create_*_descr(&x.d, ...)`).
    template <typename D, auto destroy>
    struct owned
    {
        D d                 = nullptr;
        owned()             = default;
        owned(const owned&) = delete;
        owned& operator=(const owned&) = delete;
        ~owned()
        {
            if(d)
                (void)destroy(d);
        }
        operator D() const
        {
            return d;
        }
    };

    using mat_descr
        = created<rocsparse_mat_descr, rocsparse_create_mat_descr, rocsparse_destroy_mat_descr>;
    using mat_info
        = created<rocsparse_mat_info, rocsparse_create_mat_info, rocsparse_destroy_mat_info>;
    using hyb_guard
        = created<rocsparse_hyb_mat, rocsparse_create_hyb_mat, rocsparse_destroy_hyb_mat>;
    using spmv_descr_guard
        = created<rocsparse_spmv_descr, rocsparse_create_spmv_descr, rocsparse_destroy_spmv_descr>;
    using sptrsv_descr_guard = created<rocsparse_sptrsv_descr,
                                       rocsparse_create_sptrsv_descr,
                                       rocsparse_destroy_sptrsv_descr>;
    using sptrsm_descr_guard = created<rocsparse_sptrsm_descr,
                                       rocsparse_create_sptrsm_descr,
                                       rocsparse_destroy_sptrsm_descr>;
    using spmat_guard        = owned<rocsparse_spmat_descr, rocsparse_destroy_spmat_descr>;
    using dnvec_guard        = owned<rocsparse_dnvec_descr, rocsparse_destroy_dnvec_descr>;
    using dnmat_guard        = owned<rocsparse_dnmat_descr, rocsparse_destroy_dnmat_descr>;

    struct scoped_pointer_mode
    {
        rocsparse_handle       handle;
        rocsparse_pointer_mode saved = rocsparse_pointer_mode_host;

        scoped_pointer_mode(rocsparse_handle h, rocsparse_pointer_mode mode)
            : handle(h)
        {
            EXPECT_EQ(rocsparse_get_pointer_mode(handle, &saved), rocsparse_status_success);
            EXPECT_EQ(rocsparse_set_pointer_mode(handle, mode), rocsparse_status_success);
        }

        scoped_pointer_mode(rocsparse_handle h, bool device)
            : scoped_pointer_mode(
                h, device ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host)
        {
        }

        ~scoped_pointer_mode()
        {
            (void)rocsparse_set_pointer_mode(handle, saved);
        }

        scoped_pointer_mode(const scoped_pointer_mode&) = delete;
        scoped_pointer_mode& operator=(const scoped_pointer_mode&) = delete;
    };

    inline rocsparse_index_base to_base(int base)
    {
        return base == 0 ? rocsparse_index_base_zero : rocsparse_index_base_one;
    }

    // ----- host containers --------------------------------------------------
    template <typename T>
    struct host_csr
    {
        int32_t              m    = 0;
        int32_t              n    = 0;
        int                  base = 0;
        std::vector<int32_t> ptr;
        std::vector<int32_t> ind;
        std::vector<T>       val;

        int32_t nnz() const
        {
            return static_cast<int32_t>(ind.size());
        }
    };

    // Applies a zero-based pattern to the requested index base.
    template <typename T>
    inline void rebase(host_csr<T>& a, int base)
    {
        if(a.base == base)
            return;
        const int d = base - a.base;
        for(auto& p : a.ptr)
            p += d;
        for(auto& i : a.ind)
            i += d;
        a.base = base;
    }

    template <typename T>
    inline T conj_of(const T& v)
    {
        return v;
    }
    inline rocsparse_float_complex conj_of(const rocsparse_float_complex& v)
    {
        return rocsparse_float_complex(std::real(v), -std::imag(v));
    }
    inline rocsparse_double_complex conj_of(const rocsparse_double_complex& v)
    {
        return rocsparse_double_complex(std::real(v), -std::imag(v));
    }

    // y = alpha * op(A) * x + beta * y for a host CSR matrix; the rows
    // [0, m) are all used. `trans` is none / transpose / conjugate_transpose.
    template <typename T>
    inline std::vector<T> csr_gemv(const host_csr<T>&    a,
                                   rocsparse_operation   trans,
                                   T                     alpha,
                                   const std::vector<T>& x,
                                   T                     beta,
                                   const std::vector<T>& y0)
    {
        std::vector<T> y(y0);
        for(auto& v : y)
            v = beta * v;
        for(int32_t i = 0; i < a.m; ++i)
        {
            for(int32_t k = a.ptr[i] - a.base; k < a.ptr[i + 1] - a.base; ++k)
            {
                const int32_t j = a.ind[k] - a.base;
                if(trans == rocsparse_operation_none)
                    y[i] = y[i] + alpha * a.val[k] * x[j];
                else if(trans == rocsparse_operation_transpose)
                    y[j] = y[j] + alpha * a.val[k] * x[i];
                else
                    y[j] = y[j] + alpha * conj_of(a.val[k]) * x[i];
            }
        }
        return y;
    }

    // A diagonally dominant, sorted n x n pattern with `per_row` off-diagonal
    // candidates per row on each side. The diagonal is present (non-zero) and
    // dominant, so triangular solves are well conditioned.
    //   lower / upper: keep only that triangle (plus the diagonal),
    //   both:          keep the full pattern.
    // `with_diag` false omits the diagonal entries (the pattern of a matrix with an
    // implicit unit diagonal).
    // Every off-diagonal entry is non-zero. The magnitude of the diagonal exceeds
    // the sum of the magnitudes of one triangle by a wide margin.
    template <typename T>
    inline host_csr<T> make_triangular_pattern(int32_t n,
                                               int32_t per_row,
                                               bool    lower,
                                               bool    upper,
                                               bool    with_diag,
                                               int     base        = 0,
                                               int32_t start_shift = 1)
    {
        host_csr<T> a;
        a.m    = n;
        a.n    = n;
        a.base = 0;
        a.ptr.push_back(0);
        int64_t counter = 0;
        for(int32_t i = 0; i < n; ++i)
        {
            std::vector<int32_t> cols;
            if(lower)
            {
                for(int32_t k = 1; k <= per_row; ++k)
                {
                    const int32_t j = i - k * start_shift;
                    if(j >= 0)
                        cols.push_back(j);
                }
            }
            if(with_diag)
                cols.push_back(i);
            if(upper)
            {
                for(int32_t k = 1; k <= per_row; ++k)
                {
                    const int32_t j = i + k * start_shift;
                    if(j < n)
                        cols.push_back(j);
                }
            }
            std::sort(cols.begin(), cols.end());
            for(int32_t j : cols)
            {
                a.ind.push_back(j);
                if(j == i)
                {
                    // dominant diagonal, real part positive
                    T d = value_of<T>(counter++, true);
                    d   = rocsparse_ut::scalar<T>(static_cast<float>(4 * per_row + 2)) + d;
                    a.val.push_back(d);
                }
                else
                {
                    T v = value_of<T>(counter++, true);
                    a.val.push_back(v);
                }
            }
            a.ptr.push_back(static_cast<int32_t>(a.ind.size()));
        }
        rebase(a, base);
        return a;
    }

    template <typename J, typename S>
    inline std::vector<J> convert_vector(const std::vector<S>& v)
    {
        return std::vector<J>(v.begin(), v.end());
    }

    // Device copy of column indices with one guard element (value -7, never a
    // valid column) on each side: the triangular-solve analysis kernels read the
    // entry that precedes or follows an empty row. `ptr` skips the first guard.
    template <typename J = rocsparse_int>
    struct guarded_indices
    {
        device_vector<J> buf;
        J*               ptr = nullptr;

        explicit guarded_indices(const std::vector<int32_t>& v)
            : buf(padded(v))
        {
            ptr = buf.ptr ? buf.ptr + 1 : nullptr;
        }
        operator J*() const
        {
            return ptr;
        }

    private:
        static std::vector<J> padded(const std::vector<int32_t>& v)
        {
            std::vector<J> p(v.size() + 2, static_cast<J>(-7));
            std::copy(v.begin(), v.end(), p.begin() + 1);
            return p;
        }
    };

    // Device copy of a host CSR matrix with row pointer type I and column index
    // type J (the columns are guarded, see guarded_indices).
    template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
    struct device_csr
    {
        device_vector<I>   ptr;
        guarded_indices<J> ind;
        device_vector<T>   val;

        explicit device_csr(const host_csr<T>& a)
            : ptr(convert_vector<I>(a.ptr))
            , ind(a.ind)
            , val(a.val.empty() ? std::vector<T>(1, T{}) : a.val)
        {
        }

        bool ok() const
        {
            return ptr.ptr && ind.ptr && val.ptr;
        }
    };

    template <typename T>
    inline std::vector<T> make_vector(size_t n, int64_t seed)
    {
        std::vector<T> v(n);
        for(size_t i = 0; i < n; ++i)
            v[i] = value_of<T>(static_cast<int64_t>(i) + seed);
        return v;
    }

    // ----- host reference ---------------------------------------------------------
    // Solves op(A) y = alpha * x with the triangle `fill` of A (and a unit or the
    // stored diagonal) by substitution on a dense copy.
    template <typename T>
    std::vector<T> host_triangular_solve(const host_csr<T>&    A,
                                         rocsparse_fill_mode   fill,
                                         rocsparse_diag_type   diag,
                                         rocsparse_operation   trans,
                                         T                     alpha,
                                         const std::vector<T>& x)
    {
        const int32_t  n = A.m;
        std::vector<T> M(static_cast<size_t>(n) * n, T{});
        for(int32_t i = 0; i < n; ++i)
        {
            for(int32_t k = A.ptr[i] - A.base; k < A.ptr[i + 1] - A.base; ++k)
            {
                const int32_t j = A.ind[k] - A.base;
                if(j == i)
                    M[static_cast<size_t>(i) * n + j] = A.val[k];
                else if(fill == rocsparse_fill_mode_lower ? (j < i) : (j > i))
                    M[static_cast<size_t>(i) * n + j] = A.val[k];
            }
            if(diag == rocsparse_diag_type_unit)
                M[static_cast<size_t>(i) * n + i] = rocsparse_ut::scalar<T>(1.0f);
        }

        std::vector<T> op(M);
        if(trans != rocsparse_operation_none)
        {
            for(int32_t r = 0; r < n; ++r)
                for(int32_t c = 0; c < n; ++c)
                {
                    const T v = M[static_cast<size_t>(c) * n + r];
                    op[static_cast<size_t>(r) * n + c]
                        = (trans == rocsparse_operation_conjugate_transpose) ? conj_of(v) : v;
                }
        }

        const bool lower_after
            = (fill == rocsparse_fill_mode_lower) == (trans == rocsparse_operation_none);
        std::vector<T> y(n, T{});
        for(int32_t s = 0; s < n; ++s)
        {
            const int32_t r   = lower_after ? s : n - 1 - s;
            T             sum = alpha * x[r];
            if(lower_after)
            {
                for(int32_t c = 0; c < r; ++c)
                    sum = sum - op[static_cast<size_t>(r) * n + c] * y[c];
            }
            else
            {
                for(int32_t c = r + 1; c < n; ++c)
                    sum = sum - op[static_cast<size_t>(r) * n + c] * y[c];
            }
            y[r] = sum / op[static_cast<size_t>(r) * n + r];
        }
        return y;
    }

    // ----- block (BSR / GEBSR) matrices ---------------------------------------
    template <typename T>
    struct host_bsr
    {
        int32_t              mb   = 0;
        int32_t              nb   = 0;
        int32_t              rbd  = 0;
        int32_t              cbd  = 0;
        int                  base = 0;
        rocsparse_direction  dir  = rocsparse_direction_row;
        std::vector<int32_t> ptr;
        std::vector<int32_t> ind;
        std::vector<T>       val;

        int32_t nnzb() const
        {
            return static_cast<int32_t>(ind.size());
        }

        // Entry (r, c) of block j (relative to the stored layout).
        const T& at(int32_t j, int32_t r, int32_t c) const
        {
            const size_t blk = static_cast<size_t>(rbd) * cbd;
            return (dir == rocsparse_direction_row) ? val[blk * j + r * cbd + c]
                                                    : val[blk * j + c * rbd + r];
        }
    };

    // `bpr` distinct sorted block columns in every block row.
    template <typename T>
    inline host_bsr<T> make_bsr(int32_t             mb,
                                int32_t             nb,
                                int32_t             bpr,
                                int32_t             rbd,
                                int32_t             cbd,
                                rocsparse_direction dir,
                                int                 base = 0)
    {
        host_bsr<T> a;
        a.mb  = mb;
        a.nb  = nb;
        a.rbd = rbd;
        a.cbd = cbd;
        a.dir = dir;
        a.ptr.push_back(0);
        const int32_t stride = std::max<int32_t>(1, nb / bpr);
        int64_t       seed   = 0;
        for(int32_t i = 0; i < mb; ++i)
        {
            std::vector<int32_t> cols;
            const int32_t        start = (i * 5) % nb;
            for(int32_t k = 0; k < bpr; ++k)
                cols.push_back((start + k * stride) % nb);
            std::sort(cols.begin(), cols.end());
            cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
            for(int32_t c : cols)
            {
                a.ind.push_back(c);
                for(int32_t e = 0; e < rbd * cbd; ++e)
                    a.val.push_back(value_of<T>(seed++));
            }
            a.ptr.push_back(static_cast<int32_t>(a.ind.size()));
        }
        if(base != 0)
        {
            for(auto& p : a.ptr)
                p += base;
            for(auto& c : a.ind)
                c += base;
            a.base = base;
        }
        return a;
    }

    // y[row range] = alpha * A_row * x + beta * y for the listed block rows.
    // `end_ptr` (when not empty) replaces ptr[i + 1] as the row end.
    template <typename T>
    inline std::vector<T> bsr_gemv(const host_bsr<T>&          a,
                                   const std::vector<int32_t>& rows,
                                   const std::vector<int32_t>& end_ptr,
                                   T                           alpha,
                                   const std::vector<T>&       x,
                                   T                           beta,
                                   const std::vector<T>&       y0)
    {
        std::vector<T> y(y0);
        for(int32_t i : rows)
        {
            const int32_t b = a.ptr[i] - a.base;
            const int32_t e = (end_ptr.empty() ? a.ptr[i + 1] : end_ptr[i]) - a.base;
            for(int32_t r = 0; r < a.rbd; ++r)
            {
                T sum = T{};
                for(int32_t j = b; j < e; ++j)
                {
                    const int32_t col = a.ind[j] - a.base;
                    for(int32_t c = 0; c < a.cbd; ++c)
                        sum = sum + a.at(j, r, c) * x[static_cast<size_t>(col) * a.cbd + c];
                }
                y[static_cast<size_t>(i) * a.rbd + r]
                    = beta * y[static_cast<size_t>(i) * a.rbd + r] + alpha * sum;
            }
        }
        return y;
    }

    inline std::vector<int32_t> all_rows(int32_t m)
    {
        std::vector<int32_t> rows(m);
        for(int32_t i = 0; i < m; ++i)
            rows[i] = i;
        return rows;
    }

    // A sorted (m x n) CSR matrix whose row i holds min(len_of(i), n) distinct
    // columns. Values are never zero.
    template <typename T, typename LenFn>
    inline host_csr<T> make_csr_by_length(int32_t m, int32_t n, int base, LenFn len_of)
    {
        host_csr<T> a;
        a.m    = m;
        a.n    = n;
        a.base = 0;
        a.ptr.push_back(0);
        int64_t counter = 0;
        for(int32_t i = 0; i < m; ++i)
        {
            // len * stride <= n, so the columns are distinct.
            const int32_t        len = std::min<int32_t>(len_of(i), n);
            std::vector<int32_t> cols(std::max<int32_t>(len, 0));
            if(len > 0)
            {
                const int64_t start  = (static_cast<int64_t>(i) * 13) % n;
                const int64_t stride = n / len;
                for(int32_t k = 0; k < len; ++k)
                    cols[k] = static_cast<int32_t>((start + k * stride) % n);
                std::sort(cols.begin(), cols.end());
            }
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

    // Rows of varying length: row i holds (i * 5 + 3) % 9 entries (none if
    // empty_every > 0 and i % empty_every == 0), row 1 holds `long_row` entries when
    // long_row > 0.
    template <typename T>
    inline host_csr<T>
        make_matrix(int32_t m, int32_t n, int base, int32_t long_row = 0, int32_t empty_every = 0)
    {
        return make_csr_by_length<T>(m, n, base, [=](int32_t i) -> int32_t {
            if(long_row > 0 && i == 1)
                return long_row;
            if(empty_every > 0 && i % empty_every == 0)
                return 0;
            return (i * 5 + 3) % 9;
        });
    }

    template <typename T>
    inline int32_t max_row_length(const host_csr<T>& a)
    {
        int32_t w = 0;
        for(int32_t i = 0; i < a.m; ++i)
            w = std::max(w, a.ptr[i + 1] - a.ptr[i]);
        return w;
    }

    template <typename T>
    struct host_coo
    {
        std::vector<int32_t> row;
        std::vector<int32_t> col;
        std::vector<T>       val;
    };

    // Row major sorted COO entries with the index base of `a`.
    template <typename T>
    inline host_coo<T> to_coo(const host_csr<T>& a)
    {
        host_coo<T> c;
        for(int32_t i = 0; i < a.m; ++i)
            for(int32_t k = a.ptr[i] - a.base; k < a.ptr[i + 1] - a.base; ++k)
            {
                c.row.push_back(i + a.base);
                c.col.push_back(a.ind[k]);
                c.val.push_back(a.val[k]);
            }
        return c;
    }

    template <typename T>
    struct host_ell
    {
        int32_t              width = 0;
        std::vector<int32_t> col; // column major: entry p of row i at p * m + i
        std::vector<T>       val;
    };

    template <typename T>
    inline host_ell<T> to_ell(const host_csr<T>& a)
    {
        host_ell<T> e;
        e.width = max_row_length(a);
        e.col.assign(static_cast<size_t>(e.width) * a.m, -1);
        e.val.assign(static_cast<size_t>(e.width) * a.m, T{});
        for(int32_t i = 0; i < a.m; ++i)
            for(int32_t k = a.ptr[i] - a.base, p = 0; k < a.ptr[i + 1] - a.base; ++k, ++p)
            {
                e.col[static_cast<size_t>(p) * a.m + i] = a.ind[k];
                e.val[static_cast<size_t>(p) * a.m + i] = a.val[k];
            }
        return e;
    }

    template <typename T>
    struct host_sell
    {
        int32_t              colval = 0;
        std::vector<int32_t> offsets; // nslices + 1, with the index base of the matrix
        std::vector<int32_t> col;
        std::vector<T>       val;
    };

    template <typename T>
    inline host_sell<T> to_sell(const host_csr<T>& a, int32_t slice_size)
    {
        host_sell<T>         s;
        const int32_t        nslices = (a.m - 1) / slice_size + 1;
        std::vector<int32_t> w(nslices, 0);
        s.offsets.assign(nslices + 1, 0);
        for(int32_t sl = 0; sl < nslices; ++sl)
        {
            for(int32_t r = sl * slice_size; r < std::min(a.m, (sl + 1) * slice_size); ++r)
                w[sl] = std::max(w[sl], a.ptr[r + 1] - a.ptr[r]);
            s.offsets[sl + 1] = s.offsets[sl] + w[sl] * slice_size;
        }
        s.colval = s.offsets[nslices];
        s.col.assign(s.colval, -1);
        s.val.assign(s.colval, T{});
        for(int32_t r = 0; r < a.m; ++r)
        {
            const int32_t sl = r / slice_size;
            const int32_t l  = r % slice_size;
            for(int32_t k = a.ptr[r] - a.base, p = 0; k < a.ptr[r + 1] - a.base; ++k, ++p)
            {
                s.col[s.offsets[sl] + p * slice_size + l] = a.ind[k];
                s.val[s.offsets[sl] + p * slice_size + l] = a.val[k];
            }
        }
        for(auto& o : s.offsets)
            o += a.base;
        return s;
    }

    inline size_t trans_size(rocsparse_operation trans, int32_t m, int32_t n, bool input)
    {
        const bool none = (trans == rocsparse_operation_none);
        return static_cast<size_t>((input == none) ? n : m);
    }

    // Scalar of type T with the given real part and, for complex types, imaginary part.
    template <typename T>
    inline T scalar_of(float re, float = 0.0f)
    {
        return rocsparse_ut::scalar<T>(re);
    }
    template <>
    inline rocsparse_float_complex scalar_of<rocsparse_float_complex>(float re, float im)
    {
        return rocsparse_float_complex(re, im);
    }
    template <>
    inline rocsparse_double_complex scalar_of<rocsparse_double_complex>(float re, float im)
    {
        return rocsparse_double_complex(re, im);
    }

    // alpha / beta of a test case: a complex scalar gets a non zero imaginary part
    // unless the real part is 0 or the neutral value (`neutral`), both of which are
    // special-cased by the library.
    template <typename T>
    inline T case_scalar(float re, float neutral)
    {
        return scalar_of<T>(re, (re != neutral && re != 0.0f) ? 0.25f : 0.0f);
    }

    // The operations of the products.
    const rocsparse_operation all_ops[] = {rocsparse_operation_none,
                                           rocsparse_operation_transpose,
                                           rocsparse_operation_conjugate_transpose};

    // ----- precision dispatch of the C API -----------------------------------
    // NAME<T>(args...) calls rocsparse_{s,d,c,z}NAME. The arguments are forwarded, so
    // device_vector and similar wrappers convert to the pointer type of the routine.
#define LEVEL2_DISPATCH(NAME)                                              \
    template <typename T, typename... Args>                                \
    inline rocsparse_status NAME(Args&&... args)                           \
    {                                                                      \
        if constexpr(std::is_same<T, float>::value)                        \
            return rocsparse_s##NAME(std::forward<Args>(args)...);         \
        else if constexpr(std::is_same<T, double>::value)                  \
            return rocsparse_d##NAME(std::forward<Args>(args)...);         \
        else if constexpr(std::is_same<T, rocsparse_float_complex>::value) \
            return rocsparse_c##NAME(std::forward<Args>(args)...);         \
        else                                                               \
            return rocsparse_z##NAME(std::forward<Args>(args)...);         \
    }

    LEVEL2_DISPATCH(bsrmv)
    LEVEL2_DISPATCH(bsrmv_analysis)
    LEVEL2_DISPATCH(bsrsv_analysis)
    LEVEL2_DISPATCH(bsrsv_buffer_size)
    LEVEL2_DISPATCH(bsrsv_solve)
    LEVEL2_DISPATCH(bsrxmv)
    LEVEL2_DISPATCH(coomv)
    LEVEL2_DISPATCH(csr2hyb)
    LEVEL2_DISPATCH(csrcolor)
    LEVEL2_DISPATCH(csritsv_analysis)
    LEVEL2_DISPATCH(csritsv_buffer_size)
    LEVEL2_DISPATCH(csritsv_solve)
    LEVEL2_DISPATCH(csritsv_solve_ex)
    LEVEL2_DISPATCH(csrmv)
    LEVEL2_DISPATCH(csrmv_analysis)
    LEVEL2_DISPATCH(doti)
    LEVEL2_DISPATCH(ellmv)
    LEVEL2_DISPATCH(gebsrmv)
    LEVEL2_DISPATCH(gemvi)
    LEVEL2_DISPATCH(gemvi_buffer_size)
    LEVEL2_DISPATCH(hybmv)
    LEVEL2_DISPATCH(roti)
#undef LEVEL2_DISPATCH

    template <typename V>
    inline rocsparse_status set_input(rocsparse_handle     handle,
                                      rocsparse_spmv_descr d,
                                      rocsparse_spmv_input input,
                                      const V&             value)
    {
        return rocsparse_spmv_set_input(handle, d, input, &value, sizeof(V), nullptr);
    }

    // Algorithm, operation and data types of a v2 spmv descriptor.
    inline rocsparse_status set_basic_inputs(rocsparse_handle     handle,
                                             rocsparse_spmv_descr d,
                                             rocsparse_spmv_alg   alg,
                                             rocsparse_operation  trans,
                                             rocsparse_datatype   scalar_type,
                                             rocsparse_datatype   compute_type)
    {
        rocsparse_status st = set_input(handle, d, rocsparse_spmv_input_alg, alg);
        if(st == rocsparse_status_success)
            st = set_input(handle, d, rocsparse_spmv_input_operation, trans);
        if(st == rocsparse_status_success)
            st = set_input(handle, d, rocsparse_spmv_input_scalar_datatype, scalar_type);
        if(st == rocsparse_status_success)
            st = set_input(handle, d, rocsparse_spmv_input_compute_datatype, compute_type);
        return st;
    }

    // Runs the analysis and compute stages of rocsparse_v2_spmv.
    inline void run_v2_spmv(rocsparse_handle      handle,
                            rocsparse_spmv_descr  descr,
                            rocsparse_spmat_descr A,
                            rocsparse_dnvec_descr x,
                            rocsparse_dnvec_descr y,
                            const void*           alpha,
                            const void*           beta)
    {
        for(const auto stage : {rocsparse_v2_spmv_stage_analysis, rocsparse_v2_spmv_stage_compute})
        {
            size_t bytes = 0;
            ASSERT_EQ(rocsparse_v2_spmv_buffer_size(handle, descr, A, x, y, stage, &bytes, nullptr),
                      rocsparse_status_success);
            device_vector<char> dbuf(std::max<size_t>(bytes, 1));
            ASSERT_TRUE(dbuf.ptr);
            ASSERT_EQ(rocsparse_v2_spmv(handle,
                                        descr,
                                        alpha,
                                        A,
                                        x,
                                        beta,
                                        y,
                                        stage,
                                        bytes,
                                        bytes ? dbuf.ptr : nullptr,
                                        nullptr),
                      rocsparse_status_success);
        }
    }

    // Argument validation table: `call` is invoked with the valid arguments `a...`, one
    // of which is replaced per check, e.g. `bad.check<M>(-1, rocsparse_status_invalid_size)`.
    // The position is a name from a local enum.
    template <typename F, typename... A>
    class arg_checker
    {
        F                m_call;
        std::tuple<A...> m_good;

    public:
        arg_checker(F call, A... good)
            : m_call(call)
            , m_good(good...)
        {
        }

        template <size_t I, typename V>
        void check(V bad, rocsparse_status want) const
        {
            auto args         = m_good;
            std::get<I>(args) = bad;
            EXPECT_EQ(std::apply(m_call, args), want) << "invalid argument #" << I;
        }

        // A rejected value: the library reports it as an invalid value or an invalid size.
        template <size_t I, typename V>
        void check_invalid(V bad) const
        {
            auto args         = m_good;
            std::get<I>(args) = bad;
            const auto status = std::apply(m_call, args);
            EXPECT_TRUE(status == rocsparse_status_invalid_value
                        || status == rocsparse_status_invalid_size)
                << "invalid argument #" << I << ", status " << status;
        }

        // For argument errors whose status code is not documented: any failure will do.
        template <size_t I, typename V>
        void check_fails(V bad) const
        {
            auto args         = m_good;
            std::get<I>(args) = bad;
            EXPECT_NE(std::apply(m_call, args), rocsparse_status_success)
                << "invalid argument #" << I;
        }
    };
} // namespace level2
