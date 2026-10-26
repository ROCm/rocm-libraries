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
// Device unit tests that drive the public C API dispatch of the generic sparse
// products and triangular solves: the matrix format and algorithm switches of SpMV
// and v2 SpMV, the segmented-atomic SpMM, the analysis policies and scalar
// conversions of SpTRSV and SpTRSM, and the csrsm block size ladder.
//
// The matrices are small bidiagonal matrices with distinct entries, the vectors have
// distinct entries and alpha != 1, beta != 0, so a transposed, conjugated or permuted
// operation or a dropped scalar changes the result, which is compared with a host
// reference. The conjugate transposed products use complex data. Only a configuration
// that the headers document as unsupported may stop with not_implemented, and its
// result is not checked; every other configuration must run.
//
#include "unit_test_device_level2_common.hpp"

using namespace rocsparse_ut;
using namespace level2;

namespace
{
    using cfloat = rocsparse_float_complex;

    constexpr auto f32   = rocsparse_datatype_f32_r;
    constexpr auto f64   = rocsparse_datatype_f64_r;
    constexpr auto op_n  = rocsparse_operation_none;
    constexpr auto op_t  = rocsparse_operation_transpose;
    constexpr auto op_c  = rocsparse_operation_conjugate_transpose;
    constexpr auto col   = rocsparse_order_column;
    constexpr auto row   = rocsparse_order_row;
    constexpr auto lo    = rocsparse_fill_mode_lower;
    constexpr auto up    = rocsparse_fill_mode_upper;
    constexpr auto nu    = rocsparse_diag_type_non_unit;
    constexpr auto un    = rocsparse_diag_type_unit;
    constexpr auto reuse = rocsparse_analysis_policy_reuse;
    constexpr auto force = rocsparse_analysis_policy_force;

    template <typename T>
    T alpha_of()
    {
        return scalar_of<T>(2.0f, 0.5f);
    }
    template <typename T>
    T beta_of()
    {
        return scalar_of<T>(-0.5f, 0.25f);
    }

    // ----- staged calls ------------------------------------------------------------

    // True when the next stage can run. A configuration documented as unsupported
    // may stop with not_implemented, any other status is a failure.
    bool proceed(rocsparse_status status, bool unsupported, const char* what)
    {
        if(status == rocsparse_status_success)
            return true;
        if(unsupported)
            EXPECT_EQ(status, rocsparse_status_not_implemented) << what;
        else
            ADD_FAILURE() << what << " returned status " << status;
        return false;
    }

    // SpMV and SpMM: one buffer, sized by the first stage, shared by the following ones.
    // Returns true when the compute stage ran.
    template <typename Stage, typename Call>
    bool run_buffered(Call  call,
                      Stage size,
                      Stage preprocess,
                      Stage compute,
                      int   preprocess_passes,
                      bool  unsupported)
    {
        size_t bytes = 0;
        if(!proceed(call(size, &bytes, nullptr), unsupported, "buffer_size"))
            return false;
        device_vector<char> buffer(std::max<size_t>(bytes, 1));
        EXPECT_NE(buffer.ptr, nullptr);
        void* p = (bytes && buffer.ptr) ? buffer.ptr : nullptr;
        for(int pass = 0; pass < preprocess_passes; ++pass)
            if(!proceed(call(preprocess, &bytes, p), unsupported, "preprocess"))
                return false;
        const bool ok = proceed(call(compute, &bytes, p), unsupported, "compute");
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        return ok;
    }

    // v2 SpMV, SpTRSV and SpTRSM: every stage queries its own buffer size.
    // Returns true when the last stage ran.
    template <typename Stage, typename Size, typename Run>
    bool run_stages(std::initializer_list<Stage> stages, Size size, Run run, bool unsupported)
    {
        for(const Stage stage : stages)
        {
            size_t bytes = 0;
            if(!proceed(size(stage, &bytes), unsupported, "buffer_size"))
                return false;
            device_vector<char> buffer(std::max<size_t>(bytes, 1));
            EXPECT_NE(buffer.ptr, nullptr);
            if(!proceed(run(stage, bytes, (bytes && buffer.ptr) ? buffer.ptr : nullptr),
                        unsupported,
                        "stage"))
                return false;
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        }
        return true;
    }

    // ----- data ----------------------------------------------------------------

    template <typename T>
    std::vector<T> values(size_t count, int64_t seed)
    {
        std::vector<T> v(count);
        for(size_t i = 0; i < count; ++i)
            v[i] = value_of<T>(seed + static_cast<int64_t>(i), true);
        return v;
    }

    // The CSR arrays of the transpose of a zero-based matrix, i.e. its CSC arrays.
    template <typename T>
    host_csr<T> transposed(const host_csr<T>& a)
    {
        host_csr<T> t;
        t.m = a.n;
        t.n = a.m;
        t.ptr.assign(a.n + 1, 0);
        for(const int32_t j : a.ind)
            ++t.ptr[j + 1];
        for(int32_t j = 0; j < a.n; ++j)
            t.ptr[j + 1] += t.ptr[j];
        t.ind.resize(a.ind.size());
        t.val.resize(a.val.size());
        std::vector<int32_t> next(t.ptr.begin(), t.ptr.end() - 1);
        for(int32_t i = 0; i < a.m; ++i)
            for(int32_t k = a.ptr[i]; k < a.ptr[i + 1]; ++k)
            {
                const int32_t p = next[a.ind[k]]++;
                t.ind[p]        = i;
                t.val[p]        = a.val[k];
            }
        return t;
    }

    enum class mat_format
    {
        csr,
        csc,
        coo,
        coo_aos,
        ell,
        bsr_row,
        bsr_col,
        sell
    };

    // An n x n bidiagonal matrix (lower, or upper with `upper`) with distinct entries and
    // a dominant diagonal, in the formats of the tests, and vectors x and y.
    template <typename T>
    struct small_matrix
    {
        const int32_t          n;
        const host_csr<T>      a; // CSR and block dimension 1 BSR arrays, COO columns
        const host_csr<T>      at; // CSC arrays
        const host_coo<T>      coo;
        const host_ell<T>      ell;
        const host_sell<T>     sell;
        const std::vector<T>   x0;
        device_vector<int32_t> ptr, ind, cptr, cind, rows, aos, ell_col, sell_off, sell_col;
        device_vector<T>       val, cval, ell_val, sell_val, x, y;

        explicit small_matrix(int32_t size = 3, bool upper = false)
            : n(size)
            , a(make_triangular_pattern<T>(size, 1, !upper, upper, true))
            , at(transposed(a))
            , coo(to_coo(a))
            , ell(to_ell(a))
            , sell(to_sell(a, size))
            , x0(values<T>(size, 100))
            , ptr(a.ptr)
            , ind(a.ind)
            , cptr(at.ptr)
            , cind(at.ind)
            , rows(coo.row)
            , aos(interleaved(coo))
            , ell_col(ell.col)
            , sell_off(sell.offsets)
            , sell_col(sell.col)
            , val(a.val)
            , cval(at.val)
            , ell_val(ell.val)
            , sell_val(sell.val)
            , x(x0)
            , y(values<T>(size, 200))
        {
        }

        bool ok() const
        {
            return ptr.ptr && ind.ptr && cptr.ptr && cind.ptr && rows.ptr && aos.ptr && ell_col.ptr
                   && sell_off.ptr && sell_col.ptr && val.ptr && cval.ptr && ell_val.ptr
                   && sell_val.ptr && x.ptr && y.ptr;
        }

        rocsparse_status create(mat_format format, spmat_guard& A) const
        {
            const auto    it  = rocsparse_indextype_i32;
            const auto    b   = rocsparse_index_base_zero;
            const auto    dt  = rocsparse_ut::dt_of<T>();
            const int32_t nnz = a.nnz();
            switch(format)
            {
            case mat_format::csr:
                return rocsparse_create_csr_descr(&A.d, n, n, nnz, ptr, ind, val, it, it, b, dt);
            case mat_format::csc:
                return rocsparse_create_csc_descr(&A.d, n, n, nnz, cptr, cind, cval, it, it, b, dt);
            case mat_format::coo:
                return rocsparse_create_coo_descr(&A.d, n, n, nnz, rows, ind, val, it, b, dt);
            case mat_format::coo_aos:
                return rocsparse_create_coo_aos_descr(&A.d, n, n, nnz, aos, val, it, b, dt);
            case mat_format::ell:
                return rocsparse_create_ell_descr(
                    &A.d, n, n, ell_col, ell_val, ell.width, it, b, dt);
            case mat_format::bsr_row:
            case mat_format::bsr_col:
                return rocsparse_create_bsr_descr(&A.d,
                                                  n,
                                                  n,
                                                  nnz,
                                                  format == mat_format::bsr_row
                                                      ? rocsparse_direction_row
                                                      : rocsparse_direction_column,
                                                  1,
                                                  ptr,
                                                  ind,
                                                  val,
                                                  it,
                                                  it,
                                                  b,
                                                  dt);
            case mat_format::sell:
                // one slice of n rows
                return rocsparse_create_sell_descr(
                    &A.d, n, n, nnz, n, sell.colval, sell_off, sell_col, sell_val, it, it, b, dt);
            }
            return rocsparse_status_invalid_value;
        }

        rocsparse_status vectors(dnvec_guard& vx, dnvec_guard& vy) const
        {
            const rocsparse_status st
                = rocsparse_create_dnvec_descr(&vx.d, n, x, rocsparse_ut::dt_of<T>());
            return st != rocsparse_status_success
                       ? st
                       : rocsparse_create_dnvec_descr(&vy.d, n, y, rocsparse_ut::dt_of<T>());
        }

    private:
        static std::vector<int32_t> interleaved(const host_coo<T>& c)
        {
            std::vector<int32_t> v;
            for(size_t k = 0; k < c.row.size(); ++k)
            {
                v.push_back(c.row[k]);
                v.push_back(c.col[k]);
            }
            return v;
        }
    };

    rocsparse_status
        set_triangular(rocsparse_spmat_descr A, rocsparse_fill_mode fill, rocsparse_diag_type diag)
    {
        const rocsparse_status st
            = rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &fill, sizeof(fill));
        return st != rocsparse_status_success
                   ? st
                   : rocsparse_spmat_set_attribute(
                       A, rocsparse_spmat_diag_type, &diag, sizeof(diag));
    }

    // A dense matrix in either order with distinct entries (zeros for a negative seed);
    // the storage has no padding.
    template <typename T>
    struct dense_matrix
    {
        const int64_t         rows, cols, ld;
        const rocsparse_order order;
        std::vector<T>        host;
        device_vector<T>      data;
        dnmat_guard           desc;

        dense_matrix(int64_t r, int64_t c, rocsparse_order o, int64_t seed)
            : rows(r)
            , cols(c)
            , ld(std::max<int64_t>(1, o == col ? r : c))
            , order(o)
            , host(seed < 0 ? std::vector<T>(storage(), T{}) : values<T>(storage(), seed))
            , data(host)
        {
            if(data.ptr)
                (void)rocsparse_create_dnmat_descr(
                    &desc.d, rows, cols, ld, data.ptr, rocsparse_ut::dt_of<T>(), order);
        }

        bool ok() const
        {
            return data.ptr && desc.d;
        }

        size_t at(int64_t i, int64_t j) const
        {
            return static_cast<size_t>(order == col ? i + j * ld : i * ld + j);
        }

        // op(this)(i, j)
        T op_at(rocsparse_operation op, int64_t i, int64_t j) const
        {
            if(op == op_n)
                return host[at(i, j)];
            return op == op_c ? conj_of(host[at(j, i)]) : host[at(j, i)];
        }

        // Copies the host values to the device again.
        bool upload() const
        {
            return hipMemcpy(data.ptr, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice)
                   == hipSuccess;
        }

    private:
        size_t storage() const
        {
            return static_cast<size_t>(std::max<int64_t>(1, ld * (order == col ? cols : rows)));
        }
    };

    struct solve_kind
    {
        rocsparse_fill_mode fill;
        rocsparse_diag_type diag;
        rocsparse_operation op;
        bool                diagonal; // D y = alpha x with the stored diagonal D
    };

    // alpha * op(A)^-1 x (alpha * D^-1 x for the diagonal solve).
    template <typename T>
    std::vector<T>
        solve_reference(const host_csr<T>& a, const solve_kind& s, T alpha, const std::vector<T>& x)
    {
        if(!s.diagonal)
            return host_triangular_solve(a, s.fill, s.diag, s.op, alpha, x);
        std::vector<T> y(x.size());
        for(int32_t i = 0; i < a.m; ++i)
            for(int32_t k = a.ptr[i]; k < a.ptr[i + 1]; ++k)
                if(a.ind[k] == i)
                    y[i] = alpha * x[i] / a.val[k];
        return y;
    }

    // Solves for every column of op(B) and stores column j of op(X) where column j of
    // op(B) is read (op_out == op_b) or into column j of `out` (op_out == none).
    template <typename T>
    std::vector<T> solve_columns(const host_csr<T>&     a,
                                 const solve_kind&      s,
                                 T                      alpha,
                                 const dense_matrix<T>& B,
                                 rocsparse_operation    op_b,
                                 const dense_matrix<T>& out,
                                 rocsparse_operation    op_out,
                                 int64_t                nrhs)
    {
        std::vector<T> want(out.host);
        std::vector<T> b(a.m);
        for(int64_t j = 0; j < nrhs; ++j)
        {
            for(int32_t i = 0; i < a.m; ++i)
                b[i] = B.op_at(op_b, i, j);
            const std::vector<T> y = solve_reference(a, s, alpha, b);
            for(int32_t i = 0; i < a.m; ++i)
                want[op_out == op_n ? out.at(i, j) : out.at(j, i)] = y[i];
        }
        return want;
    }

    template <typename V>
    rocsparse_status trs_input(rocsparse_handle       h,
                               rocsparse_sptrsv_descr d,
                               rocsparse_sptrsv_input input,
                               const V&               value)
    {
        return rocsparse_sptrsv_set_input(h, d, input, &value, sizeof(V), nullptr);
    }

    template <typename V>
    rocsparse_status trs_input(rocsparse_handle       h,
                               rocsparse_sptrsm_descr d,
                               rocsparse_sptrsm_input input,
                               const V&               value)
    {
        return rocsparse_sptrsm_set_input(h, d, input, &value, sizeof(V), nullptr);
    }

    // ----- SpMV --------------------------------------------------------------------

    struct spmv_call
    {
        rocsparse_spmv_alg  alg;
        rocsparse_operation op;
        bool                again       = false; // preprocess a second time
        bool                unsupported = false;
    };

    // The calls share one matrix: a later algorithm must not reuse analysis data of an
    // earlier one (nnz-split reads arrays the other analyses never fill).
    struct spmv_case
    {
        mat_format             format;
        std::vector<spmv_call> calls;

        bool conjugate() const
        {
            return std::any_of(
                calls.begin(), calls.end(), [](const spmv_call& k) { return k.op == op_c; });
        }
    };

    std::vector<spmv_case> spmv_cases()
    {
        const auto               def = rocsparse_spmv_alg_default;
        std::vector<spmv_case>   v;
        const rocsparse_spmv_alg csr_algs[] = {def,
                                               rocsparse_spmv_alg_csr_rowsplit,
                                               rocsparse_spmv_alg_csr_adaptive,
                                               rocsparse_spmv_alg_csr_lrb,
                                               rocsparse_spmv_alg_csr_nnzsplit};
        for(const auto alg : csr_algs)
            for(const auto op : all_ops)
                v.push_back({mat_format::csr, {{alg, op, alg == def && op == op_n}}});
        v.push_back({mat_format::csc, {{def, op_n, true}, {def, op_c}}});
        for(const auto alg : csr_algs)
            if(alg != def)
                v.push_back({mat_format::csc, {{alg, op_n}}});
        const auto coo    = rocsparse_spmv_alg_coo;
        const auto atomic = rocsparse_spmv_alg_coo_atomic;
        v.push_back(
            {mat_format::coo, {{def, op_n, true}, {coo, op_n}, {atomic, op_n}, {coo, op_t}}});
        v.push_back({mat_format::coo_aos, {{def, op_n, true}, {coo, op_n}, {atomic, op_n}}});
        v.push_back(
            {mat_format::ell,
             {{def, op_n, true}, {rocsparse_spmv_alg_ell, op_n}, {rocsparse_spmv_alg_ell, op_c}}});
        // A transposed BSR product is documented as not implemented in rocsparse_bsrmv.h,
        // rocsparse_spmv.h does not say.
        for(const auto dir : {mat_format::bsr_row, mat_format::bsr_col})
            v.push_back({dir, {{def, op_n, true}, {rocsparse_spmv_alg_bsr, op_t, false, true}}});
        return v;
    }

    // The v2 calls: the same table is used, `again` meaning a second analysis.
    std::vector<spmv_case> v2_spmv_cases()
    {
        const auto             def = rocsparse_spmv_alg_default;
        std::vector<spmv_case> v;
        // The transposed LRB product returns not_implemented, which rocsparse_v2_spmv.h
        // does not document.
        v.push_back({mat_format::csr,
                     {{rocsparse_spmv_alg_csr_nnzsplit, op_n},
                      {rocsparse_spmv_alg_csr_lrb, op_c, false, true},
                      {rocsparse_spmv_alg_csr_adaptive, op_n},
                      {rocsparse_spmv_alg_csr_adaptive, op_n}}});
        v.push_back({mat_format::csc, {{rocsparse_spmv_alg_csr_nnzsplit, op_n}, {def, op_t}}});
        v.push_back({mat_format::coo,
                     {{rocsparse_spmv_alg_coo_atomic, op_n}, {rocsparse_spmv_alg_coo, op_c}}});
        v.push_back({mat_format::coo_aos, {{rocsparse_spmv_alg_coo, op_t}}});
        v.push_back({mat_format::ell, {{rocsparse_spmv_alg_ell, op_c}}});
        v.push_back({mat_format::bsr_col, {{rocsparse_spmv_alg_bsr, op_n}}});
        v.push_back({mat_format::sell, {{rocsparse_spmv_alg_sell, op_n}, {def, op_t}}});
        return v;
    }

    template <typename T>
    void spmv_dispatch(rocsparse_handle handle, const spmv_case& c)
    {
        small_matrix<T> m;
        spmat_guard     A;
        dnvec_guard     x, y;
        ASSERT_TRUE(m.ok());
        ASSERT_EQ(m.create(c.format, A), rocsparse_status_success);
        ASSERT_EQ(m.vectors(x, y), rocsparse_status_success);
        scoped_pointer_mode host(handle, false);
        const T             alpha = alpha_of<T>();
        const T             beta  = beta_of<T>();
        for(const spmv_call& k : c.calls)
        {
            SCOPED_TRACE(::testing::Message() << "alg " << k.alg << " op " << k.op);
            const std::vector<T> y0  = to_host(m.y);
            const bool           ran = run_buffered(
                [&](rocsparse_spmv_stage stage, size_t* bytes, void* buffer) {
                    return rocsparse_spmv(handle,
                                          k.op,
                                          &alpha,
                                          A.d,
                                          x.d,
                                          &beta,
                                          y.d,
                                          rocsparse_ut::dt_of<T>(),
                                          k.alg,
                                          stage,
                                          bytes,
                                          buffer);
                },
                rocsparse_spmv_stage_buffer_size,
                rocsparse_spmv_stage_preprocess,
                rocsparse_spmv_stage_compute,
                k.again ? 2 : 1,
                k.unsupported);
            if(ran)
                expect_near(csr_gemv(m.a, k.op, alpha, m.x0, beta, y0), to_host(m.y), "spmv");
        }
    }

    class L2DeviceSpmvDispatch : public HandleTest, public ::testing::WithParamInterface<spmv_case>
    {
    };

    TEST_P(L2DeviceSpmvDispatch, matches_host)
    {
        if(GetParam().conjugate())
            spmv_dispatch<cfloat>(handle, GetParam());
        else
            spmv_dispatch<float>(handle, GetParam());
    }

    INSTANTIATE_TEST_SUITE_P(Cases, L2DeviceSpmvDispatch, ::testing::ValuesIn(spmv_cases()));

    template <typename T>
    void v2_spmv_dispatch(rocsparse_handle handle, const spmv_case& c)
    {
        small_matrix<T> m(c.format == mat_format::sell ? 2 : 3);
        spmat_guard     A;
        dnvec_guard     x, y;
        ASSERT_TRUE(m.ok());
        ASSERT_EQ(m.create(c.format, A), rocsparse_status_success);
        ASSERT_EQ(m.vectors(x, y), rocsparse_status_success);
        scoped_pointer_mode host(handle, false);
        const T             alpha = alpha_of<T>();
        const T             beta  = beta_of<T>();
        const auto          dt    = rocsparse_ut::dt_of<T>();
        for(const spmv_call& k : c.calls)
        {
            SCOPED_TRACE(::testing::Message() << "alg " << k.alg << " op " << k.op);
            spmv_descr_guard descr;
            ASSERT_TRUE(descr.d);
            ASSERT_EQ(set_basic_inputs(handle, descr.d, k.alg, k.op, dt, dt),
                      rocsparse_status_success);
            ASSERT_EQ(
                set_input(handle, descr.d, rocsparse_spmv_input_nnz_use_starting_block_ids, true),
                rocsparse_status_success);
            const std::vector<T> y0  = to_host(m.y);
            const bool           ran = run_stages(
                {rocsparse_v2_spmv_stage_analysis, rocsparse_v2_spmv_stage_compute},
                [&](rocsparse_v2_spmv_stage stage, size_t* bytes) {
                    return rocsparse_v2_spmv_buffer_size(
                        handle, descr.d, A.d, x.d, y.d, stage, bytes, nullptr);
                },
                [&](rocsparse_v2_spmv_stage stage, size_t bytes, void* buffer) {
                    return rocsparse_v2_spmv(handle,
                                             descr.d,
                                             &alpha,
                                             A.d,
                                             x.d,
                                             &beta,
                                             y.d,
                                             stage,
                                             bytes,
                                             buffer,
                                             nullptr);
                },
                k.unsupported);
            if(ran)
                expect_near(csr_gemv(m.a, k.op, alpha, m.x0, beta, y0), to_host(m.y), "v2 spmv");
        }
    }

    class L2DeviceV2SpmvDispatch : public HandleTest,
                                   public ::testing::WithParamInterface<spmv_case>
    {
    };

    TEST_P(L2DeviceV2SpmvDispatch, matches_host)
    {
        if(GetParam().conjugate())
            v2_spmv_dispatch<cfloat>(handle, GetParam());
        else
            v2_spmv_dispatch<float>(handle, GetParam());
    }

    INSTANTIATE_TEST_SUITE_P(Cases, L2DeviceV2SpmvDispatch, ::testing::ValuesIn(v2_spmv_cases()));

    class L2DeviceV2SpmvExtras : public HandleTest
    {
    };

    TEST_F(L2DeviceV2SpmvExtras, datatypes_and_batches)
    {
        spmv_descr_guard descr;
        ASSERT_TRUE(descr.d);
        auto set_types = [&](rocsparse_datatype scalar, rocsparse_datatype compute) {
            ASSERT_EQ(set_input(handle, descr.d, rocsparse_spmv_input_scalar_datatype, scalar),
                      rocsparse_status_success);
            ASSERT_EQ(set_input(handle, descr.d, rocsparse_spmv_input_compute_datatype, compute),
                      rocsparse_status_success);
        };
        auto vector = [](dnvec_guard& g, const auto& data, rocsparse_datatype type) {
            ASSERT_EQ(rocsparse_create_dnvec_descr(&g.d, data.n, data.ptr, type),
                      rocsparse_status_success);
        };

        // Enabling the extras requires extras to be set.
        EXPECT_NE(set_input(handle, descr.d, rocsparse_spmv_input_enable_extra, int32_t(1)),
                  rocsparse_status_success);

        // Integer extras.
        {
            set_types(rocsparse_datatype_i32_r, rocsparse_datatype_i32_r);
            device_vector<int32_t> gamma(std::vector<int32_t>{1, 1}),
                z0(std::vector<int32_t>(3, 1)), z1(std::vector<int32_t>(3, 2));
            dnvec_guard g, a, b;
            vector(g, gamma, rocsparse_datatype_i32_r);
            vector(a, z0, rocsparse_datatype_i32_r);
            vector(b, z1, rocsparse_datatype_i32_r);
            rocsparse_const_dnvec_descr zs[2] = {a.d, b.d};
            EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 2, g.d, zs, nullptr),
                      rocsparse_status_success);
            for(const int32_t enable : {1, 0})
                EXPECT_EQ(set_input(handle, descr.d, rocsparse_spmv_input_enable_extra, enable),
                          rocsparse_status_success);
            EXPECT_EQ(rocsparse_spmv_clear_extra(handle, descr.d, nullptr),
                      rocsparse_status_success);
        }

        // A datatype without extras.
        {
            set_types(rocsparse_datatype_f16_r, rocsparse_datatype_f16_r);
            device_vector<float> gamma(std::vector<float>(1, 1.0f)), z(std::vector<float>(3, 1.0f));
            dnvec_guard          g, a;
            vector(g, gamma, rocsparse_datatype_f16_r);
            vector(a, z, rocsparse_datatype_f16_r);
            rocsparse_const_dnvec_descr zs[1] = {a.d};
            EXPECT_NE(rocsparse_spmv_set_extra(handle, descr.d, 1, g.d, zs, nullptr),
                      rocsparse_status_success);
        }

        // A batch_count other than 1 on the matrix or a vector.
        small_matrix<float> m;
        spmat_guard         A;
        dnvec_guard         x, y;
        ASSERT_TRUE(m.ok());
        ASSERT_EQ(m.create(mat_format::csr, A), rocsparse_status_success);
        ASSERT_EQ(m.vectors(x, y), rocsparse_status_success);
        spmv_descr_guard plain;
        ASSERT_TRUE(plain.d);
        ASSERT_EQ(set_basic_inputs(handle, plain.d, rocsparse_spmv_alg_default, op_n, f32, f32),
                  rocsparse_status_success);
        const float alpha              = 2.0f;
        const float beta               = -0.5f;
        auto        expect_unsupported = [&](rocsparse_v2_spmv_stage stage) {
            size_t bytes = 0;
            EXPECT_NE(rocsparse_v2_spmv_buffer_size(
                          handle, plain.d, A.d, x.d, y.d, stage, &bytes, nullptr),
                      rocsparse_status_success);
            EXPECT_NE(
                rocsparse_v2_spmv(
                    handle, plain.d, &alpha, A.d, x.d, &beta, y.d, stage, 0, nullptr, nullptr),
                rocsparse_status_success);
        };
        ASSERT_EQ(rocsparse_csr_set_strided_batch(A.d, 2, m.n + 1, m.a.nnz()),
                  rocsparse_status_success);
        expect_unsupported(rocsparse_v2_spmv_stage_analysis);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(A.d, 1, 0, 0), rocsparse_status_success);
        ASSERT_EQ(rocsparse_dnvec_set_strided_batch(x.d, 2, m.n), rocsparse_status_success);
        expect_unsupported(rocsparse_v2_spmv_stage_analysis);
        ASSERT_EQ(rocsparse_dnvec_set_strided_batch(x.d, 1, m.n), rocsparse_status_success);
        ASSERT_EQ(rocsparse_dnvec_set_strided_batch(y.d, 2, m.n), rocsparse_status_success);
        expect_unsupported(rocsparse_v2_spmv_stage_compute);
    }

    // ----- SpMM ----------------------------------------------------------------------

    // C = alpha * A * op(B) + beta * C with a 2 x 2 COO matrix A. n selects the main or
    // remainder part of the segmented-atomic kernel, and the orders and the operation
    // of B select the clauses of its layout condition.
    struct coomm_case
    {
        int                 n;
        rocsparse_order     order_b;
        rocsparse_operation trans_b;
        rocsparse_order     order_c;
    };

    // Each clause of the layout condition with both parts of the kernel.
    std::vector<coomm_case> coomm_cases()
    {
        return {{2, row, op_t, col},
                {2, row, op_n, row},
                {5, col, op_n, col},
                {5, row, op_c, col},
                {6, col, op_c, col},
                {9, row, op_n, row},
                {12, row, op_n, row},
                {12, row, op_c, col},
                {13, col, op_c, col},
                {13, row, op_c, row},
                {14, col, op_n, col},
                {14, col, op_t, row},
                {15, col, op_n, col},
                {15, col, op_t, col}};
    }

    template <typename T>
    void coomm_dispatch(rocsparse_handle handle, const coomm_case& c)
    {
        const bool      t = (c.trans_b != op_n);
        small_matrix<T> m(2);
        spmat_guard     A;
        ASSERT_TRUE(m.ok());
        ASSERT_EQ(m.create(mat_format::coo, A), rocsparse_status_success);
        dense_matrix<T> B(t ? c.n : 2, t ? 2 : c.n, c.order_b, 300);
        dense_matrix<T> C(2, c.n, c.order_c, 400);
        ASSERT_TRUE(B.ok() && C.ok());
        scoped_pointer_mode host(handle, false);
        const T             alpha = alpha_of<T>();
        const T             beta  = beta_of<T>();
        const bool          ran   = run_buffered(
            [&](rocsparse_spmm_stage stage, size_t* bytes, void* buffer) {
                return rocsparse_spmm(handle,
                                      op_n,
                                      c.trans_b,
                                      &alpha,
                                      A.d,
                                      B.desc.d,
                                      &beta,
                                      C.desc.d,
                                      rocsparse_ut::dt_of<T>(),
                                      rocsparse_spmm_alg_coo_segmented_atomic,
                                      stage,
                                      bytes,
                                      buffer);
            },
            rocsparse_spmm_stage_buffer_size,
            rocsparse_spmm_stage_preprocess,
            rocsparse_spmm_stage_compute,
            1,
            false);
        if(!ran)
            return;
        std::vector<T> want(C.host);
        for(int32_t i = 0; i < 2; ++i)
            for(int64_t j = 0; j < c.n; ++j)
            {
                T sum{};
                for(int32_t k = m.a.ptr[i]; k < m.a.ptr[i + 1]; ++k)
                    sum = sum + m.a.val[k] * B.op_at(c.trans_b, m.a.ind[k], j);
                want[C.at(i, j)] = beta * want[C.at(i, j)] + alpha * sum;
            }
        expect_near(want, to_host(C.data), "spmm");
    }

    class L2DeviceCoommSegmentedAtomic : public HandleTest,
                                         public ::testing::WithParamInterface<coomm_case>
    {
    };

    TEST_P(L2DeviceCoommSegmentedAtomic, matches_host)
    {
        if(GetParam().trans_b == op_c)
            coomm_dispatch<cfloat>(handle, GetParam());
        else
            coomm_dispatch<float>(handle, GetParam());
    }

    INSTANTIATE_TEST_SUITE_P(Cases,
                             L2DeviceCoommSegmentedAtomic,
                             ::testing::ValuesIn(coomm_cases()));

    // ----- SpTRSV ----------------------------------------------------------------------

    struct sptrsv_options
    {
        rocsparse_operation       op          = op_n;
        rocsparse_datatype        scalar      = f32;
        rocsparse_datatype        compute     = f32;
        rocsparse_analysis_policy policy      = rocsparse_analysis_policy_reuse;
        const void*               alpha       = nullptr;
        bool                      diagonal    = false; // diagonal solve instead of triangular
        bool                      unsupported = false;
        rocsparse_singularity     singularity = rocsparse_singularity_none;
        int64_t                   singularity_position = -1;
        int64_t                   zero_pivot_position  = -1;
    };

    bool
        configure_sptrsv(rocsparse_handle handle, rocsparse_sptrsv_descr d, const sptrsv_options& o)
    {
        bool ok  = true;
        auto put = [&](rocsparse_sptrsv_input input, const auto& value) {
            ok = ok && trs_input(handle, d, input, value) == rocsparse_status_success;
        };
        put(rocsparse_sptrsv_input_alg, rocsparse_sptrsv_alg_default);
        put(rocsparse_sptrsv_input_operation, o.op);
        put(rocsparse_sptrsv_input_scalar_datatype, o.scalar);
        put(rocsparse_sptrsv_input_compute_datatype, o.compute);
        put(rocsparse_sptrsv_input_analysis_policy, o.policy);
        // the pointer itself is the input
        ok = ok
             && rocsparse_sptrsv_set_input(handle,
                                           d,
                                           rocsparse_sptrsv_input_scalar_alpha,
                                           o.alpha,
                                           sizeof(const void*),
                                           nullptr)
                    == rocsparse_status_success;
#if defined(ROCSPARSE_WITH_DIAGONAL_SOLVE)
        put(rocsparse_sptrsv_input_solve_mode,
            o.diagonal ? rocsparse_solve_mode_diagonal : rocsparse_solve_mode_triangular);
        if(o.diagonal)
            put(rocsparse_sptrsv_input_diagonal_modifier, rocsparse_diagonal_modifier_absolute);
#endif
        return ok;
    }

    // Returns true when the solve ran.
    bool run_sptrsv(rocsparse_handle      handle,
                    rocsparse_spmat_descr A,
                    rocsparse_dnvec_descr x,
                    rocsparse_dnvec_descr y,
                    const sptrsv_options& o)
    {
        sptrsv_descr_guard descr;
        EXPECT_NE(descr.d, nullptr);
        if(!descr.d
           || !(configure_sptrsv(handle, descr.d, o) || (ADD_FAILURE() << "configure", false)))
            return false;

        // The outputs are written through host pointers.
        int64_t               position    = -1;
        rocsparse_singularity singularity = rocsparse_singularity_none;
        auto                  query       = [&](rocsparse_sptrsv_output output, auto* value) {
            scoped_pointer_mode host(handle, false);
            return rocsparse_sptrsv_get_output(
                handle, descr.d, output, value, sizeof(*value), nullptr);
        };
        // Before the analysis: only the query path is exercised, the header does not
        // specify the result.
        (void)query(rocsparse_sptrsv_output_singularity_position, &position);
        (void)query(rocsparse_sptrsv_output_singularity, &singularity);

        const bool ran = run_stages(
            {rocsparse_sptrsv_stage_analysis, rocsparse_sptrsv_stage_compute},
            [&](rocsparse_sptrsv_stage stage, size_t* bytes) {
                return rocsparse_sptrsv_buffer_size(
                    handle, descr.d, A, x, y, stage, bytes, nullptr);
            },
            [&](rocsparse_sptrsv_stage stage, size_t bytes, void* buffer) {
                return rocsparse_sptrsv(handle, descr.d, A, x, y, stage, bytes, buffer, nullptr);
            },
            o.unsupported);
        if(!ran)
            return false;

        EXPECT_EQ(query(rocsparse_sptrsv_output_singularity_position, &position),
                  rocsparse_status_success);
        EXPECT_EQ(query(rocsparse_sptrsv_output_singularity, &singularity),
                  rocsparse_status_success);
        EXPECT_EQ(singularity, o.singularity);
        EXPECT_EQ(position, o.singularity_position);
        // The header documents only success, but for a singular matrix the zero pivot
        // query returns rocsparse_status_zero_pivot: rocsparse_sptrsv_get_output passes
        // the status of csrsv_zero_pivot through.
        position = -2;
        EXPECT_EQ(query(rocsparse_sptrsv_output_zero_pivot_position, &position),
                  o.zero_pivot_position < 0 ? rocsparse_status_success
                                            : rocsparse_status_zero_pivot);
        EXPECT_EQ(position, o.zero_pivot_position);
        return true;
    }

    struct sptrsv_case
    {
        mat_format                format;
        rocsparse_fill_mode       fill;
        rocsparse_diag_type       diag;
        rocsparse_analysis_policy policy;
        rocsparse_operation       op;
        bool                      diagonal_solve = false;
        bool                      unsupported    = false;
    };

    // The conjugate transpose, and a transposed ELL matrix, are documented as
    // unsupported.
    std::vector<sptrsv_case> sptrsv_cases()
    {
        using F = mat_format;
        std::vector<sptrsv_case> v{{F::coo, lo, nu, reuse, op_c, false, true},
                                   {F::coo, up, un, force, op_t},
                                   {F::csc, up, un, reuse, op_n}};
#if defined(ROCSPARSE_WITH_ELL_TRSV)
        v.push_back({F::ell, lo, nu, force, op_n});
        v.push_back({F::ell, lo, un, reuse, op_n});
        v.push_back({F::ell, up, nu, force, op_t, false, true});
#endif
#if defined(ROCSPARSE_WITH_DIAGONAL_SOLVE)
        for(const auto format : {F::csr, F::csc})
            v.push_back({format, lo, nu, force, op_n, true});
#endif
        return v;
    }

    class L2DeviceSptrsvDispatch : public HandleTest,
                                   public ::testing::WithParamInterface<sptrsv_case>
    {
    };

    TEST_P(L2DeviceSptrsvDispatch, matches_host)
    {
        const sptrsv_case&  c = GetParam();
        small_matrix<float> m(3, c.fill == up);
        spmat_guard         A;
        dnvec_guard         x, y;
        ASSERT_TRUE(m.ok());
        ASSERT_EQ(m.create(c.format, A), rocsparse_status_success);
        ASSERT_EQ(set_triangular(A.d, c.fill, c.diag), rocsparse_status_success);
        ASSERT_EQ(m.vectors(x, y), rocsparse_status_success);
        scoped_pointer_mode host(handle, false);
        const float         alpha = 2.0f;
        sptrsv_options      o;
        o.op          = c.op;
        o.policy      = c.policy;
        o.alpha       = &alpha;
        o.diagonal    = c.diagonal_solve;
        o.unsupported = c.unsupported;
        if(run_sptrsv(handle, A.d, x.d, y.d, o))
            expect_near(solve_reference(m.a, {c.fill, c.diag, c.op, c.diagonal_solve}, alpha, m.x0),
                        to_host(m.y),
                        "sptrsv");
    }

    INSTANTIATE_TEST_SUITE_P(Cases, L2DeviceSptrsvDispatch, ::testing::ValuesIn(sptrsv_cases()));

    class L2DeviceSptrsvMisc : public HandleTest
    {
    };

    TEST_F(L2DeviceSptrsvMisc, handle_based_descriptor_create_and_destroy)
    {
        rocsparse_sptrsv_descr d = nullptr;
        EXPECT_EQ(rocsparse_sptrsv_descr_create(nullptr, &d, nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(d, nullptr);
        EXPECT_EQ(rocsparse_sptrsv_descr_create(handle, nullptr, nullptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_sptrsv_descr_destroy(nullptr, nullptr, nullptr),
                  rocsparse_status_invalid_handle);
        ASSERT_EQ(rocsparse_sptrsv_descr_create(handle, &d, nullptr), rocsparse_status_success);
        ASSERT_NE(d, nullptr);
        EXPECT_EQ(rocsparse_sptrsv_descr_destroy(handle, d, nullptr), rocsparse_status_success);
    }

    TEST_F(L2DeviceSptrsvMisc, missing_diagonal_is_a_symbolic_singularity)
    {
        // Row 1 has no diagonal entry.
        device_vector<int32_t> ptr(std::vector<int32_t>{0, 1, 2}), ind(std::vector<int32_t>{0, 0});
        device_vector<float>   val(std::vector<float>(2, 1.0f)), xv(std::vector<float>(2, 1.0f)),
            yv(std::vector<float>(2, 0.0f));
        ASSERT_TRUE(ptr.ptr && ind.ptr && val.ptr && xv.ptr && yv.ptr);
        spmat_guard A;
        dnvec_guard x, y;
        ASSERT_EQ(rocsparse_create_csr_descr(&A.d,
                                             2,
                                             2,
                                             2,
                                             ptr,
                                             ind,
                                             val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             f32),
                  rocsparse_status_success);
        ASSERT_EQ(set_triangular(A.d, rocsparse_fill_mode_lower, rocsparse_diag_type_non_unit),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&x.d, 2, xv, f32), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&y.d, 2, yv, f32), rocsparse_status_success);
        const float    alpha = 1.0f;
        sptrsv_options o;
        o.alpha                = &alpha;
        o.singularity          = rocsparse_singularity_symbolic;
        o.singularity_position = 1;
        o.zero_pivot_position  = 1;
        EXPECT_TRUE(run_sptrsv(handle, A.d, x.d, y.d, o));
    }

    TEST_F(L2DeviceSptrsvMisc, scalar_type_differs_from_compute_type)
    {
        // The matrix and the vectors are f64, the scalar is f32.
        small_matrix<double> m;
        spmat_guard          A;
        dnvec_guard          x, y;
        ASSERT_TRUE(m.ok());
        ASSERT_EQ(m.create(mat_format::csr, A), rocsparse_status_success);
        ASSERT_EQ(set_triangular(A.d, lo, nu), rocsparse_status_success);
        ASSERT_EQ(m.vectors(x, y), rocsparse_status_success);
        const float           alpha_host = 2.0f;
        device_vector<double> alpha_device(std::vector<double>(1, 2.0));
        ASSERT_TRUE(alpha_device.ptr);

        sptrsv_options o;
        o.scalar  = f32;
        o.compute = f64;
        o.alpha   = &alpha_host;
        {
            scoped_pointer_mode host(handle, false);
            if(run_sptrsv(handle, A.d, x.d, y.d, o))
                expect_near(
                    solve_reference(m.a, {lo, nu, op_n, false}, 2.0, m.x0), to_host(m.y), "sptrsv");
        }
        scoped_pointer_mode device(handle, true);
        o.scalar = f64;
        o.op     = op_t;
        o.policy = rocsparse_analysis_policy_force;
        o.alpha  = alpha_device.ptr;
        if(run_sptrsv(handle, A.d, x.d, y.d, o))
            expect_near(solve_reference(m.a, {lo, nu, op_t, false}, 2.0, m.x0),
                        to_host(m.y),
                        "sptrsv device alpha");
    }

    // ----- SpTRSM ----------------------------------------------------------------------

    struct sptrsm_options
    {
        rocsparse_operation       op_a        = op_n;
        rocsparse_operation       op_x        = op_n;
        rocsparse_datatype        scalar      = f32;
        rocsparse_datatype        compute     = f32;
        rocsparse_analysis_policy policy      = rocsparse_analysis_policy_reuse;
        const void*               alpha       = nullptr;
        bool                      diagonal    = false; // diagonal solve instead of triangular
        bool                      unsupported = false;
    };

    bool
        configure_sptrsm(rocsparse_handle handle, rocsparse_sptrsm_descr d, const sptrsm_options& o)
    {
        bool ok  = true;
        auto put = [&](rocsparse_sptrsm_input input, const auto& value) {
            ok = ok && trs_input(handle, d, input, value) == rocsparse_status_success;
        };
        put(rocsparse_sptrsm_input_alg, rocsparse_sptrsm_alg_default);
        put(rocsparse_sptrsm_input_operation_A, o.op_a);
        put(rocsparse_sptrsm_input_operation_X, o.op_x);
        put(rocsparse_sptrsm_input_scalar_datatype, o.scalar);
        put(rocsparse_sptrsm_input_compute_datatype, o.compute);
        put(rocsparse_sptrsm_input_analysis_policy, o.policy);
        ok = ok
             && rocsparse_sptrsm_set_input(handle,
                                           d,
                                           rocsparse_sptrsm_input_scalar_alpha,
                                           o.alpha,
                                           sizeof(const void*),
                                           nullptr)
                    == rocsparse_status_success;
#if defined(ROCSPARSE_WITH_DIAGONAL_SOLVE)
        put(rocsparse_sptrsm_input_solve_mode,
            o.diagonal ? rocsparse_solve_mode_diagonal : rocsparse_solve_mode_triangular);
        if(o.diagonal)
            put(rocsparse_sptrsm_input_diagonal_modifier, rocsparse_diagonal_modifier_none);
#endif
        return ok;
    }

    // Returns true when the solve ran.
    bool run_sptrsm(rocsparse_handle      handle,
                    rocsparse_spmat_descr A,
                    rocsparse_dnmat_descr X,
                    rocsparse_dnmat_descr Y,
                    const sptrsm_options& o)
    {
        sptrsm_descr_guard descr;
        EXPECT_NE(descr.d, nullptr);
        if(!descr.d
           || !(configure_sptrsm(handle, descr.d, o) || (ADD_FAILURE() << "configure", false)))
            return false;
        const bool ran = run_stages(
            {rocsparse_sptrsm_stage_analysis, rocsparse_sptrsm_stage_compute},
            [&](rocsparse_sptrsm_stage stage, size_t* bytes) {
                return rocsparse_sptrsm_buffer_size(
                    handle, descr.d, A, X, Y, stage, bytes, nullptr);
            },
            [&](rocsparse_sptrsm_stage stage, size_t bytes, void* buffer) {
                return rocsparse_sptrsm(handle, descr.d, A, X, Y, stage, bytes, buffer, nullptr);
            },
            o.unsupported);
        if(!ran)
            return false;
        // The output is written through a host pointer.
        scoped_pointer_mode host(handle, false);
        int64_t             position = 0;
        EXPECT_EQ(rocsparse_sptrsm_get_output(handle,
                                              descr.d,
                                              rocsparse_sptrsm_output_zero_pivot_position,
                                              &position,
                                              sizeof(position),
                                              nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(position, -1);
        return true;
    }

    // X is op(X) = m x nrhs stored as given, Y is m x nrhs.
    struct sptrsm_case
    {
        mat_format                format;
        rocsparse_operation       op_a;
        rocsparse_operation       op_x;
        rocsparse_order           order_x;
        rocsparse_order           order_y;
        int                       nrhs;
        rocsparse_analysis_policy policy;
        bool                      diagonal_solve = false;
        bool                      unsupported    = false;
    };

    // The conjugate transpose of A or X is documented as unsupported.
    std::vector<sptrsm_case> sptrsm_cases()
    {
        using F = mat_format;
        std::vector<sptrsm_case> v{{F::csr, op_c, op_n, col, col, 0, reuse, false, true},
                                   {F::coo, op_t, op_t, row, row, 2, reuse},
                                   {F::coo, op_t, op_n, row, col, 3, reuse},
                                   {F::coo, op_t, op_t, row, row, 3, force},
                                   {F::csc, op_n, op_t, col, col, 2, reuse},
                                   {F::csc, op_c, op_c, row, col, 2, force, false, true},
                                   {F::csc, op_t, op_n, row, col, 3, reuse}};
#if defined(ROCSPARSE_WITH_DIAGONAL_SOLVE)
        v.push_back({F::csr, op_n, op_n, col, col, 2, force, true});
#endif
        return v;
    }

    class L2DeviceSptrsmDispatch : public HandleTest,
                                   public ::testing::WithParamInterface<sptrsm_case>
    {
    };

    TEST_P(L2DeviceSptrsmDispatch, matches_host)
    {
        const sptrsm_case&  c = GetParam();
        small_matrix<float> m;
        spmat_guard         A;
        ASSERT_TRUE(m.ok());
        ASSERT_EQ(m.create(c.format, A), rocsparse_status_success);
        ASSERT_EQ(set_triangular(A.d, lo, nu), rocsparse_status_success);
        const bool          t = (c.op_x != op_n);
        dense_matrix<float> X(t ? c.nrhs : m.n, t ? m.n : c.nrhs, c.order_x, 300);
        dense_matrix<float> Y(m.n, c.nrhs, c.order_y, -1);
        ASSERT_TRUE(X.ok() && Y.ok());
        scoped_pointer_mode host(handle, false);
        const float         alpha = 2.0f;
        sptrsm_options      o;
        o.op_a        = c.op_a;
        o.op_x        = c.op_x;
        o.policy      = c.policy;
        o.alpha       = &alpha;
        o.diagonal    = c.diagonal_solve;
        o.unsupported = c.unsupported;
        if(run_sptrsm(handle, A.d, X.desc.d, Y.desc.d, o))
            expect_near(
                solve_columns(
                    m.a, {lo, nu, c.op_a, c.diagonal_solve}, alpha, X, c.op_x, Y, op_n, c.nrhs),
                to_host(Y.data),
                "sptrsm");
    }

    INSTANTIATE_TEST_SUITE_P(Cases, L2DeviceSptrsmDispatch, ::testing::ValuesIn(sptrsm_cases()));

    class L2DeviceSptrsmMisc : public HandleTest
    {
    };

    TEST_F(L2DeviceSptrsmMisc, scalar_type_differs_from_compute_type)
    {
        // The matrix and the dense matrices are f64, the scalar is f32.
        small_matrix<double> m;
        spmat_guard          A;
        ASSERT_TRUE(m.ok());
        ASSERT_EQ(m.create(mat_format::csr, A), rocsparse_status_success);
        ASSERT_EQ(set_triangular(A.d, lo, un), rocsparse_status_success);
        dense_matrix<double> X(m.n, 2, rocsparse_order_column, 300);
        dense_matrix<double> Y(m.n, 2, rocsparse_order_row, -1);
        ASSERT_TRUE(X.ok() && Y.ok());
        const float          alpha_host = 2.0f;
        device_vector<float> alpha_device(std::vector<float>(1, 2.0f));
        ASSERT_TRUE(alpha_device.ptr);

        // The conjugate transpose is documented as unsupported.
        sptrsm_options o;
        o.scalar      = f32;
        o.compute     = f64;
        o.op_a        = op_c;
        o.alpha       = &alpha_host;
        o.unsupported = true;
        {
            scoped_pointer_mode host(handle, false);
            if(run_sptrsm(handle, A.d, X.desc.d, Y.desc.d, o))
                expect_near(solve_columns(m.a, {lo, un, op_c, false}, 2.0, X, op_n, Y, op_n, 2),
                            to_host(Y.data),
                            "sptrsm op_c");
        }
        ASSERT_TRUE(Y.upload());
        scoped_pointer_mode device(handle, true);
        o.op_a        = op_n;
        o.policy      = rocsparse_analysis_policy_force;
        o.alpha       = alpha_device.ptr;
        o.unsupported = false;
        if(run_sptrsm(handle, A.d, X.desc.d, Y.desc.d, o))
            expect_near(solve_columns(m.a, {lo, un, op_n, false}, 2.0, X, op_n, Y, op_n, 2),
                        to_host(Y.data),
                        "sptrsm");
    }

#if defined(ROCSPARSE_WITH_TRSM_REFACTORING)
    // Batched solves are supported, batch counts of A and X that differ from the one of
    // Y and a batch of Y with a zero stride are rejected.
    TEST_F(L2DeviceSptrsmMisc, mismatched_batches_are_rejected)
    {
        small_matrix<float> m;
        spmat_guard         A;
        ASSERT_TRUE(m.ok());
        ASSERT_EQ(m.create(mat_format::csr, A), rocsparse_status_success);
        dense_matrix<float> X(m.n, 1, rocsparse_order_column, 300);
        dense_matrix<float> Y(m.n, 1, rocsparse_order_column, -1);
        ASSERT_TRUE(X.ok() && Y.ok());
        sptrsm_descr_guard descr;
        const float        alpha = 2.0f;
        sptrsm_options     o;
        o.alpha = &alpha;
        ASSERT_TRUE(descr.d);
        ASSERT_TRUE(configure_sptrsm(handle, descr.d, o));
        // The batch checks are in rocsparse_sptrsm, not in the buffer size query.
        auto expect_rejected = [&](rocsparse_sptrsm_stage stage) {
            EXPECT_NE(rocsparse_sptrsm(
                          handle, descr.d, A.d, X.desc.d, Y.desc.d, stage, 0, nullptr, nullptr),
                      rocsparse_status_success);
        };
        ASSERT_EQ(rocsparse_csr_set_strided_batch(A.d, 2, m.n + 1, m.a.nnz()),
                  rocsparse_status_success);
        expect_rejected(rocsparse_sptrsm_stage_analysis);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(A.d, 1, 0, 0), rocsparse_status_success);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(X.desc.d, 2, m.n), rocsparse_status_success);
        expect_rejected(rocsparse_sptrsm_stage_analysis);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(X.desc.d, 1, m.n), rocsparse_status_success);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(Y.desc.d, 2, 0), rocsparse_status_success);
        expect_rejected(rocsparse_sptrsm_stage_compute);
    }
#endif

    // ----- csrsm --------------------------------------------------------------------------

    struct csrsm_case
    {
        rocsparse_int             nrhs;
        rocsparse_operation       op_a, op_b;
        rocsparse_fill_mode       fill;
        rocsparse_diag_type       diag;
        rocsparse_analysis_policy policy;
    };

    // Each operation, fill mode, diagonal type and analysis policy; nrhs = 1 stays on
    // the csrsv path. The conjugate transpose is documented as not implemented; a
    // solve that succeeds anyway is checked.
    std::vector<csrsm_case> csrsm_cases()
    {
        std::vector<csrsm_case> v{{0, op_t, op_t, up, nu, force},
                                  {1, op_n, op_n, up, un, reuse},
                                  {1, op_n, op_c, up, nu, force},
                                  {65, op_t, op_t, lo, un, reuse}};
#if defined(ROCSPARSE_WITH_TRSM_REFACTORING)
        // The refactored solve starts with a block size of 512 and shifts it to the
        // smallest power of two that is >= nrhs and >= 64: nrhs = 65 above, 129, 257 and
        // 513 select the block sizes 128 to 1024.
        v.push_back({129, op_t, op_n, up, un, force});
        v.push_back({257, op_n, op_t, lo, nu, force});
        v.push_back({513, op_t, op_t, up, un, force});
#endif
        return v;
    }

    class L2DeviceCsrsmDispatch : public HandleTest,
                                  public ::testing::WithParamInterface<csrsm_case>
    {
    };

    TEST_P(L2DeviceCsrsmDispatch, matches_host)
    {
        const csrsm_case&     c           = GetParam();
        const bool            unsupported = (c.op_a == op_c || c.op_b == op_c);
        const rocsparse_int   m           = 4;
        const bool            t           = (c.op_b != op_n);
        const host_csr<float> a
            = make_triangular_pattern<float>(m, 1, c.fill == lo, c.fill == up, true);
        device_vector<rocsparse_int> ptr(convert_vector<rocsparse_int>(a.ptr)),
            ind(convert_vector<rocsparse_int>(a.ind));
        device_vector<float> val(a.val);
        dense_matrix<float>  B(t ? c.nrhs : m, t ? m : c.nrhs, col, 300);
        ASSERT_TRUE(ptr.ptr && ind.ptr && val.ptr && B.data.ptr);
        const rocsparse_int ldb = static_cast<rocsparse_int>(B.ld);
        mat_descr           descr;
        mat_info            info;
        ASSERT_TRUE(descr.d && info.d);
        ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, c.fill), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_diag_type(descr, c.diag), rocsparse_status_success);
        scoped_pointer_mode host(handle, false);
        const float         alpha  = 2.0f;
        const auto          policy = rocsparse_solve_policy_auto;
        const rocsparse_int nnz    = a.nnz();

        size_t bytes = 0;
        if(!proceed(rocsparse_scsrsm_buffer_size(handle,
                                                 c.op_a,
                                                 c.op_b,
                                                 m,
                                                 c.nrhs,
                                                 nnz,
                                                 &alpha,
                                                 descr,
                                                 val,
                                                 ptr,
                                                 ind,
                                                 B.data,
                                                 ldb,
                                                 info,
                                                 policy,
                                                 &bytes),
                    unsupported,
                    "buffer_size"))
            return;
        device_vector<char> buffer(std::max<size_t>(bytes, 1));
        ASSERT_TRUE(buffer.ptr);
        if(!proceed(rocsparse_scsrsm_analysis(handle,
                                              c.op_a,
                                              c.op_b,
                                              m,
                                              c.nrhs,
                                              nnz,
                                              &alpha,
                                              descr,
                                              val,
                                              ptr,
                                              ind,
                                              B.data,
                                              ldb,
                                              info,
                                              c.policy,
                                              policy,
                                              buffer),
                    unsupported,
                    "analysis"))
            return;
        if(!proceed(rocsparse_scsrsm_solve(handle,
                                           c.op_a,
                                           c.op_b,
                                           m,
                                           c.nrhs,
                                           nnz,
                                           &alpha,
                                           descr,
                                           val,
                                           ptr,
                                           ind,
                                           B.data,
                                           ldb,
                                           info,
                                           policy,
                                           buffer),
                    unsupported,
                    "solve"))
            return;
        rocsparse_int pivot = 0;
        ASSERT_EQ(rocsparse_csrsm_zero_pivot(handle, info, &pivot), rocsparse_status_success);
        EXPECT_EQ(pivot, -1);
        expect_near(
            solve_columns(a, {c.fill, c.diag, c.op_a, false}, alpha, B, c.op_b, B, c.op_b, c.nrhs),
            to_host(B.data),
            "csrsm");
        EXPECT_EQ(rocsparse_csrsm_clear(handle, info), rocsparse_status_success);
    }

    INSTANTIATE_TEST_SUITE_P(Cases, L2DeviceCsrsmDispatch, ::testing::ValuesIn(csrsm_cases()));

    class L2DeviceCsrsmQuickReturn : public HandleTest
    {
    };

    TEST_F(L2DeviceCsrsmQuickReturn, empty_system)
    {
        // m == 0 returns before the pointers are checked.
        mat_descr descr;
        mat_info  info;
        ASSERT_TRUE(descr.d && info.d);
        device_vector<float>         dummy(std::vector<float>(1, 0.0f));
        device_vector<rocsparse_int> index(std::vector<rocsparse_int>(1, 0));
        ASSERT_TRUE(dummy.ptr && index.ptr);
        const float alpha = 1.0f;
        size_t      bytes = 0;
        ASSERT_EQ(rocsparse_scsrsm_buffer_size(handle,
                                               op_n,
                                               op_n,
                                               0,
                                               0,
                                               0,
                                               &alpha,
                                               descr,
                                               dummy,
                                               index,
                                               index,
                                               dummy,
                                               1,
                                               info,
                                               rocsparse_solve_policy_auto,
                                               &bytes),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_scsrsm_solve(handle,
                                         op_n,
                                         op_t,
                                         0,
                                         1,
                                         0,
                                         &alpha,
                                         descr,
                                         dummy,
                                         index,
                                         index,
                                         dummy,
                                         1,
                                         info,
                                         rocsparse_solve_policy_auto,
                                         dummy.ptr),
                  rocsparse_status_success);
    }
} // namespace
