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
// Device unit tests that execute the host dispatch and numeric group launches in
//   library/src/extra/rocsparse_csrgemm_numeric_calc.cpp
//   library/src/extra/rocsparse_csrgeam_nnz_impl.cpp
//   library/src/extra/rocsparse_spgeam.cpp
//   library/src/extra/rocsparse_spgemm.cpp
//
// Public C API only, so the instrumented library (not a second copy of the
// .cpp) is what runs. Matrices stay small: one output row whose nnz selects
// the csrgemm numeric group, plus 2x2 products for the type / index / transpose
// switches. gfx1201 is wavefront 32; the wavefront-64 arms cannot run here.
//
#include "unit_test_utils.hpp"

#include <type_traits>
#include <vector>

using namespace rocsparse_ut;

namespace rocsparse_ut
{
    template <>
    inline rocsparse_datatype dt_of<_Float16>()
    {
        return rocsparse_datatype_f16_r;
    }
    template <>
    inline rocsparse_datatype dt_of<rocsparse_bfloat16>()
    {
        return rocsparse_datatype_bf16_r;
    }
    template <>
    inline _Float16 scalar<_Float16>(float v)
    {
        return static_cast<_Float16>(v);
    }
    template <>
    inline rocsparse_bfloat16 scalar<rocsparse_bfloat16>(float v)
    {
        return rocsparse_bfloat16(v);
    }
}

namespace
{
    struct Spmat
    {
        rocsparse_spmat_descr p = nullptr;
        Spmat()                 = default;
        Spmat(const Spmat&)     = delete;
        ~Spmat()
        {
            if(p)
                (void)rocsparse_destroy_spmat_descr(p);
        }
    };

    struct MatDescr
    {
        rocsparse_mat_descr d = nullptr;
        MatDescr()
        {
            (void)rocsparse_create_mat_descr(&d);
        }
        ~MatDescr()
        {
            if(d)
                (void)rocsparse_destroy_mat_descr(d);
        }
    };

    struct MatInfo
    {
        rocsparse_mat_info i = nullptr;
        MatInfo()
        {
            (void)rocsparse_create_mat_info(&i);
        }
        ~MatInfo()
        {
            if(i)
                (void)rocsparse_destroy_mat_info(i);
        }
    };

    struct PointerModeGuard
    {
        rocsparse_handle       handle = nullptr;
        rocsparse_pointer_mode prev   = rocsparse_pointer_mode_host;
        explicit PointerModeGuard(rocsparse_handle h, rocsparse_pointer_mode mode)
            : handle(h)
        {
            (void)rocsparse_get_pointer_mode(handle, &prev);
            (void)rocsparse_set_pointer_mode(handle, mode);
        }
        ~PointerModeGuard()
        {
            (void)rocsparse_set_pointer_mode(handle, prev);
        }
    };

    float as_float(const void* p, rocsparse_datatype dt)
    {
        switch(dt)
        {
        case rocsparse_datatype_f32_r:
            return *static_cast<const float*>(p);
        case rocsparse_datatype_f64_r:
            return static_cast<float>(*static_cast<const double*>(p));
        case rocsparse_datatype_f16_r:
            return static_cast<float>(*static_cast<const _Float16*>(p));
        case rocsparse_datatype_bf16_r:
            return static_cast<float>(*static_cast<const rocsparse_bfloat16*>(p));
        case rocsparse_datatype_f32_c:
            return std::real(*static_cast<const rocsparse_float_complex*>(p));
        case rocsparse_datatype_f64_c:
            return static_cast<float>(std::real(*static_cast<const rocsparse_double_complex*>(p)));
        default:
            return 0.f;
        }
    }

    // C = alpha * A * B [+ beta * D]. A is 1 x n (dense, or empty), B is n x n
    // (identity, or empty), so the single output row has exactly n non-zeros and
    // lands in one csrgemm numeric group.
    enum class WideKind
    {
        product,
        add_only,
        product_and_add
    };

    template <typename I, typename J, typename T>
    void run_wide(rocsparse_handle    handle,
                  int                 n,
                  WideKind            kind,
                  bool                device_scalars,
                  rocsparse_index_base base = rocsparse_index_base_zero)
    {
        SCOPED_TRACE("n=" + std::to_string(n) + " kind=" + std::to_string(static_cast<int>(kind))
                     + " device=" + std::to_string(device_scalars));

        const bool     mul = kind != WideKind::add_only;
        const bool     add = kind != WideKind::product;
        const int64_t  N   = n;
        const I        b0  = static_cast<I>(base);
        const J        jb0 = static_cast<J>(base);
        std::vector<I> a_ptr{b0, static_cast<I>(b0 + (mul ? n : 0))};
        std::vector<J> a_col;
        std::vector<T> a_val;
        if(mul)
        {
            a_col.resize(n);
            a_val.resize(n);
            for(int c = 0; c < n; ++c)
            {
                a_col[c] = static_cast<J>(jb0 + c);
                a_val[c] = scalar<T>(1.f);
            }
        }
        std::vector<I> b_ptr(static_cast<size_t>(n) + 1);
        std::vector<J> b_col;
        std::vector<T> b_val;
        if(mul)
        {
            b_col.resize(n);
            b_val.resize(n);
            for(int r = 0; r < n; ++r)
            {
                b_ptr[r] = static_cast<I>(b0 + r);
                b_col[r] = static_cast<J>(jb0 + r);
                b_val[r] = scalar<T>(1.f);
            }
            b_ptr[n] = static_cast<I>(b0 + n);
        }
        else
        {
            for(int r = 0; r <= n; ++r)
                b_ptr[r] = b0;
        }

        std::vector<I> d_ptr;
        std::vector<J> d_col;
        std::vector<T> d_val;
        if(add)
        {
            d_ptr = {b0, static_cast<I>(b0 + n)};
            d_col.resize(n);
            d_val.resize(n);
            for(int c = 0; c < n; ++c)
            {
                d_col[c] = static_cast<J>(jb0 + c);
                d_val[c] = scalar<T>(1.f);
            }
        }

        if(!add)
        {
            d_ptr = {b0, b0};
            d_col = {jb0};
            d_val = {scalar<T>(0.f)};
        }
        device_vector<I> dA_ptr{a_ptr};
        device_vector<I> dB_ptr{b_ptr};
        device_vector<J> dA_col{a_col.empty() ? std::vector<J>{J{}} : a_col};
        device_vector<J> dB_col{b_col.empty() ? std::vector<J>{J{}} : b_col};
        device_vector<T> dA_val{a_val.empty() ? std::vector<T>{scalar<T>(0.f)} : a_val};
        device_vector<T> dB_val{b_val.empty() ? std::vector<T>{scalar<T>(0.f)} : b_val};
        device_vector<I> dD_ptr{d_ptr};
        device_vector<J> dD_col{d_col};
        device_vector<T> dD_val{d_val};
        ASSERT_TRUE(dA_ptr.ptr && dB_ptr.ptr && dD_ptr.ptr);

        const rocsparse_datatype  dt = dt_of<T>();
        const rocsparse_indextype it = it_of<I>();
        const rocsparse_indextype jt = it_of<J>();

        Spmat A, B, D, C;
        ASSERT_EQ(rocsparse_create_csr_descr(&A.p,
                                             1,
                                             N,
                                             mul ? N : 0,
                                             dA_ptr.ptr,
                                             mul ? dA_col.ptr : nullptr,
                                             mul ? dA_val.ptr : nullptr,
                                             it,
                                             jt,
                                             base,
                                             dt),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csr_descr(&B.p,
                                             N,
                                             N,
                                             mul ? N : 0,
                                             dB_ptr.ptr,
                                             mul ? dB_col.ptr : nullptr,
                                             mul ? dB_val.ptr : nullptr,
                                             it,
                                             jt,
                                             base,
                                             dt),
                  rocsparse_status_success);
        if(add)
        {
            ASSERT_EQ(rocsparse_create_csr_descr(&D.p,
                                                 1,
                                                 N,
                                                 N,
                                                 dD_ptr.ptr,
                                                 dD_col.ptr,
                                                 dD_val.ptr,
                                                 it,
                                                 jt,
                                                 base,
                                                 dt),
                      rocsparse_status_success);
        }
        else
        {
            ASSERT_EQ(rocsparse_create_csr_descr(
                          &D.p, 0, 0, 0, nullptr, nullptr, nullptr, it, jt, base, dt),
                      rocsparse_status_success);
        }

        device_vector<I> dC_ptr{size_t{2}};
        ASSERT_TRUE(dC_ptr.ptr);
        ASSERT_EQ(rocsparse_create_csr_descr(
                      &C.p, 1, N, 0, dC_ptr.ptr, nullptr, nullptr, it, jt, base, dt),
                  rocsparse_status_success);

        const T alpha_h = scalar<T>(2.f);
        const T beta_h  = scalar<T>(add ? 3.f : 0.f);
        device_vector<T> d_alpha{std::vector<T>{alpha_h}};
        device_vector<T> d_beta{std::vector<T>{beta_h}};
        ASSERT_TRUE(d_alpha.ptr && d_beta.ptr);
        const void* alpha = device_scalars ? static_cast<const void*>(d_alpha.ptr) : &alpha_h;
        const void* beta  = device_scalars ? static_cast<const void*>(d_beta.ptr) : &beta_h;

        PointerModeGuard mode(handle,
                              device_scalars ? rocsparse_pointer_mode_device
                                             : rocsparse_pointer_mode_host);

        size_t bsz = 0;
        ASSERT_EQ(rocsparse_spgemm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   alpha,
                                   A.p,
                                   B.p,
                                   beta,
                                   D.p,
                                   C.p,
                                   dt,
                                   rocsparse_spgemm_alg_default,
                                   rocsparse_spgemm_stage_buffer_size,
                                   &bsz,
                                   nullptr),
                  rocsparse_status_success);
        device_vector<char> tmp{bsz ? bsz : size_t{1}};
        ASSERT_TRUE(tmp.ptr);
        ASSERT_EQ(rocsparse_spgemm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   alpha,
                                   A.p,
                                   B.p,
                                   beta,
                                   D.p,
                                   C.p,
                                   dt,
                                   rocsparse_spgemm_alg_default,
                                   rocsparse_spgemm_stage_nnz,
                                   &bsz,
                                   tmp.ptr),
                  rocsparse_status_success);

        int64_t rows = 0, cols = 0, nnz = 0;
        ASSERT_EQ(rocsparse_spmat_get_size(C.p, &rows, &cols, &nnz), rocsparse_status_success);
        ASSERT_EQ(nnz, N);

        device_vector<J> dC_col{static_cast<size_t>(nnz)};
        device_vector<T> dC_val{static_cast<size_t>(nnz)};
        ASSERT_TRUE(dC_col.ptr && dC_val.ptr);
        ASSERT_EQ(rocsparse_csr_set_pointers(C.p, dC_ptr.ptr, dC_col.ptr, dC_val.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgemm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   alpha,
                                   A.p,
                                   B.p,
                                   beta,
                                   D.p,
                                   C.p,
                                   dt,
                                   rocsparse_spgemm_alg_default,
                                   rocsparse_spgemm_stage_compute,
                                   &bsz,
                                   tmp.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const float want = (kind == WideKind::product) ? 2.f : (kind == WideKind::add_only) ? 3.f : 5.f;
        std::vector<char> raw(sizeof(T));
        ASSERT_EQ(hipMemcpy(raw.data(), dC_val.ptr, sizeof(T), hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_NEAR(as_float(raw.data(), dt), want, 1.e-2f);
        if(nnz > 1)
        {
            ASSERT_EQ(hipMemcpy(raw.data(),
                                reinterpret_cast<char*>(dC_val.ptr) + (nnz - 1) * sizeof(T),
                                sizeof(T),
                                hipMemcpyDeviceToHost),
                      hipSuccess);
            EXPECT_NEAR(as_float(raw.data(), dt), want, 1.e-2f);
        }
    }

    template <typename I, typename J, typename T>
    void run_identity_spgemm(rocsparse_handle     handle,
                             rocsparse_operation  trans_A,
                             rocsparse_operation  trans_B,
                             rocsparse_status     expect,
                             bool                 symbolic)
    {
        const rocsparse_datatype  dt   = dt_of<T>();
        const rocsparse_indextype it   = it_of<I>();
        const rocsparse_indextype jt   = it_of<J>();
        const rocsparse_index_base base = rocsparse_index_base_zero;
        std::vector<I> hptr{I{0}, I{1}, I{2}};
        std::vector<J> hcol{J{0}, J{1}};
        std::vector<T> hval{scalar<T>(1.f), scalar<T>(1.f)};
        device_vector<I> Ap{hptr}, Bp{hptr}, Cp{size_t{3}};
        device_vector<J> Ac{hcol}, Bc{hcol};
        device_vector<T> Av{hval}, Bv{hval};
        ASSERT_TRUE(Ap.ptr && Bp.ptr && Cp.ptr && Ac.ptr && Bc.ptr && Av.ptr && Bv.ptr);

        Spmat A, B, D, C;
        ASSERT_EQ(rocsparse_create_csr_descr(&A.p, 2, 2, 2, Ap.ptr, Ac.ptr, Av.ptr, it, jt, base, dt),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csr_descr(&B.p, 2, 2, 2, Bp.ptr, Bc.ptr, Bv.ptr, it, jt, base, dt),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csr_descr(
                      &D.p, 0, 0, 0, nullptr, nullptr, nullptr, it, jt, base, dt),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csr_descr(
                      &C.p, 2, 2, 0, Cp.ptr, nullptr, nullptr, it, jt, base, dt),
                  rocsparse_status_success);

        const T alpha = scalar<T>(1.f);
        const T beta  = scalar<T>(0.f);
        size_t  bsz   = 0;
        rocsparse_status st = rocsparse_spgemm(handle,
                                               trans_A,
                                               trans_B,
                                               &alpha,
                                               A.p,
                                               B.p,
                                               &beta,
                                               D.p,
                                               C.p,
                                               dt,
                                               rocsparse_spgemm_alg_default,
                                               rocsparse_spgemm_stage_buffer_size,
                                               &bsz,
                                               nullptr);
        if(st == rocsparse_status_success && expect == rocsparse_status_success)
        {
            device_vector<char> tmp{bsz ? bsz : size_t{1}};
            ASSERT_TRUE(tmp.ptr);
            ASSERT_EQ(rocsparse_spgemm(handle,
                                       trans_A,
                                       trans_B,
                                       &alpha,
                                       A.p,
                                       B.p,
                                       &beta,
                                       D.p,
                                       C.p,
                                       dt,
                                       rocsparse_spgemm_alg_default,
                                       rocsparse_spgemm_stage_nnz,
                                       &bsz,
                                       tmp.ptr),
                      rocsparse_status_success);
            int64_t rows = 0, cols = 0, nnz = 0;
            ASSERT_EQ(rocsparse_spmat_get_size(C.p, &rows, &cols, &nnz), rocsparse_status_success);
            EXPECT_EQ(nnz, 2);
            device_vector<J> Cc{static_cast<size_t>(nnz > 0 ? nnz : 1)};
            device_vector<T> Cv{static_cast<size_t>(nnz > 0 ? nnz : 1)};
            ASSERT_TRUE(Cc.ptr && Cv.ptr);
            ASSERT_EQ(rocsparse_csr_set_pointers(C.p, Cp.ptr, Cc.ptr, Cv.ptr),
                      rocsparse_status_success);
            const rocsparse_spgemm_stage stage
                = symbolic ? rocsparse_spgemm_stage_symbolic : rocsparse_spgemm_stage_compute;
            ASSERT_EQ(rocsparse_spgemm(handle,
                                       trans_A,
                                       trans_B,
                                       &alpha,
                                       A.p,
                                       B.p,
                                       &beta,
                                       D.p,
                                       C.p,
                                       dt,
                                       rocsparse_spgemm_alg_default,
                                       stage,
                                       &bsz,
                                       tmp.ptr),
                      rocsparse_status_success);
            if(symbolic)
            {
                ASSERT_EQ(rocsparse_spgemm(handle,
                                           trans_A,
                                           trans_B,
                                           &alpha,
                                           A.p,
                                           B.p,
                                           &beta,
                                           D.p,
                                           C.p,
                                           dt,
                                           rocsparse_spgemm_alg_default,
                                           rocsparse_spgemm_stage_numeric,
                                           &bsz,
                                           tmp.ptr),
                          rocsparse_status_success);
            }
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        }
        else if(expect != rocsparse_status_success)
        {
            // Buffer size and nnz take the template path and accept transpose.
            // The not-implemented check runs in the numeric kernel on compute,
            // after C's column and value arrays are installed.
            device_vector<char> tmp{bsz ? bsz : size_t{1}};
            ASSERT_TRUE(tmp.ptr);
            if(st == rocsparse_status_success)
            {
                st = rocsparse_spgemm(handle,
                                      trans_A,
                                      trans_B,
                                      &alpha,
                                      A.p,
                                      B.p,
                                      &beta,
                                      D.p,
                                      C.p,
                                      dt,
                                      rocsparse_spgemm_alg_default,
                                      rocsparse_spgemm_stage_nnz,
                                      &bsz,
                                      tmp.ptr);
            }
            if(st == rocsparse_status_success)
            {
                int64_t rows = 0, cols = 0, nnz = 0;
                ASSERT_EQ(rocsparse_spmat_get_size(C.p, &rows, &cols, &nnz),
                          rocsparse_status_success);
                device_vector<J> Cc{static_cast<size_t>(nnz > 0 ? nnz : 1)};
                device_vector<T> Cv{static_cast<size_t>(nnz > 0 ? nnz : 1)};
                ASSERT_TRUE(Cc.ptr && Cv.ptr);
                ASSERT_EQ(rocsparse_csr_set_pointers(C.p, Cp.ptr, Cc.ptr, Cv.ptr),
                          rocsparse_status_success);
                st = rocsparse_spgemm(handle,
                                      trans_A,
                                      trans_B,
                                      &alpha,
                                      A.p,
                                      B.p,
                                      &beta,
                                      D.p,
                                      C.p,
                                      dt,
                                      rocsparse_spgemm_alg_default,
                                      rocsparse_spgemm_stage_compute,
                                      &bsz,
                                      tmp.ptr);
            }
            EXPECT_EQ(st, expect);
        }
        else
        {
            EXPECT_EQ(st, expect);
        }
    }

    // Group 10 (shared memory does not fit the hash on gfx1201) rejects unsorted
    // B or D from csrgemm_numeric_calc. Nnz and symbolic stay on the sorted path;
    // storage is flipped only before numeric.
    void run_unsorted_group10(rocsparse_handle handle, bool unsorted_B, bool unsorted_D)
    {
        SCOPED_TRACE(std::string("unsorted B=") + (unsorted_B ? "1" : "0")
                     + " D=" + (unsorted_D ? "1" : "0"));
        const int n = 5000;
        const int m = 1;
        std::vector<rocsparse_int> a_ptr{0, n};
        std::vector<rocsparse_int> a_col(n);
        std::vector<float>         a_val(n, 1.f);
        std::vector<rocsparse_int> b_ptr(static_cast<size_t>(n) + 1);
        std::vector<rocsparse_int> b_col(n);
        std::vector<float>         b_val(n, 1.f);
        std::vector<rocsparse_int> d_ptr{0, n};
        std::vector<rocsparse_int> d_col(n);
        std::vector<float>         d_val(n, 1.f);
        for(int i = 0; i < n; ++i)
        {
            a_col[i] = i;
            b_ptr[i] = i;
            b_col[i] = i;
            d_col[i] = i;
        }
        b_ptr[n] = n;

        device_vector<rocsparse_int> dA_ptr{a_ptr}, dA_col{a_col}, dB_ptr{b_ptr}, dB_col{b_col};
        device_vector<rocsparse_int> dD_ptr{d_ptr}, dD_col{d_col}, dC_ptr{size_t{2}};
        device_vector<float>         dA_val{a_val}, dB_val{b_val}, dD_val{d_val};
        ASSERT_TRUE(dA_ptr.ptr && dB_ptr.ptr && dD_ptr.ptr && dC_ptr.ptr);

        MatDescr descr_A, descr_B, descr_D, descr_C;
        MatInfo  info;
        ASSERT_TRUE(descr_A.d && descr_B.d && descr_D.d && descr_C.d && info.i);
        const float alpha = 2.f, beta = 3.f;
        const bool  add = unsorted_D;
        size_t      bsz = 0;
        ASSERT_EQ(rocsparse_scsrgemm_buffer_size(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 m,
                                                 n,
                                                 n,
                                                 &alpha,
                                                 descr_A.d,
                                                 n,
                                                 dA_ptr,
                                                 dA_col,
                                                 descr_B.d,
                                                 n,
                                                 dB_ptr,
                                                 dB_col,
                                                 add ? &beta : nullptr,
                                                 add ? descr_D.d : nullptr,
                                                 add ? n : 0,
                                                 add ? dD_ptr.ptr : nullptr,
                                                 add ? dD_col.ptr : nullptr,
                                                 info.i,
                                                 &bsz),
                  rocsparse_status_success);
        device_vector<char> tmp{bsz ? bsz : size_t{1}};
        ASSERT_TRUE(tmp.ptr);
        rocsparse_int nnz_C = 0;
        ASSERT_EQ(rocsparse_csrgemm_nnz(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       m,
                                       n,
                                       n,
                                       descr_A.d,
                                       n,
                                       dA_ptr,
                                       dA_col,
                                       descr_B.d,
                                       n,
                                       dB_ptr,
                                       dB_col,
                                       add ? descr_D.d : nullptr,
                                       add ? n : 0,
                                       add ? dD_ptr.ptr : nullptr,
                                       add ? dD_col.ptr : nullptr,
                                       descr_C.d,
                                       dC_ptr,
                                       &nnz_C,
                                       info.i,
                                       tmp.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(nnz_C, n);
        device_vector<rocsparse_int> dC_col{static_cast<size_t>(nnz_C)};
        device_vector<float>         dC_val{static_cast<size_t>(nnz_C)};
        ASSERT_TRUE(dC_col.ptr && dC_val.ptr);
        ASSERT_EQ(rocsparse_csrgemm_symbolic(handle,
                                            rocsparse_operation_none,
                                            rocsparse_operation_none,
                                            m,
                                            n,
                                            n,
                                            descr_A.d,
                                            n,
                                            dA_ptr,
                                            dA_col,
                                            descr_B.d,
                                            n,
                                            dB_ptr,
                                            dB_col,
                                            add ? descr_D.d : nullptr,
                                            add ? n : 0,
                                            add ? dD_ptr.ptr : nullptr,
                                            add ? dD_col.ptr : nullptr,
                                            descr_C.d,
                                            nnz_C,
                                            dC_ptr,
                                            dC_col,
                                            info.i,
                                            tmp.ptr),
                  rocsparse_status_success);
        if(unsorted_B)
            ASSERT_EQ(rocsparse_set_mat_storage_mode(descr_B.d, rocsparse_storage_mode_unsorted),
                      rocsparse_status_success);
        if(unsorted_D)
            ASSERT_EQ(rocsparse_set_mat_storage_mode(descr_D.d, rocsparse_storage_mode_unsorted),
                      rocsparse_status_success);
        EXPECT_EQ(rocsparse_scsrgemm_numeric(handle,
                                            rocsparse_operation_none,
                                            rocsparse_operation_none,
                                            m,
                                            n,
                                            n,
                                            &alpha,
                                            descr_A.d,
                                            n,
                                            dA_val,
                                            dA_ptr,
                                            dA_col,
                                            descr_B.d,
                                            n,
                                            dB_val,
                                            dB_ptr,
                                            dB_col,
                                            add ? &beta : nullptr,
                                            add ? descr_D.d : nullptr,
                                            add ? n : 0,
                                            add ? dD_val.ptr : nullptr,
                                            add ? dD_ptr.ptr : nullptr,
                                            add ? dD_col.ptr : nullptr,
                                            descr_C.d,
                                            nnz_C,
                                            dC_val,
                                            dC_ptr,
                                            dC_col,
                                            info.i,
                                            tmp.ptr),
                  rocsparse_status_requires_sorted_storage);
    }

    template <typename I, typename J, typename T>
    void run_spgeam(rocsparse_handle    handle,
                    rocsparse_operation op_A,
                    rocsparse_operation op_B,
                    bool                device_scalars,
                    bool                scalar_is_f64)
    {
        // A = [1, 2; 0, 3], B = [4, 0; 5, 6], C = A + B = [5, 2; 5, 9].
        const rocsparse_datatype  dt   = dt_of<T>();
        const rocsparse_indextype it   = it_of<I>();
        const rocsparse_indextype jt   = it_of<J>();
        const rocsparse_index_base base = rocsparse_index_base_zero;
        std::vector<I> a_ptr{I{0}, I{2}, I{3}};
        std::vector<J> a_col{J{0}, J{1}, J{1}};
        std::vector<T> a_val{scalar<T>(1.f), scalar<T>(2.f), scalar<T>(3.f)};
        std::vector<I> b_ptr{I{0}, I{1}, I{3}};
        std::vector<J> b_col{J{0}, J{0}, J{1}};
        std::vector<T> b_val{scalar<T>(4.f), scalar<T>(5.f), scalar<T>(6.f)};
        device_vector<I> dA_ptr{a_ptr}, dB_ptr{b_ptr};
        device_vector<J> dA_col{a_col}, dB_col{b_col};
        device_vector<T> dA_val{a_val}, dB_val{b_val};
        ASSERT_TRUE(dA_ptr.ptr && dB_ptr.ptr && dA_val.ptr && dB_val.ptr);

        Spmat A, B;
        ASSERT_EQ(rocsparse_create_csr_descr(
                      &A.p, 2, 2, 3, dA_ptr.ptr, dA_col.ptr, dA_val.ptr, it, jt, base, dt),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csr_descr(
                      &B.p, 2, 2, 3, dB_ptr.ptr, dB_col.ptr, dB_val.ptr, it, jt, base, dt),
                  rocsparse_status_success);

        const T      alpha_t = scalar<T>(1.f);
        const T      beta_t  = scalar<T>(1.f);
        const double alpha_d = 1.0, beta_d = 1.0;
        device_vector<T>      d_alpha_t{std::vector<T>{alpha_t}};
        device_vector<T>      d_beta_t{std::vector<T>{beta_t}};
        device_vector<double> d_alpha_d{std::vector<double>{alpha_d}};
        device_vector<double> d_beta_d{std::vector<double>{beta_d}};
        ASSERT_TRUE(d_alpha_t.ptr && d_beta_t.ptr && d_alpha_d.ptr && d_beta_d.ptr);

        const void* alpha_ptr = nullptr;
        const void* beta_ptr  = nullptr;
        if(scalar_is_f64)
        {
            alpha_ptr = device_scalars ? static_cast<const void*>(d_alpha_d.ptr) : &alpha_d;
            beta_ptr  = device_scalars ? static_cast<const void*>(d_beta_d.ptr) : &beta_d;
        }
        else
        {
            alpha_ptr = device_scalars ? static_cast<const void*>(d_alpha_t.ptr) : &alpha_t;
            beta_ptr  = device_scalars ? static_cast<const void*>(d_beta_t.ptr) : &beta_t;
        }

        rocsparse_spgeam_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_spgeam_descr(&descr), rocsparse_status_success);
        const rocsparse_spgeam_alg alg = rocsparse_spgeam_alg_default;
        const rocsparse_datatype scalar_dt
            = scalar_is_f64 ? rocsparse_datatype_f64_r : dt;
        ASSERT_EQ(rocsparse_spgeam_set_input(
                      handle, descr, rocsparse_spgeam_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                            descr,
                                            rocsparse_spgeam_input_operation_A,
                                            &op_A,
                                            sizeof(op_A),
                                            nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                            descr,
                                            rocsparse_spgeam_input_operation_B,
                                            &op_B,
                                            sizeof(op_B),
                                            nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                            descr,
                                            rocsparse_spgeam_input_scalar_datatype,
                                            &scalar_dt,
                                            sizeof(scalar_dt),
                                            nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                            descr,
                                            rocsparse_spgeam_input_compute_datatype,
                                            &dt,
                                            sizeof(dt),
                                            nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                            descr,
                                            rocsparse_spgeam_input_scalar_alpha,
                                            alpha_ptr,
                                            sizeof(void*),
                                            nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                            descr,
                                            rocsparse_spgeam_input_scalar_beta,
                                            beta_ptr,
                                            sizeof(void*),
                                            nullptr),
                  rocsparse_status_success);

        size_t bsz = 0;
        ASSERT_EQ(rocsparse_spgeam_buffer_size(
                      handle, descr, A.p, B.p, nullptr, rocsparse_spgeam_stage_analysis, &bsz, nullptr),
                  rocsparse_status_success);
        device_vector<char> tmp{bsz ? bsz : size_t{1}};
        ASSERT_TRUE(tmp.ptr);
        // mat_C == nullptr takes csrgeam_nnz_core_with_allocation.
        ASSERT_EQ(rocsparse_spgeam(handle,
                                  descr,
                                  A.p,
                                  B.p,
                                  nullptr,
                                  rocsparse_spgeam_stage_analysis,
                                  bsz,
                                  tmp.ptr,
                                  nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        int64_t nnz_C = 0;
        ASSERT_EQ(rocsparse_spgeam_get_output(
                      handle, descr, rocsparse_spgeam_output_nnz, &nnz_C, sizeof(nnz_C), nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(nnz_C, 4);

        device_vector<I> dC_ptr{size_t{3}};
        device_vector<J> dC_col{static_cast<size_t>(nnz_C)};
        device_vector<T> dC_val{static_cast<size_t>(nnz_C)};
        ASSERT_TRUE(dC_ptr.ptr && dC_col.ptr && dC_val.ptr);
        Spmat C;
        ASSERT_EQ(rocsparse_create_csr_descr(&C.p,
                                             2,
                                             2,
                                             nnz_C,
                                             dC_ptr.ptr,
                                             dC_col.ptr,
                                             dC_val.ptr,
                                             it,
                                             jt,
                                             base,
                                             dt),
                  rocsparse_status_success);

        PointerModeGuard mode(handle,
                              device_scalars ? rocsparse_pointer_mode_device
                                             : rocsparse_pointer_mode_host);
        bsz = 0;
        ASSERT_EQ(rocsparse_spgeam_buffer_size(
                      handle, descr, A.p, B.p, C.p, rocsparse_spgeam_stage_compute, &bsz, nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam(handle,
                                  descr,
                                  A.p,
                                  B.p,
                                  C.p,
                                  rocsparse_spgeam_stage_compute,
                                  bsz,
                                  tmp.ptr,
                                  nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(rocsparse_destroy_spgeam_descr(descr), rocsparse_status_success);

        // Scalar conversion still runs when the scalar type differs from the
        // compute type. The numeric kernel reads alpha in the scalar type, so
        // only the matching-type product has a stable expected value.
        if(std::is_same_v<T, float> && !scalar_is_f64 && op_A == rocsparse_operation_none
           && op_B == rocsparse_operation_none)
        {
            auto hc = to_host(dC_val);
            auto hi = to_host(dC_col);
            ASSERT_EQ(hc.size(), 4u);
            // Columns are sorted within each row.
            EXPECT_EQ(hi[0], J{0});
            EXPECT_EQ(hi[1], J{1});
            EXPECT_EQ(hi[2], J{0});
            EXPECT_EQ(hi[3], J{1});
            EXPECT_NEAR(as_float(&hc[0], dt), 5.f, 1.e-3f);
            EXPECT_NEAR(as_float(&hc[1], dt), 2.f, 1.e-3f);
            EXPECT_NEAR(as_float(&hc[2], dt), 5.f, 1.e-3f);
            EXPECT_NEAR(as_float(&hc[3], dt), 9.f, 1.e-3f);
        }
    }
}

class GemmGeamCov : public HandleTest
{
};

TEST_F(GemmGeamCov, csrgemm_numeric_groups)
{
    // Groups 0..6 and 10 for f32 (group 10 is the multipass path: the 8192-wide
    // hash does not fit in 64 KB). Group 7 fits only for 2-byte types.
    run_wide<int32_t, int32_t, float>(handle, 8, WideKind::product, false);
    run_wide<int32_t, int32_t, float>(handle, 8, WideKind::product, true);
    run_wide<int32_t, int32_t, float>(handle, 24, WideKind::product, false);
    run_wide<int32_t, int32_t, float>(handle, 24, WideKind::product, true);
    run_wide<int32_t, int32_t, float>(handle, 64, WideKind::product, false);
    run_wide<int32_t, int32_t, float>(handle, 64, WideKind::product, true);
    run_wide<int32_t, int32_t, float>(handle, 64, WideKind::product_and_add, false);
    run_wide<int32_t, int32_t, float>(handle, 300, WideKind::product, false);
    run_wide<int32_t, int32_t, float>(handle, 700, WideKind::product, false);
    run_wide<int32_t, int32_t, float>(handle, 1500, WideKind::product, false);
    run_wide<int32_t, int32_t, float>(handle, 3000, WideKind::product, false);
    run_wide<int32_t, int32_t, float>(handle, 3000, WideKind::product, true);
    run_wide<int32_t, int32_t, float>(handle, 3000, WideKind::add_only, false);
    run_wide<int32_t, int32_t, float>(handle, 5000, WideKind::product, false);
    run_wide<int32_t, int32_t, float>(handle, 5000, WideKind::product, true);
    run_wide<int32_t, int32_t, float>(handle, 5000, WideKind::product_and_add, false);
    run_wide<int32_t, int32_t, float>(handle, 5000, WideKind::add_only, false);
    run_wide<int32_t, int32_t, _Float16>(handle, 5000, WideKind::product, false);
    run_wide<int32_t, int32_t, rocsparse_bfloat16>(handle, 5000, WideKind::product, false);
    run_wide<int64_t, int32_t, float>(handle, 64, WideKind::product, false);
    run_wide<int64_t, int64_t, float>(handle, 64, WideKind::product, false);
    run_wide<int32_t, int32_t, float>(
        handle, 8, WideKind::product, false, rocsparse_index_base_one);

    run_unsorted_group10(handle, true, false);
    run_unsorted_group10(handle, false, true);
}

TEST_F(GemmGeamCov, spgemm_type_index_transpose)
{
    const rocsparse_operation N = rocsparse_operation_none;
    const rocsparse_operation T = rocsparse_operation_transpose;
    const rocsparse_operation C = rocsparse_operation_conjugate_transpose;

    run_identity_spgemm<int32_t, int32_t, float>(handle, N, N, rocsparse_status_success, false);
    run_identity_spgemm<int32_t, int32_t, float>(handle, N, N, rocsparse_status_success, true);
    run_identity_spgemm<int32_t, int32_t, double>(handle, N, N, rocsparse_status_success, false);
    run_identity_spgemm<int32_t, int32_t, rocsparse_float_complex>(
        handle, N, N, rocsparse_status_success, false);
    run_identity_spgemm<int32_t, int32_t, rocsparse_double_complex>(
        handle, N, N, rocsparse_status_success, false);
    run_identity_spgemm<int32_t, int32_t, _Float16>(handle, N, N, rocsparse_status_success, false);
    run_identity_spgemm<int32_t, int32_t, rocsparse_bfloat16>(
        handle, N, N, rocsparse_status_success, false);

    run_identity_spgemm<int64_t, int32_t, float>(handle, N, N, rocsparse_status_success, false);
    run_identity_spgemm<int64_t, int32_t, double>(handle, N, N, rocsparse_status_success, false);
    run_identity_spgemm<int64_t, int32_t, rocsparse_float_complex>(
        handle, N, N, rocsparse_status_success, false);
    run_identity_spgemm<int64_t, int32_t, rocsparse_double_complex>(
        handle, N, N, rocsparse_status_success, false);
    run_identity_spgemm<int64_t, int32_t, _Float16>(handle, N, N, rocsparse_status_success, true);
    run_identity_spgemm<int64_t, int32_t, rocsparse_bfloat16>(
        handle, N, N, rocsparse_status_success, false);

    run_identity_spgemm<int64_t, int64_t, float>(handle, N, N, rocsparse_status_success, false);
    run_identity_spgemm<int64_t, int64_t, double>(handle, N, N, rocsparse_status_success, false);
    run_identity_spgemm<int64_t, int64_t, rocsparse_float_complex>(
        handle, N, N, rocsparse_status_success, false);
    run_identity_spgemm<int64_t, int64_t, rocsparse_double_complex>(
        handle, N, N, rocsparse_status_success, false);
    run_identity_spgemm<int64_t, int64_t, _Float16>(handle, N, N, rocsparse_status_success, false);
    run_identity_spgemm<int64_t, int64_t, rocsparse_bfloat16>(
        handle, N, N, rocsparse_status_success, true);

    // Generic spgemm forwards transpose into the numeric kernel, which ignores it.
    run_identity_spgemm<int32_t, int32_t, float>(handle, T, N, rocsparse_status_success, false);
    run_identity_spgemm<int32_t, int32_t, float>(handle, N, T, rocsparse_status_success, false);
    run_identity_spgemm<int32_t, int32_t, float>(handle, T, T, rocsparse_status_success, false);
    run_identity_spgemm<int32_t, int32_t, float>(handle, C, N, rocsparse_status_success, false);
    run_identity_spgemm<int32_t, int32_t, rocsparse_float_complex>(
        handle, C, C, rocsparse_status_success, false);
    run_identity_spgemm<int64_t, int64_t, double>(handle, T, C, rocsparse_status_success, false);

    // The legacy entry points reject transpose before any kernel runs.
    device_vector<rocsparse_int> rp{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> ci{std::vector<rocsparse_int>{0, 1}};
    device_vector<float>         va{std::vector<float>{1.f, 1.f}};
    device_vector<rocsparse_int> cprow{size_t{3}};
    device_vector<rocsparse_int> ccol{size_t{2}};
    device_vector<float>         cval{size_t{2}};
    device_vector<rocsparse_int> hnnz{std::vector<rocsparse_int>{0}};
    ASSERT_TRUE(rp.ptr && ci.ptr && va.ptr && cprow.ptr && ccol.ptr && cval.ptr && hnnz.ptr);
    MatDescr da, db, dc;
    MatInfo  info;
    ASSERT_TRUE(da.d && db.d && dc.d && info.i);
    const float alpha = 1.f;
    size_t      bsz   = 0;
    for(rocsparse_operation op : {T, C})
    {
        EXPECT_EQ(rocsparse_scsrgemm_buffer_size(handle,
                                                 op,
                                                 N,
                                                 2,
                                                 2,
                                                 2,
                                                 &alpha,
                                                 da.d,
                                                 2,
                                                 rp.ptr,
                                                 ci.ptr,
                                                 db.d,
                                                 2,
                                                 rp.ptr,
                                                 ci.ptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 info.i,
                                                 &bsz),
                  rocsparse_status_not_implemented);
        EXPECT_EQ(rocsparse_scsrgemm_buffer_size(handle,
                                                 N,
                                                 op,
                                                 2,
                                                 2,
                                                 2,
                                                 &alpha,
                                                 da.d,
                                                 2,
                                                 rp.ptr,
                                                 ci.ptr,
                                                 db.d,
                                                 2,
                                                 rp.ptr,
                                                 ci.ptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 info.i,
                                                 &bsz),
                  rocsparse_status_not_implemented);
    }
    ASSERT_EQ(rocsparse_scsrgemm_buffer_size(handle,
                                             N,
                                             N,
                                             2,
                                             2,
                                             2,
                                             &alpha,
                                             da.d,
                                             2,
                                             rp.ptr,
                                             ci.ptr,
                                             db.d,
                                             2,
                                             rp.ptr,
                                             ci.ptr,
                                             nullptr,
                                             nullptr,
                                             0,
                                             nullptr,
                                             nullptr,
                                             info.i,
                                             &bsz),
              rocsparse_status_success);
    device_vector<char> tmp{bsz ? bsz : size_t{1}};
    ASSERT_TRUE(tmp.ptr);
    EXPECT_EQ(rocsparse_csrgemm_nnz(handle,
                                   T,
                                   N,
                                   2,
                                   2,
                                   2,
                                   da.d,
                                   2,
                                   rp.ptr,
                                   ci.ptr,
                                   db.d,
                                   2,
                                   rp.ptr,
                                   ci.ptr,
                                   nullptr,
                                   0,
                                   nullptr,
                                   nullptr,
                                   dc.d,
                                   cprow.ptr,
                                   hnnz.ptr,
                                   info.i,
                                   tmp.ptr),
              rocsparse_status_not_implemented);
    EXPECT_EQ(rocsparse_scsrgemm(handle,
                                C,
                                N,
                                2,
                                2,
                                2,
                                &alpha,
                                da.d,
                                2,
                                va.ptr,
                                rp.ptr,
                                ci.ptr,
                                db.d,
                                2,
                                va.ptr,
                                rp.ptr,
                                ci.ptr,
                                nullptr,
                                nullptr,
                                0,
                                nullptr,
                                nullptr,
                                nullptr,
                                dc.d,
                                cval.ptr,
                                cprow.ptr,
                                ccol.ptr,
                                info.i,
                                tmp.ptr),
              rocsparse_status_not_implemented);
}

TEST_F(GemmGeamCov, spgemm_rejected_types_formats_and_bsr)
{
    device_vector<int32_t> ip{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> ic{std::vector<int32_t>{0, 1}};
    device_vector<float>   iv{std::vector<float>{1.f, 1.f}};
    device_vector<int32_t> cp{size_t{3}};
    ASSERT_TRUE(ip.ptr && ic.ptr && iv.ptr && cp.ptr);

    auto make_csr = [&](Spmat& m, rocsparse_indextype it, rocsparse_indextype jt, rocsparse_datatype dt, bool empty) {
        if(empty)
            return rocsparse_create_csr_descr(&m.p, 0, 0, 0, nullptr, nullptr, nullptr, it, jt, rocsparse_index_base_zero, dt);
        return rocsparse_create_csr_descr(&m.p, 2, 2, 2, ip.ptr, ic.ptr, iv.ptr, it, jt, rocsparse_index_base_zero, dt);
    };

    const rocsparse_datatype ints[] = {rocsparse_datatype_i8_r,
                                       rocsparse_datatype_u8_r,
                                       rocsparse_datatype_i32_r,
                                       rocsparse_datatype_u32_r};
    const rocsparse_indextype pairs[][2] = {{rocsparse_indextype_i32, rocsparse_indextype_i32},
                                            {rocsparse_indextype_i64, rocsparse_indextype_i32},
                                            {rocsparse_indextype_i64, rocsparse_indextype_i64}};
    for(auto dt : ints)
    {
        for(auto& pr : pairs)
        {
            Spmat A, B, D, C;
            ASSERT_EQ(make_csr(A, pr[0], pr[1], dt, false), rocsparse_status_success);
            ASSERT_EQ(make_csr(B, pr[0], pr[1], dt, false), rocsparse_status_success);
            ASSERT_EQ(make_csr(D, pr[0], pr[1], dt, true), rocsparse_status_success);
            ASSERT_EQ(make_csr(C, pr[0], pr[1], dt, false), rocsparse_status_success);
            // C was created with nnz 2; that is fine for a rejected call.
            int    one = 1;
            size_t bsz = 0;
            EXPECT_EQ(rocsparse_spgemm(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       &one,
                                       A.p,
                                       B.p,
                                       &one,
                                       D.p,
                                       C.p,
                                       dt,
                                       rocsparse_spgemm_alg_default,
                                       rocsparse_spgemm_stage_buffer_size,
                                       &bsz,
                                       nullptr),
                      rocsparse_status_not_implemented);
        }
    }

    // Value 1 is the removed u16 index type. The public enumerator is hidden
    // when ROCSPARSE_WITH_U16_REMOVED is set; the library still rejects it.
    const auto u16 = static_cast<rocsparse_indextype>(1);
    const rocsparse_indextype rejected[][2]
        = {{rocsparse_indextype_i32, rocsparse_indextype_i64},
           {u16, rocsparse_indextype_i32},
           {rocsparse_indextype_i32, u16},
           {rocsparse_indextype_i64, u16},
           {u16, u16}};
    for(auto& pr : rejected)
    {
        Spmat A, B, D, C;
        ASSERT_EQ(make_csr(A, pr[0], pr[1], rocsparse_datatype_f32_r, false), rocsparse_status_success);
        ASSERT_EQ(make_csr(B, pr[0], pr[1], rocsparse_datatype_f32_r, false), rocsparse_status_success);
        ASSERT_EQ(make_csr(D, pr[0], pr[1], rocsparse_datatype_f32_r, true), rocsparse_status_success);
        ASSERT_EQ(make_csr(C, pr[0], pr[1], rocsparse_datatype_f32_r, false), rocsparse_status_success);
        const float alpha = 1.f, beta = 0.f;
        size_t      bsz = 0;
        EXPECT_EQ(rocsparse_spgemm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   &alpha,
                                   A.p,
                                   B.p,
                                   &beta,
                                   D.p,
                                   C.p,
                                   rocsparse_datatype_f32_r,
                                   rocsparse_spgemm_alg_default,
                                   rocsparse_spgemm_stage_buffer_size,
                                   &bsz,
                                   nullptr),
                  rocsparse_status_not_implemented);
    }

    // Invalid algorithm / stage, batched matrix, compute-type mismatch.
    {
        Spmat A, B, D, C;
        ASSERT_EQ(make_csr(A, rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_f32_r, false),
                  rocsparse_status_success);
        ASSERT_EQ(make_csr(B, rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_f32_r, false),
                  rocsparse_status_success);
        ASSERT_EQ(make_csr(D, rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_f32_r, true),
                  rocsparse_status_success);
        ASSERT_EQ(make_csr(C, rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_f32_r, false),
                  rocsparse_status_success);
        const float alpha = 1.f, beta = 0.f;
        size_t      bsz = 0;
        EXPECT_EQ(rocsparse_spgemm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   &alpha,
                                   A.p,
                                   B.p,
                                   &beta,
                                   D.p,
                                   C.p,
                                   rocsparse_datatype_f32_r,
                                   static_cast<rocsparse_spgemm_alg>(99),
                                   rocsparse_spgemm_stage_buffer_size,
                                   &bsz,
                                   nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_spgemm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   &alpha,
                                   A.p,
                                   B.p,
                                   &beta,
                                   D.p,
                                   C.p,
                                   rocsparse_datatype_f32_r,
                                   rocsparse_spgemm_alg_default,
                                   static_cast<rocsparse_spgemm_stage>(99),
                                   &bsz,
                                   nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_spgemm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   &alpha,
                                   A.p,
                                   B.p,
                                   &beta,
                                   D.p,
                                   C.p,
                                   rocsparse_datatype_f64_r,
                                   rocsparse_spgemm_alg_default,
                                   rocsparse_spgemm_stage_buffer_size,
                                   &bsz,
                                   nullptr),
                  rocsparse_status_not_implemented);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(A.p, 2, 3, 2), rocsparse_status_success);
        EXPECT_EQ(rocsparse_spgemm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   &alpha,
                                   A.p,
                                   B.p,
                                   &beta,
                                   D.p,
                                   C.p,
                                   rocsparse_datatype_f32_r,
                                   rocsparse_spgemm_alg_default,
                                   rocsparse_spgemm_stage_buffer_size,
                                   &bsz,
                                   nullptr),
                  rocsparse_status_not_implemented);
    }

    const rocsparse_spgemm_stage stages[] = {rocsparse_spgemm_stage_buffer_size,
                                             rocsparse_spgemm_stage_nnz,
                                             rocsparse_spgemm_stage_compute,
                                             rocsparse_spgemm_stage_symbolic,
                                             rocsparse_spgemm_stage_numeric};
    auto reject_format = [&](auto&& build) {
        Spmat A, B, D, C;
        ASSERT_TRUE(build(A) && build(B) && build(D) && build(C));
        const float alpha = 1.f, beta = 0.f;
        size_t      bsz = 0;
        for(auto stage : stages)
        {
            EXPECT_EQ(rocsparse_spgemm(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       &alpha,
                                       A.p,
                                       B.p,
                                       &beta,
                                       D.p,
                                       C.p,
                                       rocsparse_datatype_f32_r,
                                       rocsparse_spgemm_alg_default,
                                       stage,
                                       &bsz,
                                       nullptr),
                      rocsparse_status_not_implemented);
        }
    };

    device_vector<int32_t> one_idx{std::vector<int32_t>{0, 0}};
    device_vector<int32_t> colptr{std::vector<int32_t>{0, 1}};
    device_vector<float>   one_val{std::vector<float>{1.f}};
    ASSERT_TRUE(one_idx.ptr && colptr.ptr && one_val.ptr);

    reject_format([&](Spmat& m) {
        return rocsparse_create_coo_descr(&m.p,
                                         1,
                                         1,
                                         1,
                                         one_idx.ptr,
                                         one_idx.ptr,
                                         one_val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r)
               == rocsparse_status_success;
    });
    reject_format([&](Spmat& m) {
        return rocsparse_create_coo_aos_descr(&m.p,
                                             1,
                                             1,
                                             1,
                                             one_idx.ptr,
                                             one_val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r)
               == rocsparse_status_success;
    });
    reject_format([&](Spmat& m) {
        return rocsparse_create_csc_descr(&m.p,
                                         1,
                                         1,
                                         1,
                                         colptr.ptr,
                                         one_idx.ptr,
                                         one_val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r)
               == rocsparse_status_success;
    });
    reject_format([&](Spmat& m) {
        return rocsparse_create_ell_descr(&m.p,
                                         1,
                                         1,
                                         one_idx.ptr,
                                         one_val.ptr,
                                         1,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r)
               == rocsparse_status_success;
    });
    reject_format([&](Spmat& m) {
        return rocsparse_create_bell_descr(&m.p,
                                          1,
                                          1,
                                          rocsparse_direction_row,
                                          1,
                                          1,
                                          one_idx.ptr,
                                          one_val.ptr,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r)
               == rocsparse_status_success;
    });
    device_vector<int32_t> sell_off{std::vector<int32_t>{0, 1}};
    ASSERT_TRUE(sell_off.ptr);
    reject_format([&](Spmat& m) {
        return rocsparse_create_sell_descr(&m.p,
                                          2,
                                          2,
                                          1,
                                          2,
                                          1,
                                          sell_off.ptr,
                                          one_idx.ptr,
                                          one_val.ptr,
                                          rocsparse_indextype_i32,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r)
               == rocsparse_status_success;
    });

    // BSR is implemented for buffer/nnz/compute and rejected for symbolic/numeric.
    for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
    {
        for(rocsparse_datatype dt : {rocsparse_datatype_f32_r, rocsparse_datatype_f16_r, rocsparse_datatype_bf16_r})
        {
            device_vector<int32_t> bp{std::vector<int32_t>{0, 1, 2}};
            device_vector<int32_t> bc{std::vector<int32_t>{0, 1}};
            device_vector<float>   bvf{std::vector<float>{1.f, 1.f}};
            device_vector<_Float16> bvh{std::vector<_Float16>{_Float16(1.f), _Float16(1.f)}};
            device_vector<rocsparse_bfloat16> bvb{
                std::vector<rocsparse_bfloat16>{rocsparse_bfloat16(1.f), rocsparse_bfloat16(1.f)}};
            void* vals = bvf.ptr;
            if(dt == rocsparse_datatype_f16_r)
                vals = bvh.ptr;
            if(dt == rocsparse_datatype_bf16_r)
                vals = bvb.ptr;
            device_vector<int32_t> cptr{size_t{3}};
            Spmat A, B, D, C;
            auto mk = [&](Spmat& m, bool empty) {
                if(empty)
                    return rocsparse_create_bsr_descr(&m.p,
                                                     0,
                                                     0,
                                                     0,
                                                     dir,
                                                     1,
                                                     nullptr,
                                                     nullptr,
                                                     nullptr,
                                                     rocsparse_indextype_i32,
                                                     rocsparse_indextype_i32,
                                                     rocsparse_index_base_zero,
                                                     dt);
                return rocsparse_create_bsr_descr(&m.p,
                                                 2,
                                                 2,
                                                 2,
                                                 dir,
                                                 1,
                                                 bp.ptr,
                                                 bc.ptr,
                                                 vals,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 dt);
            };
            ASSERT_EQ(mk(A, false), rocsparse_status_success);
            ASSERT_EQ(mk(B, false), rocsparse_status_success);
            ASSERT_EQ(mk(D, true), rocsparse_status_success);
            ASSERT_EQ(rocsparse_create_bsr_descr(&C.p,
                                                2,
                                                2,
                                                0,
                                                dir,
                                                1,
                                                cptr.ptr,
                                                nullptr,
                                                nullptr,
                                                rocsparse_indextype_i32,
                                                rocsparse_indextype_i32,
                                                rocsparse_index_base_zero,
                                                dt),
                      rocsparse_status_success);
            float  alpha_f = 1.f, beta_f = 0.f;
            _Float16 alpha_h = _Float16(1.f), beta_h = _Float16(0.f);
            rocsparse_bfloat16 alpha_b(1.f), beta_b(0.f);
            const void* alpha = &alpha_f;
            const void* beta  = &beta_f;
            if(dt == rocsparse_datatype_f16_r)
            {
                alpha = &alpha_h;
                beta  = &beta_h;
            }
            if(dt == rocsparse_datatype_bf16_r)
            {
                alpha = &alpha_b;
                beta  = &beta_b;
            }
            size_t bsz = 0;
            ASSERT_EQ(rocsparse_spgemm(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       alpha,
                                       A.p,
                                       B.p,
                                       beta,
                                       D.p,
                                       C.p,
                                       dt,
                                       rocsparse_spgemm_alg_default,
                                       rocsparse_spgemm_stage_buffer_size,
                                       &bsz,
                                       nullptr),
                      rocsparse_status_success);
            device_vector<char> tmp{bsz ? bsz : size_t{1}};
            ASSERT_TRUE(tmp.ptr);
            ASSERT_EQ(rocsparse_spgemm(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       alpha,
                                       A.p,
                                       B.p,
                                       beta,
                                       D.p,
                                       C.p,
                                       dt,
                                       rocsparse_spgemm_alg_default,
                                       rocsparse_spgemm_stage_nnz,
                                       &bsz,
                                       tmp.ptr),
                      rocsparse_status_success);
            EXPECT_EQ(rocsparse_spgemm(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       alpha,
                                       A.p,
                                       B.p,
                                       beta,
                                       D.p,
                                       C.p,
                                       dt,
                                       rocsparse_spgemm_alg_default,
                                       rocsparse_spgemm_stage_symbolic,
                                       &bsz,
                                       tmp.ptr),
                      rocsparse_status_not_implemented);
            int64_t brows = 0, bcols = 0, bnnz = 0;
            ASSERT_EQ(rocsparse_spmat_get_size(C.p, &brows, &bcols, &bnnz),
                      rocsparse_status_success);
            device_vector<int32_t> ccol{static_cast<size_t>(bnnz > 0 ? bnnz : 1)};
            device_vector<float>   cvf{static_cast<size_t>(bnnz > 0 ? bnnz : 1)};
            device_vector<_Float16> cvh{static_cast<size_t>(bnnz > 0 ? bnnz : 1)};
            device_vector<rocsparse_bfloat16> cvb{static_cast<size_t>(bnnz > 0 ? bnnz : 1)};
            void* cvals = cvf.ptr;
            if(dt == rocsparse_datatype_f16_r)
                cvals = cvh.ptr;
            if(dt == rocsparse_datatype_bf16_r)
                cvals = cvb.ptr;
            ASSERT_TRUE(ccol.ptr && cvals);
            ASSERT_EQ(rocsparse_bsr_set_pointers(C.p, cptr.ptr, ccol.ptr, cvals),
                      rocsparse_status_success);
            // Generic BSR compute accepts transpose and does not branch on it.
            EXPECT_EQ(rocsparse_spgemm(handle,
                                       rocsparse_operation_transpose,
                                       rocsparse_operation_none,
                                       alpha,
                                       A.p,
                                       B.p,
                                       beta,
                                       D.p,
                                       C.p,
                                       dt,
                                       rocsparse_spgemm_alg_default,
                                       rocsparse_spgemm_stage_compute,
                                       &bsz,
                                       tmp.ptr),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        }
    }
}

TEST_F(GemmGeamCov, csrgeam_nnz_branches)
{
    MatDescr descr_A, descr_B, descr_C;
    ASSERT_TRUE(descr_A.d && descr_B.d && descr_C.d);
    rocsparse_int nnz = -1;

    EXPECT_EQ(rocsparse_csrgeam_nnz(nullptr,
                                   0,
                                   0,
                                   descr_A.d,
                                   0,
                                   nullptr,
                                   nullptr,
                                   descr_B.d,
                                   0,
                                   nullptr,
                                   nullptr,
                                   descr_C.d,
                                   nullptr,
                                   &nnz),
              rocsparse_status_invalid_handle);

    nnz = -1;
    EXPECT_EQ(rocsparse_csrgeam_nnz(handle,
                                   0,
                                   2,
                                   descr_A.d,
                                   0,
                                   nullptr,
                                   nullptr,
                                   descr_B.d,
                                   0,
                                   nullptr,
                                   nullptr,
                                   descr_C.d,
                                   nullptr,
                                   &nnz),
              rocsparse_status_success);
    EXPECT_EQ(nnz, 0);

    // Both inputs empty: row pointer of C is filled with the index base.
    device_vector<rocsparse_int> row_C{size_t{5}};
    ASSERT_TRUE(row_C.ptr);
    ASSERT_EQ(rocsparse_set_mat_index_base(descr_C.d, rocsparse_index_base_one),
              rocsparse_status_success);
    nnz = -1;
    EXPECT_EQ(rocsparse_csrgeam_nnz(handle,
                                   4,
                                   4,
                                   descr_A.d,
                                   0,
                                   nullptr,
                                   nullptr,
                                   descr_B.d,
                                   0,
                                   nullptr,
                                   nullptr,
                                   descr_C.d,
                                   row_C.ptr,
                                   &nnz),
              rocsparse_status_success);
    EXPECT_EQ(nnz, 0);
    auto hrow = to_host(row_C);
    for(int i = 0; i < 5; ++i)
        EXPECT_EQ(hrow[i], 1);

    {
        PointerModeGuard          mode(handle, rocsparse_pointer_mode_device);
        device_vector<rocsparse_int> dnnz{size_t{1}};
        ASSERT_TRUE(dnnz.ptr);
        ASSERT_EQ(hipMemset(dnnz.ptr, 0xff, sizeof(rocsparse_int)), hipSuccess);
        EXPECT_EQ(rocsparse_csrgeam_nnz(handle,
                                       4,
                                       4,
                                       descr_A.d,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_B.d,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_C.d,
                                       row_C.ptr,
                                       dnnz.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(to_host(dnnz)[0], 0);
    }

    ASSERT_EQ(rocsparse_set_mat_type(descr_A.d, rocsparse_matrix_type_symmetric),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_csrgeam_nnz(handle,
                                   2,
                                   2,
                                   descr_A.d,
                                   0,
                                   nullptr,
                                   nullptr,
                                   descr_B.d,
                                   0,
                                   nullptr,
                                   nullptr,
                                   descr_C.d,
                                   nullptr,
                                   &nnz),
              rocsparse_status_not_implemented);
    ASSERT_EQ(rocsparse_set_mat_type(descr_A.d, rocsparse_matrix_type_general),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_storage_mode(descr_B.d, rocsparse_storage_mode_unsorted),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_csrgeam_nnz(handle,
                                   2,
                                   2,
                                   descr_A.d,
                                   1,
                                   nullptr,
                                   nullptr,
                                   descr_B.d,
                                   1,
                                   nullptr,
                                   nullptr,
                                   descr_C.d,
                                   nullptr,
                                   &nnz),
              rocsparse_status_requires_sorted_storage);
    ASSERT_EQ(rocsparse_set_mat_storage_mode(descr_B.d, rocsparse_storage_mode_sorted),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_index_base(descr_C.d, rocsparse_index_base_zero),
              rocsparse_status_success);

    // Real addition, host pointer, base zero, then device pointer with base one
    // (index-base adjust kernel).
    device_vector<rocsparse_int> row{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> col{std::vector<rocsparse_int>{0, 1}};
    device_vector<rocsparse_int> row_out{size_t{3}};
    ASSERT_TRUE(row.ptr && col.ptr && row_out.ptr);
    nnz = -1;
    ASSERT_EQ(rocsparse_csrgeam_nnz(handle,
                                   2,
                                   2,
                                   descr_A.d,
                                   2,
                                   row.ptr,
                                   col.ptr,
                                   descr_B.d,
                                   2,
                                   row.ptr,
                                   col.ptr,
                                   descr_C.d,
                                   row_out.ptr,
                                   &nnz),
              rocsparse_status_success);
    EXPECT_EQ(nnz, 2);

    ASSERT_EQ(rocsparse_set_mat_index_base(descr_C.d, rocsparse_index_base_one),
              rocsparse_status_success);
    {
        PointerModeGuard             mode(handle, rocsparse_pointer_mode_device);
        device_vector<rocsparse_int> dnnz{size_t{1}};
        ASSERT_TRUE(dnnz.ptr);
        ASSERT_EQ(rocsparse_csrgeam_nnz(handle,
                                       2,
                                       2,
                                       descr_A.d,
                                       2,
                                       row.ptr,
                                       col.ptr,
                                       descr_B.d,
                                       2,
                                       row.ptr,
                                       col.ptr,
                                       descr_C.d,
                                       row_out.ptr,
                                       dnnz.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(to_host(dnnz)[0], 2);
    }
}

TEST_F(GemmGeamCov, spgeam_type_index_transpose_scalars)
{
    const rocsparse_operation N = rocsparse_operation_none;
    const rocsparse_operation T = rocsparse_operation_transpose;
    const rocsparse_operation C = rocsparse_operation_conjugate_transpose;

    run_spgeam<int32_t, int32_t, float>(handle, N, N, false, false);
    run_spgeam<int32_t, int32_t, float>(handle, T, N, false, false);
    run_spgeam<int32_t, int32_t, float>(handle, N, T, false, false);
    run_spgeam<int32_t, int32_t, float>(handle, T, C, false, false);
    run_spgeam<int32_t, int32_t, float>(handle, N, N, false, true);
    run_spgeam<int32_t, int32_t, float>(handle, N, N, true, true);
    run_spgeam<int32_t, int32_t, double>(handle, N, N, false, false);
    run_spgeam<int32_t, int32_t, rocsparse_float_complex>(handle, C, C, false, false);
    run_spgeam<int32_t, int32_t, rocsparse_double_complex>(handle, N, N, false, false);
    run_spgeam<int64_t, int32_t, float>(handle, N, N, false, false);
    run_spgeam<int64_t, int32_t, double>(handle, T, T, false, false);
    run_spgeam<int64_t, int64_t, float>(handle, N, N, false, false);
    run_spgeam<int64_t, int64_t, rocsparse_float_complex>(handle, N, C, false, false);
    run_spgeam<int64_t, int32_t, rocsparse_double_complex>(handle, N, N, true, false);

    // Split symbolic / numeric stages, and an unsupported format on analysis.
    {
        std::vector<int32_t> ptr{0, 1, 2};
        std::vector<int32_t> col{0, 1};
        std::vector<float>   val{1.f, 2.f};
        device_vector<int32_t> dp{ptr}, dc{col};
        device_vector<float>   dv{val};
        Spmat                  A, B;
        ASSERT_EQ(rocsparse_create_csr_descr(&A.p,
                                             2,
                                             2,
                                             2,
                                             dp.ptr,
                                             dc.ptr,
                                             dv.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csr_descr(&B.p,
                                             2,
                                             2,
                                             2,
                                             dp.ptr,
                                             dc.ptr,
                                             dv.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        rocsparse_spgeam_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_spgeam_descr(&descr), rocsparse_status_success);
        const rocsparse_spgeam_alg alg = rocsparse_spgeam_alg_default;
        const rocsparse_operation   op  = rocsparse_operation_none;
        const rocsparse_datatype    dt  = rocsparse_datatype_f32_r;
        const float                 alpha = 1.f, beta = 1.f;
        ASSERT_EQ(rocsparse_spgeam_set_input(
                      handle, descr, rocsparse_spgeam_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam_set_input(
                      handle, descr, rocsparse_spgeam_input_operation_A, &op, sizeof(op), nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam_set_input(
                      handle, descr, rocsparse_spgeam_input_operation_B, &op, sizeof(op), nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                            descr,
                                            rocsparse_spgeam_input_scalar_datatype,
                                            &dt,
                                            sizeof(dt),
                                            nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                            descr,
                                            rocsparse_spgeam_input_compute_datatype,
                                            &dt,
                                            sizeof(dt),
                                            nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                            descr,
                                            rocsparse_spgeam_input_scalar_alpha,
                                            &alpha,
                                            sizeof(void*),
                                            nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                            descr,
                                            rocsparse_spgeam_input_scalar_beta,
                                            &beta,
                                            sizeof(void*),
                                            nullptr),
                  rocsparse_status_success);
        const rocsparse_spgeam_alg bad_alg = static_cast<rocsparse_spgeam_alg>(99);
        EXPECT_EQ(rocsparse_spgeam_set_input(handle,
                                            descr,
                                            rocsparse_spgeam_input_alg,
                                            &bad_alg,
                                            sizeof(bad_alg),
                                            nullptr),
                  rocsparse_status_success);
        size_t bsz = 0;
        // Algorithm is stored unchecked. buffer_size does not read it; spgeam does.
        EXPECT_EQ(rocsparse_spgeam_buffer_size(handle,
                                              descr,
                                              A.p,
                                              B.p,
                                              nullptr,
                                              rocsparse_spgeam_stage_analysis,
                                              &bsz,
                                              nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_spgeam(handle,
                                  descr,
                                  A.p,
                                  B.p,
                                  nullptr,
                                  rocsparse_spgeam_stage_analysis,
                                  0,
                                  nullptr,
                                  nullptr),
                  rocsparse_status_invalid_value);
        ASSERT_EQ(rocsparse_spgeam_set_input(
                      handle, descr, rocsparse_spgeam_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_success);

        ASSERT_EQ(rocsparse_spgeam_buffer_size(handle,
                                              descr,
                                              A.p,
                                              B.p,
                                              nullptr,
                                              rocsparse_spgeam_stage_symbolic_analysis,
                                              &bsz,
                                              nullptr),
                  rocsparse_status_success);
        device_vector<char> tmp{bsz ? bsz : size_t{1}};
        ASSERT_EQ(rocsparse_spgeam(handle,
                                  descr,
                                  A.p,
                                  B.p,
                                  nullptr,
                                  rocsparse_spgeam_stage_symbolic_analysis,
                                  bsz,
                                  tmp.ptr,
                                  nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        int64_t nnz_C = 0;
        ASSERT_EQ(rocsparse_spgeam_get_output(
                      handle, descr, rocsparse_spgeam_output_nnz, &nnz_C, sizeof(nnz_C), nullptr),
                  rocsparse_status_success);
        device_vector<int32_t> cptr{size_t{3}};
        device_vector<int32_t> ccol{static_cast<size_t>(nnz_C > 0 ? nnz_C : 1)};
        device_vector<float>   cval{static_cast<size_t>(nnz_C > 0 ? nnz_C : 1)};
        Spmat                  Cm;
        ASSERT_EQ(rocsparse_create_csr_descr(&Cm.p,
                                             2,
                                             2,
                                             nnz_C,
                                             cptr.ptr,
                                             ccol.ptr,
                                             cval.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam(handle,
                                  descr,
                                  A.p,
                                  B.p,
                                  Cm.p,
                                  rocsparse_spgeam_stage_symbolic_compute,
                                  bsz,
                                  tmp.ptr,
                                  nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam(handle,
                                  descr,
                                  A.p,
                                  B.p,
                                  Cm.p,
                                  rocsparse_spgeam_stage_numeric_analysis,
                                  bsz,
                                  tmp.ptr,
                                  nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam(handle,
                                  descr,
                                  A.p,
                                  B.p,
                                  Cm.p,
                                  rocsparse_spgeam_stage_numeric_compute,
                                  bsz,
                                  tmp.ptr,
                                  nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spgeam_descr(descr), rocsparse_status_success);
    }

    device_vector<int32_t> idx{std::vector<int32_t>{0, 0}};
    device_vector<float>   val{std::vector<float>{1.f}};
    ASSERT_TRUE(idx.ptr && val.ptr);
    Spmat CA, CB;
    ASSERT_EQ(rocsparse_create_coo_descr(&CA.p,
                                        1,
                                        1,
                                        1,
                                        idx.ptr,
                                        idx.ptr,
                                        val.ptr,
                                        rocsparse_indextype_i32,
                                        rocsparse_index_base_zero,
                                        rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_coo_descr(&CB.p,
                                        1,
                                        1,
                                        1,
                                        idx.ptr,
                                        idx.ptr,
                                        val.ptr,
                                        rocsparse_indextype_i32,
                                        rocsparse_index_base_zero,
                                        rocsparse_datatype_f32_r),
              rocsparse_status_success);
    Spmat CC;
    ASSERT_EQ(rocsparse_create_coo_descr(&CC.p,
                                        1,
                                        1,
                                        1,
                                        idx.ptr,
                                        idx.ptr,
                                        val.ptr,
                                        rocsparse_indextype_i32,
                                        rocsparse_index_base_zero,
                                        rocsparse_datatype_f32_r),
              rocsparse_status_success);
    rocsparse_spgeam_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_spgeam_descr(&descr), rocsparse_status_success);
    const rocsparse_spgeam_alg alg = rocsparse_spgeam_alg_default;
    const rocsparse_operation   op  = rocsparse_operation_none;
    const rocsparse_datatype    dt  = rocsparse_datatype_f32_r;
    ASSERT_EQ(rocsparse_spgeam_set_input(
                  handle, descr, rocsparse_spgeam_input_alg, &alg, sizeof(alg), nullptr),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_spgeam_set_input(
                  handle, descr, rocsparse_spgeam_input_operation_A, &op, sizeof(op), nullptr),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_spgeam_set_input(
                  handle, descr, rocsparse_spgeam_input_operation_B, &op, sizeof(op), nullptr),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_spgeam_set_input(
                  handle, descr, rocsparse_spgeam_input_compute_datatype, &dt, sizeof(dt), nullptr),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_spgeam_set_input(
                  handle, descr, rocsparse_spgeam_input_scalar_datatype, &dt, sizeof(dt), nullptr),
              rocsparse_status_success);
    size_t bsz = 0;
    const rocsparse_spgeam_stage early[] = {rocsparse_spgeam_stage_analysis,
                                            rocsparse_spgeam_stage_symbolic_analysis,
                                            rocsparse_spgeam_stage_numeric_analysis};
    for(auto stage : early)
    {
        rocsparse_spgeam_descr d2 = nullptr;
        ASSERT_EQ(rocsparse_create_spgeam_descr(&d2), rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam_set_input(
                      handle, d2, rocsparse_spgeam_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam_set_input(
                      handle, d2, rocsparse_spgeam_input_operation_A, &op, sizeof(op), nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam_set_input(
                      handle, d2, rocsparse_spgeam_input_operation_B, &op, sizeof(op), nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                            d2,
                                            rocsparse_spgeam_input_compute_datatype,
                                            &dt,
                                            sizeof(dt),
                                            nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                            d2,
                                            rocsparse_spgeam_input_scalar_datatype,
                                            &dt,
                                            sizeof(dt),
                                            nullptr),
                  rocsparse_status_success);
        const float alpha = 1.f, beta = 1.f;
        ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                            d2,
                                            rocsparse_spgeam_input_scalar_alpha,
                                            &alpha,
                                            sizeof(void*),
                                            nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                            d2,
                                            rocsparse_spgeam_input_scalar_beta,
                                            &beta,
                                            sizeof(void*),
                                            nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_spgeam_buffer_size(handle, d2, CA.p, CB.p, nullptr, stage, &bsz, nullptr),
                  rocsparse_status_not_implemented);
        const rocsparse_status st
            = rocsparse_spgeam(handle, d2, CA.p, CB.p, nullptr, stage, bsz, nullptr, nullptr);
        if(stage == rocsparse_spgeam_stage_numeric_analysis)
        {
            // Numeric analysis is a no-op and does not look at the format.
            // Numeric compute then rejects every format other than CSR.
            EXPECT_EQ(st, rocsparse_status_success);
            EXPECT_EQ(rocsparse_spgeam(handle,
                                      d2,
                                      CA.p,
                                      CB.p,
                                      CC.p,
                                      rocsparse_spgeam_stage_numeric_compute,
                                      0,
                                      nullptr,
                                      nullptr),
                      rocsparse_status_not_implemented);
        }
        else
        {
            EXPECT_EQ(st, rocsparse_status_not_implemented);
        }
        EXPECT_EQ(rocsparse_destroy_spgeam_descr(d2), rocsparse_status_success);
    }
    EXPECT_EQ(rocsparse_destroy_spgeam_descr(descr), rocsparse_status_success);
}
