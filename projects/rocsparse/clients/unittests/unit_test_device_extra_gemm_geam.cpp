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
// the csrgemm numeric group, plus 2x2 products for the type / index switches
// and the legacy transpose rejection. gfx1201 is wavefront 32; the
// wavefront-64 arms cannot run here.
//
#include "unit_test_extra_common.hpp"

#include <functional>
#include <optional>
#include <type_traits>
#include <vector>

using namespace rocsparse_ut;
using extra_ut::any_error;
using extra_ut::MatDescrH;
using extra_ut::MatInfoH;
using extra_ut::PointerModeGuard;
using extra_ut::spgeam_set_inputs;
using extra_ut::SpgeamDescrH;
using extra_ut::SpmatH;
using extra_ut::status_is;

namespace
{
    using extra_ut::Bsr;
    using extra_ut::bsr_from_csr;
    using extra_ut::cplx;
    using extra_ut::Csr;
    using extra_ut::DevBuf;

    // A = [1, 2; 0, 3] (first) or B = [4, 0; 5, 6]. Complex types get
    // imaginary parts, so a dropped conjugation or imaginary part shows.
    template <typename T>
    Csr pair_operand(bool first)
    {
        const double im = extra_ut::is_complex_t<T>::value ? 1.0 : 0.0;
        Csr          M;
        M.m = M.n = 2;
        M.ptr     = first ? std::vector<int64_t>{0, 2, 3} : std::vector<int64_t>{0, 1, 3};
        M.col     = first ? std::vector<int64_t>{0, 1, 1} : std::vector<int64_t>{0, 0, 1};
        M.val     = first ? std::vector<cplx>{1.0, cplx(2.0, im), 3.0}
                          : std::vector<cplx>{4.0, cplx(5.0, -2.0 * im), 6.0};
        return M;
    }

    // One generic spgemm problem C = alpha * A * B + beta * D and the stages
    // that are issued on it.
    struct SpgemmFlow
    {
        rocsparse_handle   handle;
        rocsparse_datatype dt;
        const void*        alpha;
        const void*        beta;
        SpmatH             A, B, D, C;
        size_t             bsz = 0;

        rocsparse_status stage(rocsparse_spgemm_stage s, void* buffer)
        {
            return rocsparse_spgemm(handle,
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
                                    s,
                                    &bsz,
                                    buffer);
        }

        // Buffer size, nnz and C's arrays: after this, `compute` is the next
        // stage. Returns the status of the first failing stage.
        template <typename I, typename J, typename T>
        rocsparse_status prepare(device_vector<I>&                   C_ptr,
                                 std::optional<device_vector<J>>&    C_col,
                                 std::optional<device_vector<T>>&    C_val,
                                 std::optional<device_vector<char>>& buffer)
        {
            auto st = stage(rocsparse_spgemm_stage_buffer_size, nullptr);
            if(st != rocsparse_status_success)
                return st;
            buffer.emplace(bsz ? bsz : size_t{1});
            st = stage(rocsparse_spgemm_stage_nnz, buffer->ptr);
            if(st != rocsparse_status_success)
                return st;
            int64_t rows = 0, cols = 0, nnz = 0;
            st = rocsparse_spmat_get_size(C.p, &rows, &cols, &nnz);
            if(st != rocsparse_status_success)
                return st;
            C_col.emplace(static_cast<size_t>(nnz > 0 ? nnz : 1));
            C_val.emplace(static_cast<size_t>(nnz > 0 ? nnz : 1));
            return rocsparse_csr_set_pointers(C.p, C_ptr.ptr, C_col->ptr, C_val->ptr);
        }
    };

    // C = alpha * A * B [+ beta * D]. A is 1 x n (dense, or empty), B is n x n
    // (two entries per row, or empty), so the single output row has exactly n
    // non-zeros and lands in one csrgemm numeric group.
    enum class WideKind
    {
        product,
        add_only,
        product_and_add
    };

    template <typename I, typename J, typename T>
    void run_wide(rocsparse_handle     handle,
                  int                  n,
                  WideKind             kind,
                  bool                 device_scalars,
                  rocsparse_index_base base = rocsparse_index_base_zero)
    {
        SCOPED_TRACE("n=" + std::to_string(n) + " kind=" + std::to_string(static_cast<int>(kind))
                     + " device=" + std::to_string(device_scalars));

        const bool    mul = kind != WideKind::add_only;
        const bool    add = kind != WideKind::product;
        const int64_t N   = n;

        // A and D: one row of distinct small integers. B: entries (r, r) and
        // (r, r + 1 mod n) in every row, so every entry of C sums two products.
        // An absent term has no entries. The values keep f16 / bf16 exact.
        Csr hA, hB, hD;
        hA.m = hD.m = 1;
        hA.n = hB.m = hB.n = hD.n = N;
        hA.ptr                    = {0, mul ? N : 0};
        hD.ptr                    = {0, add ? N : 0};
        hB.ptr.assign(static_cast<size_t>(N) + 1, 0);
        for(int64_t c = 0; c < N; ++c)
        {
            const int64_t next = (c + 1) % N;
            if(mul)
            {
                hA.col.push_back(c);
                hA.val.push_back(static_cast<double>(c % 4 + 1));
                for(int64_t j : {std::min(c, next), std::max(c, next)})
                {
                    hB.col.push_back(j);
                    hB.val.push_back(j == c ? static_cast<double>(c % 3 + 1)
                                            : -static_cast<double>(c % 2 + 1));
                }
            }
            hB.ptr[static_cast<size_t>(c) + 1] = static_cast<int64_t>(hB.col.size());
            if(add)
            {
                hD.col.push_back(c);
                hD.val.push_back(static_cast<double>(c % 3 + 1));
            }
        }
        DevBuf<I>        dA_ptr, dB_ptr, dD_ptr;
        DevBuf<J>        dA_col, dB_col, dD_col;
        DevBuf<T>        dA_val, dB_val, dD_val;
        device_vector<I> dC_ptr{size_t{2}};
        ASSERT_TRUE(dC_ptr.ptr);

        const rocsparse_datatype  dt = dt_of<T>();
        const rocsparse_indextype it = it_of<I>();
        const rocsparse_indextype jt = it_of<J>();

        SpgemmFlow f{handle, dt};
        ASSERT_NO_FATAL_FAILURE(
            (extra_ut::make_csr_descr<T, I, J>(f.A, hA, dA_ptr, dA_col, dA_val, base)));
        ASSERT_NO_FATAL_FAILURE(
            (extra_ut::make_csr_descr<T, I, J>(f.B, hB, dB_ptr, dB_col, dB_val, base)));
        ASSERT_NO_FATAL_FAILURE(
            (extra_ut::make_csr_descr<T, I, J>(f.D, hD, dD_ptr, dD_col, dD_val, base)));
        ASSERT_EQ(rocsparse_create_csr_descr(
                      &f.C.p, 1, N, 0, dC_ptr.ptr, nullptr, nullptr, it, jt, base, dt),
                  rocsparse_status_success);

        const T          alpha_h = scalar<T>(2.f);
        const T          beta_h  = scalar<T>(add ? 3.f : 0.f);
        device_vector<T> d_alpha{std::vector<T>{alpha_h}};
        device_vector<T> d_beta{std::vector<T>{beta_h}};
        ASSERT_TRUE(d_alpha.ptr && d_beta.ptr);
        f.alpha = device_scalars ? static_cast<const void*>(d_alpha.ptr) : &alpha_h;
        f.beta  = device_scalars ? static_cast<const void*>(d_beta.ptr) : &beta_h;

        PointerModeGuard mode(
            handle, device_scalars ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host);

        std::optional<device_vector<J>>    dC_col;
        std::optional<device_vector<T>>    dC_val;
        std::optional<device_vector<char>> tmp;
        ASSERT_EQ((f.prepare<I, J, T>(dC_ptr, dC_col, dC_val, tmp)), rocsparse_status_success);
        int64_t rows = 0, cols_C = 0, nnz = 0;
        ASSERT_EQ(rocsparse_spmat_get_size(f.C.p, &rows, &cols_C, &nnz), rocsparse_status_success);
        ASSERT_EQ(nnz, N);
        ASSERT_EQ(f.stage(rocsparse_spgemm_stage_compute, tmp->ptr), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const Bsr want = extra_ut::bsr_gemm(
            mul, 2.0, bsr_from_csr(hA), bsr_from_csr(hB), add, 3.0, bsr_from_csr(hD), 1, N);
        extra_ut::expect_bsr_equal<T, I, J>(to_host(dC_ptr),
                                            to_host(*dC_col),
                                            to_host(*dC_val),
                                            want,
                                            rocsparse_direction_row,
                                            static_cast<int64_t>(base));
    }

    // C = 2 * A * B with the non-symmetric pair_operand matrices.
    template <typename I, typename J, typename T>
    void run_pair_spgemm(rocsparse_handle handle, bool symbolic)
    {
        const rocsparse_datatype   dt   = dt_of<T>();
        const rocsparse_indextype  it   = it_of<I>();
        const rocsparse_indextype  jt   = it_of<J>();
        const rocsparse_index_base base = rocsparse_index_base_zero;
        const Csr                  hA   = pair_operand<T>(true);
        const Csr                  hB   = pair_operand<T>(false);
        DevBuf<I>                  Ap, Bp;
        DevBuf<J>                  Ac, Bc;
        DevBuf<T>                  Av, Bv;
        device_vector<I>           Cp{size_t{3}};
        ASSERT_TRUE(Cp.ptr);

        SpgemmFlow f{handle, dt};
        const T    alpha = scalar<T>(2.f);
        const T    beta  = scalar<T>(0.f);
        f.alpha          = &alpha;
        f.beta           = &beta;
        ASSERT_NO_FATAL_FAILURE((extra_ut::make_csr_descr<T, I, J>(f.A, hA, Ap, Ac, Av)));
        ASSERT_NO_FATAL_FAILURE((extra_ut::make_csr_descr<T, I, J>(f.B, hB, Bp, Bc, Bv)));
        ASSERT_EQ(rocsparse_create_csr_descr(
                      &f.D.p, 0, 0, 0, nullptr, nullptr, nullptr, it, jt, base, dt),
                  rocsparse_status_success);
        ASSERT_EQ(
            rocsparse_create_csr_descr(&f.C.p, 2, 2, 0, Cp.ptr, nullptr, nullptr, it, jt, base, dt),
            rocsparse_status_success);

        std::optional<device_vector<J>>    Cc;
        std::optional<device_vector<T>>    Cv;
        std::optional<device_vector<char>> tmp;
        ASSERT_EQ((f.prepare<I, J, T>(Cp, Cc, Cv, tmp)), rocsparse_status_success);
        ASSERT_EQ(
            f.stage(symbolic ? rocsparse_spgemm_stage_symbolic : rocsparse_spgemm_stage_compute,
                    tmp->ptr),
            rocsparse_status_success);
        if(symbolic)
            ASSERT_EQ(f.stage(rocsparse_spgemm_stage_numeric, tmp->ptr), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const Bsr want = extra_ut::bsr_gemm(true,
                                            2.0,
                                            bsr_from_csr(hA),
                                            bsr_from_csr(hB),
                                            false,
                                            0.0,
                                            extra_ut::empty_bsr(2, 2, 1),
                                            2,
                                            2);
        extra_ut::expect_bsr_equal<T, I, J>(
            to_host(Cp), to_host(*Cc), to_host(*Cv), want, rocsparse_direction_row, 0);
    }

    // Group 10 rejects unsorted B or D. A row with more than 32768 non-zeros
    // selects that path on every wavefront size. A shorter row only falls
    // through to it when the hash does not fit in shared memory. Nnz and
    // symbolic stay on the sorted path; storage is flipped only before numeric.
    void run_unsorted_group10(rocsparse_handle handle, bool unsorted_B, bool unsorted_D)
    {
        SCOPED_TRACE(std::string("unsorted B=") + (unsorted_B ? "1" : "0")
                     + " D=" + (unsorted_D ? "1" : "0"));
        const int                  n = 32769;
        const int                  m = 1;
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

        MatDescrH descr_A, descr_B, descr_D, descr_C;
        MatInfoH  info;
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
    void run_spgeam(rocsparse_handle handle, bool device_scalars)
    {
        // C = 2 * A + 3 * B with the pair_operand matrices.
        const rocsparse_datatype   dt   = dt_of<T>();
        const rocsparse_indextype  it   = it_of<I>();
        const rocsparse_indextype  jt   = it_of<J>();
        const rocsparse_index_base base = rocsparse_index_base_zero;
        const Csr                  hA   = pair_operand<T>(true);
        const Csr                  hB   = pair_operand<T>(false);
        DevBuf<I>                  dA_ptr, dB_ptr;
        DevBuf<J>                  dA_col, dB_col;
        DevBuf<T>                  dA_val, dB_val;
        SpmatH                     A, B;
        ASSERT_NO_FATAL_FAILURE((extra_ut::make_csr_descr<T, I, J>(A, hA, dA_ptr, dA_col, dA_val)));
        ASSERT_NO_FATAL_FAILURE((extra_ut::make_csr_descr<T, I, J>(B, hB, dB_ptr, dB_col, dB_val)));
        const Bsr want = extra_ut::bsr_geam(2.0, bsr_from_csr(hA), 3.0, bsr_from_csr(hB));

        const T          alpha = scalar<T>(2.f);
        const T          beta  = scalar<T>(3.f);
        device_vector<T> d_alpha{std::vector<T>{alpha}};
        device_vector<T> d_beta{std::vector<T>{beta}};
        ASSERT_TRUE(d_alpha.ptr && d_beta.ptr);

        SpgeamDescrH                 holder;
        const rocsparse_spgeam_descr descr = holder.d;
        ASSERT_NE(descr, nullptr);
        ASSERT_NO_FATAL_FAILURE(
            spgeam_set_inputs(handle,
                              descr,
                              dt,
                              dt,
                              device_scalars ? static_cast<const void*>(d_alpha.ptr) : &alpha,
                              device_scalars ? static_cast<const void*>(d_beta.ptr) : &beta));

        size_t bsz = 0;
        ASSERT_EQ(
            rocsparse_spgeam_buffer_size(
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
        ASSERT_EQ(nnz_C, want.nnzb());

        device_vector<I> dC_ptr{size_t{3}};
        device_vector<J> dC_col{static_cast<size_t>(std::max<int64_t>(nnz_C, 1))};
        device_vector<T> dC_val{static_cast<size_t>(std::max<int64_t>(nnz_C, 1))};
        ASSERT_TRUE(dC_ptr.ptr && dC_col.ptr && dC_val.ptr);
        SpmatH C;
        ASSERT_EQ(rocsparse_create_csr_descr(
                      &C.p, 2, 2, nnz_C, dC_ptr.ptr, dC_col.ptr, dC_val.ptr, it, jt, base, dt),
                  rocsparse_status_success);

        PointerModeGuard mode(
            handle, device_scalars ? rocsparse_pointer_mode_device : rocsparse_pointer_mode_host);
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
        extra_ut::expect_bsr_equal<T, I, J>(
            to_host(dC_ptr), to_host(dC_col), to_host(dC_val), want, rocsparse_direction_row, 0);
    }

    class ExtraGemmGeam : public HandleTest
    {
    };
}

TEST_F(ExtraGemmGeam, csrgemm_numeric_groups)
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

TEST_F(ExtraGemmGeam, spgemm_type_index_transpose)
{
    const rocsparse_operation N  = rocsparse_operation_none;
    const rocsparse_operation T  = rocsparse_operation_transpose;
    const rocsparse_operation C  = rocsparse_operation_conjugate_transpose;
    const auto                ok = rocsparse_status_success;

    // Every value type for an index type pair; `symbolic_type` is the position
    // (in the order below) of the one type that runs the split symbolic flow.
    auto all_types = [&](auto i, auto j, int symbolic_type) {
        using I = decltype(i);
        using J = decltype(j);
        run_pair_spgemm<I, J, float>(handle, symbolic_type == 0);
        run_pair_spgemm<I, J, double>(handle, symbolic_type == 1);
        run_pair_spgemm<I, J, rocsparse_float_complex>(handle, symbolic_type == 2);
        run_pair_spgemm<I, J, rocsparse_double_complex>(handle, symbolic_type == 3);
        run_pair_spgemm<I, J, _Float16>(handle, symbolic_type == 4);
        run_pair_spgemm<I, J, rocsparse_bfloat16>(handle, symbolic_type == 5);
    };
    all_types(int32_t{}, int32_t{}, 0);
    all_types(int64_t{}, int32_t{}, 4);
    all_types(int64_t{}, int64_t{}, 5);
    run_pair_spgemm<int32_t, int32_t, float>(handle, true);

    // The legacy entry points reject transpose before any kernel runs.
    device_vector<rocsparse_int> rp{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> ci{std::vector<rocsparse_int>{0, 1}};
    device_vector<float>         va{std::vector<float>{1.f, 1.f}};
    device_vector<rocsparse_int> cprow{size_t{3}};
    device_vector<rocsparse_int> ccol{size_t{2}};
    device_vector<float>         cval{size_t{2}};
    device_vector<rocsparse_int> hnnz{std::vector<rocsparse_int>{0}};
    ASSERT_TRUE(rp.ptr && ci.ptr && va.ptr && cprow.ptr && ccol.ptr && cval.ptr && hnnz.ptr);
    MatDescrH da, db, dc;
    MatInfoH  info;
    ASSERT_TRUE(da.d && db.d && dc.d && info.i);
    const float alpha       = 1.f;
    size_t      bsz         = 0;
    auto        buffer_size = [&](rocsparse_operation opA, rocsparse_operation opB) {
        return rocsparse_scsrgemm_buffer_size(handle,
                                              opA,
                                              opB,
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
                                              &bsz);
    };
    for(rocsparse_operation op : {T, C})
    {
        EXPECT_EQ(buffer_size(op, N), rocsparse_status_not_implemented);
        EXPECT_EQ(buffer_size(N, op), rocsparse_status_not_implemented);
    }
    ASSERT_EQ(buffer_size(N, N), ok);
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

TEST_F(ExtraGemmGeam, spgemm_rejected_types_formats_and_bsr)
{
    constexpr auto I32 = rocsparse_indextype_i32;
    constexpr auto I64 = rocsparse_indextype_i64;
    constexpr auto ZB  = rocsparse_index_base_zero;
    constexpr auto F32 = rocsparse_datatype_f32_r;
    constexpr auto N   = rocsparse_operation_none;
    constexpr auto DEF = rocsparse_spgemm_alg_default;
    constexpr auto BUF = rocsparse_spgemm_stage_buffer_size;

    device_vector<int32_t> ip{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> ic{std::vector<int32_t>{0, 1}};
    device_vector<float>   iv{std::vector<float>{1.f, 1.f}};
    device_vector<int32_t> cp{size_t{3}};
    ASSERT_TRUE(ip.ptr && ic.ptr && iv.ptr && cp.ptr);

    // The four operands of C = A * B + D: a 2x2 A, B and C, and an empty D.
    struct Operands
    {
        SpmatH A, B, D, C;
    };
    auto make_csr =
        [&](Operands& o, rocsparse_indextype it, rocsparse_indextype jt, rocsparse_datatype dt) {
            ASSERT_EQ(
                rocsparse_create_csr_descr(&o.A.p, 2, 2, 2, ip.ptr, ic.ptr, iv.ptr, it, jt, ZB, dt),
                rocsparse_status_success);
            ASSERT_EQ(
                rocsparse_create_csr_descr(&o.B.p, 2, 2, 2, ip.ptr, ic.ptr, iv.ptr, it, jt, ZB, dt),
                rocsparse_status_success);
            ASSERT_EQ(rocsparse_create_csr_descr(
                          &o.D.p, 0, 0, 0, nullptr, nullptr, nullptr, it, jt, ZB, dt),
                      rocsparse_status_success);
            ASSERT_EQ(
                rocsparse_create_csr_descr(&o.C.p, 2, 2, 2, ip.ptr, ic.ptr, iv.ptr, it, jt, ZB, dt),
                rocsparse_status_success);
        };
    auto spgemm = [&](Operands&              o,
                      rocsparse_datatype     dt,
                      const void*            alpha,
                      const void*            beta,
                      rocsparse_spgemm_alg   alg   = DEF,
                      rocsparse_spgemm_stage stage = BUF,
                      void*                  buf   = nullptr,
                      rocsparse_operation    opA   = N) {
        size_t bsz = 0;
        return rocsparse_spgemm(
            handle, opA, N, alpha, o.A.p, o.B.p, beta, o.D.p, o.C.p, dt, alg, stage, &bsz, buf);
    };
    const float one = 1.f, zero = 0.f;

    // Integer value types.
    const int one_i = 1;
    for(auto dt : {rocsparse_datatype_i8_r,
                   rocsparse_datatype_u8_r,
                   rocsparse_datatype_i32_r,
                   rocsparse_datatype_u32_r})
        for(auto pr :
            {std::make_pair(I32, I32), std::make_pair(I64, I32), std::make_pair(I64, I64)})
        {
            Operands o;
            ASSERT_NO_FATAL_FAILURE(make_csr(o, pr.first, pr.second, dt));
            EXPECT_TRUE(status_is(spgemm(o, dt, &one_i, &one_i), any_error));
        }

    // Mixed index types, and the removed u16 index type (value 1, only named
    // inside the library under ROCSPARSE_KEEP_INTERNAL_U16), which the library
    // still rejects.
    constexpr auto u16_removed = static_cast<rocsparse_indextype>(1);
    for(auto pr : {std::make_pair(I32, I64),
                   std::make_pair(u16_removed, I32),
                   std::make_pair(I32, u16_removed),
                   std::make_pair(I64, u16_removed),
                   std::make_pair(u16_removed, u16_removed)})
    {
        Operands o;
        ASSERT_NO_FATAL_FAILURE(make_csr(o, pr.first, pr.second, F32));
        EXPECT_TRUE(status_is(spgemm(o, F32, &one, &zero), any_error));
    }

    // Invalid algorithm / stage, compute-type mismatch, batched matrix.
    {
        Operands o;
        ASSERT_NO_FATAL_FAILURE(make_csr(o, I32, I32, F32));
        EXPECT_TRUE(status_is(spgemm(o, F32, &one, &zero, static_cast<rocsparse_spgemm_alg>(99)),
                              any_error));
        EXPECT_TRUE(status_is(
            spgemm(o, F32, &one, &zero, DEF, static_cast<rocsparse_spgemm_stage>(99)), any_error));
        EXPECT_TRUE(status_is(spgemm(o, rocsparse_datatype_f64_r, &one, &zero), any_error));
        ASSERT_EQ(rocsparse_csr_set_strided_batch(o.A.p, 2, 3, 2), rocsparse_status_success);
        EXPECT_TRUE(status_is(spgemm(o, F32, &one, &zero), any_error));
    }

    // Formats other than CSR and BSR are rejected at every stage.
    device_vector<int32_t> one_idx{std::vector<int32_t>{0, 0}};
    device_vector<int32_t> colptr{std::vector<int32_t>{0, 1}};
    device_vector<float>   one_val{std::vector<float>{1.f}};
    device_vector<int32_t> sell_off{std::vector<int32_t>{0, 1}};
    ASSERT_TRUE(one_idx.ptr && colptr.ptr && one_val.ptr && sell_off.ptr);
    const auto                                                                 idx = one_idx.ptr;
    const auto                                                                 val = one_val.ptr;
    const std::vector<std::function<rocsparse_status(rocsparse_spmat_descr*)>> builders = {
        [&](auto* m) {
            return rocsparse_create_coo_descr(m, 1, 1, 1, idx, idx, val, I32, ZB, F32);
        },
        [&](auto* m) { return rocsparse_create_coo_aos_descr(m, 1, 1, 1, idx, val, I32, ZB, F32); },
        [&](auto* m) {
            return rocsparse_create_csc_descr(m, 1, 1, 1, colptr.ptr, idx, val, I32, I32, ZB, F32);
        },
        [&](auto* m) { return rocsparse_create_ell_descr(m, 1, 1, idx, val, 1, I32, ZB, F32); },
        [&](auto* m) {
            return rocsparse_create_bell_descr(
                m, 1, 1, rocsparse_direction_row, 1, 1, idx, val, I32, ZB, F32);
        },
        [&](auto* m) {
            return rocsparse_create_sell_descr(
                m, 2, 2, 1, 2, 1, sell_off.ptr, idx, val, I32, I32, ZB, F32);
        }};
    for(const auto& build : builders)
    {
        Operands o;
        ASSERT_EQ(build(&o.A.p), rocsparse_status_success);
        ASSERT_EQ(build(&o.B.p), rocsparse_status_success);
        ASSERT_EQ(build(&o.D.p), rocsparse_status_success);
        ASSERT_EQ(build(&o.C.p), rocsparse_status_success);
        for(auto stage : {BUF,
                          rocsparse_spgemm_stage_nnz,
                          rocsparse_spgemm_stage_compute,
                          rocsparse_spgemm_stage_symbolic,
                          rocsparse_spgemm_stage_numeric})
            EXPECT_TRUE(status_is(spgemm(o, F32, &one, &zero, DEF, stage), any_error));
    }

    // BSR is implemented for buffer/nnz/compute; symbolic/numeric are CSR only.
    // C = 2 * A * B with blocks of distinct values, so for bd = 2 the block
    // direction changes the result (bd = 1 takes the csrgemm path).
    auto bsr_case = [&](auto* tag, rocsparse_datatype dt, rocsparse_direction dir, int64_t bd) {
        using V = std::remove_pointer_t<decltype(tag)>;
        SCOPED_TRACE("dir=" + std::to_string(dir) + " bd=" + std::to_string(bd));
        Bsr hA, hB;
        hA.mb = hA.nb = hB.mb = hB.nb = 2;
        hA.bd = hB.bd = bd;
        hA.ptr        = {0, 2, 3};
        hA.col        = {0, 1, 1};
        hB.ptr        = {0, 1, 3};
        hB.col        = {0, 0, 1};
        for(int64_t e = 0; e < 3 * bd * bd; ++e)
        {
            hA.val.push_back(static_cast<double>((e * 5) % 7 - 3));
            hB.val.push_back(static_cast<double>((e * 3 + 1) % 7 - 3));
        }
        const Bsr want = extra_ut::bsr_gemm(
            true, 2.0, hA, hB, false, 0.0, extra_ut::empty_bsr(2, 2, bd), 2, 2);
        const V                alpha_v(2.f), zero_v(0.f);
        auto                   Ap = extra_ut::up_idx<int32_t>(hA.ptr);
        auto                   Ac = extra_ut::up_idx<int32_t>(hA.col);
        auto                   Bp = extra_ut::up_idx<int32_t>(hB.ptr);
        auto                   Bc = extra_ut::up_idx<int32_t>(hB.col);
        DevBuf<V>              Av(extra_ut::bsr_pack<V>(hA, dir));
        DevBuf<V>              Bv(extra_ut::bsr_pack<V>(hB, dir));
        device_vector<int32_t> cptr{size_t{3}};
        ASSERT_TRUE(Ap.ptr && Ac.ptr && Bp.ptr && Bc.ptr && Av.ptr && Bv.ptr && cptr.ptr);
        Operands o;
        auto     mk = [&](SpmatH& m, const DevBuf<int32_t>& p, const DevBuf<int32_t>& c, V* v) {
            return rocsparse_create_bsr_descr(
                &m.p, 2, 2, 3, dir, bd, p.ptr, c.ptr, v, I32, I32, ZB, dt);
        };
        ASSERT_EQ(mk(o.A, Ap, Ac, Av.ptr), rocsparse_status_success);
        ASSERT_EQ(mk(o.B, Bp, Bc, Bv.ptr), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_bsr_descr(
                      &o.D.p, 0, 0, 0, dir, bd, nullptr, nullptr, nullptr, I32, I32, ZB, dt),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_bsr_descr(
                      &o.C.p, 2, 2, 0, dir, bd, cptr.ptr, nullptr, nullptr, I32, I32, ZB, dt),
                  rocsparse_status_success);

        size_t bsz  = 0;
        auto   call = [&](rocsparse_spgemm_stage stage, void* buf) {
            return rocsparse_spgemm(handle,
                                    N,
                                    N,
                                    &alpha_v,
                                    o.A.p,
                                    o.B.p,
                                    &zero_v,
                                    o.D.p,
                                    o.C.p,
                                    dt,
                                    DEF,
                                    stage,
                                    &bsz,
                                    buf);
        };
        ASSERT_EQ(call(BUF, nullptr), rocsparse_status_success);
        device_vector<char> tmp{bsz ? bsz : size_t{1}};
        ASSERT_TRUE(tmp.ptr);
        ASSERT_EQ(call(rocsparse_spgemm_stage_nnz, tmp.ptr), rocsparse_status_success);
        EXPECT_TRUE(status_is(call(rocsparse_spgemm_stage_symbolic, tmp.ptr), any_error));
        int64_t brows = 0, bcols = 0, bnnz = 0;
        ASSERT_EQ(rocsparse_spmat_get_size(o.C.p, &brows, &bcols, &bnnz), rocsparse_status_success);
        ASSERT_EQ(bnnz, want.nnzb());
        device_vector<int32_t> ccol{static_cast<size_t>(bnnz)};
        device_vector<V>       cv{static_cast<size_t>(bnnz * bd * bd)};
        ASSERT_TRUE(ccol.ptr && cv.ptr);
        ASSERT_EQ(rocsparse_bsr_set_pointers(o.C.p, cptr.ptr, ccol.ptr, cv.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(call(rocsparse_spgemm_stage_compute, tmp.ptr), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        extra_ut::expect_bsr_equal<V, int32_t, int32_t>(
            to_host(cptr), to_host(ccol), to_host(cv), want, dir, 0);
    };
    for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
        for(int64_t bd : {1, 2})
        {
            bsr_case(static_cast<float*>(nullptr), F32, dir, bd);
            bsr_case(static_cast<_Float16*>(nullptr), rocsparse_datatype_f16_r, dir, bd);
            bsr_case(static_cast<rocsparse_bfloat16*>(nullptr), rocsparse_datatype_bf16_r, dir, bd);
        }
}

TEST_F(ExtraGemmGeam, csrgeam_nnz_branches)
{
    MatDescrH descr_A, descr_B, descr_C;
    ASSERT_TRUE(descr_A.d && descr_B.d && descr_C.d);
    rocsparse_int nnz = -1;

    // A and B share the same pattern in every call: the 2 x 2 identity, or
    // m x m without entries (row pointers of zeros).
    device_vector<rocsparse_int> row{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> empty_row{std::vector<rocsparse_int>(5, 0)};
    device_vector<rocsparse_int> col{std::vector<rocsparse_int>{0, 1}};
    device_vector<rocsparse_int> row_C{size_t{5}};
    ASSERT_TRUE(row.ptr && empty_row.ptr && col.ptr && row_C.ptr);
    auto call = [&](rocsparse_handle h,
                    rocsparse_int    m,
                    rocsparse_int    nnz_in,
                    rocsparse_int*   row_out,
                    rocsparse_int*   nnz_out) {
        const rocsparse_int* p = nnz_in ? row.ptr : empty_row.ptr;
        return rocsparse_csrgeam_nnz(h,
                                     m,
                                     m,
                                     descr_A.d,
                                     nnz_in,
                                     p,
                                     col.ptr,
                                     descr_B.d,
                                     nnz_in,
                                     p,
                                     col.ptr,
                                     descr_C.d,
                                     row_out,
                                     nnz_out);
    };

    EXPECT_EQ(call(nullptr, 0, 0, row_C.ptr, &nnz), rocsparse_status_invalid_handle);
    EXPECT_EQ(call(handle, 2, 0, row_C.ptr, &nnz), rocsparse_status_success);
    EXPECT_EQ(nnz, 0);

    // Both inputs empty: row pointer of C is filled with the index base.
    ASSERT_EQ(rocsparse_set_mat_index_base(descr_C.d, rocsparse_index_base_one),
              rocsparse_status_success);
    nnz = -1;
    EXPECT_EQ(call(handle, 4, 0, row_C.ptr, &nnz), rocsparse_status_success);
    EXPECT_EQ(nnz, 0);
    for(rocsparse_int r : to_host(row_C))
        EXPECT_EQ(r, 1);

    // Device pointer mode, with and without index base one for C.
    {
        PointerModeGuard             mode(handle, rocsparse_pointer_mode_device);
        device_vector<rocsparse_int> dnnz{size_t{1}};
        ASSERT_TRUE(dnnz.ptr);
        ASSERT_EQ(hipMemset(dnnz.ptr, 0xff, sizeof(rocsparse_int)), hipSuccess);
        EXPECT_EQ(call(handle, 4, 0, row_C.ptr, dnnz.ptr), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(to_host(dnnz)[0], 0);
    }

    ASSERT_EQ(rocsparse_set_mat_type(descr_A.d, rocsparse_matrix_type_symmetric),
              rocsparse_status_success);
    EXPECT_EQ(call(handle, 2, 2, row_C.ptr, &nnz), rocsparse_status_not_implemented);
    ASSERT_EQ(rocsparse_set_mat_type(descr_A.d, rocsparse_matrix_type_general),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_storage_mode(descr_B.d, rocsparse_storage_mode_unsorted),
              rocsparse_status_success);
    EXPECT_TRUE(status_is(call(handle, 2, 2, row_C.ptr, &nnz), any_error));
    ASSERT_EQ(rocsparse_set_mat_storage_mode(descr_B.d, rocsparse_storage_mode_sorted),
              rocsparse_status_success);

    // Real addition, host pointer with base zero, then device pointer with
    // base one (index-base adjust kernel).
    ASSERT_EQ(rocsparse_set_mat_index_base(descr_C.d, rocsparse_index_base_zero),
              rocsparse_status_success);
    nnz = -1;
    ASSERT_EQ(call(handle, 2, 2, row_C.ptr, &nnz), rocsparse_status_success);
    EXPECT_EQ(nnz, 2);
    auto row_head = [&] {
        const auto h = to_host(row_C);
        return std::vector<rocsparse_int>(h.begin(), h.begin() + 3);
    };
    EXPECT_EQ(row_head(), (std::vector<rocsparse_int>{0, 1, 2}));
    ASSERT_EQ(rocsparse_set_mat_index_base(descr_C.d, rocsparse_index_base_one),
              rocsparse_status_success);
    {
        PointerModeGuard             mode(handle, rocsparse_pointer_mode_device);
        device_vector<rocsparse_int> dnnz{size_t{1}};
        ASSERT_TRUE(dnnz.ptr);
        ASSERT_EQ(call(handle, 2, 2, row_C.ptr, dnnz.ptr), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(to_host(dnnz)[0], 2);
    }
    EXPECT_EQ(row_head(), (std::vector<rocsparse_int>{1, 2, 3}));
}

TEST_F(ExtraGemmGeam, spgeam_type_index_scalars)
{
    run_spgeam<int32_t, int32_t, float>(handle, false);
    run_spgeam<int32_t, int32_t, float>(handle, true);
    run_spgeam<int32_t, int32_t, double>(handle, false);
    run_spgeam<int32_t, int32_t, rocsparse_float_complex>(handle, false);
    run_spgeam<int32_t, int32_t, rocsparse_double_complex>(handle, false);
    run_spgeam<int64_t, int32_t, float>(handle, false);
    run_spgeam<int64_t, int32_t, double>(handle, false);
    run_spgeam<int64_t, int64_t, float>(handle, false);
    run_spgeam<int64_t, int64_t, rocsparse_float_complex>(handle, false);
    run_spgeam<int64_t, int32_t, rocsparse_double_complex>(handle, true);

    // Split symbolic / numeric stages: C = 2 * A + 3 * B.
    {
        const Csr       hA = pair_operand<float>(true);
        const Csr       hB = pair_operand<float>(false);
        DevBuf<int32_t> dAp, dAc, dBp, dBc;
        DevBuf<float>   dAv, dBv;
        SpmatH          A, B;
        ASSERT_NO_FATAL_FAILURE(
            (extra_ut::make_csr_descr<float, int32_t, int32_t>(A, hA, dAp, dAc, dAv)));
        ASSERT_NO_FATAL_FAILURE(
            (extra_ut::make_csr_descr<float, int32_t, int32_t>(B, hB, dBp, dBc, dBv)));
        const Bsr want = extra_ut::bsr_geam(2.0, bsr_from_csr(hA), 3.0, bsr_from_csr(hB));
        const rocsparse_datatype dt    = rocsparse_datatype_f32_r;
        const float              alpha = 2.f, beta = 3.f;
        SpgeamDescrH             holder;
        const auto               descr = holder.d;
        ASSERT_NO_FATAL_FAILURE(spgeam_set_inputs(handle, descr, dt, dt, &alpha, &beta));

        // An invalid algorithm makes the descriptor unusable: some call fails.
        const rocsparse_spgeam_alg bad_alg = static_cast<rocsparse_spgeam_alg>(99);
        size_t                     bsz     = 0;
        const rocsparse_status     set_st  = rocsparse_spgeam_set_input(
            handle, descr, rocsparse_spgeam_input_alg, &bad_alg, sizeof(bad_alg), nullptr);
        if(set_st == rocsparse_status_success)
        {
            const rocsparse_status bs_st = rocsparse_spgeam_buffer_size(
                handle, descr, A.p, B.p, nullptr, rocsparse_spgeam_stage_analysis, &bsz, nullptr);
            device_vector<char> bad_tmp{bsz ? bsz : size_t{1}};
            if(bs_st == rocsparse_status_success)
                EXPECT_TRUE(status_is(rocsparse_spgeam(handle,
                                                       descr,
                                                       A.p,
                                                       B.p,
                                                       nullptr,
                                                       rocsparse_spgeam_stage_analysis,
                                                       bsz,
                                                       bad_tmp.ptr,
                                                       nullptr),
                                      any_error));
        }
        const rocsparse_spgeam_alg alg = rocsparse_spgeam_alg_default;
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
        auto                run = [&](rocsparse_spgeam_stage stage, rocsparse_spmat_descr C) {
            return rocsparse_spgeam(handle, descr, A.p, B.p, C, stage, bsz, tmp.ptr, nullptr);
        };
        ASSERT_EQ(run(rocsparse_spgeam_stage_symbolic_analysis, nullptr), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        int64_t nnz_C = 0;
        ASSERT_EQ(rocsparse_spgeam_get_output(
                      handle, descr, rocsparse_spgeam_output_nnz, &nnz_C, sizeof(nnz_C), nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(nnz_C, want.nnzb());
        device_vector<int32_t> cptr{size_t{3}};
        device_vector<int32_t> ccol{static_cast<size_t>(nnz_C)};
        device_vector<float>   cval{static_cast<size_t>(nnz_C)};
        SpmatH                 Cm;
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
                                             dt),
                  rocsparse_status_success);
        for(auto stage : {rocsparse_spgeam_stage_symbolic_compute,
                          rocsparse_spgeam_stage_numeric_analysis,
                          rocsparse_spgeam_stage_numeric_compute})
            ASSERT_EQ(run(stage, Cm.p), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        extra_ut::expect_bsr_equal<float, int32_t, int32_t>(
            to_host(cptr), to_host(ccol), to_host(cval), want, rocsparse_direction_row, 0);
    }

    // An unsupported format (COO) on every analysis stage.
    device_vector<int32_t> idx{std::vector<int32_t>{0, 0}};
    device_vector<float>   val{std::vector<float>{1.f}};
    ASSERT_TRUE(idx.ptr && val.ptr);
    SpmatH CA, CB, CC;
    for(SpmatH* m : {&CA, &CB, &CC})
        ASSERT_EQ(rocsparse_create_coo_descr(&m->p,
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
    const rocsparse_datatype dt    = rocsparse_datatype_f32_r;
    const float              alpha = 1.f, beta = 1.f;
    size_t                   bsz = 0;
    for(auto stage : {rocsparse_spgeam_stage_analysis,
                      rocsparse_spgeam_stage_symbolic_analysis,
                      rocsparse_spgeam_stage_numeric_analysis})
    {
        SpgeamDescrH holder;
        ASSERT_NO_FATAL_FAILURE(spgeam_set_inputs(handle, holder.d, dt, dt, &alpha, &beta));
        EXPECT_TRUE(status_is(rocsparse_spgeam_buffer_size(
                                  handle, holder.d, CA.p, CB.p, nullptr, stage, &bsz, nullptr),
                              any_error));
        const rocsparse_status st
            = rocsparse_spgeam(handle, holder.d, CA.p, CB.p, nullptr, stage, bsz, nullptr, nullptr);
        if(stage != rocsparse_spgeam_stage_numeric_analysis)
        {
            EXPECT_TRUE(status_is(st, any_error));
            continue;
        }
        // Numeric analysis may or may not look at the format; numeric
        // compute has to reject it.
        EXPECT_TRUE(status_is(rocsparse_spgeam(handle,
                                               holder.d,
                                               CA.p,
                                               CB.p,
                                               CC.p,
                                               rocsparse_spgeam_stage_numeric_compute,
                                               0,
                                               nullptr,
                                               nullptr),
                              any_error));
    }
}
