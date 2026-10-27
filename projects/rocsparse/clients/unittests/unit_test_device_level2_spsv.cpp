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
// Device unit tests of the generic triangular solve rocsparse_spsv for the
// CSC, COO and (with ROCSPARSE_WITH_ELL_TRSV) ELL storage formats; the CSR path is
// covered elsewhere.
// Every solve is compared with a host substitution on a dense copy.
//
#include "unit_test_device_level2_common.hpp"

#include <memory>
#include <set>
#include <string>
#include <tuple>

using namespace level2;

namespace
{
    enum class sp_format
    {
        csc,
        coo
#ifdef ROCSPARSE_WITH_ELL_TRSV
        ,
        ell
#endif
    };

    inline const char* format_name(sp_format f)
    {
#ifdef ROCSPARSE_WITH_ELL_TRSV
        if(f == sp_format::ell)
            return "ell";
#endif
        return f == sp_format::csc ? "csc" : "coo";
    }

    // CSC arrays of A (the CSR arrays of A^T), same index base as A.
    template <typename T>
    host_csr<T> to_csc_arrays(const host_csr<T>& a)
    {
        const int   b = a.base;
        host_csr<T> c;
        c.m    = a.m;
        c.n    = a.n;
        c.base = b;
        std::vector<int32_t> count(a.n + 1, 0);
        for(int32_t k = 0; k < a.nnz(); ++k)
            ++count[a.ind[k] - b + 1];
        for(int32_t j = 0; j < a.n; ++j)
            count[j + 1] += count[j];
        std::vector<int32_t> pos(count.begin(), count.end() - 1);
        c.ind.resize(a.nnz());
        c.val.resize(a.nnz());
        for(int32_t i = 0; i < a.m; ++i)
            for(int32_t k = a.ptr[i] - b; k < a.ptr[i + 1] - b; ++k)
            {
                const int32_t d = pos[a.ind[k] - b]++;
                c.ind[d]        = i + b;
                c.val[d]        = a.val[k];
            }
        c.ptr = count;
        for(auto& p : c.ptr)
            p += b;
        return c;
    }

    struct spsv_case
    {
        sp_format           format = sp_format::csc;
        rocsparse_fill_mode fill   = rocsparse_fill_mode_lower;
        rocsparse_diag_type diag   = rocsparse_diag_type_non_unit;
        rocsparse_operation trans  = rocsparse_operation_none;
        int                 base   = 0;
        bool                device = false;
        int32_t             m      = 20;
    };

    template <typename T>
    host_csr<T> make_spsv_matrix(const spsv_case& c)
    {
        // Both triangles and the diagonal are stored; the solve must only use
        // the requested triangle and ignore the stored diagonal when it is unit.
        host_csr<T> A = make_triangular_pattern<T>(c.m, 3, true, true, true, c.base);
        if(c.diag == rocsparse_diag_type_unit)
            for(auto& v : A.val)
                v = v * rocsparse_ut::scalar<T>(0.125f);
        return A;
    }

    // Solves for one matrix description. `Idx` is the index type of the descriptor.
    template <typename T, typename Idx>
    void run_spsv(rocsparse_handle handle, const spsv_case& c)
    {
        SCOPED_TRACE(::testing::Message()
                     << "spsv " << format_name(c.format) << " fill=" << c.fill << " diag=" << c.diag
                     << " trans=" << c.trans << " base=" << c.base << " device=" << c.device
                     << " index_bytes=" << sizeof(Idx));

        const host_csr<T>    A     = make_spsv_matrix<T>(c);
        const int32_t        m     = A.m;
        const T              alpha = rocsparse_ut::scalar<T>(2.0f);
        const std::vector<T> x     = make_vector<T>(m, 3);
        const std::vector<T> want  = host_triangular_solve(A, c.fill, c.diag, c.trans, alpha, x);

        device_vector<T> dx(x), dy(std::vector<T>(m, T{})), dalpha(std::vector<T>(1, alpha));
        ASSERT_TRUE(dx.ptr && dy.ptr && dalpha.ptr);

        spmat_guard mat;
        // The device arrays must outlive the descriptor.
        host_csr<T>                              csc;
        std::unique_ptr<device_csr<T, Idx, Idx>> dcsc;
        std::unique_ptr<device_vector<Idx>>      drow, dcol;
        std::unique_ptr<device_vector<T>>        dval;
#ifdef ROCSPARSE_WITH_ELL_TRSV
        std::unique_ptr<device_vector<Idx>> dell_col;
        std::unique_ptr<device_vector<T>>   dell_val;
#endif

        switch(c.format)
        {
        case sp_format::csc:
        {
            csc  = to_csc_arrays(A);
            dcsc = std::make_unique<device_csr<T, Idx, Idx>>(csc);
            ASSERT_TRUE(dcsc->ok());
            ASSERT_EQ(rocsparse_create_csc_descr(&mat.d,
                                                 m,
                                                 m,
                                                 csc.nnz(),
                                                 dcsc->ptr,
                                                 dcsc->ind,
                                                 dcsc->val,
                                                 rocsparse_ut::it_of<Idx>(),
                                                 rocsparse_ut::it_of<Idx>(),
                                                 to_base(c.base),
                                                 rocsparse_ut::dt_of<T>()),
                      rocsparse_status_success);
            break;
        }
        case sp_format::coo:
        {
            std::vector<Idx> rows;
            std::vector<Idx> cols;
            for(int32_t i = 0; i < m; ++i)
                for(int32_t k = A.ptr[i] - A.base; k < A.ptr[i + 1] - A.base; ++k)
                {
                    rows.push_back(static_cast<Idx>(i + A.base));
                    cols.push_back(static_cast<Idx>(A.ind[k]));
                }
            drow = std::make_unique<device_vector<Idx>>(rows);
            dcol = std::make_unique<device_vector<Idx>>(cols);
            dval = std::make_unique<device_vector<T>>(A.val);
            ASSERT_TRUE(drow->ptr && dcol->ptr && dval->ptr);
            ASSERT_EQ(rocsparse_create_coo_descr(&mat.d,
                                                 m,
                                                 m,
                                                 A.nnz(),
                                                 drow->ptr,
                                                 dcol->ptr,
                                                 dval->ptr,
                                                 rocsparse_ut::it_of<Idx>(),
                                                 to_base(c.base),
                                                 rocsparse_ut::dt_of<T>()),
                      rocsparse_status_success);
            break;
        }
#ifdef ROCSPARSE_WITH_ELL_TRSV
        case sp_format::ell:
        {
            int32_t width = 0;
            for(int32_t i = 0; i < m; ++i)
                width = std::max(width, A.ptr[i + 1] - A.ptr[i]);
            std::vector<Idx> col(static_cast<size_t>(width) * m, static_cast<Idx>(-1));
            std::vector<T>   val(static_cast<size_t>(width) * m, T{});
            for(int32_t i = 0; i < m; ++i)
            {
                int32_t j = 0;
                for(int32_t k = A.ptr[i] - A.base; k < A.ptr[i + 1] - A.base; ++k, ++j)
                {
                    col[static_cast<size_t>(j) * m + i] = static_cast<Idx>(A.ind[k]);
                    val[static_cast<size_t>(j) * m + i] = A.val[k];
                }
            }
            dell_col = std::make_unique<device_vector<Idx>>(col);
            dell_val = std::make_unique<device_vector<T>>(val);
            ASSERT_TRUE(dell_col->ptr && dell_val->ptr);
            ASSERT_EQ(rocsparse_create_ell_descr(&mat.d,
                                                 m,
                                                 m,
                                                 dell_col->ptr,
                                                 dell_val->ptr,
                                                 width,
                                                 rocsparse_ut::it_of<Idx>(),
                                                 to_base(c.base),
                                                 rocsparse_ut::dt_of<T>()),
                      rocsparse_status_success);
            break;
        }
#endif
        }

        dnvec_guard vx, vy;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, m, dx, rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, m, dy, rocsparse_ut::dt_of<T>()),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spmat_set_attribute(
                      mat.d, rocsparse_spmat_fill_mode, &c.fill, sizeof(c.fill)),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spmat_set_attribute(
                      mat.d, rocsparse_spmat_diag_type, &c.diag, sizeof(c.diag)),
                  rocsparse_status_success);

        scoped_pointer_mode mode(handle, c.device);
        const void*         palpha
            = c.device ? static_cast<const void*>(dalpha.ptr) : static_cast<const void*>(&alpha);

        auto call = [&](rocsparse_spsv_stage stage, size_t* bytes, void* buffer) {
            return rocsparse_spsv(handle,
                                  c.trans,
                                  palpha,
                                  mat.d,
                                  vx.d,
                                  vy.d,
                                  rocsparse_ut::dt_of<T>(),
                                  rocsparse_spsv_alg_default,
                                  stage,
                                  bytes,
                                  buffer);
        };

        size_t bytes = 0;
        ASSERT_EQ(call(rocsparse_spsv_stage_buffer_size, &bytes, nullptr),
                  rocsparse_status_success);
        ASSERT_GT(bytes, size_t{0});
        device_vector<char> dbuf(bytes);
        ASSERT_TRUE(dbuf.ptr);
        ASSERT_EQ(call(rocsparse_spsv_stage_preprocess, nullptr, dbuf), rocsparse_status_success);
        // A second preprocess keeps the analysis that was already done.
        ASSERT_EQ(call(rocsparse_spsv_stage_preprocess, nullptr, dbuf), rocsparse_status_success);
        ASSERT_EQ(call(rocsparse_spsv_stage_compute, nullptr, dbuf), rocsparse_status_success);
        EXPECT_TRUE(expect_near(want, to_host(dy), "spsv y"));

        // Solving again with the same analysis gives the same result.
        ASSERT_EQ(hipMemset(dy.ptr, 0, sizeof(T) * m), hipSuccess);
        ASSERT_EQ(call(rocsparse_spsv_stage_compute, nullptr, dbuf), rocsparse_status_success);
        EXPECT_TRUE(expect_near(want, to_host(dy), "spsv y (repeated solve)"));
    }

    class L2DeviceSpsv : public rocsparse_ut::HandleTest
    {
    protected:
        // Every (fill mode, diagonal, operation, base, pointer mode) combination is run:
        // the base and the pointer mode are independent loops, not derived from the
        // other parameters.
        template <typename T, typename Idx>
        void sweep(sp_format format, bool all_trans)
        {
            std::set<std::tuple<int, int, int, int, int>> seen;
            for(const auto fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
                for(const auto diag : {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit})
                    for(const auto trans : {rocsparse_operation_none,
                                            rocsparse_operation_transpose,
                                            rocsparse_operation_conjugate_transpose})
                    {
#ifdef ROCSPARSE_WITH_ELL_TRSV
                        if(format == sp_format::ell && trans != rocsparse_operation_none)
                            continue; // ELL has no transposed solve (error path below)
#endif
                        if(!all_trans && trans == rocsparse_operation_transpose)
                            continue;
                        for(const int base : {0, 1})
                            for(const bool device : {false, true})
                            {
                                seen.insert(std::make_tuple(static_cast<int>(fill),
                                                            static_cast<int>(diag),
                                                            static_cast<int>(trans),
                                                            base,
                                                            device ? 1 : 0));
                                spsv_case c;
                                c.format = format;
                                c.fill   = fill;
                                c.diag   = diag;
                                c.trans  = trans;
                                c.base   = base;
                                c.device = device;
                                run_spsv<T, Idx>(handle, c);
                                if(::testing::Test::HasFailure())
                                    return;
                            }
                    }
            size_t transes = all_trans ? 3 : 2;
#ifdef ROCSPARSE_WITH_ELL_TRSV
            if(format == sp_format::ell)
                transes = 1;
#endif
            EXPECT_EQ(seen.size(), 2 * 2 * transes * 2 * 2);
        }
    };

    TEST_F(L2DeviceSpsv, csc_double_i64_and_complex)
    {
        sweep<double, int64_t>(sp_format::csc, true);
        sweep<rocsparse_float_complex, int32_t>(sp_format::csc, false);
    }

    TEST_F(L2DeviceSpsv, coo_double_i64_and_complex)
    {
        sweep<double, int64_t>(sp_format::coo, true);
        sweep<rocsparse_double_complex, int32_t>(sp_format::coo, false);
    }

#ifdef ROCSPARSE_WITH_ELL_TRSV
    TEST_F(L2DeviceSpsv, ell_double_i64_and_complex)
    {
        sweep<double, int64_t>(sp_format::ell, false);
        sweep<rocsparse_float_complex, int32_t>(sp_format::ell, false);
    }
#endif

    TEST_F(L2DeviceSpsv, csc_transposed_solves_share_one_descriptor)
    {
        // One matrix descriptor is analysed and solved with a different operation each
        // time (none, transpose, none again, conjugate transpose): every result must
        // match the host solve of its own operation, whatever analysis the descriptor
        // held before.
        const spsv_case                      base_case;
        const auto                           A = make_spsv_matrix<double>(base_case);
        const auto                           C = to_csc_arrays(A);
        device_csr<double, int32_t, int32_t> dC(C);
        ASSERT_TRUE(dC.ok());
        const int32_t             m     = A.m;
        const double              alpha = 1.5;
        const std::vector<double> x     = make_vector<double>(m, 5);

        spmat_guard mat;
        ASSERT_EQ(rocsparse_create_csc_descr(&mat.d,
                                             m,
                                             m,
                                             C.nnz(),
                                             dC.ptr,
                                             dC.ind,
                                             dC.val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f64_r),
                  rocsparse_status_success);
        const rocsparse_fill_mode fill = rocsparse_fill_mode_lower;
        const rocsparse_diag_type diag = rocsparse_diag_type_non_unit;
        ASSERT_EQ(
            rocsparse_spmat_set_attribute(mat.d, rocsparse_spmat_fill_mode, &fill, sizeof(fill)),
            rocsparse_status_success);
        ASSERT_EQ(
            rocsparse_spmat_set_attribute(mat.d, rocsparse_spmat_diag_type, &diag, sizeof(diag)),
            rocsparse_status_success);

        for(const auto trans : {rocsparse_operation_none,
                                rocsparse_operation_transpose,
                                rocsparse_operation_none,
                                rocsparse_operation_conjugate_transpose})
        {
            device_vector<double> dx(x), dy(std::vector<double>(m, 0.0));
            dnvec_guard           vx, vy;
            ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, m, dx, rocsparse_datatype_f64_r),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, m, dy, rocsparse_datatype_f64_r),
                      rocsparse_status_success);
            size_t bytes = 0;
            auto   call  = [&](rocsparse_spsv_stage stage, void* buffer) {
                return rocsparse_spsv(handle,
                                      trans,
                                      &alpha,
                                      mat.d,
                                      vx.d,
                                      vy.d,
                                      rocsparse_datatype_f64_r,
                                      rocsparse_spsv_alg_default,
                                      stage,
                                      &bytes,
                                      buffer);
            };
            ASSERT_EQ(call(rocsparse_spsv_stage_buffer_size, nullptr), rocsparse_status_success);
            device_vector<char> dbuf(bytes);
            ASSERT_TRUE(dbuf.ptr);
            ASSERT_EQ(call(rocsparse_spsv_stage_preprocess, dbuf), rocsparse_status_success);
            ASSERT_EQ(call(rocsparse_spsv_stage_compute, dbuf), rocsparse_status_success);
            const auto want = host_triangular_solve(A, fill, diag, trans, alpha, x);
            EXPECT_TRUE(expect_near(want, to_host(dy), "spsv csc y"));
        }
    }

    // ----- error paths --------------------------------------------------------------
#ifdef ROCSPARSE_WITH_ELL_TRSV
    TEST_F(L2DeviceSpsv, ell_error_paths)
    {
        const spsv_case c;
        const auto      A     = make_spsv_matrix<float>(c);
        const int32_t   m     = A.m;
        int32_t         width = 0;
        for(int32_t i = 0; i < m; ++i)
            width = std::max(width, A.ptr[i + 1] - A.ptr[i]);
        std::vector<int32_t> col(static_cast<size_t>(width) * m, -1);
        std::vector<float>   val(static_cast<size_t>(width) * m, 0.0f);
        for(int32_t i = 0; i < m; ++i)
        {
            int32_t j = 0;
            for(int32_t k = A.ptr[i]; k < A.ptr[i + 1]; ++k, ++j)
            {
                col[static_cast<size_t>(j) * m + i] = A.ind[k];
                val[static_cast<size_t>(j) * m + i] = A.val[k];
            }
        }
        device_vector<int32_t> dcol(col);
        device_vector<float> dval(val), dx(make_vector<float>(m, 1)), dy(make_vector<float>(m, 2));
        ASSERT_TRUE(dcol.ptr && dval.ptr && dx.ptr && dy.ptr);

        spmat_guard mat;
        ASSERT_EQ(rocsparse_create_ell_descr(&mat.d,
                                             m,
                                             m,
                                             dcol,
                                             dval,
                                             width,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        dnvec_guard vx, vy;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, m, dx, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, m, dy, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        const float alpha = 1.0f;
        size_t      bytes = 0;
        auto        call
            = [&](rocsparse_operation trans, rocsparse_spmat_descr a, rocsparse_dnvec_descr px) {
                  return rocsparse_spsv(handle,
                                        trans,
                                        &alpha,
                                        a,
                                        px,
                                        vy.d,
                                        rocsparse_datatype_f32_r,
                                        rocsparse_spsv_alg_default,
                                        rocsparse_spsv_stage_buffer_size,
                                        &bytes,
                                        nullptr);
              };
        EXPECT_EQ(call(rocsparse_operation_none, mat.d, vx.d), rocsparse_status_success);
        EXPECT_EQ(call(rocsparse_operation_transpose, mat.d, vx.d),
                  rocsparse_status_not_implemented);
        EXPECT_EQ(call(rocsparse_operation_conjugate_transpose, mat.d, vx.d),
                  rocsparse_status_not_implemented);

        const rocsparse_matrix_type sym = rocsparse_matrix_type_symmetric;
        ASSERT_EQ(
            rocsparse_spmat_set_attribute(mat.d, rocsparse_spmat_matrix_type, &sym, sizeof(sym)),
            rocsparse_status_success);
        EXPECT_NE(call(rocsparse_operation_none, mat.d, vx.d), rocsparse_status_success);
        const rocsparse_matrix_type gen = rocsparse_matrix_type_general;
        ASSERT_EQ(
            rocsparse_spmat_set_attribute(mat.d, rocsparse_spmat_matrix_type, &gen, sizeof(gen)),
            rocsparse_status_success);
        const rocsparse_storage_mode unsorted = rocsparse_storage_mode_unsorted;
        ASSERT_EQ(rocsparse_spmat_set_attribute(
                      mat.d, rocsparse_spmat_storage_mode, &unsorted, sizeof(unsorted)),
                  rocsparse_status_success);
        EXPECT_NE(call(rocsparse_operation_none, mat.d, vx.d), rocsparse_status_success);
        const rocsparse_storage_mode sorted = rocsparse_storage_mode_sorted;
        ASSERT_EQ(rocsparse_spmat_set_attribute(
                      mat.d, rocsparse_spmat_storage_mode, &sorted, sizeof(sorted)),
                  rocsparse_status_success);

        // A non square ELL matrix is rejected.
        spmat_guard rect;
        ASSERT_EQ(rocsparse_create_ell_descr(&rect.d,
                                             m,
                                             m + 3,
                                             dcol,
                                             dval,
                                             width,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        EXPECT_NE(call(rocsparse_operation_none, rect.d, vx.d), rocsparse_status_success);
    }
#endif

    TEST_F(L2DeviceSpsv, argument_checks_csc)
    {
        // The argument validation does not depend on the storage format; it is
        // exercised through a CSC descriptor.
        const spsv_case                     c;
        const auto                          A = make_spsv_matrix<float>(c);
        const auto                          C = to_csc_arrays(A);
        const int32_t                       m = A.m;
        device_csr<float, int32_t, int32_t> dC(C);
        device_vector<float> dx(make_vector<float>(m, 1)), dy(make_vector<float>(m, 2));
        ASSERT_TRUE(dC.ok() && dx.ptr && dy.ptr);
        spmat_guard mat;
        ASSERT_EQ(rocsparse_create_csc_descr(&mat.d,
                                             m,
                                             m,
                                             C.nnz(),
                                             dC.ptr,
                                             dC.ind,
                                             dC.val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        dnvec_guard vx, vy;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.d, m, dx, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.d, m, dy, rocsparse_datatype_f32_r),
                  rocsparse_status_success);

        const float alpha = 1.0f;
        size_t      bytes = 0;
        const auto  f32   = rocsparse_datatype_f32_r;
        const auto  op    = rocsparse_operation_none;
        const auto  alg   = rocsparse_spsv_alg_default;
        const auto  sz    = rocsparse_spsv_stage_buffer_size;

        EXPECT_EQ(
            rocsparse_spsv(handle, op, &alpha, mat.d, vx.d, vy.d, f32, alg, sz, &bytes, nullptr),
            rocsparse_status_success);
        EXPECT_EQ(
            rocsparse_spsv(nullptr, op, &alpha, mat.d, vx.d, vy.d, f32, alg, sz, &bytes, nullptr),
            rocsparse_status_invalid_handle);
        EXPECT_EQ(
            rocsparse_spsv(handle, op, nullptr, mat.d, vx.d, vy.d, f32, alg, sz, &bytes, nullptr),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            rocsparse_spsv(handle, op, &alpha, nullptr, vx.d, vy.d, f32, alg, sz, &bytes, nullptr),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            rocsparse_spsv(handle, op, &alpha, mat.d, nullptr, vy.d, f32, alg, sz, &bytes, nullptr),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            rocsparse_spsv(handle, op, &alpha, mat.d, vx.d, nullptr, f32, alg, sz, &bytes, nullptr),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            rocsparse_spsv(handle, op, &alpha, mat.d, vx.d, vy.d, f32, alg, sz, nullptr, nullptr),
            rocsparse_status_invalid_pointer);
        EXPECT_EQ(rocsparse_spsv(handle,
                                 static_cast<rocsparse_operation>(77),
                                 &alpha,
                                 mat.d,
                                 vx.d,
                                 vy.d,
                                 f32,
                                 alg,
                                 sz,
                                 &bytes,
                                 nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_spsv(handle,
                                 op,
                                 &alpha,
                                 mat.d,
                                 vx.d,
                                 vy.d,
                                 static_cast<rocsparse_datatype>(77),
                                 alg,
                                 sz,
                                 &bytes,
                                 nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_spsv(handle,
                                 op,
                                 &alpha,
                                 mat.d,
                                 vx.d,
                                 vy.d,
                                 f32,
                                 static_cast<rocsparse_spsv_alg>(77),
                                 sz,
                                 &bytes,
                                 nullptr),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(rocsparse_spsv(handle,
                                 op,
                                 &alpha,
                                 mat.d,
                                 vx.d,
                                 vy.d,
                                 f32,
                                 alg,
                                 static_cast<rocsparse_spsv_stage>(77),
                                 &bytes,
                                 nullptr),
                  rocsparse_status_invalid_value);
    }
} // namespace
