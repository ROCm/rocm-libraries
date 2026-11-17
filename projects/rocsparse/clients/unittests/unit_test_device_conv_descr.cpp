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

// Generic-API descriptor and auxiliary entry points: const descriptor constructors and
// getters, nnz / batch accessors, dense matrix copy, status strings, color info and
// the spmat printer. Round trips compare what the public getters return with what was
// passed in; the constructors are checked with one invalid argument at a time.

#include "unit_test_device_conv_common.hpp"

#include "rocsparse_internal_spmat_print.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <map>
#include <numeric>
#include <sstream>
#include <string>

using namespace convut;

namespace
{
    using ConvDescr = ConvTest;

    // Only the pointers are stored by the const descriptors, so host arrays are enough.
    struct Buf
    {
        int32_t i[64]{};
        int32_t j[64]{};
        float   v[64]{};
    };

    const auto bad_it   = static_cast<rocsparse_indextype>(100);
    const auto bad_base = static_cast<rocsparse_index_base>(100);
    const auto bad_dt   = static_cast<rocsparse_datatype>(100);
    const auto bad_dir  = static_cast<rocsparse_direction>(100);

    TEST_F(ConvDescr, StatusNameAndDescription)
    {
        // The names are the enumerator identifiers. The wording of the descriptions may
        // change, so only check that each is non-empty and differs from an unknown one.
#define S(x)  \
    {         \
        x, #x \
    }
        const struct
        {
            rocsparse_status status;
            const char*      name;
        } entries[] = {S(rocsparse_status_success),
                       S(rocsparse_status_invalid_handle),
                       S(rocsparse_status_not_implemented),
                       S(rocsparse_status_invalid_pointer),
                       S(rocsparse_status_invalid_size),
                       S(rocsparse_status_memory_error),
                       S(rocsparse_status_internal_error),
                       S(rocsparse_status_invalid_value),
                       S(rocsparse_status_arch_mismatch),
                       S(rocsparse_status_zero_pivot),
                       S(rocsparse_status_not_initialized),
                       S(rocsparse_status_type_mismatch),
                       S(rocsparse_status_requires_sorted_storage),
                       S(rocsparse_status_thrown_exception),
                       S(rocsparse_status_continue)};
#undef S
        const auto  unknown      = static_cast<rocsparse_status>(1000);
        const char* unknown_name = rocsparse_get_status_name(unknown);
        const char* unknown_desc = rocsparse_get_status_description(unknown);
        ASSERT_NE(unknown_name, nullptr);
        ASSERT_NE(unknown_desc, nullptr);
        for(const auto& e : entries)
        {
            const char* desc = rocsparse_get_status_description(e.status);
            ASSERT_NE(desc, nullptr) << e.name;
            EXPECT_STREQ(rocsparse_get_status_name(e.status), e.name);
            EXPECT_GT(std::strlen(desc), 0u) << e.name;
            EXPECT_STRNE(desc, unknown_desc) << e.name;
            EXPECT_STRNE(rocsparse_get_status_name(e.status), unknown_name) << e.name;
        }
    }

    TEST_F(ConvDescr, ConstSpvec)
    {
        Buf        b;
        ConstSpvec d, bad;
        const auto mk = [&](rocsparse_const_spvec_descr* p,
                            int64_t                      size,
                            int64_t                      nnz,
                            const void*                  idx,
                            const void*                  val,
                            rocsparse_indextype          it   = i32,
                            rocsparse_index_base         base = base0,
                            rocsparse_datatype           dt   = f32) {
            return rocsparse_create_const_spvec_descr(p, size, nnz, idx, val, it, base, dt);
        };
        ASSERT_EQ(mk(&d, 10, 4, b.i, b.v, i32, base1), st_ok);

        int64_t              size = 0, nnz = 0;
        const void *         idx = nullptr, *val = nullptr, *vals = nullptr;
        rocsparse_indextype  it   = rocsparse_indextype_i64;
        rocsparse_index_base base = base0;
        rocsparse_datatype   dt   = rocsparse_datatype_f64_r;
        ASSERT_EQ(rocsparse_const_spvec_get(d, &size, &nnz, &idx, &val, &it, &base, &dt), st_ok);
        EXPECT_EQ(size, 10);
        EXPECT_EQ(nnz, 4);
        EXPECT_EQ(idx, b.i);
        EXPECT_EQ(val, b.v);
        EXPECT_EQ(it, i32);
        EXPECT_EQ(base, base1);
        EXPECT_EQ(dt, f32);
        ASSERT_EQ(rocsparse_const_spvec_get_values(d, &vals), st_ok);
        EXPECT_EQ(vals, b.v);

        EXPECT_EQ(mk(nullptr, 4, 2, b.i, b.v), st_ptr);
        EXPECT_EQ(mk(&bad, -1, 0, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 4, -1, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 4, 5, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 4, 2, nullptr, b.v), st_ptr);
        EXPECT_EQ(mk(&bad, 4, 2, b.i, nullptr), st_ptr);
        EXPECT_EQ(mk(&bad, 4, 2, b.i, b.v, bad_it), st_val);
        EXPECT_EQ(mk(&bad, 4, 2, b.i, b.v, i32, bad_base), st_val);
        EXPECT_EQ(mk(&bad, 4, 2, b.i, b.v, i32, base0, bad_dt), st_val);
        EXPECT_EQ(bad.get(), nullptr);
    }

    TEST_F(ConvDescr, ConstCoo)
    {
        Buf        b;
        ConstSpmat d, bad;
        const auto mk = [&](rocsparse_const_spmat_descr* p,
                            int64_t                      m,
                            int64_t                      n,
                            int64_t                      nnz,
                            const void*                  ri,
                            const void*                  ci,
                            const void*                  v,
                            rocsparse_indextype          it   = i32,
                            rocsparse_index_base         base = base0) {
            return rocsparse_create_const_coo_descr(p, m, n, nnz, ri, ci, v, it, base, f32);
        };
        ASSERT_EQ(mk(&d, 6, 7, 3, b.i, b.j, b.v, i32, base1), st_ok);

        int64_t              rows = 0, cols = 0, nnz = 0;
        const void *         ri = nullptr, *ci = nullptr, *v = nullptr;
        rocsparse_indextype  it   = rocsparse_indextype_i64;
        rocsparse_index_base base = base0;
        rocsparse_datatype   dt   = rocsparse_datatype_f64_r;
        ASSERT_EQ(rocsparse_const_coo_get(d, &rows, &cols, &nnz, &ri, &ci, &v, &it, &base, &dt),
                  st_ok);
        EXPECT_EQ(std::make_tuple(rows, cols, nnz, it, base, dt),
                  std::make_tuple(6, 7, 3, i32, base1, f32));
        EXPECT_EQ(std::make_tuple(ri, ci, v), std::make_tuple<const void*>(b.i, b.j, b.v));

        rocsparse_format fmt     = rocsparse_format_csr;
        int64_t          nnz_get = 0;
        EXPECT_EQ(rocsparse_spmat_get_format(d, &fmt), st_ok);
        EXPECT_EQ(fmt, rocsparse_format_coo);
        EXPECT_EQ(rocsparse_spmat_get_nnz(d, &nnz_get), st_ok);
        EXPECT_EQ(nnz_get, 3);

        EXPECT_EQ(mk(nullptr, 3, 3, 2, b.i, b.i, b.v), st_ptr);
        EXPECT_EQ(mk(&bad, -1, 3, 2, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 3, -1, 2, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 3, 3, -1, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 3, 3, 10, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 3, 3, 2, nullptr, b.i, b.v), st_ptr);
        EXPECT_EQ(mk(&bad, 3, 3, 2, b.i, nullptr, b.v), st_ptr);
        EXPECT_EQ(mk(&bad, 3, 3, 2, b.i, b.i, nullptr), st_ptr);
        EXPECT_EQ(mk(&bad, 3, 3, 2, b.i, b.i, b.v, bad_it), st_val);
        EXPECT_EQ(bad.get(), nullptr);
    }

    TEST_F(ConvDescr, ConstCsc)
    {
        Buf        b;
        ConstSpmat d, bad;
        const auto mk = [&](rocsparse_const_spmat_descr* p,
                            int64_t                      m,
                            int64_t                      n,
                            int64_t                      nnz,
                            const void*                  cp,
                            const void*                  ri,
                            const void*                  v) {
            return rocsparse_create_const_csc_descr(p, m, n, nnz, cp, ri, v, i32, i32, base0, f32);
        };
        ASSERT_EQ(mk(&d, 5, 4, 3, b.i, b.j, b.v), st_ok);

        int64_t              rows = 0, cols = 0, nnz = 0;
        const void *         cp = nullptr, *ri = nullptr, *v = nullptr;
        rocsparse_indextype  cpt = rocsparse_indextype_i64, rit = rocsparse_indextype_i64;
        rocsparse_index_base base = base1;
        rocsparse_datatype   dt   = rocsparse_datatype_f64_r;
        ASSERT_EQ(
            rocsparse_const_csc_get(d, &rows, &cols, &nnz, &cp, &ri, &v, &cpt, &rit, &base, &dt),
            st_ok);
        EXPECT_EQ(std::make_tuple(rows, cols, nnz, cpt, rit, base, dt),
                  std::make_tuple(5, 4, 3, i32, i32, base0, f32));
        EXPECT_EQ(std::make_tuple(cp, ri, v), std::make_tuple<const void*>(b.i, b.j, b.v));
        rocsparse_format fmt = rocsparse_format_csr;
        EXPECT_EQ(rocsparse_spmat_get_format(d, &fmt), st_ok);
        EXPECT_EQ(fmt, rocsparse_format_csc);

        EXPECT_EQ(mk(nullptr, 3, 3, 2, b.i, b.i, b.v), st_ptr);
        EXPECT_EQ(mk(&bad, -1, 3, 2, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 3, -1, 2, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 3, 3, -1, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 3, 3, 10, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 3, 3, 2, nullptr, b.i, b.v), st_ptr);
        EXPECT_EQ(mk(&bad, 3, 3, 2, b.i, nullptr, b.v), st_ptr);
        EXPECT_EQ(mk(&bad, 3, 3, 2, b.i, b.i, nullptr), st_ptr);
        EXPECT_EQ(bad.get(), nullptr);
    }

    TEST_F(ConvDescr, ConstBsr)
    {
        Buf        b;
        ConstSpmat d, bad;
        const auto mk = [&](rocsparse_const_spmat_descr* p,
                            int64_t                      mb,
                            int64_t                      nb,
                            int64_t                      nnzb,
                            rocsparse_direction          dir,
                            int64_t                      bd,
                            const void*                  rp,
                            const void*                  ci,
                            const void*                  v,
                            rocsparse_index_base         base = base0) {
            return rocsparse_create_const_bsr_descr(
                p, mb, nb, nnzb, dir, bd, rp, ci, v, i32, i32, base, f32);
        };
        ASSERT_EQ(mk(&d, 3, 4, 5, dir_c, 2, b.i, b.j, b.v, base1), st_ok);

        int64_t              mb = 0, nb = 0, nnzb = 0, bd = 0;
        rocsparse_direction  dir = dir_r;
        const void *         rp = nullptr, *ci = nullptr, *v = nullptr;
        rocsparse_indextype  rpt = rocsparse_indextype_i64, cit = rocsparse_indextype_i64;
        rocsparse_index_base base = base0;
        rocsparse_datatype   dt   = rocsparse_datatype_f64_r;
        ASSERT_EQ(rocsparse_const_bsr_get(
                      d, &mb, &nb, &nnzb, &dir, &bd, &rp, &ci, &v, &rpt, &cit, &base, &dt),
                  st_ok);
        EXPECT_EQ(std::make_tuple(mb, nb, nnzb, dir, bd, rpt, cit, base, dt),
                  std::make_tuple(3, 4, 5, dir_c, 2, i32, i32, base1, f32));
        EXPECT_EQ(std::make_tuple(rp, ci, v), std::make_tuple<const void*>(b.i, b.j, b.v));

        // rocsparse_spmat_get_nnz returns the length of the value array: nnzb * block_dim^2.
        int64_t nnz = 0;
        EXPECT_EQ(rocsparse_spmat_get_nnz(d, &nnz), st_ok);
        EXPECT_EQ(nnz, 5 * 2 * 2);

        EXPECT_EQ(mk(nullptr, 2, 2, 2, dir_r, 2, b.i, b.i, b.v), st_ptr);
        EXPECT_EQ(mk(&bad, -1, 2, 2, dir_r, 2, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 2, -1, 2, dir_r, 2, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 2, 2, -1, dir_r, 2, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 2, 2, 5, dir_r, 2, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 2, 2, 2, bad_dir, 2, b.i, b.i, b.v), st_val);
        EXPECT_EQ(mk(&bad, 2, 2, 2, dir_r, -1, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 2, 2, 2, dir_r, 0, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 2, 2, 2, dir_r, 2, nullptr, b.i, b.v), st_ptr);
        EXPECT_EQ(mk(&bad, 2, 2, 2, dir_r, 2, b.i, nullptr, b.v), st_ptr);
        EXPECT_EQ(mk(&bad, 2, 2, 2, dir_r, 2, b.i, b.i, nullptr), st_ptr);
        EXPECT_EQ(bad.get(), nullptr);
    }

    TEST_F(ConvDescr, ConstBell)
    {
        Buf        b;
        ConstSpmat d, bad;
        const auto mk = [&](rocsparse_const_spmat_descr* p,
                            int64_t                      m,
                            int64_t                      n,
                            rocsparse_direction          dir,
                            int64_t                      bd,
                            int64_t                      ell_cols,
                            const void*                  ci,
                            const void*                  v) {
            return rocsparse_create_const_bell_descr(
                p, m, n, dir, bd, ell_cols, ci, v, i32, base0, f32);
        };
        ASSERT_EQ(mk(&d, 6, 8, dir_r, 2, 4, b.i, b.v), st_ok);

        int64_t              rows = 0, cols = 0, bd = 0, ell_cols = 0;
        rocsparse_direction  dir = dir_c;
        const void *         ci = nullptr, *v = nullptr;
        rocsparse_indextype  it   = rocsparse_indextype_i64;
        rocsparse_index_base base = base1;
        rocsparse_datatype   dt   = rocsparse_datatype_f64_r;
        ASSERT_EQ(rocsparse_const_bell_get(
                      d, &rows, &cols, &dir, &bd, &ell_cols, &ci, &v, &it, &base, &dt),
                  st_ok);
        EXPECT_EQ(std::make_tuple(rows, cols, dir, bd, ell_cols, it, base, dt),
                  std::make_tuple(6, 8, dir_r, 2, 4, i32, base0, f32));
        EXPECT_EQ(std::make_tuple(ci, v), std::make_tuple<const void*>(b.i, b.v));

        // The header does not pin the nnz of a blocked ELL matrix; the value array holds
        // rows * ell_cols entries, so require at least that.
        int64_t nnz = 0;
        EXPECT_EQ(rocsparse_spmat_get_nnz(d, &nnz), st_ok);
        EXPECT_GE(nnz, rows * ell_cols);
        // The nnz of a blocked ELL matrix cannot be set (the header only says it fails).
        EXPECT_REJECTED(rocsparse_spmat_set_nnz(mut(d), 3));

        EXPECT_EQ(mk(nullptr, 4, 4, dir_r, 2, 2, b.i, b.v), st_ptr);
        EXPECT_EQ(mk(&bad, -1, 4, dir_r, 2, 2, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 4, -1, dir_r, 2, 2, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 4, 4, bad_dir, 2, 2, b.i, b.v), st_val);
        // The header lists only rows and cols for invalid_size.
        EXPECT_REJECTED(mk(&bad, 4, 4, dir_r, -1, 2, b.i, b.v));
        EXPECT_REJECTED(mk(&bad, 4, 4, dir_r, 0, 2, b.i, b.v));
        EXPECT_REJECTED(mk(&bad, 4, 4, dir_r, 2, -1, b.i, b.v));
        EXPECT_REJECTED(mk(&bad, 4, 4, dir_r, 2, 5, b.i, b.v));
        EXPECT_EQ(mk(&bad, 4, 4, dir_r, 2, 2, nullptr, b.v), st_ptr);
        EXPECT_EQ(mk(&bad, 4, 4, dir_r, 2, 2, b.i, nullptr), st_ptr);
        EXPECT_EQ(bad.get(), nullptr);
    }

    TEST_F(ConvDescr, ConstSell)
    {
        Buf        b;
        ConstSpmat d, bad;
        const auto mk = [&](rocsparse_const_spmat_descr* p,
                            int64_t                      m,
                            int64_t                      n,
                            int64_t                      nnz,
                            int64_t                      slice,
                            int64_t                      colval,
                            const void*                  off,
                            const void*                  ci,
                            const void*                  v) {
            return rocsparse_create_const_sell_descr(
                p, m, n, nnz, slice, colval, off, ci, v, i32, i32, base0, f32);
        };
        ASSERT_EQ(rocsparse_create_const_sell_descr(
                      &d, 8, 9, 10, 4, 16, b.i, b.j, b.v, i32, i32, base1, f32),
                  st_ok);

        int64_t              rows = 0, cols = 0, nnz = 0, slice = 0, colval = 0;
        const void *         so = nullptr, *ci = nullptr, *v = nullptr;
        rocsparse_indextype  sot = rocsparse_indextype_i64, cit = rocsparse_indextype_i64;
        rocsparse_index_base base = base0;
        rocsparse_datatype   dt   = rocsparse_datatype_f64_r;
        ASSERT_EQ(rocsparse_const_sell_get(
                      d, &rows, &cols, &nnz, &slice, &colval, &so, &ci, &v, &sot, &cit, &base, &dt),
                  st_ok);
        EXPECT_EQ(std::make_tuple(rows, cols, nnz, slice, colval, sot, cit, base, dt),
                  std::make_tuple(8, 9, 10, 4, 16, i32, i32, base1, f32));
        EXPECT_EQ(std::make_tuple(so, ci, v), std::make_tuple<const void*>(b.i, b.j, b.v));

        // Every pointer argument of the getter is checked.
        expect_null_args(rocsparse_const_sell_get,
                         d.get(),
                         &rows,
                         &cols,
                         &nnz,
                         &slice,
                         &colval,
                         &so,
                         &ci,
                         &v,
                         &sot,
                         &cit,
                         &base,
                         &dt);

        // The nnz of a SELL matrix can be queried and updated.
        int64_t got = 0;
        EXPECT_EQ(rocsparse_spmat_get_nnz(d, &got), st_ok);
        EXPECT_EQ(got, 10);
        EXPECT_EQ(rocsparse_spmat_set_nnz(mut(d), 7), st_ok);
        EXPECT_EQ(rocsparse_spmat_get_nnz(d, &got), st_ok);
        EXPECT_EQ(got, 7);

        EXPECT_EQ(mk(nullptr, 8, 8, 4, 4, 8, b.i, b.i, b.v), st_ptr);
        EXPECT_EQ(mk(&bad, -1, 8, 4, 4, 8, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 8, -1, 4, 4, 8, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 8, 8, -1, 4, 8, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 8, 8, 4, -1, 8, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 8, 8, 4, 0, 8, b.i, b.i, b.v), st_size);
        EXPECT_EQ(mk(&bad, 8, 8, 4, 4, -1, b.i, b.i, b.v), st_size);
        // nnz > sell_colval_size and sell_slice_size > rows are not described as invalid.
        EXPECT_REJECTED(mk(&bad, 8, 8, 9, 4, 8, b.i, b.i, b.v));
        EXPECT_REJECTED(mk(&bad, 8, 8, 4, 9, 8, b.i, b.i, b.v));
        EXPECT_EQ(mk(&bad, 8, 8, 4, 4, 8, nullptr, b.i, b.v), st_ptr);
        EXPECT_EQ(mk(&bad, 8, 8, 4, 4, 8, b.i, nullptr, b.v), st_ptr);
        EXPECT_EQ(mk(&bad, 8, 8, 4, 4, 8, b.i, b.i, nullptr), st_ptr);
        EXPECT_EQ(bad.get(), nullptr);
    }
}

namespace
{
    TEST_F(ConvDescr, BellSetPointers)
    {
        Buf     b;
        int32_t other_idx[64]{};
        float   other_val[64]{};
        Spmat   d;
        // 4 rows, block_dim 2 -> 2 block rows, ell_cols 4 -> 2 blocks per block row.
        ASSERT_EQ(rocsparse_create_bell_descr(&d, 4, 6, dir_r, 2, 4, b.i, b.v, i32, base0, f32),
                  st_ok);

        expect_null_args(rocsparse_bell_set_pointers, d.get(), other_idx, other_val);
        ASSERT_EQ(rocsparse_bell_set_pointers(d, other_idx, other_val), st_ok);

        int64_t              rows = 0, cols = 0, bd = 0, ell_cols = 0;
        rocsparse_direction  dir = dir_c;
        void *               ci = nullptr, *v = nullptr;
        rocsparse_indextype  it   = rocsparse_indextype_i64;
        rocsparse_index_base base = base1;
        rocsparse_datatype   dt   = rocsparse_datatype_f64_r;
        ASSERT_EQ(
            rocsparse_bell_get(d, &rows, &cols, &dir, &bd, &ell_cols, &ci, &v, &it, &base, &dt),
            st_ok);
        EXPECT_EQ(ci, other_idx);
        EXPECT_EQ(v, other_val);
        EXPECT_EQ(std::make_tuple(rows, cols, bd, ell_cols), std::make_tuple(4, 6, 2, 4));
    }

    TEST_F(ConvDescr, EllNnzAndStridedBatch)
    {
        Buf   b;
        Spmat d;
        ASSERT_EQ(rocsparse_create_ell_descr(&d, 5, 7, b.i, b.v, 3, i32, base0, f32), st_ok);
        // ELL: nnz is ell_width * rows and cannot be set (the header only says it fails).
        int64_t nnz = 0;
        EXPECT_EQ(rocsparse_spmat_get_nnz(d, &nnz), st_ok);
        EXPECT_EQ(nnz, 3 * 5);
        EXPECT_REJECTED(rocsparse_spmat_set_nnz(d, 4));

        EXPECT_EQ(rocsparse_ell_set_strided_batch(nullptr, 2, 10), st_ptr);
        // The header documents invalid_size for an invalid batch count or stride, while
        // the implementation returns invalid_value; accept both (AISPARSE-904).
        EXPECT_STATUS_IN(rocsparse_ell_set_strided_batch(d, 0, 10), st_size, st_val);
        EXPECT_STATUS_IN(rocsparse_ell_set_strided_batch(d, 2, -1), st_size, st_val);
        // A positive stride with a batch count of one is not covered by the header; the
        // library accepts it.
        EXPECT_EQ(rocsparse_ell_set_strided_batch(d, 1, 5), st_ok);
        ASSERT_EQ(rocsparse_ell_set_strided_batch(d, 3, 20), st_ok);
        rocsparse_int batch_count = 0;
        EXPECT_EQ(rocsparse_spmat_get_strided_batch(d, &batch_count), st_ok);
        EXPECT_EQ(batch_count, 3);
    }

    TEST_F(ConvDescr, SpmatSetNnzPerFormat)
    {
        Buf   b;
        Spmat csr, csc, coo, bsr, aos;
        ASSERT_EQ(rocsparse_create_csr_descr(&csr, 4, 4, 8, b.i, b.j, b.v, i32, i32, base0, f32),
                  st_ok);
        ASSERT_EQ(rocsparse_create_csc_descr(&csc, 4, 4, 8, b.i, b.j, b.v, i32, i32, base0, f32),
                  st_ok);
        ASSERT_EQ(rocsparse_create_coo_descr(&coo, 4, 4, 8, b.i, b.j, b.v, i32, base0, f32), st_ok);
        ASSERT_EQ(rocsparse_create_coo_aos_descr(&aos, 4, 4, 8, b.i, b.v, i32, base0, f32), st_ok);
        ASSERT_EQ(rocsparse_create_bsr_descr(
                      &bsr, 2, 2, 4, dir_r, 3, b.i, b.j, b.v, i32, i32, base0, f32),
                  st_ok);

        // As documented, set_nnz takes the number of blocks for BSR, and get_nnz returns
        // the length of the value array (blocks * block_dim^2).
        const std::pair<rocsparse_spmat_descr, int64_t> cases[]
            = {{csr, 1}, {csc, 1}, {coo, 1}, {aos, 1}, {bsr, 3 * 3}};
        for(const auto& c : cases)
        {
            ASSERT_EQ(rocsparse_spmat_set_nnz(c.first, 3), st_ok);
            int64_t nnz = 0;
            ASSERT_EQ(rocsparse_spmat_get_nnz(c.first, &nnz), st_ok);
            EXPECT_EQ(nnz, 3 * c.second);
            EXPECT_EQ(rocsparse_spmat_set_nnz(c.first, -1), st_size);
        }
        int64_t out = 0;
        EXPECT_EQ(rocsparse_spmat_set_nnz(nullptr, 3), st_ptr);
        expect_null_args(rocsparse_spmat_get_nnz, csr.get(), &out);
    }

#ifdef ROCSPARSE_WITH_SPMAT_SCALE
    TEST_F(ConvDescr, DnvecScalar)
    {
        float      alpha = 2.5f, other = 1.0f;
        const auto mk = [&](rocsparse_handle       h,
                            rocsparse_dnvec_descr* p,
                            rocsparse_pointer_mode mode,
                            rocsparse_datatype     dt,
                            const float*           cv,
                            float*                 v) {
            return rocsparse_dnvec_descr_create_scalar(h, p, mode, dt, cv, v, nullptr);
        };
        int64_t            size = 0;
        const void*        vals = nullptr;
        void*              rw   = nullptr;
        rocsparse_datatype dt   = rocsparse_datatype_f64_r;
        {
            // Read-only descriptor.
            Dnvec d;
            ASSERT_EQ(mk(handle, &d, rocsparse_pointer_mode_host, f32, &alpha, nullptr), st_ok);
            ASSERT_EQ(rocsparse_const_dnvec_get(d, &size, &vals, &dt), st_ok);
            EXPECT_EQ(size, 1);
            EXPECT_EQ(vals, &alpha);
            EXPECT_EQ(dt, f32);
        }
        {
            // Read-write descriptor: the values alias the const values.
            Dnvec d;
            ASSERT_EQ(mk(handle, &d, rocsparse_pointer_mode_device, f32, &alpha, &alpha), st_ok);
            ASSERT_EQ(rocsparse_dnvec_get(d, &size, &rw, &dt), st_ok);
            EXPECT_EQ(size, 1);
            EXPECT_EQ(rw, &alpha);
        }
        Dnvec          bad;
        constexpr auto host = rocsparse_pointer_mode_host;
        EXPECT_EQ(mk(nullptr, &bad, host, f32, &alpha, nullptr), rocsparse_status_invalid_handle);
        EXPECT_EQ(mk(handle, nullptr, host, f32, &alpha, nullptr), st_ptr);
        EXPECT_EQ(mk(handle, &bad, static_cast<rocsparse_pointer_mode>(100), f32, &alpha, nullptr),
                  st_val);
        EXPECT_EQ(mk(handle, &bad, host, bad_dt, &alpha, nullptr), st_val);
        EXPECT_EQ(mk(handle, &bad, host, f32, nullptr, nullptr), st_ptr);
        // values must equal const_values; the status of a mismatch is not documented.
        EXPECT_REJECTED(mk(handle, &bad, host, f32, &alpha, &other));
        EXPECT_EQ(bad.get(), nullptr);
    }
#endif

    TEST_F(ConvDescr, ConstDnmatRowOrderLd)
    {
        float      v[64]{};
        ConstDnmat d, bad;
        // Row order: ld must be at least cols.
        EXPECT_EQ(rocsparse_create_const_dnmat_descr(&bad, 3, 5, 4, v, f32, rocsparse_order_row),
                  st_size);
        EXPECT_EQ(bad.get(), nullptr);
        ASSERT_EQ(rocsparse_create_const_dnmat_descr(&d, 3, 5, 6, v, f32, rocsparse_order_row),
                  st_ok);
        int64_t            rows = 0, cols = 0, ld = 0;
        const void*        vals = nullptr;
        rocsparse_datatype dt   = rocsparse_datatype_f64_r;
        rocsparse_order    ord  = rocsparse_order_column;
        ASSERT_EQ(rocsparse_const_dnmat_get(d, &rows, &cols, &ld, &vals, &dt, &ord), st_ok);
        EXPECT_EQ(std::make_tuple(rows, cols, ld, dt, ord),
                  std::make_tuple(3, 5, 6, f32, rocsparse_order_row));
        EXPECT_EQ(vals, v);
    }

    TEST_F(ConvDescr, CopyColorInfo)
    {
        // The color info structure is empty and the copy only validates its pointers.
        ColorInfo a, b;
        ASSERT_EQ(rocsparse_create_color_info(&a), st_ok);
        ASSERT_EQ(rocsparse_create_color_info(&b), st_ok);
        EXPECT_EQ(rocsparse_copy_color_info(nullptr, a), st_ptr);
        EXPECT_EQ(rocsparse_copy_color_info(a, nullptr), st_ptr);
        EXPECT_REJECTED(rocsparse_copy_color_info(a, a));
        EXPECT_EQ(rocsparse_copy_color_info(a, b), st_ok);
    }
}

// rocsparse_dnmat_copy_data (only declared with ROCSPARSE_WITH_TRSM_REFACTORING):
// contiguous, padded and batched layouts, scaled copies through the kernel path in both
// pointer modes (only with ROCSPARSE_WITH_SPMAT_SCALE, which provides
// rocsparse_dnvec_descr_create_scalar) and argument validation. Every logical and
// padding element of the target is checked.
#ifdef ROCSPARSE_WITH_TRSM_REFACTORING
namespace
{
    struct Layout
    {
        rocsparse_order order;
        int64_t         ld;
        int             batch;
        int64_t         stride;
    };

    enum class Alpha
    {
        none,
        host,
        device,
        device_batched
    };

    constexpr int64_t M = 5, N = 4;
    constexpr int     B   = 3;
    constexpr auto    col = rocsparse_order_column;
    constexpr auto    row = rocsparse_order_row;

    struct CopyCase
    {
        Layout src, dst;
        Alpha  alpha;
    };

    // clang-format off
    const CopyCase s_copy_cases[] = {
        // Single matrix, packed and padded; contiguous batches, also with a padded
        // target stride; interleaved batches.
        {{col, M, 1, 0}, {col, M, 1, 0}, Alpha::none},
        {{col, M + 3, 1, 0}, {col, M + 2, 1, 0}, Alpha::none},
        {{row, N, 1, 0}, {row, N + 1, 1, 0}, Alpha::none},
        {{col, M, B, M * N}, {col, M, B, M * N}, Alpha::none},
        {{col, M, B, M * N}, {col, M, B, M * N + 5}, Alpha::none},
        {{row, N, B, M * N}, {row, N, B, M * N}, Alpha::none},
        {{col, M * B, B, M}, {col, M * B, B, M}, Alpha::none},
        {{col, M * B + 2, B, M}, {col, M * B + 2, B, M}, Alpha::none},
        // Different batch layouts, a source matrix broadcast to every batch, and
        // different orders need the kernel.
        {{col, M, B, M * N}, {col, M * B, B, M}, Alpha::none},
        {{col, M, 1, 0}, {col, M, B, M * N}, Alpha::none},
        {{col, M, 1, 0}, {row, N, 1, 0}, Alpha::none},
        // Scaled copies.
        {{col, M, 1, 0}, {col, M, 1, 0}, Alpha::host},
        {{col, M + 1, 1, 0}, {col, M, 1, 0}, Alpha::device},
        {{row, N, 1, 0}, {row, N + 2, 1, 0}, Alpha::host},
        {{col, M, B, M * N}, {col, M, B, M * N}, Alpha::device},
        {{col, M, B, M * N}, {col, M, B, M * N}, Alpha::device_batched},
        {{col, M, 1, 0}, {col, M, B, M * N}, Alpha::host},
        {{row, N, 1, 0}, {col, M, 1, 0}, Alpha::host},
    };
    // clang-format on

    size_t layout_size(const Layout& l)
    {
        return static_cast<size_t>((l.batch - 1) * l.stride + l.ld * (l.order == col ? N : M));
    }
    size_t layout_index(const Layout& l, int b, int64_t i, int64_t j)
    {
        return static_cast<size_t>(b * l.stride + (l.order == col ? i + l.ld * j : i * l.ld + j));
    }

    template <typename T>
    void run_copy(rocsparse_handle handle, const CopyCase& c)
    {
        const Layout &src = c.src, &dst = c.dst;
#ifdef ROCSPARSE_WITH_SPMAT_SCALE
        const Alpha kind = c.alpha;
#else
        const Alpha kind = Alpha::none;
#endif
        // Source values depend on the logical position only.
        std::vector<T> hsrc(layout_size(src), make_val<T>(-7, -3));
        for(int b = 0; b < src.batch; ++b)
            for(int64_t i = 0; i < M; ++i)
                for(int64_t j = 0; j < N; ++j)
                    hsrc[layout_index(src, b, i, j)] = make_val<T>(1 + b * 100 + i * 10 + j, j - i);

        const T        sentinel = make_val<T>(-99, 99);
        std::vector<T> halpha;
        for(int b = 0; b < (kind == Alpha::device_batched ? dst.batch : kind != Alpha::none); ++b)
            halpha.push_back(make_val<T>(2 + b, 1));

        DevBuf<T> dsrc(hsrc), ddst(std::vector<T>(layout_size(dst), sentinel)), dalpha(halpha);
        Dnmat     dm_src, dm_dst;
        ASSERT_EQ(rocsparse_create_dnmat_descr(&dm_src, M, N, src.ld, dsrc, dt_of<T>(), src.order),
                  st_ok);
        ASSERT_EQ(rocsparse_create_dnmat_descr(&dm_dst, M, N, dst.ld, ddst, dt_of<T>(), dst.order),
                  st_ok);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(dm_src, src.batch, src.stride), st_ok);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(dm_dst, dst.batch, dst.stride), st_ok);

        Dnvec alpha;
#ifdef ROCSPARSE_WITH_SPMAT_SCALE
        if(kind != Alpha::none)
        {
            const bool  host = kind == Alpha::host;
            T*          dev  = dalpha;
            const void* ptr  = host ? static_cast<const void*>(halpha.data()) : dev;
            ASSERT_EQ(rocsparse_dnvec_descr_create_scalar(handle,
                                                          &alpha,
                                                          host ? rocsparse_pointer_mode_host
                                                               : rocsparse_pointer_mode_device,
                                                          dt_of<T>(),
                                                          ptr,
                                                          nullptr,
                                                          nullptr),
                      st_ok);
            if(kind == Alpha::device_batched)
            {
                ASSERT_EQ(rocsparse_dnvec_set_strided_batch(alpha, dst.batch, 1), st_ok);
            }
        }
#endif
        ASSERT_EQ(rocsparse_dnmat_copy_data(handle, alpha, dm_src, dm_dst, nullptr), st_ok);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        std::vector<T> expected(layout_size(dst), sentinel);
        for(int b = 0; b < dst.batch; ++b)
            for(int64_t i = 0; i < M; ++i)
                for(int64_t j = 0; j < N; ++j)
                    expected[layout_index(dst, b, i, j)]
                        = (halpha.empty() ? make_val<T>(1)
                                          : halpha[kind == Alpha::device_batched ? b : 0])
                          * hsrc[layout_index(src, src.batch == 1 ? 0 : b, i, j)];
        EXPECT_EQ(ddst.read(), expected);
    }

    template <typename T>
    void run_all_copies(rocsparse_handle handle)
    {
        for(const CopyCase& c : s_copy_cases)
        {
            run_copy<T>(handle, c);
        }
    }

    TEST_F(ConvDescr, DnmatCopyFloat)
    {
        run_all_copies<float>(handle);
    }
    TEST_F(ConvDescr, DnmatCopyDouble)
    {
        run_all_copies<double>(handle);
    }
    TEST_F(ConvDescr, DnmatCopyFloatComplex)
    {
        run_all_copies<rocsparse_float_complex>(handle);
    }
    TEST_F(ConvDescr, DnmatCopyDoubleComplex)
    {
        run_all_copies<rocsparse_double_complex>(handle);
    }

    TEST_F(ConvDescr, DnmatCopyInvalid)
    {
        DevBuf<float>  df(std::vector<float>(64, 1.0f)), df2(std::vector<float>(64, 1.0f));
        DevBuf<double> dd(std::vector<double>(64, 1.0));
        Dnmat          src, dst, wide, tall, dst_double;
        const auto     mk = [&](Dnmat& d, int64_t m, int64_t n, void* p, rocsparse_datatype dt) {
            ASSERT_EQ(rocsparse_create_dnmat_descr(&d, m, n, m, p, dt, col), st_ok);
        };
        mk(src, 4, 3, df, f32);
        mk(dst, 4, 3, df2, f32);
        mk(wide, 4, 4, df2, f32);
        mk(tall, 5, 3, df2, f32);
        mk(dst_double, 4, 3, dd, rocsparse_datatype_f64_r);

        const auto copy
            = [&](rocsparse_dnvec_descr a, rocsparse_dnmat_descr s, rocsparse_dnmat_descr d) {
                  return rocsparse_dnmat_copy_data(handle, a, s, d, nullptr);
              };
        // (invalid_handle is not in the list of documented statuses)
        EXPECT_REJECTED(rocsparse_dnmat_copy_data(nullptr, nullptr, src, dst, nullptr));
        EXPECT_EQ(copy(nullptr, nullptr, dst), st_ptr);
        EXPECT_EQ(copy(nullptr, src, nullptr), st_ptr);
        // The header documents invalid_value for different data types while the
        // implementation returns not_implemented; accept both (AISPARSE-904).
        EXPECT_STATUS_IN(copy(nullptr, src, dst_double), st_val, rocsparse_status_not_implemented);
        EXPECT_EQ(copy(nullptr, src, tall), st_val);
        EXPECT_EQ(copy(nullptr, src, wide), st_val);
        // Different batch counts.
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(src, 2, 12), st_ok);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(dst, 3, 12), st_ok);
        EXPECT_EQ(copy(nullptr, src, dst), st_val);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(src, 1, 0), st_ok);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(dst, 1, 0), st_ok);

#ifdef ROCSPARSE_WITH_SPMAT_SCALE
        const auto scalar = [&](Dnvec& a, rocsparse_datatype dt, const void* p) {
            ASSERT_EQ(rocsparse_dnvec_descr_create_scalar(
                          handle, &a, rocsparse_pointer_mode_host, dt, p, nullptr, nullptr),
                      st_ok);
        };
        // The scalar must have the data type of the matrices.
        double alpha_d = 2.0;
        Dnvec  alpha_double;
        scalar(alpha_double, rocsparse_datatype_f64_r, &alpha_d);
        EXPECT_STATUS_IN(copy(alpha_double, src, dst), st_val, rocsparse_status_not_implemented);
        // The scalar batch count must match the target or be one with a zero stride.
        float alpha_f[3] = {2.0f, 3.0f, 4.0f};
        Dnvec alpha;
        scalar(alpha, f32, alpha_f);
        ASSERT_EQ(rocsparse_dnvec_set_strided_batch(alpha, 2, 1), st_ok);
        EXPECT_EQ(copy(alpha, src, dst), st_val);
#endif
    }
} // namespace
#endif // ROCSPARSE_WITH_TRSM_REFACTORING

// rocsparse::internal_spmat_print writes a description of a sparse matrix descriptor
// (header fields, then optionally the index arrays and the values) to a std::ostream.
// Nothing in the library calls it and its symbol is hidden in librocsparse, so the
// defining translation unit is compiled into the device unit test binary; it is not
// reachable through the public API. The output is not a documented format, so the
// tests check the content that follows from the descriptor (sizes, nnz, the printed
// arrays) and not the spacing of the header fields.
namespace
{
    struct Printed
    {
        rocsparse_status                   status = st_ok;
        std::map<std::string, std::string> fields; // "- key : value" lines
        std::vector<std::string>           body; // every other line
    };

    std::string trim(const std::string& s)
    {
        const auto b = s.find_first_not_of(" \t\r\n");
        return b == std::string::npos ? "" : s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
    }

    Printed print(rocsparse_const_spmat_descr d, bool symbolic, bool numeric)
    {
        Printed            p;
        std::ostringstream os;
        p.status = rocsparse::internal_spmat_print(os, d, symbolic, numeric);
        std::istringstream is(os.str());
        for(std::string line; std::getline(is, line);)
        {
            const size_t colon = line.find(':');
            if(line.compare(0, 2, "- ") != 0)
            {
                p.body.push_back(line);
            }
            else if(colon == std::string::npos)
            {
                ADD_FAILURE() << "header line without a colon: " << line;
            }
            else
            {
                p.fields[trim(line.substr(2, colon - 2))] = trim(line.substr(colon + 1));
            }
        }
        return p;
    }

    // "[i] = v" lines of one printed array.
    template <typename T>
    std::vector<std::string> lines(const std::vector<T>& v)
    {
        std::vector<std::string> out;
        for(size_t i = 0; i < v.size(); ++i)
        {
            std::ostringstream os;
            os << "[" << i << "] = " << v[i];
            out.push_back(os.str());
        }
        return out;
    }

    // Rows of a printed column-major m x w matrix with leading dimension m.
    template <typename T>
    std::vector<std::string> matrix_rows(const std::vector<T>& v, int m, int w)
    {
        std::vector<std::string> out(m);
        for(int i = 0; i < m; ++i)
        {
            std::ostringstream os;
            for(int j = 0; j < w; ++j)
                os << " " << v[j * m + i];
            out[i] = os.str();
        }
        return out;
    }

    std::vector<std::string> cat(std::vector<std::string> a, const std::vector<std::string>& b)
    {
        a.insert(a.end(), b.begin(), b.end());
        return a;
    }

    template <typename T>
    std::vector<T> shifted(std::vector<T> v, int base)
    {
        for(T& x : v)
            x += base;
        return v;
    }

    // Prints the descriptor with every combination of the symbolic / numeric flags and
    // compares the header fields and the body. An expected field value starting with '~'
    // only has to be contained (case insensitive) in the printed one.
    void check_print(rocsparse_const_spmat_descr               d,
                     const std::map<std::string, std::string>& fields,
                     const std::vector<std::string>&           sym,
                     const std::vector<std::string>&           num)
    {
        for(int flags = 0; flags < 4; ++flags)
        {
            const bool s = flags & 1, n = flags & 2;
            SCOPED_TRACE("symbolic " + std::to_string(s) + " numeric " + std::to_string(n));
            const Printed p = print(d, s, n);
            ASSERT_EQ(p.status, st_ok);
            for(const auto& f : fields)
            {
                ASSERT_TRUE(p.fields.count(f.first)) << f.first;
                std::string got = p.fields.at(f.first), want = f.second;
                if(want[0] == '~')
                {
                    for(std::string* t : {&got, &want})
                        std::transform(t->begin(), t->end(), t->begin(), ::tolower);
                    EXPECT_NE(got.find(want.substr(1)), std::string::npos)
                        << f.first << ": " << got;
                }
                else
                {
                    EXPECT_EQ(got, want) << f.first;
                }
            }
            EXPECT_EQ(
                p.body,
                cat(s ? sym : std::vector<std::string>{}, n ? num : std::vector<std::string>{}));
        }
    }

    // The 3 x 4 matrix used by csr, csc, coo and coo_aos:
    //   [1 0 0 2]
    //   [0 3 0 0]
    //   [4 0 5 0]
    const std::vector<float>   s_val     = {1, 2, 3, 4, 5};
    const std::vector<int32_t> s_coo_row = {0, 0, 1, 2, 2}, s_coo_col = {0, 3, 1, 0, 2};

    TEST_F(ConvDescr, PrintFormats)
    {
        const std::map<std::string, std::string> mn   = {{"m", "3"}, {"n", "4"}, {"nnz", "5"}};
        const auto                               with = [&](std::map<std::string, std::string> f) {
            f.insert(mn.begin(), mn.end());
            return f;
        };
        for(int base : {0, 1})
        {
            const auto      ptr = shifted<int32_t>({0, 2, 3, 5}, base);
            const auto      ind = shifted<int32_t>({0, 3, 1, 0, 2}, base);
            DevBuf<int32_t> d_ptr(ptr), d_ind(ind);
            DevBuf<float>   d_val(s_val);
            Spmat           d;
            ASSERT_EQ(rocsparse_create_csr_descr(
                          &d, 3, 4, 5, d_ptr, d_ind, d_val, i32, i32, base_of(base), f32),
                      st_ok);
            check_print(d,
                        with({{"format", "~csr"},
                              {"base", std::to_string(base)},
                              {"ptr_type", "~i32"},
                              {"ind_type", "~i32"},
                              {"data_type", "~f32"}}),
                        cat(lines(ptr), lines(ind)),
                        lines(s_val));
        }
        {
            // The same matrix stored by columns: col_ptr has n + 1 entries.
            const std::vector<int32_t> ptr = {1, 3, 4, 5, 6}, ind = {1, 3, 2, 3, 1};
            const std::vector<float>   val = {1, 4, 3, 5, 2};
            DevBuf<int32_t>            d_ptr(ptr), d_ind(ind);
            DevBuf<float>              d_val(val);
            Spmat                      d;
            ASSERT_EQ(
                rocsparse_create_csc_descr(&d, 3, 4, 5, d_ptr, d_ind, d_val, i32, i32, base1, f32),
                st_ok);
            check_print(d,
                        with({{"format", "~csc"}, {"base", "1"}}),
                        cat(lines(ptr), lines(ind)),
                        lines(val));
        }
        {
            DevBuf<int32_t> d_row(s_coo_row), d_col(s_coo_col);
            DevBuf<float>   d_val(s_val);
            Spmat           d;
            ASSERT_EQ(rocsparse_create_coo_descr(&d, 3, 4, 5, d_row, d_col, d_val, i32, base0, f32),
                      st_ok);
            check_print(d,
                        with({{"format", "~coo"},
                              {"base", "0"},
                              {"row_type", "~i32"},
                              {"col_type", "~i32"}}),
                        cat(lines(s_coo_row), lines(s_coo_col)),
                        lines(s_val));
        }
        {
            // (row, col) pairs.
            const std::vector<int32_t> ind = {0, 0, 0, 3, 1, 1, 2, 0, 2, 2};
            DevBuf<int32_t>            d_ind(ind);
            DevBuf<float>              d_val(s_val);
            Spmat                      d;
            ASSERT_EQ(rocsparse_create_coo_aos_descr(&d, 3, 4, 5, d_ind, d_val, i32, base0, f32),
                      st_ok);
            check_print(d, mn, lines(ind), lines(s_val));
        }
        {
            // Column-major 3 x 2 arrays; a padding entry has the column index -1.
            const std::vector<int32_t> ind = {0, 1, 0, 3, -1, 2};
            const std::vector<float>   val = {1, 3, 4, 2, 0, 5};
            DevBuf<int32_t>            d_ind(ind);
            DevBuf<float>              d_val(val);
            Spmat                      d;
            ASSERT_EQ(rocsparse_create_ell_descr(&d, 3, 4, d_ind, d_val, 2, i32, base0, f32),
                      st_ok);
            check_print(d,
                        {{"format", "~ell"}, {"m", "3"}, {"n", "4"}, {"ell width", "2"}},
                        matrix_rows(ind, 3, 2),
                        matrix_rows(val, 3, 2));
        }
        {
            // 2 x 2 blocks of size 2, three of them stored.
            const std::vector<int32_t> ptr = {0, 2, 3}, ind = {0, 1, 1};
            std::vector<float>         val(12);
            std::iota(val.begin(), val.end(), 1.0f);
            DevBuf<int32_t> d_ptr(ptr), d_ind(ind);
            DevBuf<float>   d_val(val);
            Spmat           d;
            ASSERT_EQ(rocsparse_create_bsr_descr(
                          &d, 2, 2, 3, dir_c, 2, d_ptr, d_ind, d_val, i32, i32, base0, f32),
                      st_ok);
            check_print(d,
                        {{"format", "~bsr"},
                         {"mb", "2"},
                         {"nb", "2"},
                         {"nnzb", "3"},
                         {"dimb", "2"},
                         {"m", "4"},
                         {"n", "4"},
                         {"nnz", "12"},
                         {"dirb", "~column"}},
                        cat(lines(ptr), lines(ind)),
                        lines(val));
        }
        {
            // Only the header is printed for a SELL matrix, whatever the flags.
            DevBuf<int32_t> d_off(std::vector<int32_t>{0, 4, 8}), d_ind(8);
            DevBuf<float>   d_val(8);
            Spmat           d;
            ASSERT_EQ(rocsparse_create_sell_descr(
                          &d, 4, 4, 6, 2, 8, d_off, d_ind, d_val, i32, i32, base0, f32),
                      st_ok);
            check_print(d,
                        {{"format", "~sell"},
                         {"m", "4"},
                         {"n", "4"},
                         {"nnz", "6"},
                         {"sell_slice_size", "2"},
                         {"sell_colval_size", "8"}},
                        {},
                        {});
        }
    }

    TEST_F(ConvDescr, PrintBlockedEll)
    {
        // Library bug (AISPARSE-921): for a blocked ELL matrix, internal_spmat_print
        // reads rows * width indices and width * rows * dim^2 values, more than the
        // descriptor's arrays hold for a block size above one. The buffers are
        // over-allocated so that the read stays in bounds, and only the part that
        // follows from the descriptor is checked.
        constexpr int      rows = 4, cols = 6, dim = 2, width = 4;
        std::vector<float> val(rows * width);
        std::iota(val.begin(), val.end(), 1.0f);
        DevBuf<int32_t> d_ind(std::vector<int32_t>{0, 2, 1, 2}, rows * width);
        DevBuf<float>   d_val(val, rows * width * dim * dim);
        Spmat           d;
        ASSERT_EQ(rocsparse_create_bell_descr(
                      &d, rows, cols, dir_r, dim, width, d_ind, d_val, i32, base0, f32),
                  st_ok);
        for(int flags = 0; flags < 4; ++flags)
        {
            const bool    s = flags & 1, n = flags & 2;
            const Printed p = print(d, s, n);
            ASSERT_EQ(p.status, st_ok);
            EXPECT_EQ(p.fields.at("m"), "4");
            EXPECT_EQ(p.fields.at("n"), "6");
            EXPECT_EQ(p.fields.at("width"), "4");
            EXPECT_EQ(p.fields.at("dimb"), "2");
            if(!s && !n)
            {
                EXPECT_TRUE(p.body.empty());
            }
            if(s)
            {
                // One printed row per matrix row.
                EXPECT_GE(p.body.size(), size_t(rows));
            }
            if(n && !s)
            {
                const auto head = lines(val);
                ASSERT_GE(p.body.size(), head.size());
                EXPECT_TRUE(std::equal(head.begin(), head.end(), p.body.begin()));
            }
        }
    }

    // The four values 1..4 of every value type: raw device bytes and printed form.
    struct Typed
    {
        rocsparse_datatype       type;
        std::vector<uint8_t>     raw;
        std::vector<std::string> printed;
    };

    template <typename F>
    Typed typed(rocsparse_datatype dt, F make)
    {
        Typed tv{dt, {}, {}};
        for(int i = 1; i <= 4; ++i)
        {
            const auto         v = make(i);
            const uint8_t*     b = reinterpret_cast<const uint8_t*>(&v);
            std::ostringstream os;
            os << v;
            tv.raw.insert(tv.raw.end(), b, b + sizeof(v));
            tv.printed.push_back(os.str());
        }
        return tv;
    }

    // 16-bit floating point values given by their bit patterns.
    Typed typed16(rocsparse_datatype dt, const uint16_t (&bits)[4])
    {
        Typed tv{dt, {}, {}};
        for(int i = 0; i < 4; ++i)
        {
            const uint8_t* b = reinterpret_cast<const uint8_t*>(&bits[i]);
            tv.raw.insert(tv.raw.end(), b, b + 2);
            tv.printed.push_back(std::to_string(i + 1));
        }
        return tv;
    }

    TEST_F(ConvDescr, PrintValueTypes)
    {
        // The printed value of the small integers 1..4 is the integer itself for every
        // type; complex values print as the host stream does.
        const std::vector<Typed> all
            = {typed16(rocsparse_datatype_f16_r, {0x3C00, 0x4000, 0x4200, 0x4400}),
               typed16(rocsparse_datatype_bf16_r, {0x3F80, 0x4000, 0x4040, 0x4080}),
               typed(f32, [](int i) { return static_cast<float>(i); }),
               typed(rocsparse_datatype_f64_r, [](int i) { return static_cast<double>(i); }),
               typed(rocsparse_datatype_f32_c,
                     [](int i) { return rocsparse_float_complex(i, -1.0f); }),
               typed(rocsparse_datatype_f64_c,
                     [](int i) { return rocsparse_double_complex(i, -1.0); }),
               typed(rocsparse_datatype_i32_r, [](int i) { return static_cast<int32_t>(i); }),
               typed(rocsparse_datatype_u32_r, [](int i) { return static_cast<uint32_t>(i); }),
               typed(rocsparse_datatype_i8_r, [](int i) { return static_cast<int8_t>(64 + i); }),
               typed(rocsparse_datatype_u8_r, [](int i) { return static_cast<uint8_t>(64 + i); })};

        // CSR: 1 x 4 row, the values are the four numbers. ELL: 2 x 2, width 2.
        DevBuf<int32_t> d_ptr(std::vector<int32_t>{0, 4}), d_ind(std::vector<int32_t>{0, 1, 2, 3});
        DevBuf<int32_t> d_ell(std::vector<int32_t>{0, 1, 1, 0});
        for(const Typed& tv : all)
        {
            SCOPED_TRACE("data type " + std::to_string(static_cast<int>(tv.type)));
            DevBuf<uint8_t> d_val(tv.raw);
            Spmat           csr, ell;
            ASSERT_EQ(rocsparse_create_csr_descr(
                          &csr, 1, 4, 4, d_ptr, d_ind, d_val, i32, i32, base0, tv.type),
                      st_ok);
            ASSERT_EQ(rocsparse_create_ell_descr(&ell, 2, 2, d_ell, d_val, 2, i32, base0, tv.type),
                      st_ok);
            const Printed pc = print(csr, false, true), pe = print(ell, false, true);
            ASSERT_EQ(pc.status, st_ok);
            ASSERT_EQ(pe.status, st_ok);
            ASSERT_EQ(pc.body.size(), size_t(4));
            ASSERT_EQ(pe.body.size(), size_t(2));
            std::vector<std::string> want;
            for(size_t i = 0; i < 4; ++i)
                want.push_back("[" + std::to_string(i) + "] = " + tv.printed[i]);
            EXPECT_EQ(pc.body, want);
            // Entry (i, j) of the ELL matrix is value number j * 2 + i.
            EXPECT_EQ(pe.body[0], " " + tv.printed[0] + " " + tv.printed[2]);
            EXPECT_EQ(pe.body[1], " " + tv.printed[1] + " " + tv.printed[3]);
        }
    }

    TEST_F(ConvDescr, PrintIndexType64)
    {
        const std::vector<int64_t> row(s_coo_row.begin(), s_coo_row.end());
        const std::vector<int64_t> col(s_coo_col.begin(), s_coo_col.end());
        const std::vector<int64_t> ell_ind = {0, 1, 0, 3, -1, 2}, ptr = {0, 1, 2}, ind = {0, 1};
        const auto                 i64 = rocsparse_indextype_i64;
        DevBuf<int64_t>            d_row(row), d_col(col), d_ell(ell_ind), d_ptr(ptr), d_ind(ind);
        DevBuf<float>              d_val(std::vector<float>(6, 1.0f));
        Spmat                      coo, ell, bsr;
        ASSERT_EQ(rocsparse_create_coo_descr(&coo, 3, 4, 5, d_row, d_col, d_val, i64, base0, f32),
                  st_ok);
        ASSERT_EQ(rocsparse_create_ell_descr(&ell, 3, 4, d_ell, d_val, 2, i64, base0, f32), st_ok);
        ASSERT_EQ(rocsparse_create_bsr_descr(
                      &bsr, 2, 2, 2, dir_r, 1, d_ptr, d_ind, d_val, i64, i64, base0, f32),
                  st_ok);
        EXPECT_EQ(print(coo, true, false).body, cat(lines(row), lines(col)));
        EXPECT_EQ(print(ell, true, false).body, matrix_rows(ell_ind, 3, 2));
        EXPECT_EQ(print(bsr, true, true).body,
                  cat(cat(lines(ptr), lines(ind)), lines(std::vector<float>(2, 1.0f))));
    }

    TEST_F(ConvDescr, PrintEmptyAndInvalid)
    {
        // A CSR matrix without entries prints only the row pointer; an ELL matrix of
        // width zero prints no matrix.
        const std::vector<int32_t> ptr(4, 0);
        DevBuf<int32_t>            d_ptr(ptr), d_ind(0);
        DevBuf<float>              d_val(0);
        Spmat                      csr, ell;
        ASSERT_EQ(
            rocsparse_create_csr_descr(&csr, 3, 4, 0, d_ptr, d_ind, d_val, i32, i32, base0, f32),
            st_ok);
        ASSERT_EQ(rocsparse_create_ell_descr(&ell, 3, 4, d_ind, d_val, 0, i32, base0, f32), st_ok);
        const Printed pc = print(csr, true, true), pe = print(ell, true, true);
        ASSERT_EQ(pc.status, st_ok);
        ASSERT_EQ(pe.status, st_ok);
        EXPECT_EQ(pc.fields.at("nnz"), "0");
        EXPECT_EQ(pc.body, lines(ptr));
        EXPECT_EQ(pe.fields.at("ell width"), "0");
        EXPECT_TRUE(pe.body.empty());

        std::ostringstream os;
        EXPECT_REJECTED(rocsparse::internal_spmat_print(os, nullptr, true, true));
        EXPECT_TRUE(os.str().empty());
    }
} // namespace
