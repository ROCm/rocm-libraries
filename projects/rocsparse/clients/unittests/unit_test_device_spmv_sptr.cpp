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

// Device unit tests that drive the public C API into sparse product and
// triangular-solve host dispatch that quick/pre_checkin leave dark:
//   coomm segmented-atomic remainder widths and B layout/transpose clauses,
//   SpMV / v2 SpMV format and algorithm switches plus a second analysis,
//   v2 extra-vector datatype dispatch and batch_count != 1,
//   SpTrSV / SpTrSM analysis policies, scalar/compute conversion, zero pivot,
//   csrsm block-size ladder (nrhs selects blockdim 64..1024).
//
// gfx908-only csrsm kernels and the inactive wavefront copy of coomm are not
// reachable on this GPU. These tests link roc::rocsparse; no library TU is
// compiled in, and nothing here requires rocBLAS.

#include "unit_test_utils.hpp"

#include <cstdint>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    inline bool ok_or_skip(rocsparse_status s)
    {
        return s == rocsparse_status_success || s == rocsparse_status_not_implemented
               || s == rocsparse_status_zero_pivot;
    }

    inline bool expect_ran(rocsparse_status s, const char* what)
    {
        EXPECT_TRUE(ok_or_skip(s)) << what << " status " << static_cast<int>(s);
        return ok_or_skip(s);
    }

    struct SpMat
    {
        rocsparse_spmat_descr d = nullptr;
        ~SpMat()
        {
            if(d)
                rocsparse_destroy_spmat_descr(d);
        }
    };
    struct DnVec
    {
        rocsparse_dnvec_descr d = nullptr;
        ~DnVec()
        {
            if(d)
                rocsparse_destroy_dnvec_descr(d);
        }
    };
    struct DnMat
    {
        rocsparse_dnmat_descr d = nullptr;
        ~DnMat()
        {
            if(d)
                rocsparse_destroy_dnmat_descr(d);
        }
    };
    struct SpmvDescr
    {
        rocsparse_spmv_descr d = nullptr;
        ~SpmvDescr()
        {
            if(d)
                rocsparse_destroy_spmv_descr(d);
        }
    };
    struct SptrsvDescr
    {
        rocsparse_sptrsv_descr d = nullptr;
        ~SptrsvDescr()
        {
            if(d)
                rocsparse_destroy_sptrsv_descr(d);
        }
    };
    struct SptrsmDescr
    {
        rocsparse_sptrsm_descr d = nullptr;
        ~SptrsmDescr()
        {
            if(d)
                rocsparse_destroy_sptrsm_descr(d);
        }
    };
    struct MatDescr
    {
        rocsparse_mat_descr d = nullptr;
        ~MatDescr()
        {
            if(d)
                rocsparse_destroy_mat_descr(d);
        }
    };
    struct MatInfo
    {
        rocsparse_mat_info d = nullptr;
        ~MatInfo()
        {
            if(d)
                rocsparse_destroy_mat_info(d);
        }
    };

    inline bool set_input(rocsparse_handle        handle,
                          rocsparse_spmv_descr    descr,
                          rocsparse_spmv_input    which,
                          const void*             data,
                          size_t                  bytes)
    {
        const rocsparse_status s
            = rocsparse_spmv_set_input(handle, descr, which, data, bytes, nullptr);
        return expect_ran(s, "spmv_set_input");
    }

    // 3x3 identity, zero-based.
    struct Id
    {
        device_vector<rocsparse_int> rp{std::vector<rocsparse_int>{0, 1, 2, 3}};
        device_vector<rocsparse_int> ci{std::vector<rocsparse_int>{0, 1, 2}};
        device_vector<rocsparse_int> aos{std::vector<rocsparse_int>{0, 0, 1, 1, 2, 2}};
        device_vector<float>         val{std::vector<float>{1.f, 1.f, 1.f}};
        device_vector<float>         x{std::vector<float>{1.f, 1.f, 1.f}};
        device_vector<float>         y{std::vector<float>(3, 0.f)};
        static constexpr int64_t     n   = 3;
        static constexpr int64_t     nnz = 3;
        bool                         ok() const
        {
            return rp.ptr && ci.ptr && aos.ptr && val.ptr && x.ptr && y.ptr;
        }
    };

    inline bool make_csr(SpMat& A, Id& id)
    {
        return rocsparse_create_csr_descr(&A.d,
                                          Id::n,
                                          Id::n,
                                          Id::nnz,
                                          id.rp,
                                          id.ci,
                                          id.val,
                                          rocsparse_indextype_i32,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r)
               == rocsparse_status_success;
    }

    inline bool make_vecs(DnVec& x, DnVec& y, Id& id)
    {
        return rocsparse_create_dnvec_descr(&x.d, Id::n, id.x, rocsparse_datatype_f32_r)
                   == rocsparse_status_success
               && rocsparse_create_dnvec_descr(&y.d, Id::n, id.y, rocsparse_datatype_f32_r)
                      == rocsparse_status_success;
    }

    bool run_spmv(rocsparse_handle     handle,
                  rocsparse_spmat_descr A,
                  rocsparse_dnvec_descr x,
                  rocsparse_dnvec_descr y,
                  rocsparse_operation   trans,
                  rocsparse_spmv_alg    alg,
                  bool                  again)
    {
        const float alpha = 1.f;
        const float beta  = 0.f;
        size_t      bs    = 0;
        rocsparse_status st
            = rocsparse_spmv(handle,
                             trans,
                             &alpha,
                             A,
                             x,
                             &beta,
                             y,
                             rocsparse_datatype_f32_r,
                             alg,
                             rocsparse_spmv_stage_buffer_size,
                             &bs,
                             nullptr);
        if(!expect_ran(st, "spmv buffer_size") || st != rocsparse_status_success)
            return st == rocsparse_status_not_implemented;

        device_vector<char> buf(bs ? bs : size_t{1});
        if(!buf.ptr)
        {
            ADD_FAILURE() << "spmv buffer alloc";
            return false;
        }
        void* p = bs ? static_cast<void*>(buf.ptr) : nullptr;
        for(int pass = 0; pass < (again ? 2 : 1); ++pass)
        {
            st = rocsparse_spmv(handle,
                                trans,
                                &alpha,
                                A,
                                x,
                                &beta,
                                y,
                                rocsparse_datatype_f32_r,
                                alg,
                                rocsparse_spmv_stage_preprocess,
                                &bs,
                                p);
            if(!expect_ran(st, "spmv preprocess") || st != rocsparse_status_success)
                return st == rocsparse_status_not_implemented;
        }
        st = rocsparse_spmv(handle,
                            trans,
                            &alpha,
                            A,
                            x,
                            &beta,
                            y,
                            rocsparse_datatype_f32_r,
                            alg,
                            rocsparse_spmv_stage_compute,
                            &bs,
                            p);
        if(hipDeviceSynchronize() != hipSuccess)
        {
            ADD_FAILURE() << "spmv sync";
            return false;
        }
        return expect_ran(st, "spmv compute");
    }

    bool run_v2(rocsparse_handle      handle,
                rocsparse_spmat_descr A,
                rocsparse_dnvec_descr x,
                rocsparse_dnvec_descr y,
                rocsparse_operation   trans,
                rocsparse_spmv_alg    alg,
                rocsparse_datatype    scalar_type,
                rocsparse_datatype    compute_type,
                const void*           alpha,
                const void*           beta)
    {
        SpmvDescr descr;
        if(rocsparse_create_spmv_descr(&descr.d) != rocsparse_status_success)
        {
            ADD_FAILURE() << "create spmv descr";
            return false;
        }
        if(!set_input(handle, descr.d, rocsparse_spmv_input_alg, &alg, sizeof(alg))
           || !set_input(handle, descr.d, rocsparse_spmv_input_operation, &trans, sizeof(trans))
           || !set_input(handle,
                         descr.d,
                         rocsparse_spmv_input_scalar_datatype,
                         &scalar_type,
                         sizeof(scalar_type))
           || !set_input(handle,
                         descr.d,
                         rocsparse_spmv_input_compute_datatype,
                         &compute_type,
                         sizeof(compute_type)))
            return false;

        const bool use_blocks = true;
        if(!set_input(handle,
                      descr.d,
                      rocsparse_spmv_input_nnz_use_starting_block_ids,
                      &use_blocks,
                      sizeof(use_blocks)))
            return false;

        size_t bs = 0;
        rocsparse_status st = rocsparse_v2_spmv_buffer_size(
            handle, descr.d, A, x, y, rocsparse_v2_spmv_stage_analysis, &bs, nullptr);
        if(!expect_ran(st, "v2 buffer analysis") || st != rocsparse_status_success)
            return st == rocsparse_status_not_implemented;
        device_vector<char> buf(bs ? bs : size_t{1});
        if(!buf.ptr)
        {
            ADD_FAILURE() << "v2 buffer alloc";
            return false;
        }
        void* p = bs ? static_cast<void*>(buf.ptr) : nullptr;
        st      = rocsparse_v2_spmv(handle,
                               descr.d,
                               alpha,
                               A,
                               x,
                               beta,
                               y,
                               rocsparse_v2_spmv_stage_analysis,
                               bs,
                               p,
                               nullptr);
        if(!expect_ran(st, "v2 analysis") || st != rocsparse_status_success)
            return st == rocsparse_status_not_implemented;

        bs = 0;
        st = rocsparse_v2_spmv_buffer_size(
            handle, descr.d, A, x, y, rocsparse_v2_spmv_stage_compute, &bs, nullptr);
        if(!expect_ran(st, "v2 buffer compute") || st != rocsparse_status_success)
            return st == rocsparse_status_not_implemented;
        device_vector<char> bufc(bs ? bs : size_t{1});
        if(!bufc.ptr)
        {
            ADD_FAILURE() << "v2 compute buffer alloc";
            return false;
        }
        void* pc = bs ? static_cast<void*>(bufc.ptr) : nullptr;
        st       = rocsparse_v2_spmv(handle,
                               descr.d,
                               alpha,
                               A,
                               x,
                               beta,
                               y,
                               rocsparse_v2_spmv_stage_compute,
                               bs,
                               pc,
                               nullptr);
        if(hipDeviceSynchronize() != hipSuccess)
        {
            ADD_FAILURE() << "v2 sync";
            return false;
        }
        return expect_ran(st, "v2 compute");
    }

    bool run_spmm(rocsparse_handle      handle,
                  rocsparse_spmat_descr A,
                  rocsparse_dnmat_descr B,
                  rocsparse_dnmat_descr C,
                  rocsparse_operation   trans_b,
                  rocsparse_spmm_alg    alg)
    {
        const float alpha = 1.f;
        const float beta  = 0.f;
        size_t      bs    = 0;
        rocsparse_status st = rocsparse_spmm(handle,
                                             rocsparse_operation_none,
                                             trans_b,
                                             &alpha,
                                             A,
                                             B,
                                             &beta,
                                             C,
                                             rocsparse_datatype_f32_r,
                                             alg,
                                             rocsparse_spmm_stage_buffer_size,
                                             &bs,
                                             nullptr);
        if(!expect_ran(st, "spmm buffer") || st != rocsparse_status_success)
            return st == rocsparse_status_not_implemented;
        device_vector<char> buf(bs ? bs : size_t{1});
        if(!buf.ptr)
        {
            ADD_FAILURE() << "spmm buffer alloc";
            return false;
        }
        void* p = bs ? static_cast<void*>(buf.ptr) : nullptr;
        st      = rocsparse_spmm(handle,
                            rocsparse_operation_none,
                            trans_b,
                            &alpha,
                            A,
                            B,
                            &beta,
                            C,
                            rocsparse_datatype_f32_r,
                            alg,
                            rocsparse_spmm_stage_preprocess,
                            &bs,
                            p);
        if(!expect_ran(st, "spmm preprocess") || st != rocsparse_status_success)
            return st == rocsparse_status_not_implemented;
        st = rocsparse_spmm(handle,
                            rocsparse_operation_none,
                            trans_b,
                            &alpha,
                            A,
                            B,
                            &beta,
                            C,
                            rocsparse_datatype_f32_r,
                            alg,
                            rocsparse_spmm_stage_compute,
                            &bs,
                            p);
        if(hipDeviceSynchronize() != hipSuccess)
        {
            ADD_FAILURE() << "spmm sync";
            return false;
        }
        return expect_ran(st, "spmm compute");
    }

    struct Layout
    {
        int64_t rows = 0;
        int64_t cols = 0;
        int64_t ld   = 1;
        int64_t n    = 1;
    };

    inline Layout rhs_layout(int64_t m, int64_t nrhs, rocsparse_operation trans, rocsparse_order order)
    {
        Layout g;
        g.rows = (trans == rocsparse_operation_none) ? m : nrhs;
        g.cols = (trans == rocsparse_operation_none) ? nrhs : m;
        if(order == rocsparse_order_column)
            g.ld = (trans == rocsparse_operation_none) ? m : nrhs;
        else
            g.ld = (trans == rocsparse_operation_none) ? nrhs : m;
        if(g.ld < 1)
            g.ld = 1;
        g.n = (order == rocsparse_order_column) ? g.ld * g.cols : g.ld * g.rows;
        if(g.n < 1)
            g.n = 1;
        return g;
    }

    inline Layout sol_layout(int64_t m, int64_t nrhs, rocsparse_order order)
    {
        Layout g;
        g.rows = m;
        g.cols = nrhs;
        g.ld   = (order == rocsparse_order_column) ? m : nrhs;
        if(g.ld < 1)
            g.ld = 1;
        g.n = (order == rocsparse_order_column) ? g.ld * nrhs : g.ld * m;
        if(g.n < 1)
            g.n = 1;
        return g;
    }

    bool set_tri(rocsparse_spmat_descr A, rocsparse_fill_mode fill, rocsparse_diag_type diag)
    {
        if(rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &fill, sizeof(fill))
           != rocsparse_status_success)
            return false;
        if(rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &diag, sizeof(diag))
           != rocsparse_status_success)
            return false;
        return true;
    }

} // namespace

class CoommSegmentedAtomic : public HandleTest
{
};

TEST_F(CoommSegmentedAtomic, remainders)
{
    // 2x2 identity COO. n selects the segmented-atomic main/remainder kernel
    // (n%8 and the n<4 / n<8 buckets). Each (order, trans) pair takes one
    // clause of the B-layout condition, including conjugate transpose.
    device_vector<rocsparse_int> row{std::vector<rocsparse_int>{0, 1}};
    device_vector<rocsparse_int> col{std::vector<rocsparse_int>{0, 1}};
    device_vector<float>         val{std::vector<float>{1.f, 1.f}};
    ASSERT_TRUE(row.ptr && col.ptr && val.ptr);

    SpMat A;
    ASSERT_EQ(rocsparse_create_coo_descr(&A.d,
                                         2,
                                         2,
                                         2,
                                         row,
                                         col,
                                         val,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    const int64_t              m = 2;
    const int64_t              k = 2;
    const int                  ns[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const rocsparse_order      orders[] = {rocsparse_order_column, rocsparse_order_row};
    const rocsparse_operation  trans[]  = {rocsparse_operation_none,
                                          rocsparse_operation_transpose,
                                          rocsparse_operation_conjugate_transpose};

    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
              rocsparse_status_success);

    for(int n : ns)
    {
        for(rocsparse_order order_b : orders)
        {
            for(rocsparse_operation tb : trans)
            {
                for(rocsparse_order order_c : orders)
                {
                    const Layout bg = rhs_layout(k, n, tb, order_b);
                    // rhs_layout treats the "m" argument as the untransposed row
                    // count. For SpMM, B is k x n (or n x k when transposed).
                    Layout bfix = bg;
                    bfix.rows   = (tb == rocsparse_operation_none) ? k : n;
                    bfix.cols   = (tb == rocsparse_operation_none) ? n : k;
                    if(order_b == rocsparse_order_column)
                        bfix.ld = bfix.rows < 1 ? 1 : bfix.rows;
                    else
                        bfix.ld = bfix.cols < 1 ? 1 : bfix.cols;
                    bfix.n = (order_b == rocsparse_order_column) ? bfix.ld * bfix.cols
                                                                 : bfix.ld * bfix.rows;
                    if(bfix.n < 1)
                        bfix.n = 1;

                    Layout cg;
                    cg.rows = m;
                    cg.cols = n;
                    cg.ld   = (order_c == rocsparse_order_column) ? m : n;
                    if(cg.ld < 1)
                        cg.ld = 1;
                    cg.n = (order_c == rocsparse_order_column) ? cg.ld * n : cg.ld * m;
                    if(cg.n < 1)
                        cg.n = 1;

                    device_vector<float> bval(std::vector<float>(static_cast<size_t>(bfix.n), 1.f));
                    device_vector<float> cval(std::vector<float>(static_cast<size_t>(cg.n), 0.f));
                    ASSERT_TRUE(bval.ptr && cval.ptr);

                    DnMat B;
                    DnMat C;
                    ASSERT_EQ(rocsparse_create_dnmat_descr(&B.d,
                                                           bfix.rows,
                                                           bfix.cols,
                                                           bfix.ld,
                                                           bval,
                                                           rocsparse_datatype_f32_r,
                                                           order_b),
                              rocsparse_status_success);
                    ASSERT_EQ(rocsparse_create_dnmat_descr(&C.d,
                                                           cg.rows,
                                                           cg.cols,
                                                           cg.ld,
                                                           cval,
                                                           rocsparse_datatype_f32_r,
                                                           order_c),
                              rocsparse_status_success);

                    ASSERT_TRUE(run_spmm(handle,
                                         A.d,
                                         B.d,
                                         C.d,
                                         tb,
                                         rocsparse_spmm_alg_coo_segmented_atomic));
                }
            }
        }
    }
}

class SpmvDispatch : public HandleTest
{
};

TEST_F(SpmvDispatch, formats)
{
    Id id;
    ASSERT_TRUE(id.ok());
    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
              rocsparse_status_success);

    const rocsparse_spmv_alg csr_algs[] = {rocsparse_spmv_alg_default,
                                           rocsparse_spmv_alg_csr_rowsplit,
                                           rocsparse_spmv_alg_csr_adaptive,
                                           rocsparse_spmv_alg_csr_lrb,
                                           rocsparse_spmv_alg_csr_nnzsplit};
    const rocsparse_operation ops[]
        = {rocsparse_operation_none, rocsparse_operation_transpose, rocsparse_operation_conjugate_transpose};

    // Each algorithm caches analysis on the matrix. A later algorithm must not
    // reuse that cache: nnz-split reads arrays the other analyses never fill.
    for(rocsparse_spmv_alg alg : csr_algs)
    {
        for(rocsparse_operation op : ops)
        {
            SpMat A;
            DnVec x;
            DnVec y;
            ASSERT_TRUE(make_csr(A, id));
            ASSERT_TRUE(make_vecs(x, y, id));
            const bool again
                = (alg == rocsparse_spmv_alg_default && op == rocsparse_operation_none);
            ASSERT_TRUE(run_spmv(handle, A.d, x.d, y.d, op, alg, again));
        }
    }

    {
        SpMat A;
        DnVec x;
        DnVec y;
        ASSERT_EQ(rocsparse_create_csc_descr(&A.d,
                                             Id::n,
                                             Id::n,
                                             Id::nnz,
                                             id.rp,
                                             id.ci,
                                             id.val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_TRUE(make_vecs(x, y, id));
        ASSERT_TRUE(run_spmv(handle, A.d, x.d, y.d, rocsparse_operation_none, rocsparse_spmv_alg_default, true));
        ASSERT_TRUE(run_spmv(
            handle, A.d, x.d, y.d, rocsparse_operation_conjugate_transpose, rocsparse_spmv_alg_default, false));
    }
    for(rocsparse_spmv_alg alg : csr_algs)
    {
        if(alg == rocsparse_spmv_alg_default)
            continue;
        SpMat A;
        DnVec x;
        DnVec y;
        ASSERT_EQ(rocsparse_create_csc_descr(&A.d,
                                             Id::n,
                                             Id::n,
                                             Id::nnz,
                                             id.rp,
                                             id.ci,
                                             id.val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_TRUE(make_vecs(x, y, id));
        ASSERT_TRUE(run_spmv(handle, A.d, x.d, y.d, rocsparse_operation_none, alg, false));
    }

    const rocsparse_spmv_alg coo_algs[]
        = {rocsparse_spmv_alg_default, rocsparse_spmv_alg_coo, rocsparse_spmv_alg_coo_atomic};
    {
        SpMat A;
        DnVec x;
        DnVec y;
        ASSERT_EQ(rocsparse_create_coo_descr(&A.d,
                                             Id::n,
                                             Id::n,
                                             Id::nnz,
                                             id.ci,
                                             id.ci,
                                             id.val,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_TRUE(make_vecs(x, y, id));
        bool first = true;
        for(rocsparse_spmv_alg alg : coo_algs)
        {
            ASSERT_TRUE(run_spmv(handle, A.d, x.d, y.d, rocsparse_operation_none, alg, first));
            first = false;
        }
        ASSERT_TRUE(run_spmv(
            handle, A.d, x.d, y.d, rocsparse_operation_transpose, rocsparse_spmv_alg_coo, false));
    }
    {
        SpMat A;
        DnVec x;
        DnVec y;
        ASSERT_EQ(rocsparse_create_coo_aos_descr(&A.d,
                                                 Id::n,
                                                 Id::n,
                                                 Id::nnz,
                                                 id.aos,
                                                 id.val,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_TRUE(make_vecs(x, y, id));
        bool first = true;
        for(rocsparse_spmv_alg alg : coo_algs)
        {
            ASSERT_TRUE(run_spmv(handle, A.d, x.d, y.d, rocsparse_operation_none, alg, first));
            first = false;
        }
    }
    {
        SpMat A;
        DnVec x;
        DnVec y;
        ASSERT_EQ(rocsparse_create_ell_descr(&A.d,
                                             Id::n,
                                             Id::n,
                                             id.ci,
                                             id.val,
                                             1,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_TRUE(make_vecs(x, y, id));
        ASSERT_TRUE(run_spmv(
            handle, A.d, x.d, y.d, rocsparse_operation_none, rocsparse_spmv_alg_default, true));
        ASSERT_TRUE(
            run_spmv(handle, A.d, x.d, y.d, rocsparse_operation_none, rocsparse_spmv_alg_ell, false));
        ASSERT_TRUE(run_spmv(handle,
                             A.d,
                             x.d,
                             y.d,
                             rocsparse_operation_conjugate_transpose,
                             rocsparse_spmv_alg_ell,
                             false));
    }
    {
        for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
        {
            SpMat local;
            DnVec x;
            DnVec y;
            ASSERT_EQ(rocsparse_create_bsr_descr(&local.d,
                                                 Id::n,
                                                 Id::n,
                                                 Id::nnz,
                                                 dir,
                                                 1,
                                                 id.rp,
                                                 id.ci,
                                                 id.val,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            ASSERT_TRUE(make_vecs(x, y, id));
            ASSERT_TRUE(run_spmv(handle,
                                 local.d,
                                 x.d,
                                 y.d,
                                 rocsparse_operation_none,
                                 rocsparse_spmv_alg_default,
                                 true));
            ASSERT_TRUE(run_spmv(
                handle, local.d, x.d, y.d, rocsparse_operation_transpose, rocsparse_spmv_alg_bsr, false));
        }
    }
}

class V2SpmvDispatch : public HandleTest
{
};

TEST_F(V2SpmvDispatch, formats)
{
    Id id;
    ASSERT_TRUE(id.ok());
    const float alpha = 1.f;
    const float beta  = 0.f;
    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
              rocsparse_status_success);

    auto once = [&](rocsparse_spmat_descr A, rocsparse_spmv_alg alg, rocsparse_operation op) {
        DnVec x;
        DnVec y;
        ASSERT_TRUE(make_vecs(x, y, id));
        ASSERT_TRUE(run_v2(handle,
                           A,
                           x.d,
                           y.d,
                           op,
                           alg,
                           rocsparse_datatype_f32_r,
                           rocsparse_datatype_f32_r,
                           &alpha,
                           &beta));
    };

    {
        SpMat A;
        ASSERT_TRUE(make_csr(A, id));
        once(A.d, rocsparse_spmv_alg_csr_nnzsplit, rocsparse_operation_none);
        once(A.d, rocsparse_spmv_alg_csr_lrb, rocsparse_operation_conjugate_transpose);
        // Second analysis on the same matrix takes the "info already present" side.
        once(A.d, rocsparse_spmv_alg_csr_adaptive, rocsparse_operation_none);
        once(A.d, rocsparse_spmv_alg_csr_adaptive, rocsparse_operation_none);
    }
    {
        SpMat A;
        ASSERT_EQ(rocsparse_create_csc_descr(&A.d,
                                             Id::n,
                                             Id::n,
                                             Id::nnz,
                                             id.rp,
                                             id.ci,
                                             id.val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        once(A.d, rocsparse_spmv_alg_csr_nnzsplit, rocsparse_operation_none);
        once(A.d, rocsparse_spmv_alg_default, rocsparse_operation_transpose);
    }
    {
        SpMat A;
        ASSERT_EQ(rocsparse_create_coo_descr(&A.d,
                                             Id::n,
                                             Id::n,
                                             Id::nnz,
                                             id.ci,
                                             id.ci,
                                             id.val,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        once(A.d, rocsparse_spmv_alg_coo_atomic, rocsparse_operation_none);
        once(A.d, rocsparse_spmv_alg_coo, rocsparse_operation_conjugate_transpose);
    }
    {
        SpMat A;
        ASSERT_EQ(rocsparse_create_coo_aos_descr(&A.d,
                                                 Id::n,
                                                 Id::n,
                                                 Id::nnz,
                                                 id.aos,
                                                 id.val,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        once(A.d, rocsparse_spmv_alg_coo, rocsparse_operation_transpose);
    }
    {
        SpMat A;
        ASSERT_EQ(rocsparse_create_ell_descr(&A.d,
                                             Id::n,
                                             Id::n,
                                             id.ci,
                                             id.val,
                                             1,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        once(A.d, rocsparse_spmv_alg_ell, rocsparse_operation_conjugate_transpose);
    }
    {
        SpMat A;
        ASSERT_EQ(rocsparse_create_bsr_descr(&A.d,
                                             Id::n,
                                             Id::n,
                                             Id::nnz,
                                             rocsparse_direction_column,
                                             1,
                                             id.rp,
                                             id.ci,
                                             id.val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        once(A.d, rocsparse_spmv_alg_bsr, rocsparse_operation_none);
    }
    {
        // One SELL slice of two rows.
        device_vector<rocsparse_int> off{std::vector<rocsparse_int>{0, 2}};
        device_vector<rocsparse_int> scol{std::vector<rocsparse_int>{0, 1}};
        device_vector<float>         sval{std::vector<float>{1.f, 1.f}};
        ASSERT_TRUE(off.ptr && scol.ptr && sval.ptr);
        SpMat A;
        ASSERT_EQ(rocsparse_create_sell_descr(&A.d,
                                              2,
                                              2,
                                              2,
                                              2,
                                              2,
                                              off,
                                              scol,
                                              sval,
                                              rocsparse_indextype_i32,
                                              rocsparse_indextype_i32,
                                              rocsparse_index_base_zero,
                                              rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        device_vector<float> sx{std::vector<float>{1.f, 1.f}};
        device_vector<float> sy{std::vector<float>{0.f, 0.f}};
        DnVec               x;
        DnVec               y;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&x.d, 2, sx, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&y.d, 2, sy, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_TRUE(run_v2(handle,
                           A.d,
                           x.d,
                           y.d,
                           rocsparse_operation_none,
                           rocsparse_spmv_alg_sell,
                           rocsparse_datatype_f32_r,
                           rocsparse_datatype_f32_r,
                           &alpha,
                           &beta));
        ASSERT_TRUE(run_v2(handle,
                           A.d,
                           x.d,
                           y.d,
                           rocsparse_operation_transpose,
                           rocsparse_spmv_alg_default,
                           rocsparse_datatype_f32_r,
                           rocsparse_datatype_f32_r,
                           &alpha,
                           &beta));
    }
}

TEST_F(V2SpmvDispatch, extras_and_batch)
{
    // i32 extra extraction, the unsupported-datatype else, enable-before-set,
    // and batch_count != 1 on the matrix and on each dense vector.
    SpmvDescr descr;
    ASSERT_EQ(rocsparse_create_spmv_descr(&descr.d), rocsparse_status_success);

    const int32_t enable = 1;
    EXPECT_EQ(rocsparse_spmv_set_input(handle,
                                       descr.d,
                                       rocsparse_spmv_input_enable_extra,
                                       &enable,
                                       sizeof(enable),
                                       nullptr),
              rocsparse_status_invalid_value);

    auto set_types = [&](rocsparse_datatype scalar_type, rocsparse_datatype compute_type) {
        ASSERT_EQ(rocsparse_spmv_set_input(handle,
                                           descr.d,
                                           rocsparse_spmv_input_scalar_datatype,
                                           &scalar_type,
                                           sizeof(scalar_type),
                                           nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spmv_set_input(handle,
                                           descr.d,
                                           rocsparse_spmv_input_compute_datatype,
                                           &compute_type,
                                           sizeof(compute_type),
                                           nullptr),
                  rocsparse_status_success);
    };

    {
        set_types(rocsparse_datatype_i32_r, rocsparse_datatype_i32_r);
        device_vector<int32_t> gamma{std::vector<int32_t>{1, 1}};
        device_vector<int32_t> z0{std::vector<int32_t>{1, 1, 1}};
        device_vector<int32_t> z1{std::vector<int32_t>{2, 2, 2}};
        ASSERT_TRUE(gamma.ptr && z0.ptr && z1.ptr);
        DnVec g;
        DnVec a;
        DnVec b;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&g.d, 2, gamma, rocsparse_datatype_i32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&a.d, 3, z0, rocsparse_datatype_i32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&b.d, 3, z1, rocsparse_datatype_i32_r),
                  rocsparse_status_success);
        rocsparse_const_dnvec_descr zs[2] = {a.d, b.d};
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 2, g.d, zs, nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmv_set_input(handle,
                                           descr.d,
                                           rocsparse_spmv_input_enable_extra,
                                           &enable,
                                           sizeof(enable),
                                           nullptr),
                  rocsparse_status_success);
        const int32_t disable = 0;
        EXPECT_EQ(rocsparse_spmv_set_input(handle,
                                           descr.d,
                                           rocsparse_spmv_input_enable_extra,
                                           &disable,
                                           sizeof(disable),
                                           nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_spmv_clear_extra(handle, descr.d, nullptr), rocsparse_status_success);
    }

    {
        // f16/f16 is not one of the extract instantiations.
        set_types(rocsparse_datatype_f16_r, rocsparse_datatype_f16_r);
        device_vector<float> gamma{std::vector<float>{1.f}};
        device_vector<float> z{std::vector<float>{1.f, 1.f, 1.f}};
        DnVec               g;
        DnVec               a;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&g.d, 1, gamma, rocsparse_datatype_f16_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&a.d, 3, z, rocsparse_datatype_f16_r),
                  rocsparse_status_success);
        rocsparse_const_dnvec_descr zs[1] = {a.d};
        EXPECT_EQ(rocsparse_spmv_set_extra(handle, descr.d, 1, g.d, zs, nullptr),
                  rocsparse_status_not_implemented);
    }

    Id id;
    ASSERT_TRUE(id.ok());
    SpMat A;
    DnVec x;
    DnVec y;
    ASSERT_TRUE(make_csr(A, id));
    ASSERT_TRUE(make_vecs(x, y, id));
    const float          alpha = 1.f;
    const float          beta  = 0.f;
    const rocsparse_spmv_alg alg = rocsparse_spmv_alg_default;
    SpmvDescr            d2;
    ASSERT_EQ(rocsparse_create_spmv_descr(&d2.d), rocsparse_status_success);
    set_types(rocsparse_datatype_f32_r, rocsparse_datatype_f32_r);
    // set_types wrote into descr, not d2. Set d2 explicitly.
    ASSERT_EQ(rocsparse_spmv_set_input(
                  handle, d2.d, rocsparse_spmv_input_alg, &alg, sizeof(alg), nullptr),
              rocsparse_status_success);
    const rocsparse_operation op = rocsparse_operation_none;
    ASSERT_EQ(rocsparse_spmv_set_input(
                  handle, d2.d, rocsparse_spmv_input_operation, &op, sizeof(op), nullptr),
              rocsparse_status_success);
    const rocsparse_datatype f32 = rocsparse_datatype_f32_r;
    ASSERT_EQ(rocsparse_spmv_set_input(handle,
                                       d2.d,
                                       rocsparse_spmv_input_scalar_datatype,
                                       &f32,
                                       sizeof(f32),
                                       nullptr),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_spmv_set_input(handle,
                                       d2.d,
                                       rocsparse_spmv_input_compute_datatype,
                                       &f32,
                                       sizeof(f32),
                                       nullptr),
              rocsparse_status_success);

    // buffer_size and v2_spmv each have their own batch_count != 1 checks.
    auto expect_batch = [&](rocsparse_v2_spmv_stage stage) {
        size_t bs = 0;
        EXPECT_EQ(rocsparse_v2_spmv_buffer_size(handle, d2.d, A.d, x.d, y.d, stage, &bs, nullptr),
                  rocsparse_status_not_implemented);
        EXPECT_EQ(rocsparse_v2_spmv(handle,
                                    d2.d,
                                    &alpha,
                                    A.d,
                                    x.d,
                                    &beta,
                                    y.d,
                                    stage,
                                    0,
                                    nullptr,
                                    nullptr),
                  rocsparse_status_not_implemented);
    };

    ASSERT_EQ(rocsparse_csr_set_strided_batch(A.d, 2, Id::n + 1, Id::nnz), rocsparse_status_success);
    expect_batch(rocsparse_v2_spmv_stage_analysis);
    ASSERT_EQ(rocsparse_csr_set_strided_batch(A.d, 1, Id::n + 1, Id::nnz), rocsparse_status_success);

    ASSERT_EQ(rocsparse_dnvec_set_strided_batch(x.d, 2, Id::n), rocsparse_status_success);
    expect_batch(rocsparse_v2_spmv_stage_analysis);
    ASSERT_EQ(rocsparse_dnvec_set_strided_batch(x.d, 1, Id::n), rocsparse_status_success);

    ASSERT_EQ(rocsparse_dnvec_set_strided_batch(y.d, 2, Id::n), rocsparse_status_success);
    expect_batch(rocsparse_v2_spmv_stage_compute);
}

class SptrsvDispatch : public HandleTest
{
};

namespace
{
    bool configure_sptrsv(rocsparse_handle            handle,
                          rocsparse_sptrsv_descr      descr,
                          rocsparse_operation         op,
                          rocsparse_datatype          scalar_type,
                          rocsparse_datatype          compute_type,
                          rocsparse_analysis_policy   policy,
                          const void*                 alpha)
    {
        const rocsparse_sptrsv_alg alg = rocsparse_sptrsv_alg_default;
        auto                       put = [&](rocsparse_sptrsv_input which, const void* data, size_t n) {
            return rocsparse_sptrsv_set_input(handle, descr, which, data, n, nullptr);
        };
        if(put(rocsparse_sptrsv_input_alg, &alg, sizeof(alg)) != rocsparse_status_success)
            return false;
        if(put(rocsparse_sptrsv_input_operation, &op, sizeof(op)) != rocsparse_status_success)
            return false;
        if(put(rocsparse_sptrsv_input_scalar_datatype, &scalar_type, sizeof(scalar_type))
           != rocsparse_status_success)
            return false;
        if(put(rocsparse_sptrsv_input_compute_datatype, &compute_type, sizeof(compute_type))
           != rocsparse_status_success)
            return false;
        if(put(rocsparse_sptrsv_input_analysis_policy, &policy, sizeof(policy))
           != rocsparse_status_success)
            return false;
        if(put(rocsparse_sptrsv_input_scalar_alpha, alpha, sizeof(const void*))
           != rocsparse_status_success)
            return false;
        const rocsparse_solve_mode mode = rocsparse_solve_mode_triangular;
        if(put(rocsparse_sptrsv_input_solve_mode, &mode, sizeof(mode)) != rocsparse_status_success)
            return false;
        return true;
    }

    bool run_sptrsv(rocsparse_handle          handle,
                    rocsparse_spmat_descr     A,
                    rocsparse_dnvec_descr     x,
                    rocsparse_dnvec_descr     y,
                    rocsparse_operation       op,
                    rocsparse_datatype        scalar_type,
                    rocsparse_datatype        compute_type,
                    rocsparse_analysis_policy policy,
                    const void*               alpha,
                    bool                      diagonal)
    {
        SptrsvDescr descr;
        if(rocsparse_create_sptrsv_descr(&descr.d) != rocsparse_status_success)
        {
            ADD_FAILURE() << "create sptrsv";
            return false;
        }

        int64_t position = -1;
        // Output queries write through the handle pointer mode. The position
        // lives on the host, so query in host mode and restore around compute.
        rocsparse_pointer_mode saved_mode = rocsparse_pointer_mode_host;
        if(rocsparse_get_pointer_mode(handle, &saved_mode) != rocsparse_status_success)
        {
            ADD_FAILURE() << "get pointer mode";
            return false;
        }
        if(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host) != rocsparse_status_success)
        {
            ADD_FAILURE() << "host pointer mode";
            return false;
        }
        rocsparse_singularity sing{};
        (void)rocsparse_sptrsv_get_output(handle,
                                          descr.d,
                                          rocsparse_sptrsv_output_singularity_position,
                                          &position,
                                          sizeof(position),
                                          nullptr);
        (void)rocsparse_sptrsv_get_output(handle,
                                          descr.d,
                                          rocsparse_sptrsv_output_singularity,
                                          &sing,
                                          sizeof(sing),
                                          nullptr);
        if(rocsparse_set_pointer_mode(handle, saved_mode) != rocsparse_status_success)
        {
            ADD_FAILURE() << "restore pointer mode";
            return false;
        }

        if(!configure_sptrsv(handle, descr.d, op, scalar_type, compute_type, policy, alpha))
        {
            ADD_FAILURE() << "configure sptrsv";
            return false;
        }
        if(diagonal)
        {
            const rocsparse_solve_mode       mode = rocsparse_solve_mode_diagonal;
            const rocsparse_diagonal_modifier mod  = rocsparse_diagonal_modifier_absolute;
            if(rocsparse_sptrsv_set_input(handle,
                                          descr.d,
                                          rocsparse_sptrsv_input_solve_mode,
                                          &mode,
                                          sizeof(mode),
                                          nullptr)
                   != rocsparse_status_success
               || rocsparse_sptrsv_set_input(handle,
                                             descr.d,
                                             rocsparse_sptrsv_input_diagonal_modifier,
                                             &mod,
                                             sizeof(mod),
                                             nullptr)
                      != rocsparse_status_success)
            {
                ADD_FAILURE() << "diagonal mode";
                return false;
            }
        }

        size_t bs = 0;
        rocsparse_status st = rocsparse_sptrsv_buffer_size(
            handle, descr.d, A, x, y, rocsparse_sptrsv_stage_analysis, &bs, nullptr);
        if(!expect_ran(st, "sptrsv buffer") || st != rocsparse_status_success)
            return st == rocsparse_status_not_implemented;
        device_vector<char> buf(bs ? bs : size_t{1});
        if(!buf.ptr)
        {
            ADD_FAILURE() << "sptrsv buffer";
            return false;
        }
        void* p = bs ? static_cast<void*>(buf.ptr) : nullptr;
        st      = rocsparse_sptrsv(
            handle, descr.d, A, x, y, rocsparse_sptrsv_stage_analysis, bs, p, nullptr);
        if(!expect_ran(st, "sptrsv analysis") || st != rocsparse_status_success)
            return st == rocsparse_status_not_implemented;

        if(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host) != rocsparse_status_success)
        {
            ADD_FAILURE() << "host pointer mode";
            return false;
        }
        st = rocsparse_sptrsv_get_output(handle,
                                         descr.d,
                                         rocsparse_sptrsv_output_singularity_position,
                                         &position,
                                         sizeof(position),
                                         nullptr);
        if(!expect_ran(st, "singularity position"))
            return false;
        st = rocsparse_sptrsv_get_output(handle,
                                         descr.d,
                                         rocsparse_sptrsv_output_singularity,
                                         &sing,
                                         sizeof(sing),
                                         nullptr);
        if(!expect_ran(st, "singularity"))
            return false;
        st = rocsparse_sptrsv_get_output(handle,
                                         descr.d,
                                         rocsparse_sptrsv_output_zero_pivot_position,
                                         &position,
                                         sizeof(position),
                                         nullptr);
        if(!expect_ran(st, "zero pivot"))
            return false;
        if(rocsparse_set_pointer_mode(handle, saved_mode) != rocsparse_status_success)
        {
            ADD_FAILURE() << "restore pointer mode";
            return false;
        }

        bs = 0;
        st = rocsparse_sptrsv_buffer_size(
            handle, descr.d, A, x, y, rocsparse_sptrsv_stage_compute, &bs, nullptr);
        if(!expect_ran(st, "sptrsv compute buffer") || st != rocsparse_status_success)
            return st == rocsparse_status_not_implemented;
        device_vector<char> bufc(bs ? bs : size_t{1});
        if(!bufc.ptr)
        {
            ADD_FAILURE() << "sptrsv compute buffer";
            return false;
        }
        void* pc = bs ? static_cast<void*>(bufc.ptr) : nullptr;
        st       = rocsparse_sptrsv(
            handle, descr.d, A, x, y, rocsparse_sptrsv_stage_compute, bs, pc, nullptr);
        if(hipDeviceSynchronize() != hipSuccess)
        {
            ADD_FAILURE() << "sptrsv sync";
            return false;
        }
        return expect_ran(st, "sptrsv compute");
    }
} // namespace

TEST_F(SptrsvDispatch, triangular)
{
    Id id;
    ASSERT_TRUE(id.ok());
    const float alpha = 1.f;
    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
              rocsparse_status_success);

    const rocsparse_analysis_policy policies[]
        = {rocsparse_analysis_policy_reuse, rocsparse_analysis_policy_force};
    const rocsparse_operation ops[] = {rocsparse_operation_none,
                                       rocsparse_operation_transpose,
                                       rocsparse_operation_conjugate_transpose};
    const rocsparse_fill_mode fills[] = {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper};
    const rocsparse_diag_type diags[] = {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit};

    auto solve_id = [&](auto create) {
        for(rocsparse_fill_mode fill : fills)
        {
            for(rocsparse_diag_type diag : diags)
            {
                for(rocsparse_analysis_policy policy : policies)
                {
                    for(rocsparse_operation op : ops)
                    {
                        SpMat A;
                        DnVec x;
                        DnVec y;
                        ASSERT_TRUE(create(A));
                        ASSERT_TRUE(set_tri(A.d, fill, diag));
                        ASSERT_TRUE(make_vecs(x, y, id));
                        ASSERT_TRUE(run_sptrsv(handle,
                                               A.d,
                                               x.d,
                                               y.d,
                                               op,
                                               rocsparse_datatype_f32_r,
                                               rocsparse_datatype_f32_r,
                                               policy,
                                               &alpha,
                                               false));
                    }
                }
            }
        }
    };

    solve_id([&](SpMat& A) { return make_csr(A, id); });
    solve_id([&](SpMat& A) {
        return rocsparse_create_coo_descr(&A.d,
                                          Id::n,
                                          Id::n,
                                          Id::nnz,
                                          id.ci,
                                          id.ci,
                                          id.val,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r)
               == rocsparse_status_success;
    });
    solve_id([&](SpMat& A) {
        return rocsparse_create_csc_descr(&A.d,
                                          Id::n,
                                          Id::n,
                                          Id::nnz,
                                          id.rp,
                                          id.ci,
                                          id.val,
                                          rocsparse_indextype_i32,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r)
               == rocsparse_status_success;
    });
    solve_id([&](SpMat& A) {
        return rocsparse_create_ell_descr(&A.d,
                                          Id::n,
                                          Id::n,
                                          id.ci,
                                          id.val,
                                          1,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r)
               == rocsparse_status_success;
    });

    // Diagonal solve is a separate compute switch (CSR and CSC only).
    for(int which = 0; which < 2; ++which)
    {
        SpMat A;
        DnVec x;
        DnVec y;
        if(which == 0)
            ASSERT_TRUE(make_csr(A, id));
        else
            ASSERT_EQ(rocsparse_create_csc_descr(&A.d,
                                                 Id::n,
                                                 Id::n,
                                                 Id::nnz,
                                                 id.rp,
                                                 id.ci,
                                                 id.val,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);
        ASSERT_TRUE(set_tri(A.d, rocsparse_fill_mode_lower, rocsparse_diag_type_non_unit));
        ASSERT_TRUE(make_vecs(x, y, id));
        ASSERT_TRUE(run_sptrsv(handle,
                               A.d,
                               x.d,
                               y.d,
                               rocsparse_operation_none,
                               rocsparse_datatype_f32_r,
                               rocsparse_datatype_f32_r,
                               rocsparse_analysis_policy_force,
                               &alpha,
                               true));
    }

    // Missing diagonal, non-unit: zero-pivot status from the output query.
    {
        device_vector<rocsparse_int> rp{std::vector<rocsparse_int>{0, 1, 2}};
        device_vector<rocsparse_int> ci{std::vector<rocsparse_int>{0, 0}};
        device_vector<float>         val{std::vector<float>{1.f, 1.f}};
        device_vector<float>         xh{std::vector<float>{1.f, 1.f}};
        device_vector<float>         yh{std::vector<float>{0.f, 0.f}};
        ASSERT_TRUE(rp.ptr && ci.ptr && val.ptr && xh.ptr && yh.ptr);
        SpMat A;
        DnVec x;
        DnVec y;
        ASSERT_EQ(rocsparse_create_csr_descr(&A.d,
                                             2,
                                             2,
                                             2,
                                             rp,
                                             ci,
                                             val,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_TRUE(set_tri(A.d, rocsparse_fill_mode_lower, rocsparse_diag_type_non_unit));
        ASSERT_EQ(rocsparse_create_dnvec_descr(&x.d, 2, xh, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&y.d, 2, yh, rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_TRUE(run_sptrsv(handle,
                               A.d,
                               x.d,
                               y.d,
                               rocsparse_operation_none,
                               rocsparse_datatype_f32_r,
                               rocsparse_datatype_f32_r,
                               rocsparse_analysis_policy_reuse,
                               &alpha,
                               false));
    }
}

TEST_F(SptrsvDispatch, scalar_mismatch)
{
    // Matrix and vectors are f64; the scalar is f32, converted on host and device.
    device_vector<rocsparse_int> rp{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> ci{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<double>        val{std::vector<double>{1.0, 1.0, 1.0}};
    device_vector<double>        xh{std::vector<double>{1.0, 1.0, 1.0}};
    device_vector<double>        yh{std::vector<double>{0.0, 0.0, 0.0}};
    ASSERT_TRUE(rp.ptr && ci.ptr && val.ptr && xh.ptr && yh.ptr);
    SpMat A;
    DnVec x;
    DnVec y;
    ASSERT_EQ(rocsparse_create_csr_descr(&A.d,
                                         3,
                                         3,
                                         3,
                                         rp,
                                         ci,
                                         val,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f64_r),
              rocsparse_status_success);
    ASSERT_TRUE(set_tri(A.d, rocsparse_fill_mode_lower, rocsparse_diag_type_non_unit));
    ASSERT_EQ(rocsparse_create_dnvec_descr(&x.d, 3, xh, rocsparse_datatype_f64_r),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnvec_descr(&y.d, 3, yh, rocsparse_datatype_f64_r),
              rocsparse_status_success);

    const float          alpha_h = 1.f;
    device_vector<float> alpha_d{std::vector<float>{1.f}};
    ASSERT_TRUE(alpha_d.ptr);

    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
              rocsparse_status_success);
    ASSERT_TRUE(run_sptrsv(handle,
                           A.d,
                           x.d,
                           y.d,
                           rocsparse_operation_none,
                           rocsparse_datatype_f32_r,
                           rocsparse_datatype_f64_r,
                           rocsparse_analysis_policy_reuse,
                           &alpha_h,
                           false));

    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
              rocsparse_status_success);
    ASSERT_TRUE(run_sptrsv(handle,
                           A.d,
                           x.d,
                           y.d,
                           rocsparse_operation_transpose,
                           rocsparse_datatype_f32_r,
                           rocsparse_datatype_f64_r,
                           rocsparse_analysis_policy_force,
                           alpha_d.ptr,
                           false));
}

class SptrsmDispatch : public HandleTest
{
};

namespace
{
    bool configure_sptrsm(rocsparse_handle          handle,
                          rocsparse_sptrsm_descr    descr,
                          rocsparse_operation       op_a,
                          rocsparse_operation       op_x,
                          rocsparse_datatype        scalar_type,
                          rocsparse_datatype        compute_type,
                          rocsparse_analysis_policy policy,
                          const void*               alpha)
    {
        const rocsparse_sptrsm_alg alg  = rocsparse_sptrsm_alg_default;
        const rocsparse_solve_mode mode = rocsparse_solve_mode_triangular;
        auto                       put
            = [&](rocsparse_sptrsm_input which, const void* data, size_t n) {
                  return rocsparse_sptrsm_set_input(handle, descr, which, data, n, nullptr);
              };
        if(put(rocsparse_sptrsm_input_alg, &alg, sizeof(alg)) != rocsparse_status_success)
            return false;
        if(put(rocsparse_sptrsm_input_operation_A, &op_a, sizeof(op_a)) != rocsparse_status_success)
            return false;
        if(put(rocsparse_sptrsm_input_operation_X, &op_x, sizeof(op_x)) != rocsparse_status_success)
            return false;
        if(put(rocsparse_sptrsm_input_scalar_datatype, &scalar_type, sizeof(scalar_type))
           != rocsparse_status_success)
            return false;
        if(put(rocsparse_sptrsm_input_compute_datatype, &compute_type, sizeof(compute_type))
           != rocsparse_status_success)
            return false;
        if(put(rocsparse_sptrsm_input_analysis_policy, &policy, sizeof(policy))
           != rocsparse_status_success)
            return false;
        if(put(rocsparse_sptrsm_input_scalar_alpha, alpha, sizeof(const void*))
           != rocsparse_status_success)
            return false;
        if(put(rocsparse_sptrsm_input_solve_mode, &mode, sizeof(mode)) != rocsparse_status_success)
            return false;
        return true;
    }

    bool run_sptrsm(rocsparse_handle          handle,
                    rocsparse_spmat_descr     A,
                    rocsparse_dnmat_descr     X,
                    rocsparse_dnmat_descr     Y,
                    rocsparse_operation       op_a,
                    rocsparse_operation       op_x,
                    rocsparse_datatype        scalar_type,
                    rocsparse_datatype        compute_type,
                    rocsparse_analysis_policy policy,
                    const void*               alpha,
                    bool                      diagonal)
    {
        SptrsmDescr descr;
        if(rocsparse_create_sptrsm_descr(&descr.d) != rocsparse_status_success)
        {
            ADD_FAILURE() << "create sptrsm";
            return false;
        }
        if(!configure_sptrsm(
               handle, descr.d, op_a, op_x, scalar_type, compute_type, policy, alpha))
        {
            ADD_FAILURE() << "configure sptrsm";
            return false;
        }
        if(diagonal)
        {
            const rocsparse_solve_mode       mode = rocsparse_solve_mode_diagonal;
            const rocsparse_diagonal_modifier mod  = rocsparse_diagonal_modifier_none;
            if(rocsparse_sptrsm_set_input(handle,
                                          descr.d,
                                          rocsparse_sptrsm_input_solve_mode,
                                          &mode,
                                          sizeof(mode),
                                          nullptr)
                   != rocsparse_status_success
               || rocsparse_sptrsm_set_input(handle,
                                             descr.d,
                                             rocsparse_sptrsm_input_diagonal_modifier,
                                             &mod,
                                             sizeof(mod),
                                             nullptr)
                      != rocsparse_status_success)
            {
                ADD_FAILURE() << "sptrsm diagonal";
                return false;
            }
        }

        size_t bs = 0;
        rocsparse_status st = rocsparse_sptrsm_buffer_size(
            handle, descr.d, A, X, Y, rocsparse_sptrsm_stage_analysis, &bs, nullptr);
        if(!expect_ran(st, "sptrsm buffer") || st != rocsparse_status_success)
            return st == rocsparse_status_not_implemented;
        device_vector<char> buf(bs ? bs : size_t{1});
        if(!buf.ptr)
        {
            ADD_FAILURE() << "sptrsm buffer";
            return false;
        }
        void* p = bs ? static_cast<void*>(buf.ptr) : nullptr;
        st = rocsparse_sptrsm(handle, descr.d, A, X, Y, rocsparse_sptrsm_stage_analysis, bs, p, nullptr);
        if(!expect_ran(st, "sptrsm analysis") || st != rocsparse_status_success)
            return st == rocsparse_status_not_implemented;

        bs = 0;
        st = rocsparse_sptrsm_buffer_size(
            handle, descr.d, A, X, Y, rocsparse_sptrsm_stage_compute, &bs, nullptr);
        if(!expect_ran(st, "sptrsm compute buffer") || st != rocsparse_status_success)
            return st == rocsparse_status_not_implemented;
        device_vector<char> bufc(bs ? bs : size_t{1});
        if(!bufc.ptr)
        {
            ADD_FAILURE() << "sptrsm compute buffer";
            return false;
        }
        void* pc = bs ? static_cast<void*>(bufc.ptr) : nullptr;
        st = rocsparse_sptrsm(
            handle, descr.d, A, X, Y, rocsparse_sptrsm_stage_compute, bs, pc, nullptr);
        if(hipDeviceSynchronize() != hipSuccess)
        {
            ADD_FAILURE() << "sptrsm sync";
            return false;
        }
        if(!expect_ran(st, "sptrsm compute"))
            return false;

        rocsparse_pointer_mode saved_mode = rocsparse_pointer_mode_host;
        if(rocsparse_get_pointer_mode(handle, &saved_mode) != rocsparse_status_success)
        {
            ADD_FAILURE() << "get pointer mode";
            return false;
        }
        if(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host) != rocsparse_status_success)
        {
            ADD_FAILURE() << "host pointer mode";
            return false;
        }
        int64_t position = -1;
        st               = rocsparse_sptrsm_get_output(handle,
                                         descr.d,
                                         rocsparse_sptrsm_output_zero_pivot_position,
                                         &position,
                                         sizeof(position),
                                         nullptr);
        const bool pivot_ok = expect_ran(st, "sptrsm zero pivot");
        if(rocsparse_set_pointer_mode(handle, saved_mode) != rocsparse_status_success)
        {
            ADD_FAILURE() << "restore pointer mode";
            return false;
        }
        return pivot_ok;
    }
} // namespace

TEST_F(SptrsmDispatch, cases)
{
    Id id;
    ASSERT_TRUE(id.ok());
    const float alpha = 1.f;
    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
              rocsparse_status_success);

    // (op_x, order_x, order_y) covers both clauses of B_is_transposed and both
    // C orders, which is the four sptrsm_case values.
    struct Cfg
    {
        rocsparse_operation op_x;
        rocsparse_order     order_x;
        rocsparse_order     order_y;
        int64_t             nrhs;
    };
    const Cfg cfgs[] = {
        {rocsparse_operation_none, rocsparse_order_column, rocsparse_order_column, 2},
        {rocsparse_operation_none, rocsparse_order_row, rocsparse_order_row, 2},
        {rocsparse_operation_transpose, rocsparse_order_column, rocsparse_order_column, 2},
        {rocsparse_operation_transpose, rocsparse_order_row, rocsparse_order_row, 2},
        {rocsparse_operation_conjugate_transpose, rocsparse_order_column, rocsparse_order_row, 2},
        {rocsparse_operation_conjugate_transpose, rocsparse_order_row, rocsparse_order_column, 2},
        {rocsparse_operation_none, rocsparse_order_column, rocsparse_order_column, 0},
        {rocsparse_operation_none, rocsparse_order_row, rocsparse_order_column, 0},
        {rocsparse_operation_transpose, rocsparse_order_column, rocsparse_order_row, 0},
        {rocsparse_operation_transpose, rocsparse_order_row, rocsparse_order_row, 0},
    };

    const rocsparse_operation op_as[]
        = {rocsparse_operation_none, rocsparse_operation_transpose, rocsparse_operation_conjugate_transpose};
    const rocsparse_analysis_policy policies[]
        = {rocsparse_analysis_policy_reuse, rocsparse_analysis_policy_force};

    auto run_format = [&](auto create) {
        for(const Cfg& cfg : cfgs)
        {
            for(rocsparse_operation op_a : op_as)
            {
                for(rocsparse_analysis_policy policy : policies)
                {
                    SpMat A;
                    ASSERT_TRUE(create(A));
                    ASSERT_TRUE(
                        set_tri(A.d, rocsparse_fill_mode_lower, rocsparse_diag_type_non_unit));

                    // B stored shape follows op_x; solution is always m x nrhs.
                    const int64_t m    = Id::n;
                    const int64_t nrhs = cfg.nrhs;
                    Layout        bx   = rhs_layout(m, nrhs, cfg.op_x, cfg.order_x);
                    Layout        by   = sol_layout(m, nrhs, cfg.order_y);
                    // rhs_layout used m as the untransposed row count, which matches
                    // SpTrSM (A is m x m, nrhs is K).
                    device_vector<float> xv(std::vector<float>(static_cast<size_t>(bx.n), 1.f));
                    device_vector<float> yv(std::vector<float>(static_cast<size_t>(by.n), 0.f));
                    ASSERT_TRUE(xv.ptr && yv.ptr);
                    DnMat X;
                    DnMat Y;
                    ASSERT_EQ(rocsparse_create_dnmat_descr(&X.d,
                                                           bx.rows,
                                                           bx.cols,
                                                           bx.ld,
                                                           xv,
                                                           rocsparse_datatype_f32_r,
                                                           cfg.order_x),
                              rocsparse_status_success);
                    ASSERT_EQ(rocsparse_create_dnmat_descr(&Y.d,
                                                           by.rows,
                                                           by.cols,
                                                           by.ld,
                                                           yv,
                                                           rocsparse_datatype_f32_r,
                                                           cfg.order_y),
                              rocsparse_status_success);
                    ASSERT_TRUE(run_sptrsm(handle,
                                           A.d,
                                           X.d,
                                           Y.d,
                                           op_a,
                                           cfg.op_x,
                                           rocsparse_datatype_f32_r,
                                           rocsparse_datatype_f32_r,
                                           policy,
                                           &alpha,
                                           false));
                }
            }
        }
    };

    run_format([&](SpMat& A) { return make_csr(A, id); });
    run_format([&](SpMat& A) {
        return rocsparse_create_coo_descr(&A.d,
                                          Id::n,
                                          Id::n,
                                          Id::nnz,
                                          id.ci,
                                          id.ci,
                                          id.val,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r)
               == rocsparse_status_success;
    });
    run_format([&](SpMat& A) {
        return rocsparse_create_csc_descr(&A.d,
                                          Id::n,
                                          Id::n,
                                          Id::nnz,
                                          id.rp,
                                          id.ci,
                                          id.val,
                                          rocsparse_indextype_i32,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r)
               == rocsparse_status_success;
    });

    // Diagonal solve on CSR, one layout.
    {
        SpMat A;
        ASSERT_TRUE(make_csr(A, id));
        ASSERT_TRUE(set_tri(A.d, rocsparse_fill_mode_upper, rocsparse_diag_type_unit));
        Layout               bx = rhs_layout(Id::n, 2, rocsparse_operation_none, rocsparse_order_column);
        Layout               by = sol_layout(Id::n, 2, rocsparse_order_column);
        device_vector<float> xv(std::vector<float>(static_cast<size_t>(bx.n), 1.f));
        device_vector<float> yv(std::vector<float>(static_cast<size_t>(by.n), 0.f));
        DnMat                X;
        DnMat                Y;
        ASSERT_EQ(rocsparse_create_dnmat_descr(
                      &X.d, bx.rows, bx.cols, bx.ld, xv, rocsparse_datatype_f32_r, rocsparse_order_column),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnmat_descr(
                      &Y.d, by.rows, by.cols, by.ld, yv, rocsparse_datatype_f32_r, rocsparse_order_column),
                  rocsparse_status_success);
        ASSERT_TRUE(run_sptrsm(handle,
                               A.d,
                               X.d,
                               Y.d,
                               rocsparse_operation_none,
                               rocsparse_operation_none,
                               rocsparse_datatype_f32_r,
                               rocsparse_datatype_f32_r,
                               rocsparse_analysis_policy_force,
                               &alpha,
                               true));
    }

    // batch_count != 1 on A, X, and Y.
    {
        SpMat A;
        ASSERT_TRUE(make_csr(A, id));
        Layout               bx = rhs_layout(Id::n, 1, rocsparse_operation_none, rocsparse_order_column);
        Layout               by = sol_layout(Id::n, 1, rocsparse_order_column);
        device_vector<float> xv(std::vector<float>(static_cast<size_t>(bx.n), 1.f));
        device_vector<float> yv(std::vector<float>(static_cast<size_t>(by.n), 0.f));
        DnMat                X;
        DnMat                Y;
        ASSERT_EQ(rocsparse_create_dnmat_descr(
                      &X.d, bx.rows, bx.cols, bx.ld, xv, rocsparse_datatype_f32_r, rocsparse_order_column),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnmat_descr(
                      &Y.d, by.rows, by.cols, by.ld, yv, rocsparse_datatype_f32_r, rocsparse_order_column),
                  rocsparse_status_success);
        SptrsmDescr descr;
        ASSERT_EQ(rocsparse_create_sptrsm_descr(&descr.d), rocsparse_status_success);
        ASSERT_TRUE(configure_sptrsm(handle,
                                     descr.d,
                                     rocsparse_operation_none,
                                     rocsparse_operation_none,
                                     rocsparse_datatype_f32_r,
                                     rocsparse_datatype_f32_r,
                                     rocsparse_analysis_policy_reuse,
                                     &alpha));
        // The batch_count != 1 checks live on rocsparse_sptrsm, not buffer_size.
        auto expect_batch = [&](rocsparse_sptrsm_stage stage) {
            EXPECT_EQ(rocsparse_sptrsm(
                          handle, descr.d, A.d, X.d, Y.d, stage, 0, nullptr, nullptr),
                      rocsparse_status_not_implemented);
        };
        ASSERT_EQ(rocsparse_csr_set_strided_batch(A.d, 2, Id::n + 1, Id::nnz),
                  rocsparse_status_success);
        expect_batch(rocsparse_sptrsm_stage_analysis);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(A.d, 1, Id::n + 1, Id::nnz),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(X.d, 2, bx.n), rocsparse_status_success);
        expect_batch(rocsparse_sptrsm_stage_analysis);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(X.d, 1, bx.n), rocsparse_status_success);
        ASSERT_EQ(rocsparse_dnmat_set_strided_batch(Y.d, 2, by.n), rocsparse_status_success);
        expect_batch(rocsparse_sptrsm_stage_compute);
    }
}

TEST_F(SptrsmDispatch, scalar_mismatch)
{
    device_vector<rocsparse_int> rp{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> ci{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<double>        val{std::vector<double>{1.0, 1.0, 1.0}};
    ASSERT_TRUE(rp.ptr && ci.ptr && val.ptr);
    SpMat A;
    ASSERT_EQ(rocsparse_create_csr_descr(&A.d,
                                         3,
                                         3,
                                         3,
                                         rp,
                                         ci,
                                         val,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f64_r),
              rocsparse_status_success);
    ASSERT_TRUE(set_tri(A.d, rocsparse_fill_mode_lower, rocsparse_diag_type_unit));

    Layout               bx = rhs_layout(3, 2, rocsparse_operation_none, rocsparse_order_column);
    Layout               by = sol_layout(3, 2, rocsparse_order_row);
    device_vector<double> xv(std::vector<double>(static_cast<size_t>(bx.n), 1.0));
    device_vector<double> yv(std::vector<double>(static_cast<size_t>(by.n), 0.0));
    DnMat                X;
    DnMat                Y;
    ASSERT_EQ(rocsparse_create_dnmat_descr(
                  &X.d, bx.rows, bx.cols, bx.ld, xv, rocsparse_datatype_f64_r, rocsparse_order_column),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnmat_descr(
                  &Y.d, by.rows, by.cols, by.ld, yv, rocsparse_datatype_f64_r, rocsparse_order_row),
              rocsparse_status_success);

    const float          alpha_h = 1.f;
    device_vector<float> alpha_d{std::vector<float>{1.f}};
    ASSERT_TRUE(alpha_d.ptr);

    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
              rocsparse_status_success);
    ASSERT_TRUE(run_sptrsm(handle,
                           A.d,
                           X.d,
                           Y.d,
                           rocsparse_operation_conjugate_transpose,
                           rocsparse_operation_none,
                           rocsparse_datatype_f32_r,
                           rocsparse_datatype_f64_r,
                           rocsparse_analysis_policy_reuse,
                           &alpha_h,
                           false));

    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
              rocsparse_status_success);
    ASSERT_TRUE(run_sptrsm(handle,
                           A.d,
                           X.d,
                           Y.d,
                           rocsparse_operation_none,
                           rocsparse_operation_none,
                           rocsparse_datatype_f32_r,
                           rocsparse_datatype_f64_r,
                           rocsparse_analysis_policy_force,
                           alpha_d.ptr,
                           false));
}

class CsrsmSolveDispatch : public HandleTest
{
};

TEST_F(CsrsmSolveDispatch, blockdims)
{
    // blockdim starts at 512 and shifts until it is the smallest power of two
    // that is >= nrhs and >= 64, then the matching kernel is launched. nrhs
    // 1 stays on the csrsv path; 2, 65, 129, 257, 513 select 64..1024.
    device_vector<rocsparse_int> rp{std::vector<rocsparse_int>{0, 1, 2, 3, 4}};
    device_vector<rocsparse_int> ci{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<float>         val{std::vector<float>{1.f, 1.f, 1.f, 1.f}};
    ASSERT_TRUE(rp.ptr && ci.ptr && val.ptr);

    MatDescr descr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr.d), rocsparse_status_success);

    const rocsparse_int            m   = 4;
    const rocsparse_int            nnz = 4;
    const rocsparse_int            ks[] = {0, 1, 2, 65, 129, 257, 513};
    const rocsparse_operation      tas[]
        = {rocsparse_operation_none, rocsparse_operation_transpose, rocsparse_operation_conjugate_transpose};
    const rocsparse_operation      tbs[]
        = {rocsparse_operation_none, rocsparse_operation_transpose, rocsparse_operation_conjugate_transpose};
    const rocsparse_fill_mode      fills[] = {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper};
    const rocsparse_diag_type      diags[] = {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit};
    const rocsparse_analysis_policy pols[]
        = {rocsparse_analysis_policy_reuse, rocsparse_analysis_policy_force};

    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
              rocsparse_status_success);
    const float alpha = 1.f;

    for(rocsparse_int nrhs : ks)
    {
        for(rocsparse_operation ta : tas)
        {
            for(rocsparse_operation tb : tbs)
            {
                for(rocsparse_fill_mode fill : fills)
                {
                    for(rocsparse_diag_type diag : diags)
                    {
                        for(rocsparse_analysis_policy pol : pols)
                        {
                            ASSERT_EQ(rocsparse_set_mat_fill_mode(descr.d, fill),
                                      rocsparse_status_success);
                            ASSERT_EQ(rocsparse_set_mat_diag_type(descr.d, diag),
                                      rocsparse_status_success);

                            const rocsparse_int ldb
                                = (tb == rocsparse_operation_none) ? m : (nrhs > 0 ? nrhs : 1);
                            const size_t nb = static_cast<size_t>(ldb) * static_cast<size_t>(nrhs > 0 ? nrhs : 1);
                            device_vector<float> B(std::vector<float>(nb, 1.f));
                            ASSERT_TRUE(B.ptr);
                            MatInfo info;
                            ASSERT_EQ(rocsparse_create_mat_info(&info.d), rocsparse_status_success);

                            size_t bs = 0;
                            rocsparse_status st
                                = rocsparse_scsrsm_buffer_size(handle,
                                                               ta,
                                                               tb,
                                                               m,
                                                               nrhs,
                                                               nnz,
                                                               &alpha,
                                                               descr.d,
                                                               val,
                                                               rp,
                                                               ci,
                                                               B,
                                                               ldb,
                                                               info.d,
                                                               rocsparse_solve_policy_auto,
                                                               &bs);
                            if(!expect_ran(st, "csrsm buffer") || st != rocsparse_status_success)
                                continue;
                            device_vector<char> buf(bs ? bs : size_t{1});
                            ASSERT_TRUE(buf.ptr);
                            st = rocsparse_scsrsm_analysis(handle,
                                                           ta,
                                                           tb,
                                                           m,
                                                           nrhs,
                                                           nnz,
                                                           &alpha,
                                                           descr.d,
                                                           val,
                                                           rp,
                                                           ci,
                                                           B,
                                                           ldb,
                                                           info.d,
                                                           pol,
                                                           rocsparse_solve_policy_auto,
                                                           bs ? static_cast<void*>(buf.ptr) : nullptr);
                            if(!expect_ran(st, "csrsm analysis") || st != rocsparse_status_success)
                                continue;
                            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                            st = rocsparse_scsrsm_solve(handle,
                                                        ta,
                                                        tb,
                                                        m,
                                                        nrhs,
                                                        nnz,
                                                        &alpha,
                                                        descr.d,
                                                        val,
                                                        rp,
                                                        ci,
                                                        B,
                                                        ldb,
                                                        info.d,
                                                        rocsparse_solve_policy_auto,
                                                        bs ? static_cast<void*>(buf.ptr) : nullptr);
                            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                            ASSERT_TRUE(expect_ran(st, "csrsm solve"));

                            rocsparse_int pivot = -1;
                            st = rocsparse_csrsm_zero_pivot(handle, info.d, &pivot);
                            ASSERT_TRUE(expect_ran(st, "csrsm zero pivot"));
                            EXPECT_EQ(rocsparse_csrsm_clear(handle, info.d), rocsparse_status_success);
                        }
                    }
                }
            }
        }
    }

    // m == 0 quick return, before the pointer checks inside the impl.
    {
        MatInfo info;
        ASSERT_EQ(rocsparse_create_mat_info(&info.d), rocsparse_status_success);
        size_t bs = 0;
        device_vector<float> dummy{std::vector<float>{0.f}};
        device_vector<rocsparse_int> one{std::vector<rocsparse_int>{0}};
        ASSERT_TRUE(dummy.ptr && one.ptr);
        rocsparse_status st = rocsparse_scsrsm_buffer_size(handle,
                                                           rocsparse_operation_none,
                                                           rocsparse_operation_none,
                                                           0,
                                                           0,
                                                           0,
                                                           &alpha,
                                                           descr.d,
                                                           dummy,
                                                           one,
                                                           dummy.ptr ? reinterpret_cast<rocsparse_int*>(one.ptr) : nullptr,
                                                           dummy,
                                                           1,
                                                           info.d,
                                                           rocsparse_solve_policy_auto,
                                                           &bs);
        ASSERT_TRUE(expect_ran(st, "csrsm m0 buffer"));
        st = rocsparse_scsrsm_solve(handle,
                                    rocsparse_operation_none,
                                    rocsparse_operation_transpose,
                                    0,
                                    1,
                                    0,
                                    &alpha,
                                    descr.d,
                                    dummy,
                                    one,
                                    one,
                                    dummy,
                                    1,
                                    info.d,
                                    rocsparse_solve_policy_auto,
                                    dummy.ptr);
        ASSERT_TRUE(expect_ran(st, "csrsm m0 solve"));
    }
}
