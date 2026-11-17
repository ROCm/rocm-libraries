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
// Host-path unit tests for the GENERIC descriptor-based C API
// (library/src/generic + library/include/internal/generic).
//
// These drive the public generic entry points -- rocsparse_spmv, spmm, spgemm,
// spsv, spsm, sddmm, sparse_to_dense, dense_to_sparse, check_spmat -- so they
// exercise the *host* dispatch / validation / analysis code that rocSPARSE code
// coverage counts (coverage is host-only). Every routine gets at least one valid
// TINY call plus argument-validation guards (invalid_handle / invalid_pointer /
// invalid_value). Inputs are kept tiny (3x3 identity / diagonal) so the card's
// memory guard is never triggered.
//
// The generic level-1 vector ops (rot/gather/scatter/axpby) are already covered
// by unit_test_host_level1_generic.cpp and are intentionally NOT duplicated here.
//
#include "unit_test_utils.hpp"

using namespace rocsparse_ut;

namespace
{
    // ---- shared tiny constants ---------------------------------------------
    constexpr rocsparse_indextype  IT   = rocsparse_indextype_i32;
    constexpr rocsparse_index_base BASE = rocsparse_index_base_zero;
    constexpr rocsparse_datatype   DT   = rocsparse_datatype_f32_r;

    // A 3x3 identity in CSR, owning its device buffers + the spmat descriptor.
    struct IdentityCsr
    {
        device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1, 2, 3}};
        device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1, 2}};
        device_vector<float>   val{std::vector<float>{1.0f, 1.0f, 1.0f}};
        rocsparse_spmat_descr  A = nullptr;

        bool create()
        {
            if(!row_ptr.ptr || !col_ind.ptr || !val.ptr)
                return false;
            return rocsparse_create_csr_descr(&A, 3, 3, 3, row_ptr, col_ind, val, IT, IT, BASE, DT)
                   == rocsparse_status_success;
        }
        ~IdentityCsr()
        {
            if(A)
                (void)rocsparse_destroy_spmat_descr(A);
        }
    };

    // A 3x3 identity in COO, owning its device buffers + the spmat descriptor.
    struct IdentityCoo
    {
        device_vector<int32_t> row_ind{std::vector<int32_t>{0, 1, 2}};
        device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1, 2}};
        device_vector<float>   val{std::vector<float>{1.0f, 1.0f, 1.0f}};
        rocsparse_spmat_descr  A = nullptr;

        bool create()
        {
            if(!row_ind.ptr || !col_ind.ptr || !val.ptr)
                return false;
            return rocsparse_create_coo_descr(&A, 3, 3, 3, row_ind, col_ind, val, IT, BASE, DT)
                   == rocsparse_status_success;
        }
        ~IdentityCoo()
        {
            if(A)
                (void)rocsparse_destroy_spmat_descr(A);
        }
    };
} // namespace

// ===========================================================================
// rocsparse_spmv : y = alpha*A*x + beta*y   (A = 3x3 identity)
// ===========================================================================
class GenericSpmv : public HandleTest
{
};

TEST_F(GenericSpmv, csr_buffer_size_then_compute)
{
    IdentityCsr mat;
    ASSERT_TRUE(mat.create());

    device_vector<float> x{std::vector<float>{1.0f, 2.0f, 3.0f}};
    device_vector<float> y{std::vector<float>{0.0f, 0.0f, 0.0f}};
    ASSERT_TRUE(x.ptr && y.ptr);

    rocsparse_dnvec_descr vx = nullptr, vy = nullptr;
    ASSERT_EQ(rocsparse_create_dnvec_descr(&vx, 3, x.ptr, DT), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnvec_descr(&vy, 3, y.ptr, DT), rocsparse_status_success);

    const float alpha = 1.0f, beta = 0.0f;
    size_t      buffer_size = 0;
    ASSERT_EQ(rocsparse_spmv(handle,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             vx,
                             &beta,
                             vy,
                             DT,
                             rocsparse_spmv_alg_default,
                             rocsparse_spmv_stage_buffer_size,
                             &buffer_size,
                             nullptr),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    ASSERT_EQ(rocsparse_spmv(handle,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             vx,
                             &beta,
                             vy,
                             DT,
                             rocsparse_spmv_alg_default,
                             rocsparse_spmv_stage_preprocess,
                             &buffer_size,
                             tmp.ptr),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_spmv(handle,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             vx,
                             &beta,
                             vy,
                             DT,
                             rocsparse_spmv_alg_default,
                             rocsparse_spmv_stage_compute,
                             &buffer_size,
                             tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // y == x for the identity matrix.
    std::vector<float> hy(3, -1.0f);
    UT_CHECK_HIP(hipMemcpy(hy.data(), y.ptr, 3 * sizeof(float), hipMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(hy[0], 1.0f);
    EXPECT_FLOAT_EQ(hy[1], 2.0f);
    EXPECT_FLOAT_EQ(hy[2], 3.0f);

    EXPECT_EQ(rocsparse_destroy_dnvec_descr(vx), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnvec_descr(vy), rocsparse_status_success);
}

TEST_F(GenericSpmv, coo_buffer_size_then_compute)
{
    IdentityCoo mat;
    ASSERT_TRUE(mat.create());

    device_vector<float> x{std::vector<float>{4.0f, 5.0f, 6.0f}};
    device_vector<float> y{std::vector<float>{0.0f, 0.0f, 0.0f}};
    ASSERT_TRUE(x.ptr && y.ptr);

    rocsparse_dnvec_descr vx = nullptr, vy = nullptr;
    ASSERT_EQ(rocsparse_create_dnvec_descr(&vx, 3, x.ptr, DT), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnvec_descr(&vy, 3, y.ptr, DT), rocsparse_status_success);

    const float alpha = 1.0f, beta = 0.0f;
    size_t      buffer_size = 0;
    ASSERT_EQ(rocsparse_spmv(handle,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             vx,
                             &beta,
                             vy,
                             DT,
                             rocsparse_spmv_alg_default,
                             rocsparse_spmv_stage_buffer_size,
                             &buffer_size,
                             nullptr),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    ASSERT_EQ(rocsparse_spmv(handle,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             vx,
                             &beta,
                             vy,
                             DT,
                             rocsparse_spmv_alg_default,
                             rocsparse_spmv_stage_preprocess,
                             &buffer_size,
                             tmp.ptr),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_spmv(handle,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             vx,
                             &beta,
                             vy,
                             DT,
                             rocsparse_spmv_alg_default,
                             rocsparse_spmv_stage_compute,
                             &buffer_size,
                             tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_destroy_dnvec_descr(vx), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnvec_descr(vy), rocsparse_status_success);
}

TEST_F(GenericSpmv, bad_args)
{
    IdentityCsr mat;
    ASSERT_TRUE(mat.create());

    device_vector<float>  x{std::vector<float>{1.0f, 2.0f, 3.0f}};
    device_vector<float>  y{std::vector<float>{0.0f, 0.0f, 0.0f}};
    rocsparse_dnvec_descr vx = nullptr, vy = nullptr;
    ASSERT_EQ(rocsparse_create_dnvec_descr(&vx, 3, x.ptr, DT), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnvec_descr(&vy, 3, y.ptr, DT), rocsparse_status_success);

    const float alpha = 1.0f, beta = 0.0f;
    size_t      buffer_size = 0;

    // null handle -> invalid_handle
    EXPECT_EQ(rocsparse_spmv(nullptr,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             vx,
                             &beta,
                             vy,
                             DT,
                             rocsparse_spmv_alg_default,
                             rocsparse_spmv_stage_buffer_size,
                             &buffer_size,
                             nullptr),
              rocsparse_status_invalid_handle);

    // null matrix descriptor -> invalid_pointer
    EXPECT_EQ(rocsparse_spmv(handle,
                             rocsparse_operation_none,
                             &alpha,
                             nullptr,
                             vx,
                             &beta,
                             vy,
                             DT,
                             rocsparse_spmv_alg_default,
                             rocsparse_spmv_stage_buffer_size,
                             &buffer_size,
                             nullptr),
              rocsparse_status_invalid_pointer);

    // null buffer_size out-param -> invalid_pointer
    EXPECT_EQ(rocsparse_spmv(handle,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             vx,
                             &beta,
                             vy,
                             DT,
                             rocsparse_spmv_alg_default,
                             rocsparse_spmv_stage_buffer_size,
                             nullptr,
                             nullptr),
              rocsparse_status_invalid_pointer);

    // invalid operation enum -> invalid_value
    EXPECT_EQ(rocsparse_spmv(handle,
                             (rocsparse_operation)99,
                             &alpha,
                             mat.A,
                             vx,
                             &beta,
                             vy,
                             DT,
                             rocsparse_spmv_alg_default,
                             rocsparse_spmv_stage_buffer_size,
                             &buffer_size,
                             nullptr),
              rocsparse_status_invalid_value);

    EXPECT_EQ(rocsparse_destroy_dnvec_descr(vx), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnvec_descr(vy), rocsparse_status_success);
}

// ===========================================================================
// rocsparse_spmm : C = alpha*A*B + beta*C  (A = 3x3 identity, B/C dense 3x2)
// ===========================================================================
class GenericSpmm : public HandleTest
{
};

TEST_F(GenericSpmm, csr_dense_buffer_size_then_compute)
{
    IdentityCsr mat;
    ASSERT_TRUE(mat.create());

    const int n = 2; // number of RHS columns
    // column-major 3x2 dense B and C
    device_vector<float> B{std::vector<float>{1, 2, 3, 4, 5, 6}};
    device_vector<float> C{std::vector<float>(3 * n, 0.0f)};
    ASSERT_TRUE(B.ptr && C.ptr);

    rocsparse_dnmat_descr mB = nullptr, mC = nullptr;
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mB, 3, n, 3, B.ptr, DT, rocsparse_order_column),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mC, 3, n, 3, C.ptr, DT, rocsparse_order_column),
              rocsparse_status_success);

    const float alpha = 1.0f, beta = 0.0f;
    size_t      buffer_size = 0;
    ASSERT_EQ(rocsparse_spmm(handle,
                             rocsparse_operation_none,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             mB,
                             &beta,
                             mC,
                             DT,
                             rocsparse_spmm_alg_default,
                             rocsparse_spmm_stage_buffer_size,
                             &buffer_size,
                             nullptr),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    ASSERT_EQ(rocsparse_spmm(handle,
                             rocsparse_operation_none,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             mB,
                             &beta,
                             mC,
                             DT,
                             rocsparse_spmm_alg_default,
                             rocsparse_spmm_stage_preprocess,
                             &buffer_size,
                             tmp.ptr),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_spmm(handle,
                             rocsparse_operation_none,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             mB,
                             &beta,
                             mC,
                             DT,
                             rocsparse_spmm_alg_default,
                             rocsparse_spmm_stage_compute,
                             &buffer_size,
                             tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // C == B for the identity matrix.
    std::vector<float> hC(3 * n, -1.0f);
    UT_CHECK_HIP(hipMemcpy(hC.data(), C.ptr, hC.size() * sizeof(float), hipMemcpyDeviceToHost));
    for(int i = 0; i < 3 * n; ++i)
        EXPECT_FLOAT_EQ(hC[i], static_cast<float>(i + 1));

    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mB), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mC), rocsparse_status_success);
}

TEST_F(GenericSpmm, bad_args)
{
    IdentityCsr mat;
    ASSERT_TRUE(mat.create());

    const int             n = 2;
    device_vector<float>  B{std::vector<float>(3 * n, 1.0f)};
    device_vector<float>  C{std::vector<float>(3 * n, 0.0f)};
    rocsparse_dnmat_descr mB = nullptr, mC = nullptr;
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mB, 3, n, 3, B.ptr, DT, rocsparse_order_column),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mC, 3, n, 3, C.ptr, DT, rocsparse_order_column),
              rocsparse_status_success);

    const float alpha = 1.0f, beta = 0.0f;
    size_t      buffer_size = 0;

    EXPECT_EQ(rocsparse_spmm(nullptr,
                             rocsparse_operation_none,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             mB,
                             &beta,
                             mC,
                             DT,
                             rocsparse_spmm_alg_default,
                             rocsparse_spmm_stage_buffer_size,
                             &buffer_size,
                             nullptr),
              rocsparse_status_invalid_handle);

    EXPECT_EQ(rocsparse_spmm(handle,
                             rocsparse_operation_none,
                             rocsparse_operation_none,
                             &alpha,
                             nullptr,
                             mB,
                             &beta,
                             mC,
                             DT,
                             rocsparse_spmm_alg_default,
                             rocsparse_spmm_stage_buffer_size,
                             &buffer_size,
                             nullptr),
              rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_spmm(handle,
                             rocsparse_operation_none,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             mB,
                             &beta,
                             mC,
                             DT,
                             rocsparse_spmm_alg_default,
                             rocsparse_spmm_stage_buffer_size,
                             nullptr,
                             nullptr),
              rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mB), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mC), rocsparse_status_success);
}

// ===========================================================================
// rocsparse_spgemm : C = alpha*A*B + beta*D  (A = B = 3x3 identity, D empty)
// ===========================================================================
class GenericSpgemm : public HandleTest
{
};

TEST_F(GenericSpgemm, csr_identity_full_pipeline)
{
    IdentityCsr matA, matB;
    ASSERT_TRUE(matA.create());
    ASSERT_TRUE(matB.create());

    // C: only row_ptr initially allocated (nnz unknown), col/val set later.
    device_vector<int32_t> C_row_ptr{(size_t)4};
    ASSERT_TRUE(C_row_ptr.ptr);

    rocsparse_spmat_descr matC = nullptr, matD = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(
                  &matC, 3, 3, 0, C_row_ptr.ptr, nullptr, nullptr, IT, IT, BASE, DT),
              rocsparse_status_success);
    // Empty D (beta = 0).
    ASSERT_EQ(
        rocsparse_create_csr_descr(&matD, 0, 0, 0, nullptr, nullptr, nullptr, IT, IT, BASE, DT),
        rocsparse_status_success);

    const float alpha = 1.0f, beta = 0.0f;
    size_t      buffer_size = 0;
    ASSERT_EQ(rocsparse_spgemm(handle,
                               rocsparse_operation_none,
                               rocsparse_operation_none,
                               &alpha,
                               matA.A,
                               matB.A,
                               &beta,
                               matD,
                               matC,
                               DT,
                               rocsparse_spgemm_alg_default,
                               rocsparse_spgemm_stage_buffer_size,
                               &buffer_size,
                               nullptr),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    ASSERT_EQ(rocsparse_spgemm(handle,
                               rocsparse_operation_none,
                               rocsparse_operation_none,
                               &alpha,
                               matA.A,
                               matB.A,
                               &beta,
                               matD,
                               matC,
                               DT,
                               rocsparse_spgemm_alg_default,
                               rocsparse_spgemm_stage_nnz,
                               &buffer_size,
                               tmp.ptr),
              rocsparse_status_success);

    int64_t rows_C = 0, cols_C = 0, nnz_C = 0;
    ASSERT_EQ(rocsparse_spmat_get_size(matC, &rows_C, &cols_C, &nnz_C), rocsparse_status_success);
    EXPECT_EQ(nnz_C, 3); // identity * identity = identity

    device_vector<int32_t> C_col_ind{(size_t)(nnz_C > 0 ? nnz_C : 1)};
    device_vector<float>   C_val{(size_t)(nnz_C > 0 ? nnz_C : 1)};
    ASSERT_TRUE(C_col_ind.ptr && C_val.ptr);
    ASSERT_EQ(rocsparse_csr_set_pointers(matC, C_row_ptr.ptr, C_col_ind.ptr, C_val.ptr),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_spgemm(handle,
                               rocsparse_operation_none,
                               rocsparse_operation_none,
                               &alpha,
                               matA.A,
                               matB.A,
                               &beta,
                               matD,
                               matC,
                               DT,
                               rocsparse_spgemm_alg_default,
                               rocsparse_spgemm_stage_compute,
                               &buffer_size,
                               tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(matC), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(matD), rocsparse_status_success);
}

TEST_F(GenericSpgemm, bad_args)
{
    IdentityCsr matA, matB;
    ASSERT_TRUE(matA.create());
    ASSERT_TRUE(matB.create());

    device_vector<int32_t> C_row_ptr{(size_t)4};
    rocsparse_spmat_descr  matC = nullptr, matD = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(
                  &matC, 3, 3, 0, C_row_ptr.ptr, nullptr, nullptr, IT, IT, BASE, DT),
              rocsparse_status_success);
    ASSERT_EQ(
        rocsparse_create_csr_descr(&matD, 0, 0, 0, nullptr, nullptr, nullptr, IT, IT, BASE, DT),
        rocsparse_status_success);

    const float alpha = 1.0f, beta = 0.0f;
    size_t      buffer_size = 0;

    EXPECT_EQ(rocsparse_spgemm(nullptr,
                               rocsparse_operation_none,
                               rocsparse_operation_none,
                               &alpha,
                               matA.A,
                               matB.A,
                               &beta,
                               matD,
                               matC,
                               DT,
                               rocsparse_spgemm_alg_default,
                               rocsparse_spgemm_stage_buffer_size,
                               &buffer_size,
                               nullptr),
              rocsparse_status_invalid_handle);

    EXPECT_EQ(rocsparse_spgemm(handle,
                               rocsparse_operation_none,
                               rocsparse_operation_none,
                               &alpha,
                               nullptr,
                               matB.A,
                               &beta,
                               matD,
                               matC,
                               DT,
                               rocsparse_spgemm_alg_default,
                               rocsparse_spgemm_stage_buffer_size,
                               &buffer_size,
                               nullptr),
              rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(matC), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(matD), rocsparse_status_success);
}

// ===========================================================================
// rocsparse_spsv : solve op(A)*y = alpha*x  (A = 3x3 diagonal, triangular)
// ===========================================================================
namespace
{
    // 3x3 diagonal CSR (values 2,4,8) -- well-conditioned lower/upper triangular.
    struct DiagCsr
    {
        device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1, 2, 3}};
        device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1, 2}};
        device_vector<float>   val{std::vector<float>{2.0f, 4.0f, 8.0f}};
        rocsparse_spmat_descr  A = nullptr;

        bool create()
        {
            if(!row_ptr.ptr || !col_ind.ptr || !val.ptr)
                return false;
            if(rocsparse_create_csr_descr(&A, 3, 3, 3, row_ptr, col_ind, val, IT, IT, BASE, DT)
               != rocsparse_status_success)
                return false;
            // Mark as lower-triangular, non-unit diagonal for the solve.
            rocsparse_fill_mode fill = rocsparse_fill_mode_lower;
            rocsparse_diag_type diag = rocsparse_diag_type_non_unit;
            return rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &fill, sizeof(fill))
                       == rocsparse_status_success
                   && rocsparse_spmat_set_attribute(
                          A, rocsparse_spmat_diag_type, &diag, sizeof(diag))
                          == rocsparse_status_success;
        }
        ~DiagCsr()
        {
            if(A)
                (void)rocsparse_destroy_spmat_descr(A);
        }
    };
} // namespace

class GenericSpsv : public HandleTest
{
};

TEST_F(GenericSpsv, csr_diagonal_full_pipeline)
{
    DiagCsr mat;
    ASSERT_TRUE(mat.create());

    device_vector<float> x{std::vector<float>{2.0f, 4.0f, 8.0f}};
    device_vector<float> y{std::vector<float>{0.0f, 0.0f, 0.0f}};
    ASSERT_TRUE(x.ptr && y.ptr);

    rocsparse_dnvec_descr vx = nullptr, vy = nullptr;
    ASSERT_EQ(rocsparse_create_dnvec_descr(&vx, 3, x.ptr, DT), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnvec_descr(&vy, 3, y.ptr, DT), rocsparse_status_success);

    const float alpha       = 1.0f;
    size_t      buffer_size = 0;
    ASSERT_EQ(rocsparse_spsv(handle,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             vx,
                             vy,
                             DT,
                             rocsparse_spsv_alg_default,
                             rocsparse_spsv_stage_buffer_size,
                             &buffer_size,
                             nullptr),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    ASSERT_EQ(rocsparse_spsv(handle,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             vx,
                             vy,
                             DT,
                             rocsparse_spsv_alg_default,
                             rocsparse_spsv_stage_preprocess,
                             &buffer_size,
                             tmp.ptr),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_spsv(handle,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             vx,
                             vy,
                             DT,
                             rocsparse_spsv_alg_default,
                             rocsparse_spsv_stage_compute,
                             &buffer_size,
                             tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // diag(2,4,8) * y = x=(2,4,8) -> y = (1,1,1)
    std::vector<float> hy(3, -1.0f);
    UT_CHECK_HIP(hipMemcpy(hy.data(), y.ptr, 3 * sizeof(float), hipMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(hy[0], 1.0f);
    EXPECT_FLOAT_EQ(hy[1], 1.0f);
    EXPECT_FLOAT_EQ(hy[2], 1.0f);

    EXPECT_EQ(rocsparse_destroy_dnvec_descr(vx), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnvec_descr(vy), rocsparse_status_success);
}

TEST_F(GenericSpsv, bad_args)
{
    DiagCsr mat;
    ASSERT_TRUE(mat.create());

    device_vector<float>  x{std::vector<float>{2.0f, 4.0f, 8.0f}};
    device_vector<float>  y{std::vector<float>{0.0f, 0.0f, 0.0f}};
    rocsparse_dnvec_descr vx = nullptr, vy = nullptr;
    ASSERT_EQ(rocsparse_create_dnvec_descr(&vx, 3, x.ptr, DT), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnvec_descr(&vy, 3, y.ptr, DT), rocsparse_status_success);

    const float alpha       = 1.0f;
    size_t      buffer_size = 0;

    EXPECT_EQ(rocsparse_spsv(nullptr,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             vx,
                             vy,
                             DT,
                             rocsparse_spsv_alg_default,
                             rocsparse_spsv_stage_buffer_size,
                             &buffer_size,
                             nullptr),
              rocsparse_status_invalid_handle);

    EXPECT_EQ(rocsparse_spsv(handle,
                             rocsparse_operation_none,
                             &alpha,
                             nullptr,
                             vx,
                             vy,
                             DT,
                             rocsparse_spsv_alg_default,
                             rocsparse_spsv_stage_buffer_size,
                             &buffer_size,
                             nullptr),
              rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_destroy_dnvec_descr(vx), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnvec_descr(vy), rocsparse_status_success);
}

// ===========================================================================
// rocsparse_spsm : solve op(A)*C = alpha*op(B)  (A = 3x3 diagonal, B/C dense)
// ===========================================================================
class GenericSpsm : public HandleTest
{
};

TEST_F(GenericSpsm, csr_diagonal_full_pipeline)
{
    DiagCsr mat;
    ASSERT_TRUE(mat.create());

    const int            n = 1; // single RHS column
    device_vector<float> B{std::vector<float>{2.0f, 4.0f, 8.0f}};
    device_vector<float> C{std::vector<float>(3 * n, 0.0f)};
    ASSERT_TRUE(B.ptr && C.ptr);

    rocsparse_dnmat_descr mB = nullptr, mC = nullptr;
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mB, 3, n, 3, B.ptr, DT, rocsparse_order_column),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mC, 3, n, 3, C.ptr, DT, rocsparse_order_column),
              rocsparse_status_success);

    const float alpha       = 1.0f;
    size_t      buffer_size = 0;
    ASSERT_EQ(rocsparse_spsm(handle,
                             rocsparse_operation_none,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             mB,
                             mC,
                             DT,
                             rocsparse_spsm_alg_default,
                             rocsparse_spsm_stage_buffer_size,
                             &buffer_size,
                             nullptr),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    ASSERT_EQ(rocsparse_spsm(handle,
                             rocsparse_operation_none,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             mB,
                             mC,
                             DT,
                             rocsparse_spsm_alg_default,
                             rocsparse_spsm_stage_preprocess,
                             &buffer_size,
                             tmp.ptr),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_spsm(handle,
                             rocsparse_operation_none,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             mB,
                             mC,
                             DT,
                             rocsparse_spsm_alg_default,
                             rocsparse_spsm_stage_compute,
                             &buffer_size,
                             tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // diag(2,4,8) * C = B=(2,4,8) -> C = (1,1,1)
    std::vector<float> hC(3 * n, -1.0f);
    UT_CHECK_HIP(hipMemcpy(hC.data(), C.ptr, hC.size() * sizeof(float), hipMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(hC[0], 1.0f);
    EXPECT_FLOAT_EQ(hC[1], 1.0f);
    EXPECT_FLOAT_EQ(hC[2], 1.0f);

    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mB), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mC), rocsparse_status_success);
}

TEST_F(GenericSpsm, bad_args)
{
    DiagCsr mat;
    ASSERT_TRUE(mat.create());

    const int             n = 1;
    device_vector<float>  B{std::vector<float>{2.0f, 4.0f, 8.0f}};
    device_vector<float>  C{std::vector<float>(3 * n, 0.0f)};
    rocsparse_dnmat_descr mB = nullptr, mC = nullptr;
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mB, 3, n, 3, B.ptr, DT, rocsparse_order_column),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mC, 3, n, 3, C.ptr, DT, rocsparse_order_column),
              rocsparse_status_success);

    const float alpha       = 1.0f;
    size_t      buffer_size = 0;

    EXPECT_EQ(rocsparse_spsm(nullptr,
                             rocsparse_operation_none,
                             rocsparse_operation_none,
                             &alpha,
                             mat.A,
                             mB,
                             mC,
                             DT,
                             rocsparse_spsm_alg_default,
                             rocsparse_spsm_stage_buffer_size,
                             &buffer_size,
                             nullptr),
              rocsparse_status_invalid_handle);

    EXPECT_EQ(rocsparse_spsm(handle,
                             rocsparse_operation_none,
                             rocsparse_operation_none,
                             &alpha,
                             nullptr,
                             mB,
                             mC,
                             DT,
                             rocsparse_spsm_alg_default,
                             rocsparse_spsm_stage_buffer_size,
                             &buffer_size,
                             nullptr),
              rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mB), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mC), rocsparse_status_success);
}

// ===========================================================================
// rocsparse_sddmm : C = alpha*(A*B) .* spy(C) + beta*C   (A,B dense; C sparse)
// ===========================================================================
class GenericSddmm : public HandleTest
{
};

TEST_F(GenericSddmm, csr_full_pipeline)
{
    // C sparsity = 3x3 identity; A,B dense 3x3 (column-major).
    IdentityCsr matC;
    ASSERT_TRUE(matC.create());

    device_vector<float> A{std::vector<float>(3 * 3, 1.0f)};
    device_vector<float> B{std::vector<float>(3 * 3, 1.0f)};
    ASSERT_TRUE(A.ptr && B.ptr);

    rocsparse_dnmat_descr mA = nullptr, mB = nullptr;
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mA, 3, 3, 3, A.ptr, DT, rocsparse_order_column),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mB, 3, 3, 3, B.ptr, DT, rocsparse_order_column),
              rocsparse_status_success);

    const float alpha = 1.0f, beta = 0.0f;
    size_t      buffer_size = 0;
    ASSERT_EQ(rocsparse_sddmm_buffer_size(handle,
                                          rocsparse_operation_none,
                                          rocsparse_operation_none,
                                          &alpha,
                                          mA,
                                          mB,
                                          &beta,
                                          matC.A,
                                          DT,
                                          rocsparse_sddmm_alg_default,
                                          &buffer_size),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    ASSERT_EQ(rocsparse_sddmm_preprocess(handle,
                                         rocsparse_operation_none,
                                         rocsparse_operation_none,
                                         &alpha,
                                         mA,
                                         mB,
                                         &beta,
                                         matC.A,
                                         DT,
                                         rocsparse_sddmm_alg_default,
                                         tmp.ptr),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_sddmm(handle,
                              rocsparse_operation_none,
                              rocsparse_operation_none,
                              &alpha,
                              mA,
                              mB,
                              &beta,
                              matC.A,
                              DT,
                              rocsparse_sddmm_alg_default,
                              tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mA), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mB), rocsparse_status_success);
}

TEST_F(GenericSddmm, bad_args)
{
    IdentityCsr matC;
    ASSERT_TRUE(matC.create());

    device_vector<float>  A{std::vector<float>(3 * 3, 1.0f)};
    device_vector<float>  B{std::vector<float>(3 * 3, 1.0f)};
    rocsparse_dnmat_descr mA = nullptr, mB = nullptr;
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mA, 3, 3, 3, A.ptr, DT, rocsparse_order_column),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mB, 3, 3, 3, B.ptr, DT, rocsparse_order_column),
              rocsparse_status_success);

    const float alpha = 1.0f, beta = 0.0f;
    size_t      buffer_size = 0;

    EXPECT_EQ(rocsparse_sddmm_buffer_size(nullptr,
                                          rocsparse_operation_none,
                                          rocsparse_operation_none,
                                          &alpha,
                                          mA,
                                          mB,
                                          &beta,
                                          matC.A,
                                          DT,
                                          rocsparse_sddmm_alg_default,
                                          &buffer_size),
              rocsparse_status_invalid_handle);

    EXPECT_EQ(rocsparse_sddmm_buffer_size(handle,
                                          rocsparse_operation_none,
                                          rocsparse_operation_none,
                                          &alpha,
                                          nullptr,
                                          mB,
                                          &beta,
                                          matC.A,
                                          DT,
                                          rocsparse_sddmm_alg_default,
                                          &buffer_size),
              rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mA), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mB), rocsparse_status_success);
}

// ===========================================================================
// rocsparse_sparse_to_dense : CSR (3x3 identity) -> dense
// ===========================================================================
class GenericSparseToDense : public HandleTest
{
};

TEST_F(GenericSparseToDense, csr_to_dense)
{
    IdentityCsr mat;
    ASSERT_TRUE(mat.create());

    device_vector<float> dense{std::vector<float>(3 * 3, -1.0f)};
    ASSERT_TRUE(dense.ptr);

    rocsparse_dnmat_descr mB = nullptr;
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mB, 3, 3, 3, dense.ptr, DT, rocsparse_order_column),
              rocsparse_status_success);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_sparse_to_dense(
                  handle, mat.A, mB, rocsparse_sparse_to_dense_alg_default, &buffer_size, nullptr),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    ASSERT_EQ(rocsparse_sparse_to_dense(
                  handle, mat.A, mB, rocsparse_sparse_to_dense_alg_default, &buffer_size, tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // Column-major identity.
    std::vector<float> h(9, -2.0f);
    UT_CHECK_HIP(hipMemcpy(h.data(), dense.ptr, 9 * sizeof(float), hipMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(h[0], 1.0f);
    EXPECT_FLOAT_EQ(h[4], 1.0f);
    EXPECT_FLOAT_EQ(h[8], 1.0f);
    EXPECT_FLOAT_EQ(h[1], 0.0f);

    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mB), rocsparse_status_success);
}

TEST_F(GenericSparseToDense, bad_args)
{
    IdentityCsr mat;
    ASSERT_TRUE(mat.create());

    device_vector<float>  dense{std::vector<float>(3 * 3, 0.0f)};
    rocsparse_dnmat_descr mB = nullptr;
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mB, 3, 3, 3, dense.ptr, DT, rocsparse_order_column),
              rocsparse_status_success);

    size_t buffer_size = 0;

    EXPECT_EQ(rocsparse_sparse_to_dense(
                  nullptr, mat.A, mB, rocsparse_sparse_to_dense_alg_default, &buffer_size, nullptr),
              rocsparse_status_invalid_handle);

    EXPECT_EQ(
        rocsparse_sparse_to_dense(
            handle, nullptr, mB, rocsparse_sparse_to_dense_alg_default, &buffer_size, nullptr),
        rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mB), rocsparse_status_success);
}

// ===========================================================================
// rocsparse_dense_to_sparse : dense (3x3 identity) -> CSR
// ===========================================================================
class GenericDenseToSparse : public HandleTest
{
};

TEST_F(GenericDenseToSparse, dense_to_csr)
{
    // Column-major 3x3 identity dense input.
    std::vector<float> hdense(9, 0.0f);
    hdense[0] = hdense[4] = hdense[8] = 1.0f;
    device_vector<float> dense{hdense};
    ASSERT_TRUE(dense.ptr);

    rocsparse_dnmat_descr mA = nullptr;
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mA, 3, 3, 3, dense.ptr, DT, rocsparse_order_column),
              rocsparse_status_success);

    device_vector<int32_t> row_ptr{(size_t)4};
    ASSERT_TRUE(row_ptr.ptr);

    rocsparse_spmat_descr mB = nullptr;
    ASSERT_EQ(
        rocsparse_create_csr_descr(&mB, 3, 3, 0, row_ptr.ptr, nullptr, nullptr, IT, IT, BASE, DT),
        rocsparse_status_success);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_dense_to_sparse(
                  handle, mA, mB, rocsparse_dense_to_sparse_alg_default, &buffer_size, nullptr),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    // Analysis stage (buffer_size == nullptr) computes nnz / row_ptr.
    ASSERT_EQ(rocsparse_dense_to_sparse(
                  handle, mA, mB, rocsparse_dense_to_sparse_alg_default, nullptr, tmp.ptr),
              rocsparse_status_success);

    int64_t rows = 0, cols = 0, nnz = 0;
    ASSERT_EQ(rocsparse_spmat_get_size(mB, &rows, &cols, &nnz), rocsparse_status_success);
    EXPECT_EQ(nnz, 3);

    device_vector<int32_t> col_ind{(size_t)(nnz > 0 ? nnz : 1)};
    device_vector<float>   val{(size_t)(nnz > 0 ? nnz : 1)};
    ASSERT_TRUE(col_ind.ptr && val.ptr);
    ASSERT_EQ(rocsparse_csr_set_pointers(mB, row_ptr.ptr, col_ind.ptr, val.ptr),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_dense_to_sparse(
                  handle, mA, mB, rocsparse_dense_to_sparse_alg_default, &buffer_size, tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(mB), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mA), rocsparse_status_success);
}

TEST_F(GenericDenseToSparse, bad_args)
{
    device_vector<float>  dense{std::vector<float>(9, 0.0f)};
    rocsparse_dnmat_descr mA = nullptr;
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mA, 3, 3, 3, dense.ptr, DT, rocsparse_order_column),
              rocsparse_status_success);

    device_vector<int32_t> row_ptr{(size_t)4};
    rocsparse_spmat_descr  mB = nullptr;
    ASSERT_EQ(
        rocsparse_create_csr_descr(&mB, 3, 3, 0, row_ptr.ptr, nullptr, nullptr, IT, IT, BASE, DT),
        rocsparse_status_success);

    size_t buffer_size = 0;

    EXPECT_EQ(rocsparse_dense_to_sparse(
                  nullptr, mA, mB, rocsparse_dense_to_sparse_alg_default, &buffer_size, nullptr),
              rocsparse_status_invalid_handle);

    EXPECT_EQ(
        rocsparse_dense_to_sparse(
            handle, nullptr, mB, rocsparse_dense_to_sparse_alg_default, &buffer_size, nullptr),
        rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(mB), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mA), rocsparse_status_success);
}

// ===========================================================================
// rocsparse_check_spmat : validate a well-formed CSR / COO matrix.
// ===========================================================================
class GenericCheckSpmat : public HandleTest
{
};

TEST_F(GenericCheckSpmat, csr_valid)
{
    IdentityCsr mat;
    ASSERT_TRUE(mat.create());

    rocsparse_data_status data_status = rocsparse_data_status_inf;
    size_t                buffer_size = 0;
    ASSERT_EQ(rocsparse_check_spmat(handle,
                                    mat.A,
                                    &data_status,
                                    rocsparse_check_spmat_stage_buffer_size,
                                    &buffer_size,
                                    nullptr),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    ASSERT_EQ(rocsparse_check_spmat(handle,
                                    mat.A,
                                    &data_status,
                                    rocsparse_check_spmat_stage_compute,
                                    &buffer_size,
                                    tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(data_status, rocsparse_data_status_success);
}

TEST_F(GenericCheckSpmat, coo_valid)
{
    IdentityCoo mat;
    ASSERT_TRUE(mat.create());

    rocsparse_data_status data_status = rocsparse_data_status_inf;
    size_t                buffer_size = 0;
    ASSERT_EQ(rocsparse_check_spmat(handle,
                                    mat.A,
                                    &data_status,
                                    rocsparse_check_spmat_stage_buffer_size,
                                    &buffer_size,
                                    nullptr),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    ASSERT_EQ(rocsparse_check_spmat(handle,
                                    mat.A,
                                    &data_status,
                                    rocsparse_check_spmat_stage_compute,
                                    &buffer_size,
                                    tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(data_status, rocsparse_data_status_success);
}

TEST_F(GenericCheckSpmat, bad_args)
{
    IdentityCsr mat;
    ASSERT_TRUE(mat.create());

    rocsparse_data_status data_status = rocsparse_data_status_success;
    size_t                buffer_size = 0;

    EXPECT_EQ(rocsparse_check_spmat(nullptr,
                                    mat.A,
                                    &data_status,
                                    rocsparse_check_spmat_stage_buffer_size,
                                    &buffer_size,
                                    nullptr),
              rocsparse_status_invalid_handle);

    EXPECT_EQ(rocsparse_check_spmat(handle,
                                    nullptr,
                                    &data_status,
                                    rocsparse_check_spmat_stage_buffer_size,
                                    &buffer_size,
                                    nullptr),
              rocsparse_status_invalid_pointer);

    EXPECT_EQ(
        rocsparse_check_spmat(
            handle, mat.A, &data_status, (rocsparse_check_spmat_stage)99, &buffer_size, nullptr),
        rocsparse_status_invalid_value);
}
