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

// The rocSPARSE and rocBLAS wrappers called directly with device pointers and
// the library's handles, against host references

#include "unit_test_common.hpp"
#include "unit_test_reference.hpp"
#include "unit_test_types.hpp"

#include "base/hip/hip_blas.hpp"
#include "base/hip/hip_sparse.hpp"

#include <cmath>

using namespace rocalution_unit;

namespace
{
    // Device copy of a host array, freed on destruction
    template <typename T>
    class DeviceBuffer
    {
    public:
        explicit DeviceBuffer(const std::vector<T>& host)
            : size_(host.size())
        {
            EXPECT_EQ(hipMalloc(&ptr_, sizeof(T) * size_), hipSuccess);
            EXPECT_EQ(hipMemcpy(ptr_, host.data(), sizeof(T) * size_, hipMemcpyHostToDevice),
                      hipSuccess);
        }

        ~DeviceBuffer()
        {
            (void)hipFree(ptr_);
        }

        DeviceBuffer(const DeviceBuffer&)            = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        T* get()
        {
            return ptr_;
        }

        std::vector<T> to_host() const
        {
            std::vector<T> host(size_);
            EXPECT_EQ(hipMemcpy(host.data(), ptr_, sizeof(T) * size_, hipMemcpyDeviceToHost),
                      hipSuccess);
            return host;
        }

    private:
        T*     ptr_ = nullptr;
        size_t size_;
    };

    // hip_utils.hpp has ROCSPARSE_HANDLE and ROCBLAS_HANDLE for these, but it contains device code
    rocsparse_handle sparse_handle()
    {
        return *static_cast<rocsparse_handle*>(descriptor().ROC_sparse_handle);
    }

    rocblas_handle blas_handle()
    {
        return *static_cast<rocblas_handle*>(descriptor().ROC_blas_handle);
    }

    class MatDescr
    {
    public:
        MatDescr()
        {
            EXPECT_EQ(rocsparse_create_mat_descr(&descr_), rocsparse_status_success);
        }

        ~MatDescr()
        {
            (void)rocsparse_destroy_mat_descr(descr_);
        }

        MatDescr(const MatDescr&)            = delete;
        MatDescr& operator=(const MatDescr&) = delete;

        rocsparse_mat_descr get() const
        {
            return descr_;
        }

    private:
        rocsparse_mat_descr descr_ = nullptr;
    };

    class MatInfo
    {
    public:
        MatInfo()
        {
            EXPECT_EQ(rocsparse_create_mat_info(&info_), rocsparse_status_success);
        }

        ~MatInfo()
        {
            (void)rocsparse_destroy_mat_info(info_);
        }

        MatInfo(const MatInfo&)            = delete;
        MatInfo& operator=(const MatInfo&) = delete;

        rocsparse_mat_info get() const
        {
            return info_;
        }

    private:
        rocsparse_mat_info info_ = nullptr;
    };

    // re + i im for complex T, re for real T
    template <typename T>
    T value(double re, double im = 0.0)
    {
        if constexpr(is_complex_v<T>)
        {
            return T(static_cast<real_t<T>>(re), static_cast<real_t<T>>(im));
        }
        else
        {
            return static_cast<T>(re);
        }
    }

    template <typename T>
    void test_gthr()
    {
        std::vector<T>   y   = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5};
        std::vector<int> ind = {7, 2, 9, 0};

        DeviceBuffer<T>   d_y(y);
        DeviceBuffer<int> d_ind(ind);
        DeviceBuffer<T>   d_x(std::vector<T>(ind.size(), static_cast<T>(-1)));

        ASSERT_EQ(rocalution::rocsparseTgthr(sparse_handle(),
                                             static_cast<int>(ind.size()),
                                             d_y.get(),
                                             d_x.get(),
                                             d_ind.get(),
                                             rocsparse_index_base_zero),
                  rocsparse_status_success);

        std::vector<T> x = d_x.to_host();
        for(size_t i = 0; i < ind.size(); ++i)
        {
            EXPECT_EQ(x[i], y[ind[i]]) << "at index " << i;
        }
    }
}

TEST(hip_sparse, gthr_float)
{
    REQUIRE_BACKEND(Backend::Accelerator);
    test_gthr<float>();
}

TEST(hip_sparse, gthr_double)
{
    REQUIRE_BACKEND(Backend::Accelerator);
    test_gthr<double>();
}

TEST(hip_sparse, datatype_of_each_value_type)
{
    EXPECT_EQ(rocalution::rocsparseTdatatype<float>(), rocsparse_datatype_f32_r);
    EXPECT_EQ(rocalution::rocsparseTdatatype<double>(), rocsparse_datatype_f64_r);
    EXPECT_EQ(rocalution::rocsparseTdatatype<cfloat>(), rocsparse_datatype_f32_c);
    EXPECT_EQ(rocalution::rocsparseTdatatype<cdouble>(), rocsparse_datatype_f64_c);
    EXPECT_EQ(rocalution::rocsparseTdatatype<int8_t>(), rocsparse_datatype_i8_r);
    EXPECT_EQ(rocalution::rocsparseTdatatype<uint8_t>(), rocsparse_datatype_u8_r);
    EXPECT_EQ(rocalution::rocsparseTdatatype<int32_t>(), rocsparse_datatype_i32_r);
    EXPECT_EQ(rocalution::rocsparseTdatatype<uint32_t>(), rocsparse_datatype_u32_r);
}

template <typename P>
class hip_sparse_wrapper : public TypedBackendTest<P>
{
};

TYPED_TEST_SUITE(hip_sparse_wrapper, AcceleratorOnly<FloatingTypes>, TypedName);

TYPED_TEST(hip_sparse_wrapper, nnz_and_dense2csr_keep_the_nonzeros_of_a_column_major_matrix)
{
    using T = typename TypeParam::value_type;

    // 3 x 4 with lda 4: the padding row must be ignored. (0, 4) is a zero for real T.
    const int      m = 3, n = 4, lda = 4;
    const T        z    = T(0);
    std::vector<T> rows = {value<T>(1, 2),
                           z,
                           value<T>(2),
                           z,
                           z,
                           z,
                           z,
                           z,
                           value<T>(3),
                           value<T>(0, 4),
                           z,
                           value<T>(-5, 1)};
    std::vector<T> dense(lda * n, value<T>(99, 9));
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            dense[i + j * lda] = rows[i * n + j];
        }
    }

    std::vector<int> ref_row_ptr = {0};
    std::vector<int> ref_col;
    std::vector<T>   ref_val;
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            if(rows[i * n + j] != z)
            {
                ref_col.push_back(j);
                ref_val.push_back(rows[i * n + j]);
            }
        }
        ref_row_ptr.push_back(static_cast<int>(ref_col.size()));
    }

    MatDescr          descr;
    DeviceBuffer<T>   d_dense(dense);
    DeviceBuffer<int> d_nnz_per_row(std::vector<int>(m, -1));
    int               nnz_total = -1;

    ASSERT_EQ(rocalution::rocsparseTnnz(sparse_handle(),
                                        rocsparse_direction_row,
                                        m,
                                        n,
                                        descr.get(),
                                        d_dense.get(),
                                        lda,
                                        d_nnz_per_row.get(),
                                        &nnz_total),
              rocsparse_status_success);
    ASSERT_EQ(nnz_total, ref_row_ptr.back());
    std::vector<int> nnz_per_row = d_nnz_per_row.to_host();
    for(int i = 0; i < m; ++i)
    {
        EXPECT_EQ(nnz_per_row[i], ref_row_ptr[i + 1] - ref_row_ptr[i]) << "in row " << i;
    }

    DeviceBuffer<int> d_row_ptr(std::vector<int>(m + 1, -1));
    DeviceBuffer<int> d_col(std::vector<int>(nnz_total, -1));
    DeviceBuffer<T>   d_val(std::vector<T>(nnz_total, value<T>(-1)));

    ASSERT_EQ(rocalution::rocsparseTdense2csr(sparse_handle(),
                                              m,
                                              n,
                                              descr.get(),
                                              d_dense.get(),
                                              lda,
                                              d_nnz_per_row.get(),
                                              d_val.get(),
                                              d_row_ptr.get(),
                                              d_col.get()),
              rocsparse_status_success);
    EXPECT_EQ(d_row_ptr.to_host(), ref_row_ptr);
    EXPECT_EQ(d_col.to_host(), ref_col);
    EXPECT_EQ(d_val.to_host(), ref_val);
}

TYPED_TEST(hip_sparse_wrapper, compress_removes_the_entries_below_the_tolerance)
{
    using T = typename TypeParam::value_type;

    // (0.05, 0.3) is below the tolerance as a real value, above it as a complex one
    const int        m = 3, n = 4;
    std::vector<int> row_ptr = {0, 3, 5, 7};
    std::vector<int> col     = {0, 1, 3, 1, 2, 0, 2};
    std::vector<T>   val     = {value<T>(1, 0.5),
                                value<T>(0.01, 0.02),
                                value<T>(-2, 1),
                                value<T>(0.03, -0.04),
                                value<T>(0.05, 0.3),
                                value<T>(-0.02, 0.01),
                                value<T>(3, -2)};
    const T          tol     = value<T>(0.1);

    std::vector<int> ref_nnz_per_row;
    std::vector<int> ref_row_ptr = {0};
    std::vector<int> ref_col;
    std::vector<T>   ref_val;
    for(int i = 0; i < m; ++i)
    {
        for(int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
        {
            if(std::abs(val[k]) > std::real(tol))
            {
                ref_col.push_back(col[k]);
                ref_val.push_back(val[k]);
            }
        }
        ref_row_ptr.push_back(static_cast<int>(ref_col.size()));
        ref_nnz_per_row.push_back(ref_row_ptr[i + 1] - ref_row_ptr[i]);
    }

    MatDescr          descr;
    DeviceBuffer<int> d_row_ptr(row_ptr);
    DeviceBuffer<int> d_col(col);
    DeviceBuffer<T>   d_val(val);
    DeviceBuffer<int> d_nnz_per_row(std::vector<int>(m, -1));
    int               nnz_c = -1;

    ASSERT_EQ(rocalution::rocsparseTnnz_compress(sparse_handle(),
                                                 m,
                                                 descr.get(),
                                                 d_val.get(),
                                                 d_row_ptr.get(),
                                                 d_nnz_per_row.get(),
                                                 &nnz_c,
                                                 tol),
              rocsparse_status_success);
    ASSERT_EQ(nnz_c, ref_row_ptr.back());
    EXPECT_EQ(d_nnz_per_row.to_host(), ref_nnz_per_row);

    DeviceBuffer<int> d_row_ptr_c(std::vector<int>(m + 1, -1));
    DeviceBuffer<int> d_col_c(std::vector<int>(nnz_c, -1));
    DeviceBuffer<T>   d_val_c(std::vector<T>(nnz_c, value<T>(-1)));

    ASSERT_EQ(rocalution::rocsparseTcsr2csr_compress(sparse_handle(),
                                                     m,
                                                     n,
                                                     descr.get(),
                                                     d_val.get(),
                                                     d_row_ptr.get(),
                                                     d_col.get(),
                                                     static_cast<int>(val.size()),
                                                     d_nnz_per_row.get(),
                                                     d_val_c.get(),
                                                     d_row_ptr_c.get(),
                                                     d_col_c.get(),
                                                     tol),
              rocsparse_status_success);
    EXPECT_EQ(d_row_ptr_c.to_host(), ref_row_ptr);
    EXPECT_EQ(d_col_c.to_host(), ref_col);
    EXPECT_EQ(d_val_c.to_host(), ref_val);
}

// Block tridiagonal A = L L^H with block bidiagonal L: the zero fill-in
// factorization is the exact Cholesky factorization, so bsric0 returns L
TYPED_TEST(hip_sparse_wrapper, bsric0_factors_a_block_tridiagonal_matrix)
{
    using T = typename TypeParam::value_type;

    const int mb = 3, bs = 2, nrow = mb * bs;

    std::vector<T> L(nrow * nrow, T(0));
    auto           l = [&](int i, int j) -> T& { return L[i * nrow + j]; };
    l(0, 0)          = value<T>(2);
    l(1, 0)          = value<T>(1, 1);
    l(1, 1)          = value<T>(3);
    l(2, 0)          = value<T>(1, -1);
    l(2, 1)          = value<T>(-1, 0.5);
    l(2, 2)          = value<T>(2.5);
    l(3, 0)          = value<T>(0.5);
    l(3, 1)          = value<T>(1, 2);
    l(3, 2)          = value<T>(-0.5, 1);
    l(3, 3)          = value<T>(2);
    l(4, 2)          = value<T>(-1);
    l(4, 3)          = value<T>(0.5, -0.5);
    l(4, 4)          = value<T>(3);
    l(5, 2)          = value<T>(1, 1);
    l(5, 3)          = value<T>(2);
    l(5, 4)          = value<T>(-1, 0.5);
    l(5, 5)          = value<T>(1.5);

    std::vector<T> A(nrow * nrow, T(0));
    for(int i = 0; i < nrow; ++i)
    {
        for(int j = 0; j < nrow; ++j)
        {
            for(int k = 0; k < nrow; ++k)
            {
                A[i * nrow + j] += L[i * nrow + k] * conj_of(L[j * nrow + k]);
            }
        }
    }

    std::vector<int> row_ptr = {0, 2, 5, 7};
    std::vector<int> col     = {0, 1, 0, 1, 2, 1, 2};

    for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column})
    {
        SCOPED_TRACE(dir == rocsparse_direction_row ? "row blocks" : "column blocks");

        auto offset = [&](int k, int r, int c) {
            return k * bs * bs + (dir == rocsparse_direction_row ? r * bs + c : c * bs + r);
        };

        std::vector<T> val(col.size() * bs * bs);
        for(int bi = 0; bi < mb; ++bi)
        {
            for(int k = row_ptr[bi]; k < row_ptr[bi + 1]; ++k)
            {
                for(int r = 0; r < bs; ++r)
                {
                    for(int c = 0; c < bs; ++c)
                    {
                        val[offset(k, r, c)] = A[(bi * bs + r) * nrow + col[k] * bs + c];
                    }
                }
            }
        }

        MatDescr          descr;
        MatInfo           info;
        DeviceBuffer<int> d_row_ptr(row_ptr);
        DeviceBuffer<int> d_col(col);
        DeviceBuffer<T>   d_val(val);
        const int         nnzb = static_cast<int>(col.size());

        size_t buffer_size = 0;
        ASSERT_EQ(rocalution::rocsparseTbsric0_buffer_size(sparse_handle(),
                                                           dir,
                                                           mb,
                                                           nnzb,
                                                           descr.get(),
                                                           d_val.get(),
                                                           d_row_ptr.get(),
                                                           d_col.get(),
                                                           bs,
                                                           info.get(),
                                                           &buffer_size),
                  rocsparse_status_success);
        DeviceBuffer<char> buffer(std::vector<char>(std::max<size_t>(buffer_size, 1)));

        ASSERT_EQ(rocalution::rocsparseTbsric0_analysis(sparse_handle(),
                                                        dir,
                                                        mb,
                                                        nnzb,
                                                        descr.get(),
                                                        d_val.get(),
                                                        d_row_ptr.get(),
                                                        d_col.get(),
                                                        bs,
                                                        info.get(),
                                                        rocsparse_analysis_policy_reuse,
                                                        rocsparse_solve_policy_auto,
                                                        buffer.get()),
                  rocsparse_status_success);
        ASSERT_EQ(rocalution::rocsparseTbsric0(sparse_handle(),
                                               dir,
                                               mb,
                                               nnzb,
                                               descr.get(),
                                               d_val.get(),
                                               d_row_ptr.get(),
                                               d_col.get(),
                                               bs,
                                               info.get(),
                                               rocsparse_solve_policy_auto,
                                               buffer.get()),
                  rocsparse_status_success);

        int position = 0;
        EXPECT_EQ(rocsparse_bsric0_zero_pivot(sparse_handle(), info.get(), &position),
                  rocsparse_status_success);
        EXPECT_EQ(position, -1);

        // Only the lower triangle holds the factor
        std::vector<T> result = d_val.to_host();
        for(int bi = 0; bi < mb; ++bi)
        {
            for(int k = row_ptr[bi]; k < row_ptr[bi + 1]; ++k)
            {
                for(int r = 0; r < bs; ++r)
                {
                    for(int c = 0; c < bs; ++c)
                    {
                        const int i = bi * bs + r, j = col[k] * bs + c;
                        if(j <= i)
                        {
                            EXPECT_TRUE(is_near(result[offset(k, r, c)], L[i * nrow + j]))
                                << "at (" << i << ", " << j << ")";
                        }
                    }
                }
            }
        }
    }
}

// HIPAcceleratorMatrixCSR::ItILU0Factorize reads the history of complex
// factorizations through the real specializations, so these have to agree with
// the complex ones on a complex buffer
TYPED_TEST(hip_sparse_wrapper, csritilu0_history_of_a_converged_factorization)
{
    using T = typename TypeParam::value_type;
    using R = real_t<T>;

    const int        m       = 8;
    std::vector<int> row_ptr = {0};
    std::vector<int> col;
    std::vector<T>   val;
    for(int i = 0; i < m; ++i)
    {
        for(int j = std::max(0, i - 1); j <= std::min(m - 1, i + 1); ++j)
        {
            col.push_back(j);
            val.push_back(i == j ? value<T>(4) : value<T>(-1, j > i ? 0.5 : -0.5));
        }
        row_ptr.push_back(static_cast<int>(col.size()));
    }
    const int nnz = static_cast<int>(col.size());

    const rocsparse_itilu0_alg alg     = rocsparse_itilu0_alg_default;
    const rocsparse_int        options = rocsparse_itilu0_option_stopping_criteria
                                  | rocsparse_itilu0_option_compute_nrm_residual
                                  | rocsparse_itilu0_option_convergence_history;
    const rocsparse_int max_iter  = 50;
    const R             tolerance = static_cast<R>(tol<T>());

    DeviceBuffer<int> d_row_ptr(row_ptr);
    DeviceBuffer<int> d_col(col);
    DeviceBuffer<T>   d_val(val);
    DeviceBuffer<T>   d_ilu0(std::vector<T>(nnz, T(0)));

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_csritilu0_buffer_size(sparse_handle(),
                                              alg,
                                              options,
                                              max_iter,
                                              m,
                                              nnz,
                                              d_row_ptr.get(),
                                              d_col.get(),
                                              rocsparse_index_base_zero,
                                              rocalution::rocsparseTdatatype<T>(),
                                              &buffer_size),
              rocsparse_status_success);
    DeviceBuffer<char> buffer(std::vector<char>(std::max<size_t>(buffer_size, 1)));
    ASSERT_EQ(rocsparse_csritilu0_preprocess(sparse_handle(),
                                             alg,
                                             options,
                                             max_iter,
                                             m,
                                             nnz,
                                             d_row_ptr.get(),
                                             d_col.get(),
                                             rocsparse_index_base_zero,
                                             rocalution::rocsparseTdatatype<T>(),
                                             buffer_size,
                                             buffer.get()),
              rocsparse_status_success);

    rocsparse_int niter = max_iter;
    ASSERT_EQ(rocalution::rocsparseTcsritilu0_compute(sparse_handle(),
                                                      alg,
                                                      options,
                                                      &niter,
                                                      tolerance,
                                                      m,
                                                      nnz,
                                                      d_row_ptr.get(),
                                                      d_col.get(),
                                                      d_val.get(),
                                                      d_ilu0.get(),
                                                      rocsparse_index_base_zero,
                                                      buffer_size,
                                                      buffer.get()),
              rocsparse_status_success);
    ASSERT_GT(niter, 0);
    ASSERT_LT(niter, max_iter);

    std::vector<R> history(2 * max_iter, R(-1));
    rocsparse_int  history_niter = -1;
    ASSERT_EQ(rocalution::rocsparseTcsritilu0_history<T>(
                  sparse_handle(), alg, &history_niter, history.data(), buffer_size, buffer.get()),
              rocsparse_status_success);
    ASSERT_EQ(history_niter, niter);
    for(int i = 0; i < niter; ++i)
    {
        const R residual = history[niter + i];
        EXPECT_TRUE(std::isfinite(residual) && residual >= R(0)) << residual << " at " << i;
    }
    EXPECT_LE(history[2 * niter - 1], tolerance);

    if constexpr(is_complex_v<T>)
    {
        std::vector<R> real_history(2 * max_iter, R(-1));
        rocsparse_int  real_niter = -1;
        ASSERT_EQ(
            rocalution::rocsparseTcsritilu0_history<R>(
                sparse_handle(), alg, &real_niter, real_history.data(), buffer_size, buffer.get()),
            rocsparse_status_success);
        EXPECT_EQ(real_niter, history_niter);
        EXPECT_EQ(real_history, history);
    }
}

template <typename P>
class hip_blas_wrapper : public TypedBackendTest<P>
{
};

TYPED_TEST_SUITE(hip_blas_wrapper, AcceleratorOnly<FloatingTypes>, TypedName);

// C = alpha A^T + beta B, column major, with A stored as n x m
TYPED_TEST(hip_blas_wrapper, geam_adds_a_transposed_matrix)
{
    using T = typename TypeParam::value_type;

    const int      m = 3, n = 2;
    std::vector<T> A     = {value<T>(1, 1),
                            value<T>(2),
                            value<T>(-3, 0.5),
                            value<T>(4),
                            value<T>(0, -2),
                            value<T>(5, 1)};
    std::vector<T> B     = {value<T>(0.5),
                            value<T>(-1, 1),
                            value<T>(2, -1),
                            value<T>(3),
                            value<T>(1, 0.25),
                            value<T>(-2)};
    const T        alpha = value<T>(2, 1);
    const T        beta  = value<T>(-1, 0.5);

    std::vector<T> ref(m * n);
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            ref[i + j * m] = alpha * A[j + i * n] + beta * B[i + j * m];
        }
    }

    DeviceBuffer<T> d_A(A);
    DeviceBuffer<T> d_B(B);
    DeviceBuffer<T> d_C(std::vector<T>(m * n, value<T>(-7)));

    ASSERT_EQ(rocalution::rocblasTgeam(blas_handle(),
                                       rocblas_operation_transpose,
                                       rocblas_operation_none,
                                       m,
                                       n,
                                       &alpha,
                                       d_A.get(),
                                       n,
                                       &beta,
                                       d_B.get(),
                                       m,
                                       d_C.get(),
                                       m),
              rocblas_status_success);
    expect_all_near(d_C.to_host(), ref, tol<T>());
}

// The HIP DENSE MatMatMult is a gemm that overwrites C. C is compared as a
// dense matrix, as its zero entry is dropped in the conversion to CSR.
TYPED_TEST(hip_blas_wrapper, dense_mat_mat_mult_is_the_product)
{
    using T = typename TypeParam::value_type;

    auto checks = [] {
        Csr<T> a(3, 4), b(4, 2), c(3, 2);
        for(int i = 0; i < 3; ++i)
        {
            for(int j = 0; j < 4; ++j)
            {
                a.add(j, from_int<T>(i * 4 + j + 1));
            }
            a.end_row();
            c.add(0, from_int<T>(7));
            c.add(1, from_int<T>(7));
            c.end_row();
        }
        for(int i = 0; i < 4; ++i)
        {
            b.add(0, from_int<T>(i - 2));
            b.add(1, from_int<T>(2 * i + 1));
            b.end_row();
        }

        const Format dense_format = {rocalution::DENSE, 1};
        auto         A            = make_matrix(Backend::Accelerator, dense_format, *a.host_csr());
        auto         B            = make_matrix(Backend::Accelerator, dense_format, *b.host_csr());
        auto         C            = make_matrix(Backend::Accelerator, dense_format, *c.host_csr());
        ASSERT_NE(A, nullptr);
        ASSERT_NE(B, nullptr);
        ASSERT_NE(C, nullptr);

        ASSERT_TRUE(C->MatMatMult(*A, *B));
        expect_all_near(dense(csr_of(*C)), dense_product(dense(a), dense(b), 3, 4, 2), tol<T>());
    };

    if constexpr(is_complex_v<T>)
    {
        KNOWN_BUG("the complex rocblasTgemm of HIP DENSE MatMatMult is a FATAL_ERROR stub");
        expect_in_child(checks);
        return;
    }
    checks();
}
