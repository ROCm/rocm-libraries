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
}

TEST(hip_sparse, datatype_of_each_value_type)
{
    using rocalution::rocalution_datatype_traits;

    EXPECT_EQ(rocalution_datatype_traits<float>::value, rocsparse_datatype_f32_r);
    EXPECT_EQ(rocalution_datatype_traits<double>::value, rocsparse_datatype_f64_r);
    EXPECT_EQ(rocalution_datatype_traits<cfloat>::value, rocsparse_datatype_f32_c);
    EXPECT_EQ(rocalution_datatype_traits<cdouble>::value, rocsparse_datatype_f64_c);
    EXPECT_EQ(rocalution_datatype_traits<int32_t>::value, rocsparse_datatype_i32_r);

    // rocALUTION has vectors of these, rocSPARSE has no datatype for them
    EXPECT_FALSE(rocalution_datatype_traits<bool>::is_supported);
    EXPECT_FALSE(rocalution_datatype_traits<int64_t>::is_supported);
}

TEST(hip_sparse, indextype_of_each_index_type)
{
    EXPECT_EQ(rocalution::rocalution_indextype_traits<int32_t>::value, rocsparse_indextype_i32);
    EXPECT_EQ(rocalution::rocalution_indextype_traits<int64_t>::value, rocsparse_indextype_i64);
}

template <typename P>
class hip_sparse_wrapper : public TypedBackendTest<P>
{
};

TYPED_TEST_SUITE(hip_sparse_wrapper, AcceleratorOnly<FloatingTypes>, TypedName);

// Block tridiagonal A = L L^H with block bidiagonal L: the zero fill-in
// factorization is the exact Cholesky factorization, so spic0 returns L
TYPED_TEST(hip_sparse_wrapper, spic0_factors_a_block_tridiagonal_bsr_matrix)
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

        DeviceBuffer<int> d_row_ptr(row_ptr);
        DeviceBuffer<int> d_col(col);
        DeviceBuffer<T>   d_val(val);
        const int         nnzb = static_cast<int>(col.size());

        rocsparse_spmat_descr mat = nullptr;
        ASSERT_EQ(rocsparse_create_bsr_descr(&mat,
                                             mb,
                                             mb,
                                             nnzb,
                                             dir,
                                             bs,
                                             d_row_ptr.get(),
                                             d_col.get(),
                                             d_val.get(),
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocalution::rocalution_datatype_traits<T>::value),
                  rocsparse_status_success);
        rocalution::HIPSpIC0<T>(sparse_handle(), mat);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);

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
                                              rocalution::rocalution_datatype_traits<T>::value,
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
                                             rocalution::rocalution_datatype_traits<T>::value,
                                             buffer_size,
                                             buffer.get()),
              rocsparse_status_success);

    rocsparse_int niter = max_iter;
    ASSERT_EQ(rocalution::rocsparseTcsritilu0_compute_ex(sparse_handle(),
                                                         alg,
                                                         options,
                                                         &niter,
                                                         0,
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
