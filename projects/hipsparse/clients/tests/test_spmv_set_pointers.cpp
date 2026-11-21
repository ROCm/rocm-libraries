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

#include "utility.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <vector>

// hipsparseSpMV after hipsparseCsrSetPointers, hipsparseCscSetPointers or
// hipsparseCooSetPointers on a matrix descriptor that SpMV already analyzed: new
// arrays, new values only, and SpMV without a new preprocessing. The pre_checkin
// prefix is the name CI selects.
class spmv_set_pointers : public testing::TestWithParam<int>
{
};

namespace
{
    template <typename T>
    struct device_array
    {
        T*     ptr = nullptr;
        size_t n   = 0;

        explicit device_array(const std::vector<T>& host)
            : n(host.size())
        {
            CHECK_HIP_ERROR(hipMalloc(&ptr, n * sizeof(T)));
            CHECK_HIP_ERROR(hipMemcpy(ptr, host.data(), n * sizeof(T), hipMemcpyHostToDevice));
        }
        device_array(const device_array&) = delete;
        device_array& operator=(const device_array&) = delete;
        ~device_array()
        {
            (void)hipFree(ptr);
        }
    };

    // The CSR arrays (or the CSC arrays, by_column) of a row-major m x n matrix.
    struct compressed
    {
        std::vector<int32_t> ptr{0};
        std::vector<int32_t> ind;
        std::vector<double>  val;

        compressed(const std::vector<double>& a, int64_t m, int64_t n, bool by_column)
        {
            const int64_t outer = by_column ? n : m;
            const int64_t inner = by_column ? m : n;
            for(int64_t o = 0; o < outer; ++o)
            {
                for(int64_t i = 0; i < inner; ++i)
                {
                    const double v = by_column ? a[i * n + o] : a[o * n + i];
                    if(v != 0.0)
                    {
                        ind.push_back(static_cast<int32_t>(i));
                        val.push_back(v);
                    }
                }
                ptr.push_back(static_cast<int32_t>(ind.size()));
            }
        }
    };

    std::vector<double> gemv(const std::vector<double>& a,
                             int64_t                    m,
                             int64_t                    n,
                             bool                       trans,
                             double                     alpha,
                             const std::vector<double>& x,
                             double                     beta,
                             std::vector<double>        y)
    {
        for(size_t r = 0; r < y.size(); ++r)
        {
            double sum = 0.0;
            for(size_t c = 0; c < x.size(); ++c)
            {
                sum += (trans ? a[c * n + r] : a[r * n + c]) * x[c];
            }
            y[r] = alpha * sum + beta * y[r];
        }
        return y;
    }

    void run(bool by_column, hipsparseOperation_t op, hipsparseSpMVAlg_t alg)
    {
        SCOPED_TRACE(testing::Message() << "csc=" << by_column << " op=" << op << " alg=" << alg);

        // The matrices of #15871. a3 has the pattern of a2 with other values.
        const int64_t             m = 3;
        const int64_t             n = 2;
        const std::vector<double> a1{2, 3, 0, 4, 0, 0};
        const std::vector<double> a2{0, 2, 3, 4, 0, 0};
        const std::vector<double> a3{0, 5, -1, 6, 0, 0};
        const bool                trans = op != HIPSPARSE_OPERATION_NON_TRANSPOSE;

        const compressed h1(a1, m, n, by_column), h2(a2, m, n, by_column), h3(a3, m, n, by_column);
        ASSERT_EQ(h1.val.size(), h2.val.size());
        ASSERT_EQ(h2.val.size(), h3.val.size());

        const std::vector<double> x
            = trans ? std::vector<double>{5, 7, -2} : std::vector<double>{5, 7};
        const std::vector<double> y0
            = trans ? std::vector<double>{1, 2} : std::vector<double>{1, 2, 3};
        const double alpha = 2.0;
        const double beta  = 3.0;

        device_array<int32_t> p1(h1.ptr), i1(h1.ind), p2(h2.ptr), i2(h2.ind);
        device_array<double>  v1(h1.val), v2(h2.val), v3(h3.val), dx(x), dy(y0);

        hipsparseHandle_t handle = nullptr;
        ASSERT_EQ(hipsparseCreate(&handle), HIPSPARSE_STATUS_SUCCESS);

        hipsparseSpMatDescr_t   mat    = nullptr;
        const hipsparseStatus_t create = by_column ? hipsparseCreateCsc(&mat,
                                                                        m,
                                                                        n,
                                                                        h1.val.size(),
                                                                        p1.ptr,
                                                                        i1.ptr,
                                                                        v1.ptr,
                                                                        HIPSPARSE_INDEX_32I,
                                                                        HIPSPARSE_INDEX_32I,
                                                                        HIPSPARSE_INDEX_BASE_ZERO,
                                                                        HIP_R_64F)
                                                   : hipsparseCreateCsr(&mat,
                                                                        m,
                                                                        n,
                                                                        h1.val.size(),
                                                                        p1.ptr,
                                                                        i1.ptr,
                                                                        v1.ptr,
                                                                        HIPSPARSE_INDEX_32I,
                                                                        HIPSPARSE_INDEX_32I,
                                                                        HIPSPARSE_INDEX_BASE_ZERO,
                                                                        HIP_R_64F);
        ASSERT_EQ(create, HIPSPARSE_STATUS_SUCCESS);

        hipsparseDnVecDescr_t vx = nullptr;
        hipsparseDnVecDescr_t vy = nullptr;
        ASSERT_EQ(hipsparseCreateDnVec(&vx, x.size(), dx.ptr, HIP_R_64F), HIPSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(hipsparseCreateDnVec(&vy, y0.size(), dy.ptr, HIP_R_64F),
                  HIPSPARSE_STATUS_SUCCESS);

        void*  buffer = nullptr;
        size_t bytes  = 0;

        auto set_pointers = [&](int32_t* ptr, int32_t* ind, double* val) {
            return by_column ? hipsparseCscSetPointers(mat, ptr, ind, val)
                             : hipsparseCsrSetPointers(mat, ptr, ind, val);
        };

        // SpMV, after the buffer size query and the preprocessing when preprocess is set.
        auto spmv = [&](const std::vector<double>& a, bool preprocess, const char* what) {
            SCOPED_TRACE(what);
            if(preprocess)
            {
                size_t need = 0;
                ASSERT_EQ(hipsparseSpMV_bufferSize(
                              handle, op, &alpha, mat, vx, &beta, vy, HIP_R_64F, alg, &need),
                          HIPSPARSE_STATUS_SUCCESS);
                // rocSPARSE rejects a buffer that is not null when the size is zero.
                if(need != bytes)
                {
                    ASSERT_EQ(hipFree(buffer), hipSuccess);
                    buffer = nullptr;
                    bytes  = need;
                    if(bytes > 0)
                    {
                        ASSERT_EQ(hipMalloc(&buffer, bytes), hipSuccess);
                    }
                }
                ASSERT_EQ(hipsparseSpMV_preprocess(
                              handle, op, &alpha, mat, vx, &beta, vy, HIP_R_64F, alg, buffer),
                          HIPSPARSE_STATUS_SUCCESS);
            }
            ASSERT_EQ(
                hipMemcpy(dy.ptr, y0.data(), y0.size() * sizeof(double), hipMemcpyHostToDevice),
                hipSuccess);
            ASSERT_EQ(hipsparseSpMV(handle, op, &alpha, mat, vx, &beta, vy, HIP_R_64F, alg, buffer),
                      HIPSPARSE_STATUS_SUCCESS);
            std::vector<double> got(y0.size());
            ASSERT_EQ(
                hipMemcpy(got.data(), dy.ptr, got.size() * sizeof(double), hipMemcpyDeviceToHost),
                hipSuccess);
            const std::vector<double> want = gemv(a, m, n, trans, alpha, x, beta, y0);
            for(size_t k = 0; k < want.size(); ++k)
            {
                EXPECT_DOUBLE_EQ(got[k], want[k]) << "y[" << k << "]";
            }
        };

        spmv(a1, true, "first arrays");
        EXPECT_EQ(set_pointers(p2.ptr, i2.ptr, v2.ptr), HIPSPARSE_STATUS_SUCCESS);
        spmv(a2, true, "new arrays");
        EXPECT_EQ(set_pointers(p2.ptr, i2.ptr, v3.ptr), HIPSPARSE_STATUS_SUCCESS);
        spmv(a3, true, "new values only");
        EXPECT_EQ(set_pointers(p1.ptr, i1.ptr, v1.ptr), HIPSPARSE_STATUS_SUCCESS);
        spmv(a1, false, "first arrays again, without a new preprocessing");

        EXPECT_EQ(hipFree(buffer), hipSuccess);
        EXPECT_EQ(hipsparseDestroyDnVec(vx), HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(hipsparseDestroyDnVec(vy), HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(hipsparseDestroySpMat(mat), HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(hipsparseDestroy(handle), HIPSPARSE_STATUS_SUCCESS);
    }
}

namespace
{
    // The COO arrays (row, column and value of each nonzero) of a row-major m x n matrix.
    struct coo
    {
        std::vector<int32_t> row;
        std::vector<int32_t> col;
        std::vector<double>  val;

        coo(const std::vector<double>& a, int64_t m, int64_t n)
        {
            for(int64_t r = 0; r < m; ++r)
            {
                for(int64_t c = 0; c < n; ++c)
                {
                    if(a[r * n + c] != 0.0)
                    {
                        row.push_back(static_cast<int32_t>(r));
                        col.push_back(static_cast<int32_t>(c));
                        val.push_back(a[r * n + c]);
                    }
                }
            }
        }
    };

    // hipsparseSpMV on COO arrays after hipsparseCooSetPointers: new arrays, new
    // values only, and SpMV without a new preprocessing for the arrays of the first
    // matrix again. Each product is checked against the dense reference.
    void run_coo(hipsparseSpMVAlg_t alg)
    {
        SCOPED_TRACE(testing::Message() << "coo alg=" << alg);

        const int64_t             m = 3;
        const int64_t             n = 2;
        const std::vector<double> a1{2, 3, 0, 4, 0, 0};
        const std::vector<double> a2{0, 2, 3, 4, 0, 0};
        const std::vector<double> a3{0, 5, -1, 6, 0, 0};
        const coo                 h1(a1, m, n), h2(a2, m, n), h3(a3, m, n);
        ASSERT_EQ(h1.val.size(), h2.val.size());
        ASSERT_EQ(h2.val.size(), h3.val.size());

        const std::vector<double>  x{5, 7};
        const std::vector<double>  y0{1, 2, 3};
        const double               alpha = 2.0;
        const double               beta  = 3.0;
        const hipsparseOperation_t op    = HIPSPARSE_OPERATION_NON_TRANSPOSE;

        device_array<int32_t> r1(h1.row), c1(h1.col), r2(h2.row), c2(h2.col);
        device_array<double>  v1(h1.val), v2(h2.val), v3(h3.val), dx(x), dy(y0);

        hipsparseHandle_t handle = nullptr;
        ASSERT_EQ(hipsparseCreate(&handle), HIPSPARSE_STATUS_SUCCESS);

        hipsparseSpMatDescr_t mat = nullptr;
        ASSERT_EQ(hipsparseCreateCoo(&mat,
                                     m,
                                     n,
                                     h1.val.size(),
                                     r1.ptr,
                                     c1.ptr,
                                     v1.ptr,
                                     HIPSPARSE_INDEX_32I,
                                     HIPSPARSE_INDEX_BASE_ZERO,
                                     HIP_R_64F),
                  HIPSPARSE_STATUS_SUCCESS);
        hipsparseDnVecDescr_t vx = nullptr;
        hipsparseDnVecDescr_t vy = nullptr;
        ASSERT_EQ(hipsparseCreateDnVec(&vx, x.size(), dx.ptr, HIP_R_64F), HIPSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(hipsparseCreateDnVec(&vy, y0.size(), dy.ptr, HIP_R_64F),
                  HIPSPARSE_STATUS_SUCCESS);

        void*  buffer = nullptr;
        size_t bytes  = 0;

        // SpMV, after the buffer size query and the preprocessing when preprocess is set.
        auto spmv = [&](const std::vector<double>& a, bool preprocess, const char* what) {
            SCOPED_TRACE(what);
            if(preprocess)
            {
                size_t need = 0;
                ASSERT_EQ(hipsparseSpMV_bufferSize(
                              handle, op, &alpha, mat, vx, &beta, vy, HIP_R_64F, alg, &need),
                          HIPSPARSE_STATUS_SUCCESS);
                if(need != bytes)
                {
                    ASSERT_EQ(hipFree(buffer), hipSuccess);
                    buffer = nullptr;
                    bytes  = need;
                    if(bytes > 0)
                    {
                        ASSERT_EQ(hipMalloc(&buffer, bytes), hipSuccess);
                    }
                }
                ASSERT_EQ(hipsparseSpMV_preprocess(
                              handle, op, &alpha, mat, vx, &beta, vy, HIP_R_64F, alg, buffer),
                          HIPSPARSE_STATUS_SUCCESS);
            }
            ASSERT_EQ(
                hipMemcpy(dy.ptr, y0.data(), y0.size() * sizeof(double), hipMemcpyHostToDevice),
                hipSuccess);
            ASSERT_EQ(hipsparseSpMV(handle, op, &alpha, mat, vx, &beta, vy, HIP_R_64F, alg, buffer),
                      HIPSPARSE_STATUS_SUCCESS);
            std::vector<double> got(y0.size());
            ASSERT_EQ(
                hipMemcpy(got.data(), dy.ptr, got.size() * sizeof(double), hipMemcpyDeviceToHost),
                hipSuccess);
            const std::vector<double> want = gemv(a, m, n, false, alpha, x, beta, y0);
            for(size_t k = 0; k < want.size(); ++k)
            {
                EXPECT_DOUBLE_EQ(got[k], want[k]) << "y[" << k << "]";
            }
        };

        spmv(a1, true, "first arrays");
        EXPECT_EQ(hipsparseCooSetPointers(mat, r2.ptr, c2.ptr, v2.ptr), HIPSPARSE_STATUS_SUCCESS);
        spmv(a2, true, "new arrays");
        EXPECT_EQ(hipsparseCooSetPointers(mat, r2.ptr, c2.ptr, v3.ptr), HIPSPARSE_STATUS_SUCCESS);
        spmv(a3, true, "new values only");
        EXPECT_EQ(hipsparseCooSetPointers(mat, r1.ptr, c1.ptr, v1.ptr), HIPSPARSE_STATUS_SUCCESS);
        spmv(a1, false, "first arrays again, without a new preprocessing");

        EXPECT_EQ(hipFree(buffer), hipSuccess);
        EXPECT_EQ(hipsparseDestroyDnVec(vx), HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(hipsparseDestroyDnVec(vy), HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(hipsparseDestroySpMat(mat), HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(hipsparseDestroy(handle), HIPSPARSE_STATUS_SUCCESS);
    }
}

namespace
{
    // The first hipsparseSpMV after hipsparseCsrSetPointers would analyze the new
    // arrays, which synchronizes the device. While the handle stream is being
    // captured, it returns HIPSPARSE_STATUS_INVALID_VALUE and leaves the capture
    // valid. The next call outside the capture computes the new product.
    void run_capture(hipsparseSpMVAlg_t alg)
    {
        SCOPED_TRACE(testing::Message() << "alg=" << alg);

        const int64_t             m = 3;
        const int64_t             n = 2;
        const std::vector<double> a1{2, 3, 0, 4, 0, 0};
        const std::vector<double> a2{0, 2, 3, 4, 0, 0};
        const compressed          h1(a1, m, n, false), h2(a2, m, n, false);
        ASSERT_EQ(h1.val.size(), h2.val.size());

        const std::vector<double>  x{5, 7};
        const std::vector<double>  y0{1, 2, 3};
        const double               alpha = 2.0;
        const double               beta  = 3.0;
        const hipsparseOperation_t op    = HIPSPARSE_OPERATION_NON_TRANSPOSE;

        device_array<int32_t> p1(h1.ptr), i1(h1.ind), p2(h2.ptr), i2(h2.ind);
        device_array<double>  v1(h1.val), v2(h2.val), dx(x), dy(y0);

        hipsparseHandle_t handle = nullptr;
        ASSERT_EQ(hipsparseCreate(&handle), HIPSPARSE_STATUS_SUCCESS);
        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
        ASSERT_EQ(hipsparseSetStream(handle, stream), HIPSPARSE_STATUS_SUCCESS);

        hipsparseSpMatDescr_t mat = nullptr;
        ASSERT_EQ(hipsparseCreateCsr(&mat,
                                     m,
                                     n,
                                     h1.val.size(),
                                     p1.ptr,
                                     i1.ptr,
                                     v1.ptr,
                                     HIPSPARSE_INDEX_32I,
                                     HIPSPARSE_INDEX_32I,
                                     HIPSPARSE_INDEX_BASE_ZERO,
                                     HIP_R_64F),
                  HIPSPARSE_STATUS_SUCCESS);
        hipsparseDnVecDescr_t vx = nullptr;
        hipsparseDnVecDescr_t vy = nullptr;
        ASSERT_EQ(hipsparseCreateDnVec(&vx, x.size(), dx.ptr, HIP_R_64F), HIPSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(hipsparseCreateDnVec(&vy, y0.size(), dy.ptr, HIP_R_64F),
                  HIPSPARSE_STATUS_SUCCESS);

        size_t bytes = 0;
        ASSERT_EQ(hipsparseSpMV_bufferSize(
                      handle, op, &alpha, mat, vx, &beta, vy, HIP_R_64F, alg, &bytes),
                  HIPSPARSE_STATUS_SUCCESS);
        void* buffer = nullptr;
        if(bytes > 0)
        {
            ASSERT_EQ(hipMalloc(&buffer, bytes), hipSuccess);
        }
        ASSERT_EQ(hipsparseSpMV_preprocess(
                      handle, op, &alpha, mat, vx, &beta, vy, HIP_R_64F, alg, buffer),
                  HIPSPARSE_STATUS_SUCCESS);

        auto spmv = [&]() {
            return hipsparseSpMV(handle, op, &alpha, mat, vx, &beta, vy, HIP_R_64F, alg, buffer);
        };
        auto check = [&](const std::vector<double>& a, const char* what) {
            SCOPED_TRACE(what);
            std::vector<double> got(y0.size());
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            ASSERT_EQ(
                hipMemcpy(got.data(), dy.ptr, got.size() * sizeof(double), hipMemcpyDeviceToHost),
                hipSuccess);
            const std::vector<double> want = gemv(a, m, n, false, alpha, x, beta, y0);
            for(size_t k = 0; k < want.size(); ++k)
            {
                EXPECT_DOUBLE_EQ(got[k], want[k]) << "y[" << k << "]";
            }
            ASSERT_EQ(
                hipMemcpy(dy.ptr, y0.data(), y0.size() * sizeof(double), hipMemcpyHostToDevice),
                hipSuccess);
        };

        ASSERT_EQ(spmv(), HIPSPARSE_STATUS_SUCCESS);
        check(a1, "first arrays");
        ASSERT_EQ(hipsparseCsrSetPointers(mat, p2.ptr, i2.ptr, v2.ptr), HIPSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
        hipsparseStatus_t captured = HIPSPARSE_STATUS_SUCCESS;
        EXPECT_NO_THROW(captured = spmv());
        EXPECT_EQ(captured, HIPSPARSE_STATUS_INVALID_VALUE) << "SpMV during the capture";
        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        EXPECT_EQ(hipStreamIsCapturing(stream, &status), hipSuccess);
        EXPECT_EQ(status, hipStreamCaptureStatusActive);
        hipGraph_t graph = nullptr;
        EXPECT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
        if(graph != nullptr)
        {
            EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
        }

        EXPECT_EQ(spmv(), HIPSPARSE_STATUS_SUCCESS) << "SpMV after the capture";
        check(a2, "new arrays, after the capture");

        EXPECT_EQ(hipFree(buffer), hipSuccess);
        EXPECT_EQ(hipsparseDestroyDnVec(vx), HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(hipsparseDestroyDnVec(vy), HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(hipsparseDestroySpMat(mat), HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(hipsparseDestroy(handle), HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
    }
}

TEST_P(spmv_set_pointers, csr_capture)
{
    run_capture(HIPSPARSE_SPMV_ALG_DEFAULT);
    run_capture(HIPSPARSE_SPMV_CSR_ALG1);
}

TEST_P(spmv_set_pointers, coo)
{
    run_coo(HIPSPARSE_SPMV_ALG_DEFAULT);
    run_coo(HIPSPARSE_SPMV_COO_ALG1);
    run_coo(HIPSPARSE_SPMV_COO_ALG2);
}

TEST_P(spmv_set_pointers, csr)
{
    run(false, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_SPMV_ALG_DEFAULT);
    run(false, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_SPMV_CSR_ALG1);
}

TEST_P(spmv_set_pointers, csc)
{
    run(true, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_SPMV_ALG_DEFAULT);
    run(true, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_SPMV_ALG_DEFAULT);
}

INSTANTIATE_TEST_SUITE_P(pre_checkin, spmv_set_pointers, testing::Values(0));
