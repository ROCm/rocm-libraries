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

#include "hipsparse_local_dnvec.hpp"
#include "hipsparse_local_handle.hpp"
#include "hipsparse_local_spmat.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

using namespace hipsparse_test;

// hipsparseSpMV after hipsparseCsrSetPointers, hipsparseCscSetPointers or
// hipsparseCooSetPointers on a matrix descriptor that SpMV already analyzed: new
// arrays, new values only, and SpMV without a new preprocessing. The pre_checkin
// prefix is the name CI selects.
class spmv_set_pointers : public testing::TestWithParam<int>
{
};

namespace
{
    using stream_ptr
        = std::unique_ptr<std::remove_pointer_t<hipStream_t>, hipError_t (*)(hipStream_t)>;

    stream_ptr new_stream()
    {
        hipStream_t stream = nullptr;
        CHECK_HIP_ERROR(hipStreamCreate(&stream));
        return stream_ptr(stream, hipStreamDestroy);
    }

    template <typename T>
    hipsparse_unique_ptr to_device(const std::vector<T>& host)
    {
        auto d = hipsparse_unique_ptr{device_malloc(sizeof(T) * host.size()), device_free};
        CHECK_HIP_ERROR(
            hipMemcpy(d.get(), host.data(), sizeof(T) * host.size(), hipMemcpyHostToDevice));
        return d;
    }

    // The matrices of #15871, row-major m x n. a3 has the pattern of a2 with other values.
    constexpr int64_t         m = 3;
    constexpr int64_t         n = 2;
    const std::vector<double> a1{2, 3, 0, 4, 0, 0};
    const std::vector<double> a2{0, 2, 3, 4, 0, 0};
    const std::vector<double> a3{0, 5, -1, 6, 0, 0};

    // The CSR arrays (or the CSC arrays, by_column) of a row-major m x n matrix.
    struct compressed
    {
        std::vector<int32_t> ptr{0};
        std::vector<int32_t> ind;
        std::vector<double>  val;

        compressed(const std::vector<double>& a, bool by_column)
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

    // The COO arrays (row, column and value of each nonzero) of a row-major m x n matrix.
    struct coo
    {
        std::vector<int32_t> row;
        std::vector<int32_t> col;
        std::vector<double>  val;

        explicit coo(const std::vector<double>& a)
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

    // x, y, the scalars and the buffer of y = alpha * op(A) * x + beta * y with
    // hipsparseSpMV on the m x n matrices above, checked against a dense product.
    class spmv_check
    {
        hipsparseHandle_t    m_handle;
        hipsparseOperation_t m_op;
        hipsparseSpMVAlg_t   m_alg;
        bool                 m_trans;
        std::vector<double>  m_x;
        std::vector<double>  m_y0;
        hipsparse_unique_ptr m_dx;
        hipsparse_unique_ptr m_dy;
        hipsparseLocalDnVec  m_vx;
        hipsparseLocalDnVec  m_vy;
        hipsparse_unique_ptr m_buffer{nullptr, device_free};
        size_t               m_bytes = 0;

    public:
        const double alpha = 2.0;
        const double beta  = 3.0;

        spmv_check(hipsparseHandle_t handle, hipsparseOperation_t op, hipsparseSpMVAlg_t alg)
            : m_handle(handle)
            , m_op(op)
            , m_alg(alg)
            , m_trans(op != HIPSPARSE_OPERATION_NON_TRANSPOSE)
            , m_x(m_trans ? std::vector<double>{5, 7, -2} : std::vector<double>{5, 7})
            , m_y0(m_trans ? std::vector<double>{1, 2} : std::vector<double>{1, 2, 3})
            , m_dx(to_device(m_x))
            , m_dy(to_device(m_y0))
            , m_vx(m_x.size(), m_dx.get(), HIP_R_64F)
            , m_vy(m_y0.size(), m_dy.get(), HIP_R_64F)
        {
        }

        hipsparseStatus_t buffer_size(hipsparseSpMatDescr_t mat, size_t* bytes)
        {
            return hipsparseSpMV_bufferSize(
                m_handle, m_op, &alpha, mat, m_vx, &beta, m_vy, HIP_R_64F, m_alg, bytes);
        }

        hipsparseStatus_t preprocess(hipsparseSpMatDescr_t mat)
        {
            return hipsparseSpMV_preprocess(
                m_handle, m_op, &alpha, mat, m_vx, &beta, m_vy, HIP_R_64F, m_alg, m_buffer.get());
        }

        hipsparseStatus_t spmv(hipsparseSpMatDescr_t mat)
        {
            return hipsparseSpMV(
                m_handle, m_op, &alpha, mat, m_vx, &beta, m_vy, HIP_R_64F, m_alg, m_buffer.get());
        }

        // The buffer size query, and a buffer of that size.
        void size_buffer(hipsparseSpMatDescr_t mat)
        {
            size_t need = 0;
            ASSERT_EQ(buffer_size(mat, &need), HIPSPARSE_STATUS_SUCCESS);
            // rocSPARSE rejects a buffer that is not null when the size is zero.
            if(need != m_bytes)
            {
                m_buffer.reset(need > 0 ? device_malloc(need) : nullptr);
                m_bytes = need;
            }
        }

        void reset_y()
        {
            ASSERT_EQ(
                hipMemcpy(
                    m_dy.get(), m_y0.data(), m_y0.size() * sizeof(double), hipMemcpyHostToDevice),
                hipSuccess);
        }

        // y against the product of the row-major matrix a.
        void check(const std::vector<double>& a)
        {
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            std::vector<double> got(m_y0.size());
            ASSERT_EQ(
                hipMemcpy(
                    got.data(), m_dy.get(), got.size() * sizeof(double), hipMemcpyDeviceToHost),
                hipSuccess);
            for(size_t r = 0; r < m_y0.size(); ++r)
            {
                double sum = 0.0;
                for(size_t c = 0; c < m_x.size(); ++c)
                {
                    sum += (m_trans ? a[c * n + r] : a[r * n + c]) * m_x[c];
                }
                EXPECT_DOUBLE_EQ(got[r], alpha * sum + beta * m_y0[r]) << "y[" << r << "]";
            }
        }

        // SpMV, after the buffer size query and the preprocessing when preprocess is
        // set, compared with the product of a.
        void run(hipsparseSpMatDescr_t      mat,
                 const std::vector<double>& a,
                 bool                       preprocess,
                 const char*                what)
        {
            SCOPED_TRACE(what);
            if(preprocess)
            {
                size_buffer(mat);
                ASSERT_FALSE(testing::Test::HasFatalFailure());
                ASSERT_EQ(this->preprocess(mat), HIPSPARSE_STATUS_SUCCESS);
            }
            reset_y();
            ASSERT_EQ(spmv(mat), HIPSPARSE_STATUS_SUCCESS);
            check(a);
        }
    };

    void run(bool by_column, hipsparseOperation_t op, hipsparseSpMVAlg_t alg)
    {
        SCOPED_TRACE(testing::Message() << "csc=" << by_column << " op=" << op << " alg=" << alg);

        const compressed h1(a1, by_column), h2(a2, by_column), h3(a3, by_column);
        ASSERT_EQ(h1.val.size(), h2.val.size());
        ASSERT_EQ(h2.val.size(), h3.val.size());

        auto p1 = to_device(h1.ptr), i1 = to_device(h1.ind), p2 = to_device(h2.ptr),
             i2 = to_device(h2.ind);
        auto v1 = to_device(h1.val), v2 = to_device(h2.val), v3 = to_device(h3.val);

        hipsparseLocalHandle_t             handle;
        std::unique_ptr<hipsparseLocalCsr> csr;
        std::unique_ptr<hipsparseLocalCsc> csc;
        if(by_column)
        {
            csc.reset(new hipsparseLocalCsc(m,
                                            n,
                                            h1.val.size(),
                                            p1.get(),
                                            i1.get(),
                                            v1.get(),
                                            HIPSPARSE_INDEX_32I,
                                            HIPSPARSE_INDEX_32I,
                                            HIPSPARSE_INDEX_BASE_ZERO,
                                            HIP_R_64F));
        }
        else
        {
            csr.reset(new hipsparseLocalCsr(m,
                                            n,
                                            h1.val.size(),
                                            p1.get(),
                                            i1.get(),
                                            v1.get(),
                                            HIPSPARSE_INDEX_32I,
                                            HIPSPARSE_INDEX_32I,
                                            HIPSPARSE_INDEX_BASE_ZERO,
                                            HIP_R_64F));
        }
        const hipsparseSpMatDescr_t mat
            = by_column ? hipsparseSpMatDescr_t(*csc) : hipsparseSpMatDescr_t(*csr);

        auto set_pointers = [&](void* ptr, void* ind, void* val) {
            return by_column ? hipsparseCscSetPointers(mat, ptr, ind, val)
                             : hipsparseCsrSetPointers(mat, ptr, ind, val);
        };

        spmv_check s(handle, op, alg);
        s.run(mat, a1, true, "first arrays");
        EXPECT_EQ(set_pointers(p2.get(), i2.get(), v2.get()), HIPSPARSE_STATUS_SUCCESS);
        s.run(mat, a2, true, "new arrays");
        EXPECT_EQ(set_pointers(p2.get(), i2.get(), v3.get()), HIPSPARSE_STATUS_SUCCESS);
        s.run(mat, a3, true, "new values only");
        EXPECT_EQ(set_pointers(p1.get(), i1.get(), v1.get()), HIPSPARSE_STATUS_SUCCESS);
        s.run(mat, a1, false, "first arrays again, without a new preprocessing");
    }

    // hipsparseSpMV on COO arrays after hipsparseCooSetPointers: new arrays, new
    // values only, and SpMV without a new preprocessing for the arrays of the first
    // matrix again.
    void run_coo(hipsparseSpMVAlg_t alg)
    {
        SCOPED_TRACE(testing::Message() << "coo alg=" << alg);

        const coo h1(a1), h2(a2), h3(a3);
        ASSERT_EQ(h1.val.size(), h2.val.size());
        ASSERT_EQ(h2.val.size(), h3.val.size());

        auto r1 = to_device(h1.row), c1 = to_device(h1.col), r2 = to_device(h2.row),
             c2 = to_device(h2.col);
        auto v1 = to_device(h1.val), v2 = to_device(h2.val), v3 = to_device(h3.val);

        hipsparseLocalHandle_t handle;
        hipsparseLocalCoo      mat(m,
                              n,
                              h1.val.size(),
                              r1.get(),
                              c1.get(),
                              v1.get(),
                              HIPSPARSE_INDEX_32I,
                              HIPSPARSE_INDEX_BASE_ZERO,
                              HIP_R_64F);

        spmv_check s(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, alg);
        s.run(mat, a1, true, "first arrays");
        EXPECT_EQ(hipsparseCooSetPointers(mat, r2.get(), c2.get(), v2.get()),
                  HIPSPARSE_STATUS_SUCCESS);
        s.run(mat, a2, true, "new arrays");
        EXPECT_EQ(hipsparseCooSetPointers(mat, r2.get(), c2.get(), v3.get()),
                  HIPSPARSE_STATUS_SUCCESS);
        s.run(mat, a3, true, "new values only");
        EXPECT_EQ(hipsparseCooSetPointers(mat, r1.get(), c1.get(), v1.get()),
                  HIPSPARSE_STATUS_SUCCESS);
        s.run(mat, a1, false, "first arrays again, without a new preprocessing");
    }

    using graph_exec_ptr
        = std::unique_ptr<std::remove_pointer_t<hipGraphExec_t>, hipError_t (*)(hipGraphExec_t)>;

    // Runs call while stream is being captured in global mode and checks that the
    // capture is still active after it. Returns the status of call. When exec is
    // not null, it receives the instantiated graph.
    template <typename F>
    hipsparseStatus_t during_capture(hipStream_t stream, F&& call, graph_exec_ptr* exec = nullptr)
    {
        hipsparseStatus_t st = HIPSPARSE_STATUS_INTERNAL_ERROR;
        EXPECT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
        EXPECT_NO_THROW(st = call());
        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        EXPECT_EQ(hipStreamIsCapturing(stream, &status), hipSuccess);
        EXPECT_EQ(status, hipStreamCaptureStatusActive) << "the capture was invalidated";
        hipGraph_t graph = nullptr;
        EXPECT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
        if(graph != nullptr)
        {
            if(exec != nullptr)
            {
                hipGraphExec_t instance = nullptr;
                EXPECT_EQ(hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0), hipSuccess);
                exec->reset(instance);
            }
            EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
        }
        return st;
    }

    // a1 with other values.
    const std::vector<double> a4{-1, 5, 0, 7, 0, 0};

    // A CSR matrix of a1 on a handle with its own stream, and the arrays of a2 and a4.
    struct captured_csr
    {
        const compressed       h1{a1, false};
        const compressed       h2{a2, false};
        const compressed       h4{a4, false};
        hipsparse_unique_ptr   p1, i1, v1, p2, i2, v2, v4;
        const stream_ptr       stream;
        hipsparseLocalHandle_t handle;
        hipsparseLocalCsr      mat;
        spmv_check             s;

        explicit captured_csr(hipsparseSpMVAlg_t alg)
            : p1(to_device(h1.ptr))
            , i1(to_device(h1.ind))
            , v1(to_device(h1.val))
            , p2(to_device(h2.ptr))
            , i2(to_device(h2.ind))
            , v2(to_device(h2.val))
            , v4(to_device(h4.val))
            , stream(new_stream())
            , mat(m,
                  n,
                  h1.val.size(),
                  p1.get(),
                  i1.get(),
                  v1.get(),
                  HIPSPARSE_INDEX_32I,
                  HIPSPARSE_INDEX_32I,
                  HIPSPARSE_INDEX_BASE_ZERO,
                  HIP_R_64F)
            , s(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, alg)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseSetStream(handle, stream.get()));
        }

        hipsparseStatus_t capture_spmv(graph_exec_ptr* exec = nullptr)
        {
            return during_capture(
                stream.get(), [&] { return s.spmv(mat); }, exec);
        }

        // Launches exec on the initial y and compares y with the product of a.
        void launch(const graph_exec_ptr& exec, const std::vector<double>& a)
        {
            ASSERT_NE(exec.get(), nullptr);
            s.reset_y();
            ASSERT_FALSE(testing::Test::HasFatalFailure());
            ASSERT_EQ(hipGraphLaunch(exec.get(), stream.get()), hipSuccess);
            s.check(a);
        }
    };

    // The first hipsparseSpMV_bufferSize, hipsparseSpMV_preprocess or hipsparseSpMV
    // after hipsparseCsrSetPointers would destroy the retired state and analyze the
    // new arrays, which synchronizes the device. While the handle stream is being
    // captured, each returns HIPSPARSE_STATUS_INVALID_VALUE, whatever the algorithm,
    // and leaves the capture valid. One hipsparseSpMV outside the capture computes
    // the new product, and a hipsparseSpMV captured after it records the new
    // product.
    void run_capture(hipsparseSpMVAlg_t alg)
    {
        SCOPED_TRACE(testing::Message() << "alg=" << alg);

        captured_csr c(alg);
        c.s.run(c.mat, a1, true, "first arrays");
        ASSERT_FALSE(testing::Test::HasFailure());
        ASSERT_EQ(hipsparseCsrSetPointers(c.mat, c.p2.get(), c.i2.get(), c.v2.get()),
                  HIPSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        EXPECT_EQ(c.capture_spmv(), HIPSPARSE_STATUS_INVALID_VALUE) << "SpMV during the capture";
        size_t bytes = 0;
        EXPECT_EQ(during_capture(c.stream.get(), [&] { return c.s.buffer_size(c.mat, &bytes); }),
                  HIPSPARSE_STATUS_INVALID_VALUE)
            << "bufferSize during the capture";
        EXPECT_EQ(during_capture(c.stream.get(), [&] { return c.s.preprocess(c.mat); }),
                  HIPSPARSE_STATUS_INVALID_VALUE)
            << "preprocess during the capture";
        ASSERT_FALSE(testing::Test::HasFailure());

        c.s.reset_y();
        EXPECT_EQ(c.s.spmv(c.mat), HIPSPARSE_STATUS_SUCCESS) << "SpMV after the capture";
        c.s.check(a2);

        graph_exec_ptr exec(nullptr, hipGraphExecDestroy);
        EXPECT_EQ(c.capture_spmv(&exec), HIPSPARSE_STATUS_SUCCESS)
            << "SpMV captured after the new arrays were analyzed";
        c.launch(exec, a2);
    }

    // hipsparseSpMatSetValues keeps the analysis: a hipsparseSpMV captured right
    // after it records the product of the new values. hipsparseCsrSetPointers with
    // the same row and column arrays and new values retires the analysis as new
    // arrays do.
    void run_capture_new_values(hipsparseSpMVAlg_t alg)
    {
        SCOPED_TRACE(testing::Message() << "alg=" << alg);

        captured_csr c(alg);
        c.s.run(c.mat, a1, true, "first arrays");
        ASSERT_FALSE(testing::Test::HasFailure());

        ASSERT_EQ(hipsparseSpMatSetValues(c.mat, c.v4.get()), HIPSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        graph_exec_ptr exec(nullptr, hipGraphExecDestroy);
        EXPECT_EQ(c.capture_spmv(&exec), HIPSPARSE_STATUS_SUCCESS)
            << "SpMV captured after hipsparseSpMatSetValues";
        c.launch(exec, a4);
        ASSERT_FALSE(testing::Test::HasFailure());

        ASSERT_EQ(hipsparseCsrSetPointers(c.mat, c.p1.get(), c.i1.get(), c.v1.get()),
                  HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(c.capture_spmv(), HIPSPARSE_STATUS_INVALID_VALUE)
            << "SpMV captured after a values-only hipsparseCsrSetPointers";
        c.s.reset_y();
        EXPECT_EQ(c.s.spmv(c.mat), HIPSPARSE_STATUS_SUCCESS) << "SpMV after the capture";
        c.s.check(a1);
    }

    // With the null stream as handle stream, querying its capture status fails with
    // hipErrorStreamCaptureImplicit while a blocking stream is being captured. The
    // first hipsparseSpMV after hipsparseCsrSetPointers returns
    // HIPSPARSE_STATUS_INVALID_VALUE, as during a capture of the handle stream.
    void run_null_stream_capture()
    {
        const compressed h1(a1, false), h2(a2, false);
        ASSERT_EQ(h1.val.size(), h2.val.size());

        auto p1 = to_device(h1.ptr), i1 = to_device(h1.ind), p2 = to_device(h2.ptr),
             i2 = to_device(h2.ind);
        auto v1 = to_device(h1.val), v2 = to_device(h2.val);
        auto scratch = hipsparse_unique_ptr{device_malloc(256), device_free};

        hipsparseLocalHandle_t handle;
        hipsparseLocalCsr      mat(m,
                              n,
                              h1.val.size(),
                              p1.get(),
                              i1.get(),
                              v1.get(),
                              HIPSPARSE_INDEX_32I,
                              HIPSPARSE_INDEX_32I,
                              HIPSPARSE_INDEX_BASE_ZERO,
                              HIP_R_64F);

        spmv_check s(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_SPMV_ALG_DEFAULT);
        s.run(mat, a1, true, "first arrays");
        ASSERT_FALSE(testing::Test::HasFailure());
        ASSERT_EQ(hipsparseCsrSetPointers(mat, p2.get(), i2.get(), v2.get()),
                  HIPSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const stream_ptr capture = new_stream();
        ASSERT_EQ(hipStreamBeginCapture(capture.get(), hipStreamCaptureModeGlobal), hipSuccess);
        EXPECT_EQ(hipMemsetAsync(scratch.get(), 0, 256, capture.get()), hipSuccess);
        hipsparseStatus_t captured = HIPSPARSE_STATUS_SUCCESS;
        EXPECT_NO_THROW(captured = s.spmv(mat));
        EXPECT_EQ(captured, HIPSPARSE_STATUS_INVALID_VALUE) << "SpMV during the capture";
        hipGraph_t graph = nullptr;
        EXPECT_EQ(hipStreamEndCapture(capture.get(), &graph), hipSuccess);
        if(graph != nullptr)
        {
            EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
        }

        s.reset_y();
        EXPECT_EQ(s.spmv(mat), HIPSPARSE_STATUS_SUCCESS) << "SpMV after the capture";
        s.check(a2);
    }
}

TEST_P(spmv_set_pointers, csr_capture)
{
    run_capture(HIPSPARSE_SPMV_ALG_DEFAULT);
    run_capture(HIPSPARSE_SPMV_CSR_ALG1);
    run_capture(HIPSPARSE_SPMV_CSR_ALG2);
}

TEST_P(spmv_set_pointers, csr_capture_new_values)
{
    run_capture_new_values(HIPSPARSE_SPMV_ALG_DEFAULT);
    run_capture_new_values(HIPSPARSE_SPMV_CSR_ALG1);
}

TEST_P(spmv_set_pointers, csr_null_stream_capture)
{
    run_null_stream_capture();
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
