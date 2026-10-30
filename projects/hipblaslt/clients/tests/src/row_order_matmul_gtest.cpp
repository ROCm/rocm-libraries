// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// hipblasLtMatmul with HIPBLASLT_ORDER_ROW layouts (ROCM-29960). The library rewrites a
// row-major problem as a column-major one; these tests check the numbers it produces.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    constexpr size_t kWorkspace = size_t(32) << 20;

    // A stored r x c matrix with a leading dimension padded past the packed one, so a
    // leading dimension taken from the wrong axis reads the wrong elements.
    struct Stored
    {
        int64_t          rows, cols, ld, batch;
        hipblasLtOrder_t order;

        int64_t padded() const
        {
            return order == HIPBLASLT_ORDER_ROW ? cols + 2 : rows + 3;
        }
        int64_t stride() const
        {
            return ld * (order == HIPBLASLT_ORDER_ROW ? rows : cols);
        }
        int64_t index(int64_t b, int64_t i, int64_t j) const
        {
            return b * stride() + (order == HIPBLASLT_ORDER_ROW ? i * ld + j : i + j * ld);
        }
        size_t elements() const
        {
            return size_t(stride() * batch);
        }
    };

    Stored stored(int64_t rows, int64_t cols, int64_t batch, hipblasLtOrder_t order)
    {
        Stored s{rows, cols, 0, batch, order};
        s.ld = s.padded();
        return s;
    }

    // Small integers, so every product and sum below is exact in fp16 and fp32 and the
    // results compare with ==.
    float valueA(int64_t b, int64_t i, int64_t j)
    {
        return float((i * 7 + j * 3 + b) % 7 - 3);
    }
    float valueB(int64_t b, int64_t i, int64_t j)
    {
        return float((i * 5 + j * 2 + b) % 5 - 2);
    }
    float valueC(int64_t b, int64_t i, int64_t j)
    {
        return float((i * 3 + j * 5 + b) % 3 - 1);
    }

    template <typename T>
    void* upload(const std::vector<T>& host)
    {
        void* device = nullptr;
        EXPECT_EQ(hipMalloc(&device, host.size() * sizeof(T)), hipSuccess);
        EXPECT_EQ(hipMemcpy(device, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
                  hipSuccess);
        return device;
    }

    hipblasLtMatrixLayout_t makeLayout(hipDataType type, const Stored& s)
    {
        hipblasLtMatrixLayout_t layout = nullptr;
        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&layout, type, s.rows, s.cols, s.ld),
                  HIPBLAS_STATUS_SUCCESS);
        const int32_t order = s.order;
        EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                      layout, HIPBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
                  HIPBLAS_STATUS_SUCCESS);
        if(s.batch > 1)
        {
            const int32_t count  = int32_t(s.batch);
            const int64_t stride = s.stride();
            EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                          layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)),
                      HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(
                hipblasLtMatrixLayoutSetAttribute(
                    layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)),
                HIPBLAS_STATUS_SUCCESS);
        }
        return layout;
    }

    struct Case
    {
        hipDataType        type;
        hipblasOperation_t opA, opB;
        hipblasLtOrder_t   orderA, orderB, orderCD;
        int64_t            batch;

        std::string name() const
        {
            auto o = [](hipblasLtOrder_t x) { return x == HIPBLASLT_ORDER_ROW ? "row" : "col"; };
            auto t = [](hipblasOperation_t x) { return x == HIPBLAS_OP_N ? "N" : "T"; };
            std::ostringstream s;
            s << (type == HIP_R_16F ? "f16" : "f32") << " " << t(opA) << t(opB) << " A "
              << o(orderA) << " B " << o(orderB) << " C/D " << o(orderCD) << " batch " << batch;
            return s.str();
        }
    };

    // Odd, unequal sizes, so swapping m with n or reading a transposed index changes the result.
    constexpr int64_t kM = 37, kN = 19, kK = 11;

    template <typename T>
    void runCase(hipblasLtHandle_t handle, const Case& tc)
    {
        SCOPED_TRACE(tc.name());
        const bool   tA = tc.opA == HIPBLAS_OP_T, tB = tc.opB == HIPBLAS_OP_T;
        const Stored sA = stored(tA ? kK : kM, tA ? kM : kK, tc.batch, tc.orderA);
        const Stored sB = stored(tB ? kN : kK, tB ? kK : kN, tc.batch, tc.orderB);
        const Stored sC = stored(kM, kN, tc.batch, tc.orderCD);

        std::vector<T> hA(sA.elements()), hB(sB.elements()), hC(sC.elements());
        for(int64_t b = 0; b < tc.batch; b++)
        {
            for(int64_t i = 0; i < sA.rows; i++)
                for(int64_t j = 0; j < sA.cols; j++)
                    hA[sA.index(b, i, j)] = T(valueA(b, i, j));
            for(int64_t i = 0; i < sB.rows; i++)
                for(int64_t j = 0; j < sB.cols; j++)
                    hB[sB.index(b, i, j)] = T(valueB(b, i, j));
            for(int64_t i = 0; i < kM; i++)
                for(int64_t j = 0; j < kN; j++)
                    hC[sC.index(b, i, j)] = T(valueC(b, i, j));
        }

        const float alpha = 2.0f, beta = 3.0f;
        void*       dA   = upload(hA);
        void*       dB   = upload(hB);
        void*       dC   = upload(hC);
        void*       dD   = upload(std::vector<T>(sC.elements(), T(0)));
        void*       work = nullptr;
        ASSERT_EQ(hipMalloc(&work, kWorkspace), hipSuccess);

        hipblasLtMatmulDesc_t desc = nullptr;
        ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_TRANSA, &tc.opA, sizeof(tc.opA)),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tc.opB, sizeof(tc.opB)),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasLtMatrixLayout_t lA = makeLayout(tc.type, sA), lB = makeLayout(tc.type, sB);
        hipblasLtMatrixLayout_t lC = makeLayout(tc.type, sC), lD = makeLayout(tc.type, sC);

        hipblasLtMatmulPreference_t pref = nullptr;
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(
            hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspace, sizeof(kWorkspace)),
            HIPBLAS_STATUS_SUCCESS);
        hipblasLtMatmulHeuristicResult_t result{};
        int                              returned = 0;
        ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                      handle, desc, lA, lB, lC, lD, pref, 1, &result, &returned),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_GT(returned, 0);
        ASSERT_EQ(hipblasLtMatmul(handle,
                                  desc,
                                  &alpha,
                                  dA,
                                  lA,
                                  dB,
                                  lB,
                                  &beta,
                                  dC,
                                  lC,
                                  dD,
                                  lD,
                                  &result.algo,
                                  work,
                                  kWorkspace,
                                  nullptr),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        std::vector<T> hD(sC.elements());
        ASSERT_EQ(hipMemcpy(hD.data(), dD, hD.size() * sizeof(T), hipMemcpyDeviceToHost),
                  hipSuccess);
        int mismatches = 0;
        for(int64_t b = 0; b < tc.batch; b++)
            for(int64_t i = 0; i < kM; i++)
                for(int64_t j = 0; j < kN; j++)
                {
                    float sum = 0;
                    for(int64_t l = 0; l < kK; l++)
                        sum += (tA ? valueA(b, l, i) : valueA(b, i, l))
                               * (tB ? valueB(b, j, l) : valueB(b, l, j));
                    const float expected = alpha * sum + beta * valueC(b, i, j);
                    const float actual   = float(hD[sC.index(b, i, j)]);
                    if(actual != expected && mismatches++ < 3)
                        ADD_FAILURE() << "D(" << b << ", " << i << ", " << j << ") is " << actual
                                      << ", expected " << expected;
                }
        EXPECT_EQ(mismatches, 0);

        hipblasLtMatmulPreferenceDestroy(pref);
        for(auto l : {lA, lB, lC, lD})
            hipblasLtMatrixLayoutDestroy(l);
        hipblasLtMatmulDescDestroy(desc);
        for(void* p : {dA, dB, dC, dD, work})
            static_cast<void>(hipFree(p));
    }

    class RowOrderMatmul : public ::testing::Test
    {
    protected:
        hipblasLtHandle_t handle = nullptr;

        void SetUp() override
        {
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
        }
        void TearDown() override
        {
            if(handle)
                hipblasLtDestroy(handle);
        }
    };

    // Every op and order combination for fp32 and fp16, plus a strided batch for fp32, against
    // a reference computed from the logical matrices on the host.
    TEST_F(RowOrderMatmul, smoke_MatchesReferenceForEveryOrder)
    {
        for(hipDataType type : {HIP_R_32F, HIP_R_16F})
            for(int64_t batch : {1, 2})
            {
                if(type == HIP_R_16F && batch > 1)
                    continue;
                for(auto opA : {HIPBLAS_OP_N, HIPBLAS_OP_T})
                    for(auto opB : {HIPBLAS_OP_N, HIPBLAS_OP_T})
                        for(auto oA : {HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW})
                            for(auto oB : {HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW})
                                for(auto oCD : {HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW})
                                {
                                    const Case tc{type, opA, opB, oA, oB, oCD, batch};
                                    if(type == HIP_R_16F)
                                        runCase<_Float16>(handle, tc);
                                    else
                                        runCase<float>(handle, tc);
                                }
            }
    }

    // The case from the report: fp32 2x3 * 3x4 with every layout row-major and a packed
    // leading dimension. Before the fix D came back as 9 12 0 0; 29 40 0 0 with success.
    TEST_F(RowOrderMatmul, smoke_ReportedCase)
    {
        const std::vector<float> hA = {1, 2, 3, 4, 5, 6};
        const std::vector<float> hB = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        void*                    dA = upload(hA);
        void*                    dB = upload(hB);
        void*                    dD = upload(std::vector<float>(8, 0.0f));

        hipblasLtMatmulDesc_t desc = nullptr;
        ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasLtMatrixLayout_t lA = makeLayout(HIP_R_32F, {2, 3, 3, 1, HIPBLASLT_ORDER_ROW});
        hipblasLtMatrixLayout_t lB = makeLayout(HIP_R_32F, {3, 4, 4, 1, HIPBLASLT_ORDER_ROW});
        hipblasLtMatrixLayout_t lD = makeLayout(HIP_R_32F, {2, 4, 4, 1, HIPBLASLT_ORDER_ROW});

        const float alpha = 1.0f, beta = 0.0f;
        ASSERT_EQ(hipblasLtMatmul(handle,
                                  desc,
                                  &alpha,
                                  dA,
                                  lA,
                                  dB,
                                  lB,
                                  &beta,
                                  dD,
                                  lD,
                                  dD,
                                  lD,
                                  nullptr,
                                  nullptr,
                                  0,
                                  nullptr),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        std::vector<float> hD(8);
        ASSERT_EQ(hipMemcpy(hD.data(), dD, hD.size() * sizeof(float), hipMemcpyDeviceToHost),
                  hipSuccess);
        EXPECT_EQ(hD, (std::vector<float>{38, 44, 50, 56, 83, 98, 113, 128}));

        for(auto l : {lA, lB, lD})
            hipblasLtMatrixLayoutDestroy(l);
        hipblasLtMatmulDescDestroy(desc);
        for(void* p : {dA, dB, dD})
            static_cast<void>(hipFree(p));
    }

    // A row-major D is computed as its transpose, so anything tied to an axis of D would land
    // on the wrong axis and is refused. The same settings with a column-major D are the
    // control: there only A is rewritten and nothing moves between axes.
    TEST_F(RowOrderMatmul, smoke_RefusesWhatTheRewriteCannotExpress)
    {
        const Stored            sq  = stored(64, 64, 1, HIPBLASLT_ORDER_COL);
        const Stored            sr  = stored(64, 64, 1, HIPBLASLT_ORDER_ROW);
        hipblasLtMatrixLayout_t col = makeLayout(HIP_R_32F, sq), row = makeLayout(HIP_R_32F, sr);

        hipblasLtMatmulPreference_t pref = nullptr;
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        auto heuristic = [&](hipblasLtMatmulDesc_t   desc,
                             hipblasLtMatrixLayout_t a,
                             hipblasLtMatrixLayout_t c,
                             hipblasLtMatrixLayout_t d) {
            hipblasLtMatmulHeuristicResult_t result{};
            int                              returned = -1;
            const hipblasStatus_t            status   = hipblasLtMatmulAlgoGetHeuristic(
                handle, desc, a, col, c, d, pref, 1, &result, &returned);
            if(status != HIPBLAS_STATUS_SUCCESS)
                EXPECT_EQ(returned, 0);
            return status;
        };
        auto makeDesc = [] {
            hipblasLtMatmulDesc_t desc = nullptr;
            EXPECT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            return desc;
        };

        hipblasLtMatmulDesc_t plain = makeDesc();
        EXPECT_EQ(heuristic(plain, col, col, row), HIPBLAS_STATUS_NOT_SUPPORTED)
            << "only D row-major";
        EXPECT_EQ(heuristic(plain, col, row, col), HIPBLAS_STATUS_NOT_SUPPORTED)
            << "only C row-major";

        hipblasLtMatmulDesc_t     bias     = makeDesc();
        const hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
        EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                      bias, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(bias, col, row, row), HIPBLAS_STATUS_NOT_SUPPORTED)
            << "bias, D row-major";
        EXPECT_EQ(heuristic(bias, row, col, col), HIPBLAS_STATUS_SUCCESS) << "bias, A row-major";

        hipblasLtMatmulDesc_t        alphaVector = makeDesc();
        const hipblasLtPointerMode_t mode = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
        EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                      alphaVector, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &mode, sizeof(mode)),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(alphaVector, col, row, row), HIPBLAS_STATUS_NOT_SUPPORTED)
            << "alpha vector, D row-major";
        EXPECT_NE(heuristic(alphaVector, row, col, col), HIPBLAS_STATUS_NOT_SUPPORTED)
            << "alpha vector, A row-major";

        hipblasLtMatmulDesc_t              blockScale = makeDesc();
        const hipblasLtMatmulMatrixScale_t scaleMode  = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        EXPECT_EQ(
            hipblasLtMatmulDescSetAttribute(
                blockScale, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)),
            HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(blockScale, row, col, col), HIPBLAS_STATUS_NOT_SUPPORTED)
            << "block scale, A row-major";

        // A missing argument keeps its usual status even when the orders cannot be rewritten.
        auto nullHandle = [&](hipblasLtMatrixLayout_t d) {
            hipblasLtMatmulHeuristicResult_t result{};
            int                              returned = 0;
            return hipblasLtMatmulAlgoGetHeuristic(
                nullptr, plain, col, col, col, d, pref, 1, &result, &returned);
        };
        EXPECT_NE(nullHandle(col), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(nullHandle(row), nullHandle(col)) << "null handle, only D row-major";

        for(auto desc : {plain, bias, alphaVector, blockScale})
            hipblasLtMatmulDescDestroy(desc);
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatrixLayoutDestroy(col);
        hipblasLtMatrixLayoutDestroy(row);
    }
} // namespace
