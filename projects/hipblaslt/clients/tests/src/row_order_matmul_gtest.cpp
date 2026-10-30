// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// hipblasLtMatmul with HIPBLASLT_ORDER_ROW layouts (ROCM-29960). The library rewrites a
// row-major A or B as its column-major transpose; these tests check the numbers it produces
// and the combinations it refuses.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <functional>
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
        hipblasLtOrder_t   orderA, orderB;
        int64_t            batch;

        std::string name() const
        {
            auto o = [](hipblasLtOrder_t x) { return x == HIPBLASLT_ORDER_ROW ? "row" : "col"; };
            auto t = [](hipblasOperation_t x) { return x == HIPBLAS_OP_N ? "N" : "T"; };
            std::ostringstream s;
            s << (type == HIP_R_16F ? "f16" : "f32") << " " << t(opA) << t(opB) << " A "
              << o(orderA) << " B " << o(orderB) << " batch " << batch;
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
        const Stored sC = stored(kM, kN, tc.batch, HIPBLASLT_ORDER_COL);

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

    // Every op combination and order of A and B for fp32 and fp16, plus a strided batch for
    // fp32, against a reference computed from the logical matrices on the host.
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
                            {
                                const Case tc{type, opA, opB, oA, oB, batch};
                                if(type == HIP_R_16F)
                                    runCase<_Float16>(handle, tc);
                                else
                                    runCase<float>(handle, tc);
                            }
            }
    }

    // The matrices from the report: fp32 2x3 * 3x4 with A and B row-major and packed leading
    // dimensions, as reported, and D column-major. The reported row-major D is refused (see
    // smoke_RefusesWhatTheRewriteCannotExpress).
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
        hipblasLtMatrixLayout_t lD = makeLayout(HIP_R_32F, {2, 4, 2, 1, HIPBLASLT_ORDER_COL});

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
        EXPECT_EQ(hD, (std::vector<float>{38, 83, 44, 98, 50, 113, 56, 128}));

        for(auto l : {lA, lB, lD})
            hipblasLtMatrixLayoutDestroy(l);
        hipblasLtMatmulDescDestroy(desc);
        for(void* p : {dA, dB, dD})
            static_cast<void>(hipFree(p));
    }

    struct Guard
    {
        std::string                                name;
        hipblasComputeType_t                       compute;
        std::function<void(hipblasLtMatmulDesc_t)> configure;
        hipDataType                                typeA, typeB;
        bool                                       rowA, rowB, rowC, rowD;
        hipblasStatus_t                            expected;
    };

    template <typename T>
    void setAttribute(hipblasLtMatmulDesc_t desc, hipblasLtMatmulDescAttributes_t attr, T value)
    {
        EXPECT_EQ(hipblasLtMatmulDescSetAttribute(desc, attr, &value, sizeof(value)),
                  HIPBLAS_STATUS_SUCCESS);
    }

    // Every combination the rewrite refuses, beside controls that differ in one setting and are
    // accepted. Only A and B are rewritten, so a row-major C or D is refused, as is anything whose
    // layout is tied to how A or B is stored.
    TEST_F(RowOrderMatmul, smoke_RefusesWhatTheRewriteCannotExpress)
    {
        constexpr int64_t kSq = 64;
        auto              set = [](auto attr, auto value) {
            return [=](hipblasLtMatmulDesc_t desc) { setAttribute(desc, attr, value); };
        };
        const auto none        = [](hipblasLtMatmulDesc_t) {};
        const auto opC         = set(HIPBLASLT_MATMUL_DESC_TRANSA, HIPBLAS_OP_C);
        const auto opBC        = set(HIPBLASLT_MATMUL_DESC_TRANSB, HIPBLAS_OP_C);
        const auto opJunk      = set(HIPBLASLT_MATMUL_DESC_TRANSA, hipblasOperation_t(5));
        const auto bias        = set(HIPBLASLT_MATMUL_DESC_EPILOGUE, HIPBLASLT_EPILOGUE_BIAS);
        const auto usoOn       = set(HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT, int32_t(1));
        const auto alphaVector = set(HIPBLASLT_MATMUL_DESC_POINTER_MODE,
                                     HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST);
        const auto scaleA      = [&](hipblasLtMatmulMatrixScale_t m) {
            return set(HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, m);
        };
        const auto scaleB = [&](hipblasLtMatmulMatrixScale_t m) {
            return set(HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, m);
        };
        constexpr auto NS  = HIPBLAS_STATUS_NOT_SUPPORTED;
        constexpr auto OK  = HIPBLAS_STATUS_SUCCESS;
        constexpr auto F   = HIP_R_32F;
        constexpr auto C32 = HIPBLAS_COMPUTE_32F;
        constexpr auto V32 = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        constexpr auto VEC = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;

        const std::vector<Guard> guards = {
            {"only C row-major", C32, none, F, F, false, false, true, false, NS},
            {"only D row-major", C32, none, F, F, false, false, false, true, NS},
            {"C and D row-major", C32, none, F, F, false, false, true, true, NS},
            {"every layout row-major, as reported", C32, none, F, F, true, true, true, true, NS},
            {"op C on A, A row-major", C32, opC, F, F, true, false, false, false, NS},
            {"op C on B, B row-major", C32, opBC, F, F, false, true, false, false, NS},
            {"unrecognized op on A, B row-major", C32, opJunk, F, F, false, true, false, false, NS},
            {"block scale on A, A row-major",
             C32,
             scaleA(V32),
             F,
             F,
             true,
             false,
             false,
             false,
             NS},
            {"block scale on B, A row-major",
             C32,
             scaleB(V32),
             F,
             F,
             true,
             false,
             false,
             false,
             NS},
            {"vector scale on A, B row-major",
             C32,
             scaleA(VEC),
             F,
             F,
             false,
             true,
             false,
             false,
             NS},
            {"vector scale on B, B row-major",
             C32,
             scaleB(VEC),
             F,
             F,
             false,
             true,
             false,
             false,
             NS},
            {"int4 A, A row-major", C32, none, HIP_R_4I, F, true, false, false, false, NS},
            {"int4 B, B row-major", C32, none, F, HIP_R_4I, false, true, false, false, NS},
            {"control: bias, A row-major", C32, bias, F, F, true, false, false, false, OK},
            {"control: alpha vector, A row-major",
             C32,
             alphaVector,
             F,
             F,
             true,
             false,
             false,
             false,
             OK},
            {"control: uniform summation, A and B row-major",
             C32,
             usoOn,
             F,
             F,
             true,
             true,
             false,
             false,
             OK},
            {"control: A and B types differ, A row-major",
             C32,
             none,
             HIP_R_16F,
             F,
             true,
             false,
             false,
             false,
             OK},
        };

        hipblasLtMatmulPreference_t pref = nullptr;
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        void* buffer = nullptr;
        ASSERT_EQ(hipMalloc(&buffer, kSq * kSq * sizeof(float)), hipSuccess);
        const float alpha = 1.0f, beta = 0.0f;

        for(const Guard& g : guards)
        {
            SCOPED_TRACE(g.name);
            hipblasLtMatmulDesc_t desc = nullptr;
            ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, g.compute, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            g.configure(desc);
            auto layout = [&](hipDataType type, bool row) {
                return makeLayout(
                    type, stored(kSq, kSq, 1, row ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL));
            };
            hipblasLtMatrixLayout_t lA = layout(g.typeA, g.rowA), lB = layout(g.typeB, g.rowB);
            hipblasLtMatrixLayout_t lC = layout(F, g.rowC), lD = layout(F, g.rowD);

            hipblasLtMatmulHeuristicResult_t result{};
            int                              returned = -1;
            EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                          handle, desc, lA, lB, lC, lD, pref, 1, &result, &returned),
                      g.expected);
            if(g.expected == NS)
            {
                EXPECT_EQ(returned, 0);
                EXPECT_EQ(hipblasLtMatmul(handle,
                                          desc,
                                          &alpha,
                                          buffer,
                                          lA,
                                          buffer,
                                          lB,
                                          &beta,
                                          buffer,
                                          lC,
                                          buffer,
                                          lD,
                                          nullptr,
                                          nullptr,
                                          0,
                                          nullptr),
                          NS);
            }

            for(auto l : {lA, lB, lC, lD})
                hipblasLtMatrixLayoutDestroy(l);
            hipblasLtMatmulDescDestroy(desc);
        }

        static_cast<void>(hipFree(buffer));
        hipblasLtMatmulPreferenceDestroy(pref);
    }

    // HIPBLASLT_OPERATION_INVALID is reported by normal validation whatever the order, and a
    // missing argument keeps its usual status even when the orders cannot be rewritten.
    TEST_F(RowOrderMatmul, smoke_InvalidArgumentsKeepTheirStatus)
    {
        constexpr int64_t           kSq  = 64;
        hipblasLtMatmulPreference_t pref = nullptr;
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        void* buffer = nullptr;
        ASSERT_EQ(hipMalloc(&buffer, kSq * kSq * sizeof(float)), hipSuccess);
        hipblasLtMatrixLayout_t col
            = makeLayout(HIP_R_32F, stored(kSq, kSq, 1, HIPBLASLT_ORDER_COL));
        hipblasLtMatrixLayout_t row
            = makeLayout(HIP_R_32F, stored(kSq, kSq, 1, HIPBLASLT_ORDER_ROW));

        hipblasLtMatmulDesc_t invalidOp = nullptr;
        ASSERT_EQ(hipblasLtMatmulDescCreate(&invalidOp, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        setAttribute(invalidOp, HIPBLASLT_MATMUL_DESC_TRANSA, HIPBLASLT_OPERATION_INVALID);

        const float alpha = 1.0f, beta = 0.0f;
        auto        heuristic
            = [&](hipblasLtHandle_t h, hipblasLtMatrixLayout_t a, hipblasLtMatrixLayout_t d) {
                  hipblasLtMatmulHeuristicResult_t result{};
                  int                              returned = -1;
                  const hipblasStatus_t            status   = hipblasLtMatmulAlgoGetHeuristic(
                      h, invalidOp, a, col, col, d, pref, 1, &result, &returned);
                  return std::make_pair(status, returned);
              };
        auto matmul = [&](hipblasLtMatrixLayout_t a, hipblasLtMatrixLayout_t d) {
            return hipblasLtMatmul(handle,
                                   invalidOp,
                                   &alpha,
                                   buffer,
                                   a,
                                   buffer,
                                   col,
                                   &beta,
                                   buffer,
                                   col,
                                   buffer,
                                   d,
                                   nullptr,
                                   nullptr,
                                   0,
                                   nullptr);
        };

        EXPECT_EQ(heuristic(handle, row, col), heuristic(handle, col, col))
            << "invalid op, A row-major";
        EXPECT_EQ(heuristic(handle, col, col).second, 0) << "invalid op gets no algorithms";
        EXPECT_NE(matmul(col, col), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(matmul(row, col), matmul(col, col)) << "invalid op, A row-major";
        EXPECT_EQ(matmul(col, row), matmul(col, col)) << "invalid op, only D row-major";

        EXPECT_NE(heuristic(nullptr, col, col).first, HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(nullptr, col, row).first, heuristic(nullptr, col, col).first)
            << "null handle, only D row-major";

        hipblasLtMatmulDescDestroy(invalidOp);
        hipblasLtMatrixLayoutDestroy(col);
        hipblasLtMatrixLayoutDestroy(row);
        static_cast<void>(hipFree(buffer));
        hipblasLtMatmulPreferenceDestroy(pref);
    }
} // namespace
