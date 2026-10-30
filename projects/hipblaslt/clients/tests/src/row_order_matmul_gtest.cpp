// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// hipblasLtMatmul with HIPBLASLT_ORDER_ROW layouts (ROCM-29960). The library rewrites a
// row-major A or B as its column-major transpose; these tests check the numbers it produces
// and the combinations it refuses.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt_float8.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    constexpr size_t kWorkspace = size_t(32) << 20;

    // Owners for the objects a test creates, so a failed assertion still releases them.
    using LayoutPtr
        = std::unique_ptr<hipblasLtMatrixLayoutOpaque_t, decltype(&hipblasLtMatrixLayoutDestroy)>;
    using DescPtr
        = std::unique_ptr<hipblasLtMatmulDescOpaque_t, decltype(&hipblasLtMatmulDescDestroy)>;
    using PrefPtr = std::unique_ptr<hipblasLtMatmulPreferenceOpaque_t,
                                    decltype(&hipblasLtMatmulPreferenceDestroy)>;

    class DeviceBuffer
    {
    public:
        explicit DeviceBuffer(size_t bytes)
        {
            EXPECT_EQ(hipMalloc(&ptr_, bytes), hipSuccess);
        }
        DeviceBuffer(DeviceBuffer&& other) noexcept
            : ptr_(other.ptr_)
        {
            other.ptr_ = nullptr;
        }
        DeviceBuffer(const DeviceBuffer&)            = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(DeviceBuffer&&)      = delete;
        ~DeviceBuffer()
        {
            if(ptr_)
                static_cast<void>(hipFree(ptr_));
        }
        void* get() const
        {
            return ptr_;
        }

    private:
        void* ptr_ = nullptr;
    };

    template <typename T>
    DeviceBuffer upload(const std::vector<T>& host)
    {
        DeviceBuffer device(host.size() * sizeof(T));
        EXPECT_EQ(
            hipMemcpy(device.get(), host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
            hipSuccess);
        return device;
    }

    DescPtr makeDesc(hipblasComputeType_t compute = HIPBLAS_COMPUTE_32F)
    {
        hipblasLtMatmulDesc_t desc = nullptr;
        EXPECT_EQ(hipblasLtMatmulDescCreate(&desc, compute, HIP_R_32F), HIPBLAS_STATUS_SUCCESS);
        return DescPtr(desc, &hipblasLtMatmulDescDestroy);
    }

    PrefPtr makePref(uint64_t workspace = 0)
    {
        hipblasLtMatmulPreference_t pref = nullptr;
        EXPECT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(
            hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace)),
            HIPBLAS_STATUS_SUCCESS);
        return PrefPtr(pref, &hipblasLtMatmulPreferenceDestroy);
    }

    template <typename T>
    void setAttribute(hipblasLtMatmulDesc_t desc, hipblasLtMatmulDescAttributes_t attr, T value)
    {
        EXPECT_EQ(hipblasLtMatmulDescSetAttribute(desc, attr, &value, sizeof(value)),
                  HIPBLAS_STATUS_SUCCESS);
    }

    // A stored r x c matrix. `stored` pads the leading dimension past the packed one, so a
    // leading dimension taken from the wrong axis reads the wrong elements.
    struct Stored
    {
        int64_t          rows, cols, ld, batch;
        hipblasLtOrder_t order;

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
        const int64_t ld = order == HIPBLASLT_ORDER_ROW ? cols + 2 : rows + 3;
        return Stored{rows, cols, ld, batch, order};
    }

    Stored packed(int64_t rows, int64_t cols, hipblasLtOrder_t order)
    {
        return Stored{rows, cols, order == HIPBLASLT_ORDER_ROW ? cols : rows, 1, order};
    }

    LayoutPtr makeLayout(hipDataType type, const Stored& s)
    {
        hipblasLtMatrixLayout_t layout = nullptr;
        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&layout, type, s.rows, s.cols, s.ld),
                  HIPBLAS_STATUS_SUCCESS);
        LayoutPtr     owner(layout, &hipblasLtMatrixLayoutDestroy);
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
        return owner;
    }

    LayoutPtr squareLayout(bool rowMajor, hipDataType type = HIP_R_32F)
    {
        constexpr int64_t kSq = 64;
        return makeLayout(
            type, stored(kSq, kSq, 1, rowMajor ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL));
    }

    // Small integers, so every product and sum below is exact in fp8, fp16 and fp32 and the
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
    std::vector<T> fill(const Stored& s, float (*value)(int64_t, int64_t, int64_t))
    {
        std::vector<T> host(s.elements());
        for(int64_t b = 0; b < s.batch; b++)
            for(int64_t i = 0; i < s.rows; i++)
                for(int64_t j = 0; j < s.cols; j++)
                    host[s.index(b, i, j)] = T(value(b, i, j));
        return host;
    }

    // Compares D, a column-major m x n matrix, with alpha * scale * op(A) * op(B) + beta * C
    // computed from the logical matrices.
    template <typename T>
    void expectProduct(const std::vector<T>& hD,
                       const Stored&         sD,
                       int64_t               k,
                       bool                  tA,
                       bool                  tB,
                       float                 alphaScale,
                       float                 beta)
    {
        int mismatches = 0;
        for(int64_t b = 0; b < sD.batch; b++)
            for(int64_t i = 0; i < sD.rows; i++)
                for(int64_t j = 0; j < sD.cols; j++)
                {
                    float sum = 0;
                    for(int64_t l = 0; l < k; l++)
                        sum += (tA ? valueA(b, l, i) : valueA(b, i, l))
                               * (tB ? valueB(b, j, l) : valueB(b, l, j));
                    const float expected = alphaScale * sum + beta * valueC(b, i, j);
                    const float actual   = float(hD[sD.index(b, i, j)]);
                    if(actual != expected && mismatches++ < 3)
                        ADD_FAILURE() << "D(" << b << ", " << i << ", " << j << ") is " << actual
                                      << ", expected " << expected;
                }
        EXPECT_EQ(mismatches, 0);
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

        const float        alpha = 2.0f, beta = 3.0f;
        const DeviceBuffer dA = upload(fill<T>(sA, valueA));
        const DeviceBuffer dB = upload(fill<T>(sB, valueB));
        const DeviceBuffer dC = upload(fill<T>(sC, valueC));
        const DeviceBuffer dD = upload(std::vector<T>(sC.elements(), T(0)));
        const DeviceBuffer work(kWorkspace);

        const DescPtr desc = makeDesc();
        setAttribute(desc.get(), HIPBLASLT_MATMUL_DESC_TRANSA, tc.opA);
        setAttribute(desc.get(), HIPBLASLT_MATMUL_DESC_TRANSB, tc.opB);
        const LayoutPtr lA = makeLayout(tc.type, sA), lB = makeLayout(tc.type, sB);
        const LayoutPtr lC = makeLayout(tc.type, sC), lD = makeLayout(tc.type, sC);
        const PrefPtr   pref = makePref(kWorkspace);

        hipblasLtMatmulHeuristicResult_t result{};
        int                              returned = 0;
        ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                  desc.get(),
                                                  lA.get(),
                                                  lB.get(),
                                                  lC.get(),
                                                  lD.get(),
                                                  pref.get(),
                                                  1,
                                                  &result,
                                                  &returned),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_GT(returned, 0);
        ASSERT_EQ(hipblasLtMatmul(handle,
                                  desc.get(),
                                  &alpha,
                                  dA.get(),
                                  lA.get(),
                                  dB.get(),
                                  lB.get(),
                                  &beta,
                                  dC.get(),
                                  lC.get(),
                                  dD.get(),
                                  lD.get(),
                                  &result.algo,
                                  work.get(),
                                  kWorkspace,
                                  nullptr),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        std::vector<T> hD(sC.elements());
        ASSERT_EQ(hipMemcpy(hD.data(), dD.get(), hD.size() * sizeof(T), hipMemcpyDeviceToHost),
                  hipSuccess);
        expectProduct(hD, sC, kK, tA, tB, alpha, beta);
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

    // Runs one scalar-scaled FP8 GEMM with the given orders and transposes, and checks D. Each
    // case the test passes in reduces to the TN problem FP8 kernels are tuned for. Returns false
    // when the library has no kernel for the problem.
    template <typename F8, typename TD>
    bool runScaledFp8(hipblasLtHandle_t  handle,
                      hipDataType        typeAB,
                      hipDataType        typeD,
                      hipblasOperation_t opA,
                      hipblasOperation_t opB,
                      hipblasLtOrder_t   orderA,
                      hipblasLtOrder_t   orderB)
    {
        constexpr int64_t m = 64, n = 32, k = 128;
        const bool        tA = opA == HIPBLAS_OP_T, tB = opB == HIPBLAS_OP_T;
        const Stored      sA = packed(tA ? k : m, tA ? m : k, orderA);
        const Stored      sB = packed(tB ? n : k, tB ? k : n, orderB);
        const Stored      sD = packed(m, n, HIPBLASLT_ORDER_COL);

        // Distinct scales, so a scale applied to the wrong operand or dropped changes D.
        const float        scaleA = 2.0f, scaleB = 3.0f, alpha = 1.0f, beta = 0.0f;
        const DeviceBuffer dScaleA = upload(std::vector<float>{scaleA});
        const DeviceBuffer dScaleB = upload(std::vector<float>{scaleB});
        const DeviceBuffer dA      = upload(fill<F8>(sA, valueA));
        const DeviceBuffer dB      = upload(fill<F8>(sB, valueB));
        const DeviceBuffer dD      = upload(std::vector<TD>(sD.elements(), TD(0)));
        const DeviceBuffer work(kWorkspace);

        const DescPtr desc = makeDesc();
        setAttribute(desc.get(), HIPBLASLT_MATMUL_DESC_TRANSA, opA);
        setAttribute(desc.get(), HIPBLASLT_MATMUL_DESC_TRANSB, opB);
        setAttribute(desc.get(),
                     HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                     HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);
        setAttribute(desc.get(),
                     HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                     HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);
        setAttribute(desc.get(), HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, dScaleA.get());
        setAttribute(desc.get(), HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, dScaleB.get());
        const LayoutPtr lA = makeLayout(typeAB, sA), lB = makeLayout(typeAB, sB);
        const LayoutPtr lD   = makeLayout(typeD, sD);
        const PrefPtr   pref = makePref(kWorkspace);

        hipblasLtMatmulHeuristicResult_t result{};
        int                              returned = 0;
        EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                  desc.get(),
                                                  lA.get(),
                                                  lB.get(),
                                                  lD.get(),
                                                  lD.get(),
                                                  pref.get(),
                                                  1,
                                                  &result,
                                                  &returned),
                  HIPBLAS_STATUS_SUCCESS);
        if(returned == 0)
            return false;
        EXPECT_EQ(hipblasLtMatmul(handle,
                                  desc.get(),
                                  &alpha,
                                  dA.get(),
                                  lA.get(),
                                  dB.get(),
                                  lB.get(),
                                  &beta,
                                  dD.get(),
                                  lD.get(),
                                  dD.get(),
                                  lD.get(),
                                  &result.algo,
                                  work.get(),
                                  kWorkspace,
                                  nullptr),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        std::vector<TD> hD(sD.elements());
        EXPECT_EQ(hipMemcpy(hD.data(), dD.get(), hD.size() * sizeof(TD), hipMemcpyDeviceToHost),
                  hipSuccess);
        expectProduct(hD, sD, k, tA, tB, alpha * scaleA * scaleB, beta);
        return true;
    }

    // A row-major A or B with scalar A and B scales, the scaling the rewrite accepts. Runs for
    // each FP8 type the loaded library has a TN kernel for (gfx942 and gfx950 ship them), and
    // skips where it has none.
    TEST_F(RowOrderMatmul, smoke_ScalarScaledFp8MatchesReference)
    {
        constexpr auto N = HIPBLAS_OP_N, T = HIPBLAS_OP_T;
        constexpr auto COL = HIPBLASLT_ORDER_COL, ROW = HIPBLASLT_ORDER_ROW;
        auto           runAll = [&](auto f8, hipDataType typeAB) {
            using F8 = decltype(f8);
            for(hipDataType typeD : {HIP_R_32F, HIP_R_16F})
            {
                auto run = [&](hipblasOperation_t opA,
                               hipblasOperation_t opB,
                               hipblasLtOrder_t   orderA,
                               hipblasLtOrder_t   orderB) {
                    return typeD == HIP_R_32F
                                         ? runScaledFp8<F8, float>(
                                     handle, typeAB, typeD, opA, opB, orderA, orderB)
                                         : runScaledFp8<F8, _Float16>(
                                     handle, typeAB, typeD, opA, opB, orderA, orderB);
                };
                SCOPED_TRACE("fp8 type " + std::to_string(int(typeAB)) + ", D type "
                             + std::to_string(int(typeD)));
                // The column-major TN control decides whether this library has the kernels.
                if(!run(T, N, COL, COL))
                    continue;
                EXPECT_TRUE(run(N, N, ROW, COL)) << "A row-major: no algorithm";
                EXPECT_TRUE(run(T, T, COL, ROW)) << "B row-major: no algorithm";
                EXPECT_TRUE(run(N, T, ROW, ROW)) << "A and B row-major: no algorithm";
                return true;
            }
            return false;
        };
        const bool ranFnuz = runAll(hipblaslt_f8_fnuz(0.0f), HIP_R_8F_E4M3_FNUZ);
        const bool ranOcp  = runAll(hipblaslt_f8(0.0f), HIP_R_8F_E4M3);
        if(!ranFnuz && !ranOcp)
            GTEST_SKIP() << "the loaded library has no scalar-scaled FP8 TN kernel for this GPU";
    }

    // The matrices from the report: fp32 2x3 * 3x4 with A and B row-major and packed leading
    // dimensions, as reported, and D column-major. The reported row-major D is refused (see
    // smoke_RefusesWhatTheRewriteCannotExpress).
    TEST_F(RowOrderMatmul, smoke_ReportedCase)
    {
        const DeviceBuffer dA = upload(std::vector<float>{1, 2, 3, 4, 5, 6});
        const DeviceBuffer dB = upload(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        const DeviceBuffer dD = upload(std::vector<float>(8, 0.0f));

        const DescPtr   desc = makeDesc();
        const LayoutPtr lA   = makeLayout(HIP_R_32F, {2, 3, 3, 1, HIPBLASLT_ORDER_ROW});
        const LayoutPtr lB   = makeLayout(HIP_R_32F, {3, 4, 4, 1, HIPBLASLT_ORDER_ROW});
        const LayoutPtr lD   = makeLayout(HIP_R_32F, {2, 4, 2, 1, HIPBLASLT_ORDER_COL});

        const float alpha = 1.0f, beta = 0.0f;
        ASSERT_EQ(hipblasLtMatmul(handle,
                                  desc.get(),
                                  &alpha,
                                  dA.get(),
                                  lA.get(),
                                  dB.get(),
                                  lB.get(),
                                  &beta,
                                  dD.get(),
                                  lD.get(),
                                  dD.get(),
                                  lD.get(),
                                  nullptr,
                                  nullptr,
                                  0,
                                  nullptr),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        std::vector<float> hD(8);
        ASSERT_EQ(hipMemcpy(hD.data(), dD.get(), hD.size() * sizeof(float), hipMemcpyDeviceToHost),
                  hipSuccess);
        EXPECT_EQ(hD, (std::vector<float>{38, 83, 44, 98, 50, 113, 56, 128}));
    }

    struct Guard
    {
        std::string                                name;
        std::function<void(hipblasLtMatmulDesc_t)> configure;
        hipDataType                                typeA, typeB;
        bool                                       rowA, rowB, rowC, rowD;
        hipblasStatus_t                            expected;
    };

    // Every combination the rewrite refuses, beside controls that differ in one setting and are
    // accepted. Only A and B are rewritten, so a row-major C or D is refused, as is anything whose
    // layout is tied to how A or B is stored.
    TEST_F(RowOrderMatmul, smoke_RefusesWhatTheRewriteCannotExpress)
    {
        auto set = [](auto attr, auto value) {
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
        constexpr auto V32 = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        constexpr auto VEC = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;

        // clang-format off
        const std::vector<Guard> guards = {
            {"only C row-major",                              none,        F, F, false, false, true,  false, NS},
            {"only D row-major",                              none,        F, F, false, false, false, true,  NS},
            {"C and D row-major",                             none,        F, F, false, false, true,  true,  NS},
            {"every layout row-major, as reported",           none,        F, F, true,  true,  true,  true,  NS},
            {"op C on A, A row-major",                        opC,         F, F, true,  false, false, false, NS},
            {"op C on B, B row-major",                        opBC,        F, F, false, true,  false, false, NS},
            {"unrecognized op on A, B row-major",             opJunk,      F, F, false, true,  false, false, NS},
            {"block scale on A, A row-major",                 scaleA(V32), F, F, true,  false, false, false, NS},
            {"block scale on B, A row-major",                 scaleB(V32), F, F, true,  false, false, false, NS},
            {"vector scale on A, B row-major",                scaleA(VEC), F, F, false, true,  false, false, NS},
            {"vector scale on B, B row-major",                scaleB(VEC), F, F, false, true,  false, false, NS},
            {"int4 A, A row-major",                           none, HIP_R_4I, F, true,  false, false, false, NS},
            {"int4 B, B row-major",                           none, F, HIP_R_4I, false, true,  false, false, NS},
            {"control: bias, A row-major",                    bias,        F, F, true,  false, false, false, OK},
            {"control: alpha vector, A row-major",            alphaVector, F, F, true,  false, false, false, OK},
            {"control: uniform summation, A and B row-major", usoOn,       F, F, true,  true,  false, false, OK},
        };
        // clang-format on

        const PrefPtr      pref = makePref();
        const DeviceBuffer buffer(64 * 64 * sizeof(float));
        const float        alpha = 1.0f, beta = 0.0f;

        for(const Guard& g : guards)
        {
            SCOPED_TRACE(g.name);
            const DescPtr desc = makeDesc();
            g.configure(desc.get());
            const LayoutPtr lA = squareLayout(g.rowA, g.typeA), lB = squareLayout(g.rowB, g.typeB);
            const LayoutPtr lC = squareLayout(g.rowC), lD = squareLayout(g.rowD);

            hipblasLtMatmulHeuristicResult_t result{};
            int                              returned = -1;
            EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                      desc.get(),
                                                      lA.get(),
                                                      lB.get(),
                                                      lC.get(),
                                                      lD.get(),
                                                      pref.get(),
                                                      1,
                                                      &result,
                                                      &returned),
                      g.expected);
            if(g.expected == OK)
            {
                EXPECT_GT(returned, 0) << "an accepted combination must get an algorithm";
                continue;
            }
            EXPECT_EQ(returned, 0);
            EXPECT_EQ(hipblasLtMatmul(handle,
                                      desc.get(),
                                      &alpha,
                                      buffer.get(),
                                      lA.get(),
                                      buffer.get(),
                                      lB.get(),
                                      &beta,
                                      buffer.get(),
                                      lC.get(),
                                      buffer.get(),
                                      lD.get(),
                                      nullptr,
                                      nullptr,
                                      0,
                                      nullptr),
                      NS);
        }
    }

    // A missing argument or HIPBLASLT_OPERATION_INVALID returns the same status with row-major
    // layouts as with column-major ones, including where the order would be refused.
    TEST_F(RowOrderMatmul, smoke_InvalidArgumentsKeepTheirStatus)
    {
        const PrefPtr           pref = makePref();
        const DeviceBuffer      buffer(64 * 64 * sizeof(float));
        const LayoutPtr         colOwner = squareLayout(false), rowOwner = squareLayout(true);
        hipblasLtMatrixLayout_t col = colOwner.get(), row = rowOwner.get();

        const DescPtr plain     = makeDesc();
        const DescPtr invalidOp = makeDesc();
        setAttribute(invalidOp.get(), HIPBLASLT_MATMUL_DESC_TRANSA, HIPBLASLT_OPERATION_INVALID);

        const float alpha = 1.0f, beta = 0.0f;
        void* const data      = buffer.get();
        auto        heuristic = [&](hipblasLtHandle_t       h,
                             hipblasLtMatmulDesc_t   desc,
                             hipblasLtMatrixLayout_t a,
                             hipblasLtMatrixLayout_t d) {
            hipblasLtMatmulHeuristicResult_t result{};
            int                              returned = -1;
            const hipblasStatus_t            status   = hipblasLtMatmulAlgoGetHeuristic(
                h, desc, a, col, col, d, pref.get(), 1, &result, &returned);
            return std::make_pair(status, returned);
        };
        auto matmul = [&](hipblasLtHandle_t       h,
                          hipblasLtMatmulDesc_t   desc,
                          const void*             alphaPtr,
                          void*                   dPtr,
                          hipblasLtMatrixLayout_t a,
                          hipblasLtMatrixLayout_t d) {
            return hipblasLtMatmul(h,
                                   desc,
                                   alphaPtr,
                                   data,
                                   a,
                                   data,
                                   col,
                                   &beta,
                                   data,
                                   col,
                                   dPtr,
                                   d,
                                   nullptr,
                                   nullptr,
                                   0,
                                   nullptr);
        };

        // Each call is made with column-major layouts and with the row-major layout in the
        // position named, and must fail the same way both times.
        struct Call
        {
            std::string                                 name;
            std::function<int(hipblasLtMatrixLayout_t)> run;
        };
        const std::vector<Call> calls = {
            {"heuristic, null handle, only D row-major",
             [&](hipblasLtMatrixLayout_t l) {
                 return int(heuristic(nullptr, plain.get(), col, l).first);
             }},
            {"matmul, invalid op, A row-major",
             [&](hipblasLtMatrixLayout_t l) {
                 return int(matmul(handle, invalidOp.get(), &alpha, data, l, col));
             }},
            {"matmul, invalid op, only D row-major",
             [&](hipblasLtMatrixLayout_t l) {
                 return int(matmul(handle, invalidOp.get(), &alpha, data, col, l));
             }},
            {"matmul, null handle, only D row-major",
             [&](hipblasLtMatrixLayout_t l) {
                 return int(matmul(nullptr, plain.get(), &alpha, data, col, l));
             }},
            {"matmul, null alpha, only D row-major",
             [&](hipblasLtMatrixLayout_t l) {
                 return int(matmul(handle, plain.get(), nullptr, data, col, l));
             }},
            {"matmul, null D, only D row-major",
             [&](hipblasLtMatrixLayout_t l) {
                 return int(matmul(handle, plain.get(), &alpha, nullptr, col, l));
             }},
        };
        for(const Call& c : calls)
        {
            const int columnMajor = c.run(col);
            EXPECT_NE(columnMajor, int(HIPBLAS_STATUS_SUCCESS)) << c.name;
            EXPECT_EQ(c.run(row), columnMajor) << c.name;
        }

        // The heuristic reports an invalid op as success with no algorithms.
        EXPECT_EQ(heuristic(handle, invalidOp.get(), col, col),
                  std::make_pair(HIPBLAS_STATUS_SUCCESS, 0));
        EXPECT_EQ(heuristic(handle, invalidOp.get(), row, col),
                  std::make_pair(HIPBLAS_STATUS_SUCCESS, 0))
            << "invalid op, A row-major";
    }
} // namespace
