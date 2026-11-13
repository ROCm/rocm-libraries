// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Error paths of algorithm selection and use, which must return a status rather than crash.

#include "scoped_env.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{
    using test_env::ScopedEnv;

    // Square, so one buffer size and one layout serve A, B, C and D.
    constexpr int64_t     kSize = 128;
    constexpr hipDataType kType = HIP_R_16F;

    // No solution library holds this many solutions.
    constexpr int kUnknownIndex = 0x3fffffff;

    using LayoutPtr
        = std::unique_ptr<hipblasLtMatrixLayoutOpaque_t, decltype(&hipblasLtMatrixLayoutDestroy)>;

    // A square layout with the given order, owned so a failed assertion still destroys it.
    LayoutPtr layoutWithOrder(hipblasLtOrder_t order)
    {
        hipblasLtMatrixLayout_t l = nullptr;
        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&l, kType, kSize, kSize, kSize),
                  HIPBLAS_STATUS_SUCCESS);
        const int32_t value = order;
        EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                      l, HIPBLASLT_MATRIX_LAYOUT_ORDER, &value, sizeof(value)),
                  HIPBLAS_STATUS_SUCCESS);
        return LayoutPtr(l, &hipblasLtMatrixLayoutDestroy);
    }

    hipblasLtMatmulAlgo_t algoWithUnknownIndex()
    {
        hipblasLtMatmulAlgo_t algo{};
        *reinterpret_cast<int*>(algo.data) = kUnknownIndex;
        return algo;
    }

    // One FP16 NN problem. The buffers are never read: every call under test
    // returns before a kernel is launched.
    class AlgoErrors : public ::testing::Test
    {
    protected:
        hipblasLtHandle_t           handle = nullptr;
        hipblasLtMatmulDesc_t       desc   = nullptr;
        hipblasLtMatrixLayout_t     layout = nullptr;
        hipblasLtMatmulPreference_t pref   = nullptr;
        void*                       a      = nullptr;
        void*                       b      = nullptr;
        void*                       c      = nullptr;
        void*                       d      = nullptr;
        float                       alpha  = 1.0f;
        float                       beta   = 0.0f;

        void SetUp() override
        {
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layout, kType, kSize, kSize, kSize),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
            for(void** buffer : {&a, &b, &c, &d})
                ASSERT_EQ(hipMalloc(buffer, kSize * kSize * sizeof(uint16_t)), hipSuccess);
        }

        void TearDown() override
        {
            for(void* buffer : {a, b, c, d})
                static_cast<void>(hipFree(buffer));
            if(pref)
                hipblasLtMatmulPreferenceDestroy(pref);
            if(layout)
                hipblasLtMatrixLayoutDestroy(layout);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
            if(handle)
                hipblasLtDestroy(handle);
        }

        hipblaslt_ext::GemmInputs inputs()
        {
            hipblaslt_ext::GemmInputs in;
            in.setA(a);
            in.setB(b);
            in.setC(c);
            in.setD(d);
            in.setAlpha(&alpha);
            in.setBeta(&beta);
            return in;
        }

        hipblasStatus_t matmulWithUnknownIndex()
        {
            const hipblasLtMatmulAlgo_t algo = algoWithUnknownIndex();
            return hipblasLtMatmul(handle,
                                   desc,
                                   &alpha,
                                   a,
                                   layout,
                                   b,
                                   layout,
                                   &beta,
                                   c,
                                   layout,
                                   d,
                                   layout,
                                   &algo,
                                   nullptr,
                                   0,
                                   nullptr);
        }
    };

    TEST_F(AlgoErrors, smoke_GemmInitializeRejectsUnknownIndex)
    {
        hipblaslt_ext::Gemm gemm(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N, kType, kType, kType, kType, HIPBLAS_COMPUTE_32F);
        hipblaslt_ext::GemmEpilogue epilogue;
        hipblaslt_ext::GemmInputs   in = inputs();
        ASSERT_EQ(gemm.setProblem(kSize, kSize, kSize, 1, epilogue, in), HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(gemm.initialize(algoWithUnknownIndex(), nullptr), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(AlgoErrors, smoke_GroupedGemmInitializeRejectsUnknownIndex)
    {
        hipblaslt_ext::GroupedGemm gemm(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N, kType, kType, kType, kType, HIPBLAS_COMPUTE_32F);
        std::vector<int64_t>                     size(2, kSize), batch(2, 1);
        std::vector<hipblaslt_ext::GemmEpilogue> epilogue(2);
        std::vector<hipblaslt_ext::GemmInputs>   in(2, inputs());
        ASSERT_EQ(gemm.setProblem(size, size, size, batch, epilogue, in), HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(gemm.initialize(algoWithUnknownIndex(), nullptr), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(AlgoErrors, smoke_HeuristicReportsZeroAlgosWhenRequestIsRejected)
    {
        hipblasLtMatmulHeuristicResult_t result{};
        int                              returned = -1;
        EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                      handle, desc, layout, layout, layout, layout, pref, 0, &result, &returned),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(returned, 0);
    }

    // Kernels take batch strides as 32-bit values, so a strided batch whose stride does not fit
    // must get no algorithms, while the largest stride that fits still gets some.
    TEST_F(AlgoErrors, smoke_HeuristicRefusesBatchStridesPast32Bits)
    {
        auto algosFor = [&](int64_t stride) {
            hipblasLtMatrixLayout_t batched = nullptr;
            EXPECT_EQ(hipblasLtMatrixLayoutCreate(&batched, kType, kSize, kSize, kSize),
                      HIPBLAS_STATUS_SUCCESS);
            const int32_t count = 2;
            EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                          batched, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)),
                      HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(
                hipblasLtMatrixLayoutSetAttribute(
                    batched, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)),
                HIPBLAS_STATUS_SUCCESS);
            hipblasLtMatmulHeuristicResult_t result{};
            int                              returned = 0;
            static_cast<void>(hipblasLtMatmulAlgoGetHeuristic(
                handle, desc, batched, batched, batched, batched, pref, 1, &result, &returned));
            hipblasLtMatrixLayoutDestroy(batched);
            return returned;
        };
        EXPECT_GT(algosFor(std::numeric_limits<uint32_t>::max()), 0);
        EXPECT_EQ(algosFor(int64_t(1) << 32), 0);
    }

    // Kernels take M, N, K and the leading dimensions as 32-bit values too. With both
    // operations N, A is M x K, B is K x N, and C and D are M x N, so a K of 2^32 also makes
    // ldb 2^32 and an M of 2^32 also makes ldc and ldd 2^32.
    TEST_F(AlgoErrors, smoke_HeuristicRefusesSizesPast32Bits)
    {
        auto algosFor = [&](int64_t m, int64_t n, int64_t k, int64_t lda) {
            hipblasLtMatrixLayout_t a = nullptr, b = nullptr, cd = nullptr;
            EXPECT_EQ(hipblasLtMatrixLayoutCreate(&a, kType, m, k, lda), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipblasLtMatrixLayoutCreate(&b, kType, k, n, k), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipblasLtMatrixLayoutCreate(&cd, kType, m, n, m), HIPBLAS_STATUS_SUCCESS);
            hipblasLtMatmulHeuristicResult_t result{};
            int                              returned = 0;
            static_cast<void>(hipblasLtMatmulAlgoGetHeuristic(
                handle, desc, a, b, cd, cd, pref, 1, &result, &returned));
            hipblasLtMatrixLayoutDestroy(a);
            hipblasLtMatrixLayoutDestroy(b);
            hipblasLtMatrixLayoutDestroy(cd);
            return returned;
        };
        constexpr int64_t past = int64_t(1) << 32;
        EXPECT_GT(algosFor(kSize, kSize, kSize, kSize), 0);
        EXPECT_EQ(algosFor(kSize, kSize, kSize, past), 0);
        EXPECT_EQ(algosFor(kSize, past, kSize, kSize), 0);
        EXPECT_EQ(algosFor(kSize, kSize, past, kSize), 0);
        EXPECT_EQ(algosFor(past, kSize, kSize, past), 0);
    }

    // Before the fix the search ran on these anyway: minutes on gfx942, SIGFPE on gfx950.
    TEST_F(AlgoErrors, smoke_HeuristicReportsNoAlgosForEmptyOrInvalidProblems)
    {
        auto algosFor = [&](int64_t m, int64_t n, int32_t batchA) {
            hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, lcd = nullptr;
            EXPECT_EQ(hipblasLtMatrixLayoutCreate(&la, kType, m, kSize, kSize),
                      HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipblasLtMatrixLayoutCreate(&lb, kType, kSize, n, kSize),
                      HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipblasLtMatrixLayoutCreate(&lcd, kType, m, n, kSize),
                      HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                          la, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchA, sizeof(batchA)),
                      HIPBLAS_STATUS_SUCCESS);
            hipblasLtMatmulHeuristicResult_t result{};
            int                              returned = -1;
            EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                          handle, desc, la, lb, lcd, lcd, pref, 1, &result, &returned),
                      HIPBLAS_STATUS_SUCCESS);
            for(auto l : {la, lb, lcd})
                hipblasLtMatrixLayoutDestroy(l);
            return returned;
        };
        EXPECT_GT(algosFor(kSize, kSize, 1), 0);
        EXPECT_EQ(algosFor(0, kSize, 1), 0) << "M = 0";
        EXPECT_EQ(algosFor(kSize, 0, 1), 0) << "N = 0";
        EXPECT_EQ(algosFor(kSize, kSize, 2), 0) << "A batch count differs from B, C and D";
    }

    TEST_F(AlgoErrors, smoke_MatmulRejectsUnknownIndex)
    {
        EXPECT_NE(matmulWithUnknownIndex(), HIPBLAS_STATUS_SUCCESS);
    }

    // Extended-profile logging looks up the kernel and solution names of the unknown index
    // before the matmul rejects it. The logger reads its environment once per process, so
    // the call runs in a newly started copy of this binary.
    TEST_F(AlgoErrors, smoke_MatmulRejectsUnknownIndexWithAllLogging)
    {
        GTEST_FLAG_SET(death_test_style, "threadsafe");
        ScopedEnv level("HIPBLASLT_LOG_LEVEL", nullptr);
        ScopedEnv mask("HIPBLASLT_LOG_MASK", "255");
        ScopedEnv file("HIPBLASLT_LOG_FILE", nullptr);

        EXPECT_EXIT(
            {
                const auto status = matmulWithUnknownIndex();
                // std::exit skips GoogleTest's fixture teardown, so invoke it manually.
                TearDown();
                std::exit(status != HIPBLAS_STATUS_SUCCESS ? 0 : 1);
            },
            ::testing::ExitedWithCode(0),
            "");
    }

#ifdef HIPBLASLT_USE_ROCROLLER
    // HIPBLASLT_USE_ROCROLLER=1 sends every problem down the rocRoller path, whose indices have
    // the top bit set. A non-negative index there belongs to a Tensile algo; decoded as a
    // rocRoller index it names an arbitrary tile, and building that kernel can take minutes.
    TEST_F(AlgoErrors, smoke_RocRollerHandleRejectsNonRocRollerIndex)
    {
        int             device = 0;
        hipDeviceProp_t props{};
        ASSERT_EQ(hipGetDevice(&device), hipSuccess);
        ASSERT_EQ(hipGetDeviceProperties(&props, device), hipSuccess);
        if(std::string(props.gcnArchName).find("gfx1250") != std::string::npos)
            GTEST_SKIP() << "gfx1250 handles never use rocRoller";

        using HandlePtr = std::unique_ptr<void, decltype(&hipblasLtDestroy)>;
        auto createWith = [](const char* useRocRoller) {
            ScopedEnv         use("HIPBLASLT_USE_ROCROLLER", useRocRoller);
            hipblasLtHandle_t h = nullptr;
            EXPECT_EQ(hipblasLtCreate(&h), HIPBLAS_STATUS_SUCCESS);
            return HandlePtr(h, &hipblasLtDestroy);
        };
        HandlePtr         rocrollerOwner = createWith("1");
        HandlePtr         tensileOwner   = createWith("0");
        hipblasLtHandle_t rocroller      = rocrollerOwner.get();
        ASSERT_NE(rocroller, nullptr);
        ASSERT_NE(tensileOwner.get(), nullptr);

        auto isSupported = [&](hipblasLtMatmulAlgo_t algo) {
            size_t workspace = 0;
            return hipblaslt_ext::matmulIsAlgoSupported(
                rocroller, desc, &alpha, layout, layout, &beta, layout, layout, algo, workspace);
        };

        hipblasLtMatmulHeuristicResult_t result{};
        int                              count = 0;
        ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                      rocroller, desc, layout, layout, layout, layout, pref, 1, &result, &count),
                  HIPBLAS_STATUS_SUCCESS);
        if(count == 0)
            GTEST_SKIP() << "rocRoller has no kernel for this problem on " << props.gcnArchName;

        int index = 0;
        std::memcpy(&index, result.algo.data, sizeof(index));
        ASSERT_LT(index, 0);
        EXPECT_EQ(isSupported(result.algo), HIPBLAS_STATUS_SUCCESS);

        // Clearing only the top bit leaves an index that decodes to the same kernel, so only
        // the top-bit check can refuse it.
        hipblasLtMatmulAlgo_t unmarked = result.algo;
        const int             cleared  = index & std::numeric_limits<int>::max();
        std::memcpy(unmarked.data, &cleared, sizeof(cleared));
        EXPECT_EQ(isSupported(unmarked), HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtMatmul(rocroller,
                                  desc,
                                  &alpha,
                                  a,
                                  layout,
                                  b,
                                  layout,
                                  &beta,
                                  c,
                                  layout,
                                  d,
                                  layout,
                                  &unmarked,
                                  nullptr,
                                  0,
                                  nullptr),
                  HIPBLAS_STATUS_INVALID_VALUE);

        // The case from the field: an algo a Tensile handle chose, passed to a rocRoller handle.
        hipblasLtMatmulHeuristicResult_t tensile{};
        ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(tensileOwner.get(),
                                                  desc,
                                                  layout,
                                                  layout,
                                                  layout,
                                                  layout,
                                                  pref,
                                                  1,
                                                  &tensile,
                                                  &count),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_GT(count, 0);
        std::memcpy(&index, tensile.algo.data, sizeof(index));
        ASSERT_GE(index, 0);
        EXPECT_EQ(isSupported(tensile.algo), HIPBLAS_STATUS_INVALID_VALUE);
    }
#endif

    // No kernel applies an aux scale, so both APIs refuse a non-null one.
    TEST_F(AlgoErrors, smoke_AuxScaleIsRefused)
    {
        void* scaleAux = c;
        EXPECT_EQ(
            hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER, &scaleAux, sizeof(void*)),
            HIPBLAS_STATUS_NOT_SUPPORTED);

        void* noScale = nullptr;
        EXPECT_EQ(
            hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER, &noScale, sizeof(void*)),
            HIPBLAS_STATUS_SUCCESS);

        hipblaslt_ext::GemmEpilogue epilogue;
        hipblaslt_ext::GemmInputs   in = inputs();
        in.setScaleAux(scaleAux);

        hipblaslt_ext::Gemm gemm(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N, kType, kType, kType, kType, HIPBLAS_COMPUTE_32F);
        EXPECT_EQ(gemm.setProblem(kSize, kSize, kSize, 1, epilogue, in),
                  HIPBLAS_STATUS_NOT_SUPPORTED);

        hipblaslt_ext::GroupedGemm grouped(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N, kType, kType, kType, kType, HIPBLAS_COMPUTE_32F);
        std::vector<int64_t>                     size(2, kSize), batch(2, 1);
        std::vector<hipblaslt_ext::GemmEpilogue> epilogues(2);
        std::vector<hipblaslt_ext::GemmInputs>   ins{inputs(), in};
        EXPECT_EQ(grouped.setProblem(size, size, size, batch, epilogues, ins),
                  HIPBLAS_STATUS_NOT_SUPPORTED);
    }

    // Matmul computes every layout as column-major, so a row-major A, B, C or D is refused
    // rather than multiplied as if it were column-major (ROCM-29960). ORDER_COL set
    // explicitly on all four is the control: it must still get algorithms and pass every call.
    TEST_F(AlgoErrors, smoke_RowOrderIsRefused)
    {
        const LayoutPtr         colOwner = layoutWithOrder(HIPBLASLT_ORDER_COL);
        const LayoutPtr         rowOwner = layoutWithOrder(HIPBLASLT_ORDER_ROW);
        hipblasLtMatrixLayout_t col = colOwner.get(), row = rowOwner.get();

        hipblasLtMatmulHeuristicResult_t control{};
        int                              controlCount = 0;
        ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                      handle, desc, col, col, col, col, pref, 1, &control, &controlCount),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_GT(controlCount, 0);

        // rowIndex -1 is the all-column control; 0..3 puts ROW on A, B, C or D alone.
        for(int rowIndex = -1; rowIndex < 4; rowIndex++)
        {
            hipblasLtMatrixLayout_t mats[4] = {col, col, col, col};
            if(rowIndex >= 0)
                mats[rowIndex] = row;
            const hipblasStatus_t expected
                = rowIndex < 0 ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_NOT_SUPPORTED;
            const std::string which = rowIndex < 0 ? "control" : std::string(1, "ABCD"[rowIndex]);

            hipblasLtMatmulHeuristicResult_t result{};
            int                              returned = -1;
            EXPECT_EQ(
                hipblasLtMatmulAlgoGetHeuristic(
                    handle, desc, mats[0], mats[1], mats[2], mats[3], pref, 1, &result, &returned),
                expected)
                << which;
            if(rowIndex >= 0)
                EXPECT_EQ(returned, 0) << which;
            {
                EXPECT_EQ(hipblasLtMatmul(handle,
                                          desc,
                                          &alpha,
                                          a,
                                          mats[0],
                                          b,
                                          mats[1],
                                          &beta,
                                          c,
                                          mats[2],
                                          d,
                                          mats[3],
                                          nullptr,
                                          nullptr,
                                          0,
                                          nullptr),
                          expected)
                    << which;
            }

            hipblaslt_ext::Gemm gemm(handle,
                                     HIPBLAS_OP_N,
                                     HIPBLAS_OP_N,
                                     kType,
                                     kType,
                                     kType,
                                     kType,
                                     HIPBLAS_COMPUTE_32F);
            EXPECT_EQ(gemm.setProblem(
                          desc, &alpha, a, mats[0], b, mats[1], &beta, c, mats[2], d, mats[3]),
                      expected)
                << which;

            hipblaslt_ext::GroupedGemm           grouped(handle,
                                               HIPBLAS_OP_N,
                                               HIPBLAS_OP_N,
                                               kType,
                                               kType,
                                               kType,
                                               kType,
                                               HIPBLAS_COMPUTE_32F);
            std::vector<hipblasLtMatmulDesc_t>   descs{desc, desc};
            std::vector<void*>                   alphas{&alpha, &alpha}, betas{&beta, &beta};
            std::vector<void*>                   as{a, a}, bs{b, b}, cs{c, c}, ds{d, d};
            std::vector<hipblasLtMatrixLayout_t> la{col, mats[0]}, lb{col, mats[1]};
            std::vector<hipblasLtMatrixLayout_t> lc{col, mats[2]}, ld{col, mats[3]};
            EXPECT_EQ(grouped.setProblem(descs, alphas, as, la, bs, lb, betas, cs, lc, ds, ld),
                      expected)
                << which << " in the second of two grouped GEMMs";

            size_t workspace = 0;
            EXPECT_EQ(hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                           desc,
                                                           &alpha,
                                                           mats[0],
                                                           mats[1],
                                                           &beta,
                                                           mats[2],
                                                           mats[3],
                                                           control.algo,
                                                           workspace),
                      expected)
                << which;

            hipblasLtMatmulHeuristicResult_t supported      = control;
            int                              supportedCount = 0;
            EXPECT_EQ(hipblaslt_ext::isSolutionSupported(&supported,
                                                         handle,
                                                         desc,
                                                         &alpha,
                                                         mats[0],
                                                         mats[1],
                                                         &beta,
                                                         mats[2],
                                                         mats[3],
                                                         nullptr,
                                                         &supportedCount),
                      expected)
                << which;

            // 1 is tuned and 0 is untuned; -1 is a problem the library cannot answer for.
            const int tuned
                = hipblaslt_ext::matmulIsTuned(handle, desc, mats[0], mats[1], mats[2], mats[3]);
            if(rowIndex < 0)
                EXPECT_GE(tuned, 0) << which;
            else
                EXPECT_EQ(tuned, -1) << which;
        }

        // GemmProblemType carries an order for A and B only.
        for(int rowIndex = -1; rowIndex < 2; rowIndex++)
        {
            const hipblasStatus_t expected
                = rowIndex < 0 ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_NOT_SUPPORTED;
            const std::string which = rowIndex < 0 ? "control" : std::string(1, "AB"[rowIndex]);

            hipblaslt_ext::Gemm            gemm(handle,
                                     HIPBLAS_OP_N,
                                     HIPBLAS_OP_N,
                                     kType,
                                     kType,
                                     kType,
                                     kType,
                                     HIPBLAS_COMPUTE_32F);
            hipblaslt_ext::GemmProblemType problemType = gemm.getProblemTypes();
            problemType.setOrderA(rowIndex == 0 ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL);
            problemType.setOrderB(rowIndex == 1 ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL);

            hipblaslt_ext::GemmEpilogue epilogue;
            hipblaslt_ext::GemmInputs   in     = inputs();
            const int64_t               stride = kSize * kSize;
            EXPECT_EQ(gemm.setProblem(kSize,
                                      kSize,
                                      kSize,
                                      1,
                                      kSize,
                                      kSize,
                                      kSize,
                                      kSize,
                                      stride,
                                      stride,
                                      stride,
                                      stride,
                                      epilogue,
                                      in,
                                      problemType),
                      expected)
                << which;

            hipblaslt_ext::GroupedGemm grouped(handle,
                                               HIPBLAS_OP_N,
                                               HIPBLAS_OP_N,
                                               kType,
                                               kType,
                                               kType,
                                               kType,
                                               HIPBLAS_COMPUTE_32F);
            std::vector<int64_t>       size(2, kSize), batch(2, 1), strides(2, stride);
            std::vector<hipblaslt_ext::GemmEpilogue> epilogues(2);
            std::vector<hipblaslt_ext::GemmInputs>   ins(2, inputs());
            EXPECT_EQ(grouped.setProblem(size,
                                         size,
                                         size,
                                         batch,
                                         size,
                                         size,
                                         size,
                                         size,
                                         strides,
                                         strides,
                                         strides,
                                         strides,
                                         epilogues,
                                         ins,
                                         problemType),
                      expected)
                << which;
        }
    }

    // The kernels write D, and read C, column-major only; swizzled orders exist only for A and B.
    // A swizzled order, or a value no enumerator names, on C or D is refused rather than computed
    // as column-major.
    TEST_F(AlgoErrors, smoke_OutputOrderOtherThanColIsRefused)
    {
        const LayoutPtr         colOwner = layoutWithOrder(HIPBLASLT_ORDER_COL);
        hipblasLtMatrixLayout_t col      = colOwner.get();

        hipblasLtMatmulHeuristicResult_t control{};
        int                              controlCount = 0;
        ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                      handle, desc, col, col, col, col, pref, 1, &control, &controlCount),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_GT(controlCount, 0);

        for(const auto order : {HIPBLASLT_ORDER_COL16_4R8,
                                HIPBLASLT_ORDER_COL16_4R16,
                                HIPBLASLT_ORDER_COL16_4R32,
                                hipblasLtOrder_t(7)})
            for(int index = 2; index < 4; index++)
            {
                const LayoutPtr         owner   = layoutWithOrder(order);
                hipblasLtMatrixLayout_t mats[4] = {col, col, col, col};
                mats[index]                     = owner.get();
                const std::string which
                    = std::string(1, "ABCD"[index]) + " order " + std::to_string(int(order));

                hipblasLtMatmulHeuristicResult_t result{};
                int                              returned = -1;
                EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                          desc,
                                                          mats[0],
                                                          mats[1],
                                                          mats[2],
                                                          mats[3],
                                                          pref,
                                                          1,
                                                          &result,
                                                          &returned),
                          HIPBLAS_STATUS_NOT_SUPPORTED)
                    << which;
                EXPECT_EQ(returned, 0) << which;
                EXPECT_EQ(hipblasLtMatmul(handle,
                                          desc,
                                          &alpha,
                                          a,
                                          mats[0],
                                          b,
                                          mats[1],
                                          &beta,
                                          c,
                                          mats[2],
                                          d,
                                          mats[3],
                                          nullptr,
                                          nullptr,
                                          0,
                                          nullptr),
                          HIPBLAS_STATUS_NOT_SUPPORTED)
                    << which;

                hipblaslt_ext::Gemm gemm(handle,
                                         HIPBLAS_OP_N,
                                         HIPBLAS_OP_N,
                                         kType,
                                         kType,
                                         kType,
                                         kType,
                                         HIPBLAS_COMPUTE_32F);
                EXPECT_EQ(gemm.setProblem(
                              desc, &alpha, a, mats[0], b, mats[1], &beta, c, mats[2], d, mats[3]),
                          HIPBLAS_STATUS_NOT_SUPPORTED)
                    << which;

                size_t workspace = 0;
                EXPECT_EQ(hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                               desc,
                                                               &alpha,
                                                               mats[0],
                                                               mats[1],
                                                               &beta,
                                                               mats[2],
                                                               mats[3],
                                                               control.algo,
                                                               workspace),
                          HIPBLAS_STATUS_NOT_SUPPORTED)
                    << which;
                hipblasLtMatmulHeuristicResult_t supported      = control;
                int                              supportedCount = 0;
                EXPECT_EQ(hipblaslt_ext::isSolutionSupported(&supported,
                                                             handle,
                                                             desc,
                                                             &alpha,
                                                             mats[0],
                                                             mats[1],
                                                             &beta,
                                                             mats[2],
                                                             mats[3],
                                                             nullptr,
                                                             &supportedCount),
                          HIPBLAS_STATUS_NOT_SUPPORTED)
                    << which;
                EXPECT_EQ(
                    hipblaslt_ext::matmulIsTuned(handle, desc, mats[0], mats[1], mats[2], mats[3]),
                    -1)
                    << which;
            }
    }

    // A missing handle, descriptor, layout or pointer is reported the same way whatever the
    // layouts' order is: each invalid call below must return the status it returns with
    // column-major layouts, not HIPBLAS_STATUS_NOT_SUPPORTED.
    TEST_F(AlgoErrors, smoke_RowOrderKeepsArgumentValidationFirst)
    {
        const LayoutPtr         colOwner = layoutWithOrder(HIPBLASLT_ORDER_COL);
        const LayoutPtr         rowOwner = layoutWithOrder(HIPBLASLT_ORDER_ROW);
        hipblasLtMatrixLayout_t col = colOwner.get(), row = rowOwner.get();

        hipblasLtMatmulHeuristicResult_t control{};
        int                              controlCount = 0;
        ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                      handle, desc, col, col, col, col, pref, 1, &control, &controlCount),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_GT(controlCount, 0);

        // Each case gets the layout under test as A (it decides the order) and column-major
        // layouts elsewhere.
        using Call     = std::function<int(hipblasLtMatrixLayout_t)>;
        auto heuristic = [&](hipblasLtHandle_t           h,
                             hipblasLtMatmulDesc_t       d,
                             hipblasLtMatmulPreference_t p,
                             int                         requested,
                             bool                        count) {
            return [=](hipblasLtMatrixLayout_t l) {
                hipblasLtMatmulHeuristicResult_t result{};
                int                              returned = 0;
                return int(hipblasLtMatmulAlgoGetHeuristic(
                    h, d, l, col, col, col, p, requested, &result, count ? &returned : nullptr));
            };
        };
        auto matmul = [&](hipblasLtHandle_t       h,
                          hipblasLtMatmulDesc_t   d,
                          const void*             alphaPtr,
                          const void*             aPtr,
                          hipblasLtMatrixLayout_t layoutB,
                          void*                   dPtr,
                          size_t                  workspaceSize) {
            return [=, bPtr = b, cPtr = c, betaPtr = &beta, algo = control.algo](
                       hipblasLtMatrixLayout_t l) {
                return int(hipblasLtMatmul(h,
                                           d,
                                           alphaPtr,
                                           aPtr,
                                           l,
                                           bPtr,
                                           layoutB,
                                           betaPtr,
                                           cPtr,
                                           col,
                                           dPtr,
                                           col,
                                           &algo,
                                           nullptr,
                                           workspaceSize,
                                           nullptr));
            };
        };
        // The layout under test goes in A, or in B when A is the one left null.
        auto layoutsWithNull = [col](hipblasLtMatrixLayout_t l, int nullSlot) {
            std::array<hipblasLtMatrixLayout_t, 4> mats{l, col, col, col};
            if(nullSlot == 0)
                mats[1] = l;
            if(nullSlot >= 0)
                mats[nullSlot] = nullptr;
            return mats;
        };
        auto heuristicNullLayout = [&](int nullSlot) {
            return [=, this](hipblasLtMatrixLayout_t l) {
                const auto                       mats = layoutsWithNull(l, nullSlot);
                hipblasLtMatmulHeuristicResult_t result{};
                int                              returned = 0;
                return int(hipblasLtMatmulAlgoGetHeuristic(
                    handle, desc, mats[0], mats[1], mats[2], mats[3], pref, 1, &result, &returned));
            };
        };
        auto matmulWith = [&](const void* betaPtr, const void* cPtr, int nullSlot) {
            return [=, this, alphaPtr = &alpha, aPtr = a, bPtr = b, dPtr = d, algo = control.algo](
                       hipblasLtMatrixLayout_t l) {
                const auto mats = layoutsWithNull(l, nullSlot);
                return int(hipblasLtMatmul(handle,
                                           desc,
                                           alphaPtr,
                                           aPtr,
                                           mats[0],
                                           bPtr,
                                           mats[1],
                                           betaPtr,
                                           cPtr,
                                           mats[2],
                                           dPtr,
                                           mats[3],
                                           &algo,
                                           nullptr,
                                           0,
                                           nullptr));
            };
        };
        auto gemmNullLayout = [&](int nullSlot) {
            return [=, this](hipblasLtMatrixLayout_t l) {
                const auto          mats = layoutsWithNull(l, nullSlot);
                hipblaslt_ext::Gemm gemm(handle,
                                         HIPBLAS_OP_N,
                                         HIPBLAS_OP_N,
                                         kType,
                                         kType,
                                         kType,
                                         kType,
                                         HIPBLAS_COMPUTE_32F);
                return int(gemm.setProblem(
                    desc, &alpha, a, mats[0], b, mats[1], &beta, c, mats[2], d, mats[3]));
            };
        };
        auto algoSupported
            = [&](hipblasLtMatmulDesc_t matmulDesc, const void* alphaPtr, const void* betaPtr) {
                  return [=, this, algo = control.algo](hipblasLtMatrixLayout_t l) mutable {
                      size_t workspace = 0;
                      return int(hipblaslt_ext::matmulIsAlgoSupported(
                          handle, matmulDesc, alphaPtr, l, col, betaPtr, col, col, algo, workspace));
                  };
              };
        auto solutionSupported
            = [&](hipblasLtMatmulDesc_t matmulDesc, const void* alphaPtr, const void* betaPtr) {
                  return [=, this, result = control](hipblasLtMatrixLayout_t l) mutable {
                      int count = 0;
                      return int(hipblaslt_ext::isSolutionSupported(
                          &result, handle, matmulDesc, alphaPtr, l, col, betaPtr, col, col, nullptr, &count));
                  };
              };
        // The extension cases vary one scalar or data pointer; the layout under test is A, or
        // for GemmProblemType the order it gives A.
        auto gemmDesc = [&](const void* alphaPtr,
                            const void* betaPtr,
                            const void* aPtr,
                            const void* cPtr,
                            void*       dPtr) {
            return [=, this](hipblasLtMatrixLayout_t l) {
                hipblaslt_ext::Gemm gemm(handle,
                                         HIPBLAS_OP_N,
                                         HIPBLAS_OP_N,
                                         kType,
                                         kType,
                                         kType,
                                         kType,
                                         HIPBLAS_COMPUTE_32F);
                return int(gemm.setProblem(
                    desc, alphaPtr, aPtr, l, b, col, betaPtr, cPtr, col, dPtr, col));
            };
        };
        auto groupedDesc = [&](void* alphaPtr, void* betaPtr, void* aPtr, void* dPtr) {
            return [=, this](hipblasLtMatrixLayout_t l) {
                hipblaslt_ext::GroupedGemm         grouped(handle,
                                                   HIPBLAS_OP_N,
                                                   HIPBLAS_OP_N,
                                                   kType,
                                                   kType,
                                                   kType,
                                                   kType,
                                                   HIPBLAS_COMPUTE_32F);
                std::vector<hipblasLtMatmulDesc_t> descs{desc, desc};
                std::vector<void*>                 alphas{&alpha, alphaPtr}, betas{&beta, betaPtr};
                std::vector<void*>                 as{a, aPtr}, bs{b, b}, cs{c, c}, ds{d, dPtr};
                std::vector<hipblasLtMatrixLayout_t> la{col, l}, lb{col, col}, lc{col, col},
                    ld{col, col};
                return int(
                    grouped.setProblem(descs, alphas, as, la, bs, lb, betas, cs, lc, ds, ld));
            };
        };
        auto inputsWith = [&](const void* alphaPtr, const void* betaPtr, void* aPtr, void* dPtr) {
            hipblaslt_ext::GemmInputs in = inputs();
            in.setAlpha(alphaPtr);
            in.setBeta(betaPtr);
            in.setA(aPtr);
            in.setD(dPtr);
            return in;
        };
        auto problemTypeFor
            = [](hipblaslt_ext::GemmProblemType type, hipblasLtMatrixLayout_t l, bool rowMajor) {
                  type.setOrderA(rowMajor ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL);
                  return type;
              };
        const int64_t stride          = kSize * kSize;
        auto          gemmProblemType = [&](const void* alphaPtr,
                                   const void* betaPtr,
                                   void*       aPtr,
                                   void*       dPtr,
                                   int64_t     k = kSize) {
            return [=, this](hipblasLtMatrixLayout_t l) {
                hipblaslt_ext::Gemm            gemm(handle,
                                         HIPBLAS_OP_N,
                                         HIPBLAS_OP_N,
                                         kType,
                                         kType,
                                         kType,
                                         kType,
                                         HIPBLAS_COMPUTE_32F);
                hipblaslt_ext::GemmProblemType type
                    = problemTypeFor(gemm.getProblemTypes(), l, l == row);
                hipblaslt_ext::GemmEpilogue epilogue;
                hipblaslt_ext::GemmInputs   in = inputsWith(alphaPtr, betaPtr, aPtr, dPtr);
                return int(gemm.setProblem(kSize,
                                           kSize,
                                           k,
                                           1,
                                           kSize,
                                           kSize,
                                           kSize,
                                           kSize,
                                           stride,
                                           stride,
                                           stride,
                                           stride,
                                           epilogue,
                                           in,
                                           type));
            };
        };
        auto groupedProblemType = [&](const void* alphaPtr,
                                      const void* betaPtr,
                                      void*       aPtr,
                                      void*       dPtr,
                                      int64_t     k = kSize) {
            return [=, this](hipblasLtMatrixLayout_t l) {
                hipblaslt_ext::GroupedGemm     grouped(handle,
                                                   HIPBLAS_OP_N,
                                                   HIPBLAS_OP_N,
                                                   kType,
                                                   kType,
                                                   kType,
                                                   kType,
                                                   HIPBLAS_COMPUTE_32F);
                hipblaslt_ext::GemmProblemType type
                    = problemTypeFor(grouped.getProblemTypes()[0], l, l == row);
                std::vector<int64_t> size(2, kSize), batch(2, 1), strides(2, stride);
                std::vector<int64_t> ks{kSize, k};
                std::vector<hipblaslt_ext::GemmEpilogue> epilogues(2);
                std::vector<hipblaslt_ext::GemmInputs>   ins{
                    inputs(), inputsWith(alphaPtr, betaPtr, aPtr, dPtr)};
                return int(grouped.setProblem(size,
                                              size,
                                              ks,
                                              batch,
                                              size,
                                              size,
                                              size,
                                              size,
                                              strides,
                                              strides,
                                              strides,
                                              strides,
                                              epilogues,
                                              ins,
                                              type));
            };
        };
        const std::vector<std::pair<std::string, Call>> cases = {
            {"heuristic, null handle", heuristic(nullptr, desc, pref, 1, true)},
            {"heuristic, null descriptor", heuristic(handle, nullptr, pref, 1, true)},
            {"heuristic, null preference", heuristic(handle, desc, nullptr, 1, true)},
            {"heuristic, zero algos requested", heuristic(handle, desc, pref, 0, true)},
            {"heuristic, null algo count", heuristic(handle, desc, pref, 1, false)},
            {"heuristic, null A layout", heuristicNullLayout(0)},
            {"heuristic, null B layout", heuristicNullLayout(1)},
            {"heuristic, null C layout", heuristicNullLayout(2)},
            {"heuristic, null D layout", heuristicNullLayout(3)},
            {"matmul, null handle", matmul(nullptr, desc, &alpha, a, col, d, 0)},
            {"matmul, null descriptor", matmul(handle, nullptr, &alpha, a, col, d, 0)},
            {"matmul, null B layout", matmul(handle, desc, &alpha, a, nullptr, d, 0)},
            {"matmul, null alpha", matmul(handle, desc, nullptr, a, col, d, 0)},
            {"matmul, null A with nonzero alpha", matmul(handle, desc, &alpha, nullptr, col, d, 0)},
            {"matmul, null D", matmul(handle, desc, &alpha, a, col, nullptr, 0)},
            {"matmul, null workspace with a size", matmul(handle, desc, &alpha, a, col, d, 1024)},
            {"matmul, null A layout", matmulWith(&beta, c, 0)},
            {"matmul, null C layout", matmulWith(&beta, c, 2)},
            {"matmul, null D layout", matmulWith(&beta, c, 3)},
            {"matmul, null beta", matmulWith(nullptr, c, -1)},
            {"matmul, null C", matmulWith(&beta, nullptr, -1)},
            {"Gemm::setProblem, null A layout", gemmNullLayout(0)},
            {"Gemm::setProblem, null B layout", gemmNullLayout(1)},
            {"Gemm::setProblem, null C layout", gemmNullLayout(2)},
            {"Gemm::setProblem, null D layout", gemmNullLayout(3)},
            {"Gemm::setProblem, null descriptor",
             [&](hipblasLtMatrixLayout_t l) {
                 hipblaslt_ext::Gemm gemm(handle,
                                          HIPBLAS_OP_N,
                                          HIPBLAS_OP_N,
                                          kType,
                                          kType,
                                          kType,
                                          kType,
                                          HIPBLAS_COMPUTE_32F);
                 return int(gemm.setProblem(nullptr, &alpha, a, l, b, col, &beta, c, col, d, col));
             }},
            {"GroupedGemm::setProblem, null second descriptor",
             [&](hipblasLtMatrixLayout_t l) {
                 hipblaslt_ext::GroupedGemm           grouped(handle,
                                                    HIPBLAS_OP_N,
                                                    HIPBLAS_OP_N,
                                                    kType,
                                                    kType,
                                                    kType,
                                                    kType,
                                                    HIPBLAS_COMPUTE_32F);
                 std::vector<hipblasLtMatmulDesc_t>   descs{desc, nullptr};
                 std::vector<void*>                   alphas{&alpha, &alpha}, betas{&beta, &beta};
                 std::vector<void*>                   as{a, a}, bs{b, b}, cs{c, c}, ds{d, d};
                 std::vector<hipblasLtMatrixLayout_t> la{l, col}, lb{col, col}, lc{col, col},
                     ld{col, col};
                 return int(
                     grouped.setProblem(descs, alphas, as, la, bs, lb, betas, cs, lc, ds, ld));
             }},
            {"Gemm::setProblem, null alpha", gemmDesc(nullptr, &beta, a, c, d)},
            {"Gemm::setProblem, null beta", gemmDesc(&alpha, nullptr, a, c, d)},
            {"Gemm::setProblem, null A with nonzero alpha", gemmDesc(&alpha, &beta, nullptr, c, d)},
            {"Gemm::setProblem, null C", gemmDesc(&alpha, &beta, a, nullptr, d)},
            {"Gemm::setProblem, null D", gemmDesc(&alpha, &beta, a, c, nullptr)},
            {"GroupedGemm::setProblem, null second alpha", groupedDesc(nullptr, &beta, a, d)},
            {"GroupedGemm::setProblem, null second beta", groupedDesc(&alpha, nullptr, a, d)},
            {"GroupedGemm::setProblem, null second A", groupedDesc(&alpha, &beta, nullptr, d)},
            {"GroupedGemm::setProblem, null second D", groupedDesc(&alpha, &beta, a, nullptr)},
            {"Gemm::setProblem with GemmProblemType, null alpha",
             gemmProblemType(nullptr, &beta, a, d)},
            {"Gemm::setProblem with GemmProblemType, null beta",
             gemmProblemType(&alpha, nullptr, a, d)},
            {"Gemm::setProblem with GemmProblemType, null A",
             gemmProblemType(&alpha, &beta, nullptr, d)},
            {"Gemm::setProblem with GemmProblemType, null D",
             gemmProblemType(&alpha, &beta, a, nullptr)},
            {"GroupedGemm::setProblem with GemmProblemType, null second alpha",
             groupedProblemType(nullptr, &beta, a, d)},
            {"GroupedGemm::setProblem with GemmProblemType, null second beta",
             groupedProblemType(&alpha, nullptr, a, d)},
            {"GroupedGemm::setProblem with GemmProblemType, null second A",
             groupedProblemType(&alpha, &beta, nullptr, d)},
            // Creation, after the argument checks, rejects a null alpha even when K is 0.
            {"Gemm::setProblem with GemmProblemType, K of 0, null alpha",
             gemmProblemType(nullptr, &beta, a, d, 0)},
            {"GroupedGemm::setProblem with GemmProblemType, K of 0, null second alpha",
             groupedProblemType(nullptr, &beta, a, d, 0)},
            {"GroupedGemm::setProblem with GemmProblemType, null second D",
             groupedProblemType(&alpha, &beta, a, nullptr)},
            {"matmulIsAlgoSupported, null handle",
             [&](hipblasLtMatrixLayout_t l) {
                 size_t workspace = 0;
                 return int(hipblaslt_ext::matmulIsAlgoSupported(
                     nullptr, desc, &alpha, l, col, &beta, col, col, control.algo, workspace));
             }},
            {"matmulIsAlgoSupported, null alpha",
             [&](hipblasLtMatrixLayout_t l) {
                 size_t workspace = 0;
                 return int(hipblaslt_ext::matmulIsAlgoSupported(
                     handle, desc, nullptr, l, col, &beta, col, col, control.algo, workspace));
             }},
            {"matmulIsAlgoSupported, null descriptor", algoSupported(nullptr, &alpha, &beta)},
            {"matmulIsAlgoSupported, null beta", algoSupported(desc, &alpha, nullptr)},
            {"isSolutionSupported, null descriptor", solutionSupported(nullptr, &alpha, &beta)},
            {"isSolutionSupported, null alpha", solutionSupported(desc, nullptr, &beta)},
            {"isSolutionSupported, null beta", solutionSupported(desc, &alpha, nullptr)},
            {"isSolutionSupported, null handle",
             [&](hipblasLtMatrixLayout_t l) {
                 hipblasLtMatmulHeuristicResult_t supported = control;
                 int                              count     = 0;
                 return int(hipblaslt_ext::isSolutionSupported(
                     &supported, nullptr, desc, &alpha, l, col, &beta, col, col, nullptr, &count));
             }},
            {"matmulIsTuned, null handle",
             [&](hipblasLtMatrixLayout_t l) {
                 return hipblaslt_ext::matmulIsTuned(nullptr, desc, l, col, col, col);
             }},
        };
        for(const auto& [name, call] : cases)
        {
            const int columnMajor = call(col);
            EXPECT_NE(columnMajor, int(HIPBLAS_STATUS_SUCCESS)) << name;
            EXPECT_NE(columnMajor, int(HIPBLAS_STATUS_NOT_SUPPORTED)) << name;
            EXPECT_EQ(call(row), columnMajor) << name;
        }
    }

    // An empty problem returns before its pointers are checked, so a null alpha is accepted with
    // column-major layouts, and a row-major one is still refused.
    TEST_F(AlgoErrors, smoke_EmptyRowMajorProblemIsRefused)
    {
        for(const bool rowMajor : {false, true})
        {
            hipblaslt_ext::Gemm gemm(handle,
                                     HIPBLAS_OP_N,
                                     HIPBLAS_OP_N,
                                     kType,
                                     kType,
                                     kType,
                                     kType,
                                     HIPBLAS_COMPUTE_32F);
            hipblaslt_ext::GemmProblemType type = gemm.getProblemTypes();
            type.setOrderA(rowMajor ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL);
            hipblaslt_ext::GemmEpilogue epilogue;
            hipblaslt_ext::GemmInputs   in = inputs();
            in.setAlpha(nullptr);
            const int64_t stride = kSize * kSize;
            EXPECT_EQ(gemm.setProblem(0,
                                      kSize,
                                      kSize,
                                      1,
                                      kSize,
                                      kSize,
                                      kSize,
                                      kSize,
                                      stride,
                                      stride,
                                      stride,
                                      stride,
                                      epilogue,
                                      in,
                                      type),
                      rowMajor ? HIPBLAS_STATUS_NOT_SUPPORTED : HIPBLAS_STATUS_SUCCESS)
                << (rowMajor ? "row-major A" : "column-major A");
        }
    }

    // The extension API never sends grouped layouts to the grouped path, so a grouped layout
    // must not let a row-major one through. The C API sends the call to the grouped path, which
    // rejects the mix itself.
    TEST_F(AlgoErrors, smoke_RowOrderIsRefusedBesideAGroupedLayout)
    {
        const LayoutPtr         colOwner = layoutWithOrder(HIPBLASLT_ORDER_COL);
        const LayoutPtr         rowOwner = layoutWithOrder(HIPBLASLT_ORDER_ROW);
        hipblasLtMatrixLayout_t col = colOwner.get(), row = rowOwner.get();

        hipblasLtMatrixLayout_t groupedLayout = nullptr;
        ASSERT_EQ(
            hipblasLtGroupedMatrixLayoutCreate(&groupedLayout, kType, 2, nullptr, nullptr, nullptr),
            HIPBLAS_STATUS_SUCCESS);
        const LayoutPtr groupedOwner(groupedLayout, &hipblasLtMatrixLayoutDestroy);
        const uint64_t  extent = kSize;
        const int64_t   ld     = kSize;
        for(auto attr : {HIPBLASLT_MATRIX_LAYOUT_ROWS, HIPBLASLT_MATRIX_LAYOUT_COLS})
            ASSERT_EQ(
                hipblasLtMatrixLayoutSetAttribute(groupedLayout, attr, &extent, sizeof(extent)),
                HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutSetAttribute(
                      groupedLayout, HIPBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)),
                  HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatmulHeuristicResult_t control{};
        int                              controlCount = 0;
        ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                      handle, desc, col, col, col, col, pref, 1, &control, &controlCount),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_GT(controlCount, 0);

        hipblaslt_ext::Gemm gemm(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N, kType, kType, kType, kType, HIPBLAS_COMPUTE_32F);
        EXPECT_EQ(gemm.setProblem(desc, &alpha, a, row, b, col, &beta, c, groupedLayout, d, col),
                  HIPBLAS_STATUS_NOT_SUPPORTED);

        hipblaslt_ext::GroupedGemm grouped(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N, kType, kType, kType, kType, HIPBLAS_COMPUTE_32F);
        std::vector<hipblasLtMatmulDesc_t>   descs{desc, desc};
        std::vector<void*>                   alphas{&alpha, &alpha}, betas{&beta, &beta};
        std::vector<void*>                   as{a, a}, bs{b, b}, cs{c, c}, ds{d, d};
        std::vector<hipblasLtMatrixLayout_t> la{col, row}, lb{col, col}, lc{col, groupedLayout},
            ldl{col, col};
        EXPECT_EQ(grouped.setProblem(descs, alphas, as, la, bs, lb, betas, cs, lc, ds, ldl),
                  HIPBLAS_STATUS_NOT_SUPPORTED);

        size_t workspace = 0;
        EXPECT_EQ(
            hipblaslt_ext::matmulIsAlgoSupported(
                handle, desc, &alpha, row, col, &beta, groupedLayout, col, control.algo, workspace),
            HIPBLAS_STATUS_NOT_SUPPORTED);
        hipblasLtMatmulHeuristicResult_t supported      = control;
        int                              supportedCount = 0;
        EXPECT_EQ(hipblaslt_ext::isSolutionSupported(&supported,
                                                     handle,
                                                     desc,
                                                     &alpha,
                                                     row,
                                                     col,
                                                     &beta,
                                                     groupedLayout,
                                                     col,
                                                     nullptr,
                                                     &supportedCount),
                  HIPBLAS_STATUS_NOT_SUPPORTED);
        EXPECT_EQ(hipblaslt_ext::matmulIsTuned(handle, desc, row, col, groupedLayout, col), -1);

        hipblasLtMatmulHeuristicResult_t result{};
        int                              returned = -1;
        EXPECT_NE(hipblasLtMatmulAlgoGetHeuristic(
                      handle, desc, row, col, groupedLayout, col, pref, 1, &result, &returned),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_NE(hipblasLtMatmul(handle,
                                  desc,
                                  &alpha,
                                  a,
                                  row,
                                  b,
                                  col,
                                  &beta,
                                  c,
                                  groupedLayout,
                                  d,
                                  col,
                                  nullptr,
                                  nullptr,
                                  0,
                                  nullptr),
                  HIPBLAS_STATUS_SUCCESS);
    }

    struct WorkspaceConfig
    {
        hipDataType typeAB;
        hipDataType typeCD;
        bool        scaled; // device scalar A/B scale pointers
    };

    // Returns how many algos the C API rejects for an undersized workspace.
    int checkUndersizedWorkspace(const WorkspaceConfig& config)
    {
        constexpr int64_t m = 256, n = 256, k = 65536;

        const size_t elementAB = config.typeAB == HIP_R_16BF ? 2 : 1;
        const size_t elementCD = 2;

        hipblasLtHandle_t handle = nullptr;
        EXPECT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatmulDesc_t desc = nullptr;
        EXPECT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        const hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

        float* scale = nullptr;
        if(config.scaled)
        {
            EXPECT_EQ(hipMalloc(&scale, sizeof(float)), hipSuccess);
            const float one = 1.0f;
            EXPECT_EQ(hipMemcpy(scale, &one, sizeof(float), hipMemcpyHostToDevice), hipSuccess);
            hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale, sizeof(scale));
            hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale, sizeof(scale));
        }

        hipblasLtMatrixLayout_t layoutA = nullptr, layoutB = nullptr, layoutCD = nullptr;
        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&layoutA, config.typeAB, k, m, k),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&layoutB, config.typeAB, k, n, k),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&layoutCD, config.typeCD, m, n, m),
                  HIPBLAS_STATUS_SUCCESS);

        void *a = nullptr, *b = nullptr, *d = nullptr, *workspaceBuffer = nullptr;
        EXPECT_EQ(hipMalloc(&a, k * m * elementAB), hipSuccess);
        EXPECT_EQ(hipMalloc(&b, k * n * elementAB), hipSuccess);
        EXPECT_EQ(hipMalloc(&d, m * n * elementCD), hipSuccess);
        constexpr size_t maxWorkspace = size_t(1) << 30;
        EXPECT_EQ(hipMalloc(&workspaceBuffer, maxWorkspace), hipSuccess);

        hipStream_t stream = nullptr;
        EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);

        const float         alpha = 1.0f, beta = 0.0f;
        hipblaslt_ext::Gemm gemm(
            handle, desc, &alpha, a, layoutA, b, layoutB, &beta, d, layoutCD, d, layoutCD);
        hipblaslt_ext::GemmPreference pref;
        pref.setMaxWorkspaceBytes(maxWorkspace);
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        EXPECT_EQ(gemm.algoGetHeuristic(2000, pref, results), HIPBLAS_STATUS_SUCCESS);

        int rejected = 0;
        for(size_t i = 0; i < results.size(); i++)
        {
            auto&  algo     = results[i].algo;
            size_t required = 0;
            if(gemm.isAlgoSupported(algo, required) != HIPBLAS_STATUS_SUCCESS || required == 0)
                continue;

            // The C API is the reference for which solutions need the full workspace.
            const hipblasStatus_t reference = hipblasLtMatmul(handle,
                                                              desc,
                                                              &alpha,
                                                              a,
                                                              layoutA,
                                                              b,
                                                              layoutB,
                                                              &beta,
                                                              d,
                                                              layoutCD,
                                                              d,
                                                              layoutCD,
                                                              &algo,
                                                              workspaceBuffer,
                                                              required - 1,
                                                              stream);
            if(reference == HIPBLAS_STATUS_INVALID_VALUE)
            {
                rejected++;
                gemm.setMaxWorkspaceBytes(required - 1);
                EXPECT_EQ(gemm.initialize(algo, workspaceBuffer, false, stream),
                          HIPBLAS_STATUS_INVALID_VALUE)
                    << "algo " << i << ": " << hipblaslt_ext::getSolutionNameFromAlgo(handle, algo);
            }

            gemm.setMaxWorkspaceBytes(required);
            EXPECT_EQ(gemm.initialize(algo, workspaceBuffer, false, stream), HIPBLAS_STATUS_SUCCESS)
                << "algo " << i << ": " << hipblaslt_ext::getSolutionNameFromAlgo(handle, algo);
        }

        static_cast<void>(hipStreamSynchronize(stream));
        static_cast<void>(hipStreamDestroy(stream));
        for(void* buffer : {a, b, d, workspaceBuffer, static_cast<void*>(scale)})
            static_cast<void>(hipFree(buffer));
        hipblasLtMatrixLayoutDestroy(layoutA);
        hipblasLtMatrixLayoutDestroy(layoutB);
        hipblasLtMatrixLayoutDestroy(layoutCD);
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtDestroy(handle);
        return rejected;
    }

    // Wherever the C API rejects a workspace smaller than a solution requires, the ext API
    // must too. Stream-K solutions run with a short workspace in both APIs.
    TEST(AlgoWorkspace, smoke_GemmInitializeRejectsUndersizedWorkspace)
    {
        int rejected = 0;
        for(const WorkspaceConfig& config : {WorkspaceConfig{HIP_R_8F_E4M3, HIP_R_16BF, true},
                                             WorkspaceConfig{HIP_R_16BF, HIP_R_16BF, false}})
            rejected += checkUndersizedWorkspace(config);

        if(rejected == 0)
            GTEST_SKIP() << "no solution in the loaded library requires a fixed workspace";
    }
} // namespace
