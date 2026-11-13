// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Argument validation of HIPBLASLT_ORDER_COL256_R128_EXT and
// HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT through the C, ext and grouped ext
// APIs. Nothing is launched: an invalid combination is refused before solution
// selection on every architecture, and a valid one is accepted whether or not a kernel
// exists. The C heuristic searches for a valid one, so those cases run on gfx950 only.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <ostream>
#include <string>
#include <vector>

namespace
{
    constexpr auto kTiled = HIPBLASLT_ORDER_COL256_R128_EXT;
    constexpr auto kCol   = HIPBLASLT_ORDER_COL;
    constexpr auto kScale = HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT;
    constexpr auto kBlk32 = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;

    struct Fp6Case
    {
        const char*                  name;
        bool                         valid;
        hipblasStatus_t              rejectedAs = HIPBLAS_STATUS_INVALID_VALUE;
        hipblasLtOrder_t             orderA = kTiled;
        hipblasLtOrder_t             orderB = kTiled;
        hipblasLtMatmulMatrixScale_t scaleA = kScale;
        hipblasLtMatmulMatrixScale_t scaleB = kScale;
        hipDataType                  typeA  = HIP_R_6F_E2M3;
        hipDataType                  typeB  = HIP_R_6F_E2M3;
        hipblasOperation_t           opA    = HIPBLAS_OP_T;
        int64_t                      m      = 256;
        int64_t                      n      = 256;
        int64_t                      k      = 128;
        int64_t                      ldaPad = 0;
        int64_t                      ldbPad = 0;

        int64_t lda() const
        {
            return (opA == HIPBLAS_OP_N ? m : k) + ldaPad;
        }
        int64_t ldb() const
        {
            return k + ldbPad;
        }
    };

    std::ostream& operator<<(std::ostream& os, const Fp6Case& c)
    {
        return os << c.name;
    }

    Fp6Case make(const char* name, bool valid)
    {
        Fp6Case c{};
        c.name  = name;
        c.valid = valid;
        return c;
    }

    std::vector<Fp6Case> cases()
    {
        std::vector<Fp6Case> v;
        v.push_back(make("both_tiled", true));
        v.push_back(make("e3m2_both_tiled", true));
        v.back().typeA = v.back().typeB = HIP_R_6F_E3M2;
        v.push_back(make("plain_b", true));
        v.back().orderB = kCol;
        v.back().n      = 128;
        v.push_back(make("no_tile_ordinary_scales", true));
        v.back().orderA = v.back().orderB = kCol;
        v.back().scaleA = v.back().scaleB = kBlk32;

        v.push_back(make("tiled_b_plain_a", false));
        v.back().orderA = kCol;
        v.push_back(make("order_without_scale", false));
        v.back().scaleA = v.back().scaleB = kBlk32;
        v.push_back(make("scale_without_order", false));
        v.back().orderA = v.back().orderB = kCol;
        v.push_back(make("scale_a_only", false));
        v.back().scaleB = kBlk32;
        // Refused by the row-major order handling before this validation.
        v.push_back(make("b_row_order", false));
        v.back().orderB     = HIPBLASLT_ORDER_ROW;
        v.back().rejectedAs = HIPBLAS_STATUS_NOT_SUPPORTED;
        v.push_back(make("fp8", false));
        v.back().typeA = v.back().typeB = HIP_R_8F_E4M3;
        v.push_back(make("op_a_n", false));
        v.back().opA = HIPBLAS_OP_N;
        v.push_back(make("other_fp6_swizzle_order", false));
        v.back().orderA = HIPBLASLT_ORDER_COL16_4R16;
        v.back().orderB = kCol;
        v.back().scaleA = v.back().scaleB = kBlk32;
        v.push_back(make("m_not_multiple_of_256", false));
        v.back().orderB = kCol;
        v.back().m      = 384;
        v.back().n      = 128;
        v.push_back(make("k_not_multiple_of_128", false));
        v.back().k = 192;
        v.push_back(make("tiled_b_n_not_multiple_of_256", false));
        v.back().n = 384;
        v.push_back(make("plain_b_n_not_multiple_of_128", false));
        v.back().orderB = kCol;
        v.back().n      = 192;
        v.push_back(make("plain_b_ldb_above_k", false));
        v.back().orderB = kCol;
        v.back().n      = 128;
        v.back().ldbPad = 128;
        v.push_back(make("tiled_lda_above_k", false));
        v.back().ldaPad = 128;
        return v;
    }

    class Fp6Tile128x4Args_smoke : public ::testing::TestWithParam<Fp6Case>
    {
    protected:
        hipblasLtHandle_t handle  = nullptr;
        void*             storage = nullptr;
        const float       alpha   = 1.0f;
        const float       beta    = 0.0f;

        void SetUp() override
        {
            int             count = 0;
            hipDeviceProp_t props{};
            if(hipGetDeviceCount(&count) != hipSuccess || count == 0
               || hipGetDeviceProperties(&props, 0) != hipSuccess)
                GTEST_SKIP() << "No GPU available";
            if(GetParam().valid && std::string(props.gcnArchName).rfind("gfx950", 0) != 0)
                GTEST_SKIP() << "A valid problem reaches solution selection, which these "
                                "tests cover on gfx950 only";
            ASSERT_EQ(hipMalloc(&storage, 256), hipSuccess);
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
        }

        void TearDown() override
        {
            if(handle)
                EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
            if(storage)
                EXPECT_EQ(hipFree(storage), hipSuccess);
        }

        hipblasStatus_t expected() const
        {
            return GetParam().valid ? HIPBLAS_STATUS_SUCCESS : GetParam().rejectedAs;
        }

        hipblaslt_ext::GemmProblemType problemType() const
        {
            const Fp6Case&                 c = GetParam();
            hipblaslt_ext::GemmProblemType p(
                c.opA, HIPBLAS_OP_N, c.typeA, c.typeB, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F);
            p.setOrderA(c.orderA);
            p.setOrderB(c.orderB);
            return p;
        }

        hipblaslt_ext::GemmEpilogue epilogue() const
        {
            hipblaslt_ext::GemmEpilogue e;
            e.setScalingAType(GetParam().scaleA);
            e.setScalingBType(GetParam().scaleB);
            return e;
        }

        hipblaslt_ext::GemmInputs inputs() const
        {
            hipblaslt_ext::GemmInputs in;
            in.setA(storage);
            in.setB(storage);
            in.setC(storage);
            in.setD(storage);
            in.setAlpha(&alpha);
            in.setBeta(&beta);
            in.setScaleA(storage);
            in.setScaleB(storage);
            return in;
        }
    };

    TEST_P(Fp6Tile128x4Args_smoke, CApiHeuristic)
    {
        const Fp6Case& c = GetParam();
        const int64_t  rowsA = c.opA == HIPBLAS_OP_N ? c.m : c.k;
        const int64_t  colsA = c.opA == HIPBLAS_OP_N ? c.k : c.m;

        hipblasLtMatmulDesc_t       desc = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;
        hipblasLtMatrixLayout_t     a = nullptr, b = nullptr, cl = nullptr, d = nullptr;
        ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&a, c.typeA, rowsA, colsA, c.lda()),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&b, c.typeB, c.k, c.n, c.ldb()),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&cl, HIP_R_16BF, c.m, c.n, c.m),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&d, HIP_R_16BF, c.m, c.n, c.m),
                  HIPBLAS_STATUS_SUCCESS);
        const int32_t orderA = c.orderA;
        const int32_t orderB = c.orderB;
        ASSERT_EQ(
            hipblasLtMatrixLayoutSetAttribute(a, HIPBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA)),
            HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(
            hipblasLtMatrixLayoutSetAttribute(b, HIPBLASLT_MATRIX_LAYOUT_ORDER, &orderB, sizeof(orderB)),
            HIPBLAS_STATUS_SUCCESS);
        const hipblasOperation_t opB    = HIPBLAS_OP_N;
        const uint32_t           scaleA = c.scaleA;
        const uint32_t           scaleB = c.scaleB;
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_TRANSA, &c.opA, sizeof(c.opA)),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleA, sizeof(scaleA)),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleB, sizeof(scaleB)),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &storage, sizeof(storage)),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &storage, sizeof(storage)),
                  HIPBLAS_STATUS_SUCCESS);

        // The C heuristic answers invalid arguments with no solution; hipblasLtMatmul
        // rejects them during argument validation, before any algo or launch.
        hipblasLtMatmulHeuristicResult_t result;
        int                              count = -1;
        const hipblasStatus_t            heuristic
            = hipblasLtMatmulAlgoGetHeuristic(handle, desc, a, b, cl, d, pref, 1, &result, &count);
        if(c.valid)
            EXPECT_EQ(heuristic, HIPBLAS_STATUS_SUCCESS);
        else
        {
            EXPECT_EQ(count, 0);
            EXPECT_EQ(hipblasLtMatmul(handle,
                                      desc,
                                      &alpha,
                                      storage,
                                      a,
                                      storage,
                                      b,
                                      &beta,
                                      storage,
                                      cl,
                                      storage,
                                      d,
                                      nullptr,
                                      nullptr,
                                      0,
                                      nullptr),
                      c.rejectedAs);
        }

        for(auto* layout : {a, b, cl, d})
            hipblasLtMatrixLayoutDestroy(layout);
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatmulDescDestroy(desc);
    }

    TEST_P(Fp6Tile128x4Args_smoke, ExtSetProblem)
    {
        const Fp6Case&                 c  = GetParam();
        hipblaslt_ext::GemmProblemType pt = problemType();
        hipblaslt_ext::GemmEpilogue    ep = epilogue();
        hipblaslt_ext::GemmInputs      in = inputs();
        hipblaslt_ext::Gemm            gemm(
            handle, c.opA, HIPBLAS_OP_N, c.typeA, c.typeB, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F);
        EXPECT_EQ(gemm.setProblem(c.m,
                                  c.n,
                                  c.k,
                                  1,
                                  c.lda(),
                                  c.ldb(),
                                  c.m,
                                  c.m,
                                  c.lda() * (c.opA == HIPBLAS_OP_N ? c.k : c.m),
                                  c.ldb() * c.n,
                                  c.m * c.n,
                                  c.m * c.n,
                                  ep,
                                  in,
                                  pt),
                  expected());
    }

    TEST_P(Fp6Tile128x4Args_smoke, GroupedExtSetProblem)
    {
        const Fp6Case&                 c  = GetParam();
        hipblaslt_ext::GemmProblemType pt = problemType();
        hipblaslt_ext::GroupedGemm     grouped(
            handle, c.opA, HIPBLAS_OP_N, c.typeA, c.typeB, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F);
        std::vector<int64_t> m{c.m, c.m}, n{c.n, c.n}, k{c.k, c.k}, batch{1, 1};
        std::vector<int64_t> lda{c.lda(), c.lda()}, ldb{c.ldb(), c.ldb()}, ldc{c.m, c.m},
            ldd{c.m, c.m};
        const int64_t        strideA = c.lda() * (c.opA == HIPBLAS_OP_N ? c.k : c.m);
        std::vector<int64_t> sa{strideA, strideA}, sb{c.ldb() * c.n, c.ldb() * c.n},
            sc{c.m * c.n, c.m * c.n}, sd{c.m * c.n, c.m * c.n};
        std::vector<hipblaslt_ext::GemmEpilogue> ep{epilogue(), epilogue()};
        std::vector<hipblaslt_ext::GemmInputs>   in{inputs(), inputs()};
        EXPECT_EQ(grouped.setProblem(m, n, k, batch, lda, ldb, ldc, ldd, sa, sb, sc, sd, ep, in, pt),
                  expected());
    }

    INSTANTIATE_TEST_SUITE_P(Fp6Tile128x4Args_smoke,
                             Fp6Tile128x4Args_smoke,
                             ::testing::ValuesIn(cases()),
                             [](const ::testing::TestParamInfo<Fp6Case>& info) {
                                 return std::string(info.param.name);
                             });
}
