// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// gfx950 keeps HostPreSwizzle kernels (name token MXSFHPS) for scale 1001 and
// drops them for packed scale 3. Both scale modes search the same Tensile
// problem, which ranks the HostPreSwizzle kernels first, so these tests fail if
// the layout checks in tensile_host.cpp are removed from the heuristic, the
// support check, or the launch, in the C API or the hipblaslt_ext::Gemm API.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <cstdlib>
#include <string>
#include <vector>

namespace
{
    constexpr int64_t kM          = 128;
    constexpr int64_t kN          = 128;
    constexpr int64_t kK          = 256;
    constexpr size_t  kWorkspace  = 64ull << 20;
    constexpr size_t  kScaleBytes = 1u << 20;
    constexpr int     kAlgoCount  = 32;
    const float       kAlpha      = 1.f;
    const float       kBeta       = 0.f;

    bool archIsGfx950()
    {
        hipDeviceProp_t prop{};
        if(hipGetDeviceProperties(&prop, 0) != hipSuccess)
            return false;
        return std::string(prop.gcnArchName).find("gfx950") != std::string::npos;
    }

    bool isPreswizzled(const std::string& kernelName)
    {
        return kernelName.find("MXSFHPS") != std::string::npos;
    }

    // The handle reads HIPBLASLT_USE_ROCROLLER in its constructor.
    struct RocRollerEnv
    {
        bool        had = false;
        std::string old;

        explicit RocRollerEnv(const char* value)
        {
            if(const char* cur = std::getenv("HIPBLASLT_USE_ROCROLLER"))
            {
                had = true;
                old = cur;
            }
            setenv("HIPBLASLT_USE_ROCROLLER", value, 1);
        }

        ~RocRollerEnv()
        {
            if(had)
                setenv("HIPBLASLT_USE_ROCROLLER", old.c_str(), 1);
            else
                unsetenv("HIPBLASLT_USE_ROCROLLER");
        }
    };

    // FP8 E4M3, A is K x M (transA = T), B is K x N (transB = N), FP32 D.
    // The scale buffers cover the 32x8 pre-swizzle of this shape.
    struct Problem
    {
        hipblasLtHandle_t           handle = nullptr;
        hipblasLtMatmulDesc_t       desc   = nullptr;
        hipblasLtMatrixLayout_t     a      = nullptr;
        hipblasLtMatrixLayout_t     b      = nullptr;
        hipblasLtMatrixLayout_t     d      = nullptr;
        hipblasLtMatmulPreference_t pref   = nullptr;
        void*                       dA     = nullptr;
        void*                       dB     = nullptr;
        void*                       dD     = nullptr;
        void*                       scaleA = nullptr;
        void*                       scaleB = nullptr;
        void*                       ws     = nullptr;

        ~Problem()
        {
            if(pref)
                hipblasLtMatmulPreferenceDestroy(pref);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
            for(auto layout : {a, b, d})
                if(layout)
                    hipblasLtMatrixLayoutDestroy(layout);
            for(void* p : {dA, dB, dD, scaleA, scaleB, ws})
                if(p)
                    static_cast<void>(hipFree(p));
        }
    };

    void allocZeroed(void** p, size_t bytes)
    {
        ASSERT_EQ(hipMalloc(p, bytes), hipSuccess);
        ASSERT_EQ(hipMemset(*p, 0, bytes), hipSuccess);
    }

    void makeProblem(Problem& q, hipblasLtHandle_t handle, hipblasLtMatmulMatrixScale_t mode)
    {
        q.handle = handle;
        ASSERT_EQ(hipblasLtMatmulDescCreate(&q.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasOperation_t opA = HIPBLAS_OP_T;
        hipblasOperation_t opB = HIPBLAS_OP_N;
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      q.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      q.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)),
                  HIPBLAS_STATUS_SUCCESS);
        for(auto attr : {HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE})
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(q.desc, attr, &mode, sizeof(mode)),
                      HIPBLAS_STATUS_SUCCESS);

        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&q.a, HIP_R_8F_E4M3, kK, kM, kK),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&q.b, HIP_R_8F_E4M3, kK, kN, kK),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&q.d, HIP_R_32F, kM, kN, kM), HIPBLAS_STATUS_SUCCESS);

        allocZeroed(&q.dA, kK * kM);
        allocZeroed(&q.dB, kK * kN);
        allocZeroed(&q.dD, kM * kN * sizeof(float));
        allocZeroed(&q.scaleA, kScaleBytes);
        allocZeroed(&q.scaleB, kScaleBytes);
        allocZeroed(&q.ws, kWorkspace);
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      q.desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &q.scaleA, sizeof(void*)),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      q.desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &q.scaleB, sizeof(void*)),
                  HIPBLAS_STATUS_SUCCESS);

        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&q.pref), HIPBLAS_STATUS_SUCCESS);
        uint64_t workspace = kWorkspace;
        ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(q.pref,
                                                        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                        &workspace,
                                                        sizeof(workspace)),
                  HIPBLAS_STATUS_SUCCESS);
    }

    std::vector<std::string> names(hipblasLtHandle_t                              handle,
                                   std::vector<hipblasLtMatmulHeuristicResult_t>& results)
    {
        std::vector<std::string> out;
        for(auto& result : results)
            out.push_back(hipblaslt_ext::getKernelNameFromAlgo(handle, result.algo));
        return out;
    }

    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic(Problem& q)
    {
        hipblasLtMatmulHeuristicResult_t raw[kAlgoCount]{};
        int                              count = 0;
        EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                      q.handle, q.desc, q.a, q.b, q.d, q.d, q.pref, kAlgoCount, raw, &count),
                  HIPBLAS_STATUS_SUCCESS);
        std::vector<hipblasLtMatmulHeuristicResult_t> out;
        for(int i = 0; i < count; ++i)
            if(raw[i].state == HIPBLAS_STATUS_SUCCESS)
                out.push_back(raw[i]);
        return out;
    }

    hipblasStatus_t matmul(Problem& q, const hipblasLtMatmulHeuristicResult_t& result)
    {
        return hipblasLtMatmul(q.handle,
                               q.desc,
                               &kAlpha,
                               q.dA,
                               q.a,
                               q.dB,
                               q.b,
                               &kBeta,
                               q.dD,
                               q.d,
                               q.dD,
                               q.d,
                               &result.algo,
                               q.ws,
                               kWorkspace,
                               nullptr);
    }

    hipblasStatus_t supported(Problem& q, const hipblasLtMatmulHeuristicResult_t& result)
    {
        hipblasLtMatmulHeuristicResult_t copy      = result;
        size_t                           workspace = 0;
        int                              count     = 0;
        return hipblaslt_ext::isSolutionSupported(
            &copy, q.handle, q.desc, &kAlpha, q.a, q.b, &kBeta, q.d, q.d, &workspace, &count);
    }

    hipblaslt_ext::Gemm extGemm(Problem& q)
    {
        return hipblaslt_ext::Gemm(
            q.handle, q.desc, &kAlpha, q.dA, q.a, q.dB, q.b, &kBeta, q.dD, q.d, q.dD, q.d);
    }

    std::vector<hipblasLtMatmulHeuristicResult_t> extHeuristic(hipblaslt_ext::Gemm& gemm)
    {
        hipblaslt_ext::GemmPreference pref;
        pref.setMaxWorkspaceBytes(kWorkspace);
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        EXPECT_EQ(gemm.algoGetHeuristic(kAlgoCount, pref, results), HIPBLAS_STATUS_SUCCESS);
        return results;
    }

    class Gfx950MxScaleLayout : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if(!archIsGfx950())
                GTEST_SKIP() << "requires gfx950";
        }
    };

    // The setting under which the packed problem used to run a HostPreSwizzle
    // kernel and fail its numerical check.
    TEST_F(Gfx950MxScaleLayout, smoke_MatmulApiRocRollerOff)
    {
        RocRollerEnv      env("0");
        hipblasLtHandle_t handle = nullptr;
        ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
        {
            Problem packed;
            Problem preswizzled;
            makeProblem(packed, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
            makeProblem(preswizzled, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT);
            if(HasFatalFailure())
                return;

            auto preResults = heuristic(preswizzled);
            ASSERT_FALSE(preResults.empty()) << "scale 1001 returned no kernel";
            for(const auto& name : names(handle, preResults))
                EXPECT_TRUE(isPreswizzled(name)) << name;

            auto packedResults = heuristic(packed);
            for(const auto& name : names(handle, packedResults))
                EXPECT_FALSE(isPreswizzled(name)) << name;

            // isSolutionSupported skips an algo whose first data byte is 0.
            const hipblasLtMatmulHeuristicResult_t* algo = nullptr;
            for(const auto& result : preResults)
                if(result.algo.data[0] != 0)
                {
                    algo = &result;
                    break;
                }
            ASSERT_NE(algo, nullptr);

            EXPECT_EQ(supported(preswizzled, *algo), HIPBLAS_STATUS_SUCCESS);
            EXPECT_NE(supported(packed, *algo), HIPBLAS_STATUS_SUCCESS);

            // runContractionProblem returns not_implemented, which hipblasLtMatmul
            // reports as an internal error.
            EXPECT_EQ(matmul(preswizzled, *algo), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
            EXPECT_EQ(matmul(packed, *algo), HIPBLAS_STATUS_INTERNAL_ERROR);
        }
        hipblasLtDestroy(handle);
    }

    // Packed scale 3 goes to rocRoller. FP8 E4M3 with scale 1001 on both sides
    // stays on Tensile, and still gets its HostPreSwizzle kernels.
    TEST_F(Gfx950MxScaleLayout, smoke_MatmulApiRocRollerOn)
    {
        RocRollerEnv      env("1");
        hipblasLtHandle_t handle = nullptr;
        ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
        {
            Problem packed;
            Problem preswizzled;
            makeProblem(packed, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
            makeProblem(preswizzled, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT);
            if(HasFatalFailure())
                return;

            auto preResults = heuristic(preswizzled);
            ASSERT_FALSE(preResults.empty()) << "scale 1001 returned no kernel";
            for(const auto& name : names(handle, preResults))
                EXPECT_TRUE(isPreswizzled(name)) << name;

            auto packedResults = heuristic(packed);
            for(const auto& name : names(handle, packedResults))
                EXPECT_FALSE(isPreswizzled(name)) << name;
        }
        hipblasLtDestroy(handle);
    }

    // A descriptor keeps its Tensile problem between queries. After a scale-1001
    // query, scalar scales must not keep searching MX kernels.
    TEST_F(Gfx950MxScaleLayout, smoke_ReusedDescriptorLeavesMxScaling)
    {
        RocRollerEnv      env("0");
        hipblasLtHandle_t handle = nullptr;
        ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
        {
            Problem q;
            makeProblem(q, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT);
            if(HasFatalFailure())
                return;

            auto preResults = heuristic(q);
            ASSERT_FALSE(preResults.empty()) << "scale 1001 returned no kernel";

            const hipblasLtMatmulMatrixScale_t scalar = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
            for(auto attr :
                {HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE})
                ASSERT_EQ(hipblasLtMatmulDescSetAttribute(q.desc, attr, &scalar, sizeof(scalar)),
                          HIPBLAS_STATUS_SUCCESS);

            auto scalarResults = heuristic(q);
            ASSERT_FALSE(scalarResults.empty()) << "scalar scales returned no kernel";
            for(const auto& name : names(handle, scalarResults))
                EXPECT_EQ(name.find("MXAE8"), std::string::npos) << name;
        }
        hipblasLtDestroy(handle);
    }

    // hipblaslt_ext::Gemm has no rocRoller dispatch, so it reaches the Tensile
    // filter whatever HIPBLASLT_USE_ROCROLLER says.
    TEST_F(Gfx950MxScaleLayout, smoke_ExtGemmApi)
    {
        hipblasLtHandle_t handle = nullptr;
        ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
        {
            Problem packed;
            Problem preswizzled;
            makeProblem(packed, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
            makeProblem(preswizzled, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT);
            if(HasFatalFailure())
                return;

            auto preGemm    = extGemm(preswizzled);
            auto packedGemm = extGemm(packed);

            auto preResults = extHeuristic(preGemm);
            ASSERT_FALSE(preResults.empty()) << "scale 1001 returned no kernel";
            for(const auto& name : names(handle, preResults))
                EXPECT_TRUE(isPreswizzled(name)) << name;

            auto packedResults = extHeuristic(packedGemm);
            for(const auto& name : names(handle, packedResults))
                EXPECT_FALSE(isPreswizzled(name)) << name;

            hipblasLtMatmulAlgo_t algo      = preResults[0].algo;
            size_t                workspace = 0;
            EXPECT_EQ(preGemm.isAlgoSupported(algo, workspace), HIPBLAS_STATUS_SUCCESS);
            EXPECT_NE(packedGemm.isAlgoSupported(algo, workspace), HIPBLAS_STATUS_SUCCESS);

            preGemm.setMaxWorkspaceBytes(kWorkspace);
            packedGemm.setMaxWorkspaceBytes(kWorkspace);
            EXPECT_EQ(preGemm.initialize(algo, preswizzled.ws), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(preGemm.run(nullptr), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
            EXPECT_NE(packedGemm.initialize(algo, packed.ws), HIPBLAS_STATUS_SUCCESS);
        }
        hipblasLtDestroy(handle);
    }
}
