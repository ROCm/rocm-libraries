// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// gfx950 keeps HostPreSwizzle kernels (name token MXSFHPS) for scale 1001 and
// refuses them for packed scales. Both modes search the same Tensile problem,
// so these tests fail if the layout check is removed from the heuristic, the
// support check, the launch, the tuning candidates or the tuning-file
// override, in the C API or the hipblaslt_ext Gemm and GroupedGemm APIs.
//
// Every gfx950 kernel for the FP8 and FP4 MX types these tests use is
// HostPreSwizzle: their logic files leave MXScaleFormat at Auto, which gfx950
// resolves to HostPreSwizzle, and the rocRoller custom kernels, which are
// NoSwizzle, cover only FP6 and mixed-type pairs. Off the rocRoller path,
// packed scales therefore have no kernel for these types today. The tests
// check the rule, so a kernel that reads packed scales is launched once one
// ships.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <Tensile/SolutionLibrary.hpp>

#include "../../../library/src/amd_detail/rocblaslt/src/include/get_all_search_type_for_test.hpp"
#include "scoped_env.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <vector>

extern "C" void   hipblaslt_tuning_reset_for_test();
extern "C" size_t hipblaslt_tuning_candidates_for_test(void*  handle,
                                                       void*  matmulDesc,
                                                       void*  matA,
                                                       void*  matB,
                                                       void*  matC,
                                                       void*  matD,
                                                       size_t workspaceBytes,
                                                       int*   indexes,
                                                       size_t count);

namespace
{
    using test_env::ScopedEnv;

    constexpr int64_t kM          = 128;
    constexpr int64_t kN          = 128;
    constexpr int64_t kK          = 256;
    constexpr size_t  kWorkspace  = 64ull << 20;
    constexpr size_t  kScaleBytes = 1u << 20;
    constexpr int     kAlgoCount  = 32;
    float             kAlpha      = 1.f;
    float             kBeta       = 0.f;

    bool archIsGfx950()
    {
        int             device = 0;
        hipDeviceProp_t prop{};
        if(hipGetDevice(&device) != hipSuccess
           || hipGetDeviceProperties(&prop, device) != hipSuccess)
            return false;
        return std::string(prop.gcnArchName).find("gfx950") != std::string::npos;
    }

    bool isPreswizzled(const std::string& kernelName)
    {
        return kernelName.find("MXSFHPS") != std::string::npos;
    }

    void expectNonePreswizzled(const std::vector<std::string>& kernelNames)
    {
        for(const auto& name : kernelNames)
            EXPECT_FALSE(isPreswizzled(name)) << name;
    }

    // Destroyed on every exit from a test, including a skip or a failed assertion.
    struct OwnedHandle
    {
        hipblasLtHandle_t handle = nullptr;
        ~OwnedHandle()
        {
            if(handle)
                hipblasLtDestroy(handle);
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

    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic(Problem& q, int count)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> raw(count);
        int                                           returned = 0;
        EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                      q.handle, q.desc, q.a, q.b, q.d, q.d, q.pref, count, raw.data(), &returned),
                  HIPBLAS_STATUS_SUCCESS);
        std::vector<hipblasLtMatmulHeuristicResult_t> out;
        for(int i = 0; i < returned; ++i)
            if(raw[i].state == HIPBLAS_STATUS_SUCCESS)
                out.push_back(raw[i]);
        return out;
    }

    hipblasStatus_t matmul(Problem& q, const hipblasLtMatmulAlgo_t* algo)
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
                               algo,
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

    // Two groups built from the same problem descriptors.
    hipblaslt_ext::GroupedGemm extGroupedGemm(Problem& q)
    {
        hipblaslt_ext::GroupedGemm           gemm(q.handle,
                                                  HIPBLAS_OP_T,
                                                  HIPBLAS_OP_N,
                                                  HIP_R_8F_E4M3,
                                                  HIP_R_8F_E4M3,
                                                  HIP_R_32F,
                                                  HIP_R_32F,
                                                  HIPBLAS_COMPUTE_32F);
        std::vector<hipblasLtMatmulDesc_t>   desc(2, q.desc);
        std::vector<void*>                   alpha(2, &kAlpha), beta(2, &kBeta);
        std::vector<void*>                   a(2, q.dA), b(2, q.dB), d(2, q.dD);
        std::vector<hipblasLtMatrixLayout_t> layA(2, q.a), layB(2, q.b), layD(2, q.d);
        EXPECT_EQ(gemm.setProblem(desc, alpha, a, layA, b, layB, beta, d, layD, d, layD),
                  HIPBLAS_STATUS_SUCCESS);
        return gemm;
    }

    std::vector<hipblasLtMatmulHeuristicResult_t> extHeuristic(hipblaslt_ext::GemmInstance& gemm,
                                                               int                          count)
    {
        hipblaslt_ext::GemmPreference pref;
        pref.setMaxWorkspaceBytes(kWorkspace);
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        EXPECT_EQ(gemm.algoGetHeuristic(count, pref, results), HIPBLAS_STATUS_SUCCESS);
        return results;
    }

    // isSolutionSupported skips an algo whose first data byte is 0.
    const hipblasLtMatmulHeuristicResult_t*
        checkableAlgo(const std::vector<hipblasLtMatmulHeuristicResult_t>& results)
    {
        for(const auto& result : results)
            if(result.algo.data[0] != 0)
                return &result;
        return nullptr;
    }

    // The indexes tune mode would measure for this problem. Tune mode runs
    // inside a matmul, after its heuristic query has set up the descriptor.
    std::vector<int> tuningCandidates(Problem& q)
    {
        static_cast<void>(heuristic(q, 1));
        std::vector<int> indexes(hipblaslt_tuning_candidates_for_test(
            q.handle, q.desc, q.a, q.b, q.d, q.d, kWorkspace, nullptr, 0));
        hipblaslt_tuning_candidates_for_test(
            q.handle, q.desc, q.a, q.b, q.d, q.d, kWorkspace, indexes.data(), indexes.size());
        return indexes;
    }

    std::vector<std::string> namesOf(hipblasLtHandle_t handle, std::vector<int> indexes)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        EXPECT_EQ(hipblaslt_ext::getAlgosFromIndex(handle, indexes, results),
                  HIPBLAS_STATUS_SUCCESS);
        return names(handle, results);
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

    // The setting under which a packed problem used to run a HostPreSwizzle
    // kernel and return wrong values.
    TEST_F(Gfx950MxScaleLayout, smoke_MatmulApiRocRollerOff)
    {
        ScopedEnv    env("HIPBLASLT_USE_ROCROLLER", "0");
        OwnedHandle  owner;
        ASSERT_EQ(hipblasLtCreate(&owner.handle), HIPBLAS_STATUS_SUCCESS);
        hipblasLtHandle_t handle = owner.handle;
        {
            Problem packed;
            Problem preswizzled;
            makeProblem(packed, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
            makeProblem(preswizzled, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT);
            if(HasFatalFailure())
                return;

            // One algo is what hipblasLtMatmul asks for when given none.
            for(int count : {1, kAlgoCount})
            {
                SCOPED_TRACE(count);
                auto preResults = heuristic(preswizzled, count);
                ASSERT_FALSE(preResults.empty()) << "scale 1001 returned no kernel";
                for(const auto& name : names(handle, preResults))
                    EXPECT_TRUE(isPreswizzled(name)) << name;
                auto packedResults = heuristic(packed, count);
                expectNonePreswizzled(names(handle, packedResults));
            }

            auto preResults = heuristic(preswizzled, kAlgoCount);
            const auto* algo       = checkableAlgo(preResults);
            ASSERT_NE(algo, nullptr);

            EXPECT_EQ(supported(preswizzled, *algo), HIPBLAS_STATUS_SUCCESS);
            // hipblaslt_ext::isSolutionSupported casts the rocblaslt status
            // directly, so invalid_value arrives as HIPBLAS_STATUS_NOT_SUPPORTED.
            EXPECT_EQ(supported(packed, *algo), HIPBLAS_STATUS_NOT_SUPPORTED);

            EXPECT_EQ(matmul(preswizzled, &algo->algo), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(matmul(preswizzled, nullptr), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
            EXPECT_EQ(matmul(packed, &algo->algo), HIPBLAS_STATUS_INVALID_VALUE);

            auto packedResults = heuristic(packed, 1);
            if(packedResults.empty())
            {
                // No kernel at all, which hipblasLtMatmul reports as for any problem.
                EXPECT_EQ(matmul(packed, nullptr), HIPBLAS_STATUS_INTERNAL_ERROR);
            }
            else
            {
                EXPECT_EQ(matmul(packed, &packedResults[0].algo), HIPBLAS_STATUS_SUCCESS);
                EXPECT_EQ(matmul(packed, nullptr), HIPBLAS_STATUS_SUCCESS);
                EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
            }
        }
    }

    // With HIPBLASLT_USE_ROCROLLER unset, block-scaled problems are routed to
    // rocRoller. Where rocRoller is compiled out, that route still launches
    // through runContractionProblem, so a caller-supplied HostPreSwizzle algo
    // must be refused there too.
    TEST_F(Gfx950MxScaleLayout, smoke_RocRollerRouteRefusesPreswizzledAlgo)
    {
#ifdef HIPBLASLT_USE_ROCROLLER
        GTEST_SKIP() << "rocRoller is compiled in and serves packed scales itself";
#endif
        ScopedEnv    env("HIPBLASLT_USE_ROCROLLER", nullptr);
        OwnedHandle  owner;
        ASSERT_EQ(hipblasLtCreate(&owner.handle), HIPBLAS_STATUS_SUCCESS);
        hipblasLtHandle_t handle = owner.handle;
        {
            Problem packed;
            Problem preswizzled;
            makeProblem(packed, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
            makeProblem(preswizzled, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT);
            if(HasFatalFailure())
                return;

            // FP8 E4M3 with scale 1001 on both sides stays on Tensile.
            auto        preResults = heuristic(preswizzled, kAlgoCount);
            const auto* algo       = checkableAlgo(preResults);
            ASSERT_NE(algo, nullptr);
            hipblasLtMatmulAlgo_t preAlgo = algo->algo;
            ASSERT_TRUE(isPreswizzled(hipblaslt_ext::getKernelNameFromAlgo(handle, preAlgo)));

            auto packedResults = heuristic(packed, kAlgoCount);
            for(const auto& name : names(handle, packedResults))
                EXPECT_FALSE(isPreswizzled(name)) << name;
            EXPECT_EQ(supported(packed, *algo), HIPBLAS_STATUS_NOT_SUPPORTED);
            EXPECT_EQ(matmul(packed, &algo->algo), HIPBLAS_STATUS_INVALID_VALUE);
        }
    }

    // Both the all-kernels enumeration (getAllSolutions) and the ranked prefix.
    TEST_F(Gfx950MxScaleLayout, smoke_TuningCandidates)
    {
        ScopedEnv rocRoller("HIPBLASLT_USE_ROCROLLER", "0");
        for(const char* allKernels : {"1", "0"})
        {
            SCOPED_TRACE(allKernels);
            ScopedEnv    policy("HIPBLASLT_TUNING_ALL_KERNELS", allKernels);
            OwnedHandle  owner;
            ASSERT_EQ(hipblasLtCreate(&owner.handle), HIPBLAS_STATUS_SUCCESS);
            hipblasLtHandle_t handle = owner.handle;
            {
                Problem packed;
                Problem preswizzled;
                makeProblem(packed, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
                makeProblem(
                    preswizzled, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT);
                if(HasFatalFailure())
                    return;

                // All-kernels mode enumerates more than this problem type;
                // tune mode's support check removes the rest. Scale 1001's
                // candidates are this type's HostPreSwizzle kernels, and packed
                // scales must get none of them.
                auto preIndexes = tuningCandidates(preswizzled);
                ASSERT_FALSE(preIndexes.empty()) << "scale 1001 has no tuning candidate";
                for(const auto& name : namesOf(handle, preIndexes))
                    EXPECT_TRUE(isPreswizzled(name)) << name;
                const std::set<int> preswizzledSet(preIndexes.begin(), preIndexes.end());
                for(int index : tuningCandidates(packed))
                    EXPECT_EQ(preswizzledSet.count(index), 0u) << index;
            }
        }
    }

    // A tuning-file row is keyed on shape and types, so a row recorded for scale
    // 1001 also matches packed scales of the same shape. It must serve only
    // scale 1001.
    TEST_F(Gfx950MxScaleLayout, smoke_TuningFileRowServesOnlyItsScaleLayout)
    {
        ScopedEnv    rocRoller("HIPBLASLT_USE_ROCROLLER", "0");
        int          rowIndex = -1;
        std::string  rowName;
        {
            OwnedHandle owner;
            ASSERT_EQ(hipblasLtCreate(&owner.handle), HIPBLAS_STATUS_SUCCESS);
            hipblasLtHandle_t handle = owner.handle;
            {
                Problem q;
                makeProblem(q, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT);
                if(HasFatalFailure())
                    return;
                // Not the first rank, so serving it shows the row was used.
                auto results = heuristic(q, kAlgoCount);
                ASSERT_GE(results.size(), 2u);
                rowIndex = hipblaslt_ext::getIndexFromAlgo(results.back().algo);
                rowName  = hipblaslt_ext::getKernelNameFromAlgo(handle, results.back().algo);
            }
        }

        const std::string path = ::testing::TempDir() + "gfx950_mx_scale_layout_tuning.txt";
        // Declared before the variable it outlives: the tuning state is reloaded
        // after the override file is unset.
        struct Cleanup
        {
            std::string path;
            ~Cleanup()
            {
                std::remove(path.c_str());
                hipblaslt_tuning_reset_for_test();
            }
        } cleanup{path};
        {
            std::ofstream out(path, std::ios::trunc);
            out << "Git Version: gfx950_mx_scale_layout\n"
                << "transA,transB,batch_count,m,n,k,a_type,b_type,c_type,compute_type,"
                   "solution_index,kernel_name\n"
                << "T,N,1," << kM << "," << kN << "," << kK << ",f8_r,f8_r,f32_r,f32_r," << rowIndex
                << "," << rowName << "\n";
            ASSERT_TRUE(out.good());
        }
        ScopedEnv overrideFile("HIPBLASLT_TUNING_OVERRIDE_FILE", path.c_str());
        hipblaslt_tuning_reset_for_test();

        OwnedHandle owner;
        ASSERT_EQ(hipblasLtCreate(&owner.handle), HIPBLAS_STATUS_SUCCESS);
        hipblasLtHandle_t handle = owner.handle;
        {
            Problem packed;
            Problem preswizzled;
            makeProblem(packed, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
            makeProblem(preswizzled, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT);
            if(!HasFatalFailure())
            {
                auto preResults = heuristic(preswizzled, 1);
                ASSERT_EQ(preResults.size(), 1u);
                EXPECT_EQ(hipblaslt_ext::getIndexFromAlgo(preResults[0].algo), rowIndex)
                    << "the tuning-file row was not used";
                for(int count : {1, kAlgoCount})
                {
                    auto packedResults = heuristic(packed, count);
                    expectNonePreswizzled(names(handle, packedResults));
                    for(auto& result : packedResults)
                        EXPECT_NE(hipblaslt_ext::getIndexFromAlgo(result.algo), rowIndex);
                }
            }
        }
    }

    // The size-independent top-up loads every shard of the operation. When the
    // layout rule empties the ranked list, the heuristic skips it; any other short
    // list is still topped up.
    TEST_F(Gfx950MxScaleLayout, smoke_LayoutEmptiedHeuristicSkipsTheTopUp)
    {
        constexpr int kNoCall = -1;
        constexpr int kNoRank
            = static_cast<int>(TensileLite::SolutionLibrarySearchType::GEMM_TYPE_ONLY_NO_RANKING);
        constexpr int kMany = 100000;

        ScopedEnv   env("HIPBLASLT_USE_ROCROLLER", "0");
        OwnedHandle owner;
        ASSERT_EQ(hipblasLtCreate(&owner.handle), HIPBLAS_STATUS_SUCCESS);
        hipblasLtHandle_t handle = owner.handle;
        {
            Problem packed;
            Problem preswizzled;
            makeProblem(packed, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
            makeProblem(preswizzled, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT);
            if(HasFatalFailure())
                return;

            for(int count : {1, kAlgoCount})
            {
                SCOPED_TRACE(count);
                rocblaslt_testing_reset_last_get_all_search_type_for_test();
                if(heuristic(packed, count).empty())
                    EXPECT_EQ(rocblaslt_testing_last_get_all_search_type_for_test(), kNoCall);

                auto packedGemm = extGemm(packed);
                rocblaslt_testing_reset_last_get_all_search_type_for_test();
                if(extHeuristic(packedGemm, count).empty())
                    EXPECT_EQ(rocblaslt_testing_last_get_all_search_type_for_test(), kNoCall);
            }

            rocblaslt_testing_reset_last_get_all_search_type_for_test();
            auto preResults = heuristic(preswizzled, kMany);
            ASSERT_FALSE(preResults.empty()) << "scale 1001 returned no kernel";
            ASSERT_LT(preResults.size(), static_cast<size_t>(kMany));
            EXPECT_EQ(rocblaslt_testing_last_get_all_search_type_for_test(), kNoRank);
        }
    }

    // A descriptor keeps its Tensile problem between queries. After a scale-1001
    // query, scalar scales must not keep searching MX kernels.
    TEST_F(Gfx950MxScaleLayout, smoke_ReusedDescriptorLeavesMxScaling)
    {
        ScopedEnv    env("HIPBLASLT_USE_ROCROLLER", "0");
        OwnedHandle  owner;
        ASSERT_EQ(hipblasLtCreate(&owner.handle), HIPBLAS_STATUS_SUCCESS);
        hipblasLtHandle_t handle = owner.handle;
        {
            Problem q;
            makeProblem(q, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT);
            if(HasFatalFailure())
                return;

            ASSERT_FALSE(heuristic(q, kAlgoCount).empty()) << "scale 1001 returned no kernel";

            const hipblasLtMatmulMatrixScale_t scalar = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
            for(auto attr :
                {HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE})
                ASSERT_EQ(hipblasLtMatmulDescSetAttribute(q.desc, attr, &scalar, sizeof(scalar)),
                          HIPBLAS_STATUS_SUCCESS);

            auto scalarResults = heuristic(q, kAlgoCount);
            ASSERT_FALSE(scalarResults.empty()) << "scalar scales returned no kernel";
            for(const auto& name : names(handle, scalarResults))
                EXPECT_EQ(name.find("MXAE8"), std::string::npos) << name;
        }
    }

    // hipblaslt_ext::Gemm has no rocRoller dispatch, so it reaches the Tensile
    // check whatever HIPBLASLT_USE_ROCROLLER says.
    TEST_F(Gfx950MxScaleLayout, smoke_ExtGemmApi)
    {
        OwnedHandle owner;
        ASSERT_EQ(hipblasLtCreate(&owner.handle), HIPBLAS_STATUS_SUCCESS);
        hipblasLtHandle_t handle = owner.handle;
        {
            Problem packed;
            Problem preswizzled;
            makeProblem(packed, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
            makeProblem(preswizzled, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT);
            if(HasFatalFailure())
                return;

            auto preGemm    = extGemm(preswizzled);
            auto packedGemm = extGemm(packed);

            for(int count : {1, kAlgoCount})
            {
                SCOPED_TRACE(count);
                auto preResults = extHeuristic(preGemm, count);
                ASSERT_FALSE(preResults.empty()) << "scale 1001 returned no kernel";
                for(const auto& name : names(handle, preResults))
                    EXPECT_TRUE(isPreswizzled(name)) << name;
                auto packedResults = extHeuristic(packedGemm, count);
                expectNonePreswizzled(names(handle, packedResults));
            }

            hipblasLtMatmulAlgo_t algo      = extHeuristic(preGemm, 1)[0].algo;
            size_t                workspace = 0;
            EXPECT_EQ(preGemm.isAlgoSupported(algo, workspace), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(packedGemm.isAlgoSupported(algo, workspace), HIPBLAS_STATUS_INVALID_VALUE);

            preGemm.setMaxWorkspaceBytes(kWorkspace);
            packedGemm.setMaxWorkspaceBytes(kWorkspace);
            EXPECT_EQ(preGemm.initialize(algo, preswizzled.ws), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(preGemm.run(nullptr), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
            EXPECT_EQ(packedGemm.initialize(algo, packed.ws), HIPBLAS_STATUS_INVALID_VALUE);

            auto packedResults = extHeuristic(packedGemm, 1);
            if(!packedResults.empty())
            {
                EXPECT_EQ(packedGemm.initialize(packedResults[0].algo, packed.ws),
                          HIPBLAS_STATUS_SUCCESS);
                EXPECT_EQ(packedGemm.run(nullptr), HIPBLAS_STATUS_SUCCESS);
                EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
            }
        }
    }

    TEST_F(Gfx950MxScaleLayout, smoke_ExtGroupedGemmApi)
    {
        OwnedHandle owner;
        ASSERT_EQ(hipblasLtCreate(&owner.handle), HIPBLAS_STATUS_SUCCESS);
        hipblasLtHandle_t handle = owner.handle;
        {
            Problem packed;
            Problem preswizzled;
            makeProblem(packed, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
            makeProblem(preswizzled, handle, HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT);
            if(HasFatalFailure())
                return;

            auto preGemm    = extGroupedGemm(preswizzled);
            auto packedGemm = extGroupedGemm(packed);

            auto preResults = extHeuristic(preGemm, kAlgoCount);
            if(preResults.empty())
                GTEST_SKIP() << "no grouped MX kernel in the loaded library";
            for(const auto& name : names(handle, preResults))
                EXPECT_TRUE(isPreswizzled(name)) << name;
            auto packedResults = extHeuristic(packedGemm, kAlgoCount);
            expectNonePreswizzled(names(handle, packedResults));

            hipblasLtMatmulAlgo_t algo      = preResults[0].algo;
            size_t                workspace = 0;
            EXPECT_EQ(preGemm.isAlgoSupported(algo, workspace), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(packedGemm.isAlgoSupported(algo, workspace), HIPBLAS_STATUS_INVALID_VALUE);

            preGemm.setMaxWorkspaceBytes(kWorkspace);
            packedGemm.setMaxWorkspaceBytes(kWorkspace);
            EXPECT_EQ(preGemm.initialize(algo, preswizzled.ws, false), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(packedGemm.initialize(algo, packed.ws, false), HIPBLAS_STATUS_INVALID_VALUE);

            if(!packedResults.empty())
            {
                EXPECT_EQ(packedGemm.initialize(packedResults[0].algo, packed.ws, false),
                          HIPBLAS_STATUS_SUCCESS);
                EXPECT_EQ(packedGemm.run(nullptr), HIPBLAS_STATUS_SUCCESS);
                EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
            }
        }
    }
}
