// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "solution_entry.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <set>
#include <string>
#include <vector>

// Heuristic queries under HIPBLASLT_JIT. Each CTest entry runs one Catch2 case,
// which sets HIPBLASLT_JIT and points HIPBLASLT_JIT_LIBRARY_PATH at a scratch
// directory. The first query is a miss and generation runs. The second query in
// this process is a hit and generation does not run.
namespace
{
    using hipblaslt_jit_test::Device;
    using hipblaslt_jit_test::endsWith;

#define HIP(expression) hipblaslt_jit_test::checkHip((expression), #expression)

#ifdef HIPBLASLT_JIT_HIPKITTENS
    constexpr int M = 256;
    constexpr int N = 256;
    constexpr int kWide = 128;
    constexpr int kNarrow = 256;
    constexpr hipblasOperation_t opA = HIPBLAS_OP_T;
    constexpr const char* wideSuffix = "HK_gemm_f16_TN";
    constexpr const char* narrowSuffix = "HK_gemm_f16_TN";
#else
    constexpr int M = hipblaslt_jit_test::Fp16Gemm::rows;
    constexpr int N = hipblaslt_jit_test::Fp16Gemm::cols;
    constexpr int kWide = 512;
    constexpr int kNarrow = 256;
    constexpr hipblasOperation_t opA = HIPBLAS_OP_N;
    constexpr const char* wideSuffix = "_K512_WGM8";
    constexpr const char* narrowSuffix = "_WGM1";
#endif
    constexpr uint64_t workspaceLimit = 64ull << 20;

    bool nameMatches(const std::string& name, const std::string& token)
    {
        if(!token.empty() && token.front() == '_')
            return endsWith(name, token);
        return name.find(token) != std::string::npos;
    }

    struct Layouts
    {
        hipblasLtMatrixLayout_t a{}, b{}, c{}, d{};
        Layouts(int rowsA, int colsA, int ldA, int rowsB, int colsB, int ldB)
        {
            {
                INFO(("layout A"));
                REQUIRE((hipblasLtMatrixLayoutCreate(&a, HIP_R_16F, rowsA, colsA, ldA)
                    == HIPBLAS_STATUS_SUCCESS));
            }
            {
                INFO(("layout B"));
                REQUIRE((hipblasLtMatrixLayoutCreate(&b, HIP_R_16F, rowsB, colsB, ldB)
                    == HIPBLAS_STATUS_SUCCESS));
            }
            {
                INFO(("layout C"));
                REQUIRE((
                    hipblasLtMatrixLayoutCreate(&c, HIP_R_16F, M, N, M) == HIPBLAS_STATUS_SUCCESS));
            }
            {
                INFO(("layout D"));
                REQUIRE((
                    hipblasLtMatrixLayoutCreate(&d, HIP_R_16F, M, N, M) == HIPBLAS_STATUS_SUCCESS));
            }
        }
        ~Layouts()
        {
            hipblasLtMatrixLayoutDestroy(a);
            hipblasLtMatrixLayoutDestroy(b);
            hipblasLtMatrixLayoutDestroy(c);
            hipblasLtMatrixLayoutDestroy(d);
        }
        Layouts(const Layouts&)            = delete;
        Layouts& operator=(const Layouts&) = delete;
    };

    struct Listed
    {
        hipblasStatus_t                              status{};
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
    };

    bool isJit(hipblasLtMatmulAlgo_t algo)
    {
        // Heuristic results are JIT solution library indices, from 2^30.
        return hipblaslt_ext::getIndexFromAlgo(algo) >= (1 << 30);
    }

    Listed queryC(hipblasLtHandle_t            handle,
                  hipblasLtMatmulDesc_t        desc,
                  hipblasLtMatmulPreference_t  pref,
                  const Layouts&               layouts,
                  int                          requested)
    {
        Listed listed;
        listed.results.resize(requested);
        int count = 0;
        listed.status = hipblasLtMatmulAlgoGetHeuristic(handle,
                                                        desc,
                                                        layouts.a,
                                                        layouts.b,
                                                        layouts.c,
                                                        layouts.d,
                                                        pref,
                                                        requested,
                                                        listed.results.data(),
                                                        &count);
        {
            INFO(("C heuristic returned a count outside the request"));
            REQUIRE((count >= 0 && count <= requested));
        }
        listed.results.resize(count);
        return listed;
    }

    Listed queryCpp(hipblasLtHandle_t           handle,
                    hipblasLtMatmulDesc_t       desc,
                    hipblasOperation_t          opA,
                    hipblasOperation_t          opB,
                    const Layouts&              layouts,
                    void*                       A,
                    void*                       B,
                    void*                       C,
                    void*                       D,
                    int                         requested)
    {
        hipblaslt_ext::Gemm gemm(handle,
                                 opA,
                                 opB,
                                 HIP_R_16F,
                                 HIP_R_16F,
                                 HIP_R_16F,
                                 HIP_R_16F,
                                 HIPBLAS_COMPUTE_32F);
        float               alpha = 1.25f, beta = 0.5f;
        {
            INFO(("Gemm::setProblem"));
            REQUIRE((gemm.setProblem(desc, &alpha, A, layouts.a, B, layouts.b, &beta, C, layouts.c, D,
                            layouts.d)
                == HIPBLAS_STATUS_SUCCESS));
        }
        hipblaslt_ext::GemmPreference pref;
        pref.setMaxWorkspaceBytes(workspaceLimit);
        Listed listed;
        listed.status = gemm.algoGetHeuristic(requested, pref, listed.results);
        {
            INFO(("C++ heuristic returned more results than requested"));
            REQUIRE((static_cast<int>(listed.results.size()) <= requested));
        }
        return listed;
    }

    void requireNoJit(const std::string& label, const Listed& listed)
    {
        for(const auto& result : listed.results)
        {
            INFO((label + " returned a JIT algorithm"));
            REQUIRE((!isJit(result.algo)));
        }
        if(listed.results.empty())
        {
            INFO((label + " with no results returned status " + std::to_string(listed.status)));
            REQUIRE((listed.status == HIPBLAS_STATUS_INVALID_VALUE));
        }
        else
        {
            INFO((label + " returned status " + std::to_string(listed.status)));
            REQUIRE((listed.status == HIPBLAS_STATUS_SUCCESS));
        }
    }

    void requireOnlyJit(const std::string& label, const Listed& listed, const std::string& suffix,
                        hipblasLtHandle_t handle)
    {
        {
            INFO((label + " returned status " + std::to_string(listed.status)));
            REQUIRE((listed.status == HIPBLAS_STATUS_SUCCESS));
        }
        {
            INFO((label + " returned no algorithm"));
            REQUIRE((!listed.results.empty()));
        }
        for(const auto& result : listed.results)
        {
            auto algo = result.algo;
            {
                INFO((label + " returned an algorithm that is not JIT"));
                REQUIRE((isJit(algo)));
            }
            const auto name = hipblaslt_ext::getSolutionNameFromAlgo(handle, algo);
            {
                INFO((label + " selected '" + name + "', not a solution matching " + suffix));
                REQUIRE((nameMatches(name, suffix)));
            }
        }
    }

    void requireUniqueKernels(hipblasLtHandle_t handle, const Listed& listed)
    {
        std::set<std::string> names;
        for(const auto& result : listed.results)
        {
            auto algo = result.algo;
            const auto name = hipblaslt_ext::getKernelNameFromAlgo(handle, algo);
            if(name.empty())
                continue;
            {
                INFO(("repeated kernel " + name));
                REQUIRE((names.insert(name).second));
            }
        }
    }

    // Equality results, then JIT, then the other providers. A full list must
    // end with a non-JIT result; a short list may end on JIT.
    void requireProviderOrder(const Listed& listed, int requested)
    {
        {
            INFO(("an Equality size returned no algorithm"));
            REQUIRE((!listed.results.empty()));
        }
        {
            INFO(("an Equality size did not start with an Equality result"));
            REQUIRE((!isJit(listed.results.front().algo)));
        }
        int  phase  = 0;
        bool sawJit = false;
        for(auto& result : listed.results)
        {
            const bool jit = isJit(result.algo);
            if(phase == 0 && jit)
                phase = 1;
            else if(phase == 1 && !jit)
                phase = 2;
            else if(phase == 2 && jit)
            {
                FAIL(("a JIT result followed the other providers"));
            }
            if(jit)
                sawJit = true;
        }
        {
            INFO(("an Equality size returned no JIT result after the Equality results"));
            REQUIRE((sawJit));
        }
        if(static_cast<int>(listed.results.size()) == requested)
        {
            INFO(("a full Equality-size list did not end with another provider"));
            REQUIRE((!isJit(listed.results.back().algo)));
        }
    }

    void runFirst(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc, hipStream_t stream, int K,
                  const hipblasLtMatmulHeuristicResult_t& result)
    {
        hipblaslt_jit_test::Fp16Gemm matrices(K);
        Device                       workspace(result.workspaceSize);
        float                        alpha = 1.25f, beta = 0.5f;
        matrices.matmul(handle,
                        desc,
                        stream,
                        &result.algo,
                        workspace.pointer,
                        result.workspaceSize,
                        alpha,
                        beta,
                        "");
    }

    bool equalitySize(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc, int m, int n, int k)
    {
        hipblasLtMatrixLayout_t a{}, b{}, c{};
        {
            INFO(("equality layout A"));
            REQUIRE((
                hipblasLtMatrixLayoutCreate(&a, HIP_R_16F, m, k, m) == HIPBLAS_STATUS_SUCCESS));
        }
        {
            INFO(("equality layout B"));
            REQUIRE((
                hipblasLtMatrixLayoutCreate(&b, HIP_R_16F, k, n, k) == HIPBLAS_STATUS_SUCCESS));
        }
        {
            INFO(("equality layout C"));
            REQUIRE((
                hipblasLtMatrixLayoutCreate(&c, HIP_R_16F, m, n, m) == HIPBLAS_STATUS_SUCCESS));
        }
        const int tuned = hipblaslt_ext::matmulIsTuned(handle, desc, a, b, c, c);
        hipblasLtMatrixLayoutDestroy(a);
        hipblasLtMatrixLayoutDestroy(b);
        hipblasLtMatrixLayoutDestroy(c);
        return tuned == 1;
    }

    void checkPair(const std::string& label, const Listed& c, const Listed& cpp, bool jit,
                   const std::string& suffix, hipblasLtHandle_t handle)
    {
        if(jit)
        {
            requireOnlyJit(label + " C", c, suffix, handle);
            requireOnlyJit(label + " C++", cpp, suffix, handle);
        }
        else
        {
            requireNoJit(label + " C", c);
            requireNoJit(label + " C++", cpp);
        }
        requireUniqueKernels(handle, c);
        requireUniqueKernels(handle, cpp);
    }

    void requireMissThenHit(uint64_t before, uint64_t afterFirst, uint64_t afterSecond, int first,
                             int second)
    {
        {
            INFO(("the first query did not generate"));
            REQUIRE((afterFirst == before + 1 && afterSecond == before + 1));
        }
        {
            INFO(("the second query returned another index"));
            REQUIRE((first == second));
        }
    }

    void test(const std::string& mode)
    {
        hipblasLtHandle_t           handle{};
        hipblasLtMatmulDesc_t       desc{};
        hipblasLtMatmulPreference_t pref{};
        hipStream_t                 stream{};
        {
            INFO(("hipblasLtCreate"));
            REQUIRE((hipblasLtCreate(&handle) == HIPBLAS_STATUS_SUCCESS));
        }
        {
            INFO(("hipblasLtMatmulDescCreate"));
            REQUIRE((hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                == HIPBLAS_STATUS_SUCCESS));
        }
        {
            INFO(("preference"));
            REQUIRE((hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS));
        }
        {
            INFO(("workspace preference"));
            REQUIRE((hipblasLtMatmulPreferenceSetAttribute(pref,
                                                  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspaceLimit,
                                                  sizeof(workspaceLimit))
                == HIPBLAS_STATUS_SUCCESS));
        }
        HIP(hipStreamCreate(&stream));

        constexpr int kBuffer = kWide > kNarrow ? kWide : kNarrow;
        Device        A(static_cast<size_t>(M) * kBuffer * 2);
        Device        B(static_cast<size_t>(kBuffer) * N * 2);
        Device        C(static_cast<size_t>(M) * N * 2);
        Device        D(static_cast<size_t>(M) * N * 2);
        const int equalitySizes[][3]
            = {{1024, 4096, 20}, {2048, 128, 16}, {864, 512, 432}, {128, 5120, 1024}};
        bool tuned = false;
        if(mode == "fallback")
        {
            for(const auto& size : equalitySizes)
                tuned = tuned || equalitySize(handle, desc, size[0], size[1], size[2]);
        }
#ifdef HIPBLASLT_JIT_HIPKITTENS
        const hipblasOperation_t transpose = HIPBLAS_OP_T;
        {
            INFO(("transpose A"));
            REQUIRE((hipblasLtMatmulDescSetAttribute(
                    desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose))
                == HIPBLAS_STATUS_SUCCESS));
        }
#endif

#ifdef HIPBLASLT_JIT_HIPKITTENS
        Layouts wide(kWide, M, kWide, kWide, N, kWide);
        Layouts narrow(kNarrow, M, kNarrow, kNarrow, N, kNarrow);
#else
        Layouts wide(M, kWide, M, kWide, N, kWide);
        Layouts narrow(M, kNarrow, M, kNarrow, N, kNarrow);
#endif
#ifndef HIPBLASLT_JIT_HEURISTIC_IGNORED
        const auto before = hipblaslt_jit::generationCount();
#endif
        // One result fills the request, so the C++ query is a library hit.
        const auto cWide = queryC(handle, desc, pref, wide, 1);
#ifndef HIPBLASLT_JIT_HEURISTIC_IGNORED
        const auto afterWideC = hipblaslt_jit::generationCount();
#endif
        const auto cppWide
            = queryCpp(handle, desc, opA, HIPBLAS_OP_N, wide, A.pointer, B.pointer, C.pointer,
                       D.pointer, 1);
#ifndef HIPBLASLT_JIT_HEURISTIC_IGNORED
        const auto afterWide = hipblaslt_jit::generationCount();
#endif
        const auto cNarrow = queryC(handle, desc, pref, narrow, 1);
#ifndef HIPBLASLT_JIT_HEURISTIC_IGNORED
        const auto afterNarrowC = hipblaslt_jit::generationCount();
#endif
        const auto cppNarrow = queryCpp(handle, desc, opA, HIPBLAS_OP_N, narrow, A.pointer,
                                        B.pointer, C.pointer, D.pointer, 1);
#ifndef HIPBLASLT_JIT_HEURISTIC_IGNORED
        const auto afterNarrow = hipblaslt_jit::generationCount();
#endif

        if(mode == "off" || mode == "ignored")
        {
#ifndef HIPBLASLT_JIT_HEURISTIC_IGNORED
            {
                INFO(("HIPBLASLT_JIT off generated a solution"));
                REQUIRE((afterNarrow == before));
            }
#endif
            checkPair("wide", cWide, cppWide, false, "", handle);
            checkPair("narrow", cNarrow, cppNarrow, false, "", handle);
        }
        else if(mode == "forced" || !tuned)
        {
            checkPair("wide", cWide, cppWide, true, wideSuffix, handle);
            checkPair("narrow", cNarrow, cppNarrow, true, narrowSuffix, handle);
#ifndef HIPBLASLT_JIT_HEURISTIC_IGNORED
            requireMissThenHit(before,
                               afterWideC,
                               afterWide,
                               hipblaslt_ext::getIndexFromAlgo(cWide.results.front().algo),
                               hipblaslt_ext::getIndexFromAlgo(cppWide.results.front().algo));
            requireMissThenHit(afterWide,
                               afterNarrowC,
                               afterNarrow,
                               hipblaslt_ext::getIndexFromAlgo(cNarrow.results.front().algo),
                               hipblaslt_ext::getIndexFromAlgo(cppNarrow.results.front().algo));
#endif
#ifndef HIPBLASLT_JIT_HIPKITTENS
            runFirst(handle, desc, stream, kWide, cWide.results.front());
            std::cout << "PASS wide first result\n";
#endif
            std::cout << "PASS a second query reused the JIT library index\n";
        }
        else
        {
            {
                INFO(("wide"));
                REQUIRE((cWide.status == HIPBLAS_STATUS_SUCCESS && !cWide.results.empty()));
            }
            {
                INFO(("an untuned size with library results did not start with JIT"));
                REQUIRE((isJit(cWide.results.front().algo)));
            }
#ifndef HIPBLASLT_JIT_HEURISTIC_IGNORED
            requireMissThenHit(before,
                               afterWideC,
                               afterWide,
                               hipblaslt_ext::getIndexFromAlgo(cWide.results.front().algo),
                               hipblaslt_ext::getIndexFromAlgo(cppWide.results.front().algo));
#endif
            requireUniqueKernels(handle, cWide);
            requireUniqueKernels(handle, cppWide);
#ifndef HIPBLASLT_JIT_HIPKITTENS
            runFirst(handle, desc, stream, kWide, cWide.results.front());
#endif
            std::cout << "PASS wide first result\n";
        }

        if(mode == "forced" || mode == "fallback")
        {
#ifdef HIPBLASLT_JIT_HIPKITTENS
            // NN is outside the HipKittens kernel.
            hipblasLtMatmulDesc_t outside{};
            {
                INFO(("NN descriptor"));
                REQUIRE((hipblasLtMatmulDescCreate(&outside, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                    == HIPBLAS_STATUS_SUCCESS));
            }
            Layouts    nn(M, kWide, M, kWide, N, kWide);
            const auto cT = queryC(handle, outside, pref, nn, 4);
            const auto cppT = queryCpp(handle, outside, HIPBLAS_OP_N, HIPBLAS_OP_N, nn, A.pointer,
                                       B.pointer, C.pointer, D.pointer, 4);
#else
            hipblasLtMatmulDesc_t    outside{};
            const hipblasOperation_t transpose = HIPBLAS_OP_T;
            {
                INFO(("transposed descriptor"));
                REQUIRE((hipblasLtMatmulDescCreate(&outside, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                    == HIPBLAS_STATUS_SUCCESS));
            }
            {
                INFO(("transpose A"));
                REQUIRE((hipblasLtMatmulDescSetAttribute(outside, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                        &transpose, sizeof(transpose))
                    == HIPBLAS_STATUS_SUCCESS));
            }
            Layouts    nn(kWide, M, kWide, kWide, N, kWide);
            const auto cT = queryC(handle, outside, pref, nn, 4);
            const auto cppT = queryCpp(handle, outside, HIPBLAS_OP_T, HIPBLAS_OP_N, nn, A.pointer,
                                       B.pointer, C.pointer, D.pointer, 4);
#endif
            if(mode == "forced")
            {
                {
                    INFO(("forced mode returned a solution the kernel does not solve"));
                    REQUIRE((cT.status == HIPBLAS_STATUS_SUCCESS && cT.results.empty()));
                }
                {
                    INFO(("forced mode C++ returned a solution the kernel does not solve"));
                    REQUIRE((cppT.status == HIPBLAS_STATUS_SUCCESS && cppT.results.empty()));
                }
                std::cout << "PASS a problem the kernel does not solve returns no solution\n";
            }
            hipblasLtMatmulDescDestroy(outside);
        }

        if(mode == "fallback")
        {
            if(!tuned)
                std::cout << "SKIP heuristic-provider-order: the build has no device library with "
                             "an Equality size\n";
            else
            {
                for(const auto& size : equalitySizes)
                {
                    if(!equalitySize(handle, desc, size[0], size[1], size[2]))
                        continue;
                    hipblasLtMatrixLayout_t a{}, b{}, c{}, d{};
                    {
                        INFO(("order layout A"));
                        REQUIRE((hipblasLtMatrixLayoutCreate(&a, HIP_R_16F, size[0], size[2], size[0])
                            == HIPBLAS_STATUS_SUCCESS));
                    }
                    {
                        INFO(("order layout B"));
                        REQUIRE((hipblasLtMatrixLayoutCreate(&b, HIP_R_16F, size[2], size[1], size[2])
                            == HIPBLAS_STATUS_SUCCESS));
                    }
                    {
                        INFO(("order layout C"));
                        REQUIRE((hipblasLtMatrixLayoutCreate(&c, HIP_R_16F, size[0], size[1], size[0])
                            == HIPBLAS_STATUS_SUCCESS));
                    }
                    {
                        INFO(("order layout D"));
                        REQUIRE((hipblasLtMatrixLayoutCreate(&d, HIP_R_16F, size[0], size[1], size[0])
                            == HIPBLAS_STATUS_SUCCESS));
                    }
                    std::vector<hipblasLtMatmulHeuristicResult_t> results(8);
                    int                                           count = 0;
                    const auto                                    status
                        = hipblasLtMatmulAlgoGetHeuristic(handle, desc, a, b, c, d, pref, 8,
                                                          results.data(), &count);
                    results.resize(count);
                    {
                        INFO(("Equality-size heuristic"));
                        REQUIRE((status == HIPBLAS_STATUS_SUCCESS));
                    }
                    Listed listed{status, std::move(results)};
                    requireProviderOrder(listed, 8);
                    requireUniqueKernels(handle, listed);
                    hipblasLtMatrixLayoutDestroy(a);
                    hipblasLtMatrixLayoutDestroy(b);
                    hipblasLtMatrixLayoutDestroy(c);
                    hipblasLtMatrixLayoutDestroy(d);
                    std::cout << "PASS Equality size " << size[0] << "x" << size[1] << "x"
                              << size[2] << " is Equality, then JIT, then the other providers\n";
                    break;
                }
            }
        }

        std::cout << "PASS heuristic mode " << mode << '\n';
        static_cast<void>(hipStreamDestroy(stream));
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtDestroy(handle);
    }
}

#ifdef HIPBLASLT_JIT_HEURISTIC_IGNORED
TEST_CASE("heuristic ignored", "[jit-gpu]")
{
    hipblaslt_jit_test::setJitMode("2");
    hipblaslt_jit_test::LibraryScratch library("heuristic-ignored");
    test("ignored");
}
#else
TEST_CASE("heuristic off", "[jit-gpu]")
{
    hipblaslt_jit_test::setJitMode("0");
    hipblaslt_jit_test::LibraryScratch library("heuristic-off");
    test("off");
}

TEST_CASE("heuristic fallback", "[jit-gpu]")
{
    hipblaslt_jit_test::setJitMode("1");
    hipblaslt_jit_test::LibraryScratch library("heuristic-fallback");
    test("fallback");
}

TEST_CASE("heuristic forced", "[jit-gpu]")
{
    hipblaslt_jit_test::setJitMode("2");
    hipblaslt_jit_test::LibraryScratch library("heuristic-forced");
    test("forced");
}
#endif
