// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "test_helpers.hpp"
#include <Tensile/LocalKernelIndex.hpp>
#include <catch2/catch_test_macros.hpp>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <set>
#include <string>
#include <vector>

// Heuristic queries under HIPBLASLT_JIT. Each CTest entry runs one Catch2 case,
// which sets HIPBLASLT_JIT and points HIPBLASLT_JIT_LIBRARY_PATH at a scratch
// directory. A JIT build adds the replay backend for the queries. A second
// query returns the same JIT solution library index.
namespace
{
    using hipblaslt_jit_test::Device;
    using hipblaslt_jit_test::endsWith;

    constexpr int                M              = hipblaslt_jit_test::Fp16Gemm::rows;
    constexpr int                N              = hipblaslt_jit_test::Fp16Gemm::cols;
    constexpr int                kWide          = 512;
    constexpr int                kNarrow        = 256;
    constexpr hipblasOperation_t opA            = HIPBLAS_OP_N;
    constexpr const char*        wideSuffix     = "_K512_WGM8";
    constexpr const char*        narrowSuffix   = "_WGM1";
    constexpr uint64_t           workspaceLimit = 64ull << 20;

    bool nameMatches(const std::string& name, const std::string& token)
    {
        if(!token.empty() && token.front() == '_')
            return endsWith(name, token);
        return name.find(token) != std::string::npos;
    }

    struct Layouts
    {
        hipblasLtMatrixLayout_t a{}, b{}, c{}, d{};
        Layouts(int m, int n, int k, bool transposeA = false)
        {
            using hipblaslt_jit_test::checkBlas;
            checkBlas(transposeA ? hipblasLtMatrixLayoutCreate(&a, HIP_R_16F, k, m, k)
                                 : hipblasLtMatrixLayoutCreate(&a, HIP_R_16F, m, k, m),
                      "layout A");
            checkBlas(hipblasLtMatrixLayoutCreate(&b, HIP_R_16F, k, n, k), "layout B");
            checkBlas(hipblasLtMatrixLayoutCreate(&c, HIP_R_16F, m, n, m), "layout C");
            checkBlas(hipblasLtMatrixLayoutCreate(&d, HIP_R_16F, m, n, m), "layout D");
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
        hipblasStatus_t                               status{};
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
    };

    bool isJit(hipblasLtMatmulAlgo_t algo)
    {
        return TensileLite::isJitKernelIndex(hipblaslt_ext::getIndexFromAlgo(algo));
    }

    Listed queryC(hipblasLtHandle_t           handle,
                  hipblasLtMatmulDesc_t       desc,
                  hipblasLtMatmulPreference_t pref,
                  const Layouts&              layouts,
                  int                         requested)
    {
        Listed listed;
        listed.results.resize(requested);
        int count     = 0;
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
        JIT_REQUIRE("C heuristic returned a count outside the request",
                    count >= 0 && count <= requested);
        listed.results.resize(count);
        return listed;
    }

    Listed queryCpp(hipblasLtHandle_t     handle,
                    hipblasLtMatmulDesc_t desc,
                    hipblasOperation_t    opA,
                    hipblasOperation_t    opB,
                    const Layouts&        layouts,
                    void*                 A,
                    void*                 B,
                    void*                 C,
                    void*                 D,
                    int                   requested)
    {
        hipblaslt_ext::Gemm gemm(
            handle, opA, opB, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F);
        float alpha = 1.25f, beta = 0.5f;
        JIT_REQUIRE("Gemm::setProblem",
                    gemm.setProblem(
                        desc, &alpha, A, layouts.a, B, layouts.b, &beta, C, layouts.c, D, layouts.d)
                        == HIPBLAS_STATUS_SUCCESS);
        hipblaslt_ext::GemmPreference pref;
        pref.setMaxWorkspaceBytes(workspaceLimit);
        Listed listed;
        listed.status = gemm.algoGetHeuristic(requested, pref, listed.results);
        JIT_REQUIRE("C++ heuristic returned more results than requested",
                    static_cast<int>(listed.results.size()) <= requested);
        return listed;
    }

    void requireNoJit(const std::string& label, const Listed& listed)
    {
        for(const auto& result : listed.results)
        {
            JIT_REQUIRE(label + " returned a JIT algorithm", !isJit(result.algo));
        }
        if(listed.results.empty())
        {
            JIT_REQUIRE(label + " with no results returned status " + std::to_string(listed.status),
                        listed.status == HIPBLAS_STATUS_INVALID_VALUE);
        }
        else
        {
            JIT_REQUIRE(label + " returned status " + std::to_string(listed.status),
                        listed.status == HIPBLAS_STATUS_SUCCESS);
        }
    }

    // The kernel's macro tile over this M x N, divided by the compute units.
    float expectedWaves(const std::string& kernel)
    {
        const auto at    = kernel.find("_MT");
        int        tile0 = 0, tile1 = 0;
        REQUIRE((at != std::string::npos
                 && std::sscanf(kernel.c_str() + at, "_MT%dx%d", &tile0, &tile1) == 2));
        int             device = -1;
        hipDeviceProp_t properties{};
        HIP(hipGetDevice(&device));
        HIP(hipGetDeviceProperties(&properties, device));
        const int groups = ((M + tile0 - 1) / tile0) * ((N + tile1 - 1) / tile1);
        return static_cast<float>(groups) / static_cast<float>(properties.multiProcessorCount);
    }

    void requireOnlyJit(const std::string& label,
                        const Listed&      listed,
                        const std::string& suffix,
                        hipblasLtHandle_t  handle)
    {
        JIT_REQUIRE(label + " returned status " + std::to_string(listed.status),
                    listed.status == HIPBLAS_STATUS_SUCCESS);
        JIT_REQUIRE(label + " returned no algorithm", !listed.results.empty());
        for(const auto& result : listed.results)
        {
            auto algo = result.algo;
            JIT_REQUIRE(label + " returned an algorithm that is not JIT", isJit(algo));
            const auto name = hipblaslt_ext::getSolutionNameFromAlgo(handle, algo);
            JIT_REQUIRE(label + " selected '" + name + "', not a solution matching " + suffix,
                        nameMatches(name, suffix));
            {
                const auto kernel = hipblaslt_ext::getKernelNameFromAlgo(handle, algo);
                INFO((label + " reported wavesCount " + std::to_string(result.wavesCount)));
                REQUIRE((std::abs(result.wavesCount - expectedWaves(kernel)) <= 1e-6f));
            }
        }
    }

    void requireUniqueKernels(hipblasLtHandle_t handle, const Listed& listed)
    {
        std::set<std::string> names;
        for(const auto& result : listed.results)
        {
            auto       algo = result.algo;
            const auto name = hipblaslt_ext::getKernelNameFromAlgo(handle, algo);
            if(name.empty())
                continue;
            JIT_REQUIRE("repeated kernel " + name, names.insert(name).second);
        }
    }

    // Equality results, then JIT, then the other providers. A full list must
    // end with a non-JIT result; a short list may end on JIT.
    void requireProviderOrder(const Listed& listed, int requested)
    {
        JIT_REQUIRE("an Equality size returned no algorithm", !listed.results.empty());
        JIT_REQUIRE("an Equality size did not start with an Equality result",
                    !isJit(listed.results.front().algo));
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
        JIT_REQUIRE("an Equality size returned no JIT result after the Equality results", sawJit);
        if(static_cast<int>(listed.results.size()) == requested)
        {
            JIT_REQUIRE("a full Equality-size list did not end with another provider",
                        !isJit(listed.results.back().algo));
        }
    }

    // A null result passes a null algo, which leaves the selection to hipblasLtMatmul.
    void runFirst(hipblasLtHandle_t                       handle,
                  hipblasLtMatmulDesc_t                   desc,
                  hipStream_t                             stream,
                  int                                     K,
                  const hipblasLtMatmulHeuristicResult_t* result)
    {
        hipblaslt_jit_test::Fp16Gemm matrices(K);
        const size_t                 bytes = result ? result->workspaceSize : workspaceLimit;
        Device                       workspace(bytes);
        float                        alpha = 1.25f, beta = 0.5f;
        matrices.matmul(handle,
                        desc,
                        stream,
                        result ? &result->algo : nullptr,
                        workspace.pointer,
                        bytes,
                        alpha,
                        beta,
                        "");
    }

    bool equalitySize(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc, int m, int n, int k)
    {
        const Layouts layouts(m, n, k);
        return hipblaslt_ext::matmulIsTuned(
                   handle, desc, layouts.a, layouts.b, layouts.c, layouts.c)
               == 1;
    }

    void checkPair(const std::string& label,
                   const Listed&      c,
                   const Listed&      cpp,
                   bool               jit,
                   const std::string& suffix,
                   hipblasLtHandle_t  handle)
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

    int publishedIndex(const hipblasLtMatmulHeuristicResult_t& result)
    {
        auto algo = result.algo;
        return hipblaslt_ext::getIndexFromAlgo(algo);
    }

    void requireSameIndex(int first, int second)
    {
        JIT_REQUIRE("the second query returned another index", first == second);
    }

    void test(const std::string& mode)
    {
        hipblasLtHandle_t           handle{};
        hipblasLtMatmulDesc_t       desc{};
        hipblasLtMatmulPreference_t pref{};
        hipStream_t                 stream{};
        JIT_REQUIRE("hipblasLtCreate", hipblasLtCreate(&handle) == HIPBLAS_STATUS_SUCCESS);
        JIT_REQUIRE("hipblasLtMatmulDescCreate",
                    hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                        == HIPBLAS_STATUS_SUCCESS);
        JIT_REQUIRE("preference", hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS);
        JIT_REQUIRE("workspace preference",
                    hipblasLtMatmulPreferenceSetAttribute(pref,
                                                          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &workspaceLimit,
                                                          sizeof(workspaceLimit))
                        == HIPBLAS_STATUS_SUCCESS);
        HIP(hipStreamCreate(&stream));

        constexpr int kBuffer = kWide > kNarrow ? kWide : kNarrow;
        Device        A(static_cast<size_t>(M) * kBuffer * 2);
        Device        B(static_cast<size_t>(kBuffer) * N * 2);
        Device        C(static_cast<size_t>(M) * N * 2);
        Device        D(static_cast<size_t>(M) * N * 2);
        const int     equalitySizes[][3]
            = {{1024, 4096, 20}, {2048, 128, 16}, {864, 512, 432}, {128, 5120, 1024}};
        bool tuned = false;
        if(mode == "fallback")
        {
            for(const auto& size : equalitySizes)
                tuned = tuned || equalitySize(handle, desc, size[0], size[1], size[2]);
        }

        Layouts wide(M, N, kWide);
        Layouts narrow(M, N, kNarrow);
#ifndef HIPBLASLT_JIT_HEURISTIC_IGNORED
        if(mode == "forced")
        {
            const auto none = queryC(handle, desc, pref, wide, 1);
            JIT_REQUIRE("forced mode with no backend returned status " + std::to_string(none.status)
                            + " and " + std::to_string(none.results.size()) + " results",
                        none.status == HIPBLAS_STATUS_SUCCESS && none.results.empty());
        }
        {
            const hipblaslt_jit_test::ReplayFixture replay(HIPBLASLT_JIT_DATA);
            hipblaslt_ext::experimental::jit::addHeuristicBackend(replay.backend);
        }
#endif
        // One result fills the request, so the C++ query returns the same index.
        const auto cWide   = queryC(handle, desc, pref, wide, 1);
        const auto cppWide = queryCpp(
            handle, desc, opA, HIPBLAS_OP_N, wide, A.pointer, B.pointer, C.pointer, D.pointer, 1);
        const auto cNarrow   = queryC(handle, desc, pref, narrow, 1);
        const auto cppNarrow = queryCpp(
            handle, desc, opA, HIPBLAS_OP_N, narrow, A.pointer, B.pointer, C.pointer, D.pointer, 1);

        if(mode == "off" || mode == "ignored")
        {
            checkPair("wide", cWide, cppWide, false, "", handle);
            checkPair("narrow", cNarrow, cppNarrow, false, "", handle);
        }
        else if(mode == "forced" || !tuned)
        {
            checkPair("wide", cWide, cppWide, true, wideSuffix, handle);
            checkPair("narrow", cNarrow, cppNarrow, true, narrowSuffix, handle);
#ifndef HIPBLASLT_JIT_HEURISTIC_IGNORED
            requireSameIndex(publishedIndex(cWide.results.front()),
                             publishedIndex(cppWide.results.front()));
            requireSameIndex(publishedIndex(cNarrow.results.front()),
                             publishedIndex(cppNarrow.results.front()));
#endif
            runFirst(handle, desc, stream, kWide, &cWide.results.front());
        }
        else
        {
            JIT_REQUIRE("wide", cWide.status == HIPBLAS_STATUS_SUCCESS && !cWide.results.empty());
            JIT_REQUIRE("an untuned size with library results did not start with JIT",
                        isJit(cWide.results.front().algo));
#ifndef HIPBLASLT_JIT_HEURISTIC_IGNORED
            requireSameIndex(publishedIndex(cWide.results.front()),
                             publishedIndex(cppWide.results.front()));
#endif
            requireUniqueKernels(handle, cWide);
            requireUniqueKernels(handle, cppWide);
            runFirst(handle, desc, stream, kWide, &cWide.results.front());
        }

        if(mode == "forced" || mode == "fallback")
        {
            // A null algo launches the first ranked result, the JIT one checked above.
            runFirst(handle, desc, stream, kWide, nullptr);

            hipblasLtMatmulDesc_t    outside{};
            const hipblasOperation_t transpose = HIPBLAS_OP_T;
            JIT_REQUIRE("transposed descriptor",
                        hipblasLtMatmulDescCreate(&outside, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                            == HIPBLAS_STATUS_SUCCESS);
            JIT_REQUIRE("transpose A",
                        hipblasLtMatmulDescSetAttribute(
                            outside, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose))
                            == HIPBLAS_STATUS_SUCCESS);
            Layouts    nn(M, N, kWide, true);
            const auto cT   = queryC(handle, outside, pref, nn, 4);
            const auto cppT = queryCpp(handle,
                                       outside,
                                       HIPBLAS_OP_T,
                                       HIPBLAS_OP_N,
                                       nn,
                                       A.pointer,
                                       B.pointer,
                                       C.pointer,
                                       D.pointer,
                                       4);
            if(mode == "forced")
            {
                JIT_REQUIRE("forced mode returned a solution the kernel does not solve",
                            cT.status == HIPBLAS_STATUS_SUCCESS && cT.results.empty());
                JIT_REQUIRE("forced mode C++ returned a solution the kernel does not solve",
                            cppT.status == HIPBLAS_STATUS_SUCCESS && cppT.results.empty());
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
                    const Layouts layouts(size[0], size[1], size[2]);
                    std::vector<hipblasLtMatmulHeuristicResult_t> results(8);
                    int                                           count = 0;
                    const auto status = hipblasLtMatmulAlgoGetHeuristic(handle,
                                                                        desc,
                                                                        layouts.a,
                                                                        layouts.b,
                                                                        layouts.c,
                                                                        layouts.d,
                                                                        pref,
                                                                        8,
                                                                        results.data(),
                                                                        &count);
                    results.resize(count);
                    JIT_REQUIRE("Equality-size heuristic", status == HIPBLAS_STATUS_SUCCESS);
                    Listed listed{status, std::move(results)};
                    requireProviderOrder(listed, 8);
                    requireUniqueKernels(handle, listed);
                    break;
                }
            }
        }

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
