// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "test_helpers.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

// Heuristic queries under HIPBLASLT_JIT. The mode is a process environment
// variable, so each CTest entry runs one mode. The test lists the device's
// plain-pair bundle in HIPBLASLT_JIT_TEST_REPLAY before the first query.
namespace
{
    using hipblaslt_jit_test::require;

    void hip(hipError_t status, const char* expression)
    {
        require(status == hipSuccess, std::string(expression) + ": " + hipGetErrorString(status));
    }
#define HIP(expression) hip((expression), #expression)

    constexpr int M = 256, N = 128;
    constexpr uint64_t workspaceLimit = 64ull << 20;

    struct Device
    {
        void* pointer{};
        explicit Device(size_t bytes)
        {
            HIP(hipMalloc(&pointer, bytes));
        }
        ~Device()
        {
            static_cast<void>(hipFree(pointer));
        }
        Device(const Device&)            = delete;
        Device& operator=(const Device&) = delete;
    };

    struct Layouts
    {
        hipblasLtMatrixLayout_t a{}, b{}, c{}, d{};
        Layouts(int rowsA, int colsA, int ldA, int rowsB, int colsB, int ldB)
        {
            require(hipblasLtMatrixLayoutCreate(&a, HIP_R_16F, rowsA, colsA, ldA)
                        == HIPBLAS_STATUS_SUCCESS,
                    "layout A");
            require(hipblasLtMatrixLayoutCreate(&b, HIP_R_16F, rowsB, colsB, ldB)
                        == HIPBLAS_STATUS_SUCCESS,
                    "layout B");
            require(hipblasLtMatrixLayoutCreate(&c, HIP_R_16F, M, N, M) == HIPBLAS_STATUS_SUCCESS,
                    "layout C");
            require(hipblasLtMatrixLayoutCreate(&d, HIP_R_16F, M, N, M) == HIPBLAS_STATUS_SUCCESS,
                    "layout D");
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

    bool endsWith(const std::string& text, const std::string& suffix)
    {
        return text.size() >= suffix.size()
               && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
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
        require(count >= 0 && count <= requested, "C heuristic returned a count outside the request");
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
        require(gemm.setProblem(desc, &alpha, A, layouts.a, B, layouts.b, &beta, C, layouts.c, D,
                                layouts.d)
                    == HIPBLAS_STATUS_SUCCESS,
                "Gemm::setProblem");
        hipblaslt_ext::GemmPreference pref;
        pref.setMaxWorkspaceBytes(workspaceLimit);
        Listed listed;
        listed.status = gemm.algoGetHeuristic(requested, pref, listed.results);
        require(static_cast<int>(listed.results.size()) <= requested,
                "C++ heuristic returned more results than requested");
        return listed;
    }

    void requireNoJit(const std::string& label, const Listed& listed)
    {
        for(const auto& result : listed.results)
            require(!isJit(result.algo), label + " returned a JIT algorithm");
        if(listed.results.empty())
            require(listed.status == HIPBLAS_STATUS_INVALID_VALUE,
                    label + " with no results returned status " + std::to_string(listed.status));
        else
            require(listed.status == HIPBLAS_STATUS_SUCCESS,
                    label + " returned status " + std::to_string(listed.status));
    }

    void requireOnlyJit(const std::string& label, const Listed& listed, const std::string& suffix,
                        hipblasLtHandle_t handle)
    {
        require(listed.status == HIPBLAS_STATUS_SUCCESS,
                label + " returned status " + std::to_string(listed.status));
        require(!listed.results.empty(), label + " returned no algorithm");
        for(const auto& result : listed.results)
        {
            auto algo = result.algo;
            require(isJit(algo), label + " returned an algorithm that is not JIT");
            const auto name = hipblaslt_ext::getSolutionNameFromAlgo(handle, algo);
            require(endsWith(name, suffix),
                    label + " selected '" + name + "', not a solution ending in " + suffix);
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
            require(names.insert(name).second, "repeated kernel " + name);
        }
    }

    // Equality results, then JIT, then the other providers. A full list must
    // end with a non-JIT result; a short list may end on JIT.
    void requireProviderOrder(const Listed& listed, int requested)
    {
        require(!listed.results.empty(), "an Equality size returned no algorithm");
        require(!isJit(listed.results.front().algo),
                "an Equality size did not start with an Equality result");
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
                require(false, "a JIT result followed the other providers");
            if(jit)
                sawJit = true;
        }
        require(sawJit, "an Equality size returned no JIT result after the Equality results");
        if(static_cast<int>(listed.results.size()) == requested)
            require(!isJit(listed.results.back().algo),
                    "a full Equality-size list did not end with another provider");
    }

    std::vector<__half> fill(size_t count, int a, int b, int mod, float scale)
    {
        std::vector<__half> values(count);
        for(size_t i = 0; i < count; ++i)
            values[i] = __float2half(float(int(i * a + i / 7 * b) % mod - mod / 2) * scale);
        return values;
    }

    void verify(int K, const std::vector<__half>& a, const std::vector<__half>& b,
                const std::vector<__half>& c, const void* d, float alpha, float beta)
    {
        std::vector<__half> out(M * N);
        HIP(hipMemcpy(out.data(), d, out.size() * sizeof(__half), hipMemcpyDeviceToHost));
        for(int col = 0; col < N; ++col)
            for(int row = 0; row < M; ++row)
            {
                float sum = 0;
                for(int k = 0; k < K; ++k)
                    sum += __half2float(a[row + k * M]) * __half2float(b[k + col * K]);
                const auto i        = row + col * M;
                const auto expected = __half2float(
                    __float2half(alpha * sum + beta * __half2float(c[i])));
                const auto actual = __half2float(out[i]);
                require(std::isfinite(actual)
                            && std::abs(actual - expected) <= 0.0005f + 0.001f * std::abs(expected),
                        "D[" + std::to_string(i) + "] is " + std::to_string(actual)
                            + ", expected " + std::to_string(expected));
            }
    }

    void runFirst(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc, hipStream_t stream, int K,
                  const hipblasLtMatmulHeuristicResult_t& result)
    {
        const auto hostA = fill(size_t(M) * K, 3, 5, 13, 1 / 8.0f);
        const auto hostB = fill(size_t(K) * N, 7, 2, 11, 1 / 8.0f);
        const auto hostC = fill(size_t(M) * N, 1, 3, 7, 1 / 4.0f);
        Device     A(hostA.size() * 2), B(hostB.size() * 2), C(hostC.size() * 2), D(M * N * 2);
        HIP(hipMemcpy(A.pointer, hostA.data(), hostA.size() * 2, hipMemcpyHostToDevice));
        HIP(hipMemcpy(B.pointer, hostB.data(), hostB.size() * 2, hipMemcpyHostToDevice));
        HIP(hipMemcpy(C.pointer, hostC.data(), hostC.size() * 2, hipMemcpyHostToDevice));
        Layouts layouts(M, K, M, K, N, K);
        Device  workspace(result.workspaceSize);
        float   alpha = 1.25f, beta = 0.5f;
        HIP(hipMemset(D.pointer, 0xff, M * N * 2));
        require(hipblasLtMatmul(handle, desc, &alpha, A.pointer, layouts.a, B.pointer, layouts.b,
                                &beta, C.pointer, layouts.c, D.pointer, layouts.d, &result.algo,
                                workspace.pointer, result.workspaceSize, stream)
                    == HIPBLAS_STATUS_SUCCESS,
                "hipblasLtMatmul of the first heuristic result");
        HIP(hipStreamSynchronize(stream));
        verify(K, hostA, hostB, hostC, D.pointer, alpha, beta);
    }

    bool equalitySize(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc, int m, int n, int k)
    {
        hipblasLtMatrixLayout_t a{}, b{}, c{};
        require(hipblasLtMatrixLayoutCreate(&a, HIP_R_16F, m, k, m) == HIPBLAS_STATUS_SUCCESS,
                "equality layout A");
        require(hipblasLtMatrixLayoutCreate(&b, HIP_R_16F, k, n, k) == HIPBLAS_STATUS_SUCCESS,
                "equality layout B");
        require(hipblasLtMatrixLayoutCreate(&c, HIP_R_16F, m, n, m) == HIPBLAS_STATUS_SUCCESS,
                "equality layout C");
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

    // A fresh process querying the same K=512 problem. Prints INDEX <n> and nothing else.
    void reuse(const std::string& root)
    {
        int             device = -1;
        hipDeviceProp_t properties{};
        HIP(hipGetDevice(&device));
        HIP(hipGetDeviceProperties(&properties, device));
        const auto bundles = hipblaslt_jit_test::deviceBundles(root, properties.gcnArchName);
        require(setenv("HIPBLASLT_JIT_TEST_REPLAY", (bundles / "plain-pair").u8string().c_str(), 1)
                    == 0,
                "setenv");

        hipblasLtHandle_t           handle{};
        hipblasLtMatmulDesc_t       desc{};
        hipblasLtMatmulPreference_t pref{};
        require(hipblasLtCreate(&handle) == HIPBLAS_STATUS_SUCCESS, "hipblasLtCreate");
        require(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                    == HIPBLAS_STATUS_SUCCESS,
                "hipblasLtMatmulDescCreate");
        require(hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS, "preference");
        require(hipblasLtMatmulPreferenceSetAttribute(pref,
                                                      HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &workspaceLimit,
                                                      sizeof(workspaceLimit))
                    == HIPBLAS_STATUS_SUCCESS,
                "workspace preference");
        Layouts    layouts(M, 512, M, 512, N, 512);
        const auto listed = queryC(handle, desc, pref, layouts, 1);
        require(listed.status == HIPBLAS_STATUS_SUCCESS && listed.results.size() == 1,
                "second process query");
        auto      algo  = listed.results.front().algo;
        const int index = hipblaslt_ext::getIndexFromAlgo(algo);
        std::cout << "INDEX " << index << '\n' << std::flush;
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtDestroy(handle);
    }

    void requireSameIndexInChild(const std::string& root, int expected)
    {
        int pipes[2];
        require(pipe(pipes) == 0, "pipe");
        const auto child = fork();
        require(child >= 0, "fork");
        if(child == 0)
        {
            if(dup2(pipes[1], STDOUT_FILENO) < 0)
                _exit(127);
            close(pipes[0]);
            close(pipes[1]);
            execl("/proc/self/exe",
                  "hipblaslt-jit-heuristic-test",
                  "reuse",
                  root.c_str(),
                  nullptr);
            _exit(127);
        }
        close(pipes[1]);
        std::string output;
        char        buffer[256];
        ssize_t     count = 0;
        while((count = read(pipes[0], buffer, sizeof(buffer))) > 0)
            output.append(buffer, buffer + count);
        close(pipes[0]);
        int status = 0;
        require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                "The second process failed (wait status " + std::to_string(status) + ")");
        require(output == "INDEX " + std::to_string(expected) + "\n",
                "The second process reported '" + output + "', not index "
                    + std::to_string(expected));
        std::cout << "PASS a second process reused JIT library index " << expected << '\n';
    }

    void test(const std::string& mode, const std::string& root)
    {
        int             device = -1;
        hipDeviceProp_t properties{};
        HIP(hipGetDevice(&device));
        HIP(hipGetDeviceProperties(&properties, device));
        if(mode == "ignored")
            require(setenv("HIPBLASLT_JIT_TEST_REPLAY", "ignored", 1) == 0, "setenv");
        else
        {
            const auto bundles = hipblaslt_jit_test::deviceBundles(root, properties.gcnArchName);
            require(setenv("HIPBLASLT_JIT_TEST_REPLAY", (bundles / "plain-pair").u8string().c_str(), 1)
                        == 0,
                    "setenv");
        }

        hipblasLtHandle_t           handle{};
        hipblasLtMatmulDesc_t       desc{};
        hipblasLtMatmulPreference_t pref{};
        hipStream_t                 stream{};
        require(hipblasLtCreate(&handle) == HIPBLAS_STATUS_SUCCESS, "hipblasLtCreate");
        require(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                    == HIPBLAS_STATUS_SUCCESS,
                "hipblasLtMatmulDescCreate");
        require(hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS, "preference");
        require(hipblasLtMatmulPreferenceSetAttribute(pref,
                                                      HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &workspaceLimit,
                                                      sizeof(workspaceLimit))
                    == HIPBLAS_STATUS_SUCCESS,
                "workspace preference");
        HIP(hipStreamCreate(&stream));

        constexpr int K = 512;
        Device        A(M * K * 2), B(K * N * 2), C(M * N * 2), D(M * N * 2);
        Layouts       layouts(M, K, M, K, N, K);
        const int equalitySizes[][3]
            = {{1024, 4096, 20}, {2048, 128, 16}, {864, 512, 432}, {128, 5120, 1024}};
        bool tuned = false;
        if(mode == "fallback")
        {
            for(const auto& size : equalitySizes)
                tuned = tuned || equalitySize(handle, desc, size[0], size[1], size[2]);
        }

        const auto c512 = queryC(handle, desc, pref, layouts, 4);
        const auto cpp512
            = queryCpp(handle, desc, HIPBLAS_OP_N, HIPBLAS_OP_N, layouts, A.pointer, B.pointer,
                       C.pointer, D.pointer, 4);
        Layouts    k256(M, 256, M, 256, N, 256);
        const auto c256 = queryC(handle, desc, pref, k256, 4);
        const auto cpp256 = queryCpp(handle, desc, HIPBLAS_OP_N, HIPBLAS_OP_N, k256, A.pointer,
                                     B.pointer, C.pointer, D.pointer, 4);

        if(mode == "off" || mode == "ignored")
        {
            checkPair("K=512", c512, cpp512, false, "", handle);
            checkPair("K=256", c256, cpp256, false, "", handle);
        }
        else if(mode == "forced" || !tuned)
        {
            checkPair("K=512", c512, cpp512, true, "_K512_WGM8", handle);
            checkPair("K=256", c256, cpp256, true, "_WGM1", handle);
            runFirst(handle, desc, stream, K, c512.results.front());
            std::cout << "PASS K=512 first result\n";
            auto      algo  = c512.results.front().algo;
            const int index = hipblaslt_ext::getIndexFromAlgo(algo);
            requireSameIndexInChild(root, index);
        }
        else
        {
            require(c512.status == HIPBLAS_STATUS_SUCCESS && !c512.results.empty(), "K=512");
            require(isJit(c512.results.front().algo),
                    "an untuned size with library results did not start with JIT");
            requireUniqueKernels(handle, c512);
            requireUniqueKernels(handle, cpp512);
            runFirst(handle, desc, stream, K, c512.results.front());
            std::cout << "PASS K=512 first result\n";
        }

        if(mode == "forced" || mode == "fallback")
        {
            hipblasLtMatmulDesc_t    tn{};
            const hipblasOperation_t transpose = HIPBLAS_OP_T;
            require(hipblasLtMatmulDescCreate(&tn, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                        == HIPBLAS_STATUS_SUCCESS,
                    "transposed descriptor");
            require(hipblasLtMatmulDescSetAttribute(tn, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose,
                                                    sizeof(transpose))
                        == HIPBLAS_STATUS_SUCCESS,
                    "transpose A");
            Layouts    transposed(K, M, K, K, N, K);
            const auto cT = queryC(handle, tn, pref, transposed, 4);
            const auto cppT = queryCpp(handle, tn, HIPBLAS_OP_T, HIPBLAS_OP_N, transposed, A.pointer,
                                       B.pointer, C.pointer, D.pointer, 4);
            if(mode == "forced")
            {
                require(cT.status == HIPBLAS_STATUS_SUCCESS && cT.results.empty(),
                        "forced mode returned a solution for transposed A");
                require(cppT.status == HIPBLAS_STATUS_SUCCESS && cppT.results.empty(),
                        "forced mode C++ returned a solution for transposed A");
                std::cout << "PASS transposed A returns no solution\n";
            }
            hipblasLtMatmulDescDestroy(tn);
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
                    require(hipblasLtMatrixLayoutCreate(&a, HIP_R_16F, size[0], size[2], size[0])
                                == HIPBLAS_STATUS_SUCCESS,
                            "order layout A");
                    require(hipblasLtMatrixLayoutCreate(&b, HIP_R_16F, size[2], size[1], size[2])
                                == HIPBLAS_STATUS_SUCCESS,
                            "order layout B");
                    require(hipblasLtMatrixLayoutCreate(&c, HIP_R_16F, size[0], size[1], size[0])
                                == HIPBLAS_STATUS_SUCCESS,
                            "order layout C");
                    require(hipblasLtMatrixLayoutCreate(&d, HIP_R_16F, size[0], size[1], size[0])
                                == HIPBLAS_STATUS_SUCCESS,
                            "order layout D");
                    std::vector<hipblasLtMatmulHeuristicResult_t> results(8);
                    int                                           count = 0;
                    const auto                                    status
                        = hipblasLtMatmulAlgoGetHeuristic(handle, desc, a, b, c, d, pref, 8,
                                                          results.data(), &count);
                    results.resize(count);
                    require(status == HIPBLAS_STATUS_SUCCESS, "Equality-size heuristic");
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

int main(int argc, char** argv)
{
    if(argc < 2 || (std::string(argv[1]) != "ignored" && argc != 3))
    {
        std::cerr << "Usage: " << argv[0] << " off|fallback|forced|ignored|reuse [BUNDLES]\n";
        return 2;
    }
    const std::string mode = argv[1];
    if(mode != "off" && mode != "fallback" && mode != "forced" && mode != "ignored"
       && mode != "reuse")
    {
        std::cerr << "Unknown mode " << mode << '\n';
        return 2;
    }
    try
    {
        if(mode == "reuse")
            reuse(argv[2]);
        else
            test(mode, argc == 3 ? argv[2] : "");
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
