// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt-jit-tensilelite.hpp>
#include <hipblaslt/hipblaslt-jit.hpp>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using TensileLiteOptions = hipblaslt_ext::experimental::jit::tensilelite::Options;

namespace
{
    void check(hipError_t status, const char* what)
    {
        if(status != hipSuccess)
            throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
    }
    void check(hipblasStatus_t status, const char* what)
    {
        if(status != HIPBLAS_STATUS_SUCCESS)
            throw std::runtime_error(std::string(what) + ": status " + std::to_string(status));
    }
    // A/B values are small multiples of 1/8 and C values are multiples of 1/4,
    // so FP16 and FP32 represent the inputs without rounding. At the tested
    // K <= 4096, products and sums also fit FP32's 24-bit significand exactly.
    // The CPU reference rounds the final result to FP16 before comparing D.
    int M = 256, N = 128, K = 512;
    struct Problem
    {
        hipblasLtHandle_t       handle  = nullptr;
        hipblasLtMatmulDesc_t   desc    = nullptr;
        hipblasLtMatrixLayout_t aLayout = nullptr, bLayout = nullptr, cLayout = nullptr,
                                dLayout = nullptr;
        hipStream_t         stream      = nullptr;
        __half *            a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr;
        void*               workspace = nullptr;
        float               alpha = 1.25f, beta = 0.5f;
        std::vector<__half> hostA, hostB, hostC, hostD;
        std::vector<float>  expected;

        int  seed = 0;
        void create()
        {
            check(hipblasLtCreate(&handle), "Create handle");
            check(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  "Create matmul descriptor");
            check(hipblasLtMatrixLayoutCreate(&aLayout, HIP_R_16F, M, K, M), "Create A layout");
            check(hipblasLtMatrixLayoutCreate(&bLayout, HIP_R_16F, K, N, K), "Create B layout");
            check(hipblasLtMatrixLayoutCreate(&cLayout, HIP_R_16F, M, N, M), "Create C layout");
            check(hipblasLtMatrixLayoutCreate(&dLayout, HIP_R_16F, M, N, M), "Create D layout");
            check(hipStreamCreate(&stream), "Create stream");
            hostA.resize(M * K);
            hostB.resize(K * N);
            hostC.resize(M * N);
            hostD.resize(M * N);
            expected.resize(M * N);
            for(int k = 0; k < K; ++k)
                for(int row = 0; row < M; ++row)
                    hostA[row + k * M] = __float2half(((row * 3 + k * 5 + seed) % 13 - 6) / 8.0f);
            for(int col = 0; col < N; ++col)
                for(int k = 0; k < K; ++k)
                    hostB[k + col * K] = __float2half(((k * 7 + col * 2 + seed) % 11 - 5) / 8.0f);
            for(int col = 0; col < N; ++col)
                for(int row = 0; row < M; ++row)
                {
                    const int i = row + col * M;
                    hostC[i]    = __float2half(((row + col * 3 + seed) % 7 - 3) / 4.0f);
                }
            updateReference();
            check(hipMalloc(&a, hostA.size() * sizeof(__half)), "Allocate A");
            check(hipMalloc(&b, hostB.size() * sizeof(__half)), "Allocate B");
            check(hipMalloc(&c, hostC.size() * sizeof(__half)), "Allocate C");
            check(hipMalloc(&d, hostD.size() * sizeof(__half)), "Allocate D");
            check(hipMemcpy(a, hostA.data(), hostA.size() * sizeof(__half), hipMemcpyHostToDevice),
                  "Copy A");
            check(hipMemcpy(b, hostB.data(), hostB.size() * sizeof(__half), hipMemcpyHostToDevice),
                  "Copy B");
            check(hipMemcpy(c, hostC.data(), hostC.size() * sizeof(__half), hipMemcpyHostToDevice),
                  "Copy C");
        }
        void updateReference()
        {
            for(int col = 0; col < N; ++col)
                for(int row = 0; row < M; ++row)
                {
                    const int i   = row + col * M;
                    float     sum = 0;
                    for(int k = 0; k < K; ++k)
                        sum += __half2float(hostA[row + k * M]) * __half2float(hostB[k + col * K]);
                    const float result = alpha * sum + beta * __half2float(hostC[i]);
                    expected[i]        = __half2float(__float2half(result));
                }
        }
        void reset()
        {
            // 0x7e00 is a quiet NaN in FP16; unchanged D always fails validation.
            std::fill(hostD.begin(), hostD.end(), __float2half(NAN));
            check(
                hipMemcpyAsync(
                    d, hostD.data(), hostD.size() * sizeof(__half), hipMemcpyHostToDevice, stream),
                "Reset D");
        }
        void verify(const char* label)
        {
            check(
                hipMemcpyAsync(
                    hostD.data(), d, hostD.size() * sizeof(__half), hipMemcpyDeviceToHost, stream),
                "Copy D");
            check(hipStreamSynchronize(stream), "Synchronize JIT GEMM");
            float maxError = 0;
            for(size_t i = 0; i < hostD.size(); ++i)
            {
                const float actual = __half2float(hostD[i]);
                const float error  = std::abs(actual - expected[i]);
                // Half an FP16 ULP near unit scale plus 0.1% relative tolerance.
                if(!std::isfinite(actual) || error > 0.0005f + 0.001f * std::abs(expected[i]))
                    throw std::runtime_error(
                        std::string(label) + ": mismatch at " + std::to_string(i) + ", expected "
                        + std::to_string(expected[i]) + ", actual " + std::to_string(actual));
                maxError = std::max(maxError, error);
            }
            std::cout << label << " PASS: " << hostD.size() << " elements, max error=" << maxError
                      << '\n';
        }
        ~Problem()
        {
            if(stream)
                hipStreamSynchronize(stream);
            hipFree(workspace);
            hipFree(a);
            hipFree(b);
            hipFree(c);
            hipFree(d);
            if(aLayout)
                hipblasLtMatrixLayoutDestroy(aLayout);
            if(bLayout)
                hipblasLtMatrixLayoutDestroy(bLayout);
            if(cLayout)
                hipblasLtMatrixLayoutDestroy(cLayout);
            if(dLayout)
                hipblasLtMatrixLayoutDestroy(dLayout);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
            if(handle)
                hipblasLtDestroy(handle);
            if(stream)
                hipStreamDestroy(stream);
        }
    };
    void runPublicGemm(Problem& p, const TensileLiteOptions& options)
    {
        namespace jit = hipblaslt_ext::experimental::jit;
        jit::Diagnostics diagnostics;
        auto checked = [&](hipblasStatus_t status) { check(status, diagnostics.message.c_str()); };
        // Provider settings are needed only to construct the selected backend.
        jit::Backend backend;
        checked(jit::tensilelite::createBackend(options, backend, diagnostics));
        jit::Request request;
        checked(jit::makeGemmRequest(p.handle,
                                     p.desc,
                                     &p.alpha,
                                     p.a,
                                     p.aLayout,
                                     p.b,
                                     p.bLayout,
                                     &p.beta,
                                     p.c,
                                     p.cLayout,
                                     p.d,
                                     p.dLayout,
                                     request,
                                     diagnostics));
        int device;
        check(hipGetDevice(&device), "Get current device");
        jit::Solution solution;
        checked(jit::getJitAlgo(
            device, request, backend, std::numeric_limits<size_t>::max(), solution, diagnostics));
        hipblasLtMatmulHeuristicResult_t selected{};
        checked(jit::getGemmAlgo(solution, selected, diagnostics));
        if(selected.workspaceSize)
            check(hipMalloc(&p.workspace, selected.workspaceSize), "Allocate workspace");
        p.reset();
        check(hipblasLtMatmul(p.handle,
                              p.desc,
                              &p.alpha,
                              p.a,
                              p.aLayout,
                              p.b,
                              p.bLayout,
                              &p.beta,
                              p.c,
                              p.cLayout,
                              p.d,
                              p.dLayout,
                              &selected.algo,
                              p.workspace,
                              selected.workspaceSize,
                              p.stream),
              "C API matmul");
        p.verify("hipblasLtMatmul");
        // Reuse the selected algorithm through the existing C++ execution API.
        hipblaslt_ext::Gemm gemm(p.handle,
                                 p.desc,
                                 &p.alpha,
                                 p.a,
                                 p.aLayout,
                                 p.b,
                                 p.bLayout,
                                 &p.beta,
                                 p.c,
                                 p.cLayout,
                                 p.d,
                                 p.dLayout);
        gemm.setMaxWorkspaceBytes(selected.workspaceSize);
        check(gemm.initialize(selected.algo, p.workspace, false, p.stream), "Initialize C++ Gemm");
        p.reset();
        check(gemm.run(p.stream), "Run C++ Gemm");
        p.verify("hipblaslt_ext::Gemm");
    }
}

int main(int argc, char** argv)
try
{
    if(argc != 8)
    {
        std::cerr << "Usage: " << argv[0]
                  << " PYTHON TENSILE_SOURCE PYTHONPATH YAML FRESH_OUTPUT ARCH COMPILER\n";
        return 2;
    }
    TensileLiteOptions options;
    options.pythonExecutable       = argv[1];
    options.tensileSourceDirectory = argv[2];
    options.pythonPath             = argv[3];
    options.configPath             = argv[4];
    options.outputPath             = argv[5];
    options.architecture           = argv[6];
    options.cxxCompiler            = argv[7];
    Problem problem;
    problem.create();
    runPublicGemm(problem, options);
    return 0;
}
catch(const std::exception& error)
{
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
