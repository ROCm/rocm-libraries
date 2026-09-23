// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt-jit-tensilelite.hpp>
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
    void require(bool condition, const char* what)
    {
        if(!condition)
            throw std::runtime_error(what);
    }

    // A/B values are small multiples of 1/8 and C values are multiples of 1/4,
    // so FP16 and FP32 represent the inputs without rounding. At the tested
    // K <= 4096, products and sums also fit FP32's 24-bit significand exactly.
    // The CPU reference rounds the final result to FP16 before comparing D.
    int         M = 256, N = 128, K = 128;
    bool        transposeB             = false;
    bool        outputAmax             = false;
    bool        allowWorkspaceFallback = false;
    std::string secondConfig, secondPython;
    bool        expectHelperFailure = false;
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
        float*              amax         = nullptr;
        float               expectedAmax = 0;
        std::vector<__half> hostA, hostB, hostC, hostD;
        std::vector<float>  expected;

        bool ownsHandle = true;
        int  seed       = 0;
        void create(hipblasLtHandle_t sharedHandle = nullptr)
        {
            ownsHandle = sharedHandle == nullptr;
            if(ownsHandle)
                check(hipblasLtCreate(&handle), "Create handle");
            else
                handle = sharedHandle;
            check(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  "Create matmul descriptor");
            check(hipblasLtMatrixLayoutCreate(&aLayout, HIP_R_16F, M, K, M), "Create A layout");
            check(hipblasLtMatrixLayoutCreate(&bLayout,
                                              HIP_R_16F,
                                              transposeB ? N : K,
                                              transposeB ? K : N,
                                              transposeB ? N : K),
                  "Create B layout");
            hipblasOperation_t opB = transposeB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
            check(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)),
                  "Set B operation");
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
                    hostB[transposeB ? col + k * N : k + col * K]
                        = __float2half(((k * 7 + col * 2 + seed) % 11 - 5) / 8.0f);
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
            if(outputAmax)
            {
                check(hipMalloc(&amax, sizeof(float)), "Allocate output amax");
                check(hipblasLtMatmulDescSetAttribute(
                          desc, HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER, &amax, sizeof(amax)),
                      "Set output amax pointer");
            }
            check(hipMemcpy(a, hostA.data(), hostA.size() * sizeof(__half), hipMemcpyHostToDevice),
                  "Copy A");
            check(hipMemcpy(b, hostB.data(), hostB.size() * sizeof(__half), hipMemcpyHostToDevice),
                  "Copy B");
            check(hipMemcpy(c, hostC.data(), hostC.size() * sizeof(__half), hipMemcpyHostToDevice),
                  "Copy C");
        }
        void updateReference()
        {
            expectedAmax = 0;
            for(int col = 0; col < N; ++col)
                for(int row = 0; row < M; ++row)
                {
                    const int i   = row + col * M;
                    float     sum = 0;
                    for(int k = 0; k < K; ++k)
                        sum += __half2float(hostA[row + k * M])
                               * __half2float(hostB[transposeB ? col + k * N : k + col * K]);
                    const float result = alpha * sum + beta * __half2float(hostC[i]);
                    expectedAmax       = std::max(expectedAmax, std::abs(result));
                    expected[i]        = __half2float(__float2half(result));
                }
        }
        void changeInputs()
        {
            for(size_t i = 0; i < hostA.size(); i += 3)
                hostA[i] = __float2half(-__half2float(hostA[i]));
            updateReference();
            check(
                hipMemcpyAsync(
                    a, hostA.data(), hostA.size() * sizeof(__half), hipMemcpyHostToDevice, stream),
                "Change A between runs");
        }
        void reset()
        {
            // 0x7e00 is a quiet NaN in FP16; unchanged D always fails validation.
            std::fill(hostD.begin(), hostD.end(), __float2half(NAN));
            check(
                hipMemcpyAsync(
                    d, hostD.data(), hostD.size() * sizeof(__half), hipMemcpyHostToDevice, stream),
                "Reset D");
            if(amax)
                check(hipMemsetAsync(amax, 0xff, sizeof(float), stream), "Poison output amax");
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
            if(amax)
            {
                float actualAmax = NAN;
                check(hipMemcpy(&actualAmax, amax, sizeof(float), hipMemcpyDeviceToHost),
                      "Copy output amax");
                require(std::isfinite(actualAmax)
                            && std::abs(actualAmax - expectedAmax) <= 1e-5f * expectedAmax + 1e-6f,
                        (std::string(label) + ": amax mismatch, expected "
                         + std::to_string(expectedAmax) + ", actual " + std::to_string(actualAmax))
                            .c_str());
                std::cout << label << " amax PASS: " << actualAmax << '\n';
            }
        }
        void close()
        {
            if(stream)
                check(hipStreamSynchronize(stream), "Drain stream");
            if(workspace)
            {
                check(hipFree(workspace), "Free workspace");
                workspace = nullptr;
            }
            if(a)
            {
                check(hipFree(a), "Free A");
                a = nullptr;
            }
            if(b)
            {
                check(hipFree(b), "Free B");
                b = nullptr;
            }
            if(c)
            {
                check(hipFree(c), "Free C");
                c = nullptr;
            }
            if(d)
            {
                check(hipFree(d), "Free D");
                d = nullptr;
            }
            if(amax)
            {
                check(hipFree(amax), "Free output amax");
                amax = nullptr;
            }
            if(aLayout)
            {
                check(hipblasLtMatrixLayoutDestroy(aLayout), "Destroy A layout");
                aLayout = nullptr;
            }
            if(bLayout)
            {
                check(hipblasLtMatrixLayoutDestroy(bLayout), "Destroy B layout");
                bLayout = nullptr;
            }
            if(cLayout)
            {
                check(hipblasLtMatrixLayoutDestroy(cLayout), "Destroy C layout");
                cLayout = nullptr;
            }
            if(dLayout)
            {
                check(hipblasLtMatrixLayoutDestroy(dLayout), "Destroy D layout");
                dLayout = nullptr;
            }
            if(desc)
            {
                check(hipblasLtMatmulDescDestroy(desc), "Destroy descriptor");
                desc = nullptr;
            }
            if(handle)
            {
                if(ownsHandle)
                    check(hipblasLtDestroy(handle), "Destroy handle");
                handle = nullptr;
            }
            if(stream)
            {
                check(hipStreamDestroy(stream), "Destroy stream");
                stream = nullptr;
            }
        }
        ~Problem()
        {
            try
            {
                close();
            }
            catch(const std::exception& e)
            {
                std::cerr << "Cleanup failure: " << e.what() << '\n';
            }
        }
    };
    void runPublicGemm(Problem& p, const TensileLiteOptions& options)
    {
        using namespace hipblaslt_ext;
        experimental::jit::Diagnostics   info;
        hipblasLtMatmulHeuristicResult_t selected;
        auto                             select = [&](const TensileLiteOptions& generation) {
            experimental::jit::Backend backend;
            require(experimental::jit::tensilelite::createBackend(generation, backend, info)
                        == HIPBLAS_STATUS_SUCCESS,
                    info.message.c_str());
            experimental::jit::Request request;
            auto                       status = experimental::jit::makeGemmRequest(p.handle,
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
                                                             info);
            require(status == HIPBLAS_STATUS_SUCCESS, info.message.c_str());
            int device;
            check(hipGetDevice(&device), "Get current device");
            experimental::jit::Solution solution;
            status = experimental::jit::getJitAlgo(
                device, request, backend, std::numeric_limits<size_t>::max(), solution, info);
            require(status == HIPBLAS_STATUS_SUCCESS, info.message.c_str());
            status = experimental::jit::getGemmAlgo(solution, selected, info);
            require(status == HIPBLAS_STATUS_SUCCESS, ("JIT selection: " + info.message).c_str());
        };
        select(options);
        std::cout << "Public GEMM API manifest: " << (options.outputPath + "/bundle/manifest.json")
                  << '\n';
        hipblasLtMatmulAlgo_t algo;
        std::memcpy(&algo, &selected.algo, sizeof(algo));
        require(getIndexFromAlgo(algo) == -1, "JIT algorithm exposed a prebuilt index");
        require(!getKernelNameFromAlgo(p.handle, algo).empty(),
                "JIT algorithm kernel name mismatch");
        require(!getSolutionNameFromAlgo(p.handle, algo).empty(), "JIT solution name missing");
        const auto workspaceBytes = selected.workspaceSize;
        if(workspaceBytes)
            check(hipMalloc(&p.workspace, workspaceBytes), "Allocate GEMM workspace");
        auto runC = [&](const hipblasLtMatmulAlgo_t& chosen, size_t bytes) {
            return hipblasLtMatmul(p.handle,
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
                                   &chosen,
                                   p.workspace,
                                   bytes,
                                   p.stream);
        };
        auto invalid = algo;
        std::memset(invalid.data + 9, 0, 7); // Registry never issues token zero.
        require(runC(invalid, workspaceBytes) != HIPBLAS_STATUS_SUCCESS,
                "Unknown JIT token reached C execution");
        invalid         = algo;
        invalid.data[0] = 1;
        require(runC(invalid, workspaceBytes) != HIPBLAS_STATUS_SUCCESS,
                "Invalid JIT local index reached C execution");
        if(workspaceBytes)
        {
            p.reset();
            const auto status = runC(algo, workspaceBytes - 1);
            if(allowWorkspaceFallback)
            {
                check(status, "GEMM workspace fallback");
                p.verify("GEMM workspace fallback");
            }
            else
                require(status != HIPBLAS_STATUS_SUCCESS, "C API accepted insufficient workspace");
        }
        for(int run = 0; run < 2; ++run)
        {
            if(run)
                p.changeInputs();
            p.reset();
            check(runC(algo, workspaceBytes), "C API matmul");
            p.verify(run ? "C API copied algorithm repeat" : "C API copied algorithm");
        }
        Gemm   gemm(p.handle,
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
        size_t required = 0;
        check(gemm.isAlgoSupported(algo, required), "C++ Gemm support");
        require(required == workspaceBytes, "GEMM workspace queries disagree");
        require(gemm.isAlgoSupported(invalid, required) != HIPBLAS_STATUS_SUCCESS,
                "Unknown JIT algorithm passed extension support");
        gemm.setMaxWorkspaceBytes(workspaceBytes);
        check(gemm.initialize(algo, p.workspace, false, p.stream), "C++ Gemm initialize");
        require(!gemm.getKernelName().empty() && !gemm.getSolutionName().empty(),
                "C++ Gemm lost JIT names");
        for(int run = 0; run < 2; ++run)
        {
            if(run)
                p.changeInputs();
            p.reset();
            check(gemm.run(p.stream), "C++ Gemm run");
            p.verify(run ? "C++ Gemm repeat" : "C++ Gemm");
        }
        // Fresh bundles use the same symbols and basenames. The first retained
        // algorithm must remain runnable after a second private adapter is loaded.
        auto another = options;
        another.outputPath += "-second";
        if(!secondConfig.empty())
            another.configPath = secondConfig;
        if(!secondPython.empty())
            another.pythonExecutable = secondPython;
        select(another);
        require(selected.workspaceSize <= workspaceBytes,
                "Second test recipe needs more workspace than the first");
        require(std::memcmp(algo.data, selected.algo.data, sizeof(algo.data)) != 0,
                "Separate generations reused an opaque token");
        if(expectHelperFailure)
        {
            require(workspaceBytes != 0, "Helper-failure test requires split-K workspace");
            std::vector<unsigned char> workspace(workspaceBytes);
            auto                       unchanged = [&] {
                check(hipMemcpyAsync(p.hostD.data(),
                                     p.d,
                                     p.hostD.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost,
                                     p.stream),
                      "Read D sentinel");
                check(hipMemcpyAsync(workspace.data(),
                                     p.workspace,
                                     workspaceBytes,
                                     hipMemcpyDeviceToHost,
                                     p.stream),
                      "Read workspace sentinel");
                check(hipStreamSynchronize(p.stream), "Check preflight submission boundary");
                require(std::all_of(p.hostD.begin(),
                                    p.hostD.end(),
                                    [](auto value) { return std::isnan(__half2float(value)); }),
                        "Failed helper preflight wrote D");
                require(std::all_of(workspace.begin(),
                                    workspace.end(),
                                    [](auto value) { return value == 0xa5; }),
                        "Failed helper preflight wrote workspace");
            };
            p.reset();
            check(hipMemsetAsync(p.workspace, 0xa5, workspaceBytes, p.stream),
                  "Set workspace sentinel");
            require(runC(selected.algo, workspaceBytes) != HIPBLAS_STATUS_SUCCESS,
                    "C API accepted a missing later helper");
            unchanged();
            const auto previousName = gemm.getKernelName();
            require(gemm.initialize(selected.algo, p.workspace, false, p.stream)
                        != HIPBLAS_STATUS_SUCCESS,
                    "Extension initialize accepted a missing later helper");
            unchanged();
            require(gemm.getKernelName() == previousName,
                    "Failed helper preflight changed the prepared context");
            p.changeInputs();
            p.reset();
            check(gemm.run(p.stream), "Retained extension after helper failure");
            p.verify("Retained extension after helper failure");
            std::cout
                << "C API/extension missing-helper preflight left D/workspace unchanged PASS\n";
            return;
        }
        p.reset();
        check(runC(selected.algo, workspaceBytes), "Second generated algorithm");
        p.verify("Second private bundle");
        p.reset();
        check(runC(algo, workspaceBytes), "Retained first algorithm");
        p.verify("First bundle after second registration");
        check(gemm.initialize(selected.algo, p.workspace, false, p.stream),
              "Reinitialize extension with second solution");
        p.changeInputs();
        p.reset();
        check(gemm.run(p.stream), "Run extension after solution switch");
        p.verify("Extension after solution switch");
        const auto previousName = gemm.getKernelName();
        require(gemm.initialize(invalid, p.workspace, false, p.stream) != HIPBLAS_STATUS_SUCCESS,
                "Invalid extension reinitialization unexpectedly succeeded");
        require(gemm.getKernelName() == previousName,
                "Failed reinitialization changed the prepared context");
        p.reset();
        check(gemm.run(p.stream), "Run extension after rejected reinitialization");
        p.verify("Extension after rejected reinitialization");
        int deviceCount = 0, originalDevice = 0;
        check(hipGetDeviceCount(&deviceCount), "Device count");
        check(hipGetDevice(&originalDevice), "Current device");
        if(deviceCount > 1)
        {
            check(hipSetDevice((originalDevice + 1) % deviceCount), "Switch device");
            const auto status = gemm.isAlgoSupported(algo, required);
            check(hipSetDevice(originalDevice), "Restore device");
            require(status != HIPBLAS_STATUS_SUCCESS, "JIT algorithm accepted the wrong device");
            std::cout << "Wrong-device JIT rejection PASS\n";
        }
        std::cout << "Opaque copy, invalid token/index, names and private bundles PASS\n";
    }

}

int main(int argc, char** argv)
{
    if(argc < 8)
    {
        std::cerr << "Usage: " << argv[0]
                  << " PYTHON TENSILE_SOURCE PYTHONPATH YAML FRESH_OUTPUT ARCH COMPILER "
                     "[--m M --n N --k K --trans-b N|T --amax 0|1]\n";
        return 2;
    }
    try
    {
        TensileLiteOptions options;
        options.pythonExecutable       = argv[1];
        options.tensileSourceDirectory = argv[2];
        options.pythonPath             = argv[3];
        options.configPath             = argv[4];
        options.outputPath             = argv[5];
        options.architecture           = argv[6];
        options.cxxCompiler            = argv[7];
        require(options.configPath != "-", "This sample requires an explicit YAML recipe");
        auto integer = [](const std::string& value) {
            size_t consumed = 0;
            int    result   = std::stoi(value, &consumed);
            require(consumed == value.size(), "Expected integer option value");
            return result;
        };
        for(int i = 8; i < argc; i += 2)
        {
            require(i + 1 < argc, "Option requires a value");
            std::string key(argv[i]), value(argv[i + 1]);
            if(key == "--workspace-fallback")
                allowWorkspaceFallback = integer(value) != 0;
            else if(key == "--second-yaml")
                secondConfig = value;
            else if(key == "--second-python")
                secondPython = value;
            else if(key == "--expect-helper-failure")
                expectHelperFailure = integer(value) != 0;
            else if(key == "--amax")
                outputAmax = integer(value) != 0;
            else if(key == "--trans-b")
            {
                require(value == "N" || value == "T", "--trans-b requires N or T");
                transposeB = value == "T";
            }
            else if(key == "--m")
                M = integer(value);
            else if(key == "--n")
                N = integer(value);
            else if(key == "--k")
                K = integer(value);
            else
                throw std::runtime_error("Unknown option: " + key);
        }
        require(M > 0 && N > 0 && K > 0 && M <= 1024 && N <= 1024 && K <= 8192,
                "Sample dimensions must satisfy 0<M,N<=1024 and 0<K<=8192");
        Problem problem;
        problem.create();
        runPublicGemm(problem, options);
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
