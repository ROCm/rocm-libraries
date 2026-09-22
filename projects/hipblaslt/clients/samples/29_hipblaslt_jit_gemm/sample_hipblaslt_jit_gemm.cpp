// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-gemm.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using hipblaslt_ext::experimental::GenerateOptions;
using hipblaslt_ext::experimental::JitGemm;

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
    void check(hipblasStatus_t status, const JitGemm& gemm, const char* what)
    {
        if(status != HIPBLAS_STATUS_SUCCESS)
            throw std::runtime_error(std::string(what) + ": " + gemm.lastError());
    }
    void require(bool condition, const char* what)
    {
        if(!condition)
            throw std::runtime_error(what);
    }

    // The fixture supports this non-square NN FP16/HPA shape. All input fractions
    // are binary-exact; the independent CPU oracle accumulates in FP32.
    int         M = 256, N = 128, K = 128;
    bool        transposeB             = false;
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
                        sum += __half2float(hostA[row + k * M])
                               * __half2float(hostB[transposeB ? col + k * N : k + col * K]);
                    expected[i]
                        = __half2float(__float2half(alpha * sum + beta * __half2float(hostC[i])));
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
        void bind(JitGemm& gemm)
        {
            check(gemm.setProblem(
                      desc, &alpha, a, aLayout, b, bLayout, &beta, c, cLayout, d, dLayout),
                  gemm,
                  "Bind problem");
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
    void runNormal(Problem& p, const GenerateOptions& options)
    {
        using namespace hipblaslt_ext;
        experimental::JitGemmInfo        info;
        hipblasLtMatmulHeuristicResult_t selected;
        auto                             select = [&](const GenerateOptions& generation) {
            const auto status = experimental::getJitGemmAlgo(p.handle,
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
                                                             generation,
                                                             std::numeric_limits<size_t>::max(),
                                                             selected,
                                                             info);
            require(status == HIPBLAS_STATUS_SUCCESS, ("JIT selection: " + info.error).c_str());
        };
        select(options);
        std::cout << "Normal API manifest: " << info.manifestPath << '\n';
        hipblasLtMatmulAlgo_t algo;
        std::memcpy(&algo, &selected.algo, sizeof(algo));
        require(getIndexFromAlgo(algo) == -1, "JIT algorithm exposed a prebuilt index");
        require(getKernelNameFromAlgo(p.handle, algo) == info.kernelName,
                "JIT algorithm kernel name mismatch");
        require(!getSolutionNameFromAlgo(p.handle, algo).empty(), "JIT solution name missing");
        const auto workspaceBytes = selected.workspaceSize;
        if(workspaceBytes)
            check(hipMalloc(&p.workspace, workspaceBytes), "Allocate normal workspace");
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
                check(status, "Normal workspace fallback");
                p.verify("Normal workspace fallback");
            }
            else
                require(status != HIPBLAS_STATUS_SUCCESS,
                        "Normal C accepted insufficient workspace");
        }
        for(int run = 0; run < 2; ++run)
        {
            if(run)
                p.changeInputs();
            p.reset();
            check(runC(algo, workspaceBytes), "Normal C matmul");
            p.verify(run ? "Normal C copied algorithm repeat" : "Normal C copied algorithm");
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
        check(gemm.isAlgoSupported(algo, required), "Normal extension support");
        require(required == workspaceBytes, "Normal workspace queries disagree");
        require(gemm.isAlgoSupported(invalid, required) != HIPBLAS_STATUS_SUCCESS,
                "Unknown JIT algorithm passed extension support");
        gemm.setMaxWorkspaceBytes(workspaceBytes);
        check(gemm.initialize(algo, p.workspace, false, p.stream), "Normal extension initialize");
        require(!gemm.getKernelName().empty() && !gemm.getSolutionName().empty(),
                "Normal extension lost JIT names");
        for(int run = 0; run < 2; ++run)
        {
            if(run)
                p.changeInputs();
            p.reset();
            check(gemm.run(p.stream), "Normal extension run");
            p.verify(run ? "Normal extension repeat" : "Normal extension");
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
                    "Normal C accepted a missing later helper");
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
                << "Normal C/extension missing-helper preflight left D/workspace unchanged PASS\n";
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
                     "[--m M --n N --k K --trans-b N|T] [--normal-api both] [--expect-kernels N "
                     "--expect-min-gsu N] "
                     "[--expect-configured-gsu N --expect-accumulation MODE --min-workspace BYTES] "
                     "[--expect-prepare-failure|--expect-initialize-failure|--expect-unsupported "
                     "ERROR_SUBSTRING]\n";
        return 2;
    }
    try
    {
        GenerateOptions options;
        options.pythonExecutable       = argv[1];
        options.tensileSourceDirectory = argv[2];
        options.pythonPath             = argv[3];
        options.configPath             = std::string(argv[4]) == "-" ? "" : argv[4];
        options.outputPath             = argv[5];
        options.architecture           = argv[6];
        options.cxxCompiler            = argv[7];
        std::string failureMode, failureDiagnostic, expectedAccumulation;
        bool        normalApi       = false;
        int         expectedKernels = -1, expectedMinGsu = 0, expectedConfiguredGsu = -2;
        size_t      minimumWorkspace = 0;
        auto        integer          = [](const std::string& value) {
            size_t consumed = 0;
            int    result   = std::stoi(value, &consumed);
            require(consumed == value.size(), "Expected integer option value");
            return result;
        };
        for(int i = 8; i < argc; i += 2)
        {
            require(i + 1 < argc, "Option requires a value");
            std::string key(argv[i]), value(argv[i + 1]);
            if(key == "--normal-api")
            {
                require(value == "both", "--normal-api requires both");
                normalApi = true;
            }
            else if(key == "--workspace-fallback")
                allowWorkspaceFallback = integer(value) != 0;
            else if(key == "--second-yaml")
                secondConfig = value;
            else if(key == "--second-python")
                secondPython = value;
            else if(key == "--expect-helper-failure")
                expectHelperFailure = integer(value) != 0;
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
            else if(key == "--expect-kernels")
                expectedKernels = integer(value);
            else if(key == "--expect-min-gsu")
                expectedMinGsu = integer(value);
            else if(key == "--expect-configured-gsu")
                expectedConfiguredGsu = integer(value);
            else if(key == "--expect-accumulation")
                expectedAccumulation = value;
            else if(key == "--min-workspace")
                minimumWorkspace = std::stoull(value);
            else if(key == "--expect-prepare-failure" || key == "--expect-initialize-failure"
                    || key == "--expect-unsupported")
            {
                failureMode       = key;
                failureDiagnostic = value;
            }
            else
                throw std::runtime_error("Unknown option: " + key);
        }
        require(M > 0 && N > 0 && K > 0 && M <= 1024 && N <= 1024 && K <= 8192,
                "Sample dimensions must satisfy 0<M,N<=1024 and 0<K<=8192");
        Problem problem;
        problem.create();
        if(normalApi)
        {
            require(failureMode.empty(), "Normal API mode cannot use standalone failure options");
            runNormal(problem, options);
            return 0;
        }
        if(!failureMode.empty())
        {
            if(failureMode == "--expect-unsupported")
            {
                hipblasOperation_t transpose = HIPBLAS_OP_T;
                check(
                    hipblasLtMatmulDescSetAttribute(
                        problem.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)),
                    "Set incompatible transpose");
                check(hipblasLtMatrixLayoutDestroy(problem.aLayout), "Replace A layout");
                problem.aLayout = nullptr;
                check(hipblasLtMatrixLayoutCreate(&problem.aLayout, HIP_R_16F, K, M, K),
                      "Create transposed A layout");
            }
            {
                JitGemm failure(problem.handle);
                problem.bind(failure);
                size_t unused = 0;
                if(failureMode == "--expect-initialize-failure")
                {
                    check(failure.prepare(options, unused),
                          failure,
                          "Prepare for initialization failure");
                    if(unused)
                        check(hipMalloc(&problem.workspace, unused),
                              "Allocate negative-test workspace");
                    require(failure.initialize(problem.workspace, unused, problem.stream)
                                != HIPBLAS_STATUS_SUCCESS,
                            "Invalid helper symbol unexpectedly accepted");
                }
                else
                    require(failure.prepare(options, unused) != HIPBLAS_STATUS_SUCCESS,
                            "Invalid bundle/problem unexpectedly accepted");
                require(failure.lastError().find(failureDiagnostic) != std::string::npos,
                        ("Unexpected preparation error: " + failure.lastError()).c_str());
                std::cout << "Expected pre-launch failure PASS: " << failure.lastError() << '\n';
                if(failureMode != "--expect-initialize-failure")
                    require(failure.initialize(nullptr, 0, problem.stream)
                                != HIPBLAS_STATUS_SUCCESS,
                            "Failed preparation allowed initialization");
                require(failure.run(problem.stream) != HIPBLAS_STATUS_SUCCESS,
                        "Failed preparation allowed launch");
            }
            problem.close();
            return 0;
        }
        {
            JitGemm gemm(problem.handle);
            problem.bind(gemm);
            size_t workspaceBytes = 0;
            check(gemm.prepare(options, workspaceBytes), gemm, "Generate and prepare");
            std::cout << "Manifest: " << gemm.manifestPath() << "\nKernel: " << gemm.kernelName()
                      << "\nWorkspace bytes: " << workspaceBytes << '\n';
            require(workspaceBytes >= minimumWorkspace,
                    "Required workspace below test expectation");
            if(workspaceBytes)
            {
                require(gemm.initialize(nullptr, workspaceBytes - 1, problem.stream)
                            != HIPBLAS_STATUS_SUCCESS,
                        "Insufficient workspace unexpectedly accepted");
                check(hipMalloc(&problem.workspace, workspaceBytes), "Allocate workspace");
            }
            else
                std::cout << "Workspace insufficiency case skipped: solution requires zero bytes\n";
            check(gemm.initialize(problem.workspace, workspaceBytes, problem.stream),
                  gemm,
                  "Initialize");
            const auto& dispatch = gemm.dispatchInfo();
            std::cout << "Dispatch: configured GSU=" << dispatch.configuredGlobalSplitU
                      << ", resolved GSU=" << dispatch.globalSplitU
                      << ", accumulation=" << dispatch.accumulation
                      << ", invocations=" << dispatch.kernelNames.size() << '\n';
            for(size_t i = 0; i < dispatch.kernelNames.size(); ++i)
                std::cout << "Invocation " << i << ": " << dispatch.kernelNames[i] << '\n';
            require(expectedKernels < 0 || dispatch.kernelNames.size() == size_t(expectedKernels),
                    "Unexpected invocation count");
            require(dispatch.globalSplitU >= size_t(expectedMinGsu),
                    "Resolved GSU below test expectation");
            require(expectedConfiguredGsu == -2
                        || dispatch.configuredGlobalSplitU == expectedConfiguredGsu,
                    "Unexpected configured GSU");
            require(expectedAccumulation.empty() || dispatch.accumulation == expectedAccumulation,
                    "Unexpected accumulation strategy");
            if(workspaceBytes)
            {
                require(gemm.initialize(problem.workspace, workspaceBytes - 1, problem.stream)
                            != HIPBLAS_STATUS_SUCCESS,
                        "Insufficient workspace rebind unexpectedly accepted");
                require(gemm.run(problem.stream) != HIPBLAS_STATUS_SUCCESS,
                        "Insufficient workspace rebind left stale arguments runnable");
                check(gemm.initialize(problem.workspace, workspaceBytes, problem.stream),
                      gemm,
                      "Reinitialize with sufficient workspace");
                std::cout << "Nonzero workspace insufficiency and readiness invalidation PASS\n";
            }
            require(gemm.run(nullptr) != HIPBLAS_STATUS_SUCCESS,
                    "Wrong stream unexpectedly accepted");
            for(int run = 0; run < 2; ++run)
            {
                problem.reset();
                check(gemm.run(problem.stream), gemm, "Run");
                problem.verify(run == 0 ? "First run" : "Repeated run");
            }
            require(gemm.generationCount() == 1, "Repeated run regenerated the kernel");
            std::cout << "Generation count after two runs: " << gemm.generationCount() << '\n';
            // A failed reinitialization must invalidate the previous packed arguments.
            check(hipStreamBeginCapture(problem.stream, hipStreamCaptureModeGlobal),
                  "Begin capture");
            require(gemm.initialize(problem.workspace, workspaceBytes, problem.stream)
                        != HIPBLAS_STATUS_SUCCESS,
                    "Captured stream unexpectedly accepted");
            hipGraph_t graph = nullptr;
            check(hipStreamEndCapture(problem.stream, &graph), "End capture");
            if(graph)
                check(hipGraphDestroy(graph), "Destroy empty capture graph");
            require(gemm.run(problem.stream) != HIPBLAS_STATUS_SUCCESS,
                    "Failed reinitialization left stale arguments runnable");
            check(gemm.initialize(problem.workspace, workspaceBytes, problem.stream),
                  gemm,
                  "Reinitialize after rejected capture");

            int deviceCount = 0, originalDevice = 0;
            check(hipGetDeviceCount(&deviceCount), "Count devices");
            check(hipGetDevice(&originalDevice), "Get original device");
            if(deviceCount > 1)
            {
                const int otherDevice = (originalDevice + 1) % deviceCount;
                check(hipSetDevice(otherDevice), "Select other device");
                hipStream_t foreignStream = nullptr;
                check(hipStreamCreate(&foreignStream), "Create foreign stream");
                check(hipSetDevice(originalDevice), "Restore original device");
                require(gemm.initialize(problem.workspace, workspaceBytes, foreignStream)
                            != HIPBLAS_STATUS_SUCCESS,
                        "Foreign-device stream unexpectedly accepted");
                require(gemm.run(problem.stream) != HIPBLAS_STATUS_SUCCESS,
                        "Foreign-stream rejection left stale arguments runnable");
                check(hipSetDevice(otherDevice), "Select foreign stream device");
                check(hipStreamDestroy(foreignStream), "Destroy foreign stream");
                check(hipSetDevice(originalDevice), "Restore original device");
                check(gemm.initialize(problem.workspace, workspaceBytes, problem.stream),
                      gemm,
                      "Reinitialize after rejected foreign stream");
                std::cout << "Foreign-device stream rejection PASS\n";
            }
            else
                std::cout << "Foreign-device stream test skipped: one visible device\n";
            std::cout << "Capture rejection and failed-reinitialization invalidation PASS\n";

            // Two streams and distinct data/workspace share one handle. Each JIT
            // owner must keep its own modules and MBSK synchronization storage.
            Problem other;
            other.seed = 1;
            other.create(problem.handle);
            {
                JitGemm second(problem.handle);
                other.bind(second);
                auto secondOptions = options;
                secondOptions.outputPath += "-second";
                size_t secondWorkspace = 0;
                check(second.prepare(secondOptions, secondWorkspace),
                      second,
                      "Prepare second private adapter");
                require(secondWorkspace == workspaceBytes
                            && second.kernelName() == gemm.kernelName(),
                        "Repeated generation changed fixture identity");
                if(secondWorkspace)
                    check(hipMalloc(&other.workspace, secondWorkspace),
                          "Allocate second workspace");
                check(second.initialize(other.workspace, secondWorkspace, other.stream),
                      second,
                      "Initialize second adapter");
                require(second.dispatchInfo().kernelNames == gemm.dispatchInfo().kernelNames,
                        "Repeated solution changed invocation sequence");
                problem.reset();
                other.reset();
                check(gemm.run(problem.stream), gemm, "Run first owner concurrently");
                check(second.run(other.stream),
                      second,
                      "Run second owner before immediate destruction");
            }
            other.verify("Second stream / immediate destruction");
            problem.verify("First stream / shared handle isolation");
            other.close();
            problem.reset();
            check(gemm.run(problem.stream), gemm, "Run first adapter after second destroyed");
            problem.verify("First adapter remains valid");

            // These calls must fail before a runnable owner is produced.
            JitGemm failure(problem.handle);
            problem.bind(failure);
            size_t unused = 0;
            require(failure.prepare(options, unused) != HIPBLAS_STATUS_SUCCESS,
                    "Existing output unexpectedly accepted");
            require(failure.run(problem.stream) != HIPBLAS_STATUS_SUCCESS,
                    "Failed owner became runnable");
            auto bad = options;
            bad.outputPath += "-invalid-yaml";
            bad.configPath = bad.outputPath + ".yaml";
            {
                std::ofstream yaml(bad.configPath);
                yaml << "BenchmarkProblems: [\n";
                require(bool(yaml), "Write invalid YAML");
            }
            require(failure.prepare(bad, unused) != HIPBLAS_STATUS_SUCCESS,
                    "Invalid YAML unexpectedly accepted");
            require(!std::filesystem::exists(std::filesystem::path(bad.outputPath) / "bundle"),
                    "Invalid YAML published a bundle");
            bad = options;
            bad.outputPath += "-invalid-compiler";
            bad.cxxCompiler = "/nonexistent/amdclang++";
            require(failure.prepare(bad, unused) != HIPBLAS_STATUS_SUCCESS,
                    "Missing compiler unexpectedly accepted");
            require(!std::filesystem::exists(std::filesystem::path(bad.outputPath) / "bundle"),
                    "Missing compiler published a bundle");
            std::cout
                << "Existing output, invalid YAML, unavailable compiler, unprepared run PASS\n";
        }
        problem.close();
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
