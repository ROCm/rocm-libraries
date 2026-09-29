// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-gemm-internal.hpp"
#include "hipblaslt-jit-mock.hpp"
#include "hipblaslt-jit-tensilelite-artifacts.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace jit  = hipblaslt_ext::experimental::jit;
namespace abi  = hipblaslt_ext::experimental::jit::detail;
namespace mock = hipblaslt_ext::experimental::jit::mock;

namespace
{
    void require(bool condition, const std::string& message)
    {
        if(!condition)
            throw std::runtime_error(message);
    }
    void hip(hipError_t status, const char* expression)
    {
        require(status == hipSuccess, std::string(expression) + ": " + hipGetErrorString(status));
    }
    void blas(hipblasStatus_t status, const char* expression)
    {
        require(status == HIPBLAS_STATUS_SUCCESS,
                std::string(expression) + ": status " + std::to_string(status));
    }
#define HIP(expression) hip((expression), #expression)
#define BLAS(expression) blas((expression), #expression)

    // The splitk-api replay: FP16 NN GEMM with FP32 compute and GlobalSplitU=4.
    constexpr int M = 256, N = 128, K = 512;

    template <class T>
    struct DeviceBuffer
    {
        T*     pointer{};
        size_t count;
        explicit DeviceBuffer(size_t n)
            : count(n)
        {
            HIP(hipMalloc(reinterpret_cast<void**>(&pointer), n * sizeof(T)));
        }
        ~DeviceBuffer()
        {
            if(pointer)
                static_cast<void>(hipFree(pointer));
        }
        DeviceBuffer(const DeviceBuffer&)            = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
        size_t        bytes() const
        {
            return count * sizeof(T);
        }
    };

    struct ProbeRequest final : abi::OperationRequest
    {
        std::string_view kind() const noexcept override
        {
            return "test.mock.probe.v1";
        }
    };

    struct Problem
    {
        static constexpr unsigned char sentinel = 0xa5;
        hipblasLtHandle_t              handle{};
        hipblasLtMatmulDesc_t          desc{};
        hipblasLtMatrixLayout_t        la{}, lb{}, lc{}, ld{};
        hipStream_t                    stream{};
        hipEvent_t                     start{}, stop{};
        // A/B values are small multiples of 1/8 and C values are multiples of 1/4,
        // so FP16 holds them exactly and FP32 sums them exactly at this K.
        std::vector<__half>                          hostA[2], hostB, hostC;
        DeviceBuffer<__half>                         A0{M * K}, A1{M * K}, B{K * N};
        DeviceBuffer<__half>                         C{M * N}, D{M * N};
        std::unique_ptr<DeviceBuffer<unsigned char>> W;

        Problem()
            : hostA{std::vector<__half>(M * K), std::vector<__half>(M * K)}
            , hostB(K * N)
            , hostC(M * N)
        {
            BLAS(hipblasLtCreate(&handle));
            BLAS(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
            BLAS(hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, M, K, M));
            BLAS(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, K, N, K));
            BLAS(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16F, M, N, M));
            BLAS(hipblasLtMatrixLayoutCreate(&ld, HIP_R_16F, M, N, M));
            HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            HIP(hipEventCreate(&start));
            HIP(hipEventCreate(&stop));
            for(int k = 0; k < K; ++k)
                for(int row = 0; row < M; ++row)
                {
                    const auto value = ((row * 3 + k * 5) % 13 - 6) / 8.0f;
                    const auto i     = row + k * M;
                    hostA[0][i]      = __float2half(value);
                    hostA[1][i]      = __float2half(i % 3 ? value : -value);
                }
            for(int col = 0; col < N; ++col)
                for(int k = 0; k < K; ++k)
                    hostB[k + col * K] = __float2half(((k * 7 + col * 2) % 11 - 5) / 8.0f);
            for(int col = 0; col < N; ++col)
                for(int row = 0; row < M; ++row)
                    hostC[row + col * M] = __float2half(((row + col * 3) % 7 - 3) / 4.0f);
            HIP(hipMemcpy(A0.pointer, hostA[0].data(), A0.bytes(), hipMemcpyHostToDevice));
            HIP(hipMemcpy(A1.pointer, hostA[1].data(), A1.bytes(), hipMemcpyHostToDevice));
            HIP(hipMemcpy(B.pointer, hostB.data(), B.bytes(), hipMemcpyHostToDevice));
            HIP(hipMemcpy(C.pointer, hostC.data(), C.bytes(), hipMemcpyHostToDevice));
        }
        ~Problem()
        {
            static_cast<void>(hipStreamSynchronize(stream));
            static_cast<void>(hipEventDestroy(start));
            static_cast<void>(hipEventDestroy(stop));
            static_cast<void>(hipStreamDestroy(stream));
            hipblasLtMatrixLayoutDestroy(la);
            hipblasLtMatrixLayoutDestroy(lb);
            hipblasLtMatrixLayoutDestroy(lc);
            hipblasLtMatrixLayoutDestroy(ld);
            hipblasLtMatmulDescDestroy(desc);
            hipblasLtDestroy(handle);
        }
        const __half* a(int which) const
        {
            return which ? A1.pointer : A0.pointer;
        }
        size_t workspaceBytes() const
        {
            return W ? W->bytes() : 0;
        }
        jit::Request request(float& alpha, float& beta)
        {
            jit::Request     result;
            jit::Diagnostics diagnostics;
            BLAS(jit::makeGemmRequest(handle,
                                      desc,
                                      &alpha,
                                      A0.pointer,
                                      la,
                                      B.pointer,
                                      lb,
                                      &beta,
                                      C.pointer,
                                      lc,
                                      D.pointer,
                                      ld,
                                      result,
                                      diagnostics));
            return result;
        }
        hipblasStatus_t call(const hipblasLtMatmulAlgo_t& algo, float alpha, float beta, int which)
        {
            return call(algo, alpha, beta, which, workspaceBytes());
        }
        hipblasStatus_t call(const hipblasLtMatmulAlgo_t& algo,
                             float                        alpha,
                             float                        beta,
                             int                          which,
                             size_t                       bytes)
        {
            return hipblasLtMatmul(handle,
                                   desc,
                                   &alpha,
                                   a(which),
                                   la,
                                   B.pointer,
                                   lb,
                                   &beta,
                                   C.pointer,
                                   lc,
                                   D.pointer,
                                   ld,
                                   &algo,
                                   W ? W->pointer : nullptr,
                                   bytes,
                                   stream);
        }
        // All-ones FP16 is NaN, so unchanged D always fails verification.
        void poison()
        {
            HIP(hipMemsetAsync(D.pointer, 0xff, D.bytes(), stream));
            if(W)
                HIP(hipMemsetAsync(W->pointer, sentinel, W->bytes(), stream));
        }
        void verify(const std::string& label, int which, float alpha, float beta)
        {
            std::vector<__half> hostD(M * N);
            HIP(hipMemcpyAsync(hostD.data(), D.pointer, D.bytes(), hipMemcpyDeviceToHost, stream));
            HIP(hipStreamSynchronize(stream));
            for(int col = 0; col < N; ++col)
                for(int row = 0; row < M; ++row)
                {
                    float sum = 0;
                    for(int k = 0; k < K; ++k)
                        sum += __half2float(hostA[which][row + k * M])
                               * __half2float(hostB[k + col * K]);
                    const auto i        = row + col * M;
                    const auto expected = __half2float(
                        __float2half(alpha * sum + beta * __half2float(hostC[i])));
                    const auto actual = __half2float(hostD[i]);
                    // Half an FP16 ULP near unit scale plus 0.1% relative tolerance.
                    require(std::isfinite(actual)
                                && std::abs(actual - expected)
                                       <= 0.0005f + 0.001f * std::abs(expected),
                            label + ": mismatch at " + std::to_string(i) + ", expected "
                                + std::to_string(expected) + ", actual " + std::to_string(actual));
                }
        }
        void untouched(const std::string& label)
        {
            std::vector<unsigned char> hostD(D.bytes()), hostW(workspaceBytes());
            HIP(hipMemcpyAsync(hostD.data(), D.pointer, D.bytes(), hipMemcpyDeviceToHost, stream));
            if(W)
                HIP(hipMemcpyAsync(
                    hostW.data(), W->pointer, W->bytes(), hipMemcpyDeviceToHost, stream));
            HIP(hipStreamSynchronize(stream));
            require(std::all_of(hostD.begin(), hostD.end(), [](auto b) { return b == 0xff; }),
                    label + " wrote D");
            require(std::all_of(hostW.begin(), hostW.end(), [](auto b) { return b == sentinel; }),
                    label + " wrote workspace");
        }
    };

    jit::Backend backend(const std::string& replay, mock::Options::Fault fault = {})
    {
        jit::Backend     result;
        jit::Diagnostics diagnostics;
        const auto       status = mock::createBackend({replay, fault}, result, diagnostics);
        require(status == HIPBLAS_STATUS_SUCCESS, "Mock backend: " + diagnostics.message);
        return result;
    }

    void test(const std::string& replay)
    {
        int device = -1;
        HIP(hipGetDevice(&device));
        std::vector<std::string> objects;
        const auto               envelope = jit::tensilelite::detail::artifacts::readEnvelope(
            std::filesystem::u8path(replay) / "loader.bin", objects);
        const auto kernelName   = envelope.at("main_kernel.name");
        const auto solutionName = envelope.at("solution.name");
        Problem    p;

        float alpha = 1.25f, beta = 0.5f;
        auto  request = p.request(alpha, beta);
        alpha         = 81;
        beta          = 92;
        const auto& gemm
            = dynamic_cast<const abi::GemmRequest&>(*abi::RequestAccess::get(request));
        float ownedAlpha = 0, ownedBeta = 0;
        std::memcpy(&ownedAlpha, gemm.problem.alpha, sizeof(float));
        std::memcpy(&ownedBeta, gemm.problem.beta, sizeof(float));
        require(ownedAlpha == 1.25f && ownedBeta == 0.5f,
                "makeGemmRequest retained the caller's mutable scalar pointers");
        auto             provider = backend(replay);
        jit::Solution    solution;
        jit::Diagnostics diagnostics;
        const auto       initial = jit::getJitAlgo(
            device, request, provider, std::numeric_limits<size_t>::max(), solution, diagnostics);
        require(initial == HIPBLAS_STATUS_SUCCESS,
                "Mock generation: " + diagnostics.message + " (status " + std::to_string(initial)
                    + ")");
        require(diagnostics.backend == "mock",
                "Diagnostics named backend '" + diagnostics.backend + "'");
        hipblasLtMatmulHeuristicResult_t first{}, second{};
        BLAS(jit::getGemmAlgo(solution, first, diagnostics));
        BLAS(jit::getGemmAlgo(solution, second, diagnostics));
        require(std::memcmp(&first.algo, &second.algo, sizeof(first.algo)) == 0,
                "Repeated adaptation changed algorithm identity");
        require(first.workspaceSize > 0, "The split-K replay needs workspace");
        require(hipblaslt_ext::getIndexFromAlgo(first.algo) == -1,
                "JIT algorithm exposed a prebuilt index");
        const auto bytes = first.workspaceSize;
        p.W              = std::make_unique<DeviceBuffer<unsigned char>>(bytes);
        auto                                       copiedAlgo = first.algo;
        std::weak_ptr<const abi::CompiledSolution> retained   = abi::SolutionAccess::get(solution);
        provider                                              = {};
        request                                               = {};
        solution                                              = {};
        require(!retained.expired(),
                "Adapted algorithm lost its solution after public owners were destroyed");
        require(hipblaslt_ext::getSolutionNameFromAlgo(p.handle, copiedAlgo) == solutionName,
                "Solution name lookup did not reach the replayed solution");
        require(hipblaslt_ext::getKernelNameFromAlgo(p.handle, copiedAlgo) == kernelName,
                "Kernel name lookup did not reach the replayed kernel");
        p.poison();
        BLAS(p.call(copiedAlgo, 1.25f, 0.5f, 0));
        p.verify("C GEMM", 0, 1.25f, 0.5f);
        std::cout
            << "PASS mock replay through Jit, C GEMM, owned scalars, names, copied algorithm\n";

        p.poison();
        BLAS(p.call(copiedAlgo, -0.75f, 0.25f, 1));
        p.verify("C GEMM with changed A and scalars", 1, -0.75f, 0.25f);
        std::cout << "PASS changed pointer and scalars reuse the solution\n";

        // Prebuilt Tensile handles have 64 Stream-K flag slots. This solution never
        // uses them; crossing that count must not introduce a backend restriction.
        struct Streams
        {
            Problem&                 problem;
            hipStream_t              original;
            std::vector<hipStream_t> values;
            explicit Streams(Problem& p)
                : problem(p)
                , original(p.stream)
            {
            }
            ~Streams()
            {
                problem.stream = original;
                for(auto stream : values)
                    static_cast<void>(hipStreamDestroy(stream));
            }
        } streams(p);
        for(int i = 0; i < 65; ++i)
        {
            hipStream_t stream;
            HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            streams.values.push_back(stream);
            p.stream = stream;
            BLAS(p.call(copiedAlgo, 1.25f, 0.5f, 0));
            HIP(hipStreamSynchronize(stream));
        }
        p.stream = streams.original;
        p.verify("C GEMM on 65 streams", 0, 1.25f, 0.5f);
        std::cout << "PASS mock solution does not consume Tensile Stream-K stream slots\n";

        p.poison();
        require(p.call(copiedAlgo, 1.25f, 0.5f, 0, bytes - 1) != HIPBLAS_STATUS_SUCCESS,
                "Insufficient workspace was accepted");
        p.untouched("Insufficient workspace");
        auto forged = copiedAlgo;
        std::memset(forged.data + 9, 0, 7); // The registry never issues token zero.
        require(p.call(forged, 1.25f, 0.5f, 0) != HIPBLAS_STATUS_SUCCESS,
                "A forged JIT token reached execution");
        auto indexed    = copiedAlgo;
        indexed.data[0] = 1;
        require(p.call(indexed, 1.25f, 0.5f, 0) != HIPBLAS_STATUS_SUCCESS,
                "A nonzero JIT local index reached execution");
        p.untouched("Rejected algorithms");
        std::cout << "PASS insufficient workspace, forged token and nonzero index rejected before "
                     "GPU submission\n";

        hipblaslt_ext::Gemm cpp(p.handle,
                                HIPBLAS_OP_N,
                                HIPBLAS_OP_N,
                                HIP_R_16F,
                                HIP_R_16F,
                                HIP_R_16F,
                                HIP_R_16F,
                                HIPBLAS_COMPUTE_32F);
        {
            hipblaslt_ext::GemmInputs inputs;
            inputs.setA(p.A0.pointer);
            inputs.setB(p.B.pointer);
            inputs.setC(p.C.pointer);
            inputs.setD(p.D.pointer);
            float a = 0.75f, b = -0.25f;
            inputs.setAlpha(&a);
            inputs.setBeta(&b);
            hipblaslt_ext::GemmEpilogue    epilogue;
            hipblaslt_ext::GemmProblemType type(HIPBLAS_OP_N,
                                                HIPBLAS_OP_N,
                                                HIP_R_16F,
                                                HIP_R_16F,
                                                HIP_R_16F,
                                                HIP_R_16F,
                                                HIPBLAS_COMPUTE_32F);
            BLAS(cpp.setProblem(
                M, N, K, 1, M, K, M, M, M * K, K * N, M * N, M * N, epilogue, inputs, type));
            a = 91;
            b = 92;
        }
        size_t needed = 0;
        BLAS(cpp.isAlgoSupported(copiedAlgo, needed));
        require(needed == bytes, "Dimension-based workspace mismatch");
        cpp.setMaxWorkspaceBytes(bytes);
        p.poison();
        BLAS(cpp.initialize(copiedAlgo, p.W->pointer, true, p.stream));
        BLAS(cpp.run(p.stream));
        p.verify("Dimension-based C++ GEMM", 0, 0.75f, -0.25f);
        std::cout << "PASS dimension-based C++ GEMM retains host scalar values\n";
        {
            float localAlpha = 0.5f, localBeta = -1.25f;
            BLAS(cpp.setProblem(p.desc,
                                &localAlpha,
                                p.A1.pointer,
                                p.la,
                                p.B.pointer,
                                p.lb,
                                &localBeta,
                                p.C.pointer,
                                p.lc,
                                p.D.pointer,
                                p.ld));
            localAlpha = -999;
            localBeta  = 999;
        }
        hipblaslt_ext::GemmTuning noOverrides;
        BLAS(cpp.isAlgoSupported(copiedAlgo, noOverrides, needed));
        hipblaslt_ext::GemmTuning overrides;
        overrides.setSplitK(2);
        require(cpp.isAlgoSupported(copiedAlgo, overrides, needed) == HIPBLAS_STATUS_NOT_SUPPORTED,
                "JIT support accepted a tuning override");
        BLAS(cpp.isAlgoSupported(copiedAlgo, needed));
        require(needed == bytes, "C++ support returned an incorrect workspace size");
        p.poison();
        BLAS(cpp.initialize(copiedAlgo, p.W->pointer, true, p.stream));
        BLAS(cpp.run(p.stream, p.start, p.stop));
        HIP(hipEventSynchronize(p.stop));
        float elapsed = -1;
        HIP(hipEventElapsedTime(&elapsed, p.start, p.stop));
        require(elapsed >= 0, "Run did not record the supplied events");
        p.verify("C++ GEMM", 1, 0.5f, -1.25f);
        require(cpp.getSolutionName() == solutionName && cpp.getKernelName() == kernelName,
                "C++ name dispatch failed");
        BLAS(cpp.setProblem(p.desc,
                            &alpha,
                            p.A0.pointer,
                            p.la,
                            p.B.pointer,
                            p.lb,
                            &beta,
                            p.C.pointer,
                            p.lc,
                            p.D.pointer,
                            p.ld));
        require(cpp.run(p.stream) == HIPBLAS_STATUS_NOT_INITIALIZED,
                "setProblem retained a launch bound to previous buffers/scalars");
        std::cout << "PASS C++ setProblem/support/initialize/run, scalar lifetime, events, name "
                     "dispatch\n";

        alpha    = 1.25f;
        beta     = 0.5f;
        request  = p.request(alpha, beta);
        provider = backend(replay);
        require(jit::getJitAlgo(device + 1, request, provider, bytes, solution, diagnostics)
                        == HIPBLAS_STATUS_INVALID_VALUE
                    && !abi::SolutionAccess::get(solution),
                "Generation for another device was accepted");
        int deviceCount = 0;
        HIP(hipGetDeviceCount(&deviceCount));
        if(deviceCount > 1)
        {
            HIP(hipSetDevice((device + 1) % deviceCount));
            const auto status = cpp.isAlgoSupported(copiedAlgo, needed);
            HIP(hipSetDevice(device));
            require(status != HIPBLAS_STATUS_SUCCESS, "JIT algorithm accepted the wrong device");
        }
        std::cout << "PASS wrong device rejected\n";

        const auto probe = abi::RequestAccess::make(std::make_shared<ProbeRequest>());
        require(jit::getJitAlgo(device, probe, provider, 0, solution, diagnostics)
                        == HIPBLAS_STATUS_NOT_SUPPORTED
                    && !abi::SolutionAccess::get(solution) && diagnostics.backend == "mock"
                    && diagnostics.message.find("GEMM") != std::string::npos,
                "A non-GEMM request was not declined: " + diagnostics.message);
        {
            hipblasLtMatmulDesc_t   transposed{};
            hipblasLtMatrixLayout_t layoutB{};
            BLAS(hipblasLtMatmulDescCreate(&transposed, HIPBLAS_COMPUTE_32F, HIP_R_32F));
            hipblasOperation_t opB = HIPBLAS_OP_T;
            BLAS(hipblasLtMatmulDescSetAttribute(
                transposed, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
            BLAS(hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16F, N, K, N));
            jit::Request mismatched;
            const auto   status = jit::makeGemmRequest(p.handle,
                                                     transposed,
                                                     &alpha,
                                                     p.A0.pointer,
                                                     p.la,
                                                     p.B.pointer,
                                                     layoutB,
                                                     &beta,
                                                     p.C.pointer,
                                                     p.lc,
                                                     p.D.pointer,
                                                     p.ld,
                                                     mismatched,
                                                     diagnostics);
            hipblasLtMatrixLayoutDestroy(layoutB);
            hipblasLtMatmulDescDestroy(transposed);
            BLAS(status);
            require(jit::getJitAlgo(device, mismatched, provider, bytes, solution, diagnostics)
                            == HIPBLAS_STATUS_NOT_SUPPORTED
                        && !abi::SolutionAccess::get(solution)
                        && diagnostics.message.find("does not solve") != std::string::npos,
                    "A mismatched ProblemType was not declined: " + diagnostics.message);
        }
        std::cout << "PASS non-GEMM request and mismatched ProblemType are not supported\n";

        const std::pair<mock::Options::Fault, const char*> faults[]
            = {{mock::Options::Fault::Generate, "Mock generation fault"},
               {mock::Options::Fault::Build, "is not an ELF file"}};
        for(const auto& [fault, message] : faults)
        {
            const auto status = jit::getJitAlgo(
                device, request, backend(replay, fault), bytes, solution, diagnostics);
            require(status == HIPBLAS_STATUS_INTERNAL_ERROR && !abi::SolutionAccess::get(solution)
                        && diagnostics.backend == "mock"
                        && diagnostics.message.find(message) != std::string::npos,
                    "Fault did not surface from its stage: " + diagnostics.message);
        }
        std::cout << "PASS generation and build faults clear the solution and report their stage\n";

        std::weak_ptr<const abi::CompiledSolution> released;
        std::weak_ptr<const abi::KernelBundle>     releasedBundle;
        {
            jit::Solution temporary;
            BLAS(jit::getJitAlgo(device, request, provider, bytes, temporary, diagnostics));
            released       = abi::SolutionAccess::get(temporary);
            releasedBundle = abi::SolutionAccess::get(temporary)->bundle;
        }
        require(released.expired() && releasedBundle.expired(),
                "Unadapted solution retained its bundle");
        require(!retained.expired(), "Adapted solution did not retain its bundle");
        std::cout << "PASS unadapted bundle released; adapted bundle retained for copied "
                     "algorithms\n";

        jit::Backend missing;
        require(mock::createBackend({replay + "/missing"}, missing, diagnostics)
                        == HIPBLAS_STATUS_INVALID_VALUE
                    && !abi::BackendAccess::get(missing) && !diagnostics.message.empty(),
                "A missing replay bundle was accepted");
        std::cout << "PASS missing replay bundle rejected\n";
        std::cout << "ALL MOCK BACKEND CHECKS PASSED\n";
    }
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " SPLITK_API_OUTPUT/bundle\n";
        return 2;
    }
    try
    {
        test(argv[1]);
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
