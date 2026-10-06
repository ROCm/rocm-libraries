// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-gemm-internal.hpp"
#include "hipblaslt-jit-hipkittens.hpp"
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// The HipKittens process backend on gfx950: compiled-in variants, a capturing
// stream that must not compile, one BF16 TN GEMM checked against a host sum of
// ones, and a size the kernel does not solve.
namespace jit = hipblaslt_ext::experimental::jit;
namespace hk  = hipblaslt_ext::experimental::jit::hipkittens;

namespace
{
    void require(bool condition, const std::string& message)
    {
        if(!condition)
        {
            std::cerr << "FAIL " << message << std::endl;
            std::exit(1);
        }
    }

    void hip(hipError_t status, const char* expression)
    {
        require(status == hipSuccess, std::string(expression) + ": " + hipGetErrorString(status));
    }
#define HIP(expression) hip((expression), #expression)
#define BLAS(expression)                                                         \
    do                                                                           \
    {                                                                            \
        const auto status_ = (expression);                                       \
        require(status_ == HIPBLAS_STATUS_SUCCESS,                               \
                std::string(#expression) + ": status " + std::to_string(status_) \
                    + " " + diagnostics.message);                                \
    } while(false)

    // BF16 1 and 128, which the K=128 sum of ones produces exactly.
    constexpr uint16_t bf16One = 0x3f80;
    constexpr uint16_t bf16K   = 0x4300;
    constexpr int      M = 256, N = 256, K = 128;

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

    void checkResources()
    {
        const auto& resources = hk::detail::resources();
        require(resources.variants.size() == 2, "expected the BF16 and FP16 variants");
        bool bf16 = false, fp16 = false;
        for(const auto& variant : resources.variants)
        {
            require(variant.isa == "gfx950", "variant is not gfx950");
            require(variant.resources.kernargBytes == 84 && variant.resources.ldsBytes == 160000
                        && variant.resources.vgprSpills == 0,
                    std::string(variant.kernelName) + " resources changed");
            if(variant.kernelName == "HK_gemm_bf16_TN_MT256x256x64_W2x4_gfx950_abi5")
            {
                bf16 = true;
                require(variant.resources.vgprs == 237, "BF16 VGPR count");
            }
            else if(variant.kernelName == "HK_gemm_f16_TN_MT256x256x64_W2x4_gfx950_abi5")
            {
                fp16 = true;
                require(variant.resources.vgprs == 238, "FP16 VGPR count");
            }
        }
        require(bf16 && fp16, "missing a gfx950 kernel name");
        std::cout << "PASS compiled-in gfx950 variants\n";
    }

    // A TN BF16 problem. A is stored transposed.
    void makeProblem(hipblasLtHandle_t        handle,
                     hipblasLtMatmulDesc_t    desc,
                     int                      rows,
                     int                      cols,
                     int                      k,
                     jit::Request&            request,
                     jit::Diagnostics&        diagnostics)
    {
        Device                  A(size_t(k) * rows * 2), B(size_t(k) * cols * 2);
        Device                  C(size_t(rows) * cols * 2), D(size_t(rows) * cols * 2);
        hipblasLtMatrixLayout_t la{}, lb{}, lc{}, ld{};
        float                   alpha = 1, beta = 0;
        BLAS(hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, k, rows, k));
        BLAS(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, k, cols, k));
        BLAS(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16BF, rows, cols, rows));
        BLAS(hipblasLtMatrixLayoutCreate(&ld, HIP_R_16BF, rows, cols, rows));
        BLAS(jit::makeGemmRequest(handle,
                                  desc,
                                  &alpha,
                                  A.pointer,
                                  la,
                                  B.pointer,
                                  lb,
                                  &beta,
                                  C.pointer,
                                  lc,
                                  D.pointer,
                                  ld,
                                  request,
                                  diagnostics));
        hipblasLtMatrixLayoutDestroy(la);
        hipblasLtMatrixLayoutDestroy(lb);
        hipblasLtMatrixLayoutDestroy(lc);
        hipblasLtMatrixLayoutDestroy(ld);
    }

    void captureSkipsCompile(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc)
    {
        jit::Diagnostics diagnostics;
        jit::Request     request;
        jit::Backend     backend;
        BLAS(hk::createBackend({}, backend, diagnostics));
        makeProblem(handle, desc, M, N, K, request, diagnostics);
        const auto operation = jit::detail::RequestAccess::get(request);
        const auto gemm      = dynamic_cast<const jit::detail::GemmRequest*>(operation.get());
        require(gemm != nullptr, "request is not a GEMM");
        auto problem = gemm->problem;

        int         device = -1;
        hipStream_t stream{};
        HIP(hipGetDevice(&device));
        HIP(hipStreamCreate(&stream));
        problem.stream = stream;
        auto capturing = jit::detail::RequestAccess::make(
            std::make_shared<jit::detail::GemmRequest>(problem));
        Device marker(4);
        HIP(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
        HIP(hipMemsetAsync(marker.pointer, 0, 4, stream));
        jit::Solution  skipped;
        const auto     started = std::chrono::steady_clock::now();
        const auto     status
            = jit::getJitAlgo(device, capturing, backend, 64 << 20, skipped, diagnostics);
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
        hipGraph_t graph{};
        HIP(hipStreamEndCapture(stream, &graph));
        HIP(hipGraphDestroy(graph));
        HIP(hipStreamDestroy(stream));
        require(status == HIPBLAS_STATUS_NOT_SUPPORTED, "capture status " + std::to_string(status));
        require(diagnostics.message.find("stream capture") != std::string::npos, diagnostics.message);
        require(elapsed < 20000, "generation started during stream capture ("
                                     + std::to_string(elapsed) + " ms)");
        std::cout << "PASS capturing stream did not compile (" << elapsed << " ms)\n";
    }

    void runGemm(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc, hipStream_t stream)
    {
        jit::Diagnostics diagnostics;
        int              device = -1;
        HIP(hipGetDevice(&device));
        const auto hostBytes = size_t(M) * N * sizeof(uint16_t);
        std::vector<uint16_t> ones(size_t(K) * M, bf16One);
        std::vector<uint16_t> onesN(size_t(K) * N, bf16One);
        std::vector<uint16_t> zero(size_t(M) * N, 0);
        Device                A(ones.size() * 2), B(onesN.size() * 2), C(hostBytes), D(hostBytes);
        HIP(hipMemcpy(A.pointer, ones.data(), ones.size() * 2, hipMemcpyHostToDevice));
        HIP(hipMemcpy(B.pointer, onesN.data(), onesN.size() * 2, hipMemcpyHostToDevice));
        HIP(hipMemcpy(C.pointer, zero.data(), hostBytes, hipMemcpyHostToDevice));
        hipblasLtMatrixLayout_t la{}, lb{}, lc{}, ld{};
        float                   alpha = 1, beta = 0;
        BLAS(hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, K, M, K));
        BLAS(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, K, N, K));
        BLAS(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16BF, M, N, M));
        BLAS(hipblasLtMatrixLayoutCreate(&ld, HIP_R_16BF, M, N, M));

        jit::Request  request;
        jit::Solution solution;
        jit::Backend  backend;
        BLAS(hk::createBackend({}, backend, diagnostics));
        BLAS(jit::makeGemmRequest(handle,
                                  desc,
                                  &alpha,
                                  A.pointer,
                                  la,
                                  B.pointer,
                                  lb,
                                  &beta,
                                  C.pointer,
                                  lc,
                                  D.pointer,
                                  ld,
                                  request,
                                  diagnostics));
        BLAS(jit::getJitAlgo(device, request, backend, 64 << 20, solution, diagnostics));
        hipblasLtMatmulHeuristicResult_t result{};
        BLAS(jit::getGemmAlgo(solution, result, diagnostics));
        const auto kernel = hipblaslt_ext::getKernelNameFromAlgo(handle, result.algo);
        require(kernel == "HK_gemm_bf16_TN_MT256x256x64_W2x4_gfx950_abi5",
                "selected '" + kernel + "'");
        Device workspace(result.workspaceSize);
        HIP(hipMemset(D.pointer, 0xff, hostBytes));
        BLAS(hipblasLtMatmul(handle,
                             desc,
                             &alpha,
                             A.pointer,
                             la,
                             B.pointer,
                             lb,
                             &beta,
                             C.pointer,
                             lc,
                             D.pointer,
                             ld,
                             &result.algo,
                             workspace.pointer,
                             result.workspaceSize,
                             stream));
        HIP(hipStreamSynchronize(stream));
        std::vector<uint16_t> out(size_t(M) * N);
        HIP(hipMemcpy(out.data(), D.pointer, hostBytes, hipMemcpyDeviceToHost));
        for(size_t i = 0; i < out.size(); ++i)
            require(out[i] == bf16K, "D[" + std::to_string(i) + "] is not 128");
        std::cout << "PASS BF16 TN 256x256x128\n";

        jit::Request  small;
        jit::Solution unsupported;
        makeProblem(handle, desc, 128, N, K, small, diagnostics);
        require(jit::getJitAlgo(device, small, backend, 64 << 20, unsupported, diagnostics)
                    == HIPBLAS_STATUS_NOT_SUPPORTED,
                "M=128 was accepted");
        std::cout << "PASS M=128 is not supported\n";

        hipblasLtMatrixLayoutDestroy(la);
        hipblasLtMatrixLayoutDestroy(lb);
        hipblasLtMatrixLayoutDestroy(lc);
        hipblasLtMatrixLayoutDestroy(ld);
    }
}

int main()
{
    unsetenv("HIPBLASLT_JIT_TEST_REPLAY");
    int             device = -1;
    hipDeviceProp_t properties{};
    HIP(hipGetDevice(&device));
    HIP(hipGetDeviceProperties(&properties, device));
    require(std::string(properties.gcnArchName).rfind("gfx950", 0) == 0,
            std::string("need gfx950, got ") + properties.gcnArchName);

    checkResources();
    hipblasLtHandle_t     handle{};
    hipblasLtMatmulDesc_t desc{};
    hipStream_t           stream{};
    jit::Diagnostics      diagnostics;
    BLAS(hipblasLtCreate(&handle));
    BLAS(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    const hipblasOperation_t transpose = HIPBLAS_OP_T;
    BLAS(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
    HIP(hipStreamCreate(&stream));

    captureSkipsCompile(handle, desc);
    runGemm(handle, desc, stream);

    static_cast<void>(hipStreamDestroy(stream));
    hipblasLtMatmulDescDestroy(desc);
    hipblasLtDestroy(handle);
    return 0;
}
