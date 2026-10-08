// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-gemm-internal.hpp"
#include "hipblaslt-jit-hipkittens.hpp"
#include "test_helpers.hpp"
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdint>
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
    using hipblaslt_jit_test::Device;

#define HIP(expression) hipblaslt_jit_test::checkHip((expression), #expression)
#define BLAS(expression)                                                         \
    do                                                                           \
    {                                                                            \
        const auto status_ = (expression);                                       \
        { \
            INFO(( \
                std::string(#expression) + ": status " + std::to_string(status_) + " " + diagnostics.message)); \
            REQUIRE((status_ == HIPBLAS_STATUS_SUCCESS)); \
        } \
    } while(false)

    // BF16 1 and 128, which the K=128 sum of ones produces exactly.
    constexpr uint16_t bf16One = 0x3f80;
    constexpr uint16_t bf16K   = 0x4300;
    constexpr int      M = 256, N = 256, K = 128;

    void checkResources()
    {
        const auto& resources = hk::detail::resources();
        {
            INFO(("expected the BF16 and FP16 variants"));
            REQUIRE((resources.variants.size() == 2));
        }
        bool bf16 = false, fp16 = false;
        for(const auto& variant : resources.variants)
        {
            {
                INFO(("variant is not gfx950"));
                REQUIRE((variant.isa == "gfx950"));
            }
            {
                INFO((std::string(variant.kernelName) + " resources changed"));
                REQUIRE((variant.resources.kernargBytes == 84 && variant.resources.ldsBytes == 160000
                    && variant.resources.vgprSpills == 0));
            }
            if(variant.kernelName == "HK_gemm_bf16_TN_MT256x256x64_W2x4_gfx950_abi5")
            {
                bf16 = true;
                {
                    INFO(("BF16 VGPR count"));
                    REQUIRE((variant.resources.vgprs == 237));
                }
            }
            else if(variant.kernelName == "HK_gemm_f16_TN_MT256x256x64_W2x4_gfx950_abi5")
            {
                fp16 = true;
                {
                    INFO(("FP16 VGPR count"));
                    REQUIRE((variant.resources.vgprs == 238));
                }
            }
        }
        {
            INFO(("missing a gfx950 kernel name"));
            REQUIRE((bf16 && fp16));
        }
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
        const auto operation = request.get();
        const auto gemm      = dynamic_cast<const jit::detail::GemmRequest*>(operation.get());
        {
            INFO(("request is not a GEMM"));
            REQUIRE((gemm != nullptr));
        }
        auto problem = gemm->problem;

        int         device = -1;
        hipStream_t stream{};
        HIP(hipGetDevice(&device));
        HIP(hipStreamCreate(&stream));
        problem.stream = stream;
        auto capturing = jit::Request::create(
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
        {
            INFO(("capture status " + std::to_string(status)));
            REQUIRE((status == HIPBLAS_STATUS_NOT_SUPPORTED));
        }
        {
            INFO((diagnostics.message));
            REQUIRE((diagnostics.message.find("stream capture") != std::string::npos));
        }
        {
            INFO(("generation started during stream capture ("
                + std::to_string(elapsed) + " ms)"));
            REQUIRE((elapsed < 20000));
        }
        std::cout << "PASS capturing stream did not compile (" << elapsed << " ms)\n";
    }

    // A published index may be returned during capture, and that return does not build.
    void captureReturnsPublished(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc)
    {
        jit::Diagnostics diagnostics;
        jit::Request     request;
        jit::Backend     backend;
        BLAS(hk::createBackend({}, backend, diagnostics));
        makeProblem(handle, desc, M, N, K, request, diagnostics);
        const auto operation = request.get();
        const auto gemm      = dynamic_cast<const jit::detail::GemmRequest*>(operation.get());
        {
            INFO(("request is not a GEMM"));
            REQUIRE((gemm != nullptr));
        }
        auto problem = gemm->problem;

        int         device = -1;
        hipStream_t stream{};
        HIP(hipGetDevice(&device));
        HIP(hipStreamCreate(&stream));
        problem.stream = stream;
        auto capturing = jit::Request::create(
            std::make_shared<jit::detail::GemmRequest>(problem));
        Device marker(4);
        HIP(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
        HIP(hipMemsetAsync(marker.pointer, 0, 4, stream));
        jit::Solution  found;
        const auto     started = std::chrono::steady_clock::now();
        const auto     status
            = jit::getJitAlgo(device, capturing, backend, 64 << 20, found, diagnostics);
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
        hipGraph_t graph{};
        HIP(hipStreamEndCapture(stream, &graph));
        HIP(hipGraphDestroy(graph));
        HIP(hipStreamDestroy(stream));
        {
            INFO(("capture hit status " + std::to_string(status)));
            REQUIRE((status == HIPBLAS_STATUS_SUCCESS));
        }
        {
            INFO(("capture hit generated: " + diagnostics.message));
            REQUIRE((diagnostics.message == "1 of 1 solutions came from the JIT solution library"));
        }
        hipblasLtMatmulHeuristicResult_t result{};
        BLAS(jit::getGemmAlgo(found, result, diagnostics));
        {
            INFO(("capture hit is not a JIT library index"));
            REQUIRE((hipblaslt_ext::getIndexFromAlgo(result.algo) >= (1 << 30)));
        }
        {
            INFO(("generation started during stream capture ("
                + std::to_string(elapsed) + " ms)"));
            REQUIRE((elapsed < 20000));
        }
        std::cout << "PASS capturing stream returned the published index (" << elapsed << " ms)\n";
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
        {
            INFO(("selected '" + kernel + "'"));
            REQUIRE((kernel == "HK_gemm_bf16_TN_MT256x256x64_W2x4_gfx950_abi5"));
        }
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
        {
            INFO(("D[" + std::to_string(i) + "] is not 128"));
            REQUIRE((out[i] == bf16K));
        }
        std::cout << "PASS BF16 TN 256x256x128\n";

        jit::Request  small;
        jit::Solution unsupported;
        makeProblem(handle, desc, 128, N, K, small, diagnostics);
        {
            INFO(("M=128 was accepted"));
            REQUIRE((jit::getJitAlgo(device, small, backend, 64 << 20, unsupported, diagnostics)
                == HIPBLAS_STATUS_NOT_SUPPORTED));
        }
        std::cout << "PASS M=128 is not supported\n";

        hipblasLtMatrixLayoutDestroy(la);
        hipblasLtMatrixLayoutDestroy(lb);
        hipblasLtMatrixLayoutDestroy(lc);
        hipblasLtMatrixLayoutDestroy(ld);
    }
}

TEST_CASE("the HipKittens gfx950 backend", "[jit-gpu]")
{
    int             device = -1;
    hipDeviceProp_t properties{};
    HIP(hipGetDevice(&device));
    HIP(hipGetDeviceProperties(&properties, device));
    {
        INFO((std::string("need gfx950, got ") + properties.gcnArchName));
        REQUIRE((std::string(properties.gcnArchName).rfind("gfx950", 0) == 0));
    }
    hipblaslt_jit_test::LibraryScratch library("hipkittens");

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
    captureReturnsPublished(handle, desc);

    static_cast<void>(hipStreamDestroy(stream));
    hipblasLtMatmulDescDestroy(desc);
    hipblasLtDestroy(handle);
}
