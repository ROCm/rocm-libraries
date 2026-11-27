// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-gemm-internal.hpp"
#include "hipblaslt-jit-library.hpp"
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <string>

// Replays the committed plain bundle of the current device's architecture as
// plain-pair through Jit, which publishes each solution into the JIT solution
// library. getJitAlgo returns its first solution for K=512 and its second for
// K=256 as JIT kernel indices; each runs through hipblasLtMatmul and
// hipblaslt_ext::Gemm, and through getAlgosFromIndex, and D is checked against
// the host. A second lookup returns the same JIT solution library index.
namespace jit = hipblaslt_jit;

namespace
{
    namespace fs = std::filesystem;

    using hipblaslt_jit_test::Device;
    using hipblaslt_jit_test::Fp16Gemm;

    // FP16 NN GEMMs with FP32 compute, which the plain kernel solves.
    constexpr int M = Fp16Gemm::rows, N = Fp16Gemm::cols;

    // Resolves index with getAlgosFromIndex and runs it through hipblasLtMatmul.
    void runIndex(hipblasLtHandle_t     handle,
                  hipblasLtMatmulDesc_t desc,
                  hipStream_t           stream,
                  int                   K,
                  int                   index,
                  const std::string&    suffix)
    {
        jit::Diagnostics diagnostics;
        const auto       label = "index " + std::to_string(index) + ", K=" + std::to_string(K);
        std::vector<int> wanted{index};
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        hipblaslt_jit_test::checkBlas(hipblaslt_ext::getAlgosFromIndex(handle, wanted, results),
                                      "hipblaslt_ext::getAlgosFromIndex(handle, wanted, results)",
                                      diagnostics.message);
        JIT_REQUIRE(label + " did not resolve",
                    results.size() == 1
                        && hipblaslt_ext::getIndexFromAlgo(results[0].algo) == index);
        const auto name = hipblaslt_ext::getSolutionNameFromAlgo(handle, results[0].algo);
        JIT_REQUIRE(label + " is the solution '" + name + "', not the one ending in " + suffix,
                    hipblaslt_jit_test::endsWith(name, suffix));
        Fp16Gemm matrices(K);
        float    alpha = 1.25f, beta = 0.5f;
        size_t   bytes = 0;
        hipblaslt_jit_test::checkBlas(hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                                           desc,
                                                                           &alpha,
                                                                           matrices.layoutA,
                                                                           matrices.layoutB,
                                                                           &beta,
                                                                           matrices.layoutC,
                                                                           matrices.layoutD,
                                                                           results[0].algo,
                                                                           bytes),
                                      "matmulIsAlgoSupported",
                                      diagnostics.message);
        Device workspace(std::max<size_t>(bytes, 1));
        matrices.matmul(handle,
                        desc,
                        stream,
                        &results[0].algo,
                        workspace.pointer,
                        bytes,
                        alpha,
                        beta,
                        "hipblasLtMatmul, " + label);
    }

    // Generates the solution for an M x N x K GEMM, requires that its name ends
    // with suffix, and runs it through both GEMM APIs.
    void run(hipblasLtHandle_t                                handle,
             hipblasLtMatmulDesc_t                            desc,
             hipStream_t                                      stream,
             const std::shared_ptr<const hipblaslt_jit::Jit>& backend,
             int                                              K,
             const std::string&                               suffix)
    {
        int device = -1;
        hipblaslt_jit_test::checkHip(hipGetDevice(&device), "hipGetDevice(&device)");
        jit::Diagnostics diagnostics;
        const auto       label = "K=" + std::to_string(K);
        Fp16Gemm         matrices(K);

        // Generate (replay), build with comgr, then publish a JIT library index.
        std::shared_ptr<const jit::GemmRequest> request;
        float                                   alpha = 1.25f, beta = 0.5f;
        hipblaslt_jit_test::checkBlas(jit::makeGemmRequest(handle,
                                                           desc,
                                                           &alpha,
                                                           matrices.A.pointer,
                                                           matrices.layoutA,
                                                           matrices.B.pointer,
                                                           matrices.layoutB,
                                                           &beta,
                                                           matrices.C.pointer,
                                                           matrices.layoutC,
                                                           matrices.D.pointer,
                                                           matrices.layoutD,
                                                           request,
                                                           diagnostics),
                                      "makeGemmRequest",
                                      diagnostics.message);
        hipblasLtMatmulHeuristicResult_t result{};
        hipblaslt_jit_test::checkBlas(
            jit::getJitAlgo(device, request, backend, 64 << 20, result, diagnostics),
            "jit::getJitAlgo(device, request, backend, 64 << 20, result, diagnostics)",
            diagnostics.message);
        const int index = hipblaslt_ext::getIndexFromAlgo(result.algo);
        JIT_REQUIRE(label + " returned " + std::to_string(index) + ", not a JIT library index",
                    TensileLite::isJitKernelIndex(index) && result.state == HIPBLAS_STATUS_SUCCESS);
        hipblasLtMatmulHeuristicResult_t repeat{};
        hipblaslt_jit_test::checkBlas(
            jit::getJitAlgo(device, request, backend, 64 << 20, repeat, diagnostics),
            "jit::getJitAlgo(device, request, backend, 64 << 20, repeat, diagnostics)",
            diagnostics.message);
        JIT_REQUIRE(label + " published a different index on the second lookup",
                    hipblaslt_ext::getIndexFromAlgo(repeat.algo) == index);
        Device     workspace(result.workspaceSize);
        const auto name   = hipblaslt_ext::getSolutionNameFromAlgo(handle, result.algo);
        const auto kernel = hipblaslt_ext::getKernelNameFromAlgo(handle, result.algo);
        JIT_REQUIRE(label + " selected the solution '" + name + "', not the one ending in "
                        + suffix,
                    hipblaslt_jit_test::endsWith(name, suffix));
        JIT_REQUIRE("Unexpected kernel name '" + kernel + "'", kernel.rfind("Cijk_", 0) == 0);

        matrices.matmul(handle,
                        desc,
                        stream,
                        &result.algo,
                        workspace.pointer,
                        result.workspaceSize,
                        alpha,
                        beta,
                        "hipblasLtMatmul, " + label);

        hipblaslt_ext::Gemm gemm(handle,
                                 HIPBLAS_OP_N,
                                 HIPBLAS_OP_N,
                                 HIP_R_16F,
                                 HIP_R_16F,
                                 HIP_R_16F,
                                 HIP_R_16F,
                                 HIPBLAS_COMPUTE_32F);
        float               gemmAlpha = -0.75f, gemmBeta = 0.25f;
        hipblaslt_jit_test::checkBlas(gemm.setProblem(desc,
                                                      &gemmAlpha,
                                                      matrices.A.pointer,
                                                      matrices.layoutA,
                                                      matrices.B.pointer,
                                                      matrices.layoutB,
                                                      &gemmBeta,
                                                      matrices.C.pointer,
                                                      matrices.layoutC,
                                                      matrices.D.pointer,
                                                      matrices.layoutD),
                                      "Gemm::setProblem",
                                      diagnostics.message);
        size_t needed = 0;
        hipblaslt_jit_test::checkBlas(gemm.isAlgoSupported(result.algo, needed),
                                      "gemm.isAlgoSupported(result.algo, needed)",
                                      diagnostics.message);
        JIT_REQUIRE("isAlgoSupported returned another workspace size",
                    needed == result.workspaceSize);
        hipblaslt_jit_test::checkHip(
            hipMemset(matrices.D.pointer, 0xff, static_cast<size_t>(M) * N * sizeof(__half)),
            "hipMemset(matrices.D.pointer, 0xff, static_cast<size_t>(M) * N * sizeof(__half))");
        hipblaslt_jit_test::checkBlas(
            gemm.initialize(result.algo, workspace.pointer, true, stream),
            "gemm.initialize(result.algo, workspace.pointer, true, stream)",
            diagnostics.message);
        hipblaslt_jit_test::checkBlas(gemm.run(stream), "gemm.run(stream)", diagnostics.message);
        hipblaslt_jit_test::checkHip(hipStreamSynchronize(stream), "hipStreamSynchronize(stream)");
        hipblaslt_jit_test::checkFp16("hipblaslt_ext::Gemm, " + label,
                                      M,
                                      N,
                                      K,
                                      matrices.hostA,
                                      matrices.hostB,
                                      matrices.hostC,
                                      matrices.D.pointer,
                                      gemmAlpha,
                                      gemmBeta);
        runIndex(handle, desc, stream, K, index, suffix);
    }

    void test(const std::string& root)
    {
        hipblaslt_jit_test::ReplayFixture fixture(root);
        jit::Diagnostics                  diagnostics;
        auto&                             handle = fixture.handle;
        auto&                             desc   = fixture.desc;
        const int                         device = fixture.device;

        run(handle, desc, fixture.stream, fixture.backend, 512, "_K512_WGM8");
        run(handle, desc, fixture.stream, fixture.backend, 256, "_WGM1");

        // A transposed A is another problem type, which neither solution solves.
        constexpr int            K = 512;
        Device                   A(M * K * 2), B(K * N * 2), C(M * N * 2), D(M * N * 2);
        hipblasLtMatmulDesc_t    tn{};
        hipblasLtMatrixLayout_t  lt{}, lb{}, lc{};
        const hipblasOperation_t transpose = HIPBLAS_OP_T;
        float                    alpha = 1.25f, beta = 0.5f;
        hipblaslt_jit_test::checkBlas(
            hipblasLtMatmulDescCreate(&tn, HIPBLAS_COMPUTE_32F, HIP_R_32F),
            "hipblasLtMatmulDescCreate(&tn, HIPBLAS_COMPUTE_32F, HIP_R_32F)",
            diagnostics.message);
        hipblaslt_jit_test::checkBlas(
            hipblasLtMatmulDescSetAttribute(
                tn, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)),
            "hipblasLtMatmulDescSetAttribute",
            diagnostics.message);
        hipblaslt_jit_test::checkBlas(hipblasLtMatrixLayoutCreate(&lt, HIP_R_16F, K, M, K),
                                      "hipblasLtMatrixLayoutCreate(&lt, HIP_R_16F, K, M, K)",
                                      diagnostics.message);
        hipblaslt_jit_test::checkBlas(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, K, N, K),
                                      "hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, K, N, K)",
                                      diagnostics.message);
        hipblaslt_jit_test::checkBlas(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16F, M, N, M),
                                      "hipblasLtMatrixLayoutCreate(&lc, HIP_R_16F, M, N, M)",
                                      diagnostics.message);
        std::shared_ptr<const jit::GemmRequest> transposed;
        hipblasLtMatmulHeuristicResult_t        unsolved{};
        hipblaslt_jit_test::checkBlas(jit::makeGemmRequest(handle,
                                                           tn,
                                                           &alpha,
                                                           A.pointer,
                                                           lt,
                                                           B.pointer,
                                                           lb,
                                                           &beta,
                                                           C.pointer,
                                                           lc,
                                                           D.pointer,
                                                           lc,
                                                           transposed,
                                                           diagnostics),
                                      "makeGemmRequest",
                                      diagnostics.message);
        JIT_REQUIRE(
            "The replay backend accepted a problem its bundle does not solve",
            jit::getJitAlgo(device, transposed, fixture.backend, 64 << 20, unsolved, diagnostics)
                    == HIPBLAS_STATUS_NOT_SUPPORTED
                && unsolved.state == HIPBLAS_STATUS_NOT_SUPPORTED);
        hipblasLtMatrixLayoutDestroy(lt);
        hipblasLtMatrixLayoutDestroy(lb);
        hipblasLtMatrixLayoutDestroy(lc);
        hipblasLtMatmulDescDestroy(tn);
    }

    TEST_CASE("replayed JIT GEMM solutions run end to end", "[jit-gpu]")
    {
        [[maybe_unused]] hipblaslt_jit_test::LibraryScratch root("end-to-end");
        test(HIPBLASLT_JIT_DATA);
    }

    // FP32 alpha and beta with FP16 compute: the request holds the scale type's width.
    TEST_CASE("a GEMM request copies alpha and beta at the scale type", "[jit-gpu]")
    {
        jit::Diagnostics      diagnostics;
        hipblasLtHandle_t     handle{};
        hipblasLtMatmulDesc_t desc{};
        hipblaslt_jit_test::checkBlas(
            hipblasLtCreate(&handle), "hipblasLtCreate(&handle)", diagnostics.message);
        hipblaslt_jit_test::checkBlas(
            hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_16F, HIP_R_32F),
            "hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_16F, HIP_R_32F)",
            diagnostics.message);
        Fp16Gemm                                matrices(512);
        const float                             alpha = 1.25f, beta = 0.5f;
        std::shared_ptr<const jit::GemmRequest> request;
        hipblaslt_jit_test::checkBlas(jit::makeGemmRequest(handle,
                                                           desc,
                                                           &alpha,
                                                           matrices.A.pointer,
                                                           matrices.layoutA,
                                                           matrices.B.pointer,
                                                           matrices.layoutB,
                                                           &beta,
                                                           matrices.C.pointer,
                                                           matrices.layoutC,
                                                           matrices.D.pointer,
                                                           matrices.layoutD,
                                                           request,
                                                           diagnostics),
                                      "makeGemmRequest",
                                      diagnostics.message);
        JIT_REQUIRE("The request did not keep the FP32 alpha and beta",
                    *static_cast<const float*>(request->problem.alpha) == alpha
                        && *static_cast<const float*>(request->problem.beta) == beta);
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtDestroy(handle);
    }
}
