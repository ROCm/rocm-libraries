// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-gemm-internal.hpp"
#include "hipblaslt-jit-hipkittens.hpp"
#include "hipblaslt-jit-problem-type.hpp"
#include "test_helpers.hpp"
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/LocalKernelIndex.hpp>
#include <Tensile/Predicates.hpp>
#include <catch2/catch_test_macros.hpp>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// The HipKittens backend on gfx950: compiled-in kernels, a capturing stream that
// must not compile, BF16 TN GEMMs, a size the kernel does not solve, uniform
// summation order on a default handle, and heuristic queries.
namespace jit = hipblaslt_ext::experimental::jit;
namespace hk  = hipblaslt_ext::experimental::jit::hipkittens;

namespace
{
    using hipblaslt_jit_test::Device;

    // BF16 1 and 128, which the K=128 sum of ones produces exactly.
    constexpr uint16_t bf16One = 0x3f80;
    constexpr uint16_t bf16K   = 0x4300;
    constexpr int      M = 256, N = 256, K = 128;
    constexpr uint64_t workspaceLimit = 64ull << 20;

    void requireGfx950()
    {
        int             device = -1;
        hipDeviceProp_t properties{};
        HIP(hipGetDevice(&device));
        HIP(hipGetDeviceProperties(&properties, device));
        JIT_REQUIRE(std::string("need gfx950, got ") + properties.gcnArchName,
                    std::string(properties.gcnArchName).rfind("gfx950", 0) == 0);
    }

    // HIPBLASLT_JIT_LIBRARY_PATH is read once per process, so the cases share one
    // scratch library.
    void scratchLibrary()
    {
        static const hipblaslt_jit_test::LibraryScratch library("hipkittens");
    }

    void transposeA(hipblasLtMatmulDesc_t desc, jit::Diagnostics& diagnostics)
    {
        const hipblasOperation_t transpose = HIPBLAS_OP_T;
        BLAS(hipblasLtMatmulDescSetAttribute(
                 desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)),
             diagnostics);
    }

    void checkKernels()
    {
        const auto& kernels = hk::detail::kernels();
        JIT_REQUIRE("expected the BF16 and FP16 kernels", kernels.size() == 2);
        bool bf16 = false, fp16 = false;
        for(const auto& kernel : kernels)
        {
            JIT_REQUIRE("kernel is not gfx950",
                        kernel.processor == hipblaslt_jit::Processor::gfx950);
            JIT_REQUIRE(kernel.name + " is not the compiled-in HIP template",
                        kernel.kind == hipblaslt_jit::Artifact::Kind::Hip
                            && kernel.bytes == hk::detail::resources().source);
            const auto solution = kernel.solution();
            JIT_REQUIRE(kernel.kernelName + " solution does not launch it",
                        solution->kernelName == kernel.kernelName
                            && solution->customKernel.name == kernel.kernelName
                            && solution->problemType.aType == kernel.type
                            && solution->problemType.dType == kernel.type);
            bf16 = bf16 || kernel.kernelName == "HK_gemm_bf16_TN_MT256x256x64_W2x4_gfx950_abi5";
            fp16 = fp16 || kernel.kernelName == "HK_gemm_f16_TN_MT256x256x64_W2x4_gfx950_abi5";
        }
        JIT_REQUIRE("missing a gfx950 kernel name", bf16 && fp16);
    }

    // A TN BF16 problem of ones: A is stored transposed. alpha is the host scalar.
    struct Problem
    {
        jit::Diagnostics                                diagnostics;
        Device                                          A, B, C, D;
        hipblasLtMatrixLayout_t                         la{}, lb{}, lc{}, ld{};
        float                                           alpha, beta = 0;
        std::shared_ptr<const jit::detail::GemmRequest> request;

        Problem(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc, int rows, int k, float a = 1)
            : A(size_t(k) * rows * 2)
            , B(size_t(k) * N * 2)
            , C(size_t(rows) * N * 2)
            , D(size_t(rows) * N * 2)
            , alpha(a)
        {
            const std::vector<uint16_t> ones(size_t(k) * std::max(rows, N), bf16One);
            HIP(hipMemcpy(A.pointer, ones.data(), size_t(k) * rows * 2, hipMemcpyHostToDevice));
            HIP(hipMemcpy(B.pointer, ones.data(), size_t(k) * N * 2, hipMemcpyHostToDevice));
            HIP(hipMemset(C.pointer, 0, size_t(rows) * N * 2));
            BLAS(hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, k, rows, k), diagnostics);
            BLAS(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, k, N, k), diagnostics);
            BLAS(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16BF, rows, N, rows), diagnostics);
            BLAS(hipblasLtMatrixLayoutCreate(&ld, HIP_R_16BF, rows, N, rows), diagnostics);
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
                                      diagnostics),
                 diagnostics);
        }
        ~Problem()
        {
            for(auto layout : {la, lb, lc, ld})
                hipblasLtMatrixLayoutDestroy(layout);
        }
        Problem(const Problem&)            = delete;
        Problem& operator=(const Problem&) = delete;
    };

    std::shared_ptr<const hipblaslt_jit::Jit> makeBackend()
    {
        jit::Diagnostics                          diagnostics;
        std::shared_ptr<const hipblaslt_jit::Jit> backend;
        BLAS(hk::createBackend(backend, diagnostics), diagnostics);
        return backend;
    }

    // getJitAlgo of the problem on a stream that is capturing. A published index
    // may be returned, and that return does not build; with none (-1) the call is
    // NOT_SUPPORTED. Neither starts generation.
    void capture(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc, int published)
    {
        const auto backend = makeBackend();
        Problem    problem(handle, desc, M, K);
        auto&      diagnostics = problem.diagnostics;
        REQUIRE((problem.request != nullptr));

        int         device = -1;
        hipStream_t stream{};
        HIP(hipGetDevice(&device));
        HIP(hipStreamCreate(&stream));
        auto owned       = problem.request->problem;
        owned.stream     = stream;
        auto   capturing = std::make_shared<jit::detail::GemmRequest>(owned);
        Device marker(4);
        HIP(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
        HIP(hipMemsetAsync(marker.pointer, 0, 4, stream));
        hipblasLtMatmulHeuristicResult_t result{};
        const auto                       started = std::chrono::steady_clock::now();
        const auto                       status
            = jit::getJitAlgo(device, capturing, backend, 64 << 20, result, diagnostics);
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
        hipGraph_t graph{};
        HIP(hipStreamEndCapture(stream, &graph));
        HIP(hipGraphDestroy(graph));
        HIP(hipStreamDestroy(stream));
        INFO(("capture status " + std::to_string(status) + ": " + diagnostics.message));
        if(published < 0)
        {
            REQUIRE((status == HIPBLAS_STATUS_NOT_SUPPORTED));
            REQUIRE((diagnostics.message.find("stream capture") != std::string::npos));
        }
        else
        {
            REQUIRE((status == HIPBLAS_STATUS_SUCCESS));
            REQUIRE((hipblaslt_ext::getIndexFromAlgo(result.algo) == published));
        }
        JIT_REQUIRE("generation started during stream capture (" + std::to_string(elapsed) + " ms)",
                    elapsed < 20000);
    }

    // BF16 of a finite float, rounded to nearest, ties to even, as Tensile does.
    uint16_t hostBf16(float value)
    {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        bits += 0x7fff + ((bits >> 16) & 1);
        return uint16_t(bits >> 16);
    }

    // One BF16 TN 256x256x128 GEMM of ones with a host alpha and beta 0. Each
    // element is alpha * 128 in FP32, which the kernel rounds to BF16. Returns
    // the published library index.
    int runGemm(hipblasLtHandle_t     handle,
                hipblasLtMatmulDesc_t desc,
                hipStream_t           stream,
                float                 alpha,
                float                 product,
                uint16_t              expected)
    {
        JIT_REQUIRE("alpha " + std::to_string(alpha) + " does not make the product "
                        + std::to_string(product),
                    alpha * float(K) == product);
        JIT_REQUIRE("host reference of " + std::to_string(product) + " is not the hand value",
                    hostBf16(product) == expected);
        const auto backend = makeBackend();
        Problem    problem(handle, desc, M, K, alpha);
        auto&      diagnostics = problem.diagnostics;
        int        device      = -1;
        HIP(hipGetDevice(&device));
        hipblasLtMatmulHeuristicResult_t result{};
        BLAS(jit::getJitAlgo(device, problem.request, backend, 64 << 20, result, diagnostics),
             diagnostics);
        const auto kernel = hipblaslt_ext::getKernelNameFromAlgo(handle, result.algo);
        JIT_REQUIRE("selected '" + kernel + "'",
                    kernel == "HK_gemm_bf16_TN_MT256x256x64_W2x4_gfx950_abi5");
        Device workspace(result.workspaceSize);
        HIP(hipMemset(problem.D.pointer, 0xff, size_t(M) * N * 2));
        BLAS(hipblasLtMatmul(handle,
                             desc,
                             &problem.alpha,
                             problem.A.pointer,
                             problem.la,
                             problem.B.pointer,
                             problem.lb,
                             &problem.beta,
                             problem.C.pointer,
                             problem.lc,
                             problem.D.pointer,
                             problem.ld,
                             &result.algo,
                             workspace.pointer,
                             result.workspaceSize,
                             stream),
             diagnostics);
        HIP(hipStreamSynchronize(stream));
        std::vector<uint16_t> out(size_t(M) * N);
        HIP(hipMemcpy(out.data(), problem.D.pointer, out.size() * 2, hipMemcpyDeviceToHost));
        const auto wrong = std::count_if(
            out.begin(), out.end(), [&](uint16_t value) { return value != expected; });
        JIT_REQUIRE("alpha " + std::to_string(alpha) + ": " + std::to_string(wrong)
                        + " of D are not BF16 bits " + std::to_string(expected) + "; D[0] is "
                        + std::to_string(out[0]),
                    wrong == 0);
        return hipblaslt_ext::getIndexFromAlgo(result.algo);
    }

    // Two batches. strideA is the A batch stride in elements; the other tensors stay packed.
    struct BatchedRequest
    {
        jit::Diagnostics                                diagnostics;
        Device                                          A, B, C, D;
        float                                           alpha, beta = 0;
        std::shared_ptr<const jit::detail::GemmRequest> request;

        BatchedRequest(hipblasLtHandle_t     handle,
                       hipblasLtMatmulDesc_t desc,
                       float                 alphaValue,
                       int64_t               strideA)
            : A(size_t(K) * M * 2)
            , B(size_t(K) * N * 2)
            , C(size_t(M) * N * 2)
            , D(size_t(M) * N * 2)
            , alpha(alphaValue)
        {
            hipblasLtMatrixLayout_t la{}, lb{}, lc{}, ld{};
            BLAS(hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, K, M, K), diagnostics);
            BLAS(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, K, N, K), diagnostics);
            BLAS(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16BF, M, N, M), diagnostics);
            BLAS(hipblasLtMatrixLayoutCreate(&ld, HIP_R_16BF, M, N, M), diagnostics);
            const int32_t batch    = 2;
            const auto    setBatch = [&](hipblasLtMatrixLayout_t layout, int64_t stride) {
                BLAS(hipblasLtMatrixLayoutSetAttribute(
                         layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)),
                     diagnostics);
                BLAS(hipblasLtMatrixLayoutSetAttribute(layout,
                                                       HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                       &stride,
                                                       sizeof(stride)),
                     diagnostics);
            };
            setBatch(la, strideA);
            setBatch(lb, int64_t(K) * N);
            setBatch(lc, int64_t(M) * N);
            setBatch(ld, int64_t(M) * N);
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
                                      diagnostics),
                 diagnostics);
            for(auto layout : {la, lb, lc, ld})
                hipblasLtMatrixLayoutDestroy(layout);
        }
    };

    template <typename Predicate>
    std::shared_ptr<Predicate> findPredicate(const TensileLite::ContractionSolution& solution)
    {
        using Problem = TensileLite::ContractionProblemGemm;
        auto all      = std::dynamic_pointer_cast<TensileLite::Predicates::And<Problem>>(
            solution.problemPredicate);
        if(!all)
            return {};
        for(const auto& child : all->value)
            if(auto found = std::dynamic_pointer_cast<Predicate>(child))
                return found;
        return {};
    }
}

TEST_CASE("the HipKittens gfx950 backend", "[jit-gpu]")
{
    requireGfx950();
    scratchLibrary();
    int device = -1;
    HIP(hipGetDevice(&device));

    checkKernels();
    hipblasLtHandle_t     handle{};
    hipblasLtMatmulDesc_t desc{};
    hipStream_t           stream{};
    jit::Diagnostics      diagnostics;
    BLAS(hipblasLtCreate(&handle), diagnostics);
    BLAS(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F), diagnostics);
    transposeA(desc, diagnostics);
    HIP(hipStreamCreate(&stream));

    capture(handle, desc, -1);
    const int published = runGemm(handle, desc, stream, 1, float(K), bf16K);
    capture(handle, desc, published);
    {
        const auto                       backend = makeBackend();
        Problem                          small(handle, desc, 128, K);
        hipblasLtMatmulHeuristicResult_t unsupported{};
        INFO(("M=128 was accepted"));
        REQUIRE((jit::getJitAlgo(device, small.request, backend, 64 << 20, unsupported, diagnostics)
                 == HIPBLAS_STATUS_NOT_SUPPORTED));
    }
    // Ties BF16 cannot hold: 129.5 rounds up to the even 130, where truncation
    // gives 129, and 128.5 down to the even 128, where round-half-up gives 129.
    runGemm(handle, desc, stream, 1.01171875f, 129.5f, 0x4302);
    runGemm(handle, desc, stream, 1.00390625f, 128.5f, 0x4300);

    static_cast<void>(hipStreamDestroy(stream));
    hipblasLtMatmulDescDestroy(desc);
    hipblasLtDestroy(handle);
}

// A handle leaves uniform summation order on, and the HipKittens solution is
// accepted under it: rows of A that repeat, across waves and work-groups, give
// bitwise-identical rows of D.
TEST_CASE("HipKittens under uniform summation order", "[jit-gpu]")
{
    requireGfx950();
    scratchLibrary();
    constexpr int rows = 512, k = 512, period = 64;
    int           device = -1;
    HIP(hipGetDevice(&device));
    hipblasLtHandle_t     handle{};
    hipblasLtMatmulDesc_t desc{};
    jit::Diagnostics      diagnostics;
    BLAS(hipblasLtCreate(&handle), diagnostics);
    BLAS(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F), diagnostics);
    transposeA(desc, diagnostics);
    {
        Problem problem(handle, desc, rows, k);
        // Values in [-1, 1), whose FP32 sums depend on their order. Row r of the
        // transposed A, column r of A, repeats row r % period.
        uint32_t   state  = 1;
        const auto random = [&] {
            state = state * 1664525u + 1013904223u;
            return hostBf16(float(int32_t(state) >> 8) / float(1 << 23));
        };
        std::vector<uint16_t> a(size_t(k) * rows), b(size_t(k) * N);
        for(int r = 0; r < period; ++r)
            for(int i = 0; i < k; ++i)
                a[size_t(r) * k + i] = random();
        for(int r = period; r < rows; ++r)
            std::copy_n(&a[size_t(r % period) * k], k, &a[size_t(r) * k]);
        std::generate(b.begin(), b.end(), random);
        HIP(hipMemcpy(problem.A.pointer, a.data(), a.size() * 2, hipMemcpyHostToDevice));
        HIP(hipMemcpy(problem.B.pointer, b.data(), b.size() * 2, hipMemcpyHostToDevice));

        hipblasLtMatmulHeuristicResult_t result{};
        BLAS(jit::getJitAlgo(
                 device, problem.request, makeBackend(), workspaceLimit, result, diagnostics),
             diagnostics);
        const auto kernel = hipblaslt_ext::getKernelNameFromAlgo(handle, result.algo);
        JIT_REQUIRE("selected '" + kernel + "'",
                    kernel == "HK_gemm_bf16_TN_MT256x256x64_W2x4_gfx950_abi5");
        Device workspace(result.workspaceSize);
        BLAS(hipblasLtMatmul(handle,
                             desc,
                             &problem.alpha,
                             problem.A.pointer,
                             problem.la,
                             problem.B.pointer,
                             problem.lb,
                             &problem.beta,
                             problem.C.pointer,
                             problem.lc,
                             problem.D.pointer,
                             problem.ld,
                             &result.algo,
                             workspace.pointer,
                             result.workspaceSize,
                             nullptr),
             diagnostics);
        HIP(hipDeviceSynchronize());
        std::vector<uint16_t> d(size_t(rows) * N);
        HIP(hipMemcpy(d.data(), problem.D.pointer, d.size() * 2, hipMemcpyDeviceToHost));
        size_t differ = 0, distinct = 0;
        for(int column = 0; column < N; ++column)
        {
            const auto* out = &d[size_t(column) * rows];
            distinct += out[0] != out[1];
            for(int r = period; r < rows; ++r)
                differ += out[r] != out[r % period];
        }
        JIT_REQUIRE("rows 0 and 1 of D are equal in every column", distinct > 0);
        JIT_REQUIRE(std::to_string(differ) + " elements of D differ from their repeated row",
                    differ == 0);
    }
    hipblasLtMatmulDescDestroy(desc);
    hipblasLtDestroy(handle);
}

// Under HIPBLASLT_JIT=2 a heuristic query on a default handle with no backend
// added returns the FP16 HipKittens kernel for a TN 256x256 problem whose K is a
// multiple of 128, the same index on a second query, and nothing for K=64.
TEST_CASE("HipKittens heuristic queries", "[jit-gpu][heuristic]")
{
    requireGfx950();
    hipblaslt_jit_test::setJitMode("2");
    scratchLibrary();
    hipblasLtHandle_t           handle{};
    hipblasLtMatmulDesc_t       desc{};
    hipblasLtMatmulPreference_t pref{};
    jit::Diagnostics            diagnostics;
    BLAS(hipblasLtCreate(&handle), diagnostics);
    BLAS(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F), diagnostics);
    transposeA(desc, diagnostics);
    BLAS(hipblasLtMatmulPreferenceCreate(&pref), diagnostics);
    BLAS(hipblasLtMatmulPreferenceSetAttribute(pref,
                                               HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &workspaceLimit,
                                               sizeof(workspaceLimit)),
         diagnostics);
    const auto query = [&](int k, hipblasLtMatmulHeuristicResult_t& result) {
        hipblasLtMatrixLayout_t la{}, lb{}, lc{}, ld{};
        BLAS(hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, k, M, k), diagnostics);
        BLAS(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, k, N, k), diagnostics);
        BLAS(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16F, M, N, M), diagnostics);
        BLAS(hipblasLtMatrixLayoutCreate(&ld, HIP_R_16F, M, N, M), diagnostics);
        int count = 0;
        BLAS(
            hipblasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, lc, ld, pref, 1, &result, &count),
            diagnostics);
        for(auto layout : {la, lb, lc, ld})
            hipblasLtMatrixLayoutDestroy(layout);
        return count;
    };
    hipblasLtMatmulHeuristicResult_t first{}, again{}, outside{};
    JIT_REQUIRE("K=128 returned no algorithm", query(K, first) == 1);
    const auto index  = hipblaslt_ext::getIndexFromAlgo(first.algo);
    const auto kernel = hipblaslt_ext::getKernelNameFromAlgo(handle, first.algo);
    JIT_REQUIRE("K=128 returned index " + std::to_string(index) + " '" + kernel + "'",
                TensileLite::isJitKernelIndex(index)
                    && kernel == "HK_gemm_f16_TN_MT256x256x64_W2x4_gfx950_abi5");
    JIT_REQUIRE("a second K=128 query returned another index",
                query(K, again) == 1 && hipblaslt_ext::getIndexFromAlgo(again.algo) == index);
    JIT_REQUIRE("K=64 returned an algorithm", query(64, outside) == 0);
    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatmulDescDestroy(desc);
    hipblasLtDestroy(handle);
}

TEST_CASE("HipKittens predicates reject zero alpha and a truncating stride", "[jit-cpu]")
{
    using Problem       = TensileLite::ContractionProblemGemm;
    const auto solution = hk::detail::kernels().front().solution();
    auto       alpha = findPredicate<TensileLite::Predicates::Contraction::AlphaNotZero>(*solution);
    auto       stride
        = findPredicate<TensileLite::Predicates::Contraction::StrideFitsInUint32>(*solution);
    JIT_REQUIRE("the solution does not reject zero alpha", alpha != nullptr);
    JIT_REQUIRE("the solution does not reject a stride that does not fit in uint32",
                stride != nullptr);

    const auto bf16    = rocisa::DataType::BFloat16;
    auto       problem = Problem::GEMM_Strides(true,
                                         false,
                                         bf16,
                                         bf16,
                                         bf16,
                                         bf16,
                                         256,
                                         256,
                                         128,
                                         2,
                                         128,
                                         128 * 256,
                                         128,
                                         128 * 256,
                                         256,
                                         256 * 256,
                                         256,
                                         256 * 256,
                                         0.0);
    problem.setRuntimeAlpha(1.0);
    JIT_REQUIRE("alpha 1 was rejected", (*alpha)(problem));
    problem.setRuntimeAlpha(0.0);
    JIT_REQUIRE("alpha 0 was accepted", !(*alpha)(problem));

    // stridedBatched stays true for a pointer-array batch. Batch mode is separate.
    auto batch = findPredicate<TensileLite::Predicates::Contraction::BatchModeEqual>(*solution);
    JIT_REQUIRE("the solution does not record its batch mode", batch != nullptr);
    problem.setStridedBatched(true);
    problem.setBatchMode(Problem::BATCHMODE::STRIDED);
    JIT_REQUIRE("a strided batch was rejected", (*batch)(problem));
    problem.setBatchMode(Problem::BATCHMODE::POINTER_ARRAY);
    JIT_REQUIRE("a pointer-array batch matched a strided solution", !(*batch)(problem));

    JIT_REQUIRE("a packed batch stride was rejected", (*stride)(problem));
    const size_t huge       = (size_t(1) << 32) + 32768;
    auto         truncating = Problem::GEMM_Strides(true,
                                            false,
                                            bf16,
                                            bf16,
                                            bf16,
                                            bf16,
                                            256,
                                            256,
                                            128,
                                            2,
                                            128,
                                            huge,
                                            128,
                                            128 * 256,
                                            256,
                                            256 * 256,
                                            256,
                                            256 * 256,
                                            0.0);
    JIT_REQUIRE("strideA 2^32+32768 was accepted", !(*stride)(truncating));
}

// Selects the kernel, then asks again with alpha 0 and with a batch stride that
// does not fit in the uint32 argument. Both must miss the published solution.
TEST_CASE("a published HipKittens solution rejects zero alpha and a truncating stride", "[jit-gpu]")
{
    requireGfx950();
    scratchLibrary();
    int device = -1;
    HIP(hipGetDevice(&device));
    hipblasLtHandle_t     handle{};
    hipblasLtMatmulDesc_t desc{};
    hipStream_t           stream{};
    jit::Diagnostics      diagnostics;
    BLAS(hipblasLtCreate(&handle), diagnostics);
    BLAS(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F), diagnostics);
    transposeA(desc, diagnostics);
    HIP(hipStreamCreate(&stream));

    const auto                       backend = makeBackend();
    const int64_t                    packed  = int64_t(K) * M;
    BatchedRequest                   good(handle, desc, 1.f, packed);
    hipblasLtMatmulHeuristicResult_t published{};
    BLAS(
        jit::getJitAlgo(device, good.request, backend, workspaceLimit, published, good.diagnostics),
        good.diagnostics);
    const auto algo = published.algo;

    BatchedRequest                   zero(handle, desc, 0.f, packed);
    hipblasLtMatmulHeuristicResult_t zeroResult{};
    const auto                       zeroStatus = jit::getJitAlgo(
        device, zero.request, backend, workspaceLimit, zeroResult, zero.diagnostics);
    JIT_REQUIRE("alpha 0 was selected: " + zero.diagnostics.message,
                zeroStatus == HIPBLAS_STATUS_NOT_SUPPORTED
                    && zero.diagnostics.message.find("zero-alpha") != std::string::npos);

    hipblasLtMatrixLayout_t la{}, lb{}, lc{}, ld{};
    BLAS(hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, K, M, K), diagnostics);
    BLAS(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, K, N, K), diagnostics);
    BLAS(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16BF, M, N, M), diagnostics);
    BLAS(hipblasLtMatrixLayoutCreate(&ld, HIP_R_16BF, M, N, M), diagnostics);
    const int32_t batch     = 2;
    const auto    setPacked = [&](hipblasLtMatrixLayout_t layout, int64_t stride) {
        BLAS(hipblasLtMatrixLayoutSetAttribute(
                 layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)),
             diagnostics);
        BLAS(hipblasLtMatrixLayoutSetAttribute(
                 layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)),
             diagnostics);
    };
    setPacked(la, packed);
    setPacked(lb, int64_t(K) * N);
    setPacked(lc, int64_t(M) * N);
    setPacked(ld, int64_t(M) * N);
    Device     workspace(published.workspaceSize);
    const auto launched = hipblasLtMatmul(handle,
                                          desc,
                                          &zero.alpha,
                                          zero.A.pointer,
                                          la,
                                          zero.B.pointer,
                                          lb,
                                          &zero.beta,
                                          zero.C.pointer,
                                          lc,
                                          zero.D.pointer,
                                          ld,
                                          &algo,
                                          workspace.pointer,
                                          published.workspaceSize,
                                          stream);
    JIT_REQUIRE("a published solution launched with alpha 0", launched != HIPBLAS_STATUS_SUCCESS);
    for(auto layout : {la, lb, lc, ld})
        hipblasLtMatrixLayoutDestroy(layout);

    // The host API rejects this stride before a request exists. Matching the
    // published solution is what still has to reject it, so the request is the
    // published problem with only the A batch stride changed.
    const int64_t huge         = (int64_t(1) << 32) + 32768;
    auto          wideProblem  = good.request->problem;
    wideProblem.batch_stride_a = huge;
    const auto wideRequest     = std::make_shared<jit::detail::GemmRequest>(wideProblem);
    hipblasLtMatmulHeuristicResult_t reused{};
    jit::Diagnostics                 strideDiag;
    const auto                       strideStatus
        = jit::getJitAlgo(device, wideRequest, backend, workspaceLimit, reused, strideDiag);
    JIT_REQUIRE("a truncating batch stride matched the published solution: " + strideDiag.message,
                strideStatus == HIPBLAS_STATUS_NOT_SUPPORTED);

    HIP(hipStreamDestroy(stream));
    hipblasLtMatmulDescDestroy(desc);
    hipblasLtDestroy(handle);
}

// Selects the FP16 kernel with half alpha 1, then reuses that algorithm with
// half alpha 0 and null A and B. Generation is not enough: the published
// solution has to reject the second call.
TEST_CASE("a published HipKittens solution rejects half-precision alpha zero", "[jit-gpu]")
{
    requireGfx950();
    scratchLibrary();
    int device = -1;
    HIP(hipGetDevice(&device));
    hipblasLtHandle_t     handle{};
    hipblasLtMatmulDesc_t desc{};
    jit::Diagnostics      diagnostics;
    BLAS(hipblasLtCreate(&handle), diagnostics);
    BLAS(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_16F, HIP_R_16F), diagnostics);
    transposeA(desc, diagnostics);

    hipblasLtHalf           alphaOne  = hipblasLtHalf(1.f);
    hipblasLtHalf           alphaZero = hipblasLtHalf(0.f);
    hipblasLtHalf           beta      = hipblasLtHalf(0.f);
    Device                  A(size_t(K) * M * 2);
    Device                  B(size_t(K) * N * 2);
    Device                  C(size_t(M) * N * 2);
    Device                  D(size_t(M) * N * 2);
    hipblasLtMatrixLayout_t la{}, lb{}, lc{}, ld{};
    BLAS(hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, K, M, K), diagnostics);
    BLAS(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, K, N, K), diagnostics);
    BLAS(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16F, M, N, M), diagnostics);
    BLAS(hipblasLtMatrixLayoutCreate(&ld, HIP_R_16F, M, N, M), diagnostics);

    std::shared_ptr<const jit::detail::GemmRequest> request;
    BLAS(jit::makeGemmRequest(handle,
                              desc,
                              &alphaOne,
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
                              diagnostics),
         diagnostics);
    const auto                       backend = makeBackend();
    hipblasLtMatmulHeuristicResult_t published{};
    BLAS(jit::getJitAlgo(device, request, backend, workspaceLimit, published, diagnostics),
         diagnostics);
    const auto kernel = hipblaslt_ext::getKernelNameFromAlgo(handle, published.algo);
    JIT_REQUIRE("selected '" + kernel + "'",
                kernel == "HK_gemm_f16_TN_MT256x256x64_W2x4_gfx950_abi5");

    // Do not launch. The kernel faults on alpha 0. The decoded problem has to
    // carry runtime alpha 0, and the published solution has to reject that
    // value while K stays 128.
    std::shared_ptr<const jit::detail::GemmRequest> zeroRequest;
    BLAS(jit::makeGemmRequest(handle,
                              desc,
                              &alphaZero,
                              A.pointer,
                              la,
                              B.pointer,
                              lb,
                              &beta,
                              C.pointer,
                              lc,
                              D.pointer,
                              ld,
                              zeroRequest,
                              diagnostics),
         diagnostics);
    const auto decoded = hipblaslt_jit::RocblasltToTensileContractionProblemGemm(*zeroRequest);
    JIT_REQUIRE("half alpha 0 was stored as " + std::to_string(decoded.runtimeAlpha()),
                decoded.runtimeAlpha() == 0.0);
    auto kept = hipblaslt_jit::RocblasltToTensileContractionProblemGemm(*request);
    kept.setRuntimeAlpha(decoded.runtimeAlpha());
    std::shared_ptr<TensileLite::ContractionSolution> publishedSolution;
    for(const auto& entry : hk::detail::kernels())
        if(entry.kernelName == "HK_gemm_f16_TN_MT256x256x64_W2x4_gfx950_abi5")
            publishedSolution = entry.solution();
    JIT_REQUIRE("the FP16 solution is missing", publishedSolution != nullptr);
    JIT_REQUIRE("half alpha 0 matched the published solution",
                !(*publishedSolution->problemPredicate)(kept));

    for(auto layout : {la, lb, lc, ld})
        hipblasLtMatrixLayoutDestroy(layout);
    hipblasLtMatmulDescDestroy(desc);
    hipblasLtDestroy(handle);
}

// Publishes the strided FP16 solution, then queries the same shape as a
// pointer-array batch. The library match has to exclude it. No kernel is launched.
TEST_CASE("a published strided HipKittens solution excludes a pointer-array query", "[jit-gpu]")
{
    requireGfx950();
    scratchLibrary();
    int device = -1;
    HIP(hipGetDevice(&device));
    hipblasLtHandle_t     handle{};
    hipblasLtMatmulDesc_t desc{};
    jit::Diagnostics      diagnostics;
    BLAS(hipblasLtCreate(&handle), diagnostics);
    BLAS(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_16F, HIP_R_16F), diagnostics);
    transposeA(desc, diagnostics);

    hipblasLtHalf           alpha = hipblasLtHalf(1.f);
    hipblasLtHalf           beta  = hipblasLtHalf(0.f);
    Device                  A(size_t(K) * M * 2);
    Device                  B(size_t(K) * N * 2);
    Device                  C(size_t(M) * N * 2);
    Device                  D(size_t(M) * N * 2);
    hipblasLtMatrixLayout_t la{}, lb{}, lc{}, ld{};
    BLAS(hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, K, M, K), diagnostics);
    BLAS(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, K, N, K), diagnostics);
    BLAS(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16F, M, N, M), diagnostics);
    BLAS(hipblasLtMatrixLayoutCreate(&ld, HIP_R_16F, M, N, M), diagnostics);

    std::shared_ptr<const jit::detail::GemmRequest> request;
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
                              diagnostics),
         diagnostics);
    const auto                       backend = makeBackend();
    hipblasLtMatmulHeuristicResult_t published{};
    BLAS(jit::getJitAlgo(device, request, backend, workspaceLimit, published, diagnostics),
         diagnostics);
    const auto publishedIndex = hipblaslt_ext::getIndexFromAlgo(published.algo);

    const int32_t mode = HIPBLASLT_BATCH_MODE_POINTER_ARRAY;
    for(auto layout : {la, lb, lc, ld})
        BLAS(hipblasLtMatrixLayoutSetAttribute(
                 layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &mode, sizeof(mode)),
             diagnostics);
    std::shared_ptr<const jit::detail::GemmRequest> pointerArray;
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
                              pointerArray,
                              diagnostics),
         diagnostics);
    hipblasLtMatmulHeuristicResult_t matched{};
    jit::Diagnostics                 queryDiag;
    const auto                       status
        = jit::getJitAlgo(device, pointerArray, backend, workspaceLimit, matched, queryDiag);
    const bool returnedPublished
        = status == HIPBLAS_STATUS_SUCCESS
          && hipblaslt_ext::getIndexFromAlgo(matched.algo) == publishedIndex;
    JIT_REQUIRE("a pointer-array query returned the strided solution: " + queryDiag.message,
                !returnedPublished);

    for(auto layout : {la, lb, lc, ld})
        hipblasLtMatrixLayoutDestroy(layout);
    hipblasLtMatmulDescDestroy(desc);
    hipblasLtDestroy(handle);
}
