/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// Test that batch pointer arrays are correctly re-uploaded when the problem
// changes between calls to prepareGPUInputs().
//
// Regression test for the bug where initializeGPUBatchedInputs() was moved
// inside the initial-setup branch of prepareGPUInputs(), causing the fast path
// to skip batch pointer recomputation when m_gpuInit is already true.  Since
// m_gpuInit is never reset between problems, the fast path fires for a second
// problem and returns stale batch pointers computed from the first problem's
// strides.

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/TensorDescriptor.hpp>

#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{

    // ---------------------------------------------------------------------------
    // Helper: build a minimal po::variables_map with defaults suitable for
    // constructing ClientProblemFactory + DataInitialization.
    //
    // The largest problem in |problemSizes| determines the GPU buffer sizes, so
    // it must be >= all problems that will later be passed to prepareGPUInputs().
    // ---------------------------------------------------------------------------
    po::variables_map buildArgs(std::vector<std::vector<size_t>> problemSizes)
    {
        po::variables_map args;

        auto set = [&](std::string key, std::any val) {
            args[key].value() = std::move(val);
        };

        // --- ClientProblemFactory ---
        set("problem-identifier",
            std::any(std::string("Contraction_l_Alik_Bjlk_Cijk_Dijk")));
        set("problem-size", std::any(problemSizes));

        // Data types — all Float for simplicity
        auto f32 = rocisa::DataType::Float;
        set("type", std::any(f32));
        set("a-type", std::any(f32));
        set("b-type", std::any(f32));
        set("c-type", std::any(f32));
        set("d-type", std::any(f32));
        set("e-type", std::any(rocisa::DataType::None));
        set("amaxD-type", std::any(rocisa::DataType::None));
        set("alpha-type", std::any(f32));
        set("beta-type", std::any(f32));

        set("strided-batched", std::any(false));
        set("grouped-gemm", std::any(false));
        set("high-precision-accumulate", std::any(false));
        set("deterministic-mode", std::any(false));
        set("c-equal-d", std::any(false));
        set("kernel-language", std::any(KernelLanguage::Any));
        set("performance-metric", std::any(PerformanceMetric::DeviceEfficiency));
        set("metadata-layout", std::any(int32_t(0)));
        set("mx-a-block", std::any(int(0)));
        set("mx-b-block", std::any(int(0)));

        // Tensor ops — empty
        TensorOps nop;
        set("a-ops", std::any(nop));
        set("b-ops", std::any(nop));
        set("c-ops", std::any(nop));
        set("d-ops", std::any(nop));

        // Optional args ClientProblemFactory checks via args.count() — set
        // explicitly so the factory constructor doesn't crash.
        set("max-workspace-size", std::any(size_t(32 * 1024 * 1024)));
        set("use-bias", std::any(int(0)));
        set("bias-source", std::any(int(3)));
        set("use-scaleAB", std::any(std::string("")));
        set("use-scaleCD", std::any(false));
        set("use-scaleAlphaVec", std::any(int(0)));
        set("use-e", std::any(false));
        set("use-gradient", std::any(false));
        set("output-amaxD", std::any(false));
        set("bias-type-args",
            std::any(std::vector<rocisa::DataType>{rocisa::DataType::None}));
        set("factor-dim-args", std::any(std::vector<int>{0}));
        set("activation-type", std::any(ActivationType::None));
        set("activation-no-guard", std::any(false));
        set("activation-enum-args",
            std::any(std::vector<ActivationType>{ActivationType::None}));
        set("activation-compute-type", std::any(rocisa::DataType::None));
        set("compute-input-type-A", std::any(rocisa::DataType::None));
        set("compute-input-type-B", std::any(rocisa::DataType::None));
        set("f32-xdl-math-op", std::any(rocisa::DataType::None));
        set("swizzle-tensor-a", std::any(false));
        set("swizzle-tensor-b", std::any(false));
        set("use-user-args", std::any(false));

        // --- DataInitialization ---
        set("sparse", std::any(int(0)));
        set("num-elements-to-validate", std::any(int(0)));
        set("pristine-on-gpu", std::any(true));
        set("prune-mode", std::any(PruneSparseMode::PruneRandom));
        set("rotating-buffer-size", std::any(int32_t(0)));
        set("rotating-buffer-mode", std::any(int32_t(0)));
        set("bounds-check", std::any(BoundsCheckMode::Disable));

        // Init modes for tensors
        set("init-a", std::any(InitMode::Random));
        set("init-b", std::any(InitMode::Random));
        set("init-c", std::any(InitMode::Random));
        set("init-d", std::any(InitMode::Zero));
        set("init-e", std::any(InitMode::Zero));
        set("init-bias", std::any(InitMode::One));
        set("init-scaleA", std::any(InitMode::Two));
        set("init-scaleB", std::any(InitMode::Two));
        set("init-scaleC", std::any(InitMode::Two));
        set("init-scaleD", std::any(InitMode::Two));
        set("init-scaleAlphaVec", std::any(InitMode::One));
        set("init-alpha", std::any(InitMode::Two));
        set("init-beta", std::any(InitMode::Two));

        return args;
    }

    // ---------------------------------------------------------------------------
    // Create a batched GEMM problem using GEMM_Strides.
    //
    // Not transposed, Float, with the given M, N, K, batchSize.
    // Strides are chosen as tight packing:  lda = M, aStride = M*K, etc.
    // ---------------------------------------------------------------------------
    ContractionProblemGemm makeBatchedProblem(size_t m, size_t n, size_t k, size_t batch)
    {
        auto f32 = rocisa::DataType::Float;

        size_t lda     = m;
        size_t aStride = m * k;
        size_t ldb     = k;
        size_t bStride = k * n;
        size_t ldc     = m;
        size_t cStride = m * n;
        size_t ldd     = m;
        size_t dStride = m * n;

        auto problem = ContractionProblemGemm::GEMM_Strides(
            false, false, f32, f32, f32, f32,
            m, n, k, batch,
            lda, aStride,
            ldb, bStride,
            ldc, cStride,
            ldd, dStride,
            0.0);
        problem.setStridedBatched(false);
        return problem;
    }

} // anonymous namespace

// ---------------------------------------------------------------------------
// The actual test.
//
// We call prepareGPUInputs() for problem1, which takes the slow path
// (m_gpuInit=false -> true) and correctly sets up batch pointer arrays.
// Then we call prepareGPUInputs() for problem2, which takes the fast path
// (m_gpuInit=true, boundsCheck=Disable, !problemDependentData).
//
// The batch pointer array for tensor A encodes:
//     batchA[i] = base_A + i * aStride * sizeof(float)
//
// If the fast path correctly re-uploads batch pointers, the stride between
// consecutive entries in batchA should match problem2's aStride.  If it
// incorrectly skips the re-upload, the stride matches problem1's aStride.
// ---------------------------------------------------------------------------
TEST(BatchPointerReset, StalePointersAcrossProblems)
{
    constexpr size_t BATCH = 4;

    // Problem 1: small — aStride = 32*32 = 1024 elements = 4096 bytes
    auto p1 = makeBatchedProblem(32, 32, 32, BATCH);

    // Problem 2: larger — aStride = 64*64 = 4096 elements = 16384 bytes
    auto p2 = makeBatchedProblem(64, 64, 64, BATCH);

    // Factory problem must be at least as large as the largest test problem
    // so that the allocated GPU buffers are big enough.
    // Use M=64, N=64, batch=4, K=64 — index order is {i, j, l, k}.
    auto args = buildArgs({{64, 64, BATCH, 64}});

    ClientProblemFactory factory(args);
    DataInitialization   dataInit(args, factory);

    // --- Call 1: slow path (m_gpuInit = false -> true) ---
    auto inputs1 = dataInit.prepareGPUInputs(p1);

    // Read back batchA after problem 1
    auto* ci1 = dynamic_cast<ContractionInputs*>(inputs1.get());
    ASSERT_NE(ci1, nullptr);
    ASSERT_NE(ci1->batchA, nullptr);

    void* batchA_p1[BATCH];
    HIP_CHECK_EXC(hipMemcpy(
        batchA_p1, ci1->batchA, BATCH * sizeof(void*), hipMemcpyDeviceToHost));

    // Sanity: consecutive entries should differ by p1's aStride.
    // initGPUBatchedInput adds element-strides directly to uint8_t*, so
    // the observed byte-offset equals the element-stride value.
    {
        ptrdiff_t stride1
            = (uint8_t*)batchA_p1[1] - (uint8_t*)batchA_p1[0];
        ptrdiff_t expected1 = ptrdiff_t(32 * 32); // aStride in elements
        EXPECT_EQ(stride1, expected1)
            << "Problem 1 batch pointer stride mismatch";
    }

    // --- Call 2: fast path (m_gpuInit=true, boundsCheck=Disable,
    //     !problemDependentData).
    //     Simulate what main.cpp does when the problem changes:
    //     preProblem resets m_batchInit so batch pointers are re-uploaded.
    dataInit.preProblem(nullptr);
    auto inputs2 = dataInit.prepareGPUInputs(p2);

    auto* ci2 = dynamic_cast<ContractionInputs*>(inputs2.get());
    ASSERT_NE(ci2, nullptr);
    ASSERT_NE(ci2->batchA, nullptr);

    void* batchA_p2[BATCH];
    HIP_CHECK_EXC(hipMemcpy(
        batchA_p2, ci2->batchA, BATCH * sizeof(void*), hipMemcpyDeviceToHost));

    // The stride between consecutive batchA entries must match problem 2's
    // aStride (64*64 = 4096 elements), NOT problem 1's aStride (32*32 = 1024).
    {
        ptrdiff_t stride2
            = (uint8_t*)batchA_p2[1] - (uint8_t*)batchA_p2[0];
        ptrdiff_t expected2 = ptrdiff_t(64 * 64); // aStride in elements

        EXPECT_EQ(stride2, expected2)
            << "After switching to problem 2, the batch pointer stride should "
               "reflect problem 2's aStride (" << expected2 << "), "
               "but got " << stride2 << ".  This indicates "
               "initializeGPUBatchedInputs was not re-invoked on the fast path.";
    }
}
