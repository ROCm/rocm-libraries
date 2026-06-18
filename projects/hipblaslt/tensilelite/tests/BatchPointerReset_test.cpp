// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

#include "DataInitializationTestUtils.hpp"
#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    using TensileLite::testing::makeBatchedProblem;
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
//     batchA[i] = base_A + i * aStride
//
// DataInitialization applies the tensor stride directly to a uint8_t* base
// pointer, so the observed delta between consecutive batchA entries is the
// tensor stride value in bytes, not stride * sizeof(float).
//
// If the fast path correctly re-uploads batch pointers, the delta between
// consecutive entries in batchA should match problem2's aStride. If it
// incorrectly skips the re-upload, the delta matches problem1's aStride.
// ---------------------------------------------------------------------------
TEST(BatchPointerReset, StalePointersAcrossProblems)
{
    constexpr size_t BATCH = 4;

    // Problem 1: small — A tensor stride delta = 32*32 bytes.
    auto p1 = makeBatchedProblem(32, 32, 32, BATCH);

    // Problem 2: larger — A tensor stride delta = 64*64 bytes.
    auto p2 = makeBatchedProblem(64, 64, 64, BATCH);

    // Factory problem must be at least as large as the largest test problem
    // so that the allocated GPU buffers are big enough.
    // Use M=64, N=64, batch=4, K=64 — index order is {i, j, l, k}.
    auto args = TensileLite::testing::buildBaseDataInitArgs({{64, 64, BATCH, 64}});

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

    // Sanity: consecutive entries should differ by p1's aStride in bytes.
    {
        ptrdiff_t stride1
            = (uint8_t*)batchA_p1[1] - (uint8_t*)batchA_p1[0];
        ptrdiff_t expected1 = ptrdiff_t(32 * 32); // aStride in bytes
        EXPECT_EQ(stride1, expected1)
            << "Problem 1 batch pointer stride mismatch";
    }

    // --- Call 2: fast path (m_gpuInit=true, boundsCheck=Disable,
    //     !problemDependentData).
    //     Simulate what main.cpp does when the problem changes: preProblem()
    //     resets m_batchInitProblem to nullptr, so the pointer-identity check
    //     in prepareGPUInputsInternal fires and re-uploads batch pointers.
    dataInit.preProblem(nullptr);
    auto inputs2 = dataInit.prepareGPUInputs(p2);

    auto* ci2 = dynamic_cast<ContractionInputs*>(inputs2.get());
    ASSERT_NE(ci2, nullptr);
    ASSERT_NE(ci2->batchA, nullptr);

    void* batchA_p2[BATCH];
    HIP_CHECK_EXC(hipMemcpy(
        batchA_p2, ci2->batchA, BATCH * sizeof(void*), hipMemcpyDeviceToHost));

    // The delta between consecutive batchA entries must match problem 2's
    // aStride (64*64 = 4096 bytes), NOT problem 1's aStride (32*32 = 1024 bytes).
    {
        ptrdiff_t stride2
            = (uint8_t*)batchA_p2[1] - (uint8_t*)batchA_p2[0];
        ptrdiff_t expected2 = ptrdiff_t(64 * 64); // aStride in bytes

        EXPECT_EQ(stride2, expected2)
            << "After switching to problem 2, the batch pointer stride should "
               "reflect problem 2's aStride (" << expected2 << " bytes), "
               "but got " << stride2 << ".  This indicates "
               "initializeGPUBatchedInputs was not re-invoked on the fast path.";
    }
}

// ---------------------------------------------------------------------------
// Structural invariant: switching to a different ContractionProblemGemm
// object must trigger batch-pointer re-upload even when preProblem() is not
// called in between.
//
// With the old bool m_batchInit approach, skipping preProblem() leaves
// m_batchInit=true, so initializeGPUBatchedInputs is skipped and the caller
// gets batch pointers from the first problem's strides — silently wrong.
//
// The fix replaces the bool with ContractionProblemGemm const*
// m_batchInitProblem and checks (m_batchInitProblem != &problem) in
// prepareGPUInputsInternal.  Because p1 and p2 are distinct objects, their
// addresses differ, so the check fires and re-uploads correctly — no
// preProblem() needed to make it work.
//
// This test therefore fails with the boolean implementation and passes after
// the pointer-identity fix.  It is the regression test for the structural
// guarantee, not just the call-site-discipline guarantee.
// ---------------------------------------------------------------------------
TEST(BatchPointerReset, StructuralReinitWithoutPreProblem)
{
    constexpr size_t BATCH = 4;

    // p1: small problem — A tensor stride delta = 32*32 bytes.
    auto p1 = makeBatchedProblem(32, 32, 32, BATCH);
    // p2: larger problem — A tensor stride delta = 64*64 bytes.
    auto p2 = makeBatchedProblem(64, 64, 64, BATCH);

    // Buffer must be sized for the largest problem.
    auto args = TensileLite::testing::buildBaseDataInitArgs({{64, 64, BATCH, 64}});

    ClientProblemFactory factory(args);
    DataInitialization   dataInit(args, factory);

    // First call: slow path — initialises batch pointers for p1.
    dataInit.prepareGPUInputs(p1);

    // Second call: switch to p2 WITHOUT calling preProblem().
    // The structural pointer-identity check must detect the different problem
    // object and re-upload batch pointers for p2.
    auto inputs2 = dataInit.prepareGPUInputs(p2);

    auto* ci2 = dynamic_cast<ContractionInputs*>(inputs2.get());
    ASSERT_NE(ci2, nullptr);
    ASSERT_NE(ci2->batchA, nullptr);

    void* batchA_p2[BATCH];
    HIP_CHECK_EXC(hipMemcpy(
        batchA_p2, ci2->batchA, BATCH * sizeof(void*), hipMemcpyDeviceToHost));

    ptrdiff_t stride   = (uint8_t*)batchA_p2[1] - (uint8_t*)batchA_p2[0];
    ptrdiff_t expected = ptrdiff_t(64 * 64); // p2's aStride in bytes
    EXPECT_EQ(stride, expected)
        << "Batch pointer stride must match p2 (" << expected
        << " bytes) even without an intervening preProblem() call. "
           "Got " << stride << ". This means initializeGPUBatchedInputs was "
           "skipped — the structural pointer-identity guard is missing.";
}
