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
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/TensorDescriptor.hpp>

#include <variant>

#include "DataInitializationTestUtils.hpp"
#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    using TensileLite::testing::makeBatchedProblem;
    using TensileLite::testing::makePlainProblem;

    class PredicateDataInitialization : public DataInitialization
    {
    public:
        using DataInitialization::DataInitialization;
        using DataInitialization::cpuInputsNeedRefresh;
        using DataInitialization::gpuInputsPreparedFor;
        using DataInitialization::shouldRefreshMXForSolution;
    };

    ::testing::AssertionResult hasHipDevice()
    {
        int        deviceCount = 0;
        hipError_t err         = hipGetDeviceCount(&deviceCount);
        if(err != hipSuccess)
        {
            return ::testing::AssertionFailure()
                   << "hipGetDeviceCount failed: " << hipGetErrorString(err);
        }
        if(deviceCount <= 0)
        {
            return ::testing::AssertionFailure() << "No HIP devices available";
        }
        return ::testing::AssertionSuccess();
    }

    void setBatchPointerResetPolicyArgs(Client::po::variables_map& args)
    {
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "max-enqueues-per-sync",
                                                     std::any(int(-1)));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "min-flops-per-sync",
                                                     std::any(size_t(0)));
        TensileLite::testing::detail::setDataInitArg(args, "print-tensor-a", std::any(false));
        TensileLite::testing::detail::setDataInitArg(args, "print-tensor-b", std::any(false));
        TensileLite::testing::detail::setDataInitArg(args, "print-tensor-c", std::any(false));
        TensileLite::testing::detail::setDataInitArg(args, "print-tensor-d", std::any(false));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "print-tensor-ref",
                                                     std::any(false));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "print-tensor-bias",
                                                     std::any(false));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "print-tensor-amaxd",
                                                     std::any(false));
    }

    Client::po::variables_map
        makeBatchPointerResetArgs(std::vector<std::vector<size_t>> problemSizes)
    {
        auto args = TensileLite::testing::buildBaseDataInitArgs(std::move(problemSizes));
        setBatchPointerResetPolicyArgs(args);
        return args;
    }

    Client::po::variables_map makeRingBatchPointerArgs(
        std::vector<std::vector<size_t>> problemSizes, int elementsToValidate = 1)
    {
        auto args = TensileLite::testing::buildRingArgs(std::move(problemSizes),
                                                        elementsToValidate);
        setBatchPointerResetPolicyArgs(args);
        return args;
    }

    Client::po::variables_map makeConstantCachingArgs()
    {
        auto args = makeBatchPointerResetArgs({{32, 32, 32}});
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "activation-type",
                                                     std::any(ActivationType::Clippedrelu));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "activation-enum-args",
                                                     std::any(std::vector<ActivationType>{}));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "activation-compute-type",
                                                     std::any(rocisa::DataType::Float));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "init-alpha",
                                                     std::any(Client::InitMode::Two));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "init-beta",
                                                     std::any(Client::InitMode::Two));
        TensileLite::testing::detail::setDataInitArg(
            args,
            "activation-additional-args",
            std::any(std::vector<std::vector<double>>{{3.25, -1.5}}));
        return args;
    }

    template <typename T>
    T readDeviceValue(T const* devicePtr, size_t index)
    {
        T value{};
        HIP_CHECK_EXC(
            hipMemcpy(&value, devicePtr + index, sizeof(T), hipMemcpyDeviceToHost));
        return value;
    }

#if HIPBLASLT_ENABLE_MXDATAGENERATOR
    ContractionProblemGemm makeMXBatchedProblem(size_t M,
                                                size_t N,
                                                size_t K,
                                                size_t batch,
                                                int    mxBlock)
    {
        auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                             false,
                                                             rocisa::DataType::Float4,
                                                             rocisa::DataType::Float4,
                                                             rocisa::DataType::BFloat16,
                                                             rocisa::DataType::BFloat16,
                                                             M,
                                                             N,
                                                             K,
                                                             batch,
                                                             M,
                                                             M * K,
                                                             K,
                                                             K * N,
                                                             M,
                                                             M * N,
                                                             M,
                                                             M * N,
                                                             0.0);
        problem.setStridedBatched(false);
        problem.setMXScaleA(rocisa::DataType::E8, mxBlock);
        problem.setMXScaleB(rocisa::DataType::E8, mxBlock);
        return problem;
    }

    Client::po::variables_map
        makeMXPredicateArgs(std::vector<std::vector<size_t>> problemSizes, int mxBlock)
    {
        auto args = makeBatchPointerResetArgs(std::move(problemSizes));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "a-type",
                                                     std::any(rocisa::DataType::Float4));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "b-type",
                                                     std::any(rocisa::DataType::Float4));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "c-type",
                                                     std::any(rocisa::DataType::BFloat16));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "d-type",
                                                     std::any(rocisa::DataType::BFloat16));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "mx-a-block",
                                                     std::any(mxBlock));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "mx-b-block",
                                                     std::any(mxBlock));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "mx-scale-format",
                                                     std::any(1));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "init-a",
                                                     std::any(Client::InitMode::SerialDim0));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "init-b",
                                                     std::any(Client::InitMode::SerialDim0));
        return args;
    }
#endif
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
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    constexpr size_t BATCH = 4;

    // Problem 1: small — A tensor stride delta = 32*32 bytes.
    auto p1 = makeBatchedProblem(32, 32, 32, BATCH);

    // Problem 2: larger — A tensor stride delta = 64*64 bytes.
    auto p2 = makeBatchedProblem(64, 64, 64, BATCH);

    // Factory problem must be at least as large as the largest test problem
    // so that the allocated GPU buffers are big enough.
    // Use M=64, N=64, batch=4, K=64 — index order is {i, j, l, k}.
    auto args = makeBatchPointerResetArgs({{64, 64, BATCH, 64}});

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

TEST(BatchPointerReset, SameObjectMutationReinitializesBatchPointers)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    constexpr size_t BATCH = 4;

    auto problem = makeBatchedProblem(32, 32, 32, BATCH);
    auto args    = makeBatchPointerResetArgs({{64, 64, BATCH, 64}});

    ClientProblemFactory factory(args);
    DataInitialization   dataInit(args, factory);

    auto inputs1 = dataInit.prepareGPUInputs(problem);
    auto* ci1    = dynamic_cast<ContractionInputs*>(inputs1.get());
    ASSERT_NE(ci1, nullptr);
    ASSERT_NE(ci1->batchA, nullptr);

    void* batchA_p1[BATCH];
    HIP_CHECK_EXC(
        hipMemcpy(batchA_p1, ci1->batchA, BATCH * sizeof(void*), hipMemcpyDeviceToHost));

    ptrdiff_t stride1   = (uint8_t*)batchA_p1[1] - (uint8_t*)batchA_p1[0];
    ptrdiff_t expected1 = ptrdiff_t(32 * 32);
    EXPECT_EQ(stride1, expected1);

    auto const* stableProblemAddress = &problem;
    problem = makeBatchedProblem(64, 64, 64, BATCH);
    ASSERT_EQ(&problem, stableProblemAddress);

    auto inputs2 = dataInit.prepareGPUInputs(problem);
    auto* ci2    = dynamic_cast<ContractionInputs*>(inputs2.get());
    ASSERT_NE(ci2, nullptr);
    ASSERT_NE(ci2->batchA, nullptr);

    void* batchA_p2[BATCH];
    HIP_CHECK_EXC(
        hipMemcpy(batchA_p2, ci2->batchA, BATCH * sizeof(void*), hipMemcpyDeviceToHost));

    ptrdiff_t stride2   = (uint8_t*)batchA_p2[1] - (uint8_t*)batchA_p2[0];
    ptrdiff_t expected2 = ptrdiff_t(64 * 64);
    EXPECT_EQ(stride2, expected2)
        << "Batch pointer stride must follow the mutated problem descriptor, "
           "not the original object address.";
}

TEST(BatchPointerReset, ProblemDependentCPUInputsRefreshAcrossPreparePaths)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    constexpr size_t BATCH = 4;

    auto p1 = makeBatchedProblem(32, 32, 32, BATCH);
    auto p2 = makeBatchedProblem(64, 64, 64, BATCH);

    auto args = makeBatchPointerResetArgs({{64, 64, BATCH, 64}});
    TensileLite::testing::detail::setDataInitArg(args,
                                                 "init-a",
                                                 std::any(Client::InitMode::SerialDim0));

    ClientProblemFactory        factory(args);
    PredicateDataInitialization  dataInit(args, factory);

    dataInit.prepareCPUInputs(p1);
    EXPECT_FALSE(dataInit.cpuInputsNeedRefresh(p1));
    EXPECT_TRUE(dataInit.cpuInputsNeedRefresh(p2));

    auto inputs2 = dataInit.prepareGPUInputs(p2);
    auto* ci2    = dynamic_cast<ContractionInputs*>(inputs2.get());
    ASSERT_NE(ci2, nullptr);
    ASSERT_NE(ci2->a, nullptr);

    size_t const idx = p2.a().index(48, 0, 0);
    float const  value
        = readDeviceValue(static_cast<float const*>(ci2->a), idx);
    EXPECT_FLOAT_EQ(value, 48.0f)
        << "GPU input A(m=48, k=0) should follow the SerialDim0 pattern for problem 2.";
    EXPECT_FALSE(dataInit.cpuInputsNeedRefresh(p2));
}

TEST(BatchPointerReset, GPUPrepareCachesFreshConstantInputsOnFirstSlowPath)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    TensileLite::testing::PlainProblemSpec spec;
    spec.m     = 32;
    spec.n     = 32;
    spec.k     = 32;
    spec.beta  = 1.0;

    auto problem = makePlainProblem(spec);
    problem.setActivationType(ActivationType::Clippedrelu);
    problem.setActivationComputeType(rocisa::DataType::Float);

    auto args = makeConstantCachingArgs();

    ClientProblemFactory factory(args);
    DataInitialization   dataInit(args, factory);

    auto inputs = dataInit.prepareGPUInputs(problem);
    auto* ci    = dynamic_cast<ContractionInputs*>(inputs.get());
    ASSERT_NE(ci, nullptr);

    ASSERT_TRUE(std::holds_alternative<float>(ci->alpha));
    ASSERT_TRUE(std::holds_alternative<float>(ci->beta));
    ASSERT_EQ(ci->activationArgs.size(), 2u);
    ASSERT_TRUE(std::holds_alternative<float>(ci->activationArgs[0]));
    ASSERT_TRUE(std::holds_alternative<float>(ci->activationArgs[1]));

    EXPECT_FLOAT_EQ(std::get<float>(ci->alpha), 2.0f);
    EXPECT_FLOAT_EQ(std::get<float>(ci->beta), 2.0f);
    EXPECT_FLOAT_EQ(std::get<float>(ci->activationArgs[0]), 3.25f);
    EXPECT_FLOAT_EQ(std::get<float>(ci->activationArgs[1]), -1.5f);
}

#if HIPBLASLT_ENABLE_MXDATAGENERATOR
TEST(BatchPointerReset, MXCpuFreshnessMarksGeneratedDescriptors)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    constexpr size_t BATCH   = 4;
    constexpr int    MXBLOCK = 32;

    auto problem = makeMXBatchedProblem(64, 64, 32, BATCH, MXBLOCK);
    auto args    = makeMXPredicateArgs({{64, 64, BATCH, 32}}, MXBLOCK);

    ClientProblemFactory       factory(args);
    PredicateDataInitialization dataInit(args, factory);

    dataInit.prepareCPUInputs(problem);
    EXPECT_FALSE(dataInit.cpuInputsNeedRefresh(problem));
}

TEST(BatchPointerReset, MXPreparedInputsAreEligibleForSolutionRefresh)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    constexpr size_t BATCH   = 4;
    constexpr int    MXBLOCK = 32;

    auto problem = makeMXBatchedProblem(64, 64, 32, BATCH, MXBLOCK);
    auto args    = makeMXPredicateArgs({{64, 64, BATCH, 32}}, MXBLOCK);

    ClientProblemFactory       factory(args);
    PredicateDataInitialization dataInit(args, factory);

    auto inputs = dataInit.prepareGPUInputs(problem);
    ASSERT_NE(inputs, nullptr);
    EXPECT_TRUE(dataInit.gpuInputsPreparedFor(problem));

    ContractionSolution solution;
    EXPECT_TRUE(dataInit.shouldRefreshMXForSolution(&solution, problem));
}
#endif

TEST(BatchPointerReset, CancelAsyncResetClearsWarmRingBeforeProblemSwitch)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    constexpr size_t BATCH = 4;

    auto p1 = makeBatchedProblem(32, 32, 32, BATCH);
    auto p2 = makeBatchedProblem(64, 64, 64, BATCH);

    auto args = makeRingBatchPointerArgs({{64, 64, BATCH, 64}}, 1);

    ClientProblemFactory factory(args);
    DataInitialization   dataInit(args, factory);

    auto inputs1 = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&p1));

    auto* ci1 = dynamic_cast<ContractionInputs*>(inputs1.get());
    ASSERT_NE(ci1, nullptr);
    ASSERT_NE(ci1->batchA, nullptr);

    void* batchA_p1[BATCH];
    HIP_CHECK_EXC(
        hipMemcpy(batchA_p1, ci1->batchA, BATCH * sizeof(void*), hipMemcpyDeviceToHost));

    ptrdiff_t stride1   = (uint8_t*)batchA_p1[1] - (uint8_t*)batchA_p1[0];
    ptrdiff_t expected1 = ptrdiff_t(32 * 32);
    EXPECT_EQ(stride1, expected1) << "Problem 1 batch pointer stride mismatch";

    auto* slot0BatchA = ci1->batchA;

    dataInit.beginAsyncReset(&p1);
    dataInit.beginAsyncReset(&p1);

    auto warmInputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&p1));
    dataInit.waitCopyDone(nullptr);

    ASSERT_NE(warmInputs.get(), nullptr);
    auto* warmCi = dynamic_cast<ContractionInputs*>(warmInputs.get());
    ASSERT_NE(warmCi, nullptr);
    ASSERT_NE(warmCi->batchA, nullptr);
    EXPECT_NE(warmCi->batchA, slot0BatchA);

    dataInit.preProblem(&p2);
    dataInit.cancelAsyncReset();

    auto inputs2 = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&p2));

    auto* ci2 = dynamic_cast<ContractionInputs*>(inputs2.get());
    ASSERT_NE(ci2, nullptr);
    ASSERT_NE(ci2->batchA, nullptr);
    EXPECT_EQ(ci2->batchA, slot0BatchA);

    void* batchA_p2[BATCH];
    HIP_CHECK_EXC(
        hipMemcpy(batchA_p2, ci2->batchA, BATCH * sizeof(void*), hipMemcpyDeviceToHost));

    ptrdiff_t stride2   = (uint8_t*)batchA_p2[1] - (uint8_t*)batchA_p2[0];
    ptrdiff_t expected2 = ptrdiff_t(64 * 64);
    EXPECT_EQ(stride2, expected2)
        << "After switching to problem 2, the batch pointer stride should reflect "
           "problem 2's aStride (" << expected2 << " bytes), but got " << stride2
        << ". This indicates cancelAsyncReset did not clear the warm ring.";
}

// Regression test for the lifecycle bug where cancelAsyncReset() only cleared
// warm-ring state but left m_batchInitProblem intact. The same problem object
// is reused for a logically different GEMM, so the pointer-identity guard in
// prepareGPUInputsInternal() can only pass if cancelAsyncReset() also resets
// the batch-pointer freshness cache.
TEST(BatchPointerReset, CancelAsyncResetInvalidatesBatchPointersWithoutPreProblem)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    constexpr size_t BATCH = 4;

    auto problem = makeBatchedProblem(32, 32, 32, BATCH);
    auto* stableProblemAddress = &problem;

    auto args = makeBatchPointerResetArgs({{64, 64, BATCH, 64}});

    ClientProblemFactory factory(args);
    DataInitialization   dataInit(args, factory);

    auto inputs1 = dataInit.prepareGPUInputs(problem);

    auto* ci1 = dynamic_cast<ContractionInputs*>(inputs1.get());
    ASSERT_NE(ci1, nullptr);
    ASSERT_NE(ci1->batchA, nullptr);

    void* batchA_p1[BATCH];
    HIP_CHECK_EXC(
        hipMemcpy(batchA_p1, ci1->batchA, BATCH * sizeof(void*), hipMemcpyDeviceToHost));

    ptrdiff_t stride1   = (uint8_t*)batchA_p1[1] - (uint8_t*)batchA_p1[0];
    ptrdiff_t expected1 = ptrdiff_t(32 * 32);
    EXPECT_EQ(stride1, expected1) << "Problem 1 batch pointer stride mismatch";

    problem = makeBatchedProblem(64, 64, 64, BATCH);
    ASSERT_EQ(&problem, stableProblemAddress);

    dataInit.cancelAsyncReset();

    auto inputs2 = dataInit.prepareGPUInputs(problem);

    auto* ci2 = dynamic_cast<ContractionInputs*>(inputs2.get());
    ASSERT_NE(ci2, nullptr);
    ASSERT_NE(ci2->batchA, nullptr);

    void* batchA_p2[BATCH];
    HIP_CHECK_EXC(
        hipMemcpy(batchA_p2, ci2->batchA, BATCH * sizeof(void*), hipMemcpyDeviceToHost));

    ptrdiff_t stride2   = (uint8_t*)batchA_p2[1] - (uint8_t*)batchA_p2[0];
    ptrdiff_t expected2 = ptrdiff_t(64 * 64);
    EXPECT_EQ(stride2, expected2)
        << "After switching to the larger logical problem, the batch pointer "
           "stride should reflect the new aStride (" << expected2
        << " bytes), but got " << stride2
        << ". This indicates cancelAsyncReset did not clear the batch-pointer cache.";
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
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    constexpr size_t BATCH = 4;

    // p1: small problem — A tensor stride delta = 32*32 bytes.
    auto p1 = makeBatchedProblem(32, 32, 32, BATCH);
    // p2: larger problem — A tensor stride delta = 64*64 bytes.
    auto p2 = makeBatchedProblem(64, 64, 64, BATCH);

    // Buffer must be sized for the largest problem.
    auto args = makeBatchPointerResetArgs({{64, 64, BATCH, 64}});

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
