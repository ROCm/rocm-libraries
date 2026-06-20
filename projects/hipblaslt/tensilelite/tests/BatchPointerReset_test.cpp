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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>

#include <variant>
#include <vector>

#include "DataInitializationTestUtils.hpp"
#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"
#include "RecordingCopyEngine.hpp"

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    using TensileLite::testing::makeBatchedProblem;
    using TensileLite::testing::makePlainProblem;
    using TensileLite::testing::RecordingCopyEngine;

    class PredicateDataInitialization : public DataInitialization
    {
    public:
        using DataInitialization::DataInitialization;
        using DataInitialization::cpuInputsNeedRefresh;
        using DataInitialization::gpuInputsPreparedFor;
        using DataInitialization::shouldRefreshMXForSolution;

        bool altSlotsReady() const
        {
            return m_altSlotsReady;
        }

        auto const& slotState(size_t slot) const
        {
            return m_gpuInputSlots.at(slot);
        }
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

    Client::po::variables_map
        makeBatchPointerResetArgs(std::vector<std::vector<size_t>> problemSizes)
    {
        return TensileLite::testing::buildBaseDataInitArgs(std::move(problemSizes));
    }

    Client::po::variables_map makeRingBatchPointerArgs(
        std::vector<std::vector<size_t>> problemSizes, int elementsToValidate = 1)
    {
        return TensileLite::testing::buildRingArgs(std::move(problemSizes), elementsToValidate);
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

    std::ptrdiff_t readBatchAStride(void const* const* deviceBatchA, size_t batch)
    {
        if(batch < 2)
        {
            ADD_FAILURE() << "readBatchAStride requires at least two batch entries";
            return 0;
        }

        std::vector<void const*> batchPtrs(batch);
        HIP_CHECK_EXC(hipMemcpy(batchPtrs.data(),
                                deviceBatchA,
                                batch * sizeof(void const*),
                                hipMemcpyDeviceToHost));

        auto const* first  = static_cast<std::uint8_t const*>(batchPtrs[0]);
        auto const* second = static_cast<std::uint8_t const*>(batchPtrs[1]);
        return second - first;
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

    class MXLifecycleDataInitialization : public PredicateDataInitialization
    {
    public:
        using PredicateDataInitialization::PredicateDataInitialization;
        using PristineUnit = DataInitialization::PristineUnit;

        PristineUnit const& pristineUnit(size_t tensorIndex,
                                         ContractionProblemGemm const& problem) const
        {
            auto const& desc  = problem.tensors().at(tensorIndex);
            auto const& units = m_vdata.at(tensorIndex).pristine;
            auto        it    = units.find(desc.dataType());
            if(it == units.end())
            {
                throw std::runtime_error("Missing pristine unit for tensor index.");
            }
            return it->second;
        }
    };

    size_t tensorBytes(MXLifecycleDataInitialization const& dataInit,
                       ContractionProblemGemm const&         problem,
                       size_t                               tensorIndex)
    {
        auto const& unit = dataInit.pristineUnit(tensorIndex, problem);
        auto const& desc = problem.tensors().at(tensorIndex);
        return multiplyElementSize(unit.maxElements, desc.elementBytes());
    }

    bool bytesEqual(void const* lhs, void const* rhs, size_t bytes)
    {
        return std::memcmp(lhs, rhs, bytes) == 0;
    }

    bool allBytesEqual(void const* ptr, size_t bytes, std::uint8_t value)
    {
        auto const* begin = static_cast<std::uint8_t const*>(ptr);
        return std::all_of(begin, begin + bytes, [value](std::uint8_t byte) {
            return byte == value;
        });
    }

    RecordingCopyEngine::Call const*
        findCopy(RecordingCopyEngine const& engine,
                 void const*                 dst,
                 void const*                 src,
                 size_t                      bytes,
                 hipMemcpyKind               kind)
    {
        auto const it = std::find_if(engine.calls.begin(),
                                     engine.calls.end(),
                                     [&](RecordingCopyEngine::Call const& call) {
                                         return call.type == RecordingCopyEngine::CallType::Copy
                                                && call.dst == dst && call.src == src
                                                && call.bytes == bytes
                                                && call.copyKind == kind;
                                     });
        if(it == engine.calls.end())
        {
            return nullptr;
        }
        return &*it;
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
    //     Simulate what main.cpp does when the problem changes:
    //     beginProblem(nullptr) clears the cached problem context, so the
    //     pointer-identity check in prepareGPUInputsInternal fires and
    //     re-uploads batch pointers.
    dataInit.beginProblem(nullptr);
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

TEST(BatchPointerReset,
     MXPreSolutionRefreshesThroughListenerPathAndSyncsReferenceInputs)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    constexpr size_t M       = 128;
    constexpr size_t N       = 128;
    constexpr size_t K       = 256;
    constexpr size_t BATCH   = 4;
    constexpr int    MXBLOCK = 32;

    auto problem = makeMXBatchedProblem(M, N, K, BATCH, MXBLOCK);
    auto args    = makeMXPredicateArgs({{M, N, BATCH, K}}, MXBLOCK);

    auto engine = std::make_shared<RecordingCopyEngine>();
    ClientProblemFactory          factory(args);
    MXLifecycleDataInitialization dataInit(args, factory, engine);

    dataInit.preProblem(&problem);

    auto referenceInputs = dataInit.prepareCPUInputs(problem);
    auto* reference      = dynamic_cast<ContractionInputs*>(referenceInputs.get());
    ASSERT_NE(reference, nullptr);

    auto gpuInputs = dataInit.prepareGPUInputs(problem);
    ASSERT_NE(gpuInputs, nullptr);
    auto* gpu = dynamic_cast<ContractionInputs*>(gpuInputs.get());
    ASSERT_NE(gpu, nullptr);
    ASSERT_TRUE(dataInit.gpuInputsPreparedFor(problem));

    ContractionSolution solution;
    solution.problemType.mxScaleFormat = 1;
    solution.sizeMapping.matrixInstruction = {16, 16, 128, 1};
    ASSERT_TRUE(dataInit.shouldRefreshMXForSolution(&solution, problem));

    constexpr std::array<size_t, 4> kRefreshTensors{
        ContractionProblemGemm::TENSOR::A,
        ContractionProblemGemm::TENSOR::B,
        ContractionProblemGemm::TENSOR::MXSA,
        ContractionProblemGemm::TENSOR::MXSB,
    };

    for(size_t tensorIndex : kRefreshTensors)
    {
        auto const& unit  = dataInit.pristineUnit(tensorIndex, problem);
        auto const  bytes = tensorBytes(dataInit, problem, tensorIndex);
        ASSERT_NE(unit.cpuInput.valid.get(), nullptr);
        ASSERT_NE(unit.cpuInput.current.get(), nullptr);
        std::memset(unit.cpuInput.valid.get(), 0x5A, bytes);
        std::memset(unit.cpuInput.current.get(), 0xA5, bytes);
        EXPECT_TRUE(allBytesEqual(unit.cpuInput.valid.get(), bytes, 0x5A));
        EXPECT_TRUE(allBytesEqual(unit.cpuInput.current.get(), bytes, 0xA5));
    }

    EXPECT_EQ(reference->a,
              dataInit.pristineUnit(ContractionProblemGemm::TENSOR::A, problem)
                  .cpuInput.current.get());
    EXPECT_EQ(reference->b,
              dataInit.pristineUnit(ContractionProblemGemm::TENSOR::B, problem)
                  .cpuInput.current.get());
    EXPECT_EQ(reference->mxsa,
              dataInit.pristineUnit(ContractionProblemGemm::TENSOR::MXSA, problem)
                  .cpuInput.current.get());
    EXPECT_EQ(reference->mxsb,
              dataInit.pristineUnit(ContractionProblemGemm::TENSOR::MXSB, problem)
                  .cpuInput.current.get());

    EXPECT_EQ(gpu->a,
              dataInit.pristineUnit(ContractionProblemGemm::TENSOR::A, problem)
                  .gpuInput.current.get());
    EXPECT_EQ(gpu->b,
              dataInit.pristineUnit(ContractionProblemGemm::TENSOR::B, problem)
                  .gpuInput.current.get());
    EXPECT_EQ(gpu->mxsa,
              dataInit.pristineUnit(ContractionProblemGemm::TENSOR::MXSA, problem)
                  .gpuInput.current.get());
    EXPECT_EQ(gpu->mxsb,
              dataInit.pristineUnit(ContractionProblemGemm::TENSOR::MXSB, problem)
                  .gpuInput.current.get());

    engine->clear();
    dataInit.preSolution(&solution);

    auto const expectedStream = engine->stream();

    for(size_t tensorIndex : {ContractionProblemGemm::TENSOR::A,
                              ContractionProblemGemm::TENSOR::B})
    {
        auto const& unit  = dataInit.pristineUnit(tensorIndex, problem);
        auto const  bytes = tensorBytes(dataInit, problem, tensorIndex);
        auto const  call  = findCopy(*engine,
                                    unit.gpuInput.valid.get(),
                                    unit.cpuInput.valid.get(),
                                    bytes,
                                    hipMemcpyHostToDevice);
        ASSERT_NE(call, nullptr) << "Missing H2D refresh copy for tensor index "
                                 << tensorIndex;
        EXPECT_EQ(call->stream, expectedStream);
        EXPECT_EQ(call->submissionMode, RecordingCopyEngine::CopySubmissionMode::Async);
    }

    for(size_t tensorIndex : kRefreshTensors)
    {
        auto const& unit  = dataInit.pristineUnit(tensorIndex, problem);
        auto const  bytes = tensorBytes(dataInit, problem, tensorIndex);
        auto const  call  = findCopy(*engine,
                                    unit.gpuInput.current.get(),
                                    unit.gpuInput.valid.get(),
                                    bytes,
                                    hipMemcpyDeviceToDevice);
        ASSERT_NE(call, nullptr) << "Missing D2D refresh copy for tensor index "
                                 << tensorIndex;
        EXPECT_EQ(call->stream, expectedStream);
        EXPECT_EQ(call->submissionMode, RecordingCopyEngine::CopySubmissionMode::Async);
    }

    for(size_t tensorIndex : kRefreshTensors)
    {
        auto const& unit  = dataInit.pristineUnit(tensorIndex, problem);
        auto const  bytes = tensorBytes(dataInit, problem, tensorIndex);

        EXPECT_FALSE(allBytesEqual(unit.cpuInput.valid.get(), bytes, 0x5A))
            << "Tensor " << tensorIndex << " valid buffer was not regenerated.";
        EXPECT_FALSE(allBytesEqual(unit.cpuInput.current.get(), bytes, 0xA5))
            << "Tensor " << tensorIndex << " current buffer was not resynced.";
        EXPECT_TRUE(bytesEqual(unit.cpuInput.current.get(), unit.cpuInput.valid.get(), bytes))
            << "Tensor " << tensorIndex
            << " current and valid buffers diverged after preSolution.";
    }

    EXPECT_EQ(reference->a,
              dataInit.pristineUnit(ContractionProblemGemm::TENSOR::A, problem)
                  .cpuInput.current.get());
    EXPECT_EQ(reference->b,
              dataInit.pristineUnit(ContractionProblemGemm::TENSOR::B, problem)
                  .cpuInput.current.get());
    EXPECT_EQ(reference->mxsa,
              dataInit.pristineUnit(ContractionProblemGemm::TENSOR::MXSA, problem)
                  .cpuInput.current.get());
    EXPECT_EQ(reference->mxsb,
              dataInit.pristineUnit(ContractionProblemGemm::TENSOR::MXSB, problem)
                  .cpuInput.current.get());
}
#endif

TEST(BatchPointerReset, StaleAltSlotRefreshAfterProblemSwitch)
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

    ClientProblemFactory        factory(args);
    PredicateDataInitialization  dataInit(args, factory);

    auto inputs1 = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&p1));
    auto* ci1    = dynamic_cast<ContractionInputs*>(inputs1.get());
    ASSERT_NE(ci1, nullptr);
    ASSERT_NE(ci1->batchA, nullptr);

    auto const expectedP1Stride = std::ptrdiff_t(32 * 32);
    auto const p1Slot0Stride = readBatchAStride(ci1->batchA, BATCH);
    EXPECT_EQ(p1Slot0Stride, expectedP1Stride);

    dataInit.primeNextInputSlot(&p1);
    auto p1WarmInputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&p1));
    dataInit.waitForPreparedSlot(nullptr);

    auto* p1WarmCi = dynamic_cast<ContractionInputs*>(p1WarmInputs.get());
    ASSERT_NE(p1WarmCi, nullptr);
    ASSERT_NE(p1WarmCi->batchA, nullptr);
    EXPECT_NE(p1WarmCi->batchA, ci1->batchA);
    auto const p1WarmStride = readBatchAStride(p1WarmCi->batchA, BATCH);
    EXPECT_EQ(p1WarmStride, expectedP1Stride);

    dataInit.beginProblem(&p2);
    dataInit.resetPreparedSlotsForProblem();

    auto inputs2 = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&p2));
    auto* ci2    = dynamic_cast<ContractionInputs*>(inputs2.get());
    ASSERT_NE(ci2, nullptr);
    ASSERT_NE(ci2->batchA, nullptr);

    auto const expectedP2Stride = std::ptrdiff_t(64 * 64);
    auto const p2Slot0Stride = readBatchAStride(ci2->batchA, BATCH);
    EXPECT_EQ(p2Slot0Stride, expectedP2Stride);

    ASSERT_TRUE(dataInit.altSlotsReady());
    auto const& p2Slot1 = dataInit.slotState(1);
    ASSERT_TRUE(p2Slot1.populated());
    auto const p2Slot1BatchA = reinterpret_cast<void const* const*>(
        p2Slot1.batchPtrs.at(ContractionProblemGemm::TENSOR::A));
    ASSERT_NE(p2Slot1BatchA, nullptr);
    EXPECT_NE(p2Slot1BatchA, ci2->batchA);
    auto const p2Slot1Stride = readBatchAStride(p2Slot1BatchA, BATCH);
    EXPECT_EQ(p2Slot1Stride, expectedP2Stride);
    EXPECT_NE(p2Slot1Stride, expectedP1Stride);

    dataInit.primeNextInputSlot(&p2);
    auto p2WarmInputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&p2));
    dataInit.waitForPreparedSlot(nullptr);

    auto* p2WarmCi = dynamic_cast<ContractionInputs*>(p2WarmInputs.get());
    ASSERT_NE(p2WarmCi, nullptr);
    ASSERT_NE(p2WarmCi->batchA, nullptr);
    EXPECT_EQ(p2WarmCi->batchA, p2Slot1BatchA);
    auto const p2WarmStride = readBatchAStride(p2WarmCi->batchA, BATCH);
    EXPECT_EQ(p2WarmStride, expectedP2Stride);
}

TEST(BatchPointerReset, PointerWrapperFastPathRejectsStalePreparedRingSlot)
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

    ClientProblemFactory       factory(args);
    PredicateDataInitialization dataInit(args, factory);

    auto inputs1 = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&p1));
    auto* ci1    = dynamic_cast<ContractionInputs*>(inputs1.get());
    ASSERT_NE(ci1, nullptr);
    ASSERT_NE(ci1->batchA, nullptr);

    auto const expectedP1Stride = std::ptrdiff_t(32 * 32);
    auto const p1Slot0Stride    = readBatchAStride(ci1->batchA, BATCH);
    EXPECT_EQ(p1Slot0Stride, expectedP1Stride);

    ASSERT_TRUE(dataInit.altSlotsReady());
    auto const& p1Slot1State = dataInit.slotState(1);
    ASSERT_TRUE(p1Slot1State.populated());
    auto const p1Slot1Inputs = p1Slot1State.cachedInputs;
    ASSERT_NE(p1Slot1Inputs, nullptr);
    auto const* p1Slot1Ci = dynamic_cast<ContractionInputs const*>(p1Slot1Inputs.get());
    ASSERT_NE(p1Slot1Ci, nullptr);
    ASSERT_NE(p1Slot1Ci->batchA, nullptr);

    auto const expectedP2Stride = std::ptrdiff_t(64 * 64);
    auto const p1Slot1Stride     = readBatchAStride(p1Slot1Ci->batchA, BATCH);
    EXPECT_EQ(p1Slot1Stride, expectedP1Stride);

    dataInit.primeNextInputSlot(&p1);

    auto inputs2 = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&p2));
    auto* ci2    = dynamic_cast<ContractionInputs*>(inputs2.get());
    ASSERT_NE(ci2, nullptr);
    ASSERT_NE(ci2->batchA, nullptr);

    auto const p2Slot0Stride = readBatchAStride(ci2->batchA, BATCH);
    EXPECT_EQ(p2Slot0Stride, expectedP2Stride)
        << "The pointer-wrapper fast path must not consume the stale p1 slot "
           "when preparing p2.";

    ASSERT_TRUE(dataInit.altSlotsReady());
    auto const& p2Slot1State = dataInit.slotState(1);
    ASSERT_TRUE(p2Slot1State.populated());
    auto const p2Slot1Inputs = p2Slot1State.cachedInputs;
    ASSERT_NE(p2Slot1Inputs, nullptr);
    EXPECT_NE(p2Slot1Inputs, p1Slot1Inputs);
    auto const* p2Slot1Ci = dynamic_cast<ContractionInputs const*>(p2Slot1Inputs.get());
    ASSERT_NE(p2Slot1Ci, nullptr);
    ASSERT_NE(p2Slot1Ci->batchA, nullptr);

    auto const p2Slot1Stride = readBatchAStride(p2Slot1Ci->batchA, BATCH);
    EXPECT_EQ(p2Slot1Stride, expectedP2Stride);
    EXPECT_NE(p2Slot1Stride, p1Slot1Stride);

    dataInit.primeNextInputSlot(&p2);
    auto p2WarmInputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&p2));
    dataInit.waitForPreparedSlot(nullptr);

    auto* p2WarmCi = dynamic_cast<ContractionInputs*>(p2WarmInputs.get());
    ASSERT_NE(p2WarmCi, nullptr);
    ASSERT_NE(p2WarmCi->batchA, nullptr);

    auto const p2WarmStride = readBatchAStride(p2WarmCi->batchA, BATCH);
    EXPECT_EQ(p2WarmStride, expectedP2Stride);
    EXPECT_NE(p2WarmStride, p1Slot1Stride);
}

TEST(BatchPointerReset, ResetPreparedSlotsForProblemClearsWarmRingBeforeProblemSwitch)
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

    dataInit.primeNextInputSlot(&p1);
    dataInit.primeNextInputSlot(&p1);

    auto warmInputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&p1));
    dataInit.waitForPreparedSlot(nullptr);

    ASSERT_NE(warmInputs.get(), nullptr);
    auto* warmCi = dynamic_cast<ContractionInputs*>(warmInputs.get());
    ASSERT_NE(warmCi, nullptr);
    ASSERT_NE(warmCi->batchA, nullptr);
    EXPECT_NE(warmCi->batchA, slot0BatchA);

    dataInit.beginProblem(&p2);
    dataInit.resetPreparedSlotsForProblem();

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
        << ". This indicates resetPreparedSlotsForProblem did not clear the warm ring.";
}

// Regression test for the lifecycle bug where resetPreparedSlotsForProblem()
// only cleared warm-ring state but left the cached problem context intact.
// The same problem object is reused for a logically different GEMM, so the
// pointer-identity guard in prepareGPUInputsInternal() can only pass if
// resetPreparedSlotsForProblem() also resets the batch-pointer freshness cache.
TEST(BatchPointerReset, ResetPreparedSlotsForProblemInvalidatesBatchPointersWithoutBeginProblem)
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

    dataInit.resetPreparedSlotsForProblem();

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
        << ". This indicates resetPreparedSlotsForProblem did not clear the batch-pointer cache.";
}

// ---------------------------------------------------------------------------
// Structural invariant: switching to a different ContractionProblemGemm
// object must trigger batch-pointer re-upload even when beginProblem() is not
// called in between.
//
// With the old bool m_batchInit approach, skipping beginProblem() leaves
// m_batchInit=true, so initializeGPUBatchedInputs is skipped and the caller
// gets batch pointers from the first problem's strides — silently wrong.
//
// The fix replaces the bool with ContractionProblemGemm const*
// m_batchInitProblem and checks (m_batchInitProblem != &problem) in
// prepareGPUInputsInternal.  Because p1 and p2 are distinct objects, their
// addresses differ, so the check fires and re-uploads correctly — no
// beginProblem() needed to make it work.
//
// This test therefore fails with the boolean implementation and passes after
// the pointer-identity fix.  It is the regression test for the structural
// guarantee, not just the call-site-discipline guarantee.
// ---------------------------------------------------------------------------
TEST(BatchPointerReset, StructuralReinitWithoutBeginProblem)
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

    // Second call: switch to p2 WITHOUT calling beginProblem().
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
        << " bytes) even without an intervening beginProblem() call. "
           "Got " << stride << ". This means initializeGPUBatchedInputs was "
           "skipped — the structural pointer-identity guard is missing.";
}
