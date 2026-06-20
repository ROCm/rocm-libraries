// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <any>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include <Tensile/Utils.hpp>

#include "BatchPointerLayout.hpp"
#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"
#include "DataInitializationTestUtils.hpp"
#include "RecordingCopyEngine.hpp"

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    using TensileLite::testing::PlainProblemSpec;
    using TensileLite::testing::RecordingCopyEngine;
    using TensileLite::testing::makePlainProblem;

    class CopyPlanDataInitialization : public DataInitialization
    {
    public:
        using DataInitialization::DataInitialization;
        using DataInitialization::copyInputs;
        using DataInitialization::copyInputsForSlot;
        using DataInitialization::copySwizzledToGPUBuffer;
        using DataInitialization::copyValidToGPUBuffer;
        using DataInitialization::initializeGPUBatchedInputs;
        using DataInitialization::resetOutput;
        using DataInitialization::resetOutputForSlot;

        std::vector<void*>& gpuPtrs()
        {
            return m_gpuPtrs;
        }

        std::vector<void**>& gpuBatchPtrs()
        {
            return m_gpuBatchPtrs;
        }

        std::vector<size_t>& maxElements()
        {
            return m_maxElements;
        }

        std::vector<std::vector<size_t>>& groupedOffsets()
        {
            return m_groupedOffsets;
        }

        bool altSlotsReady() const
        {
            return m_altSlotsReady;
        }

        auto const& slotState(size_t slot) const
        {
            return m_gpuInputSlots.at(slot);
        }

        PristineUnit const& pristineUnit(size_t tensorIndex,
                                         ContractionProblemGemm const& problem) const
        {
            auto const& desc  = problem.tensors().at(tensorIndex);
            auto const& units = m_vdata.at(tensorIndex).pristine;
            auto        it    = units.find(desc.dataType());
            if(it == units.end())
                throw std::runtime_error("Missing pristine unit for tensor index.");
            return it->second;
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
            return ::testing::AssertionFailure() << "No HIP devices available";

        return ::testing::AssertionSuccess();
    }

    Client::po::variables_map makeBaseArgs(std::vector<std::vector<size_t>> problemSizes)
    {
        auto args = TensileLite::testing::buildBaseDataInitArgs(std::move(problemSizes));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "num-elements-to-validate",
                                                     std::any(int(1)));
        return args;
    }

    Client::po::variables_map makeRingArgs(std::vector<std::vector<size_t>> problemSizes)
    {
        return TensileLite::testing::buildRingArgs(std::move(problemSizes), 1);
    }

    Client::po::variables_map makeGuardPageBackArgs(std::vector<std::vector<size_t>> problemSizes,
                                                    bool swizzleTensorA = false,
                                                    bool swizzleTensorB = false)
    {
        auto args = makeBaseArgs(std::move(problemSizes));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "bounds-check",
                                                     std::any(BoundsCheckMode::GuardPageBack));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "swizzle-tensor-a",
                                                     std::any(swizzleTensorA));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "swizzle-tensor-b",
                                                     std::any(swizzleTensorB));
        return args;
    }

    ContractionProblemGemm makeBatchProblem(size_t m,
                                            size_t n,
                                            size_t k,
                                            size_t batch,
                                            bool   swizzleTensorA = false,
                                            bool   swizzleTensorB = false)
    {
        PlainProblemSpec spec;
        spec.m     = m;
        spec.n     = n;
        spec.k     = k;
        spec.batch = batch;

        auto problem = makePlainProblem(spec);
        problem.setSwizzleTensorA(swizzleTensorA);
        problem.setSwizzleTensorB(swizzleTensorB);
        return problem;
    }

    void replayRecordedCopies(RecordingCopyEngine const& engine)
    {
        for(auto const& call : engine.calls)
        {
            if(call.type != RecordingCopyEngine::CallType::Copy)
                continue;

            std::memmove(call.dst, call.src, call.bytes);
        }
    }

} // namespace

TEST(DataInitializationCopyPlan, ExecutorPreservesD2DOnlyStreamForwarding)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
        GTEST_SKIP() << hipDevice.message();

    auto problem = makeBatchProblem(32, 24, 16, 4);
    auto args    = makeBaseArgs({{32, 24, 4, 16}});

    auto engine = std::make_shared<RecordingCopyEngine>(
        reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(0x1234)));

    ClientProblemFactory      factory(args);
    CopyPlanDataInitialization dataInit(args, factory, engine);

    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);
    engine->clear();

    auto const runCopy = [&](hipMemcpyKind kind, hipStream_t targetStream, hipStream_t expected) {
        std::vector<void*>              ptrs(problem.tensors().size(), nullptr);
        std::vector<void**>             batchPtrs(problem.tensors().size(), nullptr);
        std::vector<size_t>             maxElements(problem.tensors().size(), 0);
        std::vector<std::vector<size_t>> offsets(problem.tensors().size());

        dataInit.copyInputs(ptrs, batchPtrs, maxElements, offsets, problem, kind, targetStream);

        bool sawCopy = false;
        for(auto const& call : engine->calls)
        {
            if(call.type != RecordingCopyEngine::CallType::Copy)
                continue;
            sawCopy = true;
            EXPECT_EQ(call.copyKind, kind);
            EXPECT_EQ(call.stream, expected);
        }
        EXPECT_TRUE(sawCopy);
        engine->clear();
    };

    runCopy(hipMemcpyHostToHost, nullptr, nullptr);
    runCopy(hipMemcpyHostToDevice, nullptr, nullptr);
    runCopy(hipMemcpyDeviceToDevice,
            reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(0x1234)),
            reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(0x1234)));
}

TEST(DataInitializationCopyPlan, InputNaNBoundsCopiesValidSourcesAndPreservesSentinelPadding)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
        GTEST_SKIP() << hipDevice.message();

    auto problem = makeBatchProblem(17, 19, 23, 4);
    auto args    = makeBaseArgs({{33, 35, 4, 37}});
    TensileLite::testing::detail::setDataInitArg(args,
                                                 "bounds-check",
                                                 std::any(BoundsCheckMode::NaN));

    auto engine = std::make_shared<RecordingCopyEngine>();

    ClientProblemFactory      factory(args);
    CopyPlanDataInitialization dataInit(args, factory, engine);
    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);
    engine->clear();

    std::vector<void*>              ptrs(problem.tensors().size(), nullptr);
    std::vector<void**>             batchPtrs(problem.tensors().size(), nullptr);
    std::vector<size_t>             maxElements(problem.tensors().size(), 0);
    std::vector<std::vector<size_t>> offsets(problem.tensors().size());

    dataInit.copyInputs(ptrs,
                        batchPtrs,
                        maxElements,
                        offsets,
                        problem,
                        hipMemcpyHostToHost);

    auto const& aUnit = dataInit.pristineUnit(ContractionProblemGemm::TENSOR::A, problem);
    auto const& aDesc = problem.tensors().at(ContractionProblemGemm::TENSOR::A);

    ASSERT_NE(aUnit.cpuInput.current.get(), nullptr);
    ASSERT_NE(aUnit.cpuInput.valid.get(), nullptr);
    ASSERT_NE(aUnit.cpuInput.bad.get(), nullptr);
    ASSERT_GT(aUnit.maxElements, aDesc.totalAllocatedElements());

    auto const paddingElements = aUnit.maxElements - aDesc.totalAllocatedElements();
    auto const rawPaddingBytes  = multiplyElementSize(paddingElements, aDesc.elementBytes());
    auto const alignmentBytes
        = 2 * static_cast<size_t>(std::ceil(aDesc.elementBytes() < 1.0f ? 1.0f
                                                                         : aDesc.elementBytes()));
    auto const paddingBytes    = (rawPaddingBytes / alignmentBytes) * alignmentBytes;
    auto const bytesBeforeData  = paddingBytes / 2;
    auto const allocationBytes
        = multiplyElementSize(aUnit.maxElements, aDesc.elementBytes());
    auto const validBytes
        = multiplyElementSize(aDesc.totalAllocatedElements(), aDesc.elementBytes());
    auto const bytesAfterData = allocationBytes - bytesBeforeData - validBytes;

    auto* const       currentBase = static_cast<uint8_t*>(aUnit.cpuInput.current.get());
    auto const* const validBase   = static_cast<uint8_t const*>(aUnit.cpuInput.valid.get());
    auto const* const badBase     = static_cast<uint8_t const*>(aUnit.cpuInput.bad.get());
    auto* const       validDst    = currentBase + bytesBeforeData;

    bool sawSentinelFill = false;
    bool sawValidCopy    = false;
    for(auto const& call : engine->calls)
    {
        if(call.type != RecordingCopyEngine::CallType::Copy)
            continue;

        EXPECT_EQ(call.copyKind, hipMemcpyHostToHost);
        EXPECT_EQ(call.stream, nullptr);

        if(call.dst == aUnit.cpuInput.current.get())
        {
            ASSERT_FALSE(sawSentinelFill);
            sawSentinelFill = true;
            EXPECT_EQ(call.src, aUnit.cpuInput.bad.get());
            EXPECT_EQ(call.bytes, allocationBytes);
            continue;
        }

        if(call.dst == static_cast<void*>(validDst))
        {
            ASSERT_FALSE(sawValidCopy);
            sawValidCopy = true;
            EXPECT_EQ(call.src, aUnit.cpuInput.valid.get());
            EXPECT_EQ(call.bytes, validBytes);
        }
    }

    EXPECT_TRUE(sawSentinelFill);
    EXPECT_TRUE(sawValidCopy);

    replayRecordedCopies(*engine);

    EXPECT_EQ(ptrs.at(ContractionProblemGemm::TENSOR::A), static_cast<void*>(validDst));
    EXPECT_EQ(std::memcmp(currentBase, badBase, bytesBeforeData), 0);
    EXPECT_EQ(std::memcmp(validDst, validBase, validBytes), 0);
    EXPECT_EQ(std::memcmp(validDst + validBytes,
                          badBase + bytesBeforeData + validBytes,
                          bytesAfterData),
              0);
}

TEST(DataInitializationCopyPlan, InputGuardPageBackSwizzledTensorsUseCustomPadding)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
        GTEST_SKIP() << hipDevice.message();

    auto problem = makeBatchProblem(17, 19, 23, 4, true, true);
    auto args    = makeGuardPageBackArgs({{17, 19, 4, 23}}, true, true);

    auto engine = std::make_shared<RecordingCopyEngine>(
        reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(0x2468)));

    ClientProblemFactory      factory(args);
    CopyPlanDataInitialization dataInit(args, factory, engine);
    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);
    engine->clear();

    std::vector<void*>              ptrs(problem.tensors().size(), nullptr);
    std::vector<void**>             batchPtrs(problem.tensors().size(), nullptr);
    std::vector<size_t>             maxElements(problem.tensors().size(), 0);
    std::vector<std::vector<size_t>> offsets(problem.tensors().size());

    dataInit.copyInputs(ptrs,
                        batchPtrs,
                        maxElements,
                        offsets,
                        problem,
                        hipMemcpyDeviceToDevice);

    auto const& aUnit = dataInit.pristineUnit(ContractionProblemGemm::TENSOR::A, problem);
    auto const& bUnit = dataInit.pristineUnit(ContractionProblemGemm::TENSOR::B, problem);
    auto const& aDesc = problem.tensors().at(ContractionProblemGemm::TENSOR::A);
    auto const& bDesc = problem.tensors().at(ContractionProblemGemm::TENSOR::B);

    InputLayoutPolicy const policy;
    auto const              aPlan = policy.planTensorSwizzle(problem, ContractionProblemGemm::TENSOR::A);
    auto const              bPlan = policy.planTensorSwizzle(problem, ContractionProblemGemm::TENSOR::B);

    auto const expectedADst = static_cast<uint8_t*>(aUnit.gpuInput.current.get())
                              + multiplyElementSize(aUnit.maxElements - aPlan.allocatedElements,
                                                    aDesc.elementBytes());
    auto const expectedBDst = static_cast<uint8_t*>(bUnit.gpuInput.current.get())
                              + multiplyElementSize(bUnit.maxElements - bPlan.allocatedElements,
                                                    bDesc.elementBytes());

    EXPECT_EQ(ptrs.at(ContractionProblemGemm::TENSOR::A), static_cast<void*>(expectedADst));
    EXPECT_EQ(ptrs.at(ContractionProblemGemm::TENSOR::B), static_cast<void*>(expectedBDst));
    EXPECT_EQ(maxElements.at(ContractionProblemGemm::TENSOR::A), aUnit.maxElements);
    EXPECT_EQ(maxElements.at(ContractionProblemGemm::TENSOR::B), bUnit.maxElements);
    EXPECT_EQ(batchPtrs.at(ContractionProblemGemm::TENSOR::A), aUnit.gpuInput.batch.get());
    EXPECT_EQ(batchPtrs.at(ContractionProblemGemm::TENSOR::B), bUnit.gpuInput.batch.get());
}

TEST(DataInitializationCopyPlan, InputPlanUsesDefaultAndExplicitGpuSlots)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
        GTEST_SKIP() << hipDevice.message();

    auto problem = makeBatchProblem(32, 24, 16, 4);
    auto args    = makeRingArgs({{32, 24, 4, 16}});

    ClientProblemFactory      factory(args);
    CopyPlanDataInitialization dataInit(args, factory);

    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);
    ASSERT_TRUE(dataInit.altSlotsReady());

    auto const& slot0 = dataInit.slotState(0);
    auto const& slot1 = dataInit.slotState(1);
    ASSERT_TRUE(slot0.populated());
    ASSERT_TRUE(slot1.populated());

    std::vector<void*>              defaultPtrs(problem.tensors().size(), nullptr);
    std::vector<void**>             defaultBatchPtrs(problem.tensors().size(), nullptr);
    std::vector<size_t>             defaultMaxElements(problem.tensors().size(), 0);
    std::vector<std::vector<size_t>> defaultOffsets(problem.tensors().size());
    dataInit.copyInputs(defaultPtrs,
                        defaultBatchPtrs,
                        defaultMaxElements,
                        defaultOffsets,
                        problem,
                        hipMemcpyDeviceToDevice);

    std::vector<void*>              explicitPtrs(problem.tensors().size(), nullptr);
    std::vector<void**>             explicitBatchPtrs(problem.tensors().size(), nullptr);
    std::vector<size_t>             explicitMaxElements(problem.tensors().size(), 0);
    std::vector<std::vector<size_t>> explicitOffsets(problem.tensors().size());
    dataInit.copyInputsForSlot(explicitPtrs,
                               explicitBatchPtrs,
                               explicitMaxElements,
                               explicitOffsets,
                               problem,
                               hipMemcpyDeviceToDevice,
                               1);

    for(size_t i = 0; i < problem.tensors().size(); ++i)
    {
        EXPECT_EQ(defaultPtrs.at(i), slot0.ptrs.at(i));
        EXPECT_EQ(defaultBatchPtrs.at(i), slot0.batchPtrs.at(i));
        EXPECT_EQ(defaultMaxElements.at(i), slot0.maxElements.at(i));
        EXPECT_EQ(defaultOffsets.at(i), slot0.groupedOffsets.at(i));

        EXPECT_EQ(explicitPtrs.at(i), slot1.ptrs.at(i));
        EXPECT_EQ(explicitBatchPtrs.at(i), slot1.batchPtrs.at(i));
        EXPECT_EQ(explicitMaxElements.at(i), slot1.maxElements.at(i));
        EXPECT_EQ(explicitOffsets.at(i), slot1.groupedOffsets.at(i));
    }
}

TEST(DataInitializationCopyPlan, OutputResetTargetsOutputsAndKeepsNonOutputsUntouched)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
        GTEST_SKIP() << hipDevice.message();

    auto problem = makeBatchProblem(32, 24, 16, 4);
    auto args    = makeGuardPageBackArgs({{32, 24, 4, 16}});

    auto engine = std::make_shared<RecordingCopyEngine>();

    ClientProblemFactory      factory(args);
    CopyPlanDataInitialization dataInit(args, factory, engine);
    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);
    engine->clear();

    auto beforePtrs      = dataInit.gpuPtrs();
    auto beforeBatchPtrs = dataInit.gpuBatchPtrs();
    auto beforeMax       = dataInit.maxElements();
    auto beforeOffsets   = dataInit.groupedOffsets();

    dataInit.resetOutput(dataInit.gpuPtrs(),
                         dataInit.gpuBatchPtrs(),
                         dataInit.maxElements(),
                         dataInit.groupedOffsets(),
                         problem,
                         hipMemcpyDeviceToDevice);

    auto const& tensors = problem.tensors();
    auto const& dUnit   = dataInit.pristineUnit(ContractionProblemGemm::TENSOR::D, problem);

    bool sawOutputCopy = false;
    for(auto const& call : engine->calls)
    {
        if(call.type != RecordingCopyEngine::CallType::Copy)
            continue;

        sawOutputCopy |= call.dst == dUnit.gpuInput.current.get();

        EXPECT_NE(call.dst, dataInit.pristineUnit(ContractionProblemGemm::TENSOR::A, problem)
                                 .gpuInput.current.get());
        EXPECT_NE(call.dst, dataInit.pristineUnit(ContractionProblemGemm::TENSOR::B, problem)
                                 .gpuInput.current.get());
        EXPECT_NE(call.dst, dataInit.pristineUnit(ContractionProblemGemm::TENSOR::C, problem)
                                 .gpuInput.current.get());
    }

    EXPECT_TRUE(sawOutputCopy);

    for(size_t i = 0; i < tensors.size(); ++i)
    {
        if(!tensors[i].isOutput())
        {
            EXPECT_EQ(dataInit.gpuPtrs().at(i), beforePtrs.at(i));
            EXPECT_EQ(dataInit.gpuBatchPtrs().at(i), beforeBatchPtrs.at(i));
            EXPECT_EQ(dataInit.maxElements().at(i), beforeMax.at(i));
            EXPECT_EQ(dataInit.groupedOffsets().at(i), beforeOffsets.at(i));
        }
        else
        {
            EXPECT_EQ(dataInit.gpuPtrs().at(i), dUnit.gpuInput.current.get());
            EXPECT_EQ(dataInit.gpuBatchPtrs().at(i), dUnit.gpuInput.batch.get());
        }
    }
}

TEST(DataInitializationCopyPlan, OutputResetForSlotUsesExplicitGpuSlot)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
        GTEST_SKIP() << hipDevice.message();

    auto problem = makeBatchProblem(32, 24, 16, 4);
    auto args    = makeRingArgs({{32, 24, 4, 16}});

    ClientProblemFactory      factory(args);
    CopyPlanDataInitialization dataInit(args, factory);

    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);
    ASSERT_TRUE(dataInit.altSlotsReady());

    auto const& slot0 = dataInit.slotState(0);
    auto const& slot1 = dataInit.slotState(1);

    std::vector<void*>              ptrs = slot0.ptrs;
    std::vector<void**>             batchPtrs = slot0.batchPtrs;
    std::vector<size_t>             maxElements = slot0.maxElements;
    std::vector<std::vector<size_t>> offsets = slot0.groupedOffsets;

    dataInit.resetOutputForSlot(ptrs,
                                batchPtrs,
                                maxElements,
                                offsets,
                                problem,
                                hipMemcpyDeviceToDevice,
                                1);

    for(size_t i = 0; i < problem.tensors().size(); ++i)
    {
        auto const& desc = problem.tensors().at(i);
        if(!desc.isOutput())
        {
            EXPECT_EQ(ptrs.at(i), slot0.ptrs.at(i));
            EXPECT_EQ(batchPtrs.at(i), slot0.batchPtrs.at(i));
            EXPECT_EQ(maxElements.at(i), slot0.maxElements.at(i));
            EXPECT_EQ(offsets.at(i), slot0.groupedOffsets.at(i));
            continue;
        }

        EXPECT_EQ(ptrs.at(i), slot1.ptrs.at(i));
        EXPECT_EQ(batchPtrs.at(i), slot1.batchPtrs.at(i));
        EXPECT_EQ(maxElements.at(i), slot1.maxElements.at(i));
        EXPECT_EQ(offsets.at(i), slot1.groupedOffsets.at(i));
    }
}

TEST(DataInitializationCopyPlan, InputGuardPageBackSwizzledBatchPointersUsePolicyGeometry)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
        GTEST_SKIP() << hipDevice.message();

    auto problem = makeBatchProblem(17, 19, 23, 4, true, true);
    auto args    = makeGuardPageBackArgs({{17, 19, 4, 23}}, true, true);

    auto engine = std::make_shared<RecordingCopyEngine>(
        reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(0x2468)));

    ClientProblemFactory      factory(args);
    CopyPlanDataInitialization dataInit(args, factory, engine);

    dataInit.prepareGPUInputs(problem);
    engine->clear();

    dataInit.initializeGPUBatchedInputs(problem);

    InputLayoutPolicy const policy;
    auto const aPlan = policy.planTensorSwizzle(problem, ContractionProblemGemm::TENSOR::A);
    auto const bPlan = policy.planTensorSwizzle(problem, ContractionProblemGemm::TENSOR::B);
    auto const& aUnit = dataInit.pristineUnit(ContractionProblemGemm::TENSOR::A, problem);
    auto const& bUnit = dataInit.pristineUnit(ContractionProblemGemm::TENSOR::B, problem);

    auto const expectedADst = static_cast<uint8_t*>(aUnit.gpuInput.current.get())
                              + multiplyElementSize(
                                  aUnit.maxElements - aPlan.allocatedElements,
                                  problem.tensors().at(ContractionProblemGemm::TENSOR::A)
                                      .elementBytes());
    auto const expectedBDst = static_cast<uint8_t*>(bUnit.gpuInput.current.get())
                              + multiplyElementSize(
                                  bUnit.maxElements - bPlan.allocatedElements,
                                  problem.tensors().at(ContractionProblemGemm::TENSOR::B)
                                      .elementBytes());
    auto const aLayout = makeBatchPointerLayout(
        problem.a(),
        batchPointerTensorBatchIndices(problem.batchIndices(), ContractionProblemGemm::TENSOR::A));
    auto const bLayout = makeBatchPointerLayout(
        problem.b(),
        batchPointerTensorBatchIndices(problem.batchIndices(), ContractionProblemGemm::TENSOR::B));

    auto stagingMatches = [](RecordingCopyEngine::Call const& call,
                             void*                            expectedBatchArray,
                             uint8_t*                         expectedBase,
                             BatchPointerLayout const&         layout) {
        if(call.dst != expectedBatchArray)
            return false;
        if(call.bytes != layout.count() * sizeof(void*))
            return false;

        auto const* staged = static_cast<void* const*>(call.src);
        for(size_t idx = 0; idx < layout.count(); ++idx)
        {
            if(staged[idx] != expectedBase + layout.offsets[idx])
                return false;
        }
        return true;
    };

    bool sawA = false;
    bool sawB = false;
    for(auto const& call : engine->calls)
    {
        if(call.type != RecordingCopyEngine::CallType::Copy)
            continue;
        if(call.copyKind != hipMemcpyHostToDevice)
            continue;
        sawA |= stagingMatches(call, aUnit.gpuInput.batch.get(), expectedADst, aLayout);
        sawB |= stagingMatches(call, bUnit.gpuInput.batch.get(), expectedBDst, bLayout);
    }

    EXPECT_TRUE(sawA);
    EXPECT_TRUE(sawB);
}

TEST(DataInitializationCopyPlan, CopyValidSkipsPolicySpecializedTensors)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
        GTEST_SKIP() << hipDevice.message();

    auto problem = makeBatchProblem(32, 24, 16, 4, /*swizzleTensorA=*/true);
    auto args    = makeBaseArgs({{32, 24, 4, 16}});

    auto engine = std::make_shared<RecordingCopyEngine>(
        reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(0x5678)));

    ClientProblemFactory      factory(args);
    CopyPlanDataInitialization dataInit(args, factory, engine);

    dataInit.prepareCPUInputs(problem);
    engine->clear();

    dataInit.copyValidToGPUBuffer(problem, /*callerOwnsCopySync=*/false);

    auto const& aUnit = dataInit.pristineUnit(ContractionProblemGemm::TENSOR::A, problem);
    auto const& bUnit = dataInit.pristineUnit(ContractionProblemGemm::TENSOR::B, problem);
    auto const& cUnit = dataInit.pristineUnit(ContractionProblemGemm::TENSOR::C, problem);
    auto const& dUnit = dataInit.pristineUnit(ContractionProblemGemm::TENSOR::D, problem);

    bool sawA = false;
    bool sawB = false;
    bool sawC = false;
    bool sawD = false;
    for(auto const& call : engine->calls)
    {
        if(call.type != RecordingCopyEngine::CallType::Copy)
            continue;
        sawA |= call.dst == aUnit.gpuInput.valid.get();
        sawB |= call.dst == bUnit.gpuInput.valid.get();
        sawC |= call.dst == cUnit.gpuInput.valid.get();
        sawD |= call.dst == dUnit.gpuInput.valid.get();
    }

    EXPECT_FALSE(sawA);
    EXPECT_TRUE(sawB);
    EXPECT_TRUE(sawC);
    EXPECT_TRUE(sawD);

    engine->clear();
    dataInit.copySwizzledToGPUBuffer(problem);

    bool sawSwizzledA = false;
    for(auto const& call : engine->calls)
    {
        if(call.type != RecordingCopyEngine::CallType::Copy)
            continue;
        sawSwizzledA |= call.dst == aUnit.gpuInput.valid.get();
    }

    EXPECT_TRUE(sawSwizzledA);
}
