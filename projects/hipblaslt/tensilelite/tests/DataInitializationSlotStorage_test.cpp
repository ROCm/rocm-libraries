// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <any>
#include <cstdint>
#include <memory>

#include <hip/hip_runtime.h>

#include <Tensile/Utils.hpp>

#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"
#include "DataInitializationTestUtils.hpp"
#include "HipStreamGuard.hpp"
#include "RecordingCopyEngine.hpp"

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    using TensileLite::testing::HipStreamGuard;
    using TensileLite::testing::makePlainProblem;
    using TensileLite::testing::RecordingCopyEngine;

    class SlotStorageDataInitialization : public DataInitialization
    {
    public:
        using DataInitialization::DataInitialization;
        using DataInitialization::primeNextInputSlot;
        using DataInitialization::ringEligible;
        using DataInitialization::waitForPreparedSlot;

        bool ringPolicyAllowed() const
        {
            return m_ringPolicy.allowed;
        }

        bool ringPolicyAllocatesAltBuffers() const
        {
            return m_ringPolicy.allocatesAltBuffers();
        }

        size_t activeBufferCount() const
        {
            return m_ring.activeBufferCount();
        }

        size_t ringAvailableSlots() const
        {
            return m_ring.availableSlots();
        }

        bool hasAltBuffers() const
        {
            return m_hasAltBuffers;
        }

        bool warmOutputResetRequired() const
        {
            return m_warmOutputResetRequired;
        }

        bool ringHasAvailableSlot() const
        {
            return m_ring.hasAvailableSlot();
        }

        bool ringNeedsCopyBarrier() const
        {
            return m_ring.needsCopyBarrier();
        }

        size_t activeRingSlot() const
        {
            return m_ring.activeSlot();
        }

        bool altSlotsReady() const
        {
            return m_altSlotsReady;
        }

        auto const& slotState(size_t slot) const
        {
            return m_gpuInputSlots.at(slot);
        }

        auto nextPrimeSlot() const
        {
            return m_ring.nextPrimeSlot();
        }

        bool allAltGpuInputsCleared() const
        {
            for(auto const& vd : m_vdata)
            {
                for(auto const& [_, pUnit] : vd.pristine)
                {
                    for(size_t slot = 1; slot < MAX_BUFFER_SETS; ++slot)
                    {
                        if(pUnit.gpuInput.buffers[slot] || pUnit.gpuInput.batchBufs[slot])
                            return false;
                    }
                }
            }

            for(size_t slot = 1; slot < MAX_BUFFER_SETS; ++slot)
            {
                auto const& slotState = m_gpuInputSlots.at(slot);
                if(slotState.populated() || !slotState.batchPtrs.empty()
                   || slotState.cachedInputs)
                    return false;
            }

            return true;
        }

        PristineUnit const& pristineUnit(size_t tensorIndex,
                                         ContractionProblemGemm const& problem) const
        {
            auto const& desc = problem.tensors().at(tensorIndex);
            auto const& units = m_vdata.at(tensorIndex).pristine;
            auto        it    = units.find(desc.dataType());
            if(it == units.end())
            {
                throw std::runtime_error("Missing pristine unit for tensor index.");
            }
            return it->second;
        }

        bool gpuInputSlotAllocated(size_t tensorIndex,
                                   ContractionProblemGemm const& problem,
                                   size_t slot) const
        {
            auto const& gpuInput = pristineUnit(tensorIndex, problem).gpuInput;
            return gpuInput.buffers[slot] || gpuInput.batchBufs[slot];
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

    Client::po::variables_map makeRingArgs(std::vector<std::vector<size_t>> problemSizes)
    {
        return TensileLite::testing::buildRingArgs(std::move(problemSizes), 1);
    }

    BoundsCheckMode expectedCurBoundsCheck(BoundsCheckMode mode)
    {
        return mode == BoundsCheckMode::GuardPageAll ? BoundsCheckMode::GuardPageFront : mode;
    }

    void expectBoundsCheckDoesNotPrimeRing(BoundsCheckMode boundsCheckMode)
    {
        auto problem = makePlainProblem(32, 32, 32);
        auto args    = makeRingArgs({{32, 32, 32}});
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "bounds-check",
                                                     std::any(boundsCheckMode));

        auto engine = std::make_shared<RecordingCopyEngine>();

        ClientProblemFactory         factory(args);
        SlotStorageDataInitialization dataInit(args, factory, engine);

        auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
        ASSERT_NE(inputs, nullptr);

        EXPECT_TRUE(dataInit.ringPolicyAllowed());
        EXPECT_TRUE(dataInit.ringPolicyAllocatesAltBuffers());
        EXPECT_EQ(dataInit.activeBufferCount(), 3u);
        EXPECT_TRUE(dataInit.hasAltBuffers());
        EXPECT_TRUE(dataInit.altSlotsReady());
        EXPECT_EQ(dataInit.getCurBoundsCheck(), expectedCurBoundsCheck(boundsCheckMode));
        EXPECT_NE(dataInit.getCurBoundsCheck(), BoundsCheckMode::Disable);
        EXPECT_FALSE(dataInit.ringEligible());
        EXPECT_EQ(dataInit.activeRingSlot(), 0u);
        EXPECT_EQ(dataInit.ringAvailableSlots(), 0u);
        EXPECT_FALSE(dataInit.ringHasAvailableSlot());
        EXPECT_FALSE(dataInit.ringNeedsCopyBarrier());

        auto const candidateSlot = dataInit.nextPrimeSlot();
        ASSERT_TRUE(candidateSlot.has_value());
        EXPECT_EQ(*candidateSlot, 1u);

        auto const& candidateState = dataInit.slotState(*candidateSlot);
        ASSERT_NE(candidateState.cachedInputs, nullptr);
        auto const candidateInputs = candidateState.cachedInputs;

        auto const& slot0State = dataInit.slotState(0);
        ASSERT_NE(slot0State.cachedInputs, nullptr);
        auto const slot0Inputs = slot0State.cachedInputs;

        EXPECT_EQ(inputs, slot0Inputs);

        engine->clear();

        dataInit.beginAsyncReset(&problem);

        EXPECT_TRUE(engine->calls.empty());
        EXPECT_FALSE(dataInit.ringEligible());
        EXPECT_EQ(dataInit.activeRingSlot(), 0u);
        EXPECT_EQ(dataInit.ringAvailableSlots(), 0u);
        EXPECT_FALSE(dataInit.ringHasAvailableSlot());
        EXPECT_FALSE(dataInit.ringNeedsCopyBarrier());
        EXPECT_EQ(dataInit.slotState(*candidateSlot).cachedInputs, candidateInputs);

        auto secondInputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
        ASSERT_NE(secondInputs, nullptr);
        EXPECT_EQ(secondInputs, slot0Inputs);
        EXPECT_NE(secondInputs, candidateInputs);
        EXPECT_EQ(dataInit.slotState(*candidateSlot).cachedInputs, candidateInputs);
        EXPECT_EQ(dataInit.activeRingSlot(), 0u);
        EXPECT_EQ(dataInit.ringAvailableSlots(), 0u);
        EXPECT_FALSE(dataInit.ringHasAvailableSlot());
        EXPECT_FALSE(dataInit.ringNeedsCopyBarrier());

        dataInit.waitForPreparedSlot(nullptr);

        bool sawRecordCopyDone = false;
        bool sawWaitForCopyDone = false;
        for(auto const& call : engine->calls)
        {
            if(call.type == RecordingCopyEngine::CallType::RecordCopyDone)
                sawRecordCopyDone = true;
            else if(call.type == RecordingCopyEngine::CallType::WaitForCopyDone)
                sawWaitForCopyDone = true;
        }

        EXPECT_FALSE(sawRecordCopyDone);
        EXPECT_FALSE(sawWaitForCopyDone);
    }
} // namespace

TEST(DataInitializationSlotStorage, PrimingAltSlotDoesNotMutateActiveAliases)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    auto problem = makePlainProblem(32, 32, 32);
    auto args    = makeRingArgs({{32, 32, 32}});

    ClientProblemFactory         factory(args);
    SlotStorageDataInitialization dataInit(args, factory);

    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);
    ASSERT_TRUE(dataInit.ringEligible());
    ASSERT_TRUE(dataInit.altSlotsReady());

    auto const targetSlot = dataInit.nextPrimeSlot();
    ASSERT_TRUE(targetSlot.has_value());
    ASSERT_TRUE(dataInit.slotState(*targetSlot).populated());

    auto& aUnit = dataInit.pristineUnit(ContractionProblemGemm::TENSOR::A, problem);
    auto& dUnit = dataInit.pristineUnit(ContractionProblemGemm::TENSOR::D, problem);
    auto const  slot0Cached = dataInit.slotState(0).cachedInputs;
    auto const  slot0A      = aUnit.gpuInput.current.get();
    auto const  slot0D      = dUnit.gpuInput.current.get();
    auto const  slot0BatchA = aUnit.gpuInput.batch.get();
    auto const  slot0BatchD = dUnit.gpuInput.batch.get();

    dataInit.primeNextInputSlot(&problem);

    EXPECT_EQ(aUnit.gpuInput.current.get(), slot0A);
    EXPECT_EQ(aUnit.gpuInput.batch.get(), slot0BatchA);
    EXPECT_EQ(dUnit.gpuInput.current.get(), slot0D);
    EXPECT_EQ(dUnit.gpuInput.batch.get(), slot0BatchD);
    EXPECT_EQ(dataInit.slotState(0).cachedInputs, slot0Cached);

    auto warmInputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(warmInputs, nullptr);
    auto* warmCi = dynamic_cast<ContractionInputs*>(warmInputs.get());
    ASSERT_NE(warmCi, nullptr);

    auto const& targetState = dataInit.slotState(*targetSlot);
    EXPECT_EQ(warmInputs, targetState.cachedInputs);
    EXPECT_EQ(warmCi->a, targetState.ptrs.at(ContractionProblemGemm::TENSOR::A));
    EXPECT_EQ(warmCi->d, targetState.ptrs.at(ContractionProblemGemm::TENSOR::D));
    EXPECT_EQ(warmCi->batchA, targetState.batchPtrs.at(ContractionProblemGemm::TENSOR::A));
    EXPECT_EQ(warmCi->batchD, targetState.batchPtrs.at(ContractionProblemGemm::TENSOR::D));
    EXPECT_NE(warmCi->a, slot0A);
    EXPECT_NE(warmCi->d, slot0D);
    EXPECT_NE(warmCi->batchA, slot0BatchA);
    EXPECT_NE(warmCi->batchD, slot0BatchD);

    HipStreamGuard computeStream(hipStreamNonBlocking);
    dataInit.waitForPreparedSlot(computeStream.get());
}

TEST(DataInitializationSlotStorage, PristineOnGpuFalseDisablesWarmD2DReset)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    auto problem = makePlainProblem(32, 32, 32);
    auto args    = makeRingArgs({{32, 32, 32}});
    TensileLite::testing::detail::setDataInitArg(args, "pristine-on-gpu", std::any(false));

    auto engine = std::make_shared<RecordingCopyEngine>(
        reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(0x2468)));

    ClientProblemFactory         factory(args);
    SlotStorageDataInitialization dataInit(args, factory, engine);

    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);
    ASSERT_TRUE(dataInit.ringPolicyAllowed());
    ASSERT_EQ(dataInit.activeBufferCount(), 3u);
    ASSERT_TRUE(dataInit.warmOutputResetRequired());
    EXPECT_TRUE(dataInit.hasAltBuffers());
    EXPECT_FALSE(dataInit.ringEligible());
    EXPECT_FALSE(dataInit.ringHasAvailableSlot());

    engine->clear();

    dataInit.primeNextInputSlot(&problem);

    EXPECT_TRUE(engine->calls.empty());
    EXPECT_FALSE(dataInit.ringHasAvailableSlot());
    EXPECT_EQ(dataInit.activeRingSlot(), 0u);
}

TEST(DataInitializationSlotStorage, BoundsCheckDoesNotPrimeRing)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    for(BoundsCheckMode mode : {BoundsCheckMode::NaN,
                                BoundsCheckMode::GuardPageFront,
                                BoundsCheckMode::GuardPageBack,
                                BoundsCheckMode::GuardPageAll})
    {
        SCOPED_TRACE(::testing::Message() << "bounds-check=" << mode);
        expectBoundsCheckDoesNotPrimeRing(mode);
    }
}

TEST(DataInitializationSlotStorage,
     TimedBenchmarkConfigurationDoesNotAllocateAltSlotsOrAdvanceRing)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    auto problem = makePlainProblem(32, 32, 32);
    auto args    = makeRingArgs({{32, 32, 32}});
    TensileLite::testing::detail::setDataInitArg(args, "num-benchmarks", std::any(int(1)));
    TensileLite::testing::detail::setDataInitArg(
        args, "num-enqueues-per-sync", std::any(int(1)));
    TensileLite::testing::detail::setDataInitArg(
        args, "num-syncs-per-benchmark", std::any(int(1)));
    TensileLite::testing::detail::setDataInitArg(args,
                                                 "max-enqueues-per-sync",
                                                 std::any(int(-1)));
    TensileLite::testing::detail::setDataInitArg(args,
                                                 "min-flops-per-sync",
                                                 std::any(size_t(0)));

    auto engine = std::make_shared<RecordingCopyEngine>();

    ClientProblemFactory         factory(args);
    SlotStorageDataInitialization dataInit(args, factory, engine);

    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);
    auto* initialCi = dynamic_cast<ContractionInputs*>(inputs.get());
    ASSERT_NE(initialCi, nullptr);

    EXPECT_FALSE(dataInit.ringPolicyAllowed());
    EXPECT_FALSE(dataInit.ringPolicyAllocatesAltBuffers());
    EXPECT_EQ(dataInit.activeBufferCount(), 1u);
    EXPECT_FALSE(dataInit.hasAltBuffers());
    EXPECT_TRUE(dataInit.warmOutputResetRequired());
    EXPECT_FALSE(dataInit.ringEligible());
    EXPECT_FALSE(dataInit.altSlotsReady());
    EXPECT_EQ(dataInit.activeRingSlot(), 0u);
    EXPECT_EQ(dataInit.ringAvailableSlots(), 0u);
    EXPECT_FALSE(dataInit.ringHasAvailableSlot());
    EXPECT_FALSE(dataInit.ringNeedsCopyBarrier());
    EXPECT_FALSE(dataInit.nextPrimeSlot().has_value());
    EXPECT_TRUE(dataInit.allAltGpuInputsCleared());

    auto const& slot0State = dataInit.slotState(0);
    ASSERT_TRUE(slot0State.populated());
    ASSERT_NE(slot0State.cachedInputs, nullptr);

    auto const slot0Inputs = slot0State.cachedInputs;
    auto const slot0A      = slot0State.ptrs.at(ContractionProblemGemm::TENSOR::A);
    auto const slot0D      = slot0State.ptrs.at(ContractionProblemGemm::TENSOR::D);
    auto const slot0BatchA = slot0State.batchPtrs.at(ContractionProblemGemm::TENSOR::A);
    auto const slot0BatchD = slot0State.batchPtrs.at(ContractionProblemGemm::TENSOR::D);

    ASSERT_NE(slot0A, nullptr);
    ASSERT_NE(slot0D, nullptr);
    ASSERT_NE(slot0BatchA, nullptr);
    ASSERT_NE(slot0BatchD, nullptr);
    EXPECT_EQ(inputs, slot0Inputs);
    EXPECT_EQ(initialCi->a, slot0A);
    EXPECT_EQ(initialCi->d, slot0D);
    EXPECT_EQ(initialCi->batchA, slot0BatchA);
    EXPECT_EQ(initialCi->batchD, slot0BatchD);

    engine->clear();

    dataInit.primeNextInputSlot(&problem);

    EXPECT_TRUE(engine->calls.empty());
    EXPECT_EQ(dataInit.activeRingSlot(), 0u);
    EXPECT_EQ(dataInit.ringAvailableSlots(), 0u);
    EXPECT_FALSE(dataInit.ringNeedsCopyBarrier());
    EXPECT_FALSE(dataInit.altSlotsReady());
    EXPECT_TRUE(dataInit.allAltGpuInputsCleared());

    auto secondInputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(secondInputs, nullptr);
    auto* secondCi = dynamic_cast<ContractionInputs*>(secondInputs.get());
    ASSERT_NE(secondCi, nullptr);
    EXPECT_EQ(secondInputs, slot0Inputs);
    EXPECT_EQ(secondCi->a, slot0A);
    EXPECT_EQ(secondCi->d, slot0D);
    EXPECT_EQ(secondCi->batchA, slot0BatchA);
    EXPECT_EQ(secondCi->batchD, slot0BatchD);
    EXPECT_EQ(dataInit.activeRingSlot(), 0u);
    EXPECT_EQ(dataInit.ringAvailableSlots(), 0u);
    EXPECT_FALSE(dataInit.ringNeedsCopyBarrier());
    EXPECT_FALSE(dataInit.altSlotsReady());
    EXPECT_TRUE(dataInit.allAltGpuInputsCleared());

    dataInit.waitForPreparedSlot(nullptr);

    bool sawRecordCopyDone = false;
    bool sawWaitForCopyDone = false;
    for(auto const& call : engine->calls)
    {
        if(call.type == RecordingCopyEngine::CallType::RecordCopyDone)
            sawRecordCopyDone = true;
        else if(call.type == RecordingCopyEngine::CallType::WaitForCopyDone)
            sawWaitForCopyDone = true;
    }

    EXPECT_FALSE(sawRecordCopyDone);
    EXPECT_FALSE(sawWaitForCopyDone);
}

TEST(DataInitializationSlotStorage, Slot2NotAllocatedWhenActiveRingSizeIsOne)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    auto problem = makePlainProblem(32, 32, 32);
    auto args    = makeRingArgs({{32, 32, 32}});
    TensileLite::testing::detail::setDataInitArg(args, "num-benchmarks", std::any(int(1)));
    TensileLite::testing::detail::setDataInitArg(
        args, "num-enqueues-per-sync", std::any(int(1)));
    TensileLite::testing::detail::setDataInitArg(
        args, "num-syncs-per-benchmark", std::any(int(1)));
    TensileLite::testing::detail::setDataInitArg(args,
                                                 "max-enqueues-per-sync",
                                                 std::any(int(-1)));
    TensileLite::testing::detail::setDataInitArg(args,
                                                 "min-flops-per-sync",
                                                 std::any(size_t(0)));

    ClientProblemFactory         factory(args);
    SlotStorageDataInitialization dataInit(args, factory);

    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);

    EXPECT_FALSE(dataInit.ringPolicyAllowed());
    EXPECT_FALSE(dataInit.ringPolicyAllocatesAltBuffers());
    EXPECT_EQ(dataInit.activeBufferCount(), 1u);
    EXPECT_FALSE(dataInit.hasAltBuffers());
    EXPECT_FALSE(dataInit.ringEligible());
    EXPECT_TRUE(dataInit.allAltGpuInputsCleared());

    auto const& slot0State = dataInit.slotState(0);
    ASSERT_TRUE(slot0State.populated());
    ASSERT_NE(slot0State.cachedInputs, nullptr);

    auto const& slot2State = dataInit.slotState(2);
    EXPECT_FALSE(slot2State.populated());
    EXPECT_TRUE(slot2State.batchPtrs.empty());
    EXPECT_EQ(slot2State.cachedInputs, nullptr);

    for(size_t tensorIndex = 0; tensorIndex < problem.tensors().size(); ++tensorIndex)
    {
        EXPECT_FALSE(dataInit.gpuInputSlotAllocated(tensorIndex, problem, 2))
            << "tensor index " << tensorIndex;
    }

    dataInit.primeNextInputSlot(&problem);

    EXPECT_TRUE(dataInit.allAltGpuInputsCleared());
    EXPECT_FALSE(dataInit.slotState(2).populated());
    EXPECT_TRUE(dataInit.slotState(2).batchPtrs.empty());
    EXPECT_EQ(dataInit.slotState(2).cachedInputs, nullptr);

    for(size_t tensorIndex = 0; tensorIndex < problem.tensors().size(); ++tensorIndex)
    {
        EXPECT_FALSE(dataInit.gpuInputSlotAllocated(tensorIndex, problem, 2))
            << "tensor index " << tensorIndex;
    }
}

TEST(DataInitializationSlotStorage,
     ZeroEnqueueNoValidationConfigurationDoesNotClaimWarmRingBehavior)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    auto problem = makePlainProblem(32, 32, 32);
    auto args    = TensileLite::testing::buildRingArgs({{32, 32, 32}}, 0);

    auto engine = std::make_shared<RecordingCopyEngine>(
        reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(0x8642)));

    ClientProblemFactory         factory(args);
    SlotStorageDataInitialization dataInit(args, factory, engine);

    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);
    auto* initialCi = dynamic_cast<ContractionInputs*>(inputs.get());
    ASSERT_NE(initialCi, nullptr);

    EXPECT_FALSE(dataInit.ringPolicyAllowed());
    EXPECT_FALSE(dataInit.ringPolicyAllocatesAltBuffers());
    EXPECT_EQ(dataInit.activeBufferCount(), 1u);
    EXPECT_FALSE(dataInit.hasAltBuffers());
    EXPECT_FALSE(dataInit.warmOutputResetRequired());
    EXPECT_FALSE(dataInit.ringEligible());
    EXPECT_FALSE(dataInit.altSlotsReady());
    EXPECT_EQ(dataInit.activeRingSlot(), 0u);
    EXPECT_EQ(dataInit.ringAvailableSlots(), 0u);
    EXPECT_FALSE(dataInit.ringHasAvailableSlot());
    EXPECT_FALSE(dataInit.ringNeedsCopyBarrier());
    EXPECT_FALSE(dataInit.nextPrimeSlot().has_value());
    EXPECT_TRUE(dataInit.allAltGpuInputsCleared());

    auto const& slot0State = dataInit.slotState(0);
    ASSERT_TRUE(slot0State.populated());
    ASSERT_NE(slot0State.cachedInputs, nullptr);

    auto const slot0Inputs = slot0State.cachedInputs;
    auto const slot0A      = slot0State.ptrs.at(ContractionProblemGemm::TENSOR::A);
    auto const slot0D      = slot0State.ptrs.at(ContractionProblemGemm::TENSOR::D);
    auto const slot0BatchA = slot0State.batchPtrs.at(ContractionProblemGemm::TENSOR::A);
    auto const slot0BatchD = slot0State.batchPtrs.at(ContractionProblemGemm::TENSOR::D);

    ASSERT_NE(slot0A, nullptr);
    ASSERT_NE(slot0D, nullptr);
    ASSERT_NE(slot0BatchA, nullptr);
    ASSERT_NE(slot0BatchD, nullptr);
    EXPECT_EQ(inputs, slot0Inputs);
    EXPECT_EQ(initialCi->a, slot0A);
    EXPECT_EQ(initialCi->d, slot0D);
    EXPECT_EQ(initialCi->batchA, slot0BatchA);
    EXPECT_EQ(initialCi->batchD, slot0BatchD);

    engine->clear();

    dataInit.primeNextInputSlot(&problem);

    EXPECT_EQ(dataInit.activeRingSlot(), 0u);
    EXPECT_EQ(dataInit.ringAvailableSlots(), 0u);
    EXPECT_FALSE(dataInit.ringHasAvailableSlot());
    EXPECT_FALSE(dataInit.ringNeedsCopyBarrier());
    EXPECT_FALSE(dataInit.altSlotsReady());
    EXPECT_FALSE(dataInit.warmOutputResetRequired());
    EXPECT_FALSE(dataInit.ringEligible());
    EXPECT_FALSE(dataInit.nextPrimeSlot().has_value());
    EXPECT_TRUE(dataInit.allAltGpuInputsCleared());

    auto secondInputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(secondInputs, nullptr);
    auto* secondCi = dynamic_cast<ContractionInputs*>(secondInputs.get());
    ASSERT_NE(secondCi, nullptr);
    EXPECT_EQ(secondInputs, slot0Inputs);
    EXPECT_EQ(secondCi->a, slot0A);
    EXPECT_EQ(secondCi->d, slot0D);
    EXPECT_EQ(secondCi->batchA, slot0BatchA);
    EXPECT_EQ(secondCi->batchD, slot0BatchD);
    EXPECT_EQ(dataInit.activeRingSlot(), 0u);
    EXPECT_EQ(dataInit.ringAvailableSlots(), 0u);
    EXPECT_FALSE(dataInit.ringHasAvailableSlot());
    EXPECT_FALSE(dataInit.ringNeedsCopyBarrier());
    EXPECT_FALSE(dataInit.altSlotsReady());
    EXPECT_FALSE(dataInit.warmOutputResetRequired());
    EXPECT_FALSE(dataInit.ringEligible());
    EXPECT_TRUE(dataInit.allAltGpuInputsCleared());

    dataInit.waitForPreparedSlot(nullptr);

    bool sawWarmRingHook = false;
    for(auto const& call : engine->calls)
    {
        if(call.type == RecordingCopyEngine::CallType::RecordCopyDone
           || call.type == RecordingCopyEngine::CallType::WaitForCopyDone)
        {
            sawWarmRingHook = true;
            break;
        }
    }

    EXPECT_FALSE(sawWarmRingHook);
}

TEST(DataInitializationSlotStorage, FastPathReturnsDistinctValidAltSlot)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    auto problem = makePlainProblem(32, 32, 32);
    auto args    = makeRingArgs({{32, 32, 32}});
    TensileLite::testing::detail::setDataInitArg(
        args, "init-a", std::any(Client::InitMode::Two));
    TensileLite::testing::detail::setDataInitArg(
        args, "init-b", std::any(Client::InitMode::One));

    ClientProblemFactory         factory(args);
    SlotStorageDataInitialization dataInit(args, factory);

    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);
    ASSERT_TRUE(dataInit.ringEligible());
    ASSERT_TRUE(dataInit.altSlotsReady());

    auto* initialCi = dynamic_cast<ContractionInputs*>(inputs.get());
    ASSERT_NE(initialCi, nullptr);

    auto const targetSlot = dataInit.nextPrimeSlot();
    ASSERT_TRUE(targetSlot.has_value());
    ASSERT_TRUE(dataInit.slotState(*targetSlot).populated());

    auto const slot0Cached = dataInit.slotState(0).cachedInputs;
    auto const slot0A      = initialCi->a;
    auto const slot0B      = initialCi->b;
    ASSERT_NE(slot0A, nullptr);
    ASSERT_NE(slot0B, nullptr);
    EXPECT_EQ(inputs, slot0Cached);

    dataInit.primeNextInputSlot(&problem);

    auto warmInputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(warmInputs, nullptr);
    auto* warmCi = dynamic_cast<ContractionInputs*>(warmInputs.get());
    ASSERT_NE(warmCi, nullptr);
    ASSERT_NE(warmCi->a, nullptr);
    ASSERT_NE(warmCi->b, nullptr);

    auto const& targetState = dataInit.slotState(*targetSlot);
    EXPECT_EQ(warmInputs, targetState.cachedInputs);
    EXPECT_NE(warmInputs, inputs);
    EXPECT_EQ(warmCi->a, targetState.ptrs.at(ContractionProblemGemm::TENSOR::A));
    EXPECT_NE(warmCi->a, slot0A);
    EXPECT_EQ(warmCi->b, targetState.ptrs.at(ContractionProblemGemm::TENSOR::B));
    EXPECT_NE(warmCi->b, slot0B);

    dataInit.waitForPreparedSlot(nullptr);

    float aValue = 0.0f;
    float bValue = 0.0f;
    HIP_CHECK_EXC(hipMemcpy(&aValue, warmCi->a, sizeof(aValue), hipMemcpyDeviceToHost));
    HIP_CHECK_EXC(hipMemcpy(&bValue, warmCi->b, sizeof(bValue), hipMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(aValue, 2.0f);
    EXPECT_FLOAT_EQ(bValue, 1.0f);
}

TEST(DataInitializationSlotStorage, ValidationOnlyConfigurationUsesThreeSlots)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    auto problem = makePlainProblem(32, 32, 32);
    auto args    = makeRingArgs({{32, 32, 32}});

    auto engine = std::make_shared<RecordingCopyEngine>(
        reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(0x7654)));

    ClientProblemFactory         factory(args);
    SlotStorageDataInitialization dataInit(args, factory, engine);

    auto initialInputs
        = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(initialInputs, nullptr);

    EXPECT_TRUE(dataInit.ringPolicyAllowed());
    EXPECT_TRUE(dataInit.ringPolicyAllocatesAltBuffers());
    EXPECT_EQ(dataInit.activeBufferCount(), 3u);
    EXPECT_TRUE(dataInit.hasAltBuffers());
    EXPECT_TRUE(dataInit.warmOutputResetRequired());
    EXPECT_TRUE(dataInit.ringEligible());
    EXPECT_TRUE(dataInit.altSlotsReady());
    EXPECT_EQ(dataInit.activeRingSlot(), 0u);
    EXPECT_EQ(dataInit.ringAvailableSlots(), 0u);
    EXPECT_FALSE(dataInit.ringHasAvailableSlot());
    EXPECT_FALSE(dataInit.ringNeedsCopyBarrier());

    auto const slot0Inputs = dataInit.slotState(0).cachedInputs;
    auto const slot1Inputs = dataInit.slotState(1).cachedInputs;
    auto const slot2Inputs = dataInit.slotState(2).cachedInputs;

    ASSERT_TRUE(dataInit.slotState(0).populated());
    ASSERT_TRUE(dataInit.slotState(1).populated());
    ASSERT_TRUE(dataInit.slotState(2).populated());
    ASSERT_NE(slot0Inputs, nullptr);
    ASSERT_NE(slot1Inputs, nullptr);
    ASSERT_NE(slot2Inputs, nullptr);
    EXPECT_EQ(initialInputs, slot0Inputs);
    EXPECT_NE(slot0Inputs, slot1Inputs);
    EXPECT_NE(slot1Inputs, slot2Inputs);
    EXPECT_NE(slot0Inputs, slot2Inputs);

    engine->clear();

    auto const waitStream = reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(0x1357));
    size_t const expectedSlots[] = {1u, 2u, 0u};

    for(size_t expectedSlot : expectedSlots)
    {
        auto const nextSlot = dataInit.nextPrimeSlot();
        ASSERT_TRUE(nextSlot.has_value());
        EXPECT_EQ(*nextSlot, expectedSlot);

        dataInit.primeNextInputSlot(&problem);

        EXPECT_TRUE(dataInit.ringHasAvailableSlot());
        EXPECT_EQ(dataInit.ringAvailableSlots(), 1u);

        auto cycleInputs
            = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
        ASSERT_NE(cycleInputs, nullptr);
        EXPECT_EQ(dataInit.activeRingSlot(), expectedSlot);
        EXPECT_EQ(cycleInputs, dataInit.slotState(expectedSlot).cachedInputs);
        EXPECT_EQ(dataInit.ringAvailableSlots(), 0u);
        EXPECT_FALSE(dataInit.ringHasAvailableSlot());
        EXPECT_TRUE(dataInit.ringNeedsCopyBarrier());

        dataInit.waitForPreparedSlot(waitStream);
        EXPECT_FALSE(dataInit.ringNeedsCopyBarrier());
    }

    size_t relevantIndex = 0;
    for(auto const& call : engine->calls)
    {
        if(call.type != RecordingCopyEngine::CallType::RecordCopyDone
           && call.type != RecordingCopyEngine::CallType::WaitForCopyDone)
        {
            continue;
        }

        ASSERT_LT(relevantIndex, 6u);

        size_t const phase = relevantIndex / 2;
        size_t const slot  = expectedSlots[phase];
        if((relevantIndex % 2) == 0)
        {
            EXPECT_EQ(call.type, RecordingCopyEngine::CallType::RecordCopyDone);
            EXPECT_EQ(call.slot, slot);
            EXPECT_EQ(call.stream, engine->stream());
        }
        else
        {
            EXPECT_EQ(call.type, RecordingCopyEngine::CallType::WaitForCopyDone);
            EXPECT_EQ(call.slot, slot);
            EXPECT_EQ(call.computeStream, waitStream);
        }

        ++relevantIndex;
    }
    EXPECT_EQ(relevantIndex, 6u);
}

TEST(DataInitializationSlotStorage, SyncCopyStreamDelegatesToCopyEngine)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    auto args = makeRingArgs({{32, 32, 32}});
    auto engine = std::make_shared<RecordingCopyEngine>(
        reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(0x1357)));

    ClientProblemFactory         factory(args);
    SlotStorageDataInitialization dataInit(args, factory, engine);

    engine->clear();
    dataInit.syncCopyStream();

    ASSERT_EQ(engine->calls.size(), 1u);
    EXPECT_EQ(engine->calls[0].type, RecordingCopyEngine::CallType::SynchronizeDefaultStream);
    EXPECT_EQ(engine->calls[0].stream, engine->stream());
}

TEST(DataInitializationSlotStorage, RingPrimingAndWaitDelegateThroughCopyEngine)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    auto problem = makePlainProblem(32, 32, 32);
    auto args    = makeRingArgs({{32, 32, 32}});

    auto engine = std::make_shared<RecordingCopyEngine>(
        reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(0x4321)));

    ClientProblemFactory         factory(args);
    SlotStorageDataInitialization dataInit(args, factory, engine);

    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);
    ASSERT_TRUE(dataInit.ringEligible());

    engine->clear();

    dataInit.primeNextInputSlot(&problem);
    auto const waitStream
        = reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(0x8765));
    dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    dataInit.waitForPreparedSlot(waitStream);

    bool   sawRecord      = false;
    bool   sawWait        = false;
    size_t recordIndex    = engine->calls.size();
    size_t waitIndex      = engine->calls.size();
    for(size_t i = 0; i < engine->calls.size(); ++i)
    {
        auto const& call = engine->calls[i];
        if(call.type == RecordingCopyEngine::CallType::RecordCopyDone)
        {
            sawRecord   = true;
            recordIndex = i;
            EXPECT_EQ(call.slot, 1u);
            EXPECT_EQ(call.stream, engine->stream());
        }
        else if(call.type == RecordingCopyEngine::CallType::WaitForCopyDone)
        {
            sawWait   = true;
            waitIndex = i;
            EXPECT_EQ(call.slot, 1u);
            EXPECT_EQ(call.computeStream, waitStream);
        }
    }

    EXPECT_TRUE(sawRecord);
    EXPECT_TRUE(sawWait);
    EXPECT_LT(recordIndex, waitIndex);
}
