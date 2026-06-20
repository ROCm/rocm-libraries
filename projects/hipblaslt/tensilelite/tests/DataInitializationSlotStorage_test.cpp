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
