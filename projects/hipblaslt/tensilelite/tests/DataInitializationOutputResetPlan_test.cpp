// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <any>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include <hip/hip_runtime.h>

#include <Tensile/Utils.hpp>

#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"
#include "DataInitializationTestUtils.hpp"
#include "HipStreamGuard.hpp"

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    using TensileLite::testing::HipStreamGuard;
    using TensileLite::testing::makePlainProblem;

    struct EventHandle
    {
        hipEvent_t event = nullptr;

        explicit EventHandle(unsigned int flags = hipEventDisableTiming)
        {
            HIP_CHECK_EXC(hipEventCreateWithFlags(&event, flags));
        }

        ~EventHandle()
        {
            if(event)
                (void)hipEventDestroy(event);
        }

        EventHandle(EventHandle const&)            = delete;
        EventHandle& operator=(EventHandle const&) = delete;

        hipEvent_t get() const
        {
            return event;
        }
    };

    struct DeviceSignalBuffer
    {
        uint32_t* ptr = nullptr;

        ~DeviceSignalBuffer()
        {
            if(ptr)
                (void)hipFree(ptr);
        }

        void allocate()
        {
            void* raw = nullptr;
            HIP_CHECK_EXC(hipExtMallocWithFlags(&raw, sizeof(uint32_t), hipMallocSignalMemory));
            ptr = static_cast<uint32_t*>(raw);
        }

        uint32_t* get() const
        {
            return ptr;
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

    ::testing::AssertionResult hasHipDeviceAndWaitValueSupport()
    {
        auto deviceCheck = hasHipDevice();
        if(!deviceCheck)
            return deviceCheck;

        int        device = 0;
        hipError_t err    = hipGetDevice(&device);
        if(err != hipSuccess)
            device = 0;

        int canUseStreamWaitValue = 0;
        err = hipDeviceGetAttribute(&canUseStreamWaitValue,
                                    hipDeviceAttributeCanUseStreamWaitValue,
                                    device);
        if(err != hipSuccess)
        {
            return ::testing::AssertionFailure()
                   << "hipDeviceGetAttribute(CanUseStreamWaitValue) failed: "
                   << hipGetErrorString(err);
        }
        if(!canUseStreamWaitValue)
        {
            return ::testing::AssertionFailure()
                   << "Device does not support hipStreamWaitValue32";
        }

        return ::testing::AssertionSuccess();
    }

    struct SampledByte
    {
        size_t  offset = 0;
        uint8_t value  = 0;
    };

    std::vector<SampledByte> readSampledBytes(void const* devicePtr, size_t size)
    {
        std::vector<size_t> offsets;
        offsets.reserve(3);

        auto const addOffset = [&](size_t offset) {
            if(std::find(offsets.begin(), offsets.end(), offset) == offsets.end())
                offsets.push_back(offset);
        };

        addOffset(0);
        addOffset(size / 2);
        addOffset(size - 1);

        std::vector<SampledByte> samples;
        samples.reserve(offsets.size());

        HipStreamGuard probeStream(hipStreamNonBlocking);
        for(size_t offset : offsets)
        {
            samples.push_back({offset, 0});
            HIP_CHECK_EXC(hipMemcpyAsync(&samples.back().value,
                                         static_cast<uint8_t const*>(devicePtr) + offset,
                                         sizeof(samples.back().value),
                                         hipMemcpyDeviceToHost,
                                         probeStream.get()));
        }

        probeStream.synchronize();
        return samples;
    }

    class OutputResetPlanDataInitialization : public DataInitialization
    {
    public:
        using DataInitialization::DataInitialization;
        using Action = DataInitialization::OutputResetAction;
        using Plan   = DataInitialization::OutputResetPlan;
        using Reason = DataInitialization::OutputResetReason;
        using PristineUnit = DataInitialization::PristineUnit;
        using DataInitialization::planNormalWarmOutputReset;
        using DataInitialization::planRingSlotOutputReset;
        using DataInitialization::ringEligible;

        bool altSlotsReady() const
        {
            return m_altSlotsReady;
        }

        bool warmOutputResetRequired() const
        {
            return m_warmOutputResetRequired;
        }

        bool gpuInit() const
        {
            return m_gpuInit;
        }

        hipStream_t copyStream() const
        {
            return DataInitialization::copyStream();
        }

        std::optional<size_t> nextPrimeSlot() const
        {
            return m_ring.nextPrimeSlot();
        }

        auto const& ringSlot(size_t slot) const
        {
            return m_gpuInputSlots.at(slot);
        }

        void* ringTensorPtr(size_t slot, size_t tensorIndex) const
        {
            return ringSlot(slot).ptrs.at(tensorIndex);
        }

        void* ringBatchPtr(size_t slot, size_t tensorIndex) const
        {
            return ringSlot(slot).batchPtrs.at(tensorIndex);
        }

        void setAltSlotsReady(bool value)
        {
            m_altSlotsReady = value;
        }

        void setWarmOutputResetRequired(bool value)
        {
            m_warmOutputResetRequired = value;
        }

        void setGpuInit(bool value)
        {
            m_gpuInit = value;
        }

        void setHasAltBuffers(bool value)
        {
            m_hasAltBuffers = value;
        }

        void setProblemDependentData(bool value)
        {
            m_problemDependentData = value;
        }

        void setBoundsCheck(BoundsCheckMode value)
        {
            m_curBoundsCheck = value;
        }

        PristineUnit& dPristineUnit(ContractionProblemGemm const& problem)
        {
            auto const tensorIndex = ContractionProblemGemm::TENSOR::D;
            auto const& desc       = problem.tensors().at(tensorIndex);
            auto&       units      = m_vdata.at(tensorIndex).pristine;
            auto        it         = units.find(desc.dataType());
            if(it == units.end())
                throw std::runtime_error("Missing D pristine unit for problem data type.");
            return it->second;
        }
    };
} // namespace

TEST(DataInitializationOutputResetPlan, NormalWarmValidationPlansResetFromValid)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
        GTEST_SKIP() << hipDevice.message();

    auto args = TensileLite::testing::buildBaseDataInitArgs({{32, 32, 32}});
    TensileLite::testing::detail::setDataInitArg(args,
                                                 "num-elements-to-validate",
                                                 std::any(int(1)));

    ClientProblemFactory             factory(args);
    OutputResetPlanDataInitialization dataInit(args, factory);
    auto problem = makePlainProblem(32, 32, 32);

    auto inputs = dataInit.prepareGPUInputs(problem);
    ASSERT_NE(inputs, nullptr);

    auto const plan = dataInit.planNormalWarmOutputReset(problem);
    EXPECT_EQ(plan.action, OutputResetPlanDataInitialization::Action::ResetFromValid);
    EXPECT_EQ(plan.reason, OutputResetPlanDataInitialization::Reason::NormalWarmValidation);
    EXPECT_TRUE(plan.requiresPristineGpuCopy);
    EXPECT_FALSE(plan.usesExistingSlotContents);
    EXPECT_FALSE(plan.targetIsRingSlot);
}

TEST(DataInitializationOutputResetPlan, NormalWarmValidationWithoutPristineGpuPlansResetFromValid)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
        GTEST_SKIP() << hipDevice.message();

    auto args = TensileLite::testing::buildBaseDataInitArgs({{32, 32, 32}});
    TensileLite::testing::detail::setDataInitArg(args,
                                                 "pristine-on-gpu",
                                                 std::any(false));
    TensileLite::testing::detail::setDataInitArg(args,
                                                 "num-elements-to-validate",
                                                 std::any(int(1)));

    ClientProblemFactory             factory(args);
    OutputResetPlanDataInitialization dataInit(args, factory);
    auto problem = makePlainProblem(32, 32, 32);

    auto inputs = dataInit.prepareGPUInputs(problem);
    ASSERT_NE(inputs, nullptr);

    auto const plan = dataInit.planNormalWarmOutputReset(problem);
    EXPECT_EQ(plan.action, OutputResetPlanDataInitialization::Action::ResetFromValid);
    EXPECT_EQ(plan.reason, OutputResetPlanDataInitialization::Reason::NormalWarmValidation);
    EXPECT_FALSE(plan.requiresPristineGpuCopy);
    EXPECT_FALSE(plan.usesExistingSlotContents);
    EXPECT_FALSE(plan.targetIsRingSlot);
}

TEST(DataInitializationOutputResetPlan, NormalWarmWithoutValidationPlansNoReset)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
        GTEST_SKIP() << hipDevice.message();

    auto args = TensileLite::testing::buildBaseDataInitArgs({{32, 32, 32}});
    ClientProblemFactory             factory(args);
    OutputResetPlanDataInitialization dataInit(args, factory);
    auto problem = makePlainProblem(32, 32, 32);

    auto inputs = dataInit.prepareGPUInputs(problem);
    ASSERT_NE(inputs, nullptr);

    auto const plan = dataInit.planNormalWarmOutputReset(problem);
    EXPECT_EQ(plan.action, OutputResetPlanDataInitialization::Action::NoReset);
    EXPECT_EQ(plan.reason, OutputResetPlanDataInitialization::Reason::NormalWarmValidation);
    EXPECT_FALSE(plan.requiresPristineGpuCopy);
    EXPECT_TRUE(plan.usesExistingSlotContents);
    EXPECT_FALSE(plan.targetIsRingSlot);
}

TEST(DataInitializationOutputResetPlan, NormalColdOrSwizzledPlansFullFill)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
        GTEST_SKIP() << hipDevice.message();

    auto args = TensileLite::testing::buildBaseDataInitArgs({{32, 32, 32}});
    ClientProblemFactory             factory(args);
    OutputResetPlanDataInitialization dataInit(args, factory);
    auto problem = makePlainProblem(32, 32, 32);

    auto const coldPlan = dataInit.planNormalWarmOutputReset(problem);
    EXPECT_EQ(coldPlan.action, OutputResetPlanDataInitialization::Action::FullFill);
    EXPECT_EQ(coldPlan.reason, OutputResetPlanDataInitialization::Reason::ColdSlotFill);

    dataInit.setGpuInit(true);
    problem.setSwizzleTensorA(true);

    auto const swizzledPlan = dataInit.planNormalWarmOutputReset(problem);
    EXPECT_EQ(swizzledPlan.action, OutputResetPlanDataInitialization::Action::FullFill);
    EXPECT_EQ(swizzledPlan.reason, OutputResetPlanDataInitialization::Reason::ColdSlotFill);
}

TEST(DataInitializationOutputResetPlan, RingWarmValidationPlansResetFromValid)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
        GTEST_SKIP() << hipDevice.message();

    auto args = TensileLite::testing::buildRingArgs({{32, 32, 32}}, 1);
    ClientProblemFactory             factory(args);
    OutputResetPlanDataInitialization dataInit(args, factory);
    auto problem = makePlainProblem(32, 32, 32);

    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);
    ASSERT_TRUE(dataInit.ringEligible());
    ASSERT_TRUE(dataInit.altSlotsReady());

    auto const targetSlot = dataInit.nextPrimeSlot();
    ASSERT_TRUE(targetSlot.has_value());

    auto const plan = dataInit.planRingSlotOutputReset(*targetSlot, dataInit.altSlotsReady());
    EXPECT_EQ(plan.action, OutputResetPlanDataInitialization::Action::ResetFromValid);
    EXPECT_EQ(plan.reason, OutputResetPlanDataInitialization::Reason::RingWarmValidation);
    EXPECT_TRUE(plan.requiresPristineGpuCopy);
    EXPECT_FALSE(plan.usesExistingSlotContents);
    EXPECT_TRUE(plan.targetIsRingSlot);
    EXPECT_EQ(plan.targetSlot, *targetSlot);
}

TEST(DataInitializationOutputResetPlan, NoValidationPublicArgsDisableRingWarmPath)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
        GTEST_SKIP() << hipDevice.message();

    auto args = TensileLite::testing::buildRingArgs({{32, 32, 32}}, 0);
    ClientProblemFactory             factory(args);
    OutputResetPlanDataInitialization dataInit(args, factory);
    auto problem = makePlainProblem(32, 32, 32);

    EXPECT_FALSE(dataInit.ringEligible());

    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);

    EXPECT_FALSE(dataInit.ringEligible());
    EXPECT_FALSE(dataInit.altSlotsReady());
}

TEST(DataInitializationOutputResetPlan, WarmRingValidationPathResetsOutputsBeforeReuse)
{
    auto hipDevice = hasHipDeviceAndWaitValueSupport();
    if(!hipDevice)
        GTEST_SKIP() << hipDevice.message();

    auto args = TensileLite::testing::buildRingArgs({{32, 32, 32}}, 1);
    ClientProblemFactory             factory(args);
    OutputResetPlanDataInitialization dataInit(args, factory);
    auto problem = makePlainProblem(32, 32, 32);

    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);

    auto* ci = dynamic_cast<ContractionInputs*>(inputs.get());
    ASSERT_NE(ci, nullptr);

    ASSERT_TRUE(dataInit.ringEligible());
    ASSERT_TRUE(dataInit.altSlotsReady());

    auto const targetSlot = dataInit.nextPrimeSlot();
    ASSERT_TRUE(targetSlot.has_value());

    auto& dUnit = dataInit.dPristineUnit(problem);
    auto const& dDesc = problem.tensors().at(ContractionProblemGemm::TENSOR::D);
    auto const   dBytes = multiplyElementSize(dUnit.maxElements, dDesc.elementBytes());
    ASSERT_GT(dBytes, 0u);

    auto* slotD = static_cast<uint8_t*>(
        dataInit.ringTensorPtr(*targetSlot, ContractionProblemGemm::TENSOR::D));
    auto* validD = static_cast<uint8_t*>(dUnit.gpuInput.valid.get());
    ASSERT_NE(slotD, nullptr);
    ASSERT_NE(validD, nullptr);

    auto const expectedSamples = readSampledBytes(validD, dBytes);
    for(auto const& sample : expectedSamples)
    {
        SCOPED_TRACE(sample.offset);
        EXPECT_NE(sample.value, 0xA5);
    }

    HIP_CHECK_EXC(hipMemset(slotD, 0xA5, dBytes));
    auto const poisonedSamples = readSampledBytes(slotD, dBytes);
    for(auto const& sample : poisonedSamples)
    {
        SCOPED_TRACE(sample.offset);
        EXPECT_EQ(sample.value, 0xA5);
    }

    DeviceSignalBuffer gateValue;
    gateValue.allocate();
    uint32_t zero = 0;
    HIP_CHECK_EXC(hipMemcpy(gateValue.get(), &zero, sizeof(zero), hipMemcpyHostToDevice));

    HipStreamGuard gateStream(hipStreamNonBlocking);
    HipStreamGuard computeStream(hipStreamNonBlocking);
    HipStreamGuard releaseStream(hipStreamNonBlocking);
    EventHandle    gateEvent;

    HIP_CHECK_EXC(hipStreamWaitValue32(
        gateStream.get(), gateValue.get(), 1, hipStreamWaitValueEq));
    HIP_CHECK_EXC(hipEventRecord(gateEvent.get(), gateStream.get()));
    HIP_CHECK_EXC(hipStreamWaitEvent(dataInit.copyStream(), gateEvent.get(), 0));

    dataInit.primeNextInputSlot(&problem);

    auto warmInputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(warmInputs, nullptr);

    auto* warmCi = dynamic_cast<ContractionInputs*>(warmInputs.get());
    ASSERT_NE(warmCi, nullptr);
    EXPECT_EQ(warmCi->d, slotD);

    dataInit.waitForPreparedSlot(computeStream.get());
    EXPECT_EQ(hipEventQuery(gateEvent.get()), hipErrorNotReady);

    HIP_CHECK_EXC(hipStreamWriteValue32(releaseStream.get(), gateValue.get(), 1, 0));
    HIP_CHECK_EXC(hipStreamSynchronize(releaseStream.get()));
    HIP_CHECK_EXC(hipStreamSynchronize(gateStream.get()));
    HIP_CHECK_EXC(hipStreamSynchronize(computeStream.get()));

    auto const finalSamples = readSampledBytes(slotD, dBytes);
    ASSERT_EQ(finalSamples.size(), expectedSamples.size());
    for(size_t i = 0; i < finalSamples.size(); ++i)
    {
        SCOPED_TRACE(finalSamples[i].offset);
        EXPECT_EQ(finalSamples[i].value, expectedSamples[i].value);
    }
}
