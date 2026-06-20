// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"
#include "DataInitializationTestHooks.hpp"
#include "DataInitializationTestUtils.hpp"

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    class AltAllocationDataInitialization : public DataInitialization
    {
    public:
        using DataInitialization::DataInitialization;
        using DataInitialization::ringEligible;

        bool hasAltBuffers() const
        {
            return m_hasAltBuffers;
        }

        bool ringPolicyAllocatesAltBuffers() const
        {
            return m_ringPolicy.allocatesAltBuffers();
        }

        bool altSlotsReady() const
        {
            return m_altSlotsReady;
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
                   || !slotState.maxElements.empty() || !slotState.groupedOffsets.empty()
                   || slotState.cachedInputs)
                    return false;
            }

            return true;
        }

        bool slotZeroInputsPresent() const
        {
            for(auto const& vd : m_vdata)
            {
                for(auto const& [_, pUnit] : vd.pristine)
                {
                    if(!pUnit.gpuInput.current || !pUnit.gpuInput.batch
                       || !pUnit.gpuInput.buffers[0] || !pUnit.gpuInput.batchBufs[0])
                    {
                        return false;
                    }
                }
            }
            return true;
        }
    };

    class OptionalAltAllocationFailureGuard
    {
    public:
        explicit OptionalAltAllocationFailureGuard(size_t callsBeforeFailure)
        {
            TensileLite::testing::detail::setOptionalAltAllocationFailureCountdown(
                callsBeforeFailure);
        }

        ~OptionalAltAllocationFailureGuard()
        {
            TensileLite::testing::detail::clearOptionalAltAllocationFailure();
        }

        OptionalAltAllocationFailureGuard(OptionalAltAllocationFailureGuard const&) = delete;
        OptionalAltAllocationFailureGuard&
            operator=(OptionalAltAllocationFailureGuard const&) = delete;
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

    Client::po::variables_map makeAltAllocationArgs()
    {
        return TensileLite::testing::buildRingArgs({{32, 32, 32}}, 1);
    }

    void expectAltAllocationRollback(size_t callsBeforeFailure)
    {
        auto hipDevice = hasHipDevice();
        if(!hipDevice)
            GTEST_SKIP() << hipDevice.message();

        OptionalAltAllocationFailureGuard failOptionalAltAllocation(callsBeforeFailure);

        auto args = makeAltAllocationArgs();
        ClientProblemFactory factory(args);
        AltAllocationDataInitialization dataInit(args, factory);

        auto problem = TensileLite::testing::makePlainProblem(32, 32, 32);
        auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));

        ASSERT_NE(inputs, nullptr);
        EXPECT_TRUE(dataInit.ringPolicyAllocatesAltBuffers());
        EXPECT_FALSE(dataInit.hasAltBuffers());
        EXPECT_FALSE(dataInit.altSlotsReady());
        EXPECT_FALSE(dataInit.ringEligible());
        EXPECT_TRUE(dataInit.slotZeroInputsPresent());
        EXPECT_TRUE(dataInit.allAltGpuInputsCleared());
    }
} // namespace

TEST(DataInitializationAltAllocation, PartialAltAllocationFailureRollsBackAssignedSlots)
{
    // 2 successful calls (slot 1 data, slot 1 batch) then failure on slot 2 data.
    expectAltAllocationRollback(/*callsBeforeFailure=*/2);
}

TEST(DataInitializationAltAllocation, BatchAllocationFailureDoesNotStoreDanglingAltBuffer)
{
    // 1 successful call (slot 1 data) then failure on slot 1 batch.
    expectAltAllocationRollback(/*callsBeforeFailure=*/1);
}
