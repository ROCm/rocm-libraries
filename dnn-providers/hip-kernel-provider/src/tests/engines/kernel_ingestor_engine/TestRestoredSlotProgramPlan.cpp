// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/PackedPlanTestSupport.hpp"
#include "engines/kernel_ingestor_engine/RestoredSlotProgramPlan.hpp"

/**
 * @file TestRestoredSlotProgramPlan.cpp
 * @brief The plan every kind 2 payload restores as: stages run in order over the recorded
 *        workspace, and the plan is never captured.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using hipdnn_plugin_sdk::HipdnnPluginException;

namespace packs = hip_kernel_provider::kernel_ingestor_engine::testing;

using Plan = RestoredSlotProgramPlan<Handle>;

// The slot names of the stages, in the order the payload records them.
const std::vector<std::string> SLOTS{"prep", "dkdv", "dq"};

// One handler launches every stage. Each stage's prepared dispatch carries its slot name, which
// the handler records at launch beside the workspace pointer it received.
class TestRestoredSlotProgramPlan : public ::testing::Test
{
protected:
    TestRestoredSlotProgramPlan()
        : _handler(packs::MultiKernelStubHandler::Config{}, "")
    {
    }

    std::vector<Plan::Stage> stages() const
    {
        std::vector<Plan::Stage> built;
        built.reserve(SLOTS.size());
        for(const auto& slot : SLOTS)
        {
            built.push_back({slot, &_handler, std::make_unique<packs::StubForeignPrepared>(slot)});
        }
        return built;
    }

    static void execute(const Plan& plan, void* workspace)
    {
        const Handle handle;
        plan.execute(handle, nullptr, 0, workspace);
    }

    packs::MultiKernelStubHandler _handler;
};

TEST_F(TestRestoredSlotProgramPlan, ExecutesStagesInSlotOrderOnTheBaseWorkspace)
{
    const Plan plan(stages(), 1024, 4096);
    std::vector<char> workspace(4096);

    execute(plan, workspace.data());

    const auto& launched = _handler.launched();
    ASSERT_EQ(launched.size(), SLOTS.size());
    for(size_t index = 0; index < SLOTS.size(); ++index)
    {
        EXPECT_EQ(launched[index].tag, SLOTS[index]);
        // Every stage gets the base: a stage's handler adds its own offsets to it.
        EXPECT_EQ(launched[index].workspace, workspace.data());
    }
}

TEST_F(TestRestoredSlotProgramPlan, ReportsTheRecordedWorkspaceSize)
{
    const Plan plan(stages(), 1024, 4096);

    EXPECT_EQ(plan.getWorkspaceSize(Handle{}), 4096U);
}

TEST_F(TestRestoredSlotProgramPlan, ThrowsWhenAWorkspaceIsNeededAndNull)
{
    const Plan plan(stages(), 1024, 4096);

    try
    {
        execute(plan, nullptr);
        FAIL() << "expected a missing workspace to be refused";
    }
    catch(const HipdnnPluginException& refusal)
    {
        EXPECT_EQ(refusal.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE) << refusal.getMessage();
        EXPECT_NE(refusal.getMessage().find("4096"), std::string::npos) << refusal.getMessage();
    }
    EXPECT_TRUE(_handler.launched().empty());
}

TEST_F(TestRestoredSlotProgramPlan, ExecutesAStageWithNoWorkspaceWithoutOne)
{
    const Plan plan(stages(), 0, 0);

    execute(plan, nullptr);

    ASSERT_EQ(_handler.launched().size(), SLOTS.size());
    for(const auto& launch : _handler.launched())
    {
        EXPECT_EQ(launch.workspace, nullptr);
    }
}

TEST_F(TestRestoredSlotProgramPlan, IsNotSaveable)
{
    const Plan plan(stages(), 1024, 4096);

    EXPECT_EQ(plan.saveablePlan(), nullptr);
}

TEST_F(TestRestoredSlotProgramPlan, RefusesAPlanWithNoStagesNoHandlerOrNoPreparedLaunch)
{
    EXPECT_THROW(Plan({}, 0, 0), HipdnnPluginException);

    std::vector<Plan::Stage> unprepared;
    unprepared.push_back({"prep", &_handler, nullptr});
    EXPECT_THROW(Plan(std::move(unprepared), 0, 0), HipdnnPluginException);

    // A stage whose prepared dispatch is set but whose handler is null.
    std::vector<Plan::Stage> unhandled;
    unhandled.push_back({"prep", nullptr, std::make_unique<packs::StubForeignPrepared>("prep")});
    EXPECT_THROW(Plan(std::move(unhandled), 0, 0), HipdnnPluginException);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
