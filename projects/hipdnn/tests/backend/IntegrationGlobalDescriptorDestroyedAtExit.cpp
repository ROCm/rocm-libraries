// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Backend descriptors owned by a namespace-scope object and destroyed through the C API during
// process exit. See frontend/GlobalObjectDestroyedAtExitFixture.hpp for how the child process is
// checked.

#include <array>
#include <cstdlib>

#include <gtest/gtest.h>

#include "TestUtil.hpp"
#include "frontend/GlobalObjectDestroyedAtExitFixture.hpp"
#include "hipdnn_backend.h"
#include <test_plugins/TestPluginConstants.hpp>

namespace
{

#if !defined(_WIN32)
// Destroys backend descriptors from its destructor, as an application object would.
class LateBackendDescriptors
{
public:
    LateBackendDescriptors() = default;
    LateBackendDescriptors(const LateBackendDescriptors&) = delete;
    LateBackendDescriptors& operator=(const LateBackendDescriptors&) = delete;
    LateBackendDescriptors(LateBackendDescriptors&&) = delete;
    LateBackendDescriptors& operator=(LateBackendDescriptors&&) = delete;

    ~LateBackendDescriptors()
    {
        for(const auto& descriptor : _descriptors)
        {
            if(descriptor != nullptr
               && hipdnnBackendDestroyDescriptor(descriptor) != HIPDNN_STATUS_SUCCESS)
            {
                std::_Exit(hipdnn_tests::DESTROY_FAILED_AT_EXIT);
            }
        }
    }

    // Destroyed in the order given.
    void adopt(const std::array<hipdnnBackendDescriptor_t, 4>& descriptors)
    {
        _descriptors = descriptors;
    }

private:
    std::array<hipdnnBackendDescriptor_t, 4> _descriptors{};
};

hipdnn_tests::PluginUnloadProbe gPluginUnloadProbe;
LateBackendDescriptors gLateDescriptors;
#endif

class IntegrationGlobalBackendDescriptorDestroyedAtExit
    : public hipdnn_tests::IntegrationGlobalObjectDestroyedAtExit
{
};

TEST_F(IntegrationGlobalBackendDescriptorDestroyedAtExit, ExecutionPlanExitsCleanly)
{
#if defined(_WIN32)
    GTEST_SKIP() << "The backend's exit handlers run when its DLL detaches, after the "
                    "executable's static destructors, so a late hipdnnBackendDestroyDescriptor "
                    "there runs while every plugin is still loaded and releases the plugin as "
                    "before exit";
#else
    if(!isChild())
    {
        expectChildExitsCleanly();
        return;
    }

    ASSERT_NO_FATAL_FAILURE(prepareChild(gPluginUnloadProbe));
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    hipdnnBackendDescriptor_t plan = nullptr;
    hipdnnBackendDescriptor_t engineConfig = nullptr;
    hipdnnBackendDescriptor_t engine = nullptr;
    hipdnnBackendDescriptor_t graph = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_NO_FATAL_FAILURE(
        test_util::populateTestExecutionPlan(&plan,
                                             &engineConfig,
                                             &engine,
                                             &graph,
                                             handle,
                                             hipdnn_tests::plugin_constants::engineId<GoodPlugin>(),
                                             true));
    gLateDescriptors.adopt({plan, engineConfig, engine, graph});

    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
    // The finalized plan, not the handle, now holds the engine plugin.
    ASSERT_TRUE(isTestGoodPluginLoaded());
#endif
}

} // namespace
