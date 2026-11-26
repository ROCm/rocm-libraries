// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// A handle destroyed during process exit by an object created between two loads of the engine
// plugin. See frontend/GlobalObjectDestroyedAtExitFixture.hpp for how the child process is
// checked.
//
// Under HIPDNN_PLUGIN_UNLOAD_EAGER the first handle loads the plugin and its destruction unloads
// it; the second handle loads it again. Exit handlers run in reverse registration order, so the
// object's destructor runs after the reloaded plugin's static destructors and before any exit
// handler registered at the first load. Only an exit handler registered after the reload tells
// that destructor that exit has begun.

#include <gtest/gtest.h>

#include "frontend/GlobalObjectDestroyedAtExitFixture.hpp"
#include "hipdnn_backend.h"

namespace
{

#if !defined(_WIN32)
hipdnn_tests::PluginUnloadProbe gPluginUnloadProbe;

// Constructed on first use, between the two plugin loads. A namespace-scope object would be
// destroyed after every exit handler registered in main.
hipdnn_tests::LateBackendHandle& handleCreatedBetweenLoads()
{
    static hipdnn_tests::LateBackendHandle s_handle;
    return s_handle;
}
#endif

class IntegrationPluginReloadHandleDestroyedAtExit
    : public hipdnn_tests::IntegrationGlobalObjectDestroyedAtExit
{
};

TEST_F(IntegrationPluginReloadHandleDestroyedAtExit, HandleExitsCleanly)
{
#if defined(_WIN32)
    GTEST_SKIP() << "The backend's exit handlers run when its DLL detaches, after the "
                    "executable's static destructors, so a late hipdnnDestroy there runs while "
                    "every plugin is still loaded and releases the plugin as before exit";
#else
    if(!isChild())
    {
        expectChildExitsCleanly();
        return;
    }

    ASSERT_NO_FATAL_FAILURE(prepareChild(gPluginUnloadProbe));
    hipdnnHandle_t firstHandle = nullptr;
    ASSERT_EQ(hipdnnCreate(&firstHandle), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnDestroy(firstHandle), HIPDNN_STATUS_SUCCESS);
    ASSERT_FALSE(isTestGoodPluginLoaded());

    auto& lateHandle = handleCreatedBetweenLoads();
    hipdnnHandle_t secondHandle = nullptr;
    ASSERT_EQ(hipdnnCreate(&secondHandle), HIPDNN_STATUS_SUCCESS);
    lateHandle.adopt(secondHandle);
    ASSERT_TRUE(isTestGoodPluginLoaded());
#endif
}

} // namespace
