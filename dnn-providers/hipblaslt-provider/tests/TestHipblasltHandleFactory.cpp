// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipblasltHandleFactory.hpp"
#include "HipdnnEnginePluginHandle.hpp"

using namespace hipblaslt_plugin;
using namespace hipdnn_plugin_sdk;

TEST(TestHipblasltHandleFactory, ThrowsOnNullHandle)
{
    EXPECT_THROW(HipblasltHandleFactory::createHipblasltHandle(nullptr), HipdnnPluginException);
}

TEST(TestHipblasltHandleFactory, ThrowsOnDestroyNullHandle)
{
    hipdnnEnginePluginHandle_t handle = nullptr;
    EXPECT_THROW(HipblasltHandleFactory::destroyHipblasltHandle(handle), HipdnnPluginException);
}

TEST(TestGpuHipblasltHandleFactory, CreatesAndDestroysHandle)
{
    SKIP_IF_NO_DEVICES();

    hipdnnEnginePluginHandle_t handle = nullptr;
    EXPECT_NO_THROW(HipblasltHandleFactory::createHipblasltHandle(&handle));
    ASSERT_NE(handle, nullptr);
    ASSERT_NE(handle->hipblasltHandle, nullptr);

    EXPECT_NO_THROW(HipblasltHandleFactory::destroyHipblasltHandle(handle));

    delete handle;
}
