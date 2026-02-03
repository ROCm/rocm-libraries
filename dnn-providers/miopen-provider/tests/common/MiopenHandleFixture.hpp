// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <miopen/miopen.h>

namespace test_common
{

/// @brief Base test fixture that creates and destroys a raw MIOpen handle.
///
/// Use this fixture for tests that require a MIOpen handle but don't need
/// the full HipdnnEnginePluginHandle wrapper. Tests inheriting from this
/// fixture will be skipped if no GPU is available.
///
/// Usage:
/// @code
///     class MyGpuTest : public MiopenHandleFixture { ... };
///     TEST_F(MyGpuTest, SomeTest) {
///         miopenSomeApi(_miopenHandle, ...);
///     }
/// @endcode
class MiopenHandleFixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(miopenCreate(&_miopenHandle), miopenStatusSuccess);
    }

    void TearDown() override
    {
        if(_miopenHandle != nullptr)
        {
            EXPECT_EQ(miopenDestroy(_miopenHandle), miopenStatusSuccess);
        }
    }

    miopenHandle_t _miopenHandle = nullptr;
};

} // namespace test_common
