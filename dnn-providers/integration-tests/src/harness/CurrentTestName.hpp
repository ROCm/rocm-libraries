// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include <gtest/gtest.h>

namespace hipdnn_integration_tests
{

/// "Suite.Name" of the running gtest test, or an empty string outside a test.
inline std::string currentTestName()
{
    auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    if(info == nullptr)
    {
        return {};
    }
    return std::string(info->test_suite_name()) + "." + info->name();
}

} // namespace hipdnn_integration_tests
