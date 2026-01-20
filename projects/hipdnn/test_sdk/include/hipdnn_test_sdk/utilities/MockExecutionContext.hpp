// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_plugin_sdk/interfaces/IExecutionContext.hpp>

namespace hipdnn_test_sdk::utilities
{

class MockExecutionContext : public hipdnn_plugin_sdk::IExecutionContext
{
public:
    MOCK_METHOD(bool, hasValidPlan, (), (const, override));
    MOCK_METHOD(void, setPlan, (std::unique_ptr<hipdnn_plugin_sdk::IPlan> plan), (override));
    MOCK_METHOD(hipdnn_plugin_sdk::IPlan&, getPlan, (), (const, override));

    ~MockExecutionContext() override = default;
};

} // namespace hipdnn_test_sdk::utilities
