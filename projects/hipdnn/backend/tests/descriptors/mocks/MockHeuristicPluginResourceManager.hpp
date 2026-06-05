// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin/HeuristicPluginResourceManager.hpp"
#include <gmock/gmock.h>

namespace hipdnn_backend::plugin
{

class MockHeuristicPluginResourceManager : public HeuristicPluginResourceManager
{
public:
    MockHeuristicPluginResourceManager() = default;

    MOCK_METHOD(hipdnnHeuristicHandle_t,
                getHeuristicHandleForPolicyId,
                (int64_t policyId),
                (const, override));
    MOCK_METHOD(const HeuristicPlugin*,
                getPluginForPolicyId,
                (int64_t policyId),
                (const, override));
    MOCK_METHOD(void,
                setDevicePropertiesOnAllHandles,
                (const hipdnnPluginConstData_t* devicePropsSerialized),
                (const, override));
    MOCK_METHOD(std::vector<HeuristicPolicyInfo>, getHeuristicPolicyInfos, (), (const, override));
    MOCK_METHOD(std::string, toString, (), (const, override));
};

} // namespace hipdnn_backend::plugin
