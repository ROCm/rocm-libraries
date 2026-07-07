// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin/HeuristicPlugin.hpp"
#include <gmock/gmock.h>

namespace hipdnn_backend::plugin
{

/**
 * @brief Mock heuristic plugin for testing.
 *
 * This mock can be configured to simulate different heuristic behaviors:
 * - Always succeed (return sorted IDs)
 * - Always decline (return false from finalize)
 * - Return specific engine orderings
 */
class MockHeuristicPlugin : public HeuristicPlugin
{
public:
    MockHeuristicPlugin() = default;

    // Module metadata
    MOCK_METHOD(std::vector<int64_t>, getAllPolicyIds, (), (const, override));
    MOCK_METHOD(std::string_view, getPolicyName, (int64_t policyId), (const, override));
    MOCK_METHOD(std::string_view, name, (), (const, override));
    MOCK_METHOD(hipdnnPluginType_t, type, (), (const, override));

    // Handle lifecycle
    MOCK_METHOD(void,
                setDeviceProperties,
                (hipdnnHeuristicHandle_t handle,
                 const hipdnnPluginConstData_t* devicePropsSerialized),
                (const, override));

    // Policy descriptor lifecycle
    MOCK_METHOD(hipdnnHeuristicPolicyDescriptor_t,
                createPolicyDescriptor,
                (hipdnnHeuristicHandle_t pluginHandle, int64_t policyId),
                (const, override));
    MOCK_METHOD(void,
                destroyPolicyDescriptor,
                (hipdnnHeuristicPolicyDescriptor_t desc),
                (const, override));

    // Policy inputs
    MOCK_METHOD(void,
                setEngineIds,
                (hipdnnHeuristicPolicyDescriptor_t desc,
                 const int64_t* engineIds,
                 size_t engineIdCount),
                (const, override));
    MOCK_METHOD(void,
                setSerializedGraph,
                (hipdnnHeuristicPolicyDescriptor_t desc,
                 const hipdnnPluginConstData_t* serializedGraph),
                (const, override));

    // Selection execution
    MOCK_METHOD(bool, finalize, (hipdnnHeuristicPolicyDescriptor_t desc), (const, override));
    MOCK_METHOD(std::vector<int64_t>,
                getSortedEngineIds,
                (hipdnnHeuristicPolicyDescriptor_t desc),
                (const, override));
};

} // namespace hipdnn_backend::plugin
