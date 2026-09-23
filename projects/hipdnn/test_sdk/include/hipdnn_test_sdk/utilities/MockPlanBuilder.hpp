// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

namespace hipdnn_test_sdk::utilities
{

class MockPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder
{
public:
    MOCK_METHOD(bool,
                isApplicable,
                (hipdnnEnginePluginHandle_t handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph),
                (const, override));
    MOCK_METHOD(size_t,
                getMaxWorkspaceSize,
                (hipdnnEnginePluginHandle_t handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph),
                (const, override));
    MOCK_METHOD(std::unique_ptr<hipdnn_plugin_sdk::IPlan>,
                buildPlan,
                (hipdnnEnginePluginHandle_t handle,
                 const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph),
                (const, override));

    ~MockPlanBuilder() override = default;
};

} // namespace hipdnn_test_sdk::utilities
