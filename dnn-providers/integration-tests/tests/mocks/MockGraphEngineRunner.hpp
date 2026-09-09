// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>

#include <gmock/gmock.h>

#include "harness/bundle/IGraphEngineRunner.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"

namespace hipdnn_integration_tests::bundle
{

/// Stands in for the frontend graph pipeline, so a test can drive the whole
/// verification body with no handle, no device and no engine plugin loaded.
class MockGraphEngineRunner : public IGraphEngineRunner
{
public:
    MOCK_METHOD(GraphSession,
                openGraph,
                (const IntegrationTestBundle& bundle,
                 const std::optional<LoadedEngine>& engineUnderTest),
                (override));

    MOCK_METHOD(EngineOpResult,
                buildPlans,
                (GraphSession & session, const std::optional<LoadedEngine>& engineUnderTest),
                (override));

    MOCK_METHOD(EngineOpResult,
                execute,
                (GraphSession & session,
                 const std::optional<LoadedEngine>& engineUnderTest,
                 VariantPack& variantPack),
                (override));
};

} // namespace hipdnn_integration_tests::bundle
