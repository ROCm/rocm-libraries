// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "harness/bundle/IGraphEngineRunner.hpp"

namespace hipdnn_integration_tests::bundle
{

/// The real thing: drives hipdnn_frontend::graph::Graph on the shared handle.
///
/// This is the only place in the harness that needs a handle, a device, or a
/// loaded engine plugin. Kept in its own translation unit so the unit-test binary
/// can link the harness without any of them.
class FrontendGraphEngineRunner : public IGraphEngineRunner
{
public:
    GraphSession openGraph(const IntegrationTestBundle& bundle,
                           const std::optional<LoadedEngine>& engineUnderTest) override;

    EngineOpResult buildPlans(GraphSession& session,
                              const std::optional<LoadedEngine>& engineUnderTest) override;

    EngineOpResult execute(GraphSession& session,
                           const std::optional<LoadedEngine>& engineUnderTest,
                           VariantPack& variantPack) override;
};

} // namespace hipdnn_integration_tests::bundle
