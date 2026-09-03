// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

#include <gmock/gmock.h>

#include "harness/bundle/ISupportClaimObserver.hpp"

namespace hipdnn_integration_tests::bundle
{

/// Lets a test state the claim verdicts a graph produced directly.
///
/// Needed because two states the harness must handle cannot be produced by the real
/// observeSupport(): a verdict naming an engine other than the one under test, and
/// a sidecar that exists but reads back as SidecarState::NONE.
class MockSupportClaimObserver : public ISupportClaimObserver
{
public:
    MOCK_METHOD(SupportObservation,
                observe,
                (const RankedEngines& engines,
                 const SupportClaimLocator& locator,
                 const LoadedEngine& engineUnderTest,
                 std::string_view arch,
                 std::string_view platform),
                (override));
};

} // namespace hipdnn_integration_tests::bundle
