// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

#include "harness/bundle/GraphSession.hpp"
#include "harness/bundle/LoadedEngine.hpp"
#include "harness/bundle/SupportClaims.hpp"
#include "harness/bundle/SupportVerdict.hpp"

namespace hipdnn_integration_tests::bundle
{

/// Where the harness gets one graph's claim verdicts.
///
/// A seam even though observeSupport() is already pure, because two states the
/// harness must handle are ones the real observation cannot produce by
/// construction: a verdict naming an engine other than the one under test (it
/// returns at most one, for the engine under test), and a sidecar that exists but
/// reads back as SidecarState::NONE (the harness-bug case the coverage check exists
/// to catch). Without this, those tests have nothing to drive.
class ISupportClaimObserver
{
public:
    ISupportClaimObserver() = default;
    virtual ~ISupportClaimObserver() = default;

    ISupportClaimObserver(const ISupportClaimObserver&) = delete;
    ISupportClaimObserver& operator=(const ISupportClaimObserver&) = delete;
    ISupportClaimObserver(ISupportClaimObserver&&) = delete;
    ISupportClaimObserver& operator=(ISupportClaimObserver&&) = delete;

    /// Throws std::runtime_error if the sidecar exists but cannot be read.
    virtual SupportObservation observe(const RankedEngines& engines,
                                       const SupportClaimLocator& locator,
                                       const LoadedEngine& engineUnderTest,
                                       std::string_view arch,
                                       std::string_view platform)
        = 0;
};

/// The real decision: observeSupport().
class DefaultSupportClaimObserver : public ISupportClaimObserver
{
public:
    SupportObservation observe(const RankedEngines& engines,
                               const SupportClaimLocator& locator,
                               const LoadedEngine& engineUnderTest,
                               std::string_view arch,
                               std::string_view platform) override
    {
        return observeSupport(engines, locator, engineUnderTest, arch, platform);
    }
};

} // namespace hipdnn_integration_tests::bundle
