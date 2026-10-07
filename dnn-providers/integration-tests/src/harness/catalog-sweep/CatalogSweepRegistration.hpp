// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <memory>

#include "harness/catalog-sweep/CatalogSweepRun.hpp"
#include "harness/input-init/FillInputs.hpp"

namespace hipdnn_integration_tests::catalog_sweep
{

/// What registration did with the bundles it discovered, for the zero-graphs diagnostic.
struct SweepRegistrationStats
{
    std::size_t discovered = 0;
    std::size_t excludedByGTestFilter = 0;
    std::size_t excludedByShard = 0;
    std::size_t registered = 0;
};

/// The suite-name prefix of every sweep test. main() narrows --gtest_filter to this
/// prefix, which is what keeps the C++ graph tests compiled into the binary from running
/// in a sweep.
inline constexpr const char* K_SWEEP_SUITE_PREFIX = "CatalogSweep_";

/// Registers one gtest test per bundle graph this process sweeps, through the same
/// bundle discovery and loading as the verification suite -- so the graphs, their
/// tolerances and their metadata guards come for free.
///
/// The user's --gtest_filter is applied here, to the bundle's own "Suite.Test" name, and
/// then replaced by the prefix filter above; --catalog-shard keeps the graphs this shard
/// owns. Both exclusions are counted, never silent.
SweepRegistrationStats
    registerCatalogSweepTests(const std::shared_ptr<CatalogSweepRun>& run,
                              const std::shared_ptr<DeviceInputFiller>& deviceFiller);

} // namespace hipdnn_integration_tests::catalog_sweep
