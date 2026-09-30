// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Unit coverage for the optional override-execute virtual surface on `EnginePlugin`.

#include "plugin/EnginePlugin.hpp"
#include "plugins/mocks/MockEnginePlugin.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

using namespace hipdnn_backend::plugin;

TEST(TestEnginePluginOverride, ExecuteOpGraphWithOverridesForwardsAllArguments)
{
    // Validates that the virtual is reachable through a base pointer and
    // forwards arguments unchanged.
    // NOLINTNEXTLINE(misc-const-correctness) — gmock EXPECT_CALL requires non-const mock.
    MockEnginePlugin mock;

    // Sentinel handle/execution-context values: opaque to the mock, only used
    // for identity comparison in EXPECT_CALL. Concrete addresses suffice.
    int handleSentinel = 0;
    int execSentinel = 0;
    auto* const handle = reinterpret_cast<hipdnnEnginePluginHandle_t>(&handleSentinel);
    auto* const exec = reinterpret_cast<hipdnnEnginePluginExecutionContext_t>(&execSentinel);

    int dummyWorkspace = 0;
    void* const workspace = &dummyWorkspace;

    const std::vector<int64_t> uniqueIds{42, 99};
    const std::vector<uint32_t> lengths{3u, 4u};
    const std::vector<int64_t> shape0{2, 3, 4};
    const std::vector<int64_t> shape1{1, 2, 3, 4};
    const std::vector<int64_t> stride0{12, 4, 1};
    const std::vector<int64_t> stride1{24, 12, 4, 1};
    const std::array<const int64_t*, 2> shapesPerUid{shape0.data(), shape1.data()};
    const std::array<const int64_t*, 2> stridesPerUid{stride0.data(), stride1.data()};

    EXPECT_CALL(mock,
                executeOpGraphWithOverrides(handle,
                                            exec,
                                            workspace,
                                            nullptr,
                                            0u,
                                            2u,
                                            uniqueIds.data(),
                                            lengths.data(),
                                            shapesPerUid.data(),
                                            stridesPerUid.data()));

    const EnginePlugin& base = mock;
    base.executeOpGraphWithOverrides(handle,
                                     exec,
                                     workspace,
                                     nullptr,
                                     0u,
                                     2u,
                                     uniqueIds.data(),
                                     lengths.data(),
                                     shapesPerUid.data(),
                                     stridesPerUid.data());
}
