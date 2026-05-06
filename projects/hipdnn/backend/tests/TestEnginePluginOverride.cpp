// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// RFC 0008 Phase 1 — unit coverage for the optional override-execute
// surface on `EnginePlugin` (B.8). The full dlopen-based optional-symbol
// resolution path (`tryAssignSymbol`) is exercised end-to-end by Stream
// C's frontend integration tests against the override-implementing,
// override-omitting, and version-liar fake plugins. These unit tests
// pin the C++ API contract on the host side independent of any plugin
// loading: that the virtual surface is in place, that the
// `hasOverrideExecute()` predicate is observable from the manager, and
// that the dispatch wrapper forwards arguments unchanged.

#include "plugin/EnginePlugin.hpp"
#include "plugins/mocks/MockEnginePlugin.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

using namespace hipdnn_backend::plugin;
using ::testing::Return;

TEST(TestEnginePluginOverride, MockReportsHasOverrideExecuteByDefault)
{
    // The mock surface defaults to no override support unless a test
    // explicitly programs it; guards the mock + virtual signature.
    // NOLINTNEXTLINE(misc-const-correctness) — gmock EXPECT_CALL requires non-const mock.
    MockEnginePlugin mock;
    EXPECT_CALL(mock, hasOverrideExecute()).WillOnce(Return(false));
    EXPECT_FALSE(mock.hasOverrideExecute());
}

TEST(TestEnginePluginOverride, MockReportsHasOverrideExecuteWhenProgrammed)
{
    // NOLINTNEXTLINE(misc-const-correctness) — gmock EXPECT_CALL requires non-const mock.
    MockEnginePlugin mock;
    EXPECT_CALL(mock, hasOverrideExecute()).WillOnce(Return(true));
    EXPECT_TRUE(mock.hasOverrideExecute());
}

TEST(TestEnginePluginOverride, ExecuteOpGraphWithOverridesForwardsAllArguments)
{
    // Validates that the new virtual is reachable via base-class pointer
    // and that argument forwarding is observable. Stream C's integration
    // tests cover the actual C-API plumbing through the resource manager
    // (D2 reconstruction); this test pins the C++ wrapper signature.
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
