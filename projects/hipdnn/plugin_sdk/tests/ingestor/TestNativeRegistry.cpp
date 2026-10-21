// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <stdexcept>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestNativeRegistry.cpp
 * @brief Tests for the symbol-name-to-native-callable registry: registration,
 * resolution, and fail-closed behavior on unresolved symbols.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

TEST(TestIngestorNativeRegistry, ResolvesARegisteredSymbol)
{
    GraphMatchRegistry::registerSymbol("registry.resolves", acceptGraph);

    EXPECT_EQ(GraphMatchRegistry::resolve("registry.resolves"), acceptGraph);

    GraphMatchRegistry::unregisterSymbol("registry.resolves");
}

TEST(TestIngestorNativeRegistry, RejectsDuplicateRegistration)
{
    GraphMatchRegistry::registerSymbol("registry.duplicate", acceptGraph);

    EXPECT_THROW(GraphMatchRegistry::registerSymbol("registry.duplicate", rejectGraph),
                 std::runtime_error);

    GraphMatchRegistry::unregisterSymbol("registry.duplicate");
}

TEST(TestIngestorNativeRegistry, FailsClosedOnUnknownSymbol)
{
    EXPECT_THROW(GraphMatchRegistry::resolve("registry.never_registered"), std::runtime_error);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
