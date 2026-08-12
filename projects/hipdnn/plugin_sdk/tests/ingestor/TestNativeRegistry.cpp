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
    GraphMatcherRegistry::registerSymbol("registry.resolves", acceptGraph);

    EXPECT_EQ(GraphMatcherRegistry::resolve("registry.resolves"), acceptGraph);

    GraphMatcherRegistry::unregisterSymbol("registry.resolves");
}

TEST(TestIngestorNativeRegistry, RejectsDuplicateRegistration)
{
    GraphMatcherRegistry::registerSymbol("registry.duplicate", acceptGraph);

    // Two implementations behind one name: which wins depends on static-init order.
    EXPECT_THROW(GraphMatcherRegistry::registerSymbol("registry.duplicate", rejectGraph),
                 std::runtime_error);

    GraphMatcherRegistry::unregisterSymbol("registry.duplicate");
}

TEST(TestIngestorNativeRegistry, FailsClosedOnUnknownSymbol)
{
    // Unshipped symbol must error, not silently match nothing.
    EXPECT_THROW(GraphMatcherRegistry::resolve("registry.never_registered"), std::runtime_error);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
