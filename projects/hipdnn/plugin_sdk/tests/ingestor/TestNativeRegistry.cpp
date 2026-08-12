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
 * @brief Unit tests for the symbol-name-to-native-callable registry.
 *
 * Covers registration, resolution, and the fail-closed contract a descriptor naming an
 * unshipped symbol depends on -- a missing registration must surface as an error at the
 * point of use, never as an engine that silently matches nothing.
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

    // Two implementations behind one name leaves one silently unreachable, and which one
    // wins would depend on static-init order.
    EXPECT_THROW(GraphMatcherRegistry::registerSymbol("registry.duplicate", rejectGraph),
                 std::runtime_error);

    GraphMatcherRegistry::unregisterSymbol("registry.duplicate");
}

TEST(TestIngestorNativeRegistry, FailsClosedOnUnknownSymbol)
{
    // A descriptor naming a symbol the provider does not ship must surface as an error,
    // never as an engine that quietly matches nothing.
    EXPECT_THROW(GraphMatcherRegistry::resolve("registry.never_registered"), std::runtime_error);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
