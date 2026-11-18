// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <stdexcept>
#include <type_traits>

#include <gtest/gtest.h>

#include "KernelIngestorTestFixtures.hpp"
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

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

// Code written against `ingestor::NativeRegistry` shares one registry with code using the
// SDK-level template: a symbol registered through either spelling resolves through both.
TEST(TestIngestorNativeRegistry, BothSpellingsNameOneRegistry)
{
    static_assert(std::is_same_v<NativeRegistry<GraphMatchFn>, GraphMatchRegistry>);
    static_assert(
        std::is_same_v<NativeRegistry<ScoreFn>, hipdnn_plugin_sdk::NativeRegistry<ScoreFn>>);

    NativeRegistry<GraphMatchFn>::registerSymbol("registry.spellings", acceptGraph);

    EXPECT_EQ(hipdnn_plugin_sdk::NativeRegistry<GraphMatchFn>::resolve("registry.spellings"),
              acceptGraph);

    GraphMatchRegistry::unregisterSymbol("registry.spellings");
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
