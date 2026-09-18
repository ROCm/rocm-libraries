// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

/**
 * @file TestIngestorPacks.cpp
 * @brief Enforces the one invariant resetIngestorModuleCachesForTesting() depends on:
 *        every pack's `ownsModuleCache` flag must agree with whether its
 *        `resetModuleCache` pointer is set. Without this, a pack that gains a kpack
 *        cache but is wired with a null reset pointer (or vice versa) is left out of the
 *        reset sweep silently -- the resident-module bug this file exists to catch by
 *        construction rather than by chance.
 *
 * Purely host-side: ingestorPacks() returns a static table of function pointers built at
 * first call, and this test only inspects that table's shape. It calls none of the
 * pointers, opens no device, and touches no descriptor tree, so it is NOT gated behind
 * SKIP_IF_NO_DEVICES() and runs unconditionally, including in a session with no GPU.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;

TEST(TestIngestorPacksModuleCacheOwnership, OwnsModuleCacheAgreesWithResetPointerForEveryPack)
{
    for(const auto& pack : ingestorPacks())
    {
        const bool hasResetPointer = pack.resetModuleCache != nullptr;
        EXPECT_EQ(pack.ownsModuleCache, hasResetPointer)
            << "pack '" << pack.label
            << "' declares ownsModuleCache = " << (pack.ownsModuleCache ? "true" : "false")
            << " but its resetModuleCache " << "pointer is "
            << (hasResetPointer ? "non-null" : "null") << ". Fix "
            << "ingestorPacks() in IngestorPacks.cpp: a pack that owns a kpack module "
            << "cache must set ownsModuleCache = true AND wire a non-null "
            << "resetModuleCache, or set ownsModuleCache = false with a null pointer if "
            << "it owns no cache.";
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
