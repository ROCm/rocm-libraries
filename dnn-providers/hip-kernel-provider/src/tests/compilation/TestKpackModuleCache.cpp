// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>

#include <gtest/gtest.h>

#include "compilation/KpackModuleCache.hpp"

namespace hip_kernel_provider::compilation
{
namespace
{

/// The key is pinned literally, in the style of TestSdpaModuleCache.cpp. A format change
/// that merged or reordered fields would still round-trip through makeKey and pass any
/// structural check; only the literal catches it. Note what is *not* in the key: the
/// kernel symbol, so kernels differing only by entry point share one hipModule_t.
TEST(TestKpackModuleCacheKey, MakeKeyFormatsCorrectly)
{
    EXPECT_EQ(KpackModuleCache::makeKey("/opt/packs/pointwise.kpack", "lib/libhip.so#0", "gfx942"),
              "/opt/packs/pointwise.kpack::lib/libhip.so#0::gfx942");
}

TEST(TestKpackModuleCacheKey, KeyDistinguishesTocKeyAndArch)
{
    const std::string archive = "/opt/packs/pointwise.kpack";

    // Same archive, different entry: different module.
    EXPECT_NE(KpackModuleCache::makeKey(archive, "lib/libhip.so#0", "gfx942"),
              KpackModuleCache::makeKey(archive, "bin/hiptest#0", "gfx942"));

    // Same archive and entry, different device arch: also a different module, because
    // one archive holds a distinct blob per arch.
    EXPECT_NE(KpackModuleCache::makeKey(archive, "lib/libhip.so#0", "gfx942"),
              KpackModuleCache::makeKey(archive, "lib/libhip.so#0", "gfx1100"));

    // Not asserted: "::"-joining is not prefix-free, so a tocKey ending in "::" would
    // collide -- unreachable for packer-emitted "<path>#<index>" keys and [a-z0-9]+ arch
    // names, and closing it would change the format the case above pins.
}

} // namespace
} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
