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

/// The key is pinned literally, in the style of
/// tests/engines/asm_sdpa_engine/TestSdpaModuleCache.cpp:30. A format change that
/// merged two fields, or reordered them, would still round-trip through makeKey and
/// pass any structural check; only the literal catches it.
///
/// Note what is *not* here: the kernel symbol. That is deliberate and is the mechanism
/// behind AC #6 -- kernels differing only by entry point share one hipModule_t. It is
/// recorded as a comment rather than an assertion because "makeKey does not take a
/// symbol" is enforced by the signature at compile time; there is no runtime call that
/// could fail.
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

    // Not asserted: injection-freedom. "::"-joining is not a prefix-free encoding, so a
    // tocKey ending in "::" would collide with a longer one -- the same property
    // SdpaModuleCache's key has. It is unreachable for real inputs (a toc key is a
    // packer-emitted "<path>#<index>" and an arch name is [a-z0-9]+), and closing it
    // would mean changing the format MakeKeyFormatsCorrectly pins on purpose.
}

} // namespace
} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
