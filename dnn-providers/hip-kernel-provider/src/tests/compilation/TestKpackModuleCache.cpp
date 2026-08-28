// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "compilation/KpackModuleCache.hpp"

namespace hip_kernel_provider::compilation
{
namespace
{

/// rocm-kpack's own test archive, vendored beside this test. Its entries are placeholder
/// payloads rather than HSA code objects, which is what makes it useful here: it is a
/// real container, so the reader parses it, but nothing in it can load.
constexpr const char* REAL_ARCHIVE = HIPDNN_TEST_KPACK_ARCHIVE;
constexpr const char* ARCHIVE_ARCH = "gfx1100";
constexpr const char* ARCHIVE_TOC_KEY = "lib/libhip.so#0";

/// A declared digest, shaped like a real one so these cases exercise what a descriptor
/// actually carries. The load cases below fail earlier and never reach the comparison,
/// as their stage() assertions prove.
constexpr const char* DIGEST = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

TEST(TestKpackModuleCacheKey, MakeKeyFormatsCorrectly)
{
    // Pinned literally, as in TestSdpaModuleCache.cpp: a format change that merged or
    // reordered fields still round-trips through makeKey, so only the literal catches it.
    EXPECT_EQ(KpackModuleCache::makeKey(
                  "/opt/packs/pointwise.kpack", "lib/libhip.so#0", "gfx942", 0, DIGEST),
              std::string("/opt/packs/pointwise.kpack::lib/libhip.so#0::gfx942::0::") + DIGEST);
}

TEST(TestKpackModuleCacheKey, KeyDistinguishesDeclaredDigests)
{
    const std::string archive = "/opt/packs/pointwise.kpack";
    const char* other = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

    // The half load() cannot cover on its own: a cache hit never calls it.
    EXPECT_NE(KpackModuleCache::makeKey(archive, "lib/libhip.so#0", "gfx942", 0, DIGEST),
              KpackModuleCache::makeKey(archive, "lib/libhip.so#0", "gfx942", 0, other));
}

TEST(TestKpackModuleCacheKey, KeyDistinguishesTocKeyAndArch)
{
    const std::string archive = "/opt/packs/pointwise.kpack";

    // Same archive, different entry: different module.
    EXPECT_NE(KpackModuleCache::makeKey(archive, "lib/libhip.so#0", "gfx942", 0, DIGEST),
              KpackModuleCache::makeKey(archive, "bin/hiptest#0", "gfx942", 0, DIGEST));

    // Same archive and entry, different device arch: also a different module, because
    // one archive holds a distinct blob per arch.
    EXPECT_NE(KpackModuleCache::makeKey(archive, "lib/libhip.so#0", "gfx942", 0, DIGEST),
              KpackModuleCache::makeKey(archive, "lib/libhip.so#0", "gfx1100", 0, DIGEST));

    // Not asserted: "::"-joining is not prefix-free, so a tocKey ending in "::" would
    // collide -- unreachable for packer-emitted "<path>#<index>" keys and [a-z0-9]+ arch
    // names, and closing it would change the format the case above pins.
}

TEST(TestKpackModuleCacheKey, KeyDistinguishesTwoOrdinalsOfTheSameArch)
{
    const std::string archive = "/opt/packs/pointwise.kpack";

    // A hipModule_t belongs to the device current when it was loaded, so handing device 1
    // the module device 0 loaded is the defect the ordinal closes.
    EXPECT_NE(KpackModuleCache::makeKey(archive, "lib/libhip.so#0", "gfx942", 0, DIGEST),
              KpackModuleCache::makeKey(archive, "lib/libhip.so#0", "gfx942", 1, DIGEST));
}

TEST(TestKpackModuleCacheKey, KeyIgnoresArchFeatureDecoration)
{
    const std::string archive = "/opt/packs/pointwise.kpack";

    // Feature flags describe the device, not the code object, and archMatches gates on the
    // bare name, so a decorated arch must reach the entry the bare one made.
    EXPECT_EQ(
        KpackModuleCache::makeKey(archive, "lib/libhip.so#0", "gfx90a:sramecc+:xnack-", 0, DIGEST),
        KpackModuleCache::makeKey(archive, "lib/libhip.so#0", "gfx90a", 0, DIGEST));
}

TEST(TestKpackModuleCacheLoad, RejectsAPayloadThatIsNotACodeObject)
{
    ASSERT_TRUE(std::filesystem::exists(REAL_ARCHIVE))
        << "the kpack test asset named at configure time is missing: " << REAL_ARCHIVE;

    try
    {
        KpackModuleCache::load(REAL_ARCHIVE, ARCHIVE_TOC_KEY, ARCHIVE_ARCH, 0, DIGEST);
        FAIL() << "expected a payload without code-object magic to be rejected";
    }
    catch(const KpackModuleLoadFailure& failure)
    {
        // DECOMPRESS rather than MODULE_LOAD: KpackArchive checks the container magic
        // before HIP sees it, and this asset's payloads are ASCII stand-ins, not ELF.
        EXPECT_EQ(failure.stage(), KpackLoadStage::DECOMPRESS)
            << "a payload that is not a code object must be named before HIP sees it: "
            << failure.what();
        EXPECT_NE(std::string(failure.what()).find("KPACK_ERROR_INVALID_METADATA"),
                  std::string::npos)
            << failure.what();
    }

    // Clear the HIP error state left by the intentional load failure, or the
    // HipErrorHandler listener fails this test for it.
    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

TEST(TestKpackModuleCacheLoad, ReportsAnArchTheArchiveDoesNotHold)
{
    ASSERT_TRUE(std::filesystem::exists(REAL_ARCHIVE))
        << "the kpack test asset named at configure time is missing: " << REAL_ARCHIVE;

    try
    {
        KpackModuleCache::load(REAL_ARCHIVE, ARCHIVE_TOC_KEY, "gfx90a", 0, DIGEST);
        FAIL() << "expected an arch the archive does not hold to be rejected";
    }
    catch(const KpackModuleLoadFailure& failure)
    {
        // load's own pre-check, not the reader, which cannot tell "wrong GPU" from "wrong
        // toc_key" -- both are KERNEL_NOT_FOUND.
        EXPECT_EQ(failure.stage(), KpackLoadStage::ARCH_LOOKUP) << failure.what();

        const std::string message = failure.what();
        EXPECT_NE(message.find("gfx90a"), std::string::npos)
            << "the message must name the arch that was asked for: " << message;
        EXPECT_NE(message.find(ARCHIVE_ARCH), std::string::npos)
            << "the message must name the arches the archive provides: " << message;
    }
}

} // namespace
} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
