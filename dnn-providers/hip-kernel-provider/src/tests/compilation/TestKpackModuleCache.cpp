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

/// A blob that is found and decompressed but is not a code object at all.
///
/// The reader reports success on a TOC entry whose ordinal/size keys are missing, having
/// read them uninitialized, so a well-formed request can return an unrelated entry's
/// bytes. KpackArchive checks the container magic to catch that before HIP sees it, which
/// is why this stops at DECOMPRESS rather than reaching MODULE_LOAD: the payloads in this
/// asset are ASCII stand-ins, not ELF.
TEST(TestKpackModuleCacheLoad, RejectsAPayloadThatIsNotACodeObject)
{
    ASSERT_TRUE(std::filesystem::exists(REAL_ARCHIVE))
        << "the kpack test asset named at configure time is missing: " << REAL_ARCHIVE;

    try
    {
        KpackModuleCache::load(REAL_ARCHIVE, ARCHIVE_TOC_KEY, ARCHIVE_ARCH);
        FAIL() << "expected a payload without code-object magic to be rejected";
    }
    catch(const KpackModuleLoadFailure& failure)
    {
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

/// The archive holds no binary for the running device.
///
/// Raised by KpackModuleCache::load's own pre-check rather than by the reader, which
/// cannot tell "wrong GPU" from "wrong toc_key" -- both are KERNEL_NOT_FOUND. The message
/// must name the arches the archive does carry, since that is what tells an author
/// whether the packer or the descriptor is wrong.
TEST(TestKpackModuleCacheLoad, ReportsAnArchTheArchiveDoesNotHold)
{
    ASSERT_TRUE(std::filesystem::exists(REAL_ARCHIVE))
        << "the kpack test asset named at configure time is missing: " << REAL_ARCHIVE;

    try
    {
        KpackModuleCache::load(REAL_ARCHIVE, ARCHIVE_TOC_KEY, "gfx90a");
        FAIL() << "expected an arch the archive does not hold to be rejected";
    }
    catch(const KpackModuleLoadFailure& failure)
    {
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
