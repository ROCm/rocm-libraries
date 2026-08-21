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

/// rocm-kpack's own test archive, path supplied by CMake from ROCM_KPACK_SOURCE_DIR. Its
/// entries are placeholder payloads rather than HSA code objects, so an entry that comes
/// out of it intact is exactly what HIP declines to load.
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

/// The last of the staged failures, and the only one past the reader: the blob is found,
/// decompressed and handed to HIP intact, and HIP rejects it. Asserted on stage() rather
/// than on message text, because the stage is what tells a caller the archive was fine and
/// the code object was not.
TEST(TestKpackModuleCacheLoad, ReportsAnUnloadableCodeObject)
{
    // hipModuleLoadData needs a device to reject a code object *as* a code object. Without
    // one it still fails, but for want of a context, which is a different claim.
    SKIP_IF_NO_DEVICES();

    ASSERT_TRUE(std::filesystem::exists(REAL_ARCHIVE))
        << "the kpack test asset named at configure time is missing: " << REAL_ARCHIVE;

    try
    {
        KpackModuleCache::load(REAL_ARCHIVE, ARCHIVE_TOC_KEY, ARCHIVE_ARCH);
        FAIL() << "expected HIP to reject a payload that is not a code object";
    }
    catch(const KpackModuleLoadFailure& failure)
    {
        EXPECT_EQ(failure.stage(), KpackLoadStage::MODULE_LOAD)
            << "a code object that survived extraction must not be reported as a reader "
               "failure: "
            << failure.what();
        EXPECT_NE(std::string(failure.what()).find("hipModuleLoadData rejected"), std::string::npos)
            << failure.what();
    }

    // Clear the HIP error state left by the intentional load failure, or the
    // HipErrorHandler listener fails this test for it.
    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

} // namespace
} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
