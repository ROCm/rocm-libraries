// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "PackedKernelSource.hpp"
#include "TestDescriptorRoot.hpp"
#include "compilation/KpackModuleCache.hpp"

namespace hip_kernel_provider::compilation
{
namespace
{

using hip_kernel_provider::testing::findPackedArchDirectory;
using hip_kernel_provider::testing::PackedKernelSource;
using hip_kernel_provider::testing::readPackedKernelSource;
using hip_kernel_provider::testing::unitKpackRoot;

/// rocm-kpack's own test archive, vendored beside this test. Its entries are placeholder
/// payloads rather than HSA code objects, which is what makes it useful here: it is a
/// real container, so the reader parses it, but nothing in it can load.
constexpr const char* REAL_ARCHIVE = HIPKERNELPROVIDER_TEST_KPACK_ARCHIVE;
constexpr const char* ARCHIVE_ARCH = "gfx1100";
constexpr const char* ARCHIVE_TOC_KEY = "lib/libhip.so#0";

/// A declared digest, shaped like a real one so these cases exercise what a descriptor
/// actually carries. The load cases below fail earlier and never reach the comparison,
/// as their stage() assertions prove.
constexpr const char* DIGEST = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// The standalone descriptor of the packed conv set, whose archive this build produced for
/// the local arch. The ordinal case below needs a code object HIP actually accepts, which
/// REAL_ARCHIVE's placeholder payloads deliberately are not.
constexpr const char* PACKED_UKD_DESCRIPTOR = "conv_fwd_f16_block64.ukd.json";

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

TEST(TestKpackModuleCacheLoad, ASecondOrdinalDoesNotAnswerFromTheFirstOrdinalsEntry)
{
    SKIP_IF_NO_DEVICES();

    std::string arch;
    std::filesystem::path packaged;
    hipDeviceProp_t properties{};
    ASSERT_NO_FATAL_FAILURE(findPackedArchDirectory(properties, arch, packaged));
    if(packaged.empty())
    {
        GTEST_SKIP() << "nothing was packaged for this device (" << arch
                     << "): " << unitKpackRoot() / arch
                     << " does not exist. Environmental -- the build packs per arch and this "
                        "device is outside GPU_TARGETS.";
    }

    PackedKernelSource packed;
    ASSERT_NO_FATAL_FAILURE(readPackedKernelSource(packaged, PACKED_UKD_DESCRIPTOR, packed));

    // A cache of its own, so ordinal 0's entry below is the only thing ordinal 1 can hit.
    KpackModuleCache cache;

    const auto first
        = cache.getOrLoad(packed.archive.string(), packed.tocKey, arch, 0, packed.sha256);
    ASSERT_NE(first, nullptr);
    ASSERT_EQ(cache.size(), 1U);

    int devices = 0;
    ASSERT_EQ(hipGetDeviceCount(&devices), hipSuccess);

    if(devices > 1)
    {
        const auto second
            = cache.getOrLoad(packed.archive.string(), packed.tocKey, arch, 1, packed.sha256);
        ASSERT_NE(second, nullptr);
        EXPECT_NE(second, first);
        EXPECT_EQ(cache.size(), 2U);
        return;
    }

    // One device: the miss still loads, and the load throws at its bind because there is no
    // device 1 to make current. That throw is what makes this case discriminating on the
    // single-device hosts CI runs -- an ordinal-blind key would hit the entry above and
    // hand back device 0's module without a sound.
    try
    {
        cache.getOrLoad(packed.archive.string(), packed.tocKey, arch, 1, packed.sha256);
        FAIL() << "expected ordinal 1 to miss the ordinal-0 entry and fail its bind, but a "
                  "module was returned on a host with one device";
    }
    catch(const KpackModuleLoadFailure& failure)
    {
        // MODULE_LOAD: every stage that reads the archive already succeeded for ordinal 0,
        // so the only thing left to refuse is the device.
        EXPECT_EQ(failure.stage(), KpackLoadStage::MODULE_LOAD) << failure.what();
        EXPECT_NE(std::string(failure.what()).find("cannot make device 1 current"),
                  std::string::npos)
            << failure.what();
    }

    EXPECT_EQ(cache.size(), 1U);
    EXPECT_TRUE(cache.contains(packed.archive.string(), packed.tocKey, arch, 0, packed.sha256));

    // Clear the HIP error state left by the intentional hipSetDevice failure, or the
    // HipErrorHandler listener fails this test for it.
    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

} // namespace
} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
