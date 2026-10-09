// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "PackedKernelSource.hpp"
#include "TestDescriptorRoot.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Container.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

namespace hip_kernel_provider::compilation
{
namespace
{

using hip_kernel_provider::testing::findPackedArchDirectory;
using hip_kernel_provider::testing::PackedKernelSource;
using hip_kernel_provider::testing::readPackedKernelSource;
using hip_kernel_provider::testing::testKpackArchive;
using hip_kernel_provider::testing::unitKpackRoot;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::ScopedDirectory;

/// The arch and toc key of rocm-kpack's own test archive, which testKpackArchive() resolves
/// from this binary's location. Its entries are placeholder payloads rather than HSA code
/// objects, which is what makes it useful here: it is a real container, so the reader parses
/// it, but nothing in it can load.
constexpr const char* ARCHIVE_ARCH = "gfx1100";
constexpr const char* ARCHIVE_TOC_KEY = "lib/libhip.so#0";

/// An arch rocm-kpack's test archive does not hold.
constexpr const char* ABSENT_ARCH = "gfx90a";

/// A declared digest, shaped like a real one so these cases exercise what a descriptor
/// actually carries. The load cases below fail earlier and never reach the comparison,
/// as their stage() assertions prove.
constexpr const char* DIGEST = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// The standalone descriptor of the packed conv set, whose archive this build produced for
/// the local arch. The ordinal case below needs a code object HIP actually accepts, which
/// the test kpack archive's placeholder payloads deliberately are not.
constexpr const char* PACKED_UKD_DESCRIPTOR = "conv_fwd_f16_block64.ukd.json";

/// A toc_key no archive here holds. Looking it up is answered from the TOC kpack_open read
/// into memory, so it reaches codeObject() on the handle without touching the payload.
constexpr const char* ABSENT_TOC_KEY = "lib/no-such-entry.so#0";

/// Holds a SharedKpackArchives lease for a case, as a plugin Container does. Declare it
/// after the case's scratch directory, so the archives it keeps open close before the
/// directory is removed: Windows will not delete an open file.
class ScopedArchiveLease
{
public:
    void release()
    {
        _lease.reset();
    }

private:
    std::shared_ptr<void> _lease = SharedKpackArchives::lease();
};

/// Whether SharedKpackArchives keeps @p archive open once this call drops its own handle.
[[nodiscard]] bool retainedAfterOpen(const std::filesystem::path& archive)
{
    const std::weak_ptr<const OpenKpackArchive> probe = SharedKpackArchives::open(archive.string());
    return !probe.expired();
}

/// How load() fails for @p tocKey at @p arch in @p archive. Empty if it did not fail.
std::optional<KpackModuleLoadFailure> loadFailure(const std::filesystem::path& archive,
                                                  const std::string& tocKey,
                                                  const std::string& arch)
{
    try
    {
        KpackModuleCache::load(archive.string(), tocKey, arch, 0, DIGEST);
    }
    catch(const KpackModuleLoadFailure& failure)
    {
        return failure;
    }
    return std::nullopt;
}

/// How load() fails for an arch @p archive lacks: at ARCH_LOOKUP once the archive opens,
/// before any HIP call, or at OPEN_ARCHIVE when it does not. Empty if it did not fail.
std::optional<KpackModuleLoadFailure> loadForAnAbsentArch(const std::filesystem::path& archive)
{
    return loadFailure(archive, ARCHIVE_TOC_KEY, ABSENT_ARCH);
}

constexpr std::uintmax_t CORRUPTION_BYTE_COUNT = 64;

/// Overwrites @p archive in place with bytes kpack cannot open. False if the write did not
/// land, so no case mistakes an intact archive for a reused handle.
[[nodiscard]] bool corruptInPlace(const std::filesystem::path& archive)
{
    {
        std::ofstream corrupt(archive, std::ios::binary | std::ios::trunc);
        corrupt << std::string(CORRUPTION_BYTE_COUNT, '\0');
        corrupt.close();
        if(corrupt.fail())
        {
            return false;
        }
    }
    std::error_code error;
    return std::filesystem::file_size(archive, error) == CORRUPTION_BYTE_COUNT && !error;
}

/// Copies the test archive to @p archive with its TOC's `gfx_arches` key renamed to one of
/// the same length, so the reader opens it and finds no architecture list. False if the
/// key was not found or the copy did not land.
[[nodiscard]] bool copyWithoutArchitectures(const std::filesystem::path& archive)
{
    std::ifstream in(testKpackArchive(), std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::string key = "gfx_arches";
    const auto at = bytes.find(key);
    if(at == std::string::npos)
    {
        return false;
    }
    bytes[at + key.size() - 1] = 'z';

    std::ofstream out(archive, std::ios::binary | std::ios::trunc);
    out << bytes;
    out.close();
    return !out.fail();
}

/// Device 0's bare arch and the packed conv entry this build produced for it. Leaves
/// `packed.archive` empty when nothing was packaged for the device. Call through
/// ASSERT_NO_FATAL_FAILURE.
void findPackedConvEntry(std::string& arch, PackedKernelSource& packed)
{
    hipDeviceProp_t properties{};
    std::filesystem::path packaged;
    ASSERT_NO_FATAL_FAILURE(findPackedArchDirectory(properties, arch, packaged));
    if(!packaged.empty())
    {
        ASSERT_NO_FATAL_FAILURE(readPackedKernelSource(packaged, PACKED_UKD_DESCRIPTOR, packed));
    }
}

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
    ASSERT_TRUE(std::filesystem::exists(testKpackArchive()))
        << "the test kpack archive, resolved relative to this binary, is missing: "
        << testKpackArchive();

    try
    {
        KpackModuleCache::load(
            testKpackArchive().string(), ARCHIVE_TOC_KEY, ARCHIVE_ARCH, 0, DIGEST);
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
    ASSERT_TRUE(std::filesystem::exists(testKpackArchive()))
        << "the test kpack archive, resolved relative to this binary, is missing: "
        << testKpackArchive();

    try
    {
        KpackModuleCache::load(
            testKpackArchive().string(), ARCHIVE_TOC_KEY, ABSENT_ARCH, 0, DIGEST);
        FAIL() << "expected an arch the archive does not hold to be rejected";
    }
    catch(const KpackModuleLoadFailure& failure)
    {
        // load's own pre-check, not the reader, which cannot tell "wrong GPU" from "wrong
        // toc_key" -- both are KERNEL_NOT_FOUND.
        EXPECT_EQ(failure.stage(), KpackLoadStage::ARCH_LOOKUP) << failure.what();

        const std::string message = failure.what();
        EXPECT_NE(message.find(ABSENT_ARCH), std::string::npos)
            << "the message must name the arch that was asked for: " << message;
        EXPECT_NE(message.find(ARCHIVE_ARCH), std::string::npos)
            << "the message must name the arches the archive provides: " << message;
    }
}

TEST(TestKpackModuleCacheLoad, ReportsAnArchiveThatDeclaresNoArchitectures)
{
    const ScopedDirectory scratch = claimScratchDirectory("kpackmodulecache");
    const ScopedArchiveLease lease;
    const std::filesystem::path archive = scratch.path() / "no-arches.kpack";
    ASSERT_TRUE(copyWithoutArchitectures(archive)) << "could not write " << archive;

    // Twice, under a lease: a second load must refuse the same way rather than answer from
    // a handle the first retained and report the archive as built for some other GPU.
    for(int attempt = 0; attempt < 2; ++attempt)
    {
        const auto failure = loadForAnAbsentArch(archive);
        ASSERT_TRUE(failure.has_value()) << "attempt " << attempt;
        EXPECT_EQ(failure->stage(), KpackLoadStage::ARCH_LOOKUP) << failure->what();
        EXPECT_NE(std::string(failure->what()).find("declares no architectures"), std::string::npos)
            << "attempt " << attempt << ": " << failure->what();
    }
}

TEST(TestKpackModuleCacheLoad, ALaterLoadReadsEntriesThroughTheArchiveAnEarlierOneOpened)
{
    ASSERT_TRUE(std::filesystem::exists(testKpackArchive()))
        << "the test kpack archive, resolved relative to this binary, is missing: "
        << testKpackArchive();
    const ScopedDirectory scratch = claimScratchDirectory("kpackmodulecache");
    const ScopedArchiveLease lease;
    const std::filesystem::path archive = scratch.path() / "shared.kpack";
    std::filesystem::copy_file(testKpackArchive(), archive);
    const std::size_t before = SharedKpackArchives::opensForTesting();

    // An absent toc_key at an arch the archive holds: past the arch check and into
    // codeObject(), without a stage that would evict the handle.
    for(int attempt = 0; attempt < 2; ++attempt)
    {
        const auto failure = loadFailure(archive, ABSENT_TOC_KEY, ARCHIVE_ARCH);
        ASSERT_TRUE(failure.has_value()) << "attempt " << attempt;
        EXPECT_EQ(failure->stage(), KpackLoadStage::ENTRY_LOOKUP)
            << "attempt " << attempt << ": " << failure->what();
    }

    // One open between them: the second load read its arch list and entry through the
    // handle the first opened.
    EXPECT_EQ(SharedKpackArchives::opensForTesting() - before, 1U);
}

TEST(TestKpackModuleCacheLoad, AFailedOpenIsRetriedRatherThanRemembered)
{
    const ScopedDirectory scratch = claimScratchDirectory("kpackmodulecache");
    const ScopedArchiveLease lease;
    const std::filesystem::path archive = scratch.path() / "arrives-later.kpack";

    const auto absent = loadForAnAbsentArch(archive);
    ASSERT_TRUE(absent.has_value());
    EXPECT_EQ(absent->stage(), KpackLoadStage::OPEN_ARCHIVE) << absent->what();
    EXPECT_NE(std::string(absent->what()).find("does not exist"), std::string::npos)
        << absent->what();

    // The same path, once the archive exists, opens: the failure was not cached.
    std::filesystem::copy_file(testKpackArchive(), archive);
    const auto present = loadForAnAbsentArch(archive);
    ASSERT_TRUE(present.has_value());
    EXPECT_EQ(present->stage(), KpackLoadStage::ARCH_LOOKUP) << present->what();
}

TEST(TestKpackModuleCacheLoad, ADecodeFailureEvictsTheHandleButAMissingEntryOrArchDoesNot)
{
    const ScopedDirectory scratch = claimScratchDirectory("kpackmodulecache");
    const ScopedArchiveLease lease;
    const std::filesystem::path archive = scratch.path() / "evicted.kpack";
    std::filesystem::copy_file(testKpackArchive(), archive);
    const std::weak_ptr<const OpenKpackArchive> probe = SharedKpackArchives::open(archive.string());
    ASSERT_FALSE(probe.expired()) << "a lease is held, so the archive should be retained";

    // Intact in both: it lacks what was asked for. Not evicting rests on memoized discovery;
    // see KpackModuleCache::load.
    const auto absentArch = loadForAnAbsentArch(archive);
    ASSERT_TRUE(absentArch.has_value());
    ASSERT_EQ(absentArch->stage(), KpackLoadStage::ARCH_LOOKUP) << absentArch->what();
    EXPECT_FALSE(probe.expired()) << "a missing arch must not evict the handle";

    const auto absentEntry = loadFailure(archive, ABSENT_TOC_KEY, ARCHIVE_ARCH);
    ASSERT_TRUE(absentEntry.has_value());
    ASSERT_EQ(absentEntry->stage(), KpackLoadStage::ENTRY_LOOKUP) << absentEntry->what();
    EXPECT_FALSE(probe.expired()) << "a missing entry must not evict the handle";

    // An entry the TOC lists that does not decode is damage, so the handle goes.
    const auto undecodable = loadFailure(archive, ARCHIVE_TOC_KEY, ARCHIVE_ARCH);
    ASSERT_TRUE(undecodable.has_value());
    ASSERT_EQ(undecodable->stage(), KpackLoadStage::DECOMPRESS) << undecodable->what();
    EXPECT_TRUE(probe.expired()) << "a decode failure must evict the handle";
}

TEST(TestKpackModuleCacheLoad, AnArchiveReplacedAfterADecodeFailureIsReadWithoutAReset)
{
    const ScopedDirectory scratch = claimScratchDirectory("kpackmodulecache");
    const ScopedArchiveLease lease;
    const std::filesystem::path archive = scratch.path() / "replaced.kpack";
    std::filesystem::copy_file(testKpackArchive(), archive);

    const auto undecodable = loadFailure(archive, ARCHIVE_TOC_KEY, ARCHIVE_ARCH);
    ASSERT_TRUE(undecodable.has_value());
    ASSERT_EQ(undecodable->stage(), KpackLoadStage::DECOMPRESS) << undecodable->what();

    // The failure closed the archive, so replacing it does not race an open handle.
    ASSERT_TRUE(copyWithoutArchitectures(archive)) << "could not rewrite " << archive;

    // Still leased and never reset: only a reopen can see the replacement.
    const auto replaced = loadFailure(archive, ARCHIVE_TOC_KEY, ARCHIVE_ARCH);
    ASSERT_TRUE(replaced.has_value());
    EXPECT_EQ(replaced->stage(), KpackLoadStage::ARCH_LOOKUP) << replaced->what();
    EXPECT_NE(std::string(replaced->what()).find("declares no architectures"), std::string::npos)
        << replaced->what();
}

TEST(TestKpackModuleCacheLoad, ADigestMismatchEvictsTheHandle)
{
    SKIP_IF_NO_DEVICES();

    std::string arch;
    PackedKernelSource packed;
    ASSERT_NO_FATAL_FAILURE(findPackedConvEntry(arch, packed));
    if(packed.archive.empty())
    {
        GTEST_SKIP() << "nothing was packaged for this device (" << arch << ") under "
                     << unitKpackRoot();
    }

    const ScopedDirectory scratch = claimScratchDirectory("kpackmodulecache");
    const ScopedArchiveLease lease;
    const std::filesystem::path archive = scratch.path() / "mismatched.kpack";
    std::filesystem::copy_file(packed.archive, archive);
    const std::weak_ptr<const OpenKpackArchive> probe = SharedKpackArchives::open(archive.string());
    ASSERT_FALSE(probe.expired()) << "a lease is held, so the archive should be retained";

    // DIGEST never matches a real code object.
    const auto mismatch = loadFailure(archive, packed.tocKey, arch);
    ASSERT_TRUE(mismatch.has_value());
    ASSERT_EQ(mismatch->stage(), KpackLoadStage::DIGEST_MISMATCH) << mismatch->what();
    EXPECT_TRUE(probe.expired()) << "a digest mismatch must evict the handle";

    // The archive reopens for the next load.
    EXPECT_NE(KpackModuleCache::load(archive.string(), packed.tocKey, arch, 0, packed.sha256),
              nullptr);
}

TEST(TestSharedKpackArchives, NothingIsRetainedWithoutALease)
{
    const ScopedDirectory scratch = claimScratchDirectory("kpackmodulecache");
    const std::filesystem::path archive = scratch.path() / "unleased.kpack";
    std::filesystem::copy_file(testKpackArchive(), archive);

    EXPECT_FALSE(retainedAfterOpen(archive));

    // The load closed the archive on return, so the next one reads the rewritten file.
    const auto opened = loadForAnAbsentArch(archive);
    ASSERT_TRUE(opened.has_value());
    ASSERT_EQ(opened->stage(), KpackLoadStage::ARCH_LOOKUP) << opened->what();
    ASSERT_TRUE(corruptInPlace(archive)) << "could not rewrite " << archive;
    const auto reread = loadForAnAbsentArch(archive);
    ASSERT_TRUE(reread.has_value());
    EXPECT_EQ(reread->stage(), KpackLoadStage::OPEN_ARCHIVE) << reread->what();
}

TEST(TestSharedKpackArchives, ReleasingTheLastLeaseClosesEveryArchive)
{
    const ScopedDirectory scratch = claimScratchDirectory("kpackmodulecache");
    ScopedArchiveLease first;
    ScopedArchiveLease second;
    const std::filesystem::path archive = scratch.path() / "released.kpack";
    std::filesystem::copy_file(testKpackArchive(), archive);
    const std::weak_ptr<const OpenKpackArchive> probe = SharedKpackArchives::open(archive.string());
    ASSERT_FALSE(probe.expired()) << "a lease is held, so the archive should be retained";

    first.release();
    EXPECT_FALSE(probe.expired()) << "a lease is still held, so the archive must stay open";

    second.release();
    EXPECT_TRUE(probe.expired()) << "the last lease is gone, so the archive must be closed";
}

TEST(TestSharedKpackArchives, AContainerHoldsALeaseForItsLifetime)
{
    const ScopedDirectory scratch = claimScratchDirectory("kpackmodulecache");
    const std::filesystem::path archive = scratch.path() / "container.kpack";
    std::filesystem::copy_file(testKpackArchive(), archive);

    std::weak_ptr<const OpenKpackArchive> probe;
    {
        const core::Container container;
        probe = SharedKpackArchives::open(archive.string());
        EXPECT_FALSE(probe.expired()) << "a live Container must keep archives open";
    }
    EXPECT_TRUE(probe.expired()) << "destroying the last Container must close its archives";
}

TEST(TestSharedKpackArchives, ConcurrentFirstOpensOfOneArchiveShareOneHandle)
{
    const ScopedDirectory scratch = claimScratchDirectory("kpackmodulecache");
    const ScopedArchiveLease lease;
    const std::filesystem::path archive = scratch.path() / "contended.kpack";
    std::filesystem::copy_file(testKpackArchive(), archive);
    const std::string path = archive.string();
    const std::size_t before = SharedKpackArchives::opensForTesting();

    constexpr std::size_t THREAD_COUNT = 8;
    std::vector<std::shared_ptr<const OpenKpackArchive>> handles(THREAD_COUNT);
    std::atomic<std::size_t> waiting{THREAD_COUNT};
    std::vector<std::thread> threads;
    threads.reserve(THREAD_COUNT);
    for(std::size_t index = 0; index < THREAD_COUNT; ++index)
    {
        threads.emplace_back([&handles, &waiting, &path, index] {
            // Released together, so every thread reaches open() before any handle exists.
            --waiting;
            while(waiting.load() != 0)
            {
                std::this_thread::yield();
            }
            try
            {
                handles[index] = SharedKpackArchives::open(path);
            }
            catch(const KpackModuleLoadFailure&)
            {
                handles[index] = nullptr;
            }
        });
    }
    for(auto& thread : threads)
    {
        thread.join();
    }

    ASSERT_NE(handles.front(), nullptr);
    for(std::size_t index = 1; index < THREAD_COUNT; ++index)
    {
        EXPECT_EQ(handles[index], handles.front()) << "thread " << index;
    }
    EXPECT_EQ(SharedKpackArchives::opensForTesting() - before, 1U)
        << "concurrent first opens of one archive must open it once";
}

TEST(TestSharedKpackArchives, TheIngestorResetHookClosesRetainedArchivesButKeepsTheLease)
{
    const ScopedDirectory scratch = claimScratchDirectory("kpackmodulecache");
    const ScopedArchiveLease lease;
    const std::filesystem::path archive = scratch.path() / "reset.kpack";
    std::filesystem::copy_file(testKpackArchive(), archive);
    const std::weak_ptr<const OpenKpackArchive> probe = SharedKpackArchives::open(archive.string());
    ASSERT_FALSE(probe.expired()) << "a lease is held, so the archive should be retained";

    // The hook the plugin exports to the integration suite, not the class's own reset.
    kernel_ingestor_engine::resetIngestorModuleCachesForTesting();
    EXPECT_TRUE(probe.expired()) << "the reset hook must close every retained archive";
    EXPECT_TRUE(retainedAfterOpen(archive)) << "the reset must leave the lease in force";

    // The integration suite corrupts and then resets, with nothing open at the rewrite.
    // Resetting first gives the same here: nothing is left open to rewrite, and the next
    // load reads the damage.
    kernel_ingestor_engine::resetIngestorModuleCachesForTesting();
    ASSERT_TRUE(corruptInPlace(archive)) << "could not rewrite " << archive;
    const auto reread = loadForAnAbsentArch(archive);
    ASSERT_TRUE(reread.has_value());
    EXPECT_EQ(reread->stage(), KpackLoadStage::OPEN_ARCHIVE) << reread->what();
    EXPECT_NE(std::string(reread->what()).find("could not be read"), std::string::npos)
        << reread->what();
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

    // `arch` is device 0's, and the archive holds one blob per arch. A peer on a different
    // ISA would be refused by hipModuleLoadData before the load ever reached the cache
    // assertions, so the positive case below is only meaningful on a peer that can run
    // device 0's object -- ordinal 1 is not that peer by construction.
    int peer = -1;
    for(int ordinal = 1; ordinal < devices; ++ordinal)
    {
        hipDeviceProp_t peerProperties{};
        ASSERT_EQ(hipGetDeviceProperties(&peerProperties, ordinal), hipSuccess);
        if(hipdnn_plugin_sdk::stripArchFeatures(peerProperties.gcnArchName) == arch)
        {
            peer = ordinal;
            break;
        }
    }

    if(peer >= 0)
    {
        const auto second
            = cache.getOrLoad(packed.archive.string(), packed.tocKey, arch, peer, packed.sha256);
        ASSERT_NE(second, nullptr);
        EXPECT_NE(second, first);
        EXPECT_EQ(cache.size(), 2U);
        return;
    }

    // No peer that can run device 0's object -- one device, or a mixed-ISA host. An ordinal
    // past the last device stands in: the miss still loads, and the load throws at its bind
    // because there is no such device to make current. That throw is what makes this case
    // discriminating on the single-device hosts CI runs -- an ordinal-blind key would hit
    // the entry above and hand back device 0's module without a sound.
    const int absent = devices;
    try
    {
        cache.getOrLoad(packed.archive.string(), packed.tocKey, arch, absent, packed.sha256);
        FAIL() << "expected ordinal " << absent
               << " to miss the ordinal-0 entry and fail its bind, but a module was returned";
    }
    catch(const KpackModuleLoadFailure& failure)
    {
        // MODULE_LOAD: every stage that reads the archive already succeeded for ordinal 0,
        // so the only thing left to refuse is the device.
        EXPECT_EQ(failure.stage(), KpackLoadStage::MODULE_LOAD) << failure.what();
        EXPECT_NE(std::string(failure.what())
                      .find("cannot make device " + std::to_string(absent) + " current"),
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
