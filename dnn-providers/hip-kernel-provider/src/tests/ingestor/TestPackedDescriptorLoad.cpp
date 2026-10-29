// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <string>
#include <system_error>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "TestDescriptorRoot.hpp"

/**
 * @file TestPackedDescriptorLoad.cpp
 * @brief The packer/loader seam: real packer OUTPUT read by the real loader.
 *
 * Every other test on this boundary sees only one side of it. The pytest suite under
 * `descriptor-packaging/` validates the packer against the packer's own model of what it
 * emitted, and `TestDescriptorLoader.cpp` validates the loader against JSON a C++ test
 * typed out by hand -- including two cases whose own comments say they mirror
 * `pipeline.py::_rewrite_ukd_kpack`, which is a copy of the shape, not the shape itself.
 * Nothing ran the bytes the packer actually wrote through the parser that has to read
 * them.
 *
 * Two blocking defects have already lived in exactly that gap and survived a fully green
 * suite:
 *
 *   * `library` was written arch-root-relative and read descriptor-relative, so every
 *     NESTED descriptor named an archive that was not there. It resolved only for a
 *     descriptor sitting flat at the arch root -- which was every layout that existed
 *     before path preservation made nesting normal.
 *   * a descriptor tree was authored to an invented schema (version `0.1`, non-UUID ids,
 *     wrong per-type field names), so zero descriptors would have loaded.
 *
 * Both are data-shape errors only a real load can see. The cases below are written to
 * fail if either is reintroduced.
 *
 * No device is required: the staged tree is on disk whether or not this machine has a
 * GPU, so this runs in every configuration that packs anything.
 */

namespace hip_kernel_provider::ingestor_seam
{
namespace
{

using hipdnn_plugin_sdk::ingestor::DescriptorCatalog;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;
using hipdnn_plugin_sdk::ingestor::loadDescriptorCatalog;
using hipdnn_plugin_sdk::ingestor::resolveDescriptorSets;

using hip_kernel_provider::testing::unitKpackRoot;

/// Every per-arch shard this build produced.
///
/// Enumerated from the filesystem rather than from the local device on purpose: the seam
/// under test is the packer's OUTPUT SHAPE, which is device-independent. Keying on the
/// running device would turn this into a test that quietly skips on every machine whose
/// GPU differs from what CI packed.
std::vector<std::filesystem::path> packedArchShards()
{
    std::vector<std::filesystem::path> shards;

    std::error_code ec;
    const std::filesystem::path& root = unitKpackRoot();
    if(!std::filesystem::is_directory(root, ec))
    {
        return shards;
    }

    for(const auto& entry : std::filesystem::directory_iterator(root, ec))
    {
        // A shard is an arch directory holding a kpack/ child.
        std::error_code inner;
        if(entry.is_directory(inner)
           && std::filesystem::is_directory(entry.path() / "kpack", inner))
        {
            shards.push_back(entry.path());
        }
    }

    std::sort(shards.begin(), shards.end());
    return shards;
}

/// The shards, or a skip explaining that nothing was packed.
///
/// Skipping is correct only for "the packaging rule never ran" -- no hipcc, production
/// source root unset. Once a shard exists everything below is an assertion: a staged tree
/// that cannot be loaded is the defect this file exists to catch, never a reason to pass
/// quietly.
///
/// A root that is not a directory fails. The probe below reads the children of one fixed
/// level, so a tree the build lays out somewhere else answers "nothing was packed" and
/// skips every case in this file.
#define REQUIRE_PACKED_SHARDS(shards)                                                     \
    std::error_code missingRoot;                                                          \
    ASSERT_TRUE(std::filesystem::is_directory(unitKpackRoot(), missingRoot))              \
        << "the packed set root " << unitKpackRoot()                                      \
        << " is not a directory. The staged tree sits elsewhere, or this binary holds a " \
           "stale offset to it.";                                                         \
    /* NOLINTNEXTLINE(bugprone-macro-parentheses) declarator name, not an expression */   \
    const auto shards = packedArchShards();                                               \
    if((shards).empty())                                                                  \
    {                                                                                     \
        GTEST_SKIP() << "no packed arch shard under " << unitKpackRoot()                  \
                     << " -- the packaging rule did not run. Configure with "             \
                        "-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON and a discoverable hipcc, "   \
                        "then build the descriptor staging targets.";                     \
    }                                                                                     \
    static_assert(true, "swallow the trailing semicolon")

/// Every kernel source @p kernel launches: each entry of a multi-launch kernel's `sources`,
/// or `source` for a single-launch one. A multi-launch kernel leaves `source`
/// default-constructed, whose kind is EMBEDDED_SOURCE, so a check reading only `source`
/// would skip it, or fail it for the wrong reason.
std::vector<const hipdnn_plugin_sdk::ingestor::KernelSource*>
    launchSources(const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel)
{
    if(kernel.sources.empty())
    {
        return {&kernel.source};
    }
    std::vector<const hipdnn_plugin_sdk::ingestor::KernelSource*> sources;
    sources.reserve(kernel.sources.size());
    for(const auto& [launch, source] : kernel.sources)
    {
        sources.push_back(&source);
    }
    return sources;
}

/// The arch shards of a packed fixture root (RFC 0021): the packer's output for
/// `descriptor-packaging/tests/fixtures/<fixture>`, staged as a sibling of the unit and
/// integration trees so that the embedded-source verification of the unit tree does not
/// walk it.
///
/// Reached from the unit kpack root because a binary that finds one finds both. Absent
/// when the unit shards exist, the fixture's pack did not run: that fails, since a shard
/// that was never produced is exactly what these cases must not pass over.
std::filesystem::path packedFixtureRoot(const char* fixture)
{
    return unitKpackRoot().parent_path().parent_path() / fixture;
}

std::vector<std::filesystem::path> packedFixtureShards(const char* fixture)
{
    std::vector<std::filesystem::path> shards;

    std::error_code ec;
    for(const auto& entry : std::filesystem::directory_iterator(packedFixtureRoot(fixture), ec))
    {
        std::error_code inner;
        if(entry.is_directory(inner))
        {
            shards.push_back(entry.path());
        }
    }

    std::sort(shards.begin(), shards.end());
    return shards;
}

/// Calls `visit(shard, kernel)` for every kernel of every pack of every descriptor set
/// that resolves from each shard's loaded catalog: the engine's own packs and, for a
/// slotted engine, each slot's packs. Stops at the first fatal (ASSERT_) failure inside
/// `visit`, as the test would have aborted.
template <typename Visit>
void forEachPackedKernel(const std::vector<std::filesystem::path>& shards, Visit&& visit)
{
    const auto visitPack = [&visit](const auto& shard, const auto& pack) {
        for(const auto& kernel : pack.kernels)
        {
            visit(shard, kernel);
            if(::testing::Test::HasFatalFailure())
            {
                return false;
            }
        }
        return true;
    };

    for(const auto& shard : shards)
    {
        for(const auto& set : resolveDescriptorSets(loadDescriptorCatalog(shard)))
        {
            for(const auto& pack : set.packs)
            {
                if(!visitPack(shard, pack))
                {
                    return;
                }
            }
            for(const auto& slot : set.slots)
            {
                for(const auto& pack : slot.packs)
                {
                    if(!visitPack(shard, pack))
                    {
                        return;
                    }
                }
            }
        }
    }
}

/// The shards of @p fixture, a skip when nothing was packed at all, or a failure when the
/// unit pack ran and the fixture's did not.
#define REQUIRE_PACKED_FIXTURE_SHARDS(fixtureShards, fixture, packingTarget)                   \
    REQUIRE_PACKED_SHARDS(unitShards);                                                         \
    /* NOLINTNEXTLINE(bugprone-macro-parentheses) declarator name, not an expression */        \
    const auto fixtureShards = packedFixtureShards(fixture);                                   \
    ASSERT_FALSE((fixtureShards).empty())                                                      \
        << "the unit shards exist but the fixture '" << (fixture) << "' was not packed under " \
        << packedFixtureRoot(fixture) << ". Its packing root (" << (packingTarget)             \
        << ") did not run or is not a dependency of this binary.";                             \
    static_assert(true, "swallow the trailing semicolon")

/// Whether the unit tree holds a shard for @p arch, so this build packed for it.
bool unitTreeHasShard(const std::string& arch)
{
    const auto shards = packedArchShards();
    return std::any_of(shards.begin(), shards.end(), [&arch](const auto& shard) {
        return shard.filename() == arch;
    });
}

/// The shards of @p fixture, authored for the one arch @p arch. Decided from the staged
/// content, never from GPU_TARGETS or the device: a skip when the unit tree has no @p arch
/// shard, and otherwise the fixture's @p arch shard is required.
#define REQUIRE_PACKED_FIXED_ARCH_FIXTURE_SHARDS(fixtureShards, fixture, arch, packingTarget)     \
    REQUIRE_PACKED_SHARDS(unitShards);                                                            \
    if(!unitTreeHasShard(arch))                                                                   \
    {                                                                                             \
        GTEST_SKIP() << "the unit tree has no " << (arch) << " shard under " << unitKpackRoot()   \
                     << ": this build does not pack for " << (arch) << ", and the '" << (fixture) \
                     << "' fixture is " << (arch) << "-only content.";                            \
    }                                                                                             \
    /* NOLINTNEXTLINE(bugprone-macro-parentheses) declarator name, not an expression */           \
    const auto fixtureShards = packedFixtureShards(fixture);                                      \
    ASSERT_EQ((fixtureShards).size(), 1U)                                                         \
        << "the unit tree has a " << (arch) << " shard, so the fixture '" << (fixture)            \
        << "' must hold exactly its " << (arch) << " shard under " << packedFixtureRoot(fixture)  \
        << ". Its packing root (" << (packingTarget)                                              \
        << ") did not run or is not a dependency of this binary.";                                \
    ASSERT_EQ((fixtureShards).front().filename().string(), std::string(arch))                     \
        << "the fixture '" << (fixture) << "' holds " << (fixtureShards).front() << ", not its "  \
        << (arch) << " shard.";                                                                   \
    static_assert(true, "swallow the trailing semicolon")

#define REQUIRE_PACKED_SLOTTED_SHARDS(slottedShards) \
    REQUIRE_PACKED_FIXTURE_SHARDS(slottedShards, "slotted", "hkp_packaging_slotted")

#define REQUIRE_PACKED_SLOTTED_HSACO_SHARDS(hsacoShards) \
    REQUIRE_PACKED_FIXED_ARCH_FIXTURE_SHARDS(            \
        hsacoShards, "slotted_hsaco", "gfx950", "hkp_packaging_slotted_hsaco")

// The multi-launch fixture is rocKE content: it packs only in a build with rocKE on.
#ifdef HIPKERNELPROVIDER_ENABLE_ROCKE
#define REQUIRE_PACKED_SLOTTED_MULTI_LAUNCH_SHARDS(multiLaunchShards) \
    REQUIRE_PACKED_FIXED_ARCH_FIXTURE_SHARDS(multiLaunchShards,       \
                                             "slotted_multi_launch",  \
                                             "gfx1151",               \
                                             "hkp_packaging_slotted_multi_launch")
#else
// The skip is conditional on the empty list so the test body is not unreachable code
// (-Wunreachable-code) in a build without rocKE.
#define REQUIRE_PACKED_SLOTTED_MULTI_LAUNCH_SHARDS(multiLaunchShards)                     \
    const std::vector<std::filesystem::path> multiLaunchShards;                           \
    if(multiLaunchShards.empty())                                                         \
    {                                                                                     \
        GTEST_SKIP() << "the slotted multi-launch fixture is rocKE content and this "     \
                        "build packs without rocKE (HIPKERNELPROVIDER_ENABLE_ROCKE=OFF)"; \
    }                                                                                     \
    static_assert(true)
#endif

constexpr const char* SLOTTED_ENGINE = "hipkernel:SlottedFixture";
constexpr const char* SLOTTED_HSACO_ENGINE = "hipkernel:SlottedHsaco";
constexpr const char* SLOTTED_MULTI_LAUNCH_ENGINE = "hipkernel:SlottedMultiLaunch";

/// The engine named @p name in @p shard's resolved sets, or nullptr. @p sets owns it.
const hipdnn_plugin_sdk::ingestor::DescriptorSet*
    findEngine(const std::vector<hipdnn_plugin_sdk::ingestor::DescriptorSet>& sets,
               const std::string& name)
{
    const auto found = std::find_if(
        sets.begin(), sets.end(), [&name](const auto& set) { return set.engine.name == name; });
    return found == sets.end() ? nullptr : &*found;
}

/// Every launch of every packed kernel in @p shards, slot packs included, names a code
/// object that is on disk and carries the coordinates the kpack adapter needs.
void expectPackedLaunchesAreUsable(const std::vector<std::filesystem::path>& shards)
{
    size_t launches = 0;
    forEachPackedKernel(shards, [&launches](const auto& shard, const auto& kernel) {
        for(const auto* source : launchSources(kernel))
        {
            EXPECT_EQ(source->kind, KernelSourceKind::KPACK) << kernel.name << " in " << shard;
            EXPECT_FALSE(source->tocKey.empty()) << kernel.name;
            EXPECT_FALSE(source->symbol.empty()) << kernel.name;
            EXPECT_FALSE(source->sha256.empty()) << kernel.name;
            EXPECT_FALSE(source->signature.empty()) << kernel.name;
            EXPECT_TRUE(std::filesystem::exists(
                std::filesystem::weakly_canonical(kernel.originDirectory / source->library)))
                << "kernel '" << kernel.name << "' names an archive that is not there: "
                << (kernel.originDirectory / source->library);
            ++launches;
        }
    });
    EXPECT_GT(launches, 0U);
}

/// The `.ued.json`, `.kdp.json` and `.uhd.json` files under @p shard, to be compared with
/// what the loader accepted: it skips a file it cannot parse, so only a count notices one
/// that vanished.
struct DescriptorFileCounts
{
    size_t engines = 0;
    size_t packs = 0;
    size_t models = 0;
};

DescriptorFileCounts countDescriptorFiles(const std::filesystem::path& shard)
{
    DescriptorFileCounts counts;
    std::error_code ec;
    for(const auto& entry : std::filesystem::recursive_directory_iterator(shard, ec))
    {
        const auto name = entry.path().filename().string();
        const auto endsWith = [&name](const std::string& suffix) {
            return name.size() > suffix.size()
                   && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
        };
        counts.engines += endsWith(".ued.json") ? 1U : 0U;
        counts.packs += endsWith(".kdp.json") ? 1U : 0U;
        counts.models += endsWith(".uhd.json") ? 1U : 0U;
    }
    return counts;
}

/// @p shard holds the expected number of each descriptor file, and the loader accepted
/// every one (see DescriptorFileCounts).
void expectLoaderAcceptedEveryFile(const std::filesystem::path& shard,
                                   size_t engines,
                                   size_t packs,
                                   size_t models)
{
    const auto counts = countDescriptorFiles(shard);
    EXPECT_EQ(counts.engines, engines) << shard;
    EXPECT_EQ(counts.packs, packs) << shard;
    EXPECT_EQ(counts.models, models) << shard;

    const DescriptorCatalog catalog = loadDescriptorCatalog(shard);
    EXPECT_EQ(catalog.engines.size(), counts.engines) << shard;
    EXPECT_EQ(catalog.packs.size(), counts.packs) << shard;
    EXPECT_EQ(catalog.heuristics.size(), counts.models) << shard;
}

} // namespace

// ---------------------------------------------------------------------------
// The packed tree parses at all
// ---------------------------------------------------------------------------

/// EVERY descriptor the packer wrote is one the loader accepts -- counted, not sampled.
///
/// The direct guard for the schema defect, and the assertion has to be a COUNT rather
/// than a non-empty check. `loadDescriptorCatalog` never throws: it logs a file it cannot
/// parse and skips it. So a single unloadable descriptor does not fail anything, it
/// silently DISAPPEARS -- and every later assertion in this file then iterates whatever
/// survived and passes.
///
/// Mutation-testing this file caught exactly that: three mutations that made a descriptor
/// unloadable were survived by an earlier `EXPECT_FALSE(empty())` version of this test,
/// because one engine vanishing still leaves a non-empty catalog. "At least one" is the
/// wrong floor for a guard whose whole job is to notice a drop.
///
/// So: count the `.ued.json` / `.kdp.json` files actually on disk in the shard, and
/// require the loader to account for every one of them.
TEST(TestPackedDescriptorLoad, EveryPackedDescriptorInTheShardIsLoaded)
{
    REQUIRE_PACKED_SHARDS(shards);

    for(const auto& shard : shards)
    {
        const auto counts = countDescriptorFiles(shard);

        const DescriptorCatalog catalog = loadDescriptorCatalog(shard);

        EXPECT_EQ(catalog.engines.size(), counts.engines)
            << shard << " holds " << counts.engines << " .ued.json file(s) but the loader "
            << "accepted " << catalog.engines.size()
            << ". A descriptor the loader rejects is logged at ERROR and SKIPPED, never "
               "raised -- so a shortfall here is a descriptor that silently vanished.";
        EXPECT_EQ(catalog.packs.size(), counts.packs)
            << shard << " holds " << counts.packs << " .kdp.json file(s) but the loader accepted "
            << catalog.packs.size() << ".";
    }
}

/// The parsed descriptors resolve into at least one complete engine.
///
/// Distinct from the case above and separately necessary: a catalog can be non-empty and
/// still resolve to nothing, because `resolveDescriptorSets` drops any engine whose
/// matcher / dispatch / metadata cross-references do not all resolve. A packer that
/// renumbered an id or dropped one file of a set would pass the parse check and fail here.
TEST(TestPackedDescriptorLoad, PackedDescriptorsResolveIntoCompleteSets)
{
    REQUIRE_PACKED_SHARDS(shards);

    for(const auto& shard : shards)
    {
        const auto sets = resolveDescriptorSets(loadDescriptorCatalog(shard));

        ASSERT_FALSE(sets.empty())
            << shard
            << " parsed, but no engine's references all resolved -- a descriptor set "
               "is incomplete or its cross-references disagree.";

        for(const auto& set : sets)
        {
            EXPECT_FALSE(set.engine.name.empty()) << shard << " resolved an unnamed engine.";
            EXPECT_FALSE(set.packs.empty())
                << shard << " resolved engine '" << set.engine.name << "' with no kernel packs.";
        }
    }
}

// ---------------------------------------------------------------------------
// `library` resolves -- the defect that shipped
// ---------------------------------------------------------------------------

/// Every packed kernel names an archive that is on disk at the path the RUNTIME computes.
///
/// The regression guard for the `library` defect, and the reason this is worth a C++ test
/// rather than another python one. The join below is not a reimplementation of the rule:
/// `originDirectory` is filled by the loader from the parent directory of the file that
/// defined the kernel, and `originDirectory / library` is verbatim what
/// `IngestorKernelCode.hpp`'s KPACK branch does before it opens the archive. Asserting the
/// result exists therefore asserts the runtime's own open would find it.
///
/// Nested descriptors are the case that broke, and a rocKE engine is authored nested, so
/// this is load-bearing rather than historical.
TEST(TestPackedDescriptorLoad, LibraryResolvesFromTheDescriptorThatDeclaredIt)
{
    REQUIRE_PACKED_SHARDS(shards);

    size_t kpackKernelsChecked = 0;

    forEachPackedKernel(shards, [&kpackKernelsChecked](const auto& /*shard*/, const auto& kernel) {
        for(const auto* source : launchSources(kernel))
        {
            if(source->kind != KernelSourceKind::KPACK)
            {
                continue;
            }

            ASSERT_FALSE(kernel.originDirectory.empty())
                << "kernel '" << kernel.name
                << "' came from a file but carries no originDirectory, so nothing can "
                   "anchor its library.";
            ASSERT_FALSE(source->library.empty())
                << "packed kernel '" << kernel.name << "' names no archive.";

            std::error_code ec;
            const auto resolved
                = std::filesystem::weakly_canonical(kernel.originDirectory / source->library, ec);

            EXPECT_TRUE(std::filesystem::exists(resolved))
                << "kernel '" << kernel.name << "' names an archive that is not there.\n"
                << "  originDirectory : " << kernel.originDirectory << "\n"
                << "  library         : " << source->library << "\n"
                << "  resolves to     : " << resolved << "\n"
                << "This is the runtime's own join (IngestorKernelCode.hpp, KPACK "
                   "branch). A library written relative to the arch root instead of to "
                   "this descriptor's own directory fails exactly here, and only for "
                   "nested descriptors.";

            ++kpackKernelsChecked;
        }
    });

    if(HasFatalFailure())
    {
        return;
    }

    // Without this the test passes vacuously on a tree that packed no kpack kernel at all
    // -- precisely the "green suite, zero coverage" outcome this file exists to end.
    EXPECT_GT(kpackKernelsChecked, 0U)
        << "no packed kernel used kind=kpack, so the library-resolution rule above was never "
           "exercised.";
}

/// Every packed kernel passes the runtime's OWN containment guard, and at least one of
/// them is nested deeply enough to have to climb out of its directory to reach the
/// archive.
///
/// The regression guard for the defect that made every production-packaged kernel
/// unloadable. The test above replays the resolution JOIN; this one exercises the
/// CONTAINMENT decision that sits beside it, which is a different rule and was the broken
/// one. Two behaviours were each individually correct and mutually incompatible:
///
///   * the packer writes a nested descriptor's library as `../../kpack/<archive>`,
///     because one archive ships per arch shard at the shard root;
///   * the guard rejected anything resolving outside the descriptor's own directory.
///
/// Nothing caught it because every tree any test packed was FLAT, so `..` never appeared
/// and the guard never fired. Hence the second assertion below: a shard with no nested
/// descriptor cannot exercise this, and silently proving nothing is the failure mode this
/// whole file exists to end.
///
/// buildIngestorKernelCode() cannot be called here -- it needs a device and a real archive
/// open. The containment rule is reproduced instead, reading the same two loader-populated
/// fields the guard reads, so a change to either field's meaning fails here rather than
/// only on hardware.
TEST(TestPackedDescriptorLoad, PackedKernelsSatisfyTheRuntimeContainmentGuard)
{
    REQUIRE_PACKED_SHARDS(shards);

    size_t nestedChecked = 0;

    forEachPackedKernel(shards, [&nestedChecked](const auto& /*shard*/, const auto& kernel) {
        for(const auto* source : launchSources(kernel))
        {
            if(source->kind != KernelSourceKind::KPACK)
            {
                continue;
            }

            ASSERT_FALSE(kernel.treeRoot.empty())
                << "kernel '" << kernel.name
                << "' carries no treeRoot, so the guard has nothing to anchor "
                   "containment on and would fall back to its own directory.";

            std::error_code ec;
            const auto origin = std::filesystem::weakly_canonical(kernel.originDirectory, ec);
            const auto boundary = std::filesystem::weakly_canonical(kernel.treeRoot, ec);
            const auto resolved = std::filesystem::weakly_canonical(origin / source->library, ec);

            // IngestorKernelCode.hpp, KPACK branch, verbatim.
            const std::string relative = resolved.lexically_relative(boundary).generic_string();
            const bool escapes
                = resolved != boundary && (relative.empty() || relative.rfind("..", 0) == 0);

            EXPECT_FALSE(escapes) << "kernel '" << kernel.name
                                  << "' would be REFUSED by the runtime containment guard.\n"
                                  << "  originDirectory : " << origin << "\n"
                                  << "  treeRoot        : " << boundary << "\n"
                                  << "  library         : " << source->library << "\n"
                                  << "  resolves to     : " << resolved;

            if(origin != boundary)
            {
                ++nestedChecked;
            }
        }
    });

    if(HasFatalFailure())
    {
        return;
    }

    EXPECT_GT(nestedChecked, 0U)
        << "every packed descriptor sat flat at its shard root, so none of them had to "
           "climb out to reach the archive and the containment rule was never exercised. "
           "A flat-only fixture is exactly what hid this defect: keep at least one "
           "descriptor in a child folder.";
}

/// A packed kernel carries every coordinate the kpack adapter needs to reach a code object.
///
/// `parseKernelSource` requires all five to be PRESENT, but a present empty string
/// satisfies that and then fails much later inside the archive reader, as a confusing
/// runtime error. Checking them here pins the packer to emitting usable values rather than
/// merely the right keys.
TEST(TestPackedDescriptorLoad, PackedKernelsCarryCompleteKpackCoordinates)
{
    REQUIRE_PACKED_SHARDS(shards);

    forEachPackedKernel(shards, [](const auto& /*shard*/, const auto& kernel) {
        for(const auto* source : launchSources(kernel))
        {
            if(source->kind != KernelSourceKind::KPACK)
            {
                continue;
            }

            EXPECT_FALSE(source->tocKey.empty())
                << "packed kernel '" << kernel.name
                << "' has an empty toc_key; the archive cannot be indexed.";
            EXPECT_FALSE(source->symbol.empty())
                << "packed kernel '" << kernel.name
                << "' has an empty symbol; no entry point can be resolved.";
            EXPECT_FALSE(source->sha256.empty())
                << "packed kernel '" << kernel.name << "' has an empty sha256.";
            // Every kernel this tree packs takes arguments, so an empty list here is
            // the extractor having found nothing rather than a nullary kernel -- the
            // failure that would otherwise read as a signature agreeing with anything.
            EXPECT_FALSE(source->signature.empty())
                << "packed kernel '" << kernel.name
                << "' has an empty signature; nothing would be compared at dispatch.";
        }
    });
}

/// Nothing in this root reaches the staged tree still claiming a producer-side source kind.
///
/// The packer's job on a `hip` or `rocke` UKD is to REPLACE the authored producer form --
/// `hip` with its source file, `rocke` with its builder and spec -- with the `kpack`
/// coordinates of the code object it produced. A descriptor arriving in this root still
/// naming a producer-side kind means the rewrite silently did not happen for it, and the
/// runtime would try to compile a source file the shard does not carry.
///
/// `embedded_source` is not producer-side in that sense: the packer passes it through as
/// authored, and those descriptors are staged in a different set.
///
/// This also pins the rocKE path specifically: `kind: "rocke"` is a PACKAGING vocabulary
/// the runtime loader does not accept at all, so a rocKE descriptor that failed to be
/// rewritten would not merely mis-dispatch, it would fail to parse.
TEST(TestPackedDescriptorLoad, PackedKernelsWereRewrittenToKpackForm)
{
    REQUIRE_PACKED_SHARDS(shards);

    forEachPackedKernel(shards, [](const auto& shard, const auto& kernel) {
        for(const auto* source : launchSources(kernel))
        {
            EXPECT_EQ(source->kind, KernelSourceKind::KPACK)
                << "kernel '" << kernel.name << "' in " << shard
                << " reached the staged tree without being rewritten to kpack form; it "
                   "still names a producer-side source the tree does not carry.";
        }
    });
}

// ---------------------------------------------------------------------------
// A packed SLOTTED engine (RFC 0021)
// ---------------------------------------------------------------------------

/// The slotted engine resolves, every slot keeps its own pack, KMD and dispatch
/// descriptor, and the kernels each pack carried reach their slot and no other.
///
/// An engine the loader drops would leave resolveDescriptorSets() without the name; a slot
/// whose pack the loader dropped is a warning, not an error, and leaves the engine
/// registered to decline every graph. Both are silent without these assertions.
TEST(TestPackedDescriptorLoad, PackedSlottedEngineKeepsEverySlotsPack)
{
    REQUIRE_PACKED_SLOTTED_SHARDS(shards);

    for(const auto& shard : shards)
    {
        expectLoaderAcceptedEveryFile(shard, 1U, 3U, 3U);

        const auto sets = resolveDescriptorSets(loadDescriptorCatalog(shard));
        const auto engine = std::find_if(sets.begin(), sets.end(), [](const auto& set) {
            return set.engine.name == SLOTTED_ENGINE;
        });
        ASSERT_NE(engine, sets.end()) << shard << " resolved no engine named " << SLOTTED_ENGINE;

        // Selection state is per slot, so the engine-level copies stay empty.
        EXPECT_TRUE(engine->packs.empty()) << shard;
        EXPECT_EQ(engine->engine.sharedWorkspaceToken, "ws.bytes") << shard;

        std::vector<std::string> names;
        std::vector<size_t> kernelCounts;
        std::vector<std::string> dispatches;
        for(const auto& slot : engine->slots)
        {
            names.push_back(slot.name);
            ASSERT_EQ(slot.packs.size(), 1U)
                << shard << ": slot '" << slot.name << "' lost its pack.";
            kernelCounts.push_back(slot.packs.front().kernels.size());
            dispatches.push_back(
                hipdnn_plugin_sdk::ingestor::toString(slot.packs.front().dispatchId));
            EXPECT_EQ(slot.packs.front().slot, slot.name) << shard;
        }
        std::sort(dispatches.begin(), dispatches.end());
        EXPECT_EQ(std::unique(dispatches.begin(), dispatches.end()), dispatches.end())
            << shard << ": two slots share a dispatch descriptor";
        EXPECT_EQ(names, (std::vector<std::string>{"prep", "dkdv", "dq"})) << shard;
        EXPECT_EQ(kernelCounts, (std::vector<size_t>{1, 2, 2})) << shard;

        ASSERT_EQ(engine->slots.size(), 3U);
        EXPECT_NE(engine->slots[0].schema.id, engine->slots[1].schema.id) << shard;
        EXPECT_NE(engine->slots[1].schema.id, engine->slots[2].schema.id) << shard;
        EXPECT_NE(engine->slots[0].schema.id, engine->slots[2].schema.id) << shard;
    }
}

/// The models the slotted fixture ships are bound, not merely present: each slot ranker
/// and the engine estimate survive the loader's provenance check against the packs the
/// packer wrote.
///
/// A model that fails it is disabled with an error and the engine stays registered with
/// declared-order fallback, so the engine and its packs can look healthy while every
/// ranker is gone. `unavailable*` lists what the loader refused by name.
TEST(TestPackedDescriptorLoad, PackedSlottedModelsAreBoundToTheirSlots)
{
    REQUIRE_PACKED_SLOTTED_SHARDS(shards);

    for(const auto& shard : shards)
    {
        const auto sets = resolveDescriptorSets(loadDescriptorCatalog(shard));
        const auto engine = std::find_if(sets.begin(), sets.end(), [](const auto& set) {
            return set.engine.name == SLOTTED_ENGINE;
        });
        ASSERT_NE(engine, sets.end()) << shard << " resolved no engine named " << SLOTTED_ENGINE;
        ASSERT_EQ(engine->slots.size(), 3U) << shard;

        // `prep` ships no ranker; `dkdv` and `dq` ship a metric-less one each.
        EXPECT_FALSE(engine->slots[0].heuristic.has_value()) << shard;
        for(size_t index = 1; index < engine->slots.size(); ++index)
        {
            const auto& slot = engine->slots[index];
            EXPECT_TRUE(slot.unavailableHeuristicArches.empty())
                << shard << ": the ranker of slot '" << slot.name << "' was refused.";
            ASSERT_TRUE(slot.heuristic.has_value())
                << shard << ": slot '" << slot.name << "' has no usable ranker.";
            ASSERT_TRUE(slot.heuristic->trainedAgainst.has_value()) << shard;
            EXPECT_EQ(slot.heuristic->trainedAgainst->slot, slot.name) << shard;
            EXPECT_EQ(slot.heuristic->trainedAgainst->kmd.id, slot.schema.id) << shard;
            EXPECT_TRUE(slot.heuristic->score.metric.empty()) << shard;
            ASSERT_EQ(slot.heuristicsByMetric.count(""), 1U) << shard;
            EXPECT_EQ(slot.heuristicsByMetric.at("").count("default"), 1U) << shard;
        }

        EXPECT_TRUE(engine->unavailableEnginePredictionArches.empty())
            << shard << ": the engine estimate was refused.";
        ASSERT_EQ(engine->enginePredictionsByMetric.count("time"), 1U)
            << shard << ": the engine estimate is not bound.";
        const auto& estimate = engine->enginePredictionsByMetric.at("time");
        ASSERT_EQ(estimate.count("default"), 1U) << shard;
        const auto& trained = estimate.at("default").trainedAgainst;
        ASSERT_TRUE(trained.has_value()) << shard;
        ASSERT_EQ(trained->kmds.size(), engine->slots.size()) << shard;
        for(size_t index = 0; index < engine->slots.size(); ++index)
        {
            EXPECT_EQ(trained->kmds[index].slot, engine->slots[index].name) << shard;
            EXPECT_EQ(trained->kmds[index].kmd.id, engine->slots[index].schema.id) << shard;
        }
    }
}

// ---------------------------------------------------------------------------
// Consumer-shaped slotted fixtures: authored hsaco slots, and multi-launch rocKE slots
// ---------------------------------------------------------------------------

/// The hsaco fixture loads whole: every slot keeps its pack, `dq` ships its gfx950 pack
/// alone (its generic-target pack belongs to the generic's members), and `dq`'s kernel
/// carries the field naming a second symbol of its code object.
TEST(TestPackedDescriptorLoad, PackedSlottedHsacoEngineLoadsWithEverySlotPopulated)
{
    REQUIRE_PACKED_SLOTTED_HSACO_SHARDS(shards);

    for(const auto& shard : shards)
    {
        expectLoaderAcceptedEveryFile(shard, 1U, 3U, 0U);

        const auto sets = resolveDescriptorSets(loadDescriptorCatalog(shard));
        const auto* engine = findEngine(sets, SLOTTED_HSACO_ENGINE);
        ASSERT_NE(engine, nullptr) << shard << " resolved no engine named " << SLOTTED_HSACO_ENGINE;
        ASSERT_EQ(engine->slots.size(), 3U) << shard;

        const std::vector<std::string> names{"prep", "dkdv", "dq"};
        const std::vector<std::string> symbols{
            "HsacoFixtureScale", "HsacoFixtureAdd", "HsacoFixtureAdd"};
        for(size_t index = 0; index < names.size(); ++index)
        {
            const auto& slot = engine->slots[index];
            EXPECT_EQ(slot.name, names[index]) << shard;
            ASSERT_EQ(slot.packs.size(), 1U) << shard << ": slot '" << slot.name << "'";
            ASSERT_EQ(slot.packs.front().kernels.size(), 1U) << shard;
            const auto& kernel = slot.packs.front().kernels.front();
            EXPECT_TRUE(kernel.sources.empty()) << "an authored hsaco kernel is one launch";
            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << shard;
            EXPECT_EQ(kernel.source.symbol, symbols[index]) << shard;
        }

        const auto& dq = engine->slots[2];
        EXPECT_TRUE(std::any_of(dq.schema.fields.begin(),
                                dq.schema.fields.end(),
                                [](const auto& field) { return field.name == "reduce_symbol"; }))
            << shard;
        const auto& dqKernel = dq.packs.front().kernels.front();
        const auto second = dqKernel.metadata.find("reduce_symbol");
        ASSERT_NE(second, dqKernel.metadata.end()) << shard;
        ASSERT_TRUE(std::holds_alternative<std::string>(second->second)) << shard;
        EXPECT_EQ(std::get<std::string>(second->second), "HsacoFixtureScale") << shard;
        EXPECT_NE(std::get<std::string>(second->second), dqKernel.source.symbol) << shard;
    }

    expectPackedLaunchesAreUsable(shards);
}

/// The multi-launch fixture loads whole: the `dkdv` slot's kernel keeps both launches as
/// KPACK `sources` with their own symbols, and `prep` and `dq` stay single-launch.
TEST(TestPackedDescriptorLoad, PackedSlottedMultiLaunchEngineKeepsEachSlotsLaunches)
{
    REQUIRE_PACKED_SLOTTED_MULTI_LAUNCH_SHARDS(shards);

    for(const auto& shard : shards)
    {
        expectLoaderAcceptedEveryFile(shard, 1U, 3U, 0U);

        const auto sets = resolveDescriptorSets(loadDescriptorCatalog(shard));
        const auto* engine = findEngine(sets, SLOTTED_MULTI_LAUNCH_ENGINE);
        ASSERT_NE(engine, nullptr)
            << shard << " resolved no engine named " << SLOTTED_MULTI_LAUNCH_ENGINE;
        ASSERT_EQ(engine->slots.size(), 3U) << shard;

        for(const auto& slot : engine->slots)
        {
            ASSERT_EQ(slot.packs.size(), 1U) << shard << ": slot '" << slot.name << "'";
            ASSERT_EQ(slot.packs.front().kernels.size(), 1U) << shard;

            const auto& kernel = slot.packs.front().kernels.front();
            if(slot.name == "dkdv")
            {
                ASSERT_EQ(kernel.sources.size(), 2U) << shard;
                ASSERT_EQ(kernel.sources.count("decode"), 1U) << shard;
                ASSERT_EQ(kernel.sources.count("merge"), 1U) << shard;
                EXPECT_NE(kernel.sources.at("decode").symbol, kernel.sources.at("merge").symbol)
                    << shard;
            }
            else
            {
                EXPECT_TRUE(kernel.sources.empty()) << shard << ": slot '" << slot.name << "'";
                EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << shard;
            }
        }
    }

    expectPackedLaunchesAreUsable(shards);
}

} // namespace hip_kernel_provider::ingestor_seam

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
