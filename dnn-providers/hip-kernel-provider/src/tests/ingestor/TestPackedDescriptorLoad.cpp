// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

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

/// Where this build stages what it packed, one subdirectory per arch. The same define
/// `TestKpackKernelLoader.cpp` reads; CMake sets it from `HIPDNN_DESCRIPTOR_BUILD_DIR`.
constexpr const char* PACKED_DESCRIPTOR_ROOT = HIPDNN_TEST_DESCRIPTOR_DIR;

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
    const std::filesystem::path root(PACKED_DESCRIPTOR_ROOT);
    if(!std::filesystem::is_directory(root, ec))
    {
        return shards;
    }

    for(const auto& entry : std::filesystem::directory_iterator(root, ec))
    {
        // A shard is a directory holding a kpack/ sibling; the flat per-engine descriptor
        // folders staged from the source tree are not packer output and are not in scope.
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
#define REQUIRE_PACKED_SHARDS(shards)                                                      \
    const auto shards = packedArchShards();                                                \
    if((shards).empty())                                                                   \
    {                                                                                      \
        GTEST_SKIP() << "no packed arch shard under " << PACKED_DESCRIPTOR_ROOT            \
                     << " -- the packaging rule did not run. Configure with "              \
                        "-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON, a discoverable hipcc, and a " \
                        "HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT.";                       \
    }                                                                                      \
    static_assert(true, "swallow the trailing semicolon")

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
        size_t engineFiles = 0;
        size_t packFiles = 0;

        std::error_code ec;
        for(const auto& entry : std::filesystem::recursive_directory_iterator(shard, ec))
        {
            const auto name = entry.path().filename().string();
            if(name.size() > 9 && name.compare(name.size() - 9, 9, ".ued.json") == 0)
            {
                ++engineFiles;
            }
            else if(name.size() > 9 && name.compare(name.size() - 9, 9, ".kdp.json") == 0)
            {
                ++packFiles;
            }
        }

        const DescriptorCatalog catalog = loadDescriptorCatalog(shard);

        EXPECT_EQ(catalog.engines.size(), engineFiles)
            << shard << " holds " << engineFiles << " .ued.json file(s) but the loader "
            << "accepted " << catalog.engines.size()
            << ". A descriptor the loader rejects is logged at ERROR and SKIPPED, never "
               "raised -- so a shortfall here is a descriptor that silently vanished.";
        EXPECT_EQ(catalog.packs.size(), packFiles)
            << shard << " holds " << packFiles << " .kdp.json file(s) but the loader accepted "
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

    for(const auto& shard : shards)
    {
        for(const auto& set : resolveDescriptorSets(loadDescriptorCatalog(shard)))
        {
            for(const auto& pack : set.packs)
            {
                for(const auto& kernel : pack.kernels)
                {
                    if(kernel.source.kind != KernelSourceKind::KPACK)
                    {
                        continue;
                    }

                    ASSERT_FALSE(kernel.originDirectory.empty())
                        << "kernel '" << kernel.name
                        << "' came from a file but carries no originDirectory, so nothing can "
                           "anchor its library.";
                    ASSERT_FALSE(kernel.source.library.empty())
                        << "packed kernel '" << kernel.name << "' names no archive.";

                    std::error_code ec;
                    const auto resolved = std::filesystem::weakly_canonical(
                        kernel.originDirectory / kernel.source.library, ec);

                    EXPECT_TRUE(std::filesystem::exists(resolved))
                        << "kernel '" << kernel.name << "' names an archive that is not there.\n"
                        << "  originDirectory : " << kernel.originDirectory << "\n"
                        << "  library         : " << kernel.source.library << "\n"
                        << "  resolves to     : " << resolved << "\n"
                        << "This is the runtime's own join (IngestorKernelCode.hpp, KPACK "
                           "branch). A library written relative to the arch root instead of to "
                           "this descriptor's own directory fails exactly here, and only for "
                           "nested descriptors.";

                    ++kpackKernelsChecked;
                }
            }
        }
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
/// buildIngestorKernelCode() cannot be called here -- it needs a device, a compiler and a
/// real archive open. The containment rule is reproduced instead, reading the same two
/// loader-populated fields the guard reads, so a change to either field's meaning fails
/// here rather than only on hardware.
TEST(TestPackedDescriptorLoad, PackedKernelsSatisfyTheRuntimeContainmentGuard)
{
    REQUIRE_PACKED_SHARDS(shards);

    size_t nestedChecked = 0;

    for(const auto& shard : shards)
    {
        for(const auto& set : resolveDescriptorSets(loadDescriptorCatalog(shard)))
        {
            for(const auto& pack : set.packs)
            {
                for(const auto& kernel : pack.kernels)
                {
                    if(kernel.source.kind != KernelSourceKind::KPACK)
                    {
                        continue;
                    }

                    ASSERT_FALSE(kernel.treeRoot.empty())
                        << "kernel '" << kernel.name
                        << "' carries no treeRoot, so the guard has nothing to anchor "
                           "containment on and would fall back to its own directory.";

                    std::error_code ec;
                    const auto origin
                        = std::filesystem::weakly_canonical(kernel.originDirectory, ec);
                    const auto boundary = std::filesystem::weakly_canonical(kernel.treeRoot, ec);
                    const auto resolved
                        = std::filesystem::weakly_canonical(origin / kernel.source.library, ec);

                    // IngestorKernelCode.hpp, KPACK branch, verbatim.
                    const std::string relative
                        = resolved.lexically_relative(boundary).generic_string();
                    const bool escapes = resolved != boundary
                                         && (relative.empty() || relative.rfind("..", 0) == 0);

                    EXPECT_FALSE(escapes)
                        << "kernel '" << kernel.name
                        << "' would be REFUSED by the runtime containment guard.\n"
                        << "  originDirectory : " << origin << "\n"
                        << "  treeRoot        : " << boundary << "\n"
                        << "  library         : " << kernel.source.library << "\n"
                        << "  resolves to     : " << resolved;

                    if(origin != boundary)
                    {
                        ++nestedChecked;
                    }
                }
            }
        }
    }

    EXPECT_GT(nestedChecked, 0U)
        << "every packed descriptor sat flat at its shard root, so none of them had to "
           "climb out to reach the archive and the containment rule was never exercised. "
           "A flat-only fixture is exactly what hid this defect: keep at least one "
           "descriptor in a child folder.";
}

/// A packed kernel carries every coordinate the kpack adapter needs to reach a code object.
///
/// `parseKernelSource` requires all four to be PRESENT, but a present empty string
/// satisfies that and then fails much later inside the archive reader, as a confusing
/// runtime error. Checking them here pins the packer to emitting usable values rather than
/// merely the right keys.
TEST(TestPackedDescriptorLoad, PackedKernelsCarryCompleteKpackCoordinates)
{
    REQUIRE_PACKED_SHARDS(shards);

    for(const auto& shard : shards)
    {
        for(const auto& set : resolveDescriptorSets(loadDescriptorCatalog(shard)))
        {
            for(const auto& pack : set.packs)
            {
                for(const auto& kernel : pack.kernels)
                {
                    if(kernel.source.kind != KernelSourceKind::KPACK)
                    {
                        continue;
                    }

                    EXPECT_FALSE(kernel.source.tocKey.empty())
                        << "packed kernel '" << kernel.name
                        << "' has an empty toc_key; the archive cannot be indexed.";
                    EXPECT_FALSE(kernel.source.symbol.empty())
                        << "packed kernel '" << kernel.name
                        << "' has an empty symbol; no entry point can be resolved.";
                    EXPECT_FALSE(kernel.source.sha256.empty())
                        << "packed kernel '" << kernel.name << "' has an empty sha256.";
                }
            }
        }
    }
}

/// Nothing reaches the staged tree still claiming a producer-side source kind.
///
/// The packer's whole job on a UKD is to REPLACE the authored producer form -- `hip` with
/// its source file, `rocke` with its builder and spec -- with the `kpack` coordinates of
/// the code object it produced. A descriptor that arrives still naming `embedded_source`
/// means the rewrite silently did not happen for it, and the runtime would try to compile
/// a source file the shard does not carry.
///
/// This also pins the rocKE path specifically: `kind: "rocke"` is a PACKAGING vocabulary
/// the runtime loader does not accept at all, so a rocKE descriptor that failed to be
/// rewritten would not merely mis-dispatch, it would fail to parse.
TEST(TestPackedDescriptorLoad, PackedKernelsWereRewrittenToKpackForm)
{
    REQUIRE_PACKED_SHARDS(shards);

    for(const auto& shard : shards)
    {
        for(const auto& set : resolveDescriptorSets(loadDescriptorCatalog(shard)))
        {
            for(const auto& pack : set.packs)
            {
                for(const auto& kernel : pack.kernels)
                {
                    EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK)
                        << "kernel '" << kernel.name << "' in " << shard
                        << " reached the staged tree without being rewritten to kpack form; it "
                           "still names a producer-side source the tree does not carry.";
                }
            }
        }
    }
}

} // namespace hip_kernel_provider::ingestor_seam

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
