// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "compilation/KpackProgram.hpp"

namespace hip_kernel_provider::compilation
{
namespace
{

using hipdnn_plugin_sdk::HipdnnPluginException;
using hipdnn_test_sdk::utilities::ScopedDirectory;

/// rocm-kpack's own test archive, path supplied by CMake from ROCM_KPACK_SOURCE_DIR.
/// It holds gfx1100 and gfx1101 binaries under the toc keys "lib/libhip.so#0" and
/// "bin/hiptest#0". Used rather than a hand-forged file so the reader under test is
/// the pinned reader meeting an archive it actually accepts. The parse-level cases
/// need a real *container*, not a matching *device*; the device cases read this
/// build's own packed archive -- see PACKED_DESCRIPTOR_ROOT.
constexpr const char* REAL_ARCHIVE = HIPDNN_TEST_KPACK_ARCHIVE;
constexpr const char* ARCHIVE_ARCH = "gfx1100";
constexpr const char* ARCHIVE_TOC_KEY = "lib/libhip.so#0";

/// Where this build stages the descriptors it packed, one subdirectory per arch. Same
/// value main.cpp points the binary at.
constexpr const char* PACKED_DESCRIPTOR_ROOT = HIPDNN_TEST_DESCRIPTOR_DIR;

/// The two descriptors the packaged pointwise fixture stages, one per block size. Their
/// archive and toc_key are read out of the built files rather than written here: a copy
/// would silently decouple this test from the artifact it exists to read.
constexpr const char* PACKED_KDP_DESCRIPTOR = "packed_pointwise_add.kdp.json";
constexpr const char* PACKED_UKD_DESCRIPTOR = "packed_pointwise_add_b256.ukd.json";

/// The two entry points the packaged fixture's translation unit exports. Only the first
/// is named by a descriptor; the second exists so one code object carries two symbols.
/// It must match integration_tests/kernel_ingestor_engine/fixtures/packaged/PointwiseAdd.cpp.
constexpr const char* PACKED_SYMBOL = "PointwiseAdd";
constexpr const char* PACKED_SECOND_SYMBOL = "PointwiseAddSecondSymbol";

/// A symbol no code object exports, used to reach the resolution-failure path.
constexpr const char* ABSENT_SYMBOL = "there_is_no_such_symbol";

struct PackagedKernelSource
{
    std::filesystem::path archive;
    std::string tocKey;
};

/// The bare arch of device 0 and the directory this build packed for it. `directory` is
/// left empty when nothing was packed for that arch -- environmental, not a broken build.
///
/// hipGetDeviceProperties reports feature flags on some configurations ("gfx1152:xnack-")
/// while the packager uses the bare name, so everything past here uses the stripped form.
///
/// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
void findPackagedDirectory(std::string& arch, std::filesystem::path& directory)
{
    hipDeviceProp_t properties{};
    ASSERT_EQ(hipGetDeviceProperties(&properties, 0), hipSuccess);

    const std::string reported = properties.gcnArchName;
    arch = reported.substr(0, reported.find(':'));

    const std::filesystem::path candidate = std::filesystem::path(PACKED_DESCRIPTOR_ROOT) / arch;
    directory = std::filesystem::is_directory(candidate) ? candidate : std::filesystem::path{};
}

/// Reads `kernel_source` out of a built descriptor. A .kdp.json nests it under its first
/// inline kernel descriptor; a .ukd.json carries it at the top level. Parsed directly
/// rather than through DescriptorLoader, whose contract the integration tier covers.
///
/// Found by RECURSIVE search rather than a join on the arch root: the packer preserves
/// each descriptor's authored subpath, so a descriptor sits wherever its source root put
/// it. Searching by filename keeps this test indifferent to that depth, which is the
/// point -- a flat join here is what made the whole suite blind to nesting.
///
/// Asserts rather than skips -- the per-arch directory exists by the time this is
/// called, so anything missing inside it is a broken build. Call through
/// ASSERT_NO_FATAL_FAILURE.
void readPackagedKernelSource(const std::filesystem::path& directory,
                              const std::string& descriptorFile,
                              PackagedKernelSource& out)
{
    std::filesystem::path descriptor;
    std::error_code walkError;
    for(const auto& entry : std::filesystem::recursive_directory_iterator(directory, walkError))
    {
        if(entry.is_regular_file() && entry.path().filename() == descriptorFile)
        {
            descriptor = entry.path();
            break;
        }
    }
    ASSERT_FALSE(descriptor.empty()) << "the packaged descriptor is missing anywhere under "
                                     << directory << ": " << descriptorFile;

    std::ifstream in(descriptor);
    ASSERT_TRUE(in.good()) << "could not open " << descriptor;

    nlohmann::json document;
    ASSERT_NO_THROW(document = nlohmann::json::parse(in)) << descriptor;

    const nlohmann::json& kernel
        = document.contains("kernelDescriptors") ? document["kernelDescriptors"][0] : document;
    ASSERT_TRUE(kernel.contains("kernel_source")) << descriptor;

    const nlohmann::json& source = kernel["kernel_source"];
    ASSERT_TRUE(source.contains("toc_key")) << descriptor;
    ASSERT_TRUE(source.contains("library")) << descriptor;

    out.tocKey = source["toc_key"].get<std::string>();
    // `library` is relative to the directory holding the descriptor that declared it --
    // the same anchoring KernelDefinition::originDirectory describes. That directory is
    // the descriptor's OWN parent, not the arch root, so a nested descriptor resolves
    // through the `..` segments the packer wrote.
    out.archive = descriptor.parent_path() / source["library"].get<std::string>();
    ASSERT_TRUE(std::filesystem::exists(out.archive))
        << descriptor << " names an archive that is not on disk: " << out.archive;
}

/// What a descriptor-shaped label looks like where the loader is really called.
const std::string& descriptorLabel()
{
    static const std::string s_label = hipdnn_plugin_sdk::ingestor::describeDescriptor(
        "kernel", "pointwise_add_f32_kpack", hipdnn_plugin_sdk::ingestor::DescriptorId{});
    return s_label;
}

/// Every case gets its own cache: a shared one would let an earlier case's successful
/// load answer a later case's lookup and hide the failure it is asserting.
class TestKpackKernelLoader : public ::testing::Test
{
protected:
    KpackModuleCache _cache;
    KpackKernelLoader _loader{_cache};
};

TEST_F(TestKpackKernelLoader, ReportsAMissingArchive)
{
    const std::filesystem::path absent
        = std::filesystem::temp_directory_path() / "hipdnn-kpack-there-is-no-archive-here.kpack";
    ASSERT_FALSE(std::filesystem::exists(absent));

    try
    {
        _loader.load(absent, ARCHIVE_TOC_KEY, ARCHIVE_ARCH, "PointwiseAdd", descriptorLabel());
        FAIL() << "expected a missing archive to be reported";
    }
    catch(const HipdnnPluginException& error)
    {
        const std::string what = error.what();
        EXPECT_NE(what.find(descriptorLabel()), std::string::npos) << what;
        EXPECT_NE(what.find("PointwiseAdd"), std::string::npos) << what;
        EXPECT_NE(what.find("does not exist"), std::string::npos) << what;
    }
}

TEST_F(TestKpackKernelLoader, ReportsACorruptArchive)
{
    const ScopedDirectory scratch(std::filesystem::temp_directory_path()
                                  / "hipdnn-kpack-corrupt-archive");
    const std::filesystem::path garbage = scratch.path() / "corrupt.kpack";
    {
        std::ofstream out(garbage, std::ios::binary);
        // Not "KPAK": the reader rejects this at the magic, before any arch or entry
        // lookup, which is the stage this case is pinning.
        out << "this is not a kpack archive, it is a sentence";
    }
    ASSERT_TRUE(std::filesystem::exists(garbage));

    try
    {
        _loader.load(garbage, ARCHIVE_TOC_KEY, ARCHIVE_ARCH, "PointwiseAdd", descriptorLabel());
        FAIL() << "expected an unreadable archive to be reported";
    }
    catch(const HipdnnPluginException& error)
    {
        const std::string what = error.what();
        EXPECT_NE(what.find(descriptorLabel()), std::string::npos) << what;
        EXPECT_NE(what.find("PointwiseAdd"), std::string::npos) << what;
        EXPECT_NE(what.find("could not be read"), std::string::npos) << what;
        // Distinct from the missing-archive wording: "not there" and "there but
        // unusable" are told apart.
        EXPECT_EQ(what.find("does not exist"), std::string::npos) << what;
    }
}

TEST_F(TestKpackKernelLoader, ReportsAnArchMismatch)
{
    ASSERT_TRUE(std::filesystem::exists(REAL_ARCHIVE))
        << "the kpack test asset named at configure time is missing: " << REAL_ARCHIVE;

    try
    {
        _loader.load(REAL_ARCHIVE, ARCHIVE_TOC_KEY, "gfx942", "PointwiseAdd", descriptorLabel());
        FAIL() << "expected an arch mismatch to be reported";
    }
    catch(const HipdnnPluginException& error)
    {
        const std::string what = error.what();
        EXPECT_NE(what.find(descriptorLabel()), std::string::npos) << what;
        EXPECT_NE(what.find("PointwiseAdd"), std::string::npos) << what;
        // Names the device arch and what the archive holds, so packer-vs-machine is
        // visible.
        EXPECT_NE(what.find("gfx942"), std::string::npos) << what;
        EXPECT_NE(what.find("gfx1100"), std::string::npos) << what;
        EXPECT_NE(what.find("gfx1101"), std::string::npos) << what;
    }
}

TEST_F(TestKpackKernelLoader, ReportsAMissingTocKey)
{
    ASSERT_TRUE(std::filesystem::exists(REAL_ARCHIVE))
        << "the kpack test asset named at configure time is missing: " << REAL_ARCHIVE;

    try
    {
        _loader.load(
            REAL_ARCHIVE, "no/such/entry#7", ARCHIVE_ARCH, "PointwiseAdd", descriptorLabel());
        FAIL() << "expected a missing toc_key to be reported";
    }
    catch(const HipdnnPluginException& error)
    {
        const std::string what = error.what();
        EXPECT_NE(what.find(descriptorLabel()), std::string::npos) << what;
        EXPECT_NE(what.find("PointwiseAdd"), std::string::npos) << what;
        EXPECT_NE(what.find("no/such/entry#7"), std::string::npos) << what;
        // The fifth failure, distinct from a missing symbol on purpose: it is the
        // signature of packer/descriptor skew, not of a mis-spelled entry point.
        EXPECT_NE(what.find("no entry for toc_key"), std::string::npos) << what;
        EXPECT_EQ(what.find("is not present in the loaded module"), std::string::npos) << what;
    }
}

TEST_F(TestKpackKernelLoader, ReportsAMissingSymbol)
{
    SKIP_IF_NO_DEVICES();

    std::string arch;
    std::filesystem::path packaged;
    ASSERT_NO_FATAL_FAILURE(findPackagedDirectory(arch, packaged));
    if(packaged.empty())
    {
        GTEST_SKIP() << "nothing was packaged for this device (" << arch
                     << "): " << std::filesystem::path(PACKED_DESCRIPTOR_ROOT) / arch
                     << " does not exist. Environmental -- the build packs per arch and this "
                        "device is outside GPU_TARGETS.";
    }

    PackagedKernelSource packed;
    ASSERT_NO_FATAL_FAILURE(readPackagedKernelSource(packaged, PACKED_KDP_DESCRIPTOR, packed));

    // The load itself succeeds: the archive holds this arch and this toc_key. Symbol
    // resolution is a later, separate stage -- KpackKernelLoader::load never looks at the
    // symbol -- so the failure this case is after is raised by KpackProgram::getKernel
    // against a module HIP has accepted.
    const auto program
        = _loader.load(packed.archive, packed.tocKey, arch, ABSENT_SYMBOL, descriptorLabel());
    ASSERT_NE(program, nullptr);

    try
    {
        program->getKernel(ABSENT_SYMBOL);
        FAIL() << "expected a missing symbol to be reported";
    }
    catch(const HipdnnPluginException& error)
    {
        const std::string what = error.what();
        EXPECT_NE(what.find(descriptorLabel()), std::string::npos) << what;
        EXPECT_NE(what.find(ABSENT_SYMBOL), std::string::npos) << what;
        // The blob was found, decompressed and loaded; only the entry point is absent.
        EXPECT_NE(what.find("is not present in the loaded module"), std::string::npos) << what;
        EXPECT_EQ(what.find("no entry for toc_key"), std::string::npos) << what;
    }

    // Clear the HIP error state left by the intentional hipModuleGetFunction failure,
    // or the HipErrorHandler listener fails this test for it.
    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

TEST_F(TestKpackKernelLoader, TwoSymbolsResolveAgainstOneModule)
{
    SKIP_IF_NO_DEVICES();

    std::string arch;
    std::filesystem::path packaged;
    ASSERT_NO_FATAL_FAILURE(findPackagedDirectory(arch, packaged));
    if(packaged.empty())
    {
        GTEST_SKIP() << "nothing was packaged for this device (" << arch
                     << "): " << std::filesystem::path(PACKED_DESCRIPTOR_ROOT) / arch
                     << " does not exist. Environmental -- the build packs per arch and this "
                        "device is outside GPU_TARGETS.";
    }

    // One descriptor, so one toc_key, so one blob. The fixture's translation unit exports
    // two entry points into it, which is the only way two symbols can share a key.
    PackagedKernelSource packed;
    ASSERT_NO_FATAL_FAILURE(readPackagedKernelSource(packaged, PACKED_UKD_DESCRIPTOR, packed));

    const size_t before = _cache.size();

    // Both symbols resolve...
    const auto first
        = _loader.load(packed.archive, packed.tocKey, arch, PACKED_SYMBOL, descriptorLabel());
    const auto second = _loader.load(
        packed.archive, packed.tocKey, arch, PACKED_SECOND_SYMBOL, descriptorLabel());
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first->getKernel(PACKED_SYMBOL), nullptr);
    EXPECT_NE(second->getKernel(PACKED_SECOND_SYMBOL), nullptr);

    // ...against one and the same hipModule_t. Measured as a delta rather than an
    // absolute count so the assertion still means what it says if this cache ever
    // outlives one case.
    const auto* firstKpack = dynamic_cast<const KpackProgram*>(first.get());
    const auto* secondKpack = dynamic_cast<const KpackProgram*>(second.get());
    ASSERT_NE(firstKpack, nullptr);
    ASSERT_NE(secondKpack, nullptr);
    EXPECT_NE(firstKpack->module(), nullptr);
    EXPECT_EQ(firstKpack->module(), secondKpack->module());
    EXPECT_EQ(_cache.size(), before + 1);
}

} // namespace
} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
