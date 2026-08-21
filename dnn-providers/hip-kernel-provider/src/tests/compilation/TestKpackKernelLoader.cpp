// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

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
/// the pinned reader meeting an archive it actually accepts.
constexpr const char* REAL_ARCHIVE = HIPDNN_TEST_KPACK_ARCHIVE;
constexpr const char* ARCHIVE_ARCH = "gfx1100";
constexpr const char* ARCHIVE_TOC_KEY = "lib/libhip.so#0";

/// What a descriptor-shaped label looks like where the loader is really called; the
/// assertions below check the message carries it through verbatim.
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
    KpackModuleCache cache;
    KpackKernelLoader loader{cache};
};

TEST_F(TestKpackKernelLoader, ReportsAMissingArchive)
{
    const std::filesystem::path absent
        = std::filesystem::temp_directory_path() / "hipdnn-kpack-there-is-no-archive-here.kpack";
    ASSERT_FALSE(std::filesystem::exists(absent));

    try
    {
        loader.load(absent, ARCHIVE_TOC_KEY, ARCHIVE_ARCH, "PointwiseAdd", descriptorLabel());
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
        loader.load(garbage, ARCHIVE_TOC_KEY, ARCHIVE_ARCH, "PointwiseAdd", descriptorLabel());
        FAIL() << "expected an unreadable archive to be reported";
    }
    catch(const HipdnnPluginException& error)
    {
        const std::string what = error.what();
        EXPECT_NE(what.find(descriptorLabel()), std::string::npos) << what;
        EXPECT_NE(what.find("PointwiseAdd"), std::string::npos) << what;
        EXPECT_NE(what.find("could not be read"), std::string::npos) << what;
        // Distinct from the missing-archive wording: AC #7 wants "not there" and
        // "there but unusable" told apart.
        EXPECT_EQ(what.find("does not exist"), std::string::npos) << what;
    }
}

TEST_F(TestKpackKernelLoader, ReportsAnArchMismatch)
{
    ASSERT_TRUE(std::filesystem::exists(REAL_ARCHIVE))
        << "the kpack test asset named at configure time is missing: " << REAL_ARCHIVE;

    try
    {
        loader.load(REAL_ARCHIVE, ARCHIVE_TOC_KEY, "gfx942", "PointwiseAdd", descriptorLabel());
        FAIL() << "expected an arch mismatch to be reported";
    }
    catch(const HipdnnPluginException& error)
    {
        const std::string what = error.what();
        EXPECT_NE(what.find(descriptorLabel()), std::string::npos) << what;
        EXPECT_NE(what.find("PointwiseAdd"), std::string::npos) << what;
        // Names the device arch and what the archive does hold, so a reader can tell
        // at a glance whether the packer or the machine is the odd one out.
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
        loader.load(
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

    // The archive's binaries are gfx1100/gfx1101 test payloads, so this only gets past
    // the arch gate on such a device; elsewhere the arch mismatch is the honest result
    // and the symbol stage is unreachable. Both are asserted as "an error naming the
    // descriptor and the symbol", which is what AC #7 requires either way.
    try
    {
        loader.load(REAL_ARCHIVE,
                    ARCHIVE_TOC_KEY,
                    ARCHIVE_ARCH,
                    "there_is_no_such_symbol",
                    descriptorLabel());
        FAIL() << "expected a missing symbol to be reported";
    }
    catch(const HipdnnPluginException& error)
    {
        const std::string what = error.what();
        EXPECT_NE(what.find(descriptorLabel()), std::string::npos) << what;
        EXPECT_NE(what.find("there_is_no_such_symbol"), std::string::npos) << what;
    }

    // Clear the HIP error state left by the intentional hipModuleGetFunction failure,
    // or the HipErrorHandler listener fails this test for it.
    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

TEST_F(TestKpackKernelLoader, LoadsAModuleAndResolvesTheSymbol)
{
    SKIP_IF_NO_DEVICES();

    std::string deviceArch;
    {
        hipDeviceProp_t properties{};
        ASSERT_EQ(hipGetDeviceProperties(&properties, 0), hipSuccess);
        deviceArch = properties.gcnArchName;
    }
    if(deviceArch.rfind(ARCHIVE_ARCH, 0) != 0)
    {
        GTEST_SKIP() << "the kpack test asset holds gfx1100/gfx1101 binaries; this device is "
                     << deviceArch;
    }

    // TEST_APP_KERNEL___B is one of the symbols the asset's binaries export.
    const auto program = loader.load(
        REAL_ARCHIVE, ARCHIVE_TOC_KEY, deviceArch, "TEST_APP_KERNEL___B", descriptorLabel());
    ASSERT_NE(program, nullptr);
    EXPECT_NE(program->getKernel("TEST_APP_KERNEL___B"), nullptr);
}

TEST_F(TestKpackKernelLoader, TwoSymbolsResolveAgainstOneModule)
{
    SKIP_IF_NO_DEVICES();

    std::string deviceArch;
    {
        hipDeviceProp_t properties{};
        ASSERT_EQ(hipGetDeviceProperties(&properties, 0), hipSuccess);
        deviceArch = properties.gcnArchName;
    }
    if(deviceArch.rfind(ARCHIVE_ARCH, 0) != 0)
    {
        GTEST_SKIP() << "the kpack test asset holds gfx1100/gfx1101 binaries; this device is "
                     << deviceArch;
    }

    // Both symbols resolve...
    const auto first = loader.load(
        REAL_ARCHIVE, ARCHIVE_TOC_KEY, deviceArch, "HIP_KERNEL_GFX1100_AB", descriptorLabel());
    const auto second = loader.load(
        REAL_ARCHIVE, ARCHIVE_TOC_KEY, deviceArch, "TEST_APP_KERNEL___B", descriptorLabel());
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first->getKernel("HIP_KERNEL_GFX1100_AB"), nullptr);
    EXPECT_NE(second->getKernel("TEST_APP_KERNEL___B"), nullptr);

    // ...against one and the same hipModule_t. This is the behavioural half of the
    // not-keyed-by-symbol decision and the only test that can observe the sharing
    // directly.
    const auto* firstKpack = dynamic_cast<const KpackProgram*>(first.get());
    const auto* secondKpack = dynamic_cast<const KpackProgram*>(second.get());
    ASSERT_NE(firstKpack, nullptr);
    ASSERT_NE(secondKpack, nullptr);
    EXPECT_NE(firstKpack->module(), nullptr);
    EXPECT_EQ(firstKpack->module(), secondKpack->module());
    EXPECT_EQ(cache.size(), 1U);
}

} // namespace
} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
