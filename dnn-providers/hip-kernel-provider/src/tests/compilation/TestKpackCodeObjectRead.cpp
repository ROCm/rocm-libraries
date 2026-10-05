// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "PackedKernelSource.hpp"
#include "TestDescriptorRoot.hpp"
#include "compilation/KpackArchive.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdTestGraphs.hpp"
#include "utilities/Digest.hpp"

namespace hip_kernel_provider::compilation
{
namespace
{

using hip_kernel_provider::testing::copyPackedArchTree;
using hip_kernel_provider::testing::findPackedArchDirectory;
using hip_kernel_provider::testing::PackedKernelSource;
using hip_kernel_provider::testing::readPackedKernelDefinition;
using hip_kernel_provider::testing::readPackedKernelSource;
using hip_kernel_provider::testing::testKpackArchive;
using hip_kernel_provider::testing::unitKpackRoot;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::ScopedDirectory;

constexpr const char* SCRATCH_LABEL = "kpackcodeobjectread";

// rocm-kpack's own test archive holds gfx1100 and gfx1101 placeholder payloads under this
// toc key. They are not code objects.
constexpr const char* TEST_ARCHIVE_ARCH = "gfx1100";
constexpr const char* TEST_ARCHIVE_TOC_KEY = "lib/libhip.so#0";

constexpr const char* DIGEST = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

// The standalone descriptor of the packed conv set.
constexpr const char* PACKED_UKD_DESCRIPTOR = "conv_fwd_f16_block64.ukd.json";

TEST(TestKpackCodeObjectRead, ReportsAnAbsentArchive)
{
    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    const std::string absent = (scratch.path() / "absent.kpack").string();
    std::string selected = "unchanged";

    try
    {
        static_cast<void>(
            readVerifiedKpackCodeObject(absent, TEST_ARCHIVE_TOC_KEY, "gfx90a", DIGEST, selected));
        FAIL() << "expected an absent archive to be refused";
    }
    catch(const KpackModuleLoadFailure& failure)
    {
        const std::string message = failure.what();
        EXPECT_EQ(failure.stage(), KpackLoadStage::OPEN_ARCHIVE) << message;
        EXPECT_NE(message.find("does not exist"), std::string::npos) << message;
        EXPECT_NE(message.find(absent), std::string::npos) << message;
    }
    EXPECT_EQ(selected, "unchanged");
}

TEST(TestKpackCodeObjectRead, RejectsAPayloadThatIsNotACodeObject)
{
    ASSERT_TRUE(std::filesystem::exists(testKpackArchive()))
        << "the test kpack archive, resolved relative to this binary, is missing: "
        << testKpackArchive();

    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    const auto archive = scratch.path() / testKpackArchive().filename();
    std::filesystem::copy_file(testKpackArchive(), archive);

    std::string selected;
    try
    {
        static_cast<void>(readVerifiedKpackCodeObject(
            archive.string(), TEST_ARCHIVE_TOC_KEY, TEST_ARCHIVE_ARCH, DIGEST, selected));
        FAIL() << "expected a payload that is not a code object to be refused";
    }
    catch(const KpackModuleLoadFailure& failure)
    {
        EXPECT_EQ(failure.stage(), KpackLoadStage::DECOMPRESS) << failure.what();
        EXPECT_NE(std::string(failure.what()).find("KPACK_ERROR_INVALID_METADATA"),
                  std::string::npos)
            << failure.what();
    }
}

// The local device's decorated arch, and a copy of the conv set this build packed for it.
// `copy` stays empty when nothing was packed for the device.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
void copyPackedConvSet(const ScopedDirectory& scratch,
                       std::string& deviceArch,
                       std::filesystem::path& copy)
{
    hipDeviceProp_t properties{};
    std::string arch;
    std::filesystem::path packed;
    ASSERT_NO_FATAL_FAILURE(findPackedArchDirectory(properties, arch, packed));
    deviceArch = properties.gcnArchName;
    if(!packed.empty())
    {
        copy = copyPackedArchTree(packed, scratch.path());
    }
}

TEST(TestKpackCodeObjectRead, ReturnsVerifiedBytesWithoutLoadingAModule)
{
    SKIP_IF_NO_DEVICES();

    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    std::string deviceArch;
    std::filesystem::path packed;
    ASSERT_NO_FATAL_FAILURE(copyPackedConvSet(scratch, deviceArch, packed));
    if(packed.empty())
    {
        GTEST_SKIP() << "nothing was packed for this device (" << deviceArch
                     << "): " << unitKpackRoot() << " has no directory for it.";
    }

    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    ASSERT_NO_FATAL_FAILURE(readPackedKernelDefinition(packed, PACKED_UKD_DESCRIPTOR, kernel));

    hipdnn_plugin_sdk::ingestor::DeviceProperties deviceProperties;
    deviceProperties.gcnArchName = deviceArch;
    const kernel_ingestor_engine::testing::GraphFixture fixture(
        kernel_ingestor_engine::testing::buildConvFwdGraph(
            hipdnn_flatbuffers_sdk::data_objects::DataType::HALF),
        deviceProperties);

    KpackModuleCache cache;
    const KpackKernelLoader loader(cache);
    const auto code = kernel_ingestor_engine::buildIngestorKernelCode(
        loader, fixture.context(), kernel, kernel.source.signature);
    ASSERT_EQ(cache.size(), 1U);

    // The plan keeps its own module, so the cache can be emptied. A read that loaded a
    // module through the cache would fill it again.
    cache.clear();

    const auto codeObject = code.readCodeObject();

    EXPECT_EQ(cache.size(), 0U);
    EXPECT_EQ(utilities::sha256Hex(codeObject.bytes.data(), codeObject.bytes.size()),
              kernel.source.sha256);
    EXPECT_EQ(codeObject.sha256, kernel.source.sha256);
    EXPECT_EQ(codeObject.symbol, kernel.source.symbol);
    EXPECT_TRUE(hipdnn_plugin_sdk::archMatches(
        deviceArch, codeObject.target, hipdnn_plugin_sdk::ArchMatchMode::PREFIX))
        << codeObject.target << " does not serve " << deviceArch;

    KpackArchive archive;
    KpackError error;
    ASSERT_TRUE(archive.open(kernel.originDirectory / kernel.source.library, error))
        << error.codeName;
    std::vector<std::string> arches;
    ASSERT_TRUE(archive.architectures(arches, error)) << error.codeName;
    EXPECT_NE(std::find(arches.begin(), arches.end(), codeObject.target), arches.end())
        << codeObject.target << " is not an entry of the archive";
}

TEST(TestKpackCodeObjectRead, RefusesBytesWhoseDigestDisagrees)
{
    SKIP_IF_NO_DEVICES();

    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    std::string deviceArch;
    std::filesystem::path packed;
    ASSERT_NO_FATAL_FAILURE(copyPackedConvSet(scratch, deviceArch, packed));
    if(packed.empty())
    {
        GTEST_SKIP() << "nothing was packed for this device (" << deviceArch
                     << "): " << unitKpackRoot() << " has no directory for it.";
    }

    PackedKernelSource source;
    ASSERT_NO_FATAL_FAILURE(readPackedKernelSource(packed, PACKED_UKD_DESCRIPTOR, source));

    // A well-formed digest that differs from the descriptor's in one character.
    std::string wrongDigest = source.sha256;
    ASSERT_FALSE(wrongDigest.empty());
    wrongDigest[0] = wrongDigest[0] == '0' ? '1' : '0';

    std::string selected;
    try
    {
        static_cast<void>(readVerifiedKpackCodeObject(
            source.archive.string(), source.tocKey, deviceArch, wrongDigest, selected));
        FAIL() << "expected bytes that do not match the expected digest to be refused";
    }
    catch(const KpackModuleLoadFailure& failure)
    {
        const std::string message = failure.what();
        EXPECT_EQ(failure.stage(), KpackLoadStage::DIGEST_MISMATCH) << message;
        EXPECT_NE(message.find(wrongDigest), std::string::npos) << message;
    }
}

} // namespace
} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
