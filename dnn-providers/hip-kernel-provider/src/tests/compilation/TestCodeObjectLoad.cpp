// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "PackedKernelSource.hpp"
#include "TestDescriptorRoot.hpp"
#include "compilation/CodeObjectLoad.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "engines/kernel_ingestor_engine/PackedPlanTestSupport.hpp"

/**
 * @file TestCodeObjectLoad.cpp
 * @brief loadCodeObjectOnDevice: a code object loads from bytes alone, and each failure
 *        names its step and the HIP error.
 *
 * The packed conv set is read from a copy in a scratch directory. Other suites corrupt or
 * delete the staged archives.
 */
namespace hip_kernel_provider::compilation
{
namespace
{

using hip_kernel_provider::testing::readPackedKernelDefinition;
using hip_kernel_provider::testing::unitKpackRoot;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::ScopedDirectory;

namespace packs = hip_kernel_provider::kernel_ingestor_engine::testing;

constexpr const char* SCRATCH_LABEL = "codeobjectload";

// The verified code object of the packed conv kernel, read from a copy of the packed set.
// `bytes` stays empty when nothing was packed for the device. `deviceArch` receives the
// decorated arch of device 0.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
void readPackedConvCodeObject(const ScopedDirectory& scratch,
                              std::string& deviceArch,
                              std::vector<uint8_t>& bytes,
                              std::string& symbol)
{
    hipdnn_plugin_sdk::ingestor::DeviceProperties deviceProperties;
    std::filesystem::path copy;
    ASSERT_NO_FATAL_FAILURE(
        packs::copyPackedArchForDevice(unitKpackRoot(), scratch.path(), deviceProperties, copy));
    deviceArch = deviceProperties.gcnArchName;
    if(copy.empty())
    {
        return;
    }

    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    ASSERT_NO_FATAL_FAILURE(readPackedKernelDefinition(copy, packs::CONV_FWD_DESCRIPTOR, kernel));

    std::string selected;
    const auto archive
        = std::filesystem::weakly_canonical(kernel.originDirectory / kernel.source.library);
    const KpackCodeObject codeObject = readVerifiedKpackCodeObject(
        archive.string(), kernel.source.tocKey, deviceArch, kernel.source.sha256, selected);
    const auto* first = static_cast<const uint8_t*>(codeObject.data());
    bytes.assign(first, first + codeObject.size());
    symbol = kernel.source.symbol;
}

// Reads and clears both HIP error values after a refused load. Each value is either
// hipSuccess or the refused status.
void consumeRefusedHipError(hipError_t refused)
{
    const hipError_t last = hipGetLastError();
    const hipError_t command = hipExtGetLastError();
    EXPECT_TRUE(last == hipSuccess || last == refused)
        << "hipGetLastError gave " << last << ", the refused status is " << refused;
    EXPECT_TRUE(command == hipSuccess || command == refused)
        << "hipExtGetLastError gave " << command << ", the refused status is " << refused;
}

TEST(TestCodeObjectLoad, LoadsAPackagedCodeObjectFromBytes)
{
    SKIP_IF_NO_DEVICES();

    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    std::string deviceArch;
    std::vector<uint8_t> bytes;
    std::string symbol;
    ASSERT_NO_FATAL_FAILURE(readPackedConvCodeObject(scratch, deviceArch, bytes, symbol));
    if(bytes.empty())
    {
        GTEST_SKIP() << "nothing was packed for this device (" << deviceArch
                     << "): " << unitKpackRoot() << " has no directory for it.";
    }

    const size_t cachedBefore = packs::packModuleCacheEntries();

    const LoadedCodeObject loaded = loadCodeObjectOnDevice(bytes, symbol, 0, "test conv kernel");

    EXPECT_NE(loaded.program, nullptr);
    EXPECT_NE(loaded.kernel, nullptr);
    EXPECT_EQ(packs::packModuleCacheEntries(), cachedBefore);
}

TEST(TestCodeObjectLoad, ReportsTheStageAndTheHipError)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<uint8_t> notACodeObject(256, 0xAB);
    hipError_t refused = hipSuccess;
    try
    {
        static_cast<void>(loadCodeObjectOnDevice(notACodeObject, "ConvFwd", 0, "test bytes"));
        ADD_FAILURE() << "expected bytes that are not a code object to be refused";
    }
    catch(const CodeObjectLoadFailure& failure)
    {
        refused = failure.hipStatus();
        EXPECT_EQ(failure.stage(), CodeObjectLoadStage::MODULE_LOAD) << failure.what();
        EXPECT_NE(failure.hipStatus(), hipSuccess) << failure.what();
        EXPECT_NE(std::string(failure.what()).find("hipModuleLoadData"), std::string::npos)
            << failure.what();
        EXPECT_NE(std::string(failure.what()).find("test bytes"), std::string::npos)
            << failure.what();
    }
    // HIP records the refused module load as the last error, and the load leaves it there.
    EXPECT_EQ(hipPeekAtLastError(), refused);
    consumeRefusedHipError(refused);

    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    std::string deviceArch;
    std::vector<uint8_t> bytes;
    std::string symbol;
    ASSERT_NO_FATAL_FAILURE(readPackedConvCodeObject(scratch, deviceArch, bytes, symbol));
    if(bytes.empty())
    {
        GTEST_SKIP() << "nothing was packed for this device (" << deviceArch
                     << "): " << unitKpackRoot() << " has no directory for it.";
    }

    refused = hipSuccess;
    try
    {
        static_cast<void>(loadCodeObjectOnDevice(bytes, "NoSuchKernel", 0, "test conv kernel"));
        ADD_FAILURE() << "expected a symbol the code object lacks to be refused";
    }
    catch(const CodeObjectLoadFailure& failure)
    {
        refused = failure.hipStatus();
        EXPECT_EQ(failure.stage(), CodeObjectLoadStage::SYMBOL_LOOKUP) << failure.what();
        EXPECT_NE(failure.hipStatus(), hipSuccess) << failure.what();
        EXPECT_NE(std::string(failure.what()).find("hipModuleGetFunction"), std::string::npos)
            << failure.what();
        EXPECT_NE(std::string(failure.what()).find("NoSuchKernel"), std::string::npos)
            << failure.what();
    }
    consumeRefusedHipError(refused);
}

} // namespace
} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
