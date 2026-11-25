// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <filesystem>
#include <set>
#include <string>
#include <string_view>
#include <system_error>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include "TestDescriptorRoot.hpp"
#include "engines/kernel_ingestor_engine/ConvEngineGroupGate.hpp"

/**
 * @file TestConvEngineGroupGate.cpp
 * @brief The conv engine group's runtime opt-in (HIP_KERNEL_PROVIDER_ENABLE_CONV): which
 *        values enable it, and that withdrawing the group from the provider's own descriptor
 *        catalog leaves none of its engines or packs and every other engine as it was.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter;

bool isConvEngine(const std::string& name)
{
    return std::string_view(name).substr(0, CONV_ENGINE_PREFIX.size()) == CONV_ENGINE_PREFIX;
}

TEST(TestConvEngineGroupGate, DisabledOnlyByAFalseValue)
{
    for(const char* value : {"", "1", "true", "TRUE", "on", "yes", " 1 "})
    {
        const ScopedEnvironmentVariableSetter env(CONV_ENABLE_ENV, value);
        EXPECT_TRUE(convEngineGroupEnabled()) << "'" << value << "'";
    }
    for(const char* value : {"0", "false", "OFF", "no", " 0 "})
    {
        const ScopedEnvironmentVariableSetter env(CONV_ENABLE_ENV, value);
        EXPECT_FALSE(convEngineGroupEnabled()) << "'" << value << "'";
    }
}

#ifdef HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR
TEST(TestConvEngineGroupGate, WithdrawingLeavesNoConvEngineAndEveryOtherEngine)
{
    // The production tree this build packed: the shipped engines, the conv group's included.
    const auto root = hip_kernel_provider::testing::descriptorSetRoot(
        HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR);
    std::error_code absent;
    if(root.empty() || !std::filesystem::is_directory(root, absent))
    {
        GTEST_SKIP() << "no production descriptor tree under " << root;
    }
    auto catalog = hipdnn_plugin_sdk::ingestor::loadDescriptorCatalog(root);

    std::size_t convEngines = 0;
    std::set<std::string> others;
    for(const auto& [id, entry] : catalog.engines)
    {
        if(isConvEngine(entry.descriptor.name))
        {
            ++convEngines;
        }
        else
        {
            others.insert(entry.descriptor.name);
        }
    }
    const std::size_t packs = catalog.packs.size();
#ifdef HIPDNN_ENGINE_CONV_NAIVE
    EXPECT_GT(convEngines, 0u) << "this build ships the conv group; its engines must load";
#endif

    withdrawConvEngineGroup(catalog);

    std::set<std::string> remaining;
    for(const auto& [id, entry] : catalog.engines)
    {
        EXPECT_FALSE(isConvEngine(entry.descriptor.name)) << entry.descriptor.name;
        remaining.insert(entry.descriptor.name);
    }
    EXPECT_EQ(remaining, others);
    // Every pack left names an engine left: none is orphaned by the withdrawal.
    for(const auto& [key, entry] : catalog.packs)
    {
        EXPECT_NE(catalog.engines.find(entry.descriptor.engineId), catalog.engines.end())
            << entry.descriptor.name;
    }
    if(convEngines == 0)
    {
        EXPECT_EQ(catalog.packs.size(), packs);
    }
}
#endif // HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
